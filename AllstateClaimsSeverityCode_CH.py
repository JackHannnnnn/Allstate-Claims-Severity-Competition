# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 16:28:55 2016

@author: Chaofan
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb


from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from scipy.stats import skew, boxcox
import itertools

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model



SEED = 2016
SHIFT = 200


def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r

def fair_obj(preds, dtrain):
    fair_constant = 0.7
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-SHIFT,
                                      np.exp(yhat)-SHIFT)

def munge_skewed(train, test, numeric_feats):
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.25]     
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test


def preprocess(train, test):
    start = datetime.now()
    print 'Data preprocessing starts'
    
    shift = 200
    comb_feature = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
                   'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
                   'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
                   'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')
                
    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    categorical_feats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test = munge_skewed(train, test, numeric_feats)
    
    
    for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)

            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x

            train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)

    
    train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
    train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
    train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
    train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
    train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
    train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
    train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

    train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"]) + 0000.1)
    train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"]) + 0000.1)
    train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"]) + 0000.1)
    train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"]) + 0000.1)
    train_test["cont14"] = (np.maximum(train_test["cont14"] - 0.179722, 0) / 0.665122) ** 0.25

    for comb in itertools.combinations(comb_feature, 2):
        feat = comb[0] + "_" + comb[1]
        train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
        train_test[feat] = train_test[feat].apply(encode)
        
    for col in categorical_feats:      
        train_test[col] = train_test[col].apply(encode)

    ss = StandardScaler()
    train_test[numeric_feats] = \
        ss.fit_transform(train_test[numeric_feats].values)

    train = train_test.iloc[:ntrain, :].copy()
    test = train_test.iloc[ntrain:, :].copy()

    y_train = np.log(train['loss'] + shift)
    x_train = train.drop(['loss','id'], axis=1)
    x_test = test.drop(['loss','id'], axis=1)
    
    print 'Data preprecessing is done'
    print 'Time elapsed: ', datetime.now() - start
    return x_train.values, y_train.values, x_test.values
    

class SklearnWrapper(object):
    def __init__(self, clf, seed=SEED, params=None):
        self.params = params
        self.params['random_state'] = seed
        self.clf = clf(**self.params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=SEED, params=None):
        self.params = params
        self.params['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train, x_val, y_val):
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_valid = xgb.DMatrix(x_val, label=y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]
        self.xgb = xgb.train(self.params, 
                             d_train, 
                             self.nrounds, 
                             watchlist,
                             early_stopping_rounds=60,
                             obj=fair_obj,
                             feval=xg_eval_mae,
                             verbose_eval=30)

    def predict(self, x):
        return self.xgb.predict(xgb.DMatrix(x), ntree_limit=self.xgb.best_ntree_limit)

    
class NnWrapper(object):
    def __init__(self, seed=SEED, params=None):
        self.params = params
        self.model = self.nn_mlp(input_dim, self.params)
        
    def train(self, x_train, y_train, x_val, y_val):
        self.model.fit_generator(generator=self.batch_generator(x_train, y_train, self.params['batch_size'], True),
                                 nb_epoch=self.params['n_epochs'],
                                 samples_per_epoch=x_train.shape[0],
                                 verbose=0,
                                 validation_data=(x_val, y_val),
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=10), 
                                            ModelCheckpoint('keras-regressor.check', monitor='val_loss', save_best_only=True, verbose=0)])
        self.model = load_model('keras-regressor.check')
        
    def predict(self, x):
        return self.model.predict_generator(generator=self.batch_generator(x, batch_size=800, shuffle=False), 
                                            val_samples=x.shape[0])[:, 0]
    
    def nn_mlp(self, input_shape, params):      
        model = Sequential()

        for i, layer_size in enumerate(params['layers']):

            if i == 0:
                model.add(Dense(layer_size, init='he_normal', input_dim=input_shape))
            else:
                model.add(Dense(layer_size, init='he_normal'))

            if params.get('batch_norm', False):
                model.add(BatchNormalization())

            if 'dropouts' in params:
                model.add(Dropout(params['dropouts'][i]))

            model.add(PReLU())

        model.add(Dense(1, init='he_normal'))
        model.compile(loss='mae', optimizer=params['optimizer'])
        return model
    
    def batch_generator(self, X, y=None, batch_size=128, shuffle=False):
        index = np.arange(X.shape[0])
        while True:
            if shuffle:
                np.random.shuffle(index)

            batch_start = 0
            while batch_start < X.shape[0]:
                batch_index = index[batch_start:batch_start + batch_size]
                batch_start += batch_size

                X_batch = X[batch_index, :]

                if y is None:
                    yield X_batch
                else:
                    yield (X_batch, y[batch_index])
    
    
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_val = x_train[test_index]
        y_val = y_train[test_index]

        params = clf.params
        if type(clf) is NnWrapper and params.get('n_bags', None) is not None:
            pred_bag = np.zeros(x_val.shape[0])
            pred_test = np.zeros(x_test.shape[0])
            for j in xrange(params['n_bags']):
                clf.train(x_tr, y_tr, x_val, y_val)
                pred_bag += clf.predict(x_val)
                pred_test += clf.predict(x_test)
                
            oof_train[test_index] = pred_bag / params['n_bags']
            oof_test_skf[i, :] = pred_test / params['n_bags']
        elif type(clf) is XgbWrapper:
            clf.train(x_tr, y_tr, x_val, y_val)
            oof_train[test_index] = clf.predict(x_val)
            oof_test_skf[i, :] = clf.predict(x_test)
        else:
            clf.train(x_tr, y_tr)
            oof_train[test_index] = clf.predict(x_val)
            oof_test_skf[i, :] = clf.predict(x_test)
    
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) 


def weighted_average(x_train, y_train, x_test, n_iter=100):
    def mae_loss(weights):
        final_pred = 0
        for i, weight in enumerate(weights):
            final_pred += weight*x_train[:, i]
        return mean_absolute_error(np.exp(final_pred), np.exp(y_train))    
    
    start = datetime.now()
    scores = []
    weights = []
    for i in xrange(n_iter):
        bounds = [(0,1 )] * x_train.shape[1]
        cons = ({'type': 'eq', 'fun': lambda w: 1-sum(w)})
        starting_values = np.random.uniform(size=x_train.shape[1])
        try:
            res = minimize(mae_loss,
                           starting_values,
                           method='SLSQP',
                           bounds=bounds,
                           constraints=cons)
            best_score = res['fun']
            best_weights = res['x']
            scores.append(best_score)
            weights.append(best_weights)
        except ValueError:
            pass
    optimal_weights = weights[scores.index(np.array(scores).min())]
    print 'Best mae: ', scores[scores.index(np.array(scores).min())]
    print 'Best weights: ', optimal_weights
    print 'Time elapsed: ', datetime.now() - start
    
    pred_test = np.zeros(x_test.shape[0])
    for i, weight in enumerate(optimal_weights):
            pred_test += weight*x_test[:, i]
    return pred_test / len(optimal_weights)        
    
    

if __name__ == '__main__':
    directory = '~/Downloads/KaggleCompetition/allstate_claims/'
    train = pd.read_csv(directory + 'train.csv')
    test = pd.read_csv(directory + 'test.csv')
    test_ids = test['id']
    ntest = test.shape[0]
    ntrain = train.shape[0]
    
    NFOLDS = 10
    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
    
    x_train, y_train, x_test = preprocess(train, test)
    input_dim = x_train.shape[1]
    print '\n'
    
    
    et_params = {
        'n_jobs': -1,
        'n_estimators': 600,
        'max_features': 0.63,
        'max_depth': 5
    }

    rf_params = {
        'n_jobs': -1,
        'n_estimators': 1000,
        'max_features': 0.6,
        'min_samples_leaf': 5
    }

    xgb_params = {
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.03,
        'booster': 'gbtree',
        'max_depth': 12,
        'min_child_weight': 100,
        'eval_metric': 'mae',
        'nrounds': 100000
    }
    
    nn_params = {'n_epochs': 60, 
                 'batch_size': 128, 
                 'layers': [400, 200, 50], 
                 'dropouts': [0.4, 0.2, 0.2], 
                 'batch_norm': True, 
                 'optimizer': 'adadelta',
                 'n_bags': 10
    }
        
    xg = XgbWrapper(seed=SEED, params=xgb_params)
    et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
    rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
    nn = NnWrapper(seed=SEED, params=nn_params)

    print 'Xgb starts'
    xgb_oof_train, xgb_oof_test = get_oof(xg, x_train, y_train, x_test)
    print 'Xgb is done.'
    print 'Xgb-cv-mae: ', mean_absolute_error(np.exp(y_train), np.exp(xgb_oof_train))
    print '\n'
    
    print 'Extra Tree starts'
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
    print 'Extra Tree is done'
    print 'ET-cv-mae: ', mean_absolute_error(np.exp(y_train), np.exp(et_oof_train))
    print '\n'
    
    print 'Random Forest starts'
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
    print 'Random Forest is done'
    print 'RF-cv-mae: ', mean_absolute_error(np.exp(y_train), np.exp(rf_oof_train))
    print '\n'
    
    print 'Neural Network starts'
    nn_oof_train, nn_oof_test = get_oof(nn, x_train, y_train, x_test)
    print 'Neural Network is done'
    print 'NN-cv-mae: ', mean_absolute_error(np.exp(y_train), np.exp(nn_oof_train))
    print '\n'
    
    print 'Regularized Greedy Forest starts'
    rgf_oof_train = pd.read_csv(directory+'Ensemble/rgf_oof_train.txt', header=None)
    rgf_oof_test = pd.read_csv(directory+'Ensemble/rgf_oof_test.txt', header=None)
    print 'RGF-cv-mae: ', mean_absolute_error(np.exp(y_train), np.exp(rgf_oof_train))
    print '\n'
    
    
    # Stacking
    print 'Stacking...'
    l2_x_train = np.concatenate((xgb_oof_train, et_oof_train, rf_oof_train, nn_oof_train, rgf_oof_train), axis=1)
    l2_x_test = np.concatenate((xgb_oof_test, et_oof_test, rf_oof_test, nn_oof_test, rgf_oof_test), axis=1)

    l2_xgb_params = {
        'colsample_bytree': 1,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.025,
        'booster': 'gbtree',
        'max_depth': 4,
        'min_child_weight': 69,
        'eval_metric': 'mae',
        'nrounds': 100000
    }
    
    l2_xgb = XgbWrapper(seed=SEED, params=l2_xgb_params)
    l2_xgb_oof_train, l2_xgb_oof_test = get_oof(l2_xgb, l2_x_train, y_train, l2_x_test)
    print 'l2-xgb-cv-mae: ', mean_absolute_error(np.exp(y_train), np.exp(l2_xgb_oof_train))
    
    print 'Writing l2-xgb result...'
    l2_xgb_result = pd.DataFrame({'id': test_ids, 'loss': np.exp(l2_xgb_oof_test[:, 0])-SHIFT})
    l2_xgb_filename = 'l2_xgb'
    l2_xgb_result.to_csv(l2_xgb_filename + '_ensemble_result.csv', index=False)
    print '\n'
    
    l2_wa_test = weighted_average(l2_x_train, y_train, l2_x_test, n_iter=10)
    print 'Writing l2-weighted-average result...'
    l2_wa_result = pd.DataFrame({'id': test_ids, 'loss': np.exp(l2_wa_test)-SHIFT})
    l2_wa_filename = 'l2_weighted_average'
    l2_wa_result.to_csv(l2_wa_filename + '_ensemble_result.csv', index=False)
    print '\n'
    print 'End'
    