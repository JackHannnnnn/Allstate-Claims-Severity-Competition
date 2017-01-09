# Allstate_Claims_Severity_Competition
Allstate is currently developing automated methods of predicting the cost, and hence severity, of claims, which is a typical regression problem. One interesting thing about this competition is that the optimized objective is MAE (Mean Absolute Error). Finally, I ranked top 11% in this Kaggle Competition.
## Data Profile
|Data Type|Number|
|:---|:---:|
|Training Data Size | 188318|
|Test Data Size | 125546 |
|Number of Features|130|
|Number of Continuous Variables|14|
|Number of Categorical Variables|116|
## Feature Engineering
- Categorical Feature Interaction
- Categorical Feature Alphabetical Encoding
- Numeric Feature Unskewness
- Numeric Feature Normalization
- Log + Shift Transformation of Target Value

## 1st Level Models
- Xgboost
- Neural Network
- Random Forest
- Extra Trees
- Regularized Greedy Forest

## 2nd level Models
- Stacking (Xgboost)
- Weighted Average

## Tools
- Numpy
- Pandas
- Scipy
- Sklearn
- Xgboost
- Keras
