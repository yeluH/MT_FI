# MT_FI workflow
Further improvement based on master thesis


$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$

## Improvement of Regression model

$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$


### Work 0_Regression_presence_of_accident
- Focus on dependent variable: presence of accident
- Apply feature selection methods and compare the selection result
  - Sequential feature selection
    - Methods: Forward/Backward + Floating
    - Metrics used in scoring performance:
          - For linear regression: neg_mean_squared_error, r2, neg_median_absolute_error, neg_mean_absolute_error
          - For logistic regression: accuracy, f1, precision, recall, (roc_auc)
    - Regression models: linear regression, logistic regression model
    - Cross validation k: 5, 10, 15, 20...
- Apply regression model using the selected features and compare the regression result
  - Model performance: R-squared, Adj. R-squared
  - (Significant features) 



$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$

<!-- ## Other steps -->


$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$

## Issues
Errors were found in setting scoring metrics for logistic regression. (20240903) Solving...
