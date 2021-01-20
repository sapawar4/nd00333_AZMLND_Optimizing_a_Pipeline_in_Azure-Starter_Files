# Optimizing an ML Pipeline in Azure | Sandeep Pawar

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset is from a banking institution about their campaign to enroll new customers. It has features that are related to the customer (age, marital status, job etc.), macroeconomic conditions at the time pf the campaign (bond rate, employment rate etc.) and some features related to the campaign (previous campaign, mode of contact etc.). There are 32,941 observations in the dataset. The goal of the project is to predict if a customer will enroll in bank's offering, given the historical data. 

The most accurate model was a Voting Ensemble model that combines multiple classifiers. This model has accuracy of nearly 80%. 
 
## Scikit-learn Pipeline
In the first approach, a Logistic Regression model was used. In a machine learning process, it is important to create a baseline model, typically starting with linear model, to establish baseline performance. We always we want to make sure we deploy the most accurate parsimonious model. 

'duration' column from the dataset was dropped to avoid data leakage. The data was cleaned and categorical columns were one hot encoded to make the dataset suitable for using in Logistic Regression model. The dataset was split into 80% train and 20% test. Before splitting, the data was shuffled and stratification was used to ensure train and test have equal classes to avoid bias. 
 
For the Logistic Regression model, three hyper-parameters were tuned:

 - **Regularization strength ( C):** This parameter is actually inverse of the regularization (1/lambda). It is used to avoid overfitting. c = [`100, 10, 1.0, 0.1, 0.01]`
 -  **Solver:**  Solver used to fit the model and find parameters that minimize error. Three solvers were considered `['lbfgs', 'liblinear','saga','sag']`
 - **Regularization** type: Both lasso (l1) and ridge (l2) regularization were considered. 

RandomSampling was used to sample the hyper parameters optimally. Grid search would have been exhaustive and computationally expensive, while Bayesian sampling does not support early termination. Random sampling provides a more optimal approach. For early termination, Bandit Policy was used. Bandit policy lets you run the optimization on enough parameters by specifying delay evaluation and terminate if the model cannot be further optimized. Slack factor of 0.1 was used, which means if the model performance falls below 91% of the best performing run in the first 20 runs, hyperdrive will terminate.

To assess the accuracy of the models, AUC (Area Under the Curve) was used as the accuracy metric as the target label was imbalanced (~88% No, 12% yes).

The best Logistic Regression model had AUC ~77.3% with 'l1' penalty, regularization = 1 and 'liblinear' solver. 


## AutoML
For AutoML, the same dataset and data cleaning steps were used for fair comparison. 5 fold cross-validation was used to mitigate model overfitting. 
The best performing model was a voting ensemble of 6 different models.  The AUC for the best AutoML model was 79.8% which is better than the best performing Logistic Regression model.

|Algorithm  | Weight |
|--|--|
| xgboost + maxabs scaler | 0.14% |
|lightgbm  + maxabs scaler| 0.42% |
|xgb  + maxabs scaler| 0.07% |
|SGD  + minmiaxscaler| 0.07% |
|lightgbm  + standard scaler| 0.21% |
|SGD  + standard scaler| 0.07% |

## Pipeline comparison
Logistic Regression is a linear model which means the classification boundary between the label classes would be linear. In this dataset if the boundary is non linear, logistic regression will not be able to capture that. Logistic Regression produced the best model with AUC = ~77% while AutoML with an ensemble of 6 different models produced AUC ~80%. Though the difference between the two models is not significant, an ensemble model is typically less prone to overfit. Thus, we will choose the AutoML model.  

## Future work

 - Use tree based classifier such as Random Forest, Gradient Boosting Classifier instead of Logistic regression
 - One hot encoding was used, instead of that different categorical encoding such as target encoding, mean encoding could be used
 - Bayesian sampling with more time to tune parameters could be used 
 - This dataset contains personal data of many individuals. While creating ML models, its is important that the ML model is not biased toward any class of population. More study is needed and ensure fairness of the model.
 - More study is needed to understand model uncertainty and understand where the model accuracy is below the aggregate accuracy of 80% 


