## Introduction

Objective of the challenge is to analyze BankSim Kaggle dataset.

For this challenge, I covered the following steps:

 - EDA
 - Training a XGB model to predict Gender
 - Hyperparameter tuning of XGB model
 - Training a Random Forest model to predict Gender
 - Hyperparameter tuning of Random Forest model
 - Model assessment
 - CLV model implementation for customer long-term values estimation

## 1) Repository structure

Current repository in inspired from https://drivendata.github.io/cookiecutter-data-science/
```
/data:
  /output -> Predictions
  /raw -> Raw data
  /processed -> Processed data (dataset with generated features)
/models:
  *.pkl -> Pickled trained models / Hyperopt trials
/params:
  *.json -> Model parameters
/notebooks
  EDA.ipynb -> Use for EDA
  CLV.ipynb -> Use to validate CLV predictions
  assess.ipynb -> Use for assessing model performance
/src
  features.py --> Generate features for classification
  train.py --> Train ML model
  optimize.py --> Run Hyperopt with TPE
  model.py --> Class to make it easier to fit multiple models
  spaces.py --> Search space used for optimization
  clv.py --> Run CLV to predict long-term values
  /resources
    /sql
      *.sql --> SQL scripts
Makefile -> Use to simplify pipeline execution
requirements.txt -> Packages to install
setup.py -> Use to install current package
```

## 2) Quickstart

Before anything else, you must download the dataset ```bs140513_032310.csv``` from Kaggle and paste it into ```data/raw``` folder.<br/><br/>
If you would like to get started ASAP, run these make commands in the following order:<br/>
```make venv``` --> Set-up python virtual environment<br/>
```make features``` --> Generate features<br/>
```make train_xgb``` --> Train initial XGB model for gender classification<br/>
```make optimize_xgb``` --> Run Hyperopt hyperparameter optimizer for XGB model<br/>
```make train_opt_xgb``` --> Train optimal XGB model for gender classification<br/>
```make train_rf``` --> Train initial Random Forest model for gender classification<br/>
```make optimize_rf``` --> Run Hyperopt hyperparameter optimizer for Random Forest model<br/>
```make train_opt_rf``` --> Train optimal Random Forest model for gender classification<br/>
```make clv``` --> Run CLV model to predict long-term values<br/>

## 3) Set-up Environment

Run the following command:<br/>
```make venv```<br/>
It will install all necessary packages used in this challenge.

## 4) Exploratory Data Analysis

EDA is available in notebook ```notebooks/EDA.ipynb```.<br/>

In this notebook, I am exploring each variable in the dataset and how they correlate with Gender which is our ML problem outcome variable.
I also try building more insightful features which will help us achieve better results on our ML task.

## 5) Classification using XGBoost

Second model is a XGBoost model. <br/>
This model was chosen as it yields good results without much data transformation (such as normalization, clipping etc...) required. <br/>
Its parameters can be found in ```params/def_xgb_model.json```. <br/>


```
{
  "name": "xgb",
  "clf": "XGBClassifier",
  "params": {
    "max_depth": 4,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "colsample_bytree": 0.5,
    "gamma": 1
  }
}
```

### a) Training initial XGB model

I use 90% of the data for training and the remaining 10% for validation to assess that model doesn't overfit and generalizes Ill to new data.

Run the following command:<br/>
```make train_xgb```<br/>

You should reach an AUC of {} for the validation set.<br/>
This is a very low score which shows our model didn't learn successfully on our classification task.

### b) Optimize model

To try reaching better results, I optimize model parameters.<br/>
I use ```hyperopt``` package for that purpose and our objective is to maximize validation AUC using K-Folds with K=10. TPE algorithm is picked to search the space for the best parameters.<br/>

Search history is dumped at every round in ```models/opt_xgb_trials.pkl``` so that optimizer can be stopped and resumed  anytime.

To run the optimizer, run the following command:<br/>
```make optimize_xgb```<br/>

### c) Training optimized XGB model

Its parameters can be found in ```params/opt_xgb_model.json```. <br/>

```
{
	"name": "opt_xgb",
	"clf": "XGBClassifier",
	"params": {
		"colsample_bytree": 0.5,
		"eta": 0.157,
		"gamma": 0.5700000000000001,
		"max_depth": 3,
		"min_child_Iight": 5.0,
		"n_estimators": 162.0,
		"subsample": 0.1
	}
}
```

Run the following command:<br/>
```make train_opt```<br/>

Validation AUC is improved to .<br/>
It is still not great but it shows the model learnt a bit

## 6) Classification using RandomForest

I try to use another model.<br/>
I picked RandomForest as it is less prone to overfitting than Gradient Boosting Algorithm.

Its parameters can be found in ```params/def_rf_model.json```. <br/>

### a) Training initial RandomForest model

Run the following command:<br/>
```make train_rf```<br/>

### b) Optimize model

To run the optimizer, run the following command:<br/>
```make optimize_rf```<br/>

### c) Training optimized RandomForest model

Its parameters can be found in ```params/opt_rf_model.json```. <br/>

```
{
	"name": "opt_rf",
	"clf": "RandomForestClassifier",
	"params": {
		"ccp_alpha": 0.007,
		"class_weight": "balanced",
		"max_depth": 1,
		"max_features": 0.92,
		"max_leaf_nodes": 3,
		"max_samples": 0.64,
		"n_estimators": 209.0,
		"n_jobs": -1,
		"verbose": 0,
		"warm_start": false
	}
}
```

## 6) Model assessment

Model is assessed in notebook ```notebooks/assess.ipynb```.<br/>

In this notebook, I am investigating different plots and metrics to assess model performance.<br/>
I am also looking at feature importance for model interpretability.

## 7) Predict customer long-term values

To find out customer long-term values, I use a package called ```lifetimes```.<br/>
It combines 2 bayesian models (BG/NBD and Gamma-Gamma models).<br/>
Both models require at most RFM features (Recency, Frequency, Monetary).<br/>
BG/NBD predicts expected number of future purchases per customer.<br/>
Gamma-Gamma predicts expected amount for each purchase per customer.<br/>
Multiplying both provides customer long-term values.
