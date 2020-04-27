from pathlib import Path
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, \
                     Trials, \
                     fmin, \
                     tpe, \
                     space_eval
from functools import partial
from src.model import Model
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
from sklearn.model_selection import KFold
import xgboost as xgb
import click
import pickle
import json
import os

project_dir = Path(__file__).resolve().parents[1]

model_dir = "{}/{}".format(project_dir, "models")
params_dir = "{}/{}".format(project_dir, "params")
processed_dir = "{}/{}".format(project_dir, "data/processed")
dataset_filepath = "{}/{}".format(processed_dir, "features.csv")

NFOLDS = 10
TARGET_COLUMN = "gender"


def cross_validate_model(X, Y, model):

    kf = KFold(n_splits=NFOLDS)

    all_AUCs = []

    for i, (train_idxs, validation_idxs) in enumerate(kf.split(X)):

        X_train, X_validation = X[train_idxs], X[validation_idxs]
        y_train, y_validation = Y[train_idxs], Y[validation_idxs]

        fitted_model, AUCs = model.fit(X_train, y_train, eval_set=[(X_train, y_train),
                                                                   (X_validation, y_validation)])

        print("Fold {} AUCs: {}".format(i, AUCs))

        all_AUCs.append(AUCs)

    return all_AUCs


def get_model_loss(params_dict: dict,
                   dataset: Tuple[np.ndarray, np.ndarray],
                   model: Model,
                   max_mean_val_AUC: float,
                   best_params: dict,
                   trials: Trials,
                   trials_filepath: str):
    """
    Train model and compute loss (1-AUC) on fold validation sets

    Parameters:
    params_dict (dict): Model parameters
    data (Tuple[np.ndarray, np.ndarray]): Tuple of predictor and target variable arrays
    model (Model): Base Model to update with params_dict
    max_mean_val_AUC (float): Best validation AUC obtained so far
    best_params (dict): Best set of parameters (which maximizes max_mean_val_AUC)
    trials_filepath (str): Path where to store trials as pickle
    trials (Trials): Object aggregating all trial run

    Returns:
    dict: Dict with loss and round status
    """

    X = dataset[0]
    Y = dataset[1]

    print("========== NEW ROUND ===========")
    print("Round number {}".format(len(trials)))
    print("Max Mean Val AUC: {}".format(max_mean_val_AUC))
    print("Best params: {}".format(best_params))
    print("Nb trials executed: {}".format(len(trials)))
    print("Running round with {}".format(params_dict))

    model.params.update(params_dict)

    all_AUCs = cross_validate_model(X=X, Y=Y, model=model)
    mean_val_AUC = np.mean([AUCs[1] for AUCs in all_AUCs])
    loss = 1 - mean_val_AUC

    pickle.dump(trials, open(trials_filepath, "wb"))

    if max_mean_val_AUC < mean_val_AUC:
        max_mean_val_AUC = mean_val_AUC
        best_params = params_dict

    print("Round Mean Val AUC: {}".format(mean_val_AUC))

    return {"loss": loss, "status": STATUS_OK}


@click.command()
@click.option("--model_json")
def optimize(model_json):

    df_dataset = pd.read_csv(dataset_filepath, index_col=0)
    df_dataset.fillna(value=-1, inplace=True)
    df_dataset = df_dataset.sample(frac=1, random_state=0)

    columns = df_dataset.columns.tolist()
    feature_columns = sorted([column for column in columns if column != TARGET_COLUMN])

    # Normalize inputs
    df_X = df_dataset[feature_columns]
    df_X.fillna(value=0, inplace=True)
    df_X = (df_X - df_X.mean(axis=0)) / df_X.std(axis=0)

    X = df_X.values
    Y = df_dataset[TARGET_COLUMN].values

    model = Model.get_model_from_json(model_json)

    # Define search space
    if model.clf == RandomForestClassifier:
        from spaces import rf_space
        space = rf_space
    elif model.clf == xgb.XGBClassifier:
        from spaces import xgb_space
        space = xgb_space

    trials_filename = "{}_model_trials.pkl".format(model.name)
    trials_filepath = "{}/{}".format(model_dir, trials_filename)

    trials = Trials()
    max_mean_val_AUC = 0
    best_params = {}

    if os.path.exists(trials_filepath) is True:

        trials = pickle.load(open(trials_filepath, "rb"))
        trial_list = [trial for i, trial in enumerate(trials) if "loss" in trial["result"]]

        if len(trial_list) > 0:

            best_trial = trial_list[np.argmin([trial["result"]["loss"] for trial in trial_list])]
            max_mean_val_AUC = 1 - best_trial["result"]["loss"]
            best_params = pd.DataFrame.from_dict(best_trial["misc"]["vals"]) \
                                      .to_dict(orient="records")[0]
            best_params = space_eval(space, best_params)

            print("Use existing Trial object")
            print("Nb trials executed: {}".format(len(trials)))
            print("Current Max Mean Val AUC: {}".format(max_mean_val_AUC))
            print("Current Best params: {}".format(best_params))

    fn = partial(get_model_loss,
                 model=model,
                 max_mean_val_AUC=max_mean_val_AUC,
                 best_params=best_params,
                 trials_filepath=trials_filepath,
                 trials=trials,
                 dataset=(X, Y))

    best = fmin(fn=fn,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=200)

    best_params = space_eval(space, best)

    optimal_params_dict = {
        "name": "opt_{}".format(model.name),
        "clf": model.clf.__name__,
        "params": best_params
    }

    with open("{}/opt_{}_model.json".format(params_dir, model.name), "w") as f:
        json.dump(optimal_params_dict, f, cls=JsonEncoder)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


if __name__ == "__main__":
    optimize()
