from pathlib import Path
from model import Model
import pandas as pd
import pickle
import click


project_dir = Path(__file__).resolve().parents[1]
model_dir = "{}/{}".format(project_dir, "models")
params_dir = "{}/{}".format(project_dir, "params")
processed_dir = "{}/{}".format(project_dir, "data/processed")

TARGET_COLUMN = "gender"


@click.command()
@click.option("--model_json")
@click.option("--split_ratio", default=1, type=float)
def train(model_json: str,
          split_ratio: float):
    """
    Train a model given its parameters and train/validation sets

    Parameters:
    model_json (str): Filepath to JSON containing model
    split_ratio (float): Gives split proportion to generate
    train and validation datasets
    """

    dataset_filepath = "{}/{}".format(processed_dir, "features.csv")

    model = Model.get_model_from_json(model_json)
    model_filepath = "{}/{}_model.pkl".format(model_dir, model.name)

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

    train_idxs = [0, int((split_ratio) * df_dataset.shape[0])]
    validation_idxs = [int((split_ratio) * df_dataset.shape[0]), df_dataset.shape[0]]

    X_train = X[train_idxs[0]:train_idxs[1]]
    y_train = Y[train_idxs[0]:train_idxs[1]]
    X_validation = X[validation_idxs[0]:validation_idxs[1]]
    y_validation = Y[validation_idxs[0]:validation_idxs[1]]

    model, AUCs = model.fit(X=X_train,
                            y=y_train,
                            eval_set=[(X_train, y_train),
                                      (X_validation, y_validation)])

    pickle.dump(model, open(model_filepath, "wb"))


if __name__ == "__main__":
    train()
