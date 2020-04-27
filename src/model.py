from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import inspect
import json


class Model(object):

    def __init__(self,
                 name: str,
                 clf: str,
                 params: dict = {}):

        self.name = name
        self.params = params
        self.clf = clf
        self.fitted_model = None

    @staticmethod
    def get_model_from_json(model_json_filepath: str):

        with open(model_json_filepath, "r") as f:
            model_dict = json.loads(f.read())

        model = Model.get_model_from_dict(model_dict)

        return model

    @staticmethod
    def get_model_from_dict(model_dict: dict):

        name = model_dict["name"]
        clf = model_dict["clf"]
        params = model_dict["params"]

        if clf == "XGBClassifier":
            clf = XGBClassifier
        elif clf == "RandomForestClassifier":
            clf = RandomForestClassifier

        return Model(name=name,
                     clf=clf,
                     params=params)

    def fit(self, X, y, eval_set=[]):

        args = {}
        args["X"] = X
        args["y"] = y

        model = self.clf(**self.params)

        if "eval_set" in inspect.getargspec(model.fit).args:
            args.update({"eval_set": eval_set})
        print(inspect.getargspec(model.fit).args)
        if "eval_metric" in inspect.getargspec(model.fit).args:
            args.update({"eval_metric": ["auc"]})
        if "scale_pos_weight" in inspect.getargspec(model.fit).args:
            scale_pos_weight = X.shape[0] / y.shape[0]
            model.params.update({"scale_pos_weight": scale_pos_weight})
        if "random_state" in dir(model):
            model.random_state = 0
        if "n_estimators" in dir(model) and \
           "n_estimators" in self.params:
            model.n_estimators = int(self.params["n_estimators"])
        if "max_samples" in dir(model) and \
           "max_samples" in self.params and \
           self.params["max_samples"] == 1.0:
            model.max_samples = 1

        model = model.fit(**args)

        AUCs = []
        for i, eval_tuple in enumerate(eval_set):
            y_pred = model.predict_proba(eval_tuple[0])
            AUC = roc_auc_score(y_true=eval_tuple[1],
                                y_score=y_pred[:, 1])
            AUCs.append(AUC)

        print("AUCs: {}".format(AUCs))

        self.fitted_model = model

        return model, AUCs

    def predict(self, X):
        return self.fitted_model.predict(X)

    def predict_proba(self, X):
        return self.fitted_model.predict_proba(X)
