from hyperopt import hp
import numpy as np

random_state = 0

rf_space = {
    "n_estimators": hp.quniform("n_estimators", 1, 400, 1),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.01),
    "max_depth": hp.choice("max_depth", np.arange(1, 8, dtype=int)),
    "max_leaf_nodes": hp.choice("max_leaf_nodes", np.arange(2, 10, dtype=int)),
    "ccp_alpha": hp.quniform("ccp_alpha", 0, 0.01, 0.001),
    "max_samples": hp.quniform("max_samples", 0.1, 1, 0.01),
    "n_jobs": -1,
    "warm_start": False,
    "class_weight": "balanced",
    "verbose": 0
}

xgb_space = {
    "n_estimators": hp.quniform("n_estimators", 1, 400, 1),
    "eta": hp.quniform("eta", 0.001, 0.5, 0.001),
    "max_depth":  hp.choice("max_depth", np.arange(1, 8, dtype=int)),
    "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
    "subsample": hp.quniform("subsample", 0.1, 1, 0.01),
    "gamma": hp.quniform("gamma", 0.1, 1, 0.01),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.1, 1, 0.05),
    "eval_metric": "auc",
    "objective": "binary:logistic",
    "class_weight": "balanced",
    "booster": "gbtree",
    "tree_method": "exact",
    "silent": 1,
    "job": -1,
    "seed": random_state
}
