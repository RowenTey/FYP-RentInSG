from typing import Dict, Any

RANDOM_FOREST = {
    'n_estimators': list(range(200, 2001, 200)),
    'max_features': ['auto', 'sqrt'],
    'max_depth': list(range(10, 111, 10)) + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

CATBOOST = {
    "iterations": [1000],
    "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1],
    "depth": [6, 7, 8, 9, 10],
    "subsample": [0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
    "colsample_bylevel": [0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
    "min_data_in_leaf": [20, 40, 60, 80, 100],
}

DECISION_TREE = {
    "max_depth": [3, 5, 7, 10, 15],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 3, 5, 10]
}

LASSO = RIDGE = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
}

HGBR = {
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.5, 1.0],
    "min_samples_leaf": [1, 3, 5, 10]
}

XGBOOST = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "learning_rate": [0.01, 0.1],
    "min_child_weight": [1, 2, 3],
    "gamma": [0, 0.1, 0.2],
}

LGBM_PARAMS = {
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.5, 1.0],
    "n_estimators": [50, 100, 200, 500],
    "min_child_samples": [5, 10, 20, 50]
}

HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    "random_forest": RANDOM_FOREST,
    "catboost": CATBOOST,
    "decision_tree": DECISION_TREE,
    "lasso": LASSO,
    "ridge": RIDGE,
    "hgb": HGBR,
    "xgb": XGBOOST,
    "lgbm": LGBM_PARAMS
}
