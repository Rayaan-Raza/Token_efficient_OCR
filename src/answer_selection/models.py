"""Learned scoring models for RAVEN-Select (regression + LambdaRank)."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

_REG_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}

_RANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
    "label_gain": list(range(32)),
}


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def fit_logistic_best(X: np.ndarray, y_best: np.ndarray):
    """Binary classifier: is this the best route among siblings (label 1/0)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])
    model.fit(X, y_best)
    return model


def fit_lgbm_regression(X: np.ndarray, y: np.ndarray, num_boost_round: int = 300):
    dtrain = lgb.Dataset(X, label=y)
    return lgb.train(_REG_PARAMS, dtrain, num_boost_round=num_boost_round)


def fit_lgbm_lambdarank(
    X: np.ndarray,
    y: np.ndarray,
    group: list[int],
    num_boost_round: int = 200,
):
    """LambdaRank with relevance labels derived from ANLS bins."""
    # Map continuous ANLS to integer relevance 0..31
    rel = np.clip(np.round(y * 31).astype(int), 0, 31)
    dtrain = lgb.Dataset(X, label=rel, group=group)
    return lgb.train(_RANK_PARAMS, dtrain, num_boost_round=num_boost_round)


def predict_model(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)
