"""
models.py
=========

Central registry of models to compare. Each model exposes the sklearn API
(fit / predict / predict_proba) so it works with run_baseline.py and with
evaluation.evaluate_ranker.

Models included:
    - PopularityBaseline     (in baselines.py)
    - NMFRanker              : ranks directly by the NMF reconstructed score
    - LogisticRegression     : linear baseline
    - GradientBoostingClf    : non-linear boosted trees
    - MLP                    : feedforward neural network

Notes on the MLP
----------------
MLPs do not magically beat boosted trees on tabular data. What does help:
    1. Standardize features (StandardScaler) — fundamental.
    2. Moderate architecture (64 -> 32), not overly large.
    3. Early stopping to avoid overfitting.
    4. alpha > 0 for L2 regularization.
    5. Include useful NON-LINEAR features: interactions, differences
       (we already have them: interaction, abs_diff, user_sim_score, nmf_score).
A real next-level upgrade would be a "two-tower" architecture (user and
movie embeddings learned end-to-end), but that requires keras/pytorch.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from baselines import PopularityBaseline


# =========================
# Pure NMF as a ranker
# =========================
# =========================
# Pure NMF as a ranker
# =========================
class NMFRanker:
    """
    Uses the precomputed NMF reconstructed score.

    For regression, it calibrates nmf_score to the rating scale using a
    LinearRegression fitted only on the training set.

    For classification/ranking, it still uses the calibrated score to rank
    movies and convert scores into probability-like values.
    """
    def __init__(self, task: str = "classification"):
        self.task = task
        self.calibrator = LinearRegression()

    def fit(self, X, y):
        # Learn mapping from raw nmf_score to real rating using train only
        self.calibrator.fit(X[["nmf_score"]], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Predict calibrated rating-like scores
        scores = self.calibrator.predict(X[["nmf_score"]])

        # Keep predictions in valid rating range
        scores = np.clip(scores, 0.0, 5.0)

        return scores

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Convert calibrated score [0, 5] into probability-like value [0, 1]
        scores = self.predict(X)
        p = np.clip(scores / 5.0, 0.0, 1.0)

        return np.column_stack([1.0 - p, p])
# =========================
# sklearn factories
# =========================
def _logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def _linear_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def _gbm_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        # Boosted trees do not need StandardScaler; we omit it
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])


def _gbm_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])


def _mlp_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,           # L2
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )),
    ])


def _mlp_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )),
    ])


# =========================
# Public registry
# =========================
def build_models(task: str) -> Dict[str, object]:
    """
    Returns an ordered dict {name: untrained_model} with every candidate to
    compare for a given task.
    """
    if task == "classification":
        return {
            "Popularity": PopularityBaseline(task="classification", strategy="bayesian_avg"),
            "NMF Ranker": NMFRanker(task="classification"),
            "LogisticRegression": _logistic_pipeline(),
            "GradientBoosting": _gbm_classifier(),
            "MLP": _mlp_classifier(),
        }
    elif task == "regression":
        return {
            "Popularity": PopularityBaseline(task="regression", strategy="bayesian_avg"),
            "NMF Ranker": NMFRanker(task="regression"),
            "LinearRegression": _linear_pipeline(),
            "GradientBoosting": _gbm_regressor(),
            "MLP": _mlp_regressor(),
        }
    else:
        raise ValueError(f"unknown task: {task!r}")
