"""
baselines.py
============

Non-personalized baseline: recommends the most popular items to everyone.

In a recommender system you ALWAYS need to compare against this baseline. If
your "ML" model does not clearly beat weighted popularity, it is not adding
value: the user would be just as happy looking at "the most rated movies on
the site".

Formula (bayesian_avg, IMDb Top 250 style):

    score(movie) = (avg_rating * n_ratings + global_mean * m) / (n_ratings + m)

where `m` is a smoothing parameter. Prevents a movie with two 5-star ratings
from jumping above a movie with 200 ratings of 4.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityBaseline:
    """
    Ranks movies by weighted popularity. Non-personalized: every user receives
    the same ranking.

    strategy:
        - 'count'         : score = movie_count (most rated first).
        - 'bayesian_avg'  : score = (avg * n + global_mean * m) / (n + m).
                            Smooths movies with few ratings towards the global
                            mean. Same formula as IMDb Top 250, useful for
                            cold start.
    """
    def __init__(self, task: str = "classification",
                 strategy: str = "bayesian_avg", m: float = 10.0):
        self.task = task
        self.strategy = strategy
        self.m = m
        self.global_mean_ = 3.5

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # In classification we store the liked-rate rescaled to [0, 5] so it
        # matches the scale of `movie_avg`. In regression we use the raw mean.
        if self.task == "regression":
            self.global_mean_ = float(np.mean(y))
        else:
            self.global_mean_ = float(y.mean() * 5.0)
        return self

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        if self.strategy == "count":
            return X["movie_count"].values.astype(float)

        # bayesian_avg
        count = X["movie_count"].values.astype(float)
        avg = X["movie_avg"].values.astype(float)
        return (avg * count + self.global_mean_ * self.m) / (count + self.m)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._score(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._score(X)
        # Min-max to [0, 1] so we expose a predict_proba-like interface
        lo, hi = scores.min(), scores.max()
        p = (scores - lo) / (hi - lo + 1e-9)
        return np.column_stack([1.0 - p, p])
