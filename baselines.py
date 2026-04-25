"""
baselines.py
============

Baseline sin personalizacion: recomienda lo mas popular a todo el mundo.

En un sistema de recomendacion SIEMPRE hay que comparar contra este baseline.
Si tu modelo "con ML" no bate claramente a la popularidad ponderada, no
sirve: el usuario estaria igual de contento viendo "las peliculas mas votadas
de la web".

Formula (bayesian_avg, estilo IMDb Top 250):

    score(peli) = (avg_rating * n_ratings + global_mean * m) / (n_ratings + m)

donde `m` es un parametro de suavizado. Evita que una peli con 2 ratings de
5 estrellas se pase a la cima por encima de una peli con 200 ratings de 4.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityBaseline:
    """
    Rankea peliculas por popularidad ponderada. No personalizado: a todos los
    usuarios les recomienda las mismas peliculas.

    strategy:
        - 'count'         : score = movie_count (la mas vista primero).
        - 'bayesian_avg'  : score = (avg * n + global_mean * m) / (n + m).
                            Suaviza pelis con pocos ratings hacia la media
                            global. Es la formula de IMDb Top 250, util para
                            cold start.
    """
    def __init__(self, task: str = "classification",
                 strategy: str = "bayesian_avg", m: float = 10.0):
        self.task = task
        self.strategy = strategy
        self.m = m
        self.global_mean_ = 3.5

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # En classification guardamos el liked-rate re-escalado a [0, 5] para que
        # cuadre con la escala de `movie_avg`. En regression usamos la media
        # cruda.
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
        # Min-max a [0, 1] para exponer predict_proba-like
        lo, hi = scores.min(), scores.max()
        p = (scores - lo) / (hi - lo + 1e-9)
        return np.column_stack([1.0 - p, p])
