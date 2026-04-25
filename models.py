"""
models.py
=========

Registro central de modelos a comparar. Cada modelo expone la API de sklearn
(fit / predict / predict_proba) para que funcione con run_baseline.py y con
evaluation.evaluate_ranker.

Modelos incluidos:
    - PopularityBaseline     (en baselines.py)
    - NMFRanker              : usa directamente el score reconstruido por NMF
    - LogisticRegression     : baseline lineal
    - GradientBoostingClf    : arbol de decision con boosting (no lineal)
    - MLP                    : red neuronal feedforward

Notas sobre la MLP
------------------
Las MLP en tabular NO ganan por magia a los boostings. Lo que si ayuda:
    1. Estandarizar features (StandardScaler) — fundamental.
    2. Arquitectura moderada (64 -> 32) no muy grande.
    3. Early stopping para evitar sobreajuste.
    4. alpha > 0 para regularizacion L2.
    5. Incluir features NO LINEALES utiles: interacciones, diferencias
       (ya las tenemos: interaction, abs_diff, user_sim_score, nmf_score).
Si quieres mas adelante subir de nivel, el salto real llega con una red
tipo "two-tower" (embeddings de user y movie aprendidos end-to-end), pero
eso requiere keras/pytorch.
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
from sklearn.preprocessing import StandardScaler

from baselines import PopularityBaseline


# =========================
# NMF puro como ranker
# =========================
class NMFRanker:
    """
    Rankea directamente por el score reconstruido por NMF (columna 'nmf_score'
    del feature set generado por data_loader). NO entrena nada — es un lector
    de una feature ya computada.

    Util para ver cuanta senal trae por si solo el NMF.
    """
    def __init__(self, task: str = "classification"):
        self.task = task

    def fit(self, X, y):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["nmf_score"].values.astype(float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Convertimos score (rango aprox 0.5–5.0) a probabilidad [0,1]
        scores = self.predict(X)
        p = np.clip((scores - 0.5) / 4.5, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


# =========================
# Factorias sklearn
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
        # el boosting no necesita StandardScaler, lo omito
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
# Registro publico
# =========================
def build_models(task: str) -> Dict[str, object]:
    """
    Devuelve un dict ordenado {nombre: modelo_sin_entrenar} con todos los
    candidatos a comparar para una tarea dada.
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
        raise ValueError(f"task desconocida: {task!r}")
