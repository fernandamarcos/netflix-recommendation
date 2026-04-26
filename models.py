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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler

from baselines import PopularityBaseline


# =========================
# NMF puro como ranker
# =========================
class NMFRanker:
    """
    Rankea directamente por el score reconstruido por NMF (columna 'nmf_score'
    del feature set generado por data_loader). No es un modelo de ML. 
    Simplemente toma el 'nmf_score' precalculado y lo devuelve. 
    Permite comparar el performance puro de la factorizacion.
    """
    def __init__(self, task: str = "classification"):
        self.task = task
        self.scaler = LinearRegression()

    def fit(self, X, y):
        self.scaler.fit(X[["nmf_score"]], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_scores = X[["nmf_score"]]
        if self.task == "regression":
            return self.scaler.predict(raw_scores).flatten()
        return raw_scores.values.flatten()

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
        ("model", LogisticRegression(max_iter=1000, C=0.01)),
    ])


def _linear_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1000.0)),
    ])


def _gbm_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        # el boosting no necesita StandardScaler, lo omito
        ("model", GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            random_state=42,
        )),
    ])


def _gbm_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            min_samples_split=20,
            random_state=42,
        )),
    ])


def _mlp_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.1,           # L2
            learning_rate_init=0.01,
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
            alpha=0.1,
            learning_rate_init=0.005,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )),
    ])


def _select_stacked_features(X: pd.DataFrame) -> pd.DataFrame:
    features = ["nmf_score", "user_sim_score"]
    return X[features]


def _stacked_mlp_regressor() -> Pipeline:
    return Pipeline([
        ("selector", FunctionTransformer(_select_stacked_features, validate=False)),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
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
            "NMF + NN (stacked)": _stacked_mlp_regressor(),
        }
    else:
        raise ValueError(f"task desconocida: {task!r}")
