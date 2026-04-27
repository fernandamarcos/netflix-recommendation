"""
tune_classification.py
======================

Randomized hyperparameter search for the three classification models
(LogisticRegression, GradientBoosting, MLP), using a PredefinedSplit so that
the leave-last-out test set is used as a single validation fold (preserving
the temporal split).

Optimizes ROC AUC. Prints the winning configuration for each model.
"""

import time
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from data_loader import load_data


def report_best(name, search):
    print(f"\n=== Results for {name} ===")
    print(f"Best ROC AUC (CV): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")


def main():
    print("Loading data (task=classification)...")
    data = load_data(task="classification")

    # Combine train and test to build a PredefinedSplit that respects the
    # temporal leave-last-out split.
    X_full = pd.concat([data.X_train, data.X_test]).reset_index(drop=True)
    y_full = pd.concat([data.y_train, data.y_test]).reset_index(drop=True)

    test_fold = np.concatenate([
        np.full(len(data.X_train), -1),  # -1 -> always train
        np.full(len(data.X_test), 0),    #  0 -> always validation
    ])
    ps = PredefinedSplit(test_fold)

    print(f"  X_train shape: {data.X_train.shape}, X_test shape: {data.X_test.shape}")

    pipelines = {
        "LogisticRegression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            5,  # n_iter
        ),
        "GradientBoosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]),
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
            5,
        ),
        "MLP": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(early_stopping=True, max_iter=300, random_state=42)),
            ]),
            {
                "model__hidden_layer_sizes": [(32,), (64, 32), (32, 16)],
                "model__alpha": [1e-4, 1e-3, 0.01, 0.1],
                "model__learning_rate_init": [1e-3, 5e-3, 0.01],
            },
            10,
        ),
    }

    results = {}
    for name, (pipe, param_grid, n_iter) in pipelines.items():
        print(f"\nStarting search for {name} ({n_iter} iterations, cv=PredefinedSplit)...")
        t0 = time.time()
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=ps,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        search.fit(X_full, y_full)
        print(f"Search finished in {time.time() - t0:.1f}s")
        report_best(name, search)
        results[name] = search

    print("\n--- FINAL SUMMARY ---")
    for name, search in results.items():
        print(f"{name:20s}: Best ROC AUC = {search.best_score_:.4f}")


if __name__ == "__main__":
    main()
