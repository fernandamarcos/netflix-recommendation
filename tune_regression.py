import time
import argparse
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from data_loader import load_data

def report_best(name, search):
    print(f"\n=== Resultados {name} ===")
    # Randomized Search maximiza el score, como usamos neg_root_mean_squared_error, es negativo.
    print(f"Mejor RMSE CV: {-search.best_score_:.4f}")
    print(f"Mejores parámetros: {search.best_params_}")

def main():
    print("Cargando datos (task=regression)...")
    data = load_data(task="regression")
    X_train, y_train = data.X_train, data.y_train

    print(f"  X_train shape: {X_train.shape}")
    
    cv_folds = 3
    
    pipelines = {
        "Ridge": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ]),
            {
                "model__alpha": [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
            },
            6 # iterations
        ),
        "GradientBoosting": (
        Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", GradientBoostingRegressor(random_state=42))
        ]),
        {
        "model__n_estimators": [50, 100],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__min_samples_split": [10, 20]
        },
        5
        ),
        "MLP": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(early_stopping=True, max_iter=200, random_state=42))
            ]),
            {
                "model__hidden_layer_sizes": [(32,), (64, 32), (32, 16)],
                "model__alpha": [1e-4, 1e-3, 0.01, 0.1],
                "model__learning_rate_init": [1e-3, 5e-3, 0.01]
            },
            10 # iterations
        )
    }

    results = {}
    for name, (pipe, param_grid, n_iter) in pipelines.items():
        print(f"\nIniciando búsqueda para {name} ({n_iter} iteraciones, cv={cv_folds})...")
        t0 = time.time()
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv_folds,
            n_jobs=-1, # Usar todos los procesadores para ir más rápido
            verbose=1,
            random_state=42
        )
        search.fit(X_train, y_train)
        print(f"Búsqueda finalizada en {time.time() - t0:.1f}s")
        report_best(name, search)
        results[name] = search
        
    print("\n--- RESUMEN FINAL ---")
    for name, search in results.items():
        print(f"{name:20s}: Mejor RMSE = {-search.best_score_:.4f}")

if __name__ == "__main__":
    main()
