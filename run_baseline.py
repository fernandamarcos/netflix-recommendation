"""
run_baseline.py
===============

End-to-end pipeline: loads data, trains and evaluates ALL models defined in
models.build_models(). Compares the classic metric (AUC or RMSE) against
ranking metrics (P@K, R@K, NDCG@K, Hit@K, MAP@K).

Usage:
    python run_baseline.py
    python run_baseline.py --task regression
    python run_baseline.py --k 20 --max-users 200
    python run_baseline.py --candidates unseen --max-users 50
    python run_baseline.py --only "Popularity,NMF Ranker"  (filter by name)
"""

import argparse
import time

import numpy as np

from sklearn.metrics import (
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)

from data_loader import load_data
from evaluation import evaluate_ranker, pretty_print_metrics
from models import build_models


def evaluate_legacy(name, task, model, data):
    if task == "classification":
        probs = model.predict_proba(data.X_test)[:, 1]
        auc = roc_auc_score(data.y_test, probs)
        print(f"  {name:22s}  AUC = {auc:.4f}")
        return {"auc": auc}
    else:
        preds = model.predict(data.X_test)
        rmse = np.sqrt(mean_squared_error(data.y_test, preds))
        mae = mean_absolute_error(data.y_test, preds)
        r2 = r2_score(data.y_test, preds)
        print(f"  {name:22s}  RMSE={rmse:.4f}   MAE={mae:.4f}   R2={r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=100,
                        help="Number of users for ranking. Use -1 for all.")
    parser.add_argument("--candidates", choices=["unseen", "test_plus_sample"],
                        default="test_plus_sample")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of model names to run. "
                             "Example: 'Popularity,NMF Ranker'")
    args = parser.parse_args()
    max_users = None if args.max_users is not None and args.max_users < 0 else args.max_users

    # 1. Load
    t0 = time.time()
    print(f"Loading data (task={args.task})...")
    data = load_data(task=args.task)
    print(f"  done in {time.time() - t0:.1f}s. "
          f"X_train={data.X_train.shape}, X_test={data.X_test.shape}, "
          f"features={len(data.feature_columns)}")

    # 2. Models
    models = build_models(args.task)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        models = {k: v for k, v in models.items() if k in wanted}
        print(f"  Filtered models: {list(models.keys())}")

    # 3. Training
    print(f"\nTraining models...")
    trained = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(data.X_train, data.y_train)
        trained[name] = model
        print(f"  {name:22s}  trained in {time.time() - t0:.1f}s")

    # 4. Legacy metrics
    print(f"\n=== Legacy metric ({args.task}) ===")
    legacy = {}
    for name, model in trained.items():
        legacy[name] = evaluate_legacy(name, args.task, model, data)

    # 5. Ranking
    print(f"\n=== Ranking top-{args.k}  "
          f"(max_users={max_users}, candidates={args.candidates}) ===")
    results = {}
    for name, model in trained.items():
        t0 = time.time()
        metrics = evaluate_ranker(
            model, data,
            k=args.k,
            max_users=max_users,
            candidate_strategy=args.candidates,
            verbose=False,
        )
        metrics["_eval_seconds"] = round(time.time() - t0, 1)
        results[name] = metrics
        pretty_print_metrics(name, metrics)

    # 6. Compact summary
    print(f"\n=== Ranking summary top-{args.k} ===")
    header = f"{'Model':22s}  {'P@K':>7s}  {'R@K':>7s}  {'NDCG':>7s}  {'Hit':>7s}  {'MAP':>7s}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:22s}  "
              f"{m[f'precision@{args.k}']:7.4f}  "
              f"{m[f'recall@{args.k}']:7.4f}  "
              f"{m[f'ndcg@{args.k}']:7.4f}  "
              f"{m[f'hit@{args.k}']:7.4f}  "
              f"{m[f'map@{args.k}']:7.4f}")


if __name__ == "__main__":
    main()
