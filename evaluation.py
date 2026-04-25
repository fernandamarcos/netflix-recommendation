"""
evaluation.py
=============

Metricas de ranking para sistemas de recomendacion:
    - precision_at_k
    - recall_at_k
    - hit_rate_at_k
    - ndcg_at_k  (con relevancia graduada usando el rating real)
    - map_at_k

Y un wrapper `evaluate_ranker` que aplica todas estas metricas a un modelo
ajustado, por usuario, y devuelve los promedios.

Filosofia
---------
Un recomendador bueno coloca los items que el usuario QUIERE VER en la cima
de la lista. AUC o RMSE no capturan esto; necesitas metricas de top-K. Por
eso existe este modulo.

Convencion
----------
- `true_items`: iterable de movie_ids que son relevantes para el usuario
  (p.ej. los que rato con >= 4 en test, o simplemente los que aparecen en
  test si lo tratamos como implicit).
- `pred_items`: lista ordenada (mayor a menor score) de movie_ids
  recomendados por el modelo.
- `true_relevance`: dict opcional {movie_id: rating} si quieres que NDCG
  use relevancia graduada; si no se pasa, se usa relevancia binaria (0/1).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================
# Metricas puntuales por usuario
# =========================
def precision_at_k(true_items: Iterable[int], pred_items: Sequence[int], k: int) -> float:
    if k <= 0 or len(pred_items) == 0:
        return 0.0
    top_k = pred_items[:k]
    true_set = set(true_items)
    hits = sum(1 for item in top_k if item in true_set)
    return hits / k


def recall_at_k(true_items: Iterable[int], pred_items: Sequence[int], k: int) -> float:
    true_set = set(true_items)
    if len(true_set) == 0:
        return 0.0
    top_k = pred_items[:k]
    hits = sum(1 for item in top_k if item in true_set)
    return hits / len(true_set)


def hit_rate_at_k(true_items: Iterable[int], pred_items: Sequence[int], k: int) -> float:
    """1 si al menos uno de los true aparece en top-K, 0 si no."""
    true_set = set(true_items)
    return float(any(item in true_set for item in pred_items[:k]))


def average_precision_at_k(
    true_items: Iterable[int], pred_items: Sequence[int], k: int
) -> float:
    """AP@K usado para MAP."""
    true_set = set(true_items)
    if len(true_set) == 0:
        return 0.0
    score = 0.0
    hits = 0
    for i, item in enumerate(pred_items[:k], start=1):
        if item in true_set:
            hits += 1
            score += hits / i
    return score / min(len(true_set), k)


def ndcg_at_k(
    pred_items: Sequence[int],
    k: int,
    true_relevance: Optional[Dict[int, float]] = None,
    true_items: Optional[Iterable[int]] = None,
) -> float:
    """
    NDCG@K con relevancia graduada si se pasa true_relevance, binaria si no.

    Formula: DCG = sum_i (2^rel_i - 1) / log2(i + 1)
             NDCG = DCG / IDCG
    """
    if true_relevance is None:
        true_set = set(true_items) if true_items is not None else set()
        true_relevance = {item: 1.0 for item in true_set}

    if not true_relevance:
        return 0.0

    def dcg(relevances: List[float]) -> float:
        return sum(
            (2.0 ** rel - 1.0) / np.log2(idx + 2)
            for idx, rel in enumerate(relevances)
        )

    gains = [true_relevance.get(item, 0.0) for item in pred_items[:k]]
    ideal_gains = sorted(true_relevance.values(), reverse=True)[:k]

    idcg = dcg(ideal_gains)
    if idcg == 0.0:
        return 0.0
    return dcg(gains) / idcg


# =========================
# Wrapper a nivel de modelo
# =========================
def _rank_candidates_for_user(
    user_id: int,
    candidate_movie_ids: np.ndarray,
    model,
    build_features_fn,
    feature_columns: List[str],
    task: str,
) -> List[int]:
    """
    Devuelve la lista de candidate_movie_ids ordenada por score descendente segun
    el modelo.
    """
    candidates_df = build_features_fn(user_id=user_id, candidate_movie_ids=candidate_movie_ids)
    X = candidates_df[feature_columns]

    if task == "classification" and hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    order = np.argsort(-scores)
    ranked = candidates_df.iloc[order]["movie_id"].tolist()
    return ranked


def evaluate_ranker(
    model,
    data,
    k: int = 10,
    liked_threshold: float = 4.0,
    max_users: Optional[int] = None,
    candidate_strategy: str = "unseen",
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evalua un modelo sklearn-like sobre data.test_ratings usando metricas de ranking.

    Parametros
    ----------
    model : modelo ajustado con .predict o .predict_proba
    data : RecommenderData (salida de data_loader.load_data)
    k : top-K para las metricas
    liked_threshold : umbral para considerar un rating real como "relevante"
    max_users : si no es None, evalua solo los primeros `max_users` usuarios
                (util para iterar rapido)
    candidate_strategy : 'unseen' -> todas las pelis no vistas en train
                         'test_plus_sample' -> la peli de test + 99 negativas
                         (estandar leave-one-out con muestreo negativo)
    verbose : imprime progreso cada 50 usuarios

    Devuelve
    --------
    dict con 'precision@k', 'recall@k', 'ndcg@k', 'hit@k', 'map@k'
    """
    from data_loader import build_candidate_features

    test = data.test_ratings
    relevant_mask = test["rating"] >= liked_threshold
    relevant_test = test[relevant_mask]

    users = relevant_test["user_id"].unique()
    if max_users is not None:
        users = users[:max_users]

    rng = np.random.default_rng(42)
    all_movie_ids = data.movies["movie_id"].unique()

    precisions, recalls, ndcgs, hits, maps = [], [], [], [], []

    for i, user_id in enumerate(users):
        true_user = test[test["user_id"] == user_id]
        true_liked = true_user[true_user["rating"] >= liked_threshold]
        if len(true_liked) == 0:
            continue

        true_items = true_liked["movie_id"].tolist()
        true_relevance = dict(zip(true_liked["movie_id"], true_liked["rating"]))

        # Construir candidatos
        if candidate_strategy == "test_plus_sample":
            seen = data.train_ratings.loc[
                data.train_ratings["user_id"] == user_id, "movie_id"
            ].unique()
            pool = np.setdiff1d(all_movie_ids, np.union1d(seen, true_items))
            n_neg = min(99, len(pool))
            negatives = rng.choice(pool, size=n_neg, replace=False)
            candidates = np.concatenate([np.array(true_items), negatives])
        else:  # 'unseen'
            seen = data.train_ratings.loc[
                data.train_ratings["user_id"] == user_id, "movie_id"
            ].unique()
            candidates = np.setdiff1d(all_movie_ids, seen)

        def _build(user_id=user_id, candidate_movie_ids=candidates):
            return build_candidate_features(
                data, user_id=user_id, candidate_movie_ids=candidate_movie_ids
            )

        ranked = _rank_candidates_for_user(
            user_id=user_id,
            candidate_movie_ids=candidates,
            model=model,
            build_features_fn=_build,
            feature_columns=data.feature_columns,
            task=data.task,
        )

        precisions.append(precision_at_k(true_items, ranked, k))
        recalls.append(recall_at_k(true_items, ranked, k))
        ndcgs.append(ndcg_at_k(ranked, k, true_relevance=true_relevance))
        hits.append(hit_rate_at_k(true_items, ranked, k))
        maps.append(average_precision_at_k(true_items, ranked, k))

        if verbose and (i + 1) % 50 == 0:
            print(f"  [eval] {i + 1}/{len(users)} usuarios procesados")

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit@{k}": float(np.mean(hits)) if hits else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        "n_users_evaluated": len(precisions),
    }


def pretty_print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {name} ===")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:>18s} : {val:.4f}")
        else:
            print(f"  {key:>18s} : {val}")


if __name__ == "__main__":
    # Smoke test de las metricas puras
    true = [10, 20, 30]
    pred = [10, 5, 20, 40, 30, 50, 60, 70, 80, 90]
    print("P@5   =", precision_at_k(true, pred, 5))
    print("R@5   =", recall_at_k(true, pred, 5))
    print("Hit@5 =", hit_rate_at_k(true, pred, 5))
    print("MAP@5 =", average_precision_at_k(true, pred, 5))
    print("NDCG@5=", ndcg_at_k(pred, 5, true_items=true))
