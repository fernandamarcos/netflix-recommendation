"""
data_loader.py
==============

Loads and preprocesses the MovieLens (ml-latest-small) dataset consistently for
classification ('liked' = rating >= 4) and regression (rating 0-5) tasks.

Basic usage:
    from data_loader import load_data

    data = load_data(task="classification")   # or "regression"

    data.X_train, data.X_test, data.y_train, data.y_test
    data.train_ratings, data.test_ratings     # raw DataFrames with user_id, movie_id, rating, timestamp
    data.movies                               # movie catalog with genre dummies
    data.feature_columns                      # list of columns in X_*
    data.genre_columns                        # subset of feature_columns that are genres

Design choices:
    * Leave-last-out split per user (each user's most recent rating goes to test).
      Avoids the temporal leakage you'd get from a random train_test_split.
    * All aggregate stats (user_avg, movie_avg, global_mean) are computed
      using train ONLY, never test.
    * user_sim_score is vectorized (similarity matrix @ rating matrix).
      Much faster than applying row by row.
    * 'year' feature extracted from the title.
    * Configurable: 'liked' threshold, neighbor k, task, etc.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import numpy as np
import pandas as pd

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # fallback without external dependency
    def cosine_similarity(X):
        X = np.asarray(X, dtype=float)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        return Xn @ Xn.T

try:
    from sklearn.decomposition import NMF as _SklearnNMF
    _HAS_NMF = True
except ImportError:
    _HAS_NMF = False


# =========================
# Types / constants
# =========================
Task = Literal["classification", "regression"]

DEFAULT_DATA_DIR = "ml-latest-small"
LIKED_THRESHOLD_DEFAULT = 4.0
TOP_K_SIMILAR_USERS_DEFAULT = 10
NMF_N_COMPONENTS_DEFAULT = 50


@dataclass
class RecommenderData:
    """Container with everything needed to train and evaluate."""
    task: Task

    # Final matrices ready for sklearn
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    # Raw DataFrames with user_id, movie_id, rating, timestamp (and derived columns).
    # Useful when computing per-user ranking metrics.
    train_ratings: pd.DataFrame
    test_ratings: pd.DataFrame

    # Movie catalog with genre dummies and 'year'
    movies: pd.DataFrame

    feature_columns: List[str]
    genre_columns: List[str]

    # Intermediate objects useful for scoring unseen items
    user_stats: pd.DataFrame = field(repr=False)
    movie_stats: pd.DataFrame = field(repr=False)
    global_mean: float = 0.0
    # Long-format lookup (user_id, movie_id -> user_sim_score) covering ALL
    # user-movie pairs with a computable score (including movies the user
    # has not seen). Required to rank unseen candidates correctly.
    user_sim_lookup: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    # Long-format lookup of the rating reconstructed by NMF
    # (user_id, movie_id -> nmf_score). Captures latent user taste
    # beyond a movie's average popularity.
    nmf_lookup: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


# =========================
# Raw load and basic preprocessing
# =========================
def _load_raw(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))

    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    movies = movies.rename(columns={"movieId": "movie_id"})

    return ratings, movies


def _extract_year(title: str) -> Optional[int]:
    """MovieLens places the year in parentheses at the end of the title: 'Toy Story (1995)'."""
    if not isinstance(title, str):
        return None
    match = re.search(r"\((\d{4})\)\s*$", title.strip())
    return int(match.group(1)) if match else None


def _build_movies_with_features(movies: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Adds genre dummies and 'year' to movies. Returns (movies_with_features, list_of_genre_columns)."""
    m = movies.copy()
    m["genres"] = m["genres"].fillna("")

    genre_dummies = m["genres"].str.get_dummies(sep="|")
    if "(no genres listed)" in genre_dummies.columns:
        genre_dummies = genre_dummies.drop(columns=["(no genres listed)"])
    genre_dummies.columns = [f"genre_{c}" for c in genre_dummies.columns]

    m = pd.concat([m, genre_dummies], axis=1)
    m["year"] = m["title"].apply(_extract_year)
    # Impute missing year with the median in case the title has no year tag
    m["year"] = m["year"].fillna(m["year"].median())

    return m, list(genre_dummies.columns)


# =========================
# Leave-last-out split per user
# =========================
def _leave_last_out_split(
    ratings: pd.DataFrame,
    min_ratings_per_user: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user, the MOST RECENT rating goes to test and the rest to train.
    Standard recommender evaluation, avoids temporal leakage.
    """
    # Drop users that don't have enough ratings
    counts = ratings.groupby("user_id").size()
    valid_users = counts[counts >= min_ratings_per_user].index
    ratings = ratings[ratings["user_id"].isin(valid_users)].copy()

    # Sort by timestamp and pick the last per user
    ratings = ratings.sort_values(["user_id", "timestamp"])
    last_per_user = ratings.groupby("user_id").tail(1).index

    test = ratings.loc[last_per_user].copy()
    train = ratings.drop(last_per_user).copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)


# =========================
# Train-only statistics
# =========================
def _fit_train_stats(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    global_mean = float(train["rating"].mean())

    user_stats = train.groupby("user_id").agg(
        user_avg=("rating", "mean"),
        user_count=("rating", "count"),
    )

    movie_stats = train.groupby("movie_id").agg(
        movie_avg=("rating", "mean"),
        movie_count=("rating", "count"),
    )

    return user_stats, movie_stats, global_mean


# =========================
# User-user similarity (vectorized)
# =========================
def _build_user_sim_scores(
    train: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Computes user_sim_score for EACH (user_id, movie_id) pair observed in
    training and returns it as a long-format DataFrame
    (user_id, movie_id) -> score.

    This map is then used as a lookup when building features. Pairs without
    a computable score (cold start) get NaN and are filled with global_mean.

    Idea:
        Sim_matrix (U x U), Rating_matrix (U x M). For each user u and movie m,
        we want the similarity-weighted average of ratings from u's top_k most
        similar users who have rated m.

        Simplification: instead of strictly using only top_k, we use ALL
        neighbors weighted by similarity and a "has-rated" mask. This is
        much faster and approximates the strict top-k score well (the most
        similar neighbors dominate the sum).
    """
    user_movie = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
    )

    # Rating matrix (0 where no rating) and mask (1 where rating exists)
    R = user_movie.fillna(0.0).values
    M = (~user_movie.isna()).astype(float).values

    # User-user similarity over raw ratings
    sim = cosine_similarity(R)
    # Zero out the diagonal: we don't want a user to weight themselves
    np.fill_diagonal(sim, 0.0)

    # Optional: keep only top_k neighbors per user (closer to the original code).
    # We default to a "soft" version (no top_k cap) because in practice it gives
    # similar results and is simpler. For the strict top_k version, use argsort.
    if top_k is not None and top_k > 0:
        # For each row, set to 0 all similarities except the top_k largest
        n_users = sim.shape[0]
        top_k = min(top_k, n_users - 1)
        # ascending argsort: indices of the smallest values come first
        idx_small = np.argsort(sim, axis=1)[:, : n_users - top_k]
        rows = np.repeat(np.arange(n_users), idx_small.shape[1])
        cols = idx_small.flatten()
        sim[rows, cols] = 0.0

    # Numerator and denominator of the weighted average
    num = sim @ R      # (U x M)
    den = sim @ M      # (U x M) sum of similarities of neighbors who rated each movie
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(den > 0, num / den, np.nan)

    user_sim_df = pd.DataFrame(
        score,
        index=user_movie.index,
        columns=user_movie.columns,
    )

    # Convert to long format for cheap lookup by (user_id, movie_id)
    long = (
        user_sim_df.stack(future_stack=True)
        .dropna()
        .rename("user_sim_score")
        .reset_index()
    )
    return long


# =========================
# NMF latent factors (collaborative filtering)
# =========================
def _build_nmf_features(
    train: pd.DataFrame,
    n_components: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Factorizes the user-movie matrix with NMF and returns a long-format
    lookup (user_id, movie_id -> nmf_score) holding the RECONSTRUCTED rating.

    If sklearn is unavailable, returns an empty DataFrame and the feature
    will be filled with global_mean inside _add_features (pipeline still works).

    Known limitation: missing values are filled with 0, which NMF interprets as
    "the user dislikes". This is the simple approach; a better next step would
    be implicit-feedback ALS or Surprise (SVD), which handle missings properly.
    """
    if not _HAS_NMF:
        return pd.DataFrame(columns=["user_id", "movie_id", "nmf_score"])

    user_movie = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
    )
    mat = user_movie.fillna(0.0).values

    nmf = _SklearnNMF(
        n_components=n_components,
        init="nndsvd",          # deterministic init, converges faster than 'random'
        random_state=random_state,
        max_iter=500,
    )
    W = nmf.fit_transform(mat)
    H = nmf.components_
    pred = W @ H

    pred_df = pd.DataFrame(
        pred,
        index=user_movie.index,
        columns=user_movie.columns,
    )

    long = (
        pred_df.stack(future_stack=True)
        .dropna()
        .rename("nmf_score")
        .reset_index()
    )
    return long


# =========================
# Feature engineering on a frame
# =========================
def _add_features(
    ratings_frame: pd.DataFrame,
    movies_with_features: pd.DataFrame,
    user_stats: pd.DataFrame,
    movie_stats: pd.DataFrame,
    global_mean: float,
    user_sim_lookup: pd.DataFrame,
    genre_columns: List[str],
    nmf_lookup: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = ratings_frame.copy()

    df = df.merge(user_stats, on="user_id", how="left")
    df = df.merge(movie_stats, on="movie_id", how="left")

    movie_features_cols = ["movie_id", "year"] + genre_columns
    df = df.merge(movies_with_features[movie_features_cols], on="movie_id", how="left")

    df = df.merge(user_sim_lookup, on=["user_id", "movie_id"], how="left")

    # NMF feature (optional — only if sklearn is available and it was computed)
    if nmf_lookup is not None and len(nmf_lookup) > 0:
        df = df.merge(nmf_lookup, on=["user_id", "movie_id"], how="left")
    else:
        df["nmf_score"] = np.nan

    # Consistent imputations (cold start)
    df["user_avg"] = df["user_avg"].fillna(global_mean)
    df["movie_avg"] = df["movie_avg"].fillna(global_mean)
    df["user_count"] = df["user_count"].fillna(0)
    df["movie_count"] = df["movie_count"].fillna(0)
    df["user_sim_score"] = df["user_sim_score"].fillna(global_mean)
    df["nmf_score"] = df["nmf_score"].fillna(global_mean)
    df[genre_columns] = df[genre_columns].fillna(0)
    df["year"] = df["year"].fillna(movies_with_features["year"].median())

    # Interaction features
    df["interaction"] = df["user_avg"] * df["movie_avg"]
    df["diff"] = df["user_avg"] - df["movie_avg"]
    df["abs_diff"] = df["diff"].abs()

    return df


# =========================
# Public API
# =========================
def load_data(
    task: Task = "classification",
    data_dir: str = DEFAULT_DATA_DIR,
    liked_threshold: float = LIKED_THRESHOLD_DEFAULT,
    top_k_similar: int = TOP_K_SIMILAR_USERS_DEFAULT,
    min_ratings_per_user: int = 2,
    nmf_components: int = NMF_N_COMPONENTS_DEFAULT,
    save_to: Optional[str] = None,
) -> RecommenderData:
    """
    Loads MovieLens and returns a RecommenderData ready to train.

    Parameters
    ----------
    task : 'classification' or 'regression'
        - classification: y = 1 if rating >= liked_threshold, else 0
        - regression:     y = rating (float 0.5 - 5.0)
    data_dir : path to the folder with ratings.csv and movies.csv
    liked_threshold : 'liked' threshold for classification
    top_k_similar : number of neighbors used in user_sim_score (None or 0 = all)
    min_ratings_per_user : minimum ratings per user to be included
    nmf_components : number of NMF latent factors (set to 0 to disable NMF)
    save_to : if a path is passed, saves X_train, X_test, y_train, y_test as csv.

    Returns
    -------
    RecommenderData
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    # 1. Raw load
    ratings, movies = _load_raw(data_dir)

    # 2. Movie features
    movies_feat, genre_columns = _build_movies_with_features(movies)

    # 3. Leave-last-out split
    train, test = _leave_last_out_split(ratings, min_ratings_per_user=min_ratings_per_user)

    # 4. Train-only stats
    user_stats, movie_stats, global_mean = _fit_train_stats(train)

    # 5. User similarity scores (vectorized) from train
    user_sim_lookup = _build_user_sim_scores(train, top_k=top_k_similar)

    # 5b. NMF latent factors from train
    if nmf_components and nmf_components > 0:
        nmf_lookup = _build_nmf_features(train, n_components=nmf_components)
    else:
        nmf_lookup = pd.DataFrame(columns=["user_id", "movie_id", "nmf_score"])

    # 6. Feature engineering on both frames
    train_fe = _add_features(
        train, movies_feat, user_stats, movie_stats,
        global_mean, user_sim_lookup, genre_columns,
        nmf_lookup=nmf_lookup,
    )
    test_fe = _add_features(
        test, movies_feat, user_stats, movie_stats,
        global_mean, user_sim_lookup, genre_columns,
        nmf_lookup=nmf_lookup,
    )

    # 7. Define feature_columns and target
    base_features = [
        "user_avg", "user_count",
        "movie_avg", "movie_count",
        "interaction", "abs_diff",
        "user_sim_score", "nmf_score", "year",
    ]
    feature_columns = base_features + genre_columns

    if task == "classification":
        y_train = (train_fe["rating"] >= liked_threshold).astype(int)
        y_test = (test_fe["rating"] >= liked_threshold).astype(int)
    else:
        y_train = train_fe["rating"].astype(float)
        y_test = test_fe["rating"].astype(float)

    X_train = train_fe[feature_columns].copy()
    X_test = test_fe[feature_columns].copy()

    # 8. Optional persistence
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        X_train.to_csv(os.path.join(save_to, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(save_to, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(save_to, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(save_to, "y_test.csv"), index=False)

    return RecommenderData(
        task=task,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_ratings=train_fe,
        test_ratings=test_fe,
        movies=movies_feat,
        feature_columns=feature_columns,
        genre_columns=genre_columns,
        user_stats=user_stats,
        movie_stats=movie_stats,
        global_mean=global_mean,
        user_sim_lookup=user_sim_lookup,
        nmf_lookup=nmf_lookup,
    )


# =========================
# Helper to build candidate features (unseen movies)
# =========================
def build_candidate_features(
    data: RecommenderData,
    user_id: int,
    candidate_movie_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Builds X for all (user_id, movie_id) candidate pairs. Useful for producing
    top-K recommendations with any of the trained models.

    If candidate_movie_ids is None, uses all movies the user has not seen
    in train.
    """
    if candidate_movie_ids is None:
        seen = data.train_ratings.loc[
            data.train_ratings["user_id"] == user_id, "movie_id"
        ].unique()
        all_movies = data.movies["movie_id"].unique()
        candidate_movie_ids = np.setdiff1d(all_movies, seen)

    candidates = pd.DataFrame({
        "user_id": user_id,
        "movie_id": candidate_movie_ids,
    })

    # Use the FULL lookups (built in load_data with ALL user-movie pairs that
    # have a computable score, including unseen movies). This is critical for
    # candidate ranking: if we only used train_ratings, user_sim_score and
    # nmf_score would be constant on unseen candidates and the model would
    # lose discriminative power.
    candidates = _add_features(
        candidates,
        data.movies,
        data.user_stats,
        data.movie_stats,
        data.global_mean,
        data.user_sim_lookup,
        data.genre_columns,
        nmf_lookup=data.nmf_lookup,
    )

    return candidates


if __name__ == "__main__":
    # Smoke test
    for t in ("classification", "regression"):
        d = load_data(task=t)
        print(f"=== task = {t} ===")
        print("X_train:", d.X_train.shape, "X_test:", d.X_test.shape)
        print("y_train head:", d.y_train.head().tolist())
        print("features:", d.feature_columns[:8], "...")
        print("n features:", len(d.feature_columns))
        print()
