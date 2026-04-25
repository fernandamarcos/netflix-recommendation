"""
data_loader.py
==============

Carga y preprocesa el dataset MovieLens (ml-latest-small) de forma consistente
para tareas de clasificacion ('liked' = rating >= 4) y de regresion (rating 0-5).

Uso basico:
    from data_loader import load_data

    data = load_data(task="classification")   # o "regression"

    data.X_train, data.X_test, data.y_train, data.y_test
    data.train_ratings, data.test_ratings     # raw DataFrames con user_id, movie_id, rating, timestamp
    data.movies                               # DataFrame de peliculas con generos
    data.feature_columns                      # lista de columnas de X_*
    data.genre_columns                        # subset de feature_columns que son generos

Puntos clave de diseno:
    * Split leave-last-out por usuario (el rating mas reciente de cada usuario va a test).
      Evita fuga temporal que tiene el train_test_split aleatorio.
    * Todas las estadisticas agregadas (user_avg, movie_avg, global_mean) se calculan
      SOLO con train, nunca con test.
    * user_sim_score vectorizado (matriz de similitud @ matriz de ratings). Mucho mas
      rapido que aplicar fila a fila.
    * Feature 'year' extraida del titulo.
    * Configurable: threshold de 'liked', k de vecinos, tarea, etc.
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
except ImportError:  # fallback sin dependencia externa
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
# Tipos / constantes
# =========================
Task = Literal["classification", "regression"]

DEFAULT_DATA_DIR = "ml-latest-small"
LIKED_THRESHOLD_DEFAULT = 4.0
TOP_K_SIMILAR_USERS_DEFAULT = 10
NMF_N_COMPONENTS_DEFAULT = 50


@dataclass
class RecommenderData:
    """Contenedor de todo lo necesario para entrenar y evaluar."""
    task: Task

    # Matrices finales listas para sklearn
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    # Raw DataFrames con user_id, movie_id, rating, timestamp (y columnas derivadas).
    # Utiles para calcular metricas de ranking por usuario.
    train_ratings: pd.DataFrame
    test_ratings: pd.DataFrame

    # Catalogo de peliculas con dummies de generos y year
    movies: pd.DataFrame

    feature_columns: List[str]
    genre_columns: List[str]

    # Objetos intermedios utiles para hacer recomendaciones nuevas (unseen items)
    user_stats: pd.DataFrame = field(repr=False)
    movie_stats: pd.DataFrame = field(repr=False)
    global_mean: float = 0.0
    # Long-format lookup (user_id, movie_id -> user_sim_score) con TODAS las
    # parejas calculables (incluidas peliculas no vistas por el usuario),
    # necesario para rankear candidatos correctamente.
    user_sim_lookup: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    # Long-format lookup del score reconstruido por NMF
    # (user_id, movie_id -> nmf_score). Captura gustos latentes por usuario
    # mas alla de la popularidad media de la pelicula.
    nmf_lookup: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


# =========================
# Carga y preprocesado basico
# =========================
def _load_raw(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))

    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    movies = movies.rename(columns={"movieId": "movie_id"})

    return ratings, movies


def _extract_year(title: str) -> Optional[int]:
    """MovieLens pone el anio entre parentesis al final del titulo: 'Toy Story (1995)'."""
    if not isinstance(title, str):
        return None
    match = re.search(r"\((\d{4})\)\s*$", title.strip())
    return int(match.group(1)) if match else None


def _build_movies_with_features(movies: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Anade dummies de genero y year a movies. Devuelve (movies_con_features, lista_columnas_genero)."""
    m = movies.copy()
    m["genres"] = m["genres"].fillna("")

    genre_dummies = m["genres"].str.get_dummies(sep="|")
    if "(no genres listed)" in genre_dummies.columns:
        genre_dummies = genre_dummies.drop(columns=["(no genres listed)"])
    genre_dummies.columns = [f"genre_{c}" for c in genre_dummies.columns]

    m = pd.concat([m, genre_dummies], axis=1)
    m["year"] = m["title"].apply(_extract_year)
    # Imputamos year con la mediana por si alguna peli no la trae
    m["year"] = m["year"].fillna(m["year"].median())

    return m, list(genre_dummies.columns)


# =========================
# Split leave-last-out por usuario
# =========================
def _leave_last_out_split(
    ratings: pd.DataFrame,
    min_ratings_per_user: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Para cada usuario, envia su rating MAS RECIENTE a test y el resto a train.
    Estandar en evaluacion de recomendadores, evita fuga temporal.
    """
    # Excluir usuarios con menos ratings de los necesarios
    counts = ratings.groupby("user_id").size()
    valid_users = counts[counts >= min_ratings_per_user].index
    ratings = ratings[ratings["user_id"].isin(valid_users)].copy()

    # Ordenamos por timestamp y marcamos el ultimo por usuario
    ratings = ratings.sort_values(["user_id", "timestamp"])
    last_per_user = ratings.groupby("user_id").tail(1).index

    test = ratings.loc[last_per_user].copy()
    train = ratings.drop(last_per_user).copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)


# =========================
# Estadisticas de train
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
# User-user similarity (vectorizado)
# =========================
def _build_user_sim_scores(
    train: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Calcula user_sim_score para CADA par (user_id, movie_id) observado en el
    conjunto de entrenamiento y lo devuelve como DataFrame indexado por
    (user_id, movie_id) -> score.

    Este mapa se usa luego como lookup al construir features. Para pares no
    vistos (cold start) devolvemos NaN y se rellena con global_mean.

    Idea:
        Sim_matrix (U x U), Rating_matrix (U x M). Queremos, para cada user u y
        pelicula m, la media ponderada de los ratings de los top_k usuarios mas
        parecidos a u que hayan visto m.

        Simplificacion: en vez de usar solo top_k, usamos TODOS los vecinos pero
        ponderados por similitud y mascara 'ha visto la peli'. Esto es mucho mas
        rapido y suele aproximar bien el score top-k (los vecinos mas similares
        dominan el sumatorio).
    """
    user_movie = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
    )

    # Matriz de ratings (0 donde no hay rating) y mascara (1 donde si hay)
    R = user_movie.fillna(0.0).values
    M = (~user_movie.isna()).astype(float).values

    # Similitud usuario-usuario sobre ratings crudos
    sim = cosine_similarity(R)
    # Anulamos la diagonal (no queremos que el propio usuario pese)
    np.fill_diagonal(sim, 0.0)

    # Opcional: nos quedamos con top_k vecinos por usuario (mas fiel al codigo original).
    # Dejamos una version "soft" (no top_k) porque en la practica da resultados similares
    # y simplifica. Si quieres la version estricta top_k, se puede hacer un argsort.
    if top_k is not None and top_k > 0:
        # Para cada fila, dejamos 0 todas las similitudes excepto las top_k mayores
        n_users = sim.shape[0]
        top_k = min(top_k, n_users - 1)
        # argsort ascendente: los indices de los mas pequenos primero
        idx_small = np.argsort(sim, axis=1)[:, : n_users - top_k]
        rows = np.repeat(np.arange(n_users), idx_small.shape[1])
        cols = idx_small.flatten()
        sim[rows, cols] = 0.0

    # Numerador y denominador de la media ponderada
    num = sim @ R      # (U x M)
    den = sim @ M      # (U x M) suma de similitudes de vecinos que vieron cada peli
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(den > 0, num / den, np.nan)

    user_sim_df = pd.DataFrame(
        score,
        index=user_movie.index,
        columns=user_movie.columns,
    )

    # Pasamos a formato largo para que sea un lookup barato por (user_id, movie_id)
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
    Factoriza la matriz user-movie con NMF y devuelve un lookup largo
    (user_id, movie_id -> nmf_score) con el rating RECONSTRUIDO.

    Si sklearn no esta disponible, devuelve un DataFrame vacio y el feature
    se rellenara con global_mean en _add_features (no se rompe el pipeline).

    Limitacion conocida: rellenamos missings con 0, lo que NMF interpreta como
    "no le gusta". Es el enfoque simple; un siguiente paso seria usar ALS implicit
    o Surprise (SVD) que manejan missings correctamente.
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
        init="nndsvd",          # init deterministico, converge mas rapido que 'random'
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
# Feature engineering sobre un conjunto
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

    # NMF feature (opcional — solo si hay sklearn y se ha computado)
    if nmf_lookup is not None and len(nmf_lookup) > 0:
        df = df.merge(nmf_lookup, on=["user_id", "movie_id"], how="left")
    else:
        df["nmf_score"] = np.nan

    # Imputaciones consistentes (cold start)
    df["user_avg"] = df["user_avg"].fillna(global_mean)
    df["movie_avg"] = df["movie_avg"].fillna(global_mean)
    df["user_count"] = df["user_count"].fillna(0)
    df["movie_count"] = df["movie_count"].fillna(0)
    df["user_sim_score"] = df["user_sim_score"].fillna(global_mean)
    df["nmf_score"] = df["nmf_score"].fillna(global_mean)
    df[genre_columns] = df[genre_columns].fillna(0)
    df["year"] = df["year"].fillna(movies_with_features["year"].median())

    # Interacciones
    df["interaction"] = df["user_avg"] * df["movie_avg"]
    df["diff"] = df["user_avg"] - df["movie_avg"]
    df["abs_diff"] = df["diff"].abs()

    return df


# =========================
# API publica
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
    Carga MovieLens y devuelve un RecommenderData listo para entrenar.

    Parametros
    ----------
    task : 'classification' o 'regression'
        - classification: y = 1 si rating >= liked_threshold, si no 0
        - regression:     y = rating (float 0.5 - 5.0)
    data_dir : ruta a la carpeta con ratings.csv y movies.csv
    liked_threshold : umbral del 'me gusta' para clasificacion
    top_k_similar : nº de vecinos usados en user_sim_score (None o 0 = todos)
    min_ratings_per_user : minimo de ratings por usuario para ser incluido
    nmf_components : nº de factores latentes del NMF (pon 0 para desactivar NMF)
    save_to : si se pasa un path, guarda X_train, X_test, y_train, y_test como csv.

    Devuelve
    --------
    RecommenderData
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"task debe ser 'classification' o 'regression', recibi {task!r}")

    # 1. Cargar raw
    ratings, movies = _load_raw(data_dir)

    # 2. Features de peliculas
    movies_feat, genre_columns = _build_movies_with_features(movies)

    # 3. Split leave-last-out
    train, test = _leave_last_out_split(ratings, min_ratings_per_user=min_ratings_per_user)

    # 4. Stats SOLO con train
    user_stats, movie_stats, global_mean = _fit_train_stats(train)

    # 5. User similarity scores (vectorizado) a partir de train
    user_sim_lookup = _build_user_sim_scores(train, top_k=top_k_similar)

    # 5b. NMF latent factors a partir de train
    if nmf_components and nmf_components > 0:
        nmf_lookup = _build_nmf_features(train, n_components=nmf_components)
    else:
        nmf_lookup = pd.DataFrame(columns=["user_id", "movie_id", "nmf_score"])

    # 6. Feature engineering en ambos conjuntos
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

    # 7. Definir feature_columns y target
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

    # 8. Persistencia opcional
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
# Helper para construir features de candidatos (peliculas no vistas)
# =========================
def build_candidate_features(
    data: RecommenderData,
    user_id: int,
    candidate_movie_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Genera X para todos los pares (user_id, movie_id) candidatos, util para
    producir recomendaciones top-K con cualquiera de los modelos.

    Si candidate_movie_ids es None, usa todas las peliculas no vistas por el usuario
    en train.
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

    # Usamos los lookups COMPLETOS (construidos en load_data con TODAS las
    # parejas user-movie con score calculable, incluidas las que el usuario
    # no ha visto). Esto es crucial para ranking de candidatos: si solo
    # tirasemos de train_ratings, user_sim_score y nmf_score serian constantes
    # en candidatos no vistos y el modelo perderia capacidad de discriminar.
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
