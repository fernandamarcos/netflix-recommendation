"""
eda_preprocessing.py
====================

Exploratory data analysis (EDA) and preprocessing of the MovieLens
ml-latest-small dataset. Designed to generate the figures used in the slides
and report.

Sections:
    1. Load and dtypes
    2. Null values
    3. Duplicate rows
    4. Cardinalities (users, movies)
    5. Rating distribution
    6. Class balance (rating >= 4)
    7. Ratings per user (long tail)
    8. Ratings per movie (long tail)
    9. Genre distribution
   10. Temporal distribution (year)
   11. Sparsity of the user x movie matrix
   12. Correlation between engineered features
   13. Correlation of each feature with the target

Usage:
    python eda_preprocessing.py
    python eda_preprocessing.py --out figures
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse pipeline helpers to avoid duplicating logic
from data_loader import (
    _load_raw,
    _build_movies_with_features,
    _extract_year,
    load_data,
)


# =========================
# Style
# =========================
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
})

PALETTE = {
    "primary": "#E50914",     # netflix red
    "secondary": "#221F1F",   # near-black
    "accent": "#564D4D",
    "muted": "#B3B3B3",
    "good": "#46D369",
    "bad": "#E50914",
    "blue": "#1F77B4",
}


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print('=' * 72)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures",
                        help="folder where PNGs will be written")
    parser.add_argument("--data-dir", default="ml-latest-small")
    parser.add_argument("--skip-load-data", action="store_true",
                        help="skip sections 12-13 (load_data takes ~30s for NMF)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load
    # -------------------------------------------------------------
    section("1. Loading data")
    ratings, movies = _load_raw(args.data_dir)
    print(f"ratings: {ratings.shape}   "
          f"({ratings.memory_usage(deep=True).sum()/1024**2:.2f} MB)")
    print(f"movies:  {movies.shape}   "
          f"({movies.memory_usage(deep=True).sum()/1024**2:.2f} MB)")
    print("\nratings.head():")
    print(ratings.head().to_string(index=False))
    print("\nmovies.head():")
    print(movies.head().to_string(index=False))
    print("\nratings dtypes:")
    print(ratings.dtypes.to_string())
    print("\nmovies dtypes:")
    print(movies.dtypes.to_string())

    # -------------------------------------------------------------
    # 2. Nulls
    # -------------------------------------------------------------
    section("2. Null values")
    null_ratings = ratings.isna().sum()
    null_movies = movies.isna().sum()
    print("\nnulls in ratings:")
    print(null_ratings.to_string())
    print("\nnulls in movies:")
    print(null_movies.to_string())

    # Raw year (extracted from title): here we DO see real nulls
    year_raw = movies["title"].apply(_extract_year)
    n_null_year = year_raw.isna().sum()
    print(f"\nmovies without an extractable year in the title: {n_null_year} "
          f"({100*n_null_year/len(movies):.2f}%)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cats = ["ratings.userId", "ratings.movieId", "ratings.rating",
            "ratings.timestamp", "movies.movieId", "movies.title",
            "movies.genres", "movies.year (derived)"]
    vals = [
        null_ratings.get("user_id", 0),
        null_ratings.get("movie_id", 0),
        null_ratings.get("rating", 0),
        null_ratings.get("timestamp", 0),
        null_movies.get("movie_id", 0),
        null_movies.get("title", 0),
        null_movies.get("genres", 0),
        n_null_year,
    ]
    bars = ax.bar(range(len(cats)), vals,
                  color=[PALETTE["primary"] if v > 0 else PALETTE["muted"] for v in vals],
                  edgecolor="white")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("# nulls")
    ax.set_title("Null values per column")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v}",
                ha="center", va="bottom", fontsize=9)
    fig.savefig(os.path.join(args.out, "01_nulls.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. Duplicates
    # -------------------------------------------------------------
    section("3. Duplicates")
    dup_ratings_full = ratings.duplicated().sum()
    dup_ratings_pair = ratings.duplicated(subset=["user_id", "movie_id"]).sum()
    dup_movies = movies.duplicated().sum()
    dup_movie_id = movies["movie_id"].duplicated().sum()
    print(f"ratings duplicated (full row):     {dup_ratings_full}")
    print(f"ratings duplicated (user, movie):  {dup_ratings_pair}")
    print(f"movies duplicated (full row):      {dup_movies}")
    print(f"movies with duplicated movie_id:   {dup_movie_id}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cats = ["ratings\n(row)", "ratings\n(user, movie)",
            "movies\n(row)", "movies\n(movie_id)"]
    vals = [dup_ratings_full, dup_ratings_pair, dup_movies, dup_movie_id]
    bars = ax.bar(cats, vals,
                  color=[PALETTE["primary"] if v > 0 else PALETTE["good"] for v in vals],
                  edgecolor="white")
    ax.set_ylabel("# duplicates")
    ax.set_title("Duplicates (green = clean)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(1, max(vals) * 1.3))
    fig.savefig(os.path.join(args.out, "02_duplicates.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 4. Cardinalities
    # -------------------------------------------------------------
    section("4. Cardinalities")
    n_users = ratings["user_id"].nunique()
    n_movies_in_ratings = ratings["movie_id"].nunique()
    n_movies_catalog = movies["movie_id"].nunique()
    n_obs = len(ratings)
    movies_no_rating = movies[~movies["movie_id"].isin(ratings["movie_id"])]
    print(f"unique users:                    {n_users:,}")
    print(f"movies with at least one rating: {n_movies_in_ratings:,}")
    print(f"movies in catalog:               {n_movies_catalog:,}")
    print(f"movies WITHOUT any rating:       {len(movies_no_rating):,}")
    print(f"total ratings:                   {n_obs:,}")
    print(f"avg ratings per user:            {n_obs/n_users:.1f}")

    # -------------------------------------------------------------
    # 5. Rating distribution
    # -------------------------------------------------------------
    section("5. Rating distribution")
    rating_counts = ratings["rating"].value_counts().sort_index()
    print(rating_counts.to_string())
    print(f"\nmean: {ratings['rating'].mean():.3f}   "
          f"median: {ratings['rating'].median():.3f}   "
          f"std: {ratings['rating'].std():.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(rating_counts.index.astype(str), rating_counts.values,
                  color=PALETTE["primary"], edgecolor="white", linewidth=1.2)
    ax.axvline(x=ratings["rating"].mean() / 0.5 - 1, color=PALETTE["secondary"],
               linestyle="--", alpha=0.5, label=f"mean = {ratings['rating'].mean():.2f}")
    ax.set_title("Rating distribution (0.5 - 5.0)")
    ax.set_xlabel("Rating")
    ax.set_ylabel("# ratings")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{int(h):,}",
                ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.savefig(os.path.join(args.out, "03_rating_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 6. Class balance
    # -------------------------------------------------------------
    section("6. Class balance for classification (rating >= 4)")
    liked = (ratings["rating"] >= 4).astype(int)
    counts = liked.value_counts().sort_index()
    pct_pos = liked.mean() * 100
    pct_neg = 100 - pct_pos
    print(f"liked = 0 (rating <  4): {counts[0]:>7,}  ({pct_neg:.1f}%)")
    print(f"liked = 1 (rating >= 4): {counts[1]:>7,}  ({pct_pos:.1f}%)")
    print(f"ratio (pos/neg):         {counts[1]/counts[0]:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(["Not liked\n(rating < 4)", "Liked\n(rating >= 4)"],
                  counts.values,
                  color=[PALETTE["bad"], PALETTE["good"]],
                  edgecolor="white", linewidth=1.5)
    ax.set_title(f"Class balance (target = liked)\n"
                 f"{pct_neg:.1f}% / {pct_pos:.1f}% — moderately imbalanced")
    ax.set_ylabel("# ratings")
    for bar, v, p in zip(bars, counts.values, [pct_neg, pct_pos]):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.savefig(os.path.join(args.out, "04_class_balance.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 7. Ratings per user
    # -------------------------------------------------------------
    section("7. Ratings per user (long tail)")
    per_user = ratings.groupby("user_id").size()
    print(per_user.describe().to_string())
    print(f"\nusers with < 20 ratings:  {(per_user < 20).sum()}")
    print(f"users with >= 100 ratings: {(per_user >= 100).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(per_user, bins=50, color=PALETTE["primary"], edgecolor="white")
    axes[0].axvline(per_user.median(), color=PALETTE["secondary"], linestyle="--",
                    label=f"median = {int(per_user.median())}")
    axes[0].set_title("Ratings per user - linear scale")
    axes[0].set_xlabel("# ratings of the user")
    axes[0].set_ylabel("# users")
    axes[0].legend()

    bins_log = np.logspace(np.log10(1), np.log10(per_user.max()+1), 40)
    axes[1].hist(per_user, bins=bins_log, color=PALETTE["primary"], edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_title("Ratings per user - log scale")
    axes[1].set_xlabel("# ratings (log)")
    axes[1].set_ylabel("# users")
    fig.suptitle("Long tail: a few users concentrate many ratings",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(args.out, "05_ratings_per_user.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 8. Ratings per movie
    # -------------------------------------------------------------
    section("8. Ratings per movie (long tail)")
    per_movie = ratings.groupby("movie_id").size()
    print(per_movie.describe().to_string())
    print(f"\nmovies with a single rating: {(per_movie == 1).sum()} "
          f"({100*(per_movie == 1).sum()/len(per_movie):.1f}%)")
    print(f"movies with < 10 ratings:    {(per_movie < 10).sum()} "
          f"({100*(per_movie < 10).sum()/len(per_movie):.1f}%)")
    print(f"movies with >= 100 ratings:  {(per_movie >= 100).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(per_movie, bins=50, color=PALETTE["secondary"], edgecolor="white")
    axes[0].set_title("Ratings per movie - linear scale")
    axes[0].set_xlabel("# ratings")
    axes[0].set_ylabel("# movies")

    bins_log = np.logspace(0, np.log10(per_movie.max()+1), 40)
    axes[1].hist(per_movie, bins=bins_log, color=PALETTE["secondary"], edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Ratings per movie - log-log")
    axes[1].set_xlabel("# ratings (log)")
    axes[1].set_ylabel("# movies (log)")
    fig.suptitle("Classic long tail: most movies have very few ratings",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(args.out, "06_ratings_per_movie.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9. Genres
    # -------------------------------------------------------------
    section("9. Genre distribution")
    movies_feat, genre_columns = _build_movies_with_features(movies)
    genre_counts = movies_feat[genre_columns].sum().sort_values(ascending=False)
    genre_counts.index = [c.replace("genre_", "") for c in genre_counts.index]
    print(genre_counts.to_string())

    # Combine with ratings: rating volume per genre (consumption)
    ratings_with_genre = ratings.merge(
        movies_feat[["movie_id"] + genre_columns], on="movie_id", how="left"
    )
    rated_per_genre = (ratings_with_genre[genre_columns].sum()
                       .sort_values(ascending=False))
    rated_per_genre.index = [c.replace("genre_", "") for c in rated_per_genre.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    axes[0].barh(genre_counts.index[::-1], genre_counts.values[::-1],
                 color=PALETTE["primary"], edgecolor="white")
    axes[0].set_title("Catalog: movies per genre")
    axes[0].set_xlabel("# movies")

    axes[1].barh(rated_per_genre.index[::-1], rated_per_genre.values[::-1],
                 color=PALETTE["secondary"], edgecolor="white")
    axes[1].set_title("Consumption: ratings per genre")
    axes[1].set_xlabel("# ratings")
    fig.savefig(os.path.join(args.out, "07_genres.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 10. Year
    # -------------------------------------------------------------
    section("10. Temporal distribution")
    years = movies["title"].apply(_extract_year).dropna().astype(int)
    print(f"movies with extractable year: {len(years)}")
    print(f"range: {years.min()} - {years.max()}")
    print(f"median year: {int(years.median())}")
    print(f"movies in the 2010s: {((years >= 2010) & (years < 2020)).sum()}")

    # Also: when each rating was made
    ts_dt = pd.to_datetime(ratings["timestamp"], unit="s")
    print(f"\nratings span: {ts_dt.min().date()} to {ts_dt.max().date()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].hist(years, bins=50, color=PALETTE["primary"], edgecolor="white")
    axes[0].set_title("Movie production year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("# movies")
    axes[0].axvline(int(years.median()), color=PALETTE["secondary"],
                    linestyle="--", label=f"median = {int(years.median())}")
    axes[0].legend()

    ts_year = ts_dt.dt.year.value_counts().sort_index()
    axes[1].bar(ts_year.index, ts_year.values, color=PALETTE["secondary"],
                edgecolor="white")
    axes[1].set_title("Year when the rating was issued")
    axes[1].set_xlabel("Rating year")
    axes[1].set_ylabel("# ratings")
    fig.savefig(os.path.join(args.out, "08_temporal.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 11. Sparsity
    # -------------------------------------------------------------
    section("11. Sparsity of the user x movie matrix")
    total = n_users * n_movies_catalog
    sparsity = 1 - n_obs / total
    print(f"theoretical matrix: {n_users:,} x {n_movies_catalog:,} = {total:,} cells")
    print(f"observed:           {n_obs:,}")
    print(f"sparsity:           {sparsity*100:.2f}% (density = {(1-sparsity)*100:.2f}%)")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(["Observed", "Empty"], [n_obs, total - n_obs],
                  color=[PALETTE["primary"], PALETTE["muted"]],
                  edgecolor="white", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title(f"Sparsity of the user x movie matrix\n"
                 f"{sparsity*100:.2f}% empty cells "
                 f"-> motivates matrix factorization (NMF)")
    ax.set_ylabel("# cells (log)")
    for i, v in enumerate([n_obs, total - n_obs]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=11)
    fig.savefig(os.path.join(args.out, "09_sparsity.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # 12-13. Correlations (require load_data, ~30s)
    # -------------------------------------------------------------
    if args.skip_load_data:
        section("12-13. Skipped (--skip-load-data)")
        return

    section("12. Correlations between engineered features")
    print("Loading processed data (NMF + similarity), 30-60s...")
    data = load_data(task="classification")
    numeric_features = [
        "user_avg", "user_count", "movie_avg", "movie_count",
        "interaction", "abs_diff", "user_sim_score", "nmf_score", "year",
    ]
    df = data.X_train[numeric_features].copy()
    df["target"] = data.y_train.values

    # If any column has zero variance (e.g. sklearn not installed and
    # nmf_score got filled with global_mean), drop it from the heatmap so
    # we don't end up with rows/columns full of NaN.
    zero_var = [c for c in df.columns if df[c].nunique() <= 1]
    if zero_var:
        print(f"\n[warning] zero-variance columns excluded from heatmap: {zero_var}")
        df = df.drop(columns=zero_var)

    corr = df.corr()
    print("\nPearson correlation matrix:")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson correlation between features (target included)")
    ax.grid(False)
    fig.savefig(os.path.join(args.out, "10_correlation_heatmap.png"))
    plt.close(fig)

    section("13. Correlation of each feature with the target")
    target_corr = corr["target"].drop("target")
    target_corr_abs = target_corr.abs().sort_values(ascending=False)
    print(target_corr_abs.to_string())

    target_corr_signed = target_corr.reindex(target_corr_abs.index)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [PALETTE["good"] if v > 0 else PALETTE["bad"]
              for v in target_corr_signed]
    bars = ax.barh(target_corr_signed.index[::-1], target_corr_signed.values[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of each feature with the 'liked' class\n"
                 "(green = positive correlation, red = negative)")
    ax.set_xlabel("Pearson correlation")
    for bar, v in zip(bars, target_corr_signed.values[::-1]):
        ax.text(v + (0.005 if v >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", ha="left" if v >= 0 else "right",
                va="center", fontsize=9)
    fig.savefig(os.path.join(args.out, "11_correlation_with_target.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------
    section("SUMMARY")
    pngs = sorted(f for f in os.listdir(args.out) if f.endswith(".png"))
    print(f"\n{len(pngs)} figures saved in '{args.out}/':")
    for f in pngs:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
