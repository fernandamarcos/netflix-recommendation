import pandas as pd
import numpy as np

# =========================
# Load data
# =========================
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

print("========== DATA SHAPES ==========")
print("ratings shape:", ratings.shape)
print("movies shape :", movies.shape)

print("\n========== COLUMNS ==========")
print("\nratings:")
print(ratings.dtypes)
print("\nmovies:")
print(movies.dtypes)

print("\n========== HEAD ==========")
print("\nratings:")
print(ratings.head())
print("\nmovies:")
print(movies.head())

# =========================
# Missing values
# =========================
print("\n========== MISSING VALUES ==========")
print("\nratings:")
print(ratings.isnull().sum())
print("\nmovies:")
print(movies.isnull().sum())

# =========================
# Duplicates
# =========================
print("\n========== DUPLICATES ==========")
print("ratings duplicated rows:", ratings.duplicated().sum())
print("movies duplicated rows  :", movies.duplicated().sum())

# User-movie duplicate interactions
dup_user_movie = ratings.duplicated(subset=['userId', 'movieId']).sum()
print("duplicate userId-movieId pairs in ratings:", dup_user_movie)

# =========================
# Basic stats
# =========================
print("\n========== BASIC STATS ==========")
print("number of users  :", ratings['userId'].nunique())
print("number of movies :", ratings['movieId'].nunique())
print("number of ratings:", len(ratings))

print("\nrating summary:")
print(ratings['rating'].describe())

# =========================
# Binary target
# =========================
ratings['liked'] = (ratings['rating'] >= 4.0).astype(int)

print("\n========== TARGET DISTRIBUTION ==========")
print(ratings['liked'].value_counts(dropna=False))
print("\nliked proportion:")
print(ratings['liked'].value_counts(normalize=True).round(4))

# =========================
# Rating distribution
# =========================
print("\n========== RATING DISTRIBUTION ==========")
print(ratings['rating'].value_counts().sort_index())

# =========================
# User activity
# =========================
ratings_per_user = ratings.groupby('userId').size()

print("\n========== USER ACTIVITY ==========")
print(ratings_per_user.describe())

print("\nTop 10 most active users:")
print(ratings_per_user.sort_values(ascending=False).head(10))

# =========================
# Movie activity
# =========================
ratings_per_movie = ratings.groupby('movieId').size()

print("\n========== MOVIE ACTIVITY ==========")
print(ratings_per_movie.describe())

print("\nTop 10 most rated movies:")
top_movies = ratings_per_movie.sort_values(ascending=False).head(10).reset_index()
top_movies.columns = ['movieId', 'num_ratings']
top_movies = top_movies.merge(movies[['movieId', 'title']], on='movieId', how='left')
print(top_movies)

# =========================
# Average rating by movie
# =========================
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

movie_stats = movie_stats.merge(movies[['movieId', 'title']], on='movieId', how='left')

print("\n========== TOP MOVIES BY AVERAGE RATING (min 20 ratings) ==========")
print(
    movie_stats[movie_stats['num_ratings'] >= 20]
    .sort_values(['avg_rating', 'num_ratings'], ascending=[False, False])
    [['movieId', 'title', 'avg_rating', 'num_ratings']]
    .head(10)
)

# =========================
# Sparsity
# =========================
n_users = ratings['userId'].nunique()
n_movies = ratings['movieId'].nunique()
possible_interactions = n_users * n_movies
observed_interactions = len(ratings)
sparsity = 1 - (observed_interactions / possible_interactions)

print("\n========== SPARSITY ==========")
print("possible interactions:", possible_interactions)
print("observed interactions:", observed_interactions)
print("sparsity:", round(sparsity, 6))

# =========================
# Genres EDA
# =========================
movies['genres'] = movies['genres'].fillna('')

genre_dummies = movies['genres'].str.get_dummies(sep='|')

print("\n========== GENRES ==========")
print("number of unique genre columns:", genre_dummies.shape[1])

genre_counts = genre_dummies.sum().sort_values(ascending=False)
print("\nGenre counts:")
print(genre_counts)

# Remove '(no genres listed)' if present for cleaner view
if '(no genres listed)' in genre_counts.index:
    print("\nGenre counts without '(no genres listed)':")
    print(genre_counts.drop('(no genres listed)'))

# =========================
# Timestamp EDA
# =========================
if 'timestamp' in ratings.columns:
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s', errors='coerce')

    print("\n========== TIMESTAMP ==========")
    print("min datetime:", ratings['datetime'].min())
    print("max datetime:", ratings['datetime'].max())

    print("\nRatings by year:")
    print(ratings['datetime'].dt.year.value_counts().sort_index())

# =========================
# Merge quick check
# =========================
merged = ratings.merge(movies, on='movieId', how='left')

print("\n========== MERGE CHECK ==========")
print("rows after merge:", merged.shape[0])
print("movies missing title after merge:", merged['title'].isnull().sum())

print("\n========== QUICK TAKEAWAYS ==========")
print("1. Target variable 'liked' is based on rating >= 4.0")
print("2. Data is user-item interaction data, usually very sparse")
print("3. User activity and movie popularity are likely highly skewed")
print("4. Movie averages with very few ratings may be noisy")
print("5. Genres can be encoded as binary features for modeling")