import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

import os

cols = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('ml-latest-small/ratings.csv', sep=',', names=cols, header=0)
movies = pd.read_csv('ml-latest-small/movies.csv')

df['liked'] = (df['rating'] >= 4.0).astype(int)

# genres en variables binarias, excluyendo "(no genres listed)"
movies_genres = movies.copy()
movies_genres['genres'] = movies_genres['genres'].fillna('')

genre_dummies = movies_genres['genres'].str.get_dummies(sep='|')
if '(no genres listed)' in genre_dummies.columns:
    genre_dummies = genre_dummies.drop(columns=['(no genres listed)'])

genre_dummies.columns = [f'genre_{c}' for c in genre_dummies.columns]

movies_with_genres = pd.concat(
    [movies_genres[['movieId']], genre_dummies],
    axis=1
).rename(columns={'movieId': 'movie_id'})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

user_movie = train_df.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
)

user_sim_matrix = cosine_similarity(user_movie.fillna(0))

user_sim = pd.DataFrame(
    user_sim_matrix,
    index=user_movie.index,
    columns=user_movie.index
)

global_mean = train_df['rating'].mean()

user_stats = train_df.groupby('user_id').agg({
    'rating': ['mean', 'count']
})
user_stats.columns = ['user_avg', 'user_count']

movie_stats = train_df.groupby('movie_id').agg({
    'rating': ['mean', 'count']
})
movie_stats.columns = ['movie_avg', 'movie_count']


def get_similar_users_score(user_id, movie_id, top_k=10):
    if user_id not in user_sim.index or movie_id not in user_movie.columns:
        return np.nan

    sims = user_sim[user_id].drop(user_id)
    sims = sims.sort_values(ascending=False).head(top_k)

    ratings = user_movie[movie_id]
    valid = ratings.loc[sims.index].dropna()

    if len(valid) == 0:
        return np.nan

    sims = sims.loc[valid.index]

    return np.dot(sims, valid) / sims.sum()


def add_features(df):
    df = df.merge(user_stats, on='user_id', how='left')
    df = df.merge(movie_stats, on='movie_id', how='left')
    df = df.merge(movies_with_genres, on='movie_id', how='left')

    df['user_avg'] = df['user_avg'].fillna(global_mean)
    df['movie_avg'] = df['movie_avg'].fillna(global_mean)
    df['user_count'] = df['user_count'].fillna(0)
    df['movie_count'] = df['movie_count'].fillna(0)

    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    df[genre_cols] = df[genre_cols].fillna(0)

    df['interaction'] = df['user_avg'] * df['movie_avg']
    df['diff'] = df['user_avg'] - df['movie_avg']
    df['abs_diff'] = abs(df['diff'])

    df['user_sim_score'] = df.apply(
        lambda x: get_similar_users_score(x['user_id'], x['movie_id']),
        axis=1
    )
    df['user_sim_score'] = df['user_sim_score'].fillna(global_mean)

    return df


train_df = add_features(train_df)
test_df = add_features(test_df)

genre_cols = [c for c in train_df.columns if c.startswith('genre_')]

features = [
    'user_avg',
    'user_count',
    'movie_avg',
    'movie_count',
    'interaction',
    'abs_diff',
    'user_sim_score'
] + genre_cols

X_train = train_df[features]
y_train = train_df['liked']

X_test = test_df[features]
y_test = test_df['liked']

os.makedirs("data", exist_ok=True)

X_train.to_csv("data/X_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

probs = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, probs)
print("AUC:", auc)

coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': pipeline.named_steps['model'].coef_[0]
})

coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coefficient', ascending=False)

print("\n=== Feature Coefficients ===")
print(coef_df[['feature', 'coefficient']].to_string(index=False))

# print("\n=== Top positive features ===")
# print(
#     coef_df.sort_values('coefficient', ascending=False)[['feature', 'coefficient']]
#     .head(10)
#     .to_string(index=False)
# )

# print("\n=== Top negative features ===")
# print(
#     coef_df.sort_values('coefficient', ascending=True)[['feature', 'coefficient']]
#     .head(10)
#     .to_string(index=False)
# )