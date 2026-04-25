import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load data
# =========================
cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_data = pd.read_csv('ml-latest-small/ratings.csv', sep=',', names=cols, header=0)
movies_data = pd.read_csv('ml-latest-small/movies.csv')

# =========================
# Genre binary variables
# =========================
movies_genres = movies_data.copy()
movies_genres['genres'] = movies_genres['genres'].fillna('')

genre_dummies = movies_genres['genres'].str.get_dummies(sep='|')
if '(no genres listed)' in genre_dummies.columns:
    genre_dummies = genre_dummies.drop(columns=['(no genres listed)'])

genre_dummies.columns = [f'genre_{c}' for c in genre_dummies.columns]

movies_with_genres = pd.concat(
    [movies_genres[['movieId']], genre_dummies],
    axis=1
).rename(columns={'movieId': 'movie_id'})

# =========================
# Train / test split
# =========================
train_ratings, test_ratings = train_test_split(
    ratings_data,
    test_size=0.2,
    random_state=42
)

# =========================
# User-user similarity
# =========================
user_movie_matrix = train_ratings.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
)

user_sim_matrix = cosine_similarity(user_movie_matrix.fillna(0))

user_similarity = pd.DataFrame(
    user_sim_matrix,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# =========================
# Aggregated stats
# =========================
global_mean_rating = train_ratings['rating'].mean()

user_stats = train_ratings.groupby('user_id').agg({
    'rating': ['mean', 'count']
})
user_stats.columns = ['user_avg', 'user_count']

movie_stats = train_ratings.groupby('movie_id').agg({
    'rating': ['mean', 'count']
})
movie_stats.columns = ['movie_avg', 'movie_count']

# =========================
# Similar users score
# =========================
def get_similar_users_score(user_id, movie_id, top_k=10):
    if user_id not in user_similarity.index or movie_id not in user_movie_matrix.columns:
        return np.nan

    sims = user_similarity[user_id].drop(user_id)
    sims = sims.sort_values(ascending=False).head(top_k)

    ratings_for_movie = user_movie_matrix[movie_id]
    valid_ratings = ratings_for_movie.loc[sims.index].dropna()

    if len(valid_ratings) == 0:
        return np.nan

    sims = sims.loc[valid_ratings.index]

    sim_sum = sims.sum()
    if sim_sum == 0:
        return np.nan

    return np.dot(sims, valid_ratings) / sim_sum

# =========================
# Feature engineering
# =========================
def add_features(ratings_frame):
    feature_frame = ratings_frame.copy()

    feature_frame = feature_frame.merge(user_stats, on='user_id', how='left')
    feature_frame = feature_frame.merge(movie_stats, on='movie_id', how='left')
    feature_frame = feature_frame.merge(movies_with_genres, on='movie_id', how='left')

    feature_frame['user_avg'] = feature_frame['user_avg'].fillna(global_mean_rating)
    feature_frame['movie_avg'] = feature_frame['movie_avg'].fillna(global_mean_rating)
    feature_frame['user_count'] = feature_frame['user_count'].fillna(0)
    feature_frame['movie_count'] = feature_frame['movie_count'].fillna(0)

    genre_cols_local = [c for c in feature_frame.columns if c.startswith('genre_')]
    if genre_cols_local:
        feature_frame[genre_cols_local] = feature_frame[genre_cols_local].fillna(0)

    feature_frame['interaction'] = feature_frame['user_avg'] * feature_frame['movie_avg']
    feature_frame['diff'] = feature_frame['user_avg'] - feature_frame['movie_avg']
    feature_frame['abs_diff'] = feature_frame['diff'].abs()

    feature_frame['user_sim_score'] = feature_frame.apply(
        lambda row: get_similar_users_score(row['user_id'], row['movie_id']),
        axis=1
    )
    feature_frame['user_sim_score'] = feature_frame['user_sim_score'].fillna(global_mean_rating)

    return feature_frame

train_features_df = add_features(train_ratings)
test_features_df = add_features(test_ratings)

genre_feature_cols = [c for c in train_features_df.columns if c.startswith('genre_')]

feature_columns = [
    'user_avg',
    'user_count',
    'movie_avg',
    'movie_count',
    'interaction',
    'abs_diff',
    'user_sim_score'
] + genre_feature_cols

# =========================
# Regression target
# =========================
X_train = train_features_df[feature_columns]
y_train = train_features_df['rating']

X_test = test_features_df[feature_columns]
y_test = test_features_df['rating']

# =========================
# Save train and test data
# =========================
os.makedirs("data_regression", exist_ok=True)

X_train.to_csv("data_regression/X_train.csv", index=False)
y_train.to_csv("data_regression/y_train.csv", index=False)

X_test.to_csv("data_regression/X_test.csv", index=False)
y_test.to_csv("data_regression/y_test.csv", index=False)

# =========================
# Metrics helper
# =========================
def evaluate_regression_model(model_name, fitted_model, X_eval, y_eval):
    predictions = fitted_model.predict(X_eval)

    rmse = np.sqrt(mean_squared_error(y_eval, predictions))
    mae = mean_absolute_error(y_eval, predictions)
    r2 = r2_score(y_eval, predictions)

    print(f"\n===== {model_name} =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    return predictions, rmse, mae, r2

# =========================
# 1) Linear Regression
# =========================
linear_regression_model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

linear_regression_model.fit(X_train, y_train)

linear_preds, linear_rmse, linear_mae, linear_r2 = evaluate_regression_model(
    "Linear Regression",
    linear_regression_model,
    X_test,
    y_test
)

linear_coef_df = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': linear_regression_model.named_steps['model'].coef_
})

linear_coef_df['abs_coefficient'] = linear_coef_df['coefficient'].abs()
linear_coef_df = linear_coef_df.sort_values('abs_coefficient', ascending=False)

print("\n=== Linear Regression Coefficients ===")
print(linear_coef_df[['feature', 'coefficient']].to_string(index=False))

# =========================
# 2) Neural Net Regressor
# =========================
neural_net_model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ))
])

neural_net_model.fit(X_train, y_train)

nn_preds, nn_rmse, nn_mae, nn_r2 = evaluate_regression_model(
    "Neural Net Regressor",
    neural_net_model,
    X_test,
    y_test
)

# =========================
# Compare models
# =========================
results_df = pd.DataFrame({
    'model': ['Linear Regression', 'Neural Net Regressor'],
    'RMSE': [linear_rmse, nn_rmse],
    'MAE': [linear_mae, nn_mae],
    'R2': [linear_r2, nn_r2]
})

print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# =========================
# Recommendation helpers
# =========================
def recommend_movies(user_id, seen_ratings_frame, fitted_model, top_k=5):
    seen_movies = seen_ratings_frame[seen_ratings_frame['user_id'] == user_id]['movie_id'].unique()
    all_movies = movies_with_genres['movie_id'].unique()
    unseen_movies = np.setdiff1d(all_movies, seen_movies)

    candidate_movies = pd.DataFrame({
        'user_id': user_id,
        'movie_id': unseen_movies
    })

    candidate_movies = add_features(candidate_movies)
    X_candidates = candidate_movies[feature_columns]

    candidate_movies['pred_rating'] = fitted_model.predict(X_candidates)

    return candidate_movies.sort_values('pred_rating', ascending=False).head(top_k)

def get_movie_title(movie_id):
    movie_row = movies_data[movies_data['movieId'] == movie_id]

    if movie_row.empty:
        return "Movie not found"

    return movie_row.iloc[0]['title']

def explain_prediction(user_id, movie_id, fitted_model):
    prediction_row = pd.DataFrame({
        'user_id': [user_id],
        'movie_id': [movie_id]
    })

    prediction_row = add_features(prediction_row)
    X_row = prediction_row[feature_columns]

    predicted_rating = fitted_model.predict(X_row)[0]
    movie_title = get_movie_title(movie_id)

    print("====================================")
    print(f"User ID: {user_id}")
    print(f"Movie ID: {movie_id}")
    print(f"Title   : {movie_title}")
    print("====================================")

    print("\n--- Features used by the model ---")
    for col in feature_columns:
        if not col.startswith('genre_'):
            print(f"{col}: {X_row.iloc[0][col]:.4f}")

    print("\n--- Model Output ---")
    print(f"Predicted rating: {predicted_rating:.4f}")
    print("====================================\n")

    return predicted_rating

# =========================
# Examples
# =========================
print("\nTop recommendations with Linear Regression:")
print(recommend_movies(1, train_ratings, linear_regression_model, top_k=5)[['movie_id', 'pred_rating']])

print("\nPrediction explanation with Linear Regression:")
explain_prediction(user_id=275, movie_id=5745, fitted_model=linear_regression_model)

print("\nTop recommendations with Neural Net:")
print(recommend_movies(1, train_ratings, neural_net_model, top_k=5)[['movie_id', 'pred_rating']])

print("\nPrediction explanation with Neural Net:")
explain_prediction(user_id=275, movie_id=5745, fitted_model=neural_net_model)