import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import os

# MovieLens 100K format
cols = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('ml-latest-small/ratings.csv', sep=',', names=cols, header=0)
movies = pd.read_csv('ml-latest-small/movies.csv')

# Binary target
df['liked'] = (df['rating'] >= 4.0).astype(int)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

user_movie = train_df.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
)

from sklearn.metrics.pairwise import cosine_similarity

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

global_mean = train_df['rating'].mean()

def add_features(df):
    
    df = df.merge(user_stats, on='user_id', how='left')
    df = df.merge(movie_stats, on='movie_id', how='left')
    
    # Fill missing
    df['user_avg'] = df['user_avg'].fillna(global_mean)
    df['movie_avg'] = df['movie_avg'].fillna(global_mean)
    df['user_count'] = df['user_count'].fillna(0)
    df['movie_count'] = df['movie_count'].fillna(0)
    
    # Interaction
    df['interaction'] = df['user_avg'] * df['movie_avg']
    df['diff'] = df['user_avg'] - df['movie_avg']
    df['abs_diff'] = abs(df['diff'])
    
    # 🔥 NEW FEATURE: similarity-based score
    df['user_sim_score'] = df.apply(
        lambda x: get_similar_users_score(x['user_id'], x['movie_id']),
        axis=1
    )
    
    df['user_sim_score'] = df['user_sim_score'].fillna(global_mean)
    
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)


features = [
    'user_avg',
    'user_count',
    'movie_avg',
    'movie_count',
    'interaction',
    'abs_diff', 
    'user_sim_score'
]

X_train = train_df[features]
y_train = train_df['liked']

X_test = test_df[features]
y_test = test_df['liked']


os.mkdir("data")

X_train.to_csv("data/X_train")
y_train.to_csv("data/y_train")


pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

probs = pipeline.predict_proba(X_test)[:, 1]


auc = roc_auc_score(y_test, probs)
print("AUC:", auc)



def recommend_movies(user_id, df, model, top_k=5):
    # Movies user has already seen
    seen_movies = df[df['user_id'] == user_id]['movie_id']
    
    # All movies
    all_movies = df['movie_id'].unique()
    
    # Unseen movies
    unseen = np.setdiff1d(all_movies, seen_movies)
    
    # Build candidate dataframe
    candidates = pd.DataFrame({
        'user_id': user_id,
        'movie_id': unseen
    })
    
    # Add features
    candidates = add_features(candidates)
    
    X = candidates[features]
    
    # Predict probabilities
    candidates['prob'] = model.predict_proba(X)[:, 1]
    
    # Top recommendations
    return candidates.sort_values('prob', ascending=False).head(top_k)


recommend_movies(1, train_df, pipeline)

def explain_prediction(user_id, movie_id, df, model):
    
    # Build feature row
    row = pd.DataFrame({
        'user_id': [user_id],
        'movie_id': [movie_id]
    })
    
    # Add features
    row = add_features(row)
    
    X = row[features]
    
    # Prediction
    prob = model.predict_proba(X)[:, 1][0]
    
    # Movie info
    title, genres = get_movie_info(movie_id)
    
    print("====================================")
    print(f"User ID: {user_id}")
    print(f"Movie ID: {movie_id}")
    print(f"Title   : {title}")
    print(f"Genres  : {genres}")
    print("====================================")
    
    print("\n--- Features used by the model ---")
    for col in features:
        print(f"{col}: {X.iloc[0][col]:.4f}")
    
    print("\n--- Model Output ---")
    print(f"Predicted probability of liking: {prob:.4f}")
    
    print("====================================\n")
    
    return prob



def get_movie_info(movie_id):
    row = movies[movies['movieId'] == movie_id]
    
    if row.empty:
        return "Movie not found"
    
    title = row.iloc[0]['title']
    genres = row.iloc[0]['genres']
    
    return title, genres


explain_prediction(user_id=275, movie_id=5745, df=train_df, model=pipeline)