import pandas as pd
import numpy as np

from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('ml-latest-small/ratings.csv')

df = df.rename(columns={
    'userId': 'user_id',
    'movieId': 'movie_id'
})

df['liked'] = (df['rating'] >= 4.0).astype(int)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# =========================
# USER-MOVIE MATRIX
# =========================
user_movie = train_df.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
)

# Fill missing with 0 for NMF
user_movie_filled = user_movie.fillna(0)

# =========================
# TRAIN NMF
# =========================
nmf = NMF(
    n_components=20,
    init='random',
    random_state=42,
    max_iter=500
)

W = nmf.fit_transform(user_movie_filled)
H = nmf.components_

# =========================
# RECONSTRUCT RATINGS
# =========================
reconstructed = np.dot(W, H)

nmf_pred = pd.DataFrame(
    reconstructed,
    index=user_movie.index,
    columns=user_movie.columns
)

# =========================
# PREDICTION FUNCTION
# =========================
global_mean = train_df['rating'].mean()

def predict_rating(user_id, movie_id):
    if user_id in nmf_pred.index and movie_id in nmf_pred.columns:
        return nmf_pred.loc[user_id, movie_id]
    return global_mean


# =========================
# CONVERT TO PROBABILITY
# =========================
def rating_to_prob(rating):
    """
    Convert predicted rating (0–5 scale) to probability of 'liked'
    """
    return rating / 5.0  # simple + effective baseline


# =========================
# EVALUATE
# =========================
preds = []

for _, row in test_df.iterrows():
    r = predict_rating(row['user_id'], row['movie_id'])
    p = rating_to_prob(r)
    preds.append(p)

auc = roc_auc_score(test_df['liked'], preds)
print("NMF AUC:", auc)



def recommend_movies_nmf(user_id, top_k=10):
    
    if user_id not in nmf_pred.index:
        return None
    
    # Get all predicted ratings
    scores = nmf_pred.loc[user_id]
    
    # Remove already seen movies
    seen = train_df[train_df['user_id'] == user_id]['movie_id']
    scores = scores.drop(seen, errors='ignore')
    
    return scores.sort_values(ascending=False).head(top_k)

recommend_movies_nmf(1)
