# Netflix / MovieLens recommender

Recommendation system built on **MovieLens `ml-latest-small`** (100,836
ratings, 9,742 movies, 610 users). Final project for *Business with Data*.

The goal is NOT to predict the exact rating (regression), but to build a
good **top-10 list per user**. The headline metrics are
**Hit@10 / NDCG@10 / MAP@10**, not RMSE or AUC.

---

## Repo layout

```
netflix-recommendation/
├── ml-latest-small/         dataset (ratings.csv, movies.csv)
├── data_loader.py           load + features + leave-last-out split
├── baselines.py             PopularityBaseline (non-personalized)
├── models.py                NMF Ranker, Logistic / Linear, GBM, MLP
├── evaluation.py            ranking metrics (P@K, R@K, NDCG@K, Hit@K, MAP@K)
├── run_baseline.py          end-to-end pipeline (load -> train -> evaluate -> table)
├── tune_classification.py   randomized hyperparameter search (classification)
├── tune_regression.py       randomized hyperparameter search (regression)
├── show_importances.py      Gradient Boosting feature importance
├── eda_preprocessing.py     EDA + preprocessing report (figures saved to ./figures)
├── requirements.txt
├── README.md
├── report.md                findings, "AUC trap" and model comparison
├── figures/                 PNG charts produced by eda_preprocessing.py
└── legacy/                  v1 code and original EDA scripts
```

---

## The 5 models

| # | Model                | Type                              | Why it's here |
|---|----------------------|-----------------------------------|---------------|
| 1 | **Popularity**       | Non-personalized baseline         | Floor. Any "ML" model MUST beat this. |
| 2 | **NMF Ranker**       | Matrix factorization              | Reads the score reconstructed by NMF and ranks by it. |
| 3 | **LogisticRegression** (classification) / **LinearRegression** (regression) | Linear supervised | Classic ML baseline over the engineered features. |
| 4 | **GradientBoosting** | Boosted trees                     | Strong non-linear baseline. |
| 5 | **MLP**              | Small feedforward neural network  | One non-linear deep-learning model in the lineup. |

The MLP is intentionally moderate (hidden 64 -> 32) with StandardScaler,
early stopping and L2. Larger architectures or more dropout did not help
on this dataset.

---

## Data pipeline

1. Load `ratings.csv` + `movies.csv`.
2. **Movie features**: genre dummies, `year` extracted from the title.
3. **Leave-last-out split**: each user's most recent rating goes to test.
   Avoids temporal leakage.
4. **Train-only stats**: `user_avg`, `user_count`, `movie_avg`,
   `movie_count`, `global_mean`.
5. **Vectorized user-user similarity** over the user x movie matrix.
6. **NMF latent factors** (k=50) over `user_movie.fillna(0)`.
7. **Cold start**: missing -> `global_mean` / `0`.
8. **Target**:
   - `classification`: `y = (rating >= 4).astype(int)`.
   - `regression`: `y = rating` (0.5 - 5.0).

The `user_sim_score` and `nmf_score` lookups are stored **complete**
(every user-movie pair with a computable score). This is critical: without
them, when scoring unseen movies the features would collapse to a constant
and the ranking would lose meaning.

---

## Evaluation

Ranking metrics computed per user and averaged:

- **Hit@10** — 1 if any relevant movie (in test) appears in the top-10.
- **NDCG@10** — DCG / IDCG with graded relevance (uses the actual rating).
- **MAP@10** — mean average precision.
- **Precision@10 / Recall@10**.

`evaluate_ranker(model, data, k=10, candidate_strategy=...)` supports:

- `test_plus_sample` (default): 1 positive + 99 random negatives.
  Standard leave-one-out protocol used in recsys papers.
- `unseen`: ranks ALL movies the user has not seen.

---

## EDA / Preprocessing report

`eda_preprocessing.py` produces an end-to-end EDA report with 11 PNG
figures saved to `./figures/`:

- nulls and duplicates
- rating distribution
- class balance for classification
- ratings per user and per movie (long tail)
- genre distribution (catalog vs consumption)
- temporal distribution (movie year and rating year)
- sparsity of the user x movie matrix
- correlation heatmap of engineered features
- correlation of each feature with the target

```bash
python eda_preprocessing.py
python eda_preprocessing.py --skip-load-data   # skip the ~30s NMF section
```

---

## Usage

```bash
pip install -r requirements.txt

# Full pipeline: 5 models, summary table
python run_baseline.py

# Regression task (rating 0.5 - 5.0)
python run_baseline.py --task regression

# More users, top-20, candidates = all unseen movies
python run_baseline.py --k 20 --max-users 200 --candidates unseen

# Run a subset
python run_baseline.py --only "Popularity,NMF Ranker"

# Hyperparameter search
python tune_classification.py
python tune_regression.py

# Feature importance from the tuned GBM
python show_importances.py
```

Programmatic usage:

```python
from data_loader import load_data, build_candidate_features
from models import build_models
from evaluation import evaluate_ranker

data = load_data(task="classification")
models = build_models("classification")

nmf = models["NMF Ranker"]
nmf.fit(data.X_train, data.y_train)

metrics = evaluate_ranker(nmf, data, k=10, max_users=100)
print(metrics)  # {'hit@10': ..., 'ndcg@10': ..., ...}
```

---

## Reproducibility

Random seeds fixed to 42: NMF, MLP, negative sampling. The leave-last-out
split is deterministic (sorted by timestamp).

---

## Findings

See `report.md`. Summary:

> At top-K, **pure NMF** beats the classic supervised models (Logistic,
> GBM, MLP) by a wide margin. The lesson is the **AUC trap**: optimizing
> a global classification probability is a different task from ranking
> per user.
