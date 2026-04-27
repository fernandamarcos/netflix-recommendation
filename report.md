# Report — MovieLens recommender system

## 1. Problem

Given a user, return a **top-10 list of movies** they have not seen and
that they are likely to enjoy.

Dataset: **MovieLens `ml-latest-small`** — 100,836 ratings from 610 users
on 9,742 movies, ratings in the 0.5 - 5.0 range.

## 2. The right framing: ranking, not classification

We started with two "obvious" framings:

1. **Regression**: predict the exact rating (RMSE).
2. **Binary classification**: `y = 1 if rating >= 4` (AUC).

Both produced "decent" numbers in the abstract (AUC ~ 0.73, RMSE ~ 0.98)
but when we measured what actually matters — **Hit@10 in a per-user
top-10 ranking** — they collapsed.

### The "AUC trap"

A high AUC does NOT imply a good recommender. AUC measures whether,
across all (positive, negative) pairs in the test set, the model scores
the positive higher than the negative *on average*. It does not condition
on the user. In a real recommender the goal is: "for **this** user, place
their relevant items at the top *among their candidates*". That is a
different task.

Consequence: a Logistic Regression trained on the engineered features
hits AUC = 0.73 but **Hit@10 = 0.15** (worse than random — random would
be 0.10 with one positive among 100 candidates — and far below the 0.87
that pure NMF achieves).

Decision: we stopped optimizing AUC / RMSE, moved the compass to
Hit@10 / NDCG@10 / MAP@10, and structured the project around ranking.

## 3. The 5 models

### 3.1. Popularity (baseline)

Ranks all movies by Bayesian average (IMDb Top 250 formula):

```
score(movie) = (avg_rating * n + global_mean * m) / (n + m),  m = 10
```

Non-personalized: every user gets the same list. This is the **floor**:
any ML model that does not clearly beat it is not adding value.

### 3.2. LogisticRegression (classic ML)

Pipeline `SimpleImputer -> StandardScaler -> LogisticRegression(C=1.0)`
over the engineered feature set: user stats, movie stats, genre dummies,
year, user_sim_score, nmf_score.

This is the "supervised ML with rich features" representative — the
classic course approach. As we will see, it fails at the real task.

### 3.3. NMF Ranker (matrix factorization)

Non-negative factorization of the user x movie matrix:

```
R ~ W . H,     W in R^(u x k),  H in R^(k x m),  k = 50
```

Missing values are filled with 0, NMF is fit, and we rank by the
reconstructed score. **There is no (X, y) loop**: the "model" is the
fitted NMF itself.

Equivalently: a pre-trained collaborative filtering model. On this
dataset it is the strongest performer.

### 3.4. Gradient Boosting (non-linear supervised)

Pipeline `SimpleImputer -> GradientBoostingClassifier`. Boosted trees
do not need StandardScaler. Strong baseline for tabular data.

### 3.5. MLP (small feedforward neural network)

Pipeline `SimpleImputer -> StandardScaler -> MLPClassifier(hidden=(64,32),
relu, adam, alpha=1e-4, early_stopping)`. Moderate architecture, L2
regularization, early stopping. We tried larger and more aggressive
configurations: they did not help.

For the regression task the same shapes are used with `LinearRegression`,
`GradientBoostingRegressor` and `MLPRegressor`.

## 4. Results

Numbers in `classification`, `K=10`, `max_users=100`,
`candidates=test_plus_sample`:

| Model                  | AUC   | Hit@10 | NDCG@10 | MAP@10 |
|------------------------|-------|--------|---------|--------|
| Popularity             | 0.63  | 0.64   | 0.35    | 0.27   |
| LogisticRegression     | 0.73  | 0.15   | 0.09    | 0.07   |
| NMF Ranker             | 0.54  | **0.87** | **0.66** | **0.60** |
| GradientBoosting       | 0.74  | 0.10   | 0.06    | 0.05   |
| MLP                    | 0.72  | 0.12   | 0.08    | 0.06   |

*Run `python run_baseline.py` to refresh these numbers; minor variation
is expected per environment.*

### Reading the table

- **Popularity already beats Logistic at ranking**. Logistic has higher
  AUC (0.73 vs 0.63) but much lower Hit@10 (0.15 vs 0.64). The textbook
  illustration of the AUC trap.
- **NMF beats every supervised model**. 0.87 Hit@10 in a 1-pos + 99-neg
  protocol is essentially production-grade on this small dataset, and
  without training anything against `y`.
- The NMF model has **low AUC** (0.54), nearly at chance. This confirms
  again that AUC and ranking measure different things; the NMF is
  mediocre at AUC but the strongest ranker.
- Gradient Boosting and the MLP have the highest AUC but the worst
  ranking. Same trap as Logistic, with more capacity to memorize
  population-level patterns.

## 5. Why classic supervised models fail

Three chained causes:

1. **Wrong loss**. Log-loss optimizes `P(liked | features)` over random
   pairs in the dataset. The correct ranking loss is BPR (pairwise) or
   lambdarank (listwise), which condition per user.
2. **Out-of-distribution at inference time**. In training we see movies
   with `movie_count > 0`. At inference the user competes against 99
   randomly sampled movies, many of which are "cold" (`movie_count = 0`
   or `1`). Trees do not extrapolate, and linear + StandardScaler get
   miscalibrated.
3. **Excessive weight on the bayesian feature `movie_avg`**. The
   Logistic learns "high movie_avg -> liked", but at inference all the
   true positives are movies with weaker history compared to a global
   top, and the model loses.

NMF dodges all three: it does not optimize a classification loss, it
does not rely on categorical features, and its personalized score
already encodes user-movie affinity without leaning on `movie_avg`.

## 6. Limitations

- **Small dataset**: 610 users, 9,742 movies. Numbers are indicative;
  with MovieLens-1M values would shift (the picture would not).
- **Explicit feedback**: we use 0.5-5.0 ratings. In a real Netflix-like
  recommender, ~95% of the signal is implicit (watched / not watched,
  watch time). Adapting the pipeline to implicit feedback is the natural
  productive direction.
- **Hard cold-start**: new users or movies have no NMF embedding; today
  we fill them with `global_mean`. A hybrid model that uses metadata
  (genre, year) when no history exists would handle this better.
- **No ranking loss**: this would be the next real jump. It would
  involve switching libraries (LightGBM with `objective=lambdarank` or
  implicit-feedback ALS).

## 7. Conclusion

The central lesson of the project:

> **The metric dictates the model.** If you evaluate on AUC, the
> Logistic with rich features looks reasonable. If you evaluate on
> Hit@K / NDCG@K — which is what a recommender does in production — the
> Logistic collapses and a simple matrix factorization wins.

From there, the productive approach is **not** to keep tuning supervised
models, but to start from NMF as the backbone and stack only what is
needed on top of it. Trying to make an MLP learn to rank from scratch
with handcrafted features is a poor use of effort; the right next step
is a pairwise / listwise ranking loss.
