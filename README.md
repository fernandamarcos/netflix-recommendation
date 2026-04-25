# Netflix / MovieLens recommender

Sistema de recomendacion sobre **MovieLens `ml-latest-small`** (100.836
ratings, 9.742 peliculas, 610 usuarios). Proyecto de *Business with Data*.

El objetivo NO es predecir el rating exacto (regresion), sino construir un
buen **top-10 por usuario**. La metrica principal es **Hit@10 / NDCG@10 /
MAP@10**, no RMSE ni AUC.

---

## Estructura

```
netflix-recommendation/
├── ml-latest-small/        dataset (ratings.csv, movies.csv)
├── data_loader.py          carga + features + split leave-last-out
├── baselines.py            PopularityBaseline (sin personalizar)
├── models.py                NMFRanker, NMFStackedMLP, Logistic / Linear
├── evaluation.py           metricas de ranking (P@K, R@K, NDCG@K, Hit@K, MAP@K)
├── run_baseline.py         pipeline end-to-end (carga → entrena → evalua → tabla)
├── requirements.txt
├── README.md
├── report.md               informe con hallazgos, "AUC trap" y comparativa
└── legacy/                 codigo v1 y scripts EDA
```

---

## Los 4 modelos

| # | Modelo              | Tipo                        | Por que esta ahi                    |
|---|---------------------|-----------------------------|--------------------------------------|
| 1 | **Popularity**      | Baseline no personalizado    | Suelo. Todo modelo "con ML" DEBE batirlo. |
| 2 | **LogisticRegression** | Lineal supervisado        | ML clasico con 29 features. Representa el enfoque tradicional "AUC maximizer". |
| 3 | **NMF Ranker**      | Factorizacion matricial     | Nuestro ganador. Lee el score reconstruido por NMF y rankea por el. |
| 4 | **NMF + NN**        | Red neuronal pequenna stacked | Toma `nmf_score` + 3 features de contexto y aprende a combinarlos. |

La MLP stacked es intencionalmente diminuta: 1 capa oculta de 16 neuronas, 4
features (`nmf_score`, `movie_avg`, `movie_count`, `user_avg`). Con esto
evitamos el error que cometimos antes de darle las 29 features: el modelo
grande caia en el "AUC trap" y desaprendia a rankear. Aqui restringimos la
senal y la red solo tiene que reponderar lo que el NMF ya hace bien.

---

## Pipeline de datos

1. **Carga** `ratings.csv` + `movies.csv`.
2. **Movie features**: dummies de genero, `year` extraido del titulo.
3. **Split leave-last-out**: el rating mas reciente de cada usuario va a
   test. Evita fuga temporal.
4. **Stats solo con train**: `user_avg`, `user_count`, `movie_avg`,
   `movie_count`, `global_mean`.
5. **User-user similarity vectorizado** sobre la matriz user x movie.
6. **NMF latent factors** (k=50) sobre `user_movie.fillna(0)`.
7. **Cold-start**: missing → `global_mean` / `0`.
8. **Target**:
   - `classification`: `y = (rating >= 4).astype(int)`.
   - `regression`: `y = rating` (0.5 - 5.0).

Los lookups de `user_sim_score` y `nmf_score` se guardan **completos** (todas
las parejas user-movie con score calculable). Esto es clave: sin ellos, al
scorear peliculas no vistas las features colapsarian a una constante y el
ranking dejaria de tener sentido.

---

## Evaluacion

Metricas de ranking calculadas por usuario y promediadas:

- **Hit@10** — 1 si alguna peli relevante (test) aparece en el top-10.
- **NDCG@10** — DCG/IDCG con relevancia graduada (usa el rating real).
- **MAP@10** — average-precision medio.
- **Precision@10 / Recall@10**.

`evaluate_ranker(model, data, k=10, candidate_strategy=...)` soporta:

- `test_plus_sample` (default): 1 positivo + 99 negativas aleatorias.
  Protocolo estandar leave-one-out en papers de recsys.
- `unseen`: rankea TODAS las peliculas no vistas por el usuario.

---

## Uso

```bash
pip install -r requirements.txt

# Pipeline completo: 4 modelos, tabla final
python run_baseline.py

# Regresion (rating 0.5-5.0)
python run_baseline.py --task regression

# Mas usuarios, top-20, candidatos = todas las no vistas
python run_baseline.py --k 20 --max-users 200 --candidates unseen

# Solo un subconjunto
python run_baseline.py --only "Popularity,NMF Ranker,NMF + NN"
```

Uso programatico:

```python
from data_loader import load_data, build_candidate_features
from models import build_models
from evaluation import evaluate_ranker

data = load_data(task="classification")
models = build_models("classification")

nn = models["NMF + NN"]
nn.fit(data.X_train, data.y_train)

metrics = evaluate_ranker(nn, data, k=10, max_users=100)
print(metrics)  # {'hit@10': ..., 'ndcg@10': ..., ...}
```

---

## Reproducibilidad

Semillas fijadas a 42: NMF, MLP, muestreo negativo. El split leave-last-out
es deterministico (ordena por timestamp).

---

## Hallazgos

Ver `report.md`. Resumen:

> En top-K, **NMF puro** bate con diferencia al supervisado clasico
> (Logistic). Un NN pequenno sobre `nmf_score` intenta mejorarlo y cierra
> la historia. La leccion es el **AUC trap**: optimizar probabilidad
> global es otra tarea distinta a rankear.
