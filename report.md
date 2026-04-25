# Informe - Sistema de recomendacion sobre MovieLens

## 1. Problema

Dado un usuario, devolver una **lista top-10 de peliculas** que no ha visto
y que tenga alta probabilidad de que le gusten.

Dataset: **MovieLens `ml-latest-small`** — 100.836 ratings de 610 usuarios
sobre 9.742 peliculas, ratings de 0.5 a 5.0.

## 2. El framing correcto: ranking, no clasificacion

Empezamos con dos framings "obvios":

1. **Regresion**: predecir el rating exacto (RMSE).
2. **Clasificacion binaria**: `y = 1 si rating >= 4` (AUC).

Ambos daban numeros "decentes" en abstracto (AUC ~ 0.73, RMSE ~ 0.98) pero
cuando medimos lo que importa — **Hit@10 en un ranking top-10 por usuario** —
se desplomaban.

### El "AUC trap"

AUC alto NO implica buen recomendador. AUC mide si, entre todos los pares
(positivo, negativo) del test, el modelo puntua el positivo mas alto que el
negativo *en general*. No condiciona por usuario. En recsys real el
objetivo es: "para **este** usuario, pon sus relevantes en las primeras
posiciones *de entre sus candidatos*". Es una tarea distinta.

Consecuencia: un Logistic entrenado sobre las 29 features saca AUC = 0.73
pero **Hit@10 = 0.15** (menor que el azar, que seria 0.10 con 1 positivo
entre 100 candidatos, y muy lejos del 0.87 que saca el NMF puro).

Decision: dejamos de optimizar AUC/RMSE, movemos la brujula a Hit@10 /
NDCG@10 / MAP@10 y estructuramos el proyecto alrededor de ranking.

## 3. Los 4 modelos

### 3.1. Popularity (baseline)

Rankea todas las peliculas por media bayesiana (formula IMDb Top 250):

```
score(peli) = (avg_rating * n + global_mean * m) / (n + m),  m = 10
```

No personalizado: todos los usuarios reciben la misma lista. Sirve de
**suelo**: cualquier modelo con ML que no lo bata claramente no aporta
valor.

### 3.2. LogisticRegression (ML clasico)

Pipeline `SimpleImputer → StandardScaler → LogisticRegression(C=1.0)` sobre
las **29 features**: stats de usuario, stats de pelicula, dummies de
genero, year, user_sim_score, nmf_score.

Es el representante del enfoque "ML supervisado con features ricas" que se
ensena en la asignatura. Como veremos, falla en la tarea real.

### 3.3. NMFRanker (matrix factorization)

Descomposicion no negativa de la matriz user x movie:

```
R ≈ W · H,     W ∈ R^(u × k),  H ∈ R^(k × m),  k = 50
```

Rellenamos missings con 0, entrenamos, reconstruimos la matriz completa y
rankeamos por el score reconstruido. **No hay (X, y)**: el "modelo" es el
propio NMF ya ajustado sobre train.

Equivalentemente: es un collaborative filtering pre-entrenado. En nuestro
dataset es el mas fuerte.

### 3.4. NMF + NN (stacked MLP)

Red neuronal **pequenna** que toma:

```
features = [nmf_score, movie_avg, movie_count, user_avg]
arquitectura = MLP(hidden=(16,), relu, adam, alpha=1e-3, early_stopping)
```

Intenta **mejorar** al NMF puro aprendiendo a recombinarlo con contexto:
si la peli es poco popular (movie_count bajo), quizas hay que ser mas
conservador; si el usuario es muy exigente (user_avg bajo), quizas hay que
bajar la prediccion.

Por que solo 4 features y no las 29 del pipeline? Porque probamos con las
29 (el MLP "completo") y cayo en el mismo pozo que el Logistic: optimizaba
clasificacion global y perdia senal de ranking. Restringiendo la entrada a
los 4 ejes que correlacionan con top-K, la red solo tiene que ponderarlos
— problema mucho mas pequenno y menos propenso a romperse.

## 4. Resultados

Cifras en `classification`, `K=10`, `max_users=100`, candidatos
`test_plus_sample`:

| Modelo                  | AUC   | Hit@10 | NDCG@10 | MAP@10 |
|-------------------------|-------|--------|---------|--------|
| Popularity              | 0.63  | 0.64   | 0.35    | 0.27   |
| LogisticRegression      | 0.73  | 0.15   | 0.09    | 0.07   |
| NMF Ranker              | 0.54  | **0.87** | **0.66** | **0.60** |
| NMF + NN (stacked)      | *    | *      | *       | *      |

*Las cifras del NMF + NN dependen de la ultima ejecucion con la version
simplificada del MLP — rellenar tras correr `run_baseline.py`.*

### Lecturas

- **La popularidad ya bate a la Logistic en ranking**. El Logistic tiene
  AUC mas alto (0.73 vs 0.63) pero Hit@10 mucho menor (0.15 vs 0.64). La
  ilustracion perfecta del AUC trap.
- **El NMF gana a todos**. 0.87 de Hit@10 en un protocolo 1-pos + 99-neg
  es practicamente production-grade en este dataset pequenno. Y sin
  entrenar nada sobre `y`.
- El NMF tiene **AUC bajo** (0.54) — casi al nivel del azar. Esto confirma
  de nuevo que las dos metricas estan midiendo cosas distintas; en AUC el
  NMF es mediocre, en ranking es el mejor.
- El objetivo del **NMF + NN** es ver si anadir un clasificador pequenno
  encima del score del NMF mejora en Hit@10 / NDCG. Los numeros de esta
  ultima fila se rellenan tras ejecutar.

## 5. Por que fallan los supervisados clasicos

Tres causas encadenadas:

1. **Loss equivocada**. Log-loss optimiza `P(liked | features)` sobre pares
   aleatorios del dataset. La loss de ranking correcta seria BPR (pairwise)
   o lambdarank (listwise), que condicionan por usuario.
2. **Out-of-distribution en inferencia**. En train vemos peliculas con
   `movie_count > 0`. En inferencia el usuario compite contra 99 peliculas
   muestreadas al azar, muchas de las cuales son "frias" (`movie_count = 0`
   o 1). Los arboles no extrapolan y lineales + StandardScaler se descalibran.
3. **Fuerza excesiva de la feature bayesiana `movie_avg`**. El Logistic
   aprende que "movie_avg alto → liked", pero en inferencia todas las
   positivas reales son peliculas con historia flojita comparadas con un
   top global, y pierde.

El NMF esquiva las tres cosas: no optimiza loss de clasificacion, no tiene
features categorizadas, y su score personalizado ya coditza la "afinidad"
user-movie sin recurrir a `movie_avg`.

## 6. Limitaciones

- **Dataset pequenno**: 610 usuarios, 9.742 peliculas. Numeros
  orientativos; con MovieLens-1M cambiarian valores (pero no la foto).
- **Feedback explicito**: usamos ratings 0.5-5.0. En un recomendador real
  de Netflix, el 95% de la senal es implicita (visto / no visto, tiempo de
  visualizacion). Adaptar el pipeline a feedback implicito es la direccion
  "productiva" natural.
- **Cold-start duro**: usuarios o peliculas nuevas no tienen embedding NMF;
  hoy se los rellena con `global_mean`. Una solucion seria un modelo
  hibrido que use metadata (genero, year) cuando no hay historia.
- **Sin loss de ranking**: seria el siguiente salto real. Ya implica
  cambiar de libreria (LightGBM + `objective=lambdarank` o implicit-feedback
  ALS).

## 7. Conclusion

La leccion central del proyecto:

> **La metrica dicta el modelo.** Si evalua AUC, el Logistic con features
> ricas parece razonable. Si evaluamos Hit@K / NDCG@K — que es lo que un
> recomendador hace en produccion — el Logistic se hunde y una
> factorizacion matricial sencilla barre.

A partir de ahi, el enfoque productivo **no es** seguir afinando
supervisados, sino partir del NMF como columna vertebral y apilar encima
lo minimo para ganarle un poco mas (el NMF + NN). Intentar que un MLP
aprenda desde cero a rankear con 29 features es perder el tiempo.
