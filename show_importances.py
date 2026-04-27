"""
show_importances.py
===================

Trains a Gradient Boosting classifier with the winning hyperparameters from
tune_classification.py and prints the top 15 most important features.
"""

import pandas as pd
from data_loader import load_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

print("Training Gradient Boosting to inspect feature importance...")
data = load_data(task="classification")

# Winning hyperparameters from the randomized search
model = GradientBoostingClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    subsample=1.0,
    random_state=42,
)
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", model),
])

pipe.fit(data.X_train, data.y_train)

importances = pipe.named_steps["model"].feature_importances_
df = pd.DataFrame({"Feature": data.feature_columns, "Importance": importances})
df = df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

print("\n=== Top 15 most important features ===")
print(df.head(15).to_string(index=False))
