import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=SEED),
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=SEED),
    "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
}

results = []
for name, m in models.items():
    m.fit(X_train_scaled, y_train)
    y_pred = m.predict(X_test_scaled)
    f1 = f1_score(y_pred, y_test, average="macro")
    results.append({"model": name, "f1_macro": round(f1, 4)})

df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
print(df.to_string(index=False))
df.to_csv("outputs/model_comparison.csv", index=False)
print("\nSaved to outputs/model_comparison.csv")