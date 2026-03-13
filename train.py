# ── Imports ──────────────────────────────────────────────
import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Reproducibility ──────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Load Data ─────────────────────────────────────────────
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
print("Dataset shape:", X.shape)
print("Classes:", iris.target_names)

# ── Split Data ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ── Preprocessing ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Train size:", X_train_scaled.shape)
print("Test size:", X_test_scaled.shape)

# ── Model ─────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=SEED
)
model.fit(X_train_scaled, y_train)
print("Model trained!")

# ── Evaluation ────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Confusion Matrix Plot ─────────────────────────────────
os.makedirs("outputs", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to outputs/")

# ── Feature Importance ────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=iris.feature_names)
importances.sort_values().plot(kind="barh", figsize=(6, 3), title="Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()
print("Feature importance saved to outputs/")

# ── Save Training Log ─────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

with open("outputs/train_log.txt", "w") as f:
    f.write(f"seed={SEED}\n")
    f.write(f"n_estimators=100, max_depth=4\n")
    f.write(f"val_accuracy={acc:.4f}\n")
    f.write(f"val_f1_macro={f1:.4f}\n")
    f.write(f"checkpoint=random_forest_seed42\n")

print(f"\nFinal Results — Accuracy: {acc:.4f} | F1: {f1:.4f}")