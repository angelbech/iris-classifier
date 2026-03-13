import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

iris = load_iris()
X, y = iris.data, iris.target

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=4, random_state=SEED),
    X, y, cv=5, scoring="f1_macro",
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=SEED
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(7, 4))
plt.plot(train_sizes, train_mean, label="Training F1")
plt.plot(train_sizes, val_mean, label="Validation F1")
plt.xlabel("Training samples")
plt.ylabel("F1 Macro")
plt.title("Learning Curve — Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/learning_curve.png")
plt.close()
print("Learning curve saved to outputs/learning_curve.png")