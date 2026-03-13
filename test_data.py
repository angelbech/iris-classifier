import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

def test_split_sizes():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    assert X_train.shape == (120, 4), "Train size wrong"
    assert X_test.shape == (30, 4), "Test size wrong"
    print("test_split_sizes passed ✅")

def test_scaler_mean():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    mean = np.abs(X_scaled.mean(axis=0)).max()
    assert mean < 1e-10, "Scaler did not zero-center data"
    print("test_scaler_mean passed ✅")

if __name__ == "__main__":
    test_split_sizes()
    test_scaler_mean()