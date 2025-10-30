import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

def L1_distance(P: np.ndarray, Q: np.ndarray) -> np.float64:
    return np.sum(abs(P - Q))

def L2_distance(P: np.ndarray, Q: np.ndarray) -> np.float64:
    return np.sqrt(np.sum((P - Q) ** 2))

def load_mnist(normalize=True, one_hot_label=True):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target']

    if normalize:
        X = X / 255.0

    if one_hot_label:
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return (X_train, y_train), (X_test, y_test)