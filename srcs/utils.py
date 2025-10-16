import numpy as np

def L1_distance(P: np.ndarray, Q: np.ndarray) -> np.float64:
    return np.sum(abs(P - Q))

def L2_distance(P: np.ndarray, Q: np.ndarray) -> np.float64:
    return np.sqrt(np.sum((P - Q) ** 2))