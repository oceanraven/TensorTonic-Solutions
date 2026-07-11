import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)

    N, dim = X.shape

    w = np.zeros(dim)
    b = 0.0

    for step in range(steps):
        z = np.dot(X, w) + b
        
        p = _sigmoid(z)
        
        dw = np.dot(X.T, (p - y)) / N
        db = np.mean(p - y)

        w -= lr * dw
        b -= lr * db
    
    return w, b