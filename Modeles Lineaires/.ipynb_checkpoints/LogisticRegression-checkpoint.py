import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(z))


class LogisticRegression:
    def __init__(self, lr: float = 0.001, n_iters: int = 1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iters = n_iters
        
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Training
        for _ in range(self.n_iters):
            # Predictions
            linear_predictions = X.dot(self.weights) + self.bias
            predictions = sigmoid(linear_predictions)
            # Gradiant
            dw = (1 / n_samples) * np.dot(X.T, predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y)
            # MAJ
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            
    def predict(self, X, *, threshold=0.5):
        linear_predictions = X.dot(self.weights) + self.bias
        predictions = sigmoid(linear_predictions)
        y_pred = [1 if y >= threshold else 0 for y in predictions]
        return np.array(y_pred)
        
        
            
            
            