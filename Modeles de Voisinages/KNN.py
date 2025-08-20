import numpy as np
from collections import Counter

# -------- DISTANCES --------
def compute_distance(X_train, x, metric="euclidean", p=3):
    if metric == "euclidean":
        return np.linalg.norm(X_train - x, axis=1)
    elif metric == "manhattan":
        return np.sum(np.abs(X_train - x), axis=1)
    elif metric == "chebyshev":
        return np.max(np.abs(X_train - x), axis=1)
    elif metric == "minkowski":
        return np.sum(np.abs(X_train - x)**p, axis=1)**(1/p)
    elif metric == "cosine":
        num = np.dot(X_train, x)
        denom = np.linalg.norm(X_train, axis=1) * np.linalg.norm(x)
        return 1 - (num / (denom + 1e-10))  # éviter division par zéro
    elif metric == "hamming":
        return np.mean(X_train != x, axis=1)
    else:
        raise ValueError("Unknown distance metric")


# -------- CLASSIFICATION --------
class KNNClassifier:
    def __init__(self, k=3, metric="euclidean", weighted=False, p=3):
        self.k = k
        self.metric = metric
        self.weighted = weighted  # pondération par distance
        self.p = p

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = compute_distance(self.X_train, x, self.metric, self.p)
        knn_indices = np.argsort(distances)[:self.k]
        knn_labels = self.y_train[knn_indices]

        if self.weighted:
            weights = 1 / (distances[knn_indices] + 1e-5)
            label_score = {}
            for label, w in zip(knn_labels, weights):
                label_score[label] = label_score.get(label, 0) + w
            return max(label_score, key=label_score.get)
        else:
            return Counter(knn_labels).most_common(1)[0][0]


# -------- REGRESSION --------
class KNNRegressor:
    def __init__(self, k=3, metric="euclidean", weighted=False, p=3):
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.p = p

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = compute_distance(self.X_train, x, self.metric, self.p)
        knn_indices = np.argsort(distances)[:self.k]
        knn_values = self.y_train[knn_indices]

        if self.weighted:
            weights = 1 / (distances[knn_indices] + 1e-5)
            return np.average(knn_values, weights=weights)
        else:
            return np.mean(knn_values)
