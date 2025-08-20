import numpy as np
from collection import Counter

def distance(x_train, x, distance = "euclidian"):
	if distance == "euclidian"

class KNNClassifier:
	def __init__(self, k=3, distance="euclidian"):
		self.k = k
		self.distance = distance

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y_train

	def predict(self, X):
		predictions = [self._predict(x) for x in X]
		return np.array(predictions)

	def _predict(self, x):
		distances = [distance(x_train, x, self.distance)for x_train in X_train]
		
		