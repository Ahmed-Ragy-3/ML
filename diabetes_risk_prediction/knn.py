from collections import Counter

import numpy as np

from diabetes_risk_prediction.classifier import Classifier

class KNN(Classifier):
	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		predicted_labels = [self._predict(x) for x in X]
		return np.array(predicted_labels)

	def _predict(self, x):
		# Compute distances between x and all examples in the training set
		distances = np.linalg.norm(self.X_train - x, axis=1)
		# Sort by distance and return indices of the first k neighbors
		k_indices = np.argsort(distances)[:self.k]
		# Extract the labels of the k nearest neighbor
		k_nearest_labels = [self.y_train[i] for i in k_indices]
		# Return the most common class label among the neighbors
		most_common = Counter(k_nearest_labels).most_common(1)
		return most_common[0][0]
