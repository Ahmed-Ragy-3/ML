from collections import Counter
import numpy as np
from classifier import Classifier
from dataset import Dataset

class KNN(Classifier):
	def __init__(self, dataset: Dataset, dist='euclidean_distance', k=3):
		self.k = k
		self.dist = dist
		self.dataset: Dataset = dataset

	def set_k(self, k):
		self.k = k
	
	def set_dist(self, dist):
		self.dist = dist

	def predict(self, X):
		y_pred = [self._predict(x) for x in X]
		return np.array(y_pred)

	def _predict(self, x):
		if self.dist == 'euclidean_distance':
			distances = np.linalg.norm(self.dataset.x_train - x, axis=1)
		else:
			distances = np.sum(np.abs(self.dataset.x_train - x), axis=1)

		k_indices = np.argpartition(distances, self.k)[:self.k]
		k_nearest_labels = self.dataset.y_train[k_indices]

		return Counter(k_nearest_labels).most_common(1)[0][0]