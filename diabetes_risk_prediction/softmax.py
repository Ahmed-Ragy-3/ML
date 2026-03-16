import numpy as np
from diabetes_risk_prediction.classifier import Classifier

class Softmax(Classifier):
	def __init__(self, learning_rate=0.01, n_iters=1000):
		self.learning_rate = learning_rate
		self.n_iters = n_iters
		self._is_fitted = False

	def fit(self, X, y):
		n_samples, n_features = X.shape
		n_classes = len(set(y))
		self._weights = np.zeros((n_features, n_classes))
		self._biases = np.zeros(n_classes)

		for _ in range(self.n_iters):
			scores = np.dot(X, self._weights) + self._biases
			probs = self._softmax(scores)
			probs[range(n_samples), y] -= 1
			dW = np.dot(X.T, probs) / n_samples
			db = np.sum(probs, axis=0) / n_samples
			self._weights -= self.learning_rate * dW
			self._biases -= self.learning_rate * db

		self._is_fitted = True

	def predict(self, X):
		if not self._is_fitted:
			raise Exception("Model is not fitted yet.")
		scores = np.dot(X, self._weights) + self._biases
		probs = self._softmax(scores)
		return np.argmax(probs, axis=1)

	def _softmax(self, scores):
		exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
		return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)