from pandas import DataFrame, Series
import numpy as np

# Decision Stump (weak learner)
class Learner:
	def __init__(self):
		self.feature_index: int = None
		self.threshold: float = None
		self.polarity: int = 1

	def fit(self, X: DataFrame, y: Series, sample_weights: Series) -> None:
		n_samples, n_features = X.shape
		assert len(sample_weights) == n_samples, "Sample weights length must match number of samples."
		
		min_error = float('inf')
		
		for feature_i in range(n_features):
			feature_values = X.iloc[:, feature_i]
			thresholds = np.unique(feature_values)
			for threshold in thresholds:
				polarity = 1
				predictions = np.ones(n_samples)
				predictions[feature_values < threshold] = -1

				# Misclassification error weighted by sample weights
				error = np.sum(sample_weights * (predictions != y)) / np.sum(sample_weights)

				if error > 0.5:
					error = 1 - error
					polarity = -1

				if error < min_error:
					min_error = error
					self.feature_index = feature_i
					self.threshold = threshold
					self.polarity = polarity

	def predict(self, X: DataFrame) -> Series:
		if self.feature_index is None:
			raise ValueError("Learner has not been fitted yet.")

		feature_values = X.iloc[:, self.feature_index]
		predictions = np.ones(X.shape[0])
		predictions[feature_values < self.threshold] = -1
		return predictions * self.polarity
