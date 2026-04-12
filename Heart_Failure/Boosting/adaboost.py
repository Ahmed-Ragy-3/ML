from pandas import DataFrame, Series
from learner import Learner
import numpy as np

class AdaBoost:
	def __init__(self, iterations: int = 50):
		self.iterations: int = iterations
		self.weights: Series[float] = None
		
		# self.base_learner: Learner = base_learner
		self.alphas: list[float] = None 			# alpha
		self.learners: list[Learner] = None 	# h(x)

	def fit(self, X: DataFrame, y: Series) -> None:
		# assert set(np.unique(y)).issubset({-1, 1}), "Labels must be -1 or 1"
		n_samples = X.shape[0]
		self.weights = np.ones(n_samples) / n_samples  # Initialize weights uniformly
		
		self.alphas = []
		self.learners = []

		for _ in range(self.iterations):
			# Train base learner with current weights
			learner: Learner = Learner()
			learner.fit(X, y, sample_weights=self.weights)
			y_pred = learner.predict(X)

			error = self._err(y, y_pred)
			if error == 0:
				self.alphas.append(1.0)
				self.learners.append(learner)
				break

			alpha = self._alpha(error)
			self._update_weights(alpha, y, y_pred)

			# Store the learner and its alpha
			self.learners.append(learner)
			self.alphas.append(alpha)


	# H(x) = sign(Σ alpha_i * h_i(x))
	def predict(self, X: DataFrame) -> Series:
		if self.learners is None or self.alphas is None:
			raise ValueError("Model has not been fitted yet.")

		# Aggregate predictions from all learners
		final_prediction = np.zeros(X.shape[0])
		for alpha, learner in zip(self.alphas, self.learners):
			final_prediction += alpha * learner.predict(X)

		return np.where(final_prediction >= 0, 1, -1)
	
	def _err(self, y_true: Series, y_pred: Series) -> float:
		return np.sum(self.weights * (y_true != y_pred)) / np.sum(self.weights)
	
	def _alpha(self, error: float) -> float:
		return 0.5 * np.log((1 - error) / (error + 1e-10))  # add small value to avoid division by zero
		
	# Update weights: w_i = w_i * exp(-alpha * y_i * h_i(x_i))
	def _update_weights(self, alpha: float, y_true: Series, y_pred: Series) -> Series:
		self.weights *= np.exp(-alpha * (y_true * y_pred))
		# Normalized
		self.weights /= np.sum(self.weights) 