from pandas import DataFrame, Series
import numpy as np

class Learner:
	def fit(self, X: DataFrame, y: Series, sample_weight: Series) -> None:
		pass
	
	def predict(self, X: DataFrame) -> Series:
		pass

class AdaBoost:
	def __init__(self, base_learner: Learner, iterations: int = 50):
		self.iterations: int = iterations
		self.weights: Series[float] = None
		
		self.base_learner: Learner = base_learner
		self.alphas: Series[float] = None 	# alpha
		self.learners: Series[Learner] = None 	# h(x)

	def fit(self, X: DataFrame, y: Series) -> None:
		n_samples = X.shape[0]
		self.weights = np.ones(n_samples) / n_samples  # Initialize weights uniformly
		# self.alphas = []
		# self.learners = []

		for _ in range(self.iterations):
			# Train base learner with current weights
			learner: Learner = self.base_learner()
			learner.fit(X, y, sample_weight=self.weights)
			y_pred = learner.predict(X)

			error = self._err(y, y_pred)
			alpha = self._alpha(error)

			self._update_weights(alpha, y, y_pred)

			# Store the learner and its alpha
			self.learners = self.learners.append(learner) if self.learners is not None else [learner]
			self.alphas = self.alphas.append(alpha) if self.alphas is not None else [alpha]


	# H(x) = sign(Σ alpha_i * h_i(x))
	def predict(self, X: DataFrame) -> Series:
		if self.learners is None or self.alphas is None:
			raise ValueError("Model has not been fitted yet.")

		# Aggregate predictions from all learners
		final_prediction = np.zeros(X.shape[0])
		for alpha, learner in zip(self.alphas, self.learners):
			final_prediction += alpha * learner.predict(X)

		return np.sign(final_prediction)
	
	def _err(self, y_true: Series, y_pred: Series) -> float:
		return np.sum(self.weights * (y_true != y_pred)) / np.sum(self.weights)
	
	def _alpha(self, error: float) -> float:
		return 0.5 * np.log((1 - error) / (error + 1e-10))  # add small value to avoid division by zero
		
	# Update weights: w_i = w_i * exp(-alpha * y_i * h_i(x_i))
	def _update_weights(self, alpha: float, y_true: Series, y_pred: Series) -> Series:
		self.weights *= np.exp(-alpha * (y_true * y_pred))
		# Normalized
		# self.weights /= np.sum(self.weights) 