"""
The PCA Algorithm Steps

- Standardize the Data: 
		Standardize features (mean=0, variance=1) to ensure all features contribute equally.

- Calculate Covariance Matrix: 
		Determine how variables in the input dataset vary from the mean with respect to each other.

- Calculate Eigenvectors/Eigenvalues: 
		Compute the eigenvalues and eigenvectors of the covariance matrix to identify principal components.

- Sort and Select Components: 
		Sort eigenvalues in descending order and select top k eigenvectors (principal components).

- Transform Data: 
		Project the original data onto the new components to obtain a lower-dimensional dataset.
"""
import numpy as np

class PCA:
   def __init__(self, n_components=None):
      self.n_components = n_components
      self.components = None
      self.mean = None
      self.std = None

   def _standardize(self, X):
      return (X - self.mean) / self.std

   def _fit(self, X):
      self.mean = np.mean(X, axis=0)
      self.std = np.std(X, axis=0, ddof=1)
      Z = self._standardize(X)

      cov_matrix = np.cov(Z, rowvar=False)

      eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

      idxs = np.argsort(eigenvalues)[::-1]
      eigenvalues = eigenvalues[idxs]
      eigenvectors = eigenvectors[:, idxs]

      if self.n_components is not None:
         eigenvectors = eigenvectors[:, :self.n_components]

      self.components = eigenvectors

   def _transform(self, X):
      Z = self._standardize(X)
      return np.dot(Z, self.components)

   def fit_transform(self, X, path=None):
      """Fit the model with X and apply dimensionality reduction."""
      if path is not None:
         data = np.load(path, allow_pickle=True).item()
         self.components = data["components"]
         self.mean = data["mean"]
         self.std = data["std"]
      else:
         self._fit(X)

      return self._transform(X)

   def reconstruct(self, Z):
      return np.dot(Z, self.components.T) * self.std + self.mean

   def save_components(self, path):
      np.save(path, {"components": self.components,
                     "mean": self.mean,
                     "std": self.std})
