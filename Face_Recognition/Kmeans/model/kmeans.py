import numpy as np


class KMeans:
    def __init__(self, n_clusters=40, max_iter=300, tol=1e-4, random_state=42):
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.cost_ = None

    #  Initialization
    def _initialize_centroids(self, X):
        """Random initialization: pick K distinct data points as centroids."""
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.K, replace=False)
        return X[indices].copy()

    #  E-step: Assignment
    def _assign_clusters(self, X):
        n = X.shape[0]
        distances = np.zeros((n, self.K))

        for k in range(self.K):
            diff = X - self.centroids[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)  # Distance Matrix

        return np.argmin(distances, axis=1)  # Assignment Vector

    #  M-step: Update
    def _update_centroids(self, X, labels):
        """
        Recompute each centroid as the mean of all points assigned to it.
        """
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.K):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = self.centroids[k]
        return new_centroids

    def _compute_cost(self, X, labels):
        cost = 0.0
        for k in range(self.K):
            mask = labels == k
            if np.sum(mask) > 0:
                cost += np.sum((X[mask] - self.centroids[k]) ** 2)
        return cost

    def fit(self, X):
        """Fit K-Means to data X of shape (n, d)."""
        self.centroids = self._initialize_centroids(X)
        prev_labels = None

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, labels)

            # Check convergence: assignments did not change
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                print(f"  K-Means converged at iteration {iteration + 1}")
                break
            prev_labels = labels

        self.labels_ = self._assign_clusters(X)
        self.cost_ = self._compute_cost(X, self.labels_)
        return self

    def predict(self, X):
        """Assign new data points to the nearest centroid."""
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_