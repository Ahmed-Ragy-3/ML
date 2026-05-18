import numpy as np
from sklearn.cluster import KMeans

class GMM:
    def __init__(self, n_components=10, max_iter=50, tol=1e-4, reg_covar=1e-6, random_state=42):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def gaussian_log_pdf(self, X, mean, cov_inv, log_det):
        d = X.shape[1]
        diff = X - mean
        maha = np.sum((diff @ cov_inv) * diff, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

    def initialize(self, X):
        n, d = X.shape

        kmeans = KMeans(
            n_clusters=self.K,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(X)

        self.means = kmeans.cluster_centers_

        Nk = np.bincount(labels, minlength=self.K).astype(float)
        self.weights = Nk / n

        self.covariances = np.zeros((self.K, d, d))

        for k in range(self.K):
            diff = X - self.means[k]
            cov = (diff.T @ diff) / n
            cov += self.reg_covar * np.eye(d)
            self.covariances[k] = cov

    def e_step(self, X):
        n, d = X.shape
        log_resp = np.zeros((n, self.K))

        for k in range(self.K):
            cov = self.covariances[k] + self.reg_covar * np.eye(d)

            inv_cov = np.linalg.inv(cov)
            sign, log_det = np.linalg.slogdet(cov)

            log_resp[:, k] = np.log(self.weights[k] + 1e-12) + \
                self.gaussian_log_pdf(X, self.means[k], inv_cov, log_det)

        log_resp -= np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= np.sum(resp, axis=1, keepdims=True)

        return resp

    def m_step(self, X, R):
        n, d = X.shape
        Nk = R.sum(axis=0) + 1e-12

        self.weights = Nk / n
        self.means = (R.T @ X) / Nk[:, None]

        for k in range(self.K):
            diff = X - self.means[k]
            weighted = R[:, k][:, None] * diff

            cov = (weighted.T @ diff) / Nk[k]
            cov += self.reg_covar * np.eye(d)

            self.covariances[k] = cov

    def fit(self, X):
        self.initialize(X)

        for _ in range(self.max_iter):
            R = self.e_step(X)
            self.m_step(X, R)

    def predict(self, X):
        return np.argmax(self.e_step(X), axis=1)