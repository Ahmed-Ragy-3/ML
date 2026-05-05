import numpy as np


class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=42):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def gaussian_pdf(self, X, mean, cov):
        n, d = X.shape

        cov = cov + self.reg_covar * np.eye(d)
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_cov)

        diff = X - mean
        exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)

        return norm_const * np.exp(exponent)

    def initialize(self, X):
        np.random.seed(self.random_state)

        n, d = X.shape

        random_idx = np.random.choice(n, self.K, replace=False)
        self.means = X[random_idx]

        base_cov = np.cov(X.T) + self.reg_covar * np.eye(d)

        self.covs = np.array([base_cov.copy() for _ in range(self.K)])

        self.weights = np.ones(self.K) / self.K

    def e_step(self, X):
        n = X.shape[0]

        responsibilities = np.zeros((n, self.K))

        for k in range(self.K):
            responsibilities[:, k] = self.weights[k] * self.gaussian_pdf(
                X, self.means[k], self.covs[k]
            )

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        return responsibilities

    def m_step(self, X, R):
        n, d = X.shape

        Nk = R.sum(axis=0)

        self.weights = Nk / n

        self.means = (R.T @ X) / Nk[:, None]

        self.covs = []

        for k in range(self.K):
            diff = X - self.means[k]

            cov = (R[:, k][:, None] * diff).T @ diff / Nk[k]

            cov += self.reg_covar * np.eye(d)

            self.covs.append(cov)

        self.covs = np.array(self.covs)

    def compute_log_likelihood(self, X):
        n = X.shape[0]

        total = np.zeros((n, self.K))

        for k in range(self.K):
            total[:, k] = self.weights[k] * self.gaussian_pdf(
                X, self.means[k], self.covs[k]
            )

        return np.sum(np.log(total.sum(axis=1)))

    def fit(self, X):
        self.initialize(X)

        prev_ll = None

        for iteration in range(self.max_iter):

            R = self.e_step(X)

            self.m_step(X, R)

            ll = self.compute_log_likelihood(X)

            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_ll = ll

    def predict(self, X):
        R = self.e_step(X)
        return np.argmax(R, axis=1)

    def predict_proba(self, X):
        return self.e_step(X)