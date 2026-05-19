import numpy as np


class SimpleAutoencoder:
    def __init__(self, input_dim, reduced_dim, lr=1e-3, random_state=42):
        """
        Parameters
        ----------
        input_dim   : dimensionality of each input vector (D)
        reduced_dim : bottleneck / code dimensionality (K < D)
        lr          : learning rate for gradient descent
        """
        rng = np.random.default_rng(random_state)
        scale = np.sqrt(1.0 / input_dim)

        # Encoder  W1: (K x D),  b1: (K,)
        self.W1 = (rng.standard_normal((reduced_dim, input_dim)) * scale).astype(np.float32)
        self.b1 = np.zeros(reduced_dim, dtype=np.float32)

        # Decoder  W2: (D x K),  b2: (D,)
        self.W2 = (rng.standard_normal((input_dim, reduced_dim)) * scale).astype(np.float32)
        self.b2 = np.zeros(input_dim, dtype=np.float32)

        self.lr = lr
        self.train_losses = []

    @staticmethod
    def _normalise(X):
        X = X.astype(np.float32)
        mx = X.max()
        return X / mx if mx > 1.0 else X

    # Forward
    def encode(self, X):
        return X @ self.W1.T + self.b1

    def decode(self, Z):
        return Z @ self.W2.T + self.b2

    def forward(self, X):
        return self.decode(self.encode(X))

    @staticmethod
    def mse_loss(X, X_hat):
        return np.mean((X - X_hat) ** 2)

    # Backward
    def _backward_and_update(self, X, X_hat, Z, clip=5.0):
        n = X.shape[0]

        dL = 2.0 * (X_hat - X) / n

        # Decoder gradients
        dW2 = dL.T @ Z
        db2 = dL.sum(axis=0)
        dZ  = dL @ self.W2

        # Encoder gradients
        dW1 = dZ.T @ X
        db1 = dZ.sum(axis=0)

        # Gradient clipping — prevents exploding gradients
        for g in (dW2, db2, dZ, dW1, db1):
            np.clip(g, -clip, clip, out=g)

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ── fit ───────────────────────────────────────────────────────────────
    def fit(self, X, epochs=100, batch_size=32, verbose=True):
        # Normalise once before training
        X = self._normalise(X)

        n   = X.shape[0]
        rng = np.random.default_rng(0)

        for epoch in range(1, epochs + 1):
            idx          = rng.permutation(n)
            epoch_loss   = 0.0
            nb           = 0

            for s in range(0, n, batch_size):
                batch = X[idx[s:s + batch_size]]

                # Forward
                Z     = self.encode(batch)
                X_hat = self.decode(Z)

                # Loss
                epoch_loss += self.mse_loss(batch, X_hat)
                nb         += 1

                # Backward + update
                self._backward_and_update(batch, X_hat, Z)

            avg = epoch_loss / nb
            self.train_losses.append(avg)

            if verbose and (epoch == 1 or epoch % 20 == 0):
                print(f"  Epoch {epoch:>4}/{epochs}  MSE Loss: {avg:.6f}")

        return self

    def transform(self, X):
        return self.encode(self._normalise(X))

    def reconstruct(self, X):
        return self.forward(self._normalise(X))