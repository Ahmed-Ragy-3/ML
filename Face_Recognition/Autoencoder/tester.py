"""
Autoencoder Tester (Bonus Section)

Pipeline:
  1. Normalise face data (min-max to [0,1])
  2. Train Autoencoder  → latent representations
  3. Apply K-Means & GMM on the latent space for K = 20, 40, 60
  4. Compare accuracies with PCA-based results
  5. Visualise reconstructed faces
  6. Plot training loss curve
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoencoder import Autoencoder
from kmeans     import KMeans

# ── try importing GMM from the project's GMM module, else use local gmm.py ──
try:
    from GMM.model.gmm import GMM
    _gmm_source = "project"
except ImportError:
    try:
        from gmm import GMM
        _gmm_source = "local"
    except ImportError:
        GMM = None
        _gmm_source = None
        print("WARNING: GMM module not found – only K-Means will run on AE latent space.")


class StableGMM:
    """
    Log-space diagonal-covariance GMM — numerically stable for
    high-dimensional latent spaces where full-covariance det overflows.
    """
    def __init__(self, n_components=40, max_iter=50, reg=1e-3, random_state=42):
        self.K = n_components
        self.max_iter = max_iter
        self.reg = reg
        self.random_state = random_state

    def initialize(self, X):
        np.random.seed(self.random_state)
        n, d = X.shape
        idx = np.random.choice(n, self.K, replace=False)
        self.means = X[idx].copy()
        self.vars  = np.tile(X.var(axis=0) + self.reg, (self.K, 1))  # (K, d)
        self.weights = np.ones(self.K) / self.K

    def _log_gauss(self, X):
        """Return log N(x | mu_k, diag(var_k))  shape (n, K)"""
        n, d = X.shape
        log_probs = np.zeros((n, self.K))
        for k in range(self.K):
            v = self.vars[k] + self.reg
            log_det = np.sum(np.log(v))
            diff = X - self.means[k]
            maha = np.sum(diff ** 2 / v, axis=1)
            log_probs[:, k] = -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
        return log_probs

    def e_step(self, X):
        log_p = self._log_gauss(X) + np.log(self.weights + 1e-300)
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def m_step(self, X, R):
        n, d = X.shape
        Nk = R.sum(axis=0) + 1e-10
        self.weights = Nk / n
        self.means = (R.T @ X) / Nk[:, None]
        self.vars = np.zeros((self.K, d))
        for k in range(self.K):
            diff = X - self.means[k]
            self.vars[k] = (R[:, k] @ (diff ** 2)) / Nk[k] + self.reg

    def fit(self, X):
        self.initialize(X)
        for i in range(self.max_iter):
            R = self.e_step(X)
            self.m_step(X, R)
        return self


# ────────────────────────────────────────────── helpers
def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float64)
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float64)
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, y_train, X_test, y_test


def normalise(X_train, X_test):
    """Min-max normalise using training statistics."""
    lo  = X_train.min(axis=0)
    hi  = X_train.max(axis=0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return (X_train - lo) / rng, (X_test - lo) / rng, lo, hi


def denormalise(Z, lo, hi):
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return Z * rng + lo


def build_cluster_map(cluster_labels, y_true, K):
    cluster_map = {}
    for k in range(K):
        mask = cluster_labels == k
        if np.sum(mask) == 0:
            cluster_map[k] = -1
            continue
        vals, counts = np.unique(y_true[mask], return_counts=True)
        cluster_map[k] = vals[np.argmax(counts)]
    return cluster_map


def map_predictions(cluster_labels, cluster_map):
    return np.array([cluster_map[c] for c in cluster_labels])


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ────────────────────────────────────────────── main
def run(data_dir, out_dir,
        hidden_dim=512,
        latent_dim=128,
        epochs=80,
        batch_size=32,
        lr=1e-3):

    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load & normalise ──────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_data(data_dir)
    X_tr_norm, X_te_norm, lo, hi = normalise(X_train, X_test)
    input_dim = X_tr_norm.shape[1]

    print(f"Data: train={X_tr_norm.shape}, test={X_te_norm.shape}")
    print(f"Autoencoder: {input_dim} → {hidden_dim} → {latent_dim} → {hidden_dim} → {input_dim}")

    # ── 2. Train Autoencoder ─────────────────────────────────────────────
    ae = Autoencoder(input_dim=input_dim,
                     hidden_dim=hidden_dim,
                     latent_dim=latent_dim,
                     lr=lr,
                     random_state=42)
    print(f"\nTraining Autoencoder ({epochs} epochs)…")
    ae.fit(X_tr_norm, epochs=epochs, batch_size=batch_size, verbose=True)

    # Save model
    ae_path = os.path.join(out_dir, "autoencoder_weights.npy")
    ae.save(ae_path)

    # ── 3. Plot training loss ─────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.plot(ae.train_losses, color="steelblue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "ae_training_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Saved: {loss_path}")

    # ── 4. Encode data ────────────────────────────────────────────────────
    Z_train = ae.transform(X_tr_norm)
    Z_test  = ae.transform(X_te_norm)
    print(f"\nLatent space: train={Z_train.shape}, test={Z_test.shape}")

    # ── 5. Clustering on latent space ─────────────────────────────────────
    Ks = [20, 40, 60]
    km_accs  = []
    gmm_accs = []

    print("\n" + "="*55)
    print("  Clustering on Autoencoder latent space")
    print("="*55)

    for K in Ks:
        # ── K-Means ──
        print(f"\n  K-Means  K={K}")
        km = KMeans(n_clusters=K, max_iter=300, random_state=42)
        km.fit(Z_train)
        cluster_map_km  = build_cluster_map(km.labels_, y_train, K)
        test_clusters   = km.predict(Z_test)
        y_pred_km       = map_predictions(test_clusters, cluster_map_km)
        acc_km          = accuracy(y_test, y_pred_km)
        km_accs.append(acc_km)
        print(f"  Accuracy: {acc_km:.4f}")

        # ── GMM (stable diagonal-cov version for latent space) ──
        print(f"\n  GMM      K={K}")
        gmm = StableGMM(n_components=K, max_iter=50, random_state=42)
        gmm.fit(Z_train)
        R_train_gmm  = gmm.e_step(Z_train)
        cluster_map_gmm = build_cluster_map(np.argmax(R_train_gmm, axis=1), y_train, K)
        R_test_gmm   = gmm.e_step(Z_test)
        y_pred_gmm   = map_predictions(np.argmax(R_test_gmm, axis=1), cluster_map_gmm)
        acc_gmm      = accuracy(y_test, y_pred_gmm)
        gmm_accs.append(acc_gmm)
        print(f"  Accuracy: {acc_gmm:.4f}")

    # ── 6. Plot comparison ────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(Ks, km_accs, marker='o', label="K-Means (AE)")
    plt.plot(Ks, gmm_accs, marker='s', label="GMM (AE)")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Test Accuracy")
    plt.title("Autoencoder Latent Space: Clustering Accuracy vs K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    cmp_path = os.path.join(out_dir, "ae_clustering_acc_vs_K.png")
    plt.savefig(cmp_path, dpi=150)
    plt.close()
    print(f"\nSaved: {cmp_path}")

    # ── 7. Visualise reconstructed faces ─────────────────────────────────
    n_show = 8
    X_hat_norm = ae.reconstruct(X_te_norm[:n_show])
    X_hat      = denormalise(X_hat_norm, lo, hi).clip(0, 255)
    X_orig     = X_test[:n_show]

    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 5))
    for i in range(n_show):
        axes[0, i].imshow(X_orig[i].reshape(112, 92), cmap="gray")
        axes[0, i].set_title("Original", fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(X_hat[i].reshape(112, 92), cmap="gray")
        axes[1, i].set_title("Reconstructed", fontsize=7)
        axes[1, i].axis("off")

    plt.suptitle("Autoencoder – Original vs Reconstructed Faces", fontsize=11)
    plt.tight_layout()
    recon_path = os.path.join(out_dir, "ae_reconstructed_faces.png")
    plt.savefig(recon_path, dpi=150)
    plt.close()
    print(f"Saved: {recon_path}")

    # ── 8. Full evaluation for best K-Means config ────────────────────────
    best_K_idx  = int(np.argmax(km_accs))
    best_K      = Ks[best_K_idx]
    print(f"\nBest K-Means on AE space: K={best_K}, acc={km_accs[best_K_idx]:.4f}")

    km_best = KMeans(n_clusters=best_K, max_iter=300, random_state=42)
    km_best.fit(Z_train)
    cm_best     = build_cluster_map(km_best.labels_, y_train, best_K)
    y_pred_best = map_predictions(km_best.predict(Z_test), cm_best)

    f1_macro    = f1_score(y_test, y_pred_best, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_best, average="weighted", zero_division=0)
    print(f"  F1-macro:    {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")

    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(f"K-Means (AE) Confusion Matrix  K={best_K}")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "ae_kmeans_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    # ── Summary table ────────────────────────────────────────────────────
    print("\nSummary Table (Autoencoder latent space):")
    print(f"  {'K':>4} | {'KMeans Acc':>12} | {'GMM Acc':>10}")
    print(f"  {'-'*33}")
    for i, K in enumerate(Ks):
        print(f"  {K:>4} | {km_accs[i]:>12.4f} | {gmm_accs[i]:>10.4f}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "Processed_DataSet")
    OUT_DIR  = os.path.join(BASE_DIR, "outputs")
    run(DATA_DIR, OUT_DIR)