import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from model.pca import PCA


def load_data(data_dir=None):
   if data_dir is None:
      script_dir = os.path.dirname(os.path.abspath(__file__))
      data_dir = os.path.join(script_dir, "..", "Processed_DataSet")
   X_train = np.load(f"{data_dir}/X_train.npy")
   X_test  = np.load(f"{data_dir}/X_test.npy")
   # return np.vstack((X_train, X_test))
   return X_train  # use only training data for PCA fitting


def compute_and_save_full_pca(X, eigen_path="full_eigen.npy"):
   """
   Fit PCA once on X, saving BOTH eigenvalues and eigenvectors so we can
   compute variance ratios without re-fitting.

   Uses eigh (symmetric/Hermitian) for numerical stability on the covariance
   matrix.  eigh returns ascending order, so we flip immediately.

   ddof=1 is used in both std and cov for consistency.
   """
   mean = np.mean(X, axis=0)
   std = np.std(X, axis=0, ddof=1)
   Z = (X - mean) / std

   cov_matrix = np.cov(Z, rowvar=False)

   eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

   eigenvalues  = eigenvalues[::-1]
   eigenvectors = eigenvectors[:, ::-1]

   np.save(eigen_path, {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors})
   print(f"Saved full eigen-decomposition → {eigen_path}")
   return eigenvalues, eigenvectors


def get_k_from_variance(eigenvalues, alpha):
   """Return the minimum k that retains at least alpha of total variance."""
   # eigenvalues may be tiny-negative due to float errors → clip
   eigenvalues = np.clip(eigenvalues.real, 0, None)
   cumulative_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
   k = int(np.argmax(cumulative_ratio >= alpha) + 1)
   return k


def visualize_faces_in_pca(Z_reduced, X_orig, pca, alpha, img_shape, num_samples=5):
   """
   Visualise sample faces in two ways:
     - Top row   : original face images
     - Bottom row: face reconstructed from PCA subspace (to show quality)

   This is far more informative than a bar-plot of PCA coefficients.
   """
   fig = plt.figure(figsize=(num_samples * 2.5 + 1, 6))
   fig.suptitle(
       f"PCA reconstruction  |  α = {alpha}  |  k = {Z_reduced.shape[1]} components",
       fontsize=13, y=1.01
   )
   gs = gridspec.GridSpec(2, num_samples + 1,
                          width_ratios=[0.25] + [1] * num_samples,
                          hspace=0.1, wspace=0.05)

   row_labels = ["Original", "Reconstructed"]
   for row, label in enumerate(row_labels):
      ax_label = fig.add_subplot(gs[row, 0])
      ax_label.set_axis_off()
      ax_label.text(0.9, 0.5, label, va="center", ha="right",
                    fontsize=10, rotation=0,
                    transform=ax_label.transAxes)

   for col in range(num_samples):
      # ── original ──
      ax_orig = fig.add_subplot(gs[0, col + 1])
      ax_orig.imshow(X_orig[col].reshape(img_shape), cmap="gray")
      ax_orig.set_axis_off()
      ax_orig.set_title(f"#{col}", fontsize=9)

      # ── reconstructed from PCA subspace ──
      reconstructed = pca.reconstruct(Z_reduced[col:col+1])[0]
      reconstructed = np.clip(reconstructed, 0, 255)

      ax_rec = fig.add_subplot(gs[1, col + 1])
      ax_rec.imshow(reconstructed.reshape(img_shape), cmap="gray")
      ax_rec.set_axis_off()

   plt.tight_layout()
   plt.savefig(f"pca_faces_alpha_{alpha}.png", dpi=120, bbox_inches="tight")
   plt.show()
   print(f"  Saved face grid → pca_faces_alpha_{alpha}.png")


def plot_cumulative_variance(eigenvalues, alphas):
   """Plot the cumulative explained variance curve with threshold markers."""
   eigenvalues = np.clip(eigenvalues.real, 0, None)
   cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(range(1, len(cumvar) + 1), cumvar, lw=1.5, color="#2563EB")
   ax.set_xlabel("Number of components (k)")
   ax.set_ylabel("Cumulative explained variance")
   ax.set_title("PCA — cumulative explained variance")

   colors = ["#DC2626", "#D97706", "#059669", "#7C3AED"]
   for alpha, color in zip(alphas, colors):
      k = get_k_from_variance(eigenvalues, alpha)
      ax.axhline(alpha, color=color, ls="--", lw=0.9, alpha=0.8)
      ax.axvline(k,     color=color, ls="--", lw=0.9, alpha=0.8)
      ax.scatter([k], [alpha], color=color, zorder=5, s=50)
      ax.annotate(f"α={alpha}\nk={k}", xy=(k, alpha),
                  xytext=(k + len(cumvar) * 0.02, alpha - 0.03),
                  fontsize=8, color=color)

   ax.set_xlim(1, len(cumvar))
   ax.set_ylim(0, 1.02)
   ax.grid(True, ls=":", alpha=0.4)
   plt.tight_layout()
   plt.savefig("pca_cumulative_variance.png", dpi=120, bbox_inches="tight")
   plt.show()
   print("Saved cumulative variance plot → pca_cumulative_variance.png")


script_dir = os.path.dirname(os.path.abspath(__file__))
EIGEN_PATH = os.path.join(script_dir, "results", "full_eigen.npy")
ALPHAS = [0.80, 0.85, 0.90, 0.95]

# Infer image shape from data (override if you know the exact shape)
# e.g. for ORL/AT&T faces: (112, 92); for LFW crops: (62, 47); etc.
IMG_SHAPE = (112, 92)  # set to (H, W) if known, e.g. (112, 92)

X = load_data()
print(
    f"Dataset shape: {X.shape}  ({X.shape[0]} samples, {X.shape[1]} features)")

if IMG_SHAPE is None:
   side = int(np.sqrt(X.shape[1]))
   IMG_SHAPE = (side, side)
   print(f"Inferred image shape: {IMG_SHAPE}  "
         f"(set IMG_SHAPE manually if this is wrong)")

# ── Step 1: Compute full PCA once (or load from cache) ───────────────────────
if os.path.exists(EIGEN_PATH):
   print(f"\nLoading cached eigen-decomposition from {EIGEN_PATH}")
   saved = np.load(EIGEN_PATH, allow_pickle=True).item()
   eigenvalues = saved["eigenvalues"]
   eigenvectors = saved["eigenvectors"]
else:
   print("\nComputing full PCA (this may take a moment)…")
   eigenvalues, eigenvectors = compute_and_save_full_pca(X, EIGEN_PATH)

# Plot the cumulative variance curve once
plot_cumulative_variance(eigenvalues, ALPHAS)

# ── Steps 2–5: Loop over variance thresholds ─────────────────────────────────
for alpha in ALPHAS:
   print(f"\n{'='*55}")
   print(f"  Variance threshold α = {alpha}")

   # Determine k
   k = get_k_from_variance(eigenvalues, alpha)
   print(f"  Selected k           : {k}")

   # Save the top-k components for this threshold
   components_k = eigenvectors[:, :k]
   path_k = f"components_k_{alpha}.npy"
   # FIX 3: save mean/std alongside components so fit_transform(path=…) works
   mean_X = np.mean(X, axis=0)
   std_X = np.std(X, axis=0, ddof=1)
   np.save(path_k, {"components": components_k, "mean": mean_X, "std": std_X})

   # Apply PCA — FIX 4: always call fit first so mean/std are set,
   # then transform.  Loading from path also needs mean/std (fixed in pca.py).
   pca = PCA(n_components=k)
   Z_reduced = pca.fit_transform(X, path=path_k)

   print(f"  Original shape       : {X.shape}")
   print(f"  Reduced shape        : {Z_reduced.shape}")
   retained = np.clip(eigenvalues.real, 0, None)
   retained = np.sum(retained[:k]) / np.sum(retained)
   print(f"  Variance retained    : {retained:.4f}")

   # Visualise sample faces in PCA space (original + reconstructed)
   visualize_faces_in_pca(Z_reduced, X, pca, alpha, IMG_SHAPE, num_samples=5)
