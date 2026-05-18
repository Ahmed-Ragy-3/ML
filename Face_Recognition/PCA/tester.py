import numpy as np
import matplotlib.pyplot as plt
import os
from model.pca import PCA

def load_data(data_dir=None):
   if data_dir is None:
      # Get the directory of the current script and navigate to Processed_DataSet
      script_dir = os.path.dirname(os.path.abspath(__file__))
      data_dir = os.path.join(script_dir, "..", "Processed_DataSet")
   X_train = np.load(f"{data_dir}/X_train.npy")
   X_test = np.load(f"{data_dir}/X_test.npy")

   # Concatenate train + test for PCA fitting (common practice for unsupervised transform)
   X = np.vstack((X_train, X_test))
   return X


def compute_pca_full(X, save_path="eigenvectors.npy"):
   """
   Compute PCA once and save all components
   """
   pca = PCA(n_components=None)
   pca._fit(X)
   np.save(save_path, pca.components)
   return pca.components


def get_k_from_variance(eigenvalues, alpha):
   total_variance = np.sum(eigenvalues)
   cumulative_variance = np.cumsum(eigenvalues)
   ratio = cumulative_variance / total_variance
   k = np.argmax(ratio >= alpha) + 1
   return k


def visualize_samples(Z, title, num_samples=5):
   """
   Visualize samples in PCA space (as bar plots)
   """
   plt.figure(figsize=(12, 4))
   for i in range(num_samples):
      plt.subplot(1, num_samples, i+1)
      plt.bar(range(Z.shape[1]), Z[i])
      plt.title(f"Sample {i}")
      plt.xticks([])
   plt.suptitle(title)
   plt.show()


X = load_data()

# Step 1: Compute full PCA once
Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
cov_matrix = np.cov(Z, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues
idxs = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idxs]
eigenvectors = eigenvectors[:, idxs]

# Save full components
np.save("full_components.npy", eigenvectors)

# Variance thresholds
alphas = [0.8, 0.85, 0.9, 0.95]

for alpha in alphas:
   print(f"\n=== Variance Threshold: {alpha} ===")

   # Step 2: Determine k
   k = get_k_from_variance(eigenvalues, alpha)
   print(f"Selected k: {k}")

   # Step 3: Load only top-k components
   components_k = eigenvectors[:, :k]

   # Save for reuse
   path = f"components_k_{alpha}.npy"
   np.save(path, components_k)

   # Step 4: Apply PCA
   pca = PCA(n_components=k)
   Z_reduced = pca.fit_transform(X, path=path)

   print(f"Reduced shape: {Z_reduced.shape}")

   # Step 5: Visualization
   visualize_samples(Z_reduced, title=f"PCA Space (alpha={alpha})")
