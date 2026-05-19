import numpy as np
import os
from model.pca import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "Processed_DataSet")
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

EIGEN_PATH = os.path.join(results_dir, "full_eigen.npy")
ALPHAS = [0.80, 0.85, 0.90, 0.95]

X_train = np.load(f"{data_dir}/X_train.npy")
X_test  = np.load(f"{data_dir}/X_test.npy")

# Fit PCA ONLY on training data
X = X_train

if os.path.exists(EIGEN_PATH):
   saved = np.load(EIGEN_PATH, allow_pickle=True).item()
   eigenvalues = saved["eigenvalues"]
   eigenvectors = saved["eigenvectors"]
else:
   mean = np.mean(X, axis=0)
   std = np.std(X, axis=0, ddof=1)
   Z = (X - mean) / std

   cov = np.cov(Z, rowvar=False)
   eigenvalues, eigenvectors = np.linalg.eigh(cov)

   eigenvalues = eigenvalues[::-1]
   eigenvectors = eigenvectors[:, ::-1]

   np.save(EIGEN_PATH, {"eigenvalues": eigenvalues,
           "eigenvectors": eigenvectors})


def get_k(eigs, alpha):
   eigs = np.clip(eigs.real, 0, None)
   ratio = np.cumsum(eigs) / np.sum(eigs)
   return int(np.argmax(ratio >= alpha) + 1)


for alpha in ALPHAS:
   k = get_k(eigenvalues, alpha)

   components = eigenvectors[:, :k]
   mean = np.mean(X, axis=0)
   std = np.std(X, axis=0, ddof=1)

   comp_path = os.path.join(results_dir, f"components_{alpha}.npy")
   np.save(comp_path, {"components": components, "mean": mean, "std": std})

   pca = PCA(n_components=k)

   X_train_pca = pca.fit_transform(X_train, path=comp_path)
   X_test_pca  = pca.fit_transform(X_test,  path=comp_path)

   np.save(os.path.join(results_dir, f"X_train_{alpha}.npy"), X_train_pca)
   np.save(os.path.join(results_dir, f"X_test_{alpha}.npy"),  X_test_pca)

   print(f"alpha={alpha} | k={k} | train={X_train_pca.shape} | test={X_test_pca.shape}")
