import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmeans import KMeans

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, y_train, X_test, y_test


def apply_pca(X_train, X_test, alpha):
    """
    Mean-centre and project onto top-k PCA components (via SVD on the
    data matrix). Fits on train only, then applies the same transform
    to both splits.
    """
    mean    = X_train.mean(axis=0)
    Z_train = (X_train - mean).astype(np.float32)
    Z_test  = (X_test  - mean).astype(np.float32)

    U, s, Vt    = np.linalg.svd(Z_train, full_matrices=False)
    eigenvalues = (s ** 2) / (Z_train.shape[0] - 1)

    cum_var = np.cumsum(eigenvalues) / eigenvalues.sum()
    k       = int(np.argmax(cum_var >= alpha) + 1)

    W = Vt[:k].T          # shape (d, k)
    return Z_train @ W, Z_test @ W, k


def build_cluster_map(labels_pred, y_true, K):
    """Map each cluster id → majority true label."""
    cluster_map = {}
    for k in range(K):
        mask = labels_pred == k
        if np.sum(mask) == 0:
            cluster_map[k] = -1
            continue
        vals, counts  = np.unique(y_true[mask], return_counts=True)
        cluster_map[k] = vals[np.argmax(counts)]
    return cluster_map


def map_predictions(cluster_labels, cluster_map):
    return np.array([cluster_map[c] for c in cluster_labels])


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ------------------------------------------------------------------ main run
def run(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(data_dir)

    alphas = [0.80, 0.85, 0.90, 0.95]
    Ks     = [20, 40, 60]

    # results[alpha][K] = test_accuracy
    # costs[alpha][K]   = J (cost after convergence) — used for elbow plots
    results = {a: {} for a in alphas}
    costs   = {a: {} for a in alphas}

    best = {"acc": -1, "alpha": None, "K": None, "model": None,
            "X_test_pca": None, "cluster_map": None}

    for alpha in alphas:
        print(f"\n{'='*55}")
        print(f"  PCA alpha = {alpha}")
        print(f"{'='*55}")

        X_tr_pca, X_te_pca, k = apply_pca(X_train, X_test, alpha)
        print(f"  PCA dimensionality: {k}")

        for K in Ks:
            print(f"\n  --- K-Means K={K} ---")
            km = KMeans(n_clusters=K, max_iter=300, random_state=42)
            km.fit(X_tr_pca)

            # store cost J for elbow plot
            costs[alpha][K] = km.cost_
            print(f"  Cost J: {km.cost_:.2f}")

            # build cluster→label map on training set
            cluster_map = build_cluster_map(km.labels_, y_train, K)

            # evaluate on test set
            test_clusters      = km.predict(X_te_pca)
            y_pred             = map_predictions(test_clusters, cluster_map)
            acc                = accuracy(y_test, y_pred)
            results[alpha][K]  = acc
            print(f"  Test Accuracy: {acc:.4f}")

            if acc > best["acc"]:
                best.update({"acc": acc, "alpha": alpha, "K": K,
                             "model": km, "cluster_map": cluster_map,
                             "X_test_pca": X_te_pca})

    # ------------------------------------------------ Plot 1: accuracy vs K
    plt.figure(figsize=(8, 5))
    for alpha in alphas:
        accs = [results[alpha][K] for K in Ks]
        plt.plot(Ks, accs, marker='o', label=f"α={alpha}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Test Accuracy")
    plt.title("K-Means: Accuracy vs K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "kmeans_acc_vs_K.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"\nSaved: {p1}")

    # ---------------------------------------------- Plot 2: accuracy vs alpha
    plt.figure(figsize=(8, 5))
    for K in Ks:
        accs = [results[alpha][K] for alpha in alphas]
        plt.plot(alphas, accs, marker='s', label=f"K={K}")
    plt.xlabel("Variance Threshold α")
    plt.ylabel("Test Accuracy")
    plt.title("K-Means: Accuracy vs α")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "kmeans_acc_vs_alpha.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Saved: {p2}")

    # -------------------------------------------------- Plot 3: elbow vs K
    # One curve per alpha — shows how cost J drops as K increases.
    # The "elbow" is where adding more clusters stops helping much.
    plt.figure(figsize=(8, 5))
    for alpha in alphas:
        J_values = [costs[alpha][K] for K in Ks]
        plt.plot(Ks, J_values, marker='o', label=f"α={alpha}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Cost J (within-cluster sum of squares)")
    plt.title("K-Means: Elbow Plot — Cost J vs K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    p3 = os.path.join(out_dir, "kmeans_elbow_vs_K.png")
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"Saved: {p3}")

    # ----------------------------------------------- Plot 4: elbow vs alpha
    # One curve per K — shows how cost J drops as alpha (dimensionality) grows.
    plt.figure(figsize=(8, 5))
    for K in Ks:
        J_values = [costs[alpha][K] for alpha in alphas]
        plt.plot(alphas, J_values, marker='s', label=f"K={K}")
    plt.xlabel("Variance Threshold α")
    plt.ylabel("Cost J (within-cluster sum of squares)")
    plt.title("K-Means: Elbow Plot — Cost J vs α")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    p4 = os.path.join(out_dir, "kmeans_elbow_vs_alpha.png")
    plt.savefig(p4, dpi=150)
    plt.close()
    print(f"Saved: {p4}")

    # ------------------------------------------ best model full evaluation
    print(f"\n{'='*55}")
    print(f"  Best Config → alpha={best['alpha']}, K={best['K']}")
    print(f"  Test Accuracy: {best['acc']:.4f}")
    print(f"{'='*55}")

    test_clusters = best["model"].predict(best["X_test_pca"])
    y_pred_best   = map_predictions(test_clusters, best["cluster_map"])

    f1_macro    = f1_score(y_test, y_pred_best, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_best, average="weighted", zero_division=0)
    print(f"  F1-Score (macro):    {f1_macro:.4f}")
    print(f"  F1-Score (weighted): {f1_weighted:.4f}")

    # confusion matrix
    cm   = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(f"K-Means Confusion Matrix (α={best['alpha']}, K={best['K']})")
    plt.tight_layout()
    p5 = os.path.join(out_dir, "kmeans_confusion_matrix.png")
    plt.savefig(p5, dpi=150)
    plt.close()
    print(f"Saved: {p5}")

    # tabulate all results
    print("\n  Full Results Table:")
    print(f"  {'Alpha':>6} | {'K':>4} | {'Accuracy':>10} | {'Cost J':>14}")
    print(f"  {'-'*42}")
    for alpha in alphas:
        for K in Ks:
            print(f"  {alpha:>6.2f} | {K:>4} | "
                  f"{results[alpha][K]:>10.4f} | "
                  f"{costs[alpha][K]:>14.2f}")

    return results, costs, best


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "Processed_DataSet")
    OUT_DIR  = os.path.join(BASE_DIR, "outputs")
    run(DATA_DIR, OUT_DIR)