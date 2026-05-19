import os
import numpy as np
import matplotlib

from Kmeans.model.kmeans import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def apply_pca(X_train, X_test, alpha):
    """
    Mean-centre and project onto top-k PCA components (via thin SVD).
    Fits on train only; applies the same projection to the test set.
    Returns: Z_train, Z_test, k
    """
    mean    = X_train.mean(axis=0)
    Z_train = (X_train - mean).astype(np.float32)
    Z_test  = (X_test  - mean).astype(np.float32)

    _, s, Vt    = np.linalg.svd(Z_train, full_matrices=False)
    eigenvalues = (s ** 2) / (Z_train.shape[0] - 1)
    cum_var     = np.cumsum(eigenvalues) / eigenvalues.sum()
    k           = int(np.argmax(cum_var >= alpha) + 1)

    W = Vt[:k].T                        # (d, k)
    return Z_train @ W, Z_test @ W, k


def build_cluster_map(model, X_train, y_train):
    clusters = model.predict(X_train)
    mapping  = {}
    for k in range(model.K):
        idx = np.where(clusters == k)[0]
        if len(idx) == 0:
            mapping[k] = 0
        else:
            labels, counts = np.unique(y_train[idx], return_counts=True)
            mapping[k] = labels[np.argmax(counts)]
    return mapping


def run_experiment():
    ALPHAS = [0.80, 0.85, 0.90, 0.95]
    KS     = [20, 40, 60]

    DATA_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Processed_DataSet")
    )
    OUT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "outputs")
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    X_train_raw = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test_raw  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train     = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test      = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    results = []

    # ── best model tracked by macro F1 ────────────────────────────────
    best = {"f1": -1, "acc": -1, "alpha": None, "K": None,
            "model": None, "X_test_pca": None, "cmap": None}

    print("\n==================== EXPERIMENT START ====================\n")

    for alpha in ALPHAS:
        print(f"\n==================== ALPHA = {alpha} ====================")

        X_train, X_test, k = apply_pca(X_train_raw, X_test_raw, alpha)
        print(f"  PCA dimensionality: {k}")

        for K in KS:
            print(f"\n---- Training K-Means | alpha={alpha} | K={K} ----")

            model = KMeans(n_clusters=K, random_state=42)
            model.fit(X_train)

            cmap   = build_cluster_map(model, X_train, y_train)
            y_pred = np.array([cmap[c] for c in model.predict(X_test)])

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            print(f"  Cost J   : {model.cost_:.2f}")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  F1-score (macro): {f1:.4f}")

            results.append({"alpha": alpha, "K": K,
                            "accuracy": acc, "f1": f1, "cost": model.cost_})

            # ── select best model by macro F1 ─────────────────────────
            if f1 > best["f1"]:
                best.update({"f1": f1, "acc": acc, "alpha": alpha, "K": K,
                             "model": model, "X_test_pca": X_test, "cmap": cmap})

    print("\n==================== EXPERIMENT DONE ====================\n")

    # ── full results table ────────────────────────────────────────────────
    print("\n==================== FULL RESULTS TABLE ====================\n")
    print(f"  {'Alpha':>6} | {'K':>4} | {'Accuracy':>10} | {'F1 (macro)':>12} | {'Cost J':>14}")
    print(f"  {'-'*54}")
    for r in results:
        print(f"  {r['alpha']:>6.2f} | {r['K']:>4} | {r['accuracy']:>10.4f} | "
              f"{r['f1']:>12.4f} | {r['cost']:>14.2f}")

    # ── grouped summary ───────────────────────────────────────────────────
    print("\n-- Mean accuracy per alpha --")
    for alpha in ALPHAS:
        vals = [r["accuracy"] for r in results if r["alpha"] == alpha]
        print(f"  alpha={alpha} : {np.mean(vals):.4f}")

    print("\n-- Mean macro F1 per alpha --")
    for alpha in ALPHAS:
        vals = [r["f1"] for r in results if r["alpha"] == alpha]
        print(f"  alpha={alpha} : {np.mean(vals):.4f}")

    print("\n-- Mean accuracy per K --")
    for K in KS:
        vals = [r["accuracy"] for r in results if r["K"] == K]
        print(f"  K={K} : {np.mean(vals):.4f}")

    print("\n-- Mean macro F1 per K --")
    for K in KS:
        vals = [r["f1"] for r in results if r["K"] == K]
        print(f"  K={K} : {np.mean(vals):.4f}")

    # ── plots (accuracy, F1, cost) ────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    for alpha in ALPHAS:
        accs = [r["accuracy"] for r in results if r["alpha"] == alpha]
        plt.plot(KS, accs, marker='o', label=f"α={alpha}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Accuracy")
    plt.title("K-Means: Accuracy vs K")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kmeans_acc_vs_K.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for K in KS:
        accs = [r["accuracy"] for r in results if r["K"] == K]
        plt.plot(ALPHAS, accs, marker='s', label=f"K={K}")
    plt.xlabel("Variance Threshold α")
    plt.ylabel("Accuracy")
    plt.title("K-Means: Accuracy vs α")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kmeans_acc_vs_alpha.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for alpha in ALPHAS:
        f1s = [r["f1"] for r in results if r["alpha"] == alpha]
        plt.plot(KS, f1s, marker='o', label=f"α={alpha}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Macro F1-Score")
    plt.title("K-Means: Macro F1 vs K")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kmeans_f1_vs_K.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for K in KS:
        f1s = [r["f1"] for r in results if r["K"] == K]
        plt.plot(ALPHAS, f1s, marker='s', label=f"K={K}")
    plt.xlabel("Variance Threshold α")
    plt.ylabel("Macro F1-Score")
    plt.title("K-Means: Macro F1 vs α")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kmeans_f1_vs_alpha.png"), dpi=150)
    plt.close()

    # ── plot 5: elbow — cost J vs K ───────────────────────────────────────
    plt.figure(figsize=(8, 5))
    for alpha in ALPHAS:
        costs = [r["cost"] for r in results if r["alpha"] == alpha]
        plt.plot(KS, costs, marker='o', label=f"α={alpha}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Cost J  (within-cluster sum of squares)")
    plt.title("K-Means: Elbow Plot — Cost J vs K")
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kmeans_elbow_vs_K.png"), dpi=150)
    plt.close()

    # ── plot 6: elbow — cost J vs alpha ──────────────────────────────────
    plt.figure(figsize=(8, 5))
    for K in KS:
        costs = [r["cost"] for r in results if r["K"] == K]
        plt.plot(ALPHAS, costs, marker='s', label=f"K={K}")
    plt.xlabel("Variance Threshold α")
    plt.ylabel("Cost J  (within-cluster sum of squares)")
    plt.title("K-Means: Elbow Plot — Cost J vs α")
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kmeans_elbow_vs_alpha.png"), dpi=150)
    plt.close()

    # ── best model: full evaluation + confusion matrix ────────────────────
    print(f"\n==================== BEST MODEL (by Macro F1) ====================\n")
    print(f"  alpha={best['alpha']}, K={best['K']}")
    print(f"  Accuracy      : {best['acc']:.4f}")
    print(f"  Macro F1-score: {best['f1']:.4f}")

    y_pred_best = np.array([best["cmap"][c]
                            for c in best["model"].predict(best["X_test_pca"])])

    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(f"K-Means Confusion Matrix  (α={best['alpha']}, K={best['K']})  "
                 f"[Best by Macro F1={best['f1']:.4f}]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kmeans_confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"\n  Plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_experiment()
