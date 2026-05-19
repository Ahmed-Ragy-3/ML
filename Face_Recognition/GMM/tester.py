import os
import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from GMM.model.gmm import GMM

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_cluster_map(model, X_train, y_train):
    clusters = model.predict(X_train)
    mapping = {}
    for k in range(model.K):
        idx = np.where(clusters == k)[0]
        if len(idx) == 0:
            mapping[k] = 0
        else:
            labels, counts = np.unique(y_train[idx], return_counts=True)
            mapping[k] = labels[np.argmax(counts)]
    return mapping


def run_experiment():
    ALPHAS = [0.8, 0.85, 0.9, 0.95]
    KS = [20, 40, 60]

    DATA_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Processed_DataSet")
    )
    OUT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "outputs")
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []

    print("\n==================== EXPERIMENT START ====================\n")

    for alpha in ALPHAS:
        print(f"\n==================== ALPHA = {alpha} ====================")

        X_train = np.load(os.path.join(DATA_DIR, f"X_train_{alpha}.npy"))
        X_test = np.load(os.path.join(DATA_DIR, f"X_test_{alpha}.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

        for K in KS:
            print(f"\n---- Training GMM | alpha={alpha} | K={K} ----")

            model = GMM(n_components=K, random_state=42)
            model.fit(X_train)

            cmap = build_cluster_map(model, X_train, y_train)
            y_pred = np.array([cmap[c] for c in model.predict(X_test)])

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            print(f"  Accuracy : {acc:.4f}")
            print(f"  Macro F1 : {f1:.4f}")

            results.append({
                "alpha": alpha,
                "K": K,
                "accuracy": acc,
                "f1": f1,
                "model": model,
                "cmap": cmap,
                "y_test": y_test,
                "y_pred": y_pred
            })

    print("\n==================== EXPERIMENT DONE ====================\n")
    return results, OUT_DIR


def print_report(results, OUT_DIR):
    # Results table
    print("\n==================== FULL RESULTS TABLE ====================\n")
    print(f"  {'alpha':>6} | {'K':>4} | {'Accuracy':>10} | {'Macro F1':>10}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['alpha']:>6} | {r['K']:>4} | {r['accuracy']:>10.4f} | {r['f1']:>10.4f}")

    # Accuracy vs K plot
    plt.figure(figsize=(8, 5))
    for alpha in sorted(set(r['alpha'] for r in results)):
        subset = [r for r in results if r['alpha'] == alpha]
        plt.plot([r['K'] for r in subset], [r['accuracy'] for r in subset],
                 marker='o', label=f"alpha={alpha}")
    plt.xlabel("Number of Clusters K"); plt.ylabel("Accuracy")
    plt.title("GMM Accuracy vs K"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gmm_acc_vs_K.png"), dpi=150)
    plt.close()

    # F1 vs K plot
    plt.figure(figsize=(8, 5))
    for alpha in sorted(set(r['alpha'] for r in results)):
        subset = [r for r in results if r['alpha'] == alpha]
        plt.plot([r['K'] for r in subset], [r['f1'] for r in subset],
                 marker='s', label=f"alpha={alpha}")
    plt.xlabel("Number of Clusters K"); plt.ylabel("Macro F1-Score")
    plt.title("GMM Macro F1 vs K"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gmm_f1_vs_K.png"), dpi=150)
    plt.close()

    # Best model by F1 confusion matrix
    best = max(results, key=lambda r: r["f1"])
    print("\n==================== BEST MODEL (BY MACRO F1) ====================\n")
    print(f"  Alpha={best['alpha']} | K={best['K']} | Accuracy={best['accuracy']:.4f} | Macro F1={best['f1']:.4f}")

    cm = confusion_matrix(best["y_test"], best["y_pred"])
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(f"GMM Confusion Matrix alpha={best['alpha']} K={best['K']} "
                 f"[Best by Macro F1={best['f1']:.4f}]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gmm_best_confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    results, OUT_DIR = run_experiment()
    print_report(results, OUT_DIR)
