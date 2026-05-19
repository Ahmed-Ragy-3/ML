import os
import numpy as np
import matplotlib
import torch

from Autoencoder.model.autoencoder import DeepAutoencoder
from GMM.model.gmm import GMM
from Kmeans.model.kmeans import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


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
    KS = [20, 40, 60]

    DATA_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Processed_DataSet")
    )
    OUT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "outputs")
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    X_train_raw = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
    X_test_raw  = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
    y_train     = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test      = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    input_dim   = X_train_raw.shape[1]

    # Convert to torch tensors normalized to [0,1]
    X_train_tensor = torch.tensor(X_train_raw / 255.0)
    X_test_tensor  = torch.tensor(X_test_raw / 255.0)

    # ── 1. Train Autoencoder ──────────────────────────────────────────────
    print("\n==================== AUTOENCODER TRAINING ====================\n")
    print(f"Architecture: {input_dim} → 512 → 256 → 512 → {input_dim}")
    print(f"Loss: MSE  ||x - x̃||²")

    ae = DeepAutoencoder(input_dim=input_dim, hidden1=512, bottleneck=256)
    ae, losses = DeepAutoencoder.train_autoencoder(ae, X_train_tensor, epochs=200, lr=1e-3, batch_size=32)

    # Training loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(losses, color="steelblue")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ae_training_loss.png"), dpi=150)
    plt.close()

    # ── 2. Latent codes ───────────────────────────────────────────────────
    Z_train = ae.transform(X_train_tensor).detach().numpy()
    Z_test  = ae.transform(X_test_tensor).detach().numpy()
    print(f"\n  Latent code: train={Z_train.shape}, test={Z_test.shape}")

    # ── 3. Reconstructed faces (specific indices) ─────────────────────────
    indices = [0, 5, 12, 20, 33, 47, 50, 75]  # choose specific test samples
    X_selected = X_test_tensor[indices]

    X_hat_norm = ae.reconstruct(X_selected).detach().numpy()
    X_hat_disp = (X_hat_norm * 255).clip(0, 255)

    fig, axes = plt.subplots(2, len(indices), figsize=(2 * len(indices), 5))
    for i, idx in enumerate(indices):
        axes[0, i].imshow(X_test_raw[idx].reshape(112, 92), cmap="gray")
        axes[0, i].set_title(f"Original #{idx}", fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(X_hat_disp[i].reshape(112, 92), cmap="gray")
        axes[1, i].set_title("Reconstructed", fontsize=7)
        axes[1, i].axis("off")

    plt.suptitle("Autoencoder — Selected Original vs Reconstructed Faces", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ae_selected_reconstructed_faces.png"), dpi=150)
    plt.close()

    # ── 4. Clustering on latent space ─────────────────────────────────────
    results = []
    print("\n==================== EXPERIMENT START ====================\n")

    for K in KS:
        # K-Means
        print(f"\n---- K-Means | K={K} ----")
        km = KMeans(n_clusters=K, random_state=42)
        km.fit(Z_train)
        cmap = build_cluster_map(km, Z_train, y_train)
        y_pred = np.array([cmap[c] for c in km.predict(Z_test)])
        acc_km = accuracy_score(y_test, y_pred)
        f1_km  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"  Accuracy : {acc_km:.4f}")
        print(f"  Macro F1 : {f1_km:.4f}")

        # GMM
        print(f"\n---- GMM | K={K} ----")
        gmm = GMM(n_components=K, random_state=42)
        gmm.fit(Z_train)
        cmap_g = build_cluster_map(gmm, Z_train, y_train)
        y_pred_g = np.array([cmap_g[c] for c in gmm.predict(Z_test)])
        acc_gmm = accuracy_score(y_test, y_pred_g)
        f1_gmm  = f1_score(y_test, y_pred_g, average="macro", zero_division=0)
        print(f"  Accuracy : {acc_gmm:.4f}")
        print(f"  Macro F1 : {f1_gmm:.4f}")

        results.append({
            "K": K,
            "km_acc": acc_km, "km_f1": f1_km,
            "gmm_acc": acc_gmm, "gmm_f1": f1_gmm,
            "km_model": km, "km_cmap": cmap,
            "gmm_model": gmm, "gmm_cmap": cmap_g,
        })

    print("\n==================== EXPERIMENT DONE ====================\n")

    # ── 5. Results table ──────────────────────────────────────────────────
    print("\n==================== FULL RESULTS TABLE ====================\n")
    print(f"  {'K':>4} | {'KMeans Acc':>12} | {'KMeans Macro F1':>16} | "
          f"{'GMM Acc':>10} | {'GMM Macro F1':>14}")
    print(f"  {'-'*64}")
    for r in results:
        print(f"  {r['K']:>4} | {r['km_acc']:>12.4f} | {r['km_f1']:>16.4f} | "
              f"{r['gmm_acc']:>10.4f} | {r['gmm_f1']:>14.4f}")

    # ── 6. Accuracy vs K plot ─────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(KS, [r["km_acc"]  for r in results], marker='o', label="K-Means (AE)")
    plt.plot(KS, [r["gmm_acc"] for r in results], marker='s', label="GMM (AE)")
    plt.xlabel("Number of Clusters K"); plt.ylabel("Accuracy")
    plt.title("Autoencoder Latent Space: Accuracy vs K")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ae_clustering_acc_vs_K.png"), dpi=150)
    plt.close()

    # ── 7. Macro F1 vs K plot ─────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(KS, [r["km_f1"]  for r in results], marker='o', label="K-Means (AE)")
    plt.plot(KS, [r["gmm_f1"] for r in results], marker='s', label="GMM (AE)")
    plt.xlabel("Number of Clusters K"); plt.ylabel("Macro F1-Score")
    plt.title("Autoencoder Latent Space: Macro F1 vs K")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ae_clustering_f1_vs_K.png"), dpi=150)
    plt.close()

    # ── 8. Best models confusion matrices ─────────────────────────────────
    best_km = max(results, key=lambda r: r["km_f1"])
    y_pred_best_km = np.array([best_km["km_cmap"][c] for c in best_km["km_model"].predict(Z_test)])
    cm_km = confusion_matrix(y_test, y_pred_best_km)
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm_km).plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(f"K-Means (AE) Confusion Matrix  K={best_km['K']}  "
                 f"[Best by Macro F1={best_km['km_f1']:.4f}]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ae_kmeans_confusion_matrix.png"), dpi=150)
    plt.close()

    best_gmm = max(results, key=lambda r: r["gmm_f1"])
    y_pred_best_gmm = np.array([best_gmm["gmm_cmap"][c] for c in best_gmm["gmm_model"].predict(Z_test)])
    cm_gmm = confusion_matrix(y_test, y_pred_best_gmm)
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm_gmm).plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(f"GMM (AE) Confusion Matrix  K={best_gmm['K']}  "
                 f"[Best by Macro F1={best_gmm['gmm_f1']:.4f}]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ae_gmm_confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_experiment()
