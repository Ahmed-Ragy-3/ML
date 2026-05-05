import os
import numpy as np
import matplotlib.pyplot as plt

from GMM.model.gmm import GMM


def build_cluster_map(R_train, y_train, K):
    clusters = np.argmax(R_train, axis=1)
    y_train = np.array(y_train)

    cluster_map = {}

    for k in range(K):
        mask = clusters == k

        if np.sum(mask) == 0:
            cluster_map[k] = -1
            continue

        labels, counts = np.unique(y_train[mask], return_counts=True)
        cluster_map[k] = labels[np.argmax(counts)]

    return cluster_map


def predict(gmm, X, cluster_map):
    R = gmm.e_step(X)

    clusters = np.argmax(R, axis=1)

    preds = np.array([cluster_map[c] for c in clusters])
    return preds


def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def run():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "Processed_DataSet")

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    Ks = [20, 40, 60]
    accs = []

    for K in Ks:
        print(f"\n===== GMM K={K} =====")

        gmm = GMM(n_components=K, max_iter=50)

        gmm.fit(X_train)

        R_train = gmm.e_step(X_train)
        cluster_map = build_cluster_map(R_train, y_train, K)

        y_pred = predict(gmm, X_test, cluster_map)

        acc = accuracy(y_test, y_pred)
        accs.append(acc)

        print(f"K={K} Test Accuracy: {acc:.4f}")

    plt.plot(Ks, accs, marker='o')
    plt.xlabel("Number of Components (K)")
    plt.ylabel("Test Accuracy")
    plt.title("GMM Performance on Face Dataset")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run()