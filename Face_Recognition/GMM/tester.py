import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from GMM.model.gmm import GMM



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
            f1 = f1_score(y_test, y_pred, average="weighted")

            print(f"Accuracy: {acc:.4f}")
            print(f"F1-score : {f1:.4f}")

            results.append({
                "alpha": alpha,
                "K": K,
                "accuracy": acc,
                "f1": f1
            })

    print("\n==================== EXPERIMENT DONE ====================\n")

    df = pd.DataFrame(results)
    return df


def print_report(df):
    print("\n==================== FULL RESULTS TABLE ====================\n")
    print(df.to_string(index=False))

    print("\n==================== BEST MODELS (BY ACCURACY) ====================\n")
    print(df.sort_values("accuracy", ascending=False).head(5).to_string(index=False))

    print("\n==================== BEST MODELS (BY F1 SCORE) ====================\n")
    print(df.sort_values("f1", ascending=False).head(5).to_string(index=False))

    print("\n==================== GROUPED SUMMARY ====================\n")

    print("\n-- Mean performance per alpha --")
    print(df.groupby("alpha")[["accuracy", "f1"]].mean())

    print("\n-- Mean performance per K --")
    print(df.groupby("K")[["accuracy", "f1"]].mean())


if __name__ == "__main__":
    df = run_experiment()

    print_report(df)