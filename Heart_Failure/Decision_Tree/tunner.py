import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
from Decision_Tree.model.decision_tree import DecisionTree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "dataset", "heart.csv")

    dataset = Dataset(data_path)
    dataset.prepare()
    return dataset


def train_model(x_train, y_train, depth, min_split):
    model = DecisionTree(
        max_depth=depth,
        min_samples_split=min_split
    )
    model.fit(x_train, y_train)
    return model


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def tune_model(dataset, depths, min_samples, csv_writer):
    results_acc = {}
    results_f1 = {}

    best_f1 = -1
    best_model = None
    best_params = None

    for depth in depths:
        for min_split in min_samples:

            model = train_model(
                dataset.x_train,
                dataset.y_train,
                depth,
                min_split
            )

            val_preds = model.predict(dataset.x_val)
            acc, f1, _ = evaluate(dataset.y_val, val_preds)

            print(f"[depth={depth}, min_split={min_split}] "
                  f"Val Acc={acc:.4f}, F1={f1:.4f}")

            # SAVE TO CSV
            csv_writer.writerow([depth, min_split, acc, f1])

            results_acc[(depth, min_split)] = acc
            results_f1[(depth, min_split)] = f1

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_params = (depth, min_split)

    return best_model, best_params, results_acc, results_f1


def plot_heatmap(matrix, depths, min_samples, title):
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(title)

    plt.xticks(range(len(min_samples)), min_samples)
    plt.yticks(range(len(depths)), depths)

    plt.xlabel("min_samples_split")
    plt.ylabel("max_depth")

    plt.show()


def test_model(model, dataset):
    preds = model.predict(dataset.x_test)
    acc, f1, cm = evaluate(dataset.y_test, preds)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return acc, f1


def main():

    dataset = load_data()

    depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    min_samples = [2, 5, 7, 10, 20, 30, 40]

    with open("plots/decision_tree_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["depth", "min_samples_split", "val_accuracy", "val_f1"])

        best_model, best_params, results_acc, results_f1 = tune_model(
            dataset,
            depths,
            min_samples,
            writer
        )

    acc_matrix = np.zeros((len(depths), len(min_samples)))
    f1_matrix = np.zeros((len(depths), len(min_samples)))

    for i, d in enumerate(depths):
        for j, m in enumerate(min_samples):
            acc_matrix[i, j] = results_acc[(d, m)]
            f1_matrix[i, j] = results_f1[(d, m)]

    plot_heatmap(acc_matrix, depths, min_samples,
                 "Validation Accuracy vs Hyperparameters")

    plot_heatmap(f1_matrix, depths, min_samples,
                 "Validation F1 Score vs Hyperparameters")

    print("\n===== BEST MODEL =====")
    print(f"Max Depth: {best_params[0]}")
    print(f"Min Samples Split: {best_params[1]}")

    test_acc, test_f1 = test_model(best_model, dataset)

    # append test results to CSV
    with open("plots/decision_tree_results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["TEST", "", test_acc, test_f1])

if __name__ == "__main__":
    main()