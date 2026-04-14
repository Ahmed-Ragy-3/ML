import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from Heart_Failure.dataset import Dataset
from Heart_Failure.Random_Forest.model.random_forest import RandomForest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "dataset", "heart.csv")

    dataset = Dataset(data_path)
    dataset.prepare()
    return dataset

def train_model(x_train, y_train, n_estimators):
    model = RandomForest(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_split=2,
        random_seed=42,
    )
    model.fit(x_train, y_train)
    return model

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm

def tune_model(dataset, n_estimators_list, csv_writer):
    results_acc = {}
    results_f1 = {}

    best_f1 = -1
    best_model = None
    best_params = None

    for n in n_estimators_list:

        model = train_model(
            dataset.x_train,
            dataset.y_train,
            n,
        )

        val_preds = model.predict(dataset.x_val)
        acc, f1, _ = evaluate(dataset.y_val, val_preds)

        print(f"[n_estimators={n}] "
              f"Val Acc={acc:.4f}, F1={f1:.4f}")

        # SAVE TO CSV
        csv_writer.writerow([n, acc, f1])

        results_acc[n] = acc
        results_f1[n] = f1

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = n

    return best_model, best_params, results_acc, results_f1

def plot_results(values, n_estimators_list, title, ylabel):
    plt.figure()
    plt.plot(n_estimators_list, values, marker='o')
    plt.title(title)
    plt.xlabel("n_estimators")
    plt.ylabel(ylabel)
    plt.xticks(n_estimators_list, rotation=45)
    plt.tight_layout()
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

    n_estimators_list = [1, 5, 10, 15, 20, 30, 40, 50, 75, 100]

    with open("plots/rf_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n_estimators", "val_accuracy", "val_f1"])

        best_model, best_params, results_acc, results_f1 = tune_model(
            dataset,
            n_estimators_list,
            writer,
        )

    acc_values = [results_acc[n] for n in n_estimators_list]
    f1_values  = [results_f1[n]  for n in n_estimators_list]

    plot_results(acc_values, n_estimators_list,
                 "Validation Accuracy vs Number of Trees", "Accuracy")

    plot_results(f1_values, n_estimators_list,
                 "Validation F1 Score vs Number of Trees", "F1 Score")

    print("\n===== BEST MODEL =====")
    print(f"n_estimators: {best_params}")

    test_acc, test_f1 = test_model(best_model, dataset)

    # Append test results to CSV
    with open("plots/rf_results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["TEST", test_acc, test_f1])

if __name__ == "__main__":
    main()