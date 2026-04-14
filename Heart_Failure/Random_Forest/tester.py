import os
import matplotlib.pyplot as plt

from dataset import Dataset
from Random_Forest.model.random_forest import RandomForest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate(name, y_true, y_pred, filename):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plot_confusion_matrix(y_true, y_pred, name, filename)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["0", "1"],
        cmap=plt.cm.Blues,
    )
    plt.title(title)
    plt.savefig(os.path.join("plots", filename))  # save plot
    plt.close()

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "dataset", "heart.csv")

    dataset = Dataset(data_path)
    dataset.prepare()

    # Best n_estimators found by the tuner
    model = RandomForest(n_estimators=20, max_depth=5, min_samples_split=2, random_seed=42)
    model.fit(dataset.x_train, dataset.y_train)

    # VALIDATION
    val_preds = model.predict(dataset.x_val)
    evaluate("VALIDATION", dataset.y_val, val_preds, "random_forest_validation_confusion.png")

    # TEST
    test_preds = model.predict(dataset.x_test)
    evaluate("TEST", dataset.y_test, test_preds, "random_forest_test_confusion.png")
