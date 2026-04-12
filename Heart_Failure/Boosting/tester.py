import os
import matplotlib.pyplot as plt

from Heart_Failure.dataset import Dataset
from Heart_Failure.Boosting.adaboost import AdaBoost   # adjust import
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def evaluate(name, y_true, y_pred):
   acc = accuracy_score(y_true, y_pred)
   f1 = f1_score(y_true, y_pred)
   cm = confusion_matrix(y_true, y_pred)

   print(f"\n===== {name} =====")
   print(f"Accuracy: {acc:.4f}")
   print(f"F1 Score: {f1:.4f}")

   plot_confusion_matrix(y_true, y_pred, name)


def plot_confusion_matrix(y_true, y_pred, title):
   ConfusionMatrixDisplay.from_predictions(
       y_true,
       y_pred,
       display_labels=["-1", "1"],   # IMPORTANT FIX
       cmap=plt.cm.Blues
   )
   plt.title(title)
   plt.show()


if __name__ == "__main__":

   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(BASE_DIR, "..", "dataset", "heart.csv")

   dataset = Dataset(data_path)
   dataset.prepare()

   # REQUIRED: convert labels to {-1, +1}
   dataset.y_train = dataset.y_train.replace({0: -1, 1: 1})
   dataset.y_val = dataset.y_val.replace({0: -1, 1: 1})
   dataset.y_test = dataset.y_test.replace({0: -1, 1: 1})

   model = AdaBoost(iterations=50)
   model.fit(dataset.x_train, dataset.y_train)

   # VALIDATION
   val_preds = model.predict(dataset.x_val)
   evaluate("VALIDATION", dataset.y_val, val_preds)

   # TEST
   test_preds = model.predict(dataset.x_test)
   evaluate("TEST", dataset.y_test, test_preds)
