import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
from Boosting.adaboost import AdaBoost
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_data():
   base_dir = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_dir, "..", "dataset", "heart.csv")

   dataset = Dataset(data_path)
   dataset.prepare()

   # IMPORTANT: AdaBoost expects {-1, +1}
   dataset.y_train = dataset.y_train.replace({0: -1, 1: 1})
   dataset.y_val = dataset.y_val.replace({0: -1, 1: 1})
   dataset.y_test = dataset.y_test.replace({0: -1, 1: 1})

   return dataset


def train_model(x_train, y_train, iterations):
   model = AdaBoost(iterations=iterations)
   model.fit(x_train, y_train)
   return model


def evaluate(y_true, y_pred):
   acc = accuracy_score(y_true, y_pred)
   f1 = f1_score(y_true, y_pred)
   cm = confusion_matrix(y_true, y_pred)
   return acc, f1, cm


def tune_model(dataset, iterations_list, csv_writer):
   results_acc = {}
   results_f1 = {}

   best_f1 = -1
   best_model = None
   best_iter = None

   for it in iterations_list:

      model = train_model(
          dataset.x_train,
          dataset.y_train,
          it
      )

      val_preds = model.predict(dataset.x_val)
      acc, f1, _ = evaluate(dataset.y_val, val_preds)

      print(f"[iterations={it}] Val Acc={acc:.4f}, F1={f1:.4f}")

      # SAVE TO CSV
      csv_writer.writerow([it, acc, f1])

      results_acc[it] = acc
      results_f1[it] = f1

      if f1 > best_f1:
         best_f1 = f1
         best_model = model
         best_iter = it

   return best_model, best_iter, results_acc, results_f1


def plot_curve(values, scores, title, ylabel):
   plt.figure()
   plt.plot(values, scores, marker='o')
   plt.title(title)
   plt.xlabel("Iterations")
   plt.ylabel(ylabel)
   plt.grid()
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

   iterations_list = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]

	# Ensure directory exists
   os.makedirs("plots", exist_ok=True)

   with open("plots/boosting_results_adaboost.csv", mode="w", newline="") as file:
      writer = csv.writer(file)
      writer.writerow(["iterations", "val_accuracy", "val_f1"])

      best_model, best_iter, results_acc, results_f1 = tune_model(
          dataset,
          iterations_list,
          writer
      )

   acc_scores = [results_acc[it] for it in iterations_list]
   f1_scores = [results_f1[it] for it in iterations_list]

   plot_curve(iterations_list, acc_scores,
              "Validation Accuracy vs Iterations", "Accuracy")

   plot_curve(iterations_list, f1_scores,
              "Validation F1 Score vs Iterations", "F1 Score")

   print("\n===== BEST MODEL =====")
   print(f"Best Iterations: {best_iter}")

   test_acc, test_f1 = test_model(best_model, dataset)

   # append test results
   with open("plots/boosting_results_adaboost.csv", mode="a", newline="") as file:
      writer = csv.writer(file)
      writer.writerow([])
      writer.writerow(["TEST", test_acc, test_f1])


if __name__ == "__main__":
   main()
