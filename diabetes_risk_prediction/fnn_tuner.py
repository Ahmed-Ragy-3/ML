import itertools
import csv
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fnn import FNN
from dataset import Dataset

DATASET_PATH = "C:\\COLLEGE\\Term_8\\pattern\\ML\\diabetes_risk_prediction\\data\\diabetes_012_health_indicators_BRFSS2015.csv"
RESULTS_FILE = "fnn_results.csv"
BEST_CONFIG_FILE = "fnn_best_config.csv"
BEST_MODEL_DIR = "best_fnn_model.keras"
TENSORBOARD_LOG_DIR = "tensorboard_logs"

SEARCH_SPACE = {
    "hidden_layers": [[64, 32, 16]],
    "activations": [['relu']],
    "l2": [0.1],

    "dropout": [0.4],
    "imbalance": ['oversample'],
    "pca": [10],

    "epochs": [10],
    "batch_size": [64],
}


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
   plt.title(title)
   plt.xlabel("Predicted")
   plt.ylabel("Actual")
   plt.show()


def make_dataset(pca, imbalance):
   ds = Dataset(path=DATASET_PATH, feature_scale=True,
                feature_selection=True, pca_components=pca)
   ds.prepare()
   if imbalance:
      ds.handle_imbalance(imbalance)
   return ds


def evaluate(model, ds):
   y_pred = model.predict(ds.x_val)
   return {
       "accuracy":    accuracy_score(ds.y_val, y_pred),
       "f1_micro":    f1_score(ds.y_val, y_pred, average='micro'),
       "f1_macro":    f1_score(ds.y_val, y_pred, average='macro'),
       "f1_weighted": f1_score(ds.y_val, y_pred, average='weighted'),
   }


def make_run_name(pca, imbalance, layers, l2, dropout):
   """Build a unique, human-readable log subfolder name for each run."""
   imb = imbalance if imbalance else "none"
   layers_str = "_".join(str(n) for n in layers)
   return f"pca{pca}_{imb}_layers{layers_str}_l2{l2}_do{dropout}"


def run_search():
   results = []

   for pca, imbalance in itertools.product(SEARCH_SPACE["pca"], SEARCH_SPACE["imbalance"]):
      ds = make_dataset(pca, imbalance)

      inner = (SEARCH_SPACE[k] for k in [
               "hidden_layers", "activations", "l2", "dropout", "epochs", "batch_size"])
      for layers, acts, l2, dropout, epochs, batch in itertools.product(*inner):

         # Build a dedicated TensorBoard log directory for this run
         run_name = make_run_name(pca, imbalance, layers, l2, dropout)
         log_dir = os.path.join(TENSORBOARD_LOG_DIR, run_name)

         tensorboard_cb = tf.keras.callbacks.TensorBoard(
             log_dir=log_dir,
             histogram_freq=1,       # log weight histograms each epoch
             write_graph=False,      # skip graph to keep logs light
             update_freq="epoch",    # log loss/metrics per epoch
         )

         model = FNN(dataset=ds, hidden_layers=layers,
                     activations=acts, l2=l2, dropout=dropout)
         model.fit(epochs=epochs, batch_size=batch, callbacks=[tensorboard_cb])

         metrics = evaluate(model, ds)
         results.append({"pca": pca, "imbalance": imbalance, "hidden_layers": layers,
                         "activations": acts, "l2": l2, "dropout": dropout,
                         "epochs": epochs, "batch_size": batch, **metrics})

         print(f"[pca={pca}][{imbalance}] layers={layers} acc={metrics['accuracy']:.4f} "
               f"f1_macro={metrics['f1_macro']:.4f}  logs -> {log_dir}")

   print(f"\nTensorBoard logs saved to '{TENSORBOARD_LOG_DIR}/'")
   print(f"Launch with:  tensorboard --logdir {TENSORBOARD_LOG_DIR}")
   return results


FIELDNAMES = ["pca", "imbalance", "hidden_layers", "activations", "l2", "dropout",
              "epochs", "batch_size", "accuracy", "f1_micro", "f1_macro", "f1_weighted"]


def save(obj, path):
   if isinstance(obj, list):
      with open(path, 'w', newline='') as f:
         writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
         writer.writeheader()
         writer.writerows(obj)
   else:
      # Single config (best), save as a one-row CSV
      with open(path, 'w', newline='') as f:
         writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
         writer.writeheader()
         writer.writerow(obj)


def load(path):
   with open(path, 'r', newline='') as f:
      reader = csv.DictReader(f)
      rows = list(reader)

   # Cast numeric fields back to correct types
   for row in rows:
      row["pca"] = int(row["pca"])
      row["l2"] = float(row["l2"])
      row["dropout"] = float(row["dropout"])
      row["epochs"] = int(row["epochs"])
      row["batch_size"] = int(row["batch_size"])
      row["accuracy"] = float(row["accuracy"])
      row["f1_micro"] = float(row["f1_micro"])
      row["f1_macro"] = float(row["f1_macro"])
      row["f1_weighted"] = float(row["f1_weighted"])
      row["hidden_layers"] = [int(x) for x in row["hidden_layers"].strip("[]").split(", ")]
      row["activations"] = [
          x.strip("' ") for x in row["activations"].strip("[]").split(", ")]
      row["imbalance"] = None if row["imbalance"] == "None" else row["imbalance"]
   return rows


def already_done():
   return all(os.path.exists(p) for p in [RESULTS_FILE, BEST_CONFIG_FILE, BEST_MODEL_DIR])


def train_best(best):
   ds = make_dataset(best["pca"], best["imbalance"])
   model = FNN(dataset=ds, hidden_layers=best["hidden_layers"], activations=best["activations"],
               l2=best["l2"], dropout=best["dropout"])
   model.fit(epochs=best["epochs"], batch_size=best["batch_size"])
   return model, ds


def main():
   if already_done():
      print("Loading saved results...")
      results = load(RESULTS_FILE)
      best = load(BEST_CONFIG_FILE)[0]
      model = FNN(dataset=Dataset(DATASET_PATH, feature_scale=True))
      model.model = tf.keras.models.load_model(BEST_MODEL_DIR)
      return results, best, model

   results = run_search()
   best = max(results, key=lambda x: x["f1_macro"])  # Select best by F1 macro
   print("\nBest config:", best)

   save(results, RESULTS_FILE)
   # save(best, BEST_CONFIG_FILE)

   model, ds = train_best(best)
   model.model.save(BEST_MODEL_DIR)

   plot_confusion(ds.y_val, model.predict(ds.x_val),
                  title="Confusion Matrix (Validation Set)")
   return results, best, model


if __name__ == "__main__":
   main()
