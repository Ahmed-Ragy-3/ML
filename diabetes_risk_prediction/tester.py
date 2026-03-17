# test_models.py
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset import Dataset
from knn import KNN
from softmax import Softmax
from fnn import FNN

# Dataset path
DATASET_PATH = "C:\\COLLEGE\\Term_8\\pattern\\ML\\diabetes_risk_prediction\\data\\diabetes_012_health_indicators_BRFSS2015.csv"

# Paths where models/results are saved
FNN_MODEL_DIR = "best_fnn_model"
FNN_RESULTS_FILE = "fnn_results.pkl"
FNN_BEST_CONFIG_FILE = "fnn_best_config.pkl"

SOFTMAX_MODEL_DIR = "best_softmax_model"
SOFTMAX_RESULTS_FILE = "softmax_results.pkl"
SOFTMAX_BEST_CONFIG_FILE = "softmax_best_config.pkl"

KNN_RESULTS_FILE = "knn_results.pkl"
KNN_BEST_CONFIG_FILE = "knn_best_config.pkl"


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
   plt.title(title)
   plt.xlabel("Predicted")
   plt.ylabel("Actual")
   plt.show()


def evaluate_model(name, model, x_test, y_test):
   y_pred = model.predict(x_test)
   acc = accuracy_score(y_test, y_pred)
   f1_micro = f1_score(y_test, y_pred, average='micro')
   f1_macro = f1_score(y_test, y_pred, average='macro')
   f1_weighted = f1_score(y_test, y_pred, average='weighted')

   print(f"\n=== {name} Evaluation ===")
   print(f"Accuracy: {acc:.4f}")
   print(f"F1 Micro: {f1_micro:.4f}")
   print(f"F1 Macro: {f1_macro:.4f}")
   print(f"F1 Weighted: {f1_weighted:.4f}")

   plot_confusion(y_test, y_pred, title=f"{name} Confusion Matrix")
   return acc, f1_micro, f1_macro, f1_weighted


def load_dataset():
   ds = Dataset(path=DATASET_PATH, feature_scale=True)
   ds.prepare()
   return ds


def load_fnn_model(ds: Dataset):
   import tensorflow as tf
   if os.path.exists(FNN_MODEL_DIR):
      print("Loading saved FNN model...")
      fnn = FNN(dataset=ds)  # dummy instance
      fnn.model = tf.keras.models.load_model(FNN_MODEL_DIR)
      return fnn
   else:
      print("No saved FNN model found!")
      return None


def load_softmax_model(ds: Dataset):
   import tensorflow as tf
   if os.path.exists(SOFTMAX_MODEL_DIR):
      print("Loading saved Softmax model...")
      sm = Softmax(input_dim=ds.input_dim())  # dummy instance
      sm.model = tf.keras.models.load_model(SOFTMAX_MODEL_DIR)
      return sm
   else:
      print("No saved Softmax model found!")
      return None


def load_knn_model(ds: Dataset):
   if os.path.exists(KNN_BEST_CONFIG_FILE):
      with open(KNN_BEST_CONFIG_FILE, 'rb') as f:
         best_config = pickle.load(f)
      knn = KNN(dataset=ds)
      knn.set_k(best_config['k'])
      knn.set_dist(best_config['distance'])
      if best_config['balance_method']:
         ds.handle_imbalance(best_config['balance_method'])
      knn.fit(ds.x_train, ds.y_train)
      return knn
   else:
      print("No saved KNN best config found!")
      return None


def main():
   ds = load_dataset()

   # Load and test FNN
   fnn_model = load_fnn_model(ds)
   if fnn_model:
      evaluate_model("FNN", fnn_model, ds.x_test, ds.y_test)

   # Load and test Softmax
   softmax_model = load_softmax_model(ds)
   if softmax_model:
      evaluate_model("Softmax", softmax_model, ds.x_test, ds.y_test)

   # Load and test KNN
   knn_model = load_knn_model(ds)
   if knn_model:
      evaluate_model("KNN", knn_model, ds.x_test, ds.y_test)


if __name__ == "__main__":
   main()
