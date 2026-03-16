import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
   def __init__(self, model, x_test, y_test):
      self.model = model
      self.x_test = x_test
      self.y_test = y_test
      self.y_pred = None

   def evaluate(self):
      """Run predictions and compute all metrics."""
      self.y_pred = self.model.predict(self.x_test)
      results = {
			"accuracy": self.compute_accuracy(),
			"f1_micro": self.compute_f1(average="micro"),
			"f1_macro": self.compute_f1(average="macro"),
			"f1_weighted": self.compute_f1(average="weighted"),
      }
      return results

   def compute_accuracy(self):
      return accuracy_score(self.y_test, self.y_pred)

   def compute_f1(self, average="micro"):
      return f1_score(self.y_test, self.y_pred, average=average)

   def plot_confusion_matrix(self):
      cm = confusion_matrix(self.y_test, self.y_pred)
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
      plt.title("Confusion Matrix")
      plt.xlabel("Predicted")
      plt.ylabel("Actual")
      plt.show()