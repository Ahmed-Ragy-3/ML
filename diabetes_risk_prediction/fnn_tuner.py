import itertools
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fnn import FNN
from dataset import Dataset

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
   plt.title(title)
   plt.xlabel("Predicted")
   plt.ylabel("Actual")
   plt.show()


def fnn_hyperparameter_tuner(dataset_path,
                             hidden_layers_options=[
                                 [64, 32, 16], [128, 64, 32]],
                             activations_options=[
                                 ['relu', 'relu', 'relu'], ['tanh', 'tanh', 'relu']],
                             l2_values=[0.0, 0.001, 0.01],
                             dropout_values=[0.0, 0.2],
                             imbalance_methods=[
                                 'oversample', 'undersample', None],
                             epochs_list=[20, 50],
                             batch_sizes=[32, 64]):
   results = []

   # Load and prepare dataset
   ds = Dataset(path=dataset_path, feature_scale=True)
   ds.prepare()

   # Iterate over all combinations
   for hidden_layers, activations, l2, dropout, imbalance, epochs, batch_size in itertools.product(
       hidden_layers_options,
       activations_options,
       l2_values,
       dropout_values,
       imbalance_methods,
       epochs_list,
       batch_sizes
   ):
      # Reset dataset
      ds.prepare()

      # Handle imbalance
      if imbalance:
         ds.handle_imbalance(imbalance)

      # Build and train FNN
      model = FNN(dataset=ds, hidden_layers=hidden_layers,
                  activations=activations, l2=l2, dropout=dropout)
      model.fit(epochs=epochs, batch_size=batch_size)

      # Predict on validation set
      y_val_pred = model.predict(ds.x_val)

      # Metrics
      acc = accuracy_score(ds.y_val, y_val_pred)
      f1_micro = f1_score(ds.y_val, y_val_pred, average='micro')
      f1_macro = f1_score(ds.y_val, y_val_pred, average='macro')
      f1_weighted = f1_score(ds.y_val, y_val_pred, average='weighted')

      results.append({
          'hidden_layers': hidden_layers,
          'activations': activations,
          'l2': l2,
          'dropout': dropout,
          'imbalance': imbalance,
          'epochs': epochs,
          'batch_size': batch_size,
          'accuracy': acc,
          'f1_micro': f1_micro,
          'f1_macro': f1_macro,
          'f1_weighted': f1_weighted
      })

      # Print F1 scores for this combination
      print(f"[{imbalance}] Layers={hidden_layers}, act={activations}, L2={l2}, dropout={dropout}, "
            f"epochs={epochs}, batch={batch_size} -> acc={acc:.4f}, f1_macro={f1_macro:.4f}, "
            f"f1_micro={f1_micro:.4f}, f1_weighted={f1_weighted:.4f}")

   # Best configuration by accuracy
   best = max(results, key=lambda x: x['accuracy'])
   print("\nBest Hyperparameter Configuration:")
   print(best)

   # Confusion matrix for best config
   print("\nConfusion Matrix for Best Configuration (Validation Set):")
   ds.prepare()
   if best['imbalance']:
      ds.handle_imbalance(best['imbalance'])
   best_model = FNN(dataset=ds,
                    hidden_layers=best['hidden_layers'],
                    activations=best['activations'],
                    l2=best['l2'],
                    dropout=best['dropout'])
   best_model.fit(epochs=best['epochs'], batch_size=best['batch_size'])
   y_best_pred = best_model.predict(ds.x_val)
   plot_confusion(ds.y_val, y_best_pred,
                  title="Confusion Matrix (Validation Set)")

   return results, best


dataset_path = "C:\\COLLEGE\\Term_8\\pattern\\ML\\diabetes_risk_prediction\\data\\diabetes_012_health_indicators_BRFSS2015.csv"
results, best_config = fnn_hyperparameter_tuner(
    dataset_path,
    hidden_layers_options=[[64, 32, 16], [128, 64, 32], [128, 64, 32, 16]],
    activations_options=[['relu', 'relu', 'relu'], ['tanh', 'tanh', 'relu'], ['relu', 'relu', 'relu', 'relu']],
    l2_values=[0.0, 0.001, 0.01],
    dropout_values=[0.0, 0.2],
    imbalance_methods=['oversample', 'undersample', None],
    epochs_list=[5, 10],
    batch_sizes=[32, 64]
)