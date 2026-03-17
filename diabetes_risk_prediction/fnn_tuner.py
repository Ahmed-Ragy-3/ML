import itertools
import pickle
import os
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

def fnn_tuner_save_load(dataset_path,
                        hidden_layers_options=[[64, 32, 16], [128, 64, 32]],
                        activations_options=[
                            ['relu', 'relu', 'relu'], ['tanh', 'tanh', 'relu']],
                        l2_values=[0.0, 0.001, 0.01],
                        dropout_values=[0.0, 0.2],
                        imbalance_methods=['oversample', 'undersample', None],
                        epochs_list=[5, 10],
                        batch_sizes=[32, 64],
                        results_file="fnn_results.pkl",
                        best_config_file="fnn_best_config.pkl",
                        best_model_dir="best_fnn_model"):

   # Check if results and model already exist
   if os.path.exists(results_file) and os.path.exists(best_config_file) and os.path.exists(best_model_dir):
      print("Loading saved results, best config, and model...")
      with open(results_file, 'rb') as f:
         results = pickle.load(f)
      with open(best_config_file, 'rb') as f:
         best = pickle.load(f)
      # Load the best model
      best_model = FNN(dataset=Dataset(
          dataset_path, feature_scale=True))  # dummy instance
      import tensorflow as tf
      best_model.model = tf.keras.models.load_model(best_model_dir)
      return results, best, best_model

   results = []

   # Load dataset
   ds = Dataset(path=dataset_path, feature_scale=True)
   ds.prepare()

   for imbalance in imbalance_methods:
      ds.prepare()
      if imbalance:
         ds.handle_imbalance(imbalance)

      for hidden_layers, activations, l2, dropout, epochs, batch_size in itertools.product(
          hidden_layers_options,
          activations_options,
          l2_values,
          dropout_values,
          epochs_list,
          batch_sizes
      ):

         # Train FNN
         model = FNN(dataset=ds, hidden_layers=hidden_layers,
                     activations=activations, l2=l2, dropout=dropout)
         model.fit(epochs=epochs, batch_size=batch_size)

         # Predict on validation
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

         print(f"[{imbalance}] Layers={hidden_layers}, act={activations}, L2={l2}, dropout={dropout}, "
               f"epochs={epochs}, batch={batch_size} -> acc={acc:.4f}, "
               f"f1_micro={f1_micro:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}")

   # Best configuration by accuracy
   best = max(results, key=lambda x: x['accuracy'])
   print("\nBest Hyperparameter Configuration:")
   print(best)

   # Save results and best config
   with open(results_file, 'wb') as f:
      pickle.dump(results, f)
   with open(best_config_file, 'wb') as f:
      pickle.dump(best, f)
   print(f"Results saved to '{results_file}'")
   print(f"Best config saved to '{best_config_file}'")

   # Train best model again and save
   ds.prepare()
   if best['imbalance']:
      ds.handle_imbalance(best['imbalance'])

   best_model = FNN(dataset=ds,
                    hidden_layers=best['hidden_layers'],
                    activations=best['activations'],
                    l2=best['l2'],
                    dropout=best['dropout'])
   best_model.fit(epochs=best['epochs'], batch_size=best['batch_size'])

   import tensorflow as tf
   best_model.model.save(best_model_dir)
   print(f"Best FNN model saved to '{best_model_dir}'")

   # Confusion matrix for best model
   y_best_pred = best_model.predict(ds.x_val)
   plot_confusion(ds.y_val, y_best_pred,
                  title="Confusion Matrix (Validation Set)")

   return results, best, best_model


# Example usage
dataset_path = "C:\\COLLEGE\\Term_8\\pattern\\ML\\diabetes_risk_prediction\\data\\diabetes_012_health_indicators_BRFSS2015.csv"
results, best_config, best_model = fnn_tuner_save_load(dataset_path)
