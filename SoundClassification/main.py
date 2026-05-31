"""
main.py – Full pipeline for Assignment #4: Environmental Sound Classification

Runs the following experiments in order:
    1. GRU + Mel Spectrogram  (150 features, 22 050 Hz)
    2. GRU + Energy           (1 feature,   22 050 Hz)
    3. GRU + MFCC             (40 features, 22 050 Hz)
    4. CNN-GRU + Mel          (Bonus)

For each experiment:
    • Preprocesses the dataset (once, cached to disk)
    • Trains with early stopping (patience = 5)
    • Evaluates on the test set → accuracy, macro F1, confusion matrix

Finally:
    • Plots a comparison bar chart across all experiments
    • Generates waveform and spectrogram visualisations for one sample per class

Random seed is fixed for reproducibility.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import UrbanSoundFeatureDataset, BATCH_SIZE
from model import EnvironmentalGRUClassifier, CNNGRUClassifier
from trainer import ModelTrainer, plot_comparison
from visualize import run_visualisations


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42


def set_seed(seed: int = SEED):
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

BASE_KWARGS = {
    "root_dir":     "UrbanSound8K",
    "processed_dir": "processed",
}

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
VAL_FOLDS = [7, 8]
TEST_FOLDS = [9, 10]


def build_loaders(feature_type: str):
   """Return (train_loader, val_loader, test_loader) for a given feature type."""
   train_ds = UrbanSoundFeatureDataset(
       folds=TRAIN_FOLDS, feature_type=feature_type, **BASE_KWARGS)
   val_ds = UrbanSoundFeatureDataset(
       folds=VAL_FOLDS,   feature_type=feature_type, **BASE_KWARGS)
   test_ds = UrbanSoundFeatureDataset(
       folds=TEST_FOLDS,  feature_type=feature_type, **BASE_KWARGS)

   train_ds.preprocess_all()
   val_ds.preprocess_all()
   test_ds.preprocess_all()

   collate = UrbanSoundFeatureDataset.collate_batches

   train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                             shuffle=True,  collate_fn=collate, num_workers=0)
   val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate, num_workers=0)
   test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate, num_workers=0)

   return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(name: str, model, train_loader, val_loader, test_loader,
                   num_epochs: int = 15, lr: float = 1e-3) -> dict:
   """Train and evaluate one model; return metrics dict."""
   set_seed()
   trainer = ModelTrainer(
       model, train_loader, val_loader, test_loader,
       learning_rate=lr,
       experiment_name=name,
   )
   ckpt = trainer.train(num_epochs=num_epochs)
   return trainer.evaluate(load_path=ckpt)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
   print("=" * 70)
   print("Assignment #4 – Environmental Sound Classification")
   print("=" * 70)

   # ── Step 0: Visualisations ────────────────────────────────────────────────
   print("\n[0] Generating waveform & spectrogram visualisations …")
   try:
      run_visualisations(output_dir=".")
   except Exception as exc:
      print(f"  Visualisation skipped (dataset not found or error): {exc}")

   results = {}

   # ═════════════════════════════════════════════════════════════════════════
   # Experiment 1: GRU + Mel Spectrogram (150 features)
   # ═════════════════════════════════════════════════════════════════════════
   # print("\n" + "═" * 70)
   # print("Experiment 1 of 4 – GRU + Mel Spectrogram (150 features)")
   # print("═" * 70)
   #
   # train_l, val_l, test_l = build_loaders("mel")
   # model_mel = EnvironmentalGRUClassifier(input_size=150)
   # results["GRU-Mel"] = run_experiment(
   #     "GRU-Mel", model_mel, train_l, val_l, test_l
   # )

   # ═════════════════════════════════════════════════════════════════════════
   # Experiment 2: GRU + Energy (1 feature)
   # ═════════════════════════════════════════════════════════════════════════
   print("\n" + "═" * 70)
   print("Experiment 2 of 4 – GRU + Energy (1 feature)")
   print("═" * 70)

   train_l, val_l, test_l = build_loaders("energy")
   model_energy = EnvironmentalGRUClassifier(
       input_size=1, hidden_size=64, num_layers=2
   )
   results["GRU-Energy"] = run_experiment(
       "GRU-Energy", model_energy, train_l, val_l, test_l
   )

   # ═════════════════════════════════════════════════════════════════════════
   # Experiment 3: GRU + MFCC (40 features)
   # ═════════════════════════════════════════════════════════════════════════
   print("\n" + "═" * 70)
   print("Experiment 3 of 4 – GRU + MFCC (40 features)")
   print("═" * 70)

   train_l, val_l, test_l = build_loaders("mfcc")
   model_mfcc = EnvironmentalGRUClassifier(input_size=40)
   results["GRU-MFCC"] = run_experiment(
       "GRU-MFCC", model_mfcc, train_l, val_l, test_l
   )

   # ═════════════════════════════════════════════════════════════════════════
   # Experiment 4 (BONUS): CNN-GRU + Mel (150 features)
   # ═════════════════════════════════════════════════════════════════════════
   print("\n" + "═" * 70)
   print("Experiment 4 of 4 – CNN-GRU + Mel (BONUS) (150 features)")
   print("═" * 70)

   train_l, val_l, test_l = build_loaders("mel")
   model_cnn_gru = CNNGRUClassifier(
       input_size=150,
       cnn_channels=[32, 64],
       gru_hidden=128,
       gru_layers=2,
   )
   results["CNN-GRU-Mel"] = run_experiment(
       "CNN-GRU-Mel", model_cnn_gru, train_l, val_l, test_l
   )

   # ── Final comparison ──────────────────────────────────────────────────────
   print("\n" + "=" * 70)
   print("SUMMARY")
   print("=" * 70)
   print(f"{'Experiment':<20} {'Accuracy':>12} {'F1-Score':>12}")
   print("-" * 46)
   for name, metrics in results.items():
      print(
          f"{name:<20} {metrics['accuracy']*100:>11.2f}% {metrics['f1']:>12.4f}"
      )

   plot_comparison(results, save_path="comparison.png")
   print("\nAll experiments complete. Output files:")
   print("  • waveforms_all_classes.png")
   print("  • mel_spectrograms_all_classes.png")
   print("  • mfcc_spectrograms_all_classes.png")
   print("  • combined_visualisations.png")
   print("  • GRU-Mel_confusion_matrix.png")
   print("  • GRU-Energy_confusion_matrix.png")
   print("  • GRU-MFCC_confusion_matrix.png")
   print("  • CNN-GRU-Mel_confusion_matrix.png")
   print("  • <experiment>_history.png  (one per experiment)")
   print("  • comparison.png")


if __name__ == "__main__":
   main()
