"""
trainer.py – Training, evaluation, and visualisation utilities.

Produces per-experiment:
    • Epoch-by-epoch train/val loss and accuracy curves
    • Confusion matrix on the test set
    • Printed accuracy and macro F1-score

Also provides a standalone `plot_comparison` function that draws a
side-by-side bar chart of accuracy / F1 across all experiments.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling",        "engine_idling", "gun_shot", "jackhammer",
    "siren",           "street_music",
]


# ──────────────────────────────────────────────────────────────────────────────
# ModelTrainer
# ──────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Wraps a PyTorch model with training, validation, and test-evaluation loops.

    Parameters
    ----------
    model          : nn.Module
    train_loader   : DataLoader for training split
    val_loader     : DataLoader for validation split
    test_loader    : DataLoader for test split
    learning_rate  : Adam learning rate
    experiment_name: used for file names (model checkpoint, plots, etc.)
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        learning_rate: float   = 1e-3,
        experiment_name: str   = "experiment",
    ):
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name
        print(f"[{experiment_name}] Trainer on device: {self.device}")

        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, verbose=False
        )

        self.best_val_loss = float("inf")
        self.patience      = 5
        self.counter       = 0

        # History for plotting
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, num_epochs: int = 15, save_path: str = None):
        if save_path is None:
            save_path = f"{self.experiment_name}_best.pth"

        print(f"\n[{self.experiment_name}] ── Training for up to {num_epochs} epochs ──")

        for epoch in range(num_epochs):
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for features, lengths, labels in self.train_loader:
                features = features.to(self.device)
                lengths  = lengths.to(self.device)
                labels   = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features, lengths)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                correct      += (outputs.argmax(dim=1) == labels).sum().item()
                total        += labels.size(0)

            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc  = 100.0 * correct / total

            # ── Validation ────────────────────────────────────────────────────
            val_loss, val_acc = self._evaluate_split(self.val_loader)
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"  Epoch {epoch+1:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
            )

            # ── Early stopping & checkpointing ────────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                self.counter = 0
                print(f"    → Improvement! Checkpoint saved to '{save_path}'")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}.")
                    break

        self._plot_history(save_path.replace(".pth", "_history.png"))
        return save_path

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def _evaluate_split(self, loader):
        """Return (avg_loss, accuracy%) on a DataLoader."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for features, lengths, labels in loader:
                features = features.to(self.device)
                lengths  = lengths.to(self.device)
                labels   = labels.to(self.device)
                outputs  = self.model(features, lengths)
                loss     = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                correct    += (outputs.argmax(dim=1) == labels).sum().item()
                total      += labels.size(0)
        return total_loss / len(loader.dataset), 100.0 * correct / total

    def evaluate(self, load_path: str = None) -> dict:
        """
        Full evaluation on the test set.
        Loads the best checkpoint, computes accuracy / F1, plots confusion matrix.
        Returns a dict with 'accuracy' and 'f1' keys.
        """
        if load_path is None:
            load_path = f"{self.experiment_name}_best.pth"

        print(f"\n[{self.experiment_name}] ── Final Evaluation on Test Set ──")
        self.model.load_state_dict(
            torch.load(load_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for features, lengths, labels in self.test_loader:
                features = features.to(self.device)
                lengths  = lengths.to(self.device)
                outputs  = self.model(features, lengths)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_targets.extend(labels.numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1  = f1_score(all_targets, all_preds, average="macro")

        print(f"  Test Accuracy : {acc * 100:.2f}%")
        print(f"  Test F1-Score : {f1:.4f}")

        cm_path = f"{self.experiment_name}_confusion_matrix.png"
        self._plot_confusion_matrix(all_targets, all_preds, cm_path)

        return {"accuracy": acc, "f1": f1}

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _plot_history(self, save_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss", marker="o")
        axes[0].plot(epochs, self.history["val_loss"],   label="Val Loss",   marker="s")
        axes[0].set_title(f"{self.experiment_name} – Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history["train_acc"], label="Train Acc", marker="o")
        axes[1].plot(epochs, self.history["val_acc"],   label="Val Acc",   marker="s")
        axes[1].set_title(f"{self.experiment_name} – Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Training history saved to '{save_path}'")

    def _plot_confusion_matrix(self, targets, preds, save_path: str):
        cm = confusion_matrix(targets, preds)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            cm,
            annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_title(f"{self.experiment_name} – Confusion Matrix", fontsize=14)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved to '{save_path}'")


# ──────────────────────────────────────────────────────────────────────────────
# Comparison chart across experiments
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, save_path: str = "comparison.png"):
    """
    Draw a grouped bar chart comparing accuracy and F1 across experiments.

    Parameters
    ----------
    results   : {experiment_name: {"accuracy": float, "f1": float}}
    save_path : output filename
    """
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    f1s   = [results[n]["f1"] * 100       for n in names]

    x   = np.arange(len(names))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 6))
    bars_acc = ax.bar(x - w / 2, accs, w, label="Accuracy (%)", color="steelblue")
    bars_f1  = ax.bar(x + w / 2, f1s,  w, label="F1-Score (%)", color="darkorange")

    # Value labels
    for bar in bars_acc:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=9,
        )
    for bar in bars_f1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_title("Model Comparison – Test Accuracy & F1-Score", fontsize=14)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nComparison chart saved to '{save_path}'")