import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)

        self.best_val_loss = float('inf')
        self.patience = 3
        self.counter = 0

        self.class_names = [
            "air_conditioner", "car_horn", "children_playing", "dog_bark",
            "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
        ]

    def train(self, num_epochs=15, save_path="best_model.pth"):
        print("\n--- Commencing Training ---")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for features, lengths, labels in self.train_loader:
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features, lengths)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = 100.0 * correct / total

            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for features, lengths, labels in self.val_loader:
                    features = features.to(self.device)
                    lengths = lengths.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(features, lengths)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * labels.size(0)
                    predicted = outputs.argmax(dim=1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(self.val_loader.dataset)
            val_acc = 100.0 * val_correct / val_total

            print(f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
                  f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                self.counter = 0
                print(f"--> Improvement found! Model saved to {save_path}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    def evaluate(self, load_path="best_model.pth"):
        print("\n--- Running Final Evaluation ---")
        self.model.load_state_dict(torch.load(load_path, weights_only=True))
        self.model.eval()

        all_preds, all_targets = [], []

        with torch.no_grad():
            for features, lengths, labels in self.test_loader:
                features = features.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(features, lengths)
                predicted = outputs.argmax(dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.numpy())

        test_accuracy = accuracy_score(all_targets, all_preds)
        test_f1 = f1_score(all_targets, all_preds, average="macro")

        print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
        print(f"Test F1-Score : {test_f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title("Evaluation Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("Ground Truth Labels")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")

        print("Saved matrix to 'confusion_matrix.png'")