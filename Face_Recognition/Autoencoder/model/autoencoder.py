import torch.nn as nn
import torch.optim as optim

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden1=1024, bottleneck=128):
        super(DeepAutoencoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

    def transform(self, x):
        # encoder part only: first 4 layers
        return self.network[:4](x)

    def reconstruct(self, x):
        return self.forward(x)

    def train_autoencoder(model, X_train, epochs=100, lr=1e-3, batch_size=32, verbose=True):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for i in range(0, X_train.size(0), batch_size):
                xb = X_train[i:i+batch_size]
                optimizer.zero_grad()
                x_hat = model(xb)
                loss = criterion(x_hat, xb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / (X_train.size(0) // batch_size))
            if verbose and (epoch == 1 or epoch % 10 == 0):
                print(f"Epoch {epoch}/{epochs}, Loss={losses[-1]:.6f}")

        return model, losses
