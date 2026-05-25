import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EnvironmentalGRUClassifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=3, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.norm = nn.LayerNorm(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        max_len = gru_out.size(1)
        mask = torch.arange(max_len, device=gru_out.device)[None, :] < lengths[:, None]
        gru_out = gru_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        pooled = torch.max(gru_out, dim=1).values
        pooled = self.norm(pooled)
        return self.fc(pooled)