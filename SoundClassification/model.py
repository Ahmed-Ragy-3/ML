"""
model.py – Model architectures for UrbanSound8K classification.

    EnvironmentalGRUClassifier  – Bidirectional GRU (baseline)
    CNNGRUClassifier            – CNN feature extractor + Bidirectional GRU (bonus)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ──────────────────────────────────────────────────────────────────────────────
# 1. Baseline: Bidirectional GRU
# ──────────────────────────────────────────────────────────────────────────────

class EnvironmentalGRUClassifier(nn.Module):
   """
   Bidirectional multi-layer GRU with max-pooling over time.

   Parameters
   ----------
   input_size  : feature dimension (150 for Mel, 1 for Energy, 40 for MFCC)
   hidden_size : GRU hidden units per direction
   num_layers  : stacked GRU layers
   num_classes : number of output classes (10 for UrbanSound8K)
   dropout     : dropout probability between GRU layers and in the FC head
   """

   def __init__(
       self,
       input_size: int = 150,
       hidden_size: int = 128,
       num_layers: int = 3,
       num_classes: int = 10,
       dropout: float = 0.2,
   ):
      super().__init__()

      self.gru = nn.GRU(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          batch_first=True,
          bidirectional=True,
          dropout=dropout if num_layers > 1 else 0.0,
      )
      self.norm = nn.LayerNorm(hidden_size * 2)

      self.fc = nn.Sequential(
          nn.Linear(hidden_size * 2, 128),
          nn.ReLU(),
          nn.Dropout(0.3),
          nn.Linear(128, num_classes),
      )

   def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
      packed = pack_padded_sequence(
          x, lengths.cpu(), batch_first=True, enforce_sorted=False
      )
      packed_out, _ = self.gru(packed)
      gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)

      # Mask padding positions before max-pool
      max_len = gru_out.size(1)
      mask = (
          torch.arange(max_len, device=gru_out.device)[None, :]
          < lengths[:, None]
      )
      gru_out = gru_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))

      pooled = torch.max(gru_out, dim=1).values   # (B, hidden*2)
      pooled = self.norm(pooled)
      return self.fc(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Bonus: CNN-GRU Classifier
# ──────────────────────────────────────────────────────────────────────────────

class CNNGRUClassifier(nn.Module):
   """
   CNN-GRU hybrid architecture (Bonus).

   Architecture
   ------------
   Input  : (B, T, F)  – batch-first time × feature tensor
   ↓ Reshape to (B, 1, T, F) – treat as single-channel "image"
   ↓ 2-D CNN blocks along the feature axis → local spectral patterns
   ↓ Flatten feature dim → (B, T', C)  – new temporal sequence
   ↓ Bidirectional GRU over time
   ↓ Max-pool + FC head → (B, num_classes)

   The CNN reduces the feature dimension while capturing local correlations,
   giving the GRU a richer, compressed representation to model temporal dynamics.

   Parameters
   ----------
   input_size  : original feature dimension (F)
   cnn_channels: list of output channels for successive Conv2d blocks
   gru_hidden  : GRU hidden units per direction
   gru_layers  : stacked GRU layers
   num_classes : 10 for UrbanSound8K
   dropout     : dropout in GRU and FC head
   """

   def __init__(
       self,
       input_size: int = 150,
       cnn_channels: list = None,
       gru_hidden: int = 128,
       gru_layers: int = 2,
       num_classes: int = 10,
       dropout: float = 0.3,
   ):
      super().__init__()

      if cnn_channels is None:
         cnn_channels = [32, 64]

      # ── CNN blocks ────────────────────────────────────────────────────────
      # Each block: Conv2d (kernel 3×3, 'same' padding on freq axis) →
      #             BatchNorm → ReLU → MaxPool on freq axis only
      cnn_layers = []
      in_ch = 1
      freq_size = input_size   # tracks feature dimension after each pool

      for out_ch in cnn_channels:
         cnn_layers += [
             # (B, in_ch, T, F)
             nn.Conv2d(
                 in_ch, out_ch,
                 kernel_size=(1, 3),   # 1 along time, 3 along freq
                 padding=(0, 1),       # same padding on freq axis
             ),
             nn.BatchNorm2d(out_ch),
             nn.ReLU(),
             # Pool only on the frequency axis
             nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
         ]
         freq_size = freq_size // 2
         in_ch = out_ch

      self.cnn = nn.Sequential(*cnn_layers)

      # Flattened feature size after CNN: last_ch × reduced_freq
      cnn_out_size = cnn_channels[-1] * freq_size

      # ── GRU ───────────────────────────────────────────────────────────────
      self.gru = nn.GRU(
          input_size=cnn_out_size,
          hidden_size=gru_hidden,
          num_layers=gru_layers,
          batch_first=True,
          bidirectional=True,
          dropout=dropout if gru_layers > 1 else 0.0,
      )
      self.norm = nn.LayerNorm(gru_hidden * 2)

      # ── Classification head ───────────────────────────────────────────────
      self.fc = nn.Sequential(
          nn.Linear(gru_hidden * 2, 128),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(128, num_classes),
      )

   def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
      # x: (B, T, F)
      B, T, F = x.shape

      # ── CNN forward ───────────────────────────────────────────────────────
      cnn_in = x.unsqueeze(1)          # (B, 1, T, F)
      # (B, C, T, F')  – T unchanged, F' reduced
      cnn_out = self.cnn(cnn_in)
      B, C, T2, F2 = cnn_out.shape
      cnn_out = cnn_out.permute(0, 2, 1, 3)  # (B, T, C, F')
      cnn_out = cnn_out.reshape(B, T2, C * F2)  # (B, T, C*F')

      # ── GRU forward ───────────────────────────────────────────────────────
      packed = pack_padded_sequence(
          cnn_out, lengths.cpu(), batch_first=True, enforce_sorted=False
      )
      packed_out, _ = self.gru(packed)
      gru_out,   _ = pad_packed_sequence(packed_out, batch_first=True)

      # Mask padding before max-pool
      max_len = gru_out.size(1)
      mask = (
          torch.arange(max_len, device=gru_out.device)[None, :]
          < lengths[:, None]
      )
      gru_out = gru_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))

      pooled = torch.max(gru_out, dim=1).values   # (B, gru_hidden*2)
      pooled = self.norm(pooled)
      return self.fc(pooled)
