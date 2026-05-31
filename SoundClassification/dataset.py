"""
dataset.py – UrbanSound8K feature dataset.

Supports three feature modes:
    'mel'    – Log-Mel Spectrogram  (150 mel bins)  → input_size = 150
    'energy' – Frame-level RMS energy (1 feature)   → input_size = 1
    'mfcc'   – MFCCs                 (40 coeffs)    → input_size = 40

All modes:
    • Sample rate : 22 050 Hz
    • Batch size  : 32
    • Variable-length sequences padded inside collate_batches()
"""

import pandas as pd
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
import soundfile as sf

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22_050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 150        # Mel bins
N_MFCC = 40         # MFCC coefficients
BATCH_SIZE = 32


class UrbanSoundFeatureDataset(Dataset):
   """
   Loads UrbanSound8K audio, extracts a chosen feature type and caches
   the result as a .pt file so that subsequent runs are fast.

   Parameters
   ----------
   folds        : which dataset folds to include
   root_dir     : top-level UrbanSound8K directory
   processed_dir: sub-directory for cached .pt features
   feature_type : 'mel' | 'energy' | 'mfcc'
   """

   FEATURE_TYPES = ("mel", "energy", "mfcc")

   def __init__(
       self,
       folds=(1, 2, 3, 4, 5, 6),
       root_dir="UrbanSound8K",
       processed_dir="processed",
       feature_type="mel",
   ):
      if feature_type not in self.FEATURE_TYPES:
         raise ValueError(f"feature_type must be one of {self.FEATURE_TYPES}")

      self.root_dir = Path(root_dir)
      self.audio_dir = self.root_dir / "audio"
      self.csv_path = self.root_dir / "metadata" / "UrbanSound8K.csv"
      self.feature_type = feature_type

      # Each feature type gets its own cache sub-directory to avoid mixing
      cache_subdir = f"{processed_dir}_{feature_type}"
      self.processed_dir = self.root_dir / cache_subdir
      self.processed_dir.mkdir(parents=True, exist_ok=True)

      full_df = pd.read_csv(self.csv_path)
      self.df = full_df[full_df["fold"].isin(folds)].reset_index(drop=True)

      # ── Transforms ────────────────────────────────────────────────────────
      self.mel_transform = torchaudio.transforms.MelSpectrogram(
          sample_rate=SAMPLE_RATE,
          n_fft=N_FFT,
          hop_length=HOP_LENGTH,
          n_mels=N_MELS,
      )
      self.mfcc_transform = torchaudio.transforms.MFCC(
          sample_rate=SAMPLE_RATE,
          n_mfcc=N_MFCC,
          melkwargs={
              "n_fft":       N_FFT,
              "hop_length":  HOP_LENGTH,
              "n_mels":      N_MELS,
          },
      )
      self.resampler_cache: dict = {}
      self.samples: list = []

   # ── Internal helpers ──────────────────────────────────────────────────────

   def _processed_path(self, row) -> Path:
      stem = Path(row["slice_file_name"]).stem
      return self.processed_dir / f"fold{int(row['fold'])}_{stem}.pt"

   def _load_waveform(self, audio_path) -> torch.Tensor:
      """Read audio, convert to mono, resample to SAMPLE_RATE."""
      data, sr = sf.read(str(audio_path))
      waveform = torch.from_numpy(data).float()

      # Ensure shape (channels, time)
      if waveform.ndim == 1:
         waveform = waveform.unsqueeze(0)
      else:
         waveform = waveform.transpose(0, 1)

      # Mix down to mono
      if waveform.shape[0] > 1:
         waveform = waveform.mean(dim=0, keepdim=True)

      # Resample if necessary
      if sr != SAMPLE_RATE:
         if sr not in self.resampler_cache:
            self.resampler_cache[sr] = torchaudio.transforms.Resample(
                sr, SAMPLE_RATE
            )
         waveform = self.resampler_cache[sr](waveform)

      return waveform  # shape: (1, time)

   def _normalize(self, x: torch.Tensor) -> torch.Tensor:
       """Zero-mean, unit-variance normalisation."""
       return (x - x.mean()) / (x.std(unbiased=False) + 1e-9)

   # ── Feature extraction ────────────────────────────────────────────────────

   def extract_features(self, audio_path) -> torch.Tensor:
      """
      Returns a 2-D tensor of shape (time_frames, feature_dim).
      """
      waveform = self._load_waveform(audio_path)

      if self.feature_type == "mel":
         spec = self.mel_transform(waveform)          # (1, n_mels, T)
         spec = torch.log(spec + 1e-9)
         feat = spec.squeeze(0).transpose(0, 1)       # (T, 150)

      elif self.feature_type == "energy":

          signal = waveform.squeeze(0)

          if signal.numel() < N_FFT:
              pad = N_FFT - signal.numel()

              signal = torch.nn.functional.pad(signal, (0, pad))

          frames = signal.unfold(0, N_FFT, HOP_LENGTH)

          rms = frames.pow(2).mean(dim=-1, keepdim=True).sqrt()

          feat = rms                              # (T, 1)

      elif self.feature_type == "mfcc":
         mfcc = self.mfcc_transform(waveform)         # (1, n_mfcc, T)
         feat = mfcc.squeeze(0).transpose(0, 1)       # (T, 40)

      else:
         raise ValueError(f"Unknown feature_type: {self.feature_type}")

      return self._normalize(feat)

   # ── Pre-processing ────────────────────────────────────────────────────────

   def preprocess_all(self):
      folds_str = sorted(self.df["fold"].unique().tolist())
      print(
          f"[{self.feature_type.upper()}] Preprocessing folds {folds_str} ..."
      )
      failed = []

      for _, row in self.df.iterrows():
         out_path = self._processed_path(row)
         if out_path.exists():
            continue
         try:
            audio_path = (
                self.audio_dir
                / f"fold{int(row['fold'])}"
                / row["slice_file_name"]
            )
            features = self.extract_features(audio_path)
            torch.save(
                {"features": features, "label": int(row["classID"])},
                out_path,
            )
         except Exception as exc:
            failed.append((row["slice_file_name"], str(exc)))
            print(f"  FAILED {row['slice_file_name']}: {exc}")

      self.rebuild_index()
      print(
          f"[{self.feature_type.upper()}] Done – {len(self.samples)} samples ready."
      )
      if failed:
         print(f"  Total failed: {len(failed)}")

   def rebuild_index(self):
      self.samples = []
      for _, row in self.df.iterrows():
         fp = self._processed_path(row)
         if fp.exists():
            self.samples.append((fp, int(row["classID"])))

   # ── Dataset interface ─────────────────────────────────────────────────────

   def __len__(self) -> int:
      return len(self.samples)

   def __getitem__(self, idx):
      file_path, label = self.samples[idx]
      item = torch.load(file_path, weights_only=True)
      return item["features"], label

   # ── Collation ─────────────────────────────────────────────────────────────

   @staticmethod
   def collate_batches(batch):
      """Pad variable-length sequences to the longest in the batch."""
      features = [item[0] for item in batch]
      labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
      lengths = torch.tensor([x.size(0) for x in features], dtype=torch.long)

      padded = torch.nn.utils.rnn.pad_sequence(
          features, batch_first=True, padding_value=0.0
      )
      return padded, lengths, labels
