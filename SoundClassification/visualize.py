"""
visualize.py – Waveform and spectrogram visualisations for UrbanSound8K.

Generates, for ONE sample of EACH class (10 total):
    1. Waveform plot
    2. Mel-Spectrogram (log scale)
    3. MFCC spectrogram

All 10 × 3 panels are laid out in a single grid figure per plot type,
and also saved as a combined figure.

Usage
-----
    python visualize.py
        (expects the UrbanSound8K directory to be present)
"""

import pandas as pd
import numpy as np
import soundfile as sf
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Constants – must match dataset.py
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22_050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 150
N_MFCC = 14

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling",        "engine_idling", "gun_shot", "jackhammer",
    "siren",           "street_music",
]

ROOT_DIR = Path("UrbanSound8K")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: pick one representative file per class from all 10 folds
# ──────────────────────────────────────────────────────────────────────────────

def pick_class_samples(csv_path: Path) -> dict:
   """Return {classID: (audio_path, class_name)} for one sample per class."""
   df = pd.read_csv(csv_path)
   samples = {}
   for class_id in range(10):
      subset = df[df["classID"] == class_id]
      if subset.empty:
         continue
      row = subset.iloc[0]
      path = (
          ROOT_DIR / "audio"
          / f"fold{int(row['fold'])}"
          / row["slice_file_name"]
      )
      samples[class_id] = (path, row["class"])
   return samples


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load & resample to SAMPLE_RATE
# ──────────────────────────────────────────────────────────────────────────────

def load_waveform(path: Path):
   """Return (waveform_numpy_1d, sample_rate)."""
   data, sr = sf.read(str(path))
   waveform = torch.from_numpy(data).float()
   if waveform.ndim == 1:
      waveform = waveform.unsqueeze(0)
   else:
      waveform = waveform.transpose(0, 1)
   if waveform.shape[0] > 1:
      waveform = waveform.mean(dim=0, keepdim=True)
   if sr != SAMPLE_RATE:
      resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
      waveform = resampler(waveform)
   return waveform.squeeze(0).numpy(), SAMPLE_RATE   # (N,), int


# ──────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ──────────────────────────────────────────────────────────────────────────────

def _plot_waveform_ax(ax, signal, sr, title):
   t = np.linspace(0, len(signal) / sr, len(signal))
   ax.plot(t, signal, linewidth=0.6, color="steelblue")
   ax.set_title(title, fontsize=8, pad=3)
   ax.set_xlabel("Time (s)", fontsize=7)
   ax.set_ylabel("Amplitude", fontsize=7)
   ax.tick_params(labelsize=6)
   ax.set_xlim(0, t[-1])
   ax.grid(True, alpha=0.25)


def _plot_mel_ax(ax, signal, sr, title):
   waveform_t = torch.from_numpy(signal).unsqueeze(0)
   mel_t = torchaudio.transforms.MelSpectrogram(
       sample_rate=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
   )(waveform_t)
   log_mel = torch.log(mel_t + 1e-9).squeeze(0).numpy()

   im = ax.imshow(
       log_mel, aspect="auto", origin="lower", cmap="magma",
       extent=[0, log_mel.shape[1], 0, N_MELS]
   )
   ax.set_title(title, fontsize=8, pad=3)
   ax.set_xlabel("Frame", fontsize=7)
   ax.set_ylabel("Mel Bin", fontsize=7)
   ax.tick_params(labelsize=6)
   plt.colorbar(im, ax=ax, pad=0.02,
                fraction=0.046).ax.tick_params(labelsize=5)


def _plot_mfcc_ax(ax, signal, sr, title):
   waveform_t = torch.from_numpy(signal).unsqueeze(0)
   mfcc_t = torchaudio.transforms.MFCC(
       sample_rate=sr, n_mfcc=N_MFCC,
       melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS}
   )(waveform_t)
   mfcc = mfcc_t.squeeze(0).numpy()

   im = ax.imshow(
       mfcc, aspect="auto", origin="lower", cmap="coolwarm",
       extent=[0, mfcc.shape[1], 0, N_MFCC]
   )
   ax.set_title(title, fontsize=8, pad=3)
   ax.set_xlabel("Frame", fontsize=7)
   ax.set_ylabel("MFCC Coeff", fontsize=7)
   ax.tick_params(labelsize=6)
   plt.colorbar(im, ax=ax, pad=0.02,
                fraction=0.046).ax.tick_params(labelsize=5)


# ──────────────────────────────────────────────────────────────────────────────
# Main grid figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_waveforms(samples: dict, save_path="waveforms_all_classes.png"):
   """10 waveforms in a 2×5 grid."""
   n_cols, n_rows = 5, 2
   fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6))
   fig.suptitle("Waveforms – One Sample per Class", fontsize=13, y=1.01)

   for class_id in range(10):
      ax = axes[class_id // n_cols][class_id % n_cols]
      path, name = samples[class_id]
      try:
         signal, sr = load_waveform(path)
         _plot_waveform_ax(ax, signal, sr, name.replace("_", " ").title())
      except Exception as exc:
         ax.set_title(f"{name}\n(load error)", fontsize=7)
         print(f"  WARN: could not load {path}: {exc}")

   plt.tight_layout()
   plt.savefig(save_path, dpi=150, bbox_inches="tight")
   plt.close()
   print(f"Waveform grid saved to '{save_path}'")


def plot_mel_spectrograms(samples: dict, save_path="mel_spectrograms_all_classes.png"):
   """10 Mel spectrograms in a 2×5 grid."""
   n_cols, n_rows = 5, 2
   fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7))
   fig.suptitle(
       "Log-Mel Spectrograms – One Sample per Class (150 Mels, 22 050 Hz)", fontsize=13, y=1.01)

   for class_id in range(10):
      ax = axes[class_id // n_cols][class_id % n_cols]
      path, name = samples[class_id]
      try:
         signal, sr = load_waveform(path)
         _plot_mel_ax(ax, signal, sr, name.replace("_", " ").title())
      except Exception as exc:
         ax.set_title(f"{name}\n(load error)", fontsize=7)
         print(f"  WARN: could not load {path}: {exc}")

   plt.tight_layout()
   plt.savefig(save_path, dpi=150, bbox_inches="tight")
   plt.close()
   print(f"Mel spectrogram grid saved to '{save_path}'")


def plot_mfcc_spectrograms(samples: dict, save_path="mfcc_spectrograms_all_classes.png"):
   """10 MFCC spectrograms in a 2×5 grid."""
   n_cols, n_rows = 5, 2
   fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7))
   fig.suptitle(
       "MFCC Spectrograms – One Sample per Class (40 Coeffs, 22 050 Hz)", fontsize=13, y=1.01)

   for class_id in range(10):
      ax = axes[class_id // n_cols][class_id % n_cols]
      path, name = samples[class_id]
      try:
         signal, sr = load_waveform(path)
         _plot_mfcc_ax(ax, signal, sr, name.replace("_", " ").title())
      except Exception as exc:
         ax.set_title(f"{name}\n(load error)", fontsize=7)
         print(f"  WARN: could not load {path}: {exc}")

   plt.tight_layout()
   plt.savefig(save_path, dpi=150, bbox_inches="tight")
   plt.close()
   print(f"MFCC spectrogram grid saved to '{save_path}'")


def plot_combined(samples: dict, save_path="combined_visualisations.png"):
   """
   Combined figure: for each class, one row with 3 panels
   (waveform | mel spectrogram | MFCC spectrogram).
   Total: 10 rows × 3 columns.
   """
   n_classes = 10
   fig = plt.figure(figsize=(18, n_classes * 2.8))
   gs = gridspec.GridSpec(n_classes, 3, hspace=0.55, wspace=0.4)
   fig.suptitle(
       "UrbanSound8K – Waveform, Mel Spectrogram & MFCC per Class",
       fontsize=14, y=1.005
   )

   for class_id in range(n_classes):
      path, name = samples[class_id]
      display = name.replace("_", " ").title()
      try:
         signal, sr = load_waveform(path)
      except Exception as exc:
         print(f"  WARN skipping class {class_id} ({name}): {exc}")
         continue

      ax_wav = fig.add_subplot(gs[class_id, 0])
      ax_mel = fig.add_subplot(gs[class_id, 1])
      ax_mfc = fig.add_subplot(gs[class_id, 2])

      _plot_waveform_ax(ax_wav, signal, sr, f"{display}\n(Waveform)")
      _plot_mel_ax(ax_mel, signal, sr, f"{display}\n(Mel Spectrogram)")
      _plot_mfcc_ax(ax_mfc, signal, sr, f"{display}\n(MFCC)")

   plt.savefig(save_path, dpi=120, bbox_inches="tight")
   plt.close()
   print(f"Combined visualisation saved to '{save_path}'")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_visualisations(output_dir: str = "."):
   output_dir = Path(output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)

   csv_path = ROOT_DIR / "metadata" / "UrbanSound8K.csv"
   print("Picking one sample per class from all folds…")
   samples = pick_class_samples(csv_path)

   plot_waveforms(samples, str(output_dir / "waveforms_all_classes.png"))
   plot_mel_spectrograms(samples, str(
       output_dir / "mel_spectrograms_all_classes.png"))
   plot_mfcc_spectrograms(samples, str(
       output_dir / "mfcc_spectrograms_all_classes.png"))
   plot_combined(samples, str(output_dir / "combined_visualisations.png"))
   print("All visualisations complete.")


if __name__ == "__main__":
   run_visualisations()
