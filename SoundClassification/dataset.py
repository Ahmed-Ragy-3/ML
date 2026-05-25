import pandas as pd
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
import soundfile as sf


class UrbanSoundFeatureDataset(Dataset):
    def __init__(
        self,
        folds=(1, 2, 3, 4, 5, 6),
        root_dir="UrbanSound8K",
        processed_dir="processed",
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ):
        self.root_dir = Path(root_dir)
        self.audio_dir = self.root_dir / "audio"
        self.csv_path = self.root_dir / "metadata" / "UrbanSound8K.csv"
        self.processed_dir = self.root_dir / processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        full_df = pd.read_csv(self.csv_path)
        self.df = full_df[full_df["fold"].isin(folds)].reset_index(drop=True)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.resampler_cache = {}

        self.samples = []

    def _processed_path(self, row):
        return self.processed_dir / f"fold{int(row['fold'])}_{Path(row['slice_file_name']).stem}.pt"

    def extract_features(self, audio_path):
        data, sr = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self.resampler_cache[sr](waveform)

        features = self.mel_transform(waveform)
        features = torch.log(features + 1e-9)
        features = features.squeeze(0).transpose(0, 1)  # (time, mel)

        return (features - features.mean()) / (features.std() + 1e-9)

    def preprocess_all(self):
        print(f"Starting preprocessing for folds: {sorted(self.df['fold'].unique().tolist())}...")

        failed = []
        for _, row in self.df.iterrows():
            output_file = self._processed_path(row)

            if output_file.exists():
                continue

            try:
                audio_path = self.audio_dir / f"fold{int(row['fold'])}" / row["slice_file_name"]
                features = self.extract_features(audio_path)
                torch.save(
                    {
                        "features": features,
                        "label": int(row["classID"])
                    },
                    output_file
                )
            except Exception as e:
                failed.append((row["slice_file_name"], str(e)))
                print(f"Failed on {row['slice_file_name']}: {e}")

        self.rebuild_index()

        print(f"Preprocessing completed. Ready samples: {len(self.samples)}")
        if failed:
            print(f"Total failed files: {len(failed)}")

    def rebuild_index(self):
        self.samples = []
        for _, row in self.df.iterrows():
            file_path = self._processed_path(row)
            if file_path.exists():
                self.samples.append((file_path, int(row["classID"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        item = torch.load(file_path)
        return item["features"], label

    @staticmethod
    def collate_batches(batch):
        features = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        lengths = torch.tensor([x.size(0) for x in features], dtype=torch.long)

        padded_features = torch.nn.utils.rnn.pad_sequence(
            features,
            batch_first=True,
            padding_value=0.0
        )

        return padded_features, lengths, labels