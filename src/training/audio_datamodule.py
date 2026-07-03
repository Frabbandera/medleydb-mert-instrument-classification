"""On-the-fly isolated-stem audio loading for MERT fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.audio_io import load_audio_segment
from src.utils.paths import resolve_data_path


class StemSegmentDataset(Dataset):
    """Load fixed-duration mono stem segments from a generated subset table."""

    def __init__(
        self,
        frame: pd.DataFrame,
        medleydb_root: Path,
        sample_rate: int,
        processor: Any | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.medleydb_root = Path(medleydb_root)
        self.sample_rate = int(sample_rate)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        waveform, _ = load_audio_segment(
            resolve_data_path(self.medleydb_root, row["audio_path"]),
            float(row["start_seconds"]),
            float(row["duration_seconds"]),
            target_sample_rate=self.sample_rate,
            normalize=True,
            pad=True,
        )
        if self.processor is not None:
            processed = self.processor(
                waveform.cpu().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=False,
            )
            waveform = processed["input_values"].squeeze(0).float()
        return waveform, torch.tensor(int(row["label_id"])), torch.tensor(index)


class AudioSegmentDataModule(L.LightningDataModule):
    """DataModule for direct-audio frozen and partially fine-tuned MERT runs."""

    def __init__(self, subset_csv: Path, medleydb_root: Path, *, sample_rate: int = 24000,
                 batch_size: int = 1, num_workers: int = 0, sampler: str = "shuffle",
                 processor: Any | None = None):
        super().__init__()
        self.subset_csv = Path(subset_csv)
        self.medleydb_root = Path(medleydb_root)
        self.sample_rate = int(sample_rate)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.sampler = str(sampler)
        self.processor = processor
        self.frames: dict[str, pd.DataFrame] = {}
        self.datasets: dict[str, StemSegmentDataset] = {}
        self.num_classes = 0
        self.class_weights = torch.empty(0)

    def setup(self, stage: str | None = None) -> None:
        del stage
        frame = pd.read_csv(self.subset_csv)
        required = {"segment_id", "track_id", "audio_path", "start_seconds",
                    "duration_seconds", "label_id", "split"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Subset is missing columns: {', '.join(missing)}")
        if "coarse_label" not in frame.columns and "label_name" not in frame.columns:
            raise ValueError("Subset must contain either coarse_label or label_name")
        labels = sorted(int(value) for value in frame["label_id"].unique())
        if labels != list(range(len(labels))):
            raise ValueError(f"Class IDs must be contiguous from zero, got {labels}")
        self.num_classes = len(labels)
        track_sets: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            split_frame = frame[frame["split"] == split].copy().reset_index(drop=True)
            if split_frame.empty:
                raise ValueError(f"Subset split is empty: {split}")
            self.frames[split] = split_frame
            self.datasets[split] = StemSegmentDataset(
                split_frame, self.medleydb_root, self.sample_rate, self.processor
            )
            track_sets[split] = set(split_frame["track_id"].astype(str))
        if any(track_sets[a] & track_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("Track leakage detected between subset splits")
        train_labels = torch.tensor(self.frames["train"]["label_id"].astype(int).tolist())
        counts = torch.bincount(train_labels, minlength=self.num_classes).float()
        self.class_weights = counts.sum() / (self.num_classes * counts.clamp_min(1.0))

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        sampler = None
        if split == "train" and self.sampler == "weighted":
            labels = torch.tensor(self.frames[split]["label_id"].astype(int).tolist())
            sampler = WeightedRandomSampler(self.class_weights[labels], len(labels), replacement=True)
        return DataLoader(self.datasets[split], batch_size=self.batch_size,
                          shuffle=shuffle and sampler is None, sampler=sampler,
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", False)
