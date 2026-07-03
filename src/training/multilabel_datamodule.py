"""Lightning data module for polyphonic multi-label embedding caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_multilabel_embedding_cache(path: Path) -> dict[str, Any]:
    """Load and validate one split cache for multi-label classification."""

    if not path.is_file():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"Expected a dictionary cache in {path}")
    embeddings = cache.get("embeddings")
    labels = cache.get("labels")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim not in {2, 3}:
        raise ValueError(f"Cache embeddings must have shape [N,D] or [N,L,D]: {path}")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError(f"Cache labels must have shape [N,C]: {path}")
    if len(embeddings) != len(labels):
        raise ValueError(f"Embedding/label length mismatch in {path}")
    cache.setdefault("representation_shape", list(embeddings.shape[1:]))
    cache.setdefault("embedding_dim", int(embeddings.shape[-1]))
    cache.setdefault("hidden_state_indices", [])
    cache.setdefault("cache_schema_version", 1)
    required = {
        "mixture_ids", "modes", "track_ids", "source_track_ids", "audio_paths",
        "source_audio_paths", "source_start_seconds", "start_seconds",
        "duration_seconds", "active_labels", "k_active", "label_names",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Cache {path} is missing keys: {', '.join(missing)}")
    for key in required - {"label_names"}:
        if len(cache[key]) != len(labels):
            raise ValueError(f"Cache metadata length mismatch for {key}: {path}")
    if len(cache["label_names"]) != labels.shape[1]:
        raise ValueError(f"Cache label_names length does not match target width: {path}")
    return cache


class MultilabelCachedEmbeddingDataModule(L.LightningDataModule):
    """Serve train/validation/test tensors from immutable multi-label caches."""

    def __init__(self, cache_dir: Path, batch_size: int = 32, num_workers: int = 0) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.caches: dict[str, dict[str, Any]] = {}
        self.datasets: dict[str, TensorDataset] = {}
        self.input_dim = 0
        self.num_labels = 0
        self.num_layers: int | None = None
        self.pos_weight = torch.empty(0)

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.caches = {
            split: load_multilabel_embedding_cache(self.cache_dir / f"{split}.pt")
            for split in ("train", "val", "test")
        }
        shapes = {tuple(cache["embeddings"].shape[1:]) for cache in self.caches.values()}
        label_widths = {int(cache["labels"].shape[1]) for cache in self.caches.values()}
        fingerprints = {cache.get("manifest_fingerprint") for cache in self.caches.values()}
        if len(shapes) != 1:
            raise ValueError(f"Cache embedding shapes disagree: {sorted(shapes)}")
        if len(label_widths) != 1:
            raise ValueError("Cache target widths disagree")
        if len(fingerprints) != 1:
            raise ValueError("Cache files were generated from different manifests")
        representation_shape = shapes.pop()
        self.input_dim = int(representation_shape[-1])
        self.num_layers = int(representation_shape[0]) if len(representation_shape) == 2 else None
        self.num_labels = label_widths.pop()
        train_labels = self.caches["train"]["labels"].float()
        positives = train_labels.sum(dim=0)
        negatives = train_labels.shape[0] - positives
        self.pos_weight = negatives / positives.clamp_min(1.0)
        for split, cache in self.caches.items():
            self.datasets[split] = TensorDataset(cache["embeddings"].float(), cache["labels"].float())

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)
