"""Lightning data module for split-level MERT embedding caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def load_embedding_cache(path: Path) -> dict[str, Any]:
    """Load and validate a generated embedding cache."""

    if not path.is_file():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    embeddings = cache.get("embeddings")
    if isinstance(embeddings, torch.Tensor):
        cache.setdefault("representation_shape", list(embeddings.shape[1:]))
        cache.setdefault("embedding_dim", int(embeddings.shape[-1]))
    cache.setdefault("hidden_state_indices", [])
    cache.setdefault("cache_schema_version", 1)
    required = {"embeddings", "labels", "segment_ids", "track_ids", "label_names"}
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Cache {path} is missing keys: {', '.join(missing)}")
    embeddings, labels = cache["embeddings"], cache["labels"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim not in {2, 3}:
        raise ValueError(f"Cache embeddings must have shape [N,D] or [N,L,D]: {path}")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError(f"Cache labels must have shape [N]: {path}")
    if len(embeddings) != len(labels):
        raise ValueError(f"Embedding/label length mismatch in {path}")
    for key in ("segment_ids", "track_ids", "label_names"):
        if len(cache[key]) != len(labels):
            raise ValueError(f"Cache metadata length mismatch for {key}: {path}")
    return cache


class CachedEmbeddingDataModule(L.LightningDataModule):
    """Serve train/validation/test tensors from immutable cache files."""

    def __init__(
        self,
        cache_dir: Path,
        batch_size: int = 32,
        num_workers: int = 0,
        sampler: str = "shuffle",
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.sampler = str(sampler)
        if self.sampler not in {"shuffle", "weighted"}:
            raise ValueError("sampler must be 'shuffle' or 'weighted'")
        self.caches: dict[str, dict[str, Any]] = {}
        self.datasets: dict[str, TensorDataset] = {}
        self.input_dim = 0
        self.num_classes = 0
        self.num_layers: int | None = None
        self.class_weights = torch.empty(0)

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.caches = {
            split: load_embedding_cache(self.cache_dir / f"{split}.pt")
            for split in ("train", "val", "test")
        }
        shapes = {tuple(cache["embeddings"].shape[1:]) for cache in self.caches.values()}
        fingerprints = {cache.get("subset_fingerprint") for cache in self.caches.values()}
        if len(shapes) != 1:
            raise ValueError(f"Cache embedding shapes disagree: {sorted(shapes)}")
        if len(fingerprints) != 1:
            raise ValueError("Cache files were generated from different subsets")
        representation_shape = shapes.pop()
        self.input_dim = int(representation_shape[-1])
        self.num_layers = int(representation_shape[0]) if len(representation_shape) == 2 else None
        all_labels = torch.cat([cache["labels"].long() for cache in self.caches.values()])
        unique = sorted(int(value) for value in torch.unique(all_labels).tolist())
        if unique != list(range(len(unique))):
            raise ValueError(f"Class IDs must be contiguous from zero, got {unique}")
        self.num_classes = len(unique)
        if self.num_classes < 2:
            raise ValueError("At least two classes are required")
        train_labels = self.caches["train"]["labels"].long()
        counts = torch.bincount(train_labels, minlength=self.num_classes).float()
        self.class_weights = counts.sum() / (self.num_classes * counts.clamp_min(1.0))
        for split, cache in self.caches.items():
            self.datasets[split] = TensorDataset(
                cache["embeddings"].float(), cache["labels"].long()
            )

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        sampler = None
        if split == "train" and self.sampler == "weighted":
            labels = self.caches["train"]["labels"].long()
            sampler = WeightedRandomSampler(
                weights=self.class_weights[labels],
                num_samples=len(labels),
                replacement=True,
            )
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
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
