from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from src.data.create_balanced_subset import SPLITS, choose_capped_natural_rows
from src.models.mert_finetune_classifier import configure_mert_trainability
from src.training.datamodule import CachedEmbeddingDataModule


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Linear(2, 2)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])


@pytest.mark.parametrize("mode,expected", [("frozen", 0), ("last_1", 2), ("last_2", 4)])
def test_partial_unfreezing(mode: str, expected: int) -> None:
    model = FakeBackbone()
    configure_mert_trainability(model, mode)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable == expected * 3


def test_full_unfreezing_requires_explicit_permission() -> None:
    with pytest.raises(ValueError):
        configure_mert_trainability(FakeBackbone(), "full")
    model = FakeBackbone()
    configure_mert_trainability(model, "full", allow_full=True)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_capped_natural_is_deterministic_and_track_disjoint() -> None:
    rows = []
    for split_index, split in enumerate(SPLITS):
        for label_index, label in enumerate(("bass", "guitar")):
            for index in range(8 + label_index * 3):
                rows.append({"segment_id": f"{split}_{label}_{index}",
                             "track_id": f"{split}_track_{index % 3}",
                             "coarse_label": label, "split": split})
    active = pd.DataFrame(rows)
    first = choose_capped_natural_rows(active, ["bass", "guitar"], minimum_per_class=3,
                                       total_budget=24, val_ratio=0.25, test_ratio=0.25, seed=42)
    second = choose_capped_natural_rows(active, ["bass", "guitar"], minimum_per_class=3,
                                        total_budget=24, val_ratio=0.25, test_ratio=0.25, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) <= 24
    track_sets = {split: set(first.loc[first["split"] == split, "track_id"]) for split in SPLITS}
    assert track_sets["train"].isdisjoint(track_sets["val"])
    assert track_sets["train"].isdisjoint(track_sets["test"])


def test_weighted_sampler_is_train_only(tmp_path: Path) -> None:
    for split in SPLITS:
        labels = torch.tensor([0, 0, 0, 1])
        torch.save({"embeddings": torch.randn(4, 3), "labels": labels,
                    "segment_ids": [f"{split}{i}" for i in range(4)],
                    "track_ids": [f"t{i}" for i in range(4)],
                    "audio_paths": [f"a{i}.wav" for i in range(4)],
                    "label_names": ["bass", "bass", "bass", "guitar"],
                    "subset_fingerprint": "same"}, tmp_path / f"{split}.pt")
    module = CachedEmbeddingDataModule(tmp_path, sampler="weighted")
    module.setup()
    assert module.train_dataloader().sampler.__class__.__name__ == "WeightedRandomSampler"
    assert module.val_dataloader().sampler.__class__.__name__ != "WeightedRandomSampler"
