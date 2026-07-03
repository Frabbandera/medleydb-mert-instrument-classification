from pathlib import Path

import numpy as np
import torch

from src.training.evaluate_multilabel_classifier import _metrics
from src.training.multilabel_datamodule import load_multilabel_embedding_cache


def test_multilabel_cache_validation(tmp_path: Path) -> None:
    path = tmp_path / "train.pt"
    torch.save(
        {
            "embeddings": torch.randn(3, 4),
            "labels": torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32),
            "mixture_ids": ["m1", "m2", "m3"],
            "modes": ["synthetic_random", "synthetic_random", "synthetic_random"],
            "track_ids": ["t1", "t2", "t3"],
            "source_track_ids": ["t1", "t2", "t3"],
            "audio_paths": ["", "", ""],
            "source_audio_paths": ["a.wav", "b.wav", "c.wav"],
            "source_start_seconds": ["0", "0", "0"],
            "start_seconds": [0.0, 0.0, 0.0],
            "duration_seconds": [5.0, 5.0, 5.0],
            "active_labels": ["bass", "guitar", "bass|guitar"],
            "k_active": [1, 1, 2],
            "label_names": ["bass", "guitar"],
        },
        path,
    )
    cache = load_multilabel_embedding_cache(path)
    assert cache["embedding_dim"] == 4
    assert cache["labels"].shape == (3, 2)


def test_multilabel_metrics_handle_missing_positive_class() -> None:
    targets = np.array([[1, 0], [1, 0], [0, 0]])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.3]])
    metrics, per_class = _metrics(targets, probabilities, ["bass", "guitar"], 0.5)
    assert metrics["test_micro_f1"] == 1.0
    assert metrics["test_macro_f1"] == 0.5
    assert per_class.loc[per_class["label"] == "guitar", "roc_auc"].iloc[0] == "NA"
