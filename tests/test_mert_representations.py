from types import SimpleNamespace
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.features import extract_mert_embeddings
from src.features.extract_mert_embeddings import (
    SPLITS,
    _load_torch_cache,
    _pool,
    _selected_representation,
    dataframe_fingerprint,
)
from src.models.embedding_classifier import EmbeddingClassifier, WeightedLayerAggregation


def _outputs() -> SimpleNamespace:
    states = tuple(torch.full((2, 3, 4), float(index)) for index in range(5))
    return SimpleNamespace(hidden_states=states, last_hidden_state=states[-1])


def test_layer_selection_and_last_k_average() -> None:
    selected, indices = _selected_representation(_outputs(), "2")
    assert indices == [2]
    assert torch.all(selected == 2)
    averaged, indices = _selected_representation(_outputs(), "last3avg")
    assert indices == [2, 3, 4]
    assert torch.all(averaged == 3)
    stacked, indices = _selected_representation(_outputs(), "all")
    assert stacked.shape == (2, 5, 3, 4)
    assert indices == [0, 1, 2, 3, 4]


def test_pooling_shapes() -> None:
    hidden = torch.arange(24.0).reshape(2, 3, 4)
    assert _pool(hidden, "mean").shape == (2, 4)
    assert _pool(hidden, "max").shape == (2, 4)
    assert _pool(hidden, "meanmax").shape == (2, 8)
    layered = hidden.unsqueeze(1).repeat(1, 5, 1, 1)
    assert _pool(layered, "mean").shape == (2, 5, 4)


def test_learned_layer_weights_are_normalized() -> None:
    aggregator = WeightedLayerAggregation(5)
    assert torch.isclose(aggregator.weights.sum(), torch.tensor(1.0))
    model = EmbeddingClassifier(4, 3, num_layers=5,
                                layer_aggregation="learned_softmax")
    assert model(torch.randn(2, 5, 4)).shape == (2, 3)


def _segments() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        rows.append(
            {
                "segment_id": f"{split}_s0",
                "track_id": f"{split}_track",
                "audio_path": f"{split}.wav",
                "start_seconds": 0.0,
                "duration_seconds": 5.0,
                "label_name": "bass",
                "coarse_label": "bass",
                "label_granularity": "coarse_family",
                "subset_profile": "debug",
                "label_id": 0,
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def test_old_cache_fields_are_inferred(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    torch.save(
        {
            "embeddings": torch.randn(2, 3),
            "labels": torch.tensor([0, 1]),
            "segment_ids": ["a", "b"],
            "track_ids": ["ta", "tb"],
            "audio_paths": ["a.wav", "b.wav"],
            "label_names": ["bass", "guitar"],
        },
        path,
    )
    cache = _load_torch_cache(path)
    assert cache["representation_shape"] == [3]
    assert cache["hidden_state_indices"] == []
    assert cache["cache_schema_version"] == 1


def test_cache_hit_exits_without_loading_mert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segments = _segments()
    segment_path = tmp_path / "segments.csv"
    labels_path = tmp_path / "label_to_id.json"
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    segments.to_csv(segment_path, index=False)
    labels_path.write_text(json.dumps({"bass": 0}), encoding="utf-8")
    subset_fingerprint = dataframe_fingerprint(segments)
    for split in SPLITS:
        split_frame = segments[segments["split"] == split]
        torch.save(
            {
                "embeddings": torch.randn(1, 4),
                "labels": torch.tensor([0]),
                "segment_ids": split_frame["segment_id"].tolist(),
                "track_ids": split_frame["track_id"].tolist(),
                "audio_paths": split_frame["audio_path"].tolist(),
                "label_names": ["bass"],
                "model_name": "fake/mert",
                "model_revision": "main",
                "layer": "last",
                "pooling": "mean",
                "subset_fingerprint": subset_fingerprint,
                "subset_profile": "debug",
                "label_granularity": "coarse_family",
                "segment_seconds": 5.0,
                "split_fingerprint": dataframe_fingerprint(split_frame),
                "split": split,
            },
            out_dir / f"{split}.pt",
        )

    def fail_load(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("MERT should not be loaded for a full cache hit")

    monkeypatch.setattr(extract_mert_embeddings, "_load_mert", fail_load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "extract",
            "--segments", str(segment_path),
            "--label-to-id", str(labels_path),
            "--medleydb-root", str(tmp_path),
            "--out-dir", str(out_dir),
            "--model-name", "fake/mert",
        ],
    )
    extract_mert_embeddings.main()
    config = json.loads((out_dir / "embedding_config.json").read_text(encoding="utf-8"))
    assert config["cache_schema_version"] == 3
    assert config["representation_shape"] == [4]


def test_fingerprint_changes_with_profile_and_granularity() -> None:
    base = _segments()
    changed_profile = base.copy()
    changed_profile["subset_profile"] = "largest_balanced"
    changed_granularity = base.copy()
    changed_granularity["label_granularity"] = "medleydb_instrument"
    assert dataframe_fingerprint(base) != dataframe_fingerprint(changed_profile)
    assert dataframe_fingerprint(base) != dataframe_fingerprint(changed_granularity)
