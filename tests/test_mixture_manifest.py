from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data.create_mixture_manifest import build_mixture_manifest


def _subset(path: Path) -> None:
    rows = []
    for split in ("train", "val", "test"):
        for label_index, label in enumerate(("bass", "guitar", "vocals")):
            for item in range(2):
                rows.append({
                    "segment_id": f"{split}_{label}_{item}",
                    "track_id": f"{split}_track_{item}",
                    "stem_id": f"S{label_index}",
                    "audio_path": f"Audio/{split}_track_{item}/{label}.wav",
                    "start_seconds": float(item * 5),
                    "duration_seconds": 5.0,
                    "label_name": label,
                    "label_id": label_index,
                    "split": split,
                    "genre": "Rock",
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_synthetic_mixtures_are_deterministic_and_split_safe(tmp_path: Path) -> None:
    subset = tmp_path / "subset.csv"
    _subset(subset)
    config = {
        "mixture_dataset_id": "debug_synthetic_k",
        "mode": "synthetic_k",
        "subset_csv": subset,
        "k_values": [2],
        "mixtures_per_split_per_k": 3,
        "seed": 123,
    }
    first, labels = build_mixture_manifest(config)
    second, _ = build_mixture_manifest(config)
    pd.testing.assert_frame_equal(first, second)
    assert labels == {"bass": 0, "guitar": 1, "vocals": 2}
    assert set(first["k_active"]) == {2}
    for row in first.itertuples(index=False):
        source_splits = {part.split("_", 1)[0] for part in row.source_segment_ids.split("|")}
        assert source_splits == {row.split}


def test_same_song_same_time_manifest_groups_active_labels(tmp_path: Path) -> None:
    subset = tmp_path / "subset.csv"
    frame = pd.DataFrame([
        {
            "segment_id": f"train_{label}",
            "track_id": "track_a",
            "stem_id": f"S{index}",
            "audio_path": f"Audio/track_a/{label}.wav",
            "start_seconds": 0.0,
            "duration_seconds": 5.0,
            "label_name": label,
            "label_id": index,
            "split": "train",
        }
        for index, label in enumerate(("bass", "guitar"))
    ])
    frame.to_csv(subset, index=False)
    manifest, _ = build_mixture_manifest({
        "mixture_dataset_id": "same_time",
        "mode": "same_song_same_time",
        "subset_csv": subset,
        "seed": 1,
    })
    assert len(manifest) == 1
    assert manifest.iloc[0]["active_labels"] == "bass|guitar"
    assert int(manifest.iloc[0]["k_active"]) == 2


def test_stale_stem_index_fails_clearly(tmp_path: Path) -> None:
    subset = tmp_path / "subset.csv"
    _subset(subset)
    stale = tmp_path / "stem_index.csv"
    pd.DataFrame([{"track_id": "track_a", "stem_id": "S01"}]).to_csv(stale, index=False)
    with pytest.raises(ValueError, match="stale"):
        build_mixture_manifest({
            "mixture_dataset_id": "same_time",
            "mode": "same_song_reconstructed",
            "subset_csv": subset,
            "stem_index_csv": stale,
        })


def test_same_song_uses_broader_active_stem_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subset = tmp_path / "subset.csv"
    pd.DataFrame([
        {
            "segment_id": "train_bass_0",
            "track_id": "track_a",
            "stem_id": "S01",
            "audio_path": "Audio/track_a/bass.wav",
            "start_seconds": 0.0,
            "duration_seconds": 5.0,
            "label_name": "bass",
            "label_id": 0,
            "split": "train",
        },
        {
            "segment_id": "train_guitar_1",
            "track_id": "track_a",
            "stem_id": "S02",
            "audio_path": "Audio/track_a/guitar.wav",
            "start_seconds": 5.0,
            "duration_seconds": 5.0,
            "label_name": "guitar",
            "label_id": 1,
            "split": "train",
        },
    ]).to_csv(subset, index=False)
    stem_index = tmp_path / "stem_index.csv"
    pd.DataFrame([
        {
            "track_id": "track_a",
            "stem_id": "S01",
            "audio_path": "Audio/track_a/bass.wav",
            "raw_instrument_label": "bass",
            "coarse_label": "bass",
            "medleydb_instrument_label": "bass",
            "duration_seconds": 10.0,
            "valid": True,
            "has_bleed": False,
        },
        {
            "track_id": "track_a",
            "stem_id": "S02",
            "audio_path": "Audio/track_a/guitar.wav",
            "raw_instrument_label": "guitar",
            "coarse_label": "guitar",
            "medleydb_instrument_label": "guitar",
            "duration_seconds": 10.0,
            "valid": True,
            "has_bleed": False,
        },
    ]).to_csv(stem_index, index=False)
    monkeypatch.setattr(
        "src.data.create_mixture_manifest.load_audio_segment",
        lambda *args, **kwargs: (torch.ones(100), 100),
    )
    manifest, _ = build_mixture_manifest({
        "mixture_dataset_id": "same_time",
        "mode": "same_song_reconstructed",
        "subset_csv": subset,
        "stem_index_csv": stem_index,
        "medleydb_root": tmp_path,
        "label_granularity": "medleydb_instrument",
        "activity_threshold_dbfs": -60.0,
        "seed": 1,
    })
    assert not manifest.empty
    row = manifest.iloc[0]
    assert row["mode"] == "same_song_reconstructed"
    assert row["active_labels"] == "bass|guitar"
    assert row["activity_rule"] == "stem_rms_dbfs"


def test_original_full_mix_skips_missing_mix_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subset = tmp_path / "subset.csv"
    _subset(subset)
    stem_index = tmp_path / "stem_index.csv"
    pd.DataFrame([
        {
            "track_id": "train_track_0",
            "stem_id": "S0",
            "audio_path": "Audio/train_track_0/bass.wav",
            "raw_instrument_label": "bass",
            "coarse_label": "bass",
            "medleydb_instrument_label": "bass",
            "duration_seconds": 20.0,
            "valid": True,
            "has_bleed": False,
        }
    ]).to_csv(stem_index, index=False)
    monkeypatch.setattr(
        "src.data.create_mixture_manifest.load_audio_segment",
        lambda *args, **kwargs: (torch.ones(100), 100),
    )
    with pytest.warns(RuntimeWarning, match="full mix file was not found"):
        manifest, _ = build_mixture_manifest({
            "mixture_dataset_id": "full_mix",
            "mode": "original_full_mix",
            "subset_csv": subset,
            "stem_index_csv": stem_index,
            "medleydb_root": tmp_path,
            "label_granularity": "medleydb_instrument",
            "activity_threshold_dbfs": -60.0,
            "seed": 1,
        })
    assert manifest.empty
    assert "full_mix_exists" in manifest.columns
