"""Synthetic tests for balancing and track-level leakage prevention."""

from __future__ import annotations

import pandas as pd

from src.data.create_balanced_subset import (
    SPLITS,
    apply_dataset_config,
    allocate_split_counts,
    assign_track_splits,
    choose_balanced_rows,
    enumerate_segment_candidates,
    largest_feasible_total,
    parse_args,
    rank_eligible_classes,
)


def _candidate_frame() -> pd.DataFrame:
    rows = []
    labels = ["bass", "guitar", "vocals"]
    for track_number in range(15):
        track_id = f"track_{track_number:02d}"
        for label in labels:
            for segment_number in range(5):
                rows.append(
                    {
                        "segment_id": f"{track_id}_{label}_{segment_number}",
                        "track_id": track_id,
                        "coarse_label": label,
                        "audio_path": f"Audio/{track_id}/{label}.wav",
                        "start_seconds": float(segment_number * 5),
                        "duration_seconds": 5.0,
                        "raw_instrument_label": label,
                        "rms_dbfs": -20.0,
                    }
                )
    return pd.DataFrame(rows)


def test_class_ranking_excludes_insufficient_track_coverage() -> None:
    candidates = _candidate_frame()
    rare = pd.DataFrame(
        [
            {"segment_id": "rare_a", "track_id": "rare_1", "coarse_label": "rare"},
            {"segment_id": "rare_b", "track_id": "rare_2", "coarse_label": "rare"},
        ]
    )
    ranking = rank_eligible_classes(
        pd.concat([candidates, rare], ignore_index=True), min_segments_per_class=10
    )
    rare_row = ranking[ranking["coarse_label"] == "rare"].iloc[0]
    assert not bool(rare_row["eligible"])
    assert "fewer than 3" in rare_row["reason"]


def test_balancing_is_deterministic_and_has_no_track_leakage() -> None:
    candidates = _candidate_frame()
    labels = ["bass", "guitar", "vocals"]
    mapping_a, _ = assign_track_splits(
        candidates,
        labels,
        val_ratio=0.2,
        test_ratio=0.2,
        target_per_class=15,
        seed=42,
    )
    mapping_b, _ = assign_track_splits(
        candidates,
        labels,
        val_ratio=0.2,
        test_ratio=0.2,
        target_per_class=15,
        seed=42,
    )
    assert mapping_a == mapping_b

    active = candidates.copy()
    active["split"] = active["track_id"].map(mapping_a)
    total = largest_feasible_total(
        active,
        labels,
        minimum=9,
        maximum=15,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    assert total == 15
    selected_a = choose_balanced_rows(
        active,
        labels,
        total_per_class=total,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )
    selected_b = choose_balanced_rows(
        active,
        labels,
        total_per_class=total,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )
    pd.testing.assert_frame_equal(selected_a, selected_b)

    assert selected_a.groupby("coarse_label").size().to_dict() == {
        label: 15 for label in labels
    }
    quotas = allocate_split_counts(15, 0.2, 0.2)
    for label in labels:
        counts = selected_a[selected_a["coarse_label"] == label]["split"].value_counts()
        assert {split: int(counts.get(split, 0)) for split in SPLITS} == quotas

    track_sets = {
        split: set(selected_a.loc[selected_a["split"] == split, "track_id"])
        for split in SPLITS
    }
    assert track_sets["train"].isdisjoint(track_sets["val"])
    assert track_sets["train"].isdisjoint(track_sets["test"])
    assert track_sets["val"].isdisjoint(track_sets["test"])


def test_no_feasible_total_when_a_split_lacks_a_class() -> None:
    active = _candidate_frame()
    active["split"] = "train"
    assert (
        largest_feasible_total(
            active,
            ["bass", "guitar"],
            minimum=6,
            maximum=12,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        is None
    )


def test_candidate_enumeration_supports_label_granularities() -> None:
    index = pd.DataFrame(
        [
            {
                "track_id": "track_01",
                "stem_id": "S01",
                "audio_path": "Audio/track_01/stem.wav",
                "raw_instrument_label": "clean electric guitar",
                "coarse_label": "guitar",
                "medleydb_instrument_label": "clean_electric_guitar",
                "duration_seconds": 10.0,
                "valid": True,
                "has_bleed": False,
            }
        ]
    )
    coarse, _ = enumerate_segment_candidates(
        index, segment_seconds=5, hop_seconds=5, allow_bleed=False,
        label_granularity="coarse_family",
        subset_profile="debug",
    )
    instrument, _ = enumerate_segment_candidates(
        index, segment_seconds=5, hop_seconds=5, allow_bleed=False,
        label_granularity="medleydb_instrument",
        subset_profile="largest_balanced",
    )
    assert coarse["label_name"].unique().tolist() == ["guitar"]
    assert instrument["label_name"].unique().tolist() == ["clean_electric_guitar"]
    assert instrument["coarse_family_label"].unique().tolist() == ["guitar"]
    assert coarse["subset_profile"].unique().tolist() == ["debug"]
    assert instrument["subset_profile"].unique().tolist() == ["largest_balanced"]


def test_dataset_config_applies_capped_natural_total_budget(tmp_path, monkeypatch) -> None:
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "sampling_strategy: capped_natural\n"
        "total_budget: 123\n"
        "index: data/metadata/stem_index.csv\n"
        "medleydb_root: MedleyDB\n"
        "out: data/metadata/out.csv\n"
        "report_dir: data/reports\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["create_balanced_subset", "--config", str(config)])
    args = apply_dataset_config(parse_args())
    assert args.sampling_strategy == "capped_natural"
    assert args.total_budget == 123
