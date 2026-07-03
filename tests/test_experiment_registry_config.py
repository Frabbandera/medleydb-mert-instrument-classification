from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.experiments.config import resolve_experiment_config, validate_experiment_id
from src.experiments.registry import REGISTRY_COLUMNS, append_experiment, read_registry


def test_registry_append_duplicate_and_replace(tmp_path: Path) -> None:
    path = tmp_path / "results" / "experiment_registry.csv"
    row = {column: "NA" for column in REGISTRY_COLUMNS}
    row.update({"experiment_id": "example", "test_macro_f1": 0.5})
    append_experiment(path, row)
    assert read_registry(path).iloc[0]["experiment_id"] == "example"
    with pytest.raises(FileExistsError):
        append_experiment(path, row)
    row["test_macro_f1"] = 0.7
    append_experiment(path, row, replace=True)
    frame = read_registry(path)
    assert len(frame) == 1
    assert float(frame.iloc[0]["test_macro_f1"]) == pytest.approx(0.7)


def test_config_resolution_uses_clean_defaults(tmp_path: Path) -> None:
    config = tmp_path / "minimal.yaml"
    config.write_text(
        "experiment_id: minimal_clean\nseed: 7\n"
        "data: {}\nmodel: {}\ntraining: {}\nevaluation: {}\n",
        encoding="utf-8",
    )
    resolved = resolve_experiment_config(config)
    assert resolved["experiment_id"] == "minimal_clean"
    assert resolved["approach"] == "frozen_embeddings"
    assert resolved["representation"] == {"layer": "last", "pooling": "mean"}
    assert resolved["data"]["label_granularity"] == "medleydb_instrument"
    assert resolved["data"]["subset_profile"] == "largest_balanced"
    assert resolved["data"]["subset_csv"].endswith(
        "subset_largest_balanced_medleydb_instrument.csv"
    )
    assert resolved["data"]["label_to_id"].endswith(
        "labels_largest_balanced_medleydb_instrument_label_to_id.json"
    )


def test_registry_contains_validation_and_label_protocol_columns() -> None:
    assert "best_val_macro_f1" in REGISTRY_COLUMNS
    assert "best_val_accuracy" in REGISTRY_COLUMNS
    assert "best_epoch" in REGISTRY_COLUMNS
    assert "subset_profile" in REGISTRY_COLUMNS
    assert "label_granularity" in REGISTRY_COLUMNS
    assert "class_names" in REGISTRY_COLUMNS


def test_final_isolated_configs_resolve() -> None:
    config_names = [
        "classical_largest_balanced_medleydb_mfcc_svm.yaml",
        "isolated_largest_balanced_medleydb_mert95_last_mean_linear.yaml",
        "isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml",
        "isolated_largest_balanced_medleydb_mert95_layer6_mean_mlp_h256_d02.yaml",
        "isolated_largest_balanced_medleydb_mert95_layer9_mean_mlp_h256_d02.yaml",
        "isolated_largest_balanced_medleydb_mert95_last3avg_mean_mlp_h256_d02.yaml",
        "isolated_largest_balanced_medleydb_mert95_last6avg_mean_mlp_h256_d02.yaml",
        "isolated_largest_balanced_medleydb_mert95_all_weighted_mean_mlp_h256_d02.yaml",
        "isolated_largest_balanced_medleydb_mert330_last_mean_mlp_h256_d02.yaml",
    ]
    for name in config_names:
        resolved = resolve_experiment_config(Path("configs/experiments") / name)
        assert resolved["data"]["subset_profile"] == "largest_balanced"
        assert resolved["data"]["label_granularity"] == "medleydb_instrument"
        assert resolved["notes"]
        if resolved["approach"] != "classical_baseline":
            assert resolved["training"]["max_epochs"] == 30
            assert resolved["training"]["weight_decay"] == pytest.approx(0.01)


def test_config_override_and_identifier_validation(tmp_path: Path) -> None:
    config = tmp_path / "base.yaml"
    config.write_text("data: {}\nmodel: {}\ntraining: {}\n", encoding="utf-8")
    resolved = resolve_experiment_config(config, experiment_id="new.run-1", smoke_test=True)
    assert resolved["output"]["results_dir"] == "results/new.run-1"
    assert resolved["training"]["max_epochs"] == 1
    with pytest.raises(ValueError):
        validate_experiment_id("../escape")


def _config_paths() -> list[Path]:
    root = Path("configs")
    return sorted(root.rglob("*.yaml"))


def test_clean_configs_have_unique_ids_notes_and_no_legacy_paths() -> None:
    forbidden = (
        "data/metadata/subset_segments.csv",
        "data/metadata/label_to_id.json",
        "data/cache/mert_v1_95m_subset",
    )
    experiment_ids: list[str] = []
    for path in _config_paths():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} still references {token}"
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "experiment_id" not in data:
            continue
        resolved = resolve_experiment_config(path)
        experiment_ids.append(resolved["experiment_id"])
        assert resolved["notes"], f"{path} must explain its report role"
        cache_dir = str(resolved["data"].get("cache_dir", "NA"))
        if resolved["approach"] in {"frozen_embeddings", "polyphonic_multilabel"}:
            assert cache_dir != "NA", f"{path} must define the cache it uses"
        assert Path(resolved["output"]["results_dir"]).name == resolved["experiment_id"] or (
            resolved["output"].get("layout") == "run"
            and resolved["experiment_id"] in Path(resolved["output"]["results_dir"]).parts
        )
    assert len(experiment_ids) == len(set(experiment_ids))
