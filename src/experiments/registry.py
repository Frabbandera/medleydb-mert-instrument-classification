"""CSV registry for completed isolated-stem experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import ensure_parent

REGISTRY_COLUMNS = [
    "experiment_id", "timestamp", "task", "approach", "config_path", "subset_csv",
    "cache_dir", "model_name", "model_revision", "mert_layer", "pooling",
    "representation_mode", "classifier_type", "hidden_dim", "dropout", "lr",
    "backbone_lr", "weight_decay", "batch_size", "max_epochs", "seed",
    "unfreeze_mode", "segment_seconds", "subset_strategy", "loss_weighting",
    "sampler", "git_commit", "checkpoint_path", "results_dir", "test_accuracy",
    "test_macro_f1", "test_weighted_f1", "best_val_macro_f1",
    "best_val_accuracy", "best_epoch", "subset_profile", "label_granularity",
    "num_classes", "class_names", "label_to_id_path", "mixture_mode",
    "mixture_dataset_id", "threshold", "activity_rule",
    "activity_threshold_dbfs", "notes",
]


def read_registry(path: Path) -> pd.DataFrame:
    """Read a registry, returning an empty stable-schema frame when absent."""

    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    frame = pd.read_csv(path, keep_default_na=False)
    for column in REGISTRY_COLUMNS:
        if column not in frame:
            frame[column] = "NA"
    return frame[REGISTRY_COLUMNS]


def append_experiment(path: Path, row: dict[str, Any], *, replace: bool = False) -> None:
    """Atomically append one unique completed experiment to the registry."""

    path = Path(path)
    frame = read_registry(path)
    experiment_id = str(row.get("experiment_id", ""))
    duplicate = frame["experiment_id"].astype(str) == experiment_id
    if duplicate.any() and not replace:
        raise FileExistsError(f"Experiment already exists in registry: {experiment_id}")
    if replace:
        frame = frame.loc[~duplicate].copy()
    normalized = {column: row.get(column, "NA") for column in REGISTRY_COLUMNS}
    normalized = {key: ("NA" if value is None or value == "" else value) for key, value in normalized.items()}
    frame = pd.concat([frame, pd.DataFrame([normalized])], ignore_index=True)
    ensure_parent(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)
