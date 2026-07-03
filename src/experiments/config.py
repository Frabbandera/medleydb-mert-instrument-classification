"""Load and resolve experiment YAML files into one stable schema."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import experiment_run_dir, load_yaml, resolve_run_path

EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_experiment_id(value: str) -> str:
    """Validate an experiment identifier safe for use as a directory name."""

    value = str(value).strip()
    if not value or not EXPERIMENT_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "experiment_id may contain only letters, digits, '.', '_', and '-'"
        )
    return value


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.setdefault(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping")
    return value


def resolve_experiment_config(
    config_path: Path,
    *,
    experiment_id: str | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Return a complete config while accepting the original pipeline schema."""

    path = Path(config_path)
    raw = copy.deepcopy(load_yaml(path))
    chosen_id = experiment_id or raw.get("experiment_id") or raw.get("experiment_name")
    if not chosen_id:
        chosen_id = path.stem
    chosen_id = validate_experiment_id(str(chosen_id))
    raw["experiment_id"] = chosen_id
    raw.setdefault("experiment_name", chosen_id)
    if experiment_id is not None:
        raw["experiment_name"] = chosen_id
    raw.setdefault("approach", "frozen_embeddings")
    allowed_approaches = {
        "classical_baseline",
        "frozen_embeddings",
        "mert_finetune",
        "polyphonic_multilabel",
    }
    if raw["approach"] not in allowed_approaches:
        raise ValueError(
            "approach must be one of " + ", ".join(sorted(allowed_approaches))
        )
    raw.setdefault("notes", "")
    raw["seed"] = int(raw.get("seed", 42))

    data = _mapping(raw, "data")
    data.setdefault("subset_csv", "data/metadata/subset_largest_balanced_medleydb_instrument.csv")
    data.setdefault(
        "label_to_id",
        "data/metadata/labels_largest_balanced_medleydb_instrument_label_to_id.json",
    )
    data.setdefault("subset_profile", "largest_balanced")
    data.setdefault("label_granularity", "medleydb_instrument")
    data.setdefault(
        "cache_dir",
        "NA"
        if raw["approach"] in {"classical_baseline", "mert_finetune"}
        else "data/cache/mert_v1_95m/largest_balanced/medleydb_instrument/layer_last_pool_mean",
    )
    data.setdefault("medleydb_root", "MedleyDB")
    if os.environ.get("MEDLEYDB_ROOT"):
        data["medleydb_root"] = os.environ["MEDLEYDB_ROOT"]
    data.setdefault("batch_size", 1 if raw["approach"] == "mert_finetune" else 32)
    data.setdefault("num_workers", 0)
    data.setdefault("sampler", "shuffle")
    data.setdefault("subset_strategy", "balanced")
    for key in ("subset_csv", "label_to_id", "cache_dir", "manifest_csv"):
        if key in data and str(data[key]) != "NA":
            data[key] = resolve_run_path(data[key]).as_posix()
    if data["sampler"] not in {"shuffle", "weighted"}:
        raise ValueError("data.sampler must be 'shuffle' or 'weighted'")

    model = _mapping(raw, "model")
    model.setdefault("model_name", "m-a-p/MERT-v1-95M")
    model.setdefault("model_revision", "main")
    model.setdefault("classifier_type", "mlp")
    model.setdefault("hidden_dim", 256)
    model.setdefault("dropout", 0.3)
    model.setdefault("layer_aggregation", "none")
    model.setdefault("unfreeze_mode", "frozen")
    model.setdefault("allow_full_finetune", False)

    representation = _mapping(raw, "representation")
    representation.setdefault("layer", "last")
    representation.setdefault("pooling", "mean")

    training = _mapping(raw, "training")
    training.setdefault("learning_rate", 1e-3)
    training.setdefault("head_learning_rate", training["learning_rate"])
    training.setdefault("backbone_learning_rate", 1e-5)
    training.setdefault("weight_decay", 1e-4)
    training.setdefault("max_epochs", 50)
    training.setdefault("early_stopping_patience", 8)
    training.setdefault("accelerator", "auto")
    training.setdefault("devices", 1)
    training.setdefault("precision", "16-mixed" if raw["approach"] == "mert_finetune" else "32-true")
    training.setdefault("deterministic", True)
    training.setdefault("accumulate_grad_batches", 8 if raw["approach"] == "mert_finetune" else 1)
    training.setdefault("gradient_checkpointing", raw["approach"] == "mert_finetune")
    training.setdefault("loss_weighting", "pos_weight" if raw["approach"] == "polyphonic_multilabel" else "none")
    training.setdefault("allow_double_rebalancing", False)
    allowed_loss_weighting = (
        {"none", "pos_weight"}
        if raw["approach"] == "polyphonic_multilabel"
        else {"none", "inverse_frequency"}
    )
    if training["loss_weighting"] not in allowed_loss_weighting:
        raise ValueError(
            "training.loss_weighting must be one of "
            + ", ".join(sorted(allowed_loss_weighting))
        )
    if (
        data["sampler"] == "weighted"
        and training["loss_weighting"] == "inverse_frequency"
        and not bool(training["allow_double_rebalancing"])
    ):
        raise ValueError(
            "Weighted sampling and inverse-frequency loss cannot be combined unless "
            "training.allow_double_rebalancing is true"
        )
    if smoke_test:
        training["max_epochs"] = 1
        training["limit_train_batches"] = 2
        training["limit_val_batches"] = 1
        training["enable_progress_bar"] = False
        training["enable_model_summary"] = False
        training["log_every_n_steps"] = 1
        training["num_sanity_val_steps"] = 0
        training["logger"] = False
        raw["smoke_test"] = True
    else:
        raw["smoke_test"] = False

    output = _mapping(raw, "output")
    old_evaluation = _mapping(raw, "evaluation")
    default_results = (
        f"results/{chosen_id}"
        if experiment_id is not None
        else old_evaluation.get("results_dir", f"results/{chosen_id}")
    )
    default_checkpoint = (
        f"checkpoints/{chosen_id}"
        if experiment_id is not None
        else training.get("checkpoint_dir", f"checkpoints/{chosen_id}")
    )
    output.setdefault("results_dir", default_results)
    output.setdefault("checkpoint_dir", default_checkpoint)
    output.setdefault("registry_path", "results/experiment_registry.csv")
    if experiment_id is not None:
        output["results_dir"] = (Path(output["results_dir"]).parent / chosen_id).as_posix()
        output["checkpoint_dir"] = (Path(output["checkpoint_dir"]).parent / chosen_id).as_posix()
    if output.get("layout") == "run":
        run_id = output.get("run_id")
        results_root = resolve_run_path(str(output.get("results_root", "results")))
        checkpoints_root = resolve_run_path(str(output.get("checkpoints_root", "checkpoints")))
        output["results_dir"] = experiment_run_dir(results_root, chosen_id, raw["seed"], run_id).as_posix()
        output["checkpoint_dir"] = experiment_run_dir(checkpoints_root, chosen_id, raw["seed"], run_id).as_posix()
    else:
        output["results_dir"] = resolve_run_path(output["results_dir"]).as_posix()
        output["checkpoint_dir"] = resolve_run_path(output["checkpoint_dir"]).as_posix()
    output["registry_path"] = resolve_run_path(output["registry_path"]).as_posix()
    old_evaluation["results_dir"] = output["results_dir"]
    training["checkpoint_dir"] = output["checkpoint_dir"]
    raw["config_path"] = path.as_posix()
    return raw


def save_resolved_config(config: dict[str, Any], path: Path) -> None:
    """Write a human-readable resolved configuration."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
