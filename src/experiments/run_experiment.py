"""Train, evaluate, and register one isolated-stem experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.config import resolve_experiment_config
from src.experiments.registry import append_experiment, read_registry
from src.training.evaluate_classifier import evaluate
from src.training.train_classifier import train_from_config


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "NA"


def _embedding_metadata(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "embedding_config.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _portable_path(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _training_summary(checkpoint: Path) -> dict[str, Any]:
    path = Path(checkpoint).parent / "training_summary.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _label_metadata(data: dict[str, Any]) -> dict[str, Any]:
    label_path = Path(str(data.get("label_to_id", "")))
    names: list[str] = []
    if label_path.is_file():
        mapping = json.loads(label_path.read_text(encoding="utf-8"))
        if isinstance(mapping, dict):
            names = [label for label, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    granularity = data.get("label_granularity", "coarse_family")
    subset_profile = data.get("subset_profile", "debug")
    subset_path = Path(str(data.get("subset_csv", "")))
    if subset_path.is_file():
        subset = pd.read_csv(subset_path, nrows=1)
        if "label_granularity" in subset and not subset.empty:
            granularity = str(subset.iloc[0]["label_granularity"])
        if "subset_profile" in subset and not subset.empty:
            subset_profile = str(subset.iloc[0]["subset_profile"])
    manifest_path = Path(str(data.get("manifest_csv", "")))
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path, nrows=1)
        if "label_granularity" in data:
            granularity = str(data["label_granularity"])
        if "subset_profile" in data:
            subset_profile = str(data["subset_profile"])
        if "mixture_dataset_id" in manifest and not manifest.empty:
            subset_profile = data.get("subset_profile", subset_profile)
    return {
        "subset_profile": subset_profile,
        "label_granularity": granularity,
        "num_classes": len(names) if names else "NA",
        "class_names": "|".join(names) if names else "NA",
        "label_to_id_path": data.get("label_to_id", "NA"),
    }


def _registry_row(
    config: dict[str, Any], checkpoint: Path, metrics: dict[str, Any]
) -> dict[str, Any]:
    data, model = config["data"], config["model"]
    training, representation = config["training"], config["representation"]
    cache_dir = Path(str(data.get("cache_dir", "NA")))
    embedding = _embedding_metadata(cache_dir) if str(cache_dir) != "NA" else {}
    segment_seconds: Any = "NA"
    subset_path = Path(str(data.get("subset_csv", "")))
    if subset_path.is_file():
        subset = pd.read_csv(subset_path, nrows=1)
        if "duration_seconds" in subset and not subset.empty:
            segment_seconds = float(subset.iloc[0]["duration_seconds"])
    classifier_type = str(model.get("classifier_type", "NA"))
    summary = _training_summary(checkpoint)
    labels = _label_metadata(data)
    return {
        "experiment_id": config["experiment_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": config.get("task", "isolated_single_label"),
        "approach": config["approach"],
        "config_path": config["config_path"],
        "subset_csv": data.get("subset_csv", "NA"),
        "cache_dir": data.get("cache_dir", "NA"),
        "model_name": model.get("model_name", embedding.get("model_name", "NA")),
        "model_revision": embedding.get("model_revision", model.get("model_revision", "NA")),
        "mert_layer": embedding.get("layer", representation.get("layer", "NA")),
        "pooling": embedding.get("pooling", representation.get("pooling", "NA")),
        "representation_mode": model.get("layer_aggregation", "none"),
        "classifier_type": classifier_type,
        "hidden_dim": model.get("hidden_dim", "NA") if classifier_type == "mlp" else "NA",
        "dropout": model.get("dropout", "NA") if classifier_type == "mlp" else "NA",
        "lr": training.get("head_learning_rate", training.get("learning_rate", "NA")),
        "backbone_lr": training.get("backbone_learning_rate", "NA") if config["approach"] == "mert_finetune" else "NA",
        "weight_decay": training.get("weight_decay", "NA"),
        "batch_size": data.get("batch_size", "NA"),
        "max_epochs": training.get("max_epochs", "NA"),
        "seed": config.get("seed", "NA"),
        "unfreeze_mode": model.get("unfreeze_mode", "NA") if config["approach"] == "mert_finetune" else "NA",
        "segment_seconds": segment_seconds,
        "subset_strategy": data.get("subset_strategy", "balanced"),
        "loss_weighting": training.get("loss_weighting", "none"),
        "sampler": data.get("sampler", "shuffle"),
        "git_commit": _git_commit(),
        "checkpoint_path": _portable_path(checkpoint),
        "results_dir": Path(config["output"]["results_dir"]).as_posix(),
        "test_accuracy": metrics.get("test_accuracy", metrics.get("test_exact_match_accuracy", "NA")),
        "test_macro_f1": metrics["test_macro_f1"],
        "test_weighted_f1": metrics.get("test_weighted_f1", "NA"),
        "best_val_macro_f1": summary.get("best_val_macro_f1", "NA"),
        "best_val_accuracy": summary.get("best_val_accuracy", "NA"),
        "best_epoch": summary.get("best_epoch", "NA"),
        "subset_profile": labels["subset_profile"],
        "label_granularity": labels["label_granularity"],
        "num_classes": labels["num_classes"],
        "class_names": labels["class_names"],
        "label_to_id_path": labels["label_to_id_path"],
        "mixture_mode": data.get("mixture_mode", "NA"),
        "mixture_dataset_id": data.get("mixture_dataset_id", "NA"),
        "threshold": config.get("evaluation", {}).get("threshold", "NA"),
        "activity_rule": data.get("activity_rule", config.get("activity_rule", "NA")),
        "activity_threshold_dbfs": data.get("activity_threshold_dbfs", config.get("activity_threshold_dbfs", "NA")),
        "notes": config.get("notes", "NA"),
    }


def run_experiment(
    config_path: Path,
    *,
    experiment_id: str | None = None,
    smoke_test: bool = False,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Run one complete experiment and register it after successful evaluation."""

    config = resolve_experiment_config(
        config_path, experiment_id=experiment_id, smoke_test=smoke_test
    )
    results_dir = Path(config["output"]["results_dir"])
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    registry_path = Path(config["output"]["registry_path"])
    run_layout = config["output"].get("layout") == "run"
    for directory, expected_parent in (
        (results_dir, "results"), (checkpoint_dir, "checkpoints")
    ):
        if run_layout:
            if config["experiment_id"] not in directory.parts:
                raise ValueError(
                    f"{expected_parent} run directory must contain the experiment ID: {directory}"
                )
            continue
        if directory.name != config["experiment_id"] or not directory.parts:
            raise ValueError(
                f"{expected_parent} directory must end with the experiment ID: {directory}"
            )
    registry = read_registry(registry_path)
    already_registered = config["experiment_id"] in set(registry["experiment_id"].astype(str))
    if (already_registered or results_dir.exists() or checkpoint_dir.exists()) and not replace_existing:
        raise FileExistsError(
            f"Experiment '{config['experiment_id']}' already has outputs. "
            "Use --replace-existing to replace it deliberately."
        )
    backups: list[tuple[Path, Path]] = []
    if replace_existing:
        for directory in (results_dir, checkpoint_dir):
            backup = directory.with_name(directory.name + ".replacement_backup")
            shutil.rmtree(backup, ignore_errors=True)
            if directory.exists():
                directory.rename(backup)
                backups.append((directory, backup))
    try:
        if config["approach"] == "classical_baseline":
            from src.training.train_classical_baseline import train_and_evaluate_classical

            checkpoint, metrics = train_and_evaluate_classical(config_path, config)
        elif config["approach"] == "frozen_embeddings":
            checkpoint = train_from_config(config_path, resolved_config=config)
            metrics = evaluate(config_path, checkpoint, resolved_config=config)
        elif config["approach"] == "mert_finetune":
            from src.training.train_finetune import train_and_evaluate_finetune

            checkpoint, metrics = train_and_evaluate_finetune(config_path, config)
        else:
            from src.training.evaluate_multilabel_classifier import evaluate_multilabel
            from src.training.train_multilabel_classifier import train_from_config as train_multilabel_from_config

            checkpoint = train_multilabel_from_config(config_path, resolved_config=config)
            metrics = evaluate_multilabel(config_path, checkpoint, resolved_config=config)
        append_experiment(
            registry_path,
            _registry_row(config, checkpoint, metrics),
            replace=replace_existing,
        )
    except Exception:
        if replace_existing:
            shutil.rmtree(results_dir, ignore_errors=True)
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            for directory, backup in backups:
                if backup.exists():
                    backup.rename(directory)
        raise
    for _, backup in backups:
        shutil.rmtree(backup, ignore_errors=True)
    print(f"Registered experiment: {config['experiment_id']}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.config, experiment_id=args.experiment_id,
                   smoke_test=args.smoke_test, replace_existing=args.replace_existing)


if __name__ == "__main__":
    main()
