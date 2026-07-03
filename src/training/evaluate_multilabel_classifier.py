"""Evaluate a multi-label cached-embedding classifier and save artifacts."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.experiments.config import resolve_experiment_config, save_resolved_config
from src.training.multilabel_datamodule import MultilabelCachedEmbeddingDataModule
from src.training.train_multilabel_classifier import MertMultilabelClassifier


def _select_device(accelerator: Any) -> torch.device:
    wants_gpu = str(accelerator).lower() in {"auto", "gpu", "cuda"}
    return torch.device("cuda" if wants_gpu and torch.cuda.is_available() else "cpu")


def _safe_metric(fn, *args, **kwargs) -> float | str:  # noqa: ANN001
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = fn(*args, **kwargs)
    except ValueError:
        return "NA"
    if isinstance(value, np.ndarray):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return float(value)


def _build_predictions(
    metadata: pd.DataFrame,
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    threshold: float,
) -> pd.DataFrame:
    predictions = (probabilities >= threshold).astype(int)
    rows = metadata.copy()
    rows["true_labels"] = [
        "|".join(label for label, active in zip(label_names, target) if active)
        for target in targets.astype(int)
    ]
    rows["predicted_labels"] = [
        "|".join(label for label, active in zip(label_names, prediction) if active)
        for prediction in predictions
    ]
    rows["exact_match"] = (targets.astype(int) == predictions).all(axis=1)
    for index, label in enumerate(label_names):
        rows[f"true_{label}"] = targets[:, index].astype(int)
        rows[f"pred_{label}"] = predictions[:, index].astype(int)
        rows[f"prob_{label}"] = probabilities[:, index]
    return rows


def _metrics(targets: np.ndarray, probabilities: np.ndarray, label_names: list[str], threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "test_exact_match_accuracy": float((targets.astype(int) == predictions).all(axis=1).mean()),
        "test_accuracy": float((targets.astype(int) == predictions).all(axis=1).mean()),
        "test_micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "test_macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "test_micro_precision": float(precision_score(targets, predictions, average="micro", zero_division=0)),
        "test_macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "test_micro_recall": float(recall_score(targets, predictions, average="micro", zero_division=0)),
        "test_macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "test_map": _safe_metric(average_precision_score, targets, probabilities, average="macro"),
        "test_roc_auc": _safe_metric(roc_auc_score, targets, probabilities, average="macro"),
        "num_test_mixtures": int(len(targets)),
    }
    per_class_rows = []
    for index, label in enumerate(label_names):
        y_true = targets[:, index]
        y_pred = predictions[:, index]
        y_prob = probabilities[:, index]
        per_class_rows.append({
            "label": label,
            "support": int(y_true.sum()),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "average_precision": _safe_metric(average_precision_score, y_true, y_prob),
            "roc_auc": _safe_metric(roc_auc_score, y_true, y_prob),
        })
    return metrics, pd.DataFrame(per_class_rows)


def evaluate_multilabel(
    config_path: Path,
    checkpoint_path: Path,
    *,
    resolved_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = resolved_config or resolve_experiment_config(config_path)
    data_config = config["data"]
    training_config = config["training"]
    eval_config = config.get("evaluation", {}) if isinstance(config.get("evaluation", {}), dict) else {}
    threshold = float(eval_config.get("threshold", 0.5))
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    data_module = MultilabelCachedEmbeddingDataModule(
        cache_dir=Path(data_config["cache_dir"]),
        batch_size=int(data_config.get("batch_size", 32)),
        num_workers=int(data_config.get("num_workers", 0)),
    )
    data_module.setup("test")
    label_names = list(data_module.caches["test"]["label_names"])
    model = MertMultilabelClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    device = _select_device(training_config.get("accelerator", "auto"))
    model.to(device).eval()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for embeddings, labels in data_module.test_dataloader():
            logits = model(embeddings.to(device))
            probabilities.append(torch.sigmoid(logits).cpu())
            targets.append(labels.cpu())
    probabilities_array = torch.cat(probabilities).numpy().astype(np.float64)
    targets_array = torch.cat(targets).numpy().astype(np.int64)
    cache = data_module.caches["test"]
    metadata = pd.DataFrame({
        "mixture_id": cache["mixture_ids"],
        "mode": cache["modes"],
        "track_id": cache["track_ids"],
        "source_track_ids": cache["source_track_ids"],
        "audio_path": cache["audio_paths"],
        "source_audio_paths": cache["source_audio_paths"],
        "source_start_seconds": cache["source_start_seconds"],
        "start_seconds": cache["start_seconds"],
        "duration_seconds": cache["duration_seconds"],
        "active_labels": cache["active_labels"],
        "k_active": cache["k_active"],
        "genre": cache.get("genres", [""] * len(cache["mixture_ids"])),
        "activity_rule": cache.get("activity_rules", [""] * len(cache["mixture_ids"])),
        "activity_threshold_dbfs": cache.get("activity_threshold_dbfs", [""] * len(cache["mixture_ids"])),
        "source_activity_dbfs": cache.get("source_activity_dbfs", [""] * len(cache["mixture_ids"])),
        "full_mix_exists": cache.get("full_mix_exists", [""] * len(cache["mixture_ids"])),
    })
    metrics, per_class = _metrics(targets_array, probabilities_array, label_names, threshold)
    save_resolved_config(config, results_dir / "config_resolved.yaml")
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    per_class.to_csv(results_dir / "per_class_metrics.csv", index=False)
    _build_predictions(metadata, targets_array, probabilities_array, label_names, threshold).to_csv(results_dir / "predictions.csv", index=False)
    print(f"Test micro-F1: {metrics['test_micro_f1']:.4f}")
    print(f"Test macro-F1: {metrics['test_macro_f1']:.4f}")
    print(f"Results: {results_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_experiment_config(args.config, experiment_id=args.experiment_id)
    evaluate_multilabel(args.config, args.checkpoint, resolved_config=config)


if __name__ == "__main__":
    main()
