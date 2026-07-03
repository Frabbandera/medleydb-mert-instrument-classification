"""Shared prediction formatting and evaluation artifact generation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "medleydb_mert_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.experiments.config import save_resolved_config


def normalize_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows by true-class support, leaving empty rows at zero."""

    matrix = np.asarray(matrix, dtype=np.float64)
    denominators = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denominators, out=np.zeros_like(matrix), where=denominators != 0)


def build_predictions_frame(
    metadata: pd.DataFrame,
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """Build the stable per-example prediction table used by error analysis."""

    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(targets), len(label_names)):
        raise ValueError("Probability array shape does not match targets/classes")
    if len(metadata) != len(targets):
        raise ValueError("Prediction metadata length does not match targets")
    predictions = probabilities.argmax(axis=1)
    rows = pd.DataFrame({
        "segment_id": metadata["segment_id"].astype(str).tolist(),
        "track_id": metadata["track_id"].astype(str).tolist(),
        "audio_path": metadata["audio_path"].astype(str).tolist(),
        "start_seconds": metadata.get("start_seconds", pd.Series([""] * len(metadata))).tolist(),
        "duration_seconds": metadata.get("duration_seconds", pd.Series([""] * len(metadata))).tolist(),
        "true_label": [label_names[index] for index in targets],
        "predicted_label": [label_names[index] for index in predictions],
        "correct": targets == predictions,
        "probability_true_class": probabilities[np.arange(len(targets)), targets],
        "probability_predicted_class": probabilities[np.arange(len(targets)), predictions],
    })
    for index, label in enumerate(label_names):
        rows[f"prob_{label}"] = probabilities[:, index]
    return rows


def _plot_matrix(matrix: np.ndarray, labels: list[str], path: Path, *, normalized: bool) -> None:
    size = max(7.0, len(labels) * 0.9)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalized else None)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    ax.set(xticks=ticks, yticks=ticks, xticklabels=labels, yticklabels=labels,
           xlabel="Predicted label", ylabel="True label",
           title="Normalized confusion matrix" if normalized else "Raw confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = (matrix.max() / 2.0) if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            text = f"{matrix[row, column]:.2f}" if normalized else str(int(matrix[row, column]))
            ax.text(column, row, text, ha="center", va="center",
                    color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_artifacts(
    *,
    results_dir: Path,
    resolved_config: dict[str, Any],
    metadata: pd.DataFrame,
    targets: list[int] | np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Save metrics, reports, matrices, plots, predictions, and resolved config."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("confusion_matrix.csv", "confusion_matrix.png"):
        legacy_path = results_dir / legacy_name
        if legacy_path.is_file():
            legacy_path.unlink()
    targets_array = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(probabilities).argmax(axis=1)
    label_ids = list(range(len(label_names)))
    report = classification_report(targets_array, predictions, labels=label_ids,
                                   target_names=label_names, output_dict=True, zero_division=0)
    metrics = {
        "test_accuracy": float(accuracy_score(targets_array, predictions)),
        "test_macro_f1": float(f1_score(targets_array, predictions, labels=label_ids, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(targets_array, predictions, labels=label_ids, average="weighted", zero_division=0)),
        "num_test_segments": int(len(targets_array)),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in label_names},
    }
    save_resolved_config(resolved_config, results_dir / "config_resolved.yaml")
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_frame = pd.DataFrame(report).transpose()
    report_frame.index.name = "label"
    report_frame.to_csv(results_dir / "classification_report.csv")
    raw = confusion_matrix(targets_array, predictions, labels=label_ids)
    normalized = normalize_confusion_matrix(raw)
    pd.DataFrame(raw, index=label_names, columns=label_names).to_csv(results_dir / "confusion_matrix_raw.csv", index_label="true_label")
    pd.DataFrame(normalized, index=label_names, columns=label_names).to_csv(results_dir / "confusion_matrix_normalized.csv", index_label="true_label")
    _plot_matrix(raw, label_names, results_dir / "confusion_matrix_raw.png", normalized=False)
    _plot_matrix(normalized, label_names, results_dir / "confusion_matrix_normalized.png", normalized=True)
    build_predictions_frame(metadata, targets_array, probabilities, label_names).to_csv(results_dir / "predictions.csv", index=False)
    return metrics
