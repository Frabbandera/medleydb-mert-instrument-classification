"""Evaluate a cached-embedding classifier and save complete test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.experiments.config import resolve_experiment_config
from src.experiments.run_utils import save_evaluation_artifacts
from src.training.datamodule import CachedEmbeddingDataModule
from src.training.train_classifier import MertEmbeddingClassifier


def _load_label_map(path: Path) -> tuple[dict[str, int], list[str]]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("label_to_id must contain a JSON object")
    label_to_id = {str(label): int(index) for label, index in mapping.items()}
    if sorted(label_to_id.values()) != list(range(len(label_to_id))):
        raise ValueError("label_to_id values must be contiguous from zero")
    names = [""] * len(label_to_id)
    for label, index in label_to_id.items():
        names[index] = label
    return label_to_id, names


def _select_device(accelerator: Any) -> torch.device:
    wants_gpu = str(accelerator).lower() in {"auto", "gpu", "cuda"}
    return torch.device("cuda" if wants_gpu and torch.cuda.is_available() else "cpu")


def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    *,
    resolved_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run held-out inference and write standardized experiment artifacts."""

    config = resolved_config or resolve_experiment_config(config_path)
    data_config = config["data"]
    training_config = config["training"]
    results_dir = Path(config["output"]["results_dir"])
    _, label_names = _load_label_map(Path(data_config["label_to_id"]))
    data_module = CachedEmbeddingDataModule(
        cache_dir=Path(data_config["cache_dir"]),
        batch_size=int(data_config.get("batch_size", 32)),
        num_workers=int(data_config.get("num_workers", 0)),
        sampler=str(data_config.get("sampler", "shuffle")),
    )
    data_module.setup("test")
    if data_module.num_classes != len(label_names):
        raise ValueError("Cache class count does not match label map")
    model = MertEmbeddingClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    if int(model.hparams.input_dim) != data_module.input_dim:
        raise ValueError("Checkpoint input dimension does not match the test cache")
    device = _select_device(training_config.get("accelerator", "auto"))
    model.to(device).eval()
    probabilities: list[torch.Tensor] = []
    targets: list[int] = []
    with torch.inference_mode():
        for embeddings, labels in data_module.test_dataloader():
            logits = model(embeddings.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            targets.extend(labels.tolist())
    cache = data_module.caches["test"]
    metadata = pd.DataFrame({
        "segment_id": cache["segment_ids"],
        "track_id": cache["track_ids"],
        "audio_path": cache["audio_paths"],
        "start_seconds": cache.get("start_seconds", [""] * len(cache["segment_ids"])),
        "duration_seconds": cache.get("duration_seconds", [""] * len(cache["segment_ids"])),
    })
    metrics = save_evaluation_artifacts(
        results_dir=results_dir,
        resolved_config=config,
        metadata=metadata,
        targets=targets,
        probabilities=torch.cat(probabilities).numpy().astype(np.float64),
        label_names=label_names,
    )
    print(f"Test accuracy:    {metrics['test_accuracy']:.4f}")
    print(f"Test macro-F1:    {metrics['test_macro_f1']:.4f}")
    print(f"Test weighted-F1: {metrics['test_weighted_f1']:.4f}")
    for label, value in metrics["per_class_f1"].items():
        print(f"  {label}: {value:.4f}")
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
    evaluate(args.config, args.checkpoint, resolved_config=config)


if __name__ == "__main__":
    main()
