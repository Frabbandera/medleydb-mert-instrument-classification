"""Extract frozen MERT embeddings for polyphonic mixture manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from src.data.audio_examples import load_mixture_segment
from src.features.extract_mert_embeddings import (
    CACHE_SCHEMA_VERSION,
    _load_mert,
    _pool,
    _select_device,
    _selected_representation,
)
from src.utils.paths import atomic_replace, ensure_directory, ensure_parent, load_yaml, resolve_run_path

SPLITS = ("train", "val", "test")
LABEL_SEPARATOR = "|"


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        "mixture_id",
        "mode",
        "split",
        "source_audio_paths",
        "audio_path",
        "start_seconds",
        "duration_seconds",
        "active_labels",
        "k_active",
        "activity_rule",
        "activity_threshold_dbfs",
        "full_mix_exists",
    ]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Mixture manifest is missing columns: {', '.join(missing)}")
    canonical = frame[columns].sort_values("mixture_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _target_vector(active_labels: str, label_to_id: dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(label_to_id), dtype=torch.float32)
    for label in [item for item in str(active_labels).split(LABEL_SEPARATOR) if item]:
        if label not in label_to_id:
            raise ValueError(f"Unknown active label in manifest: {label}")
        vector[int(label_to_id[label])] = 1.0
    return vector


def _load_manifest_audio(row: Any, medleydb_root: Path, sample_rate: int) -> torch.Tensor:
    waveform, _ = load_mixture_segment(row, medleydb_root, target_sample_rate=sample_rate)
    return waveform


def _cache_matches(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        return None
    for key, value in expected.items():
        if key == "model_revision" and value == "main" and cache.get(key):
            continue
        if cache.get(key) != value:
            return None
    return cache


def _matching_existing_caches(
    out_dir: Path,
    *,
    split_frames: dict[str, pd.DataFrame],
    expected_common: dict[str, Any],
) -> dict[str, dict[str, Any]] | None:
    caches: dict[str, dict[str, Any]] = {}
    for split, frame in split_frames.items():
        expected = dict(expected_common)
        expected["split"] = split
        expected["split_fingerprint"] = dataframe_fingerprint(frame)
        cache = _cache_matches(out_dir / f"{split}.pt", expected)
        if cache is None:
            return None
        caches[split] = cache
    return caches


def extract_split(
    frame: pd.DataFrame,
    *,
    split: str,
    medleydb_root: Path,
    label_to_id: dict[str, int],
    processor: Any,
    model: Any,
    model_name: str,
    model_revision: str,
    layer: str,
    pooling: str,
    batch_size: int,
    device: torch.device,
    manifest_fingerprint: str,
    mixture_dataset_id: str,
    mixture_mode: str,
    cache_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"Cache exists and --overwrite was not set: {cache_path}")
    sample_rate = int(getattr(processor, "sampling_rate", 24000))
    label_names = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    embeddings: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metadata: dict[str, list[Any]] = {
        "mixture_ids": [],
        "modes": [],
        "track_ids": [],
        "source_track_ids": [],
        "audio_paths": [],
        "source_audio_paths": [],
        "source_start_seconds": [],
        "start_seconds": [],
        "duration_seconds": [],
        "active_labels": [],
        "k_active": [],
        "genres": [],
        "activity_rules": [],
        "activity_threshold_dbfs": [],
        "source_activity_dbfs": [],
        "full_mix_exists": [],
    }
    selected_indices: list[int] | None = None
    records = list(frame.itertuples(index=False))
    for start in tqdm(range(0, len(records), batch_size), desc=f"Extracting mixture {split}"):
        batch = records[start:start + batch_size]
        waves = [_load_manifest_audio(row, medleydb_root, sample_rate) for row in batch]
        inputs = processor([wave.numpy() for wave in waves], sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            hidden, selected_indices = _selected_representation(outputs, layer)
            pooled = _pool(hidden, pooling).detach().float().cpu()
        embeddings.append(pooled)
        for row in batch:
            targets.append(_target_vector(row.active_labels, label_to_id))
            metadata["mixture_ids"].append(str(row.mixture_id))
            metadata["modes"].append(str(row.mode))
            metadata["track_ids"].append("" if pd.isna(row.track_id) else str(row.track_id))
            metadata["source_track_ids"].append(str(row.source_track_ids))
            metadata["audio_paths"].append("" if pd.isna(row.audio_path) else str(row.audio_path))
            metadata["source_audio_paths"].append(str(row.source_audio_paths))
            metadata["source_start_seconds"].append(str(row.source_start_seconds))
            metadata["start_seconds"].append(float(row.start_seconds))
            metadata["duration_seconds"].append(float(row.duration_seconds))
            metadata["active_labels"].append(str(row.active_labels))
            metadata["k_active"].append(int(row.k_active))
            metadata["genres"].append("" if pd.isna(row.genre) else str(row.genre))
            metadata["activity_rules"].append("" if pd.isna(getattr(row, "activity_rule", "")) else str(row.activity_rule))
            metadata["activity_threshold_dbfs"].append("" if pd.isna(getattr(row, "activity_threshold_dbfs", "")) else str(row.activity_threshold_dbfs))
            metadata["source_activity_dbfs"].append("" if pd.isna(getattr(row, "source_activity_dbfs", "")) else str(row.source_activity_dbfs))
            metadata["full_mix_exists"].append("" if pd.isna(getattr(row, "full_mix_exists", "")) else str(row.full_mix_exists))
    if not embeddings:
        raise ValueError(f"Split '{split}' contains no mixtures")
    embedding_tensor = torch.cat(embeddings, dim=0).contiguous()
    target_tensor = torch.stack(targets).contiguous()
    cache: dict[str, Any] = {
        "embeddings": embedding_tensor,
        "labels": target_tensor,
        "label_names": label_names,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "layer": layer,
        "pooling": pooling,
        "sample_rate": sample_rate,
        "embedding_dim": int(embedding_tensor.shape[-1]),
        "representation_shape": list(embedding_tensor.shape[1:]),
        "hidden_state_indices": selected_indices or [],
        "manifest_fingerprint": manifest_fingerprint,
        "split_fingerprint": dataframe_fingerprint(frame),
        "mixture_dataset_id": mixture_dataset_id,
        "mixture_mode": mixture_mode,
        "split": split,
        **metadata,
    }
    ensure_parent(cache_path)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, temp_path)
    atomic_replace(temp_path, cache_path)
    print(f"Saved {split} mixture cache with shape {tuple(embedding_tensor.shape)}: {cache_path}")
    return cache


def _config_from_caches(caches: dict[str, dict[str, Any]], label_to_id: dict[str, int], expected: dict[str, Any]) -> dict[str, Any]:
    train = caches["train"]
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        **expected,
        "sample_rate": train.get("sample_rate", "NA"),
        "embedding_dim": int(train["embeddings"].shape[-1]),
        "representation_shape": train.get("representation_shape", list(train["embeddings"].shape[1:])),
        "hidden_state_indices": train.get("hidden_state_indices", []),
        "label_to_id": label_to_id,
        "split_sizes": {split: len(caches[split]["labels"]) for split in SPLITS},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--medleydb-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.experiment_config)
    data = config["data"]
    model_config = config["model"]
    representation = config["representation"]
    manifest_path = resolve_run_path(data["manifest_csv"])
    label_path = resolve_run_path(data["label_to_id"])
    out_dir = ensure_directory(resolve_run_path(data["cache_dir"]))
    manifest = pd.read_csv(manifest_path)
    label_to_id = {str(label): int(index) for label, index in json.loads(label_path.read_text(encoding="utf-8")).items()}
    manifest_fingerprint = dataframe_fingerprint(manifest)
    model_name = str(model_config.get("model_name", "m-a-p/MERT-v1-95M"))
    model_revision = str(model_config.get("model_revision", "main"))
    layer = str(representation.get("layer", "last"))
    pooling = str(representation.get("pooling", "mean"))
    mixture_dataset_id = str(data["mixture_dataset_id"])
    mixture_mode = str(data.get("mixture_mode", "synthetic_k"))
    expected = {
        "model_name": model_name,
        "model_revision": model_revision,
        "layer": layer,
        "pooling": pooling,
        "manifest_fingerprint": manifest_fingerprint,
        "mixture_dataset_id": mixture_dataset_id,
        "mixture_mode": mixture_mode,
    }
    split_frames = {split: manifest[manifest["split"] == split].copy() for split in SPLITS}
    if not args.overwrite:
        caches = _matching_existing_caches(out_dir, split_frames=split_frames, expected_common=expected)
        if caches is not None:
            (out_dir / "embedding_config.json").write_text(
                json.dumps(_config_from_caches(caches, label_to_id, expected), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"All mixture split caches already match; MERT was not loaded. Cache: {out_dir}")
            return
    batch_size = int(args.batch_size or data.get("extraction_batch_size", 1))
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = _select_device(args.device)
    print(f"Loading {model_name} on {device}...")
    processor, model = _load_mert(model_name, model_revision, device)
    medleydb_root = Path(args.medleydb_root or os.environ.get("MEDLEYDB_ROOT", data.get("medleydb_root", "MedleyDB")))
    caches = {
        split: extract_split(
            frame,
            split=split,
            medleydb_root=medleydb_root,
            label_to_id=label_to_id,
            processor=processor,
            model=model,
            model_name=model_name,
            model_revision=model_revision,
            layer=layer,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            manifest_fingerprint=manifest_fingerprint,
            mixture_dataset_id=mixture_dataset_id,
            mixture_mode=mixture_mode,
            cache_path=out_dir / f"{split}.pt",
            overwrite=args.overwrite,
        )
        for split, frame in split_frames.items()
    }
    (out_dir / "embedding_config.json").write_text(
        json.dumps(_config_from_caches(caches, label_to_id, expected), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
