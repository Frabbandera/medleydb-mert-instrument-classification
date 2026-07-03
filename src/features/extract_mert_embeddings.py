"""Extract and cache frozen MERT embeddings for balanced stem segments."""

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

from src.data.audio_io import load_audio_segment
from src.utils.paths import atomic_replace, ensure_directory, ensure_parent, load_yaml, resolve_data_path, resolve_run_path

SPLITS = ("train", "val", "test")
CACHE_SCHEMA_VERSION = 3


def label_column(frame: pd.DataFrame) -> str:
    """Return the active label-name column for old or new subset tables."""

    return "label_name" if "label_name" in frame.columns else "coarse_label"


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Hash the fields that define the audio and target of every segment."""

    columns = [
        "segment_id",
        "track_id",
        "audio_path",
        "start_seconds",
        "duration_seconds",
        "label_id",
        "split",
    ]
    columns.insert(5, label_column(frame))
    for optional in ("subset_profile", "label_granularity"):
        if optional in frame.columns:
            columns.append(optional)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Segment table is missing columns: {', '.join(missing)}")
    canonical = frame[columns].sort_values("segment_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_torch_cache(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary cache in {path}")
    _normalize_cache_dict(value)
    return value


def _normalize_cache_dict(cache: dict[str, Any]) -> dict[str, Any]:
    """Fill fields absent from older cache files without rewriting them."""

    embeddings = cache.get("embeddings")
    if isinstance(embeddings, torch.Tensor):
        cache.setdefault("representation_shape", list(embeddings.shape[1:]))
        if embeddings.ndim >= 2:
            cache.setdefault("embedding_dim", int(embeddings.shape[-1]))
    cache.setdefault("hidden_state_indices", [])
    cache.setdefault("cache_schema_version", 1)
    return cache


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _load_mert(model_name: str, revision: str | None, device: torch.device):
    try:
        from transformers import AutoFeatureExtractor, AutoModel

        common: dict[str, Any] = {"trust_remote_code": True}
        if revision:
            common["revision"] = revision
        processor = AutoFeatureExtractor.from_pretrained(model_name, **common)
        model = AutoModel.from_pretrained(model_name, **common)
    except Exception as exc:
        raise RuntimeError(
            "Could not load Hugging Face MERT. Check network/Hugging Face cache access and "
            "the installed Transformers 4.x release. If remote MERT code is incompatible, "
            "install `transformers==4.38.2` and retry. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return processor, model


def _selected_representation(outputs: Any, layer: str) -> tuple[torch.Tensor, list[int]]:
    """Select a fixed, averaged, or stacked Hugging Face hidden representation."""

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("MERT did not return hidden states")
    count = len(hidden_states)
    layer = str(layer).strip().lower()
    if layer == "last":
        return hidden_states[-1], [count - 1]
    if layer == "all":
        return torch.stack(tuple(hidden_states), dim=1), list(range(count))
    if layer.startswith("last") and layer.endswith("avg"):
        number = layer.removeprefix("last").removesuffix("avg")
        if not number.isdigit() or int(number) <= 0:
            raise ValueError("Last-k averaging must use syntax such as 'last4avg'")
        k = int(number)
        if k > count:
            raise ValueError(f"Cannot average the last {k} of only {count} hidden states")
        indices = list(range(count - k, count))
        return torch.stack(tuple(hidden_states[-k:]), dim=0).mean(dim=0), indices
    try:
        requested = int(layer)
    except ValueError as exc:
        raise ValueError(
            "--layer must be 'last', an integer, 'lastKavg', or 'all'"
        ) from exc
    resolved = requested if requested >= 0 else count + requested
    if resolved < 0 or resolved >= count:
        raise ValueError(f"Hidden-state index {requested} is invalid for {count} states")
    return hidden_states[requested], [resolved]


def _pool(hidden: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool time for either [B,T,D] or [B,L,T,D] representations."""

    time_dimension = -2
    if pooling == "mean":
        return hidden.mean(dim=time_dimension)
    if pooling == "max":
        return hidden.amax(dim=time_dimension)
    if pooling == "meanmax":
        return torch.cat(
            [hidden.mean(dim=time_dimension), hidden.amax(dim=time_dimension)], dim=-1
        )
    raise ValueError("pooling must be 'mean', 'max', or 'meanmax'")


def _cache_matches(cache: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        cached = cache.get(key)
        if key == "model_revision" and value == "main" and cached:
            # Older caches may store the resolved Hugging Face commit hash,
            # while an offline cache-hit check only knows that the requested
            # revision is the default branch.  Treat that as compatible when
            # the model name, subset, layer, pooling, and split match.
            continue
        if cached != value:
            return False
    return True


def _expected_for_split(
    split_frame: pd.DataFrame,
    *,
    split: str,
    model_name: str,
    model_revision: str,
    layer: str,
    pooling: str,
    subset_fingerprint: str,
    subset_profile: str,
    label_granularity: str,
    segment_seconds: float,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_revision": model_revision,
        "layer": layer,
        "pooling": pooling,
        "subset_fingerprint": subset_fingerprint,
        "subset_profile": subset_profile,
        "label_granularity": label_granularity,
        "segment_seconds": segment_seconds,
        "split_fingerprint": dataframe_fingerprint(split_frame),
        "split": split,
    }


def _single_metadata_value(frame: pd.DataFrame, column: str, default: str) -> str:
    """Return one stable metadata value from a subset table."""

    if column not in frame.columns or frame.empty:
        return default
    values = sorted({str(value) for value in frame[column].dropna().unique()})
    if not values:
        return default
    if len(values) > 1:
        raise ValueError(f"Segment table contains multiple {column} values: {values}")
    return values[0]


def _segment_seconds(frame: pd.DataFrame) -> float:
    """Return the common segment length used by a subset table."""

    if "duration_seconds" not in frame.columns or frame.empty:
        return 0.0
    values = sorted({round(float(value), 6) for value in frame["duration_seconds"].dropna().unique()})
    if len(values) > 1:
        raise ValueError(f"Segment table contains multiple duration_seconds values: {values}")
    return float(values[0]) if values else 0.0


def _embedding_config_from_caches(
    *,
    caches: dict[str, dict[str, Any]],
    label_to_id: dict[str, Any],
    model_name: str,
    model_revision: str,
    layer: str,
    pooling: str,
    subset_fingerprint: str,
    subset_profile: str,
    label_granularity: str,
    segment_seconds: float,
) -> dict[str, Any]:
    train = caches["train"]
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "layer": layer,
        "pooling": pooling,
        "subset_profile": subset_profile,
        "label_granularity": label_granularity,
        "segment_seconds": segment_seconds,
        "sample_rate": train.get("sample_rate", "NA"),
        "embedding_dim": int(train["embeddings"].shape[-1]),
        "representation_shape": train.get("representation_shape", list(train["embeddings"].shape[1:])),
        "hidden_state_indices": train.get("hidden_state_indices", []),
        "subset_fingerprint": subset_fingerprint,
        "label_to_id": label_to_id,
        "split_sizes": {split: len(caches[split]["labels"]) for split in SPLITS},
    }


def _matching_existing_caches(
    *,
    out_dir: Path,
    split_frames: dict[str, pd.DataFrame],
    model_name: str,
    model_revision: str,
    layer: str,
    pooling: str,
    subset_fingerprint: str,
    subset_profile: str,
    label_granularity: str,
    segment_seconds: float,
) -> dict[str, dict[str, Any]] | None:
    caches: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = out_dir / f"{split}.pt"
        if not path.exists():
            return None
        cache = _load_torch_cache(path)
        expected = _expected_for_split(
            split_frames[split],
            split=split,
            model_name=model_name,
            model_revision=model_revision,
            layer=layer,
            pooling=pooling,
            subset_fingerprint=subset_fingerprint,
            subset_profile=subset_profile,
            label_granularity=label_granularity,
            segment_seconds=segment_seconds,
        )
        if not _cache_matches(cache, expected):
            return None
        caches[split] = cache
    return caches


def extract_split(
    split_frame: pd.DataFrame,
    *,
    split: str,
    medleydb_root: Path,
    processor: Any,
    model: torch.nn.Module,
    model_name: str,
    model_revision: str,
    layer: str,
    pooling: str,
    batch_size: int,
    device: torch.device,
    subset_fingerprint: str,
    subset_profile: str,
    label_granularity: str,
    segment_seconds: float,
    cache_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Extract one split and atomically write its cache."""

    active_label_column = label_column(split_frame)
    split_fingerprint = dataframe_fingerprint(split_frame)
    expected = _expected_for_split(
        split_frame,
        split=split,
        model_name=model_name,
        model_revision=model_revision,
        layer=layer,
        pooling=pooling,
        subset_fingerprint=subset_fingerprint,
        subset_profile=subset_profile,
        label_granularity=label_granularity,
        segment_seconds=segment_seconds,
    )
    if cache_path.exists() and not overwrite:
        existing = _load_torch_cache(cache_path)
        if _cache_matches(existing, expected):
            print(f"Skipping matching existing cache: {cache_path}")
            return existing
        raise RuntimeError(
            f"Existing cache settings do not match the requested extraction: {cache_path}. "
            "Pass --overwrite to replace it."
        )

    sample_rate = int(getattr(processor, "sampling_rate", 24000))
    embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    segment_ids: list[str] = []
    track_ids: list[str] = []
    audio_paths: list[str] = []
    start_seconds: list[float] = []
    duration_seconds: list[float] = []
    label_names: list[str] = []
    selected_indices: list[int] | None = None

    rows = split_frame.reset_index(drop=True)
    progress = tqdm(range(0, len(rows), batch_size), desc=f"MERT {split}")
    try:
        with torch.inference_mode():
            for start in progress:
                batch = rows.iloc[start : start + batch_size]
                arrays = []
                for row in batch.itertuples(index=False):
                    audio_path = resolve_data_path(medleydb_root, row.audio_path)
                    waveform, loaded_rate = load_audio_segment(
                        audio_path,
                        float(row.start_seconds),
                        float(row.duration_seconds),
                        target_sample_rate=sample_rate,
                        normalize=True,
                        pad=True,
                    )
                    if loaded_rate != sample_rate:
                        raise RuntimeError(
                            f"Audio loader returned {loaded_rate} Hz; MERT expects {sample_rate} Hz"
                        )
                    arrays.append(waveform.cpu().numpy())
                inputs = processor(
                    arrays,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                hidden, batch_indices = _selected_representation(outputs, layer)
                if selected_indices is None:
                    selected_indices = batch_indices
                elif selected_indices != batch_indices:
                    raise RuntimeError("MERT hidden-state count changed during extraction")
                pooled = _pool(hidden, pooling).detach().float().cpu()
                embeddings.append(pooled)
                labels.extend(int(value) for value in batch["label_id"].tolist())
                segment_ids.extend(batch["segment_id"].astype(str).tolist())
                track_ids.extend(batch["track_id"].astype(str).tolist())
                audio_paths.extend(batch["audio_path"].astype(str).tolist())
                start_seconds.extend(float(value) for value in batch["start_seconds"].tolist())
                duration_seconds.extend(float(value) for value in batch["duration_seconds"].tolist())
                label_names.extend(batch[active_label_column].astype(str).tolist())
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA ran out of memory during MERT extraction. Retry with --batch-size 1, "
            "close other GPU processes, or use --device cpu."
        ) from exc

    if not embeddings:
        raise ValueError(f"Split '{split}' contains no segments")
    embedding_tensor = torch.cat(embeddings, dim=0).contiguous()
    cache: dict[str, Any] = {
        "embeddings": embedding_tensor,
        "labels": torch.tensor(labels, dtype=torch.long),
        "segment_ids": segment_ids,
        "track_ids": track_ids,
        "audio_paths": audio_paths,
        "start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
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
        "subset_fingerprint": subset_fingerprint,
        "subset_profile": subset_profile,
        "label_granularity": label_granularity,
        "segment_seconds": segment_seconds,
        "split_fingerprint": split_fingerprint,
        "split": split,
    }
    ensure_parent(cache_path)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, temp_path)
    atomic_replace(temp_path, cache_path)
    print(f"Saved {split} cache with shape {tuple(embedding_tensor.shape)}: {cache_path}")
    return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, help="Experiment YAML config to use as the source of truth.")
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--label-to-id", type=Path)
    parser.add_argument("--medleydb-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--model-name", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layer", default="last")
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "meanmax"])
    parser.add_argument("--subset-profile", default=None)
    parser.add_argument("--label-granularity", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_experiment_config(args: argparse.Namespace) -> argparse.Namespace:
    """Populate extractor arguments from an experiment YAML config."""

    if args.experiment_config is None:
        return args
    config = load_yaml(args.experiment_config)
    data = config.get("data", {})
    model = config.get("model", {})
    representation = config.get("representation", {})
    if not isinstance(data, dict) or not isinstance(model, dict) or not isinstance(representation, dict):
        raise ValueError("Experiment config must contain mapping sections: data, model, representation")
    args.segments = resolve_run_path(data["subset_csv"])
    args.label_to_id = resolve_run_path(data["label_to_id"])
    args.medleydb_root = Path(os.environ.get("MEDLEYDB_ROOT", data.get("medleydb_root", args.medleydb_root or "MedleyDB")))
    args.out_dir = resolve_run_path(data["cache_dir"])
    args.model_name = str(model.get("model_name", args.model_name))
    args.revision = model.get("model_revision", args.revision)
    args.layer = str(representation.get("layer", args.layer))
    args.pooling = str(representation.get("pooling", args.pooling))
    args.subset_profile = data.get("subset_profile", args.subset_profile)
    args.label_granularity = data.get("label_granularity", args.label_granularity)
    return args


def validate_required_args(args: argparse.Namespace) -> None:
    missing = [
        name for name in ("segments", "label_to_id", "medleydb_root", "out_dir")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Missing required extractor settings: "
            + ", ".join(missing)
            + ". Supply them on the CLI or via --experiment-config."
        )


def main() -> None:
    args = apply_experiment_config(parse_args())
    validate_required_args(args)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    segments = pd.read_csv(args.segments)
    active_label_column = label_column(segments)
    label_to_id = json.loads(args.label_to_id.read_text(encoding="utf-8"))
    if not isinstance(label_to_id, dict):
        raise ValueError("label_to_id.json must contain a JSON object")
    for row in segments.itertuples(index=False):
        label_value = str(getattr(row, active_label_column))
        expected_id = label_to_id.get(label_value)
        if expected_id is None or int(expected_id) != int(row.label_id):
            raise ValueError(
                f"Label mapping mismatch for segment {row.segment_id}: "
                f"{label_value}/{row.label_id}"
            )

    subset_fingerprint = dataframe_fingerprint(segments)
    subset_profile = args.subset_profile or _single_metadata_value(
        segments, "subset_profile", "unknown"
    )
    label_granularity = args.label_granularity or _single_metadata_value(
        segments, "label_granularity", "coarse_family"
    )
    segment_seconds = _segment_seconds(segments)
    out_dir = ensure_directory(args.out_dir)
    requested_revision = str(args.revision or "main")
    split_frames = {split: segments[segments["split"] == split].copy() for split in SPLITS}
    if not args.overwrite:
        existing_caches = _matching_existing_caches(
            out_dir=out_dir,
            split_frames=split_frames,
            model_name=args.model_name,
            model_revision=requested_revision,
            layer=args.layer,
            pooling=args.pooling,
            subset_fingerprint=subset_fingerprint,
            subset_profile=subset_profile,
            label_granularity=label_granularity,
            segment_seconds=segment_seconds,
        )
        if existing_caches is not None:
            config = _embedding_config_from_caches(
                caches=existing_caches,
                label_to_id=label_to_id,
                model_name=args.model_name,
                model_revision=requested_revision,
                layer=args.layer,
                pooling=args.pooling,
                subset_fingerprint=subset_fingerprint,
                subset_profile=subset_profile,
                label_granularity=label_granularity,
                segment_seconds=segment_seconds,
            )
            (out_dir / "embedding_config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"All split caches already match; MERT was not loaded. Cache: {out_dir}")
            return

    device = _select_device(args.device)
    print(f"Loading {args.model_name} on {device}...")
    processor, model = _load_mert(args.model_name, args.revision, device)
    resolved_revision = requested_revision
    caches: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_frame = split_frames[split]
        caches[split] = extract_split(
            split_frame,
            split=split,
            medleydb_root=args.medleydb_root,
            processor=processor,
            model=model,
            model_name=args.model_name,
            model_revision=resolved_revision,
            layer=args.layer,
            pooling=args.pooling,
            batch_size=args.batch_size,
            device=device,
            subset_fingerprint=subset_fingerprint,
            subset_profile=subset_profile,
            label_granularity=label_granularity,
            segment_seconds=segment_seconds,
            cache_path=out_dir / f"{split}.pt",
            overwrite=args.overwrite,
        )

    config = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model_name": args.model_name,
        "model_revision": resolved_revision,
        "layer": args.layer,
        "pooling": args.pooling,
        "subset_profile": subset_profile,
        "label_granularity": label_granularity,
        "segment_seconds": segment_seconds,
        "sample_rate": caches["train"]["sample_rate"],
        "embedding_dim": int(caches["train"]["embeddings"].shape[-1]),
        "representation_shape": caches["train"]["representation_shape"],
        "hidden_state_indices": caches["train"]["hidden_state_indices"],
        "subset_fingerprint": subset_fingerprint,
        "label_to_id": label_to_id,
        "split_sizes": {split: len(caches[split]["labels"]) for split in SPLITS},
    }
    (out_dir / "embedding_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Embedding configuration: {out_dir / 'embedding_config.json'}")


if __name__ == "__main__":
    main()
