"""Create deterministic multi-label mixture manifests from MedleyDB stems."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.audio_io import load_audio_segment, rms_dbfs
from src.data.metadata_validation import validate_stem_index_columns
from src.utils.paths import ensure_parent, load_yaml, resolve_data_path, resolve_run_path

SPLITS = ("train", "val", "test")
MODE_ALIASES = {
    "synthetic_k": "synthetic_random",
    "synthetic_random": "synthetic_random",
    "same_song_same_time": "same_song_reconstructed",
    "same_song_reconstructed": "same_song_reconstructed",
    "full_mix": "original_full_mix",
    "original_full_mix": "original_full_mix",
}
MODES = tuple(sorted(MODE_ALIASES))
CANONICAL_MODES = tuple(sorted(set(MODE_ALIASES.values())))
DEFAULT_ACTIVITY_THRESHOLD_DBFS = -50.0


def _canonical_mode(value: str) -> str:
    try:
        return MODE_ALIASES[str(value)]
    except KeyError as exc:
        raise ValueError(f"mode must be one of {MODES}") from exc


def _label_column(frame: pd.DataFrame) -> str:
    if "label_name" in frame.columns:
        return "label_name"
    if "medleydb_instrument_label" in frame.columns:
        return "medleydb_instrument_label"
    if "coarse_label" in frame.columns:
        return "coarse_label"
    raise ValueError("Metadata must contain label_name, medleydb_instrument_label, or coarse_label")


def _stem_label_column(label_granularity: str) -> str:
    if label_granularity == "medleydb_instrument":
        return "medleydb_instrument_label"
    if label_granularity == "coarse_family":
        return "coarse_label"
    raise ValueError("label_granularity must be medleydb_instrument or coarse_family")


def _pipe(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def _config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mixture_id(*parts: Any) -> str:
    payload = "||".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _read_stem_index(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(
            f"Stem index not found: {path}. Rebuild generated metadata under RUN_ROOT "
            "before creating mixture manifests."
        )
    index = pd.read_csv(path)
    validate_stem_index_columns(index, source=path)
    return index


def _load_subset_with_genre(subset_csv: Path, stem_index: pd.DataFrame | None) -> pd.DataFrame:
    subset = pd.read_csv(subset_csv)
    if "genre" in subset.columns:
        return subset
    if stem_index is None or "genre" not in stem_index.columns:
        subset["genre"] = ""
        return subset
    genre = stem_index[["track_id", "stem_id", "genre"]].drop_duplicates(["track_id", "stem_id"])
    return subset.merge(genre, on=["track_id", "stem_id"], how="left")


def _track_split_map(subset: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for track_id, values in subset.groupby("track_id")["split"]:
        splits = sorted(str(value) for value in values.dropna().unique())
        if len(splits) != 1:
            raise ValueError(f"Track {track_id} appears in multiple splits: {splits}")
        mapping[str(track_id)] = splits[0]
    return mapping


def _target_vector(labels: list[str], label_to_id: dict[str, int]) -> str:
    vector = [0] * len(label_to_id)
    for label in labels:
        vector[int(label_to_id[label])] = 1
    return " ".join(str(value) for value in vector)


def _make_row(
    *,
    dataset_id: str,
    mode: str,
    split: str,
    source_rows: pd.DataFrame,
    label_column: str,
    label_to_id: dict[str, int],
    seed: int,
    audio_path: str = "",
    activity_rule: str = "manifest_labels",
    activity_threshold_dbfs: float | None = None,
    full_mix_exists: bool | None = None,
) -> dict[str, Any]:
    active_labels = sorted({str(value) for value in source_rows[label_column].tolist()})
    track_ids = sorted({str(value) for value in source_rows["track_id"].tolist()})
    start = float(source_rows["start_seconds"].iloc[0])
    duration = float(source_rows["duration_seconds"].iloc[0])
    source_activity = (
        _pipe([round(float(value), 4) for value in source_rows["activity_dbfs"].tolist()])
        if "activity_dbfs" in source_rows.columns
        else ""
    )
    mixture_id = _mixture_id(
        dataset_id,
        mode,
        split,
        _pipe(source_rows["segment_id"].astype(str).tolist()),
        audio_path,
        seed,
    )
    return {
        "mixture_id": mixture_id,
        "mixture_dataset_id": dataset_id,
        "mode": mode,
        "split": split,
        "track_id": track_ids[0] if len(track_ids) == 1 else "",
        "source_track_ids": _pipe(track_ids),
        "source_segment_ids": _pipe(source_rows["segment_id"].astype(str).tolist()),
        "source_audio_paths": _pipe(source_rows["audio_path"].astype(str).tolist()),
        "source_start_seconds": _pipe([float(value) for value in source_rows["start_seconds"].tolist()]),
        "audio_path": audio_path,
        "start_seconds": start,
        "duration_seconds": duration,
        "active_labels": _pipe(active_labels),
        "label_ids": _pipe([label_to_id[label] for label in active_labels]),
        "target_vector": _target_vector(active_labels, label_to_id),
        "k_active": len(active_labels),
        "genre": _pipe(sorted({str(value) for value in source_rows.get("genre", pd.Series([""])).fillna("").tolist() if str(value)})),
        "seed": int(seed),
        "activity_rule": activity_rule,
        "activity_threshold_dbfs": "" if activity_threshold_dbfs is None else float(activity_threshold_dbfs),
        "source_activity_dbfs": source_activity,
        "full_mix_exists": "" if full_mix_exists is None else bool(full_mix_exists),
    }


def _infer_mix_path(medleydb_root: Path, track_id: str) -> tuple[str, bool]:
    candidates = [
        medleydb_root / "Audio" / track_id / f"{track_id}_MIX.wav",
        medleydb_root / track_id / f"{track_id}_MIX.wav",
        medleydb_root / f"{track_id}_MIX.wav",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return candidate.resolve().relative_to(medleydb_root.resolve()).as_posix(), True
            except ValueError:
                return candidate.as_posix(), True
    return candidates[0].relative_to(medleydb_root).as_posix(), False


def build_synthetic_k(
    subset: pd.DataFrame,
    *,
    dataset_id: str,
    label_column: str,
    label_to_id: dict[str, int],
    k_values: list[int],
    mixtures_per_split_per_k: int,
    seed: int,
    mode: str = "synthetic_random",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_frame = subset[subset["split"] == split].copy()
        if split_frame.empty:
            continue
        labels = sorted(split_frame[label_column].astype(str).unique().tolist())
        for k in k_values:
            if len(labels) < k:
                continue
            for index in range(mixtures_per_split_per_k):
                rng = pd.Series(labels).sample(n=k, random_state=seed + index + 1009 * k + 7919 * SPLITS.index(split)).tolist()
                parts = []
                for offset, label in enumerate(rng):
                    label_rows = split_frame[split_frame[label_column].astype(str) == label]
                    parts.append(label_rows.sample(n=1, random_state=seed + index + offset + 101 * k))
                source_rows = pd.concat(parts, ignore_index=True)
                rows.append(
                    _make_row(
                        dataset_id=dataset_id,
                        mode=mode,
                        split=split,
                        source_rows=source_rows,
                        label_column=label_column,
                        label_to_id=label_to_id,
                        seed=seed,
                        activity_rule="manifest_labels",
                    )
                )
    return pd.DataFrame(rows)


def _candidate_stems_for_window(
    stem_index: pd.DataFrame,
    *,
    track_id: str,
    split: str,
    start_seconds: float,
    duration_seconds: float,
    label_granularity: str,
    label_column: str,
    label_to_id: dict[str, int],
    medleydb_root: Path,
    activity_threshold_dbfs: float,
    allow_bleed: bool,
) -> pd.DataFrame:
    label_source = _stem_label_column(label_granularity)
    frame = stem_index[stem_index["track_id"].astype(str) == str(track_id)].copy()
    if frame.empty:
        return frame
    valid = frame["valid"].fillna(False)
    if not pd.api.types.is_bool_dtype(valid):
        valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
    known = frame[label_source].fillna("").astype(str).isin(label_to_id)
    long_enough = frame["duration_seconds"].astype(float) >= start_seconds + duration_seconds
    if allow_bleed or "has_bleed" not in frame.columns:
        no_bleed = pd.Series([True] * len(frame), index=frame.index)
    else:
        bleed = frame["has_bleed"].fillna(False)
        if not pd.api.types.is_bool_dtype(bleed):
            bleed = bleed.astype(str).str.lower().isin({"true", "1", "yes"})
        no_bleed = ~bleed
    rows = []
    for row in frame[valid & known & long_enough & no_bleed].itertuples(index=False):
        try:
            waveform, _ = load_audio_segment(
                resolve_data_path(medleydb_root, str(row.audio_path)),
                start_seconds,
                duration_seconds,
                normalize=False,
                pad=False,
            )
            activity = rms_dbfs(waveform)
        except Exception:
            continue
        if activity < activity_threshold_dbfs:
            continue
        values = row._asdict()
        values["segment_id"] = f"{track_id}__{values['stem_id']}__{int(round(start_seconds * 1000)):09d}"
        values["split"] = split
        values["start_seconds"] = float(start_seconds)
        values["duration_seconds"] = float(duration_seconds)
        values[label_column] = str(values[label_source])
        values["activity_dbfs"] = float(activity)
        rows.append(values)
    return pd.DataFrame(rows)


def build_same_song_or_full_mix(
    subset: pd.DataFrame,
    *,
    stem_index: pd.DataFrame | None,
    dataset_id: str,
    mode: str,
    label_granularity: str,
    label_column: str,
    label_to_id: dict[str, int],
    seed: int,
    medleydb_root: Path,
    activity_threshold_dbfs: float,
    allow_bleed: bool,
    min_active_labels: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["split", "track_id", "start_seconds", "duration_seconds"]
    for _, group in subset.groupby(group_cols, dropna=False):
        split = str(group["split"].iloc[0])
        track_id = str(group["track_id"].iloc[0])
        start = float(group["start_seconds"].iloc[0])
        duration = float(group["duration_seconds"].iloc[0])
        if stem_index is not None:
            source_rows = _candidate_stems_for_window(
                stem_index,
                track_id=track_id,
                split=split,
                start_seconds=start,
                duration_seconds=duration,
                label_granularity=label_granularity,
                label_column=label_column,
                label_to_id=label_to_id,
                medleydb_root=medleydb_root,
                activity_threshold_dbfs=activity_threshold_dbfs,
                allow_bleed=allow_bleed,
            )
            activity_rule = "stem_rms_dbfs"
            threshold: float | None = activity_threshold_dbfs
        else:
            source_rows = group.reset_index(drop=True)
            activity_rule = "manifest_labels"
            threshold = None
        active = sorted({str(value) for value in source_rows.get(label_column, pd.Series(dtype=str)).tolist()})
        if len(active) < min_active_labels:
            continue
        full_mix_exists: bool | None = None
        audio_path = ""
        if mode == "original_full_mix":
            audio_path, full_mix_exists = _infer_mix_path(medleydb_root, track_id)
            if not full_mix_exists:
                warnings.warn(
                    f"Skipping original_full_mix row for {track_id} at {start:.3f}s: "
                    f"full mix file was not found below {medleydb_root}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
        rows.append(
            _make_row(
                dataset_id=dataset_id,
                mode=mode,
                split=split,
                source_rows=source_rows.reset_index(drop=True),
                label_column=label_column,
                label_to_id=label_to_id,
                seed=seed,
                audio_path=audio_path,
                activity_rule=activity_rule,
                activity_threshold_dbfs=threshold,
                full_mix_exists=full_mix_exists,
            )
        )
    return pd.DataFrame(rows)


def build_mixture_manifest(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a mixture manifest and label mapping from a config dictionary."""

    dataset_id = str(config["mixture_dataset_id"])
    mode = _canonical_mode(str(config["mode"]))
    stem_index = _read_stem_index(resolve_run_path(config["stem_index_csv"])) if config.get("stem_index_csv") else None
    subset = _load_subset_with_genre(resolve_run_path(config["subset_csv"]), stem_index)
    required = {"segment_id", "track_id", "audio_path", "start_seconds", "duration_seconds", "split"}
    missing = sorted(required - set(subset.columns))
    if missing:
        raise ValueError(f"Subset metadata is missing columns: {', '.join(missing)}")
    _track_split_map(subset)
    label_column = _label_column(subset)
    label_names = sorted(str(value) for value in subset[label_column].dropna().unique())
    label_to_id = {label: index for index, label in enumerate(label_names)}
    seed = int(config.get("seed", 42))
    if mode == "synthetic_random":
        manifest = build_synthetic_k(
            subset,
            dataset_id=dataset_id,
            label_column=label_column,
            label_to_id=label_to_id,
            k_values=[int(value) for value in config.get("k_values", [1, 2, 3])],
            mixtures_per_split_per_k=int(config.get("mixtures_per_split_per_k", 8)),
            seed=seed,
            mode=mode,
        )
    else:
        manifest = build_same_song_or_full_mix(
            subset,
            stem_index=stem_index,
            dataset_id=dataset_id,
            mode=mode,
            label_granularity=str(config.get("label_granularity", "medleydb_instrument")),
            label_column=label_column,
            label_to_id=label_to_id,
            seed=seed,
            medleydb_root=Path(os.environ.get("MEDLEYDB_ROOT", config.get("medleydb_root", "MedleyDB"))),
            activity_threshold_dbfs=float(config.get("activity_threshold_dbfs", DEFAULT_ACTIVITY_THRESHOLD_DBFS)),
            allow_bleed=bool(config.get("allow_bleed", False)),
            min_active_labels=int(config.get("min_active_labels", 1 if mode == "original_full_mix" else 2)),
        )
    columns = [
        "mixture_id", "mixture_dataset_id", "mode", "split", "track_id",
        "source_track_ids", "source_segment_ids", "source_audio_paths",
        "source_start_seconds", "audio_path", "start_seconds", "duration_seconds",
        "active_labels", "label_ids", "target_vector", "k_active", "genre", "seed",
        "activity_rule", "activity_threshold_dbfs", "source_activity_dbfs",
        "full_mix_exists",
    ]
    if manifest.empty:
        manifest = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in manifest.columns:
            manifest[column] = ""
    manifest["config_fingerprint"] = _config_fingerprint({**config, "mode": mode})
    return manifest[columns + ["config_fingerprint"]].sort_values(["split", "mixture_id"]).reset_index(drop=True), label_to_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    manifest, label_to_id = build_mixture_manifest(config)
    out = resolve_run_path(config["out"])
    ensure_parent(out)
    manifest.to_csv(out, index=False)
    label_path = resolve_run_path(config["label_to_id"]) if config.get("label_to_id") else out.with_name(out.stem + "_label_to_id.json")
    ensure_parent(label_path)
    label_path.write_text(json.dumps(label_to_id, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {len(manifest)} mixtures: {out}")
    print(f"Labels: {label_path}")


if __name__ == "__main__":
    main()
