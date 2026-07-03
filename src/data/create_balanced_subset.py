"""Create a small, balanced, track-disjoint subset of MedleyDB stem segments."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.audio_io import load_audio_segment, rms_dbfs
from src.data.label_mapping import OTHER_LABEL
from src.data.metadata_validation import validate_stem_index_columns
from src.utils.paths import ensure_directory, ensure_parent, load_yaml, resolve_data_path, resolve_run_path


SPLITS = ("train", "val", "test")
LABEL_GRANULARITIES = ("coarse_family", "medleydb_instrument")
SUBSET_PROFILES = (
    "debug",
    "largest_balanced",
    "subset_40_per_class",
    "capped_natural",
)


def label_column_for_granularity(granularity: str) -> str:
    """Return the stem-index label column used by a subset protocol."""

    if granularity == "coarse_family":
        return "coarse_label"
    if granularity == "medleydb_instrument":
        return "medleydb_instrument_label"
    raise ValueError("label_granularity must be 'coarse_family' or 'medleydb_instrument'")


@dataclass
class SubsetDiagnostics:
    """Mutable counters used to produce a transparent subset report."""

    silence_rejections: Counter[str] = field(default_factory=Counter)
    decode_rejections: Counter[str] = field(default_factory=Counter)
    decode_error_examples: list[str] = field(default_factory=list)
    excluded_classes: dict[str, str] = field(default_factory=dict)
    attempted_classes: list[str] = field(default_factory=list)
    split_search_score: float | None = None


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def allocate_split_counts(
    total: int, val_ratio: float, test_ratio: float
) -> dict[str, int]:
    """Allocate an integer total with largest-remainder rounding."""

    train_ratio = 1.0 - val_ratio - test_ratio
    if total <= 0:
        raise ValueError("total must be positive")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train, validation, and test ratios must all be positive")
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    raw = {name: total * ratio for name, ratio in ratios.items()}
    result = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(SPLITS, key=lambda name: (-(raw[name] - result[name]), SPLITS.index(name)))
    for name in order[:remainder]:
        result[name] += 1
    if total >= 3:
        for name in SPLITS:
            if result[name] == 0:
                donor = max(SPLITS, key=lambda key: result[key])
                result[donor] -= 1
                result[name] += 1
    return result


def enumerate_segment_candidates(
    index: pd.DataFrame,
    *,
    segment_seconds: float,
    hop_seconds: float,
    allow_bleed: bool,
    label_granularity: str = "coarse_family",
    subset_profile: str = "debug",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expand valid stem rows into full-length candidate windows."""

    label_column = label_column_for_granularity(label_granularity)
    required = {
        "track_id",
        "stem_id",
        "audio_path",
        "raw_instrument_label",
        label_column,
        "duration_seconds",
        "valid",
        "has_bleed",
    }
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Index is missing required columns: {', '.join(missing)}")
    if segment_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("segment_seconds and hop_seconds must be positive")

    valid = _boolean_series(index["valid"])
    known = index[label_column].fillna(OTHER_LABEL).astype(str) != OTHER_LABEL
    bleed = _boolean_series(index["has_bleed"])
    usable = valid & known & (True if allow_bleed else ~bleed)
    stats = {
        "indexed_files": len(index),
        "valid_files": int(valid.sum()),
        "invalid_files": int((~valid).sum()),
        "unknown_label_files": int((valid & ~known).sum()),
        "bleed_files_excluded": int((valid & known & bleed).sum()) if not allow_bleed else 0,
        "usable_files": int(usable.sum()),
    }

    rows: list[dict[str, Any]] = []
    for row in index.loc[usable].itertuples(index=False):
        duration = float(row.duration_seconds)
        start = 0.0
        ordinal = 0
        while start + segment_seconds <= duration + 1e-9:
            start_milliseconds = int(round(start * 1000.0))
            segment_id = (
                f"{row.track_id}__{row.stem_id}__{start_milliseconds:09d}"
            )
            rows.append(
                {
                    "segment_id": segment_id,
                    "track_id": str(row.track_id),
                    "stem_id": str(row.stem_id),
                    "audio_path": str(row.audio_path),
                    "start_seconds": round(start, 6),
                    "duration_seconds": float(segment_seconds),
                    "raw_instrument_label": str(row.raw_instrument_label),
                    "coarse_family_label": str(getattr(row, "coarse_label", "")),
                    # Downstream code historically reads `coarse_label` as the
                    # active class name.  Keep that column as the selected
                    # protocol label for backward compatibility.
                    "coarse_label": str(getattr(row, label_column)),
                    "medleydb_instrument_label": str(
                        getattr(row, "medleydb_instrument_label", "")
                    ),
                    "label_name": str(getattr(row, label_column)),
                    "label_granularity": label_granularity,
                    "subset_profile": subset_profile,
                    "candidate_ordinal": ordinal,
                }
            )
            start += hop_seconds
            ordinal += 1
    return pd.DataFrame(rows), stats


def rank_eligible_classes(
    candidates: pd.DataFrame,
    *,
    min_segments_per_class: int,
    min_tracks_per_class: int = 3,
) -> pd.DataFrame:
    """Rank class families by track diversity, then segment capacity."""

    columns = ["coarse_label", "track_count", "candidate_count", "eligible", "reason"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        candidates.groupby("coarse_label")
        .agg(track_count=("track_id", "nunique"), candidate_count=("segment_id", "size"))
        .reset_index()
    )
    reasons: list[str] = []
    eligible: list[bool] = []
    for row in summary.itertuples(index=False):
        if row.track_count < min_tracks_per_class:
            eligible.append(False)
            reasons.append(f"fewer than {min_tracks_per_class} contributing tracks")
        elif row.candidate_count < min_segments_per_class:
            eligible.append(False)
            reasons.append(f"fewer than {min_segments_per_class} candidate segments")
        else:
            eligible.append(True)
            reasons.append("")
    summary["eligible"] = eligible
    summary["reason"] = reasons
    return summary.sort_values(
        ["eligible", "track_count", "candidate_count", "coarse_label"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def assign_track_splits(
    candidates: pd.DataFrame,
    selected_classes: list[str],
    *,
    val_ratio: float,
    test_ratio: float,
    target_per_class: int,
    seed: int,
    max_attempts: int = 1000,
) -> tuple[dict[str, str], float]:
    """Search deterministic group assignments with good class coverage."""

    selected = candidates[candidates["coarse_label"].isin(selected_classes)]
    tracks = sorted(selected["track_id"].unique().tolist())
    if len(tracks) < 3:
        raise ValueError("At least three tracks are required for a three-way split")
    track_targets = allocate_split_counts(len(tracks), val_ratio, test_ratio)
    segment_targets = allocate_split_counts(target_per_class, val_ratio, test_ratio)
    best_mapping: dict[str, str] | None = None
    best_score = float("inf")

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        shuffled = list(rng.permutation(tracks))
        train_end = track_targets["train"]
        val_end = train_end + track_targets["val"]
        mapping = {
            track: (
                "train" if index < train_end else "val" if index < val_end else "test"
            )
            for index, track in enumerate(shuffled)
        }
        split_values = selected["track_id"].map(mapping)
        counts = (
            selected.assign(split=split_values)
            .groupby(["coarse_label", "split"])
            .size()
            .to_dict()
        )
        missing = sum(
            1
            for label in selected_classes
            for split in SPLITS
            if counts.get((label, split), 0) == 0
        )
        if missing:
            continue
        deficit = 0.0
        distribution_error = 0.0
        for label in selected_classes:
            class_total = sum(counts.get((label, split), 0) for split in SPLITS)
            for split in SPLITS:
                count = counts.get((label, split), 0)
                desired = segment_targets[split]
                deficit += max(0, desired - count) ** 2
                distribution_error += abs(count / class_total - track_targets[split] / len(tracks))
        score = deficit * 1000.0 + distribution_error
        if score < best_score:
            best_score = score
            best_mapping = mapping
            if deficit == 0 and distribution_error < 0.05 * len(selected_classes):
                break
    if best_mapping is None:
        raise ValueError(
            "Could not create a track-level split containing every selected class in all splits"
        )
    return best_mapping, best_score


def _round_robin_indices(pool: pd.DataFrame, rng: np.random.Generator) -> list[int]:
    per_track: dict[str, deque[int]] = {}
    for track_id, group in pool.groupby("track_id", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        per_track[str(track_id)] = deque(int(value) for value in indices)
    track_order = list(per_track)
    rng.shuffle(track_order)
    output: list[int] = []
    while track_order:
        next_round: list[str] = []
        for track_id in track_order:
            queue = per_track[track_id]
            if queue:
                output.append(queue.popleft())
            if queue:
                next_round.append(track_id)
        rng.shuffle(next_round)
        track_order = next_round
    return output


def scan_active_candidates(
    candidates: pd.DataFrame,
    medleydb_root: Path,
    *,
    maximum_counts: dict[str, int],
    silence_threshold_dbfs: float,
    seed: int,
    diagnostics: SubsetDiagnostics,
) -> pd.DataFrame:
    """Decode candidates until each class/split pool reaches its maximum quota."""

    accepted: list[dict[str, Any]] = []
    groups = list(candidates.groupby(["coarse_label", "split"], sort=True))
    for group_number, ((label, split), pool) in enumerate(
        tqdm(groups, desc="Checking segment activity")
    ):
        target = maximum_counts[split]
        rng = np.random.default_rng(seed + group_number * 1009)
        accepted_count = 0
        for index in _round_robin_indices(pool, rng):
            row = candidates.loc[index]
            try:
                audio_path = resolve_data_path(medleydb_root, row["audio_path"])
                waveform, _ = load_audio_segment(
                    audio_path,
                    float(row["start_seconds"]),
                    float(row["duration_seconds"]),
                    normalize=False,
                    pad=False,
                )
                level = rms_dbfs(waveform)
            except Exception as exc:
                diagnostics.decode_rejections[str(label)] += 1
                if len(diagnostics.decode_error_examples) < 20:
                    diagnostics.decode_error_examples.append(
                        f"{row['segment_id']}: {type(exc).__name__}: {exc}"
                    )
                continue
            if level < silence_threshold_dbfs:
                diagnostics.silence_rejections[str(label)] += 1
                continue
            result = row.to_dict()
            result["rms_dbfs"] = round(level, 4)
            accepted.append(result)
            accepted_count += 1
            if accepted_count >= target:
                break
    return pd.DataFrame(accepted)


def largest_feasible_total(
    active: pd.DataFrame,
    selected_classes: list[str],
    *,
    minimum: int,
    maximum: int,
    val_ratio: float,
    test_ratio: float,
) -> int | None:
    """Return the largest common class size supported in every split."""

    counts = active.groupby(["coarse_label", "split"]).size().to_dict()
    for total in range(maximum, minimum - 1, -1):
        quotas = allocate_split_counts(total, val_ratio, test_ratio)
        if all(
            counts.get((label, split), 0) >= quotas[split]
            for label in selected_classes
            for split in SPLITS
        ):
            return total
    return None


def choose_balanced_rows(
    active: pd.DataFrame,
    selected_classes: list[str],
    *,
    total_per_class: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Select exact, deterministic quotas from already active candidates."""

    quotas = allocate_split_counts(total_per_class, val_ratio, test_ratio)
    chunks: list[pd.DataFrame] = []
    for class_index, label in enumerate(sorted(selected_classes)):
        for split_index, split in enumerate(SPLITS):
            pool = active[
                (active["coarse_label"] == label) & (active["split"] == split)
            ]
            count = quotas[split]
            if len(pool) < count:
                raise ValueError(f"Insufficient active segments for {label}/{split}")
            chunks.append(
                pool.sample(
                    n=count,
                    random_state=seed + class_index * 101 + split_index,
                    replace=False,
                )
            )
    return pd.concat(chunks, ignore_index=True)


def choose_capped_natural_rows(
    active: pd.DataFrame,
    selected_classes: list[str],
    *,
    minimum_per_class: int,
    total_budget: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Guarantee minimum coverage, then fill a global budget without class balancing."""

    minimum_quotas = allocate_split_counts(minimum_per_class, val_ratio, test_ratio)
    split_budgets = allocate_split_counts(total_budget, val_ratio, test_ratio)
    chosen_indices: set[int] = set()
    chunks: list[pd.DataFrame] = []
    for class_index, label in enumerate(sorted(selected_classes)):
        for split_index, split in enumerate(SPLITS):
            pool = active[(active["coarse_label"] == label) & (active["split"] == split)]
            count = minimum_quotas[split]
            if len(pool) < count:
                raise ValueError(f"Insufficient active segments for {label}/{split}")
            reserved = pool.sample(
                n=count, random_state=seed + class_index * 101 + split_index, replace=False
            )
            chunks.append(reserved)
            chosen_indices.update(int(index) for index in reserved.index)

    reserved_frame = pd.concat(chunks)
    fill_chunks = [reserved_frame]
    for split_index, split in enumerate(SPLITS):
        already = int((reserved_frame["split"] == split).sum())
        remaining_count = max(0, split_budgets[split] - already)
        pool = active[(active["split"] == split) & (~active.index.isin(chosen_indices))]
        order = _round_robin_indices(pool, np.random.default_rng(seed + 5003 + split_index))
        if order and remaining_count:
            fill_chunks.append(active.loc[order[:remaining_count]])
    return pd.concat(fill_chunks, ignore_index=True).drop_duplicates("segment_id")


def _write_json(path: Path, value: Any) -> None:
    ensure_parent(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_subset_report(
    report_path: Path,
    *,
    file_stats: dict[str, int],
    ranking: pd.DataFrame,
    selected: pd.DataFrame,
    diagnostics: SubsetDiagnostics,
    allow_bleed: bool,
    segment_seconds: float,
    hop_seconds: float,
    silence_threshold_dbfs: float,
    sampling_strategy: str,
    label_granularity: str,
    subset_profile: str,
    min_tracks_per_class: int,
    max_classes: int,
    max_segments_per_class: int,
    total_budget: int | None = None,
) -> None:
    """Write subset provenance and leakage checks."""

    lines = [
        "# MedleyDB subset report",
        "",
        "## Input filtering",
        "",
        f"- Indexed stem rows: {file_stats['indexed_files']}",
        f"- Valid stem rows: {file_stats['valid_files']}",
        f"- Invalid stem rows: {file_stats['invalid_files']}",
        f"- Valid rows with unknown labels: {file_stats['unknown_label_files']}",
        f"- Bleed-labelled rows excluded: {file_stats['bleed_files_excluded']}",
        f"- Usable stem rows: {file_stats['usable_files']}",
        f"- Bleed policy: {'allowed' if allow_bleed else 'excluded'}",
        f"- Segment/hop duration: {segment_seconds:g}s / {hop_seconds:g}s",
        f"- Silence threshold: {silence_threshold_dbfs:g} dBFS segment RMS",
        f"- Sampling strategy: `{sampling_strategy}`",
        f"- Subset profile: `{subset_profile}`",
        f"- Label granularity: `{label_granularity}`",
        f"- Minimum contributing tracks per class: {min_tracks_per_class}",
        (
            f"- Global capped-natural budget: {int(total_budget or max_classes * max_segments_per_class)} segments"
            if sampling_strategy == "capped_natural"
            else f"- Maximum segments per class: {max_segments_per_class}"
        ),
        "",
        "## Class ranking",
        "",
        "| Rank | Class | Tracks | Candidate segments | Eligible | Exclusion reason |",
        "|---:|---|---:|---:|---|---|",
    ]
    for rank, row in enumerate(ranking.itertuples(index=False), start=1):
        reason = diagnostics.excluded_classes.get(row.coarse_label, row.reason)
        lines.append(
            f"| {rank} | {row.coarse_label} | {row.track_count} | {row.candidate_count} | "
            f"{'yes' if row.eligible else 'no'} | {reason} |"
        )
    lines.extend(["", "## Selected subset", ""])
    if selected.empty:
        lines.append("No viable balanced subset could be created.")
    else:
        counts = (
            selected.groupby(["coarse_label", "split"]).size().unstack(fill_value=0)
        )
        lines.extend(
            [
                "| Class | Train | Validation | Test | Total |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, row in counts.sort_index().iterrows():
            train = int(row.get("train", 0))
            val = int(row.get("val", 0))
            test = int(row.get("test", 0))
            lines.append(f"| {label} | {train} | {val} | {test} | {train + val + test} |")
        lines.extend(["", "## Track-level split check", ""])
        track_sets = {
            split: set(selected.loc[selected["split"] == split, "track_id"].astype(str))
            for split in SPLITS
        }
        lines.extend(
            [
                f"- Train tracks: {len(track_sets['train'])}",
                f"- Validation tracks: {len(track_sets['val'])}",
                f"- Test tracks: {len(track_sets['test'])}",
                f"- Train/validation overlap: {len(track_sets['train'] & track_sets['val'])}",
                f"- Train/test overlap: {len(track_sets['train'] & track_sets['test'])}",
                f"- Validation/test overlap: {len(track_sets['val'] & track_sets['test'])}",
                "",
                "Every segment from a track is assigned to exactly one split.",
            ]
        )
    lines.extend(
        [
            "",
            "## Rejections during audio checks",
            "",
            f"- Mostly silent by class: `{dict(diagnostics.silence_rejections)}`",
            f"- Decode failures by class: `{dict(diagnostics.decode_rejections)}`",
            f"- Decode error examples: `{diagnostics.decode_error_examples}`",
            f"- Track split search score: `{diagnostics.split_search_score}`",
            "",
        ]
    )
    ensure_parent(report_path).write_text("\n".join(lines), encoding="utf-8")


def write_label_mapping_report(
    report_path: Path,
    index: pd.DataFrame,
    *,
    segment_seconds: float,
    hop_seconds: float,
    allow_bleed: bool,
) -> None:
    """Write a compact audit of raw labels and both supported target protocols.

    Candidate segment counts are computed from valid, labelled stems using the
    same complete-window rule as subset creation.  This report is intentionally
    descriptive: it helps verify that ``medleydb_instrument`` remains close to
    the original MedleyDB labels while ``coarse_family`` is only the controlled
    family-level protocol.
    """

    required = {
        "raw_instrument_label",
        "coarse_label",
        "medleydb_instrument_label",
        "track_id",
        "duration_seconds",
        "valid",
        "has_bleed",
    }
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Index is missing required columns for label report: {', '.join(missing)}")

    valid = _boolean_series(index["valid"])
    known = index["medleydb_instrument_label"].fillna(OTHER_LABEL).astype(str) != OTHER_LABEL
    bleed = _boolean_series(index["has_bleed"])
    usable = valid & known & (True if allow_bleed else ~bleed)
    frame = index.loc[usable].copy()

    def candidate_count(duration: Any) -> int:
        try:
            value = float(duration)
        except (TypeError, ValueError):
            return 0
        if value + 1e-9 < segment_seconds:
            return 0
        return int(math.floor((value - segment_seconds) / hop_seconds) + 1)

    frame["candidate_segments"] = frame["duration_seconds"].map(candidate_count)
    summary = (
        frame.groupby(["raw_instrument_label", "medleydb_instrument_label", "coarse_label"], dropna=False)
        .agg(
            stems=("raw_instrument_label", "size"),
            tracks=("track_id", "nunique"),
            candidate_segments=("candidate_segments", "sum"),
        )
        .reset_index()
        .sort_values(["candidate_segments", "tracks", "raw_instrument_label"], ascending=[False, False, True])
    )

    lines = [
        "# Label mapping report",
        "",
        "This report compares the original MedleyDB stem labels with the two label protocols used in this project.",
        "",
        "- `medleydb_instrument` keeps the MedleyDB label with light spelling/format normalization only.",
        "- `coarse_family` maps labels to broader instrument families for controlled debugging and comparison.",
        f"- Bleed policy for candidate counts: {'allowed' if allow_bleed else 'excluded'}.",
        f"- Candidate window rule: complete {segment_seconds:g}s windows every {hop_seconds:g}s.",
        "",
        "| Raw label | MedleyDB instrument label | Coarse family label | Stems | Tracks | Candidate segments |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.raw_instrument_label} | {row.medleydb_instrument_label} | "
            f"{row.coarse_label} | {int(row.stems)} | {int(row.tracks)} | "
            f"{int(row.candidate_segments)} |"
        )
    ensure_parent(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_subset(
    index: pd.DataFrame,
    medleydb_root: Path,
    *,
    segment_seconds: float,
    hop_seconds: float,
    max_classes: int,
    max_segments_per_class: int,
    min_segments_per_class: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    allow_bleed: bool,
    silence_threshold_dbfs: float,
    split_search_attempts: int,
    sampling_strategy: str = "balanced",
    label_granularity: str = "coarse_family",
    subset_profile: str = "debug",
    min_tracks_per_class: int = 3,
    total_budget: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SubsetDiagnostics, dict[str, int]]:
    """Construct a balanced or capped-natural subset plus provenance objects."""

    if sampling_strategy not in {"balanced", "capped_natural"}:
        raise ValueError("sampling_strategy must be 'balanced' or 'capped_natural'")

    candidates, file_stats = enumerate_segment_candidates(
        index,
        segment_seconds=segment_seconds,
        hop_seconds=hop_seconds,
        allow_bleed=allow_bleed,
        label_granularity=label_granularity,
        subset_profile=subset_profile,
    )
    ranking = rank_eligible_classes(
        candidates,
        min_segments_per_class=min_segments_per_class,
        min_tracks_per_class=min_tracks_per_class,
    )
    eligible = ranking.loc[ranking["eligible"], "coarse_label"].astype(str).tolist()
    diagnostics = SubsetDiagnostics()
    for row in ranking.loc[~ranking["eligible"]].itertuples(index=False):
        diagnostics.excluded_classes[row.coarse_label] = row.reason

    available = eligible.copy()
    selected_classes = available[:max_classes]
    reserve = available[max_classes:]
    final_active = pd.DataFrame()
    total_per_class: int | None = None

    while len(selected_classes) >= 2:
        diagnostics.attempted_classes.extend(
            label for label in selected_classes if label not in diagnostics.attempted_classes
        )
        try:
            mapping, score = assign_track_splits(
                candidates,
                selected_classes,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                target_per_class=max_segments_per_class,
                seed=seed,
                max_attempts=split_search_attempts,
            )
        except ValueError as exc:
            failed_label = selected_classes.pop()
            diagnostics.excluded_classes[failed_label] = str(exc)
            if reserve:
                selected_classes.append(reserve.pop(0))
            continue
        diagnostics.split_search_score = score
        working = candidates[candidates["coarse_label"].isin(selected_classes)].copy()
        working["split"] = working["track_id"].map(mapping)
        capped_budget = int(total_budget or max_classes * max_segments_per_class)
        maximum_counts = allocate_split_counts(
            max_segments_per_class if sampling_strategy == "balanced" else capped_budget,
            val_ratio,
            test_ratio,
        )
        active = scan_active_candidates(
            working,
            medleydb_root,
            maximum_counts=maximum_counts,
            silence_threshold_dbfs=silence_threshold_dbfs,
            seed=seed,
            diagnostics=diagnostics,
        )
        feasible_total = largest_feasible_total(
            active,
            selected_classes,
            minimum=min_segments_per_class,
            maximum=(
                max_segments_per_class
                if sampling_strategy == "balanced"
                else min_segments_per_class
            ),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        total_per_class = feasible_total
        if feasible_total is not None:
            final_active = active
            break

        minimum_counts = allocate_split_counts(
            min_segments_per_class, val_ratio, test_ratio
        )
        active_counts = active.groupby(["coarse_label", "split"]).size().to_dict()
        failing = [
            label
            for label in selected_classes
            if any(
                active_counts.get((label, split), 0) < minimum_counts[split]
                for split in SPLITS
            )
        ]
        if not failing:
            failing = [selected_classes[-1]]
        for label in failing:
            diagnostics.excluded_classes[label] = (
                "insufficient non-silent segments in at least one track-level split"
            )
            selected_classes.remove(label)
            if reserve:
                selected_classes.append(reserve.pop(0))

    if len(selected_classes) < 2 or total_per_class is None:
        return pd.DataFrame(), ranking, diagnostics, file_stats

    for label in eligible:
        if label not in selected_classes and label not in diagnostics.excluded_classes:
            diagnostics.excluded_classes[label] = (
                "eligible but outside --max-classes after track-diversity ranking"
            )

    if sampling_strategy == "balanced":
        selected = choose_balanced_rows(
            final_active,
            selected_classes,
            total_per_class=total_per_class,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        selected = choose_capped_natural_rows(
            final_active,
            selected_classes,
            minimum_per_class=min_segments_per_class,
            total_budget=int(total_budget or max_classes * max_segments_per_class),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    label_to_id = {label: index for index, label in enumerate(sorted(selected_classes))}
    selected["label_id"] = selected["coarse_label"].map(label_to_id).astype(int)
    selected["label_name"] = selected["coarse_label"].astype(str)
    selected["label_granularity"] = label_granularity
    selected["subset_profile"] = subset_profile
    selected = selected.drop(columns=["candidate_ordinal"], errors="ignore")
    split_order = pd.Categorical(selected["split"], categories=SPLITS, ordered=True)
    selected = (
        selected.assign(_split_order=split_order)
        .sort_values(
            ["_split_order", "coarse_label", "track_id", "stem_id", "start_seconds"]
        )
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )
    return selected, ranking, diagnostics, file_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Dataset YAML config. CLI values are otherwise unchanged.")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--medleydb-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--hop-seconds", type=float, default=5.0)
    parser.add_argument("--max-classes", type=int, default=8)
    parser.add_argument("--max-segments-per-class", type=int, default=80)
    parser.add_argument("--min-segments-per-class", type=int, default=20)
    parser.add_argument("--min-tracks-per-class", type=int, default=3)
    parser.add_argument(
        "--total-budget",
        type=int,
        default=None,
        help="Optional global segment budget for capped_natural sampling.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-bleed", action="store_true")
    parser.add_argument("--silence-threshold-dbfs", type=float, default=-50.0)
    parser.add_argument("--split-search-attempts", type=int, default=1000)
    parser.add_argument(
        "--sampling-strategy",
        choices=["balanced", "capped_natural"],
        default="balanced",
    )
    parser.add_argument(
        "--label-granularity",
        choices=LABEL_GRANULARITIES,
        default="coarse_family",
        help="Label protocol: coarse instrument families or lightly normalized MedleyDB labels.",
    )
    parser.add_argument(
        "--subset-profile",
        choices=SUBSET_PROFILES,
        default="debug",
        help="Named protocol profile used for output names and cache validation.",
    )
    return parser.parse_args()


def apply_dataset_config(args: argparse.Namespace) -> argparse.Namespace:
    """Apply dataset YAML values to argparse output when ``--config`` is used."""

    if args.config is None:
        return args
    config = load_yaml(args.config)
    aliases = {
        "rms_threshold_db": "silence_threshold_dbfs",
        "silence_threshold_dbfs": "silence_threshold_dbfs",
    }
    for key, value in config.items():
        target = aliases.get(key, key.replace("-", "_"))
        if hasattr(args, target):
            if target in {"index", "out", "report_dir"}:
                value = resolve_run_path(value)
            elif target == "medleydb_root":
                value = Path(os.environ.get("MEDLEYDB_ROOT", value))
            setattr(args, target, value)
    return args


def validate_required_paths(args: argparse.Namespace) -> None:
    """Fail early when neither CLI nor YAML supplied required paths."""

    missing = [
        name for name in ("index", "medleydb_root", "out", "report_dir")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Missing required dataset settings: "
            + ", ".join(missing)
            + ". Supply them on the CLI or in --config."
        )


def main() -> None:
    args = apply_dataset_config(parse_args())
    validate_required_paths(args)
    if args.max_classes < 2:
        raise ValueError("--max-classes must be at least 2")
    if args.min_segments_per_class > args.max_segments_per_class:
        raise ValueError("Minimum segments per class cannot exceed the maximum")
    if args.total_budget is not None and args.total_budget < args.min_segments_per_class * args.max_classes:
        raise ValueError("total_budget must cover the minimum quota for every selected class")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("Validation and test ratios must sum to less than 1")
    if not args.index.is_file():
        raise FileNotFoundError(f"Stem index not found: {args.index}")

    index = pd.read_csv(args.index)
    validate_stem_index_columns(index, source=args.index)
    selected, ranking, diagnostics, file_stats = create_subset(
        index,
        args.medleydb_root,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        max_classes=args.max_classes,
        max_segments_per_class=args.max_segments_per_class,
        min_segments_per_class=args.min_segments_per_class,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        allow_bleed=args.allow_bleed,
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        split_search_attempts=args.split_search_attempts,
        sampling_strategy=args.sampling_strategy,
        label_granularity=args.label_granularity,
        subset_profile=args.subset_profile,
        min_tracks_per_class=args.min_tracks_per_class,
        total_budget=args.total_budget,
    )
    report_dir = ensure_directory(args.report_dir)
    write_label_mapping_report(
        report_dir / "label_mapping_report.md",
        index,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        allow_bleed=args.allow_bleed,
    )
    report_path = report_dir / f"subset_report_{args.subset_profile}_{args.label_granularity}.md"
    write_subset_report(
        report_path,
        file_stats=file_stats,
        ranking=ranking,
        selected=selected,
        diagnostics=diagnostics,
        allow_bleed=args.allow_bleed,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        sampling_strategy=args.sampling_strategy,
        label_granularity=args.label_granularity,
        subset_profile=args.subset_profile,
        min_tracks_per_class=args.min_tracks_per_class,
        max_classes=args.max_classes,
        max_segments_per_class=args.max_segments_per_class,
        total_budget=args.total_budget,
    )
    legacy_granularity_report = report_dir / f"subset_report_{args.label_granularity}.md"
    if legacy_granularity_report != report_path:
        write_subset_report(
            legacy_granularity_report,
            file_stats=file_stats,
            ranking=ranking,
            selected=selected,
            diagnostics=diagnostics,
            allow_bleed=args.allow_bleed,
            segment_seconds=args.segment_seconds,
            hop_seconds=args.hop_seconds,
            silence_threshold_dbfs=args.silence_threshold_dbfs,
            sampling_strategy=args.sampling_strategy,
            label_granularity=args.label_granularity,
            subset_profile=args.subset_profile,
            min_tracks_per_class=args.min_tracks_per_class,
            max_classes=args.max_classes,
            max_segments_per_class=args.max_segments_per_class,
            total_budget=args.total_budget,
        )
    if args.label_granularity == "coarse_family":
        # Backward-compatible report name used by existing scripts and docs.
        write_subset_report(
            report_dir / "subset_report.md",
            file_stats=file_stats,
            ranking=ranking,
            selected=selected,
            diagnostics=diagnostics,
            allow_bleed=args.allow_bleed,
            segment_seconds=args.segment_seconds,
            hop_seconds=args.hop_seconds,
            silence_threshold_dbfs=args.silence_threshold_dbfs,
            sampling_strategy=args.sampling_strategy,
            label_granularity=args.label_granularity,
            subset_profile=args.subset_profile,
            min_tracks_per_class=args.min_tracks_per_class,
            max_classes=args.max_classes,
            max_segments_per_class=args.max_segments_per_class,
            total_budget=args.total_budget,
        )
    if selected.empty:
        raise RuntimeError(
            f"Fewer than two balanced classes were viable. See {report_path}"
        )

    ensure_parent(args.out)
    selected.to_csv(args.out, index=False)
    labels = sorted(selected["coarse_label"].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {str(index): label for label, index in label_to_id.items()}
    _write_json(
        args.out.parent / f"labels_{args.subset_profile}_{args.label_granularity}_label_to_id.json",
        label_to_id,
    )
    _write_json(
        args.out.parent / f"labels_{args.subset_profile}_{args.label_granularity}_id_to_label.json",
        id_to_label,
    )
    _write_json(args.out.parent / f"{args.label_granularity}_label_to_id.json", label_to_id)
    _write_json(args.out.parent / f"{args.label_granularity}_id_to_label.json", id_to_label)
    if args.label_granularity == "coarse_family":
        # Backward-compatible names used by the original frozen-MERT pipeline.
        _write_json(args.out.parent / "label_to_id.json", label_to_id)
        _write_json(args.out.parent / "id_to_label.json", id_to_label)
    print(
        f"Saved {len(selected)} segments across {len(labels)} classes and "
        f"{selected['track_id'].nunique()} tracks."
    )
    print(f"Subset: {args.out}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
