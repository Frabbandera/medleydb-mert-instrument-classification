"""Build a fault-tolerant index of MedleyDB isolated stem audio."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.data.audio_io import inspect_audio_file
from src.data.label_mapping import normalize_label, normalize_instrument_label
from src.utils.paths import ensure_directory, ensure_parent, load_yaml, portable_relative_path


INDEX_COLUMNS = [
    "track_id",
    "stem_id",
    "audio_path",
    "metadata_path",
    "metadata_source",
    "source_layout",
    "raw_instrument_label",
    "coarse_label",
    "medleydb_instrument_label",
    "genre",
    "has_bleed",
    "duration_seconds",
    "sample_rate",
    "num_channels",
    "num_frames",
    "valid",
    "error_message",
]


def _metadata_path_text(
    path: Path, medleydb_root: Path, code_root: Path | None
) -> str:
    try:
        return portable_relative_path(path, medleydb_root)
    except ValueError:
        if code_root is not None:
            try:
                return f"fallback:{portable_relative_path(path, code_root)}"
            except ValueError:
                pass
        return path.name


def _find_audio_base(root: Path) -> tuple[Path, str]:
    audio_dir = root / "Audio"
    if audio_dir.is_dir():
        return audio_dir, "audio_subdirectory"
    return root, "direct"


def _discover_track_directories(audio_base: Path) -> list[Path]:
    tracks: list[Path] = []
    for candidate in sorted(path for path in audio_base.iterdir() if path.is_dir()):
        has_metadata = any(candidate.glob("*_METADATA.yaml"))
        has_stems = any(candidate.glob("*_STEMS")) or any(candidate.rglob("*_STEM_*.wav"))
        has_mix = any(candidate.glob("*_MIX.wav"))
        if has_metadata or has_stems or has_mix:
            tracks.append(candidate)
    return tracks


def _local_metadata_map(root: Path, audio_base: Path) -> dict[str, Path]:
    paths = list(audio_base.glob("*/*_METADATA.yaml"))
    for metadata_dir in (root / "Metadata", root / "metadata"):
        if metadata_dir.is_dir():
            paths.extend(metadata_dir.glob("*_METADATA.yaml"))
    return {
        path.name.removesuffix("_METADATA.yaml"): path
        for path in sorted(set(paths))
    }


def _fallback_metadata_map(code_root: Path | None) -> dict[str, Path]:
    if code_root is None:
        return {}
    metadata_dir = code_root / "medleydb" / "data" / "Metadata"
    if not metadata_dir.is_dir():
        return {}
    return {
        path.name.removesuffix("_METADATA.yaml"): path
        for path in metadata_dir.glob("*_METADATA.yaml")
    }


def _resolve_stem_path(
    root: Path,
    audio_base: Path,
    track_dir: Path,
    track_id: str,
    stem_dir_name: str,
    filename: str,
) -> Path:
    candidates = [
        track_dir / stem_dir_name / filename,
        audio_base / track_id / stem_dir_name / filename,
        root / stem_dir_name / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _base_row(
    *,
    track_id: str,
    stem_id: str,
    audio_path: Path,
    metadata_path: Path | None,
    metadata_source: str,
    source_layout: str,
    raw_label: str,
    genre: str,
    has_bleed: bool | None,
    root: Path,
    code_root: Path | None,
) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "stem_id": stem_id,
        "audio_path": portable_relative_path(audio_path, root),
        "metadata_path": (
            _metadata_path_text(metadata_path, root, code_root) if metadata_path else ""
        ),
        "metadata_source": metadata_source,
        "source_layout": source_layout,
        "raw_instrument_label": raw_label,
        "coarse_label": normalize_instrument_label(raw_label),
        "medleydb_instrument_label": normalize_label(raw_label, "medleydb_instrument"),
        "genre": genre,
        "has_bleed": has_bleed,
        "duration_seconds": None,
        "sample_rate": None,
        "num_channels": None,
        "num_frames": None,
        "valid": False,
        "error_message": "",
    }


def _validate_row(row: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    try:
        info = inspect_audio_file(audio_path)
        row.update(
            {
                "duration_seconds": round(info.duration_seconds, 6),
                "sample_rate": info.sample_rate,
                "num_channels": info.num_channels,
                "num_frames": info.num_frames,
                "valid": True,
            }
        )
    except Exception as exc:  # one damaged stem must not stop the dataset scan
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    return row


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return True
    if text in {"no", "false", "0"}:
        return False
    return None


def _tracklist_coverage(track_ids: set[str]) -> list[tuple[str, int, int]]:
    """Compare local tracks with the bundled official V1/V2 name manifests."""

    resources = Path(__file__).parent / "resources"
    coverage: list[tuple[str, int, int]] = []
    if not resources.is_dir():
        return coverage
    for path in sorted(resources.glob("tracklist_*.txt")):
        expected = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        coverage.append((path.stem, len(expected), len(expected & track_ids)))
    return coverage


def build_stem_index(
    medleydb_root: Path,
    medleydb_code_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return the stem index, issue table, and report statistics."""

    root = medleydb_root.resolve()
    code_root = medleydb_code_root.resolve() if medleydb_code_root else None
    if not root.is_dir():
        raise FileNotFoundError(f"MedleyDB root does not exist: {medleydb_root}")

    audio_base, source_layout = _find_audio_base(root)
    track_dirs = _discover_track_directories(audio_base)
    local_metadata = _local_metadata_map(root, audio_base)
    fallback_metadata = _fallback_metadata_map(code_root)

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    declared_paths: set[Path] = set()
    metadata_sources: Counter[str] = Counter()

    for track_dir in tqdm(track_dirs, desc="Indexing MedleyDB tracks"):
        track_id = track_dir.name
        metadata_path = local_metadata.get(track_id) or fallback_metadata.get(track_id)
        metadata_source = "local" if track_id in local_metadata else "fallback"
        if metadata_path is None:
            issues.append(
                {
                    "track_id": track_id,
                    "stem_id": "",
                    "audio_path": "",
                    "issue_type": "missing_metadata",
                    "error_message": (
                        "No metadata file was found beside the local track or in the "
                        "optional metadata fallback"
                    ),
                }
            )
            continue
        try:
            metadata = load_yaml(metadata_path)
        except Exception as exc:
            issues.append(
                {
                    "track_id": track_id,
                    "stem_id": "",
                    "audio_path": "",
                    "issue_type": "malformed_metadata",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        metadata_sources[metadata_source] += 1
        stem_dir_name = str(metadata.get("stem_dir") or f"{track_id}_STEMS")
        genre = str(metadata.get("genre") or "")
        has_bleed = _as_bool(metadata.get("has_bleed"))
        stems = metadata.get("stems") or {}
        if not isinstance(stems, dict):
            issues.append(
                {
                    "track_id": track_id,
                    "stem_id": "",
                    "audio_path": "",
                    "issue_type": "malformed_metadata",
                    "error_message": "The 'stems' field is not a mapping",
                }
            )
            continue

        for stem_id, stem_metadata in sorted(stems.items()):
            if not isinstance(stem_metadata, dict):
                stem_metadata = {}
            filename = str(
                stem_metadata.get("filename")
                or f"{track_id}_STEM_{str(stem_id).lstrip('S')}.wav"
            )
            raw_label = str(stem_metadata.get("instrument") or "")
            audio_path = _resolve_stem_path(
                root, audio_base, track_dir, track_id, stem_dir_name, filename
            )
            declared_paths.add(audio_path.resolve())
            row = _base_row(
                track_id=track_id,
                stem_id=str(stem_id),
                audio_path=audio_path,
                metadata_path=metadata_path,
                metadata_source=metadata_source,
                source_layout=source_layout,
                raw_label=raw_label,
                genre=genre,
                has_bleed=has_bleed,
                root=root,
                code_root=code_root,
            )
            row = _validate_row(row, audio_path)
            rows.append(row)
            if not row["valid"]:
                issues.append(
                    {
                        "track_id": track_id,
                        "stem_id": str(stem_id),
                        "audio_path": row["audio_path"],
                        "issue_type": (
                            "missing_audio" if not audio_path.exists() else "invalid_audio"
                        ),
                        "error_message": row["error_message"],
                    }
                )

    disk_stems = sorted(audio_base.rglob("*_STEMS/*_STEM_*.wav"))
    for audio_path in disk_stems:
        if audio_path.resolve() in declared_paths:
            continue
        track_id = audio_path.parent.parent.name
        row = _base_row(
            track_id=track_id,
            stem_id=audio_path.stem.rsplit("_STEM_", 1)[-1],
            audio_path=audio_path,
            metadata_path=None,
            metadata_source="none",
            source_layout=source_layout,
            raw_label="",
            genre="",
            has_bleed=None,
            root=root,
            code_root=code_root,
        )
        row = _validate_row(row, audio_path)
        rows.append(row)
        issues.append(
            {
                "track_id": track_id,
                "stem_id": row["stem_id"],
                "audio_path": row["audio_path"],
                "issue_type": "orphan_audio",
                "error_message": "Stem audio was found but not declared by available metadata",
            }
        )

    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if not index.empty:
        index = index.sort_values(["track_id", "stem_id", "audio_path"]).reset_index(drop=True)
    issue_columns = ["track_id", "stem_id", "audio_path", "issue_type", "error_message"]
    bad_files = pd.DataFrame(issues, columns=issue_columns)
    if not bad_files.empty:
        bad_files = bad_files.sort_values(
            ["issue_type", "track_id", "stem_id"]
        ).reset_index(drop=True)

    valid_mask = index["valid"].fillna(False).astype(bool) if not index.empty else pd.Series(dtype=bool)
    stats = {
        "root_argument": str(medleydb_root),
        "source_layout": source_layout,
        "track_directories": len(track_dirs),
        "local_metadata_files": len(local_metadata),
        "fallback_metadata_files": len(fallback_metadata),
        "metadata_sources": dict(metadata_sources),
        "indexed_stems": len(index),
        "valid_stems": int(valid_mask.sum()),
        "invalid_stems": int((~valid_mask).sum()) if len(index) else 0,
        "orphan_stems": int((bad_files["issue_type"] == "orphan_audio").sum())
        if not bad_files.empty
        else 0,
        "issue_counts": bad_files["issue_type"].value_counts().to_dict()
        if not bad_files.empty
        else {},
        "sample_rates": {
            int(key): int(value)
            for key, value in index.loc[valid_mask, "sample_rate"].value_counts().items()
        }
        if not index.empty
        else {},
        "channel_counts": {
            int(key): int(value)
            for key, value in index.loc[valid_mask, "num_channels"].value_counts().items()
        }
        if not index.empty
        else {},
        "tracklist_coverage": _tracklist_coverage({path.name for path in track_dirs}),
    }
    return index, bad_files, stats


def write_health_report(stats: dict[str, Any], report_path: Path) -> None:
    """Write a human-readable snapshot of dataset completeness."""

    lines = [
        "# MedleyDB data health report",
        "",
        "This report is generated from the local files. Missing or damaged stems are recorded rather than treated as fatal errors.",
        "",
        "## Summary",
        "",
        f"- Dataset root argument: `{stats['root_argument']}`",
        f"- Detected layout: `{stats['source_layout']}`",
        f"- Track directories found: {stats['track_directories']}",
        f"- Local metadata files found: {stats['local_metadata_files']}",
        f"- Optional fallback metadata files available: {stats['fallback_metadata_files']}",
        f"- Indexed stem rows: {stats['indexed_stems']}",
        f"- Valid stem audio files: {stats['valid_stems']}",
        f"- Invalid or missing stem audio files: {stats['invalid_stems']}",
        f"- Orphan stem audio files: {stats['orphan_stems']}",
        "",
        "## Official track-list coverage",
        "",
        "| Track list | Present | Expected |",
        "|---|---:|---:|",
    ]
    coverage = stats.get("tracklist_coverage") or []
    if coverage:
        lines.extend(f"| {name} | {present} | {expected} |" for name, expected, present in coverage)
    else:
        lines.append("| unavailable | 0 | 0 |")
    lines.extend(
        [
            "",
            "## Audio properties",
            "",
            f"- Sample-rate counts: `{stats.get('sample_rates', {})}`",
            f"- Channel-count counts: `{stats.get('channel_counts', {})}`",
            "",
            "## Issues",
            "",
        ]
    )
    issue_counts = stats.get("issue_counts") or {}
    if issue_counts:
        lines.extend(f"- {name}: {count}" for name, count in sorted(issue_counts.items()))
    else:
        lines.append("- No issues detected.")
    lines.extend(
        [
            "",
            "See `bad_files.csv` for paths and error messages. A valid file may still carry known MedleyDB annotation or bleed caveats; those are metadata concerns, not decoder failures.",
            "",
        ]
    )
    ensure_parent(report_path).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medleydb-root", type=Path, required=True)
    parser.add_argument(
        "--medleydb-code-root",
        type=Path,
        help=(
            "Optional legacy marl/medleydb checkout used only as a metadata fallback. "
            "It is not needed when the dataset has per-track *_METADATA.yaml files."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = ensure_directory(args.report_dir)
    index, bad_files, stats = build_stem_index(
        args.medleydb_root, args.medleydb_code_root
    )
    ensure_parent(args.out)
    index.to_csv(args.out, index=False)
    bad_files.to_csv(report_dir / "bad_files.csv", index=False)
    write_health_report(stats, report_dir / "data_health_report.md")
    print(
        f"Indexed {len(index)} stems: {stats['valid_stems']} valid, "
        f"{stats['invalid_stems']} invalid."
    )
    print(f"Index: {args.out}")
    print(f"Health report: {report_dir / 'data_health_report.md'}")


if __name__ == "__main__":
    main()
