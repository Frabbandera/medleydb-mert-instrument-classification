"""Validation helpers for generated MedleyDB metadata files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_STEM_INDEX_COLUMNS = {
    "track_id",
    "stem_id",
    "audio_path",
    "raw_instrument_label",
    "coarse_label",
    "medleydb_instrument_label",
    "duration_seconds",
    "valid",
    "has_bleed",
}


def validate_stem_index_columns(
    frame: pd.DataFrame,
    *,
    source: str | Path = "stem_index.csv",
    required: set[str] | None = None,
) -> None:
    """Fail clearly when a generated stem index is stale or incomplete."""

    required_columns = required or REQUIRED_STEM_INDEX_COLUMNS
    missing = sorted(required_columns - set(frame.columns))
    if not missing:
        return
    raise ValueError(
        f"{source} is missing required columns: {', '.join(missing)}. "
        "This usually means the generated metadata is stale. Rebuild it with "
        "`python -m src.data.build_stem_index --medleydb-root <MEDLEYDB_ROOT> "
        "--out <RUN_ROOT>/data/metadata/stem_index.csv --report-dir "
        "<RUN_ROOT>/data/reports` before creating subsets or mixtures."
    )

