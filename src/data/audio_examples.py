"""Helpers for replaying and exporting exact experiment audio segments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.data.audio_io import load_audio_segment, safe_peak_normalize
from src.utils.paths import ensure_parent, resolve_data_path

LABEL_SEPARATOR = "|"


def split_pipe(value: Any) -> list[str]:
    """Split a pipe-delimited metadata field while tolerating missing values."""

    text = "" if pd.isna(value) else str(value)
    return [item for item in text.split(LABEL_SEPARATOR) if item]


def load_isolated_stem_segment(
    row: Any,
    medleydb_root: Path,
    *,
    target_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load the exact isolated stem segment described by a subset/prediction row."""

    return load_audio_segment(
        resolve_data_path(Path(medleydb_root), str(row.audio_path)),
        float(row.start_seconds),
        float(row.duration_seconds),
        target_sample_rate=target_sample_rate,
    )


def load_mixture_segment(
    row: Any,
    medleydb_root: Path,
    *,
    target_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load the exact mixture segment described by a mixture manifest row.

    If ``audio_path`` is present, the segment comes from an original full mix.
    Otherwise, source stems are decoded at their stored starts and mixed.
    """

    root = Path(medleydb_root)
    duration = float(row.duration_seconds)
    audio_path = "" if pd.isna(getattr(row, "audio_path", "")) else str(row.audio_path)
    if audio_path:
        return load_audio_segment(
            resolve_data_path(root, audio_path),
            float(row.start_seconds),
            duration,
            target_sample_rate=target_sample_rate,
        )
    paths = split_pipe(getattr(row, "source_audio_paths", ""))
    starts = [float(value) for value in split_pipe(getattr(row, "source_start_seconds", ""))]
    if len(paths) != len(starts):
        raise ValueError(f"Mixture row has mismatched source paths/start times: {getattr(row, 'mixture_id', 'unknown')}")
    if not paths:
        raise ValueError(f"Mixture row has no source audio: {getattr(row, 'mixture_id', 'unknown')}")
    waveforms: list[torch.Tensor] = []
    sample_rate: int | None = None
    for path, start in zip(paths, starts):
        waveform, current_rate = load_audio_segment(
            resolve_data_path(root, path),
            start,
            duration,
            target_sample_rate=target_sample_rate,
        )
        waveforms.append(waveform)
        sample_rate = current_rate
    length = min(waveform.numel() for waveform in waveforms)
    mixed = torch.stack([waveform[:length] for waveform in waveforms]).sum(dim=0)
    mixed = safe_peak_normalize(mixed / max(1, len(waveforms)))
    return mixed, int(sample_rate or target_sample_rate or 0)


def export_audio_segment(waveform: torch.Tensor, sample_rate: int, out_path: Path) -> Path:
    """Write a mono float waveform to disk for Project6 presentation examples."""

    ensure_parent(Path(out_path))
    audio = waveform.detach().cpu().numpy().astype(np.float32)
    sf.write(Path(out_path), audio, int(sample_rate))
    return Path(out_path)

