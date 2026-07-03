"""Robust, small-footprint audio utilities for MedleyDB WAV stems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


@dataclass(frozen=True)
class AudioFileInfo:
    """Header information returned by :func:`inspect_audio_file`."""

    duration_seconds: float
    sample_rate: int
    num_channels: int
    num_frames: int


def inspect_audio_file(path: Path, probe_frames: int = 4096) -> AudioFileInfo:
    """Validate an audio file and perform bounded reads at both ends.

    Header-only inspection can miss truncated files. Reading a short block near
    the beginning and end catches common incomplete-download failures without
    decoding the whole recording.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Audio file is empty: {path}")

    with sf.SoundFile(path, mode="r") as handle:
        sample_rate = int(handle.samplerate)
        channels = int(handle.channels)
        frames = int(len(handle))
        if sample_rate <= 0 or channels <= 0 or frames <= 0:
            raise ValueError(
                f"Invalid audio header: sample_rate={sample_rate}, channels={channels}, frames={frames}"
            )
        head = handle.read(min(probe_frames, frames), dtype="float32", always_2d=True)
        if frames > probe_frames:
            handle.seek(max(0, frames - probe_frames))
            tail = handle.read(min(probe_frames, frames), dtype="float32", always_2d=True)
        else:
            tail = head
    if not np.isfinite(head).all() or not np.isfinite(tail).all():
        raise ValueError("Audio probe contains NaN or infinite values")
    return AudioFileInfo(
        duration_seconds=frames / sample_rate,
        sample_rate=sample_rate,
        num_channels=channels,
        num_frames=frames,
    )


def calculate_rms(waveform: torch.Tensor) -> float:
    """Return root-mean-square amplitude for a float waveform."""

    if waveform.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(waveform.float().square())).item())


def rms_dbfs(waveform: torch.Tensor, floor_db: float = -120.0) -> float:
    """Return RMS in dBFS, bounded by ``floor_db`` for digital silence."""

    value = calculate_rms(waveform)
    if value <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(value))


def safe_peak_normalize(waveform: torch.Tensor, target_peak: float = 0.99) -> torch.Tensor:
    """Peak-normalize non-silent audio without amplifying numerical noise."""

    if waveform.numel() == 0:
        return waveform
    peak = float(waveform.abs().max().item())
    if not math.isfinite(peak):
        raise ValueError("Waveform contains NaN or infinite values")
    if peak <= 1e-8:
        return waveform
    return waveform * (target_peak / peak)


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    try:
        import torchaudio.functional as audio_functional

        return audio_functional.resample(waveform, source_rate, target_rate)
    except (ImportError, OSError) as torchaudio_error:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError(
                "Resampling requires a working torchaudio installation or librosa."
            ) from torchaudio_error
        output = librosa.resample(
            waveform.detach().cpu().numpy(), orig_sr=source_rate, target_sr=target_rate
        )
        return torch.from_numpy(np.asarray(output, dtype=np.float32))


def load_audio_segment(
    path: Path,
    start_seconds: float,
    duration_seconds: float,
    target_sample_rate: int | None = None,
    *,
    normalize: bool = True,
    pad: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load one mono segment as ``float32`` in the range expected by MERT.

    Parameters are expressed in seconds so metadata stays independent of the
    source file's sample rate. If ``pad`` is true, a short final read is padded
    on the right; subset creation itself only emits complete source segments.
    """

    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    try:
        with sf.SoundFile(path, mode="r") as handle:
            source_rate = int(handle.samplerate)
            start_frame = int(round(start_seconds * source_rate))
            frame_count = int(round(duration_seconds * source_rate))
            if start_frame >= len(handle):
                raise ValueError(
                    f"Segment starts beyond end of file ({start_seconds:.3f}s): {path}"
                )
            handle.seek(start_frame)
            audio = handle.read(frame_count, dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise RuntimeError(
            f"Could not decode {path} at {start_seconds:.3f}s for {duration_seconds:.3f}s: {exc}"
        ) from exc

    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).transpose(0, 1)
    waveform = waveform.mean(dim=0)  # channel-first to mono
    if waveform.numel() < frame_count:
        if not pad:
            raise ValueError(
                f"Short segment read from {path}: expected {frame_count}, got {waveform.numel()} frames"
            )
        waveform = torch.nn.functional.pad(waveform, (0, frame_count - waveform.numel()))

    output_rate = target_sample_rate or source_rate
    waveform = _resample(waveform, source_rate, output_rate).float().contiguous()
    expected_output_frames = int(round(duration_seconds * output_rate))
    if waveform.numel() < expected_output_frames and pad:
        waveform = torch.nn.functional.pad(
            waveform, (0, expected_output_frames - waveform.numel())
        )
    elif waveform.numel() > expected_output_frames:
        waveform = waveform[:expected_output_frames]
    if normalize:
        waveform = safe_peak_normalize(waveform)
    return waveform, output_rate

