"""Audio utilities use generated files and need no MedleyDB checkout."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from src.data.audio_io import (
    inspect_audio_file,
    load_audio_segment,
    rms_dbfs,
    safe_peak_normalize,
)


def test_load_mono_resampled_segment(tmp_path) -> None:
    sample_rate = 8000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 440.0 * time)
    stereo = np.stack([tone, tone * 0.5], axis=1)
    path = tmp_path / "tone.wav"
    sf.write(path, stereo, sample_rate, subtype="PCM_16")

    info = inspect_audio_file(path)
    assert info.sample_rate == sample_rate
    assert info.num_channels == 2
    assert abs(info.duration_seconds - 2.0) < 1e-3

    waveform, output_rate = load_audio_segment(
        path,
        start_seconds=0.5,
        duration_seconds=1.0,
        target_sample_rate=16000,
        normalize=False,
        pad=False,
    )
    assert output_rate == 16000
    assert waveform.ndim == 1
    assert abs(waveform.numel() - 16000) <= 1
    assert rms_dbfs(waveform) > -30.0


def test_silence_and_peak_normalization() -> None:
    silence = torch.zeros(100)
    assert rms_dbfs(silence) == -120.0
    assert torch.equal(safe_peak_normalize(silence), silence)

    waveform = torch.tensor([-0.25, 0.5])
    normalized = safe_peak_normalize(waveform)
    assert torch.isclose(normalized.abs().max(), torch.tensor(0.99))

