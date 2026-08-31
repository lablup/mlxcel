#!/usr/bin/env python3
"""Generate the pinned Inkling log-mel fixture with NumPy 2.3.2.

Provenance: Blaizzy/mlx-vlm PR #1767, head commit
0d6805bb7ef67998d8aeb655bc1df83854830d56, merged as
67bc41d818ea77908599d21510ea29f352e7a417. Its independent reference test uses
``np.random.default_rng(5)``, 2,401 samples, and ``np.fft.rfft``.

This generator intentionally does not import mlxcel or mlx-vlm. The Slaney
filterbank follows the published Auditory Toolbox piecewise mel scale and area
normalization using NumPy array operations. The Rust test consumes only the
committed JSON and never runs this generator.
"""

import json
from pathlib import Path

import numpy as np


NUMPY_VERSION = "2.3.2"
UPSTREAM_HEAD = "0d6805bb7ef67998d8aeb655bc1df83854830d56"
UPSTREAM_MERGE = "67bc41d818ea77908599d21510ea29f352e7a417"
SAMPLE_RATE = 16_000
N_FFT = 1_600
HOP_LENGTH = 800
N_MELS = 80


def hz_to_slaney_mel(frequencies: np.ndarray) -> np.ndarray:
    spacing = 200.0 / 3.0
    log_frequency = 1_000.0
    log_mel = log_frequency / spacing
    log_step = np.log(6.4) / 27.0
    return np.where(
        frequencies >= log_frequency,
        log_mel + np.log(np.maximum(frequencies, log_frequency) / log_frequency) / log_step,
        frequencies / spacing,
    )


def slaney_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    spacing = 200.0 / 3.0
    log_frequency = 1_000.0
    log_mel = log_frequency / spacing
    log_step = np.log(6.4) / 27.0
    return np.where(
        mels >= log_mel,
        log_frequency * np.exp(log_step * (mels - log_mel)),
        spacing * mels,
    )


def slaney_filterbank() -> np.ndarray:
    fft_frequencies = np.linspace(0.0, SAMPLE_RATE / 2.0, N_FFT // 2 + 1)
    mel_limits = hz_to_slaney_mel(np.array([0.0, SAMPLE_RATE / 2.0]))
    mel_points = np.linspace(mel_limits[0], mel_limits[1], N_MELS + 2)
    frequencies = slaney_mel_to_hz(mel_points)
    ramps = frequencies[:, None] - fft_frequencies[None, :]
    widths = np.diff(frequencies)
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (frequencies[2:] - frequencies[:-2]))[:, None]
    return weights.astype(np.float32)


def generate() -> dict:
    if np.__version__ != NUMPY_VERSION:
        raise RuntimeError(f"expected NumPy {NUMPY_VERSION}, got {np.__version__}")
    rng = np.random.default_rng(5)
    waveform = rng.normal(0.0, 0.1, 2_401).astype(np.float32)
    right_pad = (-len(waveform)) % HOP_LENGTH
    padded = np.pad(waveform, (N_FFT - HOP_LENGTH, right_pad))
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=-1)
    magnitudes = np.maximum(np.abs(spectrum), 1e-10).astype(np.float32)
    mel = magnitudes @ slaney_filterbank().T
    log_mel = np.log10(np.maximum(mel, 1e-10)).astype(np.float32)
    return {
        "generator": "tests/fixtures/generate_inkling_dmel_numpy.py",
        "numpy_version": NUMPY_VERSION,
        "upstream_pr": "https://github.com/Blaizzy/mlx-vlm/pull/1767",
        "upstream_head_revision": UPSTREAM_HEAD,
        "upstream_merge_revision": UPSTREAM_MERGE,
        "rng": "numpy.random.default_rng(5).normal(0.0, 0.1, 2401).astype(float32)",
        "shape": list(log_mel.shape),
        "absolute_tolerance": 2e-6,
        "waveform": waveform.tolist(),
        "expected_log_mel": log_mel.reshape(-1).tolist(),
    }


if __name__ == "__main__":
    destination = Path(__file__).with_name("inkling_dmel_numpy.json")
    destination.write_text(json.dumps(generate(), indent=2) + "\n", encoding="utf-8")
