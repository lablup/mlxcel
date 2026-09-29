// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Batched STFT / iSTFT for the codec's tiny spectrogram (n_fft 16, hop 4).
//!
//! Ports the pieces of `mlx_audio/dsp.py` the codec uses (`hanning`,
//! `stft(center=False)`, `ISTFTCache.istft(center=False,
//! constrain_value_range=True)`) and the codec's `_spectrogram` padding.
//! Kokoro's `models/kokoro/stft.rs` is host-side and fixed to Kokoro sizes,
//! so this stays a separate MLX-graph implementation.
//!
//! Layout is time-major throughout: spectra are `[B, frames, bins]`.

use mlxcel_core::{MlxArray, UniquePtr};

/// Periodic Hann window, computed in f64 and stored as f32 like
/// `hanning(n, periodic=True)`.
pub(crate) fn hann_periodic(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| {
            let phase = 2.0 * std::f64::consts::PI * n as f64 / size as f64;
            (0.5 * (1.0 - phase.cos())) as f32
        })
        .collect()
}

fn dim(value: usize) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("codec stft: size {value} exceeds i32 range"))
}

/// `_spectrogram`: pad `[B, N]` by `(n_fft - hop) / 2` on both sides
/// (right-extended to at least `n_fft`), frame without centering, window,
/// and rFFT. Returns `(real, imag)`, each `[B, frames, n_fft / 2 + 1]` f32.
pub(crate) fn spectrogram(
    waveform: &MlxArray,
    n_fft: usize,
    hop_length: usize,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
    let shape = mlxcel_core::array_shape(waveform);
    if shape.len() != 2 {
        return Err(format!(
            "waveform must have shape (batch, samples), got {shape:?}"
        ));
    }
    if shape[1] == 0 {
        return Err("waveform must contain at least one sample".to_string());
    }
    let (n, hop) = (dim(n_fft)?, dim(hop_length)?);
    let pad_left = (n - hop) / 2;
    let pad_right = n - hop - pad_left;
    let signal = mlxcel_core::astype(waveform, mlxcel_core::dtype::FLOAT32);
    let extra = (n - (shape[1] + pad_left + pad_right)).max(0);
    let padded = mlxcel_core::pad(&signal, &[0, 0, pad_left, pad_right + extra], 0.0);
    let length = shape[1] + pad_left + pad_right + extra;
    let frames = 1 + (length - n) / hop;
    let framed = mlxcel_core::as_strided(
        &padded,
        &[shape[0], frames, n],
        &[i64::from(length), i64::from(hop), 1],
        0,
    );
    let window = mlxcel_core::from_slice_f32(&hann_periodic(n_fft), &[n]);
    let spectrum = mlxcel_core::rfft(&mlxcel_core::multiply(&framed, &window), n, -1);
    Ok((
        mlxcel_core::real_part(&spectrum),
        mlxcel_core::imag_part(&spectrum),
    ))
}

/// `ISTFTCache.istft(..., center=False, constrain_value_range=True)` on
/// `[B, frames, bins]` real/imag parts. Returns `[B, (frames - 1) * hop + n_fft]`
/// f32 (the caller trims the analysis padding).
pub(crate) fn istft(
    real: &MlxArray,
    imag: &MlxArray,
    n_fft: usize,
    hop_length: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(real);
    if shape.len() != 3 || mlxcel_core::array_shape(imag) != shape {
        return Err(format!(
            "codec istft: real/imag must match as [B, F, bins], got {shape:?}"
        ));
    }
    let (batch, frames, bins) = (shape[0], shape[1], shape[2]);
    let (n, hop) = (dim(n_fft)?, dim(hop_length)?);
    if frames == 0 {
        return Err("codec istft: no spectrogram frames".to_string());
    }

    // `real + 1j * imag` is complex64: stack f32 parts and reinterpret.
    let re = mlxcel_core::astype(real, mlxcel_core::dtype::FLOAT32);
    let im = mlxcel_core::astype(imag, mlxcel_core::dtype::FLOAT32);
    let pair = mlxcel_core::stack(&[&*re as *const MlxArray, &*im as *const MlxArray], -1);
    let pair = mlxcel_core::contiguous(&pair, false);
    let complex = mlxcel_core::view(&pair, mlxcel_core::dtype::COMPLEX64);
    let complex = mlxcel_core::reshape(&complex, &[batch, frames, bins]);
    let time_frames = mlxcel_core::irfft(&complex, n, -1);

    let window_host = hann_periodic(n_fft);
    let window = mlxcel_core::from_slice_f32(&window_host, &[n]);
    let clipped = mlxcel_core::clip(&time_frames, &mlxcel_core::negative(&window), &window);
    let windowed = mlxcel_core::multiply(&clipped, &window);

    // Overlap-add: with n_fft = r * hop, frame f segment j lands on output
    // hop-block f + j, so OLA is a sum of r shifted segment planes.
    let r = n / hop;
    let blocks = mlxcel_core::reshape(&windowed, &[batch, frames, r, hop]);
    let mut output: Option<UniquePtr<MlxArray>> = None;
    for j in 0..r {
        let seg = mlxcel_core::slice(&blocks, &[0, 0, j, 0], &[batch, frames, j + 1, hop]);
        let seg = mlxcel_core::reshape(&seg, &[batch, frames, hop]);
        let seg = mlxcel_core::pad(&seg, &[0, 0, j, r - 1 - j, 0, 0], 0.0);
        output = Some(match output {
            None => seg,
            Some(acc) => mlxcel_core::add(&acc, &seg),
        });
    }
    let ola_len = (frames - 1) * hop + n;
    let output = output.ok_or_else(|| "codec istft: empty overlap-add".to_string())?;
    let output = mlxcel_core::reshape(&output, &[batch, ola_len]);

    let norm = window_envelope(&window_host, frames as usize, hop_length);
    let norm = mlxcel_core::from_slice_f32(&norm, &[1, ola_len]);
    Ok(mlxcel_core::divide(&output, &norm))
}

/// Squared-window overlap-add envelope, floored at 1e-10.
fn window_envelope(window: &[f32], frames: usize, hop: usize) -> Vec<f32> {
    let len = (frames - 1) * hop + window.len();
    let mut norm = vec![0.0f32; len];
    for f in 0..frames {
        for (i, w) in window.iter().enumerate() {
            norm[f * hop + i] += w * w;
        }
    }
    norm.iter_mut().for_each(|v| *v = v.max(1e-10));
    norm
}
