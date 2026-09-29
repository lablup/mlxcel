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

//! NeMo-style log-mel frontend for the Nemotron FastConformer speech encoder.
//!
//! Port of `log_mel_spectrogram` from
//! `mlx_audio/stt/models/nemotron_asr/audio.py` (NeMo's
//! `AudioToMelSpectrogramPreprocessor` at inference time):
//!
//! 1. optional `pad_to` padding with `pad_value`,
//! 2. preemphasis `y[n] = x[n] - 0.97 x[n-1]` (first sample unchanged),
//! 3. centered STFT: reflect-pad `n_fft / 2` samples on both sides, frames of
//!    `n_fft` samples every `hop` samples, a **symmetric** Hann window of
//!    `win_length` samples zero-padded to `n_fft`,
//! 4. power spectrum `|X|^2`,
//! 5. Slaney-scale, Slaney-normalized mel filterbank,
//! 6. natural `ln(mel + log_zero_guard_value)`,
//! 7. normalization: `"NA"` (the VoiceChat checkpoint) is a no-op; the
//!    `per_feature` / `all_features` variants are kept for config parity.
//!
//! Dither is training-only in NeMo and is not applied. The output has
//! `T = 1 + floor(len / hop)` frames, row-major `[T, features]`.
//!
//! The per-frame math lives in [`NemotronMelFrontend::frame_into`], which takes
//! one already-preemphasized, already-padded `n_fft` segment, so a streaming
//! wrapper can feed frames incrementally without duplicating the DSP.
//!
//! [`StreamingLogMel`] (in `nemotron_mel_stream.rs`) is the incremental
//! variant used by the streaming session.
//!
//! Used by: NemotronLabs VoiceChat speech perception.

use std::f64::consts::PI;

use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use super::fft::real_fft_magnitude;
use super::whisper_mel::{hz_to_mel_slaney, mel_to_hz_slaney, reflect_pad};

#[path = "nemotron_mel_stream.rs"]
mod stream;

pub use stream::StreamingLogMel;

/// Mel-spectrogram featurizer settings (`audio_config.preprocessor`).
///
/// Defaults equal the NemotronLabs VoiceChat checkpoint values.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct MelArgs {
    pub sample_rate: u32,
    pub features: usize,
    pub n_fft: usize,
    /// Window length in seconds (0.025 s = 400 samples at 16 kHz).
    pub window_size: f64,
    /// Hop in seconds (0.01 s = 160 samples at 16 kHz).
    pub window_stride: f64,
    pub window: String,
    pub preemph: f32,
    /// Training-only in NeMo; never applied here.
    pub dither: f32,
    /// `"NA"` disables normalization.
    pub normalize: String,
    pub log: bool,
    pub log_zero_guard_value: f32,
    pub pad_to: usize,
    pub pad_value: f32,
}

impl Default for MelArgs {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            features: 128,
            n_fft: 512,
            window_size: 0.025,
            window_stride: 0.01,
            window: "hann".to_string(),
            preemph: 0.97,
            dither: 1e-5,
            normalize: "NA".to_string(),
            log: true,
            log_zero_guard_value: 2f32.powi(-24),
            pad_to: 0,
            pad_value: 0.0,
        }
    }
}

impl MelArgs {
    /// Window length in samples (`int(window_size * sample_rate)`).
    pub fn win_length(&self) -> usize {
        (self.window_size * self.sample_rate as f64) as usize
    }

    /// Hop length in samples (`int(window_stride * sample_rate)`).
    pub fn hop_length(&self) -> usize {
        (self.window_stride * self.sample_rate as f64) as usize
    }

    /// Number of frames the centered STFT produces for `samples` input samples.
    pub fn num_frames(&self, samples: usize) -> usize {
        let samples = if self.pad_to > 0 {
            samples.max(self.pad_to)
        } else {
            samples
        };
        1 + samples / self.hop_length().max(1)
    }
}

/// Precomputed window and filterbank for one [`MelArgs`].
#[derive(Debug, Clone)]
pub struct NemotronMelFrontend {
    args: MelArgs,
    hop: usize,
    /// `[n_fft]` symmetric window, center-padded.
    window: Vec<f32>,
    /// `[features][n_fft / 2 + 1]` row-major.
    filters: Vec<f32>,
}

impl NemotronMelFrontend {
    pub fn new(args: &MelArgs) -> Result<Self, String> {
        let hop = args.hop_length();
        if hop == 0 || args.n_fft < 2 || args.features == 0 {
            return Err(format!(
                "invalid mel settings: hop {hop}, n_fft {}, features {}",
                args.n_fft, args.features
            ));
        }
        let win_length = args.win_length();
        if win_length == 0 || win_length > args.n_fft {
            return Err(format!(
                "mel window length {win_length} must be in 1..={}",
                args.n_fft
            ));
        }
        if !args.log {
            return Err("mel frontend requires log = true".to_string());
        }
        match args.normalize.as_str() {
            "NA" | "per_feature" | "all_features" => {}
            other => return Err(format!("unsupported mel normalize mode {other:?}")),
        }
        Ok(Self {
            args: args.clone(),
            hop,
            window: centered_window(&args.window, win_length, args.n_fft)?,
            filters: slaney_filterbank(args.sample_rate, args.n_fft, args.features),
        })
    }

    pub fn args(&self) -> &MelArgs {
        &self.args
    }

    /// Compute the log-mel values of one `n_fft`-sample segment (already
    /// preemphasized and padded) into `out[..features]`.
    pub fn frame_into(&self, segment: &[f32], out: &mut [f32]) {
        let n_fft = self.args.n_fft;
        let n_bins = n_fft / 2 + 1;
        let buf: Vec<f64> = (0..n_fft)
            .map(|i| (segment.get(i).copied().unwrap_or(0.0) * self.window[i]) as f64)
            .collect();
        let magnitude = real_fft_magnitude(&buf, n_bins);
        // The reference casts the power spectrum back to f32 before the mel
        // projection.
        let power: Vec<f32> = magnitude.iter().map(|m| (m * m) as f32).collect();
        let guard = self.args.log_zero_guard_value;
        for (mel, slot) in out.iter_mut().enumerate().take(self.args.features) {
            let row = &self.filters[mel * n_bins..(mel + 1) * n_bins];
            let acc: f64 = row
                .iter()
                .zip(&power)
                .map(|(&f, &p)| f as f64 * p as f64)
                .sum();
            *slot = (acc as f32 + guard).ln();
        }
    }

    /// Full-utterance log-mel: returns `(features [T, F] row-major, T)`.
    pub fn compute(&self, x: &[f32]) -> Result<(Vec<f32>, usize), String> {
        let mut samples = x.to_vec();
        if self.args.pad_to > 0 && samples.len() < self.args.pad_to {
            samples.resize(self.args.pad_to, self.args.pad_value);
        }
        if samples.is_empty() {
            return Err("log-mel input is empty".to_string());
        }
        let emphasized = preemphasize(&samples, self.args.preemph);
        let half = self.args.n_fft / 2;
        let padded = reflect_pad(&emphasized, half);
        let frames = 1 + samples.len() / self.hop;
        let feats = self.args.features;
        let mut out = vec![0.0f32; frames * feats];
        for (t, row) in out.chunks_exact_mut(feats).enumerate() {
            let start = t * self.hop;
            let end = (start + self.args.n_fft).min(padded.len());
            self.frame_into(&padded[start..end], row);
        }
        match self.args.normalize.as_str() {
            "per_feature" => normalize_per_feature(&mut out, frames, feats),
            "all_features" => normalize_all_features(&mut out),
            _ => {}
        }
        Ok((out, frames))
    }
}

/// Compute `(features [T, F] row-major, T)` for a mono waveform.
pub fn log_mel_spectrogram(x: &[f32], args: &MelArgs) -> Result<(Vec<f32>, usize), String> {
    NemotronMelFrontend::new(args)?.compute(x)
}

/// Compute the log-mel features as an f32 MLX array `[1, T, features]`.
pub fn log_mel_array(x: &[f32], args: &MelArgs) -> Result<UniquePtr<MlxArray>, String> {
    let (features, frames) = log_mel_spectrogram(x, args)?;
    Ok(mlxcel_core::from_slice_f32(
        &features,
        &[1, frames as i32, args.features as i32],
    ))
}

/// Preemphasis high-pass filter; the first sample passes through unchanged.
pub(crate) fn preemphasize(x: &[f32], alpha: f32) -> Vec<f32> {
    if alpha <= 0.0 || x.is_empty() {
        return x.to_vec();
    }
    let mut out = Vec::with_capacity(x.len());
    out.push(x[0]);
    out.extend(x.windows(2).map(|w| w[1] - alpha * w[0]));
    out
}

/// Symmetric window of `win_length` samples (denominator `win_length - 1`,
/// `mlx_audio.dsp.hanning(n, periodic=False)`), zero-padded to `n_fft` with
/// the extra zeros split `floor` left / `ceil` right.
pub(crate) fn centered_window(
    kind: &str,
    win_length: usize,
    n_fft: usize,
) -> Result<Vec<f32>, String> {
    let denom = win_length.saturating_sub(1).max(1) as f64;
    let window: Vec<f32> = (0..win_length)
        .map(|n| {
            let c = (2.0 * PI * n as f64 / denom).cos();
            match kind {
                "hamming" => 0.54 - 0.46 * c,
                _ => 0.5 * (1.0 - c),
            }
        })
        .map(|v| v as f32)
        .collect();
    if !matches!(kind, "hann" | "hanning" | "hamming") {
        return Err(format!("unsupported mel window {kind:?}"));
    }
    let left = (n_fft - win_length) / 2;
    let mut padded = vec![0.0f32; n_fft];
    padded[left..left + win_length].copy_from_slice(&window);
    Ok(padded)
}

/// Slaney-scale, Slaney-normalized filterbank as `[n_mels][n_fft/2 + 1]`.
pub(crate) fn slaney_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let f_max = sample_rate as f64 / 2.0;
    // `mx.linspace(0, sample_rate // 2, n_freqs)`.
    let top = (sample_rate / 2) as f64;
    let freqs: Vec<f64> = (0..n_bins)
        .map(|i| top * i as f64 / (n_bins as f64 - 1.0))
        .collect();
    let m_min = hz_to_mel_slaney(0.0);
    let m_max = hz_to_mel_slaney(f_max);
    let f_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(m_min + (m_max - m_min) * i as f64 / (n_mels as f64 + 1.0)))
        .collect();
    let mut bank = vec![0.0f32; n_mels * n_bins];
    for mel in 0..n_mels {
        let lower = f_pts[mel + 1] - f_pts[mel];
        let upper = f_pts[mel + 2] - f_pts[mel + 1];
        let enorm = 2.0 / (f_pts[mel + 2] - f_pts[mel]);
        for (bin, &freq) in freqs.iter().enumerate() {
            let down = (freq - f_pts[mel]) / lower;
            let up = (f_pts[mel + 2] - freq) / upper;
            bank[mel * n_bins + bin] = (down.min(up).max(0.0) * enorm) as f32;
        }
    }
    bank
}

/// NeMo `per_feature`: per mel bin, subtract the mean over time and divide by
/// `std + 1e-5` (unbiased variance).
fn normalize_per_feature(x: &mut [f32], frames: usize, feats: usize) {
    let n = frames.saturating_sub(1).max(1) as f64;
    for f in 0..feats {
        let mean = (0..frames).map(|t| x[t * feats + f] as f64).sum::<f64>() / frames as f64;
        let var = (0..frames)
            .map(|t| (x[t * feats + f] as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let denom = var.sqrt() + 1e-5;
        for t in 0..frames {
            let v = &mut x[t * feats + f];
            *v = ((*v as f64 - mean) / denom) as f32;
        }
    }
}

/// NeMo `all_features`: global mean / population std.
fn normalize_all_features(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let len = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / len;
    let std = (x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / len).sqrt();
    for v in x.iter_mut() {
        *v = ((*v as f64 - mean) / (std + 1e-5)) as f32;
    }
}

#[cfg(test)]
#[path = "nemotron_mel_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "nemotron_mel_stream_tests.rs"]
mod stream_tests;
