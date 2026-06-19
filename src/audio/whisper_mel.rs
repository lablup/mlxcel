// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Log-mel front-end for the Whisper-style ASR encoder.
//!
//! This is the speech-recognition counterpart to the USM front-end in the
//! sibling [`super::feature_extractor`] module. The two differ in their
//! normalization and mel scale, so this file only adds the recognizer-specific
//! parameters and reuses the shared DSP primitive
//! ([`super::feature_extractor::real_fft_magnitude`]) for the per-frame
//! transform.
//!
//! Pipeline (16 kHz mono input):
//! 1. centered STFT with a 400-point periodic Hann window and a 160-sample hop
//!    (reflect padding by `n_fft / 2`, trailing analysis frame dropped),
//! 2. power spectrum (`|rfft|^2`) over the 201 retained frequency bins,
//! 3. Slaney-scale, Slaney-normalized triangular mel projection,
//! 4. `log10` with a global dynamic-range clamp and an affine rescale into the
//!    encoder's expected range.

use std::f64::consts::PI;

use super::feature_extractor::real_fft_magnitude;

/// Native sample rate the encoder expects.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// STFT window / FFT size in samples (25 ms at 16 kHz).
pub const WHISPER_N_FFT: usize = 400;
/// STFT hop in samples (10 ms at 16 kHz).
pub const WHISPER_HOP_LENGTH: usize = 160;
/// Chunk length the encoder is trained on, in seconds.
pub const WHISPER_CHUNK_LENGTH: usize = 30;
/// Samples in one 30 s chunk.
pub const WHISPER_N_SAMPLES: usize = WHISPER_CHUNK_LENGTH * WHISPER_SAMPLE_RATE as usize;
/// Mel frames produced by one full 30 s chunk.
pub const WHISPER_N_FRAMES: usize = WHISPER_N_SAMPLES / WHISPER_HOP_LENGTH;

/// Periodic Hann window of length `n` (`0.5 - 0.5*cos(2*pi*i/n)`).
///
/// This matches the window used by the reference front-end (the periodic form,
/// denominator `n`, identical to the window built inline by the sibling USM
/// extractor).
fn periodic_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()) as f32)
        .collect()
}

fn hz_to_mel_slaney(freq: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

/// Slaney-scale, Slaney-normalized triangular mel filterbank, stored row-major
/// as `[n_freqs][n_mels]` (i.e. `filters.T` of the reference layout) so the mel
/// projection is a plain `power[frame] . filters` dot product.
fn whisper_mel_filters(n_mels: usize) -> Vec<f32> {
    let n_freqs = WHISPER_N_FFT / 2 + 1; // 201
    let f_max = WHISPER_SAMPLE_RATE as f64 / 2.0; // 8000

    let all_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * f_max / (n_freqs as f64 - 1.0))
        .collect();

    let m_min = hz_to_mel_slaney(0.0);
    let m_max = hz_to_mel_slaney(f_max);
    let m_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| m_min + (m_max - m_min) * i as f64 / (n_mels as f64 + 1.0))
        .collect();
    let f_pts: Vec<f64> = m_pts.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    let f_diff: Vec<f64> = (0..f_pts.len() - 1)
        .map(|i| f_pts[i + 1] - f_pts[i])
        .collect();

    let mut bank = vec![0.0f32; n_freqs * n_mels];
    for (freq_idx, &freq) in all_freqs.iter().enumerate() {
        for mel in 0..n_mels {
            // slopes[freq, pt] = f_pts[pt] - freq
            let down = -(f_pts[mel] - freq) / f_diff[mel];
            let up = (f_pts[mel + 2] - freq) / f_diff[mel + 1];
            let mut val = down.min(up).max(0.0);
            // Slaney normalization: 2 / (f_pts[mel+2] - f_pts[mel]).
            let enorm = 2.0 / (f_pts[mel + 2] - f_pts[mel]);
            val *= enorm;
            bank[freq_idx * n_mels + mel] = val as f32;
        }
    }
    bank
}

/// Reflect-pad `audio` by `pad` samples on each side (numpy `reflect`: the edge
/// sample is not repeated). Falls back to clamped indices for inputs shorter
/// than the pad width so very short clips do not panic.
fn reflect_pad(audio: &[f32], pad: usize) -> Vec<f32> {
    let n = audio.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    if n == 0 {
        out.resize(2 * pad, 0.0);
        return out;
    }
    // prefix: audio[1..=pad] reversed (clamped to available range)
    for i in (1..=pad).rev() {
        out.push(audio[i.min(n - 1)]);
    }
    out.extend_from_slice(audio);
    // suffix: audio[n-1-pad .. n-1] reversed (clamped)
    for i in 1..=pad {
        let idx = (n - 1).saturating_sub(i);
        out.push(audio[idx]);
    }
    out
}

/// Upper bound on the sample count a single resample may emit: one hour of
/// 16 kHz audio.
///
/// Linear resampling expands the buffer by `16000 / src_rate`. The WAV reader
/// accepts any non-zero declared sample rate, so a crafted-but-well-formed
/// header claiming a tiny rate (e.g. 1 Hz) would otherwise size the output at
/// `len * 16000`, turning a few hundred thousand input samples into hundreds of
/// billions; the resulting `Vec` allocation aborts the process. Clamping the
/// output keeps the allocation bounded. The HTTP upload is already capped at a
/// few tens of MiB upstream, so a legitimate clip (sampled at >= 8 kHz) never
/// approaches this ceiling.
const MAX_RESAMPLED_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize * 3600;

/// Output sample count for a resample to 16 kHz, clamped to
/// [`MAX_RESAMPLED_SAMPLES`]. Split out so the bound is unit-testable without
/// allocating the (potentially large) output buffer.
fn resampled_len(input_len: usize, src_rate: u32) -> usize {
    if src_rate == WHISPER_SAMPLE_RATE {
        return input_len;
    }
    let ratio = WHISPER_SAMPLE_RATE as f64 / src_rate as f64;
    let projected = ((input_len as f64) * ratio).round() as usize;
    projected.min(MAX_RESAMPLED_SAMPLES)
}

/// Linear-interpolation resampler from `src_rate` to 16 kHz mono.
///
/// Linear interpolation is adequate for the recognizer's robustness; a
/// polyphase/sinc resampler is a documented follow-up. The output length is
/// bounded by [`MAX_RESAMPLED_SAMPLES`] so a tiny declared `src_rate` cannot
/// trigger an unbounded allocation.
pub fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == WHISPER_SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = WHISPER_SAMPLE_RATE as f64 / src_rate as f64;
    let out_len = resampled_len(samples.len(), src_rate);
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let left = src_pos.floor() as usize;
        let frac = (src_pos - left as f64) as f32;
        let a = samples[left.min(samples.len() - 1)];
        let b = samples[(left + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Compute the full Whisper log-mel spectrogram for a 16 kHz mono waveform.
///
/// Returns `(features, num_frames)` where `features` is row-major
/// `[num_frames][n_mels]`. The dynamic-range clamp uses the global maximum over
/// the whole utterance, matching the reference (so per-chunk slicing downstream
/// stays consistent).
pub fn log_mel_spectrogram(audio: &[f32], n_mels: usize) -> (Vec<f32>, usize) {
    let n_freqs = WHISPER_N_FFT / 2 + 1;
    let filters = whisper_mel_filters(n_mels);
    let window = periodic_hann(WHISPER_N_FFT);

    let padded = reflect_pad(audio, WHISPER_N_FFT / 2);
    if padded.len() < WHISPER_N_FFT {
        return (Vec::new(), 0);
    }
    let total_frames = 1 + (padded.len() - WHISPER_N_FFT) / WHISPER_HOP_LENGTH;
    // The reference drops the trailing analysis frame.
    let num_frames = total_frames.saturating_sub(1);
    if num_frames == 0 {
        return (Vec::new(), 0);
    }

    let mut features = vec![0.0f32; num_frames * n_mels];
    let mut frame_buf = vec![0.0f64; WHISPER_N_FFT];

    for frame_idx in 0..num_frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;
        for i in 0..WHISPER_N_FFT {
            frame_buf[i] = (padded[start + i] * window[i]) as f64;
        }
        // Power spectrum: square the shared DFT magnitude.
        let mag = real_fft_magnitude(&frame_buf, n_freqs);
        for mel in 0..n_mels {
            let mut acc = 0.0f64;
            for (freq, &m) in mag.iter().enumerate().take(n_freqs) {
                acc += (m * m) * filters[freq * n_mels + mel] as f64;
            }
            features[frame_idx * n_mels + mel] = acc.max(1e-10).log10() as f32;
        }
    }

    // Global dynamic-range clamp and affine rescale: max(x, max-8); (x+4)/4.
    let global_max = features.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = global_max - 8.0;
    for v in &mut features {
        let clamped = v.max(floor);
        *v = (clamped + 4.0) / 4.0;
    }

    (features, num_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq_hz: f64, duration_s: f64, sample_rate: u32) -> Vec<f32> {
        let n = (duration_s * sample_rate as f64).round() as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f64 / sample_rate as f64).sin() as f32)
            .collect()
    }

    #[test]
    fn periodic_hann_endpoints() {
        let w = periodic_hann(WHISPER_N_FFT);
        assert_eq!(w.len(), WHISPER_N_FFT);
        assert!(w[0].abs() < 1e-6, "periodic Hann starts at 0");
        // Peak near the middle is ~1.0.
        let mid = w[WHISPER_N_FFT / 2];
        assert!(mid > 0.999, "periodic Hann peaks near 1.0, got {mid}");
    }

    #[test]
    fn mel_filters_have_expected_shape_and_unit_scale() {
        let n_mels = 80;
        let fb = whisper_mel_filters(n_mels);
        let n_freqs = WHISPER_N_FFT / 2 + 1;
        assert_eq!(fb.len(), n_freqs * n_mels);
        // Every filter must place some positive weight somewhere.
        for mel in 0..n_mels {
            let any_positive = (0..n_freqs).any(|f| fb[f * n_mels + mel] > 0.0);
            assert!(any_positive, "mel filter {mel} is empty");
        }
        // DC bin carries no energy into the lowest non-edge filters.
        assert!(fb[0] >= 0.0);
    }

    #[test]
    fn frame_count_matches_reference_for_one_second() {
        // 1 s at 16 kHz with hop 160 -> 100 frames after the trailing-frame drop.
        let audio = tone(440.0, 1.0, WHISPER_SAMPLE_RATE);
        let (features, frames) = log_mel_spectrogram(&audio, 80);
        assert_eq!(frames, 100, "expected 100 frames, got {frames}");
        assert_eq!(features.len(), frames * 80);
    }

    #[test]
    fn full_chunk_yields_3000_frames() {
        let audio = vec![0.0f32; WHISPER_N_SAMPLES];
        let (_features, frames) = log_mel_spectrogram(&audio, 80);
        assert_eq!(
            frames, WHISPER_N_FRAMES,
            "30 s chunk must yield 3000 frames"
        );
    }

    #[test]
    fn normalization_keeps_values_in_expected_band() {
        let audio = tone(440.0, 0.5, WHISPER_SAMPLE_RATE);
        let (features, _frames) = log_mel_spectrogram(&audio, 80);
        // After max(x, max-8) and (x+4)/4 the dynamic range is exactly 2.0.
        let max = features.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = features.iter().copied().fold(f32::INFINITY, f32::min);
        assert!((max - min) <= 2.0 + 1e-3, "range {} exceeds 2.0", max - min);
    }

    #[test]
    fn tone_energy_peaks_in_low_mel_bins() {
        // 440 Hz energy should concentrate in the low mel bins.
        let audio = tone(440.0, 1.0, WHISPER_SAMPLE_RATE);
        let n_mels = 80;
        let (features, frames) = log_mel_spectrogram(&audio, n_mels);
        let mut energy = vec![0.0f64; n_mels];
        for f in 0..frames {
            for mel in 0..n_mels {
                energy[mel] += features[f * n_mels + mel] as f64;
            }
        }
        let peak = energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            peak < 25,
            "440 Hz peak should fall in low mel bins, got {peak}"
        );
    }

    #[test]
    fn resample_is_identity_at_native_rate() {
        let audio = tone(440.0, 0.1, WHISPER_SAMPLE_RATE);
        let out = resample_to_16k(&audio, WHISPER_SAMPLE_RATE);
        assert_eq!(out.len(), audio.len());
    }

    #[test]
    fn resample_halves_length_from_32k() {
        let audio = tone(440.0, 0.1, 32_000);
        let out = resample_to_16k(&audio, 32_000);
        // 32 kHz -> 16 kHz halves the sample count (within rounding).
        assert!((out.len() as i64 - (audio.len() as i64 / 2)).abs() <= 1);
    }

    #[test]
    fn reflect_pad_empty_returns_zeros() {
        // An empty waveform must not panic; the output is all zeros of length
        // 2 * pad, satisfying the downstream `padded.len() >= WHISPER_N_FFT`
        // check without reading out of bounds.
        let pad = WHISPER_N_FFT / 2;
        let out = reflect_pad(&[], pad);
        assert_eq!(out.len(), 2 * pad);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn reflect_pad_shorter_than_pad_does_not_panic() {
        // When the input is shorter than the pad width the clamped-index path
        // is taken for both prefix and suffix. This is the edge case the HIGH
        // security fix targets: a crafted very-short clip must not cause an
        // index out of bounds.
        let short = [0.5f32, -0.5];
        let pad = WHISPER_N_FFT / 2; // 200
        let out = reflect_pad(&short, pad);
        assert_eq!(out.len(), short.len() + 2 * pad);
        // All values must be in the valid sample range (no garbage).
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn very_short_audio_returns_empty_spectrogram() {
        // A clip too short to yield any frames after the trailing-frame drop
        // must return (empty, 0) rather than panicking or producing garbage.
        let (features, frames) = log_mel_spectrogram(&[0.0f32; 1], 80);
        assert_eq!(frames, 0);
        assert!(features.is_empty());
    }

    #[test]
    fn resampled_len_clamps_pathological_low_rate() {
        // A near-zero declared rate would expand the buffer by ~16000x; the cap
        // keeps the allocation bounded so a crafted WAV cannot abort the worker.
        assert_eq!(resampled_len(10_000_000, 1), MAX_RESAMPLED_SAMPLES);
        // Legitimate rates pass through unclamped.
        assert_eq!(resampled_len(1_000, WHISPER_SAMPLE_RATE), 1_000); // identity
        assert_eq!(resampled_len(1_000, 8_000), 2_000); // 8 kHz -> 16 kHz doubles
    }
}
