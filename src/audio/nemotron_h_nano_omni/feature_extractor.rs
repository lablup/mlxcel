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

//! Parakeet-style log-mel feature extractor for Nemotron H Nano Omni.
//!
//! Faithful Rust port of upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/audio.py (SoundFeatureExtractor)
//! and the underlying `mlx_audio.dsp.stft / hanning / mel_filters` calls.
//!
//! Pipeline:
//! 1. (Optional) preemphasis: `y[0] = x[0]`, `y[i] = x[i] - α x[i-1]`
//! 2. Periodic Hann window of length `win_length`, zero-padded into a
//!    centered `n_fft`-wide window.
//! 3. STFT with `n_fft`-point FFT, hop `hop_length`, constant zero-pad.
//! 4. Power spectrogram (`|X|^2`).
//! 5. Slaney-flavour mel filterbank applied to the power spectrogram.
//! 6. Log-mel with floor `2^-24`.
//! 7. Per-clip mean/variance normalization over **valid** frames only.
//!
//! Differences from Gemma 4 (`crate::audio::feature_extractor`):
//! - Different window length than n_fft (400 vs 512), centered with zeros.
//! - Power spectrogram (Gemma 4 uses magnitude with `mel_floor.ln()`).
//! - Slaney mel scale + Slaney filterbank normalization.
//! - HTK-flavour preemphasis is **not** used here; upstream uses the
//!   "regular" `y[i] = x[i] - α x[i-1]` form.
//! - Per-clip normalization (zero mean, unit std) over valid frames.
//!
//! Used by: Nemotron H Nano Omni VLM (audio modality)
//!
//! TODO: extract a shared FFT helper with `crate::audio::feature_extractor`.

use std::f64::consts::PI;

use super::config::NemotronOmniAudioConfig;

/// Output of one extractor invocation.
#[derive(Debug)]
pub struct NemotronOmniFeatureExtractorOutput {
    /// `[B, T_frames, num_mel_bins]` log-mel features (per-clip
    /// normalized, padded to the longest clip in the batch). Stored
    /// row-major as `f32`.
    pub features: Vec<f32>,
    pub features_shape: [i32; 3],
    /// `[B, T_frames]` int32 attention mask (1 = valid, 0 = padding).
    pub attention_mask: Vec<i32>,
    pub attention_mask_shape: [i32; 2],
    /// `[B]` per-clip total frame count (before padding to `T_frames`),
    /// matching upstream `full_lengths`.
    pub feature_lengths: Vec<i32>,
}

/// Parakeet log-mel feature extractor.
///
/// Cheap to clone — holds the precomputed window and mel filterbank.
#[derive(Clone)]
pub struct NemotronOmniFeatureExtractor {
    sampling_rate: u32,
    hop_length: usize,
    n_fft: usize,
    num_mel_bins: usize,
    preemphasis: f32,
    /// Centered Hann window of length `n_fft` (with `(n_fft - win_length)`
    /// zero-padding split symmetrically around it).
    window: Vec<f32>,
    /// `[num_mel_bins, n_fft/2 + 1]` Slaney-norm mel filterbank.
    mel_filters: Vec<f32>,
}

impl NemotronOmniFeatureExtractor {
    pub fn new(config: &NemotronOmniAudioConfig) -> Self {
        let window = build_centered_hann_window(config.win_length, config.n_fft);
        let mel_filters =
            slaney_mel_filterbank(config.sampling_rate, config.n_fft, config.num_mel_bins);
        Self {
            sampling_rate: config.sampling_rate,
            hop_length: config.hop_length,
            n_fft: config.n_fft,
            num_mel_bins: config.num_mel_bins,
            preemphasis: config.preemphasis,
            window,
            mel_filters,
        }
    }

    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    pub fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Compute log-mel features for a single audio clip.
    ///
    /// Returns `(features, num_frames)` where `features` is
    /// `[num_frames * num_mel_bins]` row-major and per-clip normalized
    /// over the valid range.
    pub fn extract_clip(&self, waveform: &[f32]) -> (Vec<f32>, usize) {
        let mel_floor = (2.0f64).powi(-24);
        let num_freq_bins = self.n_fft / 2 + 1;
        let half_window = self.n_fft / 2;

        // STFT with `pad_mode="constant"`: pad waveform with `n_fft / 2`
        // zeros on each side and frame at hop_length stride.
        let padded_len = waveform.len() + 2 * half_window;
        let num_frames = padded_len / self.hop_length;
        if num_frames == 0 {
            // Edge case: extremely short clip. Match upstream's
            // behaviour of returning a single zero-mel frame.
            return (vec![0.0; self.num_mel_bins], 1);
        }

        // Apply preemphasis once on the unpadded waveform (matches
        // upstream which applies preemphasis to the raw waveform before
        // the STFT pads with zeros).
        let preemphasized = apply_preemphasis(waveform, self.preemphasis);

        let mut padded = vec![0.0f32; padded_len];
        padded[half_window..half_window + preemphasized.len()].copy_from_slice(&preemphasized);

        let mut fft_buf = vec![0.0f64; self.n_fft];
        let mut features = vec![0.0f32; num_frames * self.num_mel_bins];

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            // Slice the windowed segment. STFT may step past the padded
            // length on the last frame; pad short tail with zeros.
            for (i, slot) in fft_buf.iter_mut().enumerate().take(self.n_fft) {
                let src_idx = start + i;
                let sample = if src_idx < padded.len() {
                    padded[src_idx]
                } else {
                    0.0
                };
                *slot = (sample * self.window[i]) as f64;
            }
            let magnitude = real_fft_magnitude(&fft_buf, num_freq_bins);
            // Power spectrogram = |X|^2.
            let power: Vec<f64> = magnitude.iter().map(|m| m * m).collect();
            // Mel filterbank: features[mel] = sum_freq filters[mel, freq] * power[freq]
            for mel_idx in 0..self.num_mel_bins {
                let mut acc = 0.0f64;
                let row = &self.mel_filters[mel_idx * num_freq_bins..(mel_idx + 1) * num_freq_bins];
                for (freq_idx, &p) in power.iter().enumerate() {
                    acc += row[freq_idx] as f64 * p;
                }
                let log_mel = (acc + mel_floor).ln() as f32;
                features[frame_idx * self.num_mel_bins + mel_idx] = log_mel;
            }
        }

        // Per-clip normalization over the valid range. valid_length =
        // min(waveform.len() / hop_length, num_frames).
        let valid_length = (waveform.len() / self.hop_length).min(num_frames);
        normalize_per_clip(&mut features, num_frames, self.num_mel_bins, valid_length);

        (features, num_frames)
    }

    /// Extract a batch of clips, padding to the longest. Mirrors
    /// upstream `SoundFeatureExtractor.__call__`.
    pub fn extract_batch(&self, clips: &[&[f32]]) -> NemotronOmniFeatureExtractorOutput {
        if clips.is_empty() {
            return NemotronOmniFeatureExtractorOutput {
                features: Vec::new(),
                features_shape: [0, 0, self.num_mel_bins as i32],
                attention_mask: Vec::new(),
                attention_mask_shape: [0, 0],
                feature_lengths: Vec::new(),
            };
        }

        let mut per_clip = Vec::with_capacity(clips.len());
        for clip in clips {
            let (feats, frames) = self.extract_clip(clip);
            let valid = (clip.len() / self.hop_length).min(frames);
            per_clip.push((feats, frames, valid));
        }

        let max_len = per_clip.iter().map(|(_, full, _)| *full).max().unwrap_or(0);
        let batch = clips.len();
        let mut features = vec![0.0f32; batch * max_len * self.num_mel_bins];
        let mut attention_mask = vec![0i32; batch * max_len];
        let mut feature_lengths = Vec::with_capacity(batch);

        for (b, (feats, frames, valid)) in per_clip.into_iter().enumerate() {
            let copy_len = frames.min(max_len);
            let feat_dst = &mut features[b * max_len * self.num_mel_bins
                ..b * max_len * self.num_mel_bins + copy_len * self.num_mel_bins];
            feat_dst.copy_from_slice(&feats[..copy_len * self.num_mel_bins]);
            for j in 0..valid.min(max_len) {
                attention_mask[b * max_len + j] = 1;
            }
            feature_lengths.push(frames as i32);
        }

        NemotronOmniFeatureExtractorOutput {
            features,
            features_shape: [batch as i32, max_len as i32, self.num_mel_bins as i32],
            attention_mask,
            attention_mask_shape: [batch as i32, max_len as i32],
            feature_lengths,
        }
    }
}

fn apply_preemphasis(waveform: &[f32], alpha: f32) -> Vec<f32> {
    if waveform.is_empty() {
        return Vec::new();
    }
    if alpha == 0.0 {
        return waveform.to_vec();
    }
    let mut out = Vec::with_capacity(waveform.len());
    out.push(waveform[0]);
    for i in 1..waveform.len() {
        out.push(waveform[i] - alpha * waveform[i - 1]);
    }
    out
}

fn build_centered_hann_window(win_length: usize, n_fft: usize) -> Vec<f32> {
    // Periodic Hann (matches upstream `mlx_audio.dsp.hanning(... periodic=False)`
    // semantics — note the `periodic=False` parameter in upstream is the
    // **NumPy** convention which is the symmetric/periodic (depending on
    // version) — but the upstream STFT then uses this as the window. We
    // mirror the periodic Hann formula `0.5 - 0.5 cos(2π i / N)` because
    // that is what `mlx_audio.dsp.hanning(N, periodic=False)` returns
    // when you trace through the upstream `mlx-audio` code: for STFT use,
    // it produces a length-N window where `w[0] == 0` and the cosine
    // wraps to `2π` at `i == N`. This is equivalent to NumPy's
    // `np.hanning(N+1)[:N]` and aligns with the SoundFeatureExtractor's
    // treatment of `win_length == 400` as a 400-sample window centered
    // inside an `n_fft == 512` zero-padded buffer.
    let mut win = vec![0.0f32; win_length];
    if win_length == 0 {
        // Pathological — return a single 1.0 then center.
        let mut padded = vec![0.0f32; n_fft];
        if !padded.is_empty() {
            padded[n_fft / 2] = 1.0;
        }
        return padded;
    }
    let denom = win_length as f64;
    for (i, slot) in win.iter_mut().enumerate().take(win_length) {
        *slot = (0.5 - 0.5 * ((2.0 * PI * i as f64) / denom).cos()) as f32;
    }
    if win_length >= n_fft {
        return win[..n_fft].to_vec();
    }
    // Center the win_length-window inside an n_fft buffer with zero pad.
    let pad_total = n_fft - win_length;
    let pad_left = pad_total / 2;
    let pad_right = pad_total - pad_left;
    let mut centered = Vec::with_capacity(n_fft);
    centered.extend(std::iter::repeat_n(0.0f32, pad_left));
    centered.extend(win);
    centered.extend(std::iter::repeat_n(0.0f32, pad_right));
    centered
}

/// Slaney-flavour mel filterbank.
///
/// Returns `[num_mel_bins, num_freq_bins]` row-major.
///
/// Implements the same math as `librosa.filters.mel` with
/// `htk=False, norm='slaney'`. The Slaney mel scale is piecewise (linear
/// below 1000 Hz, logarithmic above) and the Slaney normalization
/// scales each filter so the integral under the triangular response is
/// 1.0 — i.e. the filter is multiplied by `2 / (upper - lower)`.
fn slaney_mel_filterbank(sample_rate: u32, n_fft: usize, num_mel_bins: usize) -> Vec<f32> {
    let num_freq_bins = n_fft / 2 + 1;
    let f_min = 0.0;
    let f_max = sample_rate as f64 / 2.0;
    let mel_min = hz_to_slaney_mel(f_min);
    let mel_max = hz_to_slaney_mel(f_max);

    // num_mel_bins + 2 mel-spaced points (filter edges).
    let mut mel_points = Vec::with_capacity(num_mel_bins + 2);
    for i in 0..num_mel_bins + 2 {
        let frac = i as f64 / (num_mel_bins + 1) as f64;
        mel_points.push(mel_min + frac * (mel_max - mel_min));
    }
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| slaney_mel_to_hz(m)).collect();

    // Frequency at each FFT bin centre.
    let bin_freqs: Vec<f64> = (0..num_freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let mut filters = vec![0.0f32; num_mel_bins * num_freq_bins];
    for m in 0..num_mel_bins {
        let lower = hz_points[m];
        let center = hz_points[m + 1];
        let upper = hz_points[m + 2];
        let denom_lower = (center - lower).max(1e-12);
        let denom_upper = (upper - center).max(1e-12);
        let slaney_scale = 2.0 / (upper - lower).max(1e-12);
        for (b, &freq) in bin_freqs.iter().enumerate() {
            let rising = (freq - lower) / denom_lower;
            let falling = (upper - freq) / denom_upper;
            let weight = rising.min(falling).max(0.0);
            filters[m * num_freq_bins + b] = (weight * slaney_scale) as f32;
        }
    }
    filters
}

fn hz_to_slaney_mel(freq: f64) -> f64 {
    // Slaney piecewise mel.
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
    const LOGSTEP: f64 = 0.068_751_777_56; // ln(6.4) / 27 -- librosa default
    let mel = (freq - F_MIN) / F_SP;
    if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / LOGSTEP
    } else {
        mel
    }
}

fn slaney_mel_to_hz(mel: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
    const LOGSTEP: f64 = 0.068_751_777_56;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_MIN + F_SP * mel
    }
}

/// Naive real-FFT magnitude. `input` must have length `n_fft`. Returns
/// `num_bins` complex magnitudes.
///
/// O(N²) is acceptable here because audio inference is offline and the
/// per-clip FFT count is small. The Gemma 4 path uses the same approach.
fn real_fft_magnitude(input: &[f64], num_bins: usize) -> Vec<f64> {
    let n = input.len();
    let mut magnitudes = Vec::with_capacity(num_bins);
    for k in 0..num_bins {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, &sample) in input.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * t as f64 / n as f64;
            re += sample * angle.cos();
            im += sample * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// In-place per-clip mean/variance normalization over the valid frame
/// range. Mirrors upstream's
/// `mel = ((mel - mean) / (sqrt(variance) + 1e-5)) * mask`.
fn normalize_per_clip(
    features: &mut [f32],
    num_frames: usize,
    num_mel: usize,
    valid_length: usize,
) {
    if num_frames == 0 || num_mel == 0 {
        return;
    }
    let denom = valid_length.max(1) as f64;
    let var_denom = valid_length.saturating_sub(1).max(1) as f64;

    // mean over valid frames per mel bin.
    let mut mean = vec![0.0f64; num_mel];
    for t in 0..valid_length.min(num_frames) {
        for m in 0..num_mel {
            mean[m] += features[t * num_mel + m] as f64;
        }
    }
    for m in &mut mean {
        *m /= denom;
    }

    // variance over valid frames per mel bin.
    let mut variance = vec![0.0f64; num_mel];
    for t in 0..valid_length.min(num_frames) {
        for m in 0..num_mel {
            let diff = features[t * num_mel + m] as f64 - mean[m];
            variance[m] += diff * diff;
        }
    }
    for v in &mut variance {
        *v /= var_denom;
    }

    // Apply normalization. Frames outside the valid range get zeroed.
    for t in 0..num_frames {
        if t < valid_length {
            for m in 0..num_mel {
                let std = variance[m].sqrt() as f32 + 1e-5;
                let normalized = (features[t * num_mel + m] - mean[m] as f32) / std;
                features[t * num_mel + m] = normalized;
            }
        } else {
            for m in 0..num_mel {
                features[t * num_mel + m] = 0.0;
            }
        }
    }
}

/// Convenience wrapper that mirrors
/// [`super::config::NemotronOmniAudioConfig::subsampling_output_length`]
/// so callers that only have a `&NemotronOmniFeatureExtractor` (no
/// config in scope) can compute expected encoder output frame counts.
pub fn nemotron_subsampling_output_length(
    config: &NemotronOmniAudioConfig,
    input_frames: usize,
) -> usize {
    config.subsampling_output_length(input_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> NemotronOmniAudioConfig {
        NemotronOmniAudioConfig::default()
    }

    fn generate_tone(freq: f64, duration_s: f64, sr: u32) -> Vec<f32> {
        let n = (duration_s * sr as f64).round() as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect()
    }

    #[test]
    fn one_second_tone_has_expected_frame_count() {
        // 16000 samples + 2 * (n_fft/2)=512 padding -> 16512.
        // num_frames = 16512 / 160 = 103.
        let cfg = make_config();
        let extractor = NemotronOmniFeatureExtractor::new(&cfg);
        let tone = generate_tone(440.0, 1.0, cfg.sampling_rate);
        let (feats, frames) = extractor.extract_clip(&tone);
        assert_eq!(frames, (16_000 + cfg.n_fft) / cfg.hop_length);
        assert_eq!(feats.len(), frames * cfg.num_mel_bins);
    }

    #[test]
    fn batch_extract_pads_to_longest_clip() {
        let cfg = make_config();
        let extractor = NemotronOmniFeatureExtractor::new(&cfg);
        let short = generate_tone(440.0, 0.5, cfg.sampling_rate);
        let long = generate_tone(440.0, 1.0, cfg.sampling_rate);
        let out = extractor.extract_batch(&[&short[..], &long[..]]);
        assert_eq!(out.features_shape[0], 2);
        let max_frames = out.features_shape[1];
        assert!(max_frames > 0);
        assert_eq!(out.attention_mask_shape, [2, max_frames]);
        // The longer clip should be entirely valid; the shorter clip
        // should have a trailing zero region in the mask.
        let valid_long: i32 = out.attention_mask[max_frames as usize..2 * max_frames as usize]
            .iter()
            .sum();
        let valid_short: i32 = out.attention_mask[..max_frames as usize].iter().sum();
        assert!(valid_long > valid_short);
    }

    #[test]
    fn subsampling_output_length_matches_upstream_formula() {
        let cfg = make_config();
        // Upstream formula: for each of log2(8)=3 stages,
        //   add_pad = ((3-1)//2)*2 - 3 = -1
        //   length = floor((length - 1) / 2) + 1
        // Starting from 100 → 50 → 25 → 13.
        assert_eq!(cfg.subsampling_output_length(100), 13);
        assert_eq!(cfg.subsampling_output_length(800), 100);
    }

    #[test]
    fn normalization_zero_means_within_valid_range() {
        let cfg = make_config();
        let extractor = NemotronOmniFeatureExtractor::new(&cfg);
        let tone = generate_tone(440.0, 1.0, cfg.sampling_rate);
        let (feats, frames) = extractor.extract_clip(&tone);
        let valid = (tone.len() / cfg.hop_length).min(frames);
        let mut mean = vec![0.0f64; cfg.num_mel_bins];
        for t in 0..valid {
            for m in 0..cfg.num_mel_bins {
                mean[m] += feats[t * cfg.num_mel_bins + m] as f64;
            }
        }
        for m in &mut mean {
            *m /= valid as f64;
        }
        for &m in &mean {
            assert!(m.abs() < 1e-3, "mean {m} not close to zero");
        }
    }
}
