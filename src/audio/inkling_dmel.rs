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

//! Host-side Inkling discretized-mel (dMel) feature extraction.
//!
//! Inkling consumes one 80-channel soft token every 50 ms. The feature grid is
//! deliberately different from Whisper: magnitude rather than power, a
//! 100-ms non-centered analysis window, and no global dynamic-range transform.

use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;

use super::fft::real_fft_magnitude;

pub const INKLING_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_N_MEL_BINS: usize = 80;
pub const DEFAULT_MEL_VOCAB_SIZE: usize = 16;
pub const DEFAULT_DMEL_MIN: f32 = -7.0;
pub const DEFAULT_DMEL_MAX: f32 = 2.0;
const MAX_GENERIC_DMEL_BINS: usize = 4_096;
pub const INKLING_MLX_VLM_REFERENCE_REVISION: &str =
    "Blaizzy/mlx-vlm@0d6805cfb2429d944ab828473066fd771e00aac6";

fn default_feature_size() -> usize {
    DEFAULT_N_MEL_BINS
}
fn default_sampling_rate() -> u32 {
    INKLING_SAMPLE_RATE
}
fn default_hop() -> usize {
    800
}
fn default_window() -> usize {
    1_600
}
fn default_duration() -> f64 {
    0.05
}
fn default_multiplier() -> f64 {
    2.0
}
fn default_true() -> bool {
    true
}

/// Processor-side feature extractor configuration from `processor_config.json`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InklingFeatureExtractorConfig {
    #[serde(default = "default_feature_size")]
    pub feature_size: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: u32,
    #[serde(default = "default_hop")]
    pub hop_length: usize,
    #[serde(default = "default_window")]
    pub n_fft: usize,
    #[serde(default = "default_window")]
    pub window_size: usize,
    #[serde(default = "default_duration")]
    pub audio_token_duration_s: f64,
    #[serde(default = "default_multiplier")]
    pub window_size_multiplier: f64,
    #[serde(default)]
    pub padding_value: f32,
    #[serde(default = "default_true")]
    pub return_attention_mask: bool,
}

impl Default for InklingFeatureExtractorConfig {
    fn default() -> Self {
        Self {
            feature_size: default_feature_size(),
            sampling_rate: default_sampling_rate(),
            hop_length: default_hop(),
            n_fft: default_window(),
            window_size: default_window(),
            audio_token_duration_s: default_duration(),
            window_size_multiplier: default_multiplier(),
            padding_value: 0.0,
            return_attention_mask: true,
        }
    }
}

impl InklingFeatureExtractorConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.feature_size != DEFAULT_N_MEL_BINS {
            return Err(format!(
                "Inkling audio feature_size must be {DEFAULT_N_MEL_BINS}, got {}",
                self.feature_size
            ));
        }
        if self.sampling_rate != INKLING_SAMPLE_RATE {
            return Err(format!(
                "Inkling audio extractor requires {INKLING_SAMPLE_RATE} Hz, got {} Hz",
                self.sampling_rate
            ));
        }
        if self.hop_length != default_hop()
            || self.window_size != default_window()
            || self.n_fft != default_window()
        {
            return Err("Inkling audio requires hop_length 800 and window_size/n_fft 1600".into());
        }
        let computed_hop = self.audio_token_duration_s * self.sampling_rate as f64;
        let computed_window = computed_hop * self.window_size_multiplier;
        if !computed_hop.is_finite()
            || !computed_window.is_finite()
            || (computed_hop - self.hop_length as f64).abs() > 1e-6
            || (computed_window - self.window_size as f64).abs() > 1e-6
        {
            return Err("Inkling audio duration fields disagree with sample sizes".into());
        }
        if self.n_fft > super::fft::MAX_REAL_FFT_LEN {
            return Err("Inkling audio n_fft exceeds the bounded host FFT limit".into());
        }
        if self.padding_value != 0.0 {
            return Err("Inkling audio padding_value must be zero".into());
        }
        if !self.return_attention_mask {
            return Err("Inkling audio requires valid-frame masks".into());
        }
        Ok(())
    }
}

/// A padded batch in row-major `[batch, frames, channels]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct InklingLogMelBatch {
    pub features: Vec<f32>,
    pub mask: Vec<bool>,
    pub batch_size: usize,
    pub frames_per_clip: usize,
    pub feature_size: usize,
    pub valid_frames: Vec<usize>,
}

/// Valid-only rows in clip order. Unlike the padded batch shape, this never
/// contains right-padding rows and therefore never spends an FFT on padding
/// introduced by another, longer clip in the request.
#[derive(Debug, Clone, PartialEq)]
pub struct InklingCompactLogMelBatch {
    pub features: Vec<f32>,
    pub feature_size: usize,
    pub valid_frames: Vec<usize>,
    pub transformed_frames: usize,
}

#[derive(Debug, Clone)]
pub struct InklingFeatureExtractor {
    config: InklingFeatureExtractorConfig,
    window: Vec<f32>,
    /// Row-major `[mel, frequency]`, computed in f64 and cast once.
    filters: Vec<f32>,
}

impl InklingFeatureExtractor {
    pub fn new(config: InklingFeatureExtractorConfig) -> Result<Self, String> {
        config.validate()?;
        let mut window = vec![0.0f32; config.n_fft];
        let left = (config.n_fft - config.window_size) / 2;
        for index in 0..config.window_size {
            window[left + index] =
                (0.5 - 0.5 * (2.0 * PI * index as f64 / config.window_size as f64).cos()) as f32;
        }
        let filters = slaney_mel_filters(config.sampling_rate, config.n_fft, config.feature_size);
        Ok(Self {
            config,
            window,
            filters,
        })
    }

    #[must_use]
    pub fn config(&self) -> &InklingFeatureExtractorConfig {
        &self.config
    }

    pub fn extract_log_mel(&self, clip: &[f32]) -> Result<(Vec<f32>, Vec<bool>), String> {
        let batch = self.extract_batch(&[clip])?;
        Ok((batch.features, batch.mask))
    }

    pub fn extract_batch(&self, clips: &[&[f32]]) -> Result<InklingLogMelBatch, String> {
        if clips.is_empty() {
            return Err("Inkling audio batch must contain at least one clip".into());
        }
        if clips.iter().any(|clip| clip.is_empty()) {
            return Err("Inkling audio clips must contain at least one sample".into());
        }
        if let Some((clip, sample)) = clips.iter().enumerate().find_map(|(clip, samples)| {
            samples
                .iter()
                .position(|sample| !sample.is_finite())
                .map(|sample| (clip, sample))
        }) {
            return Err(format!(
                "Inkling audio clip {clip} contains a non-finite sample at {sample}"
            ));
        }
        let padded_length = clips.iter().map(|clip| clip.len()).max().unwrap_or(0);
        let right_pad = (self.config.hop_length - padded_length % self.config.hop_length)
            % self.config.hop_length;
        let left_pad = self.config.n_fft.saturating_sub(self.config.hop_length);
        let total = left_pad
            .checked_add(padded_length)
            .and_then(|value| value.checked_add(right_pad))
            .ok_or_else(|| "Inkling audio padded length overflowed".to_string())?;
        if total < self.config.n_fft {
            return Err("Inkling audio clip is too short for configured framing".into());
        }
        let frames = 1 + (total - self.config.n_fft) / self.config.hop_length;
        let feature_count = clips
            .len()
            .checked_mul(frames)
            .and_then(|value| value.checked_mul(self.config.feature_size))
            .ok_or_else(|| "Inkling audio feature allocation overflowed".to_string())?;
        let mask_count = clips
            .len()
            .checked_mul(frames)
            .ok_or_else(|| "Inkling audio mask allocation overflowed".to_string())?;
        let mut features = vec![self.config.padding_value; feature_count];
        let mut mask = vec![false; mask_count];
        let mut valid_frames = Vec::with_capacity(clips.len());
        for (clip_index, clip) in clips.iter().enumerate() {
            let valid = clip.len().div_ceil(self.config.hop_length);
            valid_frames.push(valid);
            let mut frame_scratch = vec![0.0f64; self.config.n_fft];
            for frame in 0..frames {
                if frame < valid {
                    mask[clip_index * frames + frame] = true;
                }
                let output = (clip_index * frames + frame) * self.config.feature_size;
                self.extract_frame(
                    clip,
                    frame,
                    left_pad,
                    &mut frame_scratch,
                    &mut features[output..],
                )?;
                if frame >= valid {
                    features[output..output + self.config.feature_size]
                        .fill(self.config.padding_value);
                }
            }
        }
        Ok(InklingLogMelBatch {
            features,
            mask,
            batch_size: clips.len(),
            frames_per_clip: frames,
            feature_size: self.config.feature_size,
            valid_frames,
        })
    }

    /// Extract only valid rows, with a hard limit on the number of FFTs and a
    /// cancellation check before every frame transform.
    pub fn extract_compact_batch_cancellable(
        &self,
        clips: &[&[f32]],
        max_frames: usize,
        cancelled: &AtomicBool,
    ) -> Result<InklingCompactLogMelBatch, String> {
        self.extract_compact_batch_with_cancel(clips, max_frames, || {
            cancelled.load(Ordering::Acquire)
        })
    }

    fn extract_compact_batch_with_cancel<C>(
        &self,
        clips: &[&[f32]],
        max_frames: usize,
        mut is_cancelled: C,
    ) -> Result<InklingCompactLogMelBatch, String>
    where
        C: FnMut() -> bool,
    {
        if clips.is_empty() {
            return Err("Inkling audio batch must contain at least one clip".into());
        }
        if clips.iter().any(|clip| clip.is_empty()) {
            return Err("Inkling audio clips must contain at least one sample".into());
        }
        if let Some((clip, sample)) = clips.iter().enumerate().find_map(|(clip, samples)| {
            samples
                .iter()
                .position(|sample| !sample.is_finite())
                .map(|sample| (clip, sample))
        }) {
            return Err(format!(
                "Inkling audio clip {clip} contains a non-finite sample at {sample}"
            ));
        }

        let valid_frames = clips
            .iter()
            .map(|clip| clip.len().div_ceil(self.config.hop_length))
            .collect::<Vec<_>>();
        let total_frames = valid_frames
            .iter()
            .try_fold(0usize, |total, frames| total.checked_add(*frames))
            .ok_or_else(|| "Inkling valid frame count overflowed".to_string())?;
        if total_frames > max_frames {
            return Err(format!(
                "Inkling audio requires {total_frames} frame transforms, exceeding the request limit {max_frames}"
            ));
        }
        let feature_count = total_frames
            .checked_mul(self.config.feature_size)
            .ok_or_else(|| "Inkling compact feature allocation overflowed".to_string())?;
        let mut features = vec![self.config.padding_value; feature_count];
        let left_pad = self.config.n_fft.saturating_sub(self.config.hop_length);
        let mut transformed_frames = 0usize;
        let mut frame_scratch = vec![0.0f64; self.config.n_fft];
        for (clip, &frames) in clips.iter().zip(&valid_frames) {
            for frame in 0..frames {
                if is_cancelled() {
                    return Err("Inkling audio feature extraction was cancelled".into());
                }
                let output = transformed_frames
                    .checked_mul(self.config.feature_size)
                    .ok_or_else(|| "Inkling compact feature offset overflowed".to_string())?;
                self.extract_frame(
                    clip,
                    frame,
                    left_pad,
                    &mut frame_scratch,
                    &mut features[output..output + self.config.feature_size],
                )?;
                transformed_frames += 1;
            }
        }
        debug_assert_eq!(transformed_frames, total_frames);
        Ok(InklingCompactLogMelBatch {
            features,
            feature_size: self.config.feature_size,
            valid_frames,
            transformed_frames,
        })
    }

    fn extract_frame(
        &self,
        clip: &[f32],
        frame_index: usize,
        left_pad: usize,
        frame: &mut [f64],
        output: &mut [f32],
    ) -> Result<(), String> {
        let frame_start = frame_index
            .checked_mul(self.config.hop_length)
            .ok_or_else(|| "Inkling audio frame offset overflowed".to_string())?;
        frame.fill(0.0);
        for (index, slot) in frame.iter_mut().enumerate() {
            let padded_index = frame_start + index;
            if padded_index >= left_pad {
                let source = padded_index - left_pad;
                if source < clip.len() {
                    *slot = (clip[source] * self.window[index]) as f64;
                }
            }
        }
        let bins = self.config.n_fft / 2 + 1;
        let magnitude = real_fft_magnitude(frame, bins);
        if magnitude.len() != bins {
            return Err("Inkling audio FFT failed its bounded-size contract".into());
        }
        for (mel, slot) in output.iter_mut().take(self.config.feature_size).enumerate() {
            let mut sum = 0.0f32;
            for (frequency, value) in magnitude.iter().enumerate() {
                sum += (*value as f32).max(1e-10) * self.filters[mel * bins + frequency];
            }
            *slot = sum.max(1e-10).log10();
        }
        Ok(())
    }
}

/// Exact dMel bin boundaries represented on the f32 lattice.
#[must_use]
pub fn dmel_boundaries(num_bins: usize, min_value: f32, max_value: f32) -> Vec<f32> {
    if !(2..=MAX_GENERIC_DMEL_BINS).contains(&num_bins)
        || !min_value.is_finite()
        || !max_value.is_finite()
        || min_value >= max_value
    {
        return Vec::new();
    }
    let min = min_value as f64;
    let span = max_value as f64 - min;
    (0..num_bins - 1)
        .map(|index| {
            let lower = min + span * index as f64 / (num_bins - 1) as f64;
            let upper = min + span * (index + 1) as f64 / (num_bins - 1) as f64;
            let midpoint = (lower + upper) * 0.5;
            let rounded = midpoint as f32;
            if rounded as f64 > midpoint {
                next_f32_down(rounded)
            } else {
                rounded
            }
        })
        .collect()
}

pub fn quantize_dmel(
    log_mel: &[f32],
    num_bins: usize,
    min_value: f32,
    max_value: f32,
) -> Result<Vec<i32>, String> {
    let boundaries = dmel_boundaries(num_bins, min_value, max_value);
    if boundaries.len() + 1 != num_bins {
        return Err(
            "Inkling dMel quantizer requires finite increasing bounds and at least two bins".into(),
        );
    }
    if log_mel.iter().any(|value| !value.is_finite()) {
        return Err("Inkling dMel quantizer received a non-finite feature".into());
    }
    Ok(log_mel
        .iter()
        .map(|value| {
            let value = value.clamp(min_value, max_value);
            boundaries
                .iter()
                .filter(|boundary| value > **boundary)
                .count() as i32
        })
        .collect())
}

fn next_f32_down(value: f32) -> f32 {
    if value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

fn hz_to_mel_slaney(frequency: f64) -> f64 {
    let spacing = 200.0 / 3.0;
    let log_frequency = 1_000.0;
    let log_mel = log_frequency / spacing;
    let log_step = 6.4f64.ln() / 27.0;
    if frequency >= log_frequency {
        log_mel + (frequency / log_frequency).ln() / log_step
    } else {
        frequency / spacing
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let spacing = 200.0 / 3.0;
    let log_frequency = 1_000.0;
    let log_mel = log_frequency / spacing;
    let log_step = 6.4f64.ln() / 27.0;
    if mel >= log_mel {
        log_frequency * (log_step * (mel - log_mel)).exp()
    } else {
        spacing * mel
    }
}

fn slaney_mel_filters(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let maximum_frequency = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(maximum_frequency);
    let points: Vec<f64> = (0..n_mels + 2)
        .map(|index| {
            let mel = mel_min + (mel_max - mel_min) * index as f64 / (n_mels + 1) as f64;
            mel_to_hz_slaney(mel)
        })
        .collect();
    let mut filters = vec![0.0f32; n_mels * bins];
    for mel in 0..n_mels {
        let lower = points[mel];
        let center = points[mel + 1];
        let upper = points[mel + 2];
        let normalization = 2.0 / (upper - lower);
        for frequency in 0..bins {
            let hz = frequency as f64 * maximum_frequency / (bins - 1) as f64;
            let rising = (hz - lower) / (center - lower);
            let falling = (upper - hz) / (upper - center);
            filters[mel * bins + frequency] = (rising.min(falling).max(0.0) * normalization) as f32;
        }
    }
    filters
}

#[cfg(test)]
#[path = "inkling_dmel_tests.rs"]
mod tests;
