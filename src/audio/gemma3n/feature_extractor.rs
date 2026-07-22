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

//! Official Gemma 3n USM waveform processor.

use std::f64::consts::PI;

pub const GEMMA3N_SAMPLE_RATE: u32 = 16_000;
pub const GEMMA3N_MAX_SAMPLES: usize = 480_000;
pub const GEMMA3N_AUDIO_SOFT_TOKENS: usize = 188;
const FEATURE_SIZE: usize = 128;
const FRAME_LENGTH: usize = 512;
const HOP_LENGTH: usize = 160;
const FFT_LENGTH: usize = 1024;
const PAD_MULTIPLE: usize = 128;

#[derive(Debug, Clone)]
pub struct Gemma3nAudioFeatureBatch {
    /// Flattened `[batch, frames, 128]` log-mel values.
    pub features: Vec<f32>,
    /// Flattened `[batch, frames]`; `true` means a real input frame.
    pub valid_mask: Vec<bool>,
    pub batch_size: usize,
    pub frames: usize,
}

pub struct Gemma3nAudioFeatureExtractor {
    window: Vec<f32>,
    mel_filters: Vec<f32>,
}

impl Default for Gemma3nAudioFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemma3nAudioFeatureExtractor {
    pub fn new() -> Self {
        let window = (0..FRAME_LENGTH)
            .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / FRAME_LENGTH as f64).cos()) as f32)
            .collect();
        Self {
            window,
            mel_filters: mel_filter_bank(),
        }
    }

    /// Process already-normalized mono 16 kHz float32 clips as one batch.
    ///
    /// The pinned integration processor truncates each clip at 30 seconds,
    /// left-pads the batch to its longest clip and then to a multiple of 128
    /// samples (`AutoProcessor(..., padding_side="left")`).
    pub fn extract_batch(&self, clips: &[Vec<f32>]) -> Result<Gemma3nAudioFeatureBatch, String> {
        if clips.is_empty() {
            return Err("Gemma3n audio batch must contain at least one clip".into());
        }
        for (index, clip) in clips.iter().enumerate() {
            if clip.is_empty() {
                return Err(format!("Gemma3n audio clip {index} is empty"));
            }
            if clip.iter().any(|sample| !sample.is_finite()) {
                return Err(format!(
                    "Gemma3n audio clip {index} contains a non-finite sample"
                ));
            }
            if clip.iter().any(|sample| !(-1.0..=1.0).contains(sample)) {
                return Err(format!(
                    "Gemma3n audio clip {index} contains a sample outside [-1, 1]"
                ));
            }
        }

        let longest = clips
            .iter()
            .map(|clip| clip.len().min(GEMMA3N_MAX_SAMPLES))
            .max()
            .unwrap_or(0);
        let padded_len = longest.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
        let frames = if padded_len < FRAME_LENGTH + 1 {
            0
        } else {
            (padded_len - (FRAME_LENGTH + 1)) / HOP_LENGTH + 1
        };
        let mut features = Vec::with_capacity(clips.len() * frames * FEATURE_SIZE);
        let mut valid_mask = Vec::with_capacity(clips.len() * frames);

        for clip in clips {
            let effective_len = clip.len().min(GEMMA3N_MAX_SAMPLES);
            let mut waveform = vec![0.0f32; padded_len];
            let left_padding = padded_len - effective_len;
            waveform[left_padding..left_padding + effective_len]
                .copy_from_slice(&clip[..effective_len]);

            for frame_index in 0..frames {
                let start = frame_index * HOP_LENGTH;
                let unfolded = &waveform[start..start + FRAME_LENGTH + 1];
                let mut fft_input = vec![0.0f64; FFT_LENGTH];
                fft_input[0] = (unfolded[0] * 0.03 * self.window[0]) as f64;
                for i in 1..FRAME_LENGTH {
                    let emphasized = unfolded[i] - 0.97 * unfolded[i - 1];
                    fft_input[i] = (emphasized * self.window[i]) as f64;
                }
                if fft_input[..FRAME_LENGTH].iter().all(|value| *value == 0.0) {
                    features.extend(std::iter::repeat_n(1e-5f32.ln(), FEATURE_SIZE));
                    valid_mask.push(start >= left_padding && start < left_padding + effective_len);
                    continue;
                }
                let magnitude = real_fft_magnitude_1024(&fft_input);
                for mel in 0..FEATURE_SIZE {
                    let mut value = 0.0f64;
                    for (bin, magnitude) in magnitude.iter().enumerate() {
                        value += magnitude * self.mel_filters[bin * FEATURE_SIZE + mel] as f64;
                    }
                    features.push(value.max(1e-5).ln() as f32);
                }
                // The pinned processor samples the left-padded waveform
                // attention mask at the frame start, not at the end of the
                // analysis window.
                valid_mask.push(start >= left_padding && start < left_padding + effective_len);
            }
        }

        Ok(Gemma3nAudioFeatureBatch {
            features,
            valid_mask,
            batch_size: clips.len(),
            frames,
        })
    }
}

/// Compute the positive-frequency magnitude spectrum for the fixed 1024-point
/// Gemma 3n transform. The in-tree generic audio helper intentionally uses a
/// direct DFT, which is useful for small reference transforms but makes a
/// 30-second Gemma 3n clip prohibitively expensive. This radix-2 FFT preserves
/// the reference math while keeping frontend preprocessing practical.
fn real_fft_magnitude_1024(input: &[f64]) -> Vec<f64> {
    debug_assert_eq!(input.len(), FFT_LENGTH);
    let mut real = input.to_vec();
    let mut imaginary = vec![0.0f64; FFT_LENGTH];

    let mut reversed = 0usize;
    for index in 1..FFT_LENGTH {
        let mut bit = FFT_LENGTH >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            real.swap(index, reversed);
            imaginary.swap(index, reversed);
        }
    }

    let mut width = 2usize;
    while width <= FFT_LENGTH {
        let half = width / 2;
        let angle = -2.0 * PI / width as f64;
        let step_real = angle.cos();
        let step_imaginary = angle.sin();
        for start in (0..FFT_LENGTH).step_by(width) {
            let mut twiddle_real = 1.0f64;
            let mut twiddle_imaginary = 0.0f64;
            for offset in 0..half {
                let right = start + offset + half;
                let product_real =
                    twiddle_real * real[right] - twiddle_imaginary * imaginary[right];
                let product_imaginary =
                    twiddle_real * imaginary[right] + twiddle_imaginary * real[right];
                let left = start + offset;
                let left_real = real[left];
                let left_imaginary = imaginary[left];
                real[left] = left_real + product_real;
                imaginary[left] = left_imaginary + product_imaginary;
                real[right] = left_real - product_real;
                imaginary[right] = left_imaginary - product_imaginary;
                let next_real = twiddle_real * step_real - twiddle_imaginary * step_imaginary;
                twiddle_imaginary = twiddle_real * step_imaginary + twiddle_imaginary * step_real;
                twiddle_real = next_real;
            }
        }
        width *= 2;
    }

    (0..=FFT_LENGTH / 2)
        .map(|index| real[index].hypot(imaginary[index]))
        .collect()
}

fn mel_filter_bank() -> Vec<f32> {
    let n_freqs = FFT_LENGTH / 2 + 1;
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0);
    let min_mel = hz_to_mel(125.0);
    let max_mel = hz_to_mel(7600.0);
    let points: Vec<f64> = (0..FEATURE_SIZE + 2)
        .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f64 / (FEATURE_SIZE + 1) as f64))
        .collect();
    let mut filters = vec![0.0; n_freqs * FEATURE_SIZE];
    for bin in 0..n_freqs {
        let frequency = bin as f64 * GEMMA3N_SAMPLE_RATE as f64 / FFT_LENGTH as f64;
        for mel in 0..FEATURE_SIZE {
            let down = (frequency - points[mel]) / (points[mel + 1] - points[mel]);
            let up = (points[mel + 2] - frequency) / (points[mel + 2] - points[mel + 1]);
            filters[bin * FEATURE_SIZE + mel] = down.min(up).max(0.0) as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_boundary_emits_2997_mel_frames_and_188_tokens() {
        let batch = Gemma3nAudioFeatureExtractor::new()
            .extract_batch(&[vec![0.0; GEMMA3N_MAX_SAMPLES]])
            .unwrap();
        assert_eq!(batch.frames, 2997);
        let sscp_frames = batch.frames.div_ceil(2).div_ceil(2);
        assert_eq!(sscp_frames.div_ceil(4), GEMMA3N_AUDIO_SOFT_TOKENS);
    }

    #[test]
    fn batch_padding_keeps_clip_boundaries_and_mask() {
        let batch = Gemma3nAudioFeatureExtractor::new()
            .extract_batch(&[vec![0.0; 16_000], vec![0.0; 8_000]])
            .unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.frames, 97);
        assert!(batch.valid_mask[..97].iter().all(|valid| *valid));
        assert!(batch.valid_mask[97..97 + 50].iter().all(|valid| !*valid));
        assert!(batch.valid_mask[97 + 50..].iter().all(|valid| *valid));
    }

    #[test]
    fn rejects_empty_nonfinite_and_out_of_range_clips() {
        let extractor = Gemma3nAudioFeatureExtractor::new();
        assert!(extractor.extract_batch(&[vec![]]).is_err());
        assert!(extractor.extract_batch(&[vec![f32::NAN]]).is_err());
        assert!(extractor.extract_batch(&[vec![1.01]]).is_err());
    }

    #[test]
    fn sub_frame_clip_matches_reference_zero_frame_result() {
        let batch = Gemma3nAudioFeatureExtractor::new()
            .extract_batch(&[vec![0.25; 128]])
            .unwrap();
        assert_eq!(batch.frames, 0);
        assert!(batch.features.is_empty());
        assert!(batch.valid_mask.is_empty());
    }

    #[test]
    fn hard_mel_floor_is_log_one_e_minus_five() {
        let batch = Gemma3nAudioFeatureExtractor::new()
            .extract_batch(&[vec![0.0; 640]])
            .unwrap();
        let expected = 1e-5f32.ln();
        assert!(
            batch
                .features
                .iter()
                .all(|value| (*value - expected).abs() < 1e-6)
        );
    }

    #[test]
    fn clips_longer_than_thirty_seconds_are_truncated_at_the_reference_boundary() {
        let extractor = Gemma3nAudioFeatureExtractor::new();
        let at_limit = extractor
            .extract_batch(&[vec![0.0; GEMMA3N_MAX_SAMPLES]])
            .unwrap();
        let over_limit = extractor
            .extract_batch(&[vec![
                0.0;
                GEMMA3N_MAX_SAMPLES + GEMMA3N_SAMPLE_RATE as usize
            ]])
            .unwrap();
        assert_eq!(over_limit.frames, at_limit.frames);
        assert_eq!(over_limit.features, at_limit.features);
        assert_eq!(over_limit.valid_mask, at_limit.valid_mask);
    }

    #[test]
    fn deterministic_waveform_matches_pinned_transformers_mel_fixture() {
        // Fixture source: Transformers commit
        // 181beb3ba4c47098ed8cbc97ee250d1d45ae0107's NumPy algorithm.
        let waveform: Vec<f32> = (0..1600)
            .map(|index| {
                let time = index as f64 / GEMMA3N_SAMPLE_RATE as f64;
                (0.25 * (2.0 * PI * 440.0 * time).sin() + 0.1 * (2.0 * PI * 1000.0 * time).cos())
                    as f32
            })
            .collect();
        let batch = Gemma3nAudioFeatureExtractor::new()
            .extract_batch(&[waveform])
            .unwrap();
        assert_eq!(batch.frames, 8);
        let fixtures = [
            ((0, 0), -4.375_432),
            ((0, 10), -2.835_216_8),
            ((0, 64), -2.875_424),
            ((1, 20), -0.997_653_1),
            ((3, 100), -11.512_925),
            ((6, 127), -11.512_925),
        ];
        for ((frame, bin), expected) in fixtures {
            let actual = batch.features[frame * FEATURE_SIZE + bin];
            assert!(
                (actual - expected).abs() < 1e-3,
                "mel[{frame},{bin}] = {actual}, expected {expected}"
            );
        }

        // Full-tensor checksums from the same pinned NumPy implementation.
        // The Rust radix-2 FFT and NumPy's pocketfft use different reduction
        // orders, so accumulated tolerances are wider than the per-bin 1e-3.
        let sum: f64 = batch.features.iter().map(|value| *value as f64).sum();
        let squared_sum: f64 = batch
            .features
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum();
        let weighted_sum: f64 = batch
            .features
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f64 * *value as f64)
            .sum();
        assert!((sum - -7_277.967_449_545_86).abs() < 0.1, "sum={sum}");
        assert!(
            (squared_sum - 69_566.958_335_671_2).abs() < 2.0,
            "squared_sum={squared_sum}"
        );
        assert!(
            (weighted_sum - -4_151_055.318_334_281_4).abs() < 100.0,
            "weighted_sum={weighted_sum}"
        );
    }
}
