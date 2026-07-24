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

use std::f64::consts::PI;

use mlxcel_core::{MlxArray, UniquePtr};

use crate::audio::{AudioCancellation, AudioPreprocessCheckpoint};

const FEATURE_SIZE: usize = 80;
const PREEMPHASIS: f32 = 0.97;
const INPUT_SCALE: f32 = 32_768.0;
const COMPRESSION_RATE: usize = 8;

/// Official documentation recommends clips up to 40 seconds for best quality;
/// the model card also documents summarization up to 30 minutes. The latter is
/// the hard request limit so oversized payloads fail before frame allocation.
pub const MAX_AUDIO_DURATION_SECONDS: usize = 30 * 60;

pub struct Phi4MMAudioBatch {
    /// One `[1, frames, 80]` tensor per clip. Separate tensors preserve clip
    /// boundaries without padding influencing the causal convolution frontend.
    pub clips: Vec<UniquePtr<MlxArray>>,
    pub frame_lengths: Vec<usize>,
    pub embed_sizes: Vec<usize>,
}

pub const fn audio_embed_size(frames: usize) -> usize {
    frames.div_ceil(COMPRESSION_RATE)
}

pub struct Phi4MMAudioFeatureExtractor {
    mel_filters: Vec<f32>, // [257, 80], matching SpeechLib FbankFC
}

impl Default for Phi4MMAudioFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Phi4MMAudioFeatureExtractor {
    pub fn new() -> Self {
        Self {
            mel_filters: speechlib_mel(),
        }
    }

    pub fn extract_batch(&self, audios: &[(Vec<f32>, u32)]) -> Result<Phi4MMAudioBatch, String> {
        self.extract_batch_cancellable(audios, &std::sync::atomic::AtomicBool::new(false))
    }

    pub fn extract_batch_cancellable(
        &self,
        audios: &[(Vec<f32>, u32)],
        cancelled: &dyn AudioCancellation,
    ) -> Result<Phi4MMAudioBatch, String> {
        check_feature_cancelled(cancelled)?;
        if audios.is_empty() {
            return Err("Phi4MM audio input is empty".into());
        }
        let mut clips = Vec::with_capacity(audios.len());
        let mut frame_lengths = Vec::with_capacity(audios.len());
        let mut embed_sizes = Vec::with_capacity(audios.len());
        for (index, (samples, sample_rate)) in audios.iter().enumerate() {
            check_feature_cancelled(cancelled)?;
            let (features, frames) = self
                .extract_clip_cancellable(samples, *sample_rate, cancelled)
                .map_err(|err| format!("audio clip {}: {err}", index + 1))?;
            clips.push(features);
            frame_lengths.push(frames);
            embed_sizes.push(audio_embed_size(frames));
        }
        Ok(Phi4MMAudioBatch {
            clips,
            frame_lengths,
            embed_sizes,
        })
    }

    #[cfg(test)]
    fn extract_clip(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<(UniquePtr<MlxArray>, usize), String> {
        self.extract_clip_cancellable(
            samples,
            sample_rate,
            &std::sync::atomic::AtomicBool::new(false),
        )
    }

    fn extract_clip_cancellable(
        &self,
        samples: &[f32],
        sample_rate: u32,
        cancelled: &dyn AudioCancellation,
    ) -> Result<(UniquePtr<MlxArray>, usize), String> {
        check_feature_cancelled(cancelled)?;
        validate_waveform(samples.len(), sample_rate)?;
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err("audio waveform contains a non-finite sample".into());
        }

        // Preserve the pinned processor's integer-ratio resampling contract.
        // In particular, 16_001..31_999 Hz uses a divisor of one and is merely
        // relabeled as 16 kHz; changing this to an ideal rational resampler
        // changes the official checkpoint's frame count and greedy output.
        let (waveform, effective_rate) =
            crate::audio::preprocessing::wav::phi4mm_resample(samples, sample_rate)?;
        let waveform = waveform.as_ref();

        let (win_length, hop_length, n_fft) = if effective_rate == 8_000 {
            (200usize, 80usize, 256usize)
        } else {
            (400usize, 160usize, 512usize)
        };
        if waveform.len() < win_length + hop_length {
            return Err("audio is too short: at least two analysis frames are required".into());
        }
        let frames = (waveform.len() - win_length) / hop_length + 1;
        if frames < 2 {
            return Err("audio is too short: at least two analysis frames are required".into());
        }

        // NumPy `hamming(N)` is symmetric (denominator N-1).
        let window: Vec<f32> = (0..win_length)
            .map(|i| (0.54 - 0.46 * (2.0 * PI * i as f64 / (win_length - 1) as f64).cos()) as f32)
            .collect();
        let mut framed = vec![0.0f32; frames * n_fft];
        for frame_index in 0..frames {
            if frame_index % 64 == 0 {
                check_feature_cancelled(cancelled)?;
            }
            let start = frame_index * hop_length;
            let frame = &waveform[start..start + win_length];
            for i in 0..win_length {
                // `np.roll(frame, 1)` puts frame[0] at index 1, then the
                // reference assigns prev[0] = prev[1]. Thus the first sample
                // is preemphasized against itself, not frame[1].
                let previous = if i == 0 { frame[0] } else { frame[i - 1] };
                framed[frame_index * n_fft + i] =
                    (frame[i] - PREEMPHASIS * previous) * INPUT_SCALE * window[i];
            }
        }
        check_feature_cancelled(cancelled)?;

        let framed = mlxcel_core::from_slice_f32(&framed, &[frames as i32, n_fft as i32]);
        let spectrum = mlxcel_core::abs(&mlxcel_core::rfft(&framed, n_fft as i32, 1));
        let spectrum = if effective_rate == 8_000 {
            // Reference drops the 8 kHz Nyquist bin and appends 129 zero bins,
            // making the same 257-bin layout as the 16 kHz frontend.
            let low = mlxcel_core::slice(&spectrum, &[0, 0], &[frames as i32, 128]);
            let zeros = mlxcel_core::zeros(&[frames as i32, 129], mlxcel_core::dtype::FLOAT32);
            mlxcel_core::concatenate(&low, &zeros, 1)
        } else {
            spectrum
        };
        let power = mlxcel_core::square(&spectrum);
        let mel = mlxcel_core::from_slice_f32(&self.mel_filters, &[257, FEATURE_SIZE as i32]);
        let banks = mlxcel_core::matmul(&power, &mel);
        let floor = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
        let banks = mlxcel_core::maximum(&banks, &floor);
        let features = mlxcel_core::log(&banks);
        Ok((
            mlxcel_core::reshape(&features, &[1, frames as i32, FEATURE_SIZE as i32]),
            frames,
        ))
    }
}

fn check_feature_cancelled(cancelled: &dyn AudioCancellation) -> Result<(), String> {
    if cancelled.is_cancelled(AudioPreprocessCheckpoint::Feature) {
        Err("Phi4MM audio feature extraction was cancelled".to_string())
    } else {
        Ok(())
    }
}

fn validate_waveform(sample_count: usize, sample_rate: u32) -> Result<(), String> {
    if sample_rate < 8_000 {
        return Err(format!(
            "unsupported sample rate {sample_rate} Hz; minimum is 8000 Hz"
        ));
    }
    let max_samples = (sample_rate as usize)
        .checked_mul(MAX_AUDIO_DURATION_SECONDS)
        .ok_or("audio duration limit overflow")?;
    if sample_count > max_samples {
        return Err(format!(
            "audio duration exceeds the documented {} second limit",
            MAX_AUDIO_DURATION_SECONDS
        ));
    }
    let minimum = if sample_rate == 8_000 { 280 } else { 560 };
    if sample_count < minimum {
        return Err("audio is empty or too short to produce two analysis frames".into());
    }
    Ok(())
}

/// SpeechLib FbankFC triangles in mel space. Output is transposed to
/// `[n_fft/2+1, n_mels]`, ready for `power @ filters`.
fn speechlib_mel() -> Vec<f32> {
    let (sample_rate, n_fft, n_mels, fmax) = (16_000.0f64, 512.0f64, 80usize, 7690.0f64);
    let mel = |frequency: f64| 1127.0 * (1.0 + frequency / 700.0).ln();
    let bin_to_mel = |bin: usize| mel(bin as f64 * sample_rate / n_fft);
    let f_to_bin = |frequency: f64| (frequency * n_fft / sample_rate + 0.5) as usize;
    let klo = f_to_bin(0.0) + 1;
    let khi = f_to_bin(fmax).max(klo);
    let mlo = mel(0.0);
    let mhi = mel(fmax);
    let step = (mhi - mlo) / (n_mels + 1) as f64;
    let mut filters = vec![0.0f32; 257 * n_mels];
    for m in 0..n_mels {
        let left = mlo + step * m as f64;
        let center = left + step;
        let right = center + step;
        for bin in klo..khi {
            let mel_bin = bin_to_mel(bin);
            if left < mel_bin && mel_bin < right {
                filters[bin * n_mels + m] = (1.0 - (center - mel_bin).abs() / step) as f32;
            }
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CancelInsideFeature(std::sync::atomic::AtomicUsize);

    impl AudioCancellation for CancelInsideFeature {
        fn is_cancelled(&self, checkpoint: AudioPreprocessCheckpoint) -> bool {
            checkpoint == AudioPreprocessCheckpoint::Feature
                && self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel) >= 4
        }
    }

    #[test]
    fn token_count_is_ceiling_frames_over_eight() {
        assert_eq!(audio_embed_size(1), 1);
        assert_eq!(audio_embed_size(8), 1);
        assert_eq!(audio_embed_size(9), 2);
    }

    #[test]
    fn framing_loop_observes_live_cancellation_before_mlx_allocation() {
        let error = Phi4MMAudioFeatureExtractor::new()
            .extract_batch_cancellable(
                &[(vec![0.0; 20_000], 16_000)],
                &CancelInsideFeature(std::sync::atomic::AtomicUsize::new(0)),
            )
            .err()
            .expect("feature extraction must cancel before MLX tensor creation");
        assert!(error.contains("cancelled"));
    }

    #[test]
    fn empty_one_frame_and_oversized_audio_are_rejected_before_allocation() {
        assert!(validate_waveform(0, 16_000).is_err());
        assert!(validate_waveform(400, 16_000).is_err());
        assert!(validate_waveform(16_000 * MAX_AUDIO_DURATION_SECONDS + 1, 16_000).is_err());
        assert!(validate_waveform(560, 16_000).is_ok());
        assert!(validate_waveform(16_000 * MAX_AUDIO_DURATION_SECONDS, 16_000).is_ok());
    }

    #[test]
    fn batch_preserves_multiple_clip_boundaries_without_padding() {
        let extractor = Phi4MMAudioFeatureExtractor::new();
        let first = vec![0.0; 560];
        let second = vec![0.0; 720];
        let batch = extractor
            .extract_batch(&[(first, 16_000), (second, 16_000)])
            .unwrap();

        assert_eq!(batch.clips.len(), 2);
        assert_eq!(batch.frame_lengths, vec![2, 3]);
        assert_eq!(batch.embed_sizes, vec![1, 1]);
        assert_eq!(mlxcel_core::array_shape(&batch.clips[0]), vec![1, 2, 80]);
        assert_eq!(mlxcel_core::array_shape(&batch.clips[1]), vec![1, 3, 80]);
    }

    #[test]
    fn rejects_too_short_resampled_or_non_finite_waveforms() {
        let extractor = Phi4MMAudioFeatureExtractor::new();
        assert!(extractor.extract_clip(&vec![0.0; 560], 48_000).is_err());

        let mut non_finite = vec![0.0; 560];
        non_finite[200] = f32::NAN;
        assert!(extractor.extract_clip(&non_finite, 16_000).is_err());
    }

    #[test]
    fn phi_frontend_call_path_matches_pre_move_scipy_reference() {
        let samples: Vec<f32> = (0..64)
            .map(|index| (0.25 * (2.0 * PI * 440.0 * index as f64 / 48_000.0).sin()) as f32)
            .collect();
        for (source_rate, expected) in [
            (
                32_000,
                [0.002_246_874_2, 0.028_348_744, 0.057_180_017, 0.084_596_574],
            ),
            (
                48_000,
                [0.003_921_834_3, 0.042_409_703, 0.084_868_06, 0.123_779_45],
            ),
        ] {
            let (actual, effective_rate) =
                crate::audio::preprocessing::wav::phi4mm_resample(&samples, source_rate).unwrap();
            let down = source_rate as usize / 16_000;
            assert_eq!(effective_rate, 16_000);
            assert_eq!(actual.len(), samples.len().div_ceil(down));
            for (index, expected) in expected.into_iter().enumerate() {
                assert!(
                    (actual[index] - expected).abs() < 2e-6,
                    "down={down} sample[{index}]={} expected {expected}",
                    actual[index]
                );
            }
        }
    }

    #[test]
    fn pinned_processor_relabels_24khz_before_feature_extraction() {
        let samples: Vec<f32> = (0..24_000)
            .map(|index| (0.25 * (2.0 * PI * 440.0 * index as f64 / 24_000.0).sin()) as f32)
            .collect();
        let extractor = Phi4MMAudioFeatureExtractor::new();
        let (features, frames) = extractor.extract_clip(&samples, 24_000).unwrap();
        assert_eq!(frames, 148);
        assert_eq!(mlxcel_core::array_shape(&features), vec![1, 148, 80]);
        for (frame, mel, expected) in [
            (0, 0, 12.475_846_f32),
            (0, 10, 22.935_753_f32),
            (0, 20, 12.039_375_f32),
            (1, 10, 22.939_745_f32),
            (50, 20, 11.737_321_f32),
            (147, 79, 9.370_197_f32),
        ] {
            let value = mlxcel_core::slice(&features, &[0, frame, mel], &[1, frame + 1, mel + 1]);
            mlxcel_core::eval(&value);
            let actual = mlxcel_core::item_f32(&value);
            assert!(
                (actual - expected).abs() < 2e-3,
                "feature[{frame},{mel}]={actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn speechlib_filterbank_has_pinned_reference_entries() {
        let bank = speechlib_mel();
        assert_eq!(bank.len(), 257 * 80);
        // Values generated by the pinned official `speechlib_mel` function.
        assert!((bank[80] - 0.575_65).abs() < 1e-6);
        assert!((bank[100 * 80 + 54] - 0.616_479_5).abs() < 1e-6);
        assert_eq!(bank[256 * 80 + 79], 0.0);
    }

    #[test]
    fn deterministic_waveform_matches_pinned_speechlib_features() {
        let samples: Vec<f32> = (0..1_600)
            .map(|index| (0.25 * (2.0 * PI * 440.0 * index as f64 / 16_000.0).sin()) as f32)
            .collect();
        let extractor = Phi4MMAudioFeatureExtractor::new();
        let (features, frames) = extractor.extract_clip(&samples, 16_000).unwrap();
        assert_eq!(frames, 8);
        assert_eq!(mlxcel_core::array_shape(&features), vec![1, 8, 80]);

        for (frame, mel, expected) in [
            (0, 0, 10.075_375_f32),
            (0, 10, 10.105_929_f32),
            (0, 20, 14.493_302_f32),
            (0, 30, 12.313_075_f32),
            (1, 10, 9.982_423_f32),
            (3, 20, 13.971_609_f32),
            (7, 79, 8.648_406_f32),
        ] {
            let value = mlxcel_core::slice(&features, &[0, frame, mel], &[1, frame + 1, mel + 1]);
            mlxcel_core::eval(&value);
            let actual = mlxcel_core::item_f32(&value);
            assert!(
                (actual - expected).abs() < 2e-3,
                "feature[{frame},{mel}]={actual}, expected {expected}"
            );
        }
    }
}
