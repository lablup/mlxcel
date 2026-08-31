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

use crate::audio::inkling_processor::InklingProcessorConfig;

use super::*;

fn deterministic_noise(length: usize) -> Vec<f32> {
    let mut state = 0x1234_5678u32;
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn reference_slaney_filters(n_fft: usize, n_mels: usize) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    let maximum_frequency = INKLING_SAMPLE_RATE as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(maximum_frequency);
    let points: Vec<f64> = (0..n_mels + 2)
        .map(|index| {
            let mel = mel_min + (mel_max - mel_min) * index as f64 / (n_mels + 1) as f64;
            mel_to_hz_slaney(mel)
        })
        .collect();
    let mut filters = vec![0.0f64; n_mels * bins];
    for mel in 0..n_mels {
        let lower = points[mel];
        let center = points[mel + 1];
        let upper = points[mel + 2];
        let normalization = 2.0 / (upper - lower);
        for frequency in 0..bins {
            let hz = frequency as f64 * maximum_frequency / (bins - 1) as f64;
            filters[mel * bins + frequency] = ((hz - lower) / (center - lower))
                .min((upper - hz) / (upper - center))
                .max(0.0)
                * normalization;
        }
    }
    filters
}

fn reference_log_mel(clip: &[f32]) -> Vec<f32> {
    let n_fft = 1_600usize;
    let hop = 800usize;
    let left_pad = n_fft - hop;
    let padded_length = clip.len();
    let right_pad = (hop - padded_length % hop) % hop;
    let frames = 1 + (left_pad + padded_length + right_pad - n_fft) / hop;
    let filters = reference_slaney_filters(n_fft, 80);
    let mut output = vec![0.0f32; frames * 80];
    for frame_index in 0..frames {
        let mut frame = vec![0.0f64; n_fft];
        for index in 0..n_fft {
            let padded_index = frame_index * hop + index;
            if padded_index >= left_pad {
                let source = padded_index - left_pad;
                if source < clip.len() {
                    let window = 0.5 - 0.5 * (2.0 * PI * index as f64 / n_fft as f64).cos();
                    frame[index] = clip[source] as f64 * window;
                }
            }
        }
        let magnitudes = crate::audio::fft::real_fft_magnitude(&frame, n_fft / 2 + 1);
        for mel in 0..80 {
            let mut sum = 0.0f32;
            for (frequency, magnitude) in magnitudes.iter().enumerate() {
                sum += (*magnitude as f32).max(1e-10)
                    * filters[mel * (n_fft / 2 + 1) + frequency] as f32;
            }
            output[frame_index * 80 + mel] = sum.max(1e-10).log10();
        }
    }
    output
}

#[test]
fn log_mel_matches_f64_reference_on_random_noise() {
    let clip = deterministic_noise(2_401);
    let extractor = InklingFeatureExtractor::new(InklingFeatureExtractorConfig::default()).unwrap();
    let (actual, mask) = extractor.extract_log_mel(&clip).unwrap();
    let expected = reference_log_mel(&clip);
    assert_eq!(actual.len(), 4 * 80);
    assert_eq!(mask, vec![true; 4]);
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-6,
            "feature {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn frame_count_and_mask_follow_hop() {
    let extractor = InklingFeatureExtractor::new(InklingFeatureExtractorConfig::default()).unwrap();
    for (length, expected_frames) in [(16_000, 20), (16_001, 21)] {
        let clip = vec![0.0; length];
        let batch = extractor.extract_batch(&[&clip]).unwrap();
        assert_eq!(batch.frames_per_clip, expected_frames);
        assert_eq!(batch.valid_frames, vec![expected_frames]);
        assert!(batch.mask.iter().all(|value| *value));
    }

    let long = vec![0.0; 16_000];
    let short = vec![0.0; 8_000];
    let batch = extractor.extract_batch(&[&long, &short]).unwrap();
    assert_eq!(batch.frames_per_clip, 20);
    assert_eq!(batch.valid_frames, vec![20, 10]);
    assert!(batch.mask[..20].iter().all(|value| *value));
    assert!(batch.mask[20..30].iter().all(|value| *value));
    assert!(batch.mask[30..].iter().all(|value| !*value));
    assert!(
        batch.features[(20 + 10) * 80..]
            .iter()
            .all(|value| *value == 0.0)
    );
}

#[test]
fn dmel_boundaries_round_down_and_ties_go_low() {
    let boundaries = dmel_boundaries(16, -7.0, 2.0);
    assert_eq!(boundaries.len(), 15);
    for (index, &boundary) in boundaries.iter().enumerate() {
        let center_a = -7.0f64 + 9.0 * index as f64 / 15.0;
        let center_b = -7.0f64 + 9.0 * (index + 1) as f64 / 15.0;
        let midpoint = (center_a + center_b) * 0.5;
        assert!(boundary as f64 <= midpoint);
        let next = if boundary >= 0.0 {
            f32::from_bits(boundary.to_bits() + 1)
        } else {
            f32::from_bits(boundary.to_bits() - 1)
        };
        assert!(next as f64 > midpoint);
        assert_eq!(
            quantize_dmel(&[boundary], 16, -7.0, 2.0).unwrap()[0],
            index as i32
        );
    }
    assert_eq!(
        quantize_dmel(&[-100.0, -7.0, 2.0, 100.0], 16, -7.0, 2.0).unwrap(),
        vec![0, 0, 15, 15]
    );
}

#[test]
fn rejects_invalid_configuration_and_non_finite_input() {
    let bad = InklingFeatureExtractorConfig {
        sampling_rate: 8_000,
        ..Default::default()
    };
    assert!(InklingFeatureExtractor::new(bad).is_err());
    let bad_bins = InklingProcessorConfig {
        num_dmel_bins: 17,
        ..Default::default()
    };
    assert!(bad_bins.validate().is_err());
    let extractor = InklingFeatureExtractor::new(InklingFeatureExtractorConfig::default()).unwrap();
    assert!(extractor.extract_log_mel(&[]).is_err());
    assert!(extractor.extract_log_mel(&[f32::NAN]).is_err());
    assert!(quantize_dmel(&[f32::INFINITY], 16, -7.0, 2.0).is_err());
}
