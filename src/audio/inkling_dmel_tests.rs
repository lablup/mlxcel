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

use crate::audio::inkling_processor::InklingProcessorConfig;

use serde::Deserialize;

use super::*;

const NUMPY_FIXTURE_ABSOLUTE_TOLERANCE: f32 = 2e-6;

#[derive(Debug, Deserialize)]
struct NumPyLogMelFixture {
    numpy_version: String,
    upstream_head_revision: String,
    upstream_merge_revision: String,
    waveform: Vec<f32>,
    shape: Vec<usize>,
    expected_log_mel: Vec<f32>,
    absolute_tolerance: f32,
}

#[test]
fn log_mel_matches_pinned_numpy_fixture() {
    let fixture: NumPyLogMelFixture =
        serde_json::from_str(include_str!("../../tests/fixtures/inkling_dmel_numpy.json")).unwrap();
    assert_eq!(fixture.numpy_version, "2.3.2");
    assert_eq!(
        fixture.upstream_head_revision,
        "0d6805bb7ef67998d8aeb655bc1df83854830d56"
    );
    assert_eq!(
        fixture.upstream_merge_revision,
        "67bc41d818ea77908599d21510ea29f352e7a417"
    );
    assert_eq!(fixture.shape, [4, 80]);
    assert_eq!(fixture.waveform.len(), 2_401);
    assert_eq!(fixture.absolute_tolerance, NUMPY_FIXTURE_ABSOLUTE_TOLERANCE);
    assert_eq!(
        fixture.expected_log_mel.len(),
        fixture.shape.iter().product::<usize>()
    );

    let extractor = InklingFeatureExtractor::new(InklingFeatureExtractorConfig::default()).unwrap();
    let (actual, mask) = extractor.extract_log_mel(&fixture.waveform).unwrap();
    assert_eq!(actual.len(), fixture.expected_log_mel.len());
    assert_eq!(mask, vec![true; 4]);
    for (index, (&actual, &expected)) in actual.iter().zip(&fixture.expected_log_mel).enumerate() {
        assert!(
            (actual - expected).abs() <= NUMPY_FIXTURE_ABSOLUTE_TOLERANCE,
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
fn compact_batch_transforms_only_valid_rows_for_skewed_clips() {
    let extractor = InklingFeatureExtractor::new(InklingFeatureExtractorConfig::default()).unwrap();
    let long = vec![0.0; 16_000];
    let short = vec![0.0; 1];
    let mut clips = vec![long.as_slice()];
    clips.extend(std::iter::repeat_n(short.as_slice(), 15));
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let batch = extractor
        .extract_compact_batch_cancellable(&clips, 6_000, &cancelled)
        .unwrap();
    assert_eq!(batch.valid_frames, [vec![20], vec![1; 15]].concat());
    assert_eq!(batch.transformed_frames, 35);
    assert_eq!(batch.features.len(), 35 * 80);
}

#[test]
fn compact_batch_caps_actual_transforms_before_fft_work() {
    let extractor = InklingFeatureExtractor::new(InklingFeatureExtractorConfig::default()).unwrap();
    let clip = vec![0.0; 2_401];
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let error = extractor
        .extract_compact_batch_cancellable(&[&clip, &clip], 7, &cancelled)
        .unwrap_err();
    assert!(error.contains("8 frame transforms"));
    assert!(error.contains("request limit 7"));
}

#[test]
fn compact_batch_checks_cancellation_between_frames() {
    let extractor = InklingFeatureExtractor::new(InklingFeatureExtractorConfig::default()).unwrap();
    let clip = vec![0.0; 4_000];
    let mut checks = 0usize;
    let error = extractor
        .extract_compact_batch_with_cancel(&[&clip], 6_000, || {
            checks += 1;
            checks > 2
        })
        .unwrap_err();
    assert_eq!(checks, 3);
    assert!(error.contains("cancelled"));
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
