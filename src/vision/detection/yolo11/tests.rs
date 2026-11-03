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

//! Unit tests that need no checkpoint: letterbox geometry, NMS, config
//! routing and the preprocessing inverse mapping.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};

use super::predictor::{non_max_suppression, prepare_image};
use super::{config::Yolo11Config, predictor};

fn solid(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb([10, 20, 30])))
}

/// A `(1, 4 + classes, anchors)` prediction from per-anchor tuples.
fn prediction(classes: usize, anchors: &[(f32, f32, f32, f32, Vec<f32>)]) -> Vec<f32> {
    let count = anchors.len();
    let mut flat = vec![0.0_f32; (4 + classes) * count];
    for (index, (cx, cy, w, h, scores)) in anchors.iter().enumerate() {
        flat[index] = *cx;
        flat[count + index] = *cy;
        flat[2 * count + index] = *w;
        flat[3 * count + index] = *h;
        for (label, score) in scores.iter().enumerate() {
            flat[(4 + label) * count + index] = *score;
        }
    }
    flat
}

#[test]
fn letterbox_matches_upstream_for_a_wide_screenshot() {
    let letterbox = prepare_image(&solid(1000, 600), Some(1280));
    assert_eq!(letterbox.original_size, (1000, 600));
    assert!((letterbox.gain - 1.28).abs() < 1e-6);
    // gain 1.28 -> 1280 x 768, already multiples of 32 and centred.
    assert_eq!(
        mlxcel_core::array_shape(&letterbox.pixel_values),
        vec![1, 768, 1280, 3]
    );
    assert_eq!((letterbox.left, letterbox.top), (0, 0));
}

#[test]
fn letterbox_pads_and_centres_odd_remainders() {
    // 100 x 50 at native size pads to 128 x 64, leaving 14 and 7 pixels.
    let letterbox = prepare_image(&solid(100, 50), None);
    assert_eq!(
        mlxcel_core::array_shape(&letterbox.pixel_values),
        vec![1, 64, 128, 3]
    );
    assert_eq!((letterbox.left, letterbox.top), (14, 7));
    assert_eq!(letterbox.gain, 1.0);

    let values = mlxcel_core::utils::array_to_vec_f32(&letterbox.pixel_values);
    let pixel = |x: usize, y: usize| {
        let start = (y * 128 + x) * 3;
        [values[start], values[start + 1], values[start + 2]]
    };
    let fill = 114.0 / 255.0;
    assert!((pixel(0, 0)[0] - fill).abs() < 1e-6);
    assert!((pixel(127, 63)[0] - fill).abs() < 1e-6);
    // The image starts at (14, 7) rather than the canvas origin.
    assert!((pixel(14, 7)[0] - 10.0 / 255.0).abs() < 1e-6);
    assert!((pixel(13, 7)[0] - fill).abs() < 1e-6);
    assert!((pixel(14, 6)[0] - fill).abs() < 1e-6);
}

#[test]
fn letterbox_upscales_when_size_exceeds_the_image() {
    // 64 x 32 with --size 128 doubles both sides, padding to 128 x 64.
    let letterbox = prepare_image(&solid(64, 32), Some(128));
    assert!((letterbox.gain - 2.0).abs() < 1e-6);
    assert_eq!(
        mlxcel_core::array_shape(&letterbox.pixel_values),
        vec![1, 64, 128, 3]
    );
    assert_eq!((letterbox.left, letterbox.top), (0, 0));
}

#[test]
fn inverse_letterbox_returns_original_pixels() {
    let letterbox = prepare_image(&solid(100, 50), None);
    let gain = letterbox.gain;
    let inverse =
        |value: f32, offset: i32, limit: f32| ((value - offset as f32) / gain).clamp(0.0, limit);
    // A box drawn at the pasted image's corners maps back to (0, 0, 100, 50).
    assert_eq!(inverse(14.0, letterbox.left, 100.0), 0.0);
    assert_eq!(inverse(7.0, letterbox.top, 50.0), 0.0);
    assert_eq!(inverse(114.0, letterbox.left, 100.0), 100.0);
    assert_eq!(inverse(57.0, letterbox.top, 50.0), 50.0);
}

#[test]
fn nms_keeps_overlapping_boxes_of_different_classes() {
    let anchors = [
        (50.0, 50.0, 20.0, 20.0, vec![0.9, 0.1]),
        (50.0, 50.0, 20.0, 20.0, vec![0.1, 0.8]),
    ];
    let flat = prediction(2, &anchors);
    let kept = non_max_suppression(&flat, 2, anchors.len(), 0.05, 0.1, 300, 30_000).unwrap();
    assert_eq!(kept.len(), 2, "class-aware NMS must not merge classes");
    assert_eq!(kept[0].label, 0);
    assert_eq!(kept[1].label, 1);
}

#[test]
fn nms_suppresses_same_class_overlap_at_the_iou_threshold() {
    // IoU between these boxes is 1/3: kept at 0.4, suppressed at 0.3.
    let anchors = [
        (5.0, 5.0, 10.0, 10.0, vec![0.9]),
        (10.0, 5.0, 10.0, 10.0, vec![0.8]),
    ];
    let flat = prediction(1, &anchors);
    let kept = non_max_suppression(&flat, 1, anchors.len(), 0.05, 0.4, 300, 30_000).unwrap();
    assert_eq!(kept.len(), 2);
    let suppressed = non_max_suppression(&flat, 1, anchors.len(), 0.05, 0.3, 300, 30_000).unwrap();
    assert_eq!(suppressed.len(), 1);
    assert_eq!(suppressed[0].score, 0.9);
}

#[test]
fn nms_filters_by_confidence_and_caps_max_detections() {
    let scores: Vec<Vec<f32>> = (0..6).map(|index| vec![0.9 - index as f32 * 0.1]).collect();
    // Far apart, so only the max_det cap matters.
    let anchors: Vec<(f32, f32, f32, f32, Vec<f32>)> = scores
        .iter()
        .enumerate()
        .map(|(index, score)| {
            let offset = index as f32 * 100.0;
            (offset + 5.0, 5.0, 10.0, 10.0, score.clone())
        })
        .collect();
    let flat = prediction(1, &anchors);
    let kept = non_max_suppression(&flat, 1, anchors.len(), 0.2, 0.1, 300, 30_000).unwrap();
    assert_eq!(kept.len(), 6);
    for detection in &kept {
        assert!(detection.score > 0.2);
    }
    let capped = non_max_suppression(&flat, 1, anchors.len(), 0.05, 0.1, 2, 30_000).unwrap();
    assert_eq!(capped.len(), 2);
    assert!((capped[0].score - 0.9).abs() < 1e-6);
    assert!((capped[1].score - 0.8).abs() < 1e-6);
}

#[test]
fn nms_keeps_the_highest_scores_when_max_nms_truncates() {
    let anchors: Vec<(f32, f32, f32, f32, Vec<f32>)> = (0..5)
        .map(|index| {
            let offset = index as f32 * 100.0;
            let score = 0.1 * (index + 1) as f32;
            (offset + 5.0, 5.0, 10.0, 10.0, vec![score])
        })
        .collect();
    let flat = prediction(1, &anchors);
    let kept = non_max_suppression(&flat, 1, anchors.len(), 0.05, 0.1, 300, 2).unwrap();
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].score, 0.5);
    assert_eq!(kept[1].score, 0.4);
}

#[test]
fn nms_rejects_malformed_predictions() {
    let error = non_max_suppression(&[0.0; 3], 1, 1, 0.5, 0.1, 300, 30_000).unwrap_err();
    assert!(error.contains("expected"), "{error}");
    let error = non_max_suppression(&[f32::NAN; 5], 1, 1, 0.5, 0.1, 300, 30_000).unwrap_err();
    assert!(error.contains("not finite"), "{error}");
    let error = non_max_suppression(&[0.0; 5], 1, 1, 0.5, 2.0, 300, 30_000).unwrap_err();
    assert!(error.contains("IoU"), "{error}");
}

#[test]
fn detection_routing_accepts_yolo11_with_its_own_default() {
    let root = std::env::temp_dir().join(format!("mlxcel-yolo11-routing-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{"model_type":"yolo11","nc":1,"ch":[256,512,512],"reg_max":16,
            "names":{"0":"icon"},"stride":[8,16,32]}"#,
    )
    .unwrap();
    let info = crate::vision::detection::detector_info(&root).unwrap();
    assert_eq!(info.model_type, "yolo11");
    assert_eq!(info.default_threshold, predictor::DEFAULT_THRESHOLD);
    assert!(!info.supports_masks);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn detection_routing_rejects_a_yolo11_topology_it_cannot_run() {
    let root =
        std::env::temp_dir().join(format!("mlxcel-yolo11-bad-topology-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        r#"{"model_type":"yolo11","nc":1,"ch":[128,256,256],"reg_max":16,
            "names":{"0":"icon"},"stride":[8,16,32]}"#,
    )
    .unwrap();
    let error = crate::vision::detection::detector_info(&root).unwrap_err();
    assert!(error.contains("channels"), "{error}");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn config_rejects_a_negative_threshold_before_loading() {
    let directory = Path::new("missing-yolo11-checkpoint");
    let error = crate::vision::detection::load_detector(directory, Some(-1.0))
        .err()
        .expect("negative threshold must fail");
    assert!(error.contains("threshold"), "{error}");
}

#[test]
fn default_config_is_the_omniparser_header() {
    let config = Yolo11Config::default();
    assert_eq!(config.nc, 1);
    assert_eq!(config.reg_max, 16);
    assert_eq!(config.class_name(0), "0");
}
