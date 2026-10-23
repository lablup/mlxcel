// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn region_filter_uses_upstream_strict_iou_threshold() {
    assert!(box_iou([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]) > 0.1);
    assert_eq!(box_iou([0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]), 0.0);
}

#[test]
fn empty_detection_keeps_frame_metadata_and_empty_mask_shape() {
    let options = PredictionOptions {
        task: "track".into(),
        prompts: vec!["missing object".into()],
        boxes: Vec::new(),
        threshold: 0.15,
        nms: 0.5,
    };
    let output = empty_output(1, 4, 3, &options).unwrap();
    assert!(output.instances.is_empty());
    assert_eq!(output.meta["frames"], json!(1));
    assert_eq!(output.meta["track_step_calls"], json!(0));
    assert_eq!(
        mlxcel_core::array_shape(output.tensors.get("masks").unwrap()),
        [0, 3, 4]
    );
}

#[test]
fn tracking_mask_budget_accounts_for_fixed_tracker_canvas() {
    let config = super::super::config::TrackerConfig::default();
    assert!(validate_video_mask_budget(640, 360, 3, 9, &config).is_ok());
    assert!(validate_video_mask_budget(1, 160_000, 3, 64, &config).is_err());
    assert!(validate_video_mask_budget(640, 360, 32, 17, &config).is_err());
    assert!(validate_video_mask_budget(640, 360, 32, 12, &config).is_err());
}

#[test]
fn float_mask_downsampling_matches_pillow_mode_f_bilinear() {
    let values: Vec<f32> = (0..7)
        .flat_map(|_| [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        .collect();
    let masks = mlxcel_core::from_slice_f32(&values, &[1, 7, 7]);
    let resized = resize_float_masks(&masks, 3).unwrap();
    mlxcel_core::eval(&resized);
    let actual = mlxcel_core::utils::array_to_vec_f32(&resized);
    let expected = [
        0.42857143, 0.5294118, 0.42857143, 0.42857143, 0.5294118, 0.42857143, 0.42857143,
        0.5294118, 0.42857143,
    ];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
}

#[test]
fn tracker_output_rejects_nonfinite_logits_before_transforms() {
    let masks = mlxcel_core::from_slice_f32(&[0.0, f32::NAN, 0.0, 0.0], &[1, 1, 2, 2]);
    let scores = mlxcel_core::from_slice_f32(&[0.0], &[1, 1]);
    assert!(
        append_prediction(
            &masks,
            &scores,
            0,
            &["car".into()],
            2,
            2,
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .is_err()
    );

    let masks = mlxcel_core::zeros(&[1, 1, 2, 2], mlxcel_core::dtype::FLOAT32);
    let scores = mlxcel_core::from_slice_f32(&[f32::NEG_INFINITY], &[1, 1]);
    assert!(
        append_prediction(
            &masks,
            &scores,
            0,
            &["car".into()],
            2,
            2,
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .is_err()
    );
}
