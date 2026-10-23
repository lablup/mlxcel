// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn output(logits: &[f32], masks: Option<UniquePtr<MlxArray>>) -> DetectorOutput {
    DetectorOutput {
        logits: mlxcel_core::from_slice_f32(logits, &[1, logits.len() as i32]),
        boxes: mlxcel_core::zeros(&[1, logits.len() as i32, 4], mlxcel_core::dtype::FLOAT32),
        masks,
        presence: mlxcel_core::from_slice_f32(&[1.0], &[1, 1]),
    }
}

#[test]
fn detector_rejects_nonfinite_logits_before_confidence_transform() {
    let prediction = output(&[f32::INFINITY], None);
    assert!(select_detections(&prediction, "object", 8, 8, 0.0, 0.5, false, 0).is_err());
}

#[test]
fn detector_rejects_mismatched_or_nonfinite_masks() {
    let mismatched = output(
        &[1.0, 1.0],
        Some(mlxcel_core::zeros(
            &[1, 1, 2, 2],
            mlxcel_core::dtype::FLOAT32,
        )),
    );
    assert!(select_detections(&mismatched, "object", 8, 8, 0.0, 0.5, true, 0).is_err());

    let nonfinite = output(
        &[1.0],
        Some(mlxcel_core::from_slice_f32(
            &[0.0, f32::NAN, 0.0, 0.0],
            &[1, 1, 2, 2],
        )),
    );
    let selected = select_detections(&nonfinite, "object", 8, 8, 0.0, 0.5, true, 0).unwrap();
    assert!(
        select_and_resize_masks(nonfinite.masks.as_deref().unwrap(), &selected.queries, 8, 8,)
            .is_err()
    );
}
