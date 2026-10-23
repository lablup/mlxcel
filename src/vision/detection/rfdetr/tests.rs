// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Focused RF-DETR graph and geometry tests.

use super::backbone::{window_partition, window_unpartition};
use super::model::bbox_reparameterize;

fn values(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32))
}

#[test]
fn window_partition_round_trips_spatial_tokens() {
    let input: Vec<f32> = (0..4 * 4 * 384).map(|value| value as f32).collect();
    let input = mlxcel_core::from_slice_f32(&input, &[1, 16, 384]);
    let windows = window_partition(&input, 1, 4, 4, 2);
    assert_eq!(mlxcel_core::array_shape(&windows), [4, 4, 384]);
    let output = window_unpartition(&windows, 1, 4, 4, 2);
    assert_eq!(mlxcel_core::array_shape(&output), [1, 16, 384]);
    assert_eq!(values(&output), values(&input));
}

#[test]
fn bbox_reparameterization_matches_hand_computation() {
    let delta = mlxcel_core::from_slice_f32(&[0.5, -1.0, 2.0_f32.ln(), 0.5_f32.ln()], &[1, 1, 4]);
    let reference = mlxcel_core::from_slice_f32(&[0.4, 0.3, 0.2, 0.1], &[1, 1, 4]);
    let actual = values(&bbox_reparameterize(&delta, &reference));
    for (actual, expected) in actual.iter().zip([0.5, 0.2, 0.4, 0.05]) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }
}
