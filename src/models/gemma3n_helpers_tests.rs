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

use crate::models::gemma3n_helpers::{
    apply_softcap, mean_arrays, slice_altup_plane, slice_layer_input,
    split_altup_after_per_layer_update, stack_arrays,
};
use std::sync::{Mutex, OnceLock};

fn test_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

fn assert_allclose(actual: &mlxcel_core::MlxArray, expected: &mlxcel_core::MlxArray) {
    let close = mlxcel_core::allclose(actual, expected, 1e-5, 1e-5);
    mlxcel_core::eval(&close);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
#[ignore = "requires serial MLX execution"]
fn stack_and_mean_helpers_keep_expected_shapes() {
    let _guard = test_guard().lock().unwrap();
    let arrays = vec![
        mlxcel_core::from_slice_f32(&[1.0, 3.0], &[1, 2]),
        mlxcel_core::from_slice_f32(&[3.0, 5.0], &[1, 2]),
    ];

    let stacked = stack_arrays(&arrays, 0);
    assert_eq!(mlxcel_core::array_shape(&stacked), vec![2, 1, 2]);

    let mean = mean_arrays(&arrays);
    let expected = mlxcel_core::from_slice_f32(&[2.0, 4.0], &[1, 2]);
    assert_allclose(&mean, &expected);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn apply_softcap_limits_logit_magnitude() {
    let _guard = test_guard().lock().unwrap();
    let logits = mlxcel_core::from_slice_f32(&[10.0, -10.0], &[1, 2]);
    let capped = apply_softcap(&logits, 2.0);
    let capped_abs = mlxcel_core::abs(&capped);
    let max_abs = mlxcel_core::max_all(&capped_abs);
    mlxcel_core::eval(&max_abs);

    assert!(mlxcel_core::item_f32(&max_abs) <= 2.0 + 1e-5);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn slice_layer_input_selects_requested_layer_plane() {
    let _guard = test_guard().lock().unwrap();
    let data: Vec<f32> = (0..12).map(|n| n as f32).collect();
    let per_layer = mlxcel_core::from_slice_f32(&data, &[1, 2, 3, 2]);

    let sliced = slice_layer_input(&per_layer, 1, 1, 2, 2);
    let expected = mlxcel_core::from_slice_f32(&[2.0, 3.0, 8.0, 9.0], &[1, 2, 2]);
    assert_eq!(mlxcel_core::array_shape(&sliced), vec![1, 2, 2]);
    assert_allclose(&sliced, &expected);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn slice_altup_plane_selects_stacked_prediction_plane() {
    let _guard = test_guard().lock().unwrap();
    let data: Vec<f32> = (0..16).map(|n| n as f32).collect();
    let stacked = mlxcel_core::from_slice_f32(&data, &[4, 1, 2, 2]);

    let sliced = slice_altup_plane(&stacked, 2);
    let expected = mlxcel_core::from_slice_f32(&[8.0, 9.0, 10.0, 11.0], &[1, 2, 2]);
    assert_eq!(mlxcel_core::array_shape(&sliced), vec![1, 2, 2]);
    assert_allclose(&sliced, &expected);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn split_altup_after_per_layer_update_preserves_plane_zero_and_updates_tail() {
    let _guard = test_guard().lock().unwrap();
    let data: Vec<f32> = (0..16).map(|n| n as f32).collect();
    let stacked = mlxcel_core::from_slice_f32(&data, &[4, 1, 2, 2]);
    let delta = mlxcel_core::from_slice_f32(&[100.0, 200.0, 300.0, 400.0], &[1, 2, 2]);

    let split = split_altup_after_per_layer_update(&stacked, &delta, 4);
    assert_eq!(split.len(), 4);

    let expected0 = mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[1, 2, 2]);
    let expected1 = mlxcel_core::from_slice_f32(&[104.0, 205.0, 306.0, 407.0], &[1, 2, 2]);
    let expected3 = mlxcel_core::from_slice_f32(&[112.0, 213.0, 314.0, 415.0], &[1, 2, 2]);
    assert_allclose(&split[0], &expected0);
    assert_allclose(&split[1], &expected1);
    assert_allclose(&split[3], &expected3);
}
