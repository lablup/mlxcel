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

//! Unit tests for RT-DETRv2 primitives that don't require a checkpoint.
//!
//! These exercise the host-side helpers (anchor generation, the composed
//! pooling / upsample / grid_sample ops) against hand-computed references so a
//! regression in the math surfaces without a multi-hundred-MB model download.

use mlxcel_core::MlxArray;

use super::layers::{grid_sample, max_pool_3x3_s2_p1, upsample_nearest_2x};
use super::transformer::{generate_anchors, inverse_sigmoid};

/// Read a small MLX array to a row-major `Vec<f32>`.
fn read_f32(arr: &MlxArray) -> Vec<f32> {
    let c = mlxcel_core::contiguous(arr, false);
    let c = c.as_ref().unwrap();
    mlxcel_core::eval(c);
    mlxcel_core::array_to_raw_bytes(c)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn upsample_nearest_2x_doubles_hw() {
    // (1, 2, 2, 1) with values [[1,2],[3,4]] -> (1, 4, 4, 1) nearest.
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let x = mlxcel_core::from_slice_f32(&data, &[1, 2, 2, 1]);
    let up = upsample_nearest_2x(&x);
    let shape = mlxcel_core::array_shape(&up);
    assert_eq!(shape, vec![1, 4, 4, 1]);
    let v = read_f32(&up);
    // Row 0: 1 1 2 2 ; Row 1: 1 1 2 2 ; Row 2: 3 3 4 4 ; Row 3: 3 3 4 4.
    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    for (a, b) in v.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "got {a}, want {b}");
    }
}

#[test]
fn max_pool_3x3_s2_p1_on_4x4() {
    // 4x4 ramp 0..16, single channel. With kernel 3, stride 2, pad 1 ->
    // out = floor((4 + 2 - 3) / 2) + 1 = 2x2.
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let x = mlxcel_core::from_slice_f32(&data, &[1, 4, 4, 1]);
    let pooled = max_pool_3x3_s2_p1(&x);
    let shape = mlxcel_core::array_shape(&pooled);
    assert_eq!(shape, vec![1, 2, 2, 1]);
    // Padded grid (pad=1) has the 4x4 ramp centered; window centers at output
    // (0,0) cover rows -1..1, cols -1..1 -> max of {0,1,4,5} = 5.
    // (0,1) cover cols 1..3 -> max of {1,2,3,5,6,7} = 7.
    // (1,0) -> max of {4,5,8,9,12,13} = 13.
    // (1,1) -> max of {5,6,7,9,10,11,13,14,15} = 15.
    let v = read_f32(&pooled);
    assert_eq!(v, vec![5.0, 7.0, 13.0, 15.0]);
}

#[test]
fn grid_sample_center_is_bilinear_mean() {
    // 2x2 single-channel image [[0,1],[2,3]]. Sample at normalized (0,0) which
    // (align_corners=False) maps to pixel (0.5, 0.5) -> bilinear average of all
    // four corners = (0+1+2+3)/4 = 1.5.
    let img = [0.0f32, 1.0, 2.0, 3.0];
    let x = mlxcel_core::from_slice_f32(&img, &[1, 2, 2, 1]);
    // grid (1, 1, 1, 2) = (gx=0, gy=0).
    let grid = mlxcel_core::from_slice_f32(&[0.0, 0.0], &[1, 1, 1, 2]);
    let out = grid_sample(&x, &grid);
    let shape = mlxcel_core::array_shape(&out);
    assert_eq!(shape, vec![1, 1, 1, 1]);
    let v = read_f32(&out);
    assert!((v[0] - 1.5).abs() < 1e-5, "got {}", v[0]);
}

#[test]
fn grid_sample_out_of_bounds_is_zero_padded() {
    let img = [5.0f32, 6.0, 7.0, 8.0];
    let x = mlxcel_core::from_slice_f32(&img, &[1, 2, 2, 1]);
    // Far outside [-1,1] -> all four corners out of bounds -> zero.
    let grid = mlxcel_core::from_slice_f32(&[5.0, 5.0], &[1, 1, 1, 2]);
    let out = grid_sample(&x, &grid);
    let v = read_f32(&out);
    assert!(v[0].abs() < 1e-6, "got {}", v[0]);
}

#[test]
fn generate_anchors_shapes_and_logit_consistency() {
    // Two tiny levels: 2x2 and 1x1.
    let shapes = [(2, 2), (1, 1)];
    let (anchors, mask) = generate_anchors(&shapes);
    let a_shape = mlxcel_core::array_shape(&anchors);
    let m_shape = mlxcel_core::array_shape(&mask);
    // total = 4 + 1 = 5 positions.
    assert_eq!(a_shape, vec![1, 5, 4]);
    assert_eq!(m_shape, vec![1, 5, 1]);

    let a = read_f32(&anchors);
    let m = read_f32(&mask);
    // For each valid position, sigmoid(logit) should reconstruct the anchor
    // center within [eps, 1-eps]; for masked positions the logit is f32::MAX.
    for (i, &valid) in m.iter().enumerate() {
        if valid > 0.5 {
            // first component (cx) sigmoid in (0,1).
            let s = 1.0 / (1.0 + (-a[i * 4]).exp());
            assert!(s > 0.0 && s < 1.0);
        } else {
            assert_eq!(a[i * 4], f32::MAX);
        }
    }
}

#[test]
fn inverse_sigmoid_is_logit() {
    // inverse_sigmoid(0.5) == log(0.5/0.5) == 0.
    let x = mlxcel_core::from_slice_f32(&[0.5], &[1]);
    let out = inverse_sigmoid(&x, 1e-5);
    let v = read_f32(&out);
    assert!(v[0].abs() < 1e-4, "got {}", v[0]);

    // inverse_sigmoid then sigmoid round-trips a mid-range value.
    let x = mlxcel_core::from_slice_f32(&[0.73], &[1]);
    let inv = inverse_sigmoid(&x, 1e-5);
    let back = mlxcel_core::sigmoid(&inv);
    let v = read_f32(&back);
    assert!((v[0] - 0.73).abs() < 1e-4, "got {}", v[0]);
}
