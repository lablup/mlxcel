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

use super::{Downsampler, unwindow_tokens, window_tokens};
use mlxcel_core::MlxArray;

fn to_vec(x: &MlxArray) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(x);
    let n: i32 = shape.iter().product();
    let flat = mlxcel_core::reshape(x, &[n]);
    mlxcel_core::eval(&flat);
    (0..n)
        .map(|i| {
            let cell = mlxcel_core::slice(&flat, &[i], &[i + 1]);
            mlxcel_core::eval(&cell);
            mlxcel_core::item_f32(&cell)
        })
        .collect()
}

#[test]
fn window_unwindow_round_trip_is_identity() {
    // 1 tile, 12x12 grid, window 4 -> 9 windows of 16, C=2. Un-window must
    // recover the original.
    let (nt, s, win, c) = (1i32, 12i32, 4i32, 2i32);
    let n = (nt * s * s * c) as usize;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = mlxcel_core::from_slice_f32(&data, &[nt, s * s, c]);
    let windowed = window_tokens(&x, s, win);
    assert_eq!(
        mlxcel_core::array_shape(&windowed),
        vec![nt * (s / win) * (s / win), win * win, c]
    );
    let back = unwindow_tokens(&windowed, nt, s, win);
    assert_eq!(mlxcel_core::array_shape(&back), vec![nt, s * s, c]);
    assert_eq!(to_vec(&x), to_vec(&back));
}

#[test]
fn mean_pool_downsampler_equals_block_mean() {
    // 1 tile, 24x24 grid, C=1: the value at (r,c) is r*24 + c. A 2x2 block mean
    // at output (or, oc) averages input rows {2*or, 2*or+1} x cols {2*oc, 2*oc+1}.
    let (nt, c) = (1i32, 1i32);
    let side = 24usize;
    let data: Vec<f32> = (0..(side * side)).map(|i| i as f32).collect();
    let x = mlxcel_core::from_slice_f32(&data, &[nt, (side * side) as i32, c]);
    let pooled = Downsampler::MeanPool.apply(&x, 2); // (1, 144, 1)
    assert_eq!(mlxcel_core::array_shape(&pooled), vec![1, 144, 1]);
    let got = to_vec(&pooled);
    // Reference: output (or, oc) = mean of the 2x2 block.
    for or in 0..12usize {
        for oc in 0..12usize {
            let mut sum = 0f32;
            for dr in 0..2usize {
                for dc in 0..2usize {
                    let (r, cc) = (2 * or + dr, 2 * oc + dc);
                    sum += (r * side + cc) as f32;
                }
            }
            let want = sum / 4.0;
            assert!((got[or * 12 + oc] - want).abs() < 1e-3, "block ({or},{oc})");
        }
    }
}

#[test]
fn strided_downsampler_selects_expected_offset() {
    // Same numbered 24x24 grid; strided (row_off, col_off) selects input
    // (2*or + row_off, 2*oc + col_off).
    let side = 24usize;
    let data: Vec<f32> = (0..(side * side)).map(|i| i as f32).collect();
    let x = mlxcel_core::from_slice_f32(&data, &[1, (side * side) as i32, 1]);
    for (row_off, col_off) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let sel = Downsampler::Strided { row_off, col_off }.apply(&x, 2);
        let got = to_vec(&sel);
        for or in 0..12usize {
            for oc in 0..12usize {
                let r = 2 * or + row_off as usize;
                let cc = 2 * oc + col_off as usize;
                let want = (r * side + cc) as f32;
                assert!(
                    (got[or * 12 + oc] - want).abs() < 1e-3,
                    "offset ({row_off},{col_off}) at ({or},{oc})"
                );
            }
        }
    }
}
