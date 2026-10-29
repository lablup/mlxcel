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

use super::*;
use crate::imagegen::llada_image::conditioning::tests::host;

#[test]
fn resize_at_the_grid_size_is_the_identity() {
    let values: Vec<f32> = (0..16 * 3).map(|v| v as f32 * 0.25).collect();
    let grid = mlxcel_core::from_slice_f32(&values, &[16, 3]);
    let out = resize_positions(&grid, 4, 4, 4);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 16, 3]);
    assert_eq!(host(&out), values);
}

#[test]
fn halving_samples_between_the_cells() {
    // A 2x2 grid of scalars resized to 1x1 samples the centre: the mean.
    let grid = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 6.0], &[4, 1]);
    assert_eq!(host(&resize_positions(&grid, 2, 1, 1)), vec![3.0]);
    // Upsampling 2 -> 4 on one axis: half-pixel centres clamp at the border.
    let out = host(&resize_positions(&grid, 2, 2, 4));
    assert_eq!(out, vec![1.0, 1.25, 1.75, 2.0, 3.0, 3.75, 5.25, 6.0]);
}
