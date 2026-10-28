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
use crate::imagegen::testing::values;

#[test]
fn the_axis_selector_matches_upstream() {
    // Upstream `Ideogram4MRoPE` for head_dim 256, mrope_section (24, 20, 20):
    // indices 1, 4, ..., 58 read height and 2, 5, ..., 59 read width.
    let selector = axis_selector(256, [24, 20, 20]);
    assert_eq!(selector.len(), 128);
    for (i, &axis) in selector.iter().enumerate() {
        let want = match i {
            i if i < 60 && i % 3 == 1 => 1,
            i if i < 60 && i % 3 == 2 => 2,
            _ => 0,
        };
        assert_eq!(axis, want, "index {i}");
    }
    assert_eq!(axis_selector(16, [4, 2, 2]), [0, 1, 2, 0, 1, 2, 0, 0]);
}

#[test]
fn rotate_half_negates_the_second_half_first() {
    let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    assert_eq!(values(&rotate_half(&x)), [-3.0, -4.0, 1.0, 2.0]);
}

#[test]
fn the_norms_round_their_statistics_to_the_activation_dtype() {
    let data = [0.3_f32, -1.7, 2.25, 0.01];
    let x = mlxcel_core::from_slice_f32(&data, &[1, 4]);
    let norm = Norm {
        weight: mlxcel_core::from_slice_f32(&[1.0, 2.0, 1.0, 0.5], &[4]),
        eps: 1e-6,
    };
    let ms = data.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let inv = 1.0 / (ms + 1e-6).sqrt();
    let got = values(&norm.forward(&x));
    for ((g, v), w) in got.iter().zip(data).zip([1.0, 2.0, 1.0, 0.5]) {
        assert!((g - v * inv * w).abs() < 1e-5, "{g} vs {}", v * inv * w);
    }
    let ln = values(&layer_norm_no_affine(&x, 1e-6));
    let mean = ln.iter().sum::<f32>() / 4.0;
    let var = ln.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
    assert!(
        mean.abs() < 1e-6 && (var - 1.0).abs() < 1e-4,
        "{mean} {var}"
    );
}
