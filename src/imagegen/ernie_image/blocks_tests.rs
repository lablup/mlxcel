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
fn rotate_half_swaps_and_negates_the_halves() {
    let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
    assert_eq!(values(&rotate_half(&x)), [-3.0, -4.0, 1.0, 2.0]);
}

#[test]
fn rope_tables_repeat_each_angle_on_adjacent_entries() {
    // One token at (3, 1, 2) with axes (4, 4, 4).
    let positions = mlxcel_core::from_slice_f32(&[3.0, 1.0, 2.0], &[1, 1, 3]);
    let theta = 256.0_f32;
    let (cos, sin) = rope_tables(&positions, &[4, 4, 4], theta).unwrap();
    assert_eq!(mlxcel_core::array_shape(&cos), [1, 1, 1, 12]);
    let mut want = Vec::new();
    for pos in [3.0_f32, 1.0, 2.0] {
        for j in 0..2 {
            let omega = 1.0 / theta.powf(2.0 * j as f32 / 4.0);
            want.push(pos * omega);
            want.push(pos * omega);
        }
    }
    for (got, angle) in values(&cos).iter().zip(&want) {
        assert!((got - angle.cos()).abs() < 1e-5, "{got} vs cos {angle}");
    }
    for (got, angle) in values(&sin).iter().zip(&want) {
        assert!((got - angle.sin()).abs() < 1e-5, "{got} vs sin {angle}");
    }
    assert!(rope_tables(&positions, &[4, 4], theta).is_err());
}

#[test]
fn timestep_embedding_is_sin_then_cos() {
    let t = mlxcel_core::from_slice_f32(&[500.0, 1.0], &[2]);
    let emb = values(&timestep_embedding(&t, 8));
    assert_eq!(emb.len(), 16);
    for (row, t) in [500.0_f64, 1.0].iter().enumerate() {
        for i in 0..4 {
            let freq = (-(10_000.0_f64.ln()) * i as f64 / 4.0).exp();
            let phase = t * freq;
            let (s, c) = (emb[row * 8 + i], emb[row * 8 + 4 + i]);
            assert!(
                (f64::from(s) - phase.sin()).abs() < 1e-3,
                "sin {s} vs {}",
                phase.sin()
            );
            assert!(
                (f64::from(c) - phase.cos()).abs() < 1e-3,
                "cos {c} vs {}",
                phase.cos()
            );
        }
    }
}

#[test]
fn modulation_and_gates_compute_in_f32_and_return_the_stream_dtype() {
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]),
        dtype::BFLOAT16,
    );
    let shift = mlxcel_core::from_slice_f32(&[0.5, -0.5], &[1, 2]);
    let scale = mlxcel_core::from_slice_f32(&[1.0, 0.0], &[1, 2]);
    let y = modulate(&x, &shift, &scale);
    assert_eq!(mlxcel_core::array_dtype(&y), dtype::BFLOAT16);
    assert_eq!(values(&y), [2.5, 1.5, 6.5, 3.5]);
    let gate = mlxcel_core::from_slice_f32(&[2.0, 0.0], &[1, 2]);
    let r = gated_residual(&x, &gate, &x);
    assert_eq!(mlxcel_core::array_dtype(&r), dtype::BFLOAT16);
    assert_eq!(values(&r), [3.0, 2.0, 9.0, 4.0]);
}
