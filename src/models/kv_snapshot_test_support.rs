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

//! K/V fixtures and the byte comparison shared by the quantized snapshot
//! tests (issue #3250).
//!
//! Used by: `kv_snapshot_quantized_tests`, `kv_snapshot_parts_tests`,
//! `hybrid_snapshot_tests`, and the LFM2 and Bailing MoE Linear attention-arm
//! tests.

use mlxcel_core::{MlxArray, UniquePtr};

/// Heads and head dim of the fixture. The turbo rotation is a Walsh-Hadamard
/// transform, so the head dim has to be a power of two.
pub(crate) const QH: i32 = 2;
pub(crate) const QD: i32 = 64;

/// `n` FP16 tokens of K or V starting at absolute position `start`.
///
/// The value at `[0, h, t, d]` depends on the absolute position `start + t`
/// and never on how the tokens were chunked, so a prefix of a long push is
/// bitwise equal to a short push of the same positions. The amplitude changes
/// from token to token so that per-token scales and norms differ, which is
/// what makes a sidecar restored one row out of step visible.
pub(crate) fn kv(start: i32, n: i32, phase: f32) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity((QH * n * QD) as usize);
    for h in 0..QH {
        for t in 0..n {
            let pos = start + t;
            let amplitude = 0.25 + (pos % 7) as f32 * 0.125;
            for d in 0..QD {
                let angle = h as f32 * 1.7 + pos as f32 * 0.37 + d as f32 * 0.11 + phase;
                data.push(angle.sin() * amplitude);
            }
        }
    }
    let arr = mlxcel_core::from_slice_f32(&data, &[1, QH, n, QD]);
    mlxcel_core::astype(&arr, mlxcel_core::dtype::FLOAT16)
}

pub(crate) fn keys(start: i32, n: i32) -> UniquePtr<MlxArray> {
    kv(start, n, 0.0)
}

pub(crate) fn values(start: i32, n: i32) -> UniquePtr<MlxArray> {
    kv(start, n, 1.0)
}

/// Shape, dtype and every byte must agree.
pub(crate) fn assert_same_bytes(actual: &MlxArray, expected: &MlxArray, what: &str) {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected),
        "{what}: shape"
    );
    assert_eq!(
        mlxcel_core::array_dtype(actual),
        mlxcel_core::array_dtype(expected),
        "{what}: dtype"
    );
    mlxcel_core::eval(actual);
    mlxcel_core::eval(expected);
    assert!(
        mlxcel_core::array_to_raw_bytes(actual) == mlxcel_core::array_to_raw_bytes(expected),
        "{what}: bytes differ"
    );
}
