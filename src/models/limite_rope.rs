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

//! Partial interleaved RoPE of the Limite decoder (issue #2360), with the
//! cos/sin tables built once per forward and shared by every RoPE layer.

use super::scalar;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Partial interleaved RoPE: `n_pairs` of `head_dim / 2` pairs rotate, the
/// rest pass through (zero frequency). Pairs are adjacent lanes and rotate
/// `(a, b) -> (a cos + b sin, b cos - a sin)`; cos/sin are computed in f32
/// and rounded to bf16, as the reference pins (`rope_cos_sin_dtype`).
pub(crate) struct PartialRope {
    /// `[head_dim]` f32: each pair's frequency twice, then zeros.
    freqs: UniquePtr<MlxArray>,
    /// `[head_dim]` +1 on even lanes, -1 on odd lanes, bf16.
    lane_sign: UniquePtr<MlxArray>,
    head_dim: i32,
}

impl PartialRope {
    pub(crate) fn new(n_pairs: i32, head_dim: i32, base: f32) -> Self {
        let base = scalar(1.0 / base, dtype::FLOAT32);
        let exps = mlxcel_core::linspace(0.0, 1.0, n_pairs, dtype::FLOAT32);
        let freq = mlxcel_core::repeat(&mlxcel_core::power(&base, &exps), 2, 0);
        let pad = mlxcel_core::zeros(&[head_dim - 2 * n_pairs], dtype::FLOAT32);
        let freqs = mlxcel_core::concatenate(&freq, &pad, 0);
        let signs: Vec<f32> = (0..head_dim)
            .map(|i| if i % 2 == 1 { -1.0 } else { 1.0 })
            .collect();
        let lane_sign = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&signs, &[head_dim]),
            dtype::BFLOAT16,
        );
        Self {
            freqs,
            lane_sign,
            head_dim,
        }
    }

    /// The cos and signed sin tables at `positions` (`[P, L]` f32 with `P` 1
    /// or `B`), computed once per forward and shared by every RoPE layer.
    pub(crate) fn tables(&self, positions: &MlxArray) -> RopeTables {
        let pos_shape = mlxcel_core::array_shape(positions);
        let (p, l) = (pos_shape[0], pos_shape[1]);
        let pos = mlxcel_core::reshape(positions, &[p, l, 1, 1]);
        let freqs = mlxcel_core::reshape(&self.freqs, &[1, 1, 1, self.head_dim]);
        let theta = mlxcel_core::multiply(&pos, &freqs);
        let cos = mlxcel_core::astype(&mlxcel_core::cos(&theta), dtype::BFLOAT16);
        let sin = mlxcel_core::astype(&mlxcel_core::sin(&theta), dtype::BFLOAT16);
        let sin = mlxcel_core::multiply(&sin, &self.lane_sign);
        RopeTables {
            cos,
            sin,
            head_dim: self.head_dim,
        }
    }
}

/// `[P, L, 1, head_dim]` bf16 cos and lane-signed sin of one forward.
pub(crate) struct RopeTables {
    cos: UniquePtr<MlxArray>,
    sin: UniquePtr<MlxArray>,
    head_dim: i32,
}

impl RopeTables {
    /// Rotate `x` (`[B, L, H, head_dim]`).
    pub(crate) fn apply(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dt = mlxcel_core::array_dtype(x);
        let shape = mlxcel_core::array_shape(x);
        let mut pairs = shape.clone();
        pairs.pop();
        pairs.extend([self.head_dim / 2, 2]);
        let x2 = mlxcel_core::reshape(x, &pairs);
        let last = pairs.len() as i32 - 1;
        let a = slice_axis(&x2, last, 0, 1);
        let b = slice_axis(&x2, last, 1, 2);
        let flipped = mlxcel_core::reshape(&mlxcel_core::concatenate(&b, &a, last), &shape);
        let lhs = mlxcel_core::multiply(&mlxcel_core::astype(&self.cos, dt), x);
        let rhs = mlxcel_core::multiply(&mlxcel_core::astype(&self.sin, dt), &flipped);
        mlxcel_core::add(&lhs, &rhs)
    }
}

/// `[1, len]` f32 positions starting at `offset`.
pub(crate) fn positions_from_offset(offset: i32, len: i32) -> UniquePtr<MlxArray> {
    let pos = mlxcel_core::arange_f32(offset as f32, (offset + len) as f32, 1.0);
    mlxcel_core::reshape(&pos, &[1, len])
}

/// `[B, 1]` f32 positions, one per decode row.
pub(crate) fn positions_from_offsets(offsets: &[i32]) -> UniquePtr<MlxArray> {
    let pos: Vec<f32> = offsets.iter().map(|&o| o as f32).collect();
    mlxcel_core::from_slice_f32(&pos, &[offsets.len() as i32, 1])
}
