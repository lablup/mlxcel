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

//! Vision RoPE primitives for the Youtu-VL encoder.
//!
//! Lifted out of `youtu_vl.rs` to keep the encoder file under the 500-line
//! soft target. Mirrors `VisionRoPE`, `rotate_half`, and
//! `apply_rotary_pos_emb_vision` from upstream `vision.py`.

use mlxcel_core::{MlxArray, UniquePtr};

/// 1D RoPE frequency table indexed by 2D `(h, w)` positions.
pub(super) struct VisionRoPE {
    pub(super) dim: i32,
    pub(super) theta: f32,
}

impl VisionRoPE {
    pub(super) fn new(dim: i32) -> Self {
        Self {
            dim,
            theta: 10_000.0,
        }
    }

    /// Build the freq table of shape `[seqlen, dim/2]` containing
    /// `seq[i] * inv_freq[d]` for each (i, d).
    ///
    /// Mirrors upstream `VisionRoPE.__call__`. `mx::core::pow` does not have
    /// a scalar overload exposed through the cxx bridge, so we synthesize
    /// `theta ** (arange/dim)` via `exp(ln(theta) * arange/dim)`. The `exp`
    /// path is what `mx.fast.rope` itself uses internally for the same table.
    pub(super) fn freqs(&self, seqlen: i32) -> UniquePtr<MlxArray> {
        let arange = mlxcel_core::arange_f32(0.0, self.dim as f32, 2.0);
        // exponent = arange * (ln(theta) / dim)
        let exponent = mlxcel_core::multiply_scalar(&arange, self.theta.ln() / self.dim as f32);
        // theta ** (arange/dim) = exp(exponent)
        let theta_powed = mlxcel_core::exp(&exponent);
        // inv_freq = 1 / theta_powed
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
        let inv_freq = mlxcel_core::divide(&one, &theta_powed);

        let seq = mlxcel_core::arange_f32(0.0, seqlen as f32, 1.0);
        let seq_col = mlxcel_core::reshape(&seq, &[seqlen, 1]);
        let inv_row = mlxcel_core::reshape(&inv_freq, &[1, self.dim / 2]);
        mlxcel_core::matmul(&seq_col, &inv_row)
    }
}

/// `rotate_half` matching upstream's NumPy-style `concat([-x2, x1], -1)`.
fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last = shape[shape.len() - 1];
    let half = last / 2;
    let x1 = mlxcel_core::utils::slice_axis(x, -1, 0, half);
    let x2 = mlxcel_core::utils::slice_axis(x, -1, half, last);
    let neg_x2 = mlxcel_core::negative(&x2);
    mlxcel_core::concatenate(&neg_x2, &x1, -1)
}

/// Apply rotary embedding to `(q, k)` using precomputed `(cos, sin)` tables.
/// `q, k: [seq_len, num_heads, head_dim]`; `cos/sin: [seq_len, head_dim]`.
pub(super) fn apply_rotary_pos_emb_vision(
    q: &MlxArray,
    k: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Expand cos/sin from `[seq_len, head_dim]` to `[seq_len, 1, head_dim]`
    // so they broadcast across the heads dim.
    let cos_b = mlxcel_core::expand_dims(cos, -2);
    let sin_b = mlxcel_core::expand_dims(sin, -2);

    let q_rot = rotate_half(q);
    let k_rot = rotate_half(k);

    let q_cos = mlxcel_core::multiply(q, &cos_b);
    let q_sin = mlxcel_core::multiply(&q_rot, &sin_b);
    let q_out = mlxcel_core::add(&q_cos, &q_sin);

    let k_cos = mlxcel_core::multiply(k, &cos_b);
    let k_sin = mlxcel_core::multiply(&k_rot, &sin_b);
    let k_out = mlxcel_core::add(&k_cos, &k_sin);

    (q_out, k_out)
}
