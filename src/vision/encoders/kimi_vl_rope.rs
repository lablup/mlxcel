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

//! MoonViT 2D rotary position embedding.
//!
//! Faithful real-valued port of `Rope2DPosEmb` + `apply_rope` from upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/vision.py.
//!
//! The upstream implementation builds a complex `freqs_cis` table of shape
//! `[max_h, max_w, dim/2]` and rotates `q`/`k` by treating consecutive element
//! pairs as complex numbers (`view_as_complex(reshape(..., -1, 2))`). We compute
//! the identical rotation with real `(cos, sin)` tables and the interleaved
//! ("traditional") pair rotation, which avoids depending on complex-array
//! support in `mlxcel_core` while being numerically equivalent:
//!
//! For each token `t`, head `h`, and pair `i` with angle `θ = angle[t, i]`:
//! ```text
//!   out[..., 2i]   = q[..., 2i] * cos θ - q[..., 2i+1] * sin θ
//!   out[..., 2i+1] = q[..., 2i] * sin θ + q[..., 2i+1] * cos θ
//! ```
//! The `dim/2` per-token angles interleave the x (column) and y (row)
//! frequencies exactly as the upstream `mx.stack([x_cis, y_cis], -1)` step does:
//! even pairs use `col * freq[j]`, odd pairs use `row * freq[j]`, where
//! `freq[j] = theta ** (-4j / dim)` for `j in 0..dim/4`.

use mlxcel_core::{MlxArray, UniquePtr};

/// 2D rotary position embedding for MoonViT.
///
/// `dim` is the attention head dimension (`embed_dim / num_heads`); it must be
/// divisible by 4 so the x/y interleaving is exact (upstream asserts the same).
pub(super) struct Rope2DPosEmb {
    dim: i32,
    theta: f32,
}

impl Rope2DPosEmb {
    pub(super) fn new(dim: i32) -> Self {
        assert!(
            dim % 4 == 0,
            "MoonViT rope dim must be divisible by 4 (got {dim})"
        );
        Self {
            dim,
            theta: 10_000.0,
        }
    }

    /// Base frequencies `freq[j] = theta ** (-4j / dim)` for `j in 0..dim/4`.
    ///
    /// Synthesised as `exp(-(ln theta) * (4j / dim))`; `mx.power` with a scalar
    /// base is not exposed through the cxx bridge, and the `exp` form is what
    /// `mx.fast.rope` uses internally for the same table.
    fn base_freqs(&self) -> UniquePtr<MlxArray> {
        // dim_range = [0, 4, 8, ...] truncated to dim/4 entries.
        let dim_range = mlxcel_core::arange_f32(0.0, self.dim as f32, 4.0);
        let exponent =
            mlxcel_core::multiply_scalar(&dim_range, -(self.theta.ln()) / self.dim as f32);
        mlxcel_core::exp(&exponent)
    }

    /// Build `(cos, sin)` rotation tables of shape `[total_tokens, dim/2]` for
    /// the concatenated per-image `(height, width)` patch grids. Token order is
    /// row-major within each image, images concatenated in slice order.
    pub(super) fn cos_sin(
        &self,
        shapes: &[(i32, i32)],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let mut col_pos: Vec<f32> = Vec::new();
        let mut row_pos: Vec<f32> = Vec::new();
        for &(h, w) in shapes {
            for y in 0..h {
                for x in 0..w {
                    col_pos.push(x as f32);
                    row_pos.push(y as f32);
                }
            }
        }
        let total = col_pos.len() as i32;

        let freq = self.base_freqs(); // [dim/4]
        let col = mlxcel_core::from_slice_f32(&col_pos, &[total]);
        let row = mlxcel_core::from_slice_f32(&row_pos, &[total]);

        // x_angle[t, j] = col[t] * freq[j]; y_angle[t, j] = row[t] * freq[j].
        let x_angle = mlxcel_core::outer(&col, &freq); // [total, dim/4]
        let y_angle = mlxcel_core::outer(&row, &freq); // [total, dim/4]

        // Interleave: angle[t, 2j] = x_angle[t, j], angle[t, 2j+1] = y_angle[t, j].
        let stacked = mlxcel_core::stack_owned(&[x_angle, y_angle], -1); // [total, dim/4, 2]
        let angle = mlxcel_core::reshape(&stacked, &[total, self.dim / 2]);

        let cos = mlxcel_core::cos(&angle);
        let sin = mlxcel_core::sin(&angle);
        (cos, sin)
    }
}

/// Apply the interleaved 2D rotary embedding to `q` and `k`.
///
/// `q`, `k`: `[total_tokens, num_heads, head_dim]`.
/// `cos`, `sin`: `[total_tokens, head_dim/2]`.
pub(super) fn apply_rope(
    q: &MlxArray,
    k: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let q_out = rotate_one(q, cos, sin);
    let k_out = rotate_one(k, cos, sin);
    (q_out, k_out)
}

fn rotate_one(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let total = shape[0];
    let num_heads = shape[1];
    let head_dim = shape[2];
    let half = head_dim / 2;

    // [total, num_heads, half, 2] — consecutive pairs (even, odd).
    let x4 = mlxcel_core::reshape(x, &[total, num_heads, half, 2]);
    let even = mlxcel_core::slice(&x4, &[0, 0, 0, 0], &[total, num_heads, half, 1]);
    let even = mlxcel_core::reshape(&even, &[total, num_heads, half]);
    let odd = mlxcel_core::slice(&x4, &[0, 0, 0, 1], &[total, num_heads, half, 2]);
    let odd = mlxcel_core::reshape(&odd, &[total, num_heads, half]);

    // Broadcast cos/sin across the heads axis: [total, 1, half].
    let cos_b = mlxcel_core::expand_dims(cos, 1);
    let sin_b = mlxcel_core::expand_dims(sin, 1);

    // out_even = even*cos - odd*sin ; out_odd = even*sin + odd*cos.
    let out_even = mlxcel_core::subtract(
        &mlxcel_core::multiply(&even, &cos_b),
        &mlxcel_core::multiply(&odd, &sin_b),
    );
    let out_odd = mlxcel_core::add(
        &mlxcel_core::multiply(&even, &sin_b),
        &mlxcel_core::multiply(&odd, &cos_b),
    );

    // Re-interleave: stack on a new last axis then flatten the pair back in.
    let stacked = mlxcel_core::stack_owned(&[out_even, out_odd], -1); // [total, heads, half, 2]
    mlxcel_core::reshape(&stacked, &[total, num_heads, head_dim])
}
