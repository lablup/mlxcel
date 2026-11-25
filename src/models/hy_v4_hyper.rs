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

//! Hy4 identity hyper connections (`hy_v4/hyper_connection.py` upstream).
//!
//! The residual carried between Hy4 blocks is `[B, L, hc_mult, hidden]`. Each
//! sublayer collapses it to a `[B, L, hidden]` input with a learned `pre`
//! gate and folds its output back with a learned `post` gate:
//!
//! ```text
//! x    = rms_norm(flatten(h), no weight, eps = rms_norm_eps) @ hc_fn.T   [B, L, 2 * hc]
//! pre  = sigmoid(x[..., :hc] * hc_scale[0] + hc_base[:hc]) + hc_eps
//! post = hc_magnitude * sigmoid(x[..., hc:] * hc_scale[1] + hc_base[hc:])
//! in   = sum(pre[..., None] * h, axis = hc)
//! h'   = h + post[..., None] * out[:, :, None, :]
//! ```
//!
//! This is the DeepSeek-V4 hyper connection with the `comb` output dropped:
//! the residual streams are never mixed with each other, so there is no
//! Sinkhorn normalization and `hc_fn` has `2 * hc` rows instead of
//! `(2 + hc) * hc`. `deepseek_v4_hyper.rs` is therefore not reusable; its
//! shape check would reject these tensors.
//!
//! Everything here runs in float32 over an f32 copy of the residual, and only
//! the collapsed input and the expanded residual are cast back to the
//! activation dtype. The `hc_*` tensors ship as float32 and stay float32
//! (upstream `cast_predicate`).
//!
//! The transformers reference (`modeling_hy_v4.py`, `HYV4HyperConnection`)
//! also adds `hc_eps` to `post`. mlx-vlm does not, and this port follows
//! mlx-vlm, the parity reference; the term is 1e-6 against a gate of order 1.

use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Load `name` and require `expected`, naming `hc_mult * hidden_size` in the
/// error so a config and checkpoint that disagree are told apart from a
/// missing tensor.
fn expect_hc_shape(
    weights: &WeightMap,
    name: &str,
    expected: &[i32],
    hc_mult: i32,
    hidden_size: i32,
) -> Result<UniquePtr<MlxArray>, String> {
    let weight = weights
        .get(name)
        .ok_or_else(|| format!("Weight not found: {name}"))?;
    let shape = mlxcel_core::array_shape(weight);
    if shape != expected {
        return Err(format!(
            "{name}: expected shape {expected:?} for hc_mult {hc_mult} and hidden_size \
             {hidden_size} (hc_mult * hidden_size = {}), checkpoint ships {shape:?}",
            hc_mult * hidden_size
        ));
    }
    Ok(mlxcel_core::astype(weight, mlxcel_core::dtype::FLOAT32))
}

/// `rms_norm(flatten(h)) @ fn.T` in float32, plus the f32 copy of `h`.
fn mixes(
    h: &MlxArray,
    fn_weight: &MlxArray,
    norm_eps: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(h);
    let (b, l, hc, d) = (shape[0], shape[1], shape[2], shape[3]);
    let residual = mlxcel_core::astype(h, mlxcel_core::dtype::FLOAT32);
    let flat = mlxcel_core::reshape(&residual, &[b, l, hc * d]);
    let normalized = mlxcel_core::fast_rms_norm_no_weight(&flat, norm_eps);
    let fn_t = mlxcel_core::transpose(fn_weight);
    (mlxcel_core::matmul(&normalized, &fn_t), residual)
}

/// `sum(pre[..., None] * residual, axis = hc)` cast to `dtype`.
fn collapse(pre: &MlxArray, residual: &MlxArray, dtype: i32) -> UniquePtr<MlxArray> {
    let pre = mlxcel_core::expand_dims(pre, -1);
    let weighted = mlxcel_core::multiply(&pre, residual);
    let collapsed = mlxcel_core::sum_axis(&weighted, 2, false);
    mlxcel_core::astype(&collapsed, dtype)
}

/// Per-sublayer identity hyper connection (`hc_attn_layer.hc_pre`,
/// `hc_mlp_layer.hc_pre`).
pub(super) struct IdentityHyperConnection {
    /// `[2 * hc, hc * hidden]` float32.
    hc_fn: UniquePtr<MlxArray>,
    /// `[2 * hc]` float32.
    hc_base: UniquePtr<MlxArray>,
    /// `[2]` float32: the pre and post gate scales.
    hc_scale: UniquePtr<MlxArray>,
    hc_mult: i32,
    hc_eps: f32,
    hc_magnitude: f32,
    norm_eps: f32,
}

impl IdentityHyperConnection {
    /// Load `{prefix}.{hc_fn,hc_base,hc_scale}`.
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: i32,
        hc_mult: i32,
        hc_eps: f32,
        hc_magnitude: f32,
        norm_eps: f32,
    ) -> Result<Self, String> {
        let load = |name: &str, shape: &[i32]| {
            expect_hc_shape(
                weights,
                &format!("{prefix}.{name}"),
                shape,
                hc_mult,
                hidden_size,
            )
        };
        Ok(Self {
            hc_fn: load("hc_fn", &[2 * hc_mult, hc_mult * hidden_size])?,
            hc_base: load("hc_base", &[2 * hc_mult])?,
            hc_scale: load("hc_scale", &[2])?,
            hc_mult,
            hc_eps,
            hc_magnitude,
            norm_eps,
        })
    }

    /// Collapse the widened residual `h` (`[B, L, hc, D]`).
    ///
    /// Returns `(collapsed [B, L, D] in h's dtype, post [B, L, hc] f32)`.
    pub(super) fn forward(&self, h: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let hc = self.hc_mult;
        let dtype = mlxcel_core::array_dtype(h);
        let (mixes, residual) = mixes(h, &self.hc_fn, self.norm_eps);

        let pre_scale = slice_axis(&self.hc_scale, 0, 0, 1);
        let post_scale = slice_axis(&self.hc_scale, 0, 1, 2);
        let pre_base = slice_axis(&self.hc_base, 0, 0, hc);
        let post_base = slice_axis(&self.hc_base, 0, hc, 2 * hc);

        let pre = slice_axis(&mixes, -1, 0, hc);
        let pre = mlxcel_core::add(&mlxcel_core::multiply(&pre, &pre_scale), &pre_base);
        let eps = mlxcel_core::full_f32(&[1], self.hc_eps, mlxcel_core::dtype::FLOAT32);
        let pre = mlxcel_core::add(&mlxcel_core::sigmoid(&pre), &eps);

        let post = slice_axis(&mixes, -1, hc, 2 * hc);
        let post = mlxcel_core::add(&mlxcel_core::multiply(&post, &post_scale), &post_base);
        let post = mlxcel_core::multiply_scalar(&mlxcel_core::sigmoid(&post), self.hc_magnitude);

        (collapse(&pre, &residual, dtype), post)
    }
}

/// Fold a sublayer output `x` (`[B, L, D]`) back into the widened residual:
/// `residual + post[..., None] * x[:, :, None, :]` in float32, cast back to
/// the residual's dtype. No stream is mixed with another.
pub(super) fn hc_expand(x: &MlxArray, residual: &MlxArray, post: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(residual);
    let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let x = mlxcel_core::expand_dims(&x, 2);
    let post = mlxcel_core::expand_dims(post, -1);
    let update = mlxcel_core::multiply(&post, &x);
    let residual = mlxcel_core::astype(residual, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::astype(&mlxcel_core::add(&residual, &update), dtype)
}

/// Final collapse before `model.norm` and `lm_head` (`model.hc_head`).
pub(super) struct IdentityHyperHead {
    /// `[hc, hc * hidden]` float32.
    hc_head_fn: UniquePtr<MlxArray>,
    /// `[hc]` float32.
    hc_head_base: UniquePtr<MlxArray>,
    /// `[1]` float32.
    hc_head_scale: UniquePtr<MlxArray>,
    hc_eps: f32,
    norm_eps: f32,
}

impl IdentityHyperHead {
    /// Load `{prefix}.{hc_head_fn,hc_head_base,hc_head_scale}`.
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: i32,
        hc_mult: i32,
        hc_eps: f32,
        norm_eps: f32,
    ) -> Result<Self, String> {
        let load = |name: &str, shape: &[i32]| {
            expect_hc_shape(
                weights,
                &format!("{prefix}.{name}"),
                shape,
                hc_mult,
                hidden_size,
            )
        };
        Ok(Self {
            hc_head_fn: load("hc_head_fn", &[hc_mult, hc_mult * hidden_size])?,
            hc_head_base: load("hc_head_base", &[hc_mult])?,
            hc_head_scale: load("hc_head_scale", &[1])?,
            hc_eps,
            norm_eps,
        })
    }

    /// `h` is `[B, L, hc, D]`; returns `[B, L, D]` in `h`'s dtype.
    pub(super) fn forward(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let dtype = mlxcel_core::array_dtype(h);
        let (mixes, residual) = mixes(h, &self.hc_head_fn, self.norm_eps);
        let pre = mlxcel_core::multiply(&mixes, &self.hc_head_scale);
        let pre = mlxcel_core::add(&pre, &self.hc_head_base);
        let eps = mlxcel_core::full_f32(&[1], self.hc_eps, mlxcel_core::dtype::FLOAT32);
        let pre = mlxcel_core::add(&mlxcel_core::sigmoid(&pre), &eps);
        collapse(&pre, &residual, dtype)
    }
}

#[cfg(test)]
#[path = "hy_v4_hyper_tests.rs"]
mod tests;
