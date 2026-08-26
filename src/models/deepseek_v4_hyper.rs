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

//! DeepSeek-V4 HyperConnections (`hyper_connection.py` in the reference).
//!
//! The hidden state carried between V4 blocks is rank-4
//! `[B, L, hc_mult, hidden]`, not the usual `[B, L, hidden]`. Each sublayer
//! runs `x, post, comb = hc(h)` to collapse the widened state into a
//! `[B, L, hidden]` input, and `hc_expand(x, residual, post, comb)` to fold
//! the sublayer output back into the widened residual. `HyperHead` collapses
//! the final widened state before `model.norm` and `lm_head`.
//!
//! Two details are silently wrong if approximated:
//!
//! * **The Sinkhorn loop.** `comb` is softmaxed row-wise, `hc_eps` is added,
//!   an initial COLUMN normalization runs, and then `hc_sinkhorn_iters - 1`
//!   iterations alternate row and column normalization (each with `+ eps` in
//!   the denominator). Both the iteration count (20 on the real checkpoint)
//!   and the row/column ORDER change the output; any deviation yields finite,
//!   plausible, wrong logits.
//! * **float32 throughout.** `fn` / `base` / `scale` ship as float32 and the
//!   whole gate computation runs in float32 over an f32 copy of the widened
//!   state (`y`), with only the collapsed output cast back to the activation
//!   dtype. The reference `cast_predicate` excludes these tensors from
//!   checkpoint-time casting for the same reason.
//!
//! The reference also carries a fused Metal kernel
//! (`_hc_sinkhorn_collapse_kernel`). This port implements the pure-ops path
//! (`_hc_ops` / `_hc_split_sinkhorn_ops`) only: the ops path is the
//! correctness baseline the reference itself uses off-Metal and in training;
//! the fused kernel is a follow-up optimization, out of scope for the port.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::get_weight_copy;

/// Per-sublayer hyper connection (`attn_hc` / `ffn_hc`).
pub(crate) struct HyperConnection {
    /// `[(2 + hc) * hc, hc * hidden]` float32.
    fn_weight: UniquePtr<MlxArray>,
    /// `[(2 + hc) * hc]` float32.
    base: UniquePtr<MlxArray>,
    /// `[3]` float32: pre / post / comb gate scales.
    scale: UniquePtr<MlxArray>,
    hc_mult: i32,
    sinkhorn_iters: i32,
    hc_eps: f32,
    norm_eps: f32,
}

fn expect_shape(
    weights: &WeightMap,
    name: &str,
    expected: &[i32],
) -> Result<UniquePtr<MlxArray>, String> {
    let w = get_weight_copy(weights, name)?;
    let shape = mlxcel_core::array_shape(&w);
    if shape != expected {
        return Err(format!(
            "{name}: expected shape {expected:?}, checkpoint ships {shape:?}"
        ));
    }
    Ok(w)
}

/// `x * s + b` where `s` is a `[1]` slice of the scale vector and `b` a slice
/// of the base vector; all f32.
fn affine_gate(x: &MlxArray, s: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray> {
    let scaled = mlxcel_core::multiply(x, s);
    mlxcel_core::add(&scaled, b)
}

impl HyperConnection {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: i32,
        hc_mult: i32,
        sinkhorn_iters: i32,
        hc_eps: f32,
        norm_eps: f32,
    ) -> Result<Self, String> {
        let mix = (2 + hc_mult) * hc_mult;
        Ok(Self {
            fn_weight: expect_shape(
                weights,
                &format!("{prefix}.fn"),
                &[mix, hc_mult * hidden_size],
            )?,
            base: expect_shape(weights, &format!("{prefix}.base"), &[mix])?,
            scale: expect_shape(weights, &format!("{prefix}.scale"), &[3])?,
            hc_mult,
            sinkhorn_iters,
            hc_eps,
            norm_eps,
        })
    }

    /// Collapse the widened state.
    ///
    /// `x` is `[B, L, hc, D]`. Returns `(collapsed [B, L, D] in x's dtype,
    /// post [B, L, hc] f32, comb [B, L, hc, hc] f32)`.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let (b, l, h, d) = (shape[0], shape[1], shape[2], shape[3]);
        let hc = self.hc_mult;
        debug_assert_eq!(h, hc);
        let x_dtype = mlxcel_core::array_dtype(x);

        let y = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let z = mlxcel_core::reshape(&y, &[b, l, h * d]);
        let z = mlxcel_core::fast_rms_norm_no_weight(&z, self.norm_eps);
        let fn_t = mlxcel_core::transpose(&self.fn_weight);
        let mixes = mlxcel_core::matmul(&z, &fn_t);

        let s0 = mlxcel_core::utils::slice_axis(&self.scale, 0, 0, 1);
        let s1 = mlxcel_core::utils::slice_axis(&self.scale, 0, 1, 2);
        let s2 = mlxcel_core::utils::slice_axis(&self.scale, 0, 2, 3);
        let base_pre = mlxcel_core::utils::slice_axis(&self.base, 0, 0, hc);
        let base_post = mlxcel_core::utils::slice_axis(&self.base, 0, hc, 2 * hc);
        let base_comb = mlxcel_core::utils::slice_axis(&self.base, 0, 2 * hc, -1);
        let base_comb = mlxcel_core::reshape(&base_comb, &[hc, hc]);

        let eps = mlxcel_core::full_f32(&[1], self.hc_eps, mlxcel_core::dtype::FLOAT32);

        // pre = sigmoid(mix[..., :hc] * scale[0] + base[:hc]) + eps
        let pre_in = mlxcel_core::utils::slice_axis(&mixes, -1, 0, hc);
        let pre = mlxcel_core::sigmoid(&affine_gate(&pre_in, &s0, &base_pre));
        let pre = mlxcel_core::add(&pre, &eps);

        // post = 2 * sigmoid(mix[..., hc:2hc] * scale[1] + base[hc:2hc])
        let post_in = mlxcel_core::utils::slice_axis(&mixes, -1, hc, 2 * hc);
        let post = mlxcel_core::sigmoid(&affine_gate(&post_in, &s1, &base_post));
        let post = mlxcel_core::multiply_scalar(&post, 2.0);

        // comb: softmax rows, + eps, then Sinkhorn normalization.
        let comb_in = mlxcel_core::utils::slice_axis(&mixes, -1, 2 * hc, -1);
        let comb_in = mlxcel_core::reshape(&comb_in, &[b, l, hc, hc]);
        let comb = affine_gate(&comb_in, &s2, &base_comb);
        let comb = mlxcel_core::softmax_precise(&comb, -1);
        let mut comb = mlxcel_core::add(&comb, &eps);
        // Initial column normalization, then (iters - 1) row/column rounds.
        let col_sum = mlxcel_core::sum_axis(&comb, -2, true);
        comb = mlxcel_core::divide(&comb, &mlxcel_core::add(&col_sum, &eps));
        for _ in 0..(self.sinkhorn_iters - 1).max(0) {
            let row_sum = mlxcel_core::sum_axis(&comb, -1, true);
            comb = mlxcel_core::divide(&comb, &mlxcel_core::add(&row_sum, &eps));
            let col_sum = mlxcel_core::sum_axis(&comb, -2, true);
            comb = mlxcel_core::divide(&comb, &mlxcel_core::add(&col_sum, &eps));
        }

        // collapsed = (pre[..., None] * y).sum(axis=2), back to x's dtype.
        let pre_exp = mlxcel_core::expand_dims(&pre, -1);
        let weighted = mlxcel_core::multiply(&pre_exp, &y);
        let collapsed = mlxcel_core::sum_axis(&weighted, 2, false);
        let collapsed = mlxcel_core::astype(&collapsed, x_dtype);

        (collapsed, post, comb)
    }
}

/// Fold a sublayer output back into the widened residual:
/// `post[..., None] * x[:, :, None, :] + comb.T @ residual`, computed in f32
/// and cast back to `x`'s dtype.
pub(crate) fn hc_expand(
    x: &MlxArray,
    residual: &MlxArray,
    post: &MlxArray,
    comb: &MlxArray,
) -> UniquePtr<MlxArray> {
    let x_dtype = mlxcel_core::array_dtype(x);
    let xf = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let xf = mlxcel_core::expand_dims(&xf, 2);
    let post_exp = mlxcel_core::expand_dims(post, -1);
    let y = mlxcel_core::multiply(&post_exp, &xf);
    let comb_t = mlxcel_core::transpose_axes(comb, &[0, 1, 3, 2]);
    let res_f = mlxcel_core::astype(residual, mlxcel_core::dtype::FLOAT32);
    let mixed = mlxcel_core::matmul(&comb_t, &res_f);
    let y = mlxcel_core::add(&y, &mixed);
    mlxcel_core::astype(&y, x_dtype)
}

/// Final collapse before `model.norm` / `lm_head` (`model.hc_head`).
pub(crate) struct HyperHead {
    /// `[hc, hc * hidden]` float32.
    fn_weight: UniquePtr<MlxArray>,
    /// `[hc]` float32.
    base: UniquePtr<MlxArray>,
    /// `[1]` float32.
    scale: UniquePtr<MlxArray>,
    hc_eps: f32,
    norm_eps: f32,
}

impl HyperHead {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: i32,
        hc_mult: i32,
        hc_eps: f32,
        norm_eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            fn_weight: expect_shape(
                weights,
                &format!("{prefix}.fn"),
                &[hc_mult, hc_mult * hidden_size],
            )?,
            base: expect_shape(weights, &format!("{prefix}.base"), &[hc_mult])?,
            scale: expect_shape(weights, &format!("{prefix}.scale"), &[1])?,
            hc_eps,
            norm_eps,
        })
    }

    /// `x` is `[B, L, hc, D]`; returns `[B, L, D]` in `x`'s dtype.
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l, h, d) = (shape[0], shape[1], shape[2], shape[3]);
        let x_dtype = mlxcel_core::array_dtype(x);

        let y = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let z = mlxcel_core::reshape(&y, &[b, l, h * d]);
        let z = mlxcel_core::fast_rms_norm_no_weight(&z, self.norm_eps);
        let fn_t = mlxcel_core::transpose(&self.fn_weight);
        let mixes = mlxcel_core::matmul(&z, &fn_t);

        let pre = mlxcel_core::sigmoid(&affine_gate(&mixes, &self.scale, &self.base));
        let eps = mlxcel_core::full_f32(&[1], self.hc_eps, mlxcel_core::dtype::FLOAT32);
        let pre = mlxcel_core::add(&pre, &eps);

        let pre_exp = mlxcel_core::expand_dims(&pre, -1);
        let weighted = mlxcel_core::multiply(&pre_exp, &y);
        let collapsed = mlxcel_core::sum_axis(&weighted, 2, false);
        mlxcel_core::astype(&collapsed, x_dtype)
    }
}
