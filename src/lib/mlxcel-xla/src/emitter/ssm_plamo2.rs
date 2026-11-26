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

//! PLaMo 2's Mamba mixer (epic #564, issue #3347), assembled from the shared
//! conv and selective-scan primitives in [`super::ssm`].
//!
//! Per token `x [hidden]`:
//!
//! 1. `in_proj(x)` viewed as `[H, 2P]`; each head's first `P` lanes are the
//!    gate `z`, the next `P` the scan input (per-head interleaved, not two
//!    contiguous halves).
//! 2. A depthwise causal conv (no bias) and SiLU over the scan input only.
//! 3. `bcdt_proj(conv_x)` split `[B: N, C: N, dt: R]`; each piece is
//!    RMS-normalized over its own width and multiplied by its learned weight
//!    (no offset); then `dt_proj` maps the `R`-wide dt to the `H` heads.
//! 4. `dt = clamp(softplus(dt + dt_bias), 0.001, 100)` (MLX eager's limit; the
//!    HF CPU scan does not clamp), the Mamba2 scan with scalar `A = -exp(A_log)`
//!    per head, one shared B/C group and a per-head `D` skip.
//! 5. `out_proj(y * silu(z))`, with no gated RMSNorm.
//!
//! Padded prefill rows get `dt = 0` AFTER the clamp, so the 0.001 floor never
//! revives them: their decay is `exp(0) = 1` and their input term is zero.

use super::builder::{Builder, Val};
use super::config::{SsmConfig, SsmKind};
use super::ssm::{
    SsmLayerState, add_last, conv1d_causal_step, conv1d_over_left_context, selective_scan_prefill,
    selective_scan_step, silu, softplus, splat, tail_of_left_context, to_f32,
};

/// One PLaMo 2 Mamba mixer's weights, in the HF checkpoint layout.
#[derive(Clone, Debug)]
pub(crate) struct Plamo2MixerW {
    /// `[2 * d_inner, hidden]`, per-head interleaved `[z | x]`.
    pub in_proj: Val,
    /// `[d_inner, 1, K]`, the torch depthwise Conv1d layout.
    pub conv_w: Val,
    /// `[2N + R, d_inner]`.
    pub bcdt_proj: Val,
    /// `[H, R]`.
    pub dt_proj: Val,
    /// `[R]`, `[N]`, `[N]`: learned scales after the weightless RMSNorms.
    pub dt_norm: Val,
    pub b_norm: Val,
    pub c_norm: Val,
    /// `[H]` each.
    pub a_log: Val,
    pub d: Val,
    pub dt_bias: Val,
    /// `[hidden, d_inner]`.
    pub out_proj: Val,
}

fn dt_rank(c: &SsmConfig) -> usize {
    match c.kind {
        SsmKind::Plamo2 { dt_rank } => dt_rank,
        _ => panic!("the PLaMo 2 mixer requires SsmKind::Plamo2"),
    }
}

/// `in_proj` over `rows` inputs, split into `z` and the scan input, each
/// `[R, d_inner]` in head-major order.
fn in_projection(b: &mut Builder, c: &SsmConfig, w: &Plamo2MixerW, x: &Val) -> (Val, Val) {
    let rows = x.ty.shape[0];
    let (h, p) = (c.n_heads, c.head_dim);
    let x = to_f32(b, x);
    let zx = b.linear_seq_proj(&x, &w.in_proj);
    let zx = b.reshape(&zx, vec![rows, h, 2 * p]);
    let z = b.slice(&zx, &[(0, rows), (0, h), (0, p)]);
    let xs = b.slice(&zx, &[(0, rows), (0, h), (p, 2 * p)]);
    (
        b.reshape(&z, vec![rows, h * p]),
        b.reshape(&xs, vec![rows, h * p]),
    )
}

/// `x * rsqrt(mean(x^2) + eps) * w` over the last axis of `x [R, W]`.
fn weighted_rms(b: &mut Builder, x: &Val, w: &Val, eps: f32) -> Val {
    let (rows, width) = (x.ty.shape[0], x.ty.shape[1]);
    let square = b.multiply(x, x);
    let zero = b.const_f32(0.0);
    let sum = b.reduce_add(&square, 1, &zero);
    let denom = splat(b, width as f32, &[rows]);
    let mean = b.divide(&sum, &denom);
    let eps = splat(b, eps, &[rows]);
    let stabilized = b.add(&mean, &eps);
    let inverse = b.rsqrt(&stabilized);
    let inverse = b.broadcast(&inverse, &[0], vec![rows, width]);
    let normed = b.multiply(x, &inverse);
    let w = to_f32(b, w);
    let w = b.broadcast(&w, &[1], vec![rows, width]);
    b.multiply(&normed, &w)
}

/// B, C `[R, 1, N]` and the per-head pre-bias dt `[R, H]` from the activated
/// conv output `[R, d_inner]`.
fn bcdt(b: &mut Builder, c: &SsmConfig, w: &Plamo2MixerW, conv_x: &Val) -> (Val, Val, Val) {
    let rows = conv_x.ty.shape[0];
    let (n, r) = (c.state_size, dt_rank(c));
    let all = b.linear_seq_proj(conv_x, &w.bcdt_proj);
    let bm = b.slice(&all, &[(0, rows), (0, n)]);
    let cm = b.slice(&all, &[(0, rows), (n, 2 * n)]);
    let dt = b.slice(&all, &[(0, rows), (2 * n, 2 * n + r)]);
    let eps = c.norm_eps;
    let bm = weighted_rms(b, &bm, &w.b_norm, eps);
    let cm = weighted_rms(b, &cm, &w.c_norm, eps);
    let dt = weighted_rms(b, &dt, &w.dt_norm, eps);
    let dt = b.linear_seq_proj(&dt, &w.dt_proj);
    (
        b.reshape(&bm, vec![rows, 1, n]),
        b.reshape(&cm, vec![rows, 1, n]),
        dt,
    )
}

/// `clamp(softplus(dt + dt_bias), time_step_limit)` over `[R, H]`.
fn discretize_dt(b: &mut Builder, c: &SsmConfig, dt: &Val, dt_bias: &Val) -> Val {
    let dt = add_last(b, dt, dt_bias);
    let mut dt = softplus(b, &dt);
    let (low, high) = c.time_step_limit;
    if low > 0.0 {
        let floor = splat(b, low, &dt.ty.shape);
        dt = b.maximum(&dt, &floor);
    }
    if high.is_finite() {
        let ceiling = splat(b, high, &dt.ty.shape);
        dt = b.minimum(&dt, &ceiling);
    }
    dt
}

/// `A = -exp(A_log)` per head.
fn decay_rate(b: &mut Builder, a_log: &Val) -> Val {
    let a_log = to_f32(b, a_log);
    let e = b.exponential(&a_log);
    b.negate(&e)
}

/// `out_proj(y * silu(z))`; `y [R, H, P]`, `z [R, d_inner]`.
fn gated_out(b: &mut Builder, w: &Plamo2MixerW, y: &Val, z: &Val) -> Val {
    let (rows, di) = (z.ty.shape[0], z.ty.shape[1]);
    let y = b.reshape(y, vec![rows, di]);
    let gate = silu(b, z);
    let gated = b.multiply(&y, &gate);
    b.linear_seq_proj(&gated, &w.out_proj)
}

/// One PLaMo 2 mixer over a padded prompt `x [T, hidden]` with `real_len` live
/// rows, from the carried state `h0 [H, P, N]` and conv tail
/// `conv0 [K-1, d_inner]`. Returns the output rows, `h_T`, and the new tail.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plamo2_mixer_prefill(
    b: &mut Builder,
    c: &SsmConfig,
    w: &Plamo2MixerW,
    x: &Val,
    real_len: &Val,
    h0: &Val,
    conv0: &Val,
) -> (Val, Val, Val) {
    let t = x.ty.shape[0];
    let k = c.conv_kernel;
    let (z, xs) = in_projection(b, c, w, x);
    let conv0 = to_f32(b, conv0);
    let xp = b.concatenate(&conv0, &xs, 0);
    let conv = conv1d_over_left_context(b, &xp, &w.conv_w, None, k);
    let tail = tail_of_left_context(b, &xp, real_len, k);
    let conv_x = silu(b, &conv);
    let (bm, cm, dt) = bcdt(b, c, w, &conv_x);
    let dt = discretize_dt(b, c, &dt, &w.dt_bias);
    // Padded rows advance nothing: dt = 0 for t >= real_len, after the clamp.
    let rows = b.iota(t);
    let rows = b.broadcast(&rows, &[0], vec![t, c.n_heads]);
    let limit = b.broadcast(real_len, &[], vec![t, c.n_heads]);
    let live = b.compare("LT", &rows, &limit, "SIGNED");
    let zeros = splat(b, 0.0, &[t, c.n_heads]);
    let dt = b.select(&live, &dt, &zeros);
    let xs = b.reshape(&conv_x, vec![t, c.n_heads, c.head_dim]);
    let a = decay_rate(b, &w.a_log);
    let (y, h_t) = selective_scan_prefill(b, c, &xs, &dt, &a, &bm, &cm, &w.d, h0);
    (gated_out(b, w, &y, &z), h_t, tail)
}

/// One PLaMo 2 mixer decode step over `x [B, hidden]` from `st`.
pub(crate) fn plamo2_mixer_step(
    b: &mut Builder,
    c: &SsmConfig,
    w: &Plamo2MixerW,
    x: &Val,
    st: &SsmLayerState,
) -> (Val, SsmLayerState) {
    let batch = x.ty.shape[0];
    let (z, xs) = in_projection(b, c, w, x);
    let (conv, tail) = conv1d_causal_step(b, &xs, &st.conv, &w.conv_w, None, c.conv_kernel);
    let conv_x = silu(b, &conv);
    let (bm, cm, dt) = bcdt(b, c, w, &conv_x);
    let dt = discretize_dt(b, c, &dt, &w.dt_bias);
    let xs = b.reshape(&conv_x, vec![batch, c.n_heads, c.head_dim]);
    let a = decay_rate(b, &w.a_log);
    let (y, h) = selective_scan_step(b, c, &xs, &dt, &a, &bm, &cm, &w.d, &st.h);
    (gated_out(b, w, &y, &z), SsmLayerState { h, conv: tail })
}
