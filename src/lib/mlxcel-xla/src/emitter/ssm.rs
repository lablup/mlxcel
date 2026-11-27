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

//! Selective state-space mixer primitives (epic #564, issue #3337).
//!
//! A Mamba mixer layer is a causal depthwise conv1d followed by a selective
//! scan over a fixed-size f32 state: Mamba2 uses `h [H,P,N]`, while Mamba1 and
//! Falcon-Mamba use a diagonal `h [D,N]`. Both come in a one-step decode form
//! (with an explicit leading batch axis `B`) and a whole-prompt prefill form
//! over the padded `context_capacity` rows. `spike/openxla/ssm_oracle.py`
//! checks both families against their independent Transformers slow paths.
//!
//! Prefill invariants: rows `t >= real_len` get `dt = 0`, so their decay is
//! `exp(0) = 1` and their input term vanishes; the state passes through them
//! unchanged. The conv tail is the last `K-1` pre-conv inputs ending at
//! `real_len`, taken with `dynamic_slice` from the sequence left-padded with the
//! incoming tail (zeros for a fresh sequence). The builder has no `while` op and
//! IREE-CUDA rejects scatter, so every form uses `dot_general`, `pad`, slices
//! and `select` only (no `reduce_window`: IREE-Metal miscompiles the prefix-sum
//! form, issue #3343). The prefill form choice and its measured
//! alternatives are recorded in `spike/openxla/FINDINGS_ssm.md`.

use super::builder::{Builder, Ty, Val};
use super::config::{SsmConfig, SsmKind};
#[allow(unused_imports)] // Public within the emitter for #3349/#3356 reuse.
pub(crate) use super::ssm_mamba1::{
    BcdtNormW, Mamba1MixerW, diag_scan_prefill, diag_scan_step, mamba1_mixer_prefill,
    mamba1_mixer_step,
};
pub(crate) use super::ssm_plamo2::{Plamo2MixerW, plamo2_mixer_prefill, plamo2_mixer_step};
use super::ssm_scan::{SsmPrefillForm, selective_scan_prefill_form};

/// One Mamba2 mixer layer's weights, in the HF checkpoint layout.
#[derive(Clone, Debug)]
pub(crate) struct Mamba2MixerW {
    pub in_proj: Val,
    pub in_proj_b: Option<Val>,
    /// `[C, 1, K]`, the torch depthwise Conv1d layout.
    pub conv_w: Val,
    pub conv_b: Option<Val>,
    /// `[H]`.
    pub dt_bias: Val,
    /// `[H]`.
    pub a_log: Val,
    /// `[H]`.
    pub d: Val,
    /// `[d_inner]`.
    pub norm_w: Option<Val>,
    pub falcon: Option<super::falcon_h1_config::FalconH1Config>,
    pub out_proj: Val,
    pub out_proj_b: Option<Val>,
}

/// Recurrent mixer weights carried by one model layer.
#[derive(Clone, Debug)]
pub(crate) enum SsmMixerW {
    Mamba2(Mamba2MixerW),
    Mamba1(Mamba1MixerW),
    Plamo2(Plamo2MixerW),
}

/// One recurrent layer's carried state.
#[derive(Clone, Debug)]
pub(crate) struct SsmLayerState {
    /// Mamba2: `[B,H,P,N]`; Mamba1/Falcon-Mamba: `[B,D,N]`. Always f32.
    pub h: Val,
    /// `[B, K-1, C]` f32, the last `K-1` pre-conv inputs.
    pub conv: Val,
}

/// Cast to f32 when needed, so the recurrence never runs in a narrow type.
pub(super) fn to_f32(b: &mut Builder, v: &Val) -> Val {
    if v.ty.elt == "f32" {
        v.clone()
    } else {
        b.convert(v, "f32")
    }
}

/// Broadcast an f32 scalar constant to `shape`.
pub(super) fn splat(b: &mut Builder, value: f32, shape: &[usize]) -> Val {
    let scalar = b.const_f32(value);
    b.broadcast(&scalar, &[], shape.to_vec())
}

/// Multiply without adding operations to existing families when the factor is one.
pub(super) fn scale(b: &mut Builder, x: &Val, factor: f32) -> Val {
    if factor == 1.0 {
        return x.clone();
    }
    let f = splat(b, factor, &x.ty.shape);
    b.multiply(x, &f)
}

fn sigmoid(b: &mut Builder, x: &Val) -> Val {
    let negative = b.negate(x);
    let exponential = b.exponential(&negative);
    let one = splat(b, 1.0, &x.ty.shape);
    let denominator = b.add(&one, &exponential);
    b.divide(&one, &denominator)
}

pub(super) fn silu(b: &mut Builder, x: &Val) -> Val {
    let gate = sigmoid(b, x);
    b.multiply(x, &gate)
}

/// `softplus(x) = max(x, 0) + log1p(exp(-|x|))`, finite for every input.
pub(super) fn softplus(b: &mut Builder, x: &Val) -> Val {
    let zero = splat(b, 0.0, &x.ty.shape);
    let positive = b.maximum(x, &zero);
    let magnitude = b.abs(x);
    let negative = b.negate(&magnitude);
    let exponential = b.exponential(&negative);
    let tail = b.log_plus_one(&exponential);
    b.add(&positive, &tail)
}

/// Broadcast a per-group tensor `[.., G, N]` to per-head `[.., H, N]`; head `h`
/// reads group `h / (H / G)` (HF `repeat_interleave`).
pub(super) fn heads_from_groups(b: &mut Builder, v: &Val, n_heads: usize) -> Val {
    let rank = v.ty.shape.len();
    let lead = &v.ty.shape[..rank - 2];
    let (groups, n) = (v.ty.shape[rank - 2], v.ty.shape[rank - 1]);
    if groups == n_heads {
        return v.clone();
    }
    let repeat = n_heads / groups;
    let mut wide = lead.to_vec();
    wide.extend([groups, repeat, n]);
    let mut dims: Vec<usize> = (0..rank - 1).collect();
    dims.push(rank);
    let wide = b.broadcast(v, &dims, wide);
    let mut out = lead.to_vec();
    out.extend([n_heads, n]);
    b.reshape(&wide, out)
}

/// Add a `[C]` bias over the last axis.
pub(super) fn add_last(b: &mut Builder, x: &Val, bias: &Val) -> Val {
    let last = x.ty.shape.len() - 1;
    let bias = to_f32(b, bias);
    let bias = b.broadcast(&bias, &[last], x.ty.shape.clone());
    b.add(x, &bias)
}

/// Depthwise causal conv over a sequence that already carries its `K-1` left
/// context rows: `xp [T+K-1, C] -> [T, C]`.
///
/// Computed as `K` shifted multiply-adds in f32 rather than a convolution op,
/// so the prefill rounds exactly like the decode step ([`conv1d_causal_step`])
/// on every precision path: `Builder::convolution` demotes its operands on the
/// f16 path, and a mixed-type depthwise convolution does not compile
/// (`spike/openxla/FINDINGS_ssm.md`).
pub(super) fn conv1d_over_left_context(
    b: &mut Builder,
    xp: &Val,
    w: &Val,
    bias: Option<&Val>,
    k: usize,
) -> Val {
    let (rows, channels) = (xp.ty.shape[0], xp.ty.shape[1]);
    let t = rows + 1 - k;
    let xp = to_f32(b, xp);
    let w = to_f32(b, w);
    // torch [C, 1, K] -> [C, K]; tap j multiplies input row t + j.
    let w = b.reshape(&w, vec![channels, k]);
    let mut out: Option<Val> = None;
    for tap in 0..k {
        let window = b.slice(&xp, &[(tap, tap + t), (0, channels)]);
        let weight = b.slice(&w, &[(0, channels), (tap, tap + 1)]);
        let weight = b.reshape(&weight, vec![channels]);
        let weight = b.broadcast(&weight, &[1], vec![t, channels]);
        let product = b.multiply(&window, &weight);
        out = Some(match out {
            Some(sum) => b.add(&sum, &product),
            None => product,
        });
    }
    let out = out.expect("conv_kernel is at least 2");
    match bias {
        Some(bias) => add_last(b, &out, bias),
        None => out,
    }
}

/// Causal depthwise conv1d over a fresh sequence: `x [T, C] -> [T, C]`,
/// left-padded with `K-1` zero rows (torch `Conv1d(padding=K-1)[..T]`).
pub(crate) fn conv1d_causal_prefill(
    b: &mut Builder,
    x: &Val,
    w: &Val,
    bias: Option<&Val>,
    k: usize,
) -> Val {
    let x = to_f32(b, x);
    let zero = b.const_f32(0.0);
    let xp = b.pad(&x, &zero, &[k - 1, 0], &[0, 0]);
    conv1d_over_left_context(b, &xp, w, bias, k)
}

/// The last `K-1` pre-conv rows of `xp [T+K-1, C]` ending at `real_len`.
pub(super) fn tail_of_left_context(b: &mut Builder, xp: &Val, real_len: &Val, k: usize) -> Val {
    let channels = xp.ty.shape[1];
    let zero = b.const_i32(0);
    b.dynamic_slice(xp, &[real_len, &zero], vec![k - 1, channels])
}

/// The conv tail of a fresh sequence: the last `K-1` pre-conv inputs ending at
/// `real_len`, with leading zeros when `real_len < K-1`.
pub(crate) fn conv1d_tail(b: &mut Builder, x: &Val, real_len: &Val, k: usize) -> Val {
    let x = to_f32(b, x);
    let zero = b.const_f32(0.0);
    let xp = b.pad(&x, &zero, &[k - 1, 0], &[0, 0]);
    tail_of_left_context(b, &xp, real_len, k)
}

/// One decode step of the causal conv: the window is the carried tail plus the
/// new input. Returns the conv output and the advanced tail.
pub(crate) fn conv1d_causal_step(
    b: &mut Builder,
    x_t: &Val,
    tail: &Val,
    w: &Val,
    bias: Option<&Val>,
    k: usize,
) -> (Val, Val) {
    let (batch, channels) = (x_t.ty.shape[0], x_t.ty.shape[1]);
    let x_t = to_f32(b, x_t);
    let tail = to_f32(b, tail);
    let row = b.reshape(&x_t, vec![batch, 1, channels]);
    let window = b.concatenate(&tail, &row, 1);
    let w = to_f32(b, w);
    let w = b.reshape(&w, vec![channels, k]);
    let w = b.transpose(&w, &[1, 0]);
    let w = b.broadcast(&w, &[1, 2], vec![batch, k, channels]);
    let products = b.multiply(&window, &w);
    let zero = b.const_f32(0.0);
    let out = b.reduce_add(&products, 1, &zero);
    let out = match bias {
        Some(bias) => add_last(b, &out, bias),
        None => out,
    };
    let next = b.slice(&window, &[(0, batch), (1, k), (0, channels)]);
    (out, next)
}

/// One selective-scan step: `h' = exp(dt*A) h + dt (x outer B)`,
/// `y = C . h' + D x`. Runs in f32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn selective_scan_step(
    b: &mut Builder,
    c: &SsmConfig,
    x: &Val,
    dt: &Val,
    a: &Val,
    bm: &Val,
    cm: &Val,
    d: &Val,
    h: &Val,
) -> (Val, Val) {
    let (hh, p, n) = (c.n_heads, c.head_dim, c.state_size);
    let batch = x.ty.shape[0];
    let [x, dt, a, bm, cm, d, h] = [x, dt, a, bm, cm, d, h].map(|v| to_f32(b, v));
    let full = vec![batch, hh, p, n];
    let a_b = b.broadcast(&a, &[1], vec![batch, hh]);
    let dta = b.multiply(&dt, &a_b);
    let decay = b.exponential(&dta);
    let decay = b.broadcast(&decay, &[0, 1], full.clone());
    let dt_x = b.broadcast(&dt, &[0, 1], vec![batch, hh, p]);
    let dt_x = b.multiply(&x, &dt_x);
    let dt_x = b.broadcast(&dt_x, &[0, 1, 2], full.clone());
    let bh = heads_from_groups(b, &bm, hh);
    let bh = b.broadcast(&bh, &[0, 1, 3], full);
    let input = b.multiply(&dt_x, &bh);
    let kept = b.multiply(&h, &decay);
    let next = b.add(&kept, &input);
    let ch = heads_from_groups(b, &cm, hh);
    let y = b.dot_general_keep_f32(&next, &ch, &[0, 1], &[0, 1], &[3], &[2], vec![batch, hh, p]);
    let d_b = b.broadcast(&d, &[1], vec![batch, hh, p]);
    let skip = b.multiply(&x, &d_b);
    (b.add(&y, &skip), next)
}

/// The whole-prompt selective scan in the chosen prefill form (chunked SSD with
/// `c.chunk_size`). Returns `y [T, H, P]` and the final state `h_T [H, P, N]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn selective_scan_prefill(
    b: &mut Builder,
    c: &SsmConfig,
    x: &Val,
    dt: &Val,
    a: &Val,
    bm: &Val,
    cm: &Val,
    d: &Val,
    h0: &Val,
) -> (Val, Val) {
    selective_scan_prefill_form(b, c, SsmPrefillForm::ChunkedSsd, x, dt, a, bm, cm, d, h0)
}

/// `[z, xBC, dt]` from one in_proj over `rows` input rows.
fn in_projection(b: &mut Builder, c: &SsmConfig, w: &Mamba2MixerW, x: &Val) -> (Val, Val, Val) {
    let rows = x.ty.shape[0];
    let x = to_f32(b, x);
    let x = w
        .falcon
        .as_ref()
        .map_or_else(|| x.clone(), |f| scale(b, &x, f.ssm_in));
    let zxbcdt = b.linear_seq_proj(&x, &w.in_proj);
    let zxbcdt = match &w.in_proj_b {
        Some(bias) => add_last(b, &zxbcdt, bias),
        None => zxbcdt,
    };
    let zxbcdt = if let Some(f) = &w.falcon {
        let gn = c.n_groups * c.state_size;
        let mut start = 0;
        let mut pieces = Vec::new();
        for (len, factor) in [c.d_inner(), c.d_inner(), gn, gn, c.n_heads]
            .into_iter()
            .zip(f.ssm_channels)
        {
            let part = b.slice(&zxbcdt, &[(0, rows), (start, start + len)]);
            pieces.push(scale(b, &part, factor));
            start += len;
        }
        let mut joined = pieces[0].clone();
        for part in &pieces[1..] {
            joined = b.concatenate(&joined, part, 1);
        }
        joined
    } else {
        zxbcdt
    };
    let (di, cd) = (c.d_inner(), c.conv_dim);
    let z = b.slice(&zxbcdt, &[(0, rows), (0, di)]);
    let xbc = b.slice(&zxbcdt, &[(0, rows), (di, di + cd)]);
    let dt = b.slice(&zxbcdt, &[(0, rows), (di + cd, di + cd + c.n_heads)]);
    (z, xbc, dt)
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

/// Split the activated `xBC [R, conv_dim]` into `x [R, H, P]`, `B`, `C [R, G, N]`.
fn split_xbc(b: &mut Builder, c: &SsmConfig, xbc: &Val) -> (Val, Val, Val) {
    let rows = xbc.ty.shape[0];
    let (di, gn) = (c.d_inner(), c.n_groups * c.state_size);
    let x = b.slice(xbc, &[(0, rows), (0, di)]);
    let x = b.reshape(&x, vec![rows, c.n_heads, c.head_dim]);
    let bm = b.slice(xbc, &[(0, rows), (di, di + gn)]);
    let bm = b.reshape(&bm, vec![rows, c.n_groups, c.state_size]);
    let cm = b.slice(xbc, &[(0, rows), (di + gn, di + 2 * gn)]);
    let cm = b.reshape(&cm, vec![rows, c.n_groups, c.state_size]);
    (x, bm, cm)
}

/// `A = -exp(A_log)` per head.
fn decay_rate(b: &mut Builder, a_log: &Val) -> Val {
    let a_log = to_f32(b, a_log);
    let e = b.exponential(&a_log);
    b.negate(&e)
}

/// Gated RMSNorm `rmsnorm(y * silu(z)) * w` over groups of `norm_group_size`,
/// then out_proj. `y [R, H, P]`, `z [R, d_inner]` -> `[R, hidden]`.
fn gated_norm_out(b: &mut Builder, c: &SsmConfig, w: &Mamba2MixerW, y: &Val, z: &Val) -> Val {
    let rows = z.ty.shape[0];
    let (di, gs) = (c.d_inner(), c.norm_group_size);
    let y = b.reshape(y, vec![rows, di]);
    let gate = silu(b, z);
    let before = w.falcon.as_ref().is_some_and(|f| f.norm_before_gate);
    let g = if before && w.norm_w.is_some() {
        y
    } else {
        b.multiply(&y, &gate)
    };
    let weighted = if let Some(norm_w) = &w.norm_w {
        let grouped = b.reshape(&g, vec![rows, di / gs, gs]);
        let square = b.multiply(&grouped, &grouped);
        let zero = b.const_f32(0.0);
        let sum = b.reduce_add(&square, 2, &zero);
        let width = splat(b, gs as f32, &sum.ty.shape);
        let mean = b.divide(&sum, &width);
        let eps = splat(b, c.norm_eps, &mean.ty.shape);
        let stabilized = b.add(&mean, &eps);
        let inverse = b.rsqrt(&stabilized);
        let inverse = b.broadcast(&inverse, &[0, 1], grouped.ty.shape.clone());
        let normed = b.multiply(&grouped, &inverse);
        let normed = b.reshape(&normed, vec![rows, di]);
        let weighted = {
            let norm_w = to_f32(b, norm_w);
            let norm_w = b.broadcast(&norm_w, &[1], vec![rows, di]);
            b.multiply(&normed, &norm_w)
        };
        if before {
            b.multiply(&weighted, &gate)
        } else {
            weighted
        }
    } else {
        g
    };
    let out = b.linear_seq_proj(&weighted, &w.out_proj);
    let out = match &w.out_proj_b {
        Some(bias) => add_last(b, &out, bias),
        None => out,
    };
    w.falcon
        .as_ref()
        .map_or_else(|| out.clone(), |f| scale(b, &out, f.ssm_out))
}

/// One Mamba2 mixer layer over a padded prompt `x [T, hidden]` with `real_len`
/// live rows, from the carried state `h0 [H, P, N]` and conv tail
/// `conv0 [K-1, C]`. Returns the output rows, `h_T`, and the new conv tail.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mamba2_mixer_prefill(
    b: &mut Builder,
    c: &SsmConfig,
    w: &Mamba2MixerW,
    x: &Val,
    real_len: &Val,
    h0: &Val,
    conv0: &Val,
) -> (Val, Val, Val) {
    mamba2_mixer_prefill_form(b, c, SsmPrefillForm::ChunkedSsd, w, x, real_len, h0, conv0)
}

/// [`mamba2_mixer_prefill`] with an explicit prefill scan form (the lowering
/// spike measures the alternatives through the probe).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mamba2_mixer_prefill_form(
    b: &mut Builder,
    c: &SsmConfig,
    form: SsmPrefillForm,
    w: &Mamba2MixerW,
    x: &Val,
    real_len: &Val,
    h0: &Val,
    conv0: &Val,
) -> (Val, Val, Val) {
    let t = x.ty.shape[0];
    let k = c.conv_kernel;
    let (z, xbc, dt) = in_projection(b, c, w, x);
    let conv0 = to_f32(b, conv0);
    let xp = b.concatenate(&conv0, &xbc, 0);
    let conv = conv1d_over_left_context(b, &xp, &w.conv_w, w.conv_b.as_ref(), k);
    let tail = tail_of_left_context(b, &xp, real_len, k);
    let xbc = silu(b, &conv);
    let (xs, bm, cm) = split_xbc(b, c, &xbc);
    let dt = discretize_dt(b, c, &dt, &w.dt_bias);
    // Padded rows advance nothing: dt = 0 for t >= real_len.
    let rows = b.iota(t);
    let rows = b.broadcast(&rows, &[0], vec![t, c.n_heads]);
    let limit = b.broadcast(real_len, &[], vec![t, c.n_heads]);
    let live = b.compare("LT", &rows, &limit, "SIGNED");
    let zeros = splat(b, 0.0, &[t, c.n_heads]);
    let dt = b.select(&live, &dt, &zeros);
    let a = decay_rate(b, &w.a_log);
    let (y, h_t) = selective_scan_prefill_form(b, c, form, &xs, &dt, &a, &bm, &cm, &w.d, h0);
    let out = gated_norm_out(b, c, w, &y, &z);
    (out, h_t, tail)
}

/// One Mamba2 mixer decode step over `x [B, hidden]` from `st`.
pub(crate) fn mamba2_mixer_step(
    b: &mut Builder,
    c: &SsmConfig,
    w: &Mamba2MixerW,
    x: &Val,
    st: &SsmLayerState,
) -> (Val, SsmLayerState) {
    let (z, xbc, dt) = in_projection(b, c, w, x);
    let (conv, tail) = conv1d_causal_step(
        b,
        &xbc,
        &st.conv,
        &w.conv_w,
        w.conv_b.as_ref(),
        c.conv_kernel,
    );
    let xbc = silu(b, &conv);
    let (xs, bm, cm) = split_xbc(b, c, &xbc);
    let dt = discretize_dt(b, c, &dt, &w.dt_bias);
    let a = decay_rate(b, &w.a_log);
    let (y, h) = selective_scan_step(b, c, &xs, &dt, &a, &bm, &cm, &w.d, &st.h);
    let out = gated_norm_out(b, c, w, &y, &z);
    (out, SsmLayerState { h, conv: tail })
}

/// The decode state type for `batch` rows.
pub(crate) fn ssm_state_types(c: &SsmConfig, batch: usize) -> (Ty, Ty) {
    let h = match c.kind {
        SsmKind::Mamba2 | SsmKind::Plamo2 { .. } => {
            vec![batch, c.n_heads, c.head_dim, c.state_size]
        }
        SsmKind::Mamba1 { .. } => vec![batch, c.d_inner(), c.state_size],
    };
    (
        Ty::f32(h),
        Ty::f32(vec![batch, c.conv_kernel - 1, c.conv_dim]),
    )
}
