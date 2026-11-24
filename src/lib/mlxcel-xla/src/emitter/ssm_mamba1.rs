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

//! Mamba1 diagonal selective scan and mixer (issue #3348).
//!
//! Prefill builds at most `chunk * d_inner * state_size` recurrence operands.
//! It never forms a context-capacity-sized `[T, d_inner, state_size]` value.

use super::builder::{Builder, Val};
use super::config::{SsmConfig, SsmKind};
use super::ssm::{
    SsmLayerState, add_last, conv1d_causal_step, conv1d_over_left_context, silu, softplus, splat,
    tail_of_left_context, to_f32,
};

/// One Mamba1/Falcon-Mamba mixer layer's weights in HF checkpoint order.
#[derive(Clone, Debug)]
pub(crate) struct Mamba1MixerW {
    pub in_proj: Val,
    pub in_proj_b: Option<Val>,
    /// `[D, 1, K]`, the torch depthwise Conv1d layout.
    pub conv_w: Val,
    pub conv_b: Option<Val>,
    /// `[dt_rank + 2*N, D]`.
    pub x_proj: Val,
    /// `[D, dt_rank]` plus the always-present `[D]` bias.
    pub dt_proj: Val,
    pub dt_proj_b: Val,
    /// `[D, N]`.
    pub a_log: Val,
    /// `[D]`.
    pub d: Val,
    pub out_proj: Val,
    pub out_proj_b: Option<Val>,
}

/// One diagonal recurrence step, entirely in f32:
/// `h' = exp(log_a) * h + bx`.
pub(crate) fn diag_scan_step(b: &mut Builder, log_a: &Val, bx: &Val, h: &Val) -> Val {
    debug_assert_eq!(log_a.ty.shape, bx.ty.shape);
    debug_assert_eq!(log_a.ty.shape, h.ty.shape);
    let log_a = to_f32(b, log_a);
    let bx = to_f32(b, bx);
    let h = to_f32(b, h);
    let decay = b.exponential(&log_a);
    let kept = b.multiply(&decay, &h);
    b.add(&kept, &bx)
}

/// Shift `v` down by `by` rows, filling the new prefix with `fill`.
fn shift_rows(b: &mut Builder, v: &Val, by: usize, fill: f32) -> Val {
    let rank = v.ty.shape.len();
    let rows = v.ty.shape[0];
    debug_assert!(by < rows);
    let mut keep: Vec<(usize, usize)> = v.ty.shape.iter().map(|&size| (0, size)).collect();
    keep[0] = (0, rows - by);
    let kept = b.slice(v, &keep);
    let fill = b.const_f32(fill);
    let mut low = vec![0; rank];
    low[0] = by;
    b.pad(&kept, &fill, &low, &vec![0; rank])
}

/// Log-depth inclusive prefix scan of the affine recurrence pairs in one
/// bounded block. For `(a_t, u_t)`, composition is
/// `(a_r, u_r) o (a_l, u_l) = (a_r*a_l, u_r + a_r*u_l)`.
fn diag_scan_block(
    b: &mut Builder,
    log_a: &Val,
    bx: &Val,
    readout: Option<&Val>,
    h0: &Val,
) -> (Val, Val) {
    let [rows, d, n] = log_a.ty.shape.as_slice() else {
        panic!("diagonal scan block must be [T,D,N]");
    };
    let (rows, d, n) = (*rows, *d, *n);
    let full = vec![rows, d, n];
    let log_a = to_f32(b, log_a);
    let mut decay = b.exponential(&log_a);
    let mut acc = to_f32(b, bx);
    let mut step = 1;
    while step < rows {
        let previous_decay = shift_rows(b, &decay, step, 1.0);
        let previous_acc = shift_rows(b, &acc, step, 0.0);
        let carried = b.multiply(&decay, &previous_acc);
        acc = b.add(&acc, &carried);
        decay = b.multiply(&decay, &previous_decay);
        step *= 2;
    }
    let h0 = to_f32(b, h0);
    let h0 = b.broadcast(&h0, &[1, 2], full);
    let from_h0 = b.multiply(&decay, &h0);
    let states = b.add(&from_h0, &acc);
    let y = match readout {
        Some(readout) => {
            let readout = to_f32(b, readout);
            let readout = b.reshape(&readout, vec![rows, 1, n]);
            let readout = b.broadcast(&readout, &[0, 1, 2], vec![rows, d, n]);
            let product = b.multiply(&states, &readout);
            let zero = b.const_f32(0.0);
            b.reduce_add(&product, 2, &zero)
        }
        None => b.reshape(&states, vec![rows, d]),
    };
    let last = b.slice(&states, &[(rows - 1, rows), (0, d), (0, n)]);
    (y, b.reshape(&last, vec![d, n]))
}

/// Chunked diagonal scan over `log_a, bx [T,D,N]` from `h0 [D,N]`.
///
/// `readout [T,N]` produces `y[t,d] = sum_n readout[t,n] * h[t,d,n]`.
/// With no readout, `N` must be one and the sole state column is returned.
pub(crate) fn diag_scan_prefill(
    b: &mut Builder,
    log_a: &Val,
    bx: &Val,
    readout: Option<&Val>,
    h0: &Val,
    chunk: usize,
) -> (Val, Val) {
    assert!(chunk > 0, "diagonal scan chunk must be positive");
    let [t, d, n] = log_a.ty.shape.as_slice() else {
        panic!("diagonal scan log_a must be [T,D,N]");
    };
    let (t, d, n) = (*t, *d, *n);
    assert_eq!(bx.ty.shape, vec![t, d, n]);
    assert_eq!(h0.ty.shape, vec![d, n]);
    if let Some(c) = readout {
        assert_eq!(c.ty.shape, vec![t, n]);
    } else {
        assert_eq!(n, 1, "a diagonal scan without readout requires N == 1");
    }

    let mut h = to_f32(b, h0);
    let mut all_y: Option<Val> = None;
    for begin in (0..t).step_by(chunk) {
        let end = (begin + chunk).min(t);
        let log_block = b.slice(log_a, &[(begin, end), (0, d), (0, n)]);
        let bx_block = b.slice(bx, &[(begin, end), (0, d), (0, n)]);
        let read_block = readout.map(|c| b.slice(c, &[(begin, end), (0, n)]));
        let (block_y, next) = diag_scan_block(b, &log_block, &bx_block, read_block.as_ref(), &h);
        h = next;
        all_y = Some(match all_y {
            Some(previous) => b.concatenate(&previous, &block_y, 0),
            None => block_y,
        });
    }
    (all_y.expect("the mixer requires a non-empty sequence"), h)
}

fn kind(c: &SsmConfig) -> (usize, bool) {
    match c.kind {
        SsmKind::Mamba1 { dt_rank, bcdt_norm } => (dt_rank, bcdt_norm),
        SsmKind::Mamba2 => panic!("Mamba1 mixer requires SsmKind::Mamba1"),
    }
}

/// Weightless RMSNorm over the last axis, computed in f32.
fn rms_no_weight(b: &mut Builder, x: &Val, eps: f32) -> Val {
    let x = to_f32(b, x);
    let axis = x.ty.shape.len() - 1;
    let width = x.ty.shape[axis];
    let square = b.multiply(&x, &x);
    let zero = b.const_f32(0.0);
    let sum = b.reduce_add(&square, axis, &zero);
    let denominator = splat(b, width as f32, &sum.ty.shape);
    let mean = b.divide(&sum, &denominator);
    let epsilon = splat(b, eps, &mean.ty.shape);
    let stabilized = b.add(&mean, &epsilon);
    let inverse = b.rsqrt(&stabilized);
    let dims: Vec<usize> = (0..axis).collect();
    let inverse = b.broadcast(&inverse, &dims, x.ty.shape.clone());
    b.multiply(&x, &inverse)
}

/// Project `x [R,H]` to the pre-conv branch and SiLU gate, both `[R,D]`.
fn in_projection(b: &mut Builder, c: &SsmConfig, w: &Mamba1MixerW, x: &Val) -> (Val, Val) {
    let rows = x.ty.shape[0];
    let d = c.d_inner();
    let x = to_f32(b, x);
    let projected = b.linear_seq_proj(&x, &w.in_proj);
    let projected = match &w.in_proj_b {
        Some(bias) => add_last(b, &projected, bias),
        None => projected,
    };
    let xs = b.slice(&projected, &[(0, rows), (0, d)]);
    let z = b.slice(&projected, &[(0, rows), (d, 2 * d)]);
    (xs, z)
}

/// Project activated conv rows into discretized dt and the B/C read vectors.
fn scan_parameters(b: &mut Builder, c: &SsmConfig, w: &Mamba1MixerW, x: &Val) -> (Val, Val, Val) {
    let rows = x.ty.shape[0];
    let (rank, normalize) = kind(c);
    let n = c.state_size;
    let x = to_f32(b, x);
    let projected = b.linear_seq_proj(&x, &w.x_proj);
    let mut dt = b.slice(&projected, &[(0, rows), (0, rank)]);
    let mut bm = b.slice(&projected, &[(0, rows), (rank, rank + n)]);
    let mut cm = b.slice(&projected, &[(0, rows), (rank + n, rank + 2 * n)]);
    if normalize {
        dt = rms_no_weight(b, &dt, c.norm_eps);
        bm = rms_no_weight(b, &bm, c.norm_eps);
        cm = rms_no_weight(b, &cm, c.norm_eps);
    }
    let dt = b.linear_seq_proj(&dt, &w.dt_proj);
    let dt = add_last(b, &dt, &w.dt_proj_b);
    (softplus(b, &dt), bm, cm)
}

fn decay_rate(b: &mut Builder, a_log: &Val) -> Val {
    let a_log = to_f32(b, a_log);
    let exponential = b.exponential(&a_log);
    b.negate(&exponential)
}

fn gated_output(b: &mut Builder, w: &Mamba1MixerW, y: &Val, z: &Val) -> Val {
    let gate = silu(b, z);
    let gated = b.multiply(y, &gate);
    let out = b.linear_seq_proj(&gated, &w.out_proj);
    match &w.out_proj_b {
        Some(bias) => add_last(b, &out, bias),
        None => out,
    }
}

/// One Mamba1 mixer layer over a padded prompt. The scan is lowered one bounded
/// chunk at a time, so a capacity of 4096 never creates `[4096,D,N]` operands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mamba1_mixer_prefill(
    b: &mut Builder,
    c: &SsmConfig,
    w: &Mamba1MixerW,
    x: &Val,
    real_len: &Val,
    h0: &Val,
    conv0: &Val,
) -> (Val, Val, Val) {
    let t = x.ty.shape[0];
    let (d, n, k) = (c.d_inner(), c.state_size, c.conv_kernel);
    let (conv_input, z) = in_projection(b, c, w, x);
    let conv0 = to_f32(b, conv0);
    let xp = b.concatenate(&conv0, &conv_input, 0);
    let conv = conv1d_over_left_context(b, &xp, &w.conv_w, w.conv_b.as_ref(), k);
    let tail = tail_of_left_context(b, &xp, real_len, k);
    let xs = silu(b, &conv);
    let (dt, bm, cm) = scan_parameters(b, c, w, &xs);
    let rows = b.iota(t);
    let rows = b.broadcast(&rows, &[0], vec![t, d]);
    let limit = b.broadcast(real_len, &[], vec![t, d]);
    let live = b.compare("LT", &rows, &limit, "SIGNED");
    let zeros = splat(b, 0.0, &[t, d]);
    let dt = b.select(&live, &dt, &zeros);
    let a = decay_rate(b, &w.a_log);
    let d_skip = to_f32(b, &w.d);

    let mut h = to_f32(b, h0);
    let mut output: Option<Val> = None;
    for begin in (0..t).step_by(c.chunk_size) {
        let end = (begin + c.chunk_size).min(t);
        let rows = end - begin;
        let x_block = b.slice(&xs, &[(begin, end), (0, d)]);
        let dt_block = b.slice(&dt, &[(begin, end), (0, d)]);
        let b_block = b.slice(&bm, &[(begin, end), (0, n)]);
        let c_block = b.slice(&cm, &[(begin, end), (0, n)]);

        let dt3 = b.reshape(&dt_block, vec![rows, d, 1]);
        let dt3 = b.broadcast(&dt3, &[0, 1, 2], vec![rows, d, n]);
        let a3 = b.broadcast(&a, &[1, 2], vec![rows, d, n]);
        let log_a = b.multiply(&dt3, &a3);
        let dt_x = b.multiply(&dt_block, &x_block);
        let dt_x = b.reshape(&dt_x, vec![rows, d, 1]);
        let dt_x = b.broadcast(&dt_x, &[0, 1, 2], vec![rows, d, n]);
        let b3 = b.reshape(&b_block, vec![rows, 1, n]);
        let b3 = b.broadcast(&b3, &[0, 1, 2], vec![rows, d, n]);
        let bx = b.multiply(&dt_x, &b3);
        let (y, next) = diag_scan_prefill(b, &log_a, &bx, Some(&c_block), &h, rows);
        h = next;
        let skip = b.broadcast(&d_skip, &[1], vec![rows, d]);
        let skip = b.multiply(&skip, &x_block);
        let y = b.add(&y, &skip);
        output = Some(match output {
            Some(previous) => b.concatenate(&previous, &y, 0),
            None => y,
        });
    }
    let y = output.expect("the mixer requires a non-empty sequence");
    (gated_output(b, w, &y, &z), h, tail)
}

/// One batched Mamba1 decode step.
pub(crate) fn mamba1_mixer_step(
    b: &mut Builder,
    c: &SsmConfig,
    w: &Mamba1MixerW,
    x: &Val,
    st: &SsmLayerState,
) -> (Val, SsmLayerState) {
    let batch = x.ty.shape[0];
    let (d, n) = (c.d_inner(), c.state_size);
    let (conv_input, z) = in_projection(b, c, w, x);
    let (conv, tail) = conv1d_causal_step(
        b,
        &conv_input,
        &st.conv,
        &w.conv_w,
        w.conv_b.as_ref(),
        c.conv_kernel,
    );
    let xs = silu(b, &conv);
    let (dt, bm, cm) = scan_parameters(b, c, w, &xs);
    let a = decay_rate(b, &w.a_log);
    let dt3 = b.reshape(&dt, vec![batch, d, 1]);
    let dt3 = b.broadcast(&dt3, &[0, 1, 2], vec![batch, d, n]);
    let a3 = b.broadcast(&a, &[1, 2], vec![batch, d, n]);
    let log_a = b.multiply(&dt3, &a3);
    let dt_x = b.multiply(&dt, &xs);
    let dt_x = b.reshape(&dt_x, vec![batch, d, 1]);
    let dt_x = b.broadcast(&dt_x, &[0, 1, 2], vec![batch, d, n]);
    let b3 = b.reshape(&bm, vec![batch, 1, n]);
    let b3 = b.broadcast(&b3, &[0, 1, 2], vec![batch, d, n]);
    let bx = b.multiply(&dt_x, &b3);
    let h = diag_scan_step(b, &log_a, &bx, &st.h);
    let c3 = b.reshape(&cm, vec![batch, 1, n]);
    let c3 = b.broadcast(&c3, &[0, 1, 2], vec![batch, d, n]);
    let read = b.multiply(&h, &c3);
    let zero = b.const_f32(0.0);
    let y = b.reduce_add(&read, 2, &zero);
    let d_skip = to_f32(b, &w.d);
    let d_skip = b.broadcast(&d_skip, &[1], vec![batch, d]);
    let skip = b.multiply(&xs, &d_skip);
    let y = b.add(&y, &skip);
    let out = gated_output(b, w, &y, &z);
    (out, SsmLayerState { h, conv: tail })
}
