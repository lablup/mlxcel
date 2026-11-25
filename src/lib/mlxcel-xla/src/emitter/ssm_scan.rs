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

//! Prefill forms of the selective scan (issue #3337).
//!
//! The builder has no `while` op, so a whole-prompt scan must be expressed with
//! ops that already lower. Three forms were measured by the lowering spike
//! (`spike/openxla/FINDINGS_ssm.md`):
//!
//! * [`SsmPrefillForm::ChunkedSsd`], the chosen one: Mamba2's state-space
//!   duality. Within a chunk of `Q` rows the scan is a masked
//!   `dot_general` against a segment-sum decay matrix built with [`prefix_add`];
//!   across chunks the carried states are one `dot_general` against the
//!   chunk-level decay matrix (HF's form), so nothing is unrolled.
//! * [`SsmPrefillForm::Associative`]: a log-depth Hillis-Steele scan over
//!   `(decay, input)` pairs. It materializes `[T, H, P, N]` per level.
//! * [`SsmPrefillForm::Sequential`]: `T` unrolled decode steps, the baseline.
//!
//! Every form takes `dt = 0` on padded rows, so the state passes through them.
//!
//! No form may emit a prefix-sum `stablehlo.reduce_window`. IREE rewrites that
//! pattern to `iree_linalg_ext.scan`, and the IREE-Metal target returns
//! unrelated values for that op on any tensor of rank 2 or more (issue #3343;
//! a `[3, 3]` input already reproduces it). [`prefix_add`] is the replacement.

use super::builder::{Builder, Val};
use super::config::SsmConfig;
use super::ssm::{heads_from_groups, selective_scan_step, splat, to_f32};

/// A prefill lowering of the selective scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum SsmPrefillForm {
    ChunkedSsd,
    Associative,
    Sequential,
}

impl SsmPrefillForm {
    /// Parse the spike's `--form` spelling.
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        match text {
            "chunked" => Ok(Self::ChunkedSsd),
            "associative" => Ok(Self::Associative),
            "sequential" => Ok(Self::Sequential),
            other => Err(format!(
                "unknown SSM prefill form {other:?}; use chunked, associative or sequential"
            )),
        }
    }
}

/// Dispatch the prefill scan to `form`. Shapes: `x [T,H,P]`, `dt [T,H]`,
/// `a [H]`, `bm`/`cm [T,G,N]`, `d [H]`, `h0 [H,P,N]` -> `(y [T,H,P], h_T)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn selective_scan_prefill_form(
    b: &mut Builder,
    c: &SsmConfig,
    form: SsmPrefillForm,
    x: &Val,
    dt: &Val,
    a: &Val,
    bm: &Val,
    cm: &Val,
    d: &Val,
    h0: &Val,
) -> (Val, Val) {
    let [x, dt, a, bm, cm, d, h0] = [x, dt, a, bm, cm, d, h0].map(|v| to_f32(b, v));
    let (y, h_t) = match form {
        SsmPrefillForm::ChunkedSsd => chunked_ssd(b, c, &x, &dt, &a, &bm, &cm, &h0),
        SsmPrefillForm::Associative => associative(b, c, &x, &dt, &a, &bm, &cm, &h0),
        SsmPrefillForm::Sequential => return sequential(b, c, &x, &dt, &a, &bm, &cm, &d, &h0),
    };
    let t = x.ty.shape[0];
    let d_b = b.broadcast(&d, &[1], vec![t, c.n_heads, c.head_dim]);
    let skip = b.multiply(&x, &d_b);
    (b.add(&y, &skip), h_t)
}

/// Shift `v` by `by` positions toward higher indices along `axis`, filling the
/// vacated positions with `fill`.
fn shift_along(b: &mut Builder, v: &Val, axis: usize, by: usize, fill: f32) -> Val {
    let rank = v.ty.shape.len();
    let mut keep: Vec<(usize, usize)> = v.ty.shape.iter().map(|&s| (0, s)).collect();
    keep[axis].1 -= by;
    let kept = b.slice(v, &keep);
    let fill = b.const_f32(fill);
    let mut low = vec![0; rank];
    low[axis] = by;
    b.pad(&kept, &fill, &low, &vec![0; rank])
}

/// Inclusive f32 prefix sum of `v` along `axis`, as a log-depth chain of
/// shifted adds (`ceil(log2(len))` levels of `slice`, `pad` and `add`).
///
/// This replaces the left-padded `stablehlo.reduce_window` cumsum, which
/// IREE-Metal miscompiles at rank 2 and above (issue #3343, see the module
/// docs). The pairwise association also carries less rounding error than the
/// sequential fold that `reduce_window` lowered to.
// Used by: Mamba2, Nemotron-H, Falcon-H1 (chunked SSD prefill: `exp_segsum`,
// `chunked_ssd`).
pub(super) fn prefix_add(b: &mut Builder, v: &Val, axis: usize) -> Val {
    assert_eq!(v.ty.elt, "f32", "the scan accumulates in f32");
    let len = v.ty.shape[axis];
    let mut sums = v.clone();
    let mut step = 1;
    while step < len {
        let shifted = shift_along(b, &sums, axis, step, 0.0);
        sums = b.add(&sums, &shifted);
        step *= 2;
    }
    sums
}

/// `exp(segsum)` for per-step log decays `v [.., L]` along the last axis:
/// `out[.., i, j] = exp(sum_{k=j+1..=i} v[k])` for `i >= j`, else 0. Built as
/// HF's `segment_sum` does (a masked prefix sum, not a difference of prefix
/// sums), so the near-diagonal entries carry no cancellation error.
fn exp_segsum(b: &mut Builder, v: &Val) -> Val {
    let rank = v.ty.shape.len();
    let l = v.ty.shape[rank - 1];
    let mut square = v.ty.shape.clone();
    square.push(l);
    // rows[.., i, j] = v[.., i]
    let lead: Vec<usize> = (0..rank).collect();
    let rows = b.broadcast(v, &lead, square.clone());
    let i = b.iota(l);
    let i = b.broadcast(&i, &[rank - 1], square.clone());
    let j = b.iota(l);
    let j = b.broadcast(&j, &[rank], square.clone());
    let below = b.compare("GT", &i, &j, "SIGNED");
    let on_or_below = b.compare("GE", &i, &j, "SIGNED");
    let zeros = splat(b, 0.0, &square);
    let masked = b.select(&below, &rows, &zeros);
    let sums = prefix_add(b, &masked, rank - 1);
    let neg_inf = splat(b, f32::NEG_INFINITY, &square);
    let sums = b.select(&on_or_below, &sums, &neg_inf);
    b.exponential(&sums)
}

/// Pad `v` with zero rows on axis 0 up to `rows`.
fn pad_rows(b: &mut Builder, v: &Val, rows: usize) -> Val {
    let extra = rows - v.ty.shape[0];
    if extra == 0 {
        return v.clone();
    }
    let zero = b.const_f32(0.0);
    let rank = v.ty.shape.len();
    let mut high = vec![0; rank];
    high[0] = extra;
    b.pad(v, &zero, &vec![0; rank], &high)
}

/// Chunked SSD without the `D` skip. See the module docs.
#[allow(clippy::too_many_arguments)]
fn chunked_ssd(
    b: &mut Builder,
    c: &SsmConfig,
    x: &Val,
    dt: &Val,
    a: &Val,
    bm: &Val,
    cm: &Val,
    h0: &Val,
) -> (Val, Val) {
    let (hh, p, n, g) = (c.n_heads, c.head_dim, c.state_size, c.n_groups);
    let t = x.ty.shape[0];
    let q = c.chunk_size.min(t);
    let nc = t.div_ceil(q);
    let tp = nc * q;
    let [x, dt, bm, cm] = [x, dt, bm, cm].map(|v| pad_rows(b, v, tp));

    // Discretize: x * dt, and the per-step log decay dt * A.
    let dt_x = b.broadcast(&dt, &[0, 1], vec![tp, hh, p]);
    let xdt = b.multiply(&x, &dt_x);
    let a_b = b.broadcast(a, &[1], vec![tp, hh]);
    let ad = b.multiply(&dt, &a_b);
    let ad = b.reshape(&ad, vec![nc, q, hh]);
    let acum = prefix_add(b, &ad, 1); // [nc, q, H]

    // 1. Intra-chunk (diagonal blocks): (C B^T  o  L) (x dt).
    let ad_t = b.transpose(&ad, &[0, 2, 1]); // [nc, H, q]
    let decay_l = exp_segsum(b, &ad_t); // [nc, H, q, q]
    let bm4 = b.reshape(&bm, vec![nc, q, g, n]);
    let cm4 = b.reshape(&cm, vec![nc, q, g, n]);
    let cb = b.dot_general_keep_f32(&cm4, &bm4, &[0, 2], &[0, 2], &[3], &[3], vec![nc, g, q, q]);
    let cb = if g == hh {
        cb
    } else {
        let wide = b.broadcast(&cb, &[0, 1, 3, 4], vec![nc, g, hh / g, q, q]);
        b.reshape(&wide, vec![nc, hh, q, q])
    };
    let weights = b.multiply(&cb, &decay_l);
    let xdt4 = b.reshape(&xdt, vec![nc, q, hh, p]);
    let y_diag = b.dot_general_keep_f32(
        &weights,
        &xdt4,
        &[0, 1],
        &[0, 2],
        &[3],
        &[1],
        vec![nc, hh, q, p],
    );

    // 2. Each chunk's own end state from a zero start.
    let last = b.slice(&acum, &[(0, nc), (q - 1, q), (0, hh)]);
    let last = b.reshape(&last, vec![nc, hh]);
    let last_b = b.broadcast(&last, &[0, 2], vec![nc, q, hh]);
    let to_end = b.subtract(&last_b, &acum);
    let to_end = b.exponential(&to_end);
    let to_end = b.broadcast(&to_end, &[0, 1, 2], vec![nc, q, hh, p]);
    let x_end = b.multiply(&xdt4, &to_end);
    let bh4 = heads_from_groups(b, &bm4, hh); // [nc, q, H, N]
    let states = b.dot_general_keep_f32(
        &x_end,
        &bh4,
        &[0, 2],
        &[0, 2],
        &[1],
        &[1],
        vec![nc, hh, p, n],
    );

    // 3. Inter-chunk carry: state entering chunk z is sum_c decay[z, c] * s[c],
    // with s[0] = h0 and decay the chunk-level exp(segsum).
    let h0_row = b.reshape(h0, vec![1, hh, p, n]);
    let states = b.concatenate(&h0_row, &states, 0); // [nc+1, H, P, N]
    let states = b.reshape(&states, vec![nc + 1, hh, p * n]);
    let zero = b.const_f32(0.0);
    let totals = b.pad(&last, &zero, &[1, 0], &[0, 0]); // [nc+1, H]
    let totals = b.transpose(&totals, &[1, 0]); // [H, nc+1]
    let decay_c = exp_segsum(b, &totals); // [H, nc+1, nc+1]
    let carried = b.dot_general_keep_f32(
        &decay_c,
        &states,
        &[0],
        &[1],
        &[2],
        &[0],
        vec![hh, nc + 1, p * n],
    );
    let entering = b.slice(&carried, &[(0, hh), (0, nc), (0, p * n)]);
    let entering = b.reshape(&entering, vec![hh, nc, p, n]);
    let entering = b.transpose(&entering, &[1, 0, 2, 3]); // [nc, H, P, N]
    let h_t = b.slice(&carried, &[(0, hh), (nc, nc + 1), (0, p * n)]);
    let h_t = b.reshape(&h_t, vec![hh, p, n]);

    // 4. Off-diagonal output: C . (state entering the chunk), decayed to row t.
    let ch4 = heads_from_groups(b, &cm4, hh); // [nc, q, H, N]
    let y_off = b.dot_general_keep_f32(
        &ch4,
        &entering,
        &[0, 2],
        &[0, 1],
        &[3],
        &[3],
        vec![nc, hh, q, p],
    );
    let acum_t = b.transpose(&acum, &[0, 2, 1]); // [nc, H, q]
    let from_start = b.exponential(&acum_t);
    let from_start = b.broadcast(&from_start, &[0, 1, 2], vec![nc, hh, q, p]);
    let y_off = b.multiply(&y_off, &from_start);

    let y = b.add(&y_diag, &y_off);
    let y = b.transpose(&y, &[0, 2, 1, 3]); // [nc, q, H, P]
    let y = b.reshape(&y, vec![tp, hh, p]);
    let y = if tp == t {
        y
    } else {
        b.slice(&y, &[(0, t), (0, hh), (0, p)])
    };
    (y, h_t)
}

/// Shift `v` down by `by` rows along axis 0, filling with `fill`.
fn shift_rows(b: &mut Builder, v: &Val, by: usize, fill: f32) -> Val {
    shift_along(b, v, 0, by, fill)
}

/// Log-depth inclusive scan of `h_t = a_t h_{t-1} + u_t`, without `D`.
#[allow(clippy::too_many_arguments)]
fn associative(
    b: &mut Builder,
    c: &SsmConfig,
    x: &Val,
    dt: &Val,
    a: &Val,
    bm: &Val,
    cm: &Val,
    h0: &Val,
) -> (Val, Val) {
    let (hh, p, n) = (c.n_heads, c.head_dim, c.state_size);
    let t = x.ty.shape[0];
    let full = vec![t, hh, p, n];
    let a_b = b.broadcast(a, &[1], vec![t, hh]);
    let ad = b.multiply(dt, &a_b);
    let mut decay = b.exponential(&ad); // [T, H]
    let dt_x = b.broadcast(dt, &[0, 1], vec![t, hh, p]);
    let xdt = b.multiply(x, &dt_x);
    let xdt = b.broadcast(&xdt, &[0, 1, 2], full.clone());
    let bh = heads_from_groups(b, bm, hh);
    let bh = b.broadcast(&bh, &[0, 1, 3], full.clone());
    let mut acc = b.multiply(&xdt, &bh); // [T, H, P, N]
    let mut step = 1;
    while step < t {
        let prev_decay = shift_rows(b, &decay, step, 1.0);
        let prev_acc = shift_rows(b, &acc, step, 0.0);
        let decay_b = b.broadcast(&decay, &[0, 1], full.clone());
        let carried = b.multiply(&decay_b, &prev_acc);
        acc = b.add(&acc, &carried);
        decay = b.multiply(&decay, &prev_decay);
        step *= 2;
    }
    let decay_b = b.broadcast(&decay, &[0, 1], full.clone());
    let h0_b = b.broadcast(h0, &[1, 2, 3], full);
    let from_h0 = b.multiply(&decay_b, &h0_b);
    let states = b.add(&from_h0, &acc);
    let ch = heads_from_groups(b, cm, hh);
    let y = b.dot_general_keep_f32(&states, &ch, &[0, 1], &[0, 1], &[3], &[2], vec![t, hh, p]);
    let h_t = b.slice(&states, &[(t - 1, t), (0, hh), (0, p), (0, n)]);
    let h_t = b.reshape(&h_t, vec![hh, p, n]);
    (y, h_t)
}

/// `T` unrolled decode steps (includes `D`).
#[allow(clippy::too_many_arguments)]
fn sequential(
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
    let (hh, p, n, g) = (c.n_heads, c.head_dim, c.state_size, c.n_groups);
    let t = x.ty.shape[0];
    let mut h = b.reshape(h0, vec![1, hh, p, n]);
    let mut y = splat(b, 0.0, &[t, hh, p]);
    let zero = b.const_i32(0);
    for row in 0..t {
        let x_t = b.slice(x, &[(row, row + 1), (0, hh), (0, p)]);
        let dt_t = b.slice(dt, &[(row, row + 1), (0, hh)]);
        let b_t = b.slice(bm, &[(row, row + 1), (0, g), (0, n)]);
        let c_t = b.slice(cm, &[(row, row + 1), (0, g), (0, n)]);
        let (y_t, next) = selective_scan_step(b, c, &x_t, &dt_t, a, &b_t, &c_t, d, &h);
        let at = b.const_i32(i32::try_from(row).expect("prefill rows fit i32"));
        y = b.dynamic_update_slice(&y, &y_t, &[&at, &zero, &zero]);
        h = next;
    }
    let h_t = b.reshape(&h, vec![hh, p, n]);
    (y, h_t)
}
