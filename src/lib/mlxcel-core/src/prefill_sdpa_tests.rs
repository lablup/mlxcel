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

//! The causal sink SDPA entry and the prefill route around it (issue #3098),
//! against the explicit-mask sink entry and an f64 host reference.
//!
//! Tolerances, fixed before the causal entry was run: max abs difference
//! against the host reference of `3e-5` in float32 and `2e-3` in float16
//! (about two ulp at the output magnitude); the two device routes agree to
//! the same bound.

use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::prefill_sdpa::{causal_prefill_attention, causal_window_arg};
use crate::{UniquePtr, utils};

const SCALE: f32 = 0.125;

fn tolerance(dtype_code: i32) -> f32 {
    match dtype_code {
        dtype::FLOAT32 => 3e-5,
        dtype::FLOAT16 => 2e-3,
        other => panic!("no tolerance stated for dtype {other}"),
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn random_array(rng: &mut Rng, shape: &[i32], dtype_code: i32) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|d| *d as usize).product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let f32_arr = ffi::from_slice_f32(&data, shape);
    if dtype_code == dtype::FLOAT32 {
        f32_arr
    } else {
        ffi::astype(&f32_arr, dtype_code)
    }
}

fn to_vec_f32(arr: &MlxArray) -> Vec<f32> {
    utils::array_to_vec_f32(arr)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[derive(Clone, Copy, Debug)]
struct Case {
    q_len: i32,
    k_len: i32,
    /// Non-positive: no window.
    window: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
}

/// Sink attention on the host in f64: row `r`, aligned to the suffix of the
/// keys, sees key `c` iff `c <= r + off` and (with a window) `c > r + off -
/// window`, and its softmax denominator also holds `exp(sink[h])`.
fn host_reference(q: &[f32], k: &[f32], v: &[f32], sinks: &[f32], case: Case) -> Vec<f32> {
    let (q_len, k_len, d) = (
        case.q_len as usize,
        case.k_len as usize,
        case.head_dim as usize,
    );
    let group = (case.q_heads / case.kv_heads) as usize;
    let off = k_len as i64 - q_len as i64;
    let mut out = vec![0f32; case.q_heads as usize * q_len * d];
    for h in 0..case.q_heads as usize {
        let kv_h = h / group;
        for r in 0..q_len {
            let q_row = &q[(h * q_len + r) * d..(h * q_len + r + 1) * d];
            let hi = (r as i64 + off) as usize;
            let lo = if case.window > 0 {
                (r as i64 + off - case.window as i64 + 1).max(0) as usize
            } else {
                0
            };
            let scores: Vec<f64> = (lo..=hi)
                .map(|c| {
                    let k_row = &k[(kv_h * k_len + c) * d..(kv_h * k_len + c + 1) * d];
                    let dot: f64 = q_row
                        .iter()
                        .zip(k_row)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum();
                    dot * SCALE as f64
                })
                .collect();
            let sink = sinks[h] as f64;
            let max = scores.iter().cloned().fold(sink, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f64 = weights.iter().sum::<f64>() + (sink - max).exp();
            let out_row = &mut out[(h * q_len + r) * d..(h * q_len + r + 1) * d];
            for (w, c) in weights.iter().zip(lo..=hi) {
                let v_row = &v[(kv_h * k_len + c) * d..(kv_h * k_len + c + 1) * d];
                for (o, x) in out_row.iter_mut().zip(v_row) {
                    *o += (w / denom * *x as f64) as f32;
                }
            }
        }
    }
    out
}

struct Inputs {
    q: UniquePtr<MlxArray>,
    k: UniquePtr<MlxArray>,
    v: UniquePtr<MlxArray>,
    sinks: UniquePtr<MlxArray>,
}

fn inputs(case: Case, dtype_code: i32, seed: u64) -> Inputs {
    let mut rng = Rng::new(seed);
    let q = random_array(
        &mut rng,
        &[1, case.q_heads, case.q_len, case.head_dim],
        dtype_code,
    );
    let k = random_array(
        &mut rng,
        &[1, case.kv_heads, case.k_len, case.head_dim],
        dtype_code,
    );
    let v = random_array(
        &mut rng,
        &[1, case.kv_heads, case.k_len, case.head_dim],
        dtype_code,
    );
    // Sinks are stored in f32 (gpt-oss keeps them in the checkpoint dtype);
    // scaled so a sink competes with the scores rather than vanishing.
    let sinks = crate::multiply_scalar(
        &random_array(&mut rng, &[case.q_heads], dtype::FLOAT32),
        2.0,
    );
    Inputs { q, k, v, sinks }
}

/// The pre-#3098 call: the sink entry with the explicit band.
fn explicit_mask_call(inputs: &Inputs, case: Case) -> UniquePtr<MlxArray> {
    let mask = utils::create_causal_mask_with_window_full(
        case.q_len,
        case.k_len - case.q_len,
        (case.window > 0).then_some(case.window),
    );
    // SAFETY: `mask` and `sinks` are live arrays borrowed for the call.
    unsafe {
        ffi::fast_scaled_dot_product_attention_with_sinks(
            &inputs.q,
            &inputs.k,
            &inputs.v,
            SCALE,
            mask.as_ref().unwrap() as *const MlxArray,
            inputs.sinks.as_ref().unwrap() as *const MlxArray,
        )
    }
}

fn causal_entry_call(inputs: &Inputs, case: Case) -> UniquePtr<MlxArray> {
    ffi::ffi_fast_scaled_dot_product_attention_with_sinks_causal(
        &inputs.q,
        &inputs.k,
        &inputs.v,
        SCALE,
        causal_window_arg(case.window, case.k_len),
        &inputs.sinks,
    )
    .expect("causal sink SDPA")
}

/// Causal, causal over cached prior keys, in-window and over-window cases at
/// the gpt-oss (64) and Laguna (128) head dims, with GQA.
fn cases() -> Vec<Case> {
    let base = Case {
        q_len: 32,
        k_len: 32,
        window: 0,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 64,
    };
    vec![
        base,
        Case { k_len: 80, ..base },
        Case { window: 48, ..base },
        Case { window: 12, ..base },
        Case {
            k_len: 80,
            window: 20,
            ..base
        },
        Case {
            head_dim: 128,
            k_len: 70,
            window: 16,
            ..base
        },
        Case {
            q_len: 96,
            k_len: 96,
            head_dim: 128,
            ..base
        },
    ]
}

fn check_routes(dtype_code: i32) {
    let tol = tolerance(dtype_code);
    for (i, case) in cases().into_iter().enumerate() {
        let inputs = inputs(case, dtype_code, 0x3098 + i as u64);
        let reference = host_reference(
            &to_vec_f32(&inputs.q),
            &to_vec_f32(&inputs.k),
            &to_vec_f32(&inputs.v),
            &to_vec_f32(&inputs.sinks),
            case,
        );
        let explicit = to_vec_f32(&explicit_mask_call(&inputs, case));
        let causal = to_vec_f32(&causal_entry_call(&inputs, case));
        let routed = to_vec_f32(&causal_prefill_attention(
            &inputs.q,
            &inputs.k,
            &inputs.v,
            SCALE,
            case.window,
            Some(&inputs.sinks),
        ));
        let explicit_err = max_abs_diff(&explicit, &reference);
        let causal_err = max_abs_diff(&causal, &reference);
        let arms = max_abs_diff(&causal, &explicit);
        let routed_err = max_abs_diff(&routed, &reference);
        assert!(
            explicit_err <= tol && causal_err <= tol && arms <= tol && routed_err <= tol,
            "{case:?} dtype {dtype_code}: explicit {explicit_err}, causal {causal_err}, \
             causal-vs-explicit {arms}, routed {routed_err}, tolerance {tol}"
        );
    }
}

#[test]
fn causal_sink_entry_matches_the_explicit_mask_in_float32() {
    check_routes(dtype::FLOAT32);
}

#[test]
fn causal_sink_entry_matches_the_explicit_mask_in_float16() {
    check_routes(dtype::FLOAT16);
}

#[test]
fn causal_prefill_attention_without_sinks_matches_the_explicit_mask() {
    for (i, case) in cases().into_iter().enumerate() {
        let inputs = inputs(case, dtype::FLOAT32, 0x3098_0000 + i as u64);
        let mask = utils::create_causal_mask_with_window_full(
            case.q_len,
            case.k_len - case.q_len,
            (case.window > 0).then_some(case.window),
        );
        let explicit = to_vec_f32(&crate::layers::attention(
            &inputs.q,
            &inputs.k,
            &inputs.v,
            SCALE,
            mask.as_ref(),
            0.0,
            case.window,
        ));
        let routed = to_vec_f32(&causal_prefill_attention(
            &inputs.q,
            &inputs.k,
            &inputs.v,
            SCALE,
            case.window,
            None,
        ));
        let diff = max_abs_diff(&routed, &explicit);
        assert!(diff <= tolerance(dtype::FLOAT32), "{case:?}: {diff}");
    }
}

#[test]
fn causal_sink_entry_rejects_a_bad_window_without_aborting() {
    let case = Case {
        q_len: 16,
        k_len: 16,
        window: 0,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 64,
    };
    let inputs = inputs(case, dtype::FLOAT32, 7);
    for window in [0, -2] {
        let result = ffi::ffi_fast_scaled_dot_product_attention_with_sinks_causal(
            &inputs.q,
            &inputs.k,
            &inputs.v,
            SCALE,
            window,
            &inputs.sinks,
        );
        assert!(result.is_err(), "window {window} must be an Err");
    }
    // Sinks of the wrong length are an MLX validation error, not an abort.
    let wrong = ffi::from_slice_f32(&[0.0; 3], &[3]);
    let result = ffi::ffi_fast_scaled_dot_product_attention_with_sinks_causal(
        &inputs.q, &inputs.k, &inputs.v, SCALE, -1, &wrong,
    );
    assert!(result.is_err(), "mis-shaped sinks must be an Err");
}

#[test]
fn causal_window_arg_passes_the_window_only_once_keys_outgrow_it() {
    assert_eq!(causal_window_arg(0, 100), -1);
    assert_eq!(causal_window_arg(-1, 100), -1);
    assert_eq!(causal_window_arg(128, 128), -1);
    assert_eq!(causal_window_arg(128, 64), -1);
    assert_eq!(causal_window_arg(128, 129), 128);
}
