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

//! Numeric-parity tests for the fused residual-add RMSNorm kernel (#905).
//!
//! The fused kernel replaces the `add` + `fast_rms_norm` pair at every pre-norm
//! residual join, so the property that has to hold is that a block cannot tell
//! which path it took. These tests pin that at three levels:
//!
//! 1. **Per-dtype numeric parity** against the graph path the blocks used
//!    before, at f32 / f16 / bf16 and across the hidden sizes the decode loop
//!    actually sees.
//! 2. **The Gemma `(1 + w)` convention** expressed as `weight_bias = 1.0`,
//!    checked against a materialised `(1 + w)` tensor rather than against a
//!    re-derivation of the formula, because the whole point of the scalar bias
//!    is that it is not an approximation of the precomputed tensor.
//! 3. **Greedy argmax stability** over a multi-layer, multi-step synthetic
//!    stack: the per-element deviation is far below the argmax-flip scale, and
//!    this is what says so end to end rather than element by element.
//!
//! Tolerances are per dtype and are stated as normalized RMS and normalized max
//! deviation (both divided by the reference's own RMS), so they read the same
//! way at any activation scale. They are sized at a small multiple of one ulp
//! of the dtype, because the fused and graph paths differ only in where the
//! residual sum gets rounded, not in the arithmetic.
//!
//! GPU-only: the kernel JITs through `mx.fast.metal_kernel` / `cuda_kernel`, so
//! these tests return early on a CPU-only build, matching the convention in
//! `fused_moe_parity_tests.rs`.
//!
//! Run on Apple Silicon:
//!   cargo test --release -p mlxcel-core --lib --features metal,accelerate \
//!     fused_norm_parity_tests
//! Kill-switch pass (the same file, graph path forced):
//!   MLXCEL_FUSED_ADD_RMSNORM=0 cargo test --release -p mlxcel-core --lib \
//!     --features metal,accelerate fused_norm_parity_tests

use super::*;
use crate::layers::{FusedAddRmsNormSpec, GemmaRMSNorm, RMSNorm};

/// Normalized (RMS, max) deviation budget per activation dtype.
///
/// One ulp relative to the RMS element is `2^-24` (f32), `2^-11` (f16) and
/// `2^-8` (bf16); the max budget allows a few ulp on the largest element of a
/// standard-normal sample, which sits around 4 sigma.
fn tolerance_for(dtype: i32) -> (f64, f64) {
    if dtype == dtype::FLOAT32 {
        (1e-6, 1e-5)
    } else if dtype == dtype::FLOAT16 {
        (2e-3, 1.2e-2)
    } else {
        (1.6e-2, 7e-2)
    }
}

fn gpu_available() -> bool {
    crate::metal_is_available() || crate::cuda_is_available()
}

fn flatten_f32(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn raw_bytes(arr: &MlxArray) -> Vec<u8> {
    eval(arr);
    array_to_raw_bytes(arr)
}

/// RMS and max of `a - b`, both divided by the RMS of `b`.
fn normalized_deviation(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch in deviation check");
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    let mut max_abs = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        diff_sq += d * d;
        ref_sq += (*y as f64) * (*y as f64);
        max_abs = max_abs.max(d.abs());
    }
    let ref_rms = (ref_sq / b.len() as f64).sqrt().max(1e-20);
    (
        (diff_sq / a.len() as f64).sqrt() / ref_rms,
        max_abs / ref_rms,
    )
}

fn assert_within(label: &str, got: &MlxArray, want: &MlxArray, dtype: i32) {
    let (nrms, nmax) = normalized_deviation(&flatten_f32(got), &flatten_f32(want));
    let (rms_tol, max_tol) = tolerance_for(dtype);
    assert!(
        nrms < rms_tol && nmax < max_tol,
        "{label}: normalized rms {nrms:.3e} (tol {rms_tol:.1e}), \
         normalized max {nmax:.3e} (tol {max_tol:.1e})"
    );
}

/// Random `[rows, dim]` activations plus a random `[dim]` norm weight, all in
/// `dtype`, pre-evaluated so both paths read the very same bytes.
fn random_case(
    seed: u64,
    rows: i32,
    dim: i32,
    dtype: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    random_seed(seed);
    let delta_f32 = unsafe { random_normal(&[rows, dim], dtype::FLOAT32, std::ptr::null()) };
    let residual_f32 = unsafe { random_normal(&[rows, dim], dtype::FLOAT32, std::ptr::null()) };
    // Norm weights in a real checkpoint sit near 1, not near 0; a
    // standard-normal weight would make the reference RMS meaningless.
    let w_noise = unsafe { random_normal(&[dim], dtype::FLOAT32, std::ptr::null()) };
    let ones = full_f32(&[dim], 1.0, dtype::FLOAT32);
    let w_f32 = add(
        &ones,
        &multiply(&w_noise, &full_f32(&[1], 0.1, dtype::FLOAT32)),
    );

    let delta = astype(&delta_f32, dtype);
    let residual = astype(&residual_f32, dtype);
    let weight = astype(&w_f32, dtype);
    eval(&delta);
    eval(&residual);
    eval(&weight);
    (delta, residual, weight)
}

/// Call the fused kernel directly, bypassing the env gate so the comparison is
/// always fused-vs-graph regardless of how the test binary was launched.
fn run_fused(
    delta: &MlxArray,
    residual: &MlxArray,
    weight: &MlxArray,
    eps: f32,
    weight_bias: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let mut normed = UniquePtr::null();
    let mut new_residual = UniquePtr::null();
    crate::fused_add_rms_norm(
        delta,
        residual,
        weight,
        eps,
        weight_bias,
        &mut normed,
        &mut new_residual,
    );
    eval(&normed);
    eval(&new_residual);
    (normed, new_residual)
}

/// `(1 + w)` materialised in the weight's own dtype, which is exactly what
/// `GemmaRMSNorm::new` precomputes.
fn one_plus(weight: &MlxArray) -> UniquePtr<MlxArray> {
    let dt = array_dtype(weight);
    let one = full_f32(&[1], 1.0, dt);
    let adjusted = add(&one, weight);
    eval(&adjusted);
    adjusted
}

const EPS: f32 = 1e-6;

/// Hidden sizes and row counts the decode and prefill paths actually produce:
/// one row is batch-1 decode, five rows is small-batch decode, 33 rows is a
/// prefill chunk that is not a multiple of the threadgroup sweep.
const SHAPES: &[(i32, i32)] = &[(1, 128), (1, 2048), (5, 4096), (33, 2048), (1, 3072)];

#[test]
fn fused_add_rms_norm_matches_graph_across_dtypes_and_shapes() {
    if !gpu_available() {
        return;
    }
    for &dt in &[dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        for (i, &(rows, dim)) in SHAPES.iter().enumerate() {
            let (delta, residual, weight) = random_case(1000 + i as u64, rows, dim, dt);
            let (normed, new_residual) = run_fused(&delta, &residual, &weight, EPS, 0.0);

            let norm = RMSNorm::new(copy(&weight), EPS);
            let (want_normed, want_residual) =
                crate::layers::graph_add_rms_norm(&norm, &delta, &residual);
            eval(&want_normed);
            eval(&want_residual);

            assert_within(
                &format!("normed dt={dt} rows={rows} dim={dim}"),
                &normed,
                &want_normed,
                dt,
            );
            assert_within(
                &format!("new_residual dt={dt} rows={rows} dim={dim}"),
                &new_residual,
                &want_residual,
                dt,
            );
        }
    }
}

/// The residual output is the plain sum, so it must land on the same value the
/// graph `add` produces, not merely close to it: it is the value the *next*
/// join accumulates onto, and a systematic drift there would compound layer
/// over layer.
#[test]
fn fused_add_rms_norm_residual_output_is_the_plain_sum() {
    if !gpu_available() {
        return;
    }
    for &dt in &[dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        let (delta, residual, weight) = random_case(2024, 4, 2048, dt);
        let (_, new_residual) = run_fused(&delta, &residual, &weight, EPS, 0.0);
        let want = add(&residual, &delta);
        eval(&want);
        let got_v = flatten_f32(&new_residual);
        let want_v = flatten_f32(&want);
        let ulps_off = got_v
            .iter()
            .zip(want_v.iter())
            .filter(|(a, b)| a != b)
            .count();
        // f32 is exact (no rounding step differs); f16/bf16 may double-round
        // the sum at most, which can only move a handful of elements by 1 ulp.
        let budget = if dt == dtype::FLOAT32 {
            0
        } else {
            got_v.len() / 100
        };
        assert!(
            ulps_off <= budget,
            "dt={dt}: {ulps_off} of {} residual elements differ from the graph add (budget {budget})",
            got_v.len()
        );
    }
}

/// `weight_bias = 1.0` must reproduce Gemma's precomputed `(1 + w)` tensor,
/// not approximate it: the offset is folded in the weight's own dtype inside
/// the kernel, which is the same rounding `GemmaRMSNorm::new` performs.
#[test]
fn fused_add_rms_norm_gemma_weight_bias_matches_precomputed_one_plus_w() {
    if !gpu_available() {
        return;
    }
    for &dt in &[dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        for (i, &(rows, dim)) in SHAPES.iter().enumerate() {
            let (delta, residual, weight) = random_case(3000 + i as u64, rows, dim, dt);
            let (normed, _) = run_fused(&delta, &residual, &weight, EPS, 1.0);

            // Reference: the plain graph path fed the materialised (1 + w).
            let adjusted = one_plus(&weight);
            let sum = add(&residual, &delta);
            let want = fast_rms_norm(&sum, &adjusted, EPS);
            eval(&want);

            assert_within(
                &format!("gemma normed dt={dt} rows={rows} dim={dim}"),
                &normed,
                &want,
                dt,
            );
        }
    }
}

/// The `GemmaRMSNorm` layer type and the raw `weight_bias = 1.0` call must
/// agree, so a caller cannot get a different answer depending on whether it
/// went through the layer or the kernel.
#[test]
fn gemma_rms_norm_layer_agrees_with_weight_bias_one() {
    if !gpu_available() {
        return;
    }
    let dt = dtype::FLOAT16;
    let (delta, residual, weight) = random_case(4242, 4, 2048, dt);
    let gemma = GemmaRMSNorm::new(copy(&weight), EPS);
    assert_eq!(gemma.norm_weight_bias(), 1.0);

    let (fused_normed, _) = run_fused(&delta, &residual, gemma.raw_norm_weight(), EPS, 1.0);
    let (graph_normed, _) = crate::layers::graph_add_rms_norm(&gemma, &delta, &residual);
    eval(&graph_normed);
    assert_within(
        "gemma layer vs weight_bias=1",
        &fused_normed,
        &graph_normed,
        dt,
    );
}

/// LoRA fusion and the surgery tooling both rewrite norm weights in place
/// (a scaled or merged weight replaces the checkpoint's). The fused kernel
/// reads the weight rather than caching anything derived from it, so a rewritten
/// weight has to flow through unchanged; this pins that for the standard
/// RMSNorm convention.
#[test]
fn fused_add_rms_norm_matches_graph_with_lora_scaled_weight() {
    if !gpu_available() {
        return;
    }
    let dt = dtype::FLOAT16;
    let (delta, residual, weight) = random_case(777, 4, 4096, dt);
    // Stand-in for a fused adapter: the weight the model holds after the merge
    // is a scaled version of the checkpoint's, in the same dtype.
    let scaled = astype(
        &multiply(
            &astype(&weight, dtype::FLOAT32),
            &full_f32(&[1], 1.37, dtype::FLOAT32),
        ),
        dt,
    );
    eval(&scaled);

    let (normed, new_residual) = run_fused(&delta, &residual, &scaled, EPS, 0.0);
    let norm = RMSNorm::new(copy(&scaled), EPS);
    let (want_normed, want_residual) = crate::layers::graph_add_rms_norm(&norm, &delta, &residual);
    eval(&want_normed);
    eval(&want_residual);

    assert_within("lora-scaled normed", &normed, &want_normed, dt);
    assert_within("lora-scaled residual", &new_residual, &want_residual, dt);
}

/// Same rewrite check for the Gemma convention, where the surgery tool rewrites
/// the raw `w` and the `(1 + w)` offset is applied afterwards. Getting this
/// wrong (folding the bias into a stale cached weight) would be invisible in
/// the standard-RMSNorm test above.
#[test]
fn fused_add_rms_norm_gemma_matches_graph_with_surgery_scaled_weight() {
    if !gpu_available() {
        return;
    }
    let dt = dtype::BFLOAT16;
    let (delta, residual, weight) = random_case(778, 4, 4096, dt);
    let scaled = astype(
        &multiply(
            &astype(&weight, dtype::FLOAT32),
            &full_f32(&[1], 0.62, dtype::FLOAT32),
        ),
        dt,
    );
    eval(&scaled);

    let (normed, _) = run_fused(&delta, &residual, &scaled, EPS, 1.0);
    let gemma = GemmaRMSNorm::new(copy(&scaled), EPS);
    let (want, _) = crate::layers::graph_add_rms_norm(&gemma, &delta, &residual);
    eval(&want);
    assert_within("surgery-scaled gemma normed", &normed, &want, dt);
}

/// The kill switch has to restore the unfused path exactly, not approximately.
///
/// This test reads its own expectation from the environment, so the same test
/// body is the assertion in both directions: run the file once normally (fused
/// default-on) and once with `MLXCEL_FUSED_ADD_RMSNORM=0` (graph path, and the
/// helper's output must then be byte-identical to `graph_add_rms_norm`).
#[test]
fn fused_add_rms_norm_helper_respects_the_kill_switch() {
    if !gpu_available() {
        return;
    }
    // Expectation follows the documented precedence: an explicit truthy or
    // falsey value wins, and anything else (unset, or unrecognised) keeps the
    // compiled-in default. Deriving it as `!disabled` instead would silently
    // hard-code default-on, so flipping FUSED_ADD_RMSNORM_DEFAULT after a
    // measurement would fail this test for the wrong reason.
    let raw = std::env::var("MLXCEL_FUSED_ADD_RMSNORM").ok();
    let expected = match raw.as_deref().map(|v| v.trim().to_ascii_lowercase()) {
        Some(ref v) if v == "0" || v == "false" || v == "off" || v == "no" => false,
        Some(ref v) if v == "1" || v == "true" || v == "on" || v == "yes" => true,
        _ => crate::layers::FUSED_ADD_RMSNORM_DEFAULT,
    };
    assert_eq!(
        crate::layers::fused_add_rmsnorm_enabled(),
        expected,
        "gate does not match MLXCEL_FUSED_ADD_RMSNORM"
    );
    let disabled = !expected;

    let dt = dtype::FLOAT16;
    let (delta, residual, weight) = random_case(555, 3, 2048, dt);
    let norm = RMSNorm::new(copy(&weight), EPS);

    let (normed, new_residual) = crate::layers::fused_add_rms_norm(&norm, &delta, &residual);
    eval(&normed);
    eval(&new_residual);
    let (graph_normed, graph_residual) =
        crate::layers::graph_add_rms_norm(&norm, &delta, &residual);
    eval(&graph_normed);
    eval(&graph_residual);

    if disabled {
        assert_eq!(
            raw_bytes(&normed),
            raw_bytes(&graph_normed),
            "kill switch did not restore the graph path bit-for-bit (normed)"
        );
        assert_eq!(
            raw_bytes(&new_residual),
            raw_bytes(&graph_residual),
            "kill switch did not restore the graph path bit-for-bit (residual)"
        );
    } else {
        assert_within("gate-on normed", &normed, &graph_normed, dt);
        assert_within("gate-on residual", &new_residual, &graph_residual, dt);
    }
}

/// End-to-end greedy stability: a synthetic pre-norm stack run for many steps,
/// with the argmax of a fixed readout compared step by step.
///
/// This is the property the per-element tolerances are a proxy for. The stack
/// deliberately re-feeds its own output so the two paths' deviations compound
/// exactly the way they would across decode steps rather than being reset each
/// iteration.
#[test]
fn fused_add_rms_norm_greedy_argmax_parity_over_steps() {
    if !gpu_available() {
        return;
    }
    let dt = dtype::FLOAT16;
    let dim = 1024;
    let layers_n = 8;
    let steps = 48;
    let vocab = 512;

    random_seed(90210);
    let weights: Vec<UniquePtr<MlxArray>> = (0..layers_n)
        .map(|_| {
            let noise = unsafe { random_normal(&[dim], dtype::FLOAT32, std::ptr::null()) };
            let w = add(
                &full_f32(&[dim], 1.0, dtype::FLOAT32),
                &multiply(&noise, &full_f32(&[1], 0.1, dtype::FLOAT32)),
            );
            let w = astype(&w, dt);
            eval(&w);
            w
        })
        .collect();
    let readout = {
        let r = unsafe { random_normal(&[dim, vocab], dtype::FLOAT32, std::ptr::null()) };
        let r = astype(&r, dt);
        eval(&r);
        r
    };
    let norms: Vec<RMSNorm> = weights.iter().map(|w| RMSNorm::new(copy(w), EPS)).collect();

    let seed_x = {
        let x = unsafe { random_normal(&[1, dim], dtype::FLOAT32, std::ptr::null()) };
        let x = astype(&x, dt);
        eval(&x);
        x
    };

    let mut fused_state = copy(&seed_x);
    let mut graph_state = copy(&seed_x);
    let mut flips = 0usize;

    for step in 0..steps {
        for norm in &norms {
            // "Sublayer": a cheap deterministic nonlinearity, identical on both
            // paths, so the only difference between them is the residual join.
            let f_delta = compiled_silu(&fused_state);
            let g_delta = compiled_silu(&graph_state);

            let (f_normed, f_res) =
                run_fused(&f_delta, &fused_state, norm.raw_norm_weight(), EPS, 0.0);
            let (g_normed, g_res) = crate::layers::graph_add_rms_norm(norm, &g_delta, &graph_state);
            eval(&g_normed);
            eval(&g_res);

            // Feed the normalized output forward and keep the residual, which
            // is what a real block does between its two joins.
            fused_state = add(&f_res, &f_normed);
            graph_state = add(&g_res, &g_normed);
        }
        eval(&fused_state);
        eval(&graph_state);

        let f_logits = matmul(&fused_state, &readout);
        let g_logits = matmul(&graph_state, &readout);
        let f_tok = argmax_index(&f_logits);
        let g_tok = argmax_index(&g_logits);
        if f_tok != g_tok {
            flips += 1;
        }
        assert_eq!(
            f_tok, g_tok,
            "greedy argmax diverged at step {step} ({flips} flips so far)"
        );

        // Re-normalize the state so 48 steps of a self-feeding stack do not
        // overflow f16; both paths take the identical rescale.
        fused_state = compiled_silu(&fused_state);
        graph_state = compiled_silu(&graph_state);
    }
}

fn argmax_index(logits: &MlxArray) -> usize {
    let v = flatten_f32(logits);
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
