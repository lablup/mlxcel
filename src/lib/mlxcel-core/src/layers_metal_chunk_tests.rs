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

//! The Metal score-materialization chunk gate (issue #3033): it chunks only
//! the calls MLX itself sends to the graph fallback, judged in the mask mode
//! the call actually issues, and the chunked result matches the single call.

use super::{
    ChunkGateBackend, attention_dispatch, chunked_causal_attention, chunked_query_attention,
    metal_attention_chunk_switch, sdpa_query_chunk_within_budget,
};
use crate::ffi::MlxArray;
use crate::window_sdpa::{WindowSdpaShape, sdpa_mode_is_fused};
use crate::{dtype, ffi};
use cxx::UniquePtr;

/// 16 KiB: small enough that every multi-row shape below would be over budget
/// if it materialized (the smallest, `[1, 4, 64, 64]` scores in fp16, is
/// 32 KiB), so the gate's answer is decided by the fallback verdict.
const BUDGET: usize = 16 * 1024;

fn on_metal() -> bool {
    crate::hardware::gpu_backend_kind() == crate::hardware::GpuBackendKind::Metal
}

/// Lazily built fp16 operands; the gate reads shapes and dtypes only.
fn zeros_qkv(q_len: i32, k_len: i32, d: i32) -> [UniquePtr<MlxArray>; 3] {
    [
        ffi::zeros(&[1, 4, q_len, d], dtype::FLOAT16),
        ffi::zeros(&[1, 2, k_len, d], dtype::FLOAT16),
        ffi::zeros(&[1, 2, k_len, d], dtype::FLOAT16),
    ]
}

fn decide(
    qkv: &[UniquePtr<MlxArray>; 3],
    softcap: f32,
    arr_masked: bool,
    do_causal: bool,
    backend: ChunkGateBackend,
) -> Option<i32> {
    let [q, k, v] = qkv;
    sdpa_query_chunk_within_budget(q, k, v, softcap, arr_masked, do_causal, BUDGET, backend)
}

fn fused(qkv: &[UniquePtr<MlxArray>; 3], arr_masked: bool, do_causal: bool) -> bool {
    let [q, k, v] = qkv;
    sdpa_mode_is_fused(&WindowSdpaShape::of(q, k, v), arr_masked, do_causal)
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let a = ffi::astype(a, dtype::FLOAT32);
    let b = ffi::astype(b, dtype::FLOAT32);
    let peak = ffi::max_all(&ffi::abs(&ffi::subtract(&a, &b)));
    ffi::eval(&peak);
    ffi::item_f32(&peak)
}

/// Deterministic fp16 tensor with values in [-1, 1].
fn half_tensor(shape: &[i32]) -> UniquePtr<MlxArray> {
    let len: i32 = shape.iter().product();
    let data: Vec<f32> = (0..len)
        .map(|i| ((i as f32) * 0.7371 + 0.13).sin())
        .collect();
    ffi::astype(&crate::from_slice_f32(&data, shape), dtype::FLOAT16)
}

#[test]
fn metal_kill_switch_parses_off_values_only() {
    for off in ["0", "false", "no", "off", " OFF ", "False"] {
        assert!(!metal_attention_chunk_switch(Some(off)), "{off:?}");
    }
    for on in [None, Some("1"), Some("true"), Some(""), Some("yes")] {
        assert!(metal_attention_chunk_switch(on), "{on:?}");
    }
}

/// The gate's answer is MLX's fallback verdict for the call's own mask mode,
/// never a verdict borrowed from another mode.
#[test]
fn metal_gate_chunks_exactly_the_calls_mlx_does_not_fuse() {
    if !on_metal() {
        return;
    }
    let metal = ChunkGateBackend::Metal;
    // head_dim 256, q_len == k_len >= 2048: fused in causal mode on every
    // Metal generation MLX ships rules for. The array-mask call over the same
    // shapes is fused only where MLX says so (M5+), and the gate follows.
    let square = zeros_qkv(2048, 2048, 256);
    assert!(fused(&square, false, true));
    assert_eq!(decide(&square, 0.0, false, true, metal), None);
    assert_eq!(
        decide(&square, 0.0, true, false, metal).is_some(),
        !fused(&square, true, false)
    );

    // head_dim 256 over a longer key (a later prefill chunk): MLX's causal
    // kernel needs q_len == k_len at this head dim below 1024 queries, so it
    // falls back and is chunked within budget.
    let later = zeros_qkv(64, 128, 256);
    assert!(!fused(&later, false, true));
    let chunk = decide(&later, 0.0, false, true, metal).expect("fallback is chunked");
    // 4 heads * 128 keys * 2 bytes = 1 KiB per query row: 64 KiB of scores
    // over a 16 KiB budget is four chunks of 16 rows.
    assert_eq!(chunk, 16);

    // head_dim 128 with no mask is fused; with a softcap the composite always
    // materializes, whatever MLX would do without it.
    let plain = zeros_qkv(64, 64, 128);
    assert!(fused(&plain, false, false));
    assert_eq!(decide(&plain, 0.0, false, false, metal), None);
    assert!(decide(&plain, 30.0, false, false, metal).is_some());

    // Each mode's answer agrees with the probe for the shapes above.
    for qkv in [&square, &later, &plain] {
        for (arr, causal) in [(true, false), (false, true), (false, false)] {
            assert_eq!(
                decide(qkv, 0.0, arr, causal, metal).is_some(),
                !fused(qkv, arr, causal),
                "arr_masked={arr} do_causal={causal}"
            );
        }
    }
}

/// A call within budget, a disabled budget, a decode row and the kill
/// switch's backend never chunk, on any host.
#[test]
fn metal_gate_leaves_in_budget_and_disabled_calls_alone() {
    let later = zeros_qkv(64, 128, 256);
    let [q, k, v] = &later;
    for backend in [ChunkGateBackend::Metal, ChunkGateBackend::Unchunked] {
        assert_eq!(
            sdpa_query_chunk_within_budget(q, k, v, 0.0, false, true, 1 << 30, backend),
            None
        );
        assert_eq!(
            sdpa_query_chunk_within_budget(q, k, v, 0.0, false, true, 0, backend),
            None
        );
    }
    assert_eq!(
        decide(&later, 0.0, false, true, ChunkGateBackend::Unchunked),
        None
    );
    let decode = zeros_qkv(1, 128, 256);
    assert_eq!(
        decide(&decode, 0.0, false, true, ChunkGateBackend::Metal),
        None
    );
}

/// The chunked paths the Metal gate selects reproduce the single fallback
/// call at the head dims and dtype the issue is about.
#[test]
fn metal_chunked_paths_match_the_single_call_at_head_dim_256_fp16() {
    let scale = 1.0 / 16.0_f32;
    let q = half_tensor(&[1, 4, 48, 256]);
    let k = half_tensor(&[1, 2, 80, 256]);
    let v = half_tensor(&[1, 2, 80, 256]);

    // Maskless causal over a cached prefix, the shape of a later prefill
    // chunk.
    let native = ffi::ffi_fast_scaled_dot_product_attention_causal(&q, &k, &v, scale);
    for chunk in [7, 16, 47] {
        let chunked = chunked_causal_attention(&q, &k, &v, scale, chunk);
        assert_eq!(ffi::array_shape(&chunked), ffi::array_shape(&native));
        let diff = max_abs_diff(&native, &chunked);
        assert!(diff < 2e-3, "causal chunk={chunk} diverged by {diff}");
    }

    // Array mask (the windowed-causal mask a sliding layer builds), plain and
    // softcapped.
    let mask = crate::utils::create_causal_mask_with_window_full(48, 32, Some(40));
    let mask = mask.as_ref().unwrap();
    for softcap in [0.0, 30.0] {
        let full = attention_dispatch(&q, &k, &v, scale, Some(mask), softcap);
        let chunked = chunked_query_attention(&q, &k, &v, scale, Some(mask), softcap, 16);
        let diff = max_abs_diff(&full, &chunked);
        assert!(
            diff < 2e-3,
            "array-mask softcap={softcap} diverged by {diff}"
        );
    }
}

/// One attention call at a real prefill shape, timed and measured.
struct ProbeRun {
    peak_bytes: usize,
    millis: f64,
    out: UniquePtr<MlxArray>,
}

fn probe_run(f: impl FnOnce() -> UniquePtr<MlxArray>) -> ProbeRun {
    ffi::eval(&ffi::zeros(&[1], dtype::FLOAT32));
    crate::memory::clear_cache();
    crate::memory::reset_peak_memory();
    let start = std::time::Instant::now();
    let out = f();
    ffi::eval(&out);
    let millis = start.elapsed().as_secs_f64() * 1e3;
    ProbeRun {
        peak_bytes: ffi::get_peak_memory(),
        millis,
        out,
    }
}

/// Op-level evidence for #3033, run by hand on a GPU host:
/// `MLXCEL_SDPA_PROBE_LEN=28000 cargo test --release -p mlxcel-core --lib
/// metal_chunk_tests::sdpa_chunk_peak_memory_probe -- --ignored --nocapture`.
///
/// For the reporter's shape (24 query heads over 4 KV heads, head_dim 256,
/// fp16) it measures MLX's peak active memory and wall time of one prefill
/// attention call through the production entry points ([`super::attention`]
/// with the additive causal mask a model passes, and
/// [`crate::causal_attention`] in MLX's causal mode) against the single
/// unchunked call they would make without the gate, and the largest
/// difference between the two outputs.
#[test]
#[ignore = "allocates tens of GB; run by hand with MLXCEL_SDPA_PROBE_LEN"]
fn sdpa_chunk_peak_memory_probe() {
    let len: i32 = std::env::var("MLXCEL_SDPA_PROBE_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(28000);
    let env_i32 = |name: &str, default: i32| -> i32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let heads = env_i32("MLXCEL_SDPA_PROBE_HEADS", 24);
    // head_dim and KV heads default to the reporter's shape. On M5 and later
    // MLX fuses that array-mask call, so the fallback the gate chunks is
    // reached there with `MLXCEL_SDPA_PROBE_DIM=512` (one KV head, 16 query
    // heads: Gemma 4's global-layer shape with an array mask).
    let dim = env_i32("MLXCEL_SDPA_PROBE_DIM", 256);
    let kv_heads = env_i32("MLXCEL_SDPA_PROBE_KV_HEADS", 4);
    let normal = |shape: &[i32]| {
        // SAFETY: a null key draws from MLX's global generator.
        let x = unsafe { ffi::random_normal(shape, dtype::FLOAT16, std::ptr::null()) };
        ffi::eval(&x);
        x
    };
    crate::ffi::random_seed(3033);
    let q = normal(&[1, heads, len, dim]);
    let k = normal(&[1, kv_heads, len, dim]);
    let v = normal(&[1, kv_heads, len, dim]);
    let scale = 1.0 / (dim as f32).sqrt();
    let mask = crate::utils::create_causal_mask(len, 0);
    ffi::eval(&mask);
    let mask = mask.as_ref().unwrap();
    let gb = |b: usize| b as f64 / 1e9;
    println!(
        "backend={:?} shape=[1,{heads},{len},{dim}] fp16 kv_heads={kv_heads}",
        crate::hardware::gpu_backend_kind()
    );

    let arr_fused = sdpa_mode_is_fused(&WindowSdpaShape::of(&q, &k, &v), true, false);
    let single = probe_run(|| attention_dispatch(&q, &k, &v, scale, Some(mask), 0.0));
    let gated = probe_run(|| super::attention(&q, &k, &v, scale, Some(mask), 0.0, 0));
    println!(
        "array mask: mlx_fused={arr_fused} single_call peak={:.2} GB {:.0} ms | gated peak={:.2} GB {:.0} ms | max_abs_diff={}",
        gb(single.peak_bytes),
        single.millis,
        gb(gated.peak_bytes),
        gated.millis,
        max_abs_diff(&single.out, &gated.out)
    );
    drop((single, gated));

    let causal_fused = sdpa_mode_is_fused(&WindowSdpaShape::of(&q, &k, &v), false, true);
    let single = probe_run(|| ffi::ffi_fast_scaled_dot_product_attention_causal(&q, &k, &v, scale));
    let gated = probe_run(|| crate::causal_attention(&q, &k, &v, scale, 0.0, 0));
    println!(
        "causal mode: mlx_fused={causal_fused} single_call peak={:.2} GB {:.0} ms | gated peak={:.2} GB {:.0} ms | max_abs_diff={}",
        gb(single.peak_bytes),
        single.millis,
        gb(gated.peak_bytes),
        gated.millis,
        max_abs_diff(&single.out, &gated.out)
    );
}
