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

//! The fused ROCm SDPA prefill path at the chunked-prefill shapes (issue
//! #2251).
//!
//! The batched server prefills long prompts in 2,048-token chunks, so every
//! attention call of a chunk sees a query block of 2,048 rows against 2K to
//! 16K keys (GQA 32/8, head dim 128, causal, f16). On gfx1151 that call runs
//! in `kernel_sdpa_flash_wmma` at 0.95 to 1.49 TFLOP/s while the device's
//! dense f16 GEMM path delivers about 14 TFLOP/s at the same shape. The fix
//! defers the online-softmax O-accumulator rescale so the store/scale/reload
//! of all accumulator tiles through LDS (3 barriers per tile, per K block)
//! runs only when a row's block max outgrows its softmax frame, and stages Q
//! once per block instead of re-staging it from global memory on every K
//! block.
//!
//! Tests:
//!
//! - correctness of the causal call against an explicit f32
//!   `matmul + softmax + matmul` reference at the 2K and 16K key shapes, for
//!   f16 and bf16 (the deferred frame changes the rounding, not the result);
//! - a score ramp that drives a row's block max more than the rescale margin
//!   above its frame, so the deferred rescale branch runs, still matching the
//!   reference;
//! - a throughput reproduction: the median per-call time of the 16K-key f16
//!   call must reach the achieved TFLOP/s floor below. On gfx1151, both arms
//!   the probe's median of 5 calls on an idle GPU: before the fix the call
//!   measured 540.1 ms/call (0.95 TFLOP/s), after the fix 429.2 ms/call (1.20
//!   TFLOP/s), 1.26x; the non-WMMA fallback arm (`MLX_SDPA_NO_WMMA=1`) is
//!   another 3.5x slower (1516.8 ms/call, 0.34 TFLOP/s). A gate run under
//!   `scripts/rocm_gpu_guard.sh --hold` measured 392.6 ms/call (1.31
//!   TFLOP/s). The floor fails without the fix and passes with it.
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --profile test-fast --features rocm --test rocm_sdpa_flash_wmma_prefill -- --test-threads=1
//! ```

#![cfg(feature = "rocm")]

use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{self, MlxArray, UniquePtr, dtype};

/// Relative error of a half-precision GEMM against the f32 reference.
const TOLERANCE: f32 = 5e-3;

/// Rescale margin between a row's block max and its softmax frame. Must match
/// `kFaWmmaRescaleMargin` in `flash_attention_wmma.hip`.
const RESCALE_MARGIN: f32 = 10.0;

/// Achieved TFLOP/s floor of the 16K-key f16 chunked-prefill call. Set
/// between the before (0.95) and after (1.20 to 1.31) measurements on an
/// idle gfx1151.
const TFLOPS_FLOOR: f64 = 1.1;

fn on_rocm() -> bool {
    mlxcel_core::hardware::gpu_backend_kind() == mlxcel_core::hardware::GpuBackendKind::Rocm
}

/// Uniform(-1, 1) values from an integer hash, so every host builds the same
/// inputs.
fn pseudo_random(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

struct Inputs {
    q: UniquePtr<MlxArray>,
    k: UniquePtr<MlxArray>,
    v: UniquePtr<MlxArray>,
}

impl Inputs {
    /// `[1, heads, q_len, dim]` queries against `[1, kv_heads, k_len, dim]`
    /// keys/values in `dt`, from fixed pseudo-random f32 sources.
    fn new(heads: i32, kv_heads: i32, q_len: i32, k_len: i32, dim: i32, dt: i32) -> Self {
        let q_f32 = mlxcel_core::from_slice_f32(
            &pseudo_random((heads * q_len * dim) as usize, 0x2251),
            &[1, heads, q_len, dim],
        );
        let k_f32 = mlxcel_core::from_slice_f32(
            &pseudo_random((kv_heads * k_len * dim) as usize, 0x2252),
            &[1, kv_heads, k_len, dim],
        );
        let v_f32 = mlxcel_core::from_slice_f32(
            &pseudo_random((kv_heads * k_len * dim) as usize, 0x2253),
            &[1, kv_heads, k_len, dim],
        );
        Self {
            q: mlxcel_core::astype(&q_f32, dt),
            k: mlxcel_core::astype(&k_f32, dt),
            v: mlxcel_core::astype(&v_f32, dt),
        }
    }

    /// Constant key rows whose value ramps linearly from 0 to `ramp` across
    /// the key axis, with unit queries, so every query's score at key j is
    /// close to `ramp * j / (k_len - 1)`: the block max outgrows a row's
    /// softmax frame by more than the rescale margin across blocks, running
    /// the deferred rescale branch. Values stay pseudo-random.
    fn ramped(heads: i32, kv_heads: i32, q_len: i32, k_len: i32, dim: i32, ramp: f32) -> Self {
        let mut k_data = vec![0.0_f32; (kv_heads * k_len * dim) as usize];
        for j in 0..k_len {
            let value = ramp * (j as f32) / ((k_len - 1) as f32);
            let base = (j * dim) as usize;
            for kv in 0..kv_heads {
                let row = (kv * k_len * dim) as usize + base;
                for d in 0..dim {
                    k_data[row + d as usize] = value;
                }
            }
        }
        let k_f32 = mlxcel_core::from_slice_f32(&k_data, &[1, kv_heads, k_len, dim]);
        // Unit queries: every entry 1/sqrt(dim), so q·k_j = value * sqrt(dim)
        // and the wrapper's 1/sqrt(dim) scale brings the score back to the
        // ramp. f16 keeps the value within rounding of it.
        let unit = 1.0 / (dim as f32).sqrt();
        let q_data = vec![unit; (heads * q_len * dim) as usize];
        let q_f32 = mlxcel_core::from_slice_f32(&q_data, &[1, heads, q_len, dim]);
        let v_f32 = mlxcel_core::from_slice_f32(
            &pseudo_random((kv_heads * k_len * dim) as usize, 0x2253),
            &[1, kv_heads, k_len, dim],
        );
        Self {
            q: mlxcel_core::astype(&q_f32, dtype::FLOAT16),
            k: mlxcel_core::astype(&k_f32, dtype::FLOAT16),
            v: mlxcel_core::astype(&v_f32, dtype::FLOAT16),
        }
    }
}

fn max_abs(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, dtype::FLOAT32);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&f));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

/// Explicit f32 `matmul + softmax + matmul` reference of the maskless causal
/// call, per KV head so the [q_len, k_len] score matrix stays bounded, with a
/// bottom-right aligned causal mask over all keys.
fn reference(
    inp: &Inputs,
    heads: i32,
    kv_heads: i32,
    q_len: i32,
    k_len: i32,
    scale: f32,
) -> UniquePtr<MlxArray> {
    let gqa = heads / kv_heads;
    let mut mask_data = vec![0.0_f32; (q_len * k_len) as usize];
    let offset = k_len - q_len;
    for i in 0..q_len {
        for j in 0..k_len {
            if j > offset + i {
                mask_data[(i * k_len + j) as usize] = -1e30_f32;
            }
        }
    }
    let mask = mlxcel_core::from_slice_f32(&mask_data, &[q_len, k_len]);
    let mut parts: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(kv_heads as usize);
    for kv in 0..kv_heads {
        let q_h = mlxcel_core::slice(
            &inp.q,
            &[0, kv * gqa, 0, 0],
            &[1, (kv + 1) * gqa, q_len, head_dim(&inp.q)],
        );
        let k_h = mlxcel_core::slice(
            &inp.k,
            &[0, kv, 0, 0],
            &[1, kv + 1, k_len, head_dim(&inp.k)],
        );
        let v_h = mlxcel_core::slice(
            &inp.v,
            &[0, kv, 0, 0],
            &[1, kv + 1, k_len, head_dim(&inp.v)],
        );
        let k_t = mlxcel_core::transpose_axes(&k_h, &[0, 1, 3, 2]);
        let scores = mlxcel_core::matmul(&q_h, &k_t);
        let scores = mlxcel_core::multiply_scalar(&scores, scale);
        let scores = mlxcel_core::add(&scores, &mask);
        let probs = mlxcel_core::softmax(&scores, -1);
        let out = mlxcel_core::matmul(&probs, &v_h);
        parts.push(out);
    }
    let refs: Vec<&MlxArray> = parts.iter().map(|p| p.as_ref().unwrap()).collect();
    mlxcel_core::concatenate_many(&refs, 1)
}

fn head_dim(a: &MlxArray) -> i32 {
    *mlxcel_core::array_shape(a).last().unwrap()
}

/// Relative L2 distance of the fused output against the f32 reference.
fn check_case(dt: i32, name: &str, heads: i32, kv_heads: i32, q_len: i32, k_len: i32) {
    let inp = Inputs::new(heads, kv_heads, q_len, k_len, 128, dt);
    let scale = 1.0 / (128.0_f32).sqrt();
    let out = mlxcel_core::fast_scaled_dot_product_attention_causal(&inp.q, &inp.k, &inp.v, scale);
    if let Err(err) = mlxcel_core::try_eval(&out) {
        panic!("{name}: causal SDPA on the GPU failed: {err}");
    }
    let expect = reference(&inp, heads, kv_heads, q_len, k_len, scale);
    if let Err(err) = mlxcel_core::try_eval(&expect) {
        panic!("{name}: f32 reference failed: {err}");
    }
    let diff = mlxcel_core::subtract(&mlxcel_core::astype(&out, dtype::FLOAT32), &expect);
    let err = max_abs(&diff);
    assert!(
        err <= TOLERANCE,
        "{name}: max abs diff {err:.3e} against the f32 reference exceeds {TOLERANCE:.1e}"
    );
}

#[test]
fn causal_prefill_shapes_match_reference() {
    if !on_rocm() {
        return;
    }
    let _gpu = DefaultDeviceGuard::gpu();
    // The chunked-prefill shapes: a 2,048-query chunk against 2K and 16K keys.
    check_case(dtype::FLOAT16, "f16 2048q/2048k", 32, 8, 2048, 2048);
    check_case(dtype::FLOAT16, "f16 2048q/16384k", 32, 8, 2048, 16384);
    check_case(dtype::BFLOAT16, "bf16 2048q/16384k", 32, 8, 2048, 16384);
}

#[test]
fn score_ramp_exercises_deferred_rescale() {
    if !on_rocm() {
        return;
    }
    let _gpu = DefaultDeviceGuard::gpu();
    // Scores span 3x the margin across the key axis, so rows rescale several
    // times over the K/V loop; the result must still match the reference.
    let inp = Inputs::ramped(32, 8, 2048, 8192, 128, 3.0 * RESCALE_MARGIN);
    let scale = 1.0 / (128.0_f32).sqrt();
    let out = mlxcel_core::fast_scaled_dot_product_attention_causal(&inp.q, &inp.k, &inp.v, scale);
    if let Err(err) = mlxcel_core::try_eval(&out) {
        panic!("ramped causal SDPA on the GPU failed: {err}");
    }
    let expect = reference(&inp, 32, 8, 2048, 8192, scale);
    if let Err(err) = mlxcel_core::try_eval(&expect) {
        panic!("ramped f32 reference failed: {err}");
    }
    let diff = mlxcel_core::subtract(&mlxcel_core::astype(&out, dtype::FLOAT32), &expect);
    let err = max_abs(&diff);
    assert!(
        err <= TOLERANCE,
        "ramped: max abs diff {err:.3e} against the f32 reference exceeds {TOLERANCE:.1e}"
    );
}

/// Causal-aware FLOPs of a maskless causal SDPA call.
fn causal_flops(heads: i32, dim: i32, q_len: i32, k_len: i32) -> f64 {
    let offset = (k_len - q_len).max(0) as f64;
    let q = q_len as f64;
    4.0 * heads as f64 * dim as f64 * (q * offset + q * (q + 1.0) / 2.0)
}

#[test]
fn chunked_prefill_throughput_reaches_floor() {
    if !on_rocm() {
        return;
    }
    let _gpu = DefaultDeviceGuard::gpu();
    let inp = Inputs::new(32, 8, 2048, 16384, 128, dtype::FLOAT16);
    let scale = 1.0 / (128.0_f32).sqrt();
    for _ in 0..2 {
        let out =
            mlxcel_core::fast_scaled_dot_product_attention_causal(&inp.q, &inp.k, &inp.v, scale);
        mlxcel_core::eval(&out);
    }
    let mut per_call: Vec<f64> = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let out =
            mlxcel_core::fast_scaled_dot_product_attention_causal(&inp.q, &inp.k, &inp.v, scale);
        mlxcel_core::eval(&out);
        per_call.push(start.elapsed().as_secs_f64());
    }
    let secs = median(&mut per_call);
    let tflops = causal_flops(32, 128, 2048, 16384) / secs / 1e12;
    assert!(
        tflops >= TFLOPS_FLOOR,
        "16K-key chunked-prefill call measured {tflops:.2} TFLOP/s ({:.1} ms/call), floor {TFLOPS_FLOOR}",
        secs * 1e3
    );
}
