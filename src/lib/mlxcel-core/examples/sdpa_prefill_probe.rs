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

//! Per-call time and achieved TFLOP/s of the fused ROCm SDPA prefill path at
//! the chunked-prefill shapes (issue #2251).
//!
//! The batched server prefills long prompts in 2,048-token chunks, so every
//! attention call of a chunk sees a query block of 2,048 rows against 2K to
//! 16K keys (GQA 32/8, head dim 128, causal, f16). On gfx1151 that call is
//! dominated by `kernel_sdpa_flash_wmma` at about 192 ms per call. This probe
//! measures the call the model makes, the maskless causal
//! `fast_scaled_dot_product_attention_causal` wrapper, at those shapes,
//! reports the median per-call time and the achieved TFLOP/s from the
//! causal-aware FLOP count, and compares against a dense f16 `matmul` of the
//! same shape (the device GEMM bound). Run it a second time with
//! `MLX_SDPA_NO_WMMA=1` to measure the scalar flash arm.
//!
//! ```sh
//! cargo build --release --features rocm -p mlxcel-core --example sdpa_prefill_probe
//! target/release/examples/sdpa_prefill_probe --key-lens 2048,8192,16384
//! ```

use std::time::Instant;

use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{self, MlxArray, UniquePtr, dtype};

struct Args {
    q_len: i32,
    key_lens: Vec<i32>,
    heads: i32,
    kv_heads: i32,
    dim: i32,
    runs: usize,
    warmup: usize,
    scale: f32,
}

fn parse_args() -> Args {
    let mut args = Args {
        q_len: 2048,
        key_lens: vec![2048, 8192, 16384],
        heads: 32,
        kv_heads: 8,
        dim: 128,
        runs: 5,
        warmup: 2,
        scale: 1.0 / (128.0_f32).sqrt(),
    };
    // Both spellings: --key=value and --key value.
    let tokens: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < tokens.len() {
        let arg = tokens[i].clone();
        let (key, inline_value) = match arg.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (arg.clone(), None),
        };
        let value = match inline_value {
            Some(v) => v,
            None => {
                i += 1;
                if i >= tokens.len() {
                    eprintln!("missing value for {key}");
                    break;
                }
                tokens[i].clone()
            }
        };
        i += 1;
        let take = |field: &mut i32| {
            if let Ok(v) = value.parse() {
                *field = v;
            }
        };
        match key.as_str() {
            "--q-len" => take(&mut args.q_len),
            "--heads" => take(&mut args.heads),
            "--kv-heads" => take(&mut args.kv_heads),
            "--dim" => {
                take(&mut args.dim);
                args.scale = 1.0 / (args.dim as f32).sqrt();
            }
            "--runs" => {
                if let Ok(v) = value.parse() {
                    args.runs = v;
                }
            }
            "--warmup" => {
                if let Ok(v) = value.parse() {
                    args.warmup = v;
                }
            }
            "--scale" => {
                if let Ok(v) = value.parse() {
                    args.scale = v;
                }
            }
            "--key-lens" => {
                args.key_lens = value.split(',').filter_map(|s| s.parse().ok()).collect();
            }
            _ => eprintln!("ignoring unknown argument {arg}"),
        }
    }
    args
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

/// Uniform(-1, 1) values from an integer hash, so every host builds the same
/// inputs and the same shape is comparable across runs and builds.
fn pseudo_random(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// FLOPs of a causal SDPA call: query i (global position offset + i) attends
/// keys [0, offset + i + 1], for QK^T and P@V (4 D-FLOPs per key per head).
fn causal_flops(heads: i32, dim: i32, q_len: i32, k_len: i32) -> f64 {
    let offset = (k_len - q_len).max(0) as f64;
    let q = q_len as f64;
    4.0 * heads as f64 * dim as f64 * (q * offset + q * (q + 1.0) / 2.0)
}

fn build_inputs(
    heads: i32,
    kv_heads: i32,
    q_len: i32,
    k_len: i32,
    dim: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
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
    let q = mlxcel_core::astype(&q_f32, dtype::FLOAT16);
    let k = mlxcel_core::astype(&k_f32, dtype::FLOAT16);
    let v = mlxcel_core::astype(&v_f32, dtype::FLOAT16);
    mlxcel_core::eval(&q);
    mlxcel_core::eval(&k);
    mlxcel_core::eval(&v);
    (q, k, v)
}

/// Median per-call seconds of the maskless causal SDPA wrapper.
fn time_sdpa(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    runs: usize,
    warmup: usize,
) -> f64 {
    for _ in 0..warmup {
        let out = mlxcel_core::fast_scaled_dot_product_attention_causal(q, k, v, scale);
        mlxcel_core::eval(&out);
    }
    let mut per_call: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let out = mlxcel_core::fast_scaled_dot_product_attention_causal(q, k, v, scale);
        mlxcel_core::eval(&out);
        per_call.push(start.elapsed().as_secs_f64());
    }
    median(&mut per_call)
}

/// Median per-call seconds of a dense f16 GEMM of the QK^T shape, the device
/// GEMM bound for one of the two matmuls the flash kernel fuses.
fn time_dense_gemm(q_len: i32, k_len: i32, dim: i32, runs: usize, warmup: usize) -> f64 {
    let a = mlxcel_core::from_slice_f32(
        &pseudo_random((q_len * dim) as usize, 0x2254),
        &[q_len, dim],
    );
    let b = mlxcel_core::from_slice_f32(
        &pseudo_random((k_len * dim) as usize, 0x2255),
        &[k_len, dim],
    );
    let a = mlxcel_core::astype(&a, dtype::FLOAT16);
    let b = mlxcel_core::astype(&mlxcel_core::transpose(&b), dtype::FLOAT16);
    mlxcel_core::eval(&a);
    mlxcel_core::eval(&b);
    for _ in 0..warmup {
        let c = mlxcel_core::matmul(&a, &b);
        mlxcel_core::eval(&c);
    }
    let mut per_call: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let c = mlxcel_core::matmul(&a, &b);
        mlxcel_core::eval(&c);
        per_call.push(start.elapsed().as_secs_f64());
    }
    median(&mut per_call)
}

fn main() {
    let args = parse_args();
    let _gpu = DefaultDeviceGuard::gpu();
    let no_wmma = std::env::var("MLX_SDPA_NO_WMMA").is_ok();

    println!("=== SDPA chunked-prefill probe (issue #2251) ===");
    println!(
        "q_len={} heads={} kv_heads={} dim={} runs={} warmup={} MLX_SDPA_NO_WMMA={}",
        args.q_len, args.heads, args.kv_heads, args.dim, args.runs, args.warmup, no_wmma
    );
    println!();
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "keys", "ms/call", "TFLOP/s", "gemm_ms", "gemm_TFLOP/s"
    );
    println!("{}", "-".repeat(62));

    for k_len in &args.key_lens {
        let (q, k, v) = build_inputs(args.heads, args.kv_heads, args.q_len, *k_len, args.dim);
        let secs = time_sdpa(&q, &k, &v, args.scale, args.runs, args.warmup);
        let flops = causal_flops(args.heads, args.dim, args.q_len, *k_len);
        let tflops = flops / secs / 1e12;
        let gemm_secs = time_dense_gemm(args.q_len, *k_len, args.dim, args.runs, args.warmup);
        let gemm_flops = 2.0 * (args.q_len as f64) * (*k_len as f64) * (args.dim as f64);
        let gemm_tflops = gemm_flops / gemm_secs / 1e12;
        println!(
            "{:>8} {:>12.1} {:>12.2} {:>12.1} {:>12.2}",
            k_len,
            secs * 1e3,
            tflops,
            gemm_secs * 1e3,
            gemm_tflops
        );
    }
}
