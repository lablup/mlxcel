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

//! Per-call time of the quantized matmuls a batched decode step issues.
//!
//! Batched decode hands every projection an activation shaped `[B, 1, K]`
//! against one shared 2D weight. This probe times that shape against the same
//! rows folded into `[B, K]`, and the MoE `gather_qmm` a decode step issues
//! (`[T, 1, 1, K]` against a `[E, N, K]` expert stack with `[T, top_k]`
//! indices, unsorted and pre-sorted), so a kernel-selection change can be
//! read as effective weight bandwidth per batch size.
//!
//! Run with:
//!     cargo run --release --example qmm_fold_batch_probe -p mlxcel-core --features cuda

use std::time::Instant;

use mlxcel_core::{self, MlxArray, UniquePtr, dtype};

const WARMUP: usize = 5;
const ITERS: usize = 40;
const GROUP: i32 = 64;
const BITS: i32 = 4;

fn randn(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let key = mlxcel_core::random_key(seed);
    // SAFETY: `key` lives across the call.
    let x = unsafe {
        mlxcel_core::random_normal(shape, dtype::FLOAT32, key.as_ref().unwrap() as *const _)
    };
    mlxcel_core::astype(&x, dtype::BFLOAT16)
}

struct Quantized {
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
}

fn quantize(shape: &[i32], seed: u64) -> Quantized {
    let w = mlxcel_core::multiply_scalar(&randn(shape, seed), 0.02);
    let q = mlxcel_core::quantize_weights(&w, GROUP, BITS);
    let out = Quantized {
        w: mlxcel_core::quantized_weights_w(&q),
        scales: mlxcel_core::quantized_weights_scales(&q),
        biases: mlxcel_core::quantized_weights_biases(&q),
    };
    mlxcel_core::eval(&out.w);
    mlxcel_core::eval(&out.scales);
    mlxcel_core::eval(&out.biases);
    out
}

fn time_ms(mut f: impl FnMut() -> UniquePtr<MlxArray>) -> f64 {
    for _ in 0..WARMUP {
        mlxcel_core::eval(&f());
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        mlxcel_core::eval(&f());
    }
    start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64
}

fn dense(label: &str, n: i32, k: i32) {
    let q = quantize(&[n, k], 1);
    let weight_mb = (n as f64 * k as f64 * BITS as f64 / 8.0) / 1e6;
    println!("dense {label} N={n} K={k} ({weight_mb:.1} MB packed)");
    for b in [1, 2, 4, 8, 16] {
        let x3 = randn(&[b, 1, k], 7);
        let x2 = mlxcel_core::reshape(&x3, &[b, k]);
        mlxcel_core::eval(&x3);
        mlxcel_core::eval(&x2);
        let run = |x: &MlxArray| {
            // SAFETY: every pointer is owned by `q` for the whole call.
            unsafe {
                mlxcel_core::quantized_matmul(
                    x,
                    &q.w,
                    &q.scales,
                    q.biases.as_ref().unwrap() as *const _,
                    true,
                    GROUP,
                    BITS,
                    "affine",
                )
            }
        };
        let t3 = time_ms(|| run(&x3));
        let t2 = time_ms(|| run(&x2));
        println!(
            "  B={b:2}  [B,1,K] {t3:7.3} ms ({:6.1} GB/s)   [B,K] {t2:7.3} ms ({:6.1} GB/s)",
            weight_mb / t3,
            weight_mb / t2
        );
    }
}

fn gather(label: &str, experts: i32, n: i32, k: i32, top_k: i32) {
    let q = quantize(&[experts, n, k], 2);
    println!("gather {label} E={experts} N={n} K={k} top_k={top_k}");
    for t in [1, 2, 4, 8] {
        let pairs = t * top_k;
        // Distinct experts per token, spread over the stack.
        let idx: Vec<u32> = (0..pairs)
            .map(|p| (p as u32 * 37 + (p / top_k) as u32 * 11) % experts as u32)
            .collect();
        let indices = mlxcel_core::from_slice_u32(&idx, &[t, top_k]);
        let x = randn(&[t, 1, 1, k], 9);
        let mut sorted_idx = idx.clone();
        sorted_idx.sort_unstable();
        let sorted = mlxcel_core::from_slice_u32(&sorted_idx, &[pairs]);
        let xs = randn(&[pairs, 1, k], 11);
        mlxcel_core::eval(&x);
        mlxcel_core::eval(&xs);
        let run = |x: &MlxArray, rhs: &MlxArray, sorted: bool| {
            // SAFETY: every pointer is owned by this scope for the whole call.
            unsafe {
                mlxcel_core::gather_qmm(
                    x,
                    &q.w,
                    &q.scales,
                    q.biases.as_ref().unwrap() as *const _,
                    std::ptr::null(),
                    rhs as *const _,
                    true,
                    GROUP,
                    BITS,
                    sorted,
                    "affine",
                    std::ptr::null(),
                )
            }
        };
        let unsorted_ms = time_ms(|| run(&x, &indices, false));
        let sorted_ms = time_ms(|| run(&xs, &sorted, true));
        let pair_mb = pairs as f64 * n as f64 * k as f64 * BITS as f64 / 8.0 / 1e6;
        println!(
            "  T={t}  pairs={pairs:2}  unsorted {unsorted_ms:7.3} ms ({:6.1} GB/s)   sorted {sorted_ms:7.3} ms ({:6.1} GB/s)",
            pair_mb / unsorted_ms,
            pair_mb / sorted_ms
        );
    }
}

fn main() {
    // qwen3.8-27b MLP and a 5120-wide projection.
    dense("mlp_up", 17408, 5120);
    dense("mlp_down", 5120, 17408);
    dense("attn_o", 5120, 6144);
    // gemma-4-26b-a4b experts: gate/up 704 (not a multiple of 128), down 2816.
    gather("gemma4_gate_up", 128, 704, 2816, 8);
    gather("gemma4_down", 128, 2816, 704, 8);
}
