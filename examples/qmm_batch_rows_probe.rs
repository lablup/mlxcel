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

//! Quantized-matmul cost of a decode batch by activation layout (issue #2156).
//!
//! The batched server decode feeds each projection an activation of shape
//! `[B, 1, K]` (B sequences, one token each). The same rows can also reach
//! `quantized_matmul` as `[B, K]` (one matrix of B rows) or `[1, B, K]`.
//! A backend that reads `[B, 1, K]` as B separate one-row products streams the
//! weight B times per step. This probe times the Meta-Llama-3.1-8B projection
//! shapes (4-bit, group 64; f16 activations and f16 scales as that checkpoint
//! ships them, or bf16 as most other checkpoints do) in each layout, and
//! checks the layouts return the same values.
//!
//! Method as `qmm_gemv_microbench`: per shape, round-robin over enough weight
//! copies to keep reads out of cache, T back-to-back products folded into one
//! root per eval, best of ROUNDS.
//!
//! Usage: cargo run --release --features rocm --example qmm_batch_rows_probe
//!          [ BATCH ] [ T_PER_BATCH ] [ ROUNDS ] [ f16 | bf16 ]
//! Defaults: BATCH=4, T_PER_BATCH=64, ROUNDS=5, f16.

use mlxcel_core::{
    MlxArray, UniquePtr, abs, add, astype, dtype, eval, from_slice_f32, item_f32, max_all,
    quantize_weights_biases, quantize_weights_scales, quantize_weights_w, quantized_matmul,
    reshape, subtract, synchronize_default,
};
use std::time::Instant;

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

// Meta-Llama-3.1-8B: hidden 4096, intermediate 14336, 32 q heads, 8 kv heads.
const SHAPES: &[(&str, i32, i32)] = &[
    ("q_proj", 4096, 4096),
    ("k/v_proj", 1024, 4096),
    ("o_proj", 4096, 4096),
    ("gate/up_proj", 14336, 4096),
    ("down_proj", 4096, 14336),
];

fn make(shape: &[i32], seed: usize, dt: i32) -> UniquePtr<MlxArray> {
    let total: usize = shape.iter().map(|&d| d as usize).product();
    let data: Vec<f32> = (0..total)
        .map(|i| (((i + seed * 7919) as f32) * 0.000271).sin() * 0.05)
        .collect();
    astype(&from_slice_f32(&data, shape), dt)
}

struct QuantizedWeight {
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
}

fn quantize(out: i32, inp: i32, seed: usize, dt: i32) -> QuantizedWeight {
    let wf = make(&[out, inp], seed, dt);
    let w = quantize_weights_w(&wf, GROUP_SIZE, BITS);
    let scales = quantize_weights_scales(&wf, GROUP_SIZE, BITS);
    let biases = quantize_weights_biases(&wf, GROUP_SIZE, BITS);
    eval(&w);
    eval(&scales);
    eval(&biases);
    QuantizedWeight { w, scales, biases }
}

fn qmm(x: &MlxArray, qw: &QuantizedWeight) -> UniquePtr<MlxArray> {
    // SAFETY: every array outlives the call and the biases pointer is valid.
    unsafe {
        quantized_matmul(
            x,
            &qw.w,
            &qw.scales,
            qw.biases.as_ref().unwrap() as *const MlxArray,
            true,
            GROUP_SIZE,
            BITS,
            "affine",
        )
    }
}

fn run_batch(x: &MlxArray, weights: &[QuantizedWeight], t: usize) -> f64 {
    let start = Instant::now();
    let mut acc: Option<UniquePtr<MlxArray>> = None;
    for i in 0..t {
        let y = qmm(x, &weights[i % weights.len()]);
        acc = Some(match acc {
            None => y,
            Some(a) => add(&a, &y),
        });
    }
    let root = acc.unwrap();
    eval(&root);
    synchronize_default();
    start.elapsed().as_secs_f64()
}

fn best_us(x: &MlxArray, weights: &[QuantizedWeight], t: usize, rounds: usize) -> f64 {
    let _ = run_batch(x, weights, t);
    let mut best = f64::MAX;
    for _ in 0..rounds {
        best = best.min(run_batch(x, weights, t));
    }
    best * 1e6 / t as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let batch: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let t: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let (dt, dt_name) = match args.get(4).map(String::as_str) {
        Some("bf16") => (dtype::BFLOAT16, "bf16"),
        _ => (dtype::FLOAT16, "f16"),
    };
    println!(
        "=== qmm decode-batch layout probe === B={batch} T={t} ROUNDS={rounds} 4-bit g{GROUP_SIZE} {dt_name}"
    );
    println!(
        "{:<14} {:>6} {:>6} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "shape", "out", "in", "[1,1,K] us", "[B,1,K] us", "[B,K] us", "[1,B,K] us", "max|diff|"
    );
    let mut totals = [0.0f64; 4];
    for &(name, out, inp) in SHAPES {
        let bytes = out as i64 * inp as i64 / 2 + out as i64 * (inp / GROUP_SIZE) as i64 * 4;
        let copies = ((128 * 1024 * 1024) / bytes).clamp(2, 12) as usize;
        let weights: Vec<QuantizedWeight> =
            (0..copies).map(|s| quantize(out, inp, s + 1, dt)).collect();
        let x1 = make(&[1, 1, inp], 0, dt);
        let xb3 = make(&[batch, 1, inp], 1, dt);
        let xb2 = reshape(&xb3, &[batch, inp]);
        let x1b = reshape(&xb3, &[1, batch, inp]);
        eval(&x1);
        eval(&xb3);
        eval(&xb2);
        eval(&x1b);
        synchronize_default();
        // Same rows, different layout: the products must agree.
        let a = reshape(&qmm(&xb3, &weights[0]), &[batch, out]);
        let b = qmm(&xb2, &weights[0]);
        let c = reshape(&qmm(&x1b, &weights[0]), &[batch, out]);
        let d1 = item_f32(&max_all(&abs(&subtract(
            &astype(&a, dtype::FLOAT32),
            &astype(&b, dtype::FLOAT32),
        ))));
        let d2 = item_f32(&max_all(&abs(&subtract(
            &astype(&a, dtype::FLOAT32),
            &astype(&c, dtype::FLOAT32),
        ))));
        let us = [
            best_us(&x1, &weights, t, rounds),
            best_us(&xb3, &weights, t, rounds),
            best_us(&xb2, &weights, t, rounds),
            best_us(&x1b, &weights, t, rounds),
        ];
        for (tot, u) in totals.iter_mut().zip(us) {
            *tot += u;
        }
        println!(
            "{:<14} {:>6} {:>6} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>10.2e}",
            name,
            out,
            inp,
            us[0],
            us[1],
            us[2],
            us[3],
            d1.max(d2)
        );
    }
    println!(
        "{:<14} {:>6} {:>6} {:>12.1} {:>12.1} {:>12.1} {:>12.1}",
        "sum", "", "", totals[0], totals[1], totals[2], totals[3]
    );
}
