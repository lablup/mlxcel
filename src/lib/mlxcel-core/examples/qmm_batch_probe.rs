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

//! Few-row quantized matmul probe for batched decode.
//!
//! Batched decode multiplies `B` activation rows (one token each) by a shared
//! quantized weight. This probe times an affine `quantized_matmul` at the
//! Llama 3.1 8B projection shapes for `B` in 1..=8 and prints the time
//! relative to `B = 1`, so a kernel whose cost grows with every row shows up
//! directly. For each `B > 1` it also reports the largest absolute error of
//! the `B`-row product and of the same rows computed one at a time, both
//! against an f32 product with the dequantized weight, and whether the two are
//! bitwise equal (the contract speculative verify relies on).
//!
//! ```bash
//! cargo run --release -p mlxcel-core --example qmm_batch_probe --features metal,accelerate
//! # Upstream route on M1 and M2 class GPUs, for comparison:
//! MLXCEL_QMV_WIDE=0 cargo run --release -p mlxcel-core --example qmm_batch_probe --features metal,accelerate
//! ```
//!
//! `PROBE_DTYPE=f16` switches activations and scales from bf16 to f16, and
//! `PROBE_BITS=8` quantizes to 8 bits instead of 4.

use std::time::Instant;

use mlxcel_core::{self, MlxArray, UniquePtr, dtype};

const SHAPES: &[(&str, i32, i32)] = &[
    ("qkv", 6144, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up", 28672, 4096),
    ("down", 4096, 14336),
    ("lm_head", 128256, 4096),
];

const BATCHES: &[i32] = &[1, 2, 3, 4, 5, 6, 7, 8];

struct Weight {
    bits: i32,
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
}

impl Weight {
    fn qmm(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        unsafe {
            mlxcel_core::quantized_matmul(
                x,
                &self.w,
                &self.scales,
                &*self.biases as *const _,
                true,
                64,
                self.bits,
                "affine",
            )
        }
    }
}

fn time_ms(weight: &Weight, x: &MlxArray, iters: usize) -> f64 {
    for _ in 0..5 {
        mlxcel_core::eval(&weight.qmm(x));
    }
    // Several launches per eval so launch latency does not dominate.
    const PER_EVAL: usize = 8;
    let start = Instant::now();
    for _ in 0..iters {
        let outs: Vec<_> = (0..PER_EVAL).map(|_| weight.qmm(x)).collect();
        let refs: Vec<&MlxArray> = outs.iter().map(|o| &**o).collect();
        mlxcel_core::eval(&mlxcel_core::concatenate_many(&refs, 0));
    }
    start.elapsed().as_secs_f64() * 1e3 / (iters * PER_EVAL) as f64
}

struct RowCheck {
    batched_err: f32,
    single_err: f32,
    bitwise: bool,
}

/// Compare the `[B, K]` product, the `B` single-row products, and an f32
/// product with the dequantized weight.
fn check_rows(weight: &Weight, x: &MlxArray, batch: i32, k: i32) -> RowCheck {
    let together = weight.qmm(x);
    let singles: Vec<_> = (0..batch)
        .map(|row| weight.qmm(&mlxcel_core::slice(x, &[row, 0], &[row + 1, k])))
        .collect();
    let refs: Vec<&MlxArray> = singles.iter().map(|o| &**o).collect();
    let alone = mlxcel_core::concatenate_many(&refs, 0);

    let dense = unsafe {
        mlxcel_core::dequantize(
            &weight.w,
            &weight.scales,
            &*weight.biases as *const _,
            64,
            weight.bits,
            "affine",
        )
    };
    let dense = mlxcel_core::astype(&dense, dtype::FLOAT32);
    let reference = mlxcel_core::matmul(
        &mlxcel_core::astype(x, dtype::FLOAT32),
        &mlxcel_core::transpose_axes(&dense, &[1, 0]),
    );
    let max_err = |y: &MlxArray| {
        let diff = mlxcel_core::subtract(&mlxcel_core::astype(y, dtype::FLOAT32), &reference);
        let m = mlxcel_core::max_all(&mlxcel_core::abs(&diff));
        mlxcel_core::eval(&m);
        mlxcel_core::item_f32(&m)
    };
    let equal = mlxcel_core::array_equal(&together, &alone, false);
    mlxcel_core::eval(&equal);
    RowCheck {
        batched_err: max_err(&together),
        single_err: max_err(&alone),
        bitwise: mlxcel_core::item_bool(&equal),
    }
}

fn main() {
    let dt = match std::env::var("PROBE_DTYPE").as_deref() {
        Ok("f16") => dtype::FLOAT16,
        _ => dtype::BFLOAT16,
    };
    let bits: i32 = std::env::var("PROBE_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    println!(
        "bits={bits} dtype={} MLXCEL_QMV_WIDE={}",
        if dt == dtype::FLOAT16 { "f16" } else { "bf16" },
        std::env::var("MLXCEL_QMV_WIDE").unwrap_or_else(|_| "unset".into())
    );
    println!(
        "{:<8} {:>3} {:>10} {:>8} {:>12} {:>12} {:>8}",
        "shape", "B", "ms", "vs B=1", "err batched", "err single", "bitwise"
    );
    for &(name, n, k) in SHAPES {
        let dense = unsafe { mlxcel_core::random_normal(&[n, k], dt, std::ptr::null()) };
        let q = mlxcel_core::quantize_weights(&dense, 64, bits);
        let weight = Weight {
            bits,
            w: mlxcel_core::quantized_weights_w(&q),
            scales: mlxcel_core::quantized_weights_scales(&q),
            biases: mlxcel_core::quantized_weights_biases(&q),
        };
        mlxcel_core::eval(&weight.w);
        mlxcel_core::eval(&weight.scales);
        mlxcel_core::eval(&weight.biases);
        let mut base = 0.0;
        for &batch in BATCHES {
            let x = unsafe { mlxcel_core::random_normal(&[batch, k], dt, std::ptr::null()) };
            mlxcel_core::eval(&x);
            let ms = time_ms(&weight, &x, 20);
            if batch == 1 {
                base = ms;
            }
            if batch == 1 {
                println!("{name:<8} {batch:>3} {ms:>10.3} {:>7.2}x", 1.0);
                continue;
            }
            let check = check_rows(&weight, &x, batch, k);
            println!(
                "{name:<8} {batch:>3} {ms:>10.3} {:>7.2}x {:>12.4} {:>12.4} {:>8}",
                ms / base,
                check.batched_err,
                check.single_err,
                check.bitwise
            );
        }
    }
}
