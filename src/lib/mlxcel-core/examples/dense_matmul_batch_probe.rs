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

//! Few-row dense matmul probe for batched decode.
//!
//! Times `x @ w.T` at decode projection shapes for `B` rows and prints the
//! time relative to `B = 1`, plus whether the `B`-row product is bitwise equal
//! to the same rows computed one at a time.
//!
//! ```bash
//! cargo run --release -p mlxcel-core --example dense_matmul_batch_probe --features metal,accelerate
//! # Upstream route, for comparison:
//! MLXCEL_DENSE_GEMV_ROWS=0 cargo run --release -p mlxcel-core --example dense_matmul_batch_probe --features metal,accelerate
//! ```

use std::time::Instant;

use mlxcel_core::{self, MlxArray, dtype};

const SHAPES: &[(&str, i32, i32)] = &[
    ("qkv", 6144, 2048),
    ("o_proj", 2048, 2048),
    ("up", 8192, 2048),
    ("down", 2048, 8192),
    ("lm_head", 50304, 2048),
    ("lm_large", 126464, 2048),
];

const BATCHES: &[i32] = &[1, 2, 3, 4, 5, 6, 8];

fn time_ms(wt: &MlxArray, x: &MlxArray) -> f64 {
    for _ in 0..5 {
        mlxcel_core::eval(&mlxcel_core::matmul(x, wt));
    }
    // Several launches per eval so launch latency does not dominate.
    const PER_EVAL: usize = 8;
    const ITERS: usize = 20;
    let start = Instant::now();
    for _ in 0..ITERS {
        let outs: Vec<_> = (0..PER_EVAL).map(|_| mlxcel_core::matmul(x, wt)).collect();
        let refs: Vec<&MlxArray> = outs.iter().map(|o| &**o).collect();
        mlxcel_core::eval(&mlxcel_core::concatenate_many(&refs, 0));
    }
    start.elapsed().as_secs_f64() * 1e3 / (ITERS * PER_EVAL) as f64
}

fn bitwise_equal_to_single_rows(wt: &MlxArray, x: &MlxArray, batch: i32, k: i32) -> bool {
    let together = mlxcel_core::matmul(x, wt);
    let singles: Vec<_> = (0..batch)
        .map(|row| mlxcel_core::matmul(&mlxcel_core::slice(x, &[row, 0], &[row + 1, k]), wt))
        .collect();
    let refs: Vec<&MlxArray> = singles.iter().map(|o| &**o).collect();
    let equal =
        mlxcel_core::array_equal(&together, &mlxcel_core::concatenate_many(&refs, 0), false);
    mlxcel_core::eval(&equal);
    mlxcel_core::item_bool(&equal)
}

fn main() {
    println!(
        "MLXCEL_DENSE_GEMV_ROWS={}",
        std::env::var("MLXCEL_DENSE_GEMV_ROWS").unwrap_or_else(|_| "unset".into())
    );
    for (dt_name, dt) in [("f16", dtype::FLOAT16), ("bf16", dtype::BFLOAT16)] {
        for &(name, n, k) in SHAPES {
            let w = unsafe { mlxcel_core::random_normal(&[n, k], dt, std::ptr::null()) };
            mlxcel_core::eval(&w);
            let wt = mlxcel_core::transpose(&w);
            let mut base = 0.0;
            for &batch in BATCHES {
                let x = unsafe { mlxcel_core::random_normal(&[batch, k], dt, std::ptr::null()) };
                mlxcel_core::eval(&x);
                let ms = time_ms(&wt, &x);
                if batch == 1 {
                    base = ms;
                }
                let bitwise = batch == 1 || bitwise_equal_to_single_rows(&wt, &x, batch, k);
                println!(
                    "{dt_name:<5} {name:<8} {batch:>2} {ms:>8.3} ms {:>6.2}x bitwise={bitwise}",
                    ms / base
                );
            }
        }
    }
}
