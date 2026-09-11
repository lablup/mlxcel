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

// Bridge-overhead microbench.
//
// Isolates the per-op cost of the Rust → cxx → C++ → MLX dispatch path for
// a handful of representative operations and compares against the Python
// reference measurement produced by
// `scripts/bridge_overhead_microbench_py.py`.
//
// Two modes per op:
//
// * `no_eval`: build N graph nodes back-to-back with no `mx.eval()` in the
//   loop. MLX stays lazy, so this mostly measures FFI + unique_ptr alloc +
//   MLX primitive construction. Matches what the Python script does with
//   the same workload.
// * `eval`: evaluate every intermediate. This folds the actual Metal kernel
//   launch + completion into the measurement. Both runtimes pay the same
//   kernel cost, so the remaining gap is still overhead.
//
// Usage:
//   cargo run --release --example bridge_overhead_microbench

use mlxcel_core::{
    add, eval, expand_dims, from_slice_f32, from_slice_i32, matmul, multiply, reshape,
    synchronize_default,
};
use std::time::{Duration, Instant};

const WARMUP: usize = 200;
const ITERS: usize = 10_000;

fn fmt_per_call(label: &str, total: Duration, iters: usize) {
    let per = total.as_nanos() as f64 / iters as f64;
    println!(
        "  {:<26} total={:>8.2}ms  per_call={:>7.2}us  ({} iters)",
        label,
        total.as_secs_f64() * 1000.0,
        per / 1000.0,
        iters
    );
}

fn bench_add(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) {
    println!("add(shape=[4]):");

    // no_eval
    for _ in 0..WARMUP {
        let _ = add(a, b);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = add(a, b);
    }
    let t = start.elapsed();
    fmt_per_call("no_eval", t, ITERS);

    // eval each
    for _ in 0..WARMUP {
        let c = add(a, b);
        eval(&c);
    }
    synchronize_default();
    let start = Instant::now();
    for _ in 0..ITERS {
        let c = add(a, b);
        eval(&c);
    }
    synchronize_default();
    let t = start.elapsed();
    fmt_per_call("eval", t, ITERS);
}

fn bench_multiply(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) {
    println!("multiply(shape=[4]):");

    for _ in 0..WARMUP {
        let _ = multiply(a, b);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = multiply(a, b);
    }
    let t = start.elapsed();
    fmt_per_call("no_eval", t, ITERS);
}

fn bench_reshape(a: &mlxcel_core::MlxArray) {
    println!("reshape(shape=[4]->[2,2]):");
    let target = [2i32, 2];

    for _ in 0..WARMUP {
        let _ = reshape(a, &target);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = reshape(a, &target);
    }
    let t = start.elapsed();
    fmt_per_call("no_eval", t, ITERS);
}

fn bench_expand_dims(a: &mlxcel_core::MlxArray) {
    println!("expand_dims(axis=0):");

    for _ in 0..WARMUP {
        let _ = expand_dims(a, 0);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = expand_dims(a, 0);
    }
    let t = start.elapsed();
    fmt_per_call("no_eval", t, ITERS);
}

fn bench_matmul_512() {
    println!("matmul(512x512 @ 512x512):");
    let rows = 512i32;
    let cols = 512i32;
    let total = (rows * cols) as usize;
    let data_a: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
    let data_b: Vec<f32> = (0..total).map(|i| (i as f32) * 0.002).collect();
    let a = from_slice_f32(&data_a, &[rows, cols]);
    let b = from_slice_f32(&data_b, &[rows, cols]);
    eval(&a);
    eval(&b);
    synchronize_default();

    // no_eval
    for _ in 0..WARMUP {
        let _ = matmul(&a, &b);
    }
    let start = Instant::now();
    for _ in 0..ITERS / 10 {
        let _ = matmul(&a, &b);
    }
    let t = start.elapsed();
    fmt_per_call("no_eval", t, ITERS / 10);

    // eval
    for _ in 0..WARMUP / 10 {
        let c = matmul(&a, &b);
        eval(&c);
    }
    synchronize_default();
    let start = Instant::now();
    for _ in 0..ITERS / 10 {
        let c = matmul(&a, &b);
        eval(&c);
    }
    synchronize_default();
    let t = start.elapsed();
    fmt_per_call("eval", t, ITERS / 10);
}

fn bench_from_slice_i32() {
    println!("from_slice_i32(len=1):");
    let data = [42i32];
    let shape = [1i32];

    for _ in 0..WARMUP {
        let _ = from_slice_i32(&data, &shape);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = from_slice_i32(&data, &shape);
    }
    let t = start.elapsed();
    fmt_per_call("no_eval", t, ITERS);
}

fn main() {
    println!("=== mlxcel bridge microbench ===");
    println!("warmup={}, iters={}", WARMUP, ITERS);
    println!();

    // Small-shape inputs reused across the elementwise ops. These are the
    // ones where the actual kernel cost is minimal, so any wall-time over
    // "free" should be FFI / unique_ptr / graph construction.
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let shape = [4i32];
    let a = from_slice_f32(&data, &shape);
    let b = from_slice_f32(&data, &shape);
    eval(&a);
    eval(&b);
    synchronize_default();

    bench_add(&a, &b);
    println!();
    bench_multiply(&a, &b);
    println!();
    bench_reshape(&a);
    println!();
    bench_expand_dims(&a);
    println!();
    bench_from_slice_i32();
    println!();
    bench_matmul_512();
}
