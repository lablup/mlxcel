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

// Layer-shaped workload microbench.
//
// Motivated by `bridge_overhead_microbench.rs` showing per-op cost at or
// below Python nanobind, yet real decode runs 2× slower. The remaining
// gap candidates are graph-shape effects: op count per layer, fusion
// window, sync point placement. This bench runs a synthetic workload
// that mimics a transformer decoder block without loading a model.
//
// Workload per "layer":
//     h  = rms_norm(x, w_norm1, eps)
//     h  = matmul(h, W1)              // [1,1,H] @ [H,H] -> [1,1,H]
//     h  = rms_norm(h, w_norm2, eps)
//     h  = matmul(h, W2)
//     x  = add(x, h)                  // residual
//
// One "step" is `n_layers` iterations of the block. We time:
//   * no-sync: one eval(x) at the very end of the step
//   * per-layer-sync: eval(x) after every layer (pessimistic, breaks
//     graph fusion the same way explicit sync points in real model code
//     would)
//
// If the gap hides in graph-fusion window length, we expect the
// per-layer-sync mode to be closer to the no-sync mode on one runtime
// and much slower on the other. A matching Python script
// `scripts/layer_shaped_microbench_py.py` drives the same shapes.
//
// Usage: cargo run --release --example layer_shaped_microbench
//          [ HIDDEN ] [ N_LAYERS ] [ N_STEPS ]
// Defaults: HIDDEN=2816 (Gemma 4 26B-a4b), N_LAYERS=30, N_STEPS=50.

use mlxcel_core::{add, eval, fast_rms_norm, from_slice_f32, matmul, synchronize_default};
use std::time::Instant;

fn init_weight(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let total: i32 = shape.iter().product();
    // Use a smoothly-varying pattern; any non-NaN pattern works.
    let data: Vec<f32> = (0..total as usize)
        .map(|i| ((i as f32) * 0.0003).sin() * 0.01)
        .collect();
    from_slice_f32(&data, shape)
}

fn run_layers_no_sync(
    x0: &mlxcel_core::MlxArray,
    w_norm1: &mlxcel_core::MlxArray,
    w_norm2: &mlxcel_core::MlxArray,
    w1: &mlxcel_core::MlxArray,
    w2: &mlxcel_core::MlxArray,
    n_layers: usize,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut x = mlxcel_core::copy(x0);
    for _ in 0..n_layers {
        let h = fast_rms_norm(&x, w_norm1, 1e-6);
        let h = matmul(&h, w1);
        let h = fast_rms_norm(&h, w_norm2, 1e-6);
        let h = matmul(&h, w2);
        x = add(&x, &h);
    }
    x
}

fn run_layers_per_layer_sync(
    x0: &mlxcel_core::MlxArray,
    w_norm1: &mlxcel_core::MlxArray,
    w_norm2: &mlxcel_core::MlxArray,
    w1: &mlxcel_core::MlxArray,
    w2: &mlxcel_core::MlxArray,
    n_layers: usize,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut x = mlxcel_core::copy(x0);
    for _ in 0..n_layers {
        let h = fast_rms_norm(&x, w_norm1, 1e-6);
        let h = matmul(&h, w1);
        let h = fast_rms_norm(&h, w_norm2, 1e-6);
        let h = matmul(&h, w2);
        x = add(&x, &h);
        eval(&x);
    }
    x
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let hidden: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2816);
    let n_layers: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let n_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    println!(
        "=== mlxcel layer-shaped microbench === HIDDEN={} N_LAYERS={} N_STEPS={}",
        hidden, n_layers, n_steps
    );
    println!();

    let x0 = init_weight(&[1, 1, hidden]);
    let w_norm1 = init_weight(&[hidden]);
    let w_norm2 = init_weight(&[hidden]);
    let w1 = init_weight(&[hidden, hidden]);
    let w2 = init_weight(&[hidden, hidden]);
    eval(&x0);
    eval(&w_norm1);
    eval(&w_norm2);
    eval(&w1);
    eval(&w2);
    synchronize_default();

    // -------- warmup --------
    for _ in 0..3 {
        let y = run_layers_no_sync(&x0, &w_norm1, &w_norm2, &w1, &w2, n_layers);
        eval(&y);
    }
    synchronize_default();

    // -------- no-sync: one eval at the end of each step --------
    let start = Instant::now();
    for _ in 0..n_steps {
        let y = run_layers_no_sync(&x0, &w_norm1, &w_norm2, &w1, &w2, n_layers);
        eval(&y);
    }
    synchronize_default();
    let t_no_sync = start.elapsed();
    println!(
        "no_sync          total={:>8.2}ms  per_step={:>8.3}ms  ({} steps × {} layers)",
        t_no_sync.as_secs_f64() * 1000.0,
        t_no_sync.as_secs_f64() * 1000.0 / n_steps as f64,
        n_steps,
        n_layers
    );

    // -------- per-layer-sync: eval after every layer --------
    for _ in 0..3 {
        let _ = run_layers_per_layer_sync(&x0, &w_norm1, &w_norm2, &w1, &w2, n_layers);
    }
    synchronize_default();

    let start = Instant::now();
    for _ in 0..n_steps {
        let _ = run_layers_per_layer_sync(&x0, &w_norm1, &w_norm2, &w1, &w2, n_layers);
    }
    synchronize_default();
    let t_per_layer = start.elapsed();
    println!(
        "per_layer_sync   total={:>8.2}ms  per_step={:>8.3}ms  ({} steps × {} layers)",
        t_per_layer.as_secs_f64() * 1000.0,
        t_per_layer.as_secs_f64() * 1000.0 / n_steps as f64,
        n_steps,
        n_layers
    );

    let overhead_ratio = t_per_layer.as_secs_f64() / t_no_sync.as_secs_f64();
    println!();
    println!(
        "per_layer_sync / no_sync = {:.2}× — bigger ratio means graph fusion gives more here",
        overhead_ratio
    );
}
