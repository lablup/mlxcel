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

//! Op-level microbench for the two fused decode kernels from issue #905.
//!
//! Issue #905 lands both kernels under a measure-then-keep policy: each fusion
//! keeps its wiring only if it beats the unfused graph at op level on the
//! adopting backend and does not lose on the other. This binary is what
//! produces that evidence. It loads no model and depends on no checkpoint.
//!
//! Two comparisons, each fused against the exact graph it replaced:
//!
//! * **add + RMSNorm.** Unfused is `add(residual, delta)` then
//!   `fast_rms_norm(sum, weight, eps)`, which is what every pre-norm block
//!   built before #905. Fused is the two-output
//!   `(normed, new_residual)` kernel. Swept over hidden sizes and batch.
//! * **q/k RoPE + KV-append layout.** Unfused is
//!   `slice_last_dim` x3 -> `reshape` -> `transpose` -> `fast_rope` x2, plus a
//!   `contiguous` on V, which is the Llama3 attention fallback op for op.
//!   Fused is the three-output kernel that reads the whole projection output
//!   and emits q/k/v already in their consumers' layouts.
//!
//! # Reading the output
//!
//! Per-call microseconds and the fused/unfused speedup, one row per config, plus
//! a CSV block for pasting into `docs/benchmark_results/`. A speedup below 1.0
//! is a loss and is the signal to flip `FUSED_ADD_RMSNORM_DEFAULT` or
//! `FUSED_ROPE_APPEND_DEFAULT` in `src/lib/mlxcel-core/src/layers.rs`, or to
//! leave the corresponding kernel unwired.
//!
//! The measurement is deliberately launch-bound rather than bandwidth-bound at
//! batch 1: that is what the decode step is. Do not read a small absolute delta
//! as "no effect"; a per-layer saving multiplies by the layer count and by every
//! token.
//!
//! # Caveat on what this does *not* measure
//!
//! Both fusions also remove one full-width intermediate from the MLX graph per
//! call, which shows up as allocator and dependency-tracking pressure rather
//! than as kernel time. The op-level number is therefore a lower bound on the
//! end-to-end effect, and the end-to-end decode sweep is the deciding number.
//!
//! Reproduce (use `caffeinate -i` so the host does not idle-throttle the GPU
//! mid-run, and let the machine run cool between sweeps):
//!
//! ```text
//! caffeinate -i cargo run --release --features metal,accelerate \
//!     --example fused_norm_rope_microbench
//!
//! # CUDA
//! caffeinate -i cargo run --release --features cuda \
//!     --example fused_norm_rope_microbench
//! ```

use std::time::{Duration, Instant};

use mlxcel_core::{
    MlxArray, UniquePtr, add, astype, contiguous, dtype, eval, fast_rms_norm, fast_rope,
    from_slice_f32, fused_add_rms_norm, fused_rope_qk_append, random_normal, random_seed, reshape,
    slice_last_dim, synchronize_default, transpose_axes,
};

/// Hidden sizes from the issue's sweep: 2048 (small dense), 4096 (8B-class),
/// 8192 (70B-class).
const HIDDEN_SIZES: &[i32] = &[2048, 4096, 8192];

/// Batch sizes from the issue's sweep. Batch 1 is the launch-bound decode case
/// the fusion is aimed at; 4 and 8 start to expose bandwidth instead.
const BATCHES: &[i32] = &[1, 4, 8];

const EPS: f32 = 1e-5;
const ROPE_BASE: f32 = 500000.0;

const WARMUP: usize = 32;
const ITERS: usize = 256;

/// Time `iters` calls of `body`, evaluating each result so the lazy graph is
/// actually executed, with `synchronize_default()` bracketing the timed region
/// so queued GPU work is not attributed to the next config.
fn time_body<F>(warmup: usize, iters: usize, mut body: F) -> Duration
where
    F: FnMut() -> UniquePtr<MlxArray>,
{
    for _ in 0..warmup {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    let start = Instant::now();
    for _ in 0..iters {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    start.elapsed()
}

fn per_call_us(total: Duration, iters: usize) -> f64 {
    total.as_secs_f64() * 1e6 / iters as f64
}

fn randn(shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    // SAFETY: the null pointer is the documented "no explicit key" sentinel for
    // `random_normal`; the returned array owns its data.
    let raw = unsafe { random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
    let out = astype(&raw, dt);
    eval(&out);
    out
}

/// Norm weights sit near 1 in a real checkpoint, which matters here only in that
/// it keeps the timed values in a normal exponent range.
fn norm_weight(dim: i32, dt: i32) -> UniquePtr<MlxArray> {
    let ones: Vec<f32> = (0..dim).map(|i| 1.0 + 0.01 * ((i % 7) as f32)).collect();
    let w = from_slice_f32(&ones, &[dim]);
    let w = astype(&w, dt);
    eval(&w);
    w
}

struct Row {
    label: &'static str,
    hidden: i32,
    batch: i32,
    fused_us: f64,
    unfused_us: f64,
}

impl Row {
    fn speedup(&self) -> f64 {
        if self.fused_us <= 0.0 {
            0.0
        } else {
            self.unfused_us / self.fused_us
        }
    }
}

fn bench_add_rms_norm(rows: &mut Vec<Row>, dt: i32) {
    for &hidden in HIDDEN_SIZES {
        for &batch in BATCHES {
            let delta = randn(&[batch, 1, hidden], dt);
            let residual = randn(&[batch, 1, hidden], dt);
            let weight = norm_weight(hidden, dt);

            let unfused = time_body(WARMUP, ITERS, || {
                let sum = add(&residual, &delta);
                let normed = fast_rms_norm(&sum, &weight, EPS);
                // Keep both results live so the graph matches what a block
                // needs: the residual is carried to the next join.
                add(&normed, &sum)
            });

            let fused = time_body(WARMUP, ITERS, || {
                let mut normed = UniquePtr::null();
                let mut new_residual = UniquePtr::null();
                fused_add_rms_norm(
                    &delta,
                    &residual,
                    &weight,
                    EPS,
                    0.0,
                    &mut normed,
                    &mut new_residual,
                );
                add(&normed, &new_residual)
            });

            rows.push(Row {
                label: "add_rmsnorm",
                hidden,
                batch,
                fused_us: per_call_us(fused, ITERS),
                unfused_us: per_call_us(unfused, ITERS),
            });
        }
    }
}

/// Head geometry derived from the hidden size the way the dense families do:
/// head_dim 128, GQA ratio 4.
fn head_geometry(hidden: i32) -> (i32, i32, i32) {
    let head_dim = 128;
    let n_heads = hidden / head_dim;
    let n_kv_heads = (n_heads / 4).max(1);
    (n_heads, n_kv_heads, head_dim)
}

fn bench_rope_append(rows: &mut Vec<Row>, dt: i32) {
    for &hidden in HIDDEN_SIZES {
        let (n_heads, n_kv_heads, head_dim) = head_geometry(hidden);
        let cols = (n_heads + 2 * n_kv_heads) * head_dim;
        for &batch in BATCHES {
            let qkv = randn(&[batch, 1, cols], dt);
            let q_size = n_heads * head_dim;
            let kv_size = n_kv_heads * head_dim;
            let offset = 1024;

            // Exactly the Llama3 attention fallback: three trailing-axis slices,
            // three reshape+transpose pairs, two ropes. V needs an explicit
            // `contiguous` because the fused kernel emits it contiguous and the
            // comparison would otherwise be unfair to the unfused path's
            // consumer, which pays that copy at the cache append.
            let unfused = time_body(WARMUP, ITERS, || {
                let q = slice_last_dim(&qkv, 0, q_size);
                let k = slice_last_dim(&qkv, q_size, q_size + kv_size);
                let v = slice_last_dim(&qkv, q_size + kv_size, q_size + 2 * kv_size);
                let q = transpose_axes(&reshape(&q, &[batch, 1, n_heads, head_dim]), &[0, 2, 1, 3]);
                let k = transpose_axes(
                    &reshape(&k, &[batch, 1, n_kv_heads, head_dim]),
                    &[0, 2, 1, 3],
                );
                let v = transpose_axes(
                    &reshape(&v, &[batch, 1, n_kv_heads, head_dim]),
                    &[0, 2, 1, 3],
                );
                let q = fast_rope(&q, head_dim, false, ROPE_BASE, 1.0, offset);
                let k = fast_rope(&k, head_dim, false, ROPE_BASE, 1.0, offset);
                let v = contiguous(&v, false);
                // Group q by its KV group so the consume-the-outputs add stays
                // valid under GQA: q reshapes to [B, n_kv, ratio, D], which
                // broadcasts against k/v's [B, n_kv, 1, D]. Reshaping q to
                // [B, n_heads, 1, D] instead cannot broadcast once
                // n_heads != n_kv_heads, which is every configuration here.
                add(
                    &add(&k, &v),
                    &reshape(&q, &[batch, n_kv_heads, n_heads / n_kv_heads, head_dim]),
                )
            });

            let fused = time_body(WARMUP, ITERS, || {
                let mut q = UniquePtr::null();
                let mut k = UniquePtr::null();
                let mut v = UniquePtr::null();
                fused_rope_qk_append(
                    &qkv, n_heads, n_kv_heads, head_dim, head_dim, ROPE_BASE, 1.0, false, offset,
                    0, &mut q, &mut k, &mut v,
                );
                // Group q by its KV group so the consume-the-outputs add stays
                // valid under GQA: q reshapes to [B, n_kv, ratio, D], which
                // broadcasts against k/v's [B, n_kv, 1, D]. Reshaping q to
                // [B, n_heads, 1, D] instead cannot broadcast once
                // n_heads != n_kv_heads, which is every configuration here.
                add(
                    &add(&k, &v),
                    &reshape(&q, &[batch, n_kv_heads, n_heads / n_kv_heads, head_dim]),
                )
            });

            rows.push(Row {
                label: "rope_append",
                hidden,
                batch,
                fused_us: per_call_us(fused, ITERS),
                unfused_us: per_call_us(unfused, ITERS),
            });
        }
    }
}

fn backend_name() -> &'static str {
    if mlxcel_core::metal_is_available() {
        "metal"
    } else if mlxcel_core::cuda_is_available() {
        "cuda"
    } else {
        "cpu"
    }
}

fn main() {
    random_seed(905);

    let backend = backend_name();
    if backend == "cpu" {
        eprintln!(
            "fused_norm_rope_microbench: no GPU backend available; both kernels JIT through \
             fast::metal_kernel / cuda_kernel and cannot run here."
        );
        return;
    }

    // f16 is what `load_and_sanitize_weights` leaves non-quantized activations
    // in on Apple Silicon, so it is the dtype the decode loop actually feeds
    // these kernels.
    let dt = dtype::FLOAT16;
    let dt_name = "float16";

    println!("# fused decode kernel microbench (issue #905)");
    println!("# backend={backend} dtype={dt_name} warmup={WARMUP} iters={ITERS}");
    println!(
        "# a speedup below 1.00 means the fusion loses at op level and its default should be \
         flipped in layers.rs"
    );
    println!();

    let mut rows = Vec::new();
    bench_add_rms_norm(&mut rows, dt);
    bench_rope_append(&mut rows, dt);

    println!(
        "{:<14} {:>7} {:>6} {:>12} {:>12} {:>9}",
        "op", "hidden", "batch", "fused_us", "unfused_us", "speedup"
    );
    for r in &rows {
        println!(
            "{:<14} {:>7} {:>6} {:>12.3} {:>12.3} {:>8.2}x",
            r.label,
            r.hidden,
            r.batch,
            r.fused_us,
            r.unfused_us,
            r.speedup()
        );
    }

    println!();
    println!("# csv");
    println!("backend,dtype,op,hidden,batch,fused_us,unfused_us,speedup");
    for r in &rows {
        println!(
            "{backend},{dt_name},{},{},{},{:.4},{:.4},{:.4}",
            r.label,
            r.hidden,
            r.batch,
            r.fused_us,
            r.unfused_us,
            r.speedup()
        );
    }
}
