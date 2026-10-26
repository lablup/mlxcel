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

// Kernel-level A/B for the native sliding-window SDPA route (issue #2621).
//
// Three arms over the same q/k/v, in interleaved order per sample:
//
// - `legacy_build`: the pre-#2621 over-window prefill path, the full-width
//   windowed-causal mask built per call plus the array-mask SDPA
//   (`window_sdpa::legacy_causal_window_attention`).
// - `legacy_steady`: the same SDPA with the mask built once and reused, so
//   the mask construction can be separated from the attention itself.
// - `native`: upstream MLX's `window_size` with `mask_mode = "causal"`
//   (`window_sdpa::native_causal_window_attention`), unconditionally.
//
// Query lengths 1, 4 and 8 run the production route for them
// (`causal_attention`, which keeps the decode/verify paths) against the
// forced native entry, to see whether the shapes the route does not migrate
// would move if it did. Every row reports MLX's fused-kernel verdict for both
// arms (`window_sdpa::{native_arm_is_fused, legacy_arm_is_fused}`), the
// per-call median over the samples with the min and max, and the peak memory
// of one arm's sample.
//
// Usage:
//   cargo build --release --features metal,accelerate --example window_sdpa_microbench
//   ./target/release/examples/window_sdpa_microbench [--quick] > bench.csv
//
// `--quick` limits the sweep to query lengths up to 2048.

use std::time::{Duration, Instant};

use mlxcel_core::window_sdpa::{
    WindowSdpaShape, legacy_arm_is_fused, legacy_causal_window_attention,
    legacy_causal_window_mask, native_arm_is_fused, native_causal_window_attention,
};
use mlxcel_core::{MlxArray, UniquePtr, astype, dtype, eval, from_slice_f32, synchronize_default};

const WARMUP: usize = 5;
const SAMPLES: usize = 5;
const SCALE: f32 = 0.0625;

#[derive(Clone, Copy)]
struct Geometry {
    name: &'static str,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
}

const GEOMETRIES: [Geometry; 4] = [
    Geometry {
        name: "gemma3-1b H4/KV1/D256",
        q_heads: 4,
        kv_heads: 1,
        head_dim: 256,
    },
    Geometry {
        name: "gemma4-e4b H8/KV2/D256",
        q_heads: 8,
        kv_heads: 2,
        head_dim: 256,
    },
    Geometry {
        name: "gemma4-31b H32/KV16/D256",
        q_heads: 32,
        kv_heads: 16,
        head_dim: 256,
    },
    Geometry {
        name: "upstream-ref H32/KV8/D128",
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
    },
];

fn pseudo_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
            u / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn tensor(seed: u64, shape: &[i32], dtype_code: i32) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|d| *d as usize).product();
    let arr = from_slice_f32(&pseudo_f32(seed, n), shape);
    let arr = astype(&arr, dtype_code);
    eval(&arr);
    arr
}

fn iters_for(q_len: i32) -> usize {
    match q_len {
        0..=16 => 50,
        17..=512 => 20,
        513..=2048 => 8,
        2049..=4096 => 4,
        _ => 2,
    }
}

/// Per-call wall time of `iters` calls, GPU work synchronized at both ends.
fn time_calls<F>(iters: usize, mut body: F) -> Duration
where
    F: FnMut() -> UniquePtr<MlxArray>,
{
    synchronize_default();
    let start = Instant::now();
    for _ in 0..iters {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    start.elapsed() / iters as u32
}

fn peak_bytes_of<F>(mut body: F) -> usize
where
    F: FnMut() -> UniquePtr<MlxArray>,
{
    synchronize_default();
    mlxcel_core::reset_peak_memory();
    let out = body();
    eval(&out);
    synchronize_default();
    mlxcel_core::get_peak_memory()
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn summarize(samples: &mut [Duration]) -> (f64, f64, f64) {
    samples.sort();
    let median = micros(samples[samples.len() / 2]);
    (
        median,
        micros(samples[0]),
        micros(samples[samples.len() - 1]),
    )
}

fn dtype_name(code: i32) -> &'static str {
    match code {
        dtype::FLOAT16 => "f16",
        dtype::BFLOAT16 => "bf16",
        dtype::FLOAT32 => "f32",
        _ => "other",
    }
}

#[allow(clippy::too_many_arguments)]
fn run_row(
    geometry: Geometry,
    kind: &str,
    q_len: i32,
    k_len: i32,
    window: i32,
    dtype_code: i32,
    seed: u64,
) {
    let q = tensor(
        seed,
        &[1, geometry.q_heads, q_len, geometry.head_dim],
        dtype_code,
    );
    let k = tensor(
        seed + 1,
        &[1, geometry.kv_heads, k_len, geometry.head_dim],
        dtype_code,
    );
    let v = tensor(
        seed + 2,
        &[1, geometry.kv_heads, k_len, geometry.head_dim],
        dtype_code,
    );
    let shape = WindowSdpaShape::of(&q, &k, &v);
    let native_fused = native_arm_is_fused(&shape, window);
    let legacy_fused = legacy_arm_is_fused(&shape);
    let iters = iters_for(q_len);
    let prebuilt = legacy_causal_window_mask(&q, &k, window);
    eval(&prebuilt);

    // The production entry for the shapes the route does not migrate.
    let production = |q: &MlxArray, k: &MlxArray, v: &MlxArray| {
        mlxcel_core::causal_attention(q, k, v, SCALE, 0.0, window)
    };

    let mut arms: Vec<(&str, Box<dyn FnMut() -> UniquePtr<MlxArray>>)> = if q_len > 8 {
        vec![
            (
                "legacy_build",
                Box::new(|| legacy_causal_window_attention(&q, &k, &v, SCALE, window)),
            ),
            (
                "legacy_steady",
                Box::new(|| {
                    mlxcel_core::layers::attention(
                        &q,
                        &k,
                        &v,
                        SCALE,
                        Some(prebuilt.as_ref().unwrap()),
                        0.0,
                        window,
                    )
                }),
            ),
            (
                "native",
                Box::new(|| native_causal_window_attention(&q, &k, &v, SCALE, window)),
            ),
        ]
    } else {
        vec![
            ("production", Box::new(|| production(&q, &k, &v))),
            (
                "native_forced",
                Box::new(|| native_causal_window_attention(&q, &k, &v, SCALE, window)),
            ),
        ]
    };

    for (_, arm) in arms.iter_mut() {
        for _ in 0..WARMUP {
            let out = arm();
            eval(&out);
        }
    }
    synchronize_default();

    let mut samples: Vec<Vec<Duration>> = vec![Vec::with_capacity(SAMPLES); arms.len()];
    for sample in 0..SAMPLES {
        let order: Vec<usize> = if sample % 2 == 0 {
            (0..arms.len()).collect()
        } else {
            (0..arms.len()).rev().collect()
        };
        for idx in order {
            let d = time_calls(iters, &mut arms[idx].1);
            samples[idx].push(d);
        }
    }
    let peaks: Vec<usize> = arms.iter_mut().map(|(_, arm)| peak_bytes_of(arm)).collect();

    for (idx, (name, _)) in arms.iter().enumerate() {
        let (median, min, max) = summarize(&mut samples[idx]);
        println!(
            "{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{},{},{}",
            geometry.name,
            kind,
            dtype_name(dtype_code),
            q_len,
            k_len,
            window,
            name,
            native_fused,
            legacy_fused,
            iters,
            median,
            min,
            max,
            SAMPLES,
            WARMUP,
            peaks[idx]
        );
    }
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    if !mlxcel_core::metal_is_available() {
        eprintln!("window_sdpa_microbench: the native window route is Metal-only; nothing to run");
        return;
    }
    println!(
        "geometry,kind,dtype,q_len,k_len,window,arm,native_fused,legacy_fused,iters,median_us,min_us,max_us,samples,warmup,peak_bytes"
    );
    let q_lens: &[i32] = if quick {
        &[512, 2048]
    } else {
        &[512, 2048, 4096, 8192]
    };
    let mut seed = 0x2621u64;
    for geometry in GEOMETRIES {
        let dtypes: &[i32] = if geometry.head_dim == 256 && geometry.q_heads == 4 {
            &[dtype::FLOAT16, dtype::BFLOAT16]
        } else {
            &[dtype::FLOAT16]
        };
        for &dtype_code in dtypes {
            for &q_len in q_lens {
                for window in [512, 1024] {
                    if window >= q_len {
                        // Keys within the window: plain causal, not a window
                        // route on either arm.
                        continue;
                    }
                    seed += 10;
                    run_row(geometry, "fresh", q_len, q_len, window, dtype_code, seed);
                    seed += 10;
                    run_row(
                        geometry,
                        "continuation",
                        q_len,
                        q_len + window - 1,
                        window,
                        dtype_code,
                        seed,
                    );
                }
            }
            // Decode and small verify blocks over a rolled-over window.
            for q_len in [1, 4, 8] {
                seed += 10;
                run_row(geometry, "small", q_len, 512 + q_len, 512, dtype_code, seed);
            }
        }
    }
}
