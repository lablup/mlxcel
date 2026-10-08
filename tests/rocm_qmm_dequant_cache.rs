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

//! bf16 GEMMs routed above the fused WMMA kernel's row ceiling skip the
//! dequantized-weight cache (issue #2151).
//!
//! On RDNA 3.5, `select_qmm_route` in the ROCm overlay sends a bf16 affine
//! GEMM of `MLX_ROCM_WMMA_QMM_MAX_M` rows or more (128 by default) to
//! dequantize + hipBLASLt. That route used to store every dequantized weight
//! in the process-wide LRU (8 matrices or 256 MB), which a forward pass cycles
//! without a hit while its last entries stay alive through decode. The route
//! now dequantizes into a temporary and leaves the cache alone; the f16 and
//! non-WMMA shapes that reach the same GEMM keep the cache.
//!
//! The cache and its counters are process-wide, and the knobs are read once
//! per process, so every case runs in a fresh child process (the parent
//! re-invokes this binary on an ignored child test, one child at a time). The
//! child runs one quantized GEMM shape twice on the GPU, compares each result
//! with `dequantize` + `matmul` in f32 on the CPU stream, and prints the
//! counters `mlxcel_core::rocm_qmm_cache::dequant_cache_stats` read after each
//! pass. `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=1` makes the dequantize route
//! eligible at every row count, so only the ceiling and the dtypes decide the
//! route. The bf16 above-ceiling case also checks that the output bytes equal
//! `dequantize` + `matmul` on the GPU, the GEMM the route claims to run.
//!
//! With the route change reverted (the ceiling branch of `select_qmm_route`
//! returning `DequantGemm` again, measured on gfx1151) the two bf16
//! above-ceiling cases fail on their first pass, which reads `misses=1
//! inserts=1 bypasses=0 entries=1` instead of one bypass and an empty cache;
//! the forced-WMMA and f16 cases still pass.
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --release --features rocm --test rocm_qmm_dequant_cache -- --test-threads=1
//! ```

#![cfg(feature = "rocm")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::rocm_qmm_cache::{DequantCacheStats, dequant_cache_stats};
use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{MlxArray, dtype};

const MAX_M: &str = "MLX_ROCM_WMMA_QMM_MAX_M";
const WMMA: &str = "MLX_ROCM_WMMA_QMM";
const THRESHOLD: &str = "MLX_ROCM_QMM_DEQUANT_M_THRESHOLD";
const CACHE_SIZE: &str = "MLX_ROCM_QMM_DEQUANT_CACHE_SIZE";
const CACHE_MAX_BYTES: &str = "MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES";
const STATS: &str = "MLX_ROCM_QMM_DEQUANT_CACHE_STATS";
/// Cleared in every child before a case applies its own, so the host
/// environment cannot leak in.
const ALL_VARS: &[&str] = &[MAX_M, WMMA, THRESHOLD, CACHE_SIZE, CACHE_MAX_BYTES, STATS];

const CHILD_ENV: &str = "MLXCEL_ROCM_QMM_DEQUANT_CACHE_CHILD";
const CASE_ENV: &str = "MLXCEL_ROCM_QMM_DEQUANT_CACHE_CASE";
const CHILD_TEST: &str = "child_dequant_cache_case";
const CHILD_BUDGET: Duration = Duration::from_secs(180);
const DONE_MARKER: &str = "DEQUANT_CACHE done";
const PASS_PREFIX: &str = "DEQUANT_CACHE pass=";

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
/// Relative error of a half-precision GEMM against the f32 reference.
const TOLERANCE: f32 = 2e-2;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// What the counters must read after each of the two passes.
#[derive(Clone, Copy)]
struct Expect {
    hits: u64,
    misses: u64,
    inserts: u64,
    bypasses: u64,
    entries: u64,
}

struct Case {
    name: &'static str,
    env: &'static [(&'static str, &'static str)],
    /// Activation and weight dtype.
    dtype: i32,
    rows: i32,
    k: i32,
    n: i32,
    /// Whether the GPU output must equal `dequantize` + `matmul` bytewise.
    dense_bytes: bool,
    after: [Expect; 2],
}

const NOTHING: Expect = Expect {
    hits: 0,
    misses: 0,
    inserts: 0,
    bypasses: 0,
    entries: 0,
};

fn cases() -> Vec<Case> {
    vec![
        // 256 rows is above the default ceiling of 128 on gfx1151: the GEMM
        // is dequantized per call and the cache stays empty.
        Case {
            name: "bf16_above_default_ceiling",
            env: &[(THRESHOLD, "1")],
            dtype: dtype::BFLOAT16,
            rows: 256,
            k: 4096,
            n: 4096,
            dense_bytes: true,
            after: [
                Expect {
                    bypasses: 1,
                    ..NOTHING
                },
                Expect {
                    bypasses: 2,
                    ..NOTHING
                },
            ],
        },
        // A ceiling set through the environment also produces the bypass.
        Case {
            name: "bf16_above_env_ceiling",
            env: &[(THRESHOLD, "1"), (MAX_M, "32")],
            dtype: dtype::BFLOAT16,
            rows: 64,
            k: 1024,
            n: 1024,
            dense_bytes: true,
            after: [
                Expect {
                    bypasses: 1,
                    ..NOTHING
                },
                Expect {
                    bypasses: 2,
                    ..NOTHING
                },
            ],
        },
        // MLX_ROCM_WMMA_QMM=1 removes the ceiling: the fused kernel runs, and
        // neither the cache nor the bypass is touched.
        Case {
            name: "bf16_forced_wmma",
            env: &[(THRESHOLD, "1"), (WMMA, "1")],
            dtype: dtype::BFLOAT16,
            rows: 256,
            k: 1024,
            n: 1024,
            dense_bytes: false,
            after: [NOTHING, NOTHING],
        },
        // f16 never takes the fused kernel, so it reaches the cached route:
        // the first pass misses and inserts, the second hits.
        Case {
            name: "f16_cached",
            env: &[(THRESHOLD, "1")],
            dtype: dtype::FLOAT16,
            rows: 128,
            k: 1024,
            n: 1024,
            dense_bytes: false,
            after: [
                Expect {
                    misses: 1,
                    inserts: 1,
                    entries: 1,
                    ..NOTHING
                },
                Expect {
                    hits: 1,
                    misses: 1,
                    inserts: 1,
                    entries: 1,
                    ..NOTHING
                },
            ],
        },
    ]
}

fn find_case(name: &str) -> Case {
    cases()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("unknown case {name}"))
}

fn pseudo_random(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, dtype::FLOAT32);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&f));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn format_stats(s: &DequantCacheStats) -> String {
    format!(
        "hits={} misses={} inserts={} evictions={} bypasses={} entries={} bytes={}",
        s.hits, s.misses, s.inserts, s.evictions, s.bypasses, s.entries, s.bytes
    )
}

/// Runs the case's GEMM twice on the GPU and checks results and counters.
fn run_case(case: &Case) {
    let (rows, k, n) = (case.rows, case.k, case.n);
    let w_f32 = mlxcel_core::from_slice_f32(&pseudo_random((n * k) as usize, 0x2151), &[n, k]);
    let w = mlxcel_core::astype(&w_f32, case.dtype);
    let q = mlxcel_core::quantize_weights_with_mode(&w, GROUP_SIZE, BITS, "affine");
    let packed = mlxcel_core::quantized_weights_w(&q);
    let scales = mlxcel_core::quantized_weights_scales(&q);
    let biases = mlxcel_core::quantized_weights_biases(&q);
    let x_f32 = mlxcel_core::from_slice_f32(
        &pseudo_random((rows * k) as usize, 0x2151 ^ rows as u32),
        &[rows, k],
    );
    let x = mlxcel_core::astype(&x_f32, case.dtype);

    let reference = {
        let _cpu = DefaultDeviceGuard::cpu();
        // SAFETY: `biases` is a live array owned by this frame.
        let w_deq = unsafe {
            mlxcel_core::dequantize(
                &packed,
                &scales,
                &*biases as *const MlxArray,
                GROUP_SIZE,
                BITS,
                "affine",
            )
        };
        let w_deq = mlxcel_core::astype(&w_deq, dtype::FLOAT32);
        let x_ref = mlxcel_core::astype(&x, dtype::FLOAT32);
        let out = mlxcel_core::matmul(&x_ref, &mlxcel_core::transpose(&w_deq));
        mlxcel_core::eval(&out);
        out
    };
    let scale = max_abs(&reference).max(1e-6);

    let dense = case.dense_bytes.then(|| {
        let _gpu = DefaultDeviceGuard::gpu();
        // SAFETY: `biases` is a live array owned by this frame.
        let w_deq = unsafe {
            mlxcel_core::dequantize(
                &packed,
                &scales,
                &*biases as *const MlxArray,
                GROUP_SIZE,
                BITS,
                "affine",
            )
        };
        let out = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&w_deq));
        mlxcel_core::eval(&out);
        mlxcel_core::array_to_raw_bytes(&out)
    });
    // `dequantize` + `matmul` never goes through QuantizedMatmul, so the
    // counters still read zero here.
    assert_eq!(
        dequant_cache_stats(),
        DequantCacheStats::default(),
        "{}: counters moved before the first quantized GEMM",
        case.name
    );

    for (pass, expect) in case.after.iter().enumerate() {
        let got = {
            let _gpu = DefaultDeviceGuard::gpu();
            // SAFETY: every reference is a live array owned by this frame.
            let out = unsafe {
                mlxcel_core::quantized_matmul(
                    &x,
                    &packed,
                    &scales,
                    &*biases as *const MlxArray,
                    true,
                    GROUP_SIZE,
                    BITS,
                    "affine",
                )
            };
            if let Err(err) = mlxcel_core::try_eval(&out) {
                panic!("{} pass {pass} on the GPU failed: {err}", case.name);
            }
            out
        };
        let stats = dequant_cache_stats();
        println!("{PASS_PREFIX}{pass} {}", format_stats(&stats));

        if let Some(dense) = &dense {
            assert!(
                mlxcel_core::array_to_raw_bytes(&got) == *dense,
                "{} pass {pass}: quantized_matmul differs from dequantize + matmul",
                case.name
            );
        }
        let got = mlxcel_core::astype(&got, dtype::FLOAT32);
        let rel = max_abs(&mlxcel_core::subtract(&got, &reference)) / scale;
        // Written so a NaN fails too.
        assert!(
            rel < TOLERANCE,
            "{} pass {pass}: GPU result is {rel:e} from the f32 reference",
            case.name
        );

        let expected = (
            expect.hits,
            expect.misses,
            expect.inserts,
            expect.bypasses,
            expect.entries,
        );
        let actual = (
            stats.hits,
            stats.misses,
            stats.inserts,
            stats.bypasses,
            stats.entries,
        );
        assert_eq!(
            actual,
            expected,
            "{} pass {pass}: (hits, misses, inserts, bypasses, entries) after the GEMM; {}",
            case.name,
            format_stats(&stats)
        );
        assert_eq!(stats.evictions, 0, "{} pass {pass}: evictions", case.name);
        if expect.entries == 0 {
            assert_eq!(stats.bytes, 0, "{} pass {pass}: cached bytes", case.name);
        } else {
            let weight_bytes = (n as u64) * (k as u64) * 2;
            assert_eq!(
                stats.bytes, weight_bytes,
                "{} pass {pass}: cached bytes",
                case.name
            );
        }
    }
}

struct ChildRun {
    success: bool,
    status: String,
    stdout: String,
    stderr: String,
}

fn run_child(case: &Case) -> ChildRun {
    let exe = std::env::current_exe().expect("path of this test binary");
    let mut cmd = Command::new(exe);
    cmd.args([
        CHILD_TEST,
        "--exact",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ])
    .env(CHILD_ENV, "1")
    .env(CASE_ENV, case.name)
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    for var in ALL_VARS {
        cmd.env_remove(var);
    }
    for (k, v) in case.env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn the child test process");

    // Drain both pipes on their own threads so the child never blocks on a
    // full pipe while the parent waits for it.
    let mut stdout_pipe = child.stdout.take().expect("child stdout");
    let mut stderr_pipe = child.stderr.take().expect("child stderr");
    let stdout_reader = thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout_pipe.read_to_string(&mut text);
        text
    });
    let stderr_reader = thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr_pipe.read_to_string(&mut text);
        text
    });

    let deadline = Instant::now() + CHILD_BUDGET;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("poll the child: {err}");
            }
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(50));
    };
    let stdout = stdout_reader.join().expect("stdout reader thread");
    let stderr = stderr_reader.join().expect("stderr reader thread");
    let status = status.unwrap_or_else(|| {
        panic!(
            "the child for {} did not finish within {CHILD_BUDGET:?}\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}",
            case.name
        )
    });
    ChildRun {
        success: status.success(),
        status: status.to_string(),
        stdout,
        stderr,
    }
}

fn check_child(case: &Case) -> Option<String> {
    let run = run_child(case);
    if run.success && run.stdout.contains(DONE_MARKER) {
        eprintln!("{}: ok", case.name);
        return None;
    }
    Some(format!(
        "{}: exited with {}\n--- child stdout ---\n{}\n--- child stderr ---\n{}",
        case.name, run.status, run.stdout, run.stderr
    ))
}

/// One test, so the children run one after another and never share the GPU.
#[test]
fn ceiling_routed_gemms_skip_the_dequant_cache() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let cases = cases();
    let total = cases.len();
    let failures: Vec<String> = cases.iter().filter_map(check_child).collect();
    assert!(
        failures.is_empty(),
        "{} of {total} cases failed:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// The body the parent runs in a fresh process per case.
#[test]
#[ignore = "spawned by ceiling_routed_gemms_skip_the_dequant_cache with one case per process"]
fn child_dequant_cache_case() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: run through ceiling_routed_gemms_skip_the_dequant_cache");
        return;
    }
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let name = std::env::var(CASE_ENV).expect("the parent names the case");
    run_case(&find_case(&name));
    println!("{DONE_MARKER}");
}
