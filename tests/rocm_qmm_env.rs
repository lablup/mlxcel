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

//! The ROCm quantized-matmul integer knobs are range-checked: a bad value
//! warns once and the default applies (issue #2152).
//!
//! The overlay used to read `MLX_ROCM_WMMA_QMM_MAX_M` and its siblings with
//! `strtol` and a bare `static_cast<int>`, so `4294967297` silently became a
//! row ceiling of 1 and sent a 64-row bf16 GEMM to dequantize + hipBLASLt
//! instead of the fused WMMA kernel, `2147483648` wrapped negative and was
//! dropped silently, and `-1`, `0` and `12abc` were dropped silently too.
//! `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=4294967304` silently became 8, and
//! `MLX_ROCM_QMV_TILE_N` went through `atoi` on every call. Now every one of
//! them goes through `env_int_or_default` in the overlay's `env_int.h`: a
//! value that is not a whole decimal integer in range prints one stderr line
//! naming the variable and the fallback, and the default applies.
//!
//! The knobs are function-local statics read once per process, so each case
//! needs a fresh process: the parent re-invokes this binary on an ignored
//! child test once per case, one child at a time. The child asks
//! `quantized_matmul_matches_dense_gemm` which route a bf16 `[64, 4096]` x
//! 4-bit `[4096, 4096]` g64 GEMM takes (it shares `select_qmm_route` with
//! `QuantizedMatmul::eval_gpu`), then runs a 128-row GEMM twice (dequantize
//! path, which reads the cache size) and a one-row GEMV (qmv path, which reads
//! the tile width) on the GPU and compares each with an f32 reference on the
//! CPU stream. The parent checks the child's exit status, the route it
//! reported, and the warning lines on its stderr. `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=1`
//! is set for the ceiling and cache cases so the dequantize route is eligible
//! at every row count and only the ceiling decides the route.
//!
//! With the fix reverted (the overlay sources restored from main, measured on
//! gfx1151) 9 of the 15 cases fail: `MLX_ROCM_WMMA_QMM_MAX_M=4294967297`
//! reports the dense route, and none of the nine invalid values
//! (`4294967297`, `2147483648`, `-1`, `0` and `12abc` for the ceiling, `0`
//! for the threshold, `4294967304` for the cache, `abc` and `33` for the tile
//! width) prints a warning. With only the threshold read made per call again,
//! the threshold case fails with one warning per GEMM.
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --release --features rocm --test rocm_qmm_env -- --test-threads=1
//! ```

#![cfg(feature = "rocm")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{MlxArray, dtype};

const MAX_M: &str = "MLX_ROCM_WMMA_QMM_MAX_M";
const THRESHOLD: &str = "MLX_ROCM_QMM_DEQUANT_M_THRESHOLD";
const CACHE_SIZE: &str = "MLX_ROCM_QMM_DEQUANT_CACHE_SIZE";
const TILE_N: &str = "MLX_ROCM_QMV_TILE_N";
/// Every variable a case sets; the parent clears them all before applying a
/// case, so the host environment cannot leak in.
const ALL_VARS: &[&str] = &[MAX_M, THRESHOLD, CACHE_SIZE, TILE_N];

/// Set by the parent on the children it spawns; the child body runs only
/// when it is present, so `--include-ignored` sweeps skip it.
const CHILD_ENV: &str = "MLXCEL_ROCM_QMM_ENV_CHILD";
const CHILD_TEST: &str = "child_qmm_with_env";
const CHILD_BUDGET: Duration = Duration::from_secs(120);
const DONE_MARKER: &str = "QMM_ENV done";
const ROUTE_PREFIX: &str = "QMM_ENV route_dense=";

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
/// Relative error of a bf16 GEMM against the f32 reference.
const TOLERANCE: f32 = 2e-2;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// One child process: the variables it sets, the variable whose warning it
/// checks, the exact warning line expected (or none), and the route the probe
/// must report (or `None` when the case does not pin it).
struct Case {
    env: Vec<(&'static str, &'static str)>,
    watched: &'static str,
    warning: Option<String>,
    route_dense: Option<bool>,
}

fn warning(var: &str, value: &str, expected: &str, fallback: &str) -> Option<String> {
    Some(format!(
        "[ROCm] ignoring invalid {var}=\"{value}\" (expected {expected}); using {fallback}"
    ))
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    // The ceiling. On gfx1151 (RDNA 3.5) the per-device default is 128, so a
    // 64-row GEMM stays on the fused kernel unless a valid ceiling at or
    // below 64 is set.
    let max_m = |value: Option<&'static str>, warns: bool, dense: bool| {
        let mut env = vec![(THRESHOLD, "1")];
        if let Some(v) = value {
            env.push((MAX_M, v));
        }
        Case {
            env,
            watched: MAX_M,
            warning: if warns {
                warning(
                    MAX_M,
                    value.expect("an invalid case has a value"),
                    "a positive integer",
                    "the per-device default",
                )
            } else {
                None
            },
            route_dense: Some(dense),
        }
    };
    cases.push(max_m(None, false, false));
    cases.push(max_m(Some("128"), false, false));
    cases.push(max_m(Some(""), false, false));
    // A valid ceiling below 64 rows does move the route, so the probe can
    // tell a wrapped ceiling of 1 from the default.
    cases.push(max_m(Some("32"), false, true));
    for invalid in ["4294967297", "2147483648", "-1", "0", "12abc"] {
        cases.push(max_m(Some(invalid), true, false));
    }

    // The threshold itself. Every probe and GEMM in the child reads it, so
    // exactly one line also checks that it is read once, not per call.
    cases.push(Case {
        env: vec![(THRESHOLD, "0")],
        watched: THRESHOLD,
        warning: warning(
            THRESHOLD,
            "0",
            "a positive integer",
            "the built-in crossover",
        ),
        route_dense: Some(false),
    });

    // The dequantized-weight cache: 0 is the documented off switch.
    cases.push(Case {
        env: vec![(THRESHOLD, "1"), (CACHE_SIZE, "4294967304")],
        watched: CACHE_SIZE,
        warning: warning(
            CACHE_SIZE,
            "4294967304",
            "a non-negative integer",
            "the default 8",
        ),
        route_dense: None,
    });
    cases.push(Case {
        env: vec![(THRESHOLD, "1"), (CACHE_SIZE, "0")],
        watched: CACHE_SIZE,
        warning: None,
        route_dense: None,
    });

    // The tiled qmv width, read on the one-row GEMV. No threshold here, so
    // that GEMV takes qmv rather than dequantize.
    for (value, warns) in [("16", false), ("abc", true), ("33", true)] {
        cases.push(Case {
            env: vec![(TILE_N, value)],
            watched: TILE_N,
            warning: if warns {
                warning(
                    TILE_N,
                    value,
                    "an integer from 1 to 32",
                    "the per-architecture default",
                )
            } else {
                None
            },
            route_dense: None,
        });
    }
    cases
}

struct ChildRun {
    success: bool,
    status: String,
    stdout: String,
    stderr: String,
}

fn describe(case: &Case) -> String {
    let pairs: Vec<String> = case.env.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
    format!("[{}]", pairs.join(" "))
}

/// Runs the child test with the case's variables and returns its outcome.
/// Panics if the child outlives its budget.
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
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    for var in ALL_VARS {
        cmd.env_remove(var);
    }
    for (k, v) in &case.env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn the child test process");

    // Drain both pipes on their own threads so the child can never block on a
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
            "the child with {} did not finish within {CHILD_BUDGET:?}\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}",
            describe(case)
        )
    });
    ChildRun {
        success: status.success(),
        status: status.to_string(),
        stdout,
        stderr,
    }
}

/// Checks one child and returns a description of what went wrong, if
/// anything.
fn check_child(case: &Case) -> Option<String> {
    let run = run_child(case);
    let mut problems = Vec::new();
    if !run.success {
        problems.push(format!("exited with {}", run.status));
    }
    if !run.stdout.contains(DONE_MARKER) {
        problems.push("did not finish its GEMM checks".to_string());
    }
    let mentions: Vec<&str> = run
        .stderr
        .lines()
        .filter(|l| l.contains(case.watched))
        .collect();
    match &case.warning {
        Some(expected) => {
            if mentions != [expected.as_str()] {
                problems.push(format!(
                    "expected exactly one stderr line `{expected}`, got {mentions:?}"
                ));
            }
        }
        None => {
            if !mentions.is_empty() {
                problems.push(format!("expected no warning, got {mentions:?}"));
            }
        }
    }
    if let Some(dense) = case.route_dense {
        let reported = run
            .stdout
            .lines()
            // libtest's `test name ... ` has no newline, so the marker can
            // sit mid-line.
            .find_map(|l| l.split_once(ROUTE_PREFIX).map(|(_, rest)| rest.trim()));
        let want = if dense { "true" } else { "false" };
        if reported != Some(want) {
            problems.push(format!(
                "expected the 64-row probe to report route_dense={want}, got {reported:?}"
            ));
        }
    }
    if problems.is_empty() {
        eprintln!("{}: ok", describe(case));
        return None;
    }
    Some(format!(
        "{}: {}\n--- child stdout ---\n{}\n--- child stderr ---\n{}",
        describe(case),
        problems.join("; "),
        run.stdout,
        run.stderr
    ))
}

/// One test, not one per group, so the children run one after another and
/// never share the GPU with each other.
#[test]
fn qmm_env_integers_are_range_checked() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let cases = cases();
    let total = cases.len();
    let failures: Vec<String> = cases.iter().filter_map(check_child).collect();
    assert!(
        failures.is_empty(),
        "{} of {total} cases misbehaved:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
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

/// The route probe. It reads only shapes and dtypes, so zero-filled arrays
/// of the probed shapes are enough.
fn probe_route_dense() -> bool {
    let _gpu = DefaultDeviceGuard::gpu();
    let x = mlxcel_core::zeros(&[64, 4096], dtype::BFLOAT16);
    let w = mlxcel_core::zeros(&[4096, 4096 * BITS / 32], dtype::UINT32);
    let scales = mlxcel_core::zeros(&[4096, 4096 / GROUP_SIZE], dtype::BFLOAT16);
    let biases = mlxcel_core::zeros(&[4096, 4096 / GROUP_SIZE], dtype::BFLOAT16);
    // SAFETY: every reference is a live array owned by this frame, and the
    // bridge only reads shapes and dtypes.
    unsafe {
        mlxcel_core::quantized_matmul_matches_dense_gemm(
            &x,
            &w,
            &scales,
            &*biases as *const MlxArray,
            GROUP_SIZE,
            BITS,
        )
    }
}

/// `rows` x `k` bf16 activations times a 4-bit g64 `[n, k]` weight on the
/// GPU, against `x @ dequantize(w)^T` in f32 on the CPU stream.
fn check_gemm(label: &str, rows: i32, k: i32, n: i32, repeats: usize) {
    let w_f32 = mlxcel_core::from_slice_f32(&pseudo_random((n * k) as usize, 0x2152), &[n, k]);
    let w = mlxcel_core::astype(&w_f32, dtype::BFLOAT16);
    let q = mlxcel_core::quantize_weights_with_mode(&w, GROUP_SIZE, BITS, "affine");
    let packed = mlxcel_core::quantized_weights_w(&q);
    let scales = mlxcel_core::quantized_weights_scales(&q);
    let biases = mlxcel_core::quantized_weights_biases(&q);
    let x_f32 = mlxcel_core::from_slice_f32(
        &pseudo_random((rows * k) as usize, 0x2152 ^ rows as u32),
        &[rows, k],
    );
    let x = mlxcel_core::astype(&x_f32, dtype::BFLOAT16);

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

    for pass in 0..repeats {
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
                panic!("{label} pass {pass} on the GPU failed: {err}");
            }
            out
        };
        let got = mlxcel_core::astype(&got, dtype::FLOAT32);
        let rel = max_abs(&mlxcel_core::subtract(&got, &reference)) / scale;
        println!("{label} pass={pass} rel={rel:.3e}");
        // Written so a NaN fails too.
        assert!(
            rel < TOLERANCE,
            "{label} pass {pass}: GPU result is {rel:e} from the f32 reference"
        );
    }
}

/// The body the parent test runs in a fresh process per case.
#[test]
#[ignore = "spawned by qmm_env_integers_are_range_checked with the ROCm qmm variables set per case"]
fn child_qmm_with_env() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: run through qmm_env_integers_are_range_checked");
        return;
    }
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    println!("{ROUTE_PREFIX}{}", probe_route_dense());
    // 128 rows: dequantize + hipBLASLt on gfx1151, which reads the cache
    // size; the second pass goes through the cache when it is on.
    check_gemm("gemm 128x512x512", 128, 512, 512, 2);
    // One row: qmv, which reads the tiled qmv width, unless the threshold
    // sends it to dequantize as well.
    check_gemm("gemv 1x2048x512", 1, 2048, 512, 2);
    println!("{DONE_MARKER}");
}
