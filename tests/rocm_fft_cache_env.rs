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

//! `MLX_ROCM_FFT_CACHE_SIZE` is validated: a bad value warns once and the
//! hipFFT plan cache falls back to its default (issue #2051).
//!
//! The overlay used to parse the variable with an unchecked `std::stoul`:
//! `0` was accepted and the first plan insert popped an empty list
//! (undefined behavior), `abc`, an empty value and an overflowing value threw
//! a bare `stoul` out of every FFT, and `-1` and `8abc` were taken silently as
//! `SIZE_MAX` and 8. Now any value that is not a whole decimal integer from 1
//! to `i32::MAX` prints one stderr line naming the variable and the default,
//! and the cache uses the default.
//!
//! The plan cache is a function-local static that reads the variable once, at
//! the first device FFT, so each value needs a fresh process: the parent
//! re-invokes this binary on an ignored child test once per value, one child
//! at a time. The child
//! runs rfft on the GPU and on the CPU stream, twice, and compares them; the
//! parent checks the child's exit status and counts the warning lines on its
//! stderr. With the fix reverted (measured on gfx1151) `0` and `0x10`, which
//! `stoul` reads as 0, fail the first GPU rfft with `std::bad_alloc`, `abc`,
//! the empty value and the overflow fail it with `stoul`, and `-1`, `8abc` and
//! `2147483648` print no warning.
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_fft_cache_env
//! ```

#![cfg(feature = "rocm")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{MlxArray, dtype};

const VAR: &str = "MLX_ROCM_FFT_CACHE_SIZE";
/// The capacity `fft_plan_cache()` in the overlay's `fft.hip` passes.
const DEFAULT_CAPACITY: usize = 128;
/// Set by the parent on the children it spawns; the child body runs only
/// when it is present, so `--include-ignored` sweeps skip it.
const CHILD_ENV: &str = "MLXCEL_ROCM_FFT_CACHE_ENV_CHILD";
const CHILD_TEST: &str = "child_fft_with_cache_env";
/// A child is a few rffts, measured at a few seconds on gfx1151 including
/// rocFFT's kernel compilation; the margin only has to absorb a slow host.
const CHILD_BUDGET: Duration = Duration::from_secs(120);
const DONE_MARKER: &str = "FFT_CACHE_ENV done";
const TOLERANCE: f32 = 1e-5;
const LENGTHS: &[i32] = &[400, 512];

/// Values the parser rejects. Each must warn exactly once and leave FFT
/// working.
const INVALID: &[&str] = &[
    "0",
    "-1",
    "abc",
    "",
    "8abc",
    "0x10",
    // Past `long` (ERANGE).
    "99999999999999999999999",
    // Fits `long` but not `int`.
    "2147483648",
];
/// Values the parser accepts silently.
const VALID: &[&str] = &["16", "1", "2147483647"];

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn warning_for(value: &str) -> String {
    format!(
        "[ROCm] ignoring invalid {VAR}=\"{value}\" (expected a positive integer); using the default {DEFAULT_CAPACITY}"
    )
}

struct ChildRun {
    success: bool,
    status: String,
    stdout: String,
    stderr: String,
}

/// Runs the child test with the variable set to `value`, or removed for
/// `None`, and returns its outcome. Panics if the child outlives its budget.
fn run_child(value: Option<&str>) -> ChildRun {
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
    match value {
        Some(v) => cmd.env(VAR, v),
        None => cmd.env_remove(VAR),
    };
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
            "the child with {VAR}={value:?} did not finish within {CHILD_BUDGET:?}\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
        )
    });
    ChildRun {
        success: status.success(),
        status: status.to_string(),
        stdout,
        stderr,
    }
}

/// Checks one child against the expected warning (or its absence) and
/// returns a description of what went wrong, if anything.
fn check_child(value: Option<&str>, expect_warning: bool) -> Option<String> {
    let run = run_child(value);
    let mut problems = Vec::new();
    if !run.success {
        problems.push(format!("exited with {}", run.status));
    }
    if !run.stdout.contains(DONE_MARKER) {
        problems.push("did not finish its FFT checks".to_string());
    }
    let mentions: Vec<&str> = run.stderr.lines().filter(|l| l.contains(VAR)).collect();
    if expect_warning {
        let expected = warning_for(value.expect("an invalid case has a value"));
        if mentions != [expected.as_str()] {
            problems.push(format!(
                "expected exactly one stderr line `{expected}`, got {mentions:?}"
            ));
        }
    } else if !mentions.is_empty() {
        problems.push(format!("expected no warning, got {mentions:?}"));
    }
    if problems.is_empty() {
        eprintln!("{VAR}={value:?}: ok");
        return None;
    }
    Some(format!(
        "{VAR}={value:?}: {}\n--- child stdout ---\n{}\n--- child stderr ---\n{}",
        problems.join("; "),
        run.stdout,
        run.stderr
    ))
}

/// One test, not one per group, so the children run one after another and
/// never share the GPU with each other.
#[test]
fn fft_cache_size_is_validated_and_fft_still_works() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let cases = INVALID
        .iter()
        .map(|v| (Some(*v), true))
        .chain(VALID.iter().map(|v| (Some(*v), false)))
        .chain(std::iter::once((None, false)));
    let total = INVALID.len() + VALID.len() + 1;
    let failures: Vec<String> = cases
        .filter_map(|(value, expect_warning)| check_child(value, expect_warning))
        .collect();
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

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    max_abs(&mlxcel_core::subtract(a, b))
}

/// Real and imaginary differences summed, scaled by the reference magnitude.
fn complex_rel(got: &MlxArray, reference: &MlxArray) -> f32 {
    let re = |a: &MlxArray| mlxcel_core::real_part(a);
    let im = |a: &MlxArray| mlxcel_core::imag_part(a);
    let scale = max_abs(&re(reference)) + max_abs(&im(reference));
    let diff = max_abs_diff(&re(got), &re(reference)) + max_abs_diff(&im(got), &im(reference));
    diff / if scale > 0.0 { scale } else { 1.0 }
}

/// The body the parent tests run in a fresh process per value. The first
/// rfft on the GPU constructs the plan cache, which reads the variable.
#[test]
#[ignore = "spawned by fft_cache_size_is_validated_and_fft_still_works with MLX_ROCM_FFT_CACHE_SIZE set per case"]
fn child_fft_with_cache_env() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: run through fft_cache_size_is_validated_and_fft_still_works");
        return;
    }
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }

    // Twice over two lengths: the second pass hits cached plans, and a
    // warning printed on every lookup instead of once would show up here.
    for pass in 0..2 {
        for &n in LENGTHS {
            let values = pseudo_random((4 * n) as usize, 0xF0F0 ^ n as u32);
            let x = mlxcel_core::from_slice_f32(&values, &[4, n]);
            let gpu = {
                let _guard = DefaultDeviceGuard::gpu();
                let out = mlxcel_core::rfft(&x, n, 1);
                if let Err(err) = mlxcel_core::try_eval(&out) {
                    panic!("rfft n={n} on the GPU failed: {err}");
                }
                out
            };
            let cpu = {
                let _guard = DefaultDeviceGuard::cpu();
                let out = mlxcel_core::rfft(&x, n, 1);
                mlxcel_core::eval(&out);
                out
            };
            let rel = complex_rel(&gpu, &cpu);
            println!("pass={pass} n={n} rel={rel:.3e}");
            // Written so a NaN fails too.
            assert!(
                rel < TOLERANCE,
                "rfft n={n} on the GPU is {rel:e} from the CPU stream"
            );
        }
    }
    println!("{DONE_MARKER}");
}
