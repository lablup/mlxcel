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

//! The ROCm FFT survives hipFFT plan evictions and matches the CPU stream
//! (issue #1876).
//!
//! The hipFFT plan cache used to be capped at 8 because the first eviction
//! hung: `hipfftDestroy` frees the plan's device memory with `hipFree`, and
//! that `hipFree` waited forever. The cause was the backend setting
//! `hipDeviceScheduleBlockingSync` after the allocator had already created the
//! null stream's HIP queue, which left that queue with completion signals the
//! runtime cannot attach a handler to (`rocm::ensure_device_flags` in the
//! overlay's `device.cpp` fixes the order).
//!
//! The work runs in a child process (this binary, re-invoked on an ignored
//! test) with `MLX_ROCM_FFT_CACHE_SIZE=16`, so evictions start after eight
//! round trips and repeat dozens of times, all through the production
//! `mlxcel_core` FFT ops. The variable has to be in the environment before the
//! first FFT, which is why it is set on a fresh process rather than here. The
//! parent kills a child that does not finish in time, which is what the hang
//! looks like: with the ordering fix reverted the child never gets past the
//! first eviction (checked by reverting it).
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_fft_plan_cache
//! ```

#![cfg(feature = "rocm")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Set by the parent on the child it spawns; the child body runs only when it
/// is present, so `--include-ignored` sweeps do not run it in a shared process.
const CHILD_ENV: &str = "MLXCEL_ROCM_FFT_CHILD";
const CHILD_TEST: &str = "child_fft_plan_cache";
/// Small enough that the first eviction comes early (16 hung at the first
/// eviction before the fix), and every later transform evicts again.
const CHILD_CACHE_SIZE: &str = "16";
/// Measured at about 10 s for the whole child on gfx1151 including rocFFT's
/// kernel compilation on a cold cache; the pre-fix behaviour is an unbounded
/// wait, so the margin only has to absorb a slow host.
const CHILD_BUDGET: Duration = Duration::from_secs(120);
/// Distinct round-trip lengths: two plans (R2C and C2R) each, so 96 plans
/// through a 16-entry cache.
const ROUND_TRIP_SHAPES: i32 = 48;
const TOLERANCE: f32 = 1e-5;
/// The lengths mlxcel reaches: 2 and 8 are degenerate, 400 is Whisper's frame
/// (not a power of two), 512 the Phi-4-multimodal front end, 1200 Kokoro's
/// STFT, and 1024/2048 bracket them.
const LENGTHS: &[i32] = &[2, 8, 400, 512, 1024, 1200, 2048];
const DONE_MARKER: &str = "FFT_PLAN_CACHE done";

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

#[test]
fn fft_survives_plan_evictions_and_matches_cpu() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let exe = std::env::current_exe().expect("path of this test binary");
    let mut child = Command::new(exe)
        .args([
            CHILD_TEST,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .env("MLX_ROCM_FFT_CACHE_SIZE", CHILD_CACHE_SIZE)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child test process");

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
            "the child did not finish within {CHILD_BUDGET:?}: an FFT plan eviction hung (issue #1876)\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
        )
    });
    assert!(
        status.success(),
        "child exited with {status}\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains(DONE_MARKER),
        "the child exited without finishing its checks\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
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

fn input(batch: i32, n: i32, seed: u32) -> UniquePtr<MlxArray> {
    let values = pseudo_random((batch * n) as usize, seed);
    mlxcel_core::from_slice_f32(&values, &[batch, n])
}

fn max_abs(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, dtype::FLOAT32);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&f));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    max_abs(&mlxcel_core::subtract(
        &mlxcel_core::astype(a, dtype::FLOAT32),
        &mlxcel_core::astype(b, dtype::FLOAT32),
    ))
}

fn real_rel(got: &MlxArray, reference: &MlxArray) -> f32 {
    let scale = max_abs(reference);
    max_abs_diff(got, reference) / if scale > 0.0 { scale } else { 1.0 }
}

/// Real and imaginary differences summed, scaled by the reference magnitude.
fn complex_rel(got: &MlxArray, reference: &MlxArray) -> f32 {
    let re = |a: &MlxArray| mlxcel_core::real_part(a);
    let im = |a: &MlxArray| mlxcel_core::imag_part(a);
    let scale = max_abs(&re(reference)) + max_abs(&im(reference));
    let diff = max_abs_diff(&re(got), &re(reference)) + max_abs_diff(&im(got), &im(reference));
    diff / if scale > 0.0 { scale } else { 1.0 }
}

/// Evaluate on the GPU with the default device pinned, before the guard drops.
fn on_gpu<F: FnOnce() -> UniquePtr<MlxArray>>(f: F) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    let out = f();
    mlxcel_core::eval(&out);
    out
}

fn on_cpu<F: FnOnce() -> UniquePtr<MlxArray>>(f: F) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::cpu();
    let out = f();
    mlxcel_core::eval(&out);
    out
}

/// The body of `fft_survives_plan_evictions_and_matches_cpu`, run in the child
/// process it spawns with a 16-entry plan cache.
#[test]
#[ignore = "spawned by fft_survives_plan_evictions_and_matches_cpu with a small plan cache"]
fn child_fft_plan_cache() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: run through fft_survives_plan_evictions_and_matches_cpu");
        return;
    }
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }

    let mut failures = Vec::new();
    let mut check = |case: &str, batch: i32, n: i32, rel: f32| {
        println!("{case:<10} batch={batch:<4} n={n:<5} rel={rel:.3e}");
        // Written so a NaN fails too.
        let within = rel < TOLERANCE;
        if !within {
            failures.push(format!("{case} batch={batch} n={n}: rel {rel:.3e}"));
        }
    };

    // Eviction stress first: every length is a new pair of plans, so from the
    // ninth length on, each transform evicts and destroys an older plan.
    for i in 0..ROUND_TRIP_SHAPES {
        let n = 64 + i * 8;
        let x = input(4, n, 0x5EED ^ n as u32);
        let back = on_gpu(|| {
            let spec = mlxcel_core::rfft(&x, n, 1);
            mlxcel_core::irfft(&spec, n, 1)
        });
        check("stress", 4, n, real_rel(&back, &x));
    }

    // GPU against the CPU stream, still through the small cache.
    for &n in LENGTHS {
        let x = input(4, n, 0xC0FFEE ^ n as u32);
        let gpu = on_gpu(|| mlxcel_core::rfft(&x, n, 1));
        let cpu = on_cpu(|| mlxcel_core::rfft(&x, n, 1));
        check("rfft", 4, n, complex_rel(&gpu, &cpu));
    }
    for &n in LENGTHS {
        let x = input(4, n, 0xBEEF00 ^ n as u32);
        // The spectrum comes from the CPU so both inverses see the same input.
        let spec = on_cpu(|| mlxcel_core::rfft(&x, n, 1));
        let gpu = on_gpu(|| mlxcel_core::irfft(&spec, n, 1));
        let cpu = on_cpu(|| mlxcel_core::irfft(&spec, n, 1));
        check("irfft", 4, n, real_rel(&gpu, &cpu));
    }
    for &n in LENGTHS {
        let x = input(3, n, 0x1234 ^ n as u32);
        let back = on_gpu(|| {
            let spec = mlxcel_core::rfft(&x, n, 1);
            mlxcel_core::irfft(&spec, n, 1)
        });
        check("roundtrip", 3, n, real_rel(&back, &x));
    }
    for &n in LENGTHS {
        let x = mlxcel_core::astype(&input(4, n, 0x77 ^ n as u32), dtype::COMPLEX64);
        let gpu = on_gpu(|| mlxcel_core::fft(&x, n, 1));
        let cpu = on_cpu(|| mlxcel_core::fft(&x, n, 1));
        check("fft", 4, n, complex_rel(&gpu, &cpu));
    }
    // A batch large enough that the launch geometry is not the trivial one.
    let x = input(512, 512, 0xABCD);
    let gpu = on_gpu(|| mlxcel_core::rfft(&x, 512, 1));
    let cpu = on_cpu(|| mlxcel_core::rfft(&x, 512, 1));
    check("rfft", 512, 512, complex_rel(&gpu, &cpu));

    assert!(
        failures.is_empty(),
        "FFT results outside {TOLERANCE:e} relative of the reference:\n{}",
        failures.join("\n")
    );
    println!("{DONE_MARKER}");
}
