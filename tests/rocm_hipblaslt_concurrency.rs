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

//! The ROCm overlay's hipBLASLt GEMM state is safe across threads and
//! streams (issue #2200).
//!
//! `gemms/hipblaslt_gemm.cpp` used to publish its per-device handle through
//! a plain `bool` set before `hipblasLtCreate` ran, hand every stream on the
//! device one shared workspace buffer that `ensure_workspace` could free and
//! reallocate with no lock, and erase pipe-cache entries (layouts and a
//! matmul descriptor) that another thread could still be passing to
//! `hipblasLtMatmul`. The server runs forwards on several worker threads that
//! share one `Device`, each on its own stream, so every window is reachable.
//! Measured on gfx1151, the pipe cache was the race that fires: hipBLASLt
//! writes into the descriptor objects during `hipblasLtMatmul`, so a pipe
//! shared by threads corrupts the heap and faults the queue however it is
//! owned. The handle is now published with release after a successful
//! create, the workspace is one buffer per stream released with the stream's
//! encoder, and the pipe cache is per thread.
//!
//! The init window is once per process, so the parent runs the child in a
//! fresh process `CHILD_RUNS` times, with `MLX_NO_HIPBLASLT` and
//! `MLX_HIPBLASLT_NO_PIPE_CACHE` removed from its environment so the bf16
//! GEMMs take the hipBLASLt pipe-cache path. Each child starts `THREADS`
//! threads that install their own stream, wait on a barrier before any GPU
//! matmul has run in the process, then run `ROUNDS` rounds of three GEMMs on
//! the same shapes, so every thread contends on the same pipe keys: a bf16
//! `[64, 64] x transpose([64, 64])` (the hipBLASLt TN pipe), a bf16 batched
//! `[4, 64, 64] x [4, 64, 64]` (`hipblaslt_gemm_batched`) and an f32
//! `[64, 128] x [128, 64]` (rocBLAS, so both libraries interleave). Inputs
//! are integers: `-2..=2` for bf16 over `K = 64` keeps every output an
//! integer of magnitude at most 256, exact in bf16, and `-4..=4` for f32
//! over `K = 128` stays far below 2^24. The f32 outputs must equal the host
//! `i32` reference exactly. The bf16 outputs must lie within
//! `bf16_gemm_tolerance` of it, which is at most about 0.002 here, so any
//! wrong operand, element or tile (an error of at least 1) still fails. The
//! bf16 check cannot be exact on gfx11: hipBLASLt and rocBLAS run these
//! GEMMs on `v_wmma_f32_16x16x16_bf16`, whose f32 accumulation is not IEEE
//! (a cancelling term of opposite sign leaves a residue of about one unit in
//! the 24th bit, for example `1 + -1` gives `-2^-24`), so exact-zero outputs
//! come back as `+-2^-23` or `-2^-22` (#2206; measured in
//! `docs/benchmark_results/rocm-bf16-gemm-residue-gfx1151-2026-10-08.md`).
//! `bf16_gemm_error_is_within_the_accumulation_bound` pins that bound on
//! inputs that provoke the residue in every output. The parent also requires the
//! `[hipBLASLt caps] device` line exactly once in each child's stderr, which
//! `gemm_caps` prints on the first hipBLASLt GEMM, proving the bf16 GEMMs
//! reached hipBLASLt.
//!
//! On gfx1151 the heuristics request no workspace for these shapes, so the
//! per-stream workspace path is not exercised here (`MLX_ROCM_GEMM_DEBUG=1`
//! prints what each heuristic asked for). What this test exercises on that
//! device is the pipe cache under contention and the handle init window.
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_hipblaslt_concurrency
//! ```

#![cfg(feature = "rocm")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::dtype;
use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
};

/// Set by the parent on the children it spawns; the child body runs only
/// when it is present, so `--include-ignored` sweeps skip it.
const CHILD_ENV: &str = "MLXCEL_ROCM_HIPBLASLT_CHILD";
const CHILD_TEST: &str = "child_hipblaslt_gemms_from_many_threads";
const CHILD_BUDGET: Duration = Duration::from_secs(120);
const CHILD_RUNS: usize = 10;
/// The child prints this once every thread has finished, so a child that
/// skipped or died before the check cannot pass as a success.
const DONE_MARKER: &str = "HIPBLASLT done";
/// Printed by `gemm_caps` in `hipblaslt_gemm.cpp` on the device's first
/// hipBLASLt GEMM, once per process.
const CAPS_MARKER: &str = "[hipBLASLt caps] device";
/// Variables that would steer the bf16 GEMMs away from the path under test.
const ROUTE_ENV: [&str; 2] = ["MLX_NO_HIPBLASLT", "MLX_HIPBLASLT_NO_PIPE_CACHE"];

const THREADS: usize = 8;
const ROUNDS: usize = 33;
/// bf16 shapes: `[BF_M, BF_K] x transpose([BF_N, BF_K])` and the batched
/// `[BATCH, BF_M, BF_K] x [BATCH, BF_K, BF_N]`, inputs in `-BF_HALF..=BF_HALF`.
const BF_HALF: i32 = 2;
const BF_M: usize = 64;
const BF_K: usize = 64;
const BF_N: usize = 64;
const BATCH: usize = 4;
/// f32 shape: `[F_M, F_K] x [F_K, F_N]`, inputs in `-F_HALF..=F_HALF`.
const F_HALF: i32 = 4;
const F_M: usize = 64;
const F_K: usize = 128;
const F_N: usize = 64;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// Deterministic integers in `-half..=half` from a per-thread, per-round
/// seed.
fn integer_values(len: usize, seed: usize, half: i32) -> Vec<f32> {
    let span = (2 * half + 1) as usize;
    (0..len)
        .map(|i| (((i * 31 + seed * 17 + 7) % span) as i32 - half) as f32)
        .collect()
}

/// Host `i32` reference of `a[m, k] x b[k, n]` for one matrix, with `b`
/// addressed through `b_at(k, n)` so the transposed layout shares the code.
fn reference(
    a: &[f32],
    m: usize,
    k: usize,
    n: usize,
    b_at: impl Fn(usize, usize) -> f32,
) -> Vec<i32> {
    let mut out = vec![0i32; m * n];
    for row in 0..m {
        for kk in 0..k {
            let av = a[row * k + kk] as i32;
            for col in 0..n {
                out[row * n + col] += av * b_at(kk, col) as i32;
            }
        }
    }
    out
}

/// Per-element tolerance of a bf16 GEMM with f32 accumulation whose exact
/// result is representable in bf16: twice the standard dot-product bound
/// `K * u * sum_k |a_k * b_k|` with `u = 2^-24` (Higham, "Accuracy and
/// Stability of Numerical Algorithms", section 3.1), the factor 2 covering the
/// rounding of the f32 accumulator to the bf16 output. `f32::EPSILON` is
/// `2^-23 = 2u`. On gfx1151 the WMMA residue measured at most 0.8% of the
/// undoubled bound (#2206).
fn bf16_gemm_tolerance(k: usize, abs_sum: i32) -> f32 {
    k as f32 * f32::EPSILON * abs_sum as f32
}

/// Host `sum_k |a[m, k] * b[k, n]|` with `b` addressed through `b_at(k, n)`.
fn abs_reference(
    a: &[f32],
    m: usize,
    k: usize,
    n: usize,
    b_at: impl Fn(usize, usize) -> f32,
) -> Vec<i32> {
    reference(
        &a.iter().map(|x| x.abs()).collect::<Vec<_>>(),
        m,
        k,
        n,
        |kk, col| b_at(kk, col).abs(),
    )
}

/// Evaluates `out`, casts it to f32 and compares every element with `want`:
/// exactly when `tolerance` is `None`, else within `tolerance[i]`.
fn check(
    what: &str,
    seed: usize,
    out: &mlxcel_core::MlxArray,
    want: &[i32],
    tolerance: Option<&[f32]>,
    cols: usize,
) -> Result<(), String> {
    let out = mlxcel_core::astype(out, dtype::FLOAT32);
    mlxcel_core::try_eval(&out).map_err(|e| format!("{what} seed {seed}: eval failed: {e}"))?;
    let bytes = mlxcel_core::array_to_raw_bytes(&out);
    if bytes.len() != want.len() * 4 {
        return Err(format!(
            "{what} seed {seed}: expected {} output bytes, got {}",
            want.len() * 4,
            bytes.len()
        ));
    }
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let allowed = tolerance.map_or(0.0, |t| t[i]);
        let diff = (got - want[i] as f32).abs();
        if diff.is_nan() || diff > allowed {
            return Err(format!(
                "{what} seed {seed}: element {i} (row {}, col {}) is {got}, expected {} (tolerance {allowed})",
                i / cols,
                i % cols,
                want[i]
            ));
        }
    }
    Ok(())
}

/// bf16 `a[BF_M, BF_K] x transpose(b[BF_N, BF_K])`: the TN pipe.
fn bf16_tn_and_check(seed: usize) -> Result<(), String> {
    let a_host = integer_values(BF_M * BF_K, seed, BF_HALF);
    let b_host = integer_values(BF_N * BF_K, seed + 1_000_003, BF_HALF);
    let a = mlxcel_core::from_slice_f32(&a_host, &[BF_M as i32, BF_K as i32]);
    let b = mlxcel_core::from_slice_f32(&b_host, &[BF_N as i32, BF_K as i32]);
    let a = mlxcel_core::astype(&a, dtype::BFLOAT16);
    let b = mlxcel_core::astype(&b, dtype::BFLOAT16);
    let bt = mlxcel_core::transpose(&b);
    let out = mlxcel_core::matmul(&a, &bt);
    let want = reference(&a_host, BF_M, BF_K, BF_N, |k, n| b_host[n * BF_K + k]);
    let tol: Vec<f32> = abs_reference(&a_host, BF_M, BF_K, BF_N, |k, n| b_host[n * BF_K + k])
        .into_iter()
        .map(|abs| bf16_gemm_tolerance(BF_K, abs))
        .collect();
    check("bf16 TN", seed, &out, &want, Some(&tol), BF_N)
}

/// bf16 batched `a[BATCH, BF_M, BF_K] x b[BATCH, BF_K, BF_N]`.
fn bf16_batched_and_check(seed: usize) -> Result<(), String> {
    let a_host = integer_values(BATCH * BF_M * BF_K, seed + 7, BF_HALF);
    let b_host = integer_values(BATCH * BF_K * BF_N, seed + 2_000_003, BF_HALF);
    let a = mlxcel_core::from_slice_f32(&a_host, &[BATCH as i32, BF_M as i32, BF_K as i32]);
    let b = mlxcel_core::from_slice_f32(&b_host, &[BATCH as i32, BF_K as i32, BF_N as i32]);
    let a = mlxcel_core::astype(&a, dtype::BFLOAT16);
    let b = mlxcel_core::astype(&b, dtype::BFLOAT16);
    let out = mlxcel_core::matmul(&a, &b);
    let mut want = Vec::with_capacity(BATCH * BF_M * BF_N);
    let mut tol = Vec::with_capacity(BATCH * BF_M * BF_N);
    for batch in 0..BATCH {
        let a_mat = &a_host[batch * BF_M * BF_K..(batch + 1) * BF_M * BF_K];
        let b_mat = &b_host[batch * BF_K * BF_N..(batch + 1) * BF_K * BF_N];
        want.extend(reference(a_mat, BF_M, BF_K, BF_N, |k, n| {
            b_mat[k * BF_N + n]
        }));
        tol.extend(
            abs_reference(a_mat, BF_M, BF_K, BF_N, |k, n| b_mat[k * BF_N + n])
                .into_iter()
                .map(|abs| bf16_gemm_tolerance(BF_K, abs)),
        );
    }
    check("bf16 batched", seed, &out, &want, Some(&tol), BF_N)
}

/// f32 `a[F_M, F_K] x b[F_K, F_N]`: rocBLAS, interleaved with the hipBLASLt
/// GEMMs on the same streams.
fn f32_and_check(seed: usize) -> Result<(), String> {
    let a_host = integer_values(F_M * F_K, seed + 11, F_HALF);
    let b_host = integer_values(F_K * F_N, seed + 3_000_003, F_HALF);
    let a = mlxcel_core::from_slice_f32(&a_host, &[F_M as i32, F_K as i32]);
    let b = mlxcel_core::from_slice_f32(&b_host, &[F_K as i32, F_N as i32]);
    let out = mlxcel_core::matmul(&a, &b);
    let want = reference(&a_host, F_M, F_K, F_N, |k, n| b_host[k * F_N + n]);
    check("f32", seed, &out, &want, None, F_N)
}

/// Single-threaded pin of the bound the bf16 checks above rely on (#2206).
/// Every row of `a` cancels to zero through a pair of opposite-sign terms
/// (`[h, -h, 0, ...]` with `h` in 1, 2, 4, 8), which on gfx11 WMMA leaves a
/// residue in every output, and `b` is all ones, so the exact result is zero
/// everywhere. The test passes whether or not the device is exact; it fails
/// if an element lands outside `bf16_gemm_tolerance`, and it prints how many
/// elements were inexact so a change in the hardware or library is visible.
#[test]
fn bf16_gemm_error_is_within_the_accumulation_bound() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let mut a_host = vec![0.0f32; BF_M * BF_K];
    for row in 0..BF_M {
        let h = (1 << (row % 4)) as f32;
        a_host[row * BF_K] = h;
        a_host[row * BF_K + 1] = -h;
    }
    let b_host = vec![1.0f32; BF_N * BF_K];
    let a = mlxcel_core::from_slice_f32(&a_host, &[BF_M as i32, BF_K as i32]);
    let b = mlxcel_core::from_slice_f32(&b_host, &[BF_N as i32, BF_K as i32]);
    let a = mlxcel_core::astype(&a, dtype::BFLOAT16);
    let b = mlxcel_core::astype(&b, dtype::BFLOAT16);
    let out = mlxcel_core::matmul(&a, &mlxcel_core::transpose(&b));
    let want = reference(&a_host, BF_M, BF_K, BF_N, |k, n| b_host[n * BF_K + k]);
    let tol: Vec<f32> = abs_reference(&a_host, BF_M, BF_K, BF_N, |k, n| b_host[n * BF_K + k])
        .into_iter()
        .map(|abs| bf16_gemm_tolerance(BF_K, abs))
        .collect();
    if let Err(e) = check("bf16 cancelling pairs", 0, &out, &want, Some(&tol), BF_N) {
        panic!("{e}");
    }
    let out = mlxcel_core::astype(&out, dtype::FLOAT32);
    let inexact = mlxcel_core::array_to_raw_bytes(&out)
        .chunks_exact(4)
        .filter(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != 0.0)
        .count();
    eprintln!(
        "bf16 cancelling pairs: {inexact} of {} outputs inexact, all within the bound",
        BF_M * BF_N
    );
}

/// Child body: `THREADS` threads, each on its own stream, released together
/// before the process has run any GPU matmul, so the first hipBLASLt GEMM of
/// the process is attempted by every thread at once and every later round
/// contends on the same pipe-cache keys.
#[test]
#[ignore]
fn child_hipblaslt_gemms_from_many_threads() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: child body runs only under the parent test");
        return;
    }
    assert!(
        on_rocm(),
        "the parent only spawns this child on a ROCm device"
    );

    let barrier = Barrier::new(THREADS);
    thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let barrier = &barrier;
                s.spawn(move || -> Result<(), String> {
                    let stream = new_thread_local_generation_stream();
                    install_thread_local_default_stream(stream.as_ref());
                    barrier.wait();
                    for round in 0..ROUNDS {
                        let seed = t * ROUNDS + round;
                        bf16_tn_and_check(seed).map_err(|e| format!("round {round}: {e}"))?;
                        bf16_batched_and_check(seed).map_err(|e| format!("round {round}: {e}"))?;
                        f32_and_check(seed).map_err(|e| format!("round {round}: {e}"))?;
                    }
                    Ok(())
                })
            })
            .collect();
        let errors: Vec<String> = handles
            .into_iter()
            .enumerate()
            .filter_map(|(t, h)| match h.join() {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(format!("thread {t}: {e}")),
                Err(_) => Some(format!("thread {t}: panicked")),
            })
            .collect();
        assert!(errors.is_empty(), "{}", errors.join("; "));
    });
    println!("{DONE_MARKER}");
}

struct ChildRun {
    success: bool,
    status: String,
    stdout: String,
    stderr: String,
}

/// Runs the child in a fresh process and returns its outcome. Panics if the
/// child outlives its budget.
fn run_child() -> ChildRun {
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
    for var in ROUTE_ENV {
        cmd.env_remove(var);
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
            "the child did not finish within {CHILD_BUDGET:?}\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
        )
    });
    ChildRun {
        success: status.success(),
        status: status.to_string(),
        stdout,
        stderr,
    }
}

/// Runs one child and returns what went wrong, if anything.
fn check_child() -> Option<String> {
    let run = run_child();
    let mut problems = Vec::new();
    if !run.success {
        problems.push(format!("exited with {}", run.status));
    }
    if !run.stdout.contains(DONE_MARKER) {
        problems.push(format!("did not print {DONE_MARKER:?}"));
    }
    let caps_lines = run.stderr.matches(CAPS_MARKER).count();
    if caps_lines != 1 {
        problems.push(format!(
            "printed {CAPS_MARKER:?} {caps_lines} times, expected exactly once"
        ));
    }
    if run.stderr.contains("Warning: hipBLASLt") || run.stderr.contains("Warning: rocBLAS") {
        problems.push("printed a hipBLASLt or rocBLAS warning".to_string());
    }
    if problems.is_empty() {
        None
    } else {
        Some(format!(
            "{}\n--- child stdout ---\n{}\n--- child stderr ---\n{}",
            problems.join(", "),
            run.stdout,
            run.stderr
        ))
    }
}

#[test]
fn hipblaslt_gemms_from_many_threads_in_fresh_processes() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let failures: Vec<String> = (0..CHILD_RUNS)
        .filter_map(|run| check_child().map(|problem| format!("child run {run}: {problem}")))
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {CHILD_RUNS} child runs failed:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}
