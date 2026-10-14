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

//! The ROCm `Device` creates its rocBLAS handle once and binds the stream
//! under a lock (issue #2198).
//!
//! The overlay's `Device` used to create the handle lazily behind a plain
//! `bool` that was set before `rocblas_create_handle` ran, so a second thread
//! arriving in that window saw the flag and took back a null handle, and
//! every GEMM rebound the one shared handle to the caller's stream with no
//! lock, so one thread's GEMM could be enqueued on another thread's stream.
//! The server runs forwards on several worker threads that share one
//! `Device`, each on its own stream (scheduler, embedding, rerank, audio), so
//! both windows are reachable. Every rocBLAS call now goes through
//! `Device::acquire_rocblas(stream)`, which creates the handle under a mutex
//! on first use and returns a lease that holds that mutex while the GEMM is
//! enqueued.
//!
//! The init window is once per process, so the parent runs the child in a
//! fresh process `CHILD_RUNS` times. Each child starts `THREADS` threads that
//! install their own stream, wait on a barrier before any GPU matmul has run
//! in the process, then run `ROUNDS` f32 `[M, K] x [K, N]` matmuls with
//! integer inputs. Integers in `-4..=4` over `K = 128` keep every partial sum
//! far below 2^24, so the GPU result must equal the host `i32` reference
//! exactly, whichever rocBLAS solution runs. f32 GEMMs route to rocBLAS
//! (hipBLASLt takes only bf16 and f16), so nothing steers the route; the
//! hipBLASLt side has its own test, `rocm_hipblaslt_concurrency.rs` (issue
//! #2200).
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_rocblas_handle_concurrency
//! ```

#![cfg(feature = "rocm")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
};

/// Set by the parent on the children it spawns; the child body runs only
/// when it is present, so `--include-ignored` sweeps skip it.
const CHILD_ENV: &str = "MLXCEL_ROCM_ROCBLAS_HANDLE_CHILD";
const CHILD_TEST: &str = "child_first_gemm_from_many_threads";
const CHILD_BUDGET: Duration = Duration::from_secs(120);
const CHILD_RUNS: usize = 10;
/// The child prints this once every thread has finished, so a child that
/// skipped or died before the check cannot pass as a success.
const DONE_MARKER: &str = "ROCBLAS_HANDLE done";

const THREADS: usize = 8;
const ROUNDS: usize = 33;
const M: usize = 64;
const K: usize = 128;
const N: usize = 64;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// Deterministic integers in `-4..=4` from a per-thread, per-round seed.
fn integer_values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 31 + seed * 17 + 7) % 9) as i32 - 4) as f32)
        .collect()
}

/// Host `i32` reference of `a[M, K] x b[K, N]`.
fn reference(a: &[f32], b: &[f32]) -> Vec<i32> {
    let mut out = vec![0i32; M * N];
    for m in 0..M {
        for k in 0..K {
            let av = a[m * K + k] as i32;
            for n in 0..N {
                out[m * N + n] += av * b[k * N + n] as i32;
            }
        }
    }
    out
}

/// One f32 matmul on the calling thread's default stream, checked exactly
/// against the host reference.
fn matmul_and_check(seed: usize) -> Result<(), String> {
    let a_host = integer_values(M * K, seed);
    let b_host = integer_values(K * N, seed + 1_000_003);
    let a = mlxcel_core::from_slice_f32(&a_host, &[M as i32, K as i32]);
    let b = mlxcel_core::from_slice_f32(&b_host, &[K as i32, N as i32]);
    let out = mlxcel_core::matmul(&a, &b);
    mlxcel_core::try_eval(&out).map_err(|e| format!("seed {seed}: eval failed: {e}"))?;
    let bytes = mlxcel_core::array_to_raw_bytes(&out);
    if bytes.len() != M * N * 4 {
        return Err(format!(
            "seed {seed}: expected {} output bytes, got {}",
            M * N * 4,
            bytes.len()
        ));
    }
    let want = reference(&a_host, &b_host);
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if got != want[i] as f32 {
            return Err(format!(
                "seed {seed}: element {i} (row {}, col {}) is {got}, expected {}",
                i / N,
                i % N,
                want[i]
            ));
        }
    }
    Ok(())
}

/// Child body: `THREADS` threads, each on its own stream, released together
/// before the process has run any GPU matmul, so the first rocBLAS GEMM of
/// the process is attempted by every thread at once.
#[test]
#[ignore]
fn child_first_gemm_from_many_threads() {
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
                        matmul_and_check(t * ROUNDS + round)
                            .map_err(|e| format!("round {round}: {e}"))?;
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
    if run.stderr.contains("Warning: rocBLAS") {
        problems.push("printed a rocBLAS warning".to_string());
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
fn first_gemm_from_many_threads_in_fresh_processes() {
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
