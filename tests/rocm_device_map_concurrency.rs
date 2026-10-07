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

//! The ROCm overlay's device table (`rocm::device()`) is safe to read while
//! another thread inserts into it (issue #2197).
//!
//! `device()` in `patches-rocm/mlx/backend/rocm/device.cpp` memoises one
//! `Device` per HIP device index in a process-global map. It used to do an
//! unlocked `find` and, on a miss, an unlocked `try_emplace`; it is called for
//! every primitive `gpu::eval` launches, from whichever thread evaluates, and
//! an insert happens whenever a stream is first created on a device. The map
//! and the lock are now leaked, the lookup takes a shared lock and a first
//! construction the unique lock after a re-check, `record_stream_error` and
//! `clear_all_encoders` go through the same lock, and `Device::clear_encoders`
//! destroys encoders only after `encoders_mtx_` is released. The HIP event
//! pool in `event.hip` is leaked as well, so no `hipEventDestroy` runs at exit
//! and a release during static teardown cannot touch a destroyed map.
//!
//! The insert race itself needs two GPUs. Every insert comes from a stream's
//! creation, and `mlx::core::new_stream` holds the `all_streams()` unique lock
//! around it, so two first inserts never overlap. On a single-GPU host the only
//! key is 0, inserted by the process's first stream before any thread can
//! evaluate, and every later call is a `find` on a map that no longer changes.
//! The reachable race is a first stream on GPU 1 (the insert) while another
//! thread evaluates on GPU 0 (a `find` walking the same buckets). Nothing in
//! production places a stream on GPU 1 yet (`--main-gpu` other than 0 is
//! rejected), so this is a guard for the multi-GPU work in #486 and #488.
//!
//! Both tests run their body in a fresh child process, `CHILD_RUNS` times, so
//! the first stream of the process (the index-0 insert) happens with every
//! thread already released from the barrier:
//!
//! - `streams_created_while_others_evaluate`: 8 threads, 16 rounds; in each
//!   round every thread creates and installs a new thread-local stream, then
//!   evaluates `arange_f32(0, 256, 1) * 2 + 1` four times, so `device()`
//!   lookups overlap other threads' stream creation and evaluation on GPU 0;
//! - `second_gpu_first_stream_while_first_gpu_evaluates`: skipped unless
//!   `gpu_device_count() >= 2`. 8 threads keep evaluating on GPU 0 while one
//!   thread creates and installs a thread-local stream on GPU 1 (the first and
//!   only insert of index 1 in the process), fills `ones` on that stream and
//!   evaluates the same check.
//!
//! Every output must be exactly `2 * i + 1` (or `1.0`); the child must exit 0,
//! print its done marker, print `[mlx-rocm] bound HIP device N:` exactly once
//! per index it used (one `Device` per index however many threads raced for
//! it), and print no `[mlx-rocm] releasing a HIP handle failed` line, which is
//! what a `hipEventDestroy` failing during static teardown looks like.
//!
//! Measured on gfx1151 (one GPU, so the second test skips there): the first
//! test passed 10 of 10 runs (100 children) with the fix and 10 of 10 with
//! `device.cpp` restored to main, as the reachability above predicts, so on a
//! single-GPU host it is a crash guard for the lookup path and the teardown
//! checks, not a reproduction of the insert race. The index-1 insert under
//! index-0 lookups has not been run anywhere yet. Skips on any other backend.
//! Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_device_map_concurrency
//! ```

#![cfg(feature = "rocm")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::dtype;
use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{
    gpu_device_count, install_thread_local_default_stream, new_thread_local_generation_stream,
    new_thread_local_stream_on_gpu,
};

/// Set by the parent on the children it spawns; a child body runs only when
/// it is present, so `--include-ignored` sweeps skip it.
const CHILD_ENV: &str = "MLXCEL_ROCM_DEVICE_MAP_CHILD";
const CHILD_SINGLE_GPU: &str = "child_streams_created_while_others_evaluate";
const CHILD_TWO_GPUS: &str = "child_second_gpu_first_stream_while_first_gpu_evaluates";
const CHILD_BUDGET: Duration = Duration::from_secs(120);
const CHILD_RUNS: usize = 10;
/// A child prints this once every thread has finished, so a child that
/// skipped or died before the check cannot pass as a success.
const DONE_MARKER: &str = "DEVICE MAP done";
/// Printed by `Device::Device` in `device.cpp` once per constructed device.
const BOUND_MARKER: &str = "[mlx-rocm] bound HIP device ";
/// Printed by `HipHandle::reset` in `utils.h` when a destroy fails, once per
/// handle type; at exit that is the event pool's `hipEventDestroy`.
const RELEASE_FAILED_MARKER: &str = "[mlx-rocm] releasing a HIP handle failed";

const THREADS: usize = 8;
const ROUNDS: usize = 16;
/// Evaluations per installed stream, so a thread that got its stream early
/// is still launching while the others create theirs.
const EVALS_PER_STREAM: usize = 4;
const LEN: i32 = 256;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// Compare `arr` element by element with `want(i)`, exactly.
fn check_exact(
    what: &str,
    arr: &mlxcel_core::MlxArray,
    want: impl Fn(usize) -> f32,
) -> Result<(), String> {
    mlxcel_core::try_eval(arr).map_err(|e| format!("{what}: eval failed: {e}"))?;
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    if bytes.len() != LEN as usize * 4 {
        return Err(format!(
            "{what}: expected {} output bytes, got {}",
            LEN * 4,
            bytes.len()
        ));
    }
    for (i, b) in bytes.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let expected = want(i);
        if got != expected {
            return Err(format!("{what}: element {i} is {got}, expected {expected}"));
        }
    }
    Ok(())
}

/// `arange(LEN) * 2 + 1` on the calling thread's default stream, checked
/// exactly. Three primitives, so three `device()` lookups per evaluation.
///
/// The constants are 8-byte host-written scalars (`multiply_scalar` and
/// `full_like`). Until #2213 was fixed, a scalar read by an elementwise
/// kernel while other threads evaluated on their own streams came back as a
/// value another scalar held earlier (found by this test, 9 of 10 runs with
/// and without the #2197 fix), and this file used full-length constants to
/// keep that failure out of what it guards; `tests/rocm_scalar_pool_concurrency.rs`
/// now guards it.
fn launch_and_check(what: &str) -> Result<(), String> {
    let inp = mlxcel_core::arange_f32(0.0, LEN as f32, 1.0);
    let doubled = mlxcel_core::multiply_scalar(&inp, 2.0);
    let ones = mlxcel_core::full_like(&doubled, 1.0);
    let out = mlxcel_core::add(&doubled, &ones);
    check_exact(what, &out, |i| 2.0 * i as f32 + 1.0)
}

/// Child body: `THREADS` threads behind a barrier, each creating and
/// installing a new stream per round and evaluating on it while the others
/// are still creating theirs.
#[test]
#[ignore]
fn child_streams_created_while_others_evaluate() {
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
                    barrier.wait();
                    for round in 0..ROUNDS {
                        let stream = new_thread_local_generation_stream();
                        install_thread_local_default_stream(stream.as_ref());
                        for eval in 0..EVALS_PER_STREAM {
                            launch_and_check(&format!("thread {t} round {round} eval {eval}"))?;
                        }
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

/// Child body: `THREADS` threads evaluate on GPU 0 without pause while one
/// thread creates the process's first stream on GPU 1, the insert of index 1
/// under the other threads' lookups of index 0.
#[test]
#[ignore]
fn child_second_gpu_first_stream_while_first_gpu_evaluates() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: child body runs only under the parent test");
        return;
    }
    assert!(
        on_rocm(),
        "the parent only spawns this child on a ROCm device"
    );
    assert!(
        gpu_device_count() >= 2,
        "the parent only spawns this child with two or more GPUs"
    );

    let barrier = Barrier::new(THREADS + 1);
    let stop = AtomicBool::new(false);
    thread::scope(|s| {
        let evaluators: Vec<_> = (0..THREADS)
            .map(|t| {
                let barrier = &barrier;
                let stop = &stop;
                s.spawn(move || -> Result<(), String> {
                    let stream = new_thread_local_generation_stream();
                    install_thread_local_default_stream(stream.as_ref());
                    barrier.wait();
                    let mut eval = 0usize;
                    loop {
                        launch_and_check(&format!("gpu 0 thread {t} eval {eval}"))?;
                        eval += 1;
                        if stop.load(Ordering::Acquire) {
                            return Ok(());
                        }
                    }
                })
            })
            .collect();
        let second_gpu = {
            let barrier = &barrier;
            s.spawn(move || -> Result<(), String> {
                barrier.wait();
                let tls = new_thread_local_stream_on_gpu(1)
                    .map_err(|e| format!("gpu 1: creating the stream handle failed: {e}"))?;
                // Resolving the handle on this thread creates the stream, and
                // with it the process's first `Device` for index 1.
                install_thread_local_default_stream(Some(&tls));
                let stream = mlxcel_core::stream_from_thread_local_stream(&tls);
                for round in 0..ROUNDS {
                    let ones = mlxcel_core::ones_stream(&[LEN], dtype::FLOAT32, &stream);
                    check_exact(&format!("gpu 1 round {round} ones"), &ones, |_| 1.0)?;
                    launch_and_check(&format!("gpu 1 thread round {round}"))?;
                }
                Ok(())
            })
        };
        let second_gpu_result = second_gpu.join();
        stop.store(true, Ordering::Release);
        let mut errors: Vec<String> = evaluators
            .into_iter()
            .enumerate()
            .filter_map(|(t, h)| match h.join() {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(format!("evaluator {t}: {e}")),
                Err(_) => Some(format!("evaluator {t}: panicked")),
            })
            .collect();
        match second_gpu_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => errors.push(format!("second gpu: {e}")),
            Err(_) => errors.push("second gpu: panicked".to_string()),
        }
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

/// Runs `child_test` in a fresh process and returns its outcome. Panics if
/// the child outlives its budget.
fn run_child(child_test: &str) -> ChildRun {
    let exe = std::env::current_exe().expect("path of this test binary");
    let mut child = Command::new(exe)
        .args([
            child_test,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
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

/// Runs one child and returns what went wrong, if anything. `bound_indices`
/// are the HIP device indices the child must have constructed exactly once
/// each.
fn check_child(child_test: &str, bound_indices: &[i32]) -> Option<String> {
    let run = run_child(child_test);
    let mut problems = Vec::new();
    if !run.success {
        problems.push(format!("exited with {}", run.status));
    }
    if !run.stdout.contains(DONE_MARKER) {
        problems.push(format!("did not print {DONE_MARKER:?}"));
    }
    for index in bound_indices {
        let marker = format!("{BOUND_MARKER}{index}:");
        let bound = run.stderr.matches(marker.as_str()).count();
        if bound != 1 {
            problems.push(format!(
                "printed {marker:?} {bound} times, expected exactly once"
            ));
        }
    }
    if run.stderr.contains(RELEASE_FAILED_MARKER) {
        problems.push(format!(
            "printed {RELEASE_FAILED_MARKER:?}: a HIP handle was destroyed at teardown and the destroy failed"
        ));
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

fn run_children(child_test: &str, bound_indices: &[i32]) {
    let failures: Vec<String> = (0..CHILD_RUNS)
        .filter_map(|run| {
            check_child(child_test, bound_indices)
                .map(|problem| format!("child run {run}: {problem}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {CHILD_RUNS} child runs failed:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

#[test]
fn streams_created_while_others_evaluate() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    run_children(CHILD_SINGLE_GPU, &[0]);
}

#[test]
fn second_gpu_first_stream_while_first_gpu_evaluates() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let count = gpu_device_count();
    if count < 2 {
        eprintln!(
            "skipping: the insert race needs a second GPU and this host reports {count}; the index-1 insert under index-0 lookups was not exercised here"
        );
        return;
    }
    run_children(CHILD_TWO_GPUS, &[0, 1]);
}
