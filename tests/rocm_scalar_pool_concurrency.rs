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

//! Host-written scalars served by the ROCm overlay's 8-byte `SmallSizePool`
//! keep their value while other threads evaluate on their own streams
//! (issue #2213).
//!
//! `SmallSizePool` in `patches-rocm/mlx/backend/rocm/allocator.cpp` serves
//! every allocation of at most 8 bytes: `array(2.0f)`, `full(shape, value)`,
//! a one-token index, anything the host writes and a kernel then reads
//! through `binary_vs`, `copy_s` and friends. It used to carve its slots 8
//! bytes apart, so 16 live scalars of 16 threads shared one 128-byte line.
//! When a kernel on another queue was reading a neighbour in that line while
//! this thread rewrote its slot and launched, the launch read the line's old
//! contents: the value the slot's previous occupant held. Slots are now one
//! cache line apart, so no running kernel can hold a line that another
//! thread's host write is about to change.
//!
//! The child body is the issue's reproduction: `THREADS` threads, each on
//! its own stream, behind a barrier, evaluating
//! `arange_f32(0, 256, 1) * k + c` with `k` from `multiply_scalar` and `c`
//! from `full_like`, `ROUNDS * EVALS_PER_ROUND` times, every element checked
//! exactly. Every `k` and `c` is drawn from one process-wide increasing
//! counter, so a wrong output names whose scalar the kernel read and whether
//! that scalar was written before or after the one it should have read: a
//! stale read shows an earlier value, a slot handed to two live arrays or a
//! kernel still reading a recycled slot shows a later one. The parent runs
//! the child in a fresh process `CHILD_RUNS` times.
//!
//! Measured on gfx1151 (ROCm 7.15): with 8-byte slots the child failed 10 of
//! 10 runs on main (75 wrong scalars, every one written earlier than the
//! expected one, 0 later) and 18 of 20 with the stride set back to 8 on this
//! branch (93 wrong, all earlier), and a live-slot assertion in the pool
//! never fired across 5 failing runs; with 128-byte slots it passed 20 of 20
//! and the parent passed. Skips on any other backend. Run on a ROCm host
//! with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_scalar_pool_concurrency
//! ```

#![cfg(feature = "rocm")]

use std::collections::HashMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
};

/// Set by the parent on the children it spawns; a child body runs only when
/// it is present, so `--include-ignored` sweeps skip it.
const CHILD_ENV: &str = "MLXCEL_ROCM_SCALAR_POOL_CHILD";
const CHILD_TEST: &str = "child_scalars_keep_their_values_across_threads";
const CHILD_BUDGET: Duration = Duration::from_secs(120);
const CHILD_RUNS: usize = 10;
/// A child prints this once every thread has finished, so a child that
/// skipped or died before the check cannot pass as a success.
const DONE_MARKER: &str = "SCALAR POOL done";

const THREADS: usize = 8;
const ROUNDS: usize = 16;
const EVALS_PER_ROUND: usize = 4;
const LEN: i32 = 256;

/// Every scalar written by the child comes from this counter, so the order
/// of two values is the order their host writes happened in (to within the
/// few microseconds between the draw and the write).
static NEXT_VALUE: AtomicU32 = AtomicU32::new(2);

/// Who drew each value: (thread, round, eval, "k" or "c").
static OWNERS: Mutex<Option<HashMap<u32, (usize, usize, usize, &'static str)>>> = Mutex::new(None);

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn draw(t: usize, round: usize, eval: usize, which: &'static str) -> u32 {
    let v = NEXT_VALUE.fetch_add(1, Ordering::SeqCst);
    let mut owners = OWNERS.lock().expect("owners lock");
    owners
        .get_or_insert_with(HashMap::new)
        .insert(v, (t, round, eval, which));
    v
}

fn describe(value: f32, k: u32, c: u32) -> String {
    if value.fract() != 0.0 || value < 0.0 {
        return format!("{value} (not a drawn value)");
    }
    let v = value as u32;
    let owners = OWNERS.lock().expect("owners lock");
    let owner = owners
        .as_ref()
        .and_then(|m| m.get(&v))
        .map(|(t, r, e, w)| format!("thread {t} round {r} eval {e} {w}"))
        .unwrap_or_else(|| "not drawn by any thread".to_string());
    let when = if v > k.max(c) {
        "written LATER than this evaluation's scalars"
    } else if v < k.min(c) {
        "written EARLIER than this evaluation's scalars"
    } else {
        "this evaluation's other scalar"
    };
    format!("{v} ({owner}; {when})")
}

/// `arange(LEN) * k + c` on the calling thread's default stream, with `k`
/// and `c` as host-written 8-byte scalars (`multiply_scalar` materialises
/// `k` as a one-element array, `full_like` reads `c` through `Full`), every
/// element checked exactly.
fn launch_and_check(t: usize, round: usize, eval: usize) -> Result<(), String> {
    let k = draw(t, round, eval, "k");
    let c = draw(t, round, eval, "c");
    let what = format!("thread {t} round {round} eval {eval} (k = {k}, c = {c})");
    let inp = mlxcel_core::arange_f32(0.0, LEN as f32, 1.0);
    let scaled = mlxcel_core::multiply_scalar(&inp, k as f32);
    let offset = mlxcel_core::full_like(&scaled, c as f32);
    let out = mlxcel_core::add(&scaled, &offset);
    mlxcel_core::try_eval(&out).map_err(|e| format!("{what}: eval failed: {e}"))?;
    let bytes = mlxcel_core::array_to_raw_bytes(&out);
    if bytes.len() != LEN as usize * 4 {
        return Err(format!(
            "{what}: expected {} output bytes, got {}",
            LEN * 4,
            bytes.len()
        ));
    }
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    for (i, &g) in got.iter().enumerate() {
        let expected = (i as u32 * k + c) as f32;
        if g != expected {
            // Solve the read scalars from the first two elements.
            let c_read = got[0];
            let k_read = got[1] - got[0];
            return Err(format!(
                "{what}: element {i} is {g}, expected {expected}; kernel read k = {} and c = {}",
                describe(k_read, k, c),
                describe(c_read, k, c)
            ));
        }
    }
    Ok(())
}

/// Child body: `THREADS` threads, each on its own stream, behind a barrier,
/// evaluating `ROUNDS * EVALS_PER_ROUND` times.
#[test]
#[ignore]
fn child_scalars_keep_their_values_across_threads() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: child body runs only under the parent test");
        return;
    }
    assert!(
        on_rocm(),
        "the parent only spawns this child on a ROCm device"
    );

    let barrier = Barrier::new(THREADS);
    let errors: Vec<String> = thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let barrier = &barrier;
                s.spawn(move || -> Vec<String> {
                    let stream = new_thread_local_generation_stream();
                    install_thread_local_default_stream(stream.as_ref());
                    barrier.wait();
                    let mut errors = Vec::new();
                    for round in 0..ROUNDS {
                        for eval in 0..EVALS_PER_ROUND {
                            if let Err(e) = launch_and_check(t, round, eval) {
                                errors.push(e);
                            }
                        }
                    }
                    errors
                })
            })
            .collect();
        handles
            .into_iter()
            .enumerate()
            .flat_map(|(t, h)| match h.join() {
                Ok(errors) => errors,
                Err(_) => vec![format!("thread {t}: panicked")],
            })
            .collect()
    });
    assert!(
        errors.is_empty(),
        "{} mismatches:\n{}",
        errors.len(),
        errors.join("\n")
    );
    println!("{DONE_MARKER}");
}

struct ChildRun {
    success: bool,
    status: String,
    stdout: String,
    stderr: String,
}

/// Runs the child body in a fresh process and returns its outcome. Panics
/// if the child outlives its budget.
fn run_child() -> ChildRun {
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
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child test process");

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

fn check_child() -> Option<String> {
    let run = run_child();
    let mut problems = Vec::new();
    if !run.success {
        problems.push(format!("exited with {}", run.status));
    }
    if !run.stdout.contains(DONE_MARKER) {
        problems.push(format!("did not print {DONE_MARKER:?}"));
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
fn scalars_keep_their_values_across_threads() {
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
