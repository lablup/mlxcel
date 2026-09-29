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

//! ROCm GPU failures reach Rust as errors instead of NaN or a hang
//! (issue #1804).
//!
//! Two failure classes, each provoked on purpose through the production
//! custom-kernel path (`mlxcel_core::rocm_faults`) and observed through the
//! production `try_eval`:
//!
//! - a launch HIP rejects synchronously (a 2048-thread block): the eval
//!   returns `Err` naming the launch, and the device stays usable;
//! - an asynchronous queue fault (a write 2^40 bytes past the buffer): the
//!   eval returns `Err` within seconds instead of spinning forever, and so
//!   does every later eval, because the HIP runtime rejects every call for
//!   the rest of the process after a queue fault.
//!
//! The fault test runs in a child process (this binary, re-invoked on an
//! ignored test) for two reasons: the fault kills the device for the whole
//! process, and HIP's shutdown then hangs waiting on host callbacks the
//! faulted queue never runs, so the child leaves through `_exit`. The parent
//! kills a child that does not answer in time, which is what the pre-fix hang
//! looks like. Without the fix the launch test passes an unwritten buffer as
//! a value and the fault test hangs (checked by reverting the overlay).
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_gpu_faults
//! ```

#![cfg(feature = "rocm")]

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::rocm_faults::{RocmFaultKind, fault_probe_array};

/// Set by the parent test on the child it spawns; the child body runs only
/// when it is present, so `--ignored` sweeps do not fault a shared process.
const CHILD_ENV: &str = "MLXCEL_ROCM_FAULT_CHILD";
const CHILD_TEST: &str = "child_out_of_bounds_write";
/// Long enough for a cold hiprtc compile of the probe kernel plus the fault
/// itself (measured under a second on gfx1151), short enough that the
/// pre-fix behaviour, an unbounded spin, is unmistakable.
const CHILD_BUDGET: Duration = Duration::from_secs(120);
/// The bound the issue asks for: a fault fails the waiting eval within this.
const FAULT_REPORT_BOUND: Duration = Duration::from_secs(30);

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn ones_f32(n: i32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::ones(&[n], mlxcel_core::dtype::FLOAT32)
}

#[test]
fn oversized_block_launch_fails_the_eval_and_leaves_the_device_usable() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let probe =
        fault_probe_array(RocmFaultKind::OversizedBlock).expect("probe array builds on ROCm");
    let msg = mlxcel_core::try_eval(&probe)
        .expect_err("a 2048-thread block must be rejected at launch")
        .to_string();
    // hipModuleLaunchKernel reports the oversized block as `invalid argument`
    // (hipErrorInvalidValue) on ROCm 7.15, where hipLaunchKernel says
    // `invalid configuration argument`; either names the rejected launch.
    assert!(
        msg.contains("hipModuleLaunchKernel")
            && (msg.contains("invalid argument") || msg.contains("invalid configuration")),
        "the error must name the rejected launch, got: {msg}"
    );

    // A rejected launch leaves the device usable: a plain op still evaluates
    // and produces its values.
    let ones = ones_f32(8);
    mlxcel_core::try_eval(&ones).expect("the device stays usable after a rejected launch");
    let bytes = mlxcel_core::array_to_raw_bytes(&ones);
    assert_eq!(bytes.len(), 8 * 4);
    for value in bytes.chunks_exact(4) {
        assert_eq!(
            f32::from_le_bytes([value[0], value[1], value[2], value[3]]),
            1.0
        );
    }
}

/// One line the child prints per evaluation it timed.
#[derive(Debug)]
struct ChildReport {
    elapsed: Duration,
    error: Option<String>,
}

fn parse_report(stdout: &str, label: &str) -> ChildReport {
    let prefix = format!("FAULT_PROBE {label} ms=");
    // The harness prints `test <name> ... ` without a newline before the
    // child's first line, so the marker is searched for anywhere in a line.
    let line = stdout
        .lines()
        .find(|line| line.contains(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in the child's stdout:\n{stdout}"));
    let start = line.find(&prefix).expect("prefix located") + prefix.len();
    let rest = &line[start..];
    let (ms, outcome) = rest.split_once(' ').expect("report line has an outcome");
    let elapsed = Duration::from_millis(ms.parse().expect("elapsed milliseconds"));
    let error = outcome.strip_prefix("err=").map(str::to_string);
    assert!(
        error.is_some() || outcome == "ok",
        "unexpected outcome `{outcome}` in child line: {line}"
    );
    ChildReport { elapsed, error }
}

#[test]
fn out_of_bounds_write_fails_the_eval_within_bounded_time() {
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
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child test process");

    // Drain both pipes on their own threads so a chatty child (the HIP
    // runtime prints a queue dump on a fault) can never block on a full pipe.
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
        if let Some(status) = child.try_wait().expect("poll the child") {
            break Some(status);
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
            "the child did not finish within {CHILD_BUDGET:?}: the fault hung the eval instead of failing it\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
        )
    });
    assert!(
        status.success(),
        "child exited with {status}\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}"
    );

    let first = parse_report(&stdout, "first");
    let error = first.error.as_deref().unwrap_or_else(|| {
        panic!("the eval that waited on the fault returned Ok\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}")
    });
    assert!(
        error.contains("hipErrorIllegalAddress") || error.contains("illegal memory access"),
        "the error must name the device fault, got: {error}"
    );
    assert!(
        first.elapsed < FAULT_REPORT_BOUND,
        "the fault took {:?} to be reported, more than {FAULT_REPORT_BOUND:?}",
        first.elapsed
    );

    // The device is dead for the process after a queue fault: a later eval
    // fails too, and it fails fast rather than waiting for work that will
    // never run.
    let second = parse_report(&stdout, "second");
    assert!(
        second.error.is_some(),
        "an eval after the fault returned Ok on a device that rejects every call"
    );
    assert!(
        second.elapsed < FAULT_REPORT_BOUND,
        "the eval after the fault took {:?} to fail, more than {FAULT_REPORT_BOUND:?}",
        second.elapsed
    );
}

/// The body of `out_of_bounds_write_fails_the_eval_within_bounded_time`, run
/// in the child process it spawns. Prints one `FAULT_PROBE <label> ms=<n>
/// <ok|err=...>` line per timed evaluation, then leaves through `_exit`.
#[test]
#[ignore = "spawned by out_of_bounds_write_fails_the_eval_within_bounded_time; the fault kills the device for the whole process"]
fn child_out_of_bounds_write() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: run through out_of_bounds_write_fails_the_eval_within_bounded_time");
        return;
    }
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }

    fn report(label: &str, started: Instant, outcome: Result<(), impl std::fmt::Display>) {
        let ms = started.elapsed().as_millis();
        match outcome {
            Ok(()) => println!("FAULT_PROBE {label} ms={ms} ok"),
            Err(err) => println!(
                "FAULT_PROBE {label} ms={ms} err={}",
                err.to_string().replace('\n', " ")
            ),
        }
    }

    let probe =
        fault_probe_array(RocmFaultKind::OutOfBoundsWrite).expect("probe array builds on ROCm");
    let started = Instant::now();
    report("first", started, mlxcel_core::try_eval(&probe));

    let ones = ones_f32(8);
    let started = Instant::now();
    report("second", started, mlxcel_core::try_eval(&ones));

    // HIP's teardown waits on host callbacks the faulted queue never runs
    // (measured: a process with a pending callback hangs in exit), so leave
    // without running it. `_exit` skips the atexit handlers and static
    // destructors that would reach it; the buffered report goes out first.
    let _ = std::io::stdout().flush();
    // SAFETY: `_exit` only ends the process; nothing after it runs, and the
    // only state worth flushing (stdout) was flushed on the line above.
    unsafe { libc::_exit(0) }
}
