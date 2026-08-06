// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Guard: the CUDA `mlxcel-core` suite must run single threaded (issue #1048).
//!
//! ## Why this exists
//!
//! Driving MLX from the many host threads libtest spawns by default is not safe
//! on the CUDA backend. Measured on GB10 (sm_121) at MLX pin `2c46b953`, three
//! full runs of `cargo test --release --features cuda -p mlxcel-core --lib` on
//! an idle machine:
//!
//! | Threads | `MLX_USE_CUDA_GRAPHS` | Outcome |
//! |---|---|---|
//! | default (20) | on | SIGABRT, `cudaStreamEndCapture ... previous error during capture` |
//! | `--test-threads=1` | on | ran to a verdict in 88s, no abort |
//! | default (20) | `0` | SIGABRT, `cuLaunchKernelEx ... invalid argument` |
//!
//! The third row is the one that decides the fix. Turning graph capture off does
//! not rescue the parallel run, it only moves which CUDA call reports the
//! failure, so capture is the symptom and concurrency is the cause. Serializing
//! is therefore the remedy, and `MLX_USE_CUDA_GRAPHS=0` is not. The parallel run
//! aborts at a different test and with a different message each time, which is
//! what makes the raw SIGABRT so expensive to read: it looks like whichever test
//! happened to be running is broken.
//!
//! Serialization costs almost nothing here. 1410 tests take 88 seconds.
//!
//! ## What this does
//!
//! Fails, by name, when the suite is running with more than one test thread, and
//! says what to run instead. It cannot make the run safe; it replaces an
//! anonymous abort inside MLX with a named, actionable failure.
//!
//! Because this is an ordinary test, a narrowed run (`cargo test ... --lib
//! paged_v2::launch`) filters it out and stays parallel, which is correct:
//! scoped subsets of the suite pass parallel, and only whole-suite runs abort.
//!
//! The diagnosis is also written straight to the process stderr handle rather
//! than through `eprintln!`. libtest captures the print macros and replays them
//! in its end-of-run summary, and in a parallel run there may be no end of run:
//! the abort takes the process down first. Writing to the handle bypasses the
//! capture so the message survives.
//!
//! Set `MLXCEL_ALLOW_PARALLEL_CUDA_TESTS=1` to downgrade this to a printed
//! warning, for the case where someone is deliberately reproducing the abort.

use std::io::Write;

/// Environment escape hatch: warn instead of failing.
const ALLOW_ENV: &str = "MLXCEL_ALLOW_PARALLEL_CUDA_TESTS";

/// libtest's own environment variable for the thread count. Read here for the
/// same reason libtest reads it: `--test-threads` on the command line is not the
/// only way the count gets set.
const THREADS_ENV: &str = "RUST_TEST_THREADS";

/// The thread count the invocation asked for, or `None` when it did not ask.
///
/// Mirrors libtest's own precedence: an explicit `--test-threads` on the command
/// line wins over `RUST_TEST_THREADS`, and `None` means neither was given, so
/// libtest will fall back to `available_parallelism()`.
fn declared_test_threads(args: &[String], env: Option<&str>) -> Option<usize> {
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--test-threads=") {
            return value.trim().parse().ok();
        }
        if arg == "--test-threads" {
            return args.next().and_then(|value| value.trim().parse().ok());
        }
    }
    env.and_then(|value| value.trim().parse().ok())
}

/// How many test threads libtest is actually going to use.
fn effective_test_threads(args: &[String], env: Option<&str>) -> usize {
    declared_test_threads(args, env).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

#[test]
fn the_cuda_test_suite_must_run_single_threaded() {
    let args: Vec<String> = std::env::args().collect();
    let env = std::env::var(THREADS_ENV).ok();
    let threads = effective_test_threads(&args, env.as_deref());
    if threads <= 1 {
        return;
    }

    let message = format!(
        "the CUDA mlxcel-core suite is running with {threads} test threads and will abort (issue #1048).\n\n\
         Driving MLX from many host threads is unsafe on the CUDA backend. Measured on GB10 at MLX \
         pin 2c46b953: the default 20-thread run dies with SIGABRT from \
         `cudaStreamEndCapture ... previous error during capture`, at a different test each time, \
         while the same binary under --test-threads=1 completes in 88 seconds.\n\n\
         Do not reach for MLX_USE_CUDA_GRAPHS=0. With graph capture disabled the 20-thread run \
         still aborts, as `cuLaunchKernelEx ... invalid argument`; capture selects the symptom, \
         concurrency is the cause.\n\n\
         Run the gate instead:\n  \
         make verify-test-cuda\n  \
         cargo test --workspace --profile test-fast --features cuda --no-fail-fast -- --test-threads=1\n\n\
         Narrowed runs are fine parallel; it is whole-suite runs that abort. Set {ALLOW_ENV}=1 to \
         downgrade this to a warning."
    );

    let allowed = std::env::var(ALLOW_ENV)
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on" || v == "yes"
        })
        .unwrap_or(false);

    // Direct handle write, not `eprintln!`: see the module docs. A parallel run
    // usually aborts before libtest prints its captured output.
    let prefix = if allowed { "warning" } else { "error" };
    let _ = writeln!(std::io::stderr(), "{prefix}: {message}");

    assert!(allowed, "{message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn both_spellings_of_the_command_line_flag_are_read() {
        assert_eq!(
            declared_test_threads(&args(&["bin", "--test-threads=1"]), None),
            Some(1)
        );
        assert_eq!(
            declared_test_threads(&args(&["bin", "--test-threads", "1"]), None),
            Some(1)
        );
        assert_eq!(
            declared_test_threads(&args(&["bin", "--test-threads=8"]), None),
            Some(8)
        );
    }

    #[test]
    fn the_command_line_flag_wins_over_the_environment() {
        // libtest resolves it this way, so a guard that preferred the
        // environment would fail the one invocation that is actually correct.
        assert_eq!(
            declared_test_threads(&args(&["bin", "--test-threads=1"]), Some("16")),
            Some(1)
        );
        assert_eq!(declared_test_threads(&args(&["bin"]), Some("1")), Some(1));
    }

    #[test]
    fn an_invocation_that_says_nothing_reports_nothing() {
        assert_eq!(
            declared_test_threads(&args(&["bin", "--nocapture"]), None),
            None
        );
        // A malformed value is not a claim of serialization either.
        assert_eq!(
            declared_test_threads(&args(&["bin", "--test-threads=all"]), None),
            None
        );
        assert_eq!(
            declared_test_threads(&args(&["bin", "--test-threads"]), None),
            None
        );
    }

    #[test]
    fn a_silent_invocation_falls_back_to_the_machines_parallelism() {
        let expected = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert_eq!(effective_test_threads(&args(&["bin"]), None), expected);
        // And an explicit count is used verbatim, whatever the machine has.
        assert_eq!(
            effective_test_threads(&args(&["bin", "--test-threads=1"]), None),
            1
        );
    }
}
