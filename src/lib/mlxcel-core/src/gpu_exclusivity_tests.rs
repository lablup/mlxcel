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

//! Guard against a second `mlxcel-core` test binary sharing the GPU (issue #1008).
//!
//! ## Why this exists
//!
//! Two copies of this suite running at once on one Metal device do not merely
//! contend, they corrupt each other. Measured over six rounds of two concurrent
//! runs in separate working trees, 7 of 12 runs aborted:
//!
//! ```text
//! libc++abi: terminating due to uncaught exception of type std::runtime_error:
//!   [metal::CommandEncoder] Failed to make new command queue.
//! libc++abi: terminating due to uncaught exception of type std::runtime_error:
//!   [METAL] Command buffer execution failed: Internal Error (00000206:Internal Error).
//! ```
//!
//! The aborts are not the dangerous part. Runs that *survived* reported 3 and 4
//! failures against the single known pre-existing failure a quiet machine
//! produces, so a concurrent run can finish and name tests as broken that are
//! fine. A crash is self-evidently invalid; a failure count is not, and it costs
//! someone an investigation into a test that was never wrong.
//!
//! Capping intra-process parallelism does not help: the same experiment aborts at
//! `--test-threads=4` and `--test-threads=8`. The conflict is between processes
//! sharing one device, not between threads inside one. `make test` already passes
//! `--test-threads=1`, which orders tests within a process and does nothing here.
//!
//! ## What this does
//!
//! Fails, loudly and by name, when another test binary of this crate is running.
//! It deliberately does not try to make concurrent suites work, and it does not
//! serialize them; it makes the situation legible so the other failures in the
//! same summary are read as suspect rather than as regressions.
//!
//! Set `MLXCEL_ALLOW_CONCURRENT_GPU_TESTS=1` to downgrade it to a printed warning,
//! for the case where someone knowingly accepts unreliable results.

/// Environment escape hatch: warn instead of failing.
const ALLOW_ENV: &str = "MLXCEL_ALLOW_CONCURRENT_GPU_TESTS";

/// Substring identifying a compiled test binary of this crate. Cargo names them
/// `<target-dir>/<profile>/deps/mlxcel_core-<hash>`, so this matches a sibling
/// run in any working tree, which is exactly the case that corrupts results.
const TEST_BINARY_MARKER: &str = "/deps/mlxcel_core-";

/// Paths of other running test binaries of this crate, excluding this process.
///
/// macOS only. The failure this guards against is Metal-specific, and the
/// process listing is platform-specific, so elsewhere this reports nothing and
/// the test passes.
#[cfg(target_os = "macos")]
fn other_test_binaries() -> Vec<String> {
    use std::process::Command;

    let me = std::process::id();
    let Ok(out) = Command::new("/bin/ps").args(["-Ao", "pid=,comm="]).output() else {
        // If the listing is unavailable the guard abstains rather than failing
        // the suite for an unrelated reason.
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let (pid, comm) = line.split_once(char::is_whitespace)?;
            let pid: u32 = pid.trim().parse().ok()?;
            if pid == me {
                return None;
            }
            let comm = comm.trim();
            comm.contains(TEST_BINARY_MARKER)
                .then(|| format!("pid {pid}: {comm}"))
        })
        .collect()
}

#[cfg(not(target_os = "macos"))]
fn other_test_binaries() -> Vec<String> {
    Vec::new()
}

#[test]
fn no_other_mlxcel_core_test_binary_is_sharing_the_gpu() {
    let others = other_test_binaries();
    if others.is_empty() {
        return;
    }

    let detail = others.join("\n  ");
    let message = format!(
        "another mlxcel-core test binary is running and sharing this GPU (issue #1008):\n  \
         {detail}\n\n\
         Concurrent suites on one Metal device corrupt each other. Measured: 7 of 12 \
         concurrent runs aborted outright, and runs that completed reported 3 to 4 \
         failures against the 1 a quiet machine produces. Treat every other failure in \
         this summary as unverified until the suite is re-run alone.\n\n\
         Capping --test-threads does not help; the conflict is between processes. \
         Wait for the other run, or set {ALLOW_ENV}=1 to downgrade this to a warning."
    );

    let allowed = std::env::var(ALLOW_ENV)
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on" || v == "yes"
        })
        .unwrap_or(false);

    assert!(allowed, "{message}");
    eprintln!("warning: {message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marker_matches_a_cargo_test_binary_path_and_not_the_library() {
        assert!(
            "/Volumes/x/target/release/deps/mlxcel_core-693b1aae6e646b8d"
                .contains(TEST_BINARY_MARKER)
        );
        // The rlib and the dylib live beside the test binary and must not count,
        // or the guard would fire against its own build artifacts.
        assert!(!"/Volumes/x/target/release/deps/libmlxcel_core.rlib".contains(TEST_BINARY_MARKER));
        assert!(
            !"/Volumes/x/target/release/deps/libmlxcel_core-abc.dylib".contains(TEST_BINARY_MARKER)
        );
    }

    #[test]
    fn the_current_process_is_excluded_from_the_scan() {
        // This test is itself running from a path containing the marker, so a
        // scan that failed to exclude self would always report at least one.
        let others = other_test_binaries();
        let me = std::process::id().to_string();
        assert!(
            !others.iter().any(|o| o.starts_with(&format!("pid {me}:"))),
            "self appeared in the scan: {others:?}"
        );
    }
}
