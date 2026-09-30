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

//! Host-clock phase marks for per-kernel profiling (issue #2061).
//!
//! A kernel trace (`rocprofv3 --kernel-trace`, or any tracer that stamps
//! dispatches with a host clock) holds the warmup pass and the measured prefill
//! as well as the measured decode. `MLXCEL_BENCH_PHASE_MARKS=1` makes the bench
//! print where those phases sit on the host clocks, so a post-processor can cut
//! the measured decode out of the trace by timestamp instead of guessing it
//! from kernel names. `scripts/rocm_decode_profile.py` reads these lines.
//!
//! Each mark is one stderr line:
//!
//! ```text
//! [phase] decode_start monotonic_ns=123 boottime_ns=456
//! ```
//!
//! Both clocks are printed because tracers differ in which one they stamp
//! with; they agree on durations and differ only by time spent suspended.
//! `decode_start` is derived as `measured_end` minus the generator's own
//! `decode_time_ms`, because the decode loop's start is inside the generator.
//! The prefill that precedes it ends in a blocking `eval` of the first token,
//! so the device is idle at that instant and the cut does not split a kernel.

const ENV: &str = "MLXCEL_BENCH_PHASE_MARKS";

/// Whether phase marks were asked for. Off unless the variable is `1`, so the
/// bench's output is unchanged for `scripts/bench_decode.sh`.
pub(crate) fn enabled() -> bool {
    std::env::var(ENV).is_ok_and(|v| v.trim() == "1")
}

/// A reading of both host clocks, in nanoseconds.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stamp {
    monotonic_ns: Option<u128>,
    boottime_ns: Option<u128>,
}

impl Stamp {
    pub(crate) fn now() -> Self {
        Self {
            monotonic_ns: monotonic_ns(),
            boottime_ns: boottime_ns(),
        }
    }

    /// This stamp moved `ns` earlier on both clocks.
    pub(crate) fn earlier_by(self, ns: u128) -> Self {
        Self {
            monotonic_ns: self.monotonic_ns.map(|t| t.saturating_sub(ns)),
            boottime_ns: self.boottime_ns.map(|t| t.saturating_sub(ns)),
        }
    }

    pub(crate) fn print(self, label: &str) {
        let show = |v: Option<u128>| v.map_or_else(|| "na".to_string(), |n| n.to_string());
        eprintln!(
            "[phase] {label} monotonic_ns={} boottime_ns={}",
            show(self.monotonic_ns),
            show(self.boottime_ns)
        );
    }
}

#[cfg(unix)]
fn clock_ns(id: libc::clockid_t) -> Option<u128> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a live, writable `timespec` for the whole call, which is
    // all `clock_gettime` requires of its out-pointer.
    let rc = unsafe { libc::clock_gettime(id, &mut ts) };
    (rc == 0).then(|| ts.tv_sec as u128 * 1_000_000_000 + ts.tv_nsec as u128)
}

#[cfg(unix)]
fn monotonic_ns() -> Option<u128> {
    clock_ns(libc::CLOCK_MONOTONIC)
}

#[cfg(not(unix))]
fn monotonic_ns() -> Option<u128> {
    None
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn boottime_ns() -> Option<u128> {
    clock_ns(libc::CLOCK_BOOTTIME)
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn boottime_ns() -> Option<u128> {
    None
}
