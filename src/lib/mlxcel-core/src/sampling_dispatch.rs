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

//! Which sampling path a decode step actually took, announced at INFO
//! (issue #901).
//!
//! ## Why this exists
//!
//! Issue #899 shipped a fused decode path that never activated. Its production
//! benchmark compared the fallback against itself across a full sweep and
//! returned a clean-looking null result, because the declines were reported
//! with `tracing::debug!` and a real `mlxcel-server` run emits no `DEBUG` from
//! this crate. A diagnostic invisible in the one situation it exists for is not
//! a diagnostic.
//!
//! The sampler has the same shape: a new kernel (#901), a kill switch
//! (`MLXCEL_SAMPLING_REJECTION`), and a convergence-cap fallback to the stock
//! `argpartition` chain. So the C++ dispatcher records the outcome of every
//! `fused_sample` call, and [`report_sampling_dispatch`] announces the first
//! occurrence of each distinct outcome kind at **INFO**.
//!
//! Per kind, not one global one-shot. A single flag reports only whatever
//! happened first, which in a server is a warmup request, and a later permanent
//! fallback for another reason would never surface. That is precisely how a
//! whole sweep ran on the wrong path in silence.
//!
//! ## Cost
//!
//! [`report_sampling_dispatch`] is called once per sampling step. In steady
//! state it is one FFI call returning a `u32` bitmask that is zero, so nothing
//! is formatted, nothing is allocated, and no log macro is entered. Only a
//! genuinely new outcome kind reaches the drain-and-format path.

use crate::ffi;

/// Announce every sampling dispatch outcome that has occurred since the last
/// call and has not been reported yet, one INFO line per distinct kind.
///
/// Call it directly after a [`ffi::fused_sample`] dispatch. Callers that do not
/// call it lose nothing but the log line; the outcome stays queued until some
/// caller drains it.
///
/// Used by: [`crate::sampling::sample_token_optimized`],
/// [`crate::sampling::batched_fused_sample`], `BatchScheduler::execute_batched_decode`,
/// `audio::qwen3_omni_moe::speech_layers`.
pub fn report_sampling_dispatch() {
    if ffi::sampling_dispatch_pending_kinds() == 0 {
        return;
    }
    loop {
        let line = ffi::sampling_dispatch_drain_report();
        if line.is_empty() {
            break;
        }
        tracing::info!("sampling dispatch: {line}");
    }
}

/// Rows that exhausted the rejection round cap and forced the launch back onto
/// the `argpartition` chain, cumulative since process start.
///
/// Issue #901 asks for the cap-overflow event to be counted; this is the
/// reachable form of that count, so a benchmark or an operator can read it
/// instead of inferring it. Nonzero means the rejection kernel gave up on at
/// least one row, which should be vanishingly rare: the kernel's interval
/// halves in bit space every round, so 32 rounds isolates a single float.
#[must_use]
pub fn rejection_cap_overflow_rows() -> u64 {
    ffi::sampling_rejection_cap_overflow_rows()
}

/// Launches in which at least one row exhausted the rejection round cap.
///
/// Divide by the number of sampling steps to get the fallback rate.
#[must_use]
pub fn rejection_cap_overflow_launches() -> u64 {
    ffi::sampling_rejection_cap_overflow_launches()
}

/// Clear every recorded dispatch outcome and both cap-overflow counters.
///
/// The state is process-wide, so a test that asserts on reporting needs a clean
/// slate. Not for production use.
pub fn reset_sampling_dispatch() {
    ffi::sampling_dispatch_reset();
}

/// Serialises tests that touch the process-global sampling dispatch state.
///
/// The recorded outcomes, their one-shot "already seen" bits, the cap-overflow
/// counters, and the deferred converged-flag ring are all process-global by
/// design: they exist so a server announces each distinct outcome exactly once
/// and checks each launch's flags without ever waiting. That makes them shared
/// mutable state across the whole test binary, so any test that samples through
/// a routed `fused_sample` or asserts on the report has to take this lock,
/// including the ones in `sampling_gumbel_tests`.
#[cfg(test)]
pub(crate) fn dispatch_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draining_an_empty_queue_is_a_no_op() {
        reset_sampling_dispatch();
        assert_eq!(ffi::sampling_dispatch_pending_kinds(), 0);
        // Must not panic, must not block, must not log.
        report_sampling_dispatch();
        assert_eq!(ffi::sampling_dispatch_pending_kinds(), 0);
    }

    #[test]
    fn the_reset_clears_both_cap_overflow_counters() {
        reset_sampling_dispatch();
        assert_eq!(rejection_cap_overflow_rows(), 0);
        assert_eq!(rejection_cap_overflow_launches(), 0);
    }

    #[test]
    fn the_production_round_cap_is_the_documented_value() {
        // The kernel's bracket halves in bit space every round, so 32 rounds is
        // enough to isolate a single float from any starting interval. A change
        // here changes the cap-overflow probability and must be deliberate.
        assert_eq!(ffi::sampling_rejection_max_rounds(), 32);
    }

    #[test]
    fn the_threadgroup_size_is_a_power_of_two() {
        // The block scan and both halving-tree reductions in the kernel require
        // it, and the determinism argument requires it to be fixed.
        let tg = ffi::sampling_rejection_threadgroup_size();
        assert!(tg > 0, "threadgroup size {tg}");
        assert!(
            (tg as u32).is_power_of_two(),
            "threadgroup size {tg} is not a power of two"
        );
    }
}
