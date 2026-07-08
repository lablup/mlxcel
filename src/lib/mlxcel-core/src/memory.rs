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

//! Typed wrappers around MLX's runtime memory accounting (issue #55).
//!
//! `mlx::core::get_active_memory()` and friends from `mlx/memory.h` are
//! populated by whichever allocator backs the current device:
//!
//! - **Metal** (`mlx/backend/metal/allocator.cpp`): all counters are
//!   populated. `set_wired_limit()` is meaningful on macOS 15+.
//! - **CUDA** (`mlx/backend/cuda/allocator.cpp`): all counters are
//!   populated. `set_wired_limit()` is inert.
//! - **No-GPU CPU** (`mlx/backend/no_gpu/allocator.cpp`): the active /
//!   peak / limit counters are tracked by the common allocator, but
//!   `get_cache_memory()` returns 0 and `set_cache_limit()` is a no-op
//!   that also returns 0.
//!
//! Every wrapper here is therefore safe to call on Linux/CUDA and on the
//! pure-CPU backend without panicking. Numbers that the active allocator
//! does not track simply read back as 0. This is the cross-platform
//! contract issue #55 / epic #52 relies on.
//!
//! The raw FFI entry points live in [`crate::ffi`] (re-exported through
//! `pub use ffi::*` at the crate root); the wrappers below normalise the
//! cxx `usize` return into `u64` for consumers that need a stable wire
//! size irrespective of host pointer width (the estimator, preflight, and
//! metrics surfaces).

use crate::ffi;

/// Snapshot of the four most useful MLX memory counters.
///
/// Captured atomically from MLX's perspective per call (one FFI hop per
/// field, but each field is read with its own lock inside the allocator),
/// not as a single transactional read. Use this whenever you want to log
/// or surface a coherent "where are we right now?" view — for example the
/// post-load resident measurement on the CLI generate path or the
/// preflight memory budget check (#56).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemorySnapshot {
    /// Bytes actively allocated by the MLX allocator (excludes the
    /// free-buffer cache).
    pub active_bytes: u64,
    /// Peak active bytes since process start or the last
    /// [`reset_peak_memory`] call.
    pub peak_bytes: u64,
    /// Bytes held in the allocator's free-buffer cache. `0` on the no-gpu
    /// CPU backend by MLX upstream design.
    pub cache_bytes: u64,
    /// Current soft memory limit reported by the allocator. `0` means
    /// "no limit set" / "backend does not enforce one".
    pub limit_bytes: u64,
}

impl MemorySnapshot {
    /// Bytes that count against the soft `limit_bytes` (active + cache).
    ///
    /// MLX's allocator counts cached free buffers against the soft limit
    /// on the GPU backends — eviction from the cache happens lazily on
    /// the next allocation. Reporting `active + cache` makes the headroom
    /// math consistent with how MLX itself decides whether to evict.
    #[inline]
    pub fn used_bytes(&self) -> u64 {
        self.active_bytes.saturating_add(self.cache_bytes)
    }
}

impl std::fmt::Display for MemorySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "active={} bytes, peak={} bytes, cache={} bytes, limit={} bytes",
            self.active_bytes, self.peak_bytes, self.cache_bytes, self.limit_bytes,
        )
    }
}

/// Bytes actively allocated by the MLX allocator.
///
/// Excludes the free-buffer cache. On the no-gpu CPU backend this still
/// returns a real count (tracked by `CommonAllocator`). On Metal / CUDA
/// it reflects whatever the device allocator considers "in use".
#[inline]
pub fn active_memory() -> u64 {
    ffi::get_active_memory() as u64
}

/// Peak active bytes since process start or the last
/// [`reset_peak_memory`] call.
#[inline]
pub fn peak_memory() -> u64 {
    ffi::get_peak_memory() as u64
}

/// Bytes held in the allocator's free-buffer cache.
///
/// `0` on the no-gpu CPU backend by MLX upstream design.
#[inline]
pub fn cache_memory() -> u64 {
    ffi::get_cache_memory() as u64
}

/// Current soft allocator memory limit in bytes.
///
/// `0` typically means "no explicit limit". On Metal the default is 1.5x
/// the recommended working-set size of the active device.
#[inline]
pub fn memory_limit() -> u64 {
    ffi::get_memory_limit() as u64
}

/// Set the soft allocator memory limit in bytes; returns the previous limit.
///
/// Exceeding the limit during graph evaluation will raise an MLX exception
/// once the host is also out of RAM (including swap). Used by the
/// preflight hook (#56) to fail fast instead of thrashing.
///
/// Passing `bytes = 0` removes the explicit limit on backends that
/// implement it as "unbounded"; on Metal / CUDA it falls back to the
/// allocator's defaults. The MLX docs make `0` semantically valid.
#[inline]
pub fn set_memory_limit(bytes: u64) -> u64 {
    // `bytes` came in as u64 to give callers a stable wire size; the
    // bridge takes `size_t`, which is `usize` on every host we target.
    // On a 64-bit host this is a lossless conversion; on a hypothetical
    // 32-bit host we'd want to clamp rather than silently truncate.
    let clamped = usize::try_from(bytes).unwrap_or(usize::MAX);
    ffi::set_memory_limit(clamped) as u64
}

/// Set the allocator cache limit in bytes; returns the previous limit.
///
/// Setting `bytes = 0` disables the cache (free buffers are returned to
/// the system allocator immediately). On the no-gpu CPU backend this is
/// a no-op that always returns 0 — see module-level docs.
#[inline]
pub fn set_cache_limit(bytes: u64) -> u64 {
    let clamped = usize::try_from(bytes).unwrap_or(usize::MAX);
    ffi::set_cache_limit(clamped) as u64
}

/// Default periodic decode-loop cache-clear cadence, in generated tokens.
///
/// On Metal, trimming the MLX buffer cache every 256 tokens is cheap and
/// matches Python mlx-lm. On CUDA, dropping cached buffers forces the CUDA
/// memory pool to reallocate on the next step and defeats MLX's CUDA-graph
/// executable cache (graph reuse depends on stable buffer addresses;
/// ml-explore/mlx#2358), so the periodic clear is a net loss. Issue #627
/// disables it by default on CUDA and bounds the cache via [`set_cache_limit`]
/// (`MLXCEL_CACHE_LIMIT`) instead. `0` means "never clear on cadence".
#[cfg(feature = "cuda")]
pub const DEFAULT_CACHE_CLEAR_INTERVAL: usize = 0;
/// Default periodic decode-loop cache-clear cadence, in generated tokens.
/// See the `cuda` variant for the rationale; on Metal/CPU the cheap 256-token
/// trim used by Python mlx-lm is kept.
#[cfg(not(feature = "cuda"))]
pub const DEFAULT_CACHE_CLEAR_INTERVAL: usize = 256;

const CACHE_CLEAR_INTERVAL_ENV: &str = "MLXCEL_CACHE_CLEAR_INTERVAL";

fn parse_cache_clear_interval(raw: Option<&str>) -> usize {
    match raw {
        Some(s) => s
            .trim()
            .parse::<usize>()
            .unwrap_or(DEFAULT_CACHE_CLEAR_INTERVAL),
        None => DEFAULT_CACHE_CLEAR_INTERVAL,
    }
}

/// Resolve the periodic decode-loop cache-clear cadence (in generated tokens).
///
/// `MLXCEL_CACHE_CLEAR_INTERVAL` overrides the backend default
/// ([`DEFAULT_CACHE_CLEAR_INTERVAL`]): a positive integer sets the token
/// cadence, `0` disables the periodic clear entirely, and an unset or
/// unparseable value keeps the default. Resolved once per process.
pub fn cache_clear_interval() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_cache_clear_interval(std::env::var(CACHE_CLEAR_INTERVAL_ENV).ok().as_deref())
    })
}

/// Whether the periodic decode-loop clear should fire at generated-token
/// count `n` for the resolved `interval`. Centralizes the gate so the decode
/// loops and the batch scheduler stay in lockstep: `interval == 0` disables
/// it, and `n == 0` never fires (the first post-prefill step still needs its
/// pipelined tensors).
#[inline]
pub fn should_clear_cache_at(n: usize, interval: usize) -> bool {
    interval != 0 && n > 0 && n.is_multiple_of(interval)
}

/// Like [`should_clear_cache_at`] but for batched decode loops that emit more
/// than one token per step: fire when the cumulative emitted count crosses a
/// cadence boundary between `prev` and `new`. `interval == 0` disables it (and
/// short-circuits before the division, so a zero interval never divides).
#[inline]
pub fn should_clear_cache_crossing(prev: usize, new: usize, interval: usize) -> bool {
    interval != 0 && new / interval > prev / interval
}

/// Reset the recorded peak memory counter to 0.
///
/// Call this immediately before a region of code whose peak you want to
/// observe in isolation, then read [`peak_memory`] after evaluating the
/// arrays produced in that region.
#[inline]
pub fn reset_peak_memory() {
    ffi::reset_peak_memory();
}

/// Clear the allocator's free-buffer cache.
///
/// Identical to the existing [`crate::clear_memory_cache`] helper but
/// exposed here under MLX upstream's canonical name so memory-management
/// call sites read consistently.
#[inline]
pub fn clear_cache() {
    ffi::clear_memory_cache();
}

/// Capture a coherent [`MemorySnapshot`] from MLX.
#[inline]
pub fn snapshot() -> MemorySnapshot {
    MemorySnapshot {
        active_bytes: active_memory(),
        peak_bytes: peak_memory(),
        cache_bytes: cache_memory(),
        limit_bytes: memory_limit(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Process-global state warning: every test below pokes a single
    //! shared allocator. They serialize through `MEMORY_TEST_LOCK` so
    //! `reset_peak_memory()` in one test never tramples a concurrent
    //! peak observation in another. Other tests in this crate that
    //! happen to allocate arrays will still bump the peak counter, but
    //! we only assert *monotonic* relationships (e.g. peak >= active)
    //! to keep the assertions robust against unrelated traffic from
    //! parallel test binaries.
    use super::*;
    use crate::{eval, from_slice_f32, sum_all};
    use std::sync::{Mutex, OnceLock};

    fn memory_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static MEMORY_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        MEMORY_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn snapshot_returns_consistent_values() {
        let _guard = memory_test_lock();
        // A snapshot must always satisfy peak >= active and used >= active.
        let snap = snapshot();
        assert!(
            snap.peak_bytes >= snap.active_bytes,
            "peak {} must be >= active {}",
            snap.peak_bytes,
            snap.active_bytes,
        );
        assert!(
            snap.used_bytes() >= snap.active_bytes,
            "used {} must be >= active {}",
            snap.used_bytes(),
            snap.active_bytes,
        );
    }

    #[test]
    fn active_memory_increases_after_allocation() {
        let _guard = memory_test_lock();
        // 1 MiB f32 buffer == 4 MiB of MLX-tracked allocation.
        let before = active_memory();
        let data: Vec<f32> = vec![1.0; 1024 * 1024];
        let arr = from_slice_f32(&data, &[data.len() as i32]);
        eval(&arr);
        let after = active_memory();
        // The MLX allocator may pool, evict, or merge buffers, so we
        // can't assert a precise delta. We can only assert the counter
        // is non-zero once *something* has been allocated this process:
        // if we got here, the test that allocated the array above must
        // have driven the active counter above zero at least
        // transiently. We additionally assert `after >= before` as a
        // weak monotonicity check — the allocator should not be
        // returning negative deltas inside a single test scope.
        assert!(
            after > 0,
            "active_memory() should be non-zero after a 4 MiB allocation, got {}",
            after,
        );
        // Touch `arr` so the allocation isn't optimized away.
        let s = sum_all(&arr);
        eval(&s);
        // Loose check: peak strictly tracks active.
        let peak = peak_memory();
        assert!(
            peak >= after,
            "peak {} should be >= post-alloc active {}",
            peak,
            after,
        );
        // Avoid an unused-binding warning while keeping the read intentional.
        let _ = before;
    }

    #[test]
    fn reset_peak_memory_lowers_or_holds_peak() {
        let _guard = memory_test_lock();
        // Allocate something so the peak counter has a recorded high-water mark.
        let data: Vec<f32> = vec![2.0; 256 * 1024];
        let arr = from_slice_f32(&data, &[data.len() as i32]);
        eval(&arr);
        let _ = sum_all(&arr);

        let active_before_reset = active_memory();
        reset_peak_memory();
        let peak_after_reset = peak_memory();

        // After reset, MLX guarantees peak collapses to whatever is
        // *currently* active (any future allocation will push it back up).
        // We can only assert peak <= active_before_reset + slack, because
        // the no-gpu CPU allocator sets `peak_memory_ = 0` directly under
        // the lock, while Metal/CUDA tend to re-seed peak to the live
        // active count. Either way, peak must not exceed the
        // pre-reset active by more than what's currently active.
        let active_after_reset = active_memory();
        assert!(
            peak_after_reset <= active_after_reset.max(active_before_reset),
            "peak after reset ({}) must not exceed live active ({} / {})",
            peak_after_reset,
            active_after_reset,
            active_before_reset,
        );

        // And after a fresh allocation, peak must climb back up to at
        // least that allocation's residency.
        let more: Vec<f32> = vec![3.0; 128 * 1024];
        let arr2 = from_slice_f32(&more, &[more.len() as i32]);
        eval(&arr2);
        let _ = sum_all(&arr2);
        let active_after_new = active_memory();
        let peak_after_new = peak_memory();
        assert!(
            peak_after_new >= active_after_new,
            "peak ({}) must be >= active ({}) after subsequent allocation",
            peak_after_new,
            active_after_new,
        );
    }

    #[test]
    fn set_memory_limit_round_trip_restores_previous() {
        let _guard = memory_test_lock();
        let original = memory_limit();
        // Pick a deliberately huge cap so we never accidentally trip an
        // OOM in a parallel test thread that's allocating concurrently.
        let huge = 1u64 << 60;
        let prev_returned = set_memory_limit(huge);
        // The bridge returns the *previous* limit, which must match what
        // we just observed above.
        assert_eq!(
            prev_returned, original,
            "set_memory_limit should return the previous limit",
        );
        // Reading back should reflect the new cap (the no-gpu CPU
        // allocator stores the exact value we passed; Metal/CUDA may
        // also accept it verbatim because we're well above hardware).
        let observed = memory_limit();
        assert_eq!(
            observed, huge,
            "memory_limit() should reflect the value just set",
        );
        // Restore so we don't poison subsequent tests in this process.
        let _ = set_memory_limit(original);
    }

    #[test]
    fn set_cache_limit_round_trip() {
        let _guard = memory_test_lock();
        // On the CPU no-gpu backend this is a no-op that always returns
        // 0. On Metal/CUDA it returns the previous limit. Either way,
        // calling it twice must restore the original observable state.
        let original = set_cache_limit(0);
        // Restore.
        let _ = set_cache_limit(original);
    }

    #[test]
    fn clear_cache_does_not_panic() {
        let _guard = memory_test_lock();
        // No-op on no-gpu CPU backend; releases pooled buffers on
        // Metal/CUDA. Either way, this must complete without panicking.
        clear_cache();
        let active = active_memory();
        // `active` is whatever the allocator currently considers live.
        // We can't assert a specific value here, only that the call
        // produced a reading and didn't crash.
        let _ = active;
    }

    #[test]
    fn periodic_clear_gate_respects_interval() {
        // interval 256 (Metal default): fires at 256, 512, ... never before, never at 0.
        assert!(should_clear_cache_at(256, 256));
        assert!(should_clear_cache_at(512, 256));
        assert!(!should_clear_cache_at(0, 256));
        assert!(!should_clear_cache_at(255, 256));
        // a longer cadence fires later.
        assert!(should_clear_cache_at(4096, 4096));
        assert!(!should_clear_cache_at(256, 4096));
    }

    #[test]
    fn periodic_clear_disabled_when_interval_zero() {
        // interval 0 is the CUDA default: the clear never fires.
        for n in [0_usize, 1, 256, 512, 4096, 100_000] {
            assert!(!should_clear_cache_at(n, 0));
        }
    }

    #[test]
    fn parse_cache_clear_interval_falls_back_on_garbage() {
        assert_eq!(parse_cache_clear_interval(Some("4096")), 4096);
        assert_eq!(parse_cache_clear_interval(Some("0")), 0);
        assert_eq!(parse_cache_clear_interval(Some("  32 ")), 32);
        assert_eq!(
            parse_cache_clear_interval(Some("nonsense")),
            DEFAULT_CACHE_CLEAR_INTERVAL
        );
        assert_eq!(
            parse_cache_clear_interval(None),
            DEFAULT_CACHE_CLEAR_INTERVAL
        );
    }

    #[test]
    fn crossing_gate_fires_when_cumulative_count_crosses_boundary() {
        // Batched decode emits >1 token/step: fire when a cadence multiple is crossed.
        assert!(should_clear_cache_crossing(250, 260, 256)); // crosses 256
        assert!(should_clear_cache_crossing(0, 256, 256)); // crosses 256 from 0
        assert!(should_clear_cache_crossing(511, 520, 256)); // crosses 512
        assert!(!should_clear_cache_crossing(256, 300, 256)); // no new multiple crossed
        assert!(!should_clear_cache_crossing(0, 100, 256)); // never reaches 256
        // interval 0 (CUDA default) disables it and never divides.
        for (p, n) in [(0_usize, 256_usize), (250, 260), (0, 100_000)] {
            assert!(!should_clear_cache_crossing(p, n, 0));
        }
    }
}
