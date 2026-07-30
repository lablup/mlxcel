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

//! Cold-last-level-cache benchmark support (issue #906).
//!
//! ## The problem
//!
//! mlxcel's microbenches reuse the same input buffers on every timed
//! iteration. After the first iteration the working set is resident in the
//! last-level cache (Apple's System Level Cache, NVIDIA's L2), so every
//! subsequent iteration reads at cache bandwidth rather than at DRAM
//! bandwidth. That flatters exactly the kernels most of this epic touches:
//! paged KV gather, paged decode attention, and rmsnorm are bandwidth-bound,
//! and a warm-cache measurement of a bandwidth-bound kernel is measuring the
//! wrong memory.
//!
//! In production the KV pool is far larger than any LLC and is touched once
//! per decode step, so the cold read is the representative one. Warm numbers
//! are not wrong, they answer a different question, which is why the mode has
//! to be recorded alongside every result (see `docs/benchmarks.md`).
//!
//! ## The fix
//!
//! Allocate `N` copies of the input and advance one copy per timed iteration.
//! Choosing `N` so that `N * working_set >= 2 * last_level_cache` guarantees a
//! buffer has been evicted by the time the rotation comes back to it. The 2x
//! headroom covers the cache being shared with the rest of the system and with
//! the kernel's own output traffic.
//!
//! ## Sizing the last level cache
//!
//! [`last_level_cache_bytes`] estimates the Apple SLC from the detected
//! silicon generation and performance-core count. These are *estimates by
//! device family*, not a queried value: macOS exposes no SLC size through
//! `sysctl`. They are deliberately biased high, because over-estimating costs
//! memory for extra rotation buffers while under-estimating silently
//! reintroduces the warm-cache bias the mode exists to remove. Override with
//! [`LLC_BYTES_ENV`] when the real figure is known.
//!
//! The CUDA L2 path is [`cuda_l2_bytes`], a clearly-marked stub: reading it
//! requires `cudaDeviceProp::l2CacheSize` through an FFI helper that does not
//! exist yet, and this was written on an Apple-Silicon-only host.

use crate::hardware::{AppleSiliconGen, get_hardware};

/// Override for the estimated last-level-cache size, in bytes.
pub const LLC_BYTES_ENV: &str = "MLXCEL_BENCH_LLC_BYTES";

/// Multiple of the last-level cache the rotation set must cover before a
/// buffer is considered evicted.
pub const ROTATION_HEADROOM: u64 = 2;

/// Upper bound on rotation buffers, so a tiny working set cannot ask for
/// thousands of allocations.
pub const MAX_ROTATION: usize = 64;

/// Fallback last-level-cache estimate for an unrecognized device, in bytes.
/// 8 MiB is the smallest Apple SLC in the M-series, so it is the conservative
/// floor rather than a guess at the middle.
pub const DEFAULT_LLC_BYTES: u64 = 8 * 1024 * 1024;

/// Estimated Apple SLC size for a silicon generation and performance-core
/// count, in bytes.
///
/// The performance-core count is the tier proxy this tree already uses for
/// hardware labels (`hw.perflevel0.logicalcpu`): 4 on a base M-series, 6-8 on
/// a Pro, 8-12 on a Max, 16 on an Ultra. Two dies means two SLCs, which is why
/// the Ultra tier is the largest bucket. Values are the published
/// system-level-cache figures for the M1 family rounded up, and the same
/// tiering is applied to later generations, whose SLCs are the same size or
/// larger.
#[must_use]
pub fn apple_slc_bytes(r#gen: AppleSiliconGen, perf_cores: u32) -> u64 {
    const MIB: u64 = 1024 * 1024;
    if r#gen == AppleSiliconGen::Unknown {
        return DEFAULT_LLC_BYTES;
    }
    match perf_cores {
        0..=4 => 8 * MIB,   // base M-series
        5..=6 => 24 * MIB,  // Pro (binned)
        7..=8 => 48 * MIB,  // Pro / Max, biased to the Max figure
        9..=12 => 48 * MIB, // Max
        _ => 96 * MIB,      // Ultra (two dies)
    }
}

/// CUDA L2 cache size in bytes.
///
/// **Not implemented.** Reading this needs `cudaDeviceProp::l2CacheSize`
/// through a small FFI helper, and no CUDA host was available when the
/// rotating-buffer support was written (issue #906). Callers fall back to
/// [`last_level_cache_bytes`], which returns the [`DEFAULT_LLC_BYTES`] floor
/// off Apple Silicon; set [`LLC_BYTES_ENV`] to the device's real L2 size to get
/// a correct rotation count on CUDA in the meantime.
#[must_use]
pub fn cuda_l2_bytes() -> Option<u64> {
    None
}

/// Best available last-level-cache size in bytes.
///
/// Precedence: [`LLC_BYTES_ENV`], then the CUDA L2 (when that helper exists),
/// then the Apple SLC estimate, then [`DEFAULT_LLC_BYTES`].
#[must_use]
pub fn last_level_cache_bytes() -> u64 {
    if let Ok(raw) = std::env::var(LLC_BYTES_ENV) {
        match raw.trim().parse::<u64>() {
            Ok(v) if v > 0 => return v,
            _ => {
                tracing::warn!(
                    "{LLC_BYTES_ENV}={raw:?} is not a positive integer; using the detected estimate"
                );
            }
        }
    }
    if let Some(l2) = cuda_l2_bytes() {
        return l2;
    }
    let hw = get_hardware();
    apple_slc_bytes(hw.silicon_gen, hw.gpu_core_count)
}

/// Number of input copies needed so that a buffer is evicted from a cache of
/// `llc_bytes` before the rotation returns to it.
///
/// `ceil(ROTATION_HEADROOM * llc_bytes / working_set_bytes)`, clamped into
/// `[1, MAX_ROTATION]`. A working set that already exceeds the headroom
/// window needs no rotation and returns 1, so large-context configs do not pay
/// for redundant allocations.
#[must_use]
pub fn rotation_count_for(working_set_bytes: u64, llc_bytes: u64) -> usize {
    if working_set_bytes == 0 {
        return 1;
    }
    let target = llc_bytes.saturating_mul(ROTATION_HEADROOM);
    let n = target.div_ceil(working_set_bytes);
    (n.max(1) as usize).min(MAX_ROTATION)
}

/// [`rotation_count_for`] against the detected last-level cache.
#[must_use]
pub fn rotation_count(working_set_bytes: u64) -> usize {
    rotation_count_for(working_set_bytes, last_level_cache_bytes())
}

/// Round-robin index source for a rotating input set.
///
/// Deliberately a plain counter rather than a random source: a benchmark that
/// picks buffers randomly is not reproducible, and round-robin already
/// guarantees the maximum reuse distance for a fixed rotation count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rotation {
    count: usize,
    next: usize,
}

impl Rotation {
    /// Rotation over `count` buffers. A `count` of 0 is treated as 1 (no
    /// rotation) so callers can pass a computed value unchecked.
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            count: count.max(1),
            next: 0,
        }
    }

    /// Rotation sized for a working set, using the detected last-level cache.
    #[must_use]
    pub fn for_working_set(working_set_bytes: u64) -> Self {
        Self::new(rotation_count(working_set_bytes))
    }

    /// Number of buffers in the rotation.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether this rotation actually rotates (more than one buffer).
    #[must_use]
    pub fn is_rotating(&self) -> bool {
        self.count > 1
    }

    /// Index of the next buffer to read, advancing the rotation.
    pub fn next_index(&mut self) -> usize {
        let i = self.next;
        self.next = (self.next + 1) % self.count;
        i
    }

    /// Restart the rotation at buffer 0, so a warmup phase does not leave the
    /// timed phase starting mid-cycle.
    pub fn reset(&mut self) {
        self.next = 0;
    }

    /// Human-readable mode tag for benchmark output: `"cold-l2"` when the
    /// rotation is real, `"warm"` when there is a single buffer.
    #[must_use]
    pub fn mode_tag(&self) -> &'static str {
        if self.is_rotating() {
            "cold-l2"
        } else {
            "warm"
        }
    }
}

#[cfg(test)]
#[path = "bench_rotation_tests.rs"]
mod bench_rotation_tests;
