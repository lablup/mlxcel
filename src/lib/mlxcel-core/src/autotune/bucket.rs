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

//! Shape bucketing for the kernel autotuner (issue #906).
//!
//! A serving runtime sees a continuous stream of nearby shapes: context length
//! grows by one token per decode step, batch size drifts as requests join and
//! leave. Tuning every exact shape would make the tuning matrix unbounded and
//! the cache useless (every lookup a miss). Bucketing collapses nearby shapes
//! onto one entry by rounding each dimension **up** to the next power of two,
//! so a decode sweep from context 4097 to 8192 shares a single tuned tactic.
//!
//! Rounding up (never down) matters: a tactic profiled at the bucket ceiling
//! was measured on at least as much work as any shape that maps into the
//! bucket, so it is never chosen against a launch shape that is larger than
//! what was measured.

use std::fmt;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Largest dimension the bucketer will represent. Anything above saturates
/// here instead of overflowing; a saturating top bucket is the documented
/// "out of bucket" case that callers report and fall back to the default for.
pub const MAX_BUCKET_DIM: u32 = 1 << 30;

/// A shape rounded to power-of-two buckets.
///
/// Constructed from a raw launch shape via [`ShapeBucket::from_dims`]. The
/// [`fmt::Display`] form (`"1x32x128x8192"`) is what goes into the cache key,
/// so it is stable across runs and readable in a cache file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeBucket {
    /// Bucketed dimensions, in the order the op declared them.
    dims: SmallVec<[u32; 4]>,
}

impl ShapeBucket {
    /// Round every dimension up to the next power of two.
    ///
    /// A zero dimension buckets to 1 (an empty launch still has one bucket so
    /// lookups stay total), and anything at or above [`MAX_BUCKET_DIM`]
    /// saturates there.
    #[must_use]
    pub fn from_dims(dims: &[usize]) -> Self {
        Self {
            dims: dims.iter().map(|&d| round_up_pow2(d)).collect(),
        }
    }

    /// Build a bucket from already-bucketed values. Used by the offline
    /// `mlxcel tune` matrix, which enumerates bucket ceilings directly.
    #[must_use]
    pub fn from_exact(dims: &[u32]) -> Self {
        Self {
            dims: dims.iter().copied().collect(),
        }
    }

    /// The bucketed dimensions.
    #[must_use]
    pub fn dims(&self) -> &[u32] {
        &self.dims
    }

    /// Number of dimensions in the bucket.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dims.len()
    }

    /// Whether the bucket carries no dimensions at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    /// Dimension `i`, or `None` when the bucket has fewer dimensions. Ops read
    /// their own axes through this so a shorter-than-expected bucket degrades
    /// to the default tactic instead of panicking.
    #[must_use]
    pub fn dim(&self, i: usize) -> Option<u32> {
        self.dims.get(i).copied()
    }

    /// Whether any dimension saturated at [`MAX_BUCKET_DIM`]. A saturated
    /// bucket means the real shape was larger than the bucketer can represent,
    /// so a cached tactic for it was not measured on comparable work; callers
    /// treat this as out-of-bucket and use the default.
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        self.dims.iter().any(|&d| d >= MAX_BUCKET_DIM)
    }
}

impl fmt::Display for ShapeBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            return f.write_str("scalar");
        }
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                f.write_str("x")?;
            }
            write!(f, "{d}")?;
        }
        Ok(())
    }
}

/// Round `v` up to the next power of two, clamped into
/// `[1, MAX_BUCKET_DIM]`.
///
/// `usize::next_power_of_two` panics on overflow in debug builds, so the clamp
/// happens before the rounding rather than after.
#[must_use]
pub fn round_up_pow2(v: usize) -> u32 {
    if v <= 1 {
        return 1;
    }
    if v >= MAX_BUCKET_DIM as usize {
        return MAX_BUCKET_DIM;
    }
    // `v` is now in `2 ..= MAX_BUCKET_DIM - 1`, so the result fits in u32 and
    // the rounding cannot overflow.
    let mut p: u32 = 1;
    while (p as usize) < v {
        p <<= 1;
    }
    p
}

/// Feasible powers of two in `1 ..= cap`, ascending.
///
/// Shared by every op whose tactic space is "a power-of-two launch parameter
/// bounded by a hardware budget" (paged-decode `NumSplits`, qmm `tile_m`).
/// A `cap` below 1 yields an empty candidate list, which the resolver reports
/// as "no candidates" and answers with the default.
#[must_use]
pub fn powers_of_two_up_to(cap: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut v: u32 = 1;
    while v <= cap {
        out.push(v);
        // Guard the shift so a cap near u32::MAX cannot overflow.
        if v > u32::MAX / 2 {
            break;
        }
        v <<= 1;
    }
    out
}
