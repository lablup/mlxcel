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

//! Autotuned `pages_per_chunk` for paged decode v2 (issues #906, #898).
//!
//! This is the consumer issue #906 reserved
//! [`crate::autotune::OP_PAGED_DECODE_V2_KV_CHUNK`] for, plugged in exactly as
//! that module's extension note prescribes: register under the reserved name,
//! implement [`TunableOp`] over the feasible chunk sizes, and read the winner
//! out of the resolution.
//!
//! ## Why the chunk size is a tuned knob and not a formula
//!
//! [`crate::paged_v2::plan`] derives its default by binary-searching for the
//! largest chunk size whose CTA count still reaches an occupancy target. That
//! target is derived from a device-scale proxy, not measured, and the real
//! optimum trades three costs the proxy does not model: the partial kernel's
//! per-chunk fixed work, the merge kernel's per-partial traffic, and the tail
//! effect of a chunk count that is not a multiple of the device's resident CTA
//! capacity. Those are per-shape, which is what the shape-bucketed cache is for.
//!
//! ## Feasible set
//!
//! Powers of two between [`crate::paged_v2::min_pages_per_chunk`] (the grid-z
//! bound) and [`crate::paged_v2::max_pages_per_chunk`] (above which every
//! request is one chunk and larger values are the same plan), plus the
//! heuristic value itself. Including the heuristic matters: the profiler only
//! switches away from the default when the win clears its noise margin, which
//! requires the default to have been measured.
//!
//! ## Cost when the autotuner is off
//!
//! [`resolve_pages_per_chunk`] returns the heuristic after one `OnceLock` read
//! and one mode read: no cache read, no lock, no filesystem access. Since
//! paged decode v2 is itself off by default in issue #898, the whole path is
//! unreachable in a default process.

use std::sync::OnceLock;

use crate::autotune::bucket::{ShapeBucket, powers_of_two_up_to, round_up_pow2};
use crate::autotune::tactic::{Tactic, TunableOp, TuneError};
use crate::autotune::{OP_PAGED_DECODE_V2_KV_CHUNK, Source, mode};
use crate::paged_v2::plan::{PagedDecodePlan, max_pages_per_chunk, min_pages_per_chunk};
use crate::paged_v2::{PagedDecodeGeometry, V2Context};

/// Explicit operator override, matching the `MLXCEL_PAGED_DECODE_SPLITS` /
/// `MLXCEL_QMM_TILE_*` escape-hatch convention. Wins over any tuned or cached
/// value.
pub const CHUNK_ENV: &str = "MLXCEL_PAGED_DECODE_V2_CHUNK";

/// Tactic parameter index for the chunk size.
const PARAM_PAGES_PER_CHUNK: usize = 0;

/// The explicit [`CHUNK_ENV`] value, read once.
///
/// A non-positive or unparseable value is ignored with a warning rather than
/// clamped, so a typo degrades to the heuristic instead of pinning an arbitrary
/// chunk size. The plan clamps whatever survives into the feasible range, so an
/// over-large value is safe.
fn env_pages_per_chunk() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var(CHUNK_ENV).ok()?;
        match raw.trim().parse::<i32>() {
            Ok(v) if v >= 1 => Some(v),
            _ => {
                tracing::warn!(
                    "{CHUNK_ENV}={raw:?} is not a positive integer; ignoring it and using the planned chunk size"
                );
                None
            }
        }
    })
}

/// Candidate chunk sizes for a `[min, max]` feasible range plus the heuristic.
///
/// Sorted and deduplicated so the profiler's report reads in order and no
/// candidate is measured twice.
#[must_use]
pub fn chunk_candidates(min_pages: i32, max_pages: i32, heuristic: i32) -> Vec<i64> {
    let lo = min_pages.max(1);
    let hi = max_pages.max(lo);
    let mut out: Vec<i64> = powers_of_two_up_to(hi.max(0) as u32)
        .into_iter()
        .map(i64::from)
        .filter(|&v| v >= i64::from(lo))
        .collect();
    for extra in [i64::from(hi), i64::from(heuristic.clamp(lo, hi))] {
        if !out.contains(&extra) {
            out.push(extra);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Launch shape of one v2 decode, used as cache-key material.
///
/// `batch` and `max_pages` are bucketed because they drift continuously during
/// serving; the head counts, head dim, and page size are model or pool
/// constants that each change the kernel specialization or the feasible range,
/// so merging them across buckets would mix incomparable regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2ChunkShape {
    pub batch: usize,
    pub geometry: PagedDecodeGeometry,
    /// Largest per-request page count in the batch.
    pub max_pages: usize,
}

impl V2ChunkShape {
    #[must_use]
    pub fn bucket(&self) -> ShapeBucket {
        ShapeBucket::from_exact(&[
            round_up_pow2(self.batch),
            self.geometry.q_heads.max(0) as u32,
            self.geometry.kv_heads.max(0) as u32,
            self.geometry.head_dim.max(0) as u32,
            self.geometry.page_size.max(0) as u32,
            round_up_pow2(self.max_pages),
        ])
    }
}

/// [`TunableOp`] over a real v2 launch.
///
/// Profiling re-runs the actual partial (and, where the tactic implies one,
/// merge) launch against the actual pool and page table rather than a synthetic
/// stand-in, so the measurement includes the real scatter pattern and the real
/// merge width. The context is borrowed, so no arrays are rebuilt per candidate
/// and the sweep times the kernels rather than the setup.
pub struct PagedDecodeV2ChunkOp<'a> {
    ctx: &'a V2Context<'a>,
    page_counts: &'a [usize],
    target_ctas: usize,
    heuristic: i32,
    min_pages: i32,
    max_pages: i32,
    shape: V2ChunkShape,
}

impl<'a> PagedDecodeV2ChunkOp<'a> {
    #[must_use]
    pub fn new(
        ctx: &'a V2Context<'a>,
        page_counts: &'a [usize],
        target_ctas: usize,
        heuristic: i32,
    ) -> Self {
        let max_pages = max_pages_per_chunk(page_counts);
        Self {
            ctx,
            page_counts,
            target_ctas,
            heuristic,
            min_pages: min_pages_per_chunk(page_counts),
            max_pages,
            shape: V2ChunkShape {
                batch: page_counts.len(),
                geometry: ctx.geometry,
                max_pages: max_pages.max(0) as usize,
            },
        }
    }

    fn plan_for(&self, pages_per_chunk: i32, source: Source) -> PagedDecodePlan {
        PagedDecodePlan::with_chunk_size(
            self.ctx.geometry,
            self.page_counts,
            pages_per_chunk,
            self.target_ctas,
            source,
        )
    }
}

impl TunableOp for PagedDecodeV2ChunkOp<'_> {
    fn op_name(&self) -> &str {
        OP_PAGED_DECODE_V2_KV_CHUNK
    }

    fn runner_id(&self) -> String {
        // The launcher picks the Metal or CUDA JIT body from the live backend,
        // and the two are different kernels; keep their tactics apart.
        if crate::metal_is_available() {
            "metal".to_string()
        } else {
            "cuda".to_string()
        }
    }

    fn dtype_tag(&self) -> String {
        // Both kernels accumulate and emit f32 regardless of the pool dtype
        // (the caller casts around them), so the tactic is dtype-invariant.
        "f32".to_string()
    }

    fn bucket(&self) -> ShapeBucket {
        self.shape.bucket()
    }

    fn candidates(&self, _bucket: &ShapeBucket) -> Vec<Tactic> {
        chunk_candidates(self.min_pages, self.max_pages, self.heuristic)
            .into_iter()
            .map(|v| Tactic::scalar("pages_per_chunk", v))
            .collect()
    }

    fn default_tactic(&self, _bucket: &ShapeBucket) -> Tactic {
        Tactic::scalar("pages_per_chunk", i64::from(self.heuristic))
    }

    fn env_override(&self) -> Option<Tactic> {
        env_pages_per_chunk().map(|v| Tactic::scalar("pages_per_chunk", i64::from(v)))
    }

    fn run(&self, tactic: &Tactic) -> Result<(), TuneError> {
        let chunk = tactic
            .param(PARAM_PAGES_PER_CHUNK)
            .ok_or_else(|| TuneError::infeasible(tactic, "tactic carries no chunk size"))?;
        let chunk = i32::try_from(chunk)
            .map_err(|_| TuneError::infeasible(tactic, "chunk size out of range"))?;
        if chunk < self.min_pages || chunk > self.max_pages {
            return Err(TuneError::infeasible(
                tactic,
                format!(
                    "chunk size outside the feasible [{}, {}]",
                    self.min_pages, self.max_pages
                ),
            ));
        }
        let plan = self.plan_for(chunk, Source::Tuned);
        let out = self
            .ctx
            .launch(&plan)
            .map_err(|e| TuneError::failed(tactic, e))?;
        crate::eval(&out);
        Ok(())
    }
}

/// Resolve the chunk size for one v2 launch, with the full precedence chain
/// (env override, cached tactic, profiled tactic, heuristic).
///
/// Returns the size and where it came from, so the plan records its own
/// provenance. Never fails: every error path in the autotuner falls back to
/// `heuristic`, which is what the plan would have used on its own.
#[must_use]
pub fn resolve_pages_per_chunk(
    ctx: &V2Context<'_>,
    page_counts: &[usize],
    target_ctas: usize,
    heuristic: i32,
) -> (i32, Source) {
    if let Some(v) = env_pages_per_chunk() {
        return (v, Source::EnvOverride);
    }
    if !mode().reads_cache() {
        return (heuristic, Source::Default);
    }
    let op = PagedDecodeV2ChunkOp::new(ctx, page_counts, target_ctas, heuristic);
    let resolution = crate::autotune::resolve(&op);
    if resolution.source.is_default() {
        return (heuristic, resolution.source);
    }
    match resolution.param(PARAM_PAGES_PER_CHUNK) {
        Some(v) if v >= 1 => (i32::try_from(v).unwrap_or(heuristic), resolution.source),
        _ => (heuristic, Source::Default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> PagedDecodeGeometry {
        PagedDecodeGeometry {
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            page_size: 32,
        }
    }

    #[test]
    fn candidates_span_the_feasible_range_and_include_the_heuristic() {
        let c = chunk_candidates(1, 64, 24);
        assert_eq!(c.first(), Some(&1));
        assert!(c.contains(&24), "the heuristic must be measurable: {c:?}");
        assert!(c.contains(&64), "the ceiling must be measurable: {c:?}");
        assert!(
            c.windows(2).all(|w| w[0] < w[1]),
            "sorted and deduped: {c:?}"
        );
    }

    #[test]
    fn candidates_respect_a_raised_floor() {
        // A grid-z-bounded batch cannot use fine chunks at all.
        let c = chunk_candidates(16, 64, 16);
        assert!(c.iter().all(|&v| v >= 16), "{c:?}");
    }

    #[test]
    fn candidates_never_empty_for_a_degenerate_range() {
        let c = chunk_candidates(1, 1, 1);
        assert_eq!(c, vec![1]);
        let c = chunk_candidates(8, 4, 8);
        assert!(!c.is_empty(), "{c:?}");
    }

    #[test]
    fn the_bucket_separates_shapes_that_are_not_comparable() {
        let a = V2ChunkShape {
            batch: 4,
            geometry: geometry(),
            max_pages: 128,
        };
        let mut other = geometry();
        other.head_dim = 64;
        let b = V2ChunkShape {
            batch: 4,
            geometry: other,
            max_pages: 128,
        };
        assert_ne!(a.bucket(), b.bucket());

        // Nearby contexts share a bucket, which is the point of bucketing.
        let c = V2ChunkShape {
            batch: 4,
            geometry: geometry(),
            max_pages: 100,
        };
        assert_eq!(a.bucket(), c.bucket());
    }
}
