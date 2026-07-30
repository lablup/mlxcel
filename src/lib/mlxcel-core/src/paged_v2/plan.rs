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

//! Host-side plan for paged decode v2 (issue #898).
//!
//! The plan answers one question: how many pages does each CTA cover? Every
//! other field is a consequence of that choice, precomputed so the run step
//! does no arithmetic beyond indexing.
//!
//! ## Why a chunk size needs choosing at all
//!
//! Total CTAs are `num_chunks * kv_heads * q_groups`, and `num_chunks` is
//! `sum_r ceil(pages_r / pages_per_chunk)`. Small chunks mean many CTAs (good
//! occupancy, more merge work and more partials to write); large chunks mean
//! few CTAs (no merge at all when every request fits in one, but a long serial
//! token sweep and an idle GPU). The right value therefore depends on the batch
//! composition and the device, which is why it is computed per plan rather than
//! fixed.
//!
//! ## The search
//!
//! `num_chunks` is non-increasing in `pages_per_chunk`, so
//! [`search_pages_per_chunk`] binary-searches for the **largest** chunk size
//! whose CTA count still reaches the device target. Largest, not smallest:
//! among the sizes that saturate the device, the one with the fewest chunks
//! does the least merge work. If even one page per chunk cannot reach the
//! target (a tiny batch at short context), the search returns 1 and the device
//! is simply not saturable at this shape.
//!
//! ## Autotuner seam
//!
//! [`crate::autotune::OP_PAGED_DECODE_V2_KV_CHUNK`] was reserved by issue #906
//! for exactly this knob. [`crate::autotune::ops::paged_decode_v2_chunk`]
//! implements [`crate::autotune::tactic::TunableOp`] over the candidate chunk
//! sizes with `default_tactic` returning the heuristic below, so with the
//! autotuner off (the default) the plan is bit-for-bit what the heuristic
//! chose, and with a tuned entry present the measured value wins.

use std::sync::OnceLock;

use crate::autotune::Source;
use crate::ffi;

/// Environment override for the device CTA target used by the heuristic.
///
/// Distinct from `MLXCEL_PAGED_DECODE_V2_CHUNK`, which pins the chunk size
/// itself: this one moves the occupancy target the search aims at, which is the
/// knob to turn when the derived device parallelism is wrong.
pub const TARGET_CTAS_ENV: &str = "MLXCEL_PAGED_DECODE_V2_TARGET_CTAS";

/// CTAs the heuristic aims to keep resident per GPU parallelism unit.
///
/// Deliberately generous: a decode CTA is memory-bound and short-lived, so
/// oversubscribing hides launch and tail effects. It is a starting point, not a
/// measured optimum; the autotuner exists to replace the whole choice.
const CTAS_PER_UNIT: usize = 8;

/// CTA target when no device parallelism can be derived (any non-Apple host
/// today, including CUDA).
///
/// **Unvalidated on CUDA.** A CUDA-specific target should come from the SM
/// count; deriving it needs a CUDA host, which issue #898 did not have. Until
/// then a fixed target keeps the plan well-defined and the env override keeps
/// it adjustable.
const DEFAULT_TARGET_CTAS: usize = 512;

/// Floor on the derived target so a small or misdetected parallelism figure
/// cannot collapse the plan to a single chunk.
const MIN_TARGET_CTAS: usize = 64;

/// Largest chunk count the plan will emit.
///
/// CUDA caps `gridDim.z` at 65535 and the v2 partial kernel puts the chunk
/// index there. Metal has no such cap, but the plan is device-independent data,
/// so it holds the tighter bound on both backends rather than producing a plan
/// that is only valid on one.
pub const MAX_CHUNKS: usize = 65535;

/// Static launch geometry a plan is built against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedDecodeGeometry {
    /// Query heads (`Hq`).
    pub q_heads: i32,
    /// Key/value heads (`Hkv`).
    pub kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Tokens per pool page.
    pub page_size: i32,
}

impl PagedDecodeGeometry {
    /// GQA replication factor `Hq / Hkv`.
    #[must_use]
    pub fn n_rep(&self) -> i32 {
        if self.kv_heads > 0 {
            (self.q_heads / self.kv_heads).max(1)
        } else {
            1
        }
    }

    /// Query heads one CTA owns, from the C++ launcher so the plan's CTA count
    /// and the kernel's grid cannot drift.
    #[must_use]
    pub fn q_heads_per_cta(&self) -> i32 {
        ffi::paged_attention_v2_q_heads_per_cta(self.head_dim, self.n_rep()).max(1)
    }

    /// CTA groups the query heads of one KV head split into.
    #[must_use]
    pub fn q_groups(&self) -> i32 {
        (self.n_rep() / self.q_heads_per_cta()).max(1)
    }

    /// CTAs launched per chunk: one per `(kv head, q-head group)`.
    #[must_use]
    pub fn ctas_per_chunk(&self) -> usize {
        (self.kv_heads.max(1) as usize) * (self.q_groups() as usize)
    }

    /// SIMD groups per CTA, from the C++ launcher. Reported so a plan dump
    /// states the whole launch shape.
    #[must_use]
    pub fn num_warps(&self) -> i32 {
        ffi::paged_attention_v2_num_warps(self.head_dim, self.q_heads_per_cta()).max(1)
    }

    /// Whether the v2 kernels can serve this geometry.
    ///
    /// The kernel partitions the query heads of one KV head into equal CTA
    /// groups, so `Hq` must be a multiple of `Hkv`; everything else is a
    /// positivity check. A geometry that fails here is not an error, it is a
    /// decline: the caller falls back to v1 or to gather.
    pub fn check(&self) -> Result<(), String> {
        if self.q_heads <= 0 || self.kv_heads <= 0 || self.head_dim <= 0 || self.page_size <= 0 {
            return Err(format!(
                "paged decode v2: non-positive geometry (q_heads {}, kv_heads {}, head_dim {}, page_size {})",
                self.q_heads, self.kv_heads, self.head_dim, self.page_size
            ));
        }
        if self.q_heads % self.kv_heads != 0 {
            return Err(format!(
                "paged decode v2: q_heads {} is not a multiple of kv_heads {}",
                self.q_heads, self.kv_heads
            ));
        }
        if self.n_rep() % self.q_heads_per_cta() != 0 {
            return Err(format!(
                "paged decode v2: q_heads_per_cta {} does not divide n_rep {}",
                self.q_heads_per_cta(),
                self.n_rep()
            ));
        }
        Ok(())
    }
}

/// A built plan: plain data, cacheable and reusable across decode steps for as
/// long as the batch composition (per-request page counts) and the geometry are
/// unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedDecodePlan {
    /// Geometry the plan was built for.
    pub geometry: PagedDecodeGeometry,
    /// Requests in the batch.
    pub batch: usize,
    /// The chosen knob: pages one CTA covers.
    pub pages_per_chunk: i32,
    /// `[num_chunks]` request id of each chunk, ascending.
    pub request_indices: Vec<i32>,
    /// `[num_chunks]` chunk index within its request.
    pub kv_tile_indices: Vec<i32>,
    /// `[batch + 1]` merge grouping: output row `r` merges partial rows
    /// `[o_indptr[r], o_indptr[r + 1])`. Fed straight to the merge kernel.
    pub o_indptr: Vec<i32>,
    /// Total chunks (`request_indices.len()`).
    pub num_chunks: usize,
    /// Whether any request needs more than one chunk. When false the partial
    /// kernel's output is already the answer and no merge is launched.
    pub needs_merge: bool,
    /// `num_chunks * ctas_per_chunk`, the parallelism this plan actually
    /// produces.
    pub total_ctas: usize,
    /// The occupancy target the search aimed at.
    pub target_ctas: usize,
    /// Where `pages_per_chunk` came from.
    pub chunk_source: Source,
}

impl PagedDecodePlan {
    /// Build a plan for an explicit chunk size.
    ///
    /// Every request gets at least one chunk, including a request with no
    /// visible tokens: its chunk produces an all-empty partial (`lse = -inf`)
    /// which merges to zeros, so the output always has one row per request and
    /// the caller never has to reassemble a ragged batch.
    #[must_use]
    pub fn with_chunk_size(
        geometry: PagedDecodeGeometry,
        page_counts: &[usize],
        pages_per_chunk: i32,
        target_ctas: usize,
        chunk_source: Source,
    ) -> Self {
        let ppc = pages_per_chunk.max(1);
        let batch = page_counts.len();
        let mut request_indices: Vec<i32> = Vec::with_capacity(batch);
        let mut kv_tile_indices: Vec<i32> = Vec::with_capacity(batch);
        let mut o_indptr: Vec<i32> = Vec::with_capacity(batch + 1);
        o_indptr.push(0);
        let mut needs_merge = false;
        for (r, &pages) in page_counts.iter().enumerate() {
            let chunks = chunks_for_request(pages, ppc);
            if chunks > 1 {
                needs_merge = true;
            }
            for tile in 0..chunks {
                request_indices.push(r as i32);
                kv_tile_indices.push(tile as i32);
            }
            o_indptr.push(request_indices.len() as i32);
        }
        let num_chunks = request_indices.len();
        Self {
            geometry,
            batch,
            pages_per_chunk: ppc,
            request_indices,
            kv_tile_indices,
            o_indptr,
            num_chunks,
            needs_merge,
            total_ctas: num_chunks * geometry.ctas_per_chunk(),
            target_ctas,
            chunk_source,
        }
    }

    /// Build a plan with the heuristic chunk size for `target_ctas`.
    #[must_use]
    pub fn heuristic(
        geometry: PagedDecodeGeometry,
        page_counts: &[usize],
        target_ctas: usize,
    ) -> Self {
        let ppc = search_pages_per_chunk(page_counts, geometry.ctas_per_chunk(), target_ctas);
        Self::with_chunk_size(geometry, page_counts, ppc, target_ctas, Source::Default)
    }

    /// f32 elements of the partial-V workspace the run step will allocate.
    #[must_use]
    pub fn workspace_partial_v_elems(&self) -> usize {
        self.num_chunks
            * self.geometry.q_heads.max(0) as usize
            * self.geometry.head_dim.max(0) as usize
    }

    /// f32 elements of the LSE workspace.
    #[must_use]
    pub fn workspace_lse_elems(&self) -> usize {
        self.num_chunks * self.geometry.q_heads.max(0) as usize
    }

    /// Total workspace bytes.
    ///
    /// MLX owns the allocation (the workspace is the partial kernel's output
    /// arrays, since an MLX custom kernel cannot write through a caller-owned
    /// buffer), so this is a budgeting figure for the scheduler in issue #899,
    /// not a pointer the caller passes in.
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        (self.workspace_partial_v_elems() + self.workspace_lse_elems()) * std::mem::size_of::<f32>()
    }

    /// Whether this plan can still serve a batch with these page counts.
    ///
    /// The plan is cacheable across decode steps, and a decode step appends one
    /// token per request, which changes a page count only when it crosses a
    /// page boundary. This is the cheap check a cache holder makes before
    /// reusing a plan.
    #[must_use]
    pub fn matches(&self, geometry: &PagedDecodeGeometry, page_counts: &[usize]) -> bool {
        if self.geometry != *geometry || self.batch != page_counts.len() {
            return false;
        }
        page_counts.iter().enumerate().all(|(r, &pages)| {
            let chunks = chunks_for_request(pages, self.pages_per_chunk);
            let begin = self.o_indptr[r] as usize;
            let end = self.o_indptr[r + 1] as usize;
            end - begin == chunks
        })
    }

    /// Structural check of the emitted arrays. Cheap relative to a launch and
    /// run on every build, because a malformed plan is an out-of-bounds read
    /// inside the kernel.
    pub fn validate(&self) -> Result<(), String> {
        if self.o_indptr.len() != self.batch + 1 {
            return Err(format!(
                "PagedDecodePlan: o_indptr has {} entries, expected batch + 1 = {}",
                self.o_indptr.len(),
                self.batch + 1
            ));
        }
        if self.request_indices.len() != self.num_chunks
            || self.kv_tile_indices.len() != self.num_chunks
        {
            return Err("PagedDecodePlan: chunk arrays disagree on length".to_string());
        }
        if self.o_indptr.last().copied().unwrap_or(0) as usize != self.num_chunks {
            return Err("PagedDecodePlan: o_indptr does not end at num_chunks".to_string());
        }
        if self.num_chunks > MAX_CHUNKS {
            return Err(format!(
                "PagedDecodePlan: {} chunks exceeds the {MAX_CHUNKS} grid limit",
                self.num_chunks
            ));
        }
        if !self.needs_merge && self.num_chunks != self.batch {
            return Err(
                "PagedDecodePlan: a merge-free plan must have exactly one chunk per request"
                    .to_string(),
            );
        }
        self.geometry.check()
    }
}

/// Chunks one request needs. Always at least one, so every request owns an
/// output row even when it has no visible tokens.
#[must_use]
pub fn chunks_for_request(pages: usize, pages_per_chunk: i32) -> usize {
    let ppc = pages_per_chunk.max(1) as usize;
    pages.div_ceil(ppc).max(1)
}

/// Total chunks for a batch at a given chunk size.
#[must_use]
pub fn chunks_for_batch(page_counts: &[usize], pages_per_chunk: i32) -> usize {
    page_counts
        .iter()
        .map(|&pages| chunks_for_request(pages, pages_per_chunk))
        .sum()
}

/// Largest chunk size worth considering: any larger value produces the same
/// single-chunk-per-request plan.
#[must_use]
pub fn max_pages_per_chunk(page_counts: &[usize]) -> i32 {
    let max_pages = page_counts.iter().copied().max().unwrap_or(0).max(1);
    i32::try_from(max_pages).unwrap_or(i32::MAX)
}

/// Smallest chunk size that keeps the chunk count under [`MAX_CHUNKS`].
///
/// `num_chunks = sum_r ceil(pages_r / ppc) <= total_pages / ppc + batch`, so
/// `ppc >= total_pages / (MAX_CHUNKS - batch)` suffices. A batch at or beyond
/// `MAX_CHUNKS` requests cannot be served at all and is rejected by
/// [`PagedDecodePlan::validate`] rather than silently truncated here.
#[must_use]
pub fn min_pages_per_chunk(page_counts: &[usize]) -> i32 {
    let batch = page_counts.len();
    if batch >= MAX_CHUNKS {
        return i32::MAX;
    }
    let total_pages: usize = page_counts.iter().sum();
    let headroom = MAX_CHUNKS - batch;
    let needed = total_pages.div_ceil(headroom).max(1);
    i32::try_from(needed).unwrap_or(i32::MAX)
}

/// Binary-search the largest chunk size whose CTA count still reaches
/// `target_ctas`.
///
/// `chunks_for_batch` is non-increasing in the chunk size, so the predicate
/// "reaches the target" is monotone and the search is well-defined. The result
/// is clamped up by [`min_pages_per_chunk`] so the returned plan is always
/// launchable.
#[must_use]
pub fn search_pages_per_chunk(
    page_counts: &[usize],
    ctas_per_chunk: usize,
    target_ctas: usize,
) -> i32 {
    let lo_bound = min_pages_per_chunk(page_counts);
    let hi_bound = max_pages_per_chunk(page_counts).max(lo_bound);
    let per_chunk = ctas_per_chunk.max(1);
    let reaches =
        |ppc: i32| chunks_for_batch(page_counts, ppc).saturating_mul(per_chunk) >= target_ctas;

    if !reaches(lo_bound) {
        // Not even the finest feasible split saturates the device; take it and
        // accept the under-occupancy rather than adding merge work for nothing.
        return lo_bound;
    }
    let mut lo = lo_bound;
    let mut hi = hi_bound;
    while lo < hi {
        // Round the midpoint up so `lo` can advance and the loop terminates.
        let mid = lo + (hi - lo + 1) / 2;
        if reaches(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Device CTA target for the heuristic.
///
/// Read once per process. `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS` wins; otherwise
/// Apple hosts derive it from the reported parallelism and everything else
/// takes [`DEFAULT_TARGET_CTAS`].
///
/// The Apple figure is `hardware::gpu_core_count`, which is the
/// performance-core proxy the tree already uses as a device-scale signal (see
/// [`crate::autotune::device_label`]), not a true GPU core count. It scales
/// with the part, which is what the target needs, but it is not calibrated;
/// treat the derived target as a starting point that the autotuner or the env
/// override supersedes.
#[must_use]
pub fn device_target_ctas() -> usize {
    static TARGET: OnceLock<usize> = OnceLock::new();
    *TARGET.get_or_init(|| {
        if let Ok(raw) = std::env::var(TARGET_CTAS_ENV) {
            match raw.trim().parse::<usize>() {
                Ok(v) if v >= 1 => return v,
                _ => tracing::warn!(
                    "{TARGET_CTAS_ENV}={raw:?} is not a positive integer; using the derived target"
                ),
            }
        }
        if crate::metal_is_available() {
            let hw = crate::hardware::get_hardware();
            (hw.gpu_core_count as usize)
                .saturating_mul(CTAS_PER_UNIT)
                .max(MIN_TARGET_CTAS)
        } else {
            DEFAULT_TARGET_CTAS
        }
    })
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod plan_tests;
