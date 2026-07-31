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

//! Per-batch caching of the CSR page table and the chunk plan (issue #899).
//!
//! ## What is expensive and what is not
//!
//! Building a [`PagedCsrView`] costs one hash lookup per **visible page**: at
//! 32768 tokens with 32-token pages that is 1024 lookups per request per layer,
//! and a 32-layer model at batch 4 would pay ~131k lookups every decode step.
//! Refreshing the per-request scalars costs one pass over the **batch**: four
//! `i32` writes per request, so 16 writes for that same batch.
//!
//! A decode step appends exactly one token per request, which changes only the
//! scalars unless the request crosses a page boundary. That is the asymmetry
//! this cache exploits: keep `indices` / `indptr` (the expensive half),
//! recompute `last_page_len` / `first_page_offset` / `seq_lens` /
//! `rope_offsets` (the cheap half) from the live layer states every step.
//!
//! The cheap half is **recomputed, never incremented**. A delta ("bump
//! `last_page_len` by one") would be one missed invalidation away from feeding
//! the kernel a length that outruns its pages, which is an out-of-bounds read
//! rather than a wrong answer. Recomputing from [`PagedLayerState`] costs the
//! same O(batch) and cannot drift.
//!
//! ## Reuse predicate
//!
//! `indices` and `indptr` are a function of exactly two things:
//!
//! 1. Each request's visible **page range**, `logical_start / page_size ..=
//!    (len - 1) / page_size`, which is derived from the per-request
//!    fingerprint below.
//! 2. The block-id to physical-row mapping and the block table contents, which
//!    change only through a `&mut PagedBlockPool` method.
//!
//! (1) is checked directly against the live layer states. (2) is covered by
//! [`PagedBlockPool::block_epoch`], a counter the pool bumps on every mutation
//! that can move a block: acquire, release, row assignment, row forget, block
//! restore, sequence restore, sequence release, and token trim. A cached view
//! is reusable only when the epoch it was built at is still current.
//!
//! The epoch is what makes the issue's named invalidation events fall out for
//! free rather than needing scheduler plumbing:
//!
//! | event | what bumps the epoch |
//! |---|---|
//! | admission of a new sequence | its first `write_prefill` acquires blocks |
//! | eviction / preemption | `release_sequence` releases them |
//! | sequence finish | same |
//! | crossing a page boundary | `append_tokens` acquires the new block |
//! | prompt-cache adopt / detach | `restore_sequence` / `forget` |
//!
//! The epoch is global rather than per-layer, so a boundary crossing costs two
//! fully rebuilt steps rather than one (the crossing step, plus the following
//! step whose per-layer entries were cached at intermediate epoch values). With
//! 32-token pages that is 2 rebuilt steps in 32, or ~6% of the uncached cost.
//!
//! ## Why the pool owns it
//!
//! The issue asks for the plan to be cached "on the active batch". The pool is
//! where that lives in practice: it is the object the decode path already holds
//! mutably, it is the object every invalidation event already goes through, and
//! putting the cache anywhere else would need the scheduler to forward events
//! that the pool observes first hand. See the PR body for the trade-off.
//!
//! [`PagedBlockPool::block_epoch`]: crate::cache::PagedBlockPool::block_epoch

use crate::cache::{PagedCsrView, PagedLayerState};
use crate::paged_v2::plan::{PagedDecodeGeometry, PagedDecodePlan};

/// Everything about one request that can change `indices` / `indptr`.
///
/// Deliberately not a pointer or an id: two different sequences cannot present
/// the same fingerprint at the same epoch, because a sequence entering or
/// leaving the batch moves blocks and therefore moves the epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestFingerprint {
    /// Blocks retained by the layer (the block table length).
    pub blocks: usize,
    /// First visible page index, `logical_start / page_size`.
    pub first_page: usize,
    /// Last written page index, `(len - 1) / page_size`. Meaningless when the
    /// request has no visible tokens, in which case `visible` is false.
    pub last_page: usize,
    /// Whether the request has any visible token at all.
    pub visible: bool,
}

impl RequestFingerprint {
    /// Fingerprint one layer state at a given page size.
    #[must_use]
    pub fn of(layer: &PagedLayerState, page_size: usize) -> Self {
        let page_size = page_size.max(1);
        let visible = layer.visible_len() > 0;
        Self {
            blocks: layer.block_ids.len(),
            first_page: layer.logical_start / page_size,
            last_page: if layer.len == 0 {
                0
            } else {
                (layer.len - 1) / page_size
            },
            visible,
        }
    }
}

/// A cached page table plus the epoch and fingerprints that justify reusing it.
#[derive(Debug, Clone)]
struct CachedView {
    epoch: u64,
    page_size: usize,
    fingerprints: Vec<RequestFingerprint>,
    view: PagedCsrView,
}

/// Counters for the cache's behaviour, surfaced for tests and tracing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanCacheStats {
    /// Views served by refreshing only the per-request scalars.
    pub view_reuses: u64,
    /// Views rebuilt from the block tables.
    pub view_rebuilds: u64,
    /// Plans served from the cached plan.
    pub plan_reuses: u64,
    /// Plans rebuilt (chunk search re-run).
    pub plan_rebuilds: u64,
}

/// Per-pool cache of CSR views (one entry per layer) and the chunk plan.
#[derive(Debug, Default)]
pub struct PagedDecodeV2Cache {
    views: Vec<Option<CachedView>>,
    plan: Option<PagedDecodePlan>,
    stats: PlanCacheStats,
}

impl PagedDecodeV2Cache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop everything. Called by the pool whenever it cannot cheaply reason
    /// about what changed (layout replacement, explicit scheduler request).
    pub fn clear(&mut self) {
        self.views.clear();
        self.plan = None;
    }

    /// Cache counters since construction.
    #[must_use]
    pub fn stats(&self) -> PlanCacheStats {
        self.stats
    }

    /// The CSR view for one layer, reusing the cached page list when the batch
    /// has only advanced within its current pages.
    ///
    /// `build` is invoked only on a miss and must produce the view for exactly
    /// these layer states.
    pub fn view_for<F>(
        &mut self,
        layer_idx: usize,
        epoch: u64,
        page_size: usize,
        layers: &[&PagedLayerState],
        build: F,
    ) -> Result<&PagedCsrView, String>
    where
        F: FnOnce() -> Result<PagedCsrView, String>,
    {
        self.ensure_view(layer_idx, epoch, page_size, layers, build)?;
        Ok(self.view(layer_idx).expect("ensure_view populated the entry"))
    }

    /// The cached view for a layer, if one is present.
    #[must_use]
    pub fn view(&self, layer_idx: usize) -> Option<&PagedCsrView> {
        self.views
            .get(layer_idx)
            .and_then(Option::as_ref)
            .map(|cached| &cached.view)
    }

    /// Resolve both the layer's CSR view and the batch's chunk plan in one
    /// call, so the production path takes a single pass over the cache.
    ///
    /// Returned as a pair of shared borrows: the two live in disjoint fields,
    /// but each is populated under `&mut self`, so they can only be handed out
    /// together after both mutations are done.
    pub fn view_and_plan<FV, FP>(
        &mut self,
        layer_idx: usize,
        epoch: u64,
        page_size: usize,
        layers: &[&PagedLayerState],
        geometry: &PagedDecodeGeometry,
        build_view: FV,
        build_plan: FP,
    ) -> Result<(&PagedCsrView, &PagedDecodePlan), String>
    where
        FV: FnOnce() -> Result<PagedCsrView, String>,
        FP: FnOnce(&[usize]) -> PagedDecodePlan,
    {
        self.ensure_view(layer_idx, epoch, page_size, layers, build_view)?;
        let page_counts = self
            .view(layer_idx)
            .expect("ensure_view populated the entry")
            .page_counts();
        self.ensure_plan(geometry, &page_counts, || build_plan(&page_counts));
        Ok((
            self.view(layer_idx).expect("populated above"),
            self.plan.as_ref().expect("populated above"),
        ))
    }

    fn ensure_view<F>(
        &mut self,
        layer_idx: usize,
        epoch: u64,
        page_size: usize,
        layers: &[&PagedLayerState],
        build: F,
    ) -> Result<(), String>
    where
        F: FnOnce() -> Result<PagedCsrView, String>,
    {
        let fingerprints: Vec<RequestFingerprint> = layers
            .iter()
            .map(|layer| RequestFingerprint::of(layer, page_size))
            .collect();

        if self.views.len() <= layer_idx {
            self.views.resize_with(layer_idx + 1, || None);
        }
        let reusable = match &self.views[layer_idx] {
            Some(cached) => {
                cached.epoch == epoch
                    && cached.page_size == page_size
                    && cached.fingerprints == fingerprints
            }
            None => false,
        };

        if reusable {
            let cached = self.views[layer_idx]
                .as_mut()
                .expect("reusable implies the entry is present");
            refresh_scalars(&mut cached.view, layers, page_size);
            // Cheap (one pass over the batch, not over the pages) and the same
            // assertion the builder runs, so a reused view is held to exactly
            // the invariants a freshly built one is.
            cached.view.validate()?;
            self.stats.view_reuses += 1;
            return Ok(());
        }

        let view = build()?;
        self.stats.view_rebuilds += 1;
        self.views[layer_idx] = Some(CachedView {
            epoch,
            page_size,
            fingerprints,
            view,
        });
        Ok(())
    }

    /// The chunk plan for these page counts, reusing the cached plan when it
    /// still emits the same chunk grouping.
    ///
    /// [`PagedDecodePlan::matches`] is the plan's own O(batch) predicate: it
    /// checks that every request still needs exactly the number of chunks the
    /// cached `o_indptr` allotted it, which is what makes the plan's flat index
    /// arrays still correct.
    pub fn plan_for<F>(
        &mut self,
        geometry: &PagedDecodeGeometry,
        page_counts: &[usize],
        build: F,
    ) -> &PagedDecodePlan
    where
        F: FnOnce() -> PagedDecodePlan,
    {
        self.ensure_plan(geometry, page_counts, build);
        self.plan.as_ref().expect("plan is present after the miss")
    }

    fn ensure_plan<F>(&mut self, geometry: &PagedDecodeGeometry, page_counts: &[usize], build: F)
    where
        F: FnOnce() -> PagedDecodePlan,
    {
        let hit = self
            .plan
            .as_ref()
            .is_some_and(|plan| plan.matches(geometry, page_counts));
        if hit {
            self.stats.plan_reuses += 1;
        } else {
            self.plan = Some(build());
            self.stats.plan_rebuilds += 1;
        }
    }
}

/// Recompute the per-request scalars of a reused view from the live layer
/// states. The page list (`indices` / `indptr`) is untouched; the caller has
/// already established that it is still correct.
fn refresh_scalars(view: &mut PagedCsrView, layers: &[&PagedLayerState], page_size: usize) {
    let page_size = page_size.max(1);
    for (r, layer) in layers.iter().enumerate() {
        let visible = layer.visible_len();
        view.rope_offsets[r] = clamp_i32(layer.len);
        if visible == 0 {
            view.last_page_len[r] = 0;
            view.first_page_offset[r] = 0;
            view.seq_lens[r] = 0;
            continue;
        }
        view.last_page_len[r] = clamp_i32((layer.len - 1) % page_size + 1);
        view.first_page_offset[r] = clamp_i32(layer.logical_start % page_size);
        view.seq_lens[r] = clamp_i32(visible);
    }
}

/// Saturating `usize -> i32`, matching `build_paged_csr_view`'s conversion so a
/// refreshed view and a rebuilt view agree element for element.
fn clamp_i32(v: usize) -> i32 {
    i32::try_from(v).unwrap_or(i32::MAX)
}

#[cfg(test)]
#[path = "plan_cache_tests.rs"]
mod plan_cache_tests;
