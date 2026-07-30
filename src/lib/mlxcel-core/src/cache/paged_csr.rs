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

//! CSR page-table view of an active decode batch (issue #898).
//!
//! The v1 fused kernel takes the block table as four arrays
//! (`rows` / `row_offsets` / `logical_starts` / `visible_lens`) in which `rows`
//! carries *every* block a sequence still owns, visible or not, and the kernel
//! offsets into it by absolute token position. That works, but it cannot
//! express "start this request's page list here", which is what a cross-CTA
//! chunk split needs in order to address a chunk by page range.
//!
//! The v2 kernels take the standard compressed-sparse-row layout instead:
//!
//! | array | shape | meaning |
//! |-------|-------|---------|
//! | `indices` | `[total_pages]` | flat physical pool rows, requests concatenated |
//! | `indptr` | `[B + 1]` | prefix sums delimiting each request's pages |
//! | `last_page_len` | `[B]` | valid entries in the request's final page |
//! | `first_page_offset` | `[B]` | first visible entry in the request's first page |
//!
//! `first_page_offset` is the mlxcel extension to the canonical three-array
//! form. A sequence trimmed by a sliding window keeps
//! [`PagedLayerState::logical_start`] mid-page, and the canonical layout (which
//! assumes every request starts at entry 0 of its first page) cannot say so.
//! With it, request `r`'s visible token `i` resolves to
//!
//! ```text
//! abs   = first_page_offset[r] + i
//! page  = indices[indptr[r] + abs / page_size]
//! entry = abs % page_size
//! ```
//!
//! and its visible length is
//! `(pages - 1) * page_size + last_page_len[r] - first_page_offset[r]`.
//! [`PagedCsrView::validate`] asserts exactly that identity.
//!
//! Only pages that contain visible tokens are emitted. A sliding window that
//! has retired the first three pages of a request drops them from `indices`
//! rather than carrying them and skipping them in the kernel, so the plan's
//! page counts are the real work counts.
//!
//! Shared (refcounted) blocks need no special handling: two requests that share
//! a prefix resolve the same [`PagedBlockId`] to the same physical row, so the
//! row simply appears in both requests' slices of `indices`. That is what makes
//! the same view usable for the cascade decomposition in issue #903.

use super::paged::{PagedBlockId, PagedLayerState};

/// Flat CSR page table for one layer of one decode batch, plus the per-request
/// scalars the v2 kernels and the RoPE position need.
///
/// Plain data: no MLX arrays, no borrows of the pool. It is cheap to build, to
/// compare, and to cache across decode steps while the batch composition is
/// unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PagedCsrView {
    /// Tokens per pool page (the pool's `block_size`).
    pub page_size: i32,
    /// Flat physical pool rows, requests concatenated in block-table order.
    pub indices: Vec<i32>,
    /// `[B + 1]` prefix sums into [`Self::indices`].
    pub indptr: Vec<i32>,
    /// `[B]` valid entries in each request's final page (`0` when the request
    /// has no visible tokens).
    pub last_page_len: Vec<i32>,
    /// `[B]` index of the first visible entry inside each request's first page.
    pub first_page_offset: Vec<i32>,
    /// `[B]` visible token count, derived and kept so the plan does not
    /// recompute it.
    pub seq_lens: Vec<i32>,
    /// `[B]` RoPE position offset: the absolute position the *next* token of
    /// this request will occupy, which is the request's total written length
    /// ([`PagedLayerState::len`]) and not its visible length. A sliding window
    /// trims the visible window from the front without rewinding positions, so
    /// these two differ exactly when `logical_start > 0`.
    pub rope_offsets: Vec<i32>,
}

impl PagedCsrView {
    /// Number of requests in the batch.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.seq_lens.len()
    }

    /// Total pages across the batch (`indices.len()`).
    #[must_use]
    pub fn total_pages(&self) -> usize {
        self.indices.len()
    }

    /// Pages belonging to request `b`, or `0` when out of range.
    #[must_use]
    pub fn pages_for(&self, b: usize) -> usize {
        match (self.indptr.get(b), self.indptr.get(b + 1)) {
            (Some(&begin), Some(&end)) if end >= begin => (end - begin) as usize,
            _ => 0,
        }
    }

    /// Per-request page counts, the input the chunk plan sizes itself from.
    #[must_use]
    pub fn page_counts(&self) -> Vec<usize> {
        (0..self.batch()).map(|b| self.pages_for(b)).collect()
    }

    /// Largest visible context in the batch.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.seq_lens.iter().copied().max().unwrap_or(0).max(0) as usize
    }

    /// Whether any request has a visible token. An all-empty batch has nothing
    /// for the kernel to do, and the caller falls back rather than launching an
    /// empty grid.
    #[must_use]
    pub fn any_visible(&self) -> bool {
        self.seq_lens.iter().any(|&l| l > 0)
    }

    /// Check the structural invariants the v2 kernels rely on.
    ///
    /// Cheap (one pass over the batch, not over the pages) and called on every
    /// build, because every one of these violations is a silent out-of-bounds
    /// read inside the kernel rather than an error.
    pub fn validate(&self) -> Result<(), String> {
        let b = self.batch();
        if self.page_size <= 0 {
            return Err(format!(
                "PagedCsrView: page_size must be positive, got {}",
                self.page_size
            ));
        }
        if self.indptr.len() != b + 1 {
            return Err(format!(
                "PagedCsrView: indptr has {} entries, expected batch + 1 = {}",
                self.indptr.len(),
                b + 1
            ));
        }
        if self.last_page_len.len() != b
            || self.first_page_offset.len() != b
            || self.rope_offsets.len() != b
        {
            return Err(format!(
                "PagedCsrView: per-request arrays disagree on batch size (last_page_len {}, first_page_offset {}, rope_offsets {}, seq_lens {b})",
                self.last_page_len.len(),
                self.first_page_offset.len(),
                self.rope_offsets.len()
            ));
        }
        if self.indptr.first() != Some(&0) {
            return Err("PagedCsrView: indptr must start at 0".to_string());
        }
        if self.indptr.last().copied().unwrap_or(0) as usize != self.indices.len() {
            return Err(format!(
                "PagedCsrView: indptr ends at {:?} but indices has {} entries",
                self.indptr.last(),
                self.indices.len()
            ));
        }
        let page_size = self.page_size;
        for r in 0..b {
            let pages = i64::from(self.indptr[r + 1] - self.indptr[r]);
            if pages < 0 {
                return Err(format!(
                    "PagedCsrView: request {r} has a negative page span"
                ));
            }
            let seq_len = i64::from(self.seq_lens[r]);
            let fpo = i64::from(self.first_page_offset[r]);
            let lpl = i64::from(self.last_page_len[r]);
            if pages == 0 {
                if seq_len != 0 || fpo != 0 || lpl != 0 {
                    return Err(format!(
                        "PagedCsrView: request {r} has no pages but seq_len {seq_len}, first_page_offset {fpo}, last_page_len {lpl}"
                    ));
                }
                continue;
            }
            if fpo < 0 || fpo >= i64::from(page_size) {
                return Err(format!(
                    "PagedCsrView: request {r} first_page_offset {fpo} outside [0, {page_size})"
                ));
            }
            if lpl < 1 || lpl > i64::from(page_size) {
                return Err(format!(
                    "PagedCsrView: request {r} last_page_len {lpl} outside [1, {page_size}]"
                ));
            }
            let expected = (pages - 1) * i64::from(page_size) + lpl - fpo;
            if expected != seq_len {
                return Err(format!(
                    "PagedCsrView: request {r} seq_len {seq_len} contradicts its page geometry (pages {pages}, page_size {page_size}, last_page_len {lpl}, first_page_offset {fpo} => {expected})"
                ));
            }
        }
        Ok(())
    }
}

/// Build the CSR view for one layer of a decode batch.
///
/// `row_of` resolves a [`PagedBlockId`] to its physical pool row for the layer
/// in question; it is a closure so this builder stays free of the pool's
/// private storage and is directly testable against a synthetic mapping.
/// [`crate::cache::PagedBlockPool::paged_csr_view`] is the production caller.
///
/// Errors when a layer's `len` outruns its retained blocks (the same
/// front/back asymmetry guard `gather_visible` and `paged_decode_fused` apply)
/// or when a visible block has no pool row, which means it was never written.
pub fn build_paged_csr_view<F>(
    page_size: usize,
    layers: &[&PagedLayerState],
    mut row_of: F,
) -> Result<PagedCsrView, String>
where
    F: FnMut(PagedBlockId) -> Option<usize>,
{
    if page_size == 0 {
        return Err("build_paged_csr_view: page_size must be > 0".to_string());
    }
    let batch = layers.len();
    let mut view = PagedCsrView {
        page_size: page_size as i32,
        indices: Vec::new(),
        indptr: Vec::with_capacity(batch + 1),
        last_page_len: Vec::with_capacity(batch),
        first_page_offset: Vec::with_capacity(batch),
        seq_lens: Vec::with_capacity(batch),
        rope_offsets: Vec::with_capacity(batch),
    };
    view.indptr.push(0);

    for (r, layer) in layers.iter().enumerate() {
        let n_blocks = layer.block_ids.len();
        if layer.len > n_blocks * page_size {
            return Err(format!(
                "build_paged_csr_view: request {r} len {} exceeds {n_blocks} blocks * page_size {page_size}",
                layer.len
            ));
        }
        let visible = layer.visible_len();
        // `rope_offsets` records the absolute next-token position, so it is the
        // written length even for a front-trimmed sequence.
        view.rope_offsets.push(clamp_i32(layer.len));
        if visible == 0 {
            view.indptr.push(view.indices.len() as i32);
            view.last_page_len.push(0);
            view.first_page_offset.push(0);
            view.seq_lens.push(0);
            continue;
        }

        let first_page = layer.logical_start / page_size;
        let last_page = (layer.len - 1) / page_size;
        for page in first_page..=last_page {
            let block_id = *layer.block_ids.get(page).ok_or_else(|| {
                format!(
                    "build_paged_csr_view: request {r} visible page {page} is past its {n_blocks} retained blocks"
                )
            })?;
            let row = row_of(block_id).ok_or_else(|| {
                format!(
                    "build_paged_csr_view: request {r} block {block_id} has no pool row (was it written?)"
                )
            })?;
            view.indices.push(clamp_i32(row));
        }
        view.indptr.push(view.indices.len() as i32);
        view.last_page_len
            .push(clamp_i32((layer.len - 1) % page_size + 1));
        view.first_page_offset
            .push(clamp_i32(layer.logical_start % page_size));
        view.seq_lens.push(clamp_i32(visible));
    }

    view.validate()?;
    Ok(view)
}

/// Saturating `usize -> i32`. Every field of the view is an i32 because the
/// kernels read them as i32; a pool large enough to overflow would have failed
/// allocation long before, and saturating keeps the conversion total instead of
/// wrapping into a negative index.
fn clamp_i32(v: usize) -> i32 {
    i32::try_from(v).unwrap_or(i32::MAX)
}

#[cfg(test)]
#[path = "paged_csr_tests.rs"]
mod paged_csr_tests;
