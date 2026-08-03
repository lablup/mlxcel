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

//! Sparse attention expressed as page indirection (issue #904).
//!
//! ## The reduction
//!
//! A sparse-attention model decides *which* KV positions a query may see and
//! then, today, pays dense costs anyway: DSA gathers the selected rows into a
//! fresh contiguous copy, block-sparse models build an additive `-inf` mask and
//! let the dense kernel compute the dropped blocks before discarding them.
//!
//! Page indirection removes both. [`crate::cache::paged_csr`] already gives the
//! v2 decode kernel a page table whose `indices` are *arbitrary* physical rows:
//! the kernel resolves visible token `i` of request `r` through
//! `indices[indptr[r] + (fpo + i) / page_size]` and never assumes those rows are
//! adjacent, ordered, or complete. A sparse selection is therefore not a new
//! kernel, it is a **shorter page list**. With `page_size = 1` each "page" is one
//! token, `indices` is exactly the selected-position list, and the kernel's page
//! indirection becomes a per-token gather performed inside the attention loop
//! with no materialized gathered copy at any point.
//!
//! ## Token-exact, not block-granular, is the landing path
//!
//! Issue #904 offers two granularities. This module implements the token-exact
//! one (`page_size = 1`) and documents the other, for two reasons.
//!
//! **Semantics.** Token-exact selection attends to precisely the set the
//! indexer chose, so greedy decode can be required to match the gather/mask
//! implementation exactly. Block-granular selection is a different model, and
//! the quality delta has to be measured on a pinned oracle before it can be
//! accepted; that is a checkpoint-bound experiment, not a code change.
//!
//! **Layout.** Block-granular pages require the pool row of block `j` of request
//! `r` to be `base(r) / C + j`, i.e. the per-request row stride must be a
//! multiple of the block size `C`. That holds for a [`crate::cache::PagedBlockPool`],
//! whose rows *are* blocks, and it is the natural path for a pool-backed cache:
//! feed the selected block ids straight through as `indices` with the pool's own
//! `block_size` as `page_size`. It does **not** hold for the contiguous
//! `[B, H, Cap, D]` allocations that DSA and MiniMax-M3 actually use, whose
//! reserved capacity `Cap` grows in `step` increments unrelated to `C`. Since
//! the models this issue targets are contiguous-cache models, token-exact is
//! the only granularity that applies to them at all.
//!
//! ## Addressing a contiguous cache as a page pool
//!
//! [`ContiguousCacheLayout`] is the piece that makes this zero-copy. A dense
//! `KVCache` allocation is `[B, H, Cap, D]` row-major, and the v2 kernel wants a
//! pool `[N, page_size, Hkv, D]`. At `page_size = 1` and `Hkv = 1` the kernel's
//! address arithmetic collapses to `row * D + d`, so the pool view of the
//! allocation is `[B * H * Cap, 1, 1, D]` — a **pure reshape of the allocation,
//! not of the fetched window**. Reshaping the fetched window would copy, because
//! `KVCache::update_and_fetch` returns `slice(keys, .., .., 0..live_len, ..)` of
//! a step-padded buffer and that slice is strided on the token axis.
//!
//! The KV-head axis is folded into the *request* axis rather than left as the
//! pool's `Hkv`: request `r = b * H_attn + h` owns rows
//! `[base(b, h), base(b, h) + Cap)`. The query reshapes to match
//! (`[B, Hq, 1, D]` to `[B * Hkv, n_rep, 1, D]`, also a pure reshape, because
//! MLX GQA numbers query head `i` under KV head `i / n_rep`), so one CTA still
//! owns one KV head and a group of its query heads exactly as before. Nothing
//! about the kernel's work per CTA changes; only the request count and the page
//! list do.
//!
//! ## Why the cardinality is host-known and no device sync happens
//!
//! The selected *content* is a device value (an `argpartition` output). The
//! selected *count* is not: a top-`k` selection picks `k` of something on every
//! step by construction. That split is what keeps this path off the critical
//! path — `indptr`, `last_page_len`, `first_page_offset` and `seq_lens` are all
//! determined by `(requests, per_request)` and are built on the host without
//! reading a single device element, while `indices` stays a lazy MLX array that
//! is never brought back. [`SparseSelection::materialize`] does read it back,
//! and is for tests and the opt-in dump only.

use cxx::UniquePtr;

use crate::ffi;
use crate::ffi::MlxArray;

use super::paged_csr::PagedCsrView;

/// Page size of a token-exact sparse selection: one token per page, so
/// `indices` is the selected-position list itself.
pub const TOKEN_EXACT_PAGE_SIZE: i32 = 1;

/// Where a selection's pool-row ids live.
///
/// Production builds the device form from the indexer's `argpartition` output
/// and never reads it back. Tests build the host form so the selected set is a
/// plain `Vec<i32>` that can be asserted against directly.
pub enum SparseIndices {
    /// `[requests * per_request]` pool rows, host-resident.
    Host(Vec<i32>),
    /// `[requests * per_request]` i32 MLX array, device-resident and lazy.
    Device(UniquePtr<MlxArray>),
}

impl std::fmt::Debug for SparseIndices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host(v) => f.debug_tuple("Host").field(&v.len()).finish(),
            Self::Device(_) => f.write_str("Device(..)"),
        }
    }
}

/// A uniform-cardinality sparse KV selection, ready to become a page table.
///
/// "Uniform" means every request selects the same number of positions. That is
/// what a top-`k` indexer produces and it is the property that keeps the CSR
/// structure host-known (see the module docs).
#[derive(Debug)]
pub struct SparseSelection {
    /// CSR rows. For a contiguous cache this is `B * H_attn`, one per
    /// `(sequence, KV head)` pair.
    pub requests: usize,
    /// Positions each request selects.
    pub per_request: usize,
    /// `[requests * per_request]` pool rows, request-major.
    pub indices: SparseIndices,
}

impl SparseSelection {
    /// A host-resident selection. `indices` is `[requests * per_request]`
    /// request-major.
    pub fn from_host(
        requests: usize,
        per_request: usize,
        indices: Vec<i32>,
    ) -> Result<Self, String> {
        let sel = Self {
            requests,
            per_request,
            indices: SparseIndices::Host(indices),
        };
        sel.validate()?;
        Ok(sel)
    }

    /// A device-resident selection. `indices` must be an i32 array with
    /// `requests * per_request` elements in request-major order; its shape is
    /// otherwise free, since the kernel reads it flat.
    pub fn from_device(
        requests: usize,
        per_request: usize,
        indices: UniquePtr<MlxArray>,
    ) -> Result<Self, String> {
        let sel = Self {
            requests,
            per_request,
            indices: SparseIndices::Device(indices),
        };
        sel.validate()?;
        Ok(sel)
    }

    /// Total selected positions across the batch.
    #[must_use]
    pub fn total(&self) -> usize {
        self.requests.saturating_mul(self.per_request)
    }

    /// Structural check. Cheap and shape-only: it never touches device data, so
    /// it is safe to call on every step.
    pub fn validate(&self) -> Result<(), String> {
        if self.requests == 0 {
            return Err("SparseSelection: no requests".to_string());
        }
        if self.per_request == 0 {
            return Err("SparseSelection: per_request must be positive".to_string());
        }
        let want = self.total();
        match &self.indices {
            SparseIndices::Host(v) => {
                if v.len() != want {
                    return Err(format!(
                        "SparseSelection: {} host indices for {} requests x {} selected",
                        v.len(),
                        self.requests,
                        self.per_request
                    ));
                }
                if let Some(bad) = v.iter().find(|&&r| r < 0) {
                    return Err(format!("SparseSelection: negative pool row {bad}"));
                }
            }
            SparseIndices::Device(a) => {
                let shape = ffi::array_shape(a);
                let n: i64 = shape.iter().map(|&d| i64::from(d.max(0))).product();
                if n != want as i64 {
                    return Err(format!(
                        "SparseSelection: device indices have {n} elements ({shape:?}) for \
                         {} requests x {} selected",
                        self.requests, self.per_request
                    ));
                }
            }
        }
        Ok(())
    }

    /// The page-table scalars this selection implies, all host-derived.
    ///
    /// At `page_size = 1` every page holds one token, so a request's page count
    /// *is* its visible length: `indptr` strides by `per_request`,
    /// `last_page_len` is 1 and `first_page_offset` is 0. Those satisfy
    /// [`PagedCsrView::validate`]'s geometry identity
    /// `(pages - 1) * page_size + last_page_len - first_page_offset == seq_len`
    /// by construction.
    #[must_use]
    pub fn structure(&self) -> SparseCsrStructure {
        let s = i32::try_from(self.per_request).unwrap_or(i32::MAX);
        SparseCsrStructure {
            page_size: TOKEN_EXACT_PAGE_SIZE,
            indptr: (0..=self.requests)
                .map(|r| i32::try_from(r).unwrap_or(i32::MAX).saturating_mul(s))
                .collect(),
            last_page_len: vec![1; self.requests],
            first_page_offset: vec![0; self.requests],
            seq_lens: vec![s; self.requests],
        }
    }

    /// The selection as a full [`PagedCsrView`], for the host form only.
    ///
    /// This is the bridge to everything `paged_csr` already validates and to
    /// the plan builder's `page_counts`. The device form has no host `indices`
    /// to put in the view, so it goes through [`Self::structure`] plus the
    /// device array instead; that asymmetry is the whole point of keeping the
    /// structure separable.
    pub fn to_csr_view(&self) -> Result<PagedCsrView, String> {
        let SparseIndices::Host(indices) = &self.indices else {
            return Err(
                "SparseSelection::to_csr_view: the device form has no host indices; use \
                 structure() with the device array"
                    .to_string(),
            );
        };
        let structure = self.structure();
        let view = PagedCsrView {
            page_size: structure.page_size,
            indices: indices.clone(),
            indptr: structure.indptr,
            last_page_len: structure.last_page_len,
            first_page_offset: structure.first_page_offset,
            seq_lens: structure.seq_lens,
            rope_offsets: vec![0; self.requests],
        };
        view.validate()?;
        Ok(view)
    }

    /// Read the selected pool rows back to the host, request by request.
    ///
    /// **Synchronizes.** This is the inspection hook: it is what a test asserts
    /// against a dense reference's keep-set and what the
    /// `MLXCEL_SPARSE_PAGED_DUMP` dump prints. Never call it on the decode path.
    pub fn materialize(&self) -> Vec<Vec<i32>> {
        let flat: Vec<i32> = match &self.indices {
            SparseIndices::Host(v) => v.clone(),
            SparseIndices::Device(a) => {
                let as_i32 = ffi::astype(a, crate::dtype::INT32);
                ffi::eval(&as_i32);
                ffi::array_to_raw_bytes(&as_i32)
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes(c.try_into().unwrap_or([0; 4])))
                    .collect()
            }
        };
        flat.chunks(self.per_request.max(1))
            .map(<[i32]>::to_vec)
            .take(self.requests)
            .collect()
    }
}

/// The host-known part of a sparse CSR page table: everything except the rows.
///
/// Split out from [`PagedCsrView`] because the row ids are the only field a
/// sparse decode step cannot know without reading device memory. Keeping the
/// rest separate is what lets the production path build a complete page table
/// with zero synchronization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseCsrStructure {
    /// Always [`TOKEN_EXACT_PAGE_SIZE`] for the token-exact path.
    pub page_size: i32,
    /// `[requests + 1]` prefix sums into the row list.
    pub indptr: Vec<i32>,
    /// `[requests]`, all 1 at `page_size = 1`.
    pub last_page_len: Vec<i32>,
    /// `[requests]`, all 0: a sparse page list starts at its first selected row.
    pub first_page_offset: Vec<i32>,
    /// `[requests]` selected positions per request.
    pub seq_lens: Vec<i32>,
}

impl SparseCsrStructure {
    /// Requests in the batch.
    #[must_use]
    pub fn requests(&self) -> usize {
        self.seq_lens.len()
    }

    /// Per-request page counts, the plan builder's input. At `page_size = 1`
    /// this is the selected-position count.
    #[must_use]
    pub fn page_counts(&self) -> Vec<usize> {
        (0..self.requests())
            .map(|r| {
                let begin = self.indptr[r];
                let end = self.indptr[r + 1];
                (end - begin).max(0) as usize
            })
            .collect()
    }

    /// Selected positions across the whole launch: the work the kernel actually
    /// does, and therefore the number the dispatch floor must be applied to.
    #[must_use]
    pub fn total_selected(&self) -> usize {
        self.seq_lens.iter().map(|&l| l.max(0) as usize).sum()
    }
}

/// A dense `[B, H, Cap, D]` KV allocation addressed as a `page_size = 1` pool.
///
/// `Cap` is the **reserved** token capacity of the allocation, not the live
/// length: the buffer is grown in `step` increments and the live window is a
/// prefix of it, so the row stride between heads is `Cap`, not `live_len`. Using
/// the live length here silently reads the wrong tokens once the buffer has any
/// slack at all, which is almost always.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousCacheLayout {
    /// Sequences in the allocation.
    pub batch: i32,
    /// Heads in the *allocation*. May exceed the heads attention uses: a model
    /// that rides a side stream on the KV head axis (MiniMax-M3's index key)
    /// allocates `H_attn + 1`.
    pub buffer_heads: i32,
    /// Reserved token capacity, i.e. `shape[2]` of the allocation.
    pub capacity: i32,
}

impl ContiguousCacheLayout {
    /// Read the layout off an allocation's shape. Expects `[B, H, Cap, D]`.
    pub fn from_shape(shape: &[i32]) -> Result<Self, String> {
        if shape.len() != 4 {
            return Err(format!(
                "ContiguousCacheLayout: expected a 4-D [B, H, Cap, D] allocation, got {shape:?}"
            ));
        }
        if shape.iter().any(|&d| d <= 0) {
            return Err(format!(
                "ContiguousCacheLayout: non-positive extent in {shape:?}"
            ));
        }
        Ok(Self {
            batch: shape[0],
            buffer_heads: shape[1],
            capacity: shape[2],
        })
    }

    /// First pool row of `(sequence, head)`.
    #[must_use]
    pub fn base(&self, b: i32, h: i32) -> i32 {
        (b.saturating_mul(self.buffer_heads).saturating_add(h)).saturating_mul(self.capacity)
    }

    /// Pool row of token `t` of `(sequence, head)`.
    #[must_use]
    pub fn row(&self, b: i32, h: i32, t: i32) -> i32 {
        self.base(b, h).saturating_add(t)
    }

    /// Rows in the `[N, 1, 1, D]` pool view of this allocation.
    #[must_use]
    pub fn pool_rows(&self) -> i32 {
        self.batch
            .saturating_mul(self.buffer_heads)
            .saturating_mul(self.capacity)
    }
}

/// Whether one row list can address both the K and the V allocation.
///
/// The v2 kernel resolves K and V from the *same* `indices` entry, so a sparse
/// selection is only expressible when both allocations put `(b, h, t)` at the
/// same row. That is automatic when the two allocations have the same shape (the
/// normal case, K and V are reserved together). It also holds, for `B == 1`
/// only, when the two differ in head count: with `b == 0` the base collapses to
/// `h * Cap` on both sides and the extra heads sit past every row the selection
/// can name. Any other combination is a decline, not an error: the caller falls
/// back to its existing gather or mask path.
///
/// `attn_heads` is the number of heads attention actually uses, which is the
/// range `h` is allowed to take.
pub fn shared_row_mapping(
    k: &ContiguousCacheLayout,
    v: &ContiguousCacheLayout,
    attn_heads: i32,
) -> Result<(), String> {
    if attn_heads <= 0 {
        return Err(format!(
            "sparse page table: attn_heads {attn_heads} must be positive"
        ));
    }
    if k.batch != v.batch {
        return Err(format!(
            "sparse page table: K batch {} disagrees with V batch {}",
            k.batch, v.batch
        ));
    }
    if k.capacity != v.capacity {
        return Err(format!(
            "sparse page table: K capacity {} disagrees with V capacity {}; one row list \
             cannot address both",
            k.capacity, v.capacity
        ));
    }
    if attn_heads > k.buffer_heads || attn_heads > v.buffer_heads {
        return Err(format!(
            "sparse page table: {attn_heads} attention heads exceed the allocation \
             ({} K heads, {} V heads)",
            k.buffer_heads, v.buffer_heads
        ));
    }
    for b in 0..k.batch {
        for h in 0..attn_heads {
            if k.base(b, h) != v.base(b, h) {
                return Err(format!(
                    "sparse page table: K and V disagree on the row of (sequence {b}, head {h}) \
                     ({} vs {}); a K allocation with {} heads and a V allocation with {} heads \
                     only share a row mapping at batch 1",
                    k.base(b, h),
                    v.base(b, h),
                    k.buffer_heads,
                    v.buffer_heads
                ));
            }
        }
    }
    Ok(())
}

/// Build a host selection from per-`(sequence, head)` position lists.
///
/// `positions[b][h]` are live-window token indices, which are also buffer slots:
/// `update_and_fetch` returns the window as a prefix of the allocation, so slot
/// `i` is window index `i`. Every list must have the same length, and every
/// position must be inside `live_len` — an out-of-range position would read
/// another request's tokens, since requests are adjacent in the pool view.
///
/// Used by tests and by any caller that already has the selection on the host;
/// the production block-sparse path builds the device form instead.
pub fn selection_from_positions(
    layout: &ContiguousCacheLayout,
    attn_heads: i32,
    live_len: i32,
    positions: &[Vec<Vec<i32>>],
) -> Result<SparseSelection, String> {
    if positions.len() != layout.batch as usize {
        return Err(format!(
            "sparse selection: {} sequences of positions for a batch of {}",
            positions.len(),
            layout.batch
        ));
    }
    if live_len <= 0 || live_len > layout.capacity {
        return Err(format!(
            "sparse selection: live_len {live_len} outside (0, {}]",
            layout.capacity
        ));
    }
    let mut per_request: Option<usize> = None;
    let mut rows: Vec<i32> = Vec::new();
    for (b, heads) in positions.iter().enumerate() {
        if heads.len() != attn_heads as usize {
            return Err(format!(
                "sparse selection: sequence {b} has {} head lists, expected {attn_heads}",
                heads.len()
            ));
        }
        for (h, list) in heads.iter().enumerate() {
            match per_request {
                None => per_request = Some(list.len()),
                Some(s) if s == list.len() => {}
                Some(s) => {
                    return Err(format!(
                        "sparse selection: (sequence {b}, head {h}) selects {} positions but an \
                         earlier request selected {s}; the CSR structure is only host-known when \
                         the cardinality is uniform",
                        list.len()
                    ));
                }
            }
            for &t in list {
                if t < 0 || t >= live_len {
                    return Err(format!(
                        "sparse selection: (sequence {b}, head {h}) selects position {t} outside \
                         [0, {live_len})"
                    ));
                }
                rows.push(layout.row(b as i32, h as i32, t));
            }
        }
    }
    let per_request = per_request.unwrap_or(0);
    SparseSelection::from_host(
        (layout.batch as usize) * (attn_heads as usize),
        per_request,
        rows,
    )
}

/// Expand a device-resident block selection into a device-resident row list.
///
/// This is the block-sparse half of the reduction: a model that selects
/// fixed-size *blocks* of keys still lands on a `page_size = 1` page table,
/// because the contiguous allocations these models use have a per-request row
/// stride (the reserved capacity) that is not a multiple of the block size, so
/// blocks cannot be pages. Expanding block `j` to its `block_size` token rows
/// keeps token-exact semantics and costs one `O(selected)` device pass, which
/// is a fraction of a percent of the attention traffic it replaces.
///
/// `blocks` is `[B, budget]` i32 holding block ids drawn from
/// `[0, tail_start / block_size)`, i.e. the **whole** blocks. The final,
/// possibly partial block is not selected by score and is appended here
/// instead, contributing `live_len - tail_start` rows. Two reasons: it keeps
/// the selected cardinality host-known (the tail width is a function of
/// `live_len`, not of the scores), and it removes what would otherwise be the
/// only way for a selected block to name rows past the live window, which in a
/// pool view are another request's tokens rather than an out-of-bounds error.
/// Callers whose forced-keep rule already pins the query's own block (the
/// `local_blocks >= 1` case) lose nothing by this, since that block *is* the
/// final one at decode.
///
/// Nothing is read back: every step below is an MLX graph node, and the result
/// is a lazy `[B * attn_heads * per_request]` i32 array.
pub fn selection_from_blocks(
    layout: &ContiguousCacheLayout,
    attn_heads: i32,
    block_size: i32,
    blocks: &MlxArray,
    tail_start: i32,
    live_len: i32,
) -> Result<SparseSelection, String> {
    if block_size <= 0 {
        return Err(format!(
            "sparse selection: block_size {block_size} must be positive"
        ));
    }
    if live_len <= 0 || live_len > layout.capacity {
        return Err(format!(
            "sparse selection: live_len {live_len} outside (0, {}]",
            layout.capacity
        ));
    }
    if tail_start < 0 || tail_start >= live_len {
        return Err(format!(
            "sparse selection: tail_start {tail_start} outside [0, {live_len})"
        ));
    }
    if attn_heads <= 0 || attn_heads > layout.buffer_heads {
        return Err(format!(
            "sparse selection: {attn_heads} attention heads outside (0, {}]",
            layout.buffer_heads
        ));
    }
    let block_shape = ffi::array_shape(blocks);
    if block_shape.len() != 2 || block_shape[0] != layout.batch || block_shape[1] < 1 {
        return Err(format!(
            "sparse selection: expected a [{}, budget] block list, got {block_shape:?}",
            layout.batch
        ));
    }
    let budget = block_shape[1];
    let tail = live_len - tail_start;
    let per_request = (budget as i64) * i64::from(block_size) + i64::from(tail);
    let per_request = usize::try_from(per_request)
        .map_err(|_| format!("sparse selection: {per_request} rows per request overflows"))?;

    // Whole blocks: `blocks[b, j] * block_size + [0, block_size)`.
    let scaled = ffi::multiply(
        &ffi::astype(blocks, crate::dtype::INT32),
        &ffi::from_slice_i32(&[block_size], &[1]),
    );
    let scaled = ffi::reshape(&scaled, &[layout.batch, budget, 1]);
    let within = ffi::reshape(&ffi::arange_i32(0, block_size, 1), &[1, 1, block_size]);
    let body = ffi::add(&scaled, &within);
    let body = ffi::reshape(&body, &[layout.batch, budget * block_size]);

    // The final, possibly partial block, appended for every sequence.
    let tail_tokens = ffi::reshape(&ffi::arange_i32(tail_start, live_len, 1), &[1, tail]);
    let tail_tokens = ffi::broadcast_to(&tail_tokens, &[layout.batch, tail]);
    let tokens = crate::ops::concatenate(&body, &tail_tokens, 1);

    // Fold the head axis in: request `(b, h)` owns rows starting at
    // `(b * buffer_heads + h) * capacity`.
    let seq_base = ffi::multiply(
        &ffi::arange_i32(0, layout.batch, 1),
        &ffi::from_slice_i32(&[layout.buffer_heads.saturating_mul(layout.capacity)], &[1]),
    );
    let seq_base = ffi::reshape(&seq_base, &[layout.batch, 1, 1]);
    let head_base = ffi::multiply(
        &ffi::arange_i32(0, attn_heads, 1),
        &ffi::from_slice_i32(&[layout.capacity], &[1]),
    );
    let head_base = ffi::reshape(&head_base, &[1, attn_heads, 1]);
    let base = ffi::add(&seq_base, &head_base);

    let tokens = ffi::reshape(&tokens, &[layout.batch, 1, per_request as i32]);
    let rows = ffi::add(&base, &tokens);
    let total = (layout.batch as i64) * i64::from(attn_heads) * (per_request as i64);
    let total = i32::try_from(total)
        .map_err(|_| format!("sparse selection: {total} total rows overflows i32"))?;
    let rows = ffi::reshape(&rows, &[total]);

    SparseSelection::from_device(
        (layout.batch as usize) * (attn_heads as usize),
        per_request,
        rows,
    )
}

#[cfg(test)]
#[path = "sparse_csr_tests.rs"]
mod sparse_csr_tests;
