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

//! Paged KV cache detach / adopt / park primitives.
//!
//! Mirrors the dense surface implemented in `cache/detach.rs` for the paged
//! decode backend. Paged sequences carry two independent pieces of state:
//!
//! 1. A `PagedSequenceState` per sequence — block table plus logical lengths.
//! 2. Dense `KVCache` placeholders (`SequenceCacheSet::caches`) that the
//!    scheduler mirrors into the paged state today so model forward still
//!    sees standard dense tensors.
//!
//! Detach captures **both** halves so adopt can produce a byte-for-byte
//! equivalent sequence, and pins every physical block referenced by the
//! sequence via a refcount bump on the shared [`PagedBlockPool`]. The block
//! pool therefore refuses to recycle any block referenced by a live detached
//! set, which is what makes shared-prefix adoption safe across sequences.
//!
//! ## Copy-on-write prefix sharing
//!
//! Because detach refcounts blocks instead of copying them, calling
//! [`CachePool::detach_paged`] followed by [`CachePool::adopt_paged`] twice
//! against the same parked set yields two sequences that share the prefix
//! blocks until either of them `append_paged_tokens`. Append always asks the
//! pool for a fresh block rather than mutating a shared one, so the first
//! write triggers copy-on-write automatically — there is no per-block fork
//! bookkeeping required at the cache layer.
//!
//! ## INT8 preservation
//!
//! Just like the dense path, detached INT8 scale tensors travel inside the
//! per-layer `DetachedKVCache` entries, so paged INT8 sequences round-trip
//! without any quantization loss on top of the one already introduced by the
//! live cache.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cxx::UniquePtr;

use crate::ffi;
use crate::ffi::MlxArray;

use super::detach::{DetachedHandle, DetachedKVCache};
use super::paged::{PagedBlockId, PagedBlockPool, PagedKvLayout, PagedSequenceState};
use super::{CachePool, KVCache, KVCacheMode, SequenceCacheSet, SequenceId, SequenceStateBackend};

/// Inert snapshot of a paged-backend sequence, analogous to
/// [`super::DetachedCacheSet`] for the dense backend.
///
/// Owns every piece of state required to reconstruct the sequence later:
/// per-layer dense `KVCache` handles, the full paged block table, the layout
/// the block pool was built with, and a refcount pin on each physical block
/// reachable from the block table (so the shared [`PagedBlockPool`] cannot
/// recycle any of them while this set is alive).
///
/// Paged sets deliberately do not expose raw public fields the way the dense
/// variant does; all legitimate construction goes through
/// [`CachePool::detach_paged`] so the refcount pin is maintained as an
/// invariant.
pub struct DetachedPagedCacheSet {
    pub(super) caches: Vec<DetachedKVCache>,
    pub(super) paged_state: PagedSequenceState,
    pub(super) paged_layout: PagedKvLayout,
    /// Layout tag — always [`SequenceStateBackend::PagedKvCache`] for this
    /// struct but tracked explicitly for symmetry with the dense variant.
    pub(super) backend: SequenceStateBackend,
    pub(super) prompt_len: usize,
    pub(super) current_offset: i32,
    pub(super) created_at: Instant,
    pub(super) detached_at: Instant,
    pub(super) origin_seq_id: SequenceId,
    /// When `Some`, the set still owns block pins that must be released via
    /// `CachePool::adopt_paged` or `CachePool::release_detached_paged`. Set to
    /// `None` after successful adoption so `Drop` can run without triggering
    /// a leak warning.
    pub(super) retained_blocks: Option<Vec<PagedBlockId>>,
    /// Per-page Turbo4 sidecar tensors lifted out of the originating
    /// [`PagedBlockPool`] at detach time. Keyed by `PagedBlockId`. Empty for
    /// `Fp16`/`Int8` cache modes; on adopt the entries are reinstalled into
    /// the active pool so quantization state survives the round-trip
    /// bit-identically.
    pub(super) v_packed_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
    pub(super) v_norms_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
    pub(super) k_packed_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
    pub(super) k_norms_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
    pub(super) cold_keys_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
}

impl DetachedPagedCacheSet {
    /// Number of layers carried by this set.
    pub fn num_layers(&self) -> usize {
        self.paged_state.layers.len()
    }

    /// Logical token length of the first layer (all paged layers share the
    /// visible prefix length by construction).
    pub fn seq_len(&self) -> usize {
        self.paged_state
            .layers
            .first()
            .map(|layer| layer.visible_len())
            .unwrap_or(0)
    }

    /// Backend tag of this set (always [`SequenceStateBackend::PagedKvCache`]).
    pub fn backend(&self) -> SequenceStateBackend {
        self.backend
    }

    /// The originating sequence id, preserved across detach for observability.
    pub fn origin_seq_id(&self) -> SequenceId {
        self.origin_seq_id
    }

    /// Timestamp when this set was produced by [`CachePool::detach_paged`].
    ///
    /// Preserved across park/adopt so schedulers can compute wait-time
    /// metrics on cross-request cache handoffs.
    pub fn detached_at(&self) -> Instant {
        self.detached_at
    }

    /// Timestamp when the original sequence was first allocated in the
    /// `CachePool`.
    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    /// Prompt length at the time the sequence was detached.
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// Decode offset at the time the sequence was detached.
    pub fn current_offset(&self) -> i32 {
        self.current_offset
    }

    /// Visible read-only access to the per-layer dense placeholder caches.
    pub fn dense_caches(&self) -> &[DetachedKVCache] {
        &self.caches
    }

    /// Paged layout the set was captured under.
    pub fn layout(&self) -> &PagedKvLayout {
        &self.paged_layout
    }

    /// Paged block-table state captured at detach time.
    pub fn paged_state(&self) -> &PagedSequenceState {
        &self.paged_state
    }

    /// Total byte footprint — sum of dense tensor bytes plus the bytes
    /// reserved in the paged block pool for this sequence and any per-page
    /// Turbo4 sidecar tensors carried directly by this set.
    pub fn nbytes(&self) -> usize {
        let dense: usize = self.caches.iter().map(|c| c.nbytes()).sum();
        let paged: usize = self.paged_state.reserved_bytes(&self.paged_layout);
        let sidecars: usize = self.turbo_sidecar_bytes();
        dense + paged + sidecars
    }

    /// Summed bytes of the retained paged blocks only (excluding dense
    /// placeholders). Useful for memory-usage attribution when the caller
    /// wants to split the two contributions.
    pub fn paged_bytes(&self) -> usize {
        self.paged_state.reserved_bytes(&self.paged_layout)
    }

    /// Summed bytes of the per-page Turbo4 sidecar tensors carried by this
    /// set. Always `0` for `Fp16`/`Int8` round-trips.
    pub fn turbo_sidecar_bytes(&self) -> usize {
        let sum_map = |m: &HashMap<PagedBlockId, UniquePtr<MlxArray>>| -> usize {
            m.values().map(|a| ffi::array_nbytes(a)).sum()
        };
        sum_map(&self.v_packed_pages)
            + sum_map(&self.v_norms_pages)
            + sum_map(&self.k_packed_pages)
            + sum_map(&self.k_norms_pages)
            + sum_map(&self.cold_keys_pages)
    }

    /// Number of physical blocks this set pins across all layers.
    pub fn retained_block_count(&self) -> usize {
        self.retained_blocks
            .as_ref()
            .map(|blocks| blocks.len())
            .unwrap_or(0)
    }

    /// Cache mode the set was captured under. Used by adopt to know which
    /// Turbo4 sidecar pages to reinstall.
    pub fn cache_mode(&self) -> KVCacheMode {
        self.paged_layout.cache_mode
    }

    /// Number of distinct pages carrying any Turbo4 sidecar tensor.
    pub fn turbo_sidecar_page_count(&self) -> usize {
        let mut all: std::collections::HashSet<PagedBlockId> = std::collections::HashSet::new();
        all.extend(self.v_packed_pages.keys());
        all.extend(self.v_norms_pages.keys());
        all.extend(self.k_packed_pages.keys());
        all.extend(self.k_norms_pages.keys());
        all.extend(self.cold_keys_pages.keys());
        all.len()
    }
}

impl std::fmt::Debug for DetachedPagedCacheSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedPagedCacheSet")
            .field("backend", &self.backend)
            .field("cache_mode", &self.paged_layout.cache_mode)
            .field("num_layers", &self.num_layers())
            .field("seq_len", &self.seq_len())
            .field("prompt_len", &self.prompt_len)
            .field("current_offset", &self.current_offset)
            .field("origin_seq_id", &self.origin_seq_id)
            .field("retained_blocks", &self.retained_block_count())
            .field("turbo_sidecar_pages", &self.turbo_sidecar_page_count())
            .finish()
    }
}

impl Drop for DetachedPagedCacheSet {
    fn drop(&mut self) {
        // Successful adopt or release clears `retained_blocks`. If we still
        // own pins at drop time the caller forgot to return us to the pool;
        // warn so the leak is visible rather than silently burning block
        // budget forever.
        if let Some(blocks) = self.retained_blocks.take() {
            if !blocks.is_empty() {
                eprintln!(
                    "[mlxcel::cache::paged_detach] DetachedPagedCacheSet dropped with {} retained blocks (origin seq {}); \
                     use CachePool::release_detached_paged or adopt_paged to release pins",
                    blocks.len(),
                    self.origin_seq_id
                );
            }
        }
    }
}

/// Internal parked-cache representation. Used by the pool so both dense and
/// paged detached sets share a single [`DetachedHandle`] space.
#[allow(clippy::large_enum_variant)]
pub(super) enum ParkedCache {
    Dense(super::detach::DetachedCacheSet),
    Paged(DetachedPagedCacheSet),
}

impl ParkedCache {
    pub(super) fn nbytes(&self) -> usize {
        match self {
            ParkedCache::Dense(set) => set.nbytes(),
            ParkedCache::Paged(set) => set.nbytes(),
        }
    }
}

/// Take a shared paged-state handle by value, producing an owned
/// [`PagedSequenceState`] for an inert detached set.
///
/// In the common case the handle is the sole owner (a dense-placeholder paged
/// sequence, or a pool-backed sequence whose per-layer caches were already
/// dropped) and `Rc::try_unwrap` hands back the inner value with no copy. When
/// other `Rc` clones are still live — e.g. a pool-backed sequence detached
/// while its caches still hold sibling handles (the #121 sub-step (b) radix
/// path) — fall back to cloning the block table, which is cheap (`Vec<u64>` per
/// layer) and behaviour-equivalent: the clone carries the identical block ids,
/// and the originals are released without touching pool refcounts (dropping a
/// `PagedSequenceState` does not release blocks). The detached set's refcount
/// pins are taken separately by the caller via `retain_block`.
fn into_owned_paged_state(state: Rc<RefCell<PagedSequenceState>>) -> PagedSequenceState {
    match Rc::try_unwrap(state) {
        Ok(cell) => cell.into_inner(),
        Err(shared) => shared.borrow().clone(),
    }
}

impl CachePool {
    /// Remove a paged-backed sequence from the active set and return it as an
    /// inert [`DetachedPagedCacheSet`].
    ///
    /// Returns `None` if `seq_id` is not active or is not using the paged
    /// backend. Every physical block referenced by the sequence has its
    /// refcount incremented so the shared [`PagedBlockPool`] will not recycle
    /// any of them while the detached set is alive — the set is responsible
    /// for releasing those pins on adopt or explicit release.
    ///
    /// Used by: prompt prefix cache store, scheduler request-boundary
    /// handoff for `DecodeStorageBackend::Paged`.
    pub fn detach_paged(&mut self, seq_id: SequenceId) -> Option<DetachedPagedCacheSet> {
        {
            let sequence = self.active.get(&seq_id)?;
            if sequence.backend != SequenceStateBackend::PagedKvCache {
                return None;
            }
            if sequence.paged.is_none() || sequence.paged_layout().is_none() {
                return None;
            }
        }

        let mut sequence = self.active.remove(&seq_id)?;
        let paged_layout = sequence
            .paged_layout()
            .expect("paged sequences must carry a layout")
            .clone();
        let paged_state_rc = sequence
            .paged
            .take()
            .expect("paged sequences must carry a paged state");
        // Take the block table by value. For sub-step (a) paged sequences are
        // dense-placeholder (refcount 1) so this never copies; the clone
        // fallback inside the helper covers the forthcoming pool-backed detach
        // (#121 sub-step b).
        let paged_state = into_owned_paged_state(paged_state_rc);

        // Pin every block so the pool cannot recycle prefix pages out from
        // under us. Collect block ids for later release; this also serves as
        // the retained set exposed to callers.
        let mut retained: Vec<PagedBlockId> = Vec::new();
        for layer in &paged_state.layers {
            for &block_id in &layer.block_ids {
                retained.push(block_id);
            }
        }

        // Pin under a single pool borrow, then drop it before any mutation of
        // `self.active` (the rollback `insert` below). Pool and `active` are
        // distinct fields, but keeping the borrow scoped avoids any future
        // nesting hazard.
        let pin_failed: Option<PagedBlockId> = if let Some(pool) = self.paged_pool.as_ref() {
            let mut pool = pool.borrow_mut();
            let mut failed = None;
            for &block_id in &retained {
                // retain_block only fails if the block is unknown or already
                // at refcount zero — neither should happen for a block that
                // was just live in a sequence, so log and bail on surprise.
                if let Err(err) = pool.retain_block(block_id) {
                    eprintln!(
                        "[mlxcel::cache::paged_detach] CachePool::detach_paged: failed to pin block {block_id}: {err}; \
                         reinstalling sequence {seq_id}"
                    );
                    // Roll back the partial pins we already applied.
                    let pinned_so_far: Vec<PagedBlockId> = retained
                        .iter()
                        .copied()
                        .take_while(|id| *id != block_id)
                        .collect();
                    for id in pinned_so_far.iter().rev() {
                        let _ = pool.release_block(*id);
                    }
                    failed = Some(block_id);
                    break;
                }
            }
            failed
        } else {
            None
        };

        if pin_failed.is_some() {
            // Restore the original entry so the caller sees the sequence as if
            // detach had simply declined. The pool borrow above is already
            // dropped here.
            sequence.paged = Some(Rc::new(RefCell::new(paged_state)));
            self.active.insert(seq_id, sequence);
            return None;
        }

        let detached_caches: Vec<DetachedKVCache> = sequence
            .caches
            .iter_mut()
            .map(|cache| cache.clone_handle())
            .collect();

        // Lift per-page Turbo4 sidecar tensors out of the pool so the round-
        // trip captures them losslessly. For non-Turbo4 modes every map stays
        // empty and the dispatch below is a no-op.
        let mut v_packed_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>> = HashMap::new();
        let mut v_norms_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>> = HashMap::new();
        let mut k_packed_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>> = HashMap::new();
        let mut k_norms_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>> = HashMap::new();
        let mut cold_keys_pages: HashMap<PagedBlockId, UniquePtr<MlxArray>> = HashMap::new();
        if paged_layout.is_turbo_mode() {
            if let Some(pool) = self.paged_pool.as_ref() {
                let mut pool = pool.borrow_mut();
                for &block_id in &retained {
                    if let Some(t) = pool.take_v_packed(block_id) {
                        v_packed_pages.insert(block_id, t);
                    }
                    if let Some(t) = pool.take_v_norms(block_id) {
                        v_norms_pages.insert(block_id, t);
                    }
                    if paged_layout.cache_mode == KVCacheMode::Turbo4 {
                        if let Some(t) = pool.take_k_packed(block_id) {
                            k_packed_pages.insert(block_id, t);
                        }
                        if let Some(t) = pool.take_k_norms(block_id) {
                            k_norms_pages.insert(block_id, t);
                        }
                    }
                    if paged_layout.cache_mode == KVCacheMode::Turbo4Delegated {
                        if let Some(t) = pool.take_cold_keys(block_id) {
                            cold_keys_pages.insert(block_id, t);
                        }
                    }
                }
            }
        }

        Some(DetachedPagedCacheSet {
            caches: detached_caches,
            paged_state,
            paged_layout,
            backend: sequence.backend,
            prompt_len: sequence.prompt_len,
            current_offset: sequence.current_offset,
            created_at: sequence.created_at,
            detached_at: Instant::now(),
            origin_seq_id: sequence.seq_id,
            retained_blocks: Some(retained),
            v_packed_pages,
            v_norms_pages,
            k_packed_pages,
            k_norms_pages,
            cold_keys_pages,
        })
    }

    /// Install a previously-detached paged cache set under a fresh
    /// [`SequenceId`].
    ///
    /// The retained block pins are transferred onto the new sequence and the
    /// model's [`prepare_sequence_state`](crate::generate::LanguageModel::prepare_sequence_state)
    /// hook is invoked, matching the dense `adopt` semantics. Paged-layout
    /// mismatches with the active pool or capacity exhaustion return an error
    /// **and** release all block pins so the caller never has to hand-roll
    /// cleanup.
    ///
    /// Used by: prompt prefix cache re-entry, scheduler fast-path for
    /// paged decode sequences.
    pub fn adopt_paged(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        detached: DetachedPagedCacheSet,
    ) -> Result<SequenceId, String> {
        self.adopt_paged_preserving(model, detached)
            .map_err(|(err, set)| {
                // `adopt_paged_preserving` does not consume pins on error.
                // We release them so the error path cannot leak.
                self.release_detached_paged(set);
                err
            })
    }

    /// Like [`CachePool::adopt_paged`] but returns the original detached set
    /// back to the caller on failure so it can be retried elsewhere (e.g. on
    /// a different pool or after evicting active sequences).
    ///
    /// Retained block pins are preserved across failure paths so the caller
    /// can still call [`CachePool::release_detached_paged`] manually or retry.
    #[allow(clippy::result_large_err)]
    pub fn adopt_paged_preserving(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        detached: DetachedPagedCacheSet,
    ) -> Result<SequenceId, (String, DetachedPagedCacheSet)> {
        if detached.backend != SequenceStateBackend::PagedKvCache {
            return Err((
                format!(
                    "CachePool::adopt_paged: expected paged backend, got {:?}",
                    detached.backend
                ),
                detached,
            ));
        }

        // Validate layout compatibility with the active paged pool (if any).
        if let Some(pool) = self.paged_pool.as_ref() {
            let pool_block_size = {
                let pool = pool.borrow();
                if pool.layout() != &detached.paged_layout {
                    Some(pool.layout().block_size)
                } else {
                    None
                }
            };
            if let Some(pool_block_size) = pool_block_size {
                return Err((
                    format!(
                        "CachePool::adopt_paged: paged layout mismatch (pool block_size={}, set block_size={})",
                        pool_block_size, detached.paged_layout.block_size
                    ),
                    detached,
                ));
            }
        }

        if self.active.len() >= self.max_sequences {
            return Err((
                format!(
                    "CachePool::adopt_paged: max capacity ({}) reached, cannot adopt new sequence",
                    self.max_sequences
                ),
                detached,
            ));
        }

        if let Err(err) = self.ensure_paged_pool_from_layout(&detached.paged_layout) {
            return Err((err, detached));
        }

        // `DetachedPagedCacheSet` implements `Drop`, so we cannot destructure
        // it by value; instead move each field out in turn and then
        // `forget` the husk so the `Drop` impl does not re-run on the now
        // partially-moved value. Clearing `retained_blocks` to `None` is
        // already enough to silence the leak warning, but `std::mem::forget`
        // also avoids the fake "silent pass" warning path entirely.
        let mut detached = detached;
        let caches = std::mem::take(&mut detached.caches);
        let paged_state = std::mem::replace(
            &mut detached.paged_state,
            PagedSequenceState::new(&detached.paged_layout),
        );
        let paged_layout = detached.paged_layout.clone();
        let backend = detached.backend;
        let prompt_len = detached.prompt_len;
        let current_offset = detached.current_offset;
        let created_at = detached.created_at;
        let retained_blocks = detached.retained_blocks.take();
        // Lift the per-page Turbo4 sidecars out of the detached set so we can
        // reinstall them on the active pool below. For Fp16/Int8
        // round-trips every map is empty and the work below short-circuits.
        let v_packed_pages = std::mem::take(&mut detached.v_packed_pages);
        let v_norms_pages = std::mem::take(&mut detached.v_norms_pages);
        let k_packed_pages = std::mem::take(&mut detached.k_packed_pages);
        let k_norms_pages = std::mem::take(&mut detached.k_norms_pages);
        let cold_keys_pages = std::mem::take(&mut detached.cold_keys_pages);

        let id = SequenceId::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed));
        let state_rc = Rc::new(RefCell::new(paged_state));

        // Decide pool-backing with the SAME gate `CachePool::allocate_with_layout`
        // uses for a fresh paged allocation, so an adopted sequence is
        // byte-identical in shape to one that allocated cold:
        //
        //  * the model's NATURAL backend is the dense external `KVCache` slice
        //    (`supports_batching()` transformers — qwen3 / llama3), AND
        //  * the paged layout is `Fp16`.
        //
        // When both hold, the live per-layer caches must be POOL-BACKED
        // (`new_paged`) sharing the adopted block-table `state_rc` and the
        // active pool. The block table already points at the pinned, refcounted
        // pool blocks (detach captured it via `into_owned_paged_state` + pinned
        // every block), so reads gather straight from the shared prefix pages
        // with no copy. The empty `clone_handle` dense handles captured at
        // detach time are intentionally ignored on this path — rebuilding dense
        // caches from them (the pre-#121 behaviour) would hand the adopted
        // sequence empty buffers and make it read garbage.
        let pool_backed = model.sequence_state_layout().backend
            == SequenceStateBackend::DenseKvCache
            && paged_layout.cache_mode == KVCacheMode::Fp16;

        let live_caches: Vec<KVCache> = if pool_backed {
            let pool_rc = self
                .paged_pool
                .as_ref()
                .expect("ensure_paged_pool_from_layout installed the pool above")
                .clone();
            // One pool-backed cache per layer, sharing the adopted block-table
            // `state_rc`. Each cache's monotonic `offset` is restored from the
            // detached handle (which `clone_handle` captured from the
            // originating pool-backed cache at detach time) so RoPE positions
            // for the post-adopt prefill/decode continue from the cached prefix
            // length — a fresh `new_paged` cache would start at offset 0 and
            // mis-rotate the suffix. The pool/state already hold the prefix
            // K/V; only this dense-side offset bookkeeping needs carrying over.
            caches
                .into_iter()
                .enumerate()
                .map(|(layer_idx, handle)| {
                    let mut cache =
                        KVCache::new_paged(pool_rc.clone(), state_rc.clone(), layer_idx);
                    cache.offset = handle.offset;
                    cache
                })
                .collect()
        } else {
            // Dense-from-handles reconstruction (model-owned families, or any
            // non-`Fp16` paged layout that keeps dense quantized placeholders).
            let mut live = Vec::with_capacity(caches.len());
            for detached_cache in caches {
                let mut cache = KVCache::new_with_mode(detached_cache.mode);
                cache
                    .install_detached(detached_cache)
                    .expect("freshly constructed KVCache is empty");
                live.push(cache);
            }
            live
        };

        let mut entry = SequenceCacheSet::paged(id, paged_layout.clone());
        entry.caches = live_caches;
        entry.paged = Some(state_rc);
        entry.backend = backend;
        entry.prompt_len = prompt_len;
        entry.current_offset = current_offset;
        entry.created_at = created_at;
        self.active.insert(id, entry);

        // Reinstall the per-page Turbo4 sidecars into the active pool. The
        // installs are best-effort: if the pool fails to accept a sidecar we
        // log and continue so adopt does not partially fail with a half-
        // constructed sequence. The detach-side `take_*` calls already
        // consumed the originals, so a failure here would otherwise drop
        // them silently.
        if paged_layout.is_turbo_mode() {
            if let Some(pool) = self.paged_pool.as_ref() {
                let mut pool = pool.borrow_mut();
                for (block_id, t) in v_packed_pages {
                    if let Err(err) = pool.install_v_packed(block_id, t) {
                        eprintln!(
                            "[mlxcel::cache::paged_detach] adopt_paged: failed to reinstall v_packed for {block_id}: {err}"
                        );
                    }
                }
                for (block_id, t) in v_norms_pages {
                    if let Err(err) = pool.install_v_norms(block_id, t) {
                        eprintln!(
                            "[mlxcel::cache::paged_detach] adopt_paged: failed to reinstall v_norms for {block_id}: {err}"
                        );
                    }
                }
                for (block_id, t) in k_packed_pages {
                    if let Err(err) = pool.install_k_packed(block_id, t) {
                        eprintln!(
                            "[mlxcel::cache::paged_detach] adopt_paged: failed to reinstall k_packed for {block_id}: {err}"
                        );
                    }
                }
                for (block_id, t) in k_norms_pages {
                    if let Err(err) = pool.install_k_norms(block_id, t) {
                        eprintln!(
                            "[mlxcel::cache::paged_detach] adopt_paged: failed to reinstall k_norms for {block_id}: {err}"
                        );
                    }
                }
                for (block_id, t) in cold_keys_pages {
                    if let Err(err) = pool.install_cold_keys(block_id, t) {
                        eprintln!(
                            "[mlxcel::cache::paged_detach] adopt_paged: failed to reinstall cold_keys for {block_id}: {err}"
                        );
                    }
                }
            }
        }

        // Transfer retained pins onto the new active sequence. Each block was
        // pinned twice at detach time (once by the original block_ids vector,
        // once by our refcount bump), and the new sequence's block_ids vec
        // now holds the original pin. Releasing the detach-side bump restores
        // the invariant "refcount == number of active block_ids entries
        // owning the block".
        if let Some(blocks) = retained_blocks {
            if let Some(pool) = self.paged_pool.as_ref() {
                let mut pool = pool.borrow_mut();
                for block_id in blocks {
                    if let Err(err) = pool.release_block(block_id) {
                        // Fatal: the pool now disagrees with the sequence
                        // state. Surface loudly but do not unwind — unwinding
                        // here would leave the new sequence in an even worse
                        // state.
                        eprintln!(
                            "[mlxcel::cache::paged_detach] CachePool::adopt_paged: failed to drop detach pin on block {block_id}: {err}"
                        );
                    }
                }
            }
        }

        // The model hook must run with NO pool/state borrow held (it may, for
        // model-owned families, touch the cache pool again). All `borrow_mut`s
        // above are scoped to their `if let` blocks and already dropped here.
        model.prepare_sequence_state(id);
        // `detached.retained_blocks` is `None` now, so the `Drop` impl will
        // run quietly and not complain about leaked pins.
        drop(detached);
        Ok(id)
    }

    /// Park a paged detached set so its bytes remain counted by
    /// [`CachePool::memory_usage_bytes`] across cross-request handoffs.
    ///
    /// Returns an opaque [`DetachedHandle`] that shares the handle space with
    /// [`CachePool::park_detached`] (dense), so a single scheduler can route
    /// both backends through one map.
    pub fn park_detached_paged(&mut self, detached: DetachedPagedCacheSet) -> DetachedHandle {
        let handle = DetachedHandle::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.detached.insert(handle, ParkedCache::Paged(detached));
        handle
    }

    /// Convenience: consume a parked paged handle and re-adopt it in one call.
    pub fn adopt_parked_paged(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        handle: DetachedHandle,
    ) -> Result<SequenceId, String> {
        match self.detached.remove(&handle) {
            Some(ParkedCache::Paged(set)) => self.adopt_paged(model, set),
            Some(ParkedCache::Dense(dense)) => {
                // Put it back so the caller can retry with the correct API.
                self.detached.insert(handle, ParkedCache::Dense(dense));
                Err(format!(
                    "CachePool::adopt_parked_paged: handle {handle} belongs to a dense set; use adopt_parked instead"
                ))
            }
            None => Err(format!(
                "CachePool::adopt_parked_paged: unknown handle {handle}"
            )),
        }
    }

    /// Trim a detached paged set to `target_tokens` before adoption (#225).
    ///
    /// A partial prefix match (an APC block-clamped lookup, or a request that
    /// diverges inside the stored entry) covers only `target_tokens` of the
    /// set. The caller floors that value to the pool block size, so no
    /// partially filled tail block survives the trim: the adopting sequence's
    /// suffix re-prefill then starts on a fresh block and never mutates a
    /// shared tail (no copy-on-write needed at adopt time).
    ///
    /// Every dropped tail block releases BOTH references this set holds (the
    /// block-table allocation and the detach-time pin), mirroring
    /// [`CachePool::release_detached_paged`]. Release failures are reported
    /// after the trim finishes so the set's bookkeeping never goes
    /// inconsistent halfway. The dense placeholder handles' offsets and the
    /// set's bookkeeping lengths are clamped so the subsequent
    /// [`CachePool::adopt_paged`] restores RoPE offsets at the trimmed length.
    ///
    /// Turbo4 layouts are rejected: their per-page sidecars are keyed by the
    /// dropped blocks and no current caller trims them. Sliding-window states
    /// (`logical_start > 0`) are rejected for the same reason `write_prefill`
    /// assumes absolute indexing from zero.
    pub fn trim_detached_paged_to(
        &mut self,
        set: &mut DetachedPagedCacheSet,
        target_tokens: usize,
    ) -> Result<(), String> {
        if set.backend != SequenceStateBackend::PagedKvCache {
            return Err(format!(
                "CachePool::trim_detached_paged_to: expected paged backend, got {:?}",
                set.backend
            ));
        }
        if set.paged_layout.is_turbo_mode() {
            return Err(
                "CachePool::trim_detached_paged_to: Turbo4 sidecar trim is not supported".into(),
            );
        }
        let block_size = set.paged_layout.block_size;
        if block_size == 0 || !target_tokens.is_multiple_of(block_size) {
            return Err(format!(
                "CachePool::trim_detached_paged_to: target {target_tokens} is not a multiple of block size {block_size}"
            ));
        }
        let seq_len = set.seq_len();
        if target_tokens > seq_len {
            return Err(format!(
                "CachePool::trim_detached_paged_to: target {target_tokens} exceeds set length {seq_len}"
            ));
        }
        if set
            .paged_state
            .layers
            .iter()
            .any(|layer| layer.logical_start != 0)
        {
            return Err(
                "CachePool::trim_detached_paged_to: sliding-window state (logical_start > 0) is not supported"
                    .into(),
            );
        }
        if target_tokens == seq_len {
            return Ok(());
        }
        if set.retained_blocks.is_none() {
            return Err(
                "CachePool::trim_detached_paged_to: set no longer owns its block pins".into(),
            );
        }
        let pool = self
            .paged_pool
            .as_ref()
            .ok_or("CachePool::trim_detached_paged_to: no active paged pool")?
            .clone();

        let keep_blocks = target_tokens / block_size;
        let mut dropped: HashSet<PagedBlockId> = HashSet::new();
        let mut first_err: Option<String> = None;
        {
            let mut pool = pool.borrow_mut();
            for layer in set.paged_state.layers.iter_mut() {
                while layer.block_ids.len() > keep_blocks {
                    let block_id = layer
                        .block_ids
                        .pop()
                        .expect("len > keep_blocks implies non-empty");
                    dropped.insert(block_id);
                    // Allocation reference carried by the block table, then the
                    // detach pin from `retained_blocks`.
                    for _ in 0..2 {
                        if let Err(err) = pool.release_block(block_id) {
                            first_err.get_or_insert(format!(
                                "failed to release dropped block {block_id}: {err}"
                            ));
                        }
                    }
                }
                layer.len = target_tokens;
            }
        }
        if let Some(retained) = set.retained_blocks.as_mut() {
            retained.retain(|id| !dropped.contains(id));
        }
        for handle in set.caches.iter_mut() {
            // The adopted prefix is exactly `target_tokens` long; a freshly
            // detached pool-backed handle always satisfies
            // `offset == seq_len > target_tokens` here, so this is a clamp in
            // practice and an explicit statement of intent either way.
            debug_assert!(handle.offset >= target_tokens as i32);
            handle.offset = target_tokens as i32;
        }
        set.current_offset = set.current_offset.min(target_tokens as i32);
        set.prompt_len = set.prompt_len.min(target_tokens);
        match first_err {
            Some(err) => Err(format!("CachePool::trim_detached_paged_to: {err}")),
            None => Ok(()),
        }
    }

    /// Release every refcount held by a detached paged set. Call this when the
    /// set is no longer needed and adopt is not going to run.
    ///
    /// A detached set carries **two** pool references per block: the detach
    /// pin in `retained_blocks`, and the origin sequence's original allocation,
    /// which [`detach_paged`](CachePool::detach_paged) left in place when it
    /// removed the sequence from `active` (the block table moved into the set
    /// without a `release_sequence`). The adopt path releases the pin and lets
    /// the new sequence inherit the allocation; this discard path has no
    /// inheritor, so it must release BOTH — otherwise every block leaks at
    /// refcount 1 and never returns to the pool's free list. `retained_blocks`
    /// lists each block once (built 1:1 from the block table), and a set that
    /// is still parked in the store is the sole owner of its blocks (sharing
    /// only happens through `adopt_paged`, which consumes the set), so
    /// releasing the pin once and the allocation once drives an unshared block
    /// to refcount 0.
    ///
    /// Safe to call on an already-released set (which carries no retained
    /// blocks) — the `retained_blocks.take()` guard makes the whole body a
    /// no-op so neither reference is released twice.
    pub fn release_detached_paged(&mut self, mut detached: DetachedPagedCacheSet) {
        if let Some(blocks) = detached.retained_blocks.take() {
            if let Some(pool) = self.paged_pool.as_ref() {
                let mut pool = pool.borrow_mut();
                // Release the detach pins.
                for block_id in &blocks {
                    if let Err(err) = pool.release_block(*block_id) {
                        eprintln!(
                            "[mlxcel::cache::paged_detach] CachePool::release_detached_paged: failed to release pin for block {block_id}: {err}"
                        );
                    }
                }
                // Release the origin allocation the block table still carries.
                for layer in &detached.paged_state.layers {
                    for &block_id in &layer.block_ids {
                        if let Err(err) = pool.release_block(block_id) {
                            eprintln!(
                                "[mlxcel::cache::paged_detach] CachePool::release_detached_paged: failed to release allocation for block {block_id}: {err}"
                            );
                        }
                    }
                }
            }
        }
        // Dropping `detached` here runs the normal `Drop`, which at this
        // point sees an empty `retained_blocks` and stays silent.
        drop(detached);
    }

    /// Peek at a parked set as a paged variant. Returns `None` if the handle
    /// is unknown or points to a dense set.
    pub fn peek_parked_paged(&self, handle: DetachedHandle) -> Option<&DetachedPagedCacheSet> {
        match self.detached.get(&handle) {
            Some(ParkedCache::Paged(set)) => Some(set),
            _ => None,
        }
    }

    /// Remove a parked set as a paged variant. If the handle points to a
    /// dense set it is left in place and `None` is returned — call
    /// [`CachePool::take_parked`] to drain dense sets.
    pub fn take_parked_paged(&mut self, handle: DetachedHandle) -> Option<DetachedPagedCacheSet> {
        match self.detached.remove(&handle) {
            Some(ParkedCache::Paged(set)) => Some(set),
            Some(ParkedCache::Dense(dense)) => {
                self.detached.insert(handle, ParkedCache::Dense(dense));
                None
            }
            None => None,
        }
    }

    /// Shared helper used by paged adopt to lazily stand up the block pool
    /// when the first paged sequence arrives via the adopt path rather than
    /// a fresh allocate.
    pub(super) fn ensure_paged_pool_from_layout(
        &mut self,
        layout: &PagedKvLayout,
    ) -> Result<(), String> {
        if let Some(pool) = self.paged_pool.as_ref() {
            if pool.borrow().layout() != layout {
                return Err(
                    "CachePool::adopt_paged: paged layout mismatch for active paged backend"
                        .to_string(),
                );
            }
            return Ok(());
        }
        self.paged_pool = Some(Rc::new(RefCell::new(PagedBlockPool::new(layout.clone()))));
        Ok(())
    }
}

// Tests live in the companion `paged_detach_tests.rs` so this file stays
// focused on implementation (see `docs/code-guidelines.md`).
#[cfg(test)]
#[path = "paged_detach_tests.rs"]
mod tests;
