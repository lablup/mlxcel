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

//! A single prompt-prefix cache entry.
//!
//! An entry pairs a previously observed token-prefix with the detached
//! KV-cache set produced by [`mlxcel_core::cache::CachePool::detach`].
//! The store holds each entry behind an [`Arc`] so lookups can hand out
//! clones without bumping the cache lock.

use std::sync::Mutex;
use std::time::Instant;

use mlxcel_core::cache::{DetachedCacheSet, DetachedPagedCacheSet};

use super::block_hash::ApcBlockHash;

/// A detached KV-cache set, one variant per decode-storage backend.
///
/// * [`DetachedKvSet::Dense`] — the dense per-layer `KVCache` buffers
///   produced by [`mlxcel_core::cache::CachePool::detach`].
/// * [`DetachedKvSet::Paged`] — the paged block-table snapshot produced by
///   [`mlxcel_core::cache::CachePool::detach_paged`]. For a pool-backed
///   paged sequence the per-layer dense handles are EMPTY by design (the
///   K/V lives in the shared `PagedBlockPool`); the set carries the block
///   table plus a refcount pin on every physical block so a later
///   `adopt_paged` can share the prefix without re-prefilling it.
///
/// This is the single payload the cross-request prompt-prefix store parks
/// between a donate-back and a later adopt, so the radix store is unified
/// across both backends (#121 sub-step b).
//
// The two variants differ substantially in size (the paged variant carries
// several `HashMap`s of per-page sidecar tensors); boxing the larger arm
// would add an allocation on the hot dense path, so we accept the size gap.
#[allow(clippy::large_enum_variant)]
pub enum DetachedKvSet {
    /// Dense per-layer KV buffers (the [`SequenceStateBackend::DenseKvCache`]
    /// backend).
    ///
    /// [`SequenceStateBackend::DenseKvCache`]: mlxcel_core::cache::SequenceStateBackend::DenseKvCache
    Dense(DetachedCacheSet),
    /// Paged block-table snapshot (the [`SequenceStateBackend::PagedKvCache`]
    /// backend).
    ///
    /// [`SequenceStateBackend::PagedKvCache`]: mlxcel_core::cache::SequenceStateBackend::PagedKvCache
    Paged(DetachedPagedCacheSet),
}

impl DetachedKvSet {
    /// Total byte footprint of the detached set, dispatched per backend.
    ///
    /// Feeds [`CacheEntry::size_bytes`] so the store's byte-budget eviction
    /// accounting is consistent across dense and paged entries.
    pub fn nbytes(&self) -> usize {
        match self {
            DetachedKvSet::Dense(d) => d.nbytes(),
            DetachedKvSet::Paged(p) => p.nbytes(),
        }
    }

    /// Whether the set carries no reusable KV state.
    ///
    /// * Dense: the per-layer caches are absent or all empty (e.g. stored
    ///   against a sequence aborted before any prefill completed).
    /// * Paged: the per-layer dense handles are EMPTY by design, so we gate
    ///   on the paged block table instead — a set is empty when it exposes no
    ///   visible tokens or pins no physical blocks.
    pub fn is_empty(&self) -> bool {
        match self {
            DetachedKvSet::Dense(d) => d.caches.is_empty() || d.caches.iter().all(|c| c.is_empty()),
            DetachedKvSet::Paged(p) => p.seq_len() == 0 || p.retained_block_count() == 0,
        }
    }
}

/// Send/Sync holder for a [`DetachedKvSet`].
///
/// Both variants wrap `cxx::UniquePtr<MlxArray>` (directly for dense, or in
/// the paged variant's per-page sidecar maps) which is neither `Send` nor
/// `Sync` because the underlying FFI buffer is an opaque C++ pointer. The
/// detach/adopt API explicitly guarantees that:
///
/// * Once `CachePool::detach` / `detach_paged` returns, the originating
///   sequence no longer aliases the buffers (the pointers are moved, and the
///   paged block pins are refcounted, not cloned).
/// * Each buffer is functional: any MLX operation produces a fresh array,
///   so even if adopt runs on a different OS thread than detach, there is
///   no concurrent read/write on the same pointer.
///
/// We therefore assert `Send + Sync` on a newtype wrapper rather than the
/// upstream type. Access to the inner set is additionally serialised
/// through a [`Mutex`] inside [`CacheEntry`], which means at any moment at
/// most one thread is reading or moving the tensors. This combination is
/// sufficient to satisfy the soundness obligations documented in
/// `src/lib/mlxcel-core/src/cache/detach.rs` (the "INT8 preservation" and
/// "Aliasing with `MLXCEL_ENABLE_DIRECT_PREFILL_CACHE_STORE`" sections) and
/// the analogous discipline in `cache/paged_detach.rs`.
pub(crate) struct DetachedKvSetHolder {
    inner: Option<DetachedKvSet>,
}

impl DetachedKvSetHolder {
    pub(crate) fn new(set: DetachedKvSet) -> Self {
        Self { inner: Some(set) }
    }

    pub(crate) fn is_some(&self) -> bool {
        self.inner.is_some()
    }

    pub(crate) fn take(&mut self) -> Option<DetachedKvSet> {
        self.inner.take()
    }

    pub(crate) fn peek(&self) -> Option<&DetachedKvSet> {
        self.inner.as_ref()
    }
}

// SAFETY: See the type-level doc-comment above. The detach/adopt API
// moves buffer ownership (and refcounts paged block pins) rather than
// aliasing it, the holder sits behind a Mutex inside `CacheEntry`, and each
// buffer is only touched by one thread at a time (the worker thread at adopt
// time; no parallel MLX op is ever issued against a parked tensor). This is
// the same discipline that upstream `CachePool` uses for its own
// `DetachedMap`, which similarly only guarantees "single-threaded access" at
// any given time.
unsafe impl Send for DetachedKvSetHolder {}
unsafe impl Sync for DetachedKvSetHolder {}

/// A prompt-prefix cache entry.
///
/// * `tokens` — the canonical token-id prefix this entry represents.
/// * `detached` — the detached KV-cache set corresponding to those tokens.
///   Stored inside a [`Mutex`] so adopt paths can take ownership from a
///   shared entry without making the whole entry mutable. Once consumed,
///   the slot is left `None` to avoid double-adopt.
/// * `last_used` — monotonic timestamp of the last successful lookup (used
///   by the LRU / TTL policy).
/// * `size_bytes` — cached byte footprint at insert time. We snapshot this
///   once instead of recomputing on every eviction pass.
/// * `apc_block_hashes` — Automatic Prefix Caching (APC)
///   block-granularity hash chain over `tokens`. Populated only when APC
///   is enabled at insert time; `None` otherwise. The chain is computed
///   from `tokens`, the configured block size, the configured hash algo,
///   and the request's `MultimodalDigest`. Used by lookup paths to
///   verify block-level prefix consistency in addition to the existing
///   trie/scan candidate selection.
pub struct CacheEntry {
    pub tokens: Vec<i32>,
    pub(crate) detached: Mutex<DetachedKvSetHolder>,
    pub last_used: Mutex<Instant>,
    pub size_bytes: usize,
    pub apc_block_hashes: Option<Vec<ApcBlockHash>>,
}

impl CacheEntry {
    /// Build a new entry from a token prefix and its detached cache set.
    ///
    /// `size_bytes` is captured from [`DetachedKvSet::nbytes`] so the
    /// store's byte-budget accounting is consistent (across both dense and
    /// paged backends) even if the underlying tensors are later adopted out
    /// of the entry.
    ///
    /// The APC block-hash chain is left unset; callers that have APC
    /// enabled should attach it via [`CacheEntry::with_apc_block_hashes`]
    /// before inserting into the store.
    pub fn new(tokens: Vec<i32>, detached: DetachedKvSet) -> Self {
        let size_bytes = detached.nbytes();
        Self {
            tokens,
            detached: Mutex::new(DetachedKvSetHolder::new(detached)),
            last_used: Mutex::new(Instant::now()),
            size_bytes,
            apc_block_hashes: None,
        }
    }

    /// Builder-style: attach an APC block-hash chain to this entry.
    ///
    /// Returns `self` so callers can chain `.with_apc_block_hashes(..)`
    /// after [`CacheEntry::new`]. The store calls this internally during
    /// `insert` when APC is enabled, so most callers do not need to invoke
    /// it directly.
    #[must_use]
    pub fn with_apc_block_hashes(mut self, hashes: Vec<ApcBlockHash>) -> Self {
        self.apc_block_hashes = Some(hashes);
        self
    }

    /// Read the APC block-hash chain attached to this entry, if any.
    ///
    /// Returns `Some(&[ApcBlockHash])` when APC was enabled at insert time
    /// and the entry was constructed with a populated chain; `None`
    /// otherwise. Callers that depend on APC behaviour should treat
    /// `None` as "APC was not active for this entry" and fall back to
    /// the existing whole-prefix matcher.
    pub fn apc_block_hashes(&self) -> Option<&[ApcBlockHash]> {
        self.apc_block_hashes.as_deref()
    }

    /// Take ownership of the detached cache set for adopt. Returns `None`
    /// if already consumed.
    ///
    /// Called by the scheduler / model worker at adopt time. Safe to call
    /// from any thread because the cache set is serialised through this
    /// entry's mutex; the actual adopt path must still run on the worker
    /// thread (or wherever the model's `CachePool` lives).
    pub fn take_detached(&self) -> Option<DetachedKvSet> {
        match self.detached.lock() {
            Ok(mut g) => g.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        }
    }

    /// Run `f` against the parked set WITHOUT consuming it (#227).
    ///
    /// Returns `None` when the set was already taken (a drained shell).
    /// Used by the non-consuming clone-and-pin adoption path: the closure
    /// clones pinned block references out of a paged set while the entry
    /// stays live in the store for concurrent siblings and deeper future
    /// matches. The entry lock is held for the duration of `f`.
    pub fn with_detached<R>(&self, f: impl FnOnce(&DetachedKvSet) -> R) -> Option<R> {
        let guard = match self.detached.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.peek().map(f)
    }

    /// Update the `last_used` timestamp. Called on every successful lookup
    /// to keep the LRU ordering fresh. Poisoned lock paths fall back to
    /// overwriting the inner value so accounting keeps moving.
    pub fn touch(&self) {
        let mut guard = match self.last_used.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = Instant::now();
    }

    /// Read the current `last_used` timestamp.
    pub fn last_used(&self) -> Instant {
        match self.last_used.lock() {
            Ok(g) => *g,
            Err(poisoned) => *poisoned.into_inner(),
        }
    }

    /// Number of tokens this entry's prefix contains.
    pub fn token_len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the underlying detached cache is still available to adopt.
    pub fn has_detached(&self) -> bool {
        match self.detached.lock() {
            Ok(g) => g.is_some(),
            Err(poisoned) => poisoned.into_inner().is_some(),
        }
    }
}

#[cfg(test)]
impl CacheEntry {
    /// Test-only constructor that fabricates a zero-tensor detached set and
    /// overrides the reported `size_bytes`. This lets unit tests exercise
    /// budget / LRU paths without allocating real MLX buffers.
    ///
    /// The APC block-hash chain is left unset. Tests that want to populate
    /// it explicitly can chain [`CacheEntry::with_apc_block_hashes`] on
    /// the returned value, but the store path itself attaches the chain
    /// during `insert` so most tests do not need to.
    pub(crate) fn new_for_test(tokens: Vec<i32>, size_bytes: usize) -> Self {
        use mlxcel_core::cache::SequenceId;
        let now = Instant::now();
        let detached = DetachedCacheSet {
            caches: Vec::new(),
            backend: mlxcel_core::cache::SequenceStateBackend::DenseKvCache,
            prompt_len: 0,
            current_offset: 0,
            created_at: now,
            detached_at: now,
            origin_seq_id: SequenceId::from_raw(0),
        };
        Self {
            tokens,
            detached: Mutex::new(DetachedKvSetHolder::new(DetachedKvSet::Dense(detached))),
            last_used: Mutex::new(Instant::now()),
            size_bytes,
            apc_block_hashes: None,
        }
    }
}

impl std::fmt::Debug for CacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheEntry")
            .field("tokens_len", &self.tokens.len())
            .field("size_bytes", &self.size_bytes)
            .field("has_detached", &self.has_detached())
            .field(
                "apc_blocks",
                &self.apc_block_hashes.as_ref().map(|h| h.len()),
            )
            .finish()
    }
}
