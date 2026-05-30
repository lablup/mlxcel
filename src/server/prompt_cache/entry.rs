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

use mlxcel_core::cache::DetachedCacheSet;

use super::block_hash::ApcBlockHash;

/// Send/Sync holder for a [`DetachedCacheSet`].
///
/// The upstream type wraps `cxx::UniquePtr<MlxArray>` which is neither
/// `Send` nor `Sync` because the underlying FFI buffer is an opaque C++
/// pointer. The detach/adopt API explicitly guarantees that:
///
/// * Once `CachePool::detach` returns, the originating sequence no longer
///   aliases the buffers (the pointers are moved, not cloned).
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
/// "Aliasing with `MLXCEL_ENABLE_DIRECT_PREFILL_CACHE_STORE`" sections).
pub(crate) struct DetachedHolder {
    inner: Option<DetachedCacheSet>,
}

impl DetachedHolder {
    pub(crate) fn new(set: DetachedCacheSet) -> Self {
        Self { inner: Some(set) }
    }

    pub(crate) fn is_some(&self) -> bool {
        self.inner.is_some()
    }

    #[allow(dead_code)]
    pub(crate) fn take(&mut self) -> Option<DetachedCacheSet> {
        self.inner.take()
    }
}

// SAFETY: See the type-level doc-comment above. The detach/adopt API
// moves buffer ownership rather than aliasing it, the holder sits behind a
// Mutex inside `CacheEntry`, and each buffer is only touched by one thread
// at a time (the worker thread at adopt time; no parallel MLX op is ever
// issued against a parked tensor). This is the same discipline that
// upstream `CachePool` uses for its own `DetachedMap`, which similarly
// only guarantees "single-threaded access" at any given time.
unsafe impl Send for DetachedHolder {}
unsafe impl Sync for DetachedHolder {}

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
    pub(crate) detached: Mutex<DetachedHolder>,
    pub last_used: Mutex<Instant>,
    pub size_bytes: usize,
    pub apc_block_hashes: Option<Vec<ApcBlockHash>>,
}

impl CacheEntry {
    /// Build a new entry from a token prefix and its detached cache set.
    ///
    /// `size_bytes` is captured from [`DetachedCacheSet::nbytes`] so the
    /// store's byte-budget accounting is consistent even if the underlying
    /// tensors are later adopted out of the entry.
    ///
    /// The APC block-hash chain is left unset; callers that have APC
    /// enabled should attach it via [`CacheEntry::with_apc_block_hashes`]
    /// before inserting into the store.
    pub fn new(tokens: Vec<i32>, detached: DetachedCacheSet) -> Self {
        let size_bytes = detached.nbytes();
        Self {
            tokens,
            detached: Mutex::new(DetachedHolder::new(detached)),
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
    pub fn take_detached(&self) -> Option<DetachedCacheSet> {
        match self.detached.lock() {
            Ok(mut g) => g.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        }
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
            detached: Mutex::new(DetachedHolder::new(detached)),
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
