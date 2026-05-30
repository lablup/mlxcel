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

//! Shared LRU store for detached KV caches.
//!
//! This module provides [`PromptCacheStore`], the cross-request
//! prompt-prefix cache. The store is thread-safe via a single
//! `Arc<RwLock<Inner>>`: concurrent lookups take a read lock and match
//! prefixes, while inserts/evictions take an exclusive write lock.
//!
//! The two-tier longest-prefix matcher lives in
//! [`super::lookup`]; this module wires the matcher into the store's
//! locking + metrics discipline. See that module and
//! [`super::trie`] for the lookup algorithm and data structure choice.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::apc_lookup::{ApcStoreStats, apc_consistent_prefix_len};
use super::block_hash::{ApcBlockHash, BlockHashChain};
use super::entry::CacheEntry;
use super::key::{PromptCacheKey, PromptCacheKeyDigest};
use super::metrics::{NoopPromptCacheMetrics, PromptCacheMetrics};
use super::policy::{PromptCacheConfig, PromptCacheStats};
use super::trie::RadixTrie;
pub(super) use super::types::SessionlessBucketKey;
use super::types::{BucketKey, InsertError};

/// Internal entry bookkeeping.
pub(super) struct EntrySlot {
    pub(super) entry: Arc<CacheEntry>,
    /// Bucket identity for prefix-matching fallback paths. The digest is
    /// recoverable via the HashMap key, so we only keep the bucket key here.
    pub(super) bucket: BucketKey,
    /// Session-agnostic bucket, used to locate the radix trie on evict /
    /// replace paths without re-deriving from strings.
    sessionless: SessionlessBucketKey,
}

struct Inner {
    config: PromptCacheConfig,
    // Primary map: digest -> entry.
    entries: HashMap<PromptCacheKeyDigest, EntrySlot>,
    // Per-(model, lora, template) radix trie. Each trie stores digests
    // indexed by their stored-entry token prefix; lookups walk the trie
    // to find the longest token-prefix match in `O(L)` where `L` is the
    // matched depth. Cross-session reuse is handled at candidate-scoring
    // time inside `lookup_longest_prefix`.
    tries: HashMap<SessionlessBucketKey, RadixTrie>,
    total_bytes: usize,
    inserts: u64,
    rejections_oversized: u64,
    lookups: u64,
    hits: u64,
    evictions_lru: u64,
    evictions_ttl: u64,
}

impl Inner {
    fn new(config: PromptCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            tries: HashMap::new(),
            total_bytes: 0,
            inserts: 0,
            rejections_oversized: 0,
            lookups: 0,
            hits: 0,
            evictions_lru: 0,
            evictions_ttl: 0,
        }
    }

    fn stats(&self) -> PromptCacheStats {
        PromptCacheStats {
            entries: self.entries.len(),
            bytes: self.total_bytes,
            inserts: self.inserts,
            rejections_oversized: self.rejections_oversized,
            lookups: self.lookups,
            hits: self.hits,
            evictions_lru: self.evictions_lru,
            evictions_ttl: self.evictions_ttl,
        }
    }

    fn remove_entry(
        &mut self,
        digest: &PromptCacheKeyDigest,
    ) -> Option<(BucketKey, Arc<CacheEntry>)> {
        let slot = self.entries.remove(digest)?;
        self.total_bytes = self.total_bytes.saturating_sub(slot.entry.size_bytes);

        let trie_empty = if let Some(trie) = self.tries.get_mut(&slot.sessionless) {
            trie.remove(&slot.entry.tokens, *digest);
            trie.len() == 0
        } else {
            false
        };
        if trie_empty {
            self.tries.remove(&slot.sessionless);
        }
        Some((slot.bucket, slot.entry))
    }

    /// Sweep every entry that has been idle for longer than `config.ttl`.
    /// Returns `(bytes_freed, evicted_count)`.
    fn sweep_ttl(&mut self, now: Instant) -> (usize, usize) {
        if self.config.ttl.is_zero() || self.entries.is_empty() {
            return (0, 0);
        }
        let ttl = self.config.ttl;
        let stale: Vec<PromptCacheKeyDigest> = self
            .entries
            .iter()
            .filter(|(_, slot)| now.duration_since(slot.entry.last_used()) >= ttl)
            .map(|(d, _)| *d)
            .collect();
        let mut bytes = 0;
        for digest in &stale {
            if let Some((_, entry)) = self.remove_entry(digest) {
                bytes += entry.size_bytes;
            }
        }
        let count = stale.len();
        self.evictions_ttl = self.evictions_ttl.saturating_add(count as u64);
        (bytes, count)
    }

    /// Remove entries whose detached cache was already consumed by an adopt
    /// path. The store keeps lookup results as `Arc<CacheEntry>`, so the
    /// scheduler drains the detached payload after the store lock is released.
    /// Sweeping these drained shells on the next store touch keeps byte-budget
    /// accounting and trie candidates aligned with reusable cache state.
    fn sweep_drained(&mut self) -> (usize, usize) {
        if self.entries.is_empty() {
            return (0, 0);
        }
        let drained: Vec<PromptCacheKeyDigest> = self
            .entries
            .iter()
            .filter(|(_, slot)| !slot.entry.has_detached())
            .map(|(digest, _)| *digest)
            .collect();
        let mut bytes = 0;
        for digest in &drained {
            if let Some((_, entry)) = self.remove_entry(digest) {
                bytes += entry.size_bytes;
            }
        }
        (bytes, drained.len())
    }

    /// Evict the single oldest entry. Returns the number of bytes freed, or
    /// `0` if the store is empty.
    fn evict_oldest(&mut self) -> usize {
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, slot)| slot.entry.last_used())
            .map(|(d, _)| *d);
        match oldest {
            Some(digest) => {
                let freed = self
                    .remove_entry(&digest)
                    .map(|(_, e)| e.size_bytes)
                    .unwrap_or(0);
                if freed > 0 {
                    self.evictions_lru = self.evictions_lru.saturating_add(1);
                }
                freed
            }
            None => 0,
        }
    }

    /// Enforce both caps: max_entries, then capacity_bytes. Returns the
    /// number of bytes freed.
    fn enforce_caps(&mut self, metrics: &dyn PromptCacheMetrics) -> usize {
        let mut freed = 0;
        while self.entries.len() > self.config.max_entries {
            let n = self.evict_oldest();
            if n == 0 {
                break;
            }
            metrics.record_evict_lru(n);
            freed += n;
        }
        while self.total_bytes > self.config.capacity_bytes {
            let n = self.evict_oldest();
            if n == 0 {
                break;
            }
            metrics.record_evict_lru(n);
            freed += n;
        }
        freed
    }
}

/// Shared LRU store for detached KV caches.
///
/// Construct once via [`PromptCacheStore::new`] / [`PromptCacheStore::with_config`]
/// and share via `Arc<PromptCacheStore>`. All methods take `&self`; internal
/// mutation goes through an `RwLock`.
pub struct PromptCacheStore {
    inner: RwLock<Inner>,
    metrics: Arc<dyn PromptCacheMetrics>,
}

impl PromptCacheStore {
    /// Build a store with the default configuration.
    pub fn new() -> Self {
        Self::with_config(PromptCacheConfig::default())
    }

    /// Build a store with a caller-supplied configuration.
    pub fn with_config(config: PromptCacheConfig) -> Self {
        Self {
            inner: RwLock::new(Inner::new(config)),
            metrics: Arc::new(NoopPromptCacheMetrics),
        }
    }

    /// Build a store with a caller-supplied configuration and metrics
    /// implementor. uses this entry point to hand in the
    /// Prometheus / `BatchMetrics` bridge.
    pub fn with_metrics(config: PromptCacheConfig, metrics: Arc<dyn PromptCacheMetrics>) -> Self {
        Self {
            inner: RwLock::new(Inner::new(config)),
            metrics,
        }
    }

    /// Convenience: wrap in an `Arc` for sharing across threads and
    /// subsystems.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Whether the store will accept inserts. When `false`, `insert` short
    /// circuits to [`InsertError::Disabled`] and lookups immediately return
    /// `None`.
    pub fn is_enabled(&self) -> bool {
        self.inner
            .read()
            .map(|g| g.config.is_enabled())
            .unwrap_or(false)
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.inner.read().map(|g| g.entries.len()).unwrap_or(0)
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current cumulative byte footprint of all entries.
    pub fn bytes(&self) -> usize {
        self.inner.read().map(|g| g.total_bytes).unwrap_or(0)
    }

    /// Capacity in bytes as configured. Does not change at runtime.
    pub fn capacity_bytes(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.config.capacity_bytes)
            .unwrap_or(0)
    }

    /// Maximum entry count as configured.
    pub fn max_entries(&self) -> usize {
        self.inner.read().map(|g| g.config.max_entries).unwrap_or(0)
    }

    /// Snapshot of the store's internal counters. Safe to call concurrently
    /// with inserts/lookups; values are captured under the read lock.
    pub fn stats(&self) -> PromptCacheStats {
        self.inner.read().map(|g| g.stats()).unwrap_or_default()
    }

    /// Snapshot of APC-specific aggregate statistics across all live
    /// entries.
    ///
    /// Walks every entry under the read lock and reports:
    ///
    /// - `total_blocks_stored` — sum of per-entry APC chain lengths.
    /// - `unique_block_hashes` — number of distinct block hashes across
    ///   all entries (deduplication potential metric).
    /// - `apc_active_entries` — number of entries that actually carry an
    ///   APC chain. When APC is disabled at the store level, this is
    ///   always `0` because `insert` does not populate the chain.
    ///
    /// The walk is `O(N * B)` in entry count `N` and average chain length
    /// `B`, so the call is meant for periodic monitoring rather than the
    /// per-request hot path.
    pub fn apc_stats(&self) -> ApcStoreStats {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(_) => return ApcStoreStats::default(),
        };
        if !guard.config.apc_enabled() {
            return ApcStoreStats::default();
        }
        let mut total_blocks_stored = 0usize;
        let mut apc_active_entries = 0usize;
        // Collecting into a HashSet preserves the dedup-potential metric:
        // identical block hashes from different entries collapse to a
        // single set member.
        let mut unique: std::collections::HashSet<ApcBlockHash> = std::collections::HashSet::new();
        for slot in guard.entries.values() {
            if let Some(hashes) = slot.entry.apc_block_hashes() {
                apc_active_entries += 1;
                total_blocks_stored += hashes.len();
                unique.extend(hashes.iter().copied());
            }
        }
        ApcStoreStats {
            total_blocks_stored,
            unique_block_hashes: unique.len(),
            apc_active_entries,
        }
    }

    /// Insert an entry. Evicts older entries as needed to satisfy the
    /// entry-count and byte-budget caps. Returns [`InsertError`] if the
    /// store is disabled or if the single entry is too large to ever fit.
    ///
    /// When Automatic Prefix Caching (APC) is enabled on this
    /// store, the entry's APC block-hash chain is computed during insert
    /// from the entry's tokens, the configured block size, the configured
    /// hash algo, and the request's `MultimodalDigest` (carried by `key`).
    /// The chain is attached to the entry before it lands in the store so
    /// subsequent lookups can verify block-level prefix consistency. When
    /// APC is disabled, no extra work is performed and the entry's
    /// `apc_block_hashes` field stays `None`.
    pub fn insert(&self, key: &PromptCacheKey<'_>, entry: CacheEntry) -> Result<(), InsertError> {
        let digest = key.digest();
        let entry_bytes = entry.size_bytes;
        let bucket = BucketKey::from_key(key);
        let sessionless = SessionlessBucketKey::from_key(key);

        let mut guard = self.inner.write().expect("prompt cache inner lock");

        if !guard.config.is_enabled() {
            return Err(InsertError::Disabled);
        }
        if key.effective_prefix_len() < guard.config.min_prefix_tokens {
            return Err(InsertError::PrefixTooShort {
                got: key.effective_prefix_len(),
                min_required: guard.config.min_prefix_tokens,
            });
        }
        if entry_bytes > guard.config.capacity_bytes {
            guard.rejections_oversized = guard.rejections_oversized.saturating_add(1);
            let metrics = Arc::clone(&self.metrics);
            drop(guard);
            metrics.record_reject_oversized(entry_bytes);
            return Err(InsertError::OversizedEntry {
                entry_bytes,
                capacity_bytes: self
                    .inner
                    .read()
                    .map(|g| g.config.capacity_bytes)
                    .unwrap_or(0),
            });
        }

        // Replace an existing entry under the same digest (idempotent insert
        // semantics for repeated prefill of the same prompt).
        if let Some((_, _)) = guard.remove_entry(&digest) {
            // Treat replacement as an LRU eviction for accounting purposes.
            guard.evictions_lru = guard.evictions_lru.saturating_add(1);
        }

        // APC integration: when APC is on, fold the block-hash
        // chain into the entry. The chain is computed against the entry's
        // own token prefix and the request's mm_digest (carried by `key`),
        // so two entries with identical tokens but different multimodal
        // payloads — should they ever land in the same bucket — diverge on
        // every block hash and lookup-time block verification will reject
        // the cross-payload candidate.
        let entry = if guard.config.apc_enabled() {
            let chain = BlockHashChain::compute(
                &entry.tokens,
                guard.config.apc.block_size,
                guard.config.apc.hash,
                key.mm_digest.as_bytes(),
            );
            entry.with_apc_block_hashes(chain.hashes)
        } else {
            entry
        };

        // Speculatively account for the new bytes, then evict as needed.
        guard.total_bytes = guard.total_bytes.saturating_add(entry_bytes);
        let tokens_for_trie = entry.tokens.clone();
        let slot = EntrySlot {
            entry: Arc::new(entry),
            bucket,
            sessionless: sessionless.clone(),
        };
        guard.entries.insert(digest, slot);
        guard
            .tries
            .entry(sessionless)
            .or_default()
            .insert(&tokens_for_trie, digest);
        guard.inserts = guard.inserts.saturating_add(1);

        let metrics = Arc::clone(&self.metrics);
        metrics.record_insert(entry_bytes);

        // Enforce caps. This may evict freshly inserted entries if they are
        // already beyond capacity, which is intentional — we never exceed
        // the configured budget.
        guard.enforce_caps(metrics.as_ref());
        Ok(())
    }

    /// Find the best cached entry whose stored token prefix forms the
    /// longest common prefix of `tokens` and is reusable under `key`.
    ///
    /// Search is two-tier:
    ///
    /// 1. **Exact-session tier.** Filter candidates whose `session_key`
    ///    matches `key.session_key`. If any clear the
    ///    [`PromptCacheConfig::min_prefix_tokens`] threshold, return the
    ///    longest match; ties resolved by most-recently-used.
    /// 2. **Cross-session tier.** Fall back to candidates with a different
    ///    `session_key` (or `None`), still under the same
    ///    `(model, lora, template)` bucket. Same threshold, MRU tie-break.
    ///
    /// The cross-session tier only wins if its best match is **strictly
    /// longer** than the exact-session tier's best match — otherwise the
    /// exact-session match is preferred, matching the tie-break rule
    /// "same `session_key` first".
    ///
    /// Underlying lookup uses the per-`(model, lora, template)` radix
    /// trie from [`super::trie::RadixTrie`]: `O(L)` in the matched depth.
    ///
    /// When Automatic Prefix Caching (APC) is enabled, the
    /// candidate selected by the trie / scan tier is additionally
    /// verified against the request's APC block-hash chain. The chain is
    /// computed from the request's tokens with the same block size, hash
    /// algo, and `extra_hash` (the request's `MultimodalDigest`) used at
    /// insert time. For each full block in the candidate's matched
    /// prefix, the candidate's stored block hash must equal the request's
    /// block hash; otherwise the matched length is truncated to the last
    /// consistent block boundary. If after truncation the matched length
    /// drops below `min_prefix_tokens`, the lookup is reported as a miss.
    /// This gives APC its core safety property: a candidate cannot be
    /// adopted unless every covered block hashes identically on both
    /// sides (text *and* multimodal content). When APC is disabled the
    /// fast path is unchanged and no block hashes are touched.
    pub fn lookup_longest_prefix(
        &self,
        key: &PromptCacheKey<'_>,
        tokens: &[i32],
    ) -> Option<(Arc<CacheEntry>, usize)> {
        // Fast path: TTL sweep under a read-then-upgrade pattern would need
        // the write lock anyway. Do the sweep under the write lock so we
        // never hand out expired entries.
        {
            let now = Instant::now();
            let mut guard = self.inner.write().expect("prompt cache inner lock");
            if !guard.config.is_enabled() {
                return None;
            }
            let _ = guard.sweep_drained();
            let (freed, count) = guard.sweep_ttl(now);
            if let Some(per_entry) = freed.checked_div(count) {
                let metrics = Arc::clone(&self.metrics);
                for _ in 0..count {
                    metrics.record_evict_ttl(per_entry);
                }
            }
        }

        let sessionless = SessionlessBucketKey::from_key(key);
        let best = {
            let guard = self.inner.read().expect("prompt cache inner lock");
            let min_len = guard.config.min_prefix_tokens;
            // when APC is on, the trie / scan tiers may surface
            // candidates whose stored prefix is **not** fully contained in
            // the request. The block-hash discriminator below clamps the
            // resulting `matched` value to the last block boundary where
            // the chains agree. When APC is off, retain the legacy
            // whole-prefix-contained check inside both tiers so the
            // earlier hot path is bit-exact.
            let apc_partial_allowed = guard.config.apc_enabled();
            let trie = match guard.tries.get(&sessionless) {
                Some(t) => t,
                None => {
                    drop(guard);
                    return self.finalize_miss();
                }
            };
            super::lookup::select_best(trie, key, tokens, min_len, apc_partial_allowed, |d| {
                guard.entries.get(d)
            })
            .or_else(|| {
                select_best_by_scan(
                    &guard,
                    &sessionless,
                    key,
                    tokens,
                    min_len,
                    apc_partial_allowed,
                )
            })
        };

        // Increment lookup counters under the write lock so statistics stay
        // accurate even under concurrent readers. The hot path is the miss
        // case, which only writes a single atomic.
        let (entry, matched_len) = {
            let mut guard = self.inner.write().expect("prompt cache inner lock");
            guard.lookups = guard.lookups.saturating_add(1);
            match best {
                Some(winner) => {
                    // Snapshot every value we need from the entry slot up
                    // front so we can drop the immutable borrow of
                    // `guard.entries` before mutating `guard.hits`. The APC
                    // block-hash clone only happens when APC is actually
                    // enabled, keeping the disabled path allocation-free.
                    let apc_on = guard.config.apc_enabled();
                    let block_size = guard.config.apc.block_size;
                    let hash_algo = guard.config.apc.hash;
                    let min_prefix = guard.config.min_prefix_tokens;
                    let (entry_arc, entry_apc_hashes) = match guard.entries.get(&winner.digest) {
                        Some(s) => {
                            let hashes = if apc_on {
                                s.entry.apc_block_hashes().map(|h| h.to_vec())
                            } else {
                                None
                            };
                            (Arc::clone(&s.entry), hashes)
                        }
                        None => {
                            drop(guard);
                            let metrics = Arc::clone(&self.metrics);
                            metrics.record_lookup(false, 0);
                            return None;
                        }
                    };

                    // APC block-hash verification. When APC is on AND the
                    // candidate has a stored chain, recompute the request's
                    // chain on-demand and clamp `matched_len` to the last
                    // block boundary where both chains agree. This is the
                    // load-bearing safety property of APC: identical token
                    // prefixes with different multimodal payloads diverge
                    // on every block, so even if a candidate slipped
                    // through bucket isolation it cannot be adopted across
                    // payloads.
                    // When `apc_on` is true but `entry_apc_hashes` is `None`
                    // the entry was written before APC was enabled on this
                    // store (e.g. the store was reconfigured at runtime, or
                    // the entry was inserted by a code path that predates
                    // APC). This is not an invariant violation — it is a
                    // normal "old-format entry in a new-APC store" case.
                    // Falling through to `winner.matched` is safe: we cannot
                    // perform block-hash verification without a stored chain,
                    // so we treat the entry as if APC were off and accept the
                    // trie-reported match depth at face value. The entry will
                    // be replaced by an APC-aware version after the next
                    // insert on this key.
                    let apc_matched = if apc_on && let Some(hashes) = entry_apc_hashes.as_deref() {
                        apc_consistent_prefix_len(
                            tokens,
                            hashes,
                            block_size,
                            hash_algo,
                            key.mm_digest.as_bytes(),
                            winner.matched,
                        )
                    } else {
                        winner.matched
                    };

                    if apc_matched < min_prefix {
                        // The block-hash check truncated past the minimum
                        // useful prefix. Treat as a miss so the caller does
                        // not adopt a partial cache for fewer tokens than
                        // the configured threshold.
                        drop(guard);
                        let metrics = Arc::clone(&self.metrics);
                        metrics.record_lookup(false, 0);
                        return None;
                    }

                    guard.hits = guard.hits.saturating_add(1);
                    entry_arc.touch();
                    (Some(entry_arc), apc_matched)
                }
                None => (None, 0),
            }
        };

        let metrics = Arc::clone(&self.metrics);
        match &entry {
            Some(_) => metrics.record_lookup(true, matched_len),
            None => metrics.record_lookup(false, 0),
        }
        entry.map(|e| (e, matched_len))
    }

    /// Account a lookup miss and return `None`. Factored out so the
    /// two-tier fast-path `return` sites don't duplicate the metric /
    /// counter bookkeeping.
    fn finalize_miss(&self) -> Option<(Arc<CacheEntry>, usize)> {
        {
            let mut guard = self.inner.write().expect("prompt cache inner lock");
            guard.lookups = guard.lookups.saturating_add(1);
        }
        let metrics = Arc::clone(&self.metrics);
        metrics.record_lookup(false, 0);
        None
    }

    /// Force a sweep. Returns the total bytes freed.
    pub fn evict_if_needed(&self) -> usize {
        let mut guard = self.inner.write().expect("prompt cache inner lock");
        if !guard.config.is_enabled() {
            return 0;
        }
        let (drained_freed, _) = guard.sweep_drained();
        let now = Instant::now();
        let (ttl_freed, ttl_count) = guard.sweep_ttl(now);
        if let Some(per_entry) = ttl_freed.checked_div(ttl_count) {
            let metrics = Arc::clone(&self.metrics);
            for _ in 0..ttl_count {
                metrics.record_evict_ttl(per_entry);
            }
        }
        let metrics = Arc::clone(&self.metrics);
        let cap_freed = guard.enforce_caps(metrics.as_ref());
        drained_freed + ttl_freed + cap_freed
    }

    /// Drop every entry. Primarily for tests and shutdown paths.
    pub fn clear(&self) {
        let mut guard = self.inner.write().expect("prompt cache inner lock");
        guard.entries.clear();
        guard.tries.clear();
        guard.total_bytes = 0;
    }
}

fn better_candidate(a: &super::lookup::BestCandidate, b: &super::lookup::BestCandidate) -> bool {
    if a.matched != b.matched {
        return a.matched > b.matched;
    }
    a.last_used > b.last_used
}

/// Scan-tier fallback for `lookup_longest_prefix`. Called when the trie
/// lookup yields no candidate for the given sessionless bucket key.
///
/// **Per-entry cost note (APC on):** when `apc_partial_allowed` is `true`
/// each entry in the bucket is scored by `common_prefix_len`, an O(min(a,b))
/// comparison. For stores with many entries per bucket and long prompts this
/// is linear in both dimensions. The primary trie path (O(prompt_len)) should
/// handle the common case; this scan path is the cold fallback for entries
/// that share a bucket key but were not indexed under a matching trie prefix.
fn select_best_by_scan(
    guard: &Inner,
    sessionless: &SessionlessBucketKey,
    key: &PromptCacheKey<'_>,
    tokens: &[i32],
    min_len: usize,
    apc_partial_allowed: bool,
) -> Option<super::lookup::BestCandidate> {
    let caller_session = key.session_key;
    let mut best_same_session: Option<super::lookup::BestCandidate> = None;
    let mut best_other_session: Option<super::lookup::BestCandidate> = None;

    for (digest, slot) in &guard.entries {
        if &slot.sessionless != sessionless || !slot.entry.has_detached() {
            continue;
        }
        let token_len = slot.entry.tokens.len();
        if token_len < min_len {
            continue;
        }
        // Compute the longest common prefix between the request tokens
        // and the entry's stored tokens. When APC partial adoption is
        // enabled we surface candidates whose stored prefix
        // diverges inside the request — the caller will clamp the
        // matched length to a block boundary via the APC discriminator
        // before adopting. With APC off, the legacy "stored prefix must
        // be fully contained in request" gate is preserved bit-exactly.
        if !apc_partial_allowed {
            if token_len > tokens.len() {
                continue;
            }
            if slot.entry.tokens.as_slice() != &tokens[..token_len] {
                continue;
            }
        }
        let common = if apc_partial_allowed {
            common_prefix_len(slot.entry.tokens.as_slice(), tokens)
        } else {
            token_len
        };
        if common < min_len {
            continue;
        }

        let same_session = match (&slot.bucket.session_key, caller_session) {
            (Some(a), Some(b)) => a.as_str() == b,
            (None, None) => true,
            _ => false,
        };
        let candidate = super::lookup::BestCandidate {
            digest: *digest,
            matched: common,
            last_used: slot.entry.last_used(),
        };
        let bucket = if same_session {
            &mut best_same_session
        } else {
            &mut best_other_session
        };
        match bucket {
            None => *bucket = Some(candidate),
            Some(existing) => {
                if better_candidate(&candidate, existing) {
                    *existing = candidate;
                }
            }
        }
    }

    match (best_same_session, best_other_session) {
        (Some(s), Some(o)) if o.matched > s.matched => Some(o),
        (Some(s), _) => Some(s),
        (None, Some(o)) => Some(o),
        (None, None) => None,
    }
}

/// Length of the longest common token prefix between `a` and `b`. Used by
/// the APC partial-adoption scan path so a candidate whose
/// stored prefix diverges inside the request still surfaces with its
/// actual common-prefix length, ready for the block-hash discriminator
/// to clamp.
fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

impl Default for PromptCacheStore {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PromptCacheStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("PromptCacheStore")
            .field("stats", &stats)
            .finish()
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
