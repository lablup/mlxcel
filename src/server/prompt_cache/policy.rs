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

//! Eviction policy and configuration for the prompt prefix cache.
//!
//! The policy is intentionally small: LRU with a TTL escape hatch and both
//! byte-budget and entry-count caps. A production deployment can revisit
//! this layer without touching the store's public surface.

use std::fmt;
use std::time::Duration;

use super::block_hash::{ApcHashAlgo, DEFAULT_APC_BLOCK_SIZE};

/// Runtime configuration for the prompt-prefix cache store.
///
/// CLI/env parsing for these fields is tracked. Until then
/// construct an instance via [`PromptCacheConfig::default`] or the explicit
/// constructor and hand it to [`super::PromptCacheStore::with_config`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PromptCacheConfig {
    /// Toggle the entire feature. When `false`, callers should skip
    /// constructing a [`super::PromptCacheStore`] entirely so no memory is
    /// reserved. The store itself also honors this value as an early-out.
    pub enabled: bool,
    /// Total byte budget across all entries. Inserts that would exceed this
    /// after eviction are rejected.
    pub capacity_bytes: usize,
    /// Upper bound on the number of live cache entries. Oldest entries are
    /// evicted first once this cap is hit.
    pub max_entries: usize,
    /// Time-to-live for an entry since its last successful lookup. Lazy
    /// sweep: entries are checked and possibly expired on lookup and on
    /// [`super::PromptCacheStore::evict_if_needed`].
    pub ttl: Duration,
    /// Minimum number of prompt tokens required before an entry is eligible
    /// to be inserted. Helps avoid polluting the cache with tiny prefixes
    /// that can't really amortize the detach/adopt overhead.
    pub min_prefix_tokens: usize,
    /// Automatic Prefix Caching (APC) configuration.
    ///
    /// APC is an additive, opt-in feature layered on top of the existing
    /// prompt-prefix cache. It computes block-granularity hash chains so
    /// finer-grained KV reuse becomes possible. When [`ApcConfig::enabled`]
    /// is `false` (the default) no extra work is done and behaviour is
    /// identical to the earlier store.
    pub apc: ApcConfig,
    /// Byte budget dedicated to exact-prefix recurrent/model-owned snapshots.
    pub snapshot_capacity_bytes: usize,
    /// Maximum live exact-prefix snapshot entries.
    pub snapshot_max_entries: usize,
    /// Time-to-live for snapshot entries since their last successful restore.
    pub snapshot_ttl: Duration,
}

/// Automatic Prefix Caching configuration knobs.
///
/// Mirrors the upstream `mlx-vlm` PR #1114 / WIP #1103 surface so callers can
/// tune block size, count, and hash algorithm via CLI flags or `APC_*` env
/// vars. See [`super::block_hash`] for how `block_size` and `hash` interact.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ApcConfig {
    /// Master switch for APC. When `false`, the block-hash chain is not
    /// computed and the existing whole-prefix cache path is the only one
    /// active.
    pub enabled: bool,
    /// Number of tokens per APC block. Defaults to
    /// [`DEFAULT_APC_BLOCK_SIZE`] (16) to match upstream.
    pub block_size: usize,
    /// Optional hard cap on the number of block-hash entries APC tracks. When
    /// `None`, the budget is derived from
    /// [`PromptCacheConfig::max_entries`] (a heuristic — the actual block
    /// count is bounded by `max_entries * (avg_prefix_len / block_size)`).
    pub num_blocks: Option<usize>,
    /// Hash algorithm used for block hashes. Defaults to SHA-256 for upstream
    /// wire-compat.
    pub hash: ApcHashAlgo,
}

impl Default for ApcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            block_size: DEFAULT_APC_BLOCK_SIZE,
            num_blocks: None,
            hash: ApcHashAlgo::Sha256,
        }
    }
}

impl ApcConfig {
    /// Whether APC is fully active.
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.block_size > 0
    }
}

impl PromptCacheConfig {
    /// Default capacity in bytes: 2 GiB.
    pub const DEFAULT_CAPACITY_BYTES: usize = 2 * 1024 * 1024 * 1024;
    /// Default maximum entry count.
    pub const DEFAULT_MAX_ENTRIES: usize = 1024;
    /// Default TTL: 3600 seconds.
    pub const DEFAULT_TTL_SECONDS: u64 = 3600;
    /// Default minimum prompt-prefix length before caching kicks in.
    pub const DEFAULT_MIN_PREFIX_TOKENS: usize = 32;
    /// Default snapshot capacity: snapshots are smaller than KV blocks, so
    /// reserve one quarter of the KV budget for them by default.
    pub const DEFAULT_SNAPSHOT_CAPACITY_BYTES: usize = Self::DEFAULT_CAPACITY_BYTES / 4;
    /// Default snapshot entry count: exact-prefix snapshots are smaller, so
    /// keep more of them than detached KV entries.
    pub const DEFAULT_SNAPSHOT_MAX_ENTRIES: usize = Self::DEFAULT_MAX_ENTRIES * 4;
    /// Default snapshot TTL: allow longer multi-turn chat reuse than KV blocks.
    pub const DEFAULT_SNAPSHOT_TTL_SECONDS: u64 = Self::DEFAULT_TTL_SECONDS * 2;

    /// Build a fully-specified config. Prefer [`PromptCacheConfig::default`]
    /// unless a caller has a reason to deviate.
    pub fn new(
        enabled: bool,
        capacity_bytes: usize,
        max_entries: usize,
        ttl: Duration,
        min_prefix_tokens: usize,
    ) -> Self {
        Self {
            enabled,
            capacity_bytes,
            max_entries,
            ttl,
            min_prefix_tokens,
            apc: ApcConfig::default(),
            snapshot_capacity_bytes: Self::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            snapshot_max_entries: Self::DEFAULT_SNAPSHOT_MAX_ENTRIES,
            snapshot_ttl: Duration::from_secs(Self::DEFAULT_SNAPSHOT_TTL_SECONDS),
        }
    }

    /// Builder-style: attach APC configuration. Returns `self` so callers can
    /// chain `.with_apc(...)` after [`PromptCacheConfig::new`].
    #[must_use]
    pub fn with_apc(mut self, apc: ApcConfig) -> Self {
        self.apc = apc;
        self
    }

    /// Builder-style: override exact-prefix snapshot limits.
    #[must_use]
    pub fn with_snapshot_limits(
        mut self,
        capacity_bytes: usize,
        max_entries: usize,
        ttl: Duration,
    ) -> Self {
        self.snapshot_capacity_bytes = capacity_bytes;
        self.snapshot_max_entries = max_entries;
        self.snapshot_ttl = ttl;
        self
    }

    /// Config variant with the feature disabled. Safe to hand to the store
    /// constructor; the resulting store is a cheap no-op.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Whether the store should accept any insert at all.
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.capacity_bytes > 0 && self.max_entries > 0
    }

    /// Whether the APC block-hash mode is active. Implies the store itself is
    /// enabled — APC layers on top of the regular prompt-prefix cache.
    pub fn apc_enabled(&self) -> bool {
        self.is_enabled() && self.apc.is_enabled()
    }
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capacity_bytes: Self::DEFAULT_CAPACITY_BYTES,
            max_entries: Self::DEFAULT_MAX_ENTRIES,
            ttl: Duration::from_secs(Self::DEFAULT_TTL_SECONDS),
            min_prefix_tokens: Self::DEFAULT_MIN_PREFIX_TOKENS,
            apc: ApcConfig::default(),
            snapshot_capacity_bytes: Self::DEFAULT_SNAPSHOT_CAPACITY_BYTES,
            snapshot_max_entries: Self::DEFAULT_SNAPSHOT_MAX_ENTRIES,
            snapshot_ttl: Duration::from_secs(Self::DEFAULT_SNAPSHOT_TTL_SECONDS),
        }
    }
}

/// Aggregate store statistics, intended for both tests and the metrics
/// bridge tracked.
///
/// This is a pure snapshot: values are captured under the store's lock and
/// returned by value so callers can release the lock immediately.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PromptCacheStats {
    /// Number of live entries in the store.
    pub entries: usize,
    /// Sum of `size_bytes` across all live entries.
    pub bytes: usize,
    /// Lifetime count of successful inserts.
    pub inserts: u64,
    /// Lifetime count of inserts rejected because the single entry exceeded
    /// the configured byte budget.
    pub rejections_oversized: u64,
    /// Lifetime count of lookup calls.
    pub lookups: u64,
    /// Lifetime count of lookup calls that returned `Some(..)`.
    pub hits: u64,
    /// Lifetime count of entries evicted due to LRU pressure (entry-cap or
    /// byte-cap).
    pub evictions_lru: u64,
    /// Lifetime count of entries evicted because the TTL expired.
    pub evictions_ttl: u64,
    /// Number of live exact-prefix recurrent/model-owned snapshots.
    pub snapshot_entries: usize,
    /// Sum of bytes across live snapshot entries.
    pub snapshot_bytes: usize,
    /// Lifetime count of successful snapshot inserts.
    pub snapshot_inserts: u64,
    /// Lifetime count of snapshot inserts rejected as oversized.
    pub snapshot_rejections_oversized: u64,
    /// Lifetime count of snapshot lookups.
    pub snapshot_lookups: u64,
    /// Lifetime count of snapshot lookup hits.
    pub snapshot_hits: u64,
    /// Lifetime count of snapshot LRU evictions.
    pub snapshot_evictions_lru: u64,
    /// Lifetime count of snapshot TTL evictions.
    pub snapshot_evictions_ttl: u64,
}

impl fmt::Display for PromptCacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "entries={} bytes={} inserts={} hits={}/{} lru_evict={} ttl_evict={} reject_oversized={} snapshot_entries={} snapshot_bytes={} snapshot_hits={}/{}",
            self.entries,
            self.bytes,
            self.inserts,
            self.hits,
            self.lookups,
            self.evictions_lru,
            self.evictions_ttl,
            self.rejections_oversized,
            self.snapshot_entries,
            self.snapshot_bytes,
            self.snapshot_hits,
            self.snapshot_lookups,
        )
    }
}
