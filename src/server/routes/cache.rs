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

//! Prompt-cache observability endpoints.
//!
//! - `GET /v1/cache/stats` — returns a snapshot of the current cache state
//!   (entries, byte footprint, hit/miss counters, APC config).
//! - `POST /v1/cache/reset` — clears every live entry. The store stays alive,
//!   only its contents are dropped, so subsequent inserts work normally.
//!
//! Both endpoints work whether or not the cache is actually enabled — when
//! the store is `None` they return a stable "disabled" payload so monitoring
//! clients can poll without conditional logic.

use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use crate::server::AppState;
use crate::server::batch::ObservabilitySnapshot;

/// Paged KV block-pool view for the cache-stats body (epic #116 #122 c).
///
/// Sourced from the batch observability snapshot so the HTTP body mirrors the
/// Prometheus `mlxcel_cache_pool_paged_*` gauges. Defaults to all-zero (the
/// "no paged pool" state), which lets route-level tests build a response
/// without standing up a worker / scheduler.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PagedBlockStats {
    pub block_size: usize,
    pub blocks_allocated: u64,
    pub blocks_live: u64,
    pub blocks_free: u64,
    pub bytes_reserved: u64,
    pub bytes_in_use: u64,
    pub block_budget: u64,
}

impl PagedBlockStats {
    /// Project the paged block-pool gauges out of a batch observability
    /// snapshot.
    pub(crate) fn from_observability(snap: &ObservabilitySnapshot) -> Self {
        Self {
            block_size: snap.cache_pool_paged_block_size,
            blocks_allocated: snap.cache_pool_paged_blocks_allocated,
            blocks_live: snap.cache_pool_paged_blocks_live,
            blocks_free: snap.cache_pool_paged_blocks_free,
            bytes_reserved: snap.cache_pool_paged_bytes_reserved,
            bytes_in_use: snap.cache_pool_paged_bytes_in_use,
            block_budget: snap.cache_pool_paged_block_budget,
        }
    }
}

/// Prompt-cache reject/decline breakdown for the cache-stats body
/// (issue #774).
///
/// Sourced from the batch observability snapshot rather than the store's own
/// counters (like [`PagedBlockStats`]), because it covers BOTH the
/// donate-back (insert) path and the adopt path -- the store's own
/// `rejections_oversized` only ever sees the former.
#[derive(Debug, Clone, Default)]
pub(crate) struct RejectReasonStats {
    pub oversized: u64,
    pub disabled: u64,
    pub prefix_too_short: u64,
    pub mode_mismatch: u64,
    pub empty_set: u64,
    pub layout_constraints: u64,
    pub block_boundary_floor: u64,
    pub last_reason: Option<String>,
    pub last_seq_id: Option<u64>,
    pub last_context_len: Option<u64>,
    pub last_at_unix_ms: Option<u64>,
}

impl RejectReasonStats {
    /// Project the per-reason reject counters and last-reject snapshot out
    /// of a batch observability snapshot.
    pub(crate) fn from_observability(snap: &ObservabilitySnapshot) -> Self {
        let last = snap.prompt_cache_last_reject.as_ref();
        Self {
            oversized: snap.prompt_cache_reject_oversized,
            disabled: snap.prompt_cache_reject_disabled,
            prefix_too_short: snap.prompt_cache_reject_prefix_too_short,
            mode_mismatch: snap.prompt_cache_reject_mode_mismatch,
            empty_set: snap.prompt_cache_reject_empty_set,
            layout_constraints: snap.prompt_cache_reject_layout_constraints,
            block_boundary_floor: snap.prompt_cache_reject_block_boundary_floor,
            last_reason: last.map(|r| r.reason.to_string()),
            last_seq_id: last.and_then(|r| r.seq_id),
            last_context_len: last.map(|r| r.context_len),
            last_at_unix_ms: last.map(|r| r.at_unix_ms),
        }
    }
}

/// Snapshot of cache state returned by `GET /v1/cache/stats`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatsResponse {
    /// Whether the prompt-prefix cache is enabled at all.
    pub enabled: bool,
    /// Whether APC (block-granularity hash chains) is active.
    pub apc_enabled: bool,
    /// APC block size in tokens. Always reported even when APC is off so the
    /// configured value is visible.
    pub block_size: usize,
    /// APC hash algorithm string (`"sha256"` or `"blake3"`).
    pub hash: String,
    /// Live entries in the store.
    pub entries: usize,
    /// Total bytes consumed by live entries.
    pub bytes: usize,
    /// Configured byte capacity ceiling.
    pub capacity_bytes: usize,
    /// Configured maximum-entries ceiling.
    pub max_entries: usize,
    /// Lifetime hits.
    pub hits: u64,
    /// Lifetime lookups (hits + misses).
    pub lookups: u64,
    /// Hit rate as a fraction in `[0, 1]`. Returned as `0.0` when no lookups
    /// have happened yet, so clients never need to handle division-by-zero.
    pub hit_rate: f64,
    /// Lifetime successful inserts.
    pub inserts: u64,
    /// Lifetime LRU evictions.
    pub evictions_lru: u64,
    /// Lifetime TTL evictions.
    pub evictions_ttl: u64,
    /// Lifetime rejections of oversized inserts.
    pub rejections_oversized: u64,
    /// Total APC block hashes stored across every entry. Always `0` when
    /// APC is disabled or when no entries carry a chain.
    pub total_blocks_stored: usize,
    /// Number of distinct APC block hashes across all entries — a measure
    /// of dedup potential. `total_blocks_stored - unique_block_hashes`
    /// approximates the count of cross-entry block reuse opportunities.
    /// Always `0` when APC is disabled.
    pub unique_block_hashes: usize,
    /// Number of entries that carry a populated APC block-hash chain.
    /// Always `0` when APC is disabled.
    pub apc_active_entries: usize,
    /// Live exact-prefix recurrent/model-owned snapshot entries.
    pub snapshot_entries: usize,
    /// Bytes consumed by live snapshot entries.
    pub snapshot_bytes: usize,
    /// Configured snapshot byte capacity.
    pub snapshot_capacity_bytes: usize,
    /// Configured maximum snapshot entries.
    pub snapshot_max_entries: usize,
    /// Lifetime snapshot hits.
    pub snapshot_hits: u64,
    /// Lifetime snapshot lookups.
    pub snapshot_lookups: u64,
    /// Snapshot hit rate as a fraction in `[0, 1]`.
    pub snapshot_hit_rate: f64,
    /// Lifetime successful snapshot inserts.
    pub snapshot_inserts: u64,
    /// Lifetime snapshot LRU evictions.
    pub snapshot_evictions_lru: u64,
    /// Lifetime snapshot TTL evictions.
    pub snapshot_evictions_ttl: u64,
    /// Lifetime snapshot insert rejections due to size.
    pub snapshot_rejections_oversized: u64,

    // ── Paged KV block pool (epic #116 #122 c) ───────────────────────────────
    // Sourced from the batch observability gauges (not the prompt-cache store),
    // so these are meaningful even when the prompt cache is disabled. They
    // mirror the `mlxcel_cache_pool_paged_*` Prometheus gauges.
    /// Paged decode block size in tokens. `0` when paged decode is inactive
    /// for this worker (dense backend), so a `0` cleanly marks "no paged pool".
    pub paged_block_size: usize,
    /// Paged KV blocks tracked by the allocator (rows minted into the pool).
    pub paged_blocks_allocated: u64,
    /// Paged KV blocks currently held by live sequences.
    pub paged_blocks_live: u64,
    /// Paged KV blocks freed and retained on the pool's free list for reuse.
    pub paged_blocks_free: u64,
    /// Reserved bytes across active paged sequences.
    pub paged_bytes_reserved: u64,
    /// Visible bytes currently in use across active paged sequences.
    pub paged_bytes_in_use: u64,
    /// Configured paged KV block-budget cap (`--kv-cache-budget`). `0` means
    /// unbounded; otherwise the admission gate holds `paged_blocks_live` at or
    /// below this, so `paged_block_budget - paged_blocks_live` is the
    /// acquirable headroom before eviction / preemption kicks in.
    pub paged_block_budget: u64,

    // ── Prompt-cache reject-reason breakdown (issue #774) ────────────────────
    // Sourced from the batch observability gauges (not the prompt-cache
    // store), so -- like the paged block-pool fields above -- these are
    // populated even when the prompt cache is disabled. They cover BOTH the
    // donate-back (insert) path and the adopt path; `rejections_oversized`
    // above only ever sees the former.
    /// Single-entry-too-large declines (donate-back only).
    pub reject_oversized: u64,
    /// Store-disabled declines (donate-back only).
    pub reject_disabled: u64,
    /// Below-`min_prefix_tokens` declines, from either path.
    pub reject_prefix_too_short: u64,
    /// Cross-backend or whole-entry-policy declines (adopt path only).
    pub reject_mode_mismatch: u64,
    /// Nothing-to-cache declines, from either path.
    pub reject_empty_set: u64,
    /// Pool-level operation failures (paged clone/trim, dense truncate,
    /// snapshot restore, or the final adopt call).
    pub reject_layout_constraints: u64,
    /// Paged pool block-size floor pushed the adoptable/donatable length
    /// below `min_prefix_tokens` (adopt path only).
    pub reject_block_boundary_floor: u64,
    /// Reason label of the most recent reject/decline event, if any has
    /// happened yet.
    pub last_reject_reason: Option<String>,
    /// Sequence id of the most recent reject, when one was available at the
    /// decline site (see [`crate::server::prompt_cache::PromptCacheLastReject`]).
    pub last_reject_seq_id: Option<u64>,
    /// Matched-prefix length (adopt) or donated token count (donate) at the
    /// time of the most recent reject.
    pub last_reject_context_len: Option<u64>,
    /// Unix epoch milliseconds of the most recent reject.
    pub last_reject_at_unix_ms: Option<u64>,
}

/// Response for `POST /v1/cache/reset`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheResetResponse {
    /// Whether the call cleared anything. `true` even when the store was
    /// already empty so monitoring clients always see a successful response.
    pub cleared: bool,
    /// Bytes freed by the reset.
    pub freed_bytes: usize,
    /// Entry count that was dropped.
    pub freed_entries: usize,
}

/// `GET /v1/cache/stats` — return a snapshot of cache statistics.
pub async fn cache_stats(State(state): State<AppState>) -> Json<CacheStatsResponse> {
    let obs_snapshot = state.batch_observability.snapshot();
    let paged = PagedBlockStats::from_observability(&obs_snapshot);
    let reject = RejectReasonStats::from_observability(&obs_snapshot);
    Json(build_stats_response(
        state.prompt_cache.as_deref(),
        &state.config.prompt_cache,
        paged,
        reject,
    ))
}

/// `POST /v1/cache/reset` — drop every live cache entry.
///
/// Idempotent: calling on an empty or disabled cache returns a successful
/// response with `freed_bytes = 0, freed_entries = 0`.
pub async fn cache_reset(State(state): State<AppState>) -> Json<CacheResetResponse> {
    Json(build_reset_response(state.prompt_cache.as_deref()))
}

/// Pure helper: build a [`CacheStatsResponse`] from a prompt-cache store and
/// its configuration. Extracted so route-level integration tests can drive
/// the handler logic without constructing a full [`AppState`] (which would
/// require loading a real model).
pub(crate) fn build_stats_response(
    store: Option<&crate::server::prompt_cache::PromptCacheStore>,
    cfg: &crate::server::prompt_cache::PromptCacheConfig,
    paged: PagedBlockStats,
    reject: RejectReasonStats,
) -> CacheStatsResponse {
    let apc = &cfg.apc;
    match store {
        Some(s) => {
            let stats = s.stats();
            let apc_stats = s.apc_stats();
            let hit_rate = if stats.lookups > 0 {
                stats.hits as f64 / stats.lookups as f64
            } else {
                0.0
            };
            let snapshot_hit_rate = if stats.snapshot_lookups > 0 {
                stats.snapshot_hits as f64 / stats.snapshot_lookups as f64
            } else {
                0.0
            };
            CacheStatsResponse {
                enabled: cfg.is_enabled(),
                apc_enabled: cfg.apc_enabled(),
                block_size: apc.block_size,
                hash: apc.hash.to_string(),
                entries: stats.entries,
                bytes: stats.bytes,
                capacity_bytes: cfg.capacity_bytes,
                max_entries: cfg.max_entries,
                hits: stats.hits,
                lookups: stats.lookups,
                hit_rate,
                inserts: stats.inserts,
                evictions_lru: stats.evictions_lru,
                evictions_ttl: stats.evictions_ttl,
                rejections_oversized: stats.rejections_oversized,
                total_blocks_stored: apc_stats.total_blocks_stored,
                unique_block_hashes: apc_stats.unique_block_hashes,
                apc_active_entries: apc_stats.apc_active_entries,
                snapshot_entries: stats.snapshot_entries,
                snapshot_bytes: stats.snapshot_bytes,
                snapshot_capacity_bytes: cfg.snapshot_capacity_bytes,
                snapshot_max_entries: cfg.snapshot_max_entries,
                snapshot_hits: stats.snapshot_hits,
                snapshot_lookups: stats.snapshot_lookups,
                snapshot_hit_rate,
                snapshot_inserts: stats.snapshot_inserts,
                snapshot_evictions_lru: stats.snapshot_evictions_lru,
                snapshot_evictions_ttl: stats.snapshot_evictions_ttl,
                snapshot_rejections_oversized: stats.snapshot_rejections_oversized,
                // Paged block-pool gauges are store-independent.
                paged_block_size: paged.block_size,
                paged_blocks_allocated: paged.blocks_allocated,
                paged_blocks_live: paged.blocks_live,
                paged_blocks_free: paged.blocks_free,
                paged_bytes_reserved: paged.bytes_reserved,
                paged_bytes_in_use: paged.bytes_in_use,
                paged_block_budget: paged.block_budget,
                // Reject-reason breakdown is store-independent (issue #774).
                reject_oversized: reject.oversized,
                reject_disabled: reject.disabled,
                reject_prefix_too_short: reject.prefix_too_short,
                reject_mode_mismatch: reject.mode_mismatch,
                reject_empty_set: reject.empty_set,
                reject_layout_constraints: reject.layout_constraints,
                reject_block_boundary_floor: reject.block_boundary_floor,
                last_reject_reason: reject.last_reason,
                last_reject_seq_id: reject.last_seq_id,
                last_reject_context_len: reject.last_context_len,
                last_reject_at_unix_ms: reject.last_at_unix_ms,
            }
        }
        None => CacheStatsResponse {
            enabled: false,
            apc_enabled: false,
            block_size: apc.block_size,
            hash: apc.hash.to_string(),
            entries: 0,
            bytes: 0,
            capacity_bytes: cfg.capacity_bytes,
            max_entries: cfg.max_entries,
            hits: 0,
            lookups: 0,
            hit_rate: 0.0,
            inserts: 0,
            evictions_lru: 0,
            evictions_ttl: 0,
            rejections_oversized: 0,
            total_blocks_stored: 0,
            unique_block_hashes: 0,
            apc_active_entries: 0,
            snapshot_entries: 0,
            snapshot_bytes: 0,
            snapshot_capacity_bytes: cfg.snapshot_capacity_bytes,
            snapshot_max_entries: cfg.snapshot_max_entries,
            snapshot_hits: 0,
            snapshot_lookups: 0,
            snapshot_hit_rate: 0.0,
            snapshot_inserts: 0,
            snapshot_evictions_lru: 0,
            snapshot_evictions_ttl: 0,
            snapshot_rejections_oversized: 0,
            // Paged decode can run with the prompt cache disabled, so these
            // still reflect the live pool even on the `None` branch.
            paged_block_size: paged.block_size,
            paged_blocks_allocated: paged.blocks_allocated,
            paged_blocks_live: paged.blocks_live,
            paged_blocks_free: paged.blocks_free,
            paged_bytes_reserved: paged.bytes_reserved,
            paged_bytes_in_use: paged.bytes_in_use,
            paged_block_budget: paged.block_budget,
            // Reject events can still occur while the prompt cache is
            // disabled (e.g. `InsertError::Disabled` on an in-flight donate
            // racing a runtime disable), so these mirror the `Some` branch.
            reject_oversized: reject.oversized,
            reject_disabled: reject.disabled,
            reject_prefix_too_short: reject.prefix_too_short,
            reject_mode_mismatch: reject.mode_mismatch,
            reject_empty_set: reject.empty_set,
            reject_layout_constraints: reject.layout_constraints,
            reject_block_boundary_floor: reject.block_boundary_floor,
            last_reject_reason: reject.last_reason,
            last_reject_seq_id: reject.last_seq_id,
            last_reject_context_len: reject.last_context_len,
            last_reject_at_unix_ms: reject.last_at_unix_ms,
        },
    }
}

/// Pure helper: build a [`CacheResetResponse`] by clearing the store.
pub(crate) fn build_reset_response(
    store: Option<&crate::server::prompt_cache::PromptCacheStore>,
) -> CacheResetResponse {
    match store {
        Some(s) => {
            let snapshot = s.stats();
            s.clear();
            CacheResetResponse {
                cleared: true,
                freed_bytes: snapshot.bytes,
                freed_entries: snapshot.entries,
            }
        }
        None => CacheResetResponse {
            cleared: true,
            freed_bytes: 0,
            freed_entries: 0,
        },
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
