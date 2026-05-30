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
    Json(build_stats_response(
        state.prompt_cache.as_deref(),
        &state.config.prompt_cache,
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
