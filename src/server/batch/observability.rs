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

//! Batch observability metrics for continuous batching.
//!
//! [`BatchObservability`] extends the existing [`super::super::state::BatchMetrics`]
//! with detailed counters for prefill, decode, and cache utilization. All fields
//! are atomic for lock-free reads from HTTP handlers and writes from the single
//! scheduler thread.
//!
//! The scheduler updates these counters at key lifecycle points (prefill start,
//! decode step, sequence completion) and HTTP handlers read them from `/health`
//! and `/metrics` endpoints without locking.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use mlxcel_core::cache::PagedCacheStats;
use serde::Serialize;

/// Detailed observability counters for the batch scheduler.
///
/// These complement the coarser [`BatchMetrics`] already tracked in
/// `server/state.rs`. All operations use `Ordering::Relaxed` because
/// exactness is not required for monitoring -- slight staleness is
/// acceptable and avoids unnecessary memory barriers on the hot path.
pub struct BatchObservability {
    // -- Counters (monotonically increasing) --
    /// Total sequences that entered the prefill stage.
    pub sequences_started: AtomicU64,
    /// Total sequences that completed generation (stop or length).
    pub sequences_completed: AtomicU64,
    /// Cumulative prompt tokens processed across all prefills.
    pub total_prefill_tokens: AtomicU64,
    /// Cumulative decode tokens generated across all sequences.
    pub total_decode_tokens: AtomicU64,
    /// Number of prefill chunks processed (chunked prefill only).
    pub prefill_chunks_processed: AtomicU64,
    /// Number of decode steps executed (one per tick per batch).
    pub decode_steps_processed: AtomicU64,

    // -- Gauges (point-in-time values set by the scheduler) --
    /// Number of sequences currently in the active decode batch.
    pub current_batch_size: AtomicUsize,
    /// Number of requests waiting in the prefill queue.
    pub current_queue_depth: AtomicUsize,
    /// Number of active cache entries in the pool.
    pub cache_pool_active: AtomicUsize,
    /// Estimated memory bytes held by active KV caches.
    pub cache_pool_bytes: AtomicU64,
    /// Paged decode block size in tokens.
    pub cache_pool_paged_block_size: AtomicUsize,
    /// Total blocks currently known to the paged allocator.
    pub cache_pool_paged_blocks_allocated: AtomicU64,
    /// Blocks currently in live use by active sequences.
    pub cache_pool_paged_blocks_live: AtomicU64,
    /// Reusable blocks currently on the free list.
    pub cache_pool_paged_blocks_free: AtomicU64,
    /// Reserved bytes across active paged sequences.
    pub cache_pool_paged_bytes_reserved: AtomicU64,
    /// Visible bytes currently in use across active paged sequences.
    pub cache_pool_paged_bytes_in_use: AtomicU64,
    /// Configured paged KV block-budget cap (epic #116 #122). `0` means
    /// unbounded (no `--kv-cache-budget` set) — the acquirable headroom under
    /// a cap is `budget - blocks_live`.
    pub cache_pool_paged_block_budget: AtomicU64,
    /// Times paged decode was requested but fell back to dense.
    pub decode_storage_fallbacks: AtomicU64,

    // -- prompt-prefix cache --
    /// Cumulative count of successful prompt-cache adoptions (hits).
    pub prompt_cache_hits: AtomicU64,
    /// Cumulative count of prompt tokens that bypassed prefill because they
    /// were already present in the adopted detached KV cache.
    pub prompt_cache_hit_tokens: AtomicU64,
    /// Cumulative count of sequences that donated their cache back to the
    /// store on healthy completion.
    pub prompt_cache_inserts: AtomicU64,
    /// Cumulative count of donate-back attempts rejected by the store
    /// (e.g. `InsertError::OversizedEntry`, `InsertError::Disabled`,
    /// `InsertError::PrefixTooShort`).
    pub prompt_cache_insert_rejects: AtomicU64,
}

impl Default for BatchObservability {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchObservability {
    /// Create a zeroed observability instance.
    pub fn new() -> Self {
        Self {
            sequences_started: AtomicU64::new(0),
            sequences_completed: AtomicU64::new(0),
            total_prefill_tokens: AtomicU64::new(0),
            total_decode_tokens: AtomicU64::new(0),
            prefill_chunks_processed: AtomicU64::new(0),
            decode_steps_processed: AtomicU64::new(0),
            current_batch_size: AtomicUsize::new(0),
            current_queue_depth: AtomicUsize::new(0),
            cache_pool_active: AtomicUsize::new(0),
            cache_pool_bytes: AtomicU64::new(0),
            cache_pool_paged_block_size: AtomicUsize::new(0),
            cache_pool_paged_blocks_allocated: AtomicU64::new(0),
            cache_pool_paged_blocks_live: AtomicU64::new(0),
            cache_pool_paged_blocks_free: AtomicU64::new(0),
            cache_pool_paged_bytes_reserved: AtomicU64::new(0),
            cache_pool_paged_bytes_in_use: AtomicU64::new(0),
            cache_pool_paged_block_budget: AtomicU64::new(0),
            decode_storage_fallbacks: AtomicU64::new(0),
            prompt_cache_hits: AtomicU64::new(0),
            prompt_cache_hit_tokens: AtomicU64::new(0),
            prompt_cache_inserts: AtomicU64::new(0),
            prompt_cache_insert_rejects: AtomicU64::new(0),
        }
    }

    // -- Counter increments (called by the scheduler thread) --

    /// Record that a sequence has started prefill.
    pub fn record_prefill_start(&self, prompt_len: usize) {
        self.sequences_started.fetch_add(1, Ordering::Relaxed);
        self.total_prefill_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);
    }

    /// Record that a chunked prefill chunk was processed.
    pub fn record_prefill_chunk(&self) {
        self.prefill_chunks_processed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record that one decode step was executed for a batch of the given size.
    pub fn record_decode_step(&self, batch_size: usize) {
        self.decode_steps_processed.fetch_add(1, Ordering::Relaxed);
        self.total_decode_tokens
            .fetch_add(batch_size as u64, Ordering::Relaxed);
    }

    /// Record that a sequence has completed generation.
    pub fn record_sequence_completed(&self) {
        self.sequences_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that requested paged decode fell back to dense execution.
    pub fn record_decode_storage_fallback(&self) {
        self.decode_storage_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful prompt-cache hit.
    ///
    /// `matched_tokens` is the number of leading tokens covered by the
    /// adopted detached KV cache — exactly the count subtracted from the
    /// sequence's prefill workload.
    pub fn record_prompt_cache_hit(&self, matched_tokens: usize) {
        self.prompt_cache_hits.fetch_add(1, Ordering::Relaxed);
        self.prompt_cache_hit_tokens
            .fetch_add(matched_tokens as u64, Ordering::Relaxed);
    }

    /// Record a successful donate-back insert into the prompt cache store.
    pub fn record_prompt_cache_insert(&self) {
        self.prompt_cache_inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a donate-back attempt was rejected by the store
    /// (oversized entry, feature disabled, prefix too short, …).
    pub fn record_prompt_cache_insert_reject(&self) {
        self.prompt_cache_insert_rejects
            .fetch_add(1, Ordering::Relaxed);
    }

    // -- Gauge updates (called by the scheduler thread) --

    /// Update the point-in-time gauge values.
    pub fn update_gauges(
        &self,
        batch_size: usize,
        queue_depth: usize,
        cache_active: usize,
        cache_bytes: u64,
        paged_block_size: usize,
        paged_stats: Option<PagedCacheStats>,
        paged_block_budget: u64,
    ) {
        self.current_batch_size.store(batch_size, Ordering::Relaxed);
        self.current_queue_depth
            .store(queue_depth, Ordering::Relaxed);
        self.cache_pool_active
            .store(cache_active, Ordering::Relaxed);
        self.cache_pool_bytes.store(cache_bytes, Ordering::Relaxed);

        let stats = paged_stats.unwrap_or_default();
        self.cache_pool_paged_block_size
            .store(paged_block_size, Ordering::Relaxed);
        self.cache_pool_paged_blocks_allocated
            .store(stats.allocated_blocks as u64, Ordering::Relaxed);
        self.cache_pool_paged_blocks_live
            .store(stats.live_blocks as u64, Ordering::Relaxed);
        self.cache_pool_paged_blocks_free
            .store(stats.free_blocks as u64, Ordering::Relaxed);
        self.cache_pool_paged_bytes_reserved
            .store(stats.bytes_reserved as u64, Ordering::Relaxed);
        self.cache_pool_paged_bytes_in_use
            .store(stats.bytes_in_use as u64, Ordering::Relaxed);
        self.cache_pool_paged_block_budget
            .store(paged_block_budget, Ordering::Relaxed);
    }

    // -- Snapshot for HTTP handlers --

    /// Create a serializable snapshot of the current observability state.
    pub fn snapshot(&self) -> ObservabilitySnapshot {
        ObservabilitySnapshot {
            sequences_started: self.sequences_started.load(Ordering::Relaxed),
            sequences_completed: self.sequences_completed.load(Ordering::Relaxed),
            total_prefill_tokens: self.total_prefill_tokens.load(Ordering::Relaxed),
            total_decode_tokens: self.total_decode_tokens.load(Ordering::Relaxed),
            prefill_chunks_processed: self.prefill_chunks_processed.load(Ordering::Relaxed),
            decode_steps_processed: self.decode_steps_processed.load(Ordering::Relaxed),
            current_batch_size: self.current_batch_size.load(Ordering::Relaxed),
            current_queue_depth: self.current_queue_depth.load(Ordering::Relaxed),
            cache_pool_active: self.cache_pool_active.load(Ordering::Relaxed),
            cache_pool_bytes: self.cache_pool_bytes.load(Ordering::Relaxed),
            cache_pool_paged_block_size: self.cache_pool_paged_block_size.load(Ordering::Relaxed),
            cache_pool_paged_blocks_allocated: self
                .cache_pool_paged_blocks_allocated
                .load(Ordering::Relaxed),
            cache_pool_paged_blocks_live: self.cache_pool_paged_blocks_live.load(Ordering::Relaxed),
            cache_pool_paged_blocks_free: self.cache_pool_paged_blocks_free.load(Ordering::Relaxed),
            cache_pool_paged_bytes_reserved: self
                .cache_pool_paged_bytes_reserved
                .load(Ordering::Relaxed),
            cache_pool_paged_bytes_in_use: self
                .cache_pool_paged_bytes_in_use
                .load(Ordering::Relaxed),
            cache_pool_paged_block_budget: self
                .cache_pool_paged_block_budget
                .load(Ordering::Relaxed),
            decode_storage_fallbacks: self.decode_storage_fallbacks.load(Ordering::Relaxed),
            prompt_cache_hits: self.prompt_cache_hits.load(Ordering::Relaxed),
            prompt_cache_hit_tokens: self.prompt_cache_hit_tokens.load(Ordering::Relaxed),
            prompt_cache_inserts: self.prompt_cache_inserts.load(Ordering::Relaxed),
            prompt_cache_insert_rejects: self.prompt_cache_insert_rejects.load(Ordering::Relaxed),
        }
    }
}

/// Serializable point-in-time snapshot of batch observability metrics.
///
/// Returned by [`BatchObservability::snapshot()`] and included in the
/// `/health` endpoint response when the batch scheduler is active.
#[derive(Debug, Clone, Serialize)]
pub struct ObservabilitySnapshot {
    pub sequences_started: u64,
    pub sequences_completed: u64,
    pub total_prefill_tokens: u64,
    pub total_decode_tokens: u64,
    pub prefill_chunks_processed: u64,
    pub decode_steps_processed: u64,
    pub current_batch_size: usize,
    pub current_queue_depth: usize,
    pub cache_pool_active: usize,
    pub cache_pool_bytes: u64,
    pub cache_pool_paged_block_size: usize,
    pub cache_pool_paged_blocks_allocated: u64,
    pub cache_pool_paged_blocks_live: u64,
    pub cache_pool_paged_blocks_free: u64,
    pub cache_pool_paged_bytes_reserved: u64,
    pub cache_pool_paged_bytes_in_use: u64,
    pub cache_pool_paged_block_budget: u64,
    pub decode_storage_fallbacks: u64,
    /// successful prompt-cache adoptions.
    pub prompt_cache_hits: u64,
    /// tokens skipped due to cache hits (Σ
    /// matched-prefix lengths across all hits).
    pub prompt_cache_hit_tokens: u64,
    /// successful donate-back inserts.
    pub prompt_cache_inserts: u64,
    /// rejected donate-back inserts (oversized
    /// entry, store disabled, prefix too short, …).
    pub prompt_cache_insert_rejects: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_observability_starts_zeroed() {
        let obs = BatchObservability::new();
        let snap = obs.snapshot();
        assert_eq!(snap.sequences_started, 0);
        assert_eq!(snap.sequences_completed, 0);
        assert_eq!(snap.total_prefill_tokens, 0);
        assert_eq!(snap.total_decode_tokens, 0);
        assert_eq!(snap.prefill_chunks_processed, 0);
        assert_eq!(snap.decode_steps_processed, 0);
        assert_eq!(snap.current_batch_size, 0);
        assert_eq!(snap.current_queue_depth, 0);
        assert_eq!(snap.cache_pool_active, 0);
        assert_eq!(snap.cache_pool_bytes, 0);
        assert_eq!(snap.cache_pool_paged_block_size, 0);
        assert_eq!(snap.cache_pool_paged_blocks_allocated, 0);
        assert_eq!(snap.cache_pool_paged_blocks_live, 0);
        assert_eq!(snap.cache_pool_paged_blocks_free, 0);
        assert_eq!(snap.cache_pool_paged_bytes_reserved, 0);
        assert_eq!(snap.cache_pool_paged_bytes_in_use, 0);
        assert_eq!(snap.decode_storage_fallbacks, 0);
        assert_eq!(snap.prompt_cache_hits, 0);
        assert_eq!(snap.prompt_cache_hit_tokens, 0);
        assert_eq!(snap.prompt_cache_inserts, 0);
        assert_eq!(snap.prompt_cache_insert_rejects, 0);
    }

    #[test]
    fn record_prefill_increments_counters() {
        let obs = BatchObservability::new();
        obs.record_prefill_start(128);
        obs.record_prefill_start(256);
        let snap = obs.snapshot();
        assert_eq!(snap.sequences_started, 2);
        assert_eq!(snap.total_prefill_tokens, 384);
    }

    #[test]
    fn record_decode_step_increments_counters() {
        let obs = BatchObservability::new();
        obs.record_decode_step(4);
        obs.record_decode_step(3);
        let snap = obs.snapshot();
        assert_eq!(snap.decode_steps_processed, 2);
        assert_eq!(snap.total_decode_tokens, 7);
    }

    #[test]
    fn record_completion_increments_counter() {
        let obs = BatchObservability::new();
        obs.record_sequence_completed();
        obs.record_sequence_completed();
        obs.record_sequence_completed();
        assert_eq!(obs.snapshot().sequences_completed, 3);
    }

    #[test]
    fn update_gauges_sets_values() {
        let obs = BatchObservability::new();
        obs.update_gauges(
            4,
            10,
            8,
            1024 * 1024,
            32,
            Some(PagedCacheStats {
                allocated_blocks: 16,
                live_blocks: 12,
                free_blocks: 4,
                bytes_reserved: 8192,
                bytes_in_use: 6144,
            }),
            64,
        );
        let snap = obs.snapshot();
        assert_eq!(snap.current_batch_size, 4);
        assert_eq!(snap.current_queue_depth, 10);
        assert_eq!(snap.cache_pool_active, 8);
        assert_eq!(snap.cache_pool_bytes, 1024 * 1024);
        assert_eq!(snap.cache_pool_paged_block_size, 32);
        assert_eq!(snap.cache_pool_paged_blocks_allocated, 16);
        assert_eq!(snap.cache_pool_paged_blocks_live, 12);
        assert_eq!(snap.cache_pool_paged_blocks_free, 4);
        assert_eq!(snap.cache_pool_paged_bytes_reserved, 8192);
        assert_eq!(snap.cache_pool_paged_bytes_in_use, 6144);
        assert_eq!(snap.cache_pool_paged_block_budget, 64);
    }

    #[test]
    fn snapshot_is_serializable() {
        let obs = BatchObservability::new();
        obs.record_prefill_start(100);
        obs.record_decode_step(2);
        obs.record_decode_storage_fallback();
        let snap = obs.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"sequences_started\":1"));
        assert!(json.contains("\"total_decode_tokens\":2"));
        assert!(json.contains("\"decode_storage_fallbacks\":1"));
    }
}
