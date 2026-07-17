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

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use mlxcel_core::cache::PagedCacheStats;
use serde::Serialize;

use crate::server::prompt_cache::metrics::{PromptCacheRejectCounters, PromptCacheRejectReason};

/// Returns whether `MLXCEL_APC_TRACE=1` opt-in prompt-cache trace logging is
/// enabled (issue #774). Read exactly once via [`OnceLock`] so the store /
/// adopt / reject hot paths pay a single relaxed load per event rather than
/// an environment-variable lookup.
pub fn apc_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLXCEL_APC_TRACE").ok().as_deref() == Some("1"))
}

/// Inclusive upper bounds for the per-request time-to-first-token histogram,
/// in milliseconds. Chosen to span sub-100ms short-prompt TTFT through the
/// multi-second range that long-prompt prefill produces (epic #623 #624).
const TTFT_MS_BUCKETS: [f64; 12] = [
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0,
];

/// Inclusive upper bounds for the per-request decode-rate histogram, in
/// tokens/second. Covers the range from heavily-batched large-model decode up
/// through small-model single-request rates on accelerated hardware.
const DECODE_TOK_S_BUCKETS: [f64; 12] = [
    1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0,
];

/// Fixed-bucket cumulative histogram for per-request telemetry.
///
/// Prometheus histogram semantics: each observation lands in the first bucket
/// whose inclusive upper bound is `>= value`, otherwise the implicit `+Inf`
/// bucket. Bucket counts are stored non-cumulatively and made cumulative at
/// snapshot time. [`observe`](Self::observe) runs exactly once per request
/// completion (never per token), so the compare-exchange loop used to
/// accumulate the floating-point `_sum` is negligible overhead.
pub struct RequestHistogram {
    bounds: &'static [f64],
    counts: Vec<AtomicU64>,
    sum_bits: AtomicU64,
    count: AtomicU64,
}

impl RequestHistogram {
    fn new(bounds: &'static [f64]) -> Self {
        let counts = bounds.iter().map(|_| AtomicU64::new(0)).collect();
        Self {
            bounds,
            counts,
            sum_bits: AtomicU64::new(0.0_f64.to_bits()),
            count: AtomicU64::new(0),
        }
    }

    /// Record a single observation. Non-finite or negative values are clamped
    /// to `0.0` so a stray timing can never corrupt the running sum.
    pub fn observe(&self, value: f64) {
        let value = if value.is_finite() && value > 0.0 {
            value
        } else {
            0.0
        };
        self.count.fetch_add(1, Ordering::Relaxed);

        // Accumulate the f64 sum via a compare-exchange loop so the result is
        // correct regardless of writer count. Only reached once per request.
        let mut cur = self.sum_bits.load(Ordering::Relaxed);
        loop {
            let next = f64::from_bits(cur) + value;
            match self.sum_bits.compare_exchange_weak(
                cur,
                next.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }

        for (i, &bound) in self.bounds.iter().enumerate() {
            if value <= bound {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // Value exceeds every finite bound; it belongs to the +Inf bucket only,
        // which the snapshot derives from the total `count`.
    }

    /// Snapshot with cumulative bucket counts (Prometheus `le` semantics).
    pub fn snapshot(&self) -> HistogramSnapshot {
        let mut cumulative = 0u64;
        let buckets = self
            .bounds
            .iter()
            .enumerate()
            .map(|(i, &bound)| {
                cumulative += self.counts[i].load(Ordering::Relaxed);
                (bound, cumulative)
            })
            .collect();
        HistogramSnapshot {
            buckets,
            sum: f64::from_bits(self.sum_bits.load(Ordering::Relaxed)),
            count: self.count.load(Ordering::Relaxed),
        }
    }
}

/// Serializable snapshot of a [`RequestHistogram`].
///
/// `buckets` holds `(inclusive_upper_bound, cumulative_count)` pairs for the
/// finite buckets; the `+Inf` bucket equals `count`. `sum` is the running total
/// of observed values.
#[derive(Debug, Clone)]
pub struct HistogramSnapshot {
    pub buckets: Vec<(f64, u64)>,
    pub sum: f64,
    pub count: u64,
}

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
    /// Number of decode steps served by the lookahead async_eval pipeline
    /// (issue #632). One increment per steady pipelined tick where the batch
    /// committed a token from a prebuilt (async-scheduled) forward instead of
    /// running synchronously. Ticks that fall back to the synchronous path
    /// (admission, stop, preemption, ineligible sampling, `MLXCEL_FORCE_SYNC`)
    /// do not increment this, so the ratio against `decode_steps_processed`
    /// reports how often lookahead actually engaged.
    pub decode_lookahead_steps: AtomicU64,

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
    /// Per-reason breakdown of prompt-cache reject/decline events across
    /// both the donate-back (insert) path and the adopt path (issue #774).
    /// A superset of `prompt_cache_insert_rejects`: it additionally covers
    /// adopt-path declines (mode mismatch, empty set, layout constraints,
    /// block-boundary flooring) that never call `store.insert`/
    /// `insert_snapshot` at all, so never touch that older counter.
    pub prompt_cache_reject_reasons: PromptCacheRejectCounters,

    // -- per-request latency/throughput histograms (epic #623 #624) --
    /// Per-request time-to-first-token (prefill latency), in milliseconds.
    /// Recorded once per completed request in `finalize_completed`.
    pub request_ttft_ms: RequestHistogram,
    /// Per-request decode throughput, in tokens/second. Recorded once per
    /// completed request from `completion_tokens / generation_only_ms`.
    pub request_decode_tok_s: RequestHistogram,
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
            decode_lookahead_steps: AtomicU64::new(0),
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
            prompt_cache_reject_reasons: PromptCacheRejectCounters::new(),
            request_ttft_ms: RequestHistogram::new(&TTFT_MS_BUCKETS),
            request_decode_tok_s: RequestHistogram::new(&DECODE_TOK_S_BUCKETS),
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

    /// Record that one decode step was served by the lookahead async_eval
    /// pipeline (issue #632). Called once per steady pipelined tick.
    pub fn record_lookahead_step(&self) {
        self.decode_lookahead_steps.fetch_add(1, Ordering::Relaxed);
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

    /// Record a prompt-cache reject/decline with its specific reason
    /// (issue #774). Called from both `donate_finished_sequence_cache` and
    /// `try_adopt_cached_prefix` in the scheduler at every real decline site.
    ///
    /// `seq_id` is `None` for adopt-path declines that occur before a
    /// sequence id has been allocated (the common case: most adopt declines
    /// happen before `allocate_sequence_state`/`cache_pool.adopt*` runs); it
    /// is `Some` for donate-path declines (a finished sequence always
    /// carries one) and the handful of adopt declines that occur after a
    /// sequence id was already allocated (the snapshot-restore-failure
    /// path). `context_len` is the matched-prefix length for adopt declines,
    /// or the donated token count for donate declines.
    ///
    /// Also emits a `tracing::info!` one-liner when `MLXCEL_APC_TRACE=1` is
    /// set (see [`apc_trace_enabled`]) so reject events are greppable
    /// without needing metrics scraping.
    pub fn record_prompt_cache_reject(
        &self,
        reason: PromptCacheRejectReason,
        seq_id: Option<u64>,
        context_len: usize,
    ) {
        self.prompt_cache_reject_reasons
            .record(reason, seq_id, context_len);
        if apc_trace_enabled() {
            tracing::info!(
                apc_event = "reject",
                reason = reason.as_str(),
                seq_id = ?seq_id,
                context_len,
                "prompt-cache reject"
            );
        }
    }

    /// Record per-request completion telemetry: time-to-first-token and the
    /// observed decode rate. Called exactly once per finished sequence from the
    /// scheduler thread (never per token), so the cost is negligible.
    ///
    /// Requests that produced no completion tokens are ignored: TTFT is
    /// undefined without a first token and a zero-length generation has no
    /// decode rate. The decode-rate observation is likewise skipped when the
    /// decode phase rounded to 0ms (single-token completions), where a rate is
    /// not measurable.
    pub fn record_request_completion(
        &self,
        prompt_eval_ms: u64,
        generation_only_ms: u64,
        completion_tokens: usize,
    ) {
        if completion_tokens == 0 {
            return;
        }
        self.request_ttft_ms.observe(prompt_eval_ms as f64);
        if generation_only_ms > 0 {
            let decode_tok_s = completion_tokens as f64 * 1000.0 / generation_only_ms as f64;
            self.request_decode_tok_s.observe(decode_tok_s);
        }
    }

    /// Snapshot of the per-request TTFT histogram for the `/metrics` endpoint.
    pub fn ttft_ms_snapshot(&self) -> HistogramSnapshot {
        self.request_ttft_ms.snapshot()
    }

    /// Snapshot of the per-request decode-rate histogram for `/metrics`.
    pub fn decode_tok_s_snapshot(&self) -> HistogramSnapshot {
        self.request_decode_tok_s.snapshot()
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
            decode_lookahead_steps: self.decode_lookahead_steps.load(Ordering::Relaxed),
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
            prompt_cache_reject_oversized: self
                .prompt_cache_reject_reasons
                .count(PromptCacheRejectReason::Oversized),
            prompt_cache_reject_disabled: self
                .prompt_cache_reject_reasons
                .count(PromptCacheRejectReason::Disabled),
            prompt_cache_reject_prefix_too_short: self
                .prompt_cache_reject_reasons
                .count(PromptCacheRejectReason::PrefixTooShort),
            prompt_cache_reject_mode_mismatch: self
                .prompt_cache_reject_reasons
                .count(PromptCacheRejectReason::ModeMismatch),
            prompt_cache_reject_empty_set: self
                .prompt_cache_reject_reasons
                .count(PromptCacheRejectReason::EmptySet),
            prompt_cache_reject_layout_constraints: self
                .prompt_cache_reject_reasons
                .count(PromptCacheRejectReason::LayoutConstraints),
            prompt_cache_reject_block_boundary_floor: self
                .prompt_cache_reject_reasons
                .count(PromptCacheRejectReason::BlockBoundaryFloor),
            prompt_cache_last_reject: self.prompt_cache_reject_reasons.last().map(|r| {
                PromptCacheLastRejectSnapshot {
                    reason: r.reason,
                    seq_id: r.seq_id,
                    context_len: r.context_len as u64,
                    at_unix_ms: r.at_unix_ms,
                }
            }),
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
    /// Decode steps served by the lookahead async_eval pipeline (issue #632).
    pub decode_lookahead_steps: u64,
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
    /// Per-reason prompt-cache reject/decline counts (issue #774). See
    /// [`PromptCacheRejectReason`] for what each reason covers; this is a
    /// superset of `prompt_cache_insert_rejects` that also counts adopt-path
    /// declines.
    pub prompt_cache_reject_oversized: u64,
    pub prompt_cache_reject_disabled: u64,
    pub prompt_cache_reject_prefix_too_short: u64,
    pub prompt_cache_reject_mode_mismatch: u64,
    pub prompt_cache_reject_empty_set: u64,
    pub prompt_cache_reject_layout_constraints: u64,
    pub prompt_cache_reject_block_boundary_floor: u64,
    /// Most recent prompt-cache reject/decline event, if any has happened
    /// yet.
    pub prompt_cache_last_reject: Option<PromptCacheLastRejectSnapshot>,
}

/// Serializable snapshot of [`crate::server::prompt_cache::PromptCacheLastReject`]
/// for the `/v1/cache/stats` JSON body and the `/health` observability
/// snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PromptCacheLastRejectSnapshot {
    pub reason: &'static str,
    pub seq_id: Option<u64>,
    pub context_len: u64,
    pub at_unix_ms: u64,
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
        assert_eq!(snap.decode_lookahead_steps, 0);
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
    fn record_request_completion_populates_histograms() {
        let obs = BatchObservability::new();
        // 200ms TTFT, 100 decode tokens over 1000ms decode -> 100 tok/s.
        obs.record_request_completion(200, 1000, 100);
        // 40ms TTFT, 20 tokens over 500ms -> 40 tok/s.
        obs.record_request_completion(40, 500, 20);

        let ttft = obs.ttft_ms_snapshot();
        assert_eq!(ttft.count, 2);
        assert!((ttft.sum - 240.0).abs() < 1e-6);
        // le=50 bucket (index 3) should include only the 40ms observation.
        assert_eq!(ttft.buckets[3].0, 50.0);
        assert_eq!(ttft.buckets[3].1, 1);
        // le=250 bucket (index 5) should include both observations.
        assert_eq!(ttft.buckets[5].0, 250.0);
        assert_eq!(ttft.buckets[5].1, 2);

        let decode = obs.decode_tok_s_snapshot();
        assert_eq!(decode.count, 2);
        assert!((decode.sum - 140.0).abs() < 1e-6);
        // le=50 bucket (index 4) includes only the 40 tok/s observation.
        assert_eq!(decode.buckets[4].0, 50.0);
        assert_eq!(decode.buckets[4].1, 1);
        // le=100 bucket (index 6) includes both observations.
        assert_eq!(decode.buckets[6].0, 100.0);
        assert_eq!(decode.buckets[6].1, 2);
    }

    #[test]
    fn record_request_completion_skips_zero_token_requests() {
        let obs = BatchObservability::new();
        // Zero completion tokens: neither histogram should record anything.
        obs.record_request_completion(500, 0, 0);
        assert_eq!(obs.ttft_ms_snapshot().count, 0);
        assert_eq!(obs.decode_tok_s_snapshot().count, 0);

        // One token with a 0ms decode phase: TTFT records, decode rate skips.
        obs.record_request_completion(120, 0, 1);
        assert_eq!(obs.ttft_ms_snapshot().count, 1);
        assert_eq!(obs.decode_tok_s_snapshot().count, 0);
    }

    #[test]
    fn histogram_counts_plus_inf_observations() {
        let obs = BatchObservability::new();
        // 40000ms exceeds the largest finite TTFT bound (30000ms).
        obs.record_request_completion(40_000, 1000, 10);
        let ttft = obs.ttft_ms_snapshot();
        assert_eq!(ttft.count, 1);
        // No finite bucket captured the observation; the +Inf bucket (== count)
        // still accounts for it.
        assert_eq!(ttft.buckets.last().unwrap().1, 0);
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

    // -- Prompt-cache reject-reason breakdown (issue #774) --

    #[test]
    fn reject_reasons_start_zeroed_with_no_last_reject() {
        let obs = BatchObservability::new();
        let snap = obs.snapshot();
        assert_eq!(snap.prompt_cache_reject_oversized, 0);
        assert_eq!(snap.prompt_cache_reject_disabled, 0);
        assert_eq!(snap.prompt_cache_reject_prefix_too_short, 0);
        assert_eq!(snap.prompt_cache_reject_mode_mismatch, 0);
        assert_eq!(snap.prompt_cache_reject_empty_set, 0);
        assert_eq!(snap.prompt_cache_reject_layout_constraints, 0);
        assert_eq!(snap.prompt_cache_reject_block_boundary_floor, 0);
        assert!(snap.prompt_cache_last_reject.is_none());
    }

    #[test]
    fn record_prompt_cache_reject_increments_only_its_own_reason() {
        let obs = BatchObservability::new();
        obs.record_prompt_cache_reject(PromptCacheRejectReason::ModeMismatch, None, 12);
        obs.record_prompt_cache_reject(PromptCacheRejectReason::ModeMismatch, Some(7), 34);
        obs.record_prompt_cache_reject(PromptCacheRejectReason::BlockBoundaryFloor, Some(9), 56);

        let snap = obs.snapshot();
        assert_eq!(snap.prompt_cache_reject_mode_mismatch, 2);
        assert_eq!(snap.prompt_cache_reject_block_boundary_floor, 1);
        // Every other reason stays at zero: the counters are independent.
        assert_eq!(snap.prompt_cache_reject_oversized, 0);
        assert_eq!(snap.prompt_cache_reject_disabled, 0);
        assert_eq!(snap.prompt_cache_reject_prefix_too_short, 0);
        assert_eq!(snap.prompt_cache_reject_empty_set, 0);
        assert_eq!(snap.prompt_cache_reject_layout_constraints, 0);
    }

    #[test]
    fn record_prompt_cache_reject_updates_last_reject_snapshot() {
        let obs = BatchObservability::new();
        obs.record_prompt_cache_reject(PromptCacheRejectReason::EmptySet, Some(1), 10);
        obs.record_prompt_cache_reject(PromptCacheRejectReason::Oversized, Some(2), 20);

        // `last_reject` always reflects the most recent event, regardless of
        // which reason it was.
        let last = obs
            .snapshot()
            .prompt_cache_last_reject
            .expect("a reject has been recorded");
        assert_eq!(last.reason, "oversized");
        assert_eq!(last.seq_id, Some(2));
        assert_eq!(last.context_len, 20);
    }

    #[test]
    fn snapshot_serializes_reject_reason_fields() {
        let obs = BatchObservability::new();
        obs.record_prompt_cache_reject(PromptCacheRejectReason::LayoutConstraints, Some(5), 99);
        let json = serde_json::to_string(&obs.snapshot()).unwrap();
        assert!(json.contains("\"prompt_cache_reject_layout_constraints\":1"));
        assert!(json.contains("\"reason\":\"layout_constraints\""));
        assert!(json.contains("\"seq_id\":5"));
        assert!(json.contains("\"context_len\":99"));
    }
}
