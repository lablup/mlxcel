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

//! Shared server state and metrics containers.
//!
//! Route handlers only need these coordination structs; separating them from
//! startup/config policy keeps request handling focused on state access rather
//! than construction details.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::distributed::pipeline::{PipelineObservability, PpTracer};
use crate::tokenizer::MlxcelTokenizer;

use super::batch::BatchObservability;
use super::conversation_store::ConversationStore;
use super::prompt_cache::{PromptCacheStore, metrics::PromptCacheMetrics};
use super::responses_store::ResponsesStore;
use super::{ChatTemplateProcessor, ModelProvider, ServerConfig};

/// Server-wide metrics counters (atomic, lock-free).
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub completion_tokens_total: AtomicU64,
    pub generation_time_ms_total: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            completion_tokens_total: AtomicU64::new(0),
            generation_time_ms_total: AtomicU64::new(0),
        }
    }

    pub fn record_request(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
        generation_time_ms: u64,
    ) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_total
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens_total
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.generation_time_ms_total
            .fetch_add(generation_time_ms, Ordering::Relaxed);
    }
}

/// Batch-level metrics updated by the scheduler thread.
///
/// All fields are atomic for lock-free reads from HTTP handlers and
/// writes from the single scheduler thread.
pub struct BatchMetrics {
    /// Number of sequences currently in the active decode batch.
    pub active_count: AtomicUsize,
    /// Number of requests waiting in the prefill queue.
    pub queue_depth: AtomicUsize,
    /// Cumulative number of sequences that have completed generation.
    pub total_sequences_processed: AtomicU64,
    /// Cumulative number of tokens generated across all sequences.
    pub total_tokens_generated: AtomicU64,
    /// Cumulative number of preemptive evictions.
    pub preemptions_total: AtomicU64,

    // -- prompt-prefix cache Prometheus counters --
    /// Cumulative successful prompt-cache adoptions (hits). Incremented by the
    /// scheduler's `try_adopt_cached_prefix` when an entry is adopted.
    pub prompt_cache_hits_total: AtomicU64,
    /// Cumulative prompt-cache misses. Incremented in the scheduler when no
    /// matching prefix is found and a cold allocate is used.
    pub prompt_cache_misses_total: AtomicU64,
    /// Cumulative prefix tokens reused across all hits. Σ matched-prefix
    /// lengths from `try_adopt_cached_prefix`.
    pub prompt_cache_prefix_tokens_reused_total: AtomicU64,
    /// Cumulative evictions labeled by reason: lru / ttl / capacity.
    ///
    /// Stored as three separate atomics and combined into labeled Prometheus
    /// output by the `/metrics` handler.
    pub prompt_cache_evictions_lru_total: AtomicU64,
    pub prompt_cache_evictions_ttl_total: AtomicU64,
    pub prompt_cache_evictions_capacity_total: AtomicU64,
    /// Current byte footprint of all live prompt-cache entries (gauge).
    pub prompt_cache_bytes: AtomicU64,
    /// Current number of live prompt-cache entries (gauge).
    pub prompt_cache_entries: AtomicU64,
}

impl Default for BatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchMetrics {
    pub fn new() -> Self {
        Self {
            active_count: AtomicUsize::new(0),
            queue_depth: AtomicUsize::new(0),
            total_sequences_processed: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            preemptions_total: AtomicU64::new(0),
            prompt_cache_hits_total: AtomicU64::new(0),
            prompt_cache_misses_total: AtomicU64::new(0),
            prompt_cache_prefix_tokens_reused_total: AtomicU64::new(0),
            prompt_cache_evictions_lru_total: AtomicU64::new(0),
            prompt_cache_evictions_ttl_total: AtomicU64::new(0),
            prompt_cache_evictions_capacity_total: AtomicU64::new(0),
            prompt_cache_bytes: AtomicU64::new(0),
            prompt_cache_entries: AtomicU64::new(0),
        }
    }

    /// Current number of active decode sequences.
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Current prefill queue depth.
    pub fn queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Update active count (called by scheduler thread).
    pub fn set_active_count(&self, count: usize) {
        self.active_count.store(count, Ordering::Relaxed);
    }

    /// Update queue depth (called by scheduler thread).
    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Record a completed sequence (called by scheduler thread).
    pub fn record_sequence_completed(&self, tokens_generated: usize) {
        self.total_sequences_processed
            .fetch_add(1, Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(tokens_generated as u64, Ordering::Relaxed);
    }

    /// Record a preemptive eviction (called by scheduler thread).
    pub fn record_preemption(&self) {
        self.preemptions_total.fetch_add(1, Ordering::Relaxed);
    }

    // -- Prompt-prefix cache helpers --

    /// Record a prompt-cache hit and the number of tokens that were reused.
    ///
    /// Called by the scheduler's `try_adopt_cached_prefix` on success.
    pub fn record_prompt_cache_hit(&self, matched_tokens: usize) {
        self.prompt_cache_hits_total.fetch_add(1, Ordering::Relaxed);
        self.prompt_cache_prefix_tokens_reused_total
            .fetch_add(matched_tokens as u64, Ordering::Relaxed);
    }

    /// Record a prompt-cache miss (cold allocate path).
    pub fn record_prompt_cache_miss(&self) {
        self.prompt_cache_misses_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record an LRU eviction from the prompt-cache store.
    pub fn record_prompt_cache_eviction_lru(&self) {
        self.prompt_cache_evictions_lru_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a TTL eviction from the prompt-cache store.
    pub fn record_prompt_cache_eviction_ttl(&self) {
        self.prompt_cache_evictions_ttl_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a capacity-pressure eviction from the prompt-cache store.
    ///
    /// A capacity eviction occurs when an insert triggers a byte-budget or
    /// entry-count cap enforcement that ejects one or more existing entries.
    pub fn record_prompt_cache_eviction_capacity(&self) {
        self.prompt_cache_evictions_capacity_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Overwrite the prompt-cache byte and entry gauges.
    ///
    /// Called after every insert or eviction so `/metrics` always reflects
    /// the current store size. Safe to call from the scheduler thread only.
    pub fn update_prompt_cache_gauges(&self, bytes: usize, entries: usize) {
        self.prompt_cache_bytes
            .store(bytes as u64, Ordering::Relaxed);
        self.prompt_cache_entries
            .store(entries as u64, Ordering::Relaxed);
    }
}

/// Bridges [`BatchMetrics`] into the [`PromptCacheMetrics`] callback surface
/// consumed by [`PromptCacheStore`].
///
/// Constructed once at server startup and shared via `Arc` between the store
/// and the metrics reader so the single `BatchMetrics` instance stays as the
/// source of truth for all Prometheus output.
///
/// Used by: startup.rs (PromptCacheStore::with_metrics call site)
pub struct BatchMetricsCacheAdapter {
    inner: Arc<BatchMetrics>,
}

impl BatchMetricsCacheAdapter {
    pub fn new(metrics: Arc<BatchMetrics>) -> Self {
        Self { inner: metrics }
    }
}

impl PromptCacheMetrics for BatchMetricsCacheAdapter {
    fn record_insert(&self, _bytes: usize) {
        // Gauge is updated via `update_prompt_cache_gauges`; no separate
        // counter is defined for raw inserts at the `BatchMetrics` level.
    }

    fn record_reject_oversized(&self, _bytes: usize) {
        // No dedicated counter in `BatchMetrics` for oversized rejects;
        // `BatchObservability::prompt_cache_insert_rejects` covers this.
    }

    fn record_lookup(&self, hit: bool, matched_len: usize) {
        if hit {
            self.inner.record_prompt_cache_hit(matched_len);
        } else {
            self.inner.record_prompt_cache_miss();
        }
    }

    fn record_evict_lru(&self, _bytes: usize) {
        self.inner.record_prompt_cache_eviction_lru();
    }

    fn record_evict_ttl(&self, _bytes: usize) {
        self.inner.record_prompt_cache_eviction_ttl();
    }
}

/// Static media-input capability flags resolved once at server startup
/// Used by HTTP handlers to short-circuit unsupported requests
/// with a clear 400 before reaching the model worker.
///
/// The flags reflect the model type detected from `config.json`, not the
/// request payload — they describe what the loaded model could in principle
/// process, not what any individual request asks for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelMediaSupport {
    /// `true` when the loaded model supports `video_url` content blocks.
    /// Currently this is exactly the Gemma 4 VLM family; expand the
    /// detection logic alongside any new video-capable model.
    pub video: bool,
}

/// Shared application state passed into route handlers.
#[derive(Clone)]
pub struct AppState {
    pub model_provider: Arc<ModelProvider>,
    pub config: Arc<ServerConfig>,
    pub chat_template: Arc<ChatTemplateProcessor>,
    /// Tokenizer for tokenize/detokenize endpoints (thread-safe).
    pub tokenizer: Arc<MlxcelTokenizer>,
    /// Model directory path (for props/info).
    pub model_path: PathBuf,
    /// Static media-input capability flags resolved once at startup.
    pub media_support: ModelMediaSupport,
    /// Batch-level metrics (active sequences, queue depth) for admission
    /// control and status reporting.
    pub batch_metrics: Arc<BatchMetrics>,
    /// Detailed batch observability counters (prefill, decode, cache).
    pub batch_observability: Arc<BatchObservability>,
    /// Server metrics (request counts, token throughput).
    pub metrics: Arc<Metrics>,
    /// Pipeline-parallel observability aggregator. Always
    /// present even on non-PP deployments — counters stay at zero.
    pub pp_observability: Arc<PipelineObservability>,
    /// Optional chrome-tracing writer (`--debug-pp-trace`).
    /// `None` when tracing is disabled.
    pub pp_tracer: Option<Arc<PpTracer>>,
    /// Cross-request prompt-prefix KV cache. `None`
    /// when the feature is disabled via [`super::config::ServerConfig::prompt_cache`].
    /// The store is shared with [`ModelProvider`] so the worker thread can
    /// publish and adopt detached caches. HTTP handlers may only call
    /// read-only observation methods on this store.
    pub prompt_cache: Option<Arc<PromptCacheStore>>,
    /// in-memory store backing `POST /v1/responses` with
    /// `store=true`, `GET /v1/responses/:id`, and `previous_response_id`
    /// chaining. `None` when the operator passed
    /// `--responses-store-max-entries 0`.
    pub responses_store: Option<Arc<ResponsesStore>>,
    /// in-memory conversation transcripts referenced by the
    /// `conversation` field in Responses-API requests. `None` when the
    /// store is disabled.
    pub conversation_store: Option<Arc<ConversationStore>>,
}

impl AppState {
    /// Create `AppState` with all required components.
    pub fn new(
        model_provider: Arc<ModelProvider>,
        config: ServerConfig,
        chat_template: ChatTemplateProcessor,
        tokenizer: MlxcelTokenizer,
        model_path: PathBuf,
        batch_metrics: Arc<BatchMetrics>,
    ) -> Self {
        Self {
            model_provider,
            config: Arc::new(config),
            chat_template: Arc::new(chat_template),
            tokenizer: Arc::new(tokenizer),
            model_path,
            media_support: ModelMediaSupport::default(),
            batch_metrics,
            batch_observability: Arc::new(BatchObservability::new()),
            metrics: Arc::new(Metrics::new()),
            pp_observability: Arc::new(PipelineObservability::new()),
            pp_tracer: None,
            prompt_cache: None,
            responses_store: None,
            conversation_store: None,
        }
    }

    /// Create `AppState` with a pre-existing `BatchObservability` instance.
    ///
    /// Use this when the scheduler owns the same `Arc<BatchObservability>`
    /// so that scheduler writes are visible to HTTP handlers.
    pub fn with_observability(
        model_provider: Arc<ModelProvider>,
        config: ServerConfig,
        chat_template: ChatTemplateProcessor,
        tokenizer: MlxcelTokenizer,
        model_path: PathBuf,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Self {
        Self {
            model_provider,
            config: Arc::new(config),
            chat_template: Arc::new(chat_template),
            tokenizer: Arc::new(tokenizer),
            model_path,
            media_support: ModelMediaSupport::default(),
            batch_metrics,
            batch_observability,
            metrics: Arc::new(Metrics::new()),
            pp_observability: Arc::new(PipelineObservability::new()),
            pp_tracer: None,
            prompt_cache: None,
            responses_store: None,
            conversation_store: None,
        }
    }

    /// Override the static media-support flags resolved at startup. Used by
    /// the startup pipeline to record whether the loaded model supports
    /// `video_url` content blocks.
    #[must_use]
    pub fn with_media_support(mut self, support: ModelMediaSupport) -> Self {
        self.media_support = support;
        self
    }

    /// Attach the shared prompt-prefix cache store. Pass `None` when the
    /// feature is disabled so downstream consumers can branch on
    /// `prompt_cache.is_some()`.
    #[must_use]
    pub fn with_prompt_cache(mut self, store: Option<Arc<PromptCacheStore>>) -> Self {
        self.prompt_cache = store;
        self
    }

    /// Attach the Responses-API in-memory response store.
    /// Pass `None` to disable response persistence (and reject any request
    /// that depends on it: `GET /v1/responses/:id`, `previous_response_id`).
    #[must_use]
    pub fn with_responses_store(mut self, store: Option<Arc<ResponsesStore>>) -> Self {
        self.responses_store = store;
        self
    }

    /// Attach the Responses-API conversation store. Pass
    /// `None` to disable; requests referencing `conversation` will still
    /// be accepted but will not be replayed against the missing transcript.
    #[must_use]
    pub fn with_conversation_store(mut self, store: Option<Arc<ConversationStore>>) -> Self {
        self.conversation_store = store;
        self
    }

    /// Override the default no-op pipeline observability aggregator. Used
    /// by the startup pipeline when the coordinator holds a shared `Arc`
    /// so that scheduler writes are visible to HTTP handlers.
    #[must_use]
    pub fn with_pp_observability(mut self, pp: Arc<PipelineObservability>) -> Self {
        self.pp_observability = pp;
        self
    }

    /// Attach a chrome-tracing writer. Used by `--debug-pp-trace`.
    #[must_use]
    pub fn with_pp_tracer(mut self, tracer: Option<Arc<PpTracer>>) -> Self {
        self.pp_tracer = tracer;
        self
    }

    /// Get the display model ID (alias if set, otherwise model provider ID).
    pub fn display_model_id(&self) -> &str {
        self.config
            .model_alias
            .as_deref()
            .unwrap_or_else(|| self.model_provider.model_id())
    }

    /// Check whether the server can accept a new request based on queue depth.
    ///
    /// Returns `true` if the current queue depth is below `max_queue_depth`.
    pub fn can_accept_request(&self) -> bool {
        self.batch_metrics.queue_depth() < self.config.max_queue_depth
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
