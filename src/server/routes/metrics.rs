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

//! Prometheus-compatible metrics endpoint.
//!
//! This route is read-only and should remain separate from generation policy.
//! Includes both server-level request counters and batch observability gauges,
//! plus the pipeline-parallel families added by
//!
//! - Per-stage utilization and bubble ratio.
//! - Activation transfer latency histograms (p50/p95/p99 per stage pair).
//! - KV cache admission rejection counters (labeled by stage + reason).
//! - Elastic repartition event counter emission
//!   path.

use std::fmt::Write;
use std::sync::atomic::Ordering;

use axum::{
    extract::State,
    http::header,
    response::{IntoResponse, Response},
};

use crate::distributed::pipeline::PipelineObservabilitySnapshot;
use crate::server::AppState;
use crate::server::batch::observability::HistogramSnapshot;

#[path = "metrics_audio.rs"]
mod metrics_audio;
use metrics_audio::append_audio_preprocess_metrics;

/// GET /metrics -- Prometheus text format
///
/// Always mounted; without `--metrics` it answers b10621's own diagnostic
/// instead of a 404 (#1440). The body opens with the b10621 `llamacpp:`
/// metric families a llama-server scrape config matches, followed by
/// mlxcel's native `mlxcel_` families.
pub async fn metrics(State(state): State<AppState>) -> Response {
    if !state.config.enable_metrics_endpoint {
        return super::slots::llama_not_supported(
            "This server does not support metrics endpoint. Start it with `--metrics`",
        );
    }
    let m = &state.metrics;

    let requests = m.requests_total.load(Ordering::Relaxed);
    let prompt_tokens = m.prompt_tokens_total.load(Ordering::Relaxed);
    let completion_tokens = m.completion_tokens_total.load(Ordering::Relaxed);
    let gen_time_ms = m.generation_time_ms_total.load(Ordering::Relaxed);

    // B9 — read process-wide lang-bias observability counters.
    let lang_bias_applied = mlxcel_core::lang_bias_applied_total();
    let lang_bias_suppressed = mlxcel_core::lang_bias_tokens_suppressed_total();
    // byte-fragment suppression counter. Strict subset of
    // `lang_bias_suppressed`; tracks suppressions that came from the opt-in
    // byte-fragment classifier so operators can observe over-suppression.
    let lang_bias_byte_fragment = mlxcel_core::lang_bias_byte_fragment_suppressions_total();

    let slots_total = state.config.max_batch_size;
    let active = state.batch_metrics.active_count();
    let slots_available = slots_total.saturating_sub(active);
    let queue_depth = state.batch_metrics.queue_depth();

    // prompt-prefix cache Prometheus counters
    let pc_hits = state
        .batch_metrics
        .prompt_cache_hits_total
        .load(Ordering::Relaxed);
    let pc_misses = state
        .batch_metrics
        .prompt_cache_misses_total
        .load(Ordering::Relaxed);
    let pc_reused_tokens = state
        .batch_metrics
        .prompt_cache_prefix_tokens_reused_total
        .load(Ordering::Relaxed);
    let pc_evict_lru = state
        .batch_metrics
        .prompt_cache_evictions_lru_total
        .load(Ordering::Relaxed);
    let pc_evict_ttl = state
        .batch_metrics
        .prompt_cache_evictions_ttl_total
        .load(Ordering::Relaxed);
    let pc_evict_capacity = state
        .batch_metrics
        .prompt_cache_evictions_capacity_total
        .load(Ordering::Relaxed);
    let pc_bytes = state
        .batch_metrics
        .prompt_cache_bytes
        .load(Ordering::Relaxed);
    let pc_entries = state
        .batch_metrics
        .prompt_cache_entries
        .load(Ordering::Relaxed);
    let pc_snapshot_hits = state
        .batch_metrics
        .prompt_cache_snapshot_hits_total
        .load(Ordering::Relaxed);
    let pc_snapshot_misses = state
        .batch_metrics
        .prompt_cache_snapshot_misses_total
        .load(Ordering::Relaxed);
    let pc_snapshot_reused_tokens = state
        .batch_metrics
        .prompt_cache_snapshot_tokens_reused_total
        .load(Ordering::Relaxed);
    let pc_snapshot_evict_lru = state
        .batch_metrics
        .prompt_cache_snapshot_evictions_lru_total
        .load(Ordering::Relaxed);
    let pc_snapshot_evict_ttl = state
        .batch_metrics
        .prompt_cache_snapshot_evictions_ttl_total
        .load(Ordering::Relaxed);

    // Batch observability counters
    let obs = state.batch_observability.snapshot();

    // Prompt-cache reject-reason breakdown (issue #774). Covers both the
    // donate-back (insert) path and the adopt path.
    let pc_reject_oversized = obs.prompt_cache_reject_oversized;
    let pc_reject_disabled = obs.prompt_cache_reject_disabled;
    let pc_reject_prefix_too_short = obs.prompt_cache_reject_prefix_too_short;
    let pc_reject_mode_mismatch = obs.prompt_cache_reject_mode_mismatch;
    let pc_reject_empty_set = obs.prompt_cache_reject_empty_set;
    let pc_reject_layout_constraints = obs.prompt_cache_reject_layout_constraints;
    let pc_reject_block_boundary_floor = obs.prompt_cache_reject_block_boundary_floor;
    let pc_reject_snapshot_diverged = obs.prompt_cache_reject_snapshot_diverged;
    let pc_reject_model_owned_state = obs.prompt_cache_reject_model_owned_state;

    let pp_snapshot = state.pp_observability.snapshot();
    let body = format!(
        "# HELP mlxcel_requests_total Total number of generation requests\n\
         # TYPE mlxcel_requests_total counter\n\
         mlxcel_requests_total {requests}\n\
         # HELP mlxcel_prompt_tokens_total Total prompt tokens processed\n\
         # TYPE mlxcel_prompt_tokens_total counter\n\
         mlxcel_prompt_tokens_total {prompt_tokens}\n\
         # HELP mlxcel_completion_tokens_total Total completion tokens generated\n\
         # TYPE mlxcel_completion_tokens_total counter\n\
         mlxcel_completion_tokens_total {completion_tokens}\n\
         # HELP mlxcel_generation_time_seconds_total Total generation time in seconds\n\
         # TYPE mlxcel_generation_time_seconds_total counter\n\
         mlxcel_generation_time_seconds_total {gen_time_sec:.3}\n\
         # HELP mlxcel_slots_total Total number of parallel slots\n\
         # TYPE mlxcel_slots_total gauge\n\
         mlxcel_slots_total {slots_total}\n\
         # HELP mlxcel_slots_available Available parallel slots\n\
         # TYPE mlxcel_slots_available gauge\n\
         mlxcel_slots_available {slots_available}\n\
         # HELP mlxcel_queue_depth Current prefill queue depth\n\
         # TYPE mlxcel_queue_depth gauge\n\
         mlxcel_queue_depth {queue_depth}\n\
         # HELP mlxcel_batch_sequences_started Total sequences that entered prefill\n\
         # TYPE mlxcel_batch_sequences_started counter\n\
         mlxcel_batch_sequences_started {seq_started}\n\
         # HELP mlxcel_batch_sequences_completed Total sequences that completed generation\n\
         # TYPE mlxcel_batch_sequences_completed counter\n\
         mlxcel_batch_sequences_completed {seq_completed}\n\
         # HELP mlxcel_batch_prefill_tokens_total Cumulative prefill tokens processed\n\
         # TYPE mlxcel_batch_prefill_tokens_total counter\n\
         mlxcel_batch_prefill_tokens_total {prefill_tokens}\n\
         # HELP mlxcel_batch_decode_tokens_total Cumulative decode tokens generated\n\
         # TYPE mlxcel_batch_decode_tokens_total counter\n\
         mlxcel_batch_decode_tokens_total {decode_tokens}\n\
         # HELP mlxcel_batch_decode_steps_total Total decode steps executed\n\
         # TYPE mlxcel_batch_decode_steps_total counter\n\
         mlxcel_batch_decode_steps_total {decode_steps}\n\
         # HELP mlxcel_batch_decode_lookahead_steps_total Decode steps served by the lookahead async_eval pipeline\n\
         # TYPE mlxcel_batch_decode_lookahead_steps_total counter\n\
         mlxcel_batch_decode_lookahead_steps_total {decode_lookahead_steps}\n\
         # HELP mlxcel_batch_prefill_chunks_total Total prefill chunks processed\n\
         # TYPE mlxcel_batch_prefill_chunks_total counter\n\
         mlxcel_batch_prefill_chunks_total {prefill_chunks}\n\
         # HELP mlxcel_batch_mixed_steps_total Mixed prefill/decode ticks (issue #908 prototype; zero unless MLXCEL_MIXED_STEP is set)\n\
         # TYPE mlxcel_batch_mixed_steps_total counter\n\
         mlxcel_batch_mixed_steps_total {mixed_steps}\n\
         # HELP mlxcel_batch_prefill_grants_total Fairness grants handed to a parked chunked prefill (issue #1011; zero when --prefill-grant-interval is 0)\n\
         # TYPE mlxcel_batch_prefill_grants_total counter\n\
         mlxcel_batch_prefill_grants_total {prefill_grants}\n\
         # HELP mlxcel_batch_current_size Current active batch size\n\
         # TYPE mlxcel_batch_current_size gauge\n\
         mlxcel_batch_current_size {batch_size}\n\
         # HELP mlxcel_cache_pool_active Active cache entries\n\
         # TYPE mlxcel_cache_pool_active gauge\n\
         mlxcel_cache_pool_active {cache_active}\n\
         # HELP mlxcel_cache_pool_paged_block_size Paged decode block size in tokens\n\
         # TYPE mlxcel_cache_pool_paged_block_size gauge\n\
         mlxcel_cache_pool_paged_block_size {paged_block_size}\n\
         # HELP mlxcel_cache_pool_paged_blocks_allocated Total paged KV blocks tracked by the allocator\n\
         # TYPE mlxcel_cache_pool_paged_blocks_allocated gauge\n\
         mlxcel_cache_pool_paged_blocks_allocated {paged_blocks_allocated}\n\
         # HELP mlxcel_cache_pool_paged_blocks_live Paged KV blocks currently in live use\n\
         # TYPE mlxcel_cache_pool_paged_blocks_live gauge\n\
         mlxcel_cache_pool_paged_blocks_live {paged_blocks_live}\n\
         # HELP mlxcel_cache_pool_paged_blocks_free Paged KV blocks currently free for reuse\n\
         # TYPE mlxcel_cache_pool_paged_blocks_free gauge\n\
         mlxcel_cache_pool_paged_blocks_free {paged_blocks_free}\n\
         # HELP mlxcel_cache_pool_paged_bytes_reserved Real bytes of allocated paged pool slabs (capacity, including grow slack)\n\
         # TYPE mlxcel_cache_pool_paged_bytes_reserved gauge\n\
         mlxcel_cache_pool_paged_bytes_reserved {paged_bytes_reserved}\n\
         # HELP mlxcel_cache_pool_paged_bytes_in_use Real bytes of pool rows mapped to live blocks (active sequences plus parked prompt-cache pins)\n\
         # TYPE mlxcel_cache_pool_paged_bytes_in_use gauge\n\
         mlxcel_cache_pool_paged_bytes_in_use {paged_bytes_in_use}\n\
         # HELP mlxcel_cache_pool_paged_block_budget Configured paged KV block-budget cap (0 = unbounded)\n\
         # TYPE mlxcel_cache_pool_paged_block_budget gauge\n\
         mlxcel_cache_pool_paged_block_budget {paged_block_budget}\n\
         # HELP mlxcel_decode_storage_fallbacks_total Number of paged decode fallback events\n\
         # TYPE mlxcel_decode_storage_fallbacks_total counter\n\
         mlxcel_decode_storage_fallbacks_total {decode_storage_fallbacks}\n\
         # HELP mlxcel_paged_decode_launches_total Layer decode steps by attention path\n\
         # TYPE mlxcel_paged_decode_launches_total counter\n\
         mlxcel_paged_decode_launches_total{{path=\"fused_v2\"}} {paged_decode_v2_launches}\n\
         mlxcel_paged_decode_launches_total{{path=\"gather\"}} {paged_decode_gather_fallbacks}\n\
         mlxcel_paged_decode_launches_total{{path=\"cascade\"}} {cascade_launches}\n\
         # HELP mlxcel_cascade_shared_tokens_total Cumulative KV tokens hoisted into a shared-span cascade launch\n\
         # TYPE mlxcel_cascade_shared_tokens_total counter\n\
         mlxcel_cascade_shared_tokens_total {cascade_shared_tokens}\n\
         # HELP mlxcel_cascade_member_sequences_total Cumulative sequences sharing a cascade span\n\
         # TYPE mlxcel_cascade_member_sequences_total counter\n\
         mlxcel_cascade_member_sequences_total {cascade_member_seqs}\n\
         # HELP mlxcel_cascade_failures_total Planned cascade launches that fell back to the flat v2 launch\n\
         # TYPE mlxcel_cascade_failures_total counter\n\
         mlxcel_cascade_failures_total {cascade_failures}\n\
         # HELP mlxcel_lang_bias_applied_total Sampling steps where language token bias was applied\n\
         # TYPE mlxcel_lang_bias_applied_total counter\n\
         mlxcel_lang_bias_applied_total {lang_bias_applied}\n\
         # HELP mlxcel_lang_bias_tokens_suppressed_total Sampling steps where the pre-bias top-1 token was neg-inf suppressed\n\
         # TYPE mlxcel_lang_bias_tokens_suppressed_total counter\n\
         mlxcel_lang_bias_tokens_suppressed_total {lang_bias_suppressed}\n\
         # HELP mlxcel_lang_bias_byte_fragment_suppressions_total Suppressions where the neg-inf top-1 token was classified via UTF-8 start-byte\n\
         # TYPE mlxcel_lang_bias_byte_fragment_suppressions_total counter\n\
         mlxcel_lang_bias_byte_fragment_suppressions_total {lang_bias_byte_fragment}\n\
         # HELP mlxcel_prompt_cache_hits_total Successful prompt-prefix cache adoptions\n\
         # TYPE mlxcel_prompt_cache_hits_total counter\n\
         mlxcel_prompt_cache_hits_total {pc_hits}\n\
         # HELP mlxcel_prompt_cache_misses_total Prompt-prefix cache lookups that produced no match\n\
         # TYPE mlxcel_prompt_cache_misses_total counter\n\
         mlxcel_prompt_cache_misses_total {pc_misses}\n\
         # HELP mlxcel_prompt_cache_prefix_tokens_reused_total Total prompt tokens reused from prefix cache\n\
         # TYPE mlxcel_prompt_cache_prefix_tokens_reused_total counter\n\
         mlxcel_prompt_cache_prefix_tokens_reused_total {pc_reused_tokens}\n\
         # HELP mlxcel_prompt_cache_evictions_total Prompt-prefix cache evictions labeled by reason\n\
         # TYPE mlxcel_prompt_cache_evictions_total counter\n\
         mlxcel_prompt_cache_evictions_total{{reason=\"lru\"}} {pc_evict_lru}\n\
         mlxcel_prompt_cache_evictions_total{{reason=\"ttl\"}} {pc_evict_ttl}\n\
         mlxcel_prompt_cache_evictions_total{{reason=\"capacity\"}} {pc_evict_capacity}\n\
         # HELP mlxcel_prompt_cache_bytes Current byte footprint of all live prompt-cache entries\n\
         # TYPE mlxcel_prompt_cache_bytes gauge\n\
         mlxcel_prompt_cache_bytes {pc_bytes}\n\
         # HELP mlxcel_prompt_cache_entries Current number of live prompt-cache entries\n\
         # TYPE mlxcel_prompt_cache_entries gauge\n\
         mlxcel_prompt_cache_entries {pc_entries}\n\
         # HELP mlxcel_prompt_cache_snapshot_hits_total Successful exact-prefix recurrent-state snapshot restores\n\
         # TYPE mlxcel_prompt_cache_snapshot_hits_total counter\n\
         mlxcel_prompt_cache_snapshot_hits_total {pc_snapshot_hits}\n\
         # HELP mlxcel_prompt_cache_snapshot_misses_total Exact-prefix recurrent-state snapshot lookups that missed\n\
         # TYPE mlxcel_prompt_cache_snapshot_misses_total counter\n\
         mlxcel_prompt_cache_snapshot_misses_total {pc_snapshot_misses}\n\
         # HELP mlxcel_prompt_cache_snapshot_tokens_reused_total Total prompt tokens reused from recurrent-state snapshots\n\
         # TYPE mlxcel_prompt_cache_snapshot_tokens_reused_total counter\n\
         mlxcel_prompt_cache_snapshot_tokens_reused_total {pc_snapshot_reused_tokens}\n\
         # HELP mlxcel_prompt_cache_snapshot_evictions_total Snapshot evictions labeled by reason\n\
         # TYPE mlxcel_prompt_cache_snapshot_evictions_total counter\n\
         mlxcel_prompt_cache_snapshot_evictions_total{{reason=\"lru\"}} {pc_snapshot_evict_lru}\n\
         mlxcel_prompt_cache_snapshot_evictions_total{{reason=\"ttl\"}} {pc_snapshot_evict_ttl}\n\
         # HELP mlxcel_prompt_cache_reject_total Prompt-prefix cache reject/decline events labeled by reason (covers both the donate-back and adopt paths)\n\
         # TYPE mlxcel_prompt_cache_reject_total counter\n\
         mlxcel_prompt_cache_reject_total{{reason=\"oversized\"}} {pc_reject_oversized}\n\
         mlxcel_prompt_cache_reject_total{{reason=\"disabled\"}} {pc_reject_disabled}\n\
         mlxcel_prompt_cache_reject_total{{reason=\"prefix_too_short\"}} {pc_reject_prefix_too_short}\n\
         mlxcel_prompt_cache_reject_total{{reason=\"mode_mismatch\"}} {pc_reject_mode_mismatch}\n\
         mlxcel_prompt_cache_reject_total{{reason=\"empty_set\"}} {pc_reject_empty_set}\n\
         mlxcel_prompt_cache_reject_total{{reason=\"layout_constraints\"}} {pc_reject_layout_constraints}\n\
         mlxcel_prompt_cache_reject_total{{reason=\"block_boundary_floor\"}} {pc_reject_block_boundary_floor}\n\
         mlxcel_prompt_cache_reject_total{{reason=\"snapshot_diverged\"}} {pc_reject_snapshot_diverged}\n\
         mlxcel_prompt_cache_reject_total{{reason=\"model_owned_state\"}} {pc_reject_model_owned_state}\n",
        gen_time_sec = gen_time_ms as f64 / 1000.0,
        seq_started = obs.sequences_started,
        seq_completed = obs.sequences_completed,
        prefill_tokens = obs.total_prefill_tokens,
        decode_tokens = obs.total_decode_tokens,
        decode_steps = obs.decode_steps_processed,
        decode_lookahead_steps = obs.decode_lookahead_steps,
        prefill_chunks = obs.prefill_chunks_processed,
        mixed_steps = obs.mixed_steps_processed,
        prefill_grants = obs.prefill_grants_processed,
        batch_size = obs.current_batch_size,
        cache_active = obs.cache_pool_active,
        paged_block_size = obs.cache_pool_paged_block_size,
        paged_blocks_allocated = obs.cache_pool_paged_blocks_allocated,
        paged_blocks_live = obs.cache_pool_paged_blocks_live,
        paged_blocks_free = obs.cache_pool_paged_blocks_free,
        paged_bytes_reserved = obs.cache_pool_paged_bytes_reserved,
        paged_bytes_in_use = obs.cache_pool_paged_bytes_in_use,
        paged_block_budget = obs.cache_pool_paged_block_budget,
        decode_storage_fallbacks = obs.decode_storage_fallbacks,
        paged_decode_v2_launches = obs.paged_decode_v2_launches,
        paged_decode_gather_fallbacks = obs.paged_decode_gather_fallbacks,
        cascade_launches = obs.cascade_launches,
        cascade_shared_tokens = obs.cascade_shared_tokens,
        cascade_member_seqs = obs.cascade_member_seqs,
        cascade_failures = obs.cascade_failures,
        pc_hits = pc_hits,
        pc_misses = pc_misses,
        pc_reused_tokens = pc_reused_tokens,
        pc_evict_lru = pc_evict_lru,
        pc_evict_ttl = pc_evict_ttl,
        pc_evict_capacity = pc_evict_capacity,
        pc_bytes = pc_bytes,
        pc_entries = pc_entries,
        pc_snapshot_hits = pc_snapshot_hits,
        pc_snapshot_misses = pc_snapshot_misses,
        pc_snapshot_reused_tokens = pc_snapshot_reused_tokens,
        pc_snapshot_evict_lru = pc_snapshot_evict_lru,
        pc_snapshot_evict_ttl = pc_snapshot_evict_ttl,
        pc_reject_oversized = pc_reject_oversized,
        pc_reject_disabled = pc_reject_disabled,
        pc_reject_prefix_too_short = pc_reject_prefix_too_short,
        pc_reject_mode_mismatch = pc_reject_mode_mismatch,
        pc_reject_empty_set = pc_reject_empty_set,
        pc_reject_layout_constraints = pc_reject_layout_constraints,
        pc_reject_block_boundary_floor = pc_reject_block_boundary_floor,
        pc_reject_snapshot_diverged = pc_reject_snapshot_diverged,
        pc_reject_model_owned_state = pc_reject_model_owned_state,
    );

    let mut body = body;
    append_audio_preprocess_metrics(&mut body, &obs);

    // Per-request TTFT / decode-rate histograms (epic #623 #624). Populated
    // once per completed request by the scheduler's `finalize_completed`.
    let ttft = state.batch_observability.ttft_ms_snapshot();
    let decode_rate = state.batch_observability.decode_tok_s_snapshot();
    append_request_histogram(
        &mut body,
        "mlxcel_request_ttft_ms",
        "Per-request time to first token (prefill latency) in milliseconds",
        &ttft,
    );
    append_request_histogram(
        &mut body,
        "mlxcel_request_decode_tok_s",
        "Per-request decode throughput in tokens per second",
        &decode_rate,
    );

    append_pipeline_metrics(&mut body, &pp_snapshot);

    let mut full = String::new();
    append_llamacpp_metrics(&mut full, &state);
    full.push_str(&body);

    (
        [
            (
                header::CONTENT_TYPE.as_str(),
                "text/plain; version=0.0.4; charset=utf-8".to_string(),
            ),
            (
                // b10621 sends the server start time on every scrape so
                // Prometheus can detect counter resets across restarts.
                "Process-Start-Time-Unix",
                state.started_at_unix.to_string(),
            ),
        ],
        full,
    )
        .into_response()
}

/// The b10621 `llamacpp:` metric families (#1440), name-for-name what
/// `server_task_result_metrics::to_metrics` exports at the pinned commit
/// (https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-task.cpp),
/// so a Prometheus scrape config written for llama-server matches mlxcel
/// unchanged.
fn append_llamacpp_metrics(body: &mut String, state: &AppState) {
    let obs = &state.batch_observability;
    let load = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);

    let prompt_tokens = load(&obs.llama_prompt_tokens_processed);
    let prompt_cached = load(&obs.llama_prompt_tokens_cached);
    let prompt_us = load(&obs.llama_prompt_us_total);
    let predicted_tokens = load(&obs.llama_predicted_tokens);
    let predicted_us = load(&obs.llama_predicted_us_total);
    let n_decode = load(&obs.llama_decode_calls);
    let n_tokens_max = load(&obs.llama_n_tokens_max);
    let busy_slots = load(&obs.llama_busy_slots_total);
    let (spec_draft, spec_accepted, spec_steps) =
        crate::server::batch::observability::spec_counters::snapshot();

    // b10621 averages its throughput gauges over the window between two
    // scrapes (`reset_bucket` after each one); differencing the cumulative
    // counters against the previous scrape's baseline produces the same
    // bucket averages without a second set of hot-path counters.
    let (bucket_prompt_rate, bucket_predict_rate) = {
        let mut baseline = state
            .llama_scrape
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let d_prompt_tokens = prompt_tokens.saturating_sub(baseline.prompt_tokens);
        let d_prompt_us = prompt_us.saturating_sub(baseline.prompt_us);
        let d_pred_tokens = predicted_tokens.saturating_sub(baseline.predicted_tokens);
        let d_pred_us = predicted_us.saturating_sub(baseline.predicted_us);
        baseline.prompt_tokens = prompt_tokens;
        baseline.prompt_us = prompt_us;
        baseline.predicted_tokens = predicted_tokens;
        baseline.predicted_us = predicted_us;
        let rate = |tokens: u64, us: u64| {
            if us == 0 {
                0.0
            } else {
                tokens as f64 * 1.0e6 / us as f64
            }
        };
        (
            rate(d_prompt_tokens, d_prompt_us),
            rate(d_pred_tokens, d_pred_us),
        )
    };

    let counters: [(&str, &str, f64); 10] = [
        (
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            prompt_tokens as f64,
        ),
        (
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            prompt_cached as f64,
        ),
        (
            "prompt_seconds_total",
            "Total time spent processing prompts",
            prompt_us as f64 / 1.0e6,
        ),
        (
            "tokens_predicted_total",
            "Number of generation tokens processed",
            predicted_tokens as f64,
        ),
        (
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            predicted_us as f64 / 1.0e6,
        ),
        (
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            n_decode as f64,
        ),
        (
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            n_tokens_max as f64,
        ),
        (
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            spec_draft as f64,
        ),
        (
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            spec_accepted as f64,
        ),
        (
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            spec_steps as f64,
        ),
    ];
    let gauges: [(&str, &str, f64); 5] = [
        (
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            bucket_prompt_rate,
        ),
        (
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            bucket_predict_rate,
        ),
        (
            "requests_processing",
            "Number of requests processing",
            state.batch_metrics.active_count() as f64,
        ),
        (
            "requests_deferred",
            "Number of requests deferred",
            state.batch_metrics.queue_depth() as f64,
        ),
        (
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            busy_slots as f64 / (n_decode as f64).max(1.0),
        ),
    ];
    let mut add_items = |kind: &str, items: &[(&str, &str, f64)]| {
        for (name, help, value) in items {
            let _ = writeln!(body, "# HELP llamacpp:{name} {help}");
            let _ = writeln!(body, "# TYPE llamacpp:{name} {kind}");
            let _ = writeln!(body, "llamacpp:{name} {value}");
        }
    };
    add_items("counter", &counters);
    add_items("gauge", &gauges);
}

/// Append one per-request histogram family in Prometheus text format.
///
/// Emits the standard `_bucket{le="…"}` cumulative series (including the
/// `le="+Inf"` bucket), `_sum`, and `_count` lines. The snapshot reads its
/// atomics exactly once, so this is safe to call on every scrape.
fn append_request_histogram(body: &mut String, name: &str, help: &str, snap: &HistogramSnapshot) {
    let _ = writeln!(body, "# HELP {name} {help}");
    let _ = writeln!(body, "# TYPE {name} histogram");
    for (bound, cumulative) in &snap.buckets {
        let _ = writeln!(body, "{name}_bucket{{le=\"{bound}\"}} {cumulative}");
    }
    let _ = writeln!(body, "{name}_bucket{{le=\"+Inf\"}} {}", snap.count);
    let _ = writeln!(body, "{name}_sum {:.3}", snap.sum);
    let _ = writeln!(body, "{name}_count {}", snap.count);
}

/// Append the pipeline-parallel observability families in Prometheus text
/// format. Safe to call on every scrape — the snapshot is an owned value
/// that reads atomics / locked maps exactly once.
fn append_pipeline_metrics(body: &mut String, snap: &PipelineObservabilitySnapshot) {
    // --- Stage utilization ---
    let _ = writeln!(
        body,
        "# HELP mlxcel_pp_stage_busy_fraction Fraction of stage time spent on compute/transfer\n\
         # TYPE mlxcel_pp_stage_busy_fraction gauge"
    );
    for s in &snap.stage_utilization {
        let _ = writeln!(
            body,
            "mlxcel_pp_stage_busy_fraction{{stage=\"{}\"}} {:.6}",
            s.stage_index,
            s.busy_fraction()
        );
    }

    let _ = writeln!(
        body,
        "# HELP mlxcel_pp_stage_bubble_fraction Fraction of stage time spent idle (bubble)\n\
         # TYPE mlxcel_pp_stage_bubble_fraction gauge"
    );
    for s in &snap.stage_utilization {
        let _ = writeln!(
            body,
            "mlxcel_pp_stage_bubble_fraction{{stage=\"{}\"}} {:.6}",
            s.stage_index,
            s.bubble_fraction()
        );
    }

    // --- Mean bubble ratio ---
    let _ = writeln!(
        body,
        "# HELP mlxcel_pp_mean_bubble_ratio Rolling mean bubble ratio across recorded steps\n\
         # TYPE mlxcel_pp_mean_bubble_ratio gauge\n\
         mlxcel_pp_mean_bubble_ratio {:.6}",
        snap.mean_bubble_ratio
    );

    // --- Activation transfer latency ---
    let _ = writeln!(
        body,
        "# HELP mlxcel_pp_activation_latency_microseconds Activation transfer latency between adjacent stages\n\
         # TYPE mlxcel_pp_activation_latency_microseconds summary"
    );
    for p in &snap.activation_latency {
        let _ = writeln!(
            body,
            "mlxcel_pp_activation_latency_microseconds{{src_stage=\"{src}\",dst_stage=\"{dst}\",quantile=\"0.5\"}} {p50}\n\
             mlxcel_pp_activation_latency_microseconds{{src_stage=\"{src}\",dst_stage=\"{dst}\",quantile=\"0.95\"}} {p95}\n\
             mlxcel_pp_activation_latency_microseconds{{src_stage=\"{src}\",dst_stage=\"{dst}\",quantile=\"0.99\"}} {p99}\n\
             mlxcel_pp_activation_latency_microseconds_count{{src_stage=\"{src}\",dst_stage=\"{dst}\"}} {count}\n\
             mlxcel_pp_activation_latency_microseconds_max{{src_stage=\"{src}\",dst_stage=\"{dst}\"}} {max}",
            src = p.src_stage,
            dst = p.dst_stage,
            p50 = p.p50_us,
            p95 = p.p95_us,
            p99 = p.p99_us,
            count = p.count,
            max = p.max_us,
        );
    }

    // --- Admission rejection counters ---
    let _ = writeln!(
        body,
        "# HELP mlxcel_pp_admission_rejections_total Total KV cache admission rejections\n\
         # TYPE mlxcel_pp_admission_rejections_total counter"
    );
    for e in &snap.admission_rejections {
        let _ = writeln!(
            body,
            "mlxcel_pp_admission_rejections_total{{stage=\"{}\",reason=\"{}\"}} {}",
            e.stage_index, e.reason, e.count
        );
    }

    // --- Elastic repartition counters (emission path) ---
    let _ = writeln!(
        body,
        "# HELP mlxcel_pp_repartition_events_total Total elastic pipeline repartition events\n\
         # TYPE mlxcel_pp_repartition_events_total counter"
    );
    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = snap.repartition.counters.keys().collect();
    keys.sort();
    for key in keys {
        let count = snap.repartition.counters.get(key).copied().unwrap_or(0);
        // Key format is "<trigger>:<outcome>".
        let (trigger, outcome) = key.split_once(':').unwrap_or((key.as_str(), "unknown"));
        let _ = writeln!(
            body,
            "mlxcel_pp_repartition_events_total{{trigger=\"{trigger}\",outcome=\"{outcome}\"}} {count}"
        );
    }

    let _ = writeln!(
        body,
        "# HELP mlxcel_pp_repartition_drain_microseconds Drain duration histogram for completed repartitions\n\
         # TYPE mlxcel_pp_repartition_drain_microseconds summary\n\
         mlxcel_pp_repartition_drain_microseconds_sum {}\n\
         mlxcel_pp_repartition_drain_microseconds_count {}\n\
         mlxcel_pp_repartition_drain_microseconds_max {}",
        snap.repartition.drain_us_total,
        snap.repartition.drain_count,
        snap.repartition.drain_us_max
    );
    let _ = writeln!(
        body,
        "# HELP mlxcel_pp_repartition_total_microseconds Total repartition wall time across outcomes\n\
         # TYPE mlxcel_pp_repartition_total_microseconds summary\n\
         mlxcel_pp_repartition_total_microseconds_sum {}\n\
         mlxcel_pp_repartition_total_microseconds_count {}",
        snap.repartition.total_us_total, snap.repartition.total_count
    );
}

#[cfg(test)]
#[path = "metrics_llama_tests.rs"]
mod metrics_llama_tests;
