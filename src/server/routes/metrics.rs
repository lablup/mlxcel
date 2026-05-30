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

/// GET /metrics -- Prometheus text format
pub async fn metrics(State(state): State<AppState>) -> Response {
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

    // Batch observability counters
    let obs = state.batch_observability.snapshot();

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
         # HELP mlxcel_batch_prefill_chunks_total Total prefill chunks processed\n\
         # TYPE mlxcel_batch_prefill_chunks_total counter\n\
         mlxcel_batch_prefill_chunks_total {prefill_chunks}\n\
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
         # HELP mlxcel_cache_pool_paged_bytes_reserved Reserved bytes across paged sequences\n\
         # TYPE mlxcel_cache_pool_paged_bytes_reserved gauge\n\
         mlxcel_cache_pool_paged_bytes_reserved {paged_bytes_reserved}\n\
         # HELP mlxcel_cache_pool_paged_bytes_in_use Visible bytes in use across paged sequences\n\
         # TYPE mlxcel_cache_pool_paged_bytes_in_use gauge\n\
         mlxcel_cache_pool_paged_bytes_in_use {paged_bytes_in_use}\n\
         # HELP mlxcel_decode_storage_fallbacks_total Number of paged decode fallback events\n\
         # TYPE mlxcel_decode_storage_fallbacks_total counter\n\
         mlxcel_decode_storage_fallbacks_total {decode_storage_fallbacks}\n\
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
         mlxcel_prompt_cache_entries {pc_entries}\n",
        gen_time_sec = gen_time_ms as f64 / 1000.0,
        seq_started = obs.sequences_started,
        seq_completed = obs.sequences_completed,
        prefill_tokens = obs.total_prefill_tokens,
        decode_tokens = obs.total_decode_tokens,
        decode_steps = obs.decode_steps_processed,
        prefill_chunks = obs.prefill_chunks_processed,
        batch_size = obs.current_batch_size,
        cache_active = obs.cache_pool_active,
        paged_block_size = obs.cache_pool_paged_block_size,
        paged_blocks_allocated = obs.cache_pool_paged_blocks_allocated,
        paged_blocks_live = obs.cache_pool_paged_blocks_live,
        paged_blocks_free = obs.cache_pool_paged_blocks_free,
        paged_bytes_reserved = obs.cache_pool_paged_bytes_reserved,
        paged_bytes_in_use = obs.cache_pool_paged_bytes_in_use,
        decode_storage_fallbacks = obs.decode_storage_fallbacks,
        pc_hits = pc_hits,
        pc_misses = pc_misses,
        pc_reused_tokens = pc_reused_tokens,
        pc_evict_lru = pc_evict_lru,
        pc_evict_ttl = pc_evict_ttl,
        pc_evict_capacity = pc_evict_capacity,
        pc_bytes = pc_bytes,
        pc_entries = pc_entries,
    );

    let mut body = body;
    append_pipeline_metrics(&mut body, &pp_snapshot);

    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
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
