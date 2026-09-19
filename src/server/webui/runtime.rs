// Copyright 2025-2026 Lablup Inc.
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

//! Cheap, redacted runtime observation. No allocator/Metal calls, requests,
//! tokenization, cache lookup/touch, model initialization or scrape-baseline writes.
//! Atomics are observational samples, not a transaction across worker counters.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use crate::server::router_lifecycle::{
    MeasuredValue, RuntimeSlots, RuntimeSnapshot, SCHEMA_VERSION,
};
use crate::server::{AppState, ServerConfig};

pub(crate) fn runtime_snapshot(
    server_instance_id: String,
    model_id: String,
    revision: u64,
    snapshot_sequence: u64,
    config: &ServerConfig,
    state: Option<&AppState>,
) -> RuntimeSnapshot {
    let state = state.filter(|state| state.model_provider.is_loaded());
    let sampled_at = chrono::Utc::now().to_rfc3339();
    let mut measurements = BTreeMap::new();
    for (key, unit, scope, reason) in [
        (
            "gpu_utilization",
            "percent",
            "unknown",
            "not measured by mlxcel",
        ),
        (
            "ttft",
            "ms",
            "model",
            "request-send to first output token timing is not available in runtime counters",
        ),
        (
            "decode_rate",
            "tokens/s",
            "model",
            "no atomic interval sample; cumulative generation time is not request TTFT or instantaneous decode rate",
        ),
        (
            "process_resident_bytes",
            "bytes",
            "server",
            "process-wide resident memory has no existing cheap CPU snapshot",
        ),
        (
            "allocator_active_bytes",
            "bytes",
            "server",
            "process-wide allocator active memory has no existing cheap CPU snapshot",
        ),
        (
            "allocator_cache_bytes",
            "bytes",
            "server",
            "process-wide allocator cache memory has no existing cheap CPU snapshot",
        ),
        (
            "allocator_peak_bytes",
            "bytes",
            "server",
            "process-wide allocator peak memory has no existing cheap CPU snapshot",
        ),
        (
            "device_total_bytes",
            "bytes",
            "unknown",
            "device total memory has no existing cheap CPU snapshot; unified memory must not be summed with process memory",
        ),
        (
            "model_weights_bytes",
            "bytes",
            "model",
            "resident weights estimate is unavailable; checkpoint download bytes are not resident memory",
        ),
        (
            "kv_cache_bytes",
            "bytes",
            "model",
            "no cheap live KV byte snapshot; token occupancy cannot determine exact KV bytes",
        ),
    ] {
        measurements.insert(key.to_string(), unavailable(unit, scope, reason));
    }
    let reason = if !config.enable_metrics_endpoint {
        Some("metrics disabled; restart with --metrics")
    } else if state.is_none() {
        Some("model provider is not loaded; no live counters are available")
    } else {
        None
    };
    let values = state.map(|state| {
        let b = &state.batch_metrics;
        let m = &state.metrics;
        let o = &state.batch_observability;
        [
            b.active_count.load(Ordering::Relaxed) as u64,
            b.queue_depth.load(Ordering::Relaxed) as u64,
            m.requests_total.load(Ordering::Relaxed),
            m.completion_tokens_total.load(Ordering::Relaxed),
            m.generation_time_ms_total.load(Ordering::Relaxed),
            o.llama_predicted_tokens.load(Ordering::Relaxed),
            o.llama_predicted_us_total.load(Ordering::Relaxed),
            b.prompt_cache_bytes.load(Ordering::Relaxed),
            b.prompt_cache_entries.load(Ordering::Relaxed),
        ]
    });
    for (index, (key, unit)) in [
        ("active_requests", "requests"),
        ("queued_requests", "requests"),
        ("completed_requests_total", "requests"),
        ("completion_tokens_total", "tokens"),
        ("generation_time_ms_total", "ms"),
        ("decode_tokens_total", "tokens"),
        ("decode_time_us_total", "us"),
        ("prompt_cache_bytes", "bytes"),
        ("prompt_cache_entries", "entries"),
    ]
    .iter()
    .enumerate()
    {
        let cache_disabled = key.starts_with("prompt_cache_")
            && state.is_some_and(|state| state.model_provider.prompt_cache().is_none());
        let unsupported_single = state
            .is_some_and(|state| state.model_provider.uses_single_stream_observation())
            && matches!(
                *key,
                "active_requests" | "decode_tokens_total" | "decode_time_us_total"
            );
        let no_decode_sample = matches!(*key, "decode_tokens_total" | "decode_time_us_total")
            && values
                .as_ref()
                .is_none_or(|values| values[5] == 0 && values[6] == 0);
        let missing = reason
            .or(no_decode_sample
                .then_some("no completed decode timing sample has been published by this provider"))
            .or(unsupported_single
                .then_some("single-stream provider does not publish this batch counter"))
            .or(cache_disabled.then_some("prompt cache is disabled for this provider"));
        let measurement = match (missing, values.as_ref()) {
            (None, Some(values)) => MeasuredValue {
                value: Some(values[index] as f64),
                unit: (*unit).to_string(),
                scope: "model".to_string(),
                measured_at: Some(sampled_at.clone()),
                reason: Some(counter_source(key).to_string()),
            },
            _ => unavailable(unit, "model", missing.unwrap_or("counter unavailable")),
        };
        measurements.insert((*key).to_string(), measurement);
    }
    RuntimeSnapshot {
        schema_version: SCHEMA_VERSION.to_string(),
        server_instance_id,
        model_id,
        revision: revision.max(1),
        snapshot_sequence,
        measurements,
        slots: slots_snapshot(config, state, sampled_at),
        settings: super::api::settings_report(config),
    }
}

fn counter_source(key: &str) -> &'static str {
    match key {
        "active_requests" => {
            "BatchMetrics active decode sequences; point-in-time sample, excludes queued prefill"
        }
        "queued_requests" => {
            "BatchMetrics pending prefill / single-stream queue; point-in-time sample"
        }
        "completed_requests_total" => {
            "Metrics successfully recorded route completions, streamed and non-streamed chat alike, excluding chat streams the client cancelled; loaded-provider lifetime, not all admissions or errors"
        }
        "completion_tokens_total" => {
            "Metrics completion tokens from recorded route results, streamed and non-streamed chat alike, excluding chat streams the client cancelled; loaded-provider lifetime"
        }
        "generation_time_ms_total" => {
            "Metrics sum of route-reported generation/elapsed times; loaded-provider lifetime, not decode-only or TTFT"
        }
        "decode_tokens_total" => {
            "BatchObservability llama_predicted_tokens from completed request results; loaded-provider lifetime"
        }
        "decode_time_us_total" => {
            "BatchObservability llama_predicted_us_total from completed request generation-only times; loaded-provider lifetime"
        }
        _ => {
            "BatchMetrics prompt-cache gauge last published by scheduler; may lag eviction, not live GPU KV allocation"
        }
    }
}

fn unavailable(unit: &str, scope: &str, reason: &str) -> MeasuredValue {
    MeasuredValue {
        value: None,
        unit: unit.to_string(),
        scope: scope.to_string(),
        measured_at: None,
        reason: Some(reason.to_string()),
    }
}

fn slots_snapshot(
    config: &ServerConfig,
    state: Option<&AppState>,
    sampled_at: String,
) -> RuntimeSlots {
    let mut result = RuntimeSlots {
        available: false,
        reason: Some(
            if config.enable_slots_endpoint {
                "model provider is not loaded; slots are unavailable"
            } else {
                "slots disabled; restart with --slots"
            }
            .to_string(),
        ),
        measured_at: None,
        configured_parallelism: config.n_parallel,
        effective_parallelism: None,
        request_context_tokens: None,
        shared_pool_context_tokens: None,
        items: Vec::new(),
    };
    if !config.enable_slots_endpoint {
        return result;
    }
    let Some(state) = state else {
        return result;
    };
    let Some(items) = state.slots.runtime_snapshot() else {
        result.reason = Some("slot registry is unavailable".to_string());
        return result;
    };
    result.available = true;
    result.reason = (state.slots.total() > 256)
        .then(|| "showing the first 256 observational slots".to_string());
    result.measured_at = Some(sampled_at);
    result.request_context_tokens = nonzero(state.effective_context_size());
    result.shared_pool_context_tokens = config
        .kv_unified
        .then_some(config.context_size_total)
        .and_then(nonzero);
    // #1815 release-publishes this only when the worker clamps to non-batching.
    // Configured max_batch_size alone is NOT a proven post-load decode width.
    result.effective_parallelism = state.batch_metrics.runtime_context_size().map(|_| 1);
    result.items = items;
    result
}

fn nonzero(value: usize) -> Option<usize> {
    (value > 0).then_some(value)
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
