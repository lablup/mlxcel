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

//! JSON projections of the structured, read-only `ServerConfig` values that
//! `/settings` reports (issue #2329).
//!
//! These values reach the WebUI Settings > Server tab as user copy, so they
//! must never be Rust `Debug` output: an absent value is JSON `null`, an enum
//! is its stable CLI or wire spelling, and a struct is a flat JSON object of
//! plain scalars and string lists.

use axum::http::HeaderValue;
use serde_json::{Value, json};

use crate::distributed::ShardConfig;
use crate::memory_estimate::PagedBudgetDirective;
use crate::server::config::{PipelineParallelRuntimeConfig, RemotePipelineStageConfig};
use crate::server::gcp_compat::GcpRoutes;
use crate::server::prompt_cache::PromptCacheConfig;
use crate::server::{CorsPolicy, OriginPolicy};
use mlxcel_core::cache::BatchKvQuantConfig;

fn header_text(value: &HeaderValue) -> String {
    String::from_utf8_lossy(value.as_bytes()).into_owned()
}

/// `null` when the Vertex AI adapter is off, otherwise its two routes.
pub(super) fn gcp(routes: Option<&GcpRoutes>) -> Value {
    routes.map_or(Value::Null, |routes| {
        json!({
            "health_route": routes.path_health,
            "predict_route": routes.path_predict,
        })
    })
}

/// `null` without pipeline parallelism, otherwise the runtime and its geometry.
pub(super) fn pipeline_parallel_runtime(runtime: Option<&PipelineParallelRuntimeConfig>) -> Value {
    match runtime {
        None => Value::Null,
        Some(PipelineParallelRuntimeConfig::InProcess {
            layers,
            micro_batch_size,
        }) => json!({
            "mode": "in_process",
            "layers": layers,
            "micro_batch_size": micro_batch_size,
        }),
        Some(PipelineParallelRuntimeConfig::RemoteCoordinator(config)) => json!({
            "mode": "remote_coordinator",
            "stage_peers": config.stage_peers,
            "transport": config.transport_backend.to_string(),
            "bind_address": config.bind_address,
            "stage_timeout_ms": u64::try_from(config.stage_timeout.as_millis()).unwrap_or(u64::MAX),
        }),
    }
}

/// `null` unless this process runs as a remote pipeline stage.
pub(super) fn remote_pipeline_stage(stage: Option<&RemotePipelineStageConfig>) -> Value {
    stage.map_or(Value::Null, |stage| {
        json!({
            "bind_address": stage.bind_address,
            "stage_index": stage.stage_index,
            "num_stages": stage.num_stages,
            "upstream_peer": stage.upstream_peer,
            "downstream_peer": stage.downstream_peer,
            "transport": stage.transport_backend.to_string(),
        })
    })
}

pub(super) fn tensor_parallel(shard: &ShardConfig) -> Value {
    json!({
        "tp_size": shard.tp_size,
        "moe_mode": shard.moe_mode.to_string(),
        "embedding_mode": shard.embedding_mode.to_string(),
        "lm_head_mode": shard.lm_head_mode.to_string(),
    })
}

/// The prompt-prefix cache budget, flattened so every entry is one plain value.
pub(super) fn prompt_cache(config: &PromptCacheConfig) -> Value {
    json!({
        "enabled": config.enabled,
        "capacity_bytes": config.capacity_bytes,
        "max_entries": config.max_entries,
        "ttl_secs": config.ttl.as_secs(),
        "min_prefix_tokens": config.min_prefix_tokens,
        "apc_enabled": config.apc.enabled,
        "apc_block_size": config.apc.block_size,
        "apc_num_blocks": config.apc.num_blocks,
        "apc_hash": config.apc.hash.as_str(),
        "snapshot_capacity_bytes": config.snapshot_capacity_bytes,
        "snapshot_max_entries": config.snapshot_max_entries,
        "snapshot_ttl_secs": config.snapshot_ttl.as_secs(),
    })
}

pub(super) fn batch_kv_quant(config: &BatchKvQuantConfig) -> Value {
    json!({
        "scheme": config.scheme.to_string(),
        "bits": config.bits,
        "group_size": config.group_size,
        "skip_last_layer": config.skip_last_layer,
    })
}

/// The `--kv-cache-budget` spelling that reproduces the directive: `auto`,
/// `disabled`, or a raw byte count. `null` when the flag was absent.
pub(super) fn kv_cache_budget(directive: Option<PagedBudgetDirective>) -> Value {
    match directive {
        None => Value::Null,
        Some(PagedBudgetDirective::Auto) => json!("auto"),
        Some(PagedBudgetDirective::Disabled) => json!("disabled"),
        Some(PagedBudgetDirective::Bytes(bytes)) => json!(bytes.to_string()),
    }
}

pub(super) fn cors_policy(policy: &CorsPolicy) -> Value {
    let (origin_policy, origins) = match &policy.origins {
        OriginPolicy::Wildcard => ("wildcard", vec!["*".to_string()]),
        OriginPolicy::Localhost => ("localhost", vec!["localhost".to_string()]),
        OriginPolicy::Literal(value) => ("literal", vec![header_text(value)]),
        OriginPolicy::AllowList(values) => ("allow_list", values.iter().map(header_text).collect()),
    };
    json!({
        "origin_policy": origin_policy,
        "origins": origins,
        "methods": header_text(&policy.methods),
        "headers": header_text(&policy.headers),
        "credentials": policy.credentials,
    })
}
