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

//! Read-only `/settings` values are user copy on the WebUI Server tab, so no
//! value may be Rust `Debug` output (issue #2329).

use std::time::Duration;

use axum::http::HeaderValue;
use serde_json::{Value, json};

use super::{KnobKind, read_only_kind, read_only_value, schema};
use crate::distributed::pipeline::RemotePipelineRuntimeConfig;
use crate::distributed::{EmbeddingMode, MoeShardMode, ShardConfig, TransportBackend};
use crate::memory_estimate::PagedBudgetDirective;
use crate::server::config::{
    DecodeStorageBackend, EmbeddingServingMode, PipelineParallelRuntimeConfig, PreemptionPolicy,
    RemotePipelineStageConfig,
};
use crate::server::gcp_compat::GcpRoutes;
use crate::server::prompt_cache::{ApcConfig, ApcHashAlgo, PromptCacheConfig};
use crate::server::{CorsPolicy, OriginPolicy, ServerConfig};
use mlxcel_core::cache::{BatchKvQuantConfig, KvQuantScheme};

/// The fields that used to be rendered with `format!("{value:?}")`.
const STRUCTURED_FIELDS: [&str; 12] = [
    "gcp",
    "embedding_serving_mode",
    "preemption_policy",
    "decode_storage_backend",
    "pipeline_parallel_runtime",
    "remote_pipeline_stage",
    "tensor_parallel",
    "prompt_cache",
    "batch_kv_quant",
    "kv_cache_budget",
    "cors_policy",
    "serving_mode",
];

/// Shared with `webui/src/features/settings/startup-values.test.tsx`, which
/// renders exactly these values on the Server tab.
const FIXTURE: &str = include_str!("../../tests/fixtures/settings/startup-values.json");

/// A bare CamelCase word such as `Hybrid` or `LongestFirst`: what `Debug`
/// prints for a unit enum variant. Upper-case acronyms (`GET`) do not match.
fn is_bare_variant(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|first| first.is_ascii_uppercase())
        && text.chars().all(|c| c.is_ascii_alphanumeric())
        && text.chars().any(|c| c.is_ascii_lowercase())
}

fn debug_shapes(value: &Value, path: &str, found: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if text.contains("Some(")
                || text.contains("None")
                || text.contains(" { ")
                || text.contains("{ ")
                || is_bare_variant(text)
            {
                found.push(format!("{path} = {text:?}"));
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                debug_shapes(item, &format!("{path}[{index}]"), found);
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                debug_shapes(item, &format!("{path}.{key}"), found);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn kind_admits(kind: KnobKind, value: &Value) -> bool {
    match kind {
        KnobKind::Str => value.is_string(),
        KnobKind::StrOrNull => value.is_string() || value.is_null(),
        KnobKind::Object => value.is_object(),
        KnobKind::ObjectOrNull => value.is_object() || value.is_null(),
        KnobKind::Bool => value.is_boolean(),
        KnobKind::Int => value.is_u64() || value.is_i64(),
        KnobKind::IntOrNull => value.is_u64() || value.is_i64() || value.is_null(),
        KnobKind::Float => value.is_number(),
        KnobKind::Array => value.is_array(),
    }
}

/// Every structured field set to a non-default, `Some` variant.
fn configured() -> ServerConfig {
    ServerConfig {
        gcp: Some(GcpRoutes {
            path_health: Some("/health".to_string()),
            path_predict: "/predict".to_string(),
        }),
        embedding_serving_mode: EmbeddingServingMode::EmbeddingOnly,
        preemption_policy: PreemptionPolicy::LowestPriority,
        decode_storage_backend: DecodeStorageBackend::Paged,
        pipeline_parallel_runtime: Some(PipelineParallelRuntimeConfig::InProcess {
            layers: "0-13,14-27".to_string(),
            micro_batch_size: 2,
        }),
        remote_pipeline_stage: Some(RemotePipelineStageConfig {
            bind_address: "0.0.0.0:9100".to_string(),
            stage_index: 1,
            num_stages: 2,
            upstream_peer: Some("10.0.0.1:9100".to_string()),
            downstream_peer: None,
            transport_backend: TransportBackend::Tcp,
        }),
        tensor_parallel: ShardConfig {
            tp_size: 2,
            moe_mode: MoeShardMode::ExpertParallel,
            embedding_mode: EmbeddingMode::Replicated,
            lm_head_mode: EmbeddingMode::VocabParallel,
        },
        prompt_cache: PromptCacheConfig {
            enabled: true,
            capacity_bytes: 1_073_741_824,
            capacity_bytes_explicit: true,
            max_entries: 64,
            ttl: Duration::from_secs(600),
            min_prefix_tokens: 32,
            apc: ApcConfig {
                enabled: false,
                block_size: 16,
                num_blocks: None,
                hash: ApcHashAlgo::Sha256,
            },
            snapshot_capacity_bytes: 268_435_456,
            snapshot_capacity_bytes_explicit: false,
            snapshot_max_entries: 8,
            snapshot_ttl: Duration::from_secs(300),
        },
        batch_kv_quant: BatchKvQuantConfig {
            scheme: KvQuantScheme::TurboQuant,
            bits: 4,
            group_size: 64,
            skip_last_layer: true,
        },
        kv_cache_budget: Some(PagedBudgetDirective::Auto),
        cors_policy: CorsPolicy {
            origins: OriginPolicy::AllowList(vec![HeaderValue::from_static(
                "http://localhost:5173",
            )]),
            methods: HeaderValue::from_static("GET, POST"),
            headers: HeaderValue::from_static("Authorization, Content-Type"),
            credentials: false,
        },
        serving_mode: crate::distributed::disaggregated::ServingMode::PrefillOnly,
        ..ServerConfig::default()
    }
}

fn assert_plain(config: &ServerConfig, label: &str) {
    let mut found = Vec::new();
    for field in STRUCTURED_FIELDS {
        let value = read_only_value(config, field);
        debug_shapes(&value, field, &mut found);
        let kind = read_only_kind(field);
        assert!(
            kind_admits(kind, &value),
            "{label}: {field} = {value} does not match its schema type {kind:?}"
        );
    }
    assert!(found.is_empty(), "{label}: Debug output in {found:#?}");
}

#[test]
fn structured_read_only_values_are_plain_json_for_default_and_configured_servers() {
    assert_plain(&ServerConfig::default(), "default");
    assert_plain(&configured(), "configured");

    let mut config = configured();
    config.pipeline_parallel_runtime = Some(PipelineParallelRuntimeConfig::RemoteCoordinator(
        RemotePipelineRuntimeConfig {
            stage_peers: vec!["10.0.0.2:9100".to_string()],
            transport_backend: TransportBackend::Rdma,
            bind_address: "0.0.0.0:9000".to_string(),
            stage_timeout: Duration::from_millis(1500),
        },
    ));
    config.kv_cache_budget = Some(PagedBudgetDirective::Bytes(4_294_967_296));
    config.cors_policy.origins = OriginPolicy::Literal(HeaderValue::from_static("https://a.test"));
    assert_plain(&config, "remote coordinator");
    assert_eq!(
        read_only_value(&config, "pipeline_parallel_runtime"),
        json!({
            "mode": "remote_coordinator",
            "stage_peers": ["10.0.0.2:9100"],
            "transport": "rdma",
            "bind_address": "0.0.0.0:9000",
            "stage_timeout_ms": 1500,
        })
    );
    assert_eq!(
        read_only_value(&config, "kv_cache_budget"),
        json!("4294967296")
    );

    config.kv_cache_budget = Some(PagedBudgetDirective::Disabled);
    config.cors_policy.origins = OriginPolicy::Localhost;
    assert_plain(&config, "disabled budget");
    assert_eq!(
        read_only_value(&config, "kv_cache_budget"),
        json!("disabled")
    );
}

#[test]
fn absent_optional_values_are_json_null() {
    let config = ServerConfig {
        gcp: None,
        pipeline_parallel_runtime: None,
        remote_pipeline_stage: None,
        kv_cache_budget: None,
        ..ServerConfig::default()
    };
    for field in [
        "gcp",
        "pipeline_parallel_runtime",
        "remote_pipeline_stage",
        "kv_cache_budget",
    ] {
        assert_eq!(read_only_value(&config, field), Value::Null, "{field}");
    }
}

#[test]
fn no_read_only_default_in_the_schema_carries_debug_output() {
    for config in [ServerConfig::default(), configured()] {
        let mut found = Vec::new();
        for spec in schema(&config).iter().filter(|spec| !spec.mutable) {
            debug_shapes(&spec.default, spec.name, &mut found);
        }
        assert!(found.is_empty(), "Debug output in {found:#?}");
    }
}

#[test]
fn shared_webui_fixture_matches_the_configured_server() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture must be JSON");
    let config = configured();
    let actual: Vec<Value> = schema(&config)
        .into_iter()
        .filter(|spec| STRUCTURED_FIELDS.contains(&spec.name))
        .map(|spec| {
            json!({
                "name": spec.name,
                "type": spec.kind,
                "default": spec.default,
            })
        })
        .collect();
    assert_eq!(
        fixture["values"],
        Value::Array(actual.clone()),
        "update tests/fixtures/settings/startup-values.json `values` to:\n{}",
        serde_json::to_string_pretty(&actual).expect("values serialize")
    );
}
