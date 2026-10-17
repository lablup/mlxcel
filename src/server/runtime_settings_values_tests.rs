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
//! value may be Rust `Debug` output (issue #2329), and every enum spelling
//! needs a catalog label there (issue #2419).

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use axum::http::HeaderValue;
use serde_json::{Value, json};

use super::{KnobKind, read_only_kind, read_only_value, schema};
use crate::distributed::disaggregated::ServingMode;
use crate::distributed::pipeline::RemotePipelineRuntimeConfig;
use crate::distributed::{EmbeddingMode, MoeShardMode, ShardConfig, TransportBackend};
use crate::memory_estimate::PagedBudgetDirective;
use crate::server::config::{
    DecodeStorageBackend, EmbeddingServingMode, PipelineParallelRuntimeConfig, PreemptionPolicy,
    ReasoningAliasField, RemotePipelineStageConfig,
};
use crate::server::gcp_compat::GcpRoutes;
use crate::server::prompt_cache::{ApcConfig, ApcHashAlgo, PromptCacheConfig};
use crate::server::{CorsPolicy, OriginPolicy, ReasoningFormat, ServerConfig};
use mlxcel_core::cache::{BatchKvQuantConfig, KVCacheMode, KvQuantScheme};

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

/// Every read-only field and its sub-fields, shared with
/// `webui/src/features/settings/startup-labels.test.ts`, which fails when one
/// of them has no catalog label (issue #2334).
const READ_ONLY_FIELDS: &str = include_str!("../../tests/fixtures/settings/read-only-fields.json");

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

/// The configured server as a remote pipeline coordinator: the other
/// `pipeline_parallel_runtime` shape.
fn remote_coordinator() -> ServerConfig {
    let mut config = configured();
    config.pipeline_parallel_runtime = Some(PipelineParallelRuntimeConfig::RemoteCoordinator(
        RemotePipelineRuntimeConfig {
            stage_peers: vec!["10.0.0.2:9100".to_string()],
            transport_backend: TransportBackend::Rdma,
            bind_address: "0.0.0.0:9000".to_string(),
            stage_timeout: Duration::from_millis(1500),
        },
    ));
    config
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

    let mut config = remote_coordinator();
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

/// The dotted key path of every object member in `value`, looking through
/// lists, so a nested structure contributes `outer.inner`.
fn sub_field_paths(value: &Value, prefix: &str, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, item) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                sub_field_paths(item, &path, out);
                out.insert(path);
            }
        }
        Value::Array(items) => {
            for item in items {
                sub_field_paths(item, prefix, out);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

#[test]
fn read_only_field_fixture_lists_every_field_and_sub_field() {
    let configs = [ServerConfig::default(), configured(), remote_coordinator()];
    let mut fields: Vec<(&'static str, BTreeSet<String>)> = schema(&configs[0])
        .into_iter()
        .filter(|spec| !spec.mutable)
        .map(|spec| (spec.name, BTreeSet::new()))
        .collect();
    for config in &configs {
        for spec in schema(config).into_iter().filter(|spec| !spec.mutable) {
            let (_, sub_fields) = fields
                .iter_mut()
                .find(|(name, _)| *name == spec.name)
                .expect("every server config has the same read-only fields");
            sub_field_paths(&spec.default, "", sub_fields);
        }
    }
    let actual: Vec<Value> = fields
        .into_iter()
        .map(|(name, sub_fields)| json!({"name": name, "sub_fields": sub_fields}))
        .collect();
    let fixture: Value = serde_json::from_str(READ_ONLY_FIELDS).expect("fixture must be JSON");
    assert_eq!(
        fixture["fields"],
        Value::Array(actual.clone()),
        "update tests/fixtures/settings/read-only-fields.json `fields` to the list below, then give each new entry a label in webui/src/features/settings/startup-strings.ts:\n{}",
        serde_json::to_string_pretty(&actual).expect("fields serialize")
    );
}

/// Every variant of a unit-only enum, listed by hand. The `match` has no
/// wildcard arm, so a new variant fails to compile here until it is listed,
/// and then fails `read_only_enum_fixture_lists_every_spelling` until the
/// fixture and the WebUI catalog carry its spelling (issue #2419).
macro_rules! every_variant {
    ($ty:ty { $($variant:path),+ $(,)? }) => {{
        let all: Vec<$ty> = vec![$($variant),+];
        for value in &all {
            match value {
                $($variant)|+ => {}
            }
        }
        all
    }};
}

/// Read-only values, by dotted path, whose strings are enum spellings rather
/// than data such as a model name, an address or an origin. A new enum-valued
/// field or sub-field belongs here.
const ENUM_PATHS: [&str; 17] = [
    "reasoning_format",
    "reasoning_alias_field",
    "embedding_serving_mode",
    "preemption_policy",
    "decode_storage_backend",
    "kv_cache_mode",
    "serving_mode",
    "kv_cache_budget",
    "pipeline_parallel_runtime.mode",
    "pipeline_parallel_runtime.transport",
    "remote_pipeline_stage.transport",
    "tensor_parallel.moe_mode",
    "tensor_parallel.embedding_mode",
    "tensor_parallel.lm_head_mode",
    "prompt_cache.apc_hash",
    "batch_kv_quant.scheme",
    "cors_policy.origin_policy",
];

/// Server configs that, between them, set every variant of every enum a
/// read-only value spells: config `i` takes variant `i` of each list, wrapping.
fn every_enum_variant_configs() -> Vec<ServerConfig> {
    let reasoning_formats = every_variant!(ReasoningFormat {
        ReasoningFormat::Auto,
        ReasoningFormat::None,
        ReasoningFormat::DeepSeek,
        ReasoningFormat::DeepSeekLegacy,
    });
    let alias_fields = every_variant!(ReasoningAliasField {
        ReasoningAliasField::None,
        ReasoningAliasField::Reasoning,
    });
    let embedding_serving = every_variant!(EmbeddingServingMode {
        EmbeddingServingMode::Any,
        EmbeddingServingMode::EmbeddingOnly,
        EmbeddingServingMode::RerankOnly,
    });
    let preemption = every_variant!(PreemptionPolicy {
        PreemptionPolicy::LongestFirst,
        PreemptionPolicy::LowestPriority,
    });
    let decode_storage = every_variant!(DecodeStorageBackend {
        DecodeStorageBackend::Auto,
        DecodeStorageBackend::Dense,
        DecodeStorageBackend::Paged,
    });
    let kv_cache_modes = every_variant!(KVCacheMode {
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo3Asym,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Delegated,
    });
    let serving_modes = every_variant!(ServingMode {
        ServingMode::Hybrid,
        ServingMode::PrefillOnly,
        ServingMode::DecodeOnly,
        ServingMode::Router,
    });
    let transports = every_variant!(TransportBackend {
        TransportBackend::Tcp,
        TransportBackend::Thunderbolt,
        TransportBackend::Rdma,
    });
    let moe_modes = every_variant!(MoeShardMode {
        MoeShardMode::ExpertParallel,
        MoeShardMode::WithinExpert,
    });
    let embedding_modes = every_variant!(EmbeddingMode {
        EmbeddingMode::VocabParallel,
        EmbeddingMode::Replicated,
    });
    let apc_hashes = every_variant!(ApcHashAlgo {
        ApcHashAlgo::Sha256,
        ApcHashAlgo::Blake3,
    });
    let kv_quant_schemes = every_variant!(KvQuantScheme {
        KvQuantScheme::Uniform,
        KvQuantScheme::TurboQuant,
    });
    let budgets = [
        PagedBudgetDirective::Auto,
        PagedBudgetDirective::Disabled,
        PagedBudgetDirective::Bytes(4_294_967_296),
    ];
    for budget in &budgets {
        match budget {
            PagedBudgetDirective::Auto
            | PagedBudgetDirective::Disabled
            | PagedBudgetDirective::Bytes(_) => {}
        }
    }
    let origins = [
        OriginPolicy::Wildcard,
        OriginPolicy::Localhost,
        OriginPolicy::Literal(HeaderValue::from_static("https://a.test")),
        OriginPolicy::AllowList(vec![HeaderValue::from_static("https://b.test")]),
    ];
    for origin in &origins {
        match origin {
            OriginPolicy::Wildcard
            | OriginPolicy::Localhost
            | OriginPolicy::Literal(_)
            | OriginPolicy::AllowList(_) => {}
        }
    }
    let in_process = configured()
        .pipeline_parallel_runtime
        .expect("configured() sets an in-process pipeline runtime");
    let coordinator = remote_coordinator()
        .pipeline_parallel_runtime
        .expect("remote_coordinator() sets a remote pipeline runtime");
    let runtimes = [in_process, coordinator];
    for runtime in &runtimes {
        match runtime {
            PipelineParallelRuntimeConfig::InProcess { .. }
            | PipelineParallelRuntimeConfig::RemoteCoordinator(_) => {}
        }
    }

    let rows = [
        reasoning_formats.len(),
        kv_cache_modes.len(),
        serving_modes.len(),
        transports.len(),
        origins.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(1);
    let pick = |len: usize, row: usize| row % len;
    (0..rows)
        .map(|row| {
            let mut config = configured();
            config.reasoning_format = reasoning_formats[pick(reasoning_formats.len(), row)];
            config.reasoning_alias_field = alias_fields[pick(alias_fields.len(), row)];
            config.embedding_serving_mode = embedding_serving[pick(embedding_serving.len(), row)];
            config.preemption_policy = preemption[pick(preemption.len(), row)];
            config.decode_storage_backend = decode_storage[pick(decode_storage.len(), row)];
            config.kv_cache_mode = kv_cache_modes[pick(kv_cache_modes.len(), row)];
            config.serving_mode = serving_modes[pick(serving_modes.len(), row)];
            config.kv_cache_budget = Some(budgets[pick(budgets.len(), row)]);
            let transport = transports[pick(transports.len(), row)];
            let mut runtime = runtimes[pick(runtimes.len(), row)].clone();
            if let PipelineParallelRuntimeConfig::RemoteCoordinator(remote) = &mut runtime {
                remote.transport_backend = transport;
            }
            config.pipeline_parallel_runtime = Some(runtime);
            if let Some(stage) = config.remote_pipeline_stage.as_mut() {
                stage.transport_backend = transport;
            }
            config.tensor_parallel.moe_mode = moe_modes[pick(moe_modes.len(), row)];
            config.tensor_parallel.embedding_mode =
                embedding_modes[pick(embedding_modes.len(), row)];
            config.tensor_parallel.lm_head_mode =
                embedding_modes[pick(embedding_modes.len(), row + 1)];
            config.prompt_cache.apc.hash = apc_hashes[pick(apc_hashes.len(), row)];
            config.batch_kv_quant.scheme = kv_quant_schemes[pick(kv_quant_schemes.len(), row)];
            config.cors_policy.origins = origins[pick(origins.len(), row)].clone();
            config
        })
        .collect()
}

#[test]
fn read_only_enum_fixture_lists_every_spelling() {
    let mut spellings: BTreeMap<&str, BTreeSet<String>> = ENUM_PATHS
        .iter()
        .map(|path| (*path, BTreeSet::new()))
        .collect();
    for config in every_enum_variant_configs() {
        for path in ENUM_PATHS {
            let mut segments = path.split('.');
            let field = segments.next().expect("a path names its field");
            let mut value = read_only_value(&config, field);
            for segment in segments {
                value = value.get(segment).cloned().unwrap_or(Value::Null);
            }
            // A decimal count (the KV cache budget in bytes) is data, not a spelling.
            if let Value::String(text) = value
                && !text.bytes().all(|byte| byte.is_ascii_digit())
            {
                spellings
                    .get_mut(path)
                    .expect("every path is listed")
                    .insert(text);
            }
        }
    }
    let empty: Vec<&str> = spellings
        .iter()
        .filter(|(_, values)| values.is_empty())
        .map(|(path, _)| *path)
        .collect();
    assert!(empty.is_empty(), "no spelling observed at {empty:?}");
    let actual = json!(spellings);
    let fixture: Value = serde_json::from_str(READ_ONLY_FIELDS).expect("fixture must be JSON");
    assert_eq!(
        fixture["enum_values"],
        actual,
        "update tests/fixtures/settings/read-only-fields.json `enum_values` to the map below, then give each new spelling a label (`settings.startup.<path>.values.<spelling>`) in webui/src/features/settings/startup-strings.ts:\n{}",
        serde_json::to_string_pretty(&actual).expect("spellings serialize")
    );
}
