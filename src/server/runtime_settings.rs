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

//! Typed schema, validation, updates, and fingerprints for live server
//! settings (issue #1312).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::{LiveSettings, ServerConfig};
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::server::thinking_budget::ThinkingBudget;

const WORKER_REASON: &str = "owned by the model worker; restart required";
const SCHEDULER_REASON: &str =
    "scheduler or channel geometry is fixed at startup; restart required";
const ROUTE_REASON: &str =
    "route topology or access policy is installed at startup; restart required";
const DISTRIBUTED_REASON: &str =
    "distributed or pipeline topology is fixed at startup; restart required";
const MODEL_REASON: &str = "fixed for the loaded model provider; restart required";
const INERT_REASON: &str = "accepted for compatibility but not applied by the current runtime";
const UNSUPPORTED_LIVE_REASON: &str = "outside the supported live-update set; restart required";

/// Every field declared on ServerConfig, in source order.
///
/// decode_timeout_seconds is represented on the management API as the
/// shorter compatibility name timeout_seconds. The source-level coverage
/// test compares this list with the struct declaration so a future field
/// cannot silently miss the settings schema.
pub const CLASSIFIED_SERVER_CONFIG_FIELDS: &[&str] = &[
    "gcp",
    "reasoning_format",
    "reasoning_alias_field",
    "skip_chat_parsing",
    "no_prefill_assistant",
    "reasoning_budget_message",
    "api_keys",
    "decode_timeout_seconds",
    "api_prefix",
    "sse_ping_interval",
    "model_alias",
    "model_aliases",
    "context_size",
    "n_parallel",
    "enable_slots_endpoint",
    "enable_props_endpoint",
    "enable_metrics_endpoint",
    "enable_settings_endpoint",
    "slot_save_path",
    "model_tags",
    "lora_adapters",
    "lora_runtime",
    "embd_normalize",
    "embedding_serving_mode",
    "spm_infill",
    "default_temperature",
    "default_top_p",
    "default_top_k",
    "default_min_p",
    "default_typical_p",
    "default_top_n_sigma",
    "default_xtc_probability",
    "default_xtc_threshold",
    "default_ignore_eos",
    "default_stop_sequences",
    "default_repetition_penalty",
    "default_repetition_context_size",
    "default_max_tokens",
    "default_seed",
    "default_frequency_penalty",
    "default_presence_penalty",
    "default_dry_multiplier",
    "default_dry_base",
    "default_dry_allowed_length",
    "default_dry_penalty_last_n",
    "default_dry_sequence_breakers",
    "default_grammar",
    "default_mirostat",
    "default_mirostat_tau",
    "default_mirostat_eta",
    "default_dynatemp_range",
    "default_dynatemp_exponent",
    "default_adaptive_target",
    "default_adaptive_decay",
    "default_adaptive_p_named",
    "default_logit_bias",
    "draft_model_path",
    "num_draft_tokens",
    "draft_kind",
    "draft_block_size",
    "max_batch_size",
    "max_queue_depth",
    "audio_queue_depth",
    "audio_request_timeout_secs",
    "embedding_model_path",
    "embedding_batch_size",
    "embedding_max_length",
    "embedding_queue_depth",
    "embedding_request_timeout_secs",
    "reranker_model_path",
    "rerank_batch_size",
    "prefill_chunk_size",
    "prefill_grant_interval",
    "enable_preemption",
    "preemption_policy",
    "no_batch",
    "max_batch_prefill",
    "max_batch_prefill_tokens",
    "decode_storage_backend",
    "pipeline_parallel_runtime",
    "remote_pipeline_stage",
    "tensor_parallel",
    "vision_cache_size",
    "lang_bias_config",
    "reasoning_budget",
    "chat_template_kwargs",
    "prompt_cache",
    "kv_cache_mode",
    "batch_kv_quant",
    "max_kv_size",
    "context_shift",
    "n_keep",
    "kv_cache_budget",
    "enable_vlm_prefix_cache",
    "cors_policy",
    "serving_mode",
    "prefill_peers",
    "decode_peers",
    "serving_bind",
    "max_denoising_steps",
    "diffusion_sampler",
    "diffusion_threshold",
    "loop_detection",
    "model_is_gemma4_family",
];

/// JSON type exposed by a schema entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnobKind {
    Bool,
    Int,
    IntOrNull,
    Float,
    Str,
    StrOrNull,
    Array,
    Object,
    ObjectOrNull,
}

/// One typed runtime-setting schema entry.
#[derive(Debug, Clone, Serialize)]
pub struct KnobSpec {
    pub name: &'static str,
    #[serde(rename = "type")]
    pub kind: KnobKind,
    pub default: Value,
    pub mutable: bool,
    pub allowed: Option<Vec<&'static str>>,
    pub help: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}

/// PATCH operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    #[default]
    Merge,
    Replace,
}

/// One rejected setting from a partially successful PATCH.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Rejected {
    pub name: String,
    pub reason: String,
}

/// Pure result of validating and applying a PATCH candidate.
#[derive(Debug)]
pub struct ApplyResult {
    pub next: LiveSettings,
    pub applied: Map<String, Value>,
    pub rejected: Vec<Rejected>,
}

/// Management API name corresponding to a ServerConfig field.
#[must_use]
pub fn api_name(config_field: &'static str) -> &'static str {
    if config_field == "decode_timeout_seconds" {
        "timeout_seconds"
    } else {
        config_field
    }
}

/// Whether a management API name is live-mutable.
#[must_use]
pub fn is_mutable(name: &str) -> bool {
    matches!(
        name,
        "timeout_seconds"
            | "default_temperature"
            | "default_top_p"
            | "default_top_k"
            | "default_min_p"
            | "default_repetition_penalty"
            | "default_repetition_context_size"
            | "default_max_tokens"
            | "default_seed"
            | "default_frequency_penalty"
            | "default_presence_penalty"
            | "default_dry_multiplier"
            | "default_dry_base"
            | "default_dry_allowed_length"
            | "default_dry_penalty_last_n"
            | "default_dry_sequence_breakers"
            | "lang_bias_config"
            | "reasoning_budget"
            | "chat_template_kwargs"
            | "loop_detection"
            | "max_denoising_steps"
            | "diffusion_sampler"
            | "diffusion_threshold"
    )
}

fn reasoning_budget_value(value: Option<ThinkingBudget>) -> Value {
    let raw = match value {
        None => -1,
        Some(ThinkingBudget::ImmediateClose) => 0,
        Some(ThinkingBudget::Limited(limit)) => i64::from(limit.get()),
    };
    json!(raw)
}

/// Canonical JSON map of the mutable settings set.
#[must_use]
pub fn mutable_values(live: &LiveSettings) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("timeout_seconds".to_string(), json!(live.timeout_seconds));
    out.insert(
        "default_temperature".to_string(),
        json!(live.default_temperature),
    );
    out.insert("default_top_p".to_string(), json!(live.default_top_p));
    out.insert("default_top_k".to_string(), json!(live.default_top_k));
    out.insert("default_min_p".to_string(), json!(live.default_min_p));
    out.insert(
        "default_repetition_penalty".to_string(),
        json!(live.default_repetition_penalty),
    );
    out.insert(
        "default_repetition_context_size".to_string(),
        json!(live.default_repetition_context_size),
    );
    out.insert(
        "default_max_tokens".to_string(),
        json!(live.default_max_tokens),
    );
    out.insert("default_seed".to_string(), json!(live.default_seed));
    out.insert(
        "default_frequency_penalty".to_string(),
        json!(live.default_frequency_penalty),
    );
    out.insert(
        "default_presence_penalty".to_string(),
        json!(live.default_presence_penalty),
    );
    out.insert(
        "default_dry_multiplier".to_string(),
        json!(live.default_dry_multiplier),
    );
    out.insert("default_dry_base".to_string(), json!(live.default_dry_base));
    out.insert(
        "default_dry_allowed_length".to_string(),
        json!(live.default_dry_allowed_length),
    );
    out.insert(
        "default_dry_penalty_last_n".to_string(),
        json!(live.default_dry_penalty_last_n),
    );
    out.insert(
        "default_dry_sequence_breakers".to_string(),
        json!(live.default_dry_sequence_breakers),
    );
    out.insert(
        "lang_bias_config".to_string(),
        live.lang_bias_config
            .as_ref()
            .and_then(|value| serde_json::to_value(value).ok())
            .unwrap_or(Value::Null),
    );
    out.insert(
        "reasoning_budget".to_string(),
        reasoning_budget_value(live.reasoning_budget),
    );
    out.insert(
        "chat_template_kwargs".to_string(),
        live.chat_template_kwargs
            .as_ref()
            .map(|value| Value::Object(value.as_map().clone()))
            .unwrap_or_else(|| Value::Object(Map::new())),
    );
    out.insert(
        "loop_detection".to_string(),
        live.loop_detection
            .as_ref()
            .and_then(|value| serde_json::to_value(value).ok())
            .unwrap_or(Value::Null),
    );
    out.insert(
        "max_denoising_steps".to_string(),
        json!(live.max_denoising_steps),
    );
    out.insert(
        "diffusion_sampler".to_string(),
        json!(live.diffusion_sampler),
    );
    out.insert(
        "diffusion_threshold".to_string(),
        json!(live.diffusion_threshold),
    );
    out
}

fn mutable_metadata(name: &'static str) -> (KnobKind, Option<Vec<&'static str>>, &'static str) {
    match name {
        "timeout_seconds" => (
            KnobKind::Int,
            None,
            "Decode watchdog used by newly admitted requests.",
        ),
        "default_temperature" => (
            KnobKind::Float,
            None,
            "Sampling temperature used when the request omits temperature.",
        ),
        "default_top_p" => (KnobKind::Float, None, "Top-p sampling default in (0, 1]."),
        "default_top_k" => (KnobKind::Int, None, "Top-k sampling default."),
        "default_min_p" => (KnobKind::Float, None, "Min-p sampling default in [0, 1]."),
        "default_repetition_penalty" => (
            KnobKind::Float,
            None,
            "Repetition penalty used when a request omits it.",
        ),
        "default_repetition_context_size" => (
            KnobKind::Int,
            None,
            "Token history window used by repetition penalties.",
        ),
        "default_max_tokens" => (
            KnobKind::Int,
            None,
            "Generation budget used when the request omits max_tokens.",
        ),
        "default_seed" => (
            KnobKind::IntOrNull,
            None,
            "Random seed used when a request omits one; null is random.",
        ),
        "default_frequency_penalty" => (
            KnobKind::Float,
            None,
            "Frequency penalty used when a request omits it.",
        ),
        "default_presence_penalty" => (
            KnobKind::Float,
            None,
            "Presence penalty used when a request omits it.",
        ),
        "default_dry_multiplier" => (KnobKind::Float, None, "DRY repetition multiplier."),
        "default_dry_base" => (KnobKind::Float, None, "DRY exponential base."),
        "default_dry_allowed_length" => (KnobKind::Int, None, "DRY allowed repeat length."),
        "default_dry_penalty_last_n" => (KnobKind::Int, None, "DRY history window."),
        "default_dry_sequence_breakers" => (
            KnobKind::Array,
            None,
            "b10621 DRY sequence-breaker strings; breaker token data is derived from them per request.",
        ),
        "lang_bias_config" => (
            KnobKind::ObjectOrNull,
            None,
            "Language-bias policy for newly admitted requests.",
        ),
        "reasoning_budget" => (
            KnobKind::Int,
            None,
            "-1 is unbounded, 0 closes thinking immediately, and N caps reasoning tokens.",
        ),
        "chat_template_kwargs" => (
            KnobKind::Object,
            None,
            "Server template kwargs merged under per-request kwargs.",
        ),
        "loop_detection" => (
            KnobKind::ObjectOrNull,
            None,
            "Global N-gram loop-detection override.",
        ),
        "max_denoising_steps" => (
            KnobKind::IntOrNull,
            None,
            "Optional diffusion denoising-step cap.",
        ),
        "diffusion_sampler" => (
            KnobKind::Str,
            Some(vec!["entropy-bound", "confidence-threshold"]),
            "Default sampler for diffusion generation.",
        ),
        "diffusion_threshold" => (
            KnobKind::Float,
            None,
            "Confidence threshold for diffusion threshold sampling.",
        ),
        _ => unreachable!("mutable setting metadata missing for {name}"),
    }
}

fn read_only_reason(field: &str) -> &'static str {
    match field {
        "reasoning_format"
        | "reasoning_alias_field"
        | "skip_chat_parsing"
        | "sse_ping_interval"
        | "spm_infill"
        | "embd_normalize"
        | "default_typical_p"
        | "default_top_n_sigma"
        | "default_xtc_probability"
        | "default_xtc_threshold"
        | "default_ignore_eos"
        | "default_stop_sequences"
        | "default_grammar"
        | "default_mirostat"
        | "default_mirostat_tau"
        | "default_mirostat_eta"
        | "default_dynatemp_range"
        | "default_dynatemp_exponent"
        | "default_adaptive_target"
        | "default_adaptive_decay"
        | "default_adaptive_p_named"
        | "default_logit_bias" => UNSUPPORTED_LIVE_REASON,
        "gcp"
        | "api_keys"
        | "api_prefix"
        | "enable_slots_endpoint"
        | "enable_props_endpoint"
        | "enable_metrics_endpoint"
        | "enable_settings_endpoint"
        | "cors_policy" => ROUTE_REASON,
        "no_prefill_assistant" | "reasoning_budget_message" => INERT_REASON,
        "pipeline_parallel_runtime"
        | "remote_pipeline_stage"
        | "tensor_parallel"
        | "serving_mode"
        | "prefill_peers"
        | "decode_peers"
        | "serving_bind" => DISTRIBUTED_REASON,
        "max_batch_size"
        | "max_queue_depth"
        | "audio_queue_depth"
        | "audio_request_timeout_secs"
        | "embedding_batch_size"
        | "embedding_max_length"
        | "embedding_queue_depth"
        | "embedding_request_timeout_secs"
        | "rerank_batch_size"
        | "prefill_chunk_size"
        | "prefill_grant_interval"
        | "enable_preemption"
        | "preemption_policy"
        | "no_batch"
        | "max_batch_prefill"
        | "max_batch_prefill_tokens"
        | "decode_storage_backend"
        | "vision_cache_size"
        | "prompt_cache"
        | "kv_cache_mode"
        | "batch_kv_quant"
        | "max_kv_size"
        | "context_shift"
        | "n_keep"
        | "kv_cache_budget"
        | "enable_vlm_prefix_cache" => SCHEDULER_REASON,
        "draft_model_path" | "num_draft_tokens" | "draft_kind" | "draft_block_size" => {
            WORKER_REASON
        }
        _ => MODEL_REASON,
    }
}

fn read_only_value(config: &ServerConfig, field: &str) -> Value {
    let debug = |value: &dyn std::fmt::Debug| Value::String(format!("{value:?}"));
    match field {
        "gcp" => debug(&config.gcp),
        "reasoning_format" => json!(config.reasoning_format.as_str()),
        "reasoning_alias_field" => json!(config.reasoning_alias_field.as_str()),
        "skip_chat_parsing" => json!(config.skip_chat_parsing),
        "no_prefill_assistant" => json!(config.no_prefill_assistant),
        "reasoning_budget_message" => json!(config.reasoning_budget_message),
        "api_keys" => json!({
            "configured": !config.api_keys.is_empty(),
            "count": config.api_keys.len(),
            "redacted": true,
        }),
        "api_prefix" => json!(config.api_prefix),
        // #1485 sampling and grammar defaults: resolved at startup and
        // honoured on every request, but read from the frozen config rather
        // than the live snapshot, so the management API reports them
        // read-only.
        "default_grammar" => config
            .default_grammar
            .as_ref()
            .and_then(|g| g.gbnf.clone())
            .map_or(Value::Null, Value::String),
        "default_mirostat" => json!(config.default_mirostat),
        "default_mirostat_tau" => json!(config.default_mirostat_tau),
        "default_mirostat_eta" => json!(config.default_mirostat_eta),
        "default_dynatemp_range" => json!(config.default_dynatemp_range),
        "default_dynatemp_exponent" => json!(config.default_dynatemp_exponent),
        "default_adaptive_target" => json!(config.default_adaptive_target),
        "default_adaptive_decay" => json!(config.default_adaptive_decay),
        "default_adaptive_p_named" => json!(config.default_adaptive_p_named),
        "default_logit_bias" => json!(
            config
                .default_logit_bias
                .iter()
                .map(|&(id, bias)| json!([id, bias]))
                .collect::<Vec<_>>()
        ),
        "sse_ping_interval" => config
            .sse_ping_interval
            .map(|value| json!(value.as_secs()))
            .unwrap_or(Value::Null),
        "model_alias" => json!(config.model_alias),
        "model_aliases" => json!(config.model_aliases),
        "context_size" => json!(config.context_size),
        "n_parallel" => json!(config.n_parallel),
        "enable_slots_endpoint" => json!(config.enable_slots_endpoint),
        "enable_props_endpoint" => json!(config.enable_props_endpoint),
        "enable_metrics_endpoint" => json!(config.enable_metrics_endpoint),
        "enable_settings_endpoint" => json!(config.enable_settings_endpoint),
        "slot_save_path" => config
            .slot_save_path
            .as_ref()
            .map(|path| json!(path.to_string_lossy()))
            .unwrap_or(Value::Null),
        "model_tags" => json!(config.model_tags),
        "lora_adapters" => json!({"count": config.lora_adapters.len()}),
        // The runtime LoRA set is a shared handle the worker mutates through
        // POST /lora-adapters, not a startup knob; report only whether
        // unfused serving is active, never the handle itself (#1439).
        "lora_runtime" => json!({"unfused": config.lora_runtime.is_some()}),
        "embd_normalize" => config
            .embd_normalize
            .map(|value| json!(value.value()))
            .unwrap_or(Value::Null),
        "embedding_serving_mode" => debug(&config.embedding_serving_mode),
        "spm_infill" => json!(config.spm_infill),
        "default_typical_p" => json!(config.default_typical_p),
        "default_top_n_sigma" => json!(config.default_top_n_sigma),
        "default_xtc_probability" => json!(config.default_xtc_probability),
        "default_xtc_threshold" => json!(config.default_xtc_threshold),
        "default_ignore_eos" => json!(config.default_ignore_eos),
        "default_stop_sequences" => json!(config.default_stop_sequences),
        "draft_model_path" => config
            .draft_model_path
            .as_ref()
            .map(|path| json!(path.to_string_lossy()))
            .unwrap_or(Value::Null),
        "num_draft_tokens" => json!(config.num_draft_tokens),
        "draft_kind" => json!(config.draft_kind),
        "draft_block_size" => json!(config.draft_block_size),
        "max_batch_size" => json!(config.max_batch_size),
        "max_queue_depth" => json!(config.max_queue_depth),
        "audio_queue_depth" => json!(config.audio_queue_depth),
        "audio_request_timeout_secs" => json!(config.audio_request_timeout_secs),
        "embedding_model_path" => config
            .embedding_model_path
            .as_ref()
            .map(|path| json!(path.to_string_lossy()))
            .unwrap_or(Value::Null),
        "embedding_batch_size" => json!(config.embedding_batch_size),
        "embedding_max_length" => json!(config.embedding_max_length),
        "embedding_queue_depth" => json!(config.embedding_queue_depth),
        "embedding_request_timeout_secs" => json!(config.embedding_request_timeout_secs),
        "reranker_model_path" => config
            .reranker_model_path
            .as_ref()
            .map(|path| json!(path.to_string_lossy()))
            .unwrap_or(Value::Null),
        "rerank_batch_size" => json!(config.rerank_batch_size),
        "prefill_chunk_size" => json!(config.prefill_chunk_size),
        "prefill_grant_interval" => json!(config.prefill_grant_interval),
        "enable_preemption" => json!(config.enable_preemption),
        "preemption_policy" => debug(&config.preemption_policy),
        "no_batch" => json!(config.no_batch),
        "max_batch_prefill" => json!(config.max_batch_prefill),
        "max_batch_prefill_tokens" => json!(config.max_batch_prefill_tokens),
        "decode_storage_backend" => debug(&config.decode_storage_backend),
        "pipeline_parallel_runtime" => debug(&config.pipeline_parallel_runtime),
        "remote_pipeline_stage" => debug(&config.remote_pipeline_stage),
        "tensor_parallel" => debug(&config.tensor_parallel),
        "vision_cache_size" => json!(config.vision_cache_size),
        "prompt_cache" => debug(&config.prompt_cache),
        "kv_cache_mode" => json!(config.kv_cache_mode.to_string()),
        "batch_kv_quant" => debug(&config.batch_kv_quant),
        "max_kv_size" => json!(config.max_kv_size),
        "context_shift" => json!(config.context_shift),
        "n_keep" => json!(config.n_keep),
        "kv_cache_budget" => debug(&config.kv_cache_budget),
        "enable_vlm_prefix_cache" => json!(config.enable_vlm_prefix_cache),
        "cors_policy" => debug(&config.cors_policy),
        "serving_mode" => debug(&config.serving_mode),
        "prefill_peers" => json!(
            config
                .prefill_peers
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        ),
        "decode_peers" => json!(
            config
                .decode_peers
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        ),
        "serving_bind" => config
            .serving_bind
            .map(|address| json!(address.to_string()))
            .unwrap_or(Value::Null),
        "model_is_gemma4_family" => json!(config.model_is_gemma4_family),
        _ => unreachable!("read-only ServerConfig field missing from schema: {field}"),
    }
}

fn read_only_kind(field: &str) -> KnobKind {
    match field {
        "no_prefill_assistant"
        | "skip_chat_parsing"
        | "spm_infill"
        | "default_ignore_eos"
        | "enable_slots_endpoint"
        | "enable_props_endpoint"
        | "enable_metrics_endpoint"
        | "enable_settings_endpoint"
        | "enable_preemption"
        | "no_batch"
        | "enable_vlm_prefix_cache"
        | "context_shift"
        | "model_is_gemma4_family"
        | "default_adaptive_p_named" => KnobKind::Bool,
        "context_size"
        | "n_parallel"
        | "n_keep"
        | "default_mirostat"
        | "num_draft_tokens"
        | "max_batch_size"
        | "max_queue_depth"
        | "audio_queue_depth"
        | "audio_request_timeout_secs"
        | "embedding_batch_size"
        | "embedding_queue_depth"
        | "embedding_request_timeout_secs"
        | "rerank_batch_size"
        | "prefill_chunk_size"
        | "max_batch_prefill"
        | "vision_cache_size" => KnobKind::Int,
        "model_aliases"
        | "model_tags"
        | "prefill_peers"
        | "decode_peers"
        | "default_stop_sequences"
        | "default_logit_bias" => KnobKind::Array,
        "sse_ping_interval"
        | "embd_normalize"
        | "draft_block_size"
        | "embedding_max_length"
        | "prefill_grant_interval"
        | "max_batch_prefill_tokens"
        | "max_kv_size" => KnobKind::IntOrNull,
        "default_typical_p"
        | "default_top_n_sigma"
        | "default_xtc_probability"
        | "default_xtc_threshold"
        | "default_mirostat_tau"
        | "default_mirostat_eta"
        | "default_dynatemp_range"
        | "default_dynatemp_exponent"
        | "default_adaptive_target"
        | "default_adaptive_decay" => KnobKind::Float,
        "api_keys" | "lora_adapters" | "lora_runtime" => KnobKind::Object,
        "reasoning_budget_message"
        | "model_alias"
        | "slot_save_path"
        | "draft_model_path"
        | "draft_kind"
        | "embedding_model_path"
        | "reranker_model_path"
        | "serving_bind"
        | "default_grammar" => KnobKind::StrOrNull,
        _ => KnobKind::Str,
    }
}

/// Full typed schema, including every read-only ServerConfig field.
#[must_use]
pub fn schema(startup: &ServerConfig) -> Vec<KnobSpec> {
    let live_defaults = mutable_values(&startup.live_settings());
    CLASSIFIED_SERVER_CONFIG_FIELDS
        .iter()
        .map(|&field| {
            let name = api_name(field);
            if is_mutable(name) {
                let (kind, allowed, help) = mutable_metadata(name);
                KnobSpec {
                    name,
                    kind,
                    default: live_defaults.get(name).cloned().unwrap_or(Value::Null),
                    mutable: true,
                    allowed,
                    help,
                    reason: None,
                }
            } else {
                KnobSpec {
                    name,
                    kind: read_only_kind(field),
                    default: read_only_value(startup, field),
                    mutable: false,
                    allowed: None,
                    help: "Startup server configuration.",
                    reason: Some(read_only_reason(field)),
                }
            }
        })
        .collect()
}

/// Current values for every schema entry.
#[must_use]
pub fn current(live: &LiveSettings, config: &ServerConfig) -> Map<String, Value> {
    let mutable = mutable_values(live);
    schema(config)
        .into_iter()
        .map(|spec| {
            let value = mutable
                .get(spec.name)
                .cloned()
                .unwrap_or_else(|| spec.default.clone());
            (spec.name.to_string(), value)
        })
        .collect()
}

fn recursively_sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<_, _> = map
                .iter()
                .map(|(key, value)| (key.clone(), recursively_sorted(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.iter().map(recursively_sorted).collect()),
        other => other.clone(),
    }
}

/// SHA-256 over canonical sorted mutable name/value pairs.
#[must_use]
pub fn fingerprint(live: &LiveSettings) -> String {
    let canonical = recursively_sorted(&Value::Object(mutable_values(live)));
    let encoded = serde_json::to_vec(&canonical).expect("live settings must serialize");
    let digest = Sha256::digest(encoded);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn as_f32(name: &str, value: &Value) -> Result<f32, String> {
    let raw = value
        .as_f64()
        .ok_or_else(|| format!("{name} must be a JSON number, got {}", type_name(value)))?;
    let parsed = raw as f32;
    if !parsed.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    Ok(parsed)
}

fn as_u64(name: &str, value: &Value) -> Result<u64, String> {
    value.as_u64().ok_or_else(|| {
        format!(
            "{name} must be a non-negative integer, got {}",
            type_name(value)
        )
    })
}

fn as_usize(name: &str, value: &Value) -> Result<usize, String> {
    usize::try_from(as_u64(name, value)?).map_err(|_| format!("{name} is too large"))
}

fn as_i32(name: &str, value: &Value) -> Result<i32, String> {
    let raw = value
        .as_i64()
        .ok_or_else(|| format!("{name} must be an integer, got {}", type_name(value)))?;
    i32::try_from(raw).map_err(|_| format!("{name} is outside the i32 range"))
}

fn string_values(name: &str, value: &Value) -> Result<Vec<String>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| format!("{name} must be an array of strings"))?;
    array
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{name} entries must be strings"))
        })
        .collect()
}

fn apply_one(
    next: &mut LiveSettings,
    name: &str,
    value: &Value,
    context_size: usize,
) -> Result<Value, String> {
    match name {
        "timeout_seconds" => {
            let parsed = as_u64(name, value)?;
            if parsed == 0 {
                return Err("timeout_seconds must be greater than zero".to_string());
            }
            next.timeout_seconds = parsed;
        }
        "default_temperature" => {
            let parsed = as_f32(name, value)?;
            if parsed < 0.0 {
                return Err("default_temperature must be >= 0".to_string());
            }
            next.default_temperature = parsed;
        }
        "default_top_p" => {
            let parsed = as_f32(name, value)?;
            if !(parsed > 0.0 && parsed <= 1.0) {
                return Err("default_top_p must be in (0, 1]".to_string());
            }
            next.default_top_p = parsed;
        }
        "default_top_k" => {
            let parsed = as_i32(name, value)?;
            if parsed < 0 {
                return Err("default_top_k must be >= 0".to_string());
            }
            next.default_top_k = parsed;
        }
        "default_min_p" => {
            let parsed = as_f32(name, value)?;
            if !(0.0..=1.0).contains(&parsed) {
                return Err("default_min_p must be in [0, 1]".to_string());
            }
            next.default_min_p = parsed;
        }
        "default_repetition_penalty" => {
            next.default_repetition_penalty = as_f32(name, value)?;
        }
        "default_repetition_context_size" => {
            next.default_repetition_context_size = as_usize(name, value)?;
        }
        "default_max_tokens" => {
            let parsed = as_usize(name, value)?;
            if parsed == 0 {
                return Err("default_max_tokens must be greater than zero".to_string());
            }
            if context_size > 0 && parsed > context_size {
                return Err(format!(
                    "default_max_tokens must be <= context_size ({context_size})"
                ));
            }
            next.default_max_tokens = parsed;
        }
        "default_seed" => {
            next.default_seed = if value.is_null() {
                None
            } else {
                Some(as_u64(name, value)?)
            };
        }
        "default_frequency_penalty" => next.default_frequency_penalty = as_f32(name, value)?,
        "default_presence_penalty" => next.default_presence_penalty = as_f32(name, value)?,
        "default_dry_multiplier" => {
            let parsed = as_f32(name, value)?;
            if parsed < 0.0 {
                return Err("default_dry_multiplier must be >= 0".to_string());
            }
            next.default_dry_multiplier = parsed;
        }
        "default_dry_base" => {
            let parsed = as_f32(name, value)?;
            if parsed < 1.0 {
                return Err("default_dry_base must be >= 1".to_string());
            }
            next.default_dry_base = parsed;
        }
        "default_dry_allowed_length" => next.default_dry_allowed_length = as_usize(name, value)?,
        "default_dry_penalty_last_n" => {
            next.default_dry_penalty_last_n = as_usize(name, value)?;
        }
        "default_dry_sequence_breakers" => {
            next.default_dry_sequence_breakers = string_values(name, value)?;
        }
        "lang_bias_config" => {
            next.lang_bias_config = if value.is_null() {
                None
            } else if value.is_object() {
                Some(
                    serde_json::from_value(value.clone())
                        .map_err(|error| format!("invalid lang_bias_config: {error}"))?,
                )
            } else {
                return Err(format!(
                    "lang_bias_config must be an object or null, got {}",
                    type_name(value)
                ));
            };
        }
        "reasoning_budget" => {
            next.reasoning_budget = ThinkingBudget::from_raw_i32(as_i32(name, value)?)
                .map_err(|error| error.to_string())?;
        }
        "chat_template_kwargs" => {
            next.chat_template_kwargs = Some(
                ChatTemplateKwargs::from_json_str(&value.to_string())
                    .map_err(|error| error.to_string())?,
            );
        }
        "loop_detection" => {
            next.loop_detection = if value.is_null() {
                None
            } else if value.is_object() {
                Some(
                    serde_json::from_value(value.clone())
                        .map_err(|error| format!("invalid loop_detection: {error}"))?,
                )
            } else {
                return Err(format!(
                    "loop_detection must be an object or null, got {}",
                    type_name(value)
                ));
            };
        }
        "max_denoising_steps" => {
            next.max_denoising_steps = if value.is_null() {
                None
            } else {
                let parsed = as_usize(name, value)?;
                if parsed == 0 {
                    return Err("max_denoising_steps must be greater than zero or null".to_string());
                }
                Some(parsed)
            };
        }
        "diffusion_sampler" => {
            let parsed = value
                .as_str()
                .ok_or_else(|| "diffusion_sampler must be a string".to_string())?;
            crate::server::diffusion_worker::parse_diffusion_sampler(parsed)?;
            next.diffusion_sampler = parsed.to_string();
        }
        "diffusion_threshold" => {
            let parsed = as_f32(name, value)?;
            if !(0.0..=1.0).contains(&parsed) {
                return Err("diffusion_threshold must be in [0, 1]".to_string());
            }
            next.diffusion_threshold = parsed;
        }
        _ => return Err(format!("unknown setting: {name}")),
    }
    Ok(mutable_values(next)
        .remove(name)
        .expect("applied mutable setting must be serializable"))
}

/// Apply a merge or replace patch. Each valid knob applies even if another is
/// rejected; the caller publishes next in one swap.
#[must_use]
pub fn apply(
    startup: &LiveSettings,
    current: &LiveSettings,
    op: Op,
    values: &Map<String, Value>,
    config: &ServerConfig,
) -> ApplyResult {
    let mut next = match op {
        Op::Merge => current.clone(),
        Op::Replace => startup.clone(),
    };
    let specs = schema(config);
    let mut applied = Map::new();
    let mut rejected = Vec::new();
    for (name, value) in values {
        match specs.iter().find(|spec| spec.name == name) {
            None => rejected.push(Rejected {
                name: name.clone(),
                reason: "unknown setting".to_string(),
            }),
            Some(spec) if !spec.mutable => rejected.push(Rejected {
                name: name.clone(),
                reason: format!("read-only: {}", spec.reason.unwrap_or("restart required")),
            }),
            Some(_) => match apply_one(&mut next, name, value, config.context_size) {
                Ok(canonical) => {
                    applied.insert(name.clone(), canonical);
                }
                Err(reason) => rejected.push(Rejected {
                    name: name.clone(),
                    reason,
                }),
            },
        }
    }
    ApplyResult {
        next,
        applied,
        rejected,
    }
}

/// Parse the two accepted PATCH body shapes.
pub fn parse_patch_body(mut body: Map<String, Value>) -> Result<(Op, Map<String, Value>), String> {
    let wrapped = body.contains_key("op") || body.contains_key("values");
    if !wrapped {
        return Ok((Op::Merge, body));
    }
    let op = match body.remove("op") {
        None => Op::Merge,
        Some(value) => serde_json::from_value(value)
            .map_err(|_| "op must be \"merge\" or \"replace\"".to_string())?,
    };
    let values = match body.remove("values") {
        Some(Value::Object(values)) => values,
        Some(other) => {
            return Err(format!(
                "values must be an object, got {}",
                type_name(&other)
            ));
        }
        None => return Err("wrapped PATCH body requires a values object".to_string()),
    };
    if !body.is_empty() {
        return Err(format!(
            "wrapped PATCH body has unknown fields: {}",
            body.keys().cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    Ok((op, values))
}

#[cfg(test)]
#[path = "runtime_settings_tests.rs"]
mod tests;
