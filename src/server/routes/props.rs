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

//! Server properties endpoints (llama-server b10621 compatible, issue #1440).
//!
//! `GET /props` is always mounted, as it is in b10621, and reports the b10621
//! key set (`default_generation_settings` as `{params, n_ctx}`, model
//! identity, modalities, endpoint toggles, chat-template metadata, build
//! info, sleep state) plus mlxcel's own resolved-configuration extensions
//! (`kv_cache_mode`, `kv_bits`, `speculative`, `capabilities`), which b10621
//! clients ignore. `POST /props` is what `--props` gates, exactly upstream's
//! semantics: enabled it answers `{"success": true}` without changing
//! anything (upstream's handler body is literally empty), disabled it
//! answers upstream's 501 diagnostic.

use axum::{Json, extract::State, response::IntoResponse, response::Response};

use super::slots::llama_not_supported;
use crate::server::config::ServerConfig;
use crate::server::types::{EmbeddingCapability, RerankCapability, ServerCapabilities};
use crate::server::{AppState, LiveSettings};

/// The `default_generation_settings.params` object of the `/props` payload.
///
/// b10621 reports its full `task_params` block here; mlxcel reports the
/// resolved defaults it actually acts on, under upstream's key names, and
/// omits a key it has no analogue for rather than inventing a value (the
/// #1441 precedent for `generation_settings`, recorded as the entry's
/// divergence). The set is a contract: an operator reads it to confirm what
/// the server resolved a flag to, so a sampling default that the server
/// honours but does not report here is invisible in exactly the way
/// `--dry-sequence-breaker` was before #1103.
#[allow(dead_code)]
pub(crate) fn default_generation_settings(config: &ServerConfig) -> serde_json::Value {
    default_generation_settings_with_live(config, &config.live_settings())
}

fn default_generation_settings_with_live(
    config: &ServerConfig,
    live: &LiveSettings,
) -> serde_json::Value {
    serde_json::json!({
        "n_predict": live.default_max_tokens,
        "max_tokens": live.default_max_tokens,
        "temperature": live.default_temperature,
        "top_k": live.default_top_k,
        "top_p": live.default_top_p,
        "min_p": live.default_min_p,
        "typical_p": config.default_typical_p,
        "top_n_sigma": config.default_top_n_sigma,
        "xtc_probability": config.default_xtc_probability,
        "xtc_threshold": config.default_xtc_threshold,
        "ignore_eos": config.default_ignore_eos,
        "repeat_penalty": live.default_repetition_penalty,
        "repeat_last_n": live.default_repetition_context_size,
        "seed": live.default_seed.unwrap_or(u64::MAX),
        "frequency_penalty": live.default_frequency_penalty,
        "presence_penalty": live.default_presence_penalty,
        "dry_multiplier": live.default_dry_multiplier,
        "dry_base": live.default_dry_base,
        "dry_allowed_length": live.default_dry_allowed_length,
        "dry_penalty_last_n": live.default_dry_penalty_last_n,
        // Reported as resolved token IDs rather than as the strings the
        // operator typed, because the IDs are what the sampler compares
        // against and what a per-request `dry_sequence_breakers` overrides.
        "dry_sequence_breakers": live.default_dry_sequence_breakers,
        // b10621 context retention (#1472 gave these a real analogue, so
        // #1440's omission policy no longer applies to them): `n_keep` is the
        // server-wide `--keep` a request's own `n_keep` falls back to, and
        // `n_discard` is upstream's per-request default, which mlxcel does not
        // make server-settable and therefore always reports as upstream's 0.
        "n_keep": config.n_keep,
        "n_discard": 0,
    })
}

/// mlxcel context/batch geometry, reported as a `/props` extension block.
///
/// An operator passes `--ctx-size` and `--batch-size` and has no other way to
/// confirm what the server resolved them to (#1450). `n_ctx` is the PER-SLOT
/// window (`--ctx-size 8192 --parallel 4` gives each slot 2048), matching
/// llama-server, whose `/props` also reports the per-slot `n_ctx`. `0` means
/// the checkpoint's own trained context, which mlxcel does not clamp.
pub(crate) fn geometry_block(config: &ServerConfig) -> serde_json::Value {
    serde_json::json!({
        // The logical prefill batch `--batch-size` / `-b` resolves to. mlxcel
        // has no separate physical micro-batch (`--ubatch-size` is accepted
        // and ignored on unified memory), so the two report the same value.
        "n_batch": config.prefill_chunk_size,
        "n_ubatch": config.prefill_chunk_size,
        // The decode batch width `--max-batch-size` (or `--parallel`)
        // resolved to, before the scheduler's per-family clamp.
        "n_batch_decode": config.max_batch_size,
        // The KV live-window cap in tokens, `null` when unbounded. Folds
        // `--max-kv-size` and the per-slot share of `--ctx-size` together the
        // way `resolve_context_kv_cap` does.
        "n_kv_max": config.max_kv_size,
    })
}

/// Read a JSON file below the model directory, `None` on any failure.
fn read_model_json(state: &AppState, name: &str) -> Option<serde_json::Value> {
    let raw = std::fs::read_to_string(state.model_path.join(name)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// A special token from `tokenizer_config.json`, which stores either a bare
/// string or an added-token object with a `content` key.
fn special_token_string(config: Option<&serde_json::Value>, key: &str) -> String {
    let Some(value) = config.and_then(|c| c.get(key)) else {
        return String::new();
    };
    match value {
        serde_json::Value::String(s) => s.clone(),
        obj => obj
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or_default()
            .to_string(),
    }
}

/// The checkpoint's storage type, in the `model_ftype` position.
///
/// b10621 reports the GGUF ftype name; MLX checkpoints have no GGUF ftype, so
/// the honest analogue is the quantization the checkpoint declares (`MLX Q4`)
/// or the f16 the non-quantized weights are served at.
fn model_ftype(state: &AppState) -> String {
    let quant_bits = read_model_json(state, "config.json")
        .as_ref()
        .and_then(|c| c.get("quantization"))
        .and_then(|q| q.get("bits"))
        .and_then(|b| b.as_u64());
    match quant_bits {
        Some(bits) => format!("MLX Q{bits}"),
        None => "MLX F16".to_string(),
    }
}

/// The b10621 `chat_template_caps` map, derived from the loaded template.
///
/// The nine keys are `jinja::caps::to_map` at the pinned commit
/// (https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/jinja/caps.cpp).
/// b10621 derives them by probing the template with synthetic renders;
/// mlxcel derives them from what its own pipeline does with the template:
/// content is normalized to strings before rendering, tool support follows
/// the template's tools hint, and reasoning caps follow the template's use
/// of the corresponding variables.
fn chat_template_caps(state: &AppState) -> serde_json::Value {
    let source = state.chat_template.template_source();
    let tools = state.chat_template.supports_tools_hint();
    serde_json::json!({
        "supports_string_content": true,
        "supports_typed_content": false,
        "supports_tools": tools,
        "supports_tool_calls": tools,
        "supports_parallel_tool_calls": tools,
        "supports_system_role": source.contains("system"),
        "supports_preserve_reasoning": source.contains("reasoning_content"),
        "supports_reasoning_effort": source.contains("reasoning_effort"),
        "supports_object_arguments": false,
    })
}

/// GET /props
pub async fn props(State(state): State<AppState>) -> Json<serde_json::Value> {
    let live = state.live();
    let tokenizer_config = read_model_json(&state, "tokenizer_config.json");
    let mut body = serde_json::json!({
        // -- b10621 key set --
        "default_generation_settings": {
            "params": default_generation_settings_with_live(&state.config, &live),
            "n_ctx": state.config.context_size,
        },
        "total_slots": state.config.n_parallel,
        "model_alias": state.display_model_id(),
        "model_ftype": model_ftype(&state),
        "model_path": state.model_path.to_string_lossy(),
        "modalities": {
            "vision": state.media_support.image,
            "video": state.media_support.video,
            "audio": state.media_support.audio,
        },
        // mlxcel's chat surface has no textual media placeholder: media
        // arrives as content parts, never as a marker spliced into prompt
        // text, so there is no marker string to report where b10621 reports
        // mtmd's `<__media__>`.
        "media_marker": serde_json::Value::Null,
        "endpoint_slots": state.config.enable_slots_endpoint,
        "endpoint_props": state.config.enable_props_endpoint,
        "endpoint_metrics": state.config.enable_metrics_endpoint,
        "endpoint_settings": state.config.enable_settings_endpoint,
        // mlxcel ships no web UI and no MCP CORS proxy.
        "ui": false,
        "ui_settings": {},
        "chat_template": state.chat_template.template_source(),
        "chat_template_caps": chat_template_caps(&state),
        "bos_token": special_token_string(tokenizer_config.as_ref(), "bos_token"),
        "eos_token": special_token_string(tokenizer_config.as_ref(), "eos_token"),
        "build_info": concat!("mlxcel-", env!("CARGO_PKG_VERSION")),
        // mlxcel has no idle-sleep lifecycle yet (`--sleep-idle-seconds` is
        // deferred under #1440), so the server is truthfully never sleeping.
        "is_sleeping": false,
        "cors_proxy_enabled": false,
        // -- mlxcel extension keys, resolved-configuration reporting --
        // The effective KV mode, not the requested one: startup has already
        // substituted anything this model family cannot hold (issue #1350).
        "kv_cache_mode": state.config.kv_cache_mode.to_string(),
        "kv_bits": state.config.batch_kv_quant.bits,
        "speculative": speculative_config(&state.config),
        "geometry": geometry_block(&state.config),
    });
    body["capabilities"] = serde_json::to_value(server_capabilities(&state)).unwrap_or_default();
    Json(body)
}

/// POST /props
///
/// b10621 gates this behind `--props` and, when enabled, acknowledges with
/// `{"success": true}` without reading the body or changing anything (its
/// handler body is empty upstream). Matching that exactly means not parsing
/// the request body at all.
pub async fn post_props(State(state): State<AppState>) -> Response {
    if !state.config.enable_props_endpoint {
        return llama_not_supported(
            "This server does not support changing global properties. Start it with `--props`",
        );
    }
    Json(serde_json::json!({ "success": true })).into_response()
}

/// The resolved speculative configuration block of `/props` (#1433).
///
/// Reported so an operator can confirm what `--model-draft` /
/// `--spec-draft-n-max` / `--draft-kind` resolved to, without exposing the
/// draft checkpoint's full path (only its basename; a repo-id or URL could
/// otherwise carry a token).
pub(crate) fn speculative_config(config: &ServerConfig) -> serde_json::Value {
    serde_json::json!({
        "model": config.draft_model_path.as_ref().map(|p| {
            // Basename only (a full path or repo URL could leak layout or a
            // token); a path with no final component (trailing `/`, `..`)
            // still reads as "configured" rather than as no draft model.
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "(configured)".to_string())
        }),
        "kind": config.draft_kind,
        "n_max": config.num_draft_tokens,
    })
}

/// Build the resolved capability block (#1452).
///
/// Everything here is read from the workers that were really loaded rather than
/// from the flags that were passed, so a `--pooling` the checkpoint overrode,
/// or a `--embedding-model` that failed to load, shows up as what happened.
pub(crate) fn server_capabilities(state: &AppState) -> ServerCapabilities {
    ServerCapabilities {
        generation: !state.model_provider.is_chat_unavailable()
            && !state.config.embedding_serving_mode.blocks_generation(),
        serving_mode: state.config.embedding_serving_mode.flag(),
        embedding: state
            .embedding_model
            .as_ref()
            .map(|provider| EmbeddingCapability {
                model: provider.model_id().to_string(),
                dim: provider.dim(),
                pooling: provider.pooling().to_string(),
                // The value an unqualified request really gets: the
                // `--embd-normalize` flag when the operator set one, and the
                // checkpoint's own `normalize` flag otherwise. Reporting the
                // checkpoint's answer unconditionally would have shown 2 on a
                // server started with `--embd-normalize 1`, which is the
                // opposite of what this block is for.
                embd_normalize: state
                    .config
                    .embd_normalize
                    .unwrap_or_else(|| provider.embd_normalize())
                    .value(),
                multi_vector: provider.multi_vector(),
            }),
        reranking: state
            .rerank_model
            .as_ref()
            .map(|provider| RerankCapability {
                model: provider.model_id().to_string(),
                kind: provider.kind().as_str().to_string(),
            }),
    }
}

#[cfg(test)]
#[path = "props_tests.rs"]
mod props_tests;
