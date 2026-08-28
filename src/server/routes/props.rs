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

//! Server properties endpoint (llama-server compatible).
//!
//! This route surfaces configuration/state snapshots and should stay as a thin
//! read-only adapter.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::config::ServerConfig;
use crate::server::types::{
    EmbeddingCapability, PropsResponse, RerankCapability, ServerCapabilities,
};

/// The `default_generation_settings` object of the `/props` payload.
///
/// Split out of the handler so the reported field set can be asserted without
/// standing up an `AppState`. The set is a contract: an operator reads it to
/// confirm what the server resolved a flag to, so a sampling default that the
/// server honours but does not report here is invisible in exactly the way
/// `--dry-sequence-breaker` was before #1103.
pub(crate) fn default_generation_settings(config: &ServerConfig) -> serde_json::Value {
    serde_json::json!({
        "n_predict": config.default_max_tokens,
        "temperature": config.default_temperature,
        "top_k": config.default_top_k,
        "top_p": config.default_top_p,
        "min_p": config.default_min_p,
        "typical_p": config.default_typical_p,
        "top_n_sigma": config.default_top_n_sigma,
        "xtc_probability": config.default_xtc_probability,
        "xtc_threshold": config.default_xtc_threshold,
        "ignore_eos": config.default_ignore_eos,
        "repeat_penalty": config.default_repetition_penalty,
        "repeat_last_n": config.default_repetition_context_size,
        "seed": config.default_seed.unwrap_or(u64::MAX),
        "frequency_penalty": config.default_frequency_penalty,
        "presence_penalty": config.default_presence_penalty,
        "dry_multiplier": config.default_dry_multiplier,
        "dry_base": config.default_dry_base,
        "dry_allowed_length": config.default_dry_allowed_length,
        "dry_penalty_last_n": config.default_dry_penalty_last_n,
        // Reported as resolved token IDs rather than as the strings the
        // operator typed, because the IDs are what the sampler compares
        // against and what a per-request `dry_sequence_breakers` overrides.
        "dry_sequence_breakers": config.default_dry_sequence_breakers,
        // Context and batch geometry, reported for the same reason the
        // sampling defaults are: an operator passes `--ctx-size` and
        // `--batch-size` and has no other way to confirm what the server
        // resolved them to (#1450). `n_ctx` is the PER-SLOT window, which is
        // the number a request is actually bounded by, not the `--ctx-size`
        // total: `--ctx-size 8192 --parallel 4` gives each slot 2048. That
        // matches llama-server, whose `/props` also reports the per-slot
        // `n_ctx` rather than the aggregate. `0` means the checkpoint's own
        // trained context, which mlxcel does not clamp.
        "n_ctx": config.context_size,
        // The logical prefill batch `--batch-size` / `-b` resolves to.
        // mlxcel has no separate physical micro-batch (`--ubatch-size` is
        // accepted and ignored on unified memory), so the two are reported at
        // the same value rather than one of them being invented.
        "n_batch": config.prefill_chunk_size,
        "n_ubatch": config.prefill_chunk_size,
        // The decode batch width `--max-batch-size` (or `--parallel`)
        // resolved to, before the scheduler's per-family clamp.
        "n_batch_decode": config.max_batch_size,
        // The KV live-window cap in tokens, `null` when unbounded. Folds
        // `--max-kv-size` and the per-slot share of `--ctx-size` together the
        // way `resolve_context_kv_cap` does, so what is reported is the bound
        // that is actually enforced.
        "n_kv_max": config.max_kv_size,
    })
}

/// GET /props
pub async fn props(State(state): State<AppState>) -> Json<PropsResponse> {
    Json(PropsResponse {
        default_generation_settings: default_generation_settings(&state.config),
        total_slots: state.config.n_parallel,
        // The effective mode, not the requested one: `ServerStartupInput::
        // into_startup_config` has already substituted anything this model
        // family cannot hold (issue #1350), and `ServerConfig` carries the
        // result. Reporting it here is what makes "the mode announced is the
        // mode in force" checkable by a client rather than only greppable in
        // the startup log.
        kv_cache_mode: state.config.kv_cache_mode.to_string(),
        speculative: speculative_config(&state.config),
        kv_bits: state.config.batch_kv_quant.bits,
        capabilities: server_capabilities(&state),
    })
}

/// The resolved speculative configuration block of `/props` (#1433).
///
/// Reported so an operator can confirm what `--model-draft` /
/// `--spec-draft-n-max` / `--draft-kind` resolved to, without exposing the
/// draft checkpoint's full path (only its basename; a repo-id or URL could
/// otherwise carry a token).
pub(crate) fn speculative_config(config: &ServerConfig) -> serde_json::Value {
    serde_json::json!({
        "model": config
            .draft_model_path
            .as_ref()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())),
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
