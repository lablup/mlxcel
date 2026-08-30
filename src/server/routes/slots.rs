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

//! Slots endpoints (llama-server b10621 compatible, issue #1440).
//!
//! `GET /slots` reports each slot in b10621's shape, redacting `prompt` and
//! `generated` unless `LLAMA_SERVER_SLOTS_DEBUG` is set, and honors the
//! `fail_on_no_slot` query switch. `POST /slots/:id_slot` serves the
//! `save` / `restore` / `erase` actions behind `--slot-save-path`. Both
//! routes are always mounted and answer b10621's own diagnostics when their
//! gate is off, which is what upstream does instead of a 404.

use std::collections::HashMap;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::server::AppState;
use crate::server::config::ServerGenerateOptions;
use crate::server::slot_persist::{self, SlotPersistError};
use crate::server::slots_state::SlotCache;

/// b10621 `format_error_response` envelope: `{"error": {code, message,
/// type}}` with a numeric `code` equal to the HTTP status.
///
/// mlxcel's own [`crate::server::types::ErrorResponse`] carries a nullable
/// string `code`; the native llama-server surfaces added by #1440 need the
/// numeric form byte-for-byte, so they build it here instead of widening the
/// shared type under three concurrently developed chains.
pub(crate) fn llama_error_response(
    status: StatusCode,
    error_type: &str,
    message: &str,
) -> Response {
    let body = serde_json::json!({
        "error": {
            "code": status.as_u16(),
            "message": message,
            "type": error_type,
        }
    });
    (status, Json(body)).into_response()
}

/// b10621 `ERROR_TYPE_NOT_SUPPORTED` (501).
pub(crate) fn llama_not_supported(message: &str) -> Response {
    llama_error_response(StatusCode::NOT_IMPLEMENTED, "not_supported_error", message)
}

/// b10621 `ERROR_TYPE_INVALID_REQUEST` (400).
pub(crate) fn llama_invalid_request(message: &str) -> Response {
    llama_error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// b10621 `ERROR_TYPE_UNAVAILABLE` (503).
pub(crate) fn llama_unavailable(message: &str) -> Response {
    llama_error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "unavailable_error",
        message,
    )
}

/// The b10621 `params` snapshot a slot reports for its task.
///
/// The key set follows the #1441 precedent for `generation_settings`: the
/// settings mlxcel actually resolves are reported under upstream's names, and
/// a b10621 `task_params` key with no mlxcel analogue is omitted rather than
/// invented. That difference is the recorded divergence on the `GET /slots`
/// manifest entry.
pub(crate) fn slot_params_json(options: &ServerGenerateOptions, stream: bool) -> serde_json::Value {
    let sampling = &options.sampling;
    serde_json::json!({
        "seed": sampling.seed.unwrap_or(u64::MAX),
        "temperature": sampling.temperature,
        "top_k": sampling.top_k,
        "top_p": sampling.top_p,
        "min_p": sampling.min_p,
        "top_n_sigma": sampling.top_n_sigma,
        "xtc_probability": sampling.xtc_probability,
        "xtc_threshold": sampling.xtc_threshold,
        "typical_p": sampling.typical_p,
        "repeat_last_n": sampling.penalty_last_n,
        "repeat_penalty": sampling.repetition_penalty,
        "presence_penalty": sampling.presence_penalty,
        "frequency_penalty": sampling.frequency_penalty,
        "dry_multiplier": sampling.dry_multiplier,
        "dry_base": sampling.dry_base,
        "dry_allowed_length": sampling.dry_allowed_length,
        "dry_penalty_last_n": sampling.dry_penalty_last_n,
        "max_tokens": options.max_tokens,
        "n_predict": options.max_tokens,
        "ignore_eos": options.ignore_eos,
        "stream": stream,
        "n_probs": options.logprobs.top_k,
        "stop": options.stop_sequences.clone().unwrap_or_default(),
        "timings_per_token": false,
        // #1485 sampling remainder.
        "mirostat": sampling.mirostat,
        "mirostat_tau": sampling.mirostat_tau,
        "mirostat_eta": sampling.mirostat_eta,
        "dynatemp_range": sampling.dynatemp_range,
        "dynatemp_exponent": sampling.dynatemp_exponent,
        "adaptive_target": sampling.adaptive_target,
        "adaptive_decay": sampling.adaptive_decay,
        "min_keep": sampling.min_keep,
        "post_sampling_probs": options.post_sampling_probs,
    })
}

/// GET /slots
///
/// b10621 shape: one object per slot with `id`, `n_ctx`, `speculative`,
/// `is_processing`, and, once the slot has carried a task, the task's
/// counters. `?fail_on_no_slot=1` turns "no idle slot" into upstream's 503.
pub async fn slots(
    State(state): State<AppState>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    if !state.config.enable_slots_endpoint {
        return llama_not_supported(
            "This server does not support slots endpoint. Start it with `--slots`",
        );
    }

    let fail_on_no_slot = query.get("fail_on_no_slot").is_some_and(|v| !v.is_empty());
    if fail_on_no_slot && state.slots.idle_count() == 0 {
        return llama_unavailable("no slot available");
    }

    let speculative = state.config.draft_model_path.is_some();
    let body = state
        .slots
        .slots_json(state.config.context_size, speculative, state.slots_debug);
    Json(body).into_response()
}

/// Parse b10621's `std::stoi` semantics: optional sign and leading digits,
/// ignoring any trailing garbage, error only when no digits lead.
fn parse_stoi(value: &str) -> Option<i64> {
    let trimmed = value.trim_start();
    let (sign, rest) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1i64, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<i64>().ok().map(|n| sign * n)
}

#[derive(serde::Deserialize)]
struct SlotActionBody {
    filename: Option<String>,
}

/// POST /slots/:id_slot?action=save|restore|erase
pub async fn slot_action(
    State(state): State<AppState>,
    Path(id_slot): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    body: axum::body::Bytes,
) -> Response {
    let Some(save_root) = state.config.slot_save_path.clone() else {
        return llama_not_supported(
            "This server does not support slots action. Start it with `--slot-save-path`",
        );
    };

    let Some(id) = parse_stoi(&id_slot) else {
        return llama_invalid_request("Invalid slot ID");
    };
    if id < 0 || (id as usize) >= state.slots.total() {
        return llama_invalid_request("Invalid slot ID");
    }
    let id = id as usize;

    let action = query.get("action").map(String::as_str).unwrap_or("");
    match action {
        "save" | "restore" => {
            let Some(filename) = serde_json::from_slice::<SlotActionBody>(&body)
                .ok()
                .and_then(|b| b.filename)
            else {
                return llama_invalid_request("Invalid filename");
            };
            if action == "save" {
                slot_save(&state, &save_root, id, &filename)
            } else {
                slot_restore(&state, &save_root, id, &filename)
            }
        }
        "erase" => slot_erase(&state, id),
        _ => llama_invalid_request("Invalid action"),
    }
}

/// Materialize the token stream a slot currently caches.
///
/// A restored slot already holds ids; a slot whose last task retained text
/// re-encodes prompt + generation, which reproduces the stream the next
/// request's prefill would compute from the same text.
fn materialize_cache_tokens(
    state: &AppState,
    cache: &SlotCache,
    prompt: &str,
    generated: &str,
) -> Vec<i32> {
    match cache {
        SlotCache::Empty => Vec::new(),
        SlotCache::Restored(tokens) => tokens.clone(),
        SlotCache::FromTask => {
            let text = format!("{prompt}{generated}");
            if text.is_empty() {
                return Vec::new();
            }
            state
                .tokenizer
                .encode(&text, true)
                .map(|ids| ids.into_iter().map(|id| id as i32).collect())
                .unwrap_or_default()
        }
    }
}

fn persist_error_response(err: SlotPersistError) -> Response {
    match err {
        SlotPersistError::InvalidFilename => llama_invalid_request("Invalid filename"),
        SlotPersistError::Unreadable(msg) | SlotPersistError::Invalid(msg) => {
            llama_invalid_request(&format!("Unable to restore slot: {msg}"))
        }
        SlotPersistError::Io(msg) => {
            llama_error_response(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &msg)
        }
    }
}

fn slot_save(state: &AppState, root: &std::path::Path, id: usize, filename: &str) -> Response {
    let started = std::time::Instant::now();
    let (cache, prompt, generated) = match state.slots.cache_for_save(id) {
        None => return llama_invalid_request("Invalid slot ID"),
        Some(Err(())) => return llama_unavailable("Requested slot is processing"),
        Some(Ok(parts)) => parts,
    };
    let tokens = materialize_cache_tokens(state, &cache, &prompt, &generated);
    let fingerprint = slot_persist::tokenizer_fingerprint(&state.tokenizer);
    match slot_persist::save(
        root,
        filename,
        state.display_model_id(),
        &fingerprint,
        &tokens,
    ) {
        Ok(n_written) => Json(serde_json::json!({
            "id_slot": id,
            "filename": filename,
            "n_saved": tokens.len(),
            "n_written": n_written,
            "timings": { "save_ms": started.elapsed().as_secs_f64() * 1000.0 },
        }))
        .into_response(),
        Err(err) => persist_error_response(err),
    }
}

fn slot_restore(state: &AppState, root: &std::path::Path, id: usize, filename: &str) -> Response {
    let started = std::time::Instant::now();
    let fingerprint = slot_persist::tokenizer_fingerprint(&state.tokenizer);
    let (envelope, n_read) =
        match slot_persist::load(root, filename, state.display_model_id(), &fingerprint) {
            Ok(loaded) => loaded,
            Err(err) => return persist_error_response(err),
        };
    // The restored stream must fit the per-slot context window, b10621's
    // "Restored prompt does not fit in the slot context" refusal. A
    // `--ctx-size` of 0 means the model's own trained window applies and the
    // bound is enforced at prefill time instead.
    let n_ctx = state.config.context_size;
    if n_ctx > 0 && envelope.tokens.len() > n_ctx {
        return llama_invalid_request(
            "Unable to restore slot: Restored prompt does not fit in the slot context",
        );
    }
    let n_restored = envelope.tokens.len();
    match state.slots.install_restored(id, envelope.tokens) {
        None => llama_invalid_request("Invalid slot ID"),
        Some(Err(())) => llama_unavailable("Requested slot is processing"),
        Some(Ok(())) => Json(serde_json::json!({
            "id_slot": id,
            "filename": filename,
            "n_restored": n_restored,
            "n_read": n_read,
            "timings": { "restore_ms": started.elapsed().as_secs_f64() * 1000.0 },
        }))
        .into_response(),
    }
}

fn slot_erase(state: &AppState, id: usize) -> Response {
    let (cache, prompt, generated) = match state.slots.erase(id) {
        None => return llama_invalid_request("Invalid slot ID"),
        Some(Err(())) => return llama_unavailable("Requested slot is processing"),
        Some(Ok(parts)) => parts,
    };
    let n_erased = materialize_cache_tokens(state, &cache, &prompt, &generated).len();
    Json(serde_json::json!({
        "id_slot": id,
        "n_erased": n_erased,
    }))
    .into_response()
}

#[cfg(test)]
#[path = "slots_tests.rs"]
mod slots_tests;
