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

//! Thin HTTP route adapters.
//!
//! These files should stay focused on request/response translation. Shared
//! policy belongs in `server/request_options.rs`, `server/chat_request.rs`,
//! `server/media.rs`, `server/streaming.rs`, and `server/model_worker.rs`.
use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;

use crate::server::AppState;
use crate::server::audio_model::AudioModelKind;
use crate::server::model_provider::{ChatWorkerGoneError, QueueFullError};
use crate::server::types::ErrorResponse;

pub mod anthropic;
pub mod audio;
pub mod cache;
pub mod chat;
pub(crate) mod chat_generation;
pub mod completions;
pub mod control;
pub mod detokenize;
pub mod embeddings;
pub mod feature_disabled;
pub mod health;
pub mod images;
pub mod infill;
pub mod lora_adapters;
pub mod metrics;
pub mod models;
pub mod mtp_policy;
pub mod native_completion;
pub mod prompt_inspection;
pub mod props;
pub mod realtime;
pub mod rerank;
pub mod responses;
mod responses_auto_compact;
pub mod responses_compact;
mod responses_compact_media;
pub(crate) mod responses_run;
pub mod settings;
pub mod slots;
pub mod stream;
pub mod tokenize;
pub(crate) mod tool_call_stream;
pub mod transcription_compat;

#[cfg(test)]
#[path = "availability_tests.rs"]
mod availability_tests;

#[cfg(test)]
#[path = "realtime_route_tests.rs"]
mod realtime_route_tests;

#[cfg(test)]
#[path = "stream_route_tests.rs"]
mod stream_route_tests;

#[cfg(test)]
#[path = "feature_disabled_tests.rs"]
mod feature_disabled_tests;

#[cfg(test)]
#[path = "settings_tests.rs"]
mod settings_tests;

#[cfg(test)]
#[path = "json_body_rejection_tests.rs"]
mod json_body_rejection_tests;

pub use anthropic::{anthropic_count_tokens, anthropic_messages};
pub use audio::{audio_speech, audio_transcriptions, audio_translations};
pub use cache::{cache_reset, cache_stats};
pub use chat::chat_completions;
pub use completions::completions;
pub use control::chat_completions_control;
pub use detokenize::detokenize;
pub use embeddings::{create_embeddings, native_embeddings};
pub use feature_disabled::feature_disabled;
pub use health::health_check;
pub use images::{create_image_edit, create_image_generation};
pub use infill::infill;
pub use lora_adapters::{get_lora_adapters, post_lora_adapters};
pub use metrics::metrics;
pub use models::list_models;
pub use mtp_policy::mtp_policy;
pub use native_completion::native_completion;
pub use prompt_inspection::{apply_template, chat_input_tokens, responses_input_tokens};
pub use props::{post_props, props};
pub use rerank::create_rerank;
pub use responses::{cancel_response, create_response, delete_response, retrieve_response};
pub use responses_compact::compact_response;
pub use settings::{get_settings, patch_settings};
pub use slots::{slot_action, slots};
pub use stream::{stream_delete, stream_get, streams_lookup};
pub use tokenize::tokenize;

/// Explain a terminal generation-unavailable state without exposing worker or
/// channel implementation details.
pub(crate) fn chat_unavailable_message(state: &AppState) -> Option<String> {
    // `--image-model` with no `-m` serves image generation alone (#2825).
    // Its own wording is used rather than the embedding-only sentence below,
    // because the remedy (add `-m`, or a second server) differs from dropping
    // a restriction flag.
    if state.config.embedding_serving_mode == crate::server::config::EmbeddingServingMode::ImageOnly
    {
        return Some(
            "This server was started with --image-model and no -m and serves only \
             /v1/images/generations and /v1/images/edits; generation is disabled. Add -m <chat \
             model> to serve generation from the same process, or start a second server."
                .to_string(),
        );
    }
    // b10621's `--embeddings` / `--reranking` restrict the server to that
    // workload, so generation is refused even when a chat model is loaded next
    // to the side model (#1452). The flag is named in the body, because
    // "generation is off here" and "the chat model failed to load" are
    // different operational problems and the 501 is the only place a client
    // sees either.
    if let Some(flag) = state.config.embedding_serving_mode.flag() {
        return Some(format!(
            "This server was started with {flag} and serves only its {} routes; generation is \
             disabled. Drop {flag} to serve generation from the same process, or start a second \
             server with -m <chat model>.",
            match state.config.embedding_serving_mode {
                crate::server::config::EmbeddingServingMode::RerankOnly => "reranking",
                _ => "embedding",
            }
        ));
    }
    if !state.model_provider.is_chat_unavailable() {
        return None;
    }

    let mut routes = Vec::new();
    let mut side_model_flags = Vec::new();
    if state.embedding_model.is_some() {
        routes.push("/v1/embeddings");
        side_model_flags.push("--embedding-model <path>");
    }
    if state.rerank_model.is_some() {
        routes.push("/v1/rerank");
        side_model_flags.push("--reranker-model <path>");
    }
    if state.realtime_engine.is_some() {
        routes.push(realtime::REALTIME_PATH);
    }
    if let Some(audio) = state.audio_model.as_ref() {
        if audio.supports(AudioModelKind::Stt) {
            routes.push("/v1/audio/transcriptions");
            routes.push("/v1/audio/translations");
        }
        if audio.supports(AudioModelKind::Tts) {
            routes.push("/v1/audio/speech");
        }
    }

    if routes.is_empty() {
        return Some(
            "This server has no loaded chat model; check the startup log for the load failure."
                .to_string(),
        );
    }

    let guidance = if side_model_flags.is_empty() {
        "Start a separate server with -m <chat model> to serve generation.".to_string()
    } else {
        format!(
            "Start with -m <chat model> {} to serve chat and these side-model routes.",
            side_model_flags.join(" ")
        )
    };
    Some(format!(
        "This server has no loaded chat model; it serves {}. {guidance}",
        routes.join(", ")
    ))
}

/// Return the OpenAI-compatible capability response for a terminal no-chat
/// provider state.
pub(crate) fn chat_not_available(state: &AppState) -> Option<ErrorResponse> {
    chat_unavailable_message(state).map(ErrorResponse::not_implemented)
}

/// The same refusal, but only on a server started without a chat tokenizer:
/// image-only serving (#2825) and a primary Kokoro TTS checkpoint.
///
/// `/tokenize`, `/detokenize` and `/apply-template` are not generation routes,
/// so they stay available in every other mode, including `-m --image-model`.
/// In these two modes there is no tokenizer or chat template behind them, and
/// running on the empty tokenizer would return wrong answers rather than an
/// error. A TTS provider exists only for a primary Kokoro checkpoint.
pub(crate) fn chat_not_available_without_tokenizer(state: &AppState) -> Option<ErrorResponse> {
    let tts_only = state
        .audio_model
        .as_ref()
        .is_some_and(|audio| audio.supports(AudioModelKind::Tts));
    if state.config.embedding_serving_mode == crate::server::config::EmbeddingServingMode::ImageOnly
        || tts_only
    {
        chat_not_available(state)
    } else {
        None
    }
}

/// The status an API error carries for a rejected `Json` request body.
///
/// A body that is valid JSON but does not match the request type is a 400,
/// like every other invalid request, instead of axum's 422. The other statuses
/// are kept: 400 for malformed or empty JSON, 413 for a body over the size
/// limit, and 415 for a missing or non-JSON `Content-Type`.
///
/// Used by: [`parse_json_request`] (OpenAI envelope) and
/// `anthropic::parse_anthropic_request` (Anthropic envelope), so both
/// envelopes answer one rejection with one status.
pub(crate) fn json_rejection_status(rejection: &JsonRejection) -> StatusCode {
    match rejection.status() {
        StatusCode::UNPROCESSABLE_ENTITY => StatusCode::BAD_REQUEST,
        status => status,
    }
}

/// Longest [`json_rejection_message`], in bytes, before the truncation marker.
const MAX_JSON_REJECTION_MESSAGE_BYTES: usize = 1024;

/// The client-facing description of a rejected `Json` request body.
///
/// axum's text quotes serde's error, and serde quotes the offending value, so
/// a long unknown enum tag or map key would otherwise come back whole, up to
/// the request body limit (which multimodal configurations raise to GiB) and
/// larger still once escaped as JSON. The error kind comes before the quoted
/// value, so the bound keeps it.
///
/// Used by: [`parse_json_request`] (OpenAI envelope) and
/// `anthropic::parse_anthropic_request` (Anthropic envelope).
pub(crate) fn json_rejection_message(rejection: &JsonRejection) -> String {
    let mut message = rejection.body_text();
    if message.len() > MAX_JSON_REJECTION_MESSAGE_BYTES {
        message.truncate(message.floor_char_boundary(MAX_JSON_REJECTION_MESSAGE_BYTES));
        message.push_str("...");
    }
    message
}

/// Unwrap a typed JSON request body, or answer its rejection as an
/// OpenAI-envelope `invalid_request_error` whose message is
/// [`json_rejection_message`] and whose status comes from
/// [`json_rejection_status`].
///
/// Used by: the single-node `/v1/chat/completions`, `/v1/completions`, and
/// `/v1/responses` handlers (which also serve the `/chat/completions` and
/// `/responses` aliases), and the disaggregated router's
/// `/v1/chat/completions` and `/v1/completions`.
pub(crate) fn parse_json_request<T>(
    request: Result<Json<T>, JsonRejection>,
) -> Result<T, ErrorResponse> {
    request.map(|Json(request)| request).map_err(|rejection| {
        let mut error =
            ErrorResponse::new(json_rejection_message(&rejection), "invalid_request_error");
        error.status = json_rejection_status(&rejection);
        error
    })
}

/// Map generation dispatch failures consistently across OpenAI-compatible
/// routes.
// Used by: all serving models, all OpenAI-compatible generation routes.
pub(crate) fn generation_error_to_response(err: anyhow::Error) -> ErrorResponse {
    generation_error_response(&err)
}

/// [`generation_error_to_response`] for a caller that keeps the error, such as
/// a streaming route reporting a failure as an in-stream frame (#2877), so
/// both surfaces classify and word the same failure identically.
pub(crate) fn generation_error_response(err: &anyhow::Error) -> ErrorResponse {
    if let Some(rejection) =
        err.downcast_ref::<crate::execution::request_memory::RequestMemoryRejection>()
    {
        let mut response = ErrorResponse::new(
            rejection.to_string(),
            crate::execution::request_memory::RequestMemoryRejection::CODE,
        );
        response.error.code =
            Some(crate::execution::request_memory::RequestMemoryRejection::CODE.to_string());
        response.error.estimated_bytes = Some(rejection.estimated_bytes);
        response.error.budget_bytes = Some(rejection.budget_bytes);
        response.status = axum::http::StatusCode::SERVICE_UNAVAILABLE;
        return response;
    }

    if err.downcast_ref::<QueueFullError>().is_some() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.");
    }

    // b10621 context admission (#1472): the scheduler refuses a prompt that
    // does not fit the per-slot KV bound with upstream's exceed-context
    // wording, and upstream classifies that as ERROR_TYPE_EXCEED_CONTEXT_SIZE
    // (400, "exceed_context_size_error"; server-common.cpp), not as a server
    // error. The scheduler hands errors up as strings through
    // GenerateEvent::Error, so the classification keys on the one wording the
    // admission guard emits.
    let message = err.to_string();
    if message.contains("exceeds the available context size") {
        let mut response = ErrorResponse::new(message, "exceed_context_size_error");
        response.status = axum::http::StatusCode::BAD_REQUEST;
        // The server's own overflow class with OpenAI's code, so either kind
        // of client can recognize it.
        response.error.code =
            Some(crate::server::responses_compaction::CONTEXT_LENGTH_EXCEEDED.to_string());
        return response;
    }

    let mut response = if err.downcast_ref::<ChatWorkerGoneError>().is_some() {
        ErrorResponse::new(
            "The chat worker has exited; check the server log.",
            "server_error",
        )
    } else {
        ErrorResponse::new(format!("Generation error: {err}"), "server_error")
    };
    response.status = if err.downcast_ref::<ChatWorkerGoneError>().is_some() {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    } else {
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    };
    response
}

#[cfg(test)]
#[path = "native_route_tests.rs"]
mod native_route_tests;

#[cfg(test)]
#[path = "embedding_rerank_mode_tests.rs"]
mod embedding_rerank_mode_tests;
