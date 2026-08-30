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

//! Thin HTTP route adapters.
//!
//! These files should stay focused on request/response translation. Shared
//! policy belongs in `server/request_options.rs`, `server/chat_request.rs`,
//! `server/media.rs`, `server/streaming.rs`, and `server/model_worker.rs`.
use crate::server::AppState;
use crate::server::audio_model::AudioModelKind;
use crate::server::model_provider::{ChatWorkerGoneError, QueueFullError};
use crate::server::types::ErrorResponse;

pub mod anthropic;
pub mod audio;
pub mod cache;
pub mod chat;
pub mod completions;
pub mod control;
pub mod detokenize;
pub mod embeddings;
pub mod feature_disabled;
pub mod health;
pub mod infill;
pub mod lora_adapters;
pub mod metrics;
pub mod models;
pub mod mtp_policy;
pub mod native_completion;
pub mod prompt_inspection;
pub mod props;
pub mod rerank;
pub mod responses;
pub mod settings;
pub mod slots;
pub mod stream;
pub mod tokenize;
pub mod transcription_compat;

#[cfg(test)]
#[path = "availability_tests.rs"]
mod availability_tests;

#[cfg(test)]
#[path = "stream_route_tests.rs"]
mod stream_route_tests;

#[cfg(test)]
#[path = "feature_disabled_tests.rs"]
mod feature_disabled_tests;

#[cfg(test)]
#[path = "settings_tests.rs"]
mod settings_tests;

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
pub use settings::{get_settings, patch_settings};
pub use slots::{slot_action, slots};
pub use stream::{stream_delete, stream_get, streams_lookup};
pub use tokenize::tokenize;

/// Explain a terminal generation-unavailable state without exposing worker or
/// channel implementation details.
pub(crate) fn chat_unavailable_message(state: &AppState) -> Option<String> {
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

/// Map generation dispatch failures consistently across OpenAI-compatible
/// routes.
pub(crate) fn generation_error_to_response(err: anyhow::Error) -> ErrorResponse {
    if err.downcast_ref::<QueueFullError>().is_some() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.");
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
