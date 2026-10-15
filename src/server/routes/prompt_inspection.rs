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

//! Prompt-inspection routes: render a prompt, and count it, without generating
//! from it (issue #1442).
//!
//! Three b10621 surfaces share one property: they take a request body the
//! generation routes already accept, run it through the chat template, and
//! answer with something about the resulting prompt instead of a completion.
//!
//! | Route | Body | Answer |
//! |---|---|---|
//! | `POST /apply-template` | chat completions | `{"prompt": "..."}` |
//! | `POST /chat/completions/input_tokens`, `POST /v1/chat/completions/input_tokens` | chat completions | `{"input_tokens": N}` |
//! | `POST /responses/input_tokens`, `POST /v1/responses/input_tokens` | responses | `{"input_tokens": N}` |
//!
//! The count is the number of tokens the same body would actually have been
//! prefilled with, because it runs the identical render and the identical
//! encode the generation path runs: `prepare_chat_request_with_cache` followed
//! by the generation path's own `add_special` rule. A separate estimator would
//! drift from the prompt the server really builds, which is the failure this
//! endpoint exists to prevent. A body carrying images, clips or audio is
//! counted on the worker, through the same media expansion a generation runs,
//! because each media part becomes its soft tokens only there (issue #1788):
//! the answer equals that body's `usage.prompt_tokens`.
//! `POST /v1/messages/count_tokens` answers the same `input_tokens` key for
//! the Anthropic body shape, lives with the rest of that surface in
//! `routes/anthropic.rs`, and counts through [`count_prompt_tokens`] here.
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server.cpp>

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::server::chat_request::{
    PreparedChatRequest, prepare_chat_request_with_cache_and_video_allowlist,
};
use crate::server::request_options::{
    RequestOptionOverrides, build_server_generate_options_with_live,
};
use crate::server::responses_translator::responses_request_to_chat;
use crate::server::types::{ChatCompletionRequest, CreateResponseRequest, ErrorResponse};
use crate::server::{AppState, LiveSettings};

/// Deserialize a request body that b10621 does not require a `model` on.
///
/// `/v1/chat/completions` and `/v1/responses` follow the OpenAI schema, where
/// `model` is mandatory, and mlxcel's request types say so. The three routes in
/// this module are b10621's own, and it asks for messages or input and nothing
/// else, so a client written against `llama-server` sends no `model` at all.
/// Filling in the loaded model's id before deserializing keeps that client
/// working without loosening the schema of the generating routes, and a body
/// that does name a model is passed through unchanged.
fn parse_with_default_model<T: serde::de::DeserializeOwned>(
    state: &AppState,
    mut body: serde_json::Value,
) -> Result<T, ErrorResponse> {
    if let Some(object) = body.as_object_mut()
        && !object.contains_key("model")
    {
        object.insert(
            "model".to_string(),
            serde_json::Value::String(state.display_model_id().to_string()),
        );
    }
    serde_json::from_value(body).map_err(|err| {
        ErrorResponse::new(format!("Invalid request: {err}"), "invalid_request_error")
    })
}

/// Render one chat-completions body into the prompt the generation path would
/// prefill.
///
/// Returns the prepared request (prompt plus resolved media) or an error
/// response. The media-capability gate and the tool guard run in the order
/// `/v1/chat/completions` runs them, so neither an image the checkpoint has no
/// tower for nor an oversized tool array can reach the Jinja2 renderer through
/// a route that does not generate.
///
/// The media gate is not only about answering consistently. `prepare_chat_request_with_cache`
/// downloads every `image_url` and `input_audio` payload and opens every
/// `video_url` as part of rendering, and these routes never generate, so
/// without a gate here a text-only checkpoint would fetch a URL a client named
/// and report nothing but a token count for it. The generating routes have
/// refused that at the boundary since issue #1451; this closes the same hole
/// on the three routes that only inspect (issue #1349 moved the video+audio
/// arm of that check here too, so the combination is covered by the same
/// call).
async fn render_chat_prompt(
    state: &AppState,
    live: &LiveSettings,
    request: ChatCompletionRequest,
) -> Result<PreparedChatRequest, ErrorResponse> {
    if let Some(rejection) = crate::server::media_capability_rejection(
        &request,
        state.media_support,
        state.display_model_id(),
    ) {
        return Err(rejection);
    }
    if let Err(message) = super::chat::validate_chat_tool_inputs(&request) {
        return Err(ErrorResponse::new(message, "invalid_request_error"));
    }
    render_admitted_chat_prompt(state, live, request).await
}

/// Render a request that has already passed the media-capability gate and the
/// tool guard, the way the generating route renders it.
///
/// Split out of [`render_chat_prompt`] for `/v1/messages/count_tokens`, which
/// runs both checks itself so their refusals keep the Anthropic error envelope
/// (issue #1788).
pub(crate) async fn render_admitted_chat_prompt(
    state: &AppState,
    live: &LiveSettings,
    mut request: ChatCompletionRequest,
) -> Result<PreparedChatRequest, ErrorResponse> {
    // These routes exist to answer "what prompt would the generating route
    // build for this body?", so they have to run the same video-to-frames
    // substitution the generating route does (issue #1322); otherwise the
    // reported prompt would carry no image placeholders for a clip that
    // /v1/chat/completions expands into as many as sixteen. Every caller owns
    // the request it passes and none of them generates from it, so it is
    // rewritten in place: a copy would duplicate every base64 image payload
    // and tool definition in the body just to change its messages (issue
    // #1766).
    crate::server::chat_request::expand_request_video_parts(state, &mut request)
        .await
        .map_err(crate::server::chat_request::VideoFramesError::into_error_response)?;
    let prompt_cache_enabled = state.prompt_cache.is_some();
    prepare_chat_request_with_cache_and_video_allowlist(
        &state.chat_template,
        &request,
        live.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
        state.should_render_history_boundary_snapshot(),
        state.prefill_assistant(),
        &state.thinking_markers,
        &state.video_dir_allowlist,
    )
    .await
    .map_err(|err| ErrorResponse::new(err.to_string(), "invalid_request_error"))
}

/// Why [`count_prompt_tokens`] produced no count.
#[derive(Debug)]
pub(crate) enum PromptCountError {
    /// The rendered prompt could not be encoded: a request error.
    Tokenization(String),
    /// The worker could not count a media request: busy, gone, or refusing
    /// the media on preparation. Each route renders it the way its generating
    /// sibling renders a generation failure.
    Worker(anyhow::Error),
}

impl PromptCountError {
    /// The OpenAI-envelope response for this failure.
    fn into_error_response(self) -> ErrorResponse {
        match self {
            Self::Tokenization(message) => ErrorResponse::new(
                format!("Tokenization error: {message}"),
                "invalid_request_error",
            ),
            Self::Worker(err) => super::generation_error_to_response(err),
        }
    }
}

/// Count the tokens a prepared request would be prefilled with.
///
/// Shared by every counting route (`/chat/completions/input_tokens`,
/// `/responses/input_tokens` and `/v1/messages/count_tokens`), so they cannot
/// drift from each other or from `usage.prompt_tokens`.
///
/// A text-only prompt is counted here, on the request thread. A native
/// renderer's ids (#1338) are authoritative, so they are counted directly;
/// otherwise `add_special` follows `prompt_carries_bos`, the same rule the
/// generation path uses, which is what makes the number comparable to
/// `tokens_evaluated`. Passing `true` unconditionally over-counted by one for
/// every template that emits its own BOS (Laguna, and the whole Llama 3
/// lineage since #1347).
///
/// A prompt carrying resolved media is not countable from its text: each
/// image, clip or audio part is one placeholder there and becomes its soft
/// tokens only on the worker (issue #1788). Those requests go to the worker as
/// a count-only request, which runs the generation path's own tokenization and
/// media preparation and stops before prefill, so the answer is the
/// `prompt_tokens` the same body reports when it generates. The media bytes
/// this function receives are the ones the render already fetched; nothing is
/// read a second time.
pub(crate) async fn count_prompt_tokens(
    state: &AppState,
    live: &Arc<LiveSettings>,
    prepared: PreparedChatRequest,
) -> Result<usize, PromptCountError> {
    let carries_media = !prepared.image_data.is_empty()
        || !prepared.audio_data.is_empty()
        || !prepared.videos.is_empty();
    if !carries_media {
        if let Some(ids) = prepared.prompt_token_ids.as_ref() {
            return Ok(ids.len());
        }
        return state
            .tokenizer
            .encode(
                &prepared.prompt,
                !state.tokenizer.prompt_carries_bos(&prepared.prompt),
            )
            .map(|ids| ids.len())
            .map_err(|e| PromptCountError::Tokenization(e.to_string()));
    }

    let mut options = build_server_generate_options_with_live(
        &state.config,
        live,
        RequestOptionOverrides::default(),
    );
    options.image_soft_tokens = prepared.image_soft_tokens;
    options.pre_rendered_prompt_tokens = prepared.prompt_token_ids;
    let provider = Arc::clone(&state.model_provider);
    let live = Arc::clone(live);
    tokio::task::spawn_blocking(move || {
        provider.count_prompt_tokens_with_media(
            prepared.prompt,
            options,
            prepared.image_data,
            prepared.audio_data,
            prepared.videos,
            prepared.media,
            &live,
        )
    })
    .await
    .map_err(|err| PromptCountError::Worker(anyhow::anyhow!("prompt count task failed: {err}")))?
    .map_err(PromptCountError::Worker)
}

/// POST /apply-template
///
/// Renders the request through the loaded chat template and answers
/// `{"prompt": "..."}` without generating. Messages, tools, reasoning options
/// and `chat_template_kwargs` are honored because the render is the same call
/// `/v1/chat/completions` makes, so what comes back is the prompt that request
/// would have been served with, not a re-derivation of it.
pub async fn apply_template(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let live = state.live();
    let request: ChatCompletionRequest = match parse_with_default_model(&state, body) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    match render_chat_prompt(&state, &live, request).await {
        // A native renderer's id count is reported alongside the text so an
        // operator inspecting a Kimi K3 prompt sees the length the model
        // actually prefills, which is not what re-encoding the text gives.
        Ok(prepared) => {
            let mut body = serde_json::json!({ "prompt": prepared.prompt });
            if let Some(ids) = prepared.prompt_token_ids.as_ref()
                && let Some(object) = body.as_object_mut()
            {
                object.insert(
                    "prompt_token_count".to_string(),
                    serde_json::Value::from(ids.len()),
                );
            }
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(err) => err.into_response(),
    }
}

/// POST /chat/completions/input_tokens and POST /v1/chat/completions/input_tokens
pub async fn chat_input_tokens(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let live = state.live();
    let request: ChatCompletionRequest = match parse_with_default_model(&state, body) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    let prepared = match render_chat_prompt(&state, &live, request).await {
        Ok(prepared) => prepared,
        Err(err) => return err.into_response(),
    };
    match count_prompt_tokens(&state, &live, prepared).await {
        Ok(count) => (
            StatusCode::OK,
            Json(serde_json::json!({ "input_tokens": count })),
        )
            .into_response(),
        Err(err) => err.into_error_response().into_response(),
    }
}

/// POST /responses/input_tokens and POST /v1/responses/input_tokens
///
/// The Responses body is flattened to the chat shape by the same translator
/// `/v1/responses` uses, including `previous_response_id` and `conversation`
/// rehydration, so the count covers the whole conversation the generation call
/// would have prefilled rather than only the new turn.
pub async fn responses_input_tokens(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let live = state.live();
    let request: CreateResponseRequest = match parse_with_default_model(&state, body) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    let translated = match responses_request_to_chat(
        &request,
        state.responses_store.as_ref(),
        state.conversation_store.as_ref(),
    ) {
        Ok(translated) => translated,
        Err(err) => {
            return ErrorResponse::new(err.to_string(), "invalid_request_error").into_response();
        }
    };
    let prepared = match render_chat_prompt(&state, &live, translated.chat_request).await {
        Ok(prepared) => prepared,
        Err(err) => return err.into_response(),
    };
    match count_prompt_tokens(&state, &live, prepared).await {
        Ok(count) => (
            StatusCode::OK,
            Json(serde_json::json!({ "input_tokens": count })),
        )
            .into_response(),
        Err(err) => err.into_error_response().into_response(),
    }
}

#[cfg(test)]
#[path = "prompt_inspection_tests.rs"]
mod prompt_inspection_tests;

#[cfg(test)]
#[path = "prompt_count_route_tests.rs"]
mod prompt_count_route_tests;
