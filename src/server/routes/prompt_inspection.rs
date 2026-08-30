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
//! by `encode(prompt, /*add_special=*/true)`. A separate estimator would drift
//! from the prompt the server really builds, which is the failure this endpoint
//! exists to prevent. `POST /v1/messages/count_tokens` answers the same
//! `input_tokens` key for the Anthropic body shape and lives with the rest of
//! that surface in `routes/anthropic.rs`.
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server.cpp>

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::server::chat_request::prepare_chat_request_with_cache;
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
/// Returns the rendered prompt or a 400-shaped error. The tool guard runs
/// first, as it does on `/v1/chat/completions` and `/v1/messages/count_tokens`,
/// so an oversized tool array cannot reach the Jinja2 renderer through a route
/// that does not generate.
async fn render_chat_prompt(
    state: &AppState,
    live: &LiveSettings,
    request: &ChatCompletionRequest,
) -> Result<String, ErrorResponse> {
    if let Err(message) = super::chat::validate_chat_tool_inputs(request) {
        return Err(ErrorResponse::new(message, "invalid_request_error"));
    }
    let prompt_cache_enabled = state.prompt_cache.is_some();
    prepare_chat_request_with_cache(
        &state.chat_template,
        request,
        live.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
        state.should_render_history_boundary_snapshot(),
        state.prefill_assistant(),
    )
    .await
    .map(|prepared| prepared.prompt)
    .map_err(|err| ErrorResponse::new(err.to_string(), "invalid_request_error"))
}

/// Count the tokens a rendered prompt occupies.
///
/// `add_special` is `true` so the count includes the BOS the generation path
/// adds, which is what makes the number comparable to `tokens_evaluated`.
fn count_prompt_tokens(state: &AppState, prompt: &str) -> Result<usize, ErrorResponse> {
    state
        .tokenizer
        .encode(prompt, true)
        .map(|ids| ids.len())
        .map_err(|e| {
            ErrorResponse::new(format!("Tokenization error: {e}"), "invalid_request_error")
        })
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
    match render_chat_prompt(&state, &live, &request).await {
        Ok(prompt) => (
            StatusCode::OK,
            Json(serde_json::json!({ "prompt": prompt })),
        )
            .into_response(),
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
    let prompt = match render_chat_prompt(&state, &live, &request).await {
        Ok(prompt) => prompt,
        Err(err) => return err.into_response(),
    };
    match count_prompt_tokens(&state, &prompt) {
        Ok(count) => (
            StatusCode::OK,
            Json(serde_json::json!({ "input_tokens": count })),
        )
            .into_response(),
        Err(err) => err.into_response(),
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
    let prompt = match render_chat_prompt(&state, &live, &translated.chat_request).await {
        Ok(prompt) => prompt,
        Err(err) => return err.into_response(),
    };
    match count_prompt_tokens(&state, &prompt) {
        Ok(count) => (
            StatusCode::OK,
            Json(serde_json::json!({ "input_tokens": count })),
        )
            .into_response(),
        Err(err) => err.into_response(),
    }
}

#[cfg(test)]
#[path = "prompt_inspection_tests.rs"]
mod prompt_inspection_tests;
