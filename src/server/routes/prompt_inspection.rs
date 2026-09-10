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
/// Returns the rendered prompt or an error response. The media-capability gate
/// and the tool guard run in the order `/v1/chat/completions` runs them, so
/// neither an image the checkpoint has no tower for nor an oversized tool array
/// can reach the Jinja2 renderer through a route that does not generate.
///
/// The media gate is not only about answering consistently. `prepare_chat_request_with_cache`
/// downloads every `image_url` and `input_audio` payload and opens every
/// `video_url` as part of rendering, and these routes never reach a model
/// worker, so without a gate here a text-only checkpoint would fetch a URL a
/// client named and report nothing but a token count for it. The generating
/// routes have refused that at the boundary since issue #1451; this closes the
/// same hole on the three routes that only inspect (issue #1349 moved the
/// video+audio arm of that check here too, so the combination is covered by
/// the same call).
async fn render_chat_prompt(
    state: &AppState,
    live: &LiveSettings,
    mut request: ChatCompletionRequest,
) -> Result<RenderedPrompt, ErrorResponse> {
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
    prepare_chat_request_with_cache(
        &state.chat_template,
        &request,
        live.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
        state.should_render_history_boundary_snapshot(),
        state.prefill_assistant(),
        &state.thinking_markers,
    )
    .await
    .map(|prepared| RenderedPrompt {
        prompt: prepared.prompt,
        token_ids: prepared.prompt_token_ids,
    })
    .map_err(|err| ErrorResponse::new(err.to_string(), "invalid_request_error"))
}

/// One rendered prompt, plus the ids a native renderer produced for it.
///
/// Kimi K3's XTML renderer (#1338) emits ids directly, so `token_ids` is the
/// authoritative count for it and `prompt` is the text form the generation
/// path also carries. `token_ids` is `None` for every template-rendered
/// request, and the count then comes from encoding `prompt` exactly as before.
struct RenderedPrompt {
    prompt: String,
    token_ids: Option<Vec<i32>>,
}

/// Count the tokens a rendered prompt occupies.
///
/// A native renderer's ids (#1338) are authoritative, so they are counted
/// directly. Otherwise `add_special` follows `prompt_carries_bos`, the same
/// rule the generation path uses, which is what makes the number comparable to
/// `tokens_evaluated`. Passing `true` unconditionally over-counted by one for
/// every template that emits its own BOS (Laguna, and the whole Llama 3
/// lineage since #1347).
fn count_prompt_tokens(
    state: &AppState,
    rendered: &RenderedPrompt,
) -> Result<usize, ErrorResponse> {
    if let Some(ids) = rendered.token_ids.as_ref() {
        return Ok(ids.len());
    }
    state
        .tokenizer
        .encode(
            &rendered.prompt,
            !state.tokenizer.prompt_carries_bos(&rendered.prompt),
        )
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
    match render_chat_prompt(&state, &live, request).await {
        // A native renderer's id count is reported alongside the text so an
        // operator inspecting a Kimi K3 prompt sees the length the model
        // actually prefills, which is not what re-encoding the text gives.
        Ok(rendered) => {
            let mut body = serde_json::json!({ "prompt": rendered.prompt });
            if let Some(ids) = rendered.token_ids.as_ref()
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
    let rendered = match render_chat_prompt(&state, &live, request).await {
        Ok(rendered) => rendered,
        Err(err) => return err.into_response(),
    };
    match count_prompt_tokens(&state, &rendered) {
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
    let rendered = match render_chat_prompt(&state, &live, translated.chat_request).await {
        Ok(rendered) => rendered,
        Err(err) => return err.into_response(),
    };
    match count_prompt_tokens(&state, &rendered) {
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
