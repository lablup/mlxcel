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

//! Fill-in-the-middle endpoint (`POST /infill`, issue #1442).
//!
//! An HTTP adapter over two pieces that live elsewhere: the FIM vocabulary gate
//! in [`crate::tokenizer::fim`] and the prompt assembly in
//! [`crate::server::infill`]. Once the prompt is built the request is an
//! ordinary native completion, so it is handed to the same
//! `serve_native_completion` entry `POST /completion` uses rather than growing
//! a parallel generation path.
//!
//! The capability gate runs first and refuses a model whose vocabulary has no
//! prefix, suffix and middle tokens, naming the missing ones. That ordering is
//! upstream's and it matters: prompting a chat model with literal `<PRE>` text
//! produces a fluent completion that is silently wrong, which is exactly the
//! "accepted and ignored" failure epic #1431 exists to remove.
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server.cpp>

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use crate::server::AppState;
use crate::server::infill::{format_infill_prompt, parse_infill_inputs, reject_marker_injection};
use crate::server::types::{ErrorResponse, NativeCompletionRequest};

/// POST /infill
pub async fn infill(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<serde_json::Value>,
) -> Response {
    let live = state.live();
    if let Some(response) = super::chat_not_available(&state) {
        return response.into_response();
    }

    // Capability gate, before any request validation, as upstream orders it.
    let fim = state.tokenizer.fim_tokens();
    let triple = match fim.require_triple() {
        Ok(triple) => triple,
        Err(message) => return ErrorResponse::not_supported(message).into_response(),
    };

    let inputs = match parse_infill_inputs(&body) {
        Ok(inputs) => inputs,
        Err(message) => {
            return ErrorResponse::new(message, "invalid_request_error").into_response();
        }
    };

    if let Err(message) = reject_marker_injection(&fim, &inputs) {
        return ErrorResponse::new(message, "invalid_request_error").into_response();
    }

    let prompt = format_infill_prompt(&fim, &triple, &inputs, state.config.spm_infill);

    // The remaining fields are the `/completion` schema. Replace `prompt` with
    // the assembled one and drop the FIM-only keys so the completion request
    // deserializes from exactly what upstream hands its own shared handler.
    let Some(object) = body.as_object_mut() else {
        return ErrorResponse::new(
            "the request body must be a JSON object",
            "invalid_request_error",
        )
        .into_response();
    };
    object.insert("prompt".to_string(), serde_json::Value::String(prompt));
    object.remove("input_prefix");
    object.remove("input_suffix");
    object.remove("input_extra");

    let request: NativeCompletionRequest = match serde_json::from_value(body) {
        Ok(request) => request,
        Err(err) => {
            return ErrorResponse::new(
                format!("Invalid infill request: {err}"),
                "invalid_request_error",
            )
            .into_response();
        }
    };

    super::native_completion::serve_native_completion(state, live, &headers, request).await
}

#[cfg(test)]
#[path = "infill_route_tests.rs"]
mod infill_route_tests;
