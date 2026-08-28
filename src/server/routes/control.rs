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

//! `POST /v1/chat/completions/control` (b10621, #1444): realtime control of
//! a live completion.
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/main/tools/server/server-context.cpp>
//! (`post_control` and the `SERVER_TASK_TYPE_CONTROL` arm). The body carries
//! the completion `id` a streaming client learned from its first chunk plus
//! an `action`; `reasoning_end` is the only action b10621 defines, and it
//! requires the completion to have armed the `reasoning_control` request
//! field. Response contract, matched here:
//!
//! - missing/empty `id` — 400 `"missing completion id"`.
//! - any action other than `reasoning_end` — 400 `"unknown control action"`.
//! - no live completion with that id — 200
//!   `{"success": false, "message": "no active completion for this id"}`.
//! - live but unarmed — 200 `{"success": false, "message": "reasoning
//!   control not enabled for this completion"}`.
//! - armed — 200 `{"success": true}`; the sequence's thinking tracker closes
//!   the reasoning block at the next sampled token. Events already committed
//!   to the stream are unaffected, which is the defined event-order boundary.

use axum::{
    body::Bytes,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use crate::server::AppState;
use crate::server::completion_control::ControlOutcome;
use crate::server::types::ErrorResponse;

use super::stream::request_stream_owner;

/// `POST /v1/chat/completions/control`
pub async fn chat_completions_control(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return ErrorResponse::new(format!("invalid body: {e}"), "invalid_request_error")
                .into_response();
        }
    };
    let cmpl_id = parsed.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let action = parsed.get("action").and_then(|v| v.as_str()).unwrap_or("");
    if cmpl_id.is_empty() {
        return ErrorResponse::new("missing completion id", "invalid_request_error")
            .into_response();
    }
    if action != "reasoning_end" {
        return ErrorResponse::new("unknown control action", "invalid_request_error")
            .into_response();
    }

    let owner = request_stream_owner(&state, &headers);
    let body = match state.completion_controls.reasoning_end(cmpl_id, &owner) {
        ControlOutcome::Forced => serde_json::json!({ "success": true }),
        ControlOutcome::NotArmed => serde_json::json!({
            "success": false,
            "message": "reasoning control not enabled for this completion",
        }),
        ControlOutcome::NoActiveCompletion => serde_json::json!({
            "success": false,
            "message": "no active completion for this id",
        }),
    };
    axum::Json(body).into_response()
}
