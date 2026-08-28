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

//! b10621 resumable-stream routes (#1444): `GET /v1/stream`,
//! `POST /v1/streams/lookup`, `DELETE /v1/stream`.
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/main/tools/server/server-stream.cpp>.
//! A streaming completion that carried `X-Conversation-Id` buffered its SSE
//! bytes in a [`crate::server::stream_session::StreamSession`]; these routes
//! replay, discover, and stop those sessions. Status codes and error wording
//! follow the pinned b10621 handlers:
//!
//! - `GET /v1/stream?conv_id=<id>&from=N` — 400 without `conv_id`, 404 for an
//!   unknown or expired session, 400 for an unparsable `from`, 400 "offset
//!   lost" when `from` fell below the dropped buffer prefix, otherwise a
//!   `text/event-stream` replay that keeps following live output until the
//!   session finalizes.
//! - `POST /v1/streams/lookup` `{"conversation_ids": [...]}` — 200 with one
//!   status object per matching session; only ids the caller asked about
//!   (and, in mlxcel, owns) are answered, so the endpoint cannot enumerate
//!   other clients' sessions.
//! - `DELETE /v1/stream?conv_id=<id>` — idempotent 204; cancels the
//!   generation and evicts the buffer.
//!
//! Ownership (mlxcel hardening, recorded in the manifest entry notes): with
//! API keys configured, sessions are scoped to the key that created them.
//! Another key's `GET` answers the same 404 as an unknown id, its lookup
//! omits the session, and its `DELETE` is the same 204 no-op, so none of the
//! three is an existence oracle across keys.

use axum::{
    body::{Body, Bytes},
    extract::{Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;

use crate::server::AppState;
use crate::server::auth;
use crate::server::stream_session::{ReadChunk, StreamOwner};
use crate::server::types::ErrorResponse;

/// The owner identity for stream sessions and completion control: the
/// presented API key when authentication is configured, `None` when it is
/// disabled. The authentication middleware has already validated the
/// credential by the time any handler runs.
pub(crate) fn request_stream_owner(state: &AppState, headers: &HeaderMap) -> StreamOwner {
    if state.config.api_keys.is_empty() {
        return None;
    }
    auth::presented_credential(headers).map(str::to_string)
}

/// The `X-Conversation-Id` header value, if present and non-empty. b10621
/// attaches a resumable session to any streaming completion that carries it.
pub(crate) fn conversation_id_from_headers(headers: &HeaderMap) -> Option<String> {
    let value = headers.get("x-conversation-id")?.to_str().ok()?;
    if value.is_empty() {
        return None;
    }
    Some(value.to_string())
}

fn error_with_status(status: StatusCode, message: &str) -> Response {
    let mut err = ErrorResponse::new(message, error_type_for(status));
    err.status = status;
    err.into_response()
}

/// b10621 `error_type_to_str` for the two statuses these routes produce.
fn error_type_for(status: StatusCode) -> &'static str {
    match status {
        StatusCode::NOT_FOUND => "not_found_error",
        _ => "invalid_request_error",
    }
}

#[derive(Debug, Deserialize)]
pub struct StreamQuery {
    conv_id: Option<String>,
    from: Option<String>,
}

/// Parse the `from` offset the way b10621's `std::stoull` does: optional
/// leading whitespace and `+`, then a digit run. No digits, or overflow, is
/// the parse failure that answers 400.
fn parse_from_offset(raw: &str) -> Option<u64> {
    let t = raw.trim_start();
    let t = t.strip_prefix('+').unwrap_or(t);
    let digits: &str = {
        let end = t
            .char_indices()
            .find(|(_, c)| !c.is_ascii_digit())
            .map(|(i, _)| i)
            .unwrap_or(t.len());
        &t[..end]
    };
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// `GET /v1/stream?conv_id=<id>&from=N`
pub async fn stream_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<StreamQuery>,
) -> Response {
    let Some(conv_id) = query.conv_id.filter(|c| !c.is_empty()) else {
        return error_with_status(StatusCode::BAD_REQUEST, "Missing conversation id in path");
    };
    let owner = request_stream_owner(&state, &headers);
    let Some(session) = state.stream_sessions.get(&conv_id, &owner) else {
        return error_with_status(StatusCode::NOT_FOUND, "Stream not found or expired");
    };
    let from = match &query.from {
        None => 0,
        Some(raw) if raw.is_empty() => 0,
        Some(raw) => match parse_from_offset(raw) {
            Some(v) => v as usize,
            None => {
                return error_with_status(StatusCode::BAD_REQUEST, "Invalid 'from' offset");
            }
        },
    };
    if from < session.dropped_prefix() {
        return error_with_status(
            StatusCode::BAD_REQUEST,
            "Stream offset lost, please restart",
        );
    }

    // Replay buffered bytes from `from`, then follow live output until the
    // session finalizes. The bytes were stored in their on-wire SSE framing,
    // so they are streamed raw rather than re-wrapped in new SSE events.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::convert::Infallible>>(8);
    tokio::spawn(async move {
        let mut offset = from;
        loop {
            match session.read_chunk(offset) {
                ReadChunk::Data(data) => {
                    offset += data.len();
                    if tx.send(Ok(Bytes::from(data))).await.is_err() {
                        // Replay client disconnected; the session itself is
                        // untouched (consumers never finalize it).
                        return;
                    }
                }
                ReadChunk::Pending => {
                    if tx.is_closed() {
                        return;
                    }
                    session.wait_for_change().await;
                }
                // A slow replay that fell behind the dropped prefix ends the
                // stream, as upstream's OFFSET_LOST does mid-read; the client
                // restarts with a fresh GET and sees the 400.
                ReadChunk::Eof | ReadChunk::OffsetLost => return,
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// `POST /v1/streams/lookup` with body `{"conversation_ids": [...]}`.
pub async fn streams_lookup(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            // b10621 answers a manual 400 envelope naming the parse error.
            let mut err = ErrorResponse::new(format!("invalid body: {e}"), "invalid_request_error");
            err.status = StatusCode::BAD_REQUEST;
            return err.into_response();
        }
    };
    let requested: Vec<String> = parsed
        .get("conversation_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let owner = request_stream_owner(&state, &headers);
    let sessions = state.stream_sessions.lookup(&requested, &owner);
    let out: Vec<serde_json::Value> = sessions
        .iter()
        .map(|s| {
            serde_json::json!({
                "conversation_id": s.conversation_id(),
                "is_done": s.is_done(),
                "total_bytes": s.total_size(),
                "started_at": s.started_at(),
                "completed_at": s.completed_at(),
            })
        })
        .collect();
    axum::Json(out).into_response()
}

/// `DELETE /v1/stream?conv_id=<id>` — the explicit user Stop. Cancels the
/// generation through the session's shared cancellation token and evicts the
/// buffer. Idempotent: 204 even when the session was already gone.
pub async fn stream_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<StreamQuery>,
) -> Response {
    let Some(conv_id) = query.conv_id.filter(|c| !c.is_empty()) else {
        return error_with_status(StatusCode::BAD_REQUEST, "Missing conversation id in path");
    };
    let owner = request_stream_owner(&state, &headers);
    state.stream_sessions.evict_and_cancel(&conv_id, &owner);
    StatusCode::NO_CONTENT.into_response()
}
