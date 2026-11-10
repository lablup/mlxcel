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

//! The SSE readers of a streamed background response (issue #2900): the
//! create call of a `background: true`, `stream: true` request, and
//! `GET /v1/responses/{id}?stream=true[&starting_after=N]`.
//!
//! Both read the job's
//! [`BackgroundEventLog`](crate::server::responses_background_events::BackgroundEventLog)
//! through an independent cursor. A reader holds nothing but that cursor, so
//! a client that disconnects ends its own stream only: the job and every
//! other reader carry on, and the job's cancellation token is never touched.

use std::sync::Arc;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::server::AppState;
use crate::server::responses_background_events::{
    BackgroundEventLog, CursorError, evicted_cursor_message, log_not_retained_message, tail_events,
};
use crate::server::responses_store::ResponsesStore;
use crate::server::streaming::sse_response;
use crate::server::streaming_responses::ResponseSseKeepAlive;
use crate::server::types::ErrorResponse;
use crate::server::types::responses_response::ResponseObject;

/// Query parameters of `GET /v1/responses/{id}`. Both are kept as text so a
/// bad value gets the JSON error body instead of the extractor's plain text.
#[derive(Debug, Default, Deserialize)]
pub struct RetrieveResponseQuery {
    /// `true` asks for the event stream instead of the JSON object.
    #[serde(default)]
    stream: Option<String>,
    /// The last `sequence_number` the client received; read only with
    /// `stream=true`.
    #[serde(default)]
    starting_after: Option<String>,
}

/// The SSE response over `log` from sequence number `next`, with the
/// server's keepalive ping (`--sse-ping-interval`).
fn stream_response(
    state: &AppState,
    log: Arc<BackgroundEventLog>,
    response_id: &str,
    next: u64,
) -> Response {
    sse_response(
        tail_events(log, response_id.to_string(), next),
        ResponseSseKeepAlive::from_interval(state.config.sse_ping_interval),
    )
}

/// Answer a streamed background create: the job's events from the first one.
///
/// Used by: `create_background_response`.
pub(super) fn stream_from_start(state: &AppState, queued: ResponseObject) -> Response {
    let log = state
        .responses_store
        .as_ref()
        .and_then(|store| store.background_event_log(&queued.id));
    match log {
        Some(log) => stream_response(state, log, &queued.id, 0),
        // Only when the job already finished and its entry was evicted
        // before this line ran: the object is all there is left to answer.
        None => Json(queued).into_response(),
    }
}

/// Answer `GET /v1/responses/{id}` when it asks for the event stream.
/// `None` when it does not (`stream` absent or false): the caller then
/// returns the JSON object as before.
///
/// Used by: `retrieve_response`.
pub(super) fn resume_stream(
    state: &AppState,
    store: &ResponsesStore,
    id: &str,
    query: &RetrieveResponseQuery,
) -> Option<Response> {
    match query.stream.as_deref() {
        None | Some("false" | "0" | "") => return None,
        Some("true" | "1") => {}
        Some(other) => {
            return Some(
                invalid_request(format!("stream must be true or false, got '{other}'"))
                    .into_response(),
            );
        }
    }
    let Some(log) = store.background_event_log(id) else {
        return Some(if store.background_event_log_dropped(id) {
            invalid_request(log_not_retained_message(id)).into_response()
        } else if store.background_snapshot(id).is_some() {
            invalid_request(format!(
                "response '{id}' has no event stream: it was not created with background: true \
                 and stream: true. Retrieve it with GET /v1/responses/{id} instead"
            ))
            .into_response()
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse::new(
                    format!("response '{id}' not found"),
                    "not_found",
                )),
            )
                .into_response()
        });
    };
    let starting_after = match query.starting_after.as_deref() {
        None => None,
        Some(text) => match text.parse::<u64>() {
            Ok(sequence) => Some(sequence),
            Err(_) => {
                return Some(
                    invalid_request(format!(
                        "starting_after must be a sequence_number (a non-negative integer), \
                         got '{text}'"
                    ))
                    .into_response(),
                );
            }
        },
    };
    Some(match log.resolve_cursor(starting_after) {
        Ok(next) => stream_response(state, log, id, next),
        Err(CursorError::Evicted { oldest }) => {
            invalid_request(evicted_cursor_message(id, oldest)).into_response()
        }
        Err(CursorError::Ahead { newest }) => {
            let newest = newest.map_or_else(
                || "no event was sent yet".to_string(),
                |newest| format!("the newest event has sequence_number {newest}"),
            );
            invalid_request(format!(
                "starting_after {} is beyond the events of response '{id}': {newest}",
                starting_after.unwrap_or_default()
            ))
            .into_response()
        }
    })
}

fn invalid_request(message: String) -> ErrorResponse {
    ErrorResponse::new(message, "invalid_request_error")
}
