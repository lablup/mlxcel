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

//! Shared WebUI SSE cursor parsing and response construction.

use std::sync::Arc;

use axum::Json;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};

use crate::server::router_lifecycle::{
    ErrorBody, ErrorEnvelope, FieldError, LifecycleCoordinator, MAX_SAFE_EVENT_SEQUENCE,
    ReplaySubscribeError, ResetEventKind, UiEvent, UiReplayCursor,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct UiEventCursorError {
    field: &'static str,
    code: &'static str,
    message: &'static str,
}

impl UiEventCursorError {
    const fn new(field: &'static str, code: &'static str, message: &'static str) -> Self {
        Self {
            field,
            code,
            message,
        }
    }

    fn into_response(self) -> Response {
        invalid_event_field(self.field, self.code, self.message)
    }
}

pub(crate) fn event_to_sse(event: UiEvent) -> Event {
    let id = event.event_id.clone();
    let name = event.event_type.clone();
    let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
    Event::default().id(id).event(name).data(data)
}

fn request_id() -> String {
    format!("req_{}", chrono::Utc::now().timestamp_micros())
}

fn invalid_event_field(
    field: &str,
    field_code: &str,
    field_message: impl Into<String>,
) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorEnvelope {
            error: ErrorBody {
                code: "invalid_request".to_string(),
                message: "request does not match the WebUI contract".to_string(),
                retryable: true,
                field_errors: Some(vec![FieldError {
                    field: field.to_string(),
                    code: field_code.to_string(),
                    message: field_message.into(),
                }]),
                operation_id: None,
            },
            request_id: request_id(),
        }),
    )
        .into_response()
}

fn is_webui_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'~' | b'-')
}

fn is_webui_token(value: &str, min: usize, max: usize) -> bool {
    let bytes = value.as_bytes();
    (min..=max).contains(&bytes.len()) && bytes.iter().copied().all(is_webui_token_byte)
}

pub(crate) fn last_event_id_from_headers(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("last-event-id")
        .or_else(|| headers.get("Last-Event-ID"))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
}

pub(crate) fn parse_ui_event_cursor(
    query: Option<&str>,
    last_event_id: Option<&str>,
) -> Result<UiReplayCursor, UiEventCursorError> {
    let Some(query) = query.filter(|query| !query.is_empty()) else {
        return Ok(match last_event_id {
            Some(value) => UiReplayCursor::LastEventId(value.to_string()),
            None => UiReplayCursor::Snapshot,
        });
    };
    let mut server_instance_id: Option<String> = None;
    let mut after_sequence: Option<u64> = None;
    for pair in query.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            return Err(UiEventCursorError::new(
                "events_query",
                "malformed",
                "event replay query parameters must be key=value pairs",
            ));
        };
        match key {
            "server_instance_id" => {
                if server_instance_id.is_some() {
                    return Err(UiEventCursorError::new(
                        "server_instance_id",
                        "duplicate",
                        "server_instance_id may appear at most once",
                    ));
                }
                if !is_webui_token(value, 1, 128) {
                    return Err(UiEventCursorError::new(
                        "server_instance_id",
                        "invalid_format",
                        "server_instance_id must be a printable token",
                    ));
                }
                server_instance_id = Some(value.to_string());
            }
            "after_sequence" => {
                if after_sequence.is_some() {
                    return Err(UiEventCursorError::new(
                        "after_sequence",
                        "duplicate",
                        "after_sequence may appear at most once",
                    ));
                }
                if value.is_empty() || !value.as_bytes().iter().all(|byte| byte.is_ascii_digit()) {
                    return Err(UiEventCursorError::new(
                        "after_sequence",
                        "invalid_format",
                        "after_sequence must be a base-10 integer",
                    ));
                }
                let parsed = value.parse::<u64>().map_err(|_| {
                    UiEventCursorError::new(
                        "after_sequence",
                        "invalid_format",
                        "after_sequence must be a base-10 integer",
                    )
                })?;
                if parsed > MAX_SAFE_EVENT_SEQUENCE {
                    return Err(UiEventCursorError::new(
                        "after_sequence",
                        "out_of_range",
                        "after_sequence must be a JavaScript-safe integer",
                    ));
                }
                after_sequence = Some(parsed);
            }
            _ => {
                return Err(UiEventCursorError::new(
                    "events_query",
                    "unknown_parameter",
                    "only server_instance_id and after_sequence are accepted",
                ));
            }
        }
    }
    if last_event_id.is_some() {
        return Err(UiEventCursorError::new(
            "Last-Event-ID",
            "conflict",
            "Last-Event-ID cannot be combined with paired replay query parameters",
        ));
    }
    match (server_instance_id, after_sequence) {
        (Some(server_instance_id), Some(after_sequence)) => Ok(UiReplayCursor::Sequence {
            server_instance_id,
            after_sequence,
        }),
        (Some(_), None) => Err(UiEventCursorError::new(
            "after_sequence",
            "required",
            "after_sequence is required when server_instance_id is supplied",
        )),
        (None, Some(_)) => Err(UiEventCursorError::new(
            "server_instance_id",
            "required",
            "server_instance_id is required when after_sequence is supplied",
        )),
        (None, None) => Ok(UiReplayCursor::Snapshot),
    }
}

pub(crate) fn ui_events_response(
    coordinator: Arc<LifecycleCoordinator>,
    runtime_model_ids: Vec<String>,
    headers: HeaderMap,
    uri: Uri,
) -> Response {
    let cursor = match parse_ui_event_cursor(uri.query(), last_event_id_from_headers(&headers)) {
        Ok(cursor) => cursor,
        Err(error) => return error.into_response(),
    };
    let (receiver, replay) = match coordinator.subscribe_for_ui(cursor, runtime_model_ids) {
        Ok(subscription) => subscription,
        Err(ReplaySubscribeError::FutureSequence) => {
            return invalid_event_field(
                "after_sequence",
                "future_sequence",
                "after_sequence cannot be newer than the current server sequence",
            );
        }
    };
    let stream = futures::stream::unfold(
        (replay.into_iter(), receiver, coordinator),
        |state| async move {
            let (mut replay, mut receiver, coordinator) = state;
            if let Some(event) = replay.next() {
                return Some((
                    Ok::<Event, std::convert::Infallible>(event_to_sse(event)),
                    (replay, receiver, coordinator),
                ));
            }
            match receiver.recv().await {
                Ok(event) => Some((Ok(event_to_sse(event)), (replay, receiver, coordinator))),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    let event = coordinator.local_reset_event("gap", ResetEventKind::Gap);
                    Some((Ok(event_to_sse(event)), (replay, receiver, coordinator)))
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => None,
            }
        },
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
