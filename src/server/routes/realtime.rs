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

//! `GET /v1/realtime`: the full-duplex VoiceChat WebSocket (issue #1376).
//!
//! Port of `realtime_voicechat_endpoint` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/server/realtime.py.
//! One connection holds the engine reservation; a second gets
//! `server_busy` and close code 1013. `input_audio_buffer.append` is pushed
//! one native frame (1280 samples) per engine call and each call's events
//! are sent before the next, so a large network chunk does not delay every
//! output event until the whole chunk is processed. The task only awaits
//! engine replies; all MLX work runs on the engine thread.
//!
//! Used by: `server/app.rs` (mounted only when a VoiceChat checkpoint is
//! served), `tests/realtime_ws.rs`

use std::sync::Arc;

use axum::extract::State;
use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;

use crate::server::CorsPolicy;
use crate::server::realtime_engine::{RealtimeSessionConfig, RealtimeVoiceChatEngine};
use crate::server::realtime_protocol::{
    AudioFormat, CLOSE_CODE_POLICY, CLOSE_CODE_TRY_AGAIN_LATER, CODE_INFERENCE_ERROR,
    CODE_INVALID_REQUEST, CODE_SERVER_BUSY, CODE_SESSION_INITIALIZATION_FAILED, ClientMessage,
    DEFAULT_FRAME_SAMPLES, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, ServerEvent, SessionObject,
    new_session_id, parse_append, parse_client_message, parse_session_request, to_wire_json,
};

/// Route path of the realtime socket.
pub const REALTIME_PATH: &str = "/v1/realtime";

/// State of the realtime router: the engine and the origin policy the
/// upgrade is checked against.
#[derive(Clone)]
struct RealtimeRouteState {
    engine: Arc<RealtimeVoiceChatEngine>,
    cors: Arc<CorsPolicy>,
}

/// A router serving only [`REALTIME_PATH`]. `server/app.rs` merges it inside
/// the auth and `--api-prefix` layers, so a request failing both the API key
/// and the origin check gets 401. `cors` is the server's `--cors-origins` /
/// `--allowed-origins` policy: browsers do not apply CORS to WebSocket
/// upgrades, so the handler checks `Origin` itself (#2042).
pub fn realtime_router<S>(engine: Arc<RealtimeVoiceChatEngine>, cors: Arc<CorsPolicy>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route(REALTIME_PATH, get(realtime_ws))
        .with_state(RealtimeRouteState { engine, cors })
}

/// Upgrade handler. A disallowed `Origin` gets 403 before the upgrade is
/// even validated, so no reservation is taken and no session is created.
async fn realtime_ws(
    State(state): State<RealtimeRouteState>,
    headers: HeaderMap,
    ws: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Response {
    let origin = headers.get(header::ORIGIN);
    if !state.cors.permits_websocket_origin(origin) {
        let shown = origin.map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned());
        tracing::warn!("rejected /v1/realtime upgrade from origin {shown:?}");
        let body = json!({
            "error": {
                "message": "origin not allowed for /v1/realtime",
                "type": "invalid_request_error",
            }
        });
        return (StatusCode::FORBIDDEN, Json(body)).into_response();
    }
    let ws = match ws {
        Ok(ws) => ws,
        Err(rejection) => return rejection.into_response(),
    };
    let engine = state.engine;
    // Clients send 80 ms appends (about 3.5 KB of base64); 1 MiB (about
    // 24 s of audio per message) bounds what one message can make the
    // server decode and push before the socket is read again.
    ws.max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| run_connection(socket, engine))
}

/// Largest accepted WebSocket message or frame.
pub const MAX_MESSAGE_BYTES: usize = 1 << 20;

/// Longest session a client can request, and the limit applied when it
/// requests none: the language-model and TTS caches grow every 80 ms frame,
/// so an unbounded session on the only slot would grow until the device
/// runs out of memory.
pub const MAX_STREAMING_SECONDS: f32 = 600.0;

/// How long a connection may stay open without sending `session.update`.
pub const CONFIGURE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// How long a configured session may go without a client message (pings
/// do not count) before it is closed and the slot is released.
pub const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Why the message loop ended.
enum Exit {
    /// `commit` flushed or `cancel` stopped the session.
    Finished,
    /// The client closed or the socket failed.
    Disconnected,
    /// The client sent nothing for too long; the socket is closed with 1008.
    TimedOut,
}

/// Holds the reservation; a task dropped mid-session (server shutdown)
/// still closes its session and releases the engine.
struct Reservation {
    engine: Arc<RealtimeVoiceChatEngine>,
    session_id: String,
    configured: bool,
    /// An `open` was sent to the engine (possibly still in flight), so a
    /// session may exist on the engine thread even if `configured` is not
    /// set yet; closing an inactive session is a no-op.
    open_sent: bool,
    released: bool,
}

impl Reservation {
    async fn finish(&mut self) {
        if (self.configured || self.open_sent)
            && let Err(err) = self.engine.close(&self.session_id).await
        {
            // An `open` that failed left no session behind to close.
            if self.configured {
                tracing::warn!("failed to close realtime VoiceChat session: {err}");
            }
        }
        self.engine.release(&self.session_id);
        self.released = true;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        if self.configured || self.open_sent {
            self.engine.close_detached(&self.session_id);
        }
        self.engine.release(&self.session_id);
    }
}

async fn send(socket: &mut WebSocket, event: &ServerEvent) -> Result<(), axum::Error> {
    socket.send(Message::Text(to_wire_json(event))).await
}

async fn send_error(socket: &mut WebSocket, code: &str, message: impl Into<String>) -> bool {
    send(socket, &ServerEvent::error(code, message))
        .await
        .is_ok()
}

async fn close_with(socket: &mut WebSocket, code: u16, reason: &'static str) {
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        })))
        .await;
}

async fn run_connection(mut socket: WebSocket, engine: Arc<RealtimeVoiceChatEngine>) {
    let session_id = new_session_id();
    if !engine.try_reserve(&session_id) {
        send_error(
            &mut socket,
            CODE_SERVER_BUSY,
            "another realtime VoiceChat session is already active",
        )
        .await;
        close_with(&mut socket, CLOSE_CODE_TRY_AGAIN_LATER, "server busy").await;
        return;
    }
    let mut reservation = Reservation {
        engine: engine.clone(),
        session_id: session_id.clone(),
        configured: false,
        open_sent: false,
        released: false,
    };

    let created = ServerEvent::SessionCreated {
        session: SessionObject {
            id: session_id.clone(),
            state: "configuring".to_string(),
            model: None,
            frame_samples: None,
            input_audio_format: AudioFormat::pcm16(INPUT_SAMPLE_RATE),
            output_audio_format: AudioFormat::pcm16(OUTPUT_SAMPLE_RATE),
        },
    };
    let exit = if send(&mut socket, &created).await.is_ok() {
        message_loop(&mut socket, &mut reservation).await
    } else {
        Exit::Disconnected
    };
    reservation.finish().await;
    match exit {
        Exit::Finished => close_with(&mut socket, 1000, "").await,
        Exit::TimedOut => close_with(&mut socket, CLOSE_CODE_POLICY, "idle timeout").await,
        Exit::Disconnected => {}
    }
}

async fn message_loop(socket: &mut WebSocket, reservation: &mut Reservation) -> Exit {
    let engine = reservation.engine.clone();
    let session_id = reservation.session_id.clone();
    let mut frame_samples = DEFAULT_FRAME_SAMPLES;
    let mut last_activity = tokio::time::Instant::now();

    loop {
        let limit = if reservation.configured {
            IDLE_TIMEOUT
        } else {
            CONFIGURE_TIMEOUT
        };
        let Ok(received) = tokio::time::timeout_at(last_activity + limit, socket.recv()).await
        else {
            let reason = if reservation.configured {
                "no client message within the idle timeout; closing the session"
            } else {
                "send session.update after connecting; closing the connection"
            };
            let _ = send_error(socket, CODE_INVALID_REQUEST, reason).await;
            return Exit::TimedOut;
        };
        if matches!(received, Some(Ok(Message::Text(_) | Message::Binary(_)))) {
            last_activity = tokio::time::Instant::now();
        }
        let text = match received {
            Some(Ok(Message::Text(text))) => text,
            Some(Ok(Message::Binary(_))) => {
                if !send_error(
                    socket,
                    CODE_INVALID_REQUEST,
                    "message must be a JSON object",
                )
                .await
                {
                    return Exit::Disconnected;
                }
                continue;
            }
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return Exit::Disconnected,
        };
        let message = match parse_client_message(&text) {
            Ok(message) => message,
            Err(err) => {
                if !send_error(socket, CODE_INVALID_REQUEST, err).await {
                    return Exit::Disconnected;
                }
                continue;
            }
        };

        // Events to send and whether the loop ends after them.
        let (events, finished) = match message {
            ClientMessage::SessionUpdate { session } => {
                if reservation.configured {
                    (
                        vec![ServerEvent::error(
                            CODE_INVALID_REQUEST,
                            "the active session cannot be reconfigured",
                        )],
                        false,
                    )
                } else {
                    reservation.open_sent = true;
                    let (event, info) = configure(&engine, &session_id, &session).await;
                    if let Some(info) = info {
                        reservation.configured = true;
                        frame_samples = info.frame_samples.max(1);
                    }
                    (vec![event], false)
                }
            }
            ClientMessage::Ping => (vec![ServerEvent::SessionPong], false),
            _ if !reservation.configured => (
                vec![ServerEvent::error(
                    CODE_INVALID_REQUEST,
                    "send session.update before audio",
                )],
                false,
            ),
            ClientMessage::Append { audio, sample_rate } => {
                let (samples, rate) = match parse_append(audio.as_ref(), sample_rate.as_ref()) {
                    Ok(decoded) => decoded,
                    Err(err) => {
                        if !send_error(socket, CODE_INFERENCE_ERROR, err).await {
                            return Exit::Disconnected;
                        }
                        continue;
                    }
                };
                // One engine call per native frame; an empty append still
                // makes one (empty) call, as upstream's loop does.
                let mut start = 0;
                loop {
                    let end = (start + frame_samples).min(samples.len());
                    let result = engine
                        .push(&session_id, samples[start..end].to_vec(), rate)
                        .await;
                    let sent = match result {
                        Ok(events) => send_all(socket, &events).await,
                        Err(err) => {
                            // Upstream stops the chunk at the first failure.
                            if !send_error(socket, CODE_INFERENCE_ERROR, err.to_string()).await {
                                return Exit::Disconnected;
                            }
                            break;
                        }
                    };
                    if !sent {
                        return Exit::Disconnected;
                    }
                    start = end;
                    if start >= samples.len() {
                        break;
                    }
                }
                continue;
            }
            ClientMessage::Commit { pad_partial } => {
                if send(socket, &ServerEvent::InputAudioBufferCommitted)
                    .await
                    .is_err()
                {
                    return Exit::Disconnected;
                }
                match engine.flush(&session_id, pad_partial).await {
                    Ok(events) => (events, true),
                    Err(err) => (
                        vec![ServerEvent::error(CODE_INFERENCE_ERROR, err.to_string())],
                        false,
                    ),
                }
            }
            ClientMessage::Cancel => match engine.cancel(&session_id).await {
                Ok(events) => (events, true),
                Err(err) => (
                    vec![ServerEvent::error(CODE_INFERENCE_ERROR, err.to_string())],
                    false,
                ),
            },
            ClientMessage::Other(kind) => (
                vec![ServerEvent::error(
                    CODE_INVALID_REQUEST,
                    format!("unsupported event type '{kind}'"),
                )],
                false,
            ),
        };
        if !send_all(socket, &events).await {
            return Exit::Disconnected;
        }
        if finished {
            return Exit::Finished;
        }
    }
}

async fn send_all(socket: &mut WebSocket, events: &[ServerEvent]) -> bool {
    for event in events {
        if send(socket, event).await.is_err() {
            return false;
        }
    }
    true
}

/// Open the session for `session.update`; returns the event to send and the
/// session geometry on success.
async fn configure(
    engine: &RealtimeVoiceChatEngine,
    session_id: &str,
    session: &serde_json::Value,
) -> (
    ServerEvent,
    Option<crate::server::realtime_engine::RealtimeSessionInfo>,
) {
    let request = match parse_session_request(session) {
        Ok(request) => request,
        Err(err) => {
            return (
                ServerEvent::error(CODE_SESSION_INITIALIZATION_FAILED, err),
                None,
            );
        }
    };
    // The server runs one model: `session.model` is optional (upstream
    // requires it because it loads models by name), any value opens the
    // served model, and `session.updated` reports the served id.
    let config = RealtimeSessionConfig {
        system_prompt: request.system_prompt,
        seed: request.seed,
        // Clamp to the server limit and apply it when the client sets none.
        // An invalid value is passed through so the session reports it.
        max_streaming_seconds: Some(match request.max_streaming_seconds {
            Some(s) if s.is_finite() && s > 0.0 => s.min(MAX_STREAMING_SECONDS),
            Some(s) => s,
            None => MAX_STREAMING_SECONDS,
        }),
    };
    match engine.open(session_id, config).await {
        Ok(info) => (
            ServerEvent::SessionUpdated {
                session: SessionObject {
                    id: session_id.to_string(),
                    state: "ready".to_string(),
                    model: Some(engine.model_id().to_string()),
                    frame_samples: Some(info.frame_samples),
                    input_audio_format: AudioFormat::pcm16(info.input_sample_rate),
                    output_audio_format: AudioFormat::pcm16(info.output_sample_rate),
                },
            },
            Some(info),
        ),
        Err(err) => (
            ServerEvent::error(CODE_SESSION_INITIALIZATION_FAILED, err.to_string()),
            None,
        ),
    }
}
