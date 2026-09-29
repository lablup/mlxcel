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

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;

use crate::server::realtime_engine::{RealtimeSessionConfig, RealtimeVoiceChatEngine};
use crate::server::realtime_protocol::{
    AudioFormat, CLOSE_CODE_TRY_AGAIN_LATER, CODE_INFERENCE_ERROR, CODE_INVALID_REQUEST,
    CODE_SERVER_BUSY, CODE_SESSION_INITIALIZATION_FAILED, ClientMessage, DEFAULT_FRAME_SAMPLES,
    INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, ServerEvent, SessionObject, new_session_id,
    parse_append, parse_client_message, parse_session_request, to_wire_json,
};

/// Route path of the realtime socket.
pub const REALTIME_PATH: &str = "/v1/realtime";

/// A router serving only [`REALTIME_PATH`], with the engine as its state.
/// `server/app.rs` merges it inside the auth and `--api-prefix` layers.
pub fn realtime_router<S>(engine: Arc<RealtimeVoiceChatEngine>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route(REALTIME_PATH, get(realtime_ws))
        .with_state(engine)
}

/// Upgrade handler.
pub async fn realtime_ws(
    ws: WebSocketUpgrade,
    State(engine): State<Arc<RealtimeVoiceChatEngine>>,
) -> Response {
    ws.on_upgrade(move |socket| run_connection(socket, engine))
}

/// Why the message loop ended.
enum Exit {
    /// `commit` flushed or `cancel` stopped the session.
    Finished,
    /// The client closed or the socket failed.
    Disconnected,
}

/// Holds the reservation; a task dropped mid-session (server shutdown)
/// still closes its session and releases the engine.
struct Reservation {
    engine: Arc<RealtimeVoiceChatEngine>,
    session_id: String,
    configured: bool,
    released: bool,
}

impl Reservation {
    async fn finish(&mut self) {
        if self.configured
            && let Err(err) = self.engine.close(&self.session_id).await
        {
            tracing::warn!("failed to close realtime VoiceChat session: {err}");
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
        if self.configured {
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
    if matches!(exit, Exit::Finished) {
        close_with(&mut socket, 1000, "").await;
    }
}

async fn message_loop(socket: &mut WebSocket, reservation: &mut Reservation) -> Exit {
    let engine = reservation.engine.clone();
    let session_id = reservation.session_id.clone();
    let mut frame_samples = DEFAULT_FRAME_SAMPLES;

    loop {
        let text = match socket.recv().await {
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
        max_streaming_seconds: request.max_streaming_seconds,
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
