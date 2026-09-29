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

//! `/v1/realtime` WebSocket protocol tests against a fake VoiceChat model
//! (issue #1376). No checkpoint is needed: the fake reports each pushed slice
//! as a transcript delta whose text is the slice length, so the tests can see
//! how the socket loop sliced an append.

use std::sync::Arc;

use axum::http::HeaderValue;
use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use mlxcel::models::nemotron_voicechat::VoiceChatEvent;
use mlxcel::models::nemotron_voicechat::streaming::VoiceChatError;
use mlxcel::server::CorsPolicy;
use mlxcel::server::realtime_engine::{
    RealtimeModel, RealtimeSession, RealtimeSessionConfig, RealtimeSessionInfo,
    RealtimeVoiceChatEngine,
};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

struct FakeModel;

struct FakeSession {
    frames: u64,
    closed: bool,
}

impl RealtimeModel for FakeModel {
    fn open_session(
        &self,
        config: &RealtimeSessionConfig,
    ) -> Result<Box<dyn RealtimeSession + '_>, VoiceChatError> {
        if config.max_streaming_seconds.is_some_and(|s| s <= 0.0) {
            return Err(VoiceChatError::InvalidInput(
                "max_streaming_seconds must be positive".to_string(),
            ));
        }
        Ok(Box::new(FakeSession {
            frames: 0,
            closed: false,
        }))
    }
}

impl RealtimeSession for FakeSession {
    fn info(&self) -> RealtimeSessionInfo {
        RealtimeSessionInfo {
            input_sample_rate: 16_000,
            output_sample_rate: 22_050,
            frame_samples: 1280,
        }
    }

    fn push_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<VoiceChatEvent>, VoiceChatError> {
        if sample_rate != 16_000 {
            return Err(VoiceChatError::InvalidInput(format!(
                "expected 16000 Hz PCM, received {sample_rate} Hz"
            )));
        }
        let frame_index = self.frames;
        self.frames += 1;
        Ok(vec![
            VoiceChatEvent::UserTranscriptDelta {
                frame_index,
                delta: samples.len().to_string(),
                text: samples.len().to_string(),
            },
            VoiceChatEvent::Audio {
                frame_index,
                samples: vec![0.25; 1764],
                sample_rate: 22_050,
                audio_codes: vec![1; 31],
            },
        ])
    }

    fn flush(&mut self, _pad_partial: bool) -> Result<Vec<VoiceChatEvent>, VoiceChatError> {
        self.closed = true;
        Ok(vec![VoiceChatEvent::Done {
            frame_index: self.frames,
        }])
    }

    fn cancel(&mut self) -> Vec<VoiceChatEvent> {
        self.closed = true;
        vec![VoiceChatEvent::Cancelled {
            frame_index: self.frames,
        }]
    }

    fn is_closed(&self) -> bool {
        self.closed
    }
}

/// Serve the realtime router on a free port; returns the socket URL.
async fn start_server() -> String {
    start_server_with_policy(CorsPolicy::default()).await
}

/// [`start_server`] with an explicit origin policy (#2042).
async fn start_server_with_policy(policy: CorsPolicy) -> String {
    let engine = RealtimeVoiceChatEngine::spawn_with_loader("fake-voicechat", || Ok(FakeModel))
        .expect("fake engine");
    let app =
        mlxcel::server::routes::realtime::realtime_router::<()>(Arc::new(engine), Arc::new(policy));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("ws://{addr}/v1/realtime")
}

async fn connect(url: &str) -> Socket {
    tokio_tungstenite::connect_async(url).await.unwrap().0
}

/// Next JSON event; panics on a close frame or a closed socket.
async fn next_event(socket: &mut Socket) -> Value {
    loop {
        match socket.next().await {
            Some(Ok(Message::Text(text))) => return serde_json::from_str(&text).unwrap(),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            other => panic!("expected a JSON event, got {other:?}"),
        }
    }
}

/// Wait for the server's close frame and return its code.
async fn close_code(socket: &mut Socket) -> Option<CloseCode> {
    loop {
        match socket.next().await {
            Some(Ok(Message::Close(frame))) => return frame.map(|f| f.code),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            None | Some(Err(_)) => return None,
            Some(Ok(other)) => panic!("expected close, got {other:?}"),
        }
    }
}

async fn send(socket: &mut Socket, value: Value) {
    socket.send(Message::Text(value.to_string())).await.unwrap();
}

fn pcm16(samples: usize) -> String {
    base64::engine::general_purpose::STANDARD.encode(vec![0u8; samples * 2])
}

/// Connect, read `session.created`, configure, read `session.updated`.
async fn configured(url: &str) -> Socket {
    let mut socket = connect(url).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.created");
    send(
        &mut socket,
        json!({"type": "session.update", "session": {"system_prompt": "Be brief.", "seed": 0}}),
    )
    .await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    socket
}

#[tokio::test]
async fn session_created_then_updated() {
    let url = start_server().await;
    let mut socket = connect(&url).await;
    let created = next_event(&mut socket).await;
    assert_eq!(created["type"], "session.created");
    let event_id = created["event_id"].as_str().unwrap();
    assert!(event_id.starts_with("event_") && event_id.len() == 22);
    let session_id = created["session"]["id"].as_str().unwrap().to_string();
    assert!(session_id.starts_with("sess_") && session_id.len() == 21);
    assert_eq!(created["session"]["state"], "configuring");
    assert_eq!(
        created["session"]["input_audio_format"],
        json!({"type": "pcm16", "sample_rate": 16000})
    );
    assert_eq!(
        created["session"]["output_audio_format"],
        json!({"type": "pcm16", "sample_rate": 22050})
    );

    send(&mut socket, json!({"type": "session.ping"})).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.pong");

    // A failed open is session_initialization_failed and leaves the socket
    // configurable.
    send(
        &mut socket,
        json!({"type": "session.update", "session": {"max_streaming_seconds": -1}}),
    )
    .await;
    let failed = next_event(&mut socket).await;
    assert_eq!(failed["error"]["code"], "session_initialization_failed");

    send(
        &mut socket,
        json!({"type": "session.update", "session": {}}),
    )
    .await;
    let updated = next_event(&mut socket).await;
    assert_eq!(updated["type"], "session.updated");
    assert_eq!(updated["session"]["id"], session_id.as_str());
    assert_eq!(updated["session"]["state"], "ready");
    assert_eq!(updated["session"]["model"], "fake-voicechat");
    assert_eq!(updated["session"]["frame_samples"], 1280);
    assert_eq!(
        updated["session"]["output_audio_format"]["sample_rate"],
        22050
    );

    send(
        &mut socket,
        json!({"type": "session.update", "session": {}}),
    )
    .await;
    let again = next_event(&mut socket).await;
    assert_eq!(again["error"]["code"], "invalid_request");

    send(&mut socket, json!({"type": "bogus.event"})).await;
    let unknown = next_event(&mut socket).await;
    assert_eq!(unknown["error"]["code"], "invalid_request");
    socket.send(Message::Text("not json".into())).await.unwrap();
    let not_json = next_event(&mut socket).await;
    assert_eq!(not_json["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn append_before_update_errors() {
    let url = start_server().await;
    let mut socket = connect(&url).await;
    next_event(&mut socket).await;
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": pcm16(1280)}),
    )
    .await;
    let error = next_event(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["code"], "invalid_request");
    assert_eq!(
        error["error"]["message"],
        "send session.update before audio"
    );
}

#[tokio::test]
async fn append_streams_events_per_1280_sample_slice() {
    let url = start_server().await;
    let mut socket = configured(&url).await;
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": pcm16(3000), "sample_rate": 16000}),
    )
    .await;
    let mut slices = Vec::new();
    for expected_frame in 0..3u64 {
        let transcript = next_event(&mut socket).await;
        assert_eq!(
            transcript["type"],
            "conversation.item.input_audio_transcription.delta"
        );
        assert_eq!(transcript["frame_index"], expected_frame);
        slices.push(transcript["delta"].as_str().unwrap().to_string());
        let audio = next_event(&mut socket).await;
        assert_eq!(audio["type"], "response.audio.delta");
        assert_eq!(audio["frame_index"], expected_frame);
        assert_eq!(audio["format"], "pcm16");
        assert_eq!(audio["sample_rate"], 22050);
        assert_eq!(audio["channels"], 1);
        assert_eq!(audio["audio_codes"].as_array().unwrap().len(), 31);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(audio["delta"].as_str().unwrap())
            .unwrap();
        assert_eq!(bytes.len(), 1764 * 2);
        // round(0.25 * 32767) = 8192
        assert_eq!(i16::from_le_bytes([bytes[0], bytes[1]]), 8192);
    }
    assert_eq!(slices, vec!["1280", "1280", "440"]);

    // An odd byte count is an inference_error and the session stays usable.
    let odd = base64::engine::general_purpose::STANDARD.encode([0u8; 3]);
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": odd}),
    )
    .await;
    assert_eq!(
        next_event(&mut socket).await["error"]["code"],
        "inference_error"
    );
}

#[tokio::test]
async fn commit_emits_committed_then_done() {
    let url = start_server().await;
    let mut socket = configured(&url).await;
    send(&mut socket, json!({"type": "input_audio_buffer.commit"})).await;
    assert_eq!(
        next_event(&mut socket).await["type"],
        "input_audio_buffer.committed"
    );
    let done = next_event(&mut socket).await;
    assert_eq!(done["type"], "response.done");
    assert_eq!(done["frame_index"], 0);
    assert_eq!(close_code(&mut socket).await, Some(CloseCode::Normal));
}

#[tokio::test]
async fn second_connection_gets_server_busy_1013() {
    let url = start_server().await;
    let mut first = configured(&url).await;

    let mut second = connect(&url).await;
    let busy = next_event(&mut second).await;
    assert_eq!(busy["type"], "error");
    assert_eq!(busy["error"]["code"], "server_busy");
    assert_eq!(close_code(&mut second).await, Some(CloseCode::Again));

    // Disconnecting the first releases the engine for the next client.
    first.close(None).await.unwrap();
    drop(first);
    let mut third = None;
    for _ in 0..50 {
        let mut socket = connect(&url).await;
        let event = next_event(&mut socket).await;
        if event["type"] == "session.created" {
            third = Some(socket);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let mut third = third.expect("the reservation is released after the first disconnects");
    send(&mut third, json!({"type": "session.update", "session": {}})).await;
    assert_eq!(next_event(&mut third).await["type"], "session.updated");
}

#[tokio::test]
async fn cancel_emits_cancelled_and_releases() {
    let url = start_server().await;
    let mut socket = configured(&url).await;
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": pcm16(1280)}),
    )
    .await;
    next_event(&mut socket).await;
    next_event(&mut socket).await;
    send(&mut socket, json!({"type": "response.cancel"})).await;
    let cancelled = next_event(&mut socket).await;
    assert_eq!(cancelled["type"], "response.cancelled");
    assert_eq!(cancelled["frame_index"], 1);
    assert_eq!(close_code(&mut socket).await, Some(CloseCode::Normal));

    // The server releases before it sends the close frame.
    let mut next = connect(&url).await;
    assert_eq!(next_event(&mut next).await["type"], "session.created");
}

#[tokio::test]
async fn wrong_sample_rate_is_inference_error() {
    let url = start_server().await;
    let mut socket = configured(&url).await;
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": pcm16(1280), "sample_rate": 8000}),
    )
    .await;
    let error = next_event(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["code"], "inference_error");
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("8000 Hz"),
        "{error}"
    );
    send(&mut socket, json!({"type": "session.ping"})).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.pong");
}

#[tokio::test]
async fn abrupt_disconnect_releases_reservation() {
    let url = start_server().await;
    let mut first = configured(&url).await;
    send(
        &mut first,
        json!({"type": "input_audio_buffer.append", "audio": pcm16(640)}),
    )
    .await;
    // Drop the TCP connection without a close frame, commit or cancel.
    drop(first);
    let mut next = None;
    for _ in 0..50 {
        let mut socket = connect(&url).await;
        if next_event(&mut socket).await["type"] == "session.created" {
            next = Some(socket);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let mut next = next.expect("an abrupt disconnect releases the reservation");
    send(&mut next, json!({"type": "session.update", "session": {}})).await;
    assert_eq!(next_event(&mut next).await["type"], "session.updated");
}

#[tokio::test]
async fn whole_float_sample_rate_is_accepted() {
    let url = start_server().await;
    let mut socket = configured(&url).await;
    send(
        &mut socket,
        json!({"type": "input_audio_buffer.append", "audio": pcm16(1280), "sample_rate": 16000.0}),
    )
    .await;
    let event = next_event(&mut socket).await;
    assert_ne!(event["type"], "error", "16000.0 must be accepted: {event}");
}

/// Connect with an optional `Origin` header; the handshake result as is.
async fn connect_from(url: &str, origin: Option<&str>) -> Result<Socket, WsError> {
    let mut request = url.into_client_request().unwrap();
    if let Some(origin) = origin {
        let value = HeaderValue::from_str(origin).unwrap();
        request.headers_mut().insert("origin", value);
    }
    Ok(tokio_tungstenite::connect_async(request).await?.0)
}

fn assert_forbidden(result: Result<Socket, WsError>) {
    let err = result.err().expect("a disallowed origin must not upgrade");
    assert!(
        matches!(&err, WsError::Http(r) if r.status() == 403),
        "{err:?}"
    );
}

/// Retry until any previous session's reservation is released.
async fn created_after_release(url: &str, origin: Option<&str>) -> Socket {
    for _ in 0..50 {
        let mut socket = connect_from(url, origin).await.expect("origin upgrades");
        if next_event(&mut socket).await["type"] == "session.created" {
            return socket;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("no session.created for origin {origin:?}");
}

#[tokio::test]
async fn disallowed_origin_is_rejected_before_session() {
    let allowed = vec![HeaderValue::from_static("https://app.example.com")];
    let policy = CorsPolicy::resolve("*", "GET", "*", true, Some(allowed)).unwrap();
    let url = start_server_with_policy(policy).await;
    assert_forbidden(connect_from(&url, Some("https://evil.example")).await);

    // The rejected attempt held no reservation; allowed and absent Origin pass.
    let mut first = connect_from(&url, Some("https://app.example.com"))
        .await
        .unwrap();
    assert_eq!(next_event(&mut first).await["type"], "session.created");
    drop(first);
    created_after_release(&url, None).await;
}

#[tokio::test]
async fn localhost_policy_gates_and_default_policy_accepts_any_origin() {
    let policy = CorsPolicy::resolve("localhost", "GET", "*", true, None).unwrap();
    let url = start_server_with_policy(policy).await;
    assert_forbidden(connect_from(&url, Some("https://localhost.evil.com")).await);
    created_after_release(&url, Some("http://localhost:3000")).await;
    let url = start_server().await;
    created_after_release(&url, Some("https://evil.example")).await;
}
