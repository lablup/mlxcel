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

//! Integration tests for the b10621 resumable-stream lifecycle and the
//! chat-completion control route (#1444).
//!
//! These run the real router against a scripted streaming provider, so the
//! disconnect-and-resume flow exercises the same SSE senders, session tee,
//! and cancellation plumbing the production path uses.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use crate::server::config::ServerGenerateOptions;
use crate::server::model_provider::{ModelProvider, ScriptedStreamHandle};
use crate::server::{AppState, ChatTemplateProcessor, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn scripted_app(
    config: ServerConfig,
) -> (
    axum::Router,
    ScriptedStreamHandle,
    mpsc::Receiver<ServerGenerateOptions>,
) {
    let (options_tx, options_rx) = mpsc::channel();
    let (provider, handle) = ModelProvider::scripted_streaming_for_route_tests(options_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    );
    (create_app(state), handle, options_rx)
}

fn chat_body(reasoning_control: Option<bool>) -> String {
    let mut body = serde_json::json!({
        "model": "route-test-model",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}],
    });
    if let Some(rc) = reasoning_control {
        body["reasoning_control"] = serde_json::json!(rc);
    }
    body.to_string()
}

fn chat_request(
    conv_id: Option<&str>,
    auth: Option<&str>,
    reasoning_control: Option<bool>,
) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(cid) = conv_id {
        builder = builder.header("x-conversation-id", cid);
    }
    if let Some(key) = auth {
        builder = builder.header("authorization", format!("Bearer {key}"));
    }
    builder
        .body(Body::from(chat_body(reasoning_control)))
        .expect("request builds")
}

fn get_request(uri: &str, auth: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method("GET").uri(uri);
    if let Some(key) = auth {
        builder = builder.header("authorization", format!("Bearer {key}"));
    }
    builder.body(Body::empty()).expect("request builds")
}

fn delete_request(uri: &str, auth: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method("DELETE").uri(uri);
    if let Some(key) = auth {
        builder = builder.header("authorization", format!("Bearer {key}"));
    }
    builder.body(Body::empty()).expect("request builds")
}

fn post_json(uri: &str, body: String, auth: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(key) = auth {
        builder = builder.header("authorization", format!("Bearer {key}"));
    }
    builder.body(Body::from(body)).expect("request builds")
}

/// Read body frames until the accumulated bytes contain `needle`, with a
/// hard timeout so a broken stream fails the test rather than hanging it.
async fn read_until(body: &mut Body, acc: &mut Vec<u8>, needle: &str) {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while !String::from_utf8_lossy(acc).contains(needle) {
            let frame = body
                .frame()
                .await
                .expect("stream ended before needle")
                .expect("frame ok");
            if let Some(data) = frame.data_ref() {
                acc.extend_from_slice(data);
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {needle:?}"));
}

/// Collect an entire response body (bounded), waiting for the stream to end.
async fn collect_body(response: axum::response::Response) -> Vec<u8> {
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        axum::body::to_bytes(response.into_body(), 16 * 1024 * 1024),
    )
    .await
    .expect("body collect timeout")
    .expect("body collects")
    .to_vec()
}

async fn wait_until<F: Fn() -> bool>(what: &str, cond: F) {
    for _ in 0..200 {
        if cond() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("timed out waiting until {what}");
}

/// The core acceptance test: a disconnected client resumes without missing
/// or duplicating committed events, at byte granularity.
#[tokio::test]
async fn resumable_chat_stream_replays_across_disconnect_without_loss_or_duplication() {
    let (app, handle, _options_rx) = scripted_app(ServerConfig::default());

    let response = app
        .clone()
        .oneshot(chat_request(Some("conv-1"), None, None))
        .await
        .expect("stream request");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();

    // Two tokens reach the connected client, then it disconnects.
    let mut first_bytes: Vec<u8> = Vec::new();
    handle.token("Hello");
    read_until(&mut body, &mut first_bytes, "Hello").await;
    handle.token(" world");
    read_until(&mut body, &mut first_bytes, " world").await;
    drop(body);

    // The generation must survive the disconnect: the cancellation token
    // stays clear and later tokens are committed to the session.
    let flag = handle.cancellation_flag(0).expect("one generation");
    handle.token("!");
    handle.token("?");
    handle.finish();
    assert!(
        !flag.load(std::sync::atomic::Ordering::Acquire),
        "a resumable stream must not be cancelled by a client disconnect"
    );

    // Wait for completion (lookup reports is_done), then replay everything.
    let mut lookup_done = false;
    for _ in 0..200 {
        let response = app
            .clone()
            .oneshot(post_json(
                "/v1/streams/lookup",
                serde_json::json!({"conversation_ids": ["conv-1"]}).to_string(),
                None,
            ))
            .await
            .expect("lookup");
        assert_eq!(response.status(), StatusCode::OK);
        let parsed: serde_json::Value =
            serde_json::from_slice(&collect_body(response).await).expect("json");
        assert_eq!(parsed[0]["conversation_id"], "conv-1");
        if parsed[0]["is_done"].as_bool().unwrap_or(false) {
            assert!(parsed[0]["total_bytes"].as_u64().unwrap_or(0) > 0);
            assert!(parsed[0]["completed_at"].as_i64().unwrap_or(0) > 0);
            lookup_done = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(lookup_done, "lookup must report the completed session");
    assert!(
        !flag.load(std::sync::atomic::Ordering::Acquire),
        "the completed resumable stream must never have been cancelled"
    );

    let response = app
        .clone()
        .oneshot(get_request("/v1/stream?conv_id=conv-1&from=0", None))
        .await
        .expect("replay request");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    let full_replay = collect_body(response).await;
    let replay_text = String::from_utf8_lossy(&full_replay).to_string();

    // Every committed event is present exactly once: the four deltas, the
    // finish chunk and the [DONE] marker.
    for needle in ["Hello", " world", "!", "?"] {
        assert_eq!(
            replay_text.matches(needle).count(),
            1,
            "replay must carry {needle:?} exactly once:\n{replay_text}"
        );
    }
    assert_eq!(replay_text.matches("[DONE]").count(), 1);
    assert!(replay_text.contains("finish_reason"));

    // Byte-exact resume: what the client saw before disconnecting is a
    // prefix of the replay, and resuming from that offset yields exactly
    // the remainder. No event is lost, none is duplicated.
    assert!(
        full_replay.starts_with(&first_bytes),
        "the live bytes must be a prefix of the replay"
    );
    let response = app
        .clone()
        .oneshot(get_request(
            &format!("/v1/stream?conv_id=conv-1&from={}", first_bytes.len()),
            None,
        ))
        .await
        .expect("resume request");
    assert_eq!(response.status(), StatusCode::OK);
    let rest = collect_body(response).await;
    let mut reassembled = first_bytes.clone();
    reassembled.extend_from_slice(&rest);
    assert_eq!(
        reassembled, full_replay,
        "prefix + resumed suffix must equal the full replay"
    );
}

/// Regression guard for the pre-#1444 behavior: without `X-Conversation-Id`
/// a client disconnect still cancels the generation.
#[tokio::test]
async fn plain_stream_without_conversation_id_still_cancels_on_disconnect() {
    let (app, handle, _options_rx) = scripted_app(ServerConfig::default());

    let response = app
        .clone()
        .oneshot(chat_request(None, None, None))
        .await
        .expect("stream request");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    handle.token("A");
    read_until(&mut body, &mut bytes, "A").await;
    drop(body);

    // The next send attempts hit the dropped receiver and flip the flag.
    let flag = handle.cancellation_flag(0).expect("one generation");
    handle.token("B");
    wait_until("disconnect cancels the plain stream", || {
        handle.token("C");
        flag.load(std::sync::atomic::Ordering::Acquire)
    })
    .await;
    handle.finish();

    // And nothing is replayable: no session was created.
    let response = app
        .oneshot(get_request("/v1/stream?conv_id=conv-1&from=0", None))
        .await
        .expect("replay request");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn stream_get_validates_conv_id_and_from() {
    let (app, _handle, _options_rx) = scripted_app(ServerConfig::default());

    // Missing conv_id: b10621's 400.
    let response = app
        .clone()
        .oneshot(get_request("/v1/stream", None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = collect_body(response).await;
    assert!(String::from_utf8_lossy(&body).contains("Missing conversation id in path"));

    // Unknown conversation: 404 with b10621's wording.
    let response = app
        .clone()
        .oneshot(get_request("/v1/stream?conv_id=missing", None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = collect_body(response).await;
    assert!(String::from_utf8_lossy(&body).contains("Stream not found or expired"));

    // DELETE without conv_id: 400 as well.
    let response = app
        .clone()
        .oneshot(delete_request("/v1/stream", None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // DELETE of an unknown conversation: idempotent 204.
    let response = app
        .clone()
        .oneshot(delete_request("/v1/stream?conv_id=missing", None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    // Lookup with a malformed body: 400 naming the parse error.
    let response = app
        .clone()
        .oneshot(post_json("/v1/streams/lookup", "{not json".into(), None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = collect_body(response).await;
    assert!(String::from_utf8_lossy(&body).contains("invalid body"));

    // Lookup of unknown ids: an empty array, never an error.
    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/streams/lookup",
            serde_json::json!({"conversation_ids": ["missing"]}).to_string(),
            None,
        ))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    let body = collect_body(response).await;
    assert_eq!(String::from_utf8_lossy(&body), "[]");
}

#[tokio::test]
async fn stream_get_rejects_an_unparsable_from_offset() {
    let (app, handle, _options_rx) = scripted_app(ServerConfig::default());
    let response = app
        .clone()
        .oneshot(chat_request(Some("conv-1"), None, None))
        .await
        .expect("stream request");
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    handle.token("A");
    read_until(&mut body, &mut bytes, "A").await;

    let response = app
        .clone()
        .oneshot(get_request("/v1/stream?conv_id=conv-1&from=zebra", None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let err = collect_body(response).await;
    assert!(String::from_utf8_lossy(&err).contains("Invalid 'from' offset"));
    handle.finish();
}

#[tokio::test]
async fn delete_cancels_the_generation_and_evicts_the_session() {
    let (app, handle, _options_rx) = scripted_app(ServerConfig::default());

    let response = app
        .clone()
        .oneshot(chat_request(Some("conv-1"), None, None))
        .await
        .expect("stream request");
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    handle.token("A");
    read_until(&mut body, &mut bytes, "A").await;

    // Explicit user Stop.
    let response = app
        .clone()
        .oneshot(delete_request("/v1/stream?conv_id=conv-1", None))
        .await
        .expect("delete");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let flag = handle.cancellation_flag(0).expect("one generation");
    assert!(
        flag.load(std::sync::atomic::Ordering::Acquire),
        "DELETE /v1/stream must cancel the underlying generation"
    );
    handle.finish();

    // The session is gone for replay and lookup alike.
    let response = app
        .clone()
        .oneshot(get_request("/v1/stream?conv_id=conv-1&from=0", None))
        .await
        .expect("replay");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    // And the delete stays idempotent.
    let response = app
        .clone()
        .oneshot(delete_request("/v1/stream?conv_id=conv-1", None))
        .await
        .expect("delete again");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn stream_sessions_are_isolated_between_api_keys() {
    let config = ServerConfig {
        api_keys: crate::server::ApiKeys::from_vec(vec!["key-a".into(), "key-b".into()]),
        ..Default::default()
    };
    let (app, handle, _options_rx) = scripted_app(config);

    let response = app
        .clone()
        .oneshot(chat_request(Some("conv-1"), Some("key-a"), None))
        .await
        .expect("stream request");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    handle.token("secret");
    read_until(&mut body, &mut bytes, "secret").await;

    // Key B cannot read, discover, or delete key A's stream, and the
    // answers are indistinguishable from "no such stream".
    let response = app
        .clone()
        .oneshot(get_request(
            "/v1/stream?conv_id=conv-1&from=0",
            Some("key-b"),
        ))
        .await
        .expect("cross-key replay");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/streams/lookup",
            serde_json::json!({"conversation_ids": ["conv-1"]}).to_string(),
            Some("key-b"),
        ))
        .await
        .expect("cross-key lookup");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(String::from_utf8_lossy(&collect_body(response).await), "[]");

    let response = app
        .clone()
        .oneshot(delete_request("/v1/stream?conv_id=conv-1", Some("key-b")))
        .await
        .expect("cross-key delete");
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    let flag = handle.cancellation_flag(0).expect("one generation");
    assert!(
        !flag.load(std::sync::atomic::Ordering::Acquire),
        "another key's DELETE must not cancel the stream"
    );

    // The owner still sees and replays it.
    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/streams/lookup",
            serde_json::json!({"conversation_ids": ["conv-1"]}).to_string(),
            Some("key-a"),
        ))
        .await
        .expect("owner lookup");
    let parsed: serde_json::Value =
        serde_json::from_slice(&collect_body(response).await).expect("json");
    assert_eq!(parsed.as_array().map(Vec::len), Some(1));
    assert_eq!(parsed[0]["conversation_id"], "conv-1");

    handle.finish();
    let response = app
        .clone()
        .oneshot(get_request(
            "/v1/stream?conv_id=conv-1&from=0",
            Some("key-a"),
        ))
        .await
        .expect("owner replay");
    assert_eq!(response.status(), StatusCode::OK);
    let replay = collect_body(response).await;
    assert!(String::from_utf8_lossy(&replay).contains("secret"));

    // Unauthenticated requests never reach the handlers at all.
    let response = app
        .oneshot(get_request("/v1/stream?conv_id=conv-1&from=0", None))
        .await
        .expect("anonymous replay");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn control_route_validates_its_body() {
    let (app, _handle, _options_rx) = scripted_app(ServerConfig::default());

    // Missing id.
    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/chat/completions/control",
            serde_json::json!({"action": "reasoning_end"}).to_string(),
            None,
        ))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = collect_body(response).await;
    assert!(String::from_utf8_lossy(&body).contains("missing completion id"));

    // Unknown action.
    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/chat/completions/control",
            serde_json::json!({"id": "chatcmpl-x", "action": "sampler_swap"}).to_string(),
            None,
        ))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = collect_body(response).await;
    assert!(String::from_utf8_lossy(&body).contains("unknown control action"));

    // Unknown completion id: 200 with success=false, as upstream answers.
    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/chat/completions/control",
            serde_json::json!({"id": "chatcmpl-missing", "action": "reasoning_end"}).to_string(),
            None,
        ))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    let parsed: serde_json::Value =
        serde_json::from_slice(&collect_body(response).await).expect("json");
    assert_eq!(parsed["success"], false);
    assert_eq!(parsed["message"], "no active completion for this id");
}

#[tokio::test]
async fn reasoning_end_forces_the_armed_flag_and_expires_with_the_completion() {
    let (app, handle, options_rx) = scripted_app(ServerConfig::default());

    let response = app
        .clone()
        .oneshot(chat_request(None, None, Some(true)))
        .await
        .expect("stream request");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();

    // The first chunk carries the completion id the control call targets.
    let mut bytes = Vec::new();
    handle.token("thinking...");
    read_until(&mut body, &mut bytes, "thinking...").await;
    let text = String::from_utf8_lossy(&bytes);
    let id_start = text.find("chatcmpl-").expect("chunk carries the id");
    let cmpl_id: String = text[id_start..].chars().take_while(|c| *c != '"').collect();

    // The armed flag reached the scheduler options and is still clear.
    let options = options_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("options captured");
    let armed = options
        .reasoning_control
        .clone()
        .expect("reasoning_control armed the options flag");
    assert!(!armed.load(std::sync::atomic::Ordering::Acquire));

    // Control call: success, and the shared flag flips. The next sampled
    // token is the event-order boundary; bytes already streamed are
    // untouched (the boundary semantics are unit-tested on ThinkingState).
    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/chat/completions/control",
            serde_json::json!({"id": cmpl_id, "action": "reasoning_end"}).to_string(),
            None,
        ))
        .await
        .expect("control");
    assert_eq!(response.status(), StatusCode::OK);
    let parsed: serde_json::Value =
        serde_json::from_slice(&collect_body(response).await).expect("json");
    assert_eq!(parsed, serde_json::json!({"success": true}));
    assert!(armed.load(std::sync::atomic::Ordering::Acquire));

    // Finish the stream; once the generation task exits, the id is gone.
    handle.finish();
    let mut rest = Vec::new();
    read_until(&mut body, &mut rest, "[DONE]").await;
    drop(body);
    let mut expired = false;
    for _ in 0..200 {
        let response = app
            .clone()
            .oneshot(post_json(
                "/v1/chat/completions/control",
                serde_json::json!({"id": cmpl_id, "action": "reasoning_end"}).to_string(),
                None,
            ))
            .await
            .expect("control");
        let parsed: serde_json::Value =
            serde_json::from_slice(&collect_body(response).await).expect("json");
        if parsed["success"] == false {
            assert_eq!(parsed["message"], "no active completion for this id");
            expired = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(
        expired,
        "the control entry must expire with the generation task"
    );
}

#[tokio::test]
async fn reasoning_end_on_an_unarmed_completion_reports_not_enabled() {
    let (app, handle, _options_rx) = scripted_app(ServerConfig::default());

    let response = app
        .clone()
        .oneshot(chat_request(None, None, None))
        .await
        .expect("stream request");
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    handle.token("A");
    read_until(&mut body, &mut bytes, "A").await;
    let text = String::from_utf8_lossy(&bytes);
    let id_start = text.find("chatcmpl-").expect("chunk carries the id");
    let cmpl_id: String = text[id_start..].chars().take_while(|c| *c != '"').collect();

    let response = app
        .clone()
        .oneshot(post_json(
            "/v1/chat/completions/control",
            serde_json::json!({"id": cmpl_id, "action": "reasoning_end"}).to_string(),
            None,
        ))
        .await
        .expect("control");
    assert_eq!(response.status(), StatusCode::OK);
    let parsed: serde_json::Value =
        serde_json::from_slice(&collect_body(response).await).expect("json");
    assert_eq!(parsed["success"], false);
    assert_eq!(
        parsed["message"],
        "reasoning control not enabled for this completion"
    );
    handle.finish();
}
