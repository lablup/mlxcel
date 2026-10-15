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

//! `/v1/messages/count_tokens` through the real router (issue #1748).
//!
//! The handler renders the translated request, and rendering resolves every
//! `image_url` part, so the media gate has to run before it. These tests drive
//! the route with `create_app` + `oneshot` and point the image at a loopback
//! origin that records every connection, so "refused before anything was
//! fetched" is observed rather than inferred from the status code.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

use crate::server::state::ModelMediaSupport;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn state() -> AppState {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(
            "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}".to_string(),
        ),
        MlxcelTokenizer::stub_with_byte_fallback(),
        PathBuf::from("count-tokens-test-model"),
        batch_metrics,
    )
}

/// The stub state's default media support: a text-only checkpoint.
fn text_only_app() -> Router {
    create_app(state())
}

fn image_capable_app() -> Router {
    create_app(state().with_media_support(ModelMediaSupport {
        image: true,
        ..ModelMediaSupport::default()
    }))
}

async fn post(
    app: Router,
    path: &str,
    body: &serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

/// A loopback HTTP origin that counts every connection it accepts and answers
/// each with an empty 404, so a fetch that does happen finishes promptly.
///
/// The counter is bumped before the response is written, so by the time the
/// route that fetched has answered, its connection is already counted.
async fn recording_origin() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener binds");
    let addr = listener.local_addr().expect("listener address");
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let server = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            counter.fetch_add(1, Ordering::SeqCst);
            let mut request = [0u8; 2048];
            let _ = socket.read(&mut request).await;
            let _ = socket
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await;
        }
    });
    (format!("http://{addr}/cat.png"), hits, server)
}

fn image_body(url: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "count-tokens-test-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is in this picture"},
            {"type": "image", "source": {"type": "url", "url": url}}
        ]}]
    })
}

#[tokio::test]
async fn an_image_block_is_refused_before_its_url_is_fetched() {
    // Loopback origins are refused by the media address policy in production;
    // open them for this test so the gate is the only thing that can keep the
    // route from connecting.
    let _private = crate::server::media_net::allow_private_media_urls_in_tests();
    let (url, hits, server) = recording_origin().await;
    let body = image_body(&url);

    let (messages_status, messages_body) = post(text_only_app(), "/v1/messages", &body).await;
    assert_eq!(
        messages_status,
        StatusCode::NOT_IMPLEMENTED,
        "{messages_body}"
    );

    for path in ["/v1/messages/count_tokens", "/messages/count_tokens"] {
        let (status, response) = post(text_only_app(), path, &body).await;
        assert_eq!(status, messages_status, "{path}: {response}");
        assert_eq!(
            response, messages_body,
            "{path} answers what /v1/messages answers"
        );
        assert_eq!(response["type"], "error", "{path}: {response}");
        assert_eq!(
            response["error"]["type"], "not_supported_error",
            "{path}: {response}"
        );
        assert!(
            response["error"]["message"]
                .as_str()
                .expect("message string")
                .starts_with("image input is not supported"),
            "{path}: {response}"
        );
        assert!(response.get("input_tokens").is_none(), "{path}: {response}");
    }

    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "no route may connect to a URL the request names before refusing it"
    );
    server.abort();
}

#[tokio::test]
async fn an_image_inside_a_tool_result_is_refused_before_its_url_is_fetched() {
    // The translator surfaces tool-result images as `image_url` parts through a
    // branch of its own (PR #1200), so the user-turn case above does not cover
    // it.
    let _private = crate::server::media_net::allow_private_media_urls_in_tests();
    let (url, hits, server) = recording_origin().await;
    let body = serde_json::json!({
        "model": "count-tokens-test-model",
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "look at this"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "camera", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "image", "source": {"type": "url", "url": url}}
                ]}
            ]}
        ]
    });
    for path in ["/v1/messages/count_tokens", "/messages/count_tokens"] {
        let (status, response) = post(text_only_app(), path, &body).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {response}");
        assert_eq!(
            response["error"]["type"], "not_supported_error",
            "{path}: {response}"
        );
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    server.abort();
}

#[tokio::test]
async fn the_recording_origin_does_see_a_fetch_the_gate_admits() {
    // Positive control for the test above: on a checkpoint that can consume
    // images, the same body passes the gate and the render resolves the URL,
    // so the origin records the connection. Without this, a zero count above
    // could mean the harness never observes fetches at all.
    let _private = crate::server::media_net::allow_private_media_urls_in_tests();
    let (url, hits, server) = recording_origin().await;
    let _ = post(
        image_capable_app(),
        "/v1/messages/count_tokens",
        &image_body(&url),
    )
    .await;
    assert!(
        hits.load(Ordering::SeqCst) >= 1,
        "an admitted image URL is fetched by the render"
    );
    server.abort();
}

#[tokio::test]
async fn a_text_only_body_is_still_counted() {
    let body = serde_json::json!({
        "model": "count-tokens-test-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let (status, response) = post(text_only_app(), "/v1/messages/count_tokens", &body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert!(response["input_tokens"].as_u64().is_some(), "{response}");
}
