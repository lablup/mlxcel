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

//! Authentication behavior on the real router (#1437).
//!
//! The unit tests in `auth_tests.rs` cover parsing and header extraction;
//! these drive `create_app` so the assertions are about what a client
//! actually gets back: which paths stay public, which status and body an
//! unauthenticated request receives, and that every configured key works
//! independently on both the `/v1` and the native route spellings.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

/// Two independent keys, standing in for `--api-key a,b`.
const KEY_ONE: &str = "first-canary-key";
const KEY_TWO: &str = "second-canary-key";

fn app_with_keys(keys: &[&str]) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let owned: Vec<String> = keys.iter().map(|k| (*k).to_string()).collect();
    let state = AppState::new(
        provider,
        ServerConfig {
            api_keys: crate::server::resolve_api_keys(&owned, &[]).expect("valid key set"),
            enable_slots_endpoint: true,
            enable_props_endpoint: true,
            enable_metrics_endpoint: true,
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("auth-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn send(
    app: Router,
    method: Method,
    path: &str,
    credential: Option<(&str, &str)>,
) -> axum::http::Response<Body> {
    let needs_body = method == Method::POST;
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some((name, value)) = credential {
        builder = builder.header(name, value);
    }
    let body = if needs_body {
        Body::from("{}")
    } else {
        Body::empty()
    };
    app.oneshot(builder.body(body).expect("request builds"))
        .await
        .expect("router responds")
}

async fn status(
    app: Router,
    method: Method,
    path: &str,
    credential: Option<(&str, &str)>,
) -> StatusCode {
    send(app, method, path, credential).await.status()
}

async fn body_text(response: axum::http::Response<Body>) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("body collects");
    String::from_utf8(bytes.to_vec()).expect("utf-8 body")
}

#[tokio::test]
async fn every_configured_key_authenticates_independently() {
    for key in [KEY_ONE, KEY_TWO] {
        let status = status(
            app_with_keys(&[KEY_ONE, KEY_TWO]),
            Method::GET,
            "/v1/models",
            Some((header::AUTHORIZATION.as_str(), &format!("Bearer {key}"))),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{key} must authenticate");
    }
}

#[tokio::test]
async fn an_unknown_key_is_rejected() {
    let status = status(
        app_with_keys(&[KEY_ONE, KEY_TWO]),
        Method::GET,
        "/v1/models",
        Some((header::AUTHORIZATION.as_str(), "Bearer not-a-key")),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn a_missing_credential_and_an_unknown_one_are_indistinguishable() {
    let missing = send(app_with_keys(&[KEY_ONE]), Method::GET, "/v1/models", None).await;
    let unknown = send(
        app_with_keys(&[KEY_ONE]),
        Method::GET,
        "/v1/models",
        Some((header::AUTHORIZATION.as_str(), "Bearer wrong")),
    )
    .await;

    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(unknown.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        body_text(missing).await,
        body_text(unknown).await,
        "a probe must not be able to tell a missing key from a wrong one"
    );
}

#[tokio::test]
async fn the_rejection_body_matches_b10621_and_leaks_no_key() {
    let response = send(
        app_with_keys(&[KEY_ONE, KEY_TWO]),
        Method::POST,
        "/v1/chat/completions",
        Some((header::AUTHORIZATION.as_str(), "Bearer wrong")),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("application/json; charset=utf-8")
    );

    let text = body_text(response).await;
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("json body");
    assert_eq!(parsed["error"]["message"], "Invalid API Key");
    assert_eq!(parsed["error"]["type"], "authentication_error");
    assert_eq!(parsed["error"]["code"], 401);
    assert!(
        !text.contains("canary"),
        "the body must not echo a key: {text}"
    );
}

#[tokio::test]
async fn a_malformed_authorization_header_is_rejected_without_leaking() {
    for header_value in ["", "Bearer", "Bearer ", "Basic dXNlcjpwYXNz", "Bearer  "] {
        let response = send(
            app_with_keys(&[KEY_ONE]),
            Method::GET,
            "/v1/models",
            Some((header::AUTHORIZATION.as_str(), header_value)),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "{header_value:?} must not authenticate"
        );
        let text = body_text(response).await;
        assert!(!text.contains("canary"), "{text}");
    }
}

#[tokio::test]
async fn the_anthropic_header_authenticates_too() {
    let status = status(
        app_with_keys(&[KEY_ONE]),
        Method::GET,
        "/v1/models",
        Some(("x-api-key", KEY_ONE)),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn the_public_endpoints_stay_open() {
    // b10621's public set: /health, /v1/health, and the front-end paths, of
    // which mlxcel has only "/".
    for path in ["/", "/health", "/v1/health"] {
        let status = status(app_with_keys(&[KEY_ONE]), Method::GET, path, None).await;
        assert_eq!(status, StatusCode::OK, "{path} must stay public");
    }
}

#[tokio::test]
async fn the_observability_endpoints_are_protected() {
    // Upstream's public set holds neither /props, /slots, /metrics nor
    // /models, so an unauthenticated scrape gets 401 on both servers.
    for path in ["/props", "/slots", "/metrics", "/models", "/v1/models"] {
        let status = status(app_with_keys(&[KEY_ONE]), Method::GET, path, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{path} must be protected");
    }
}

#[tokio::test]
async fn both_the_v1_and_the_native_route_spellings_are_protected() {
    let protected = [
        (Method::POST, "/props"),
        (Method::POST, "/slots/0"),
        (Method::POST, "/v1/chat/completions"),
        (Method::POST, "/chat/completions"),
        (Method::POST, "/v1/completions"),
        (Method::POST, "/completions"),
        (Method::POST, "/completion"),
        (Method::POST, "/v1/embeddings"),
        (Method::POST, "/embeddings"),
        (Method::POST, "/tokenize"),
        (Method::POST, "/detokenize"),
        (Method::POST, "/v1/rerank"),
        (Method::POST, "/rerank"),
        (Method::POST, "/v1/messages"),
    ];
    for (method, path) in protected {
        let unauthenticated = status(app_with_keys(&[KEY_ONE]), method.clone(), path, None).await;
        assert_eq!(
            unauthenticated,
            StatusCode::UNAUTHORIZED,
            "{method} {path} must require a key"
        );

        let authenticated = status(
            app_with_keys(&[KEY_ONE]),
            method.clone(),
            path,
            Some((header::AUTHORIZATION.as_str(), &format!("Bearer {KEY_ONE}"))),
        )
        .await;
        assert_ne!(
            authenticated,
            StatusCode::UNAUTHORIZED,
            "{method} {path} must accept the configured key"
        );
    }
}

#[tokio::test]
async fn an_unconfigured_server_authenticates_nothing() {
    // The single-key deployments that existed before #1437 keep working, and
    // so does a server with no key at all.
    let status = status(app_with_keys(&[]), Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_single_key_deployment_keeps_working() {
    let status = status(
        app_with_keys(&[KEY_ONE]),
        Method::GET,
        "/v1/models",
        Some((header::AUTHORIZATION.as_str(), &format!("Bearer {KEY_ONE}"))),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_bare_authorization_value_without_bearer_authenticates() {
    let status = status(
        app_with_keys(&[KEY_ONE]),
        Method::GET,
        "/v1/models",
        Some((header::AUTHORIZATION.as_str(), KEY_ONE)),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_preflight_is_still_answered_without_a_credential() {
    // The CORS middleware sits outside authentication, so a browser preflight
    // to a protected route succeeds even though the request that follows it
    // needs a key. This is upstream's ordering.
    let status = status(
        app_with_keys(&[KEY_ONE]),
        Method::OPTIONS,
        "/v1/chat/completions",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn no_authenticated_endpoint_echoes_a_configured_key() {
    // "Keep secret values out of help, logs, debug output, metrics, and error
    // bodies": the observability surfaces are reachable WITH a key, so this
    // asserts the bodies they return carry no key material.
    for (method, path) in [
        (Method::GET, "/props"),
        (Method::GET, "/slots"),
        (Method::GET, "/metrics"),
        (Method::GET, "/v1/models"),
        (Method::GET, "/health"),
    ] {
        let response = send(
            app_with_keys(&[KEY_ONE, KEY_TWO]),
            method.clone(),
            path,
            Some((header::AUTHORIZATION.as_str(), &format!("Bearer {KEY_ONE}"))),
        )
        .await;
        let text = body_text(response).await;
        assert!(
            !text.contains("canary"),
            "{method} {path} echoed a configured key: {text}"
        );
    }
}
