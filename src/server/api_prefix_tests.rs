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

//! Route discovery under `--api-prefix` (#1432).
//!
//! `create_app` nests the whole route set under the configured prefix, so
//! these tests assert what a client can actually reach: the prefixed paths
//! resolve, the unprefixed ones stop resolving, and the authentication
//! boundary lands where `llama-server` b10621 puts it.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn app(api_prefix: &str, api_key: Option<&str>) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig {
            api_prefix: api_prefix.to_string(),
            api_keys: crate::server::resolve_api_keys(
                &api_key.map(str::to_string).into_iter().collect::<Vec<_>>(),
                &[],
            )
            .expect("valid key set"),
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("prefix-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn status(app: Router, method: Method, path: &str) -> StatusCode {
    let needs_body = method == Method::POST;
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(if needs_body {
            Body::from("{}")
        } else {
            Body::empty()
        })
        .expect("request builds");
    app.oneshot(request)
        .await
        .expect("router responds")
        .status()
}

fn resolved(status: StatusCode) -> bool {
    status != StatusCode::NOT_FOUND && status != StatusCode::METHOD_NOT_ALLOWED
}

#[tokio::test]
async fn without_a_prefix_the_routes_stay_at_the_root() {
    assert_eq!(
        status(app("", None), Method::GET, "/health").await,
        StatusCode::OK
    );
    assert!(
        !resolved(status(app("", None), Method::GET, "/llama/health").await),
        "an unconfigured prefix must not be served"
    );
}

#[tokio::test]
async fn a_prefix_moves_every_route_under_it() {
    for (method, path) in [
        (Method::GET, "/llama/health"),
        (Method::GET, "/llama/v1/models"),
        (Method::POST, "/llama/v1/chat/completions"),
        (Method::POST, "/llama/completion"),
        (Method::POST, "/llama/tokenize"),
    ] {
        let status = status(app("/llama", None), method.clone(), path).await;
        assert!(
            resolved(status),
            "{method} {path} must resolve under --api-prefix /llama, got {status}"
        );
    }
}

#[tokio::test]
async fn a_prefix_removes_the_unprefixed_paths() {
    for (method, path) in [
        (Method::GET, "/health"),
        (Method::GET, "/v1/models"),
        (Method::POST, "/v1/chat/completions"),
    ] {
        let status = status(app("/llama", None), method.clone(), path).await;
        assert!(
            !resolved(status),
            "{method} {path} must stop resolving once a prefix is set, got {status}"
        );
    }
}

#[tokio::test]
async fn a_nested_prefix_is_served_verbatim() {
    assert_eq!(
        status(app("/a/b", None), Method::GET, "/a/b/health").await,
        StatusCode::OK
    );
}

#[tokio::test]
async fn the_prefixed_health_route_requires_authentication_like_b10621() {
    // Upstream compares its public-endpoint set against `req.path`, which
    // carries the prefix, while the set holds the unprefixed "/health" and
    // "/v1/health". A prefixed health check therefore needs the API key.
    // mlxcel matches that, and startup warns when both are configured.
    let status = status(app("/llama", Some("secret")), Method::GET, "/llama/health").await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "a prefixed health endpoint is not in the public set"
    );
}

#[tokio::test]
async fn an_authenticated_request_reaches_a_prefixed_route() {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/llama/health")
        .header(header::AUTHORIZATION, "Bearer secret")
        .body(Body::empty())
        .expect("request builds");
    let response = app("/llama", Some("secret"))
        .oneshot(request)
        .await
        .expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn the_unprefixed_health_path_is_public_but_unmounted() {
    // The public-path check happens before routing, so /health passes
    // authentication and then 404s because no route is mounted there. This is
    // the same pair of decisions upstream makes.
    let status = status(app("/llama", Some("secret")), Method::GET, "/health").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
