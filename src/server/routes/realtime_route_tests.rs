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

//! App-level wiring of `/v1/realtime` (#1376): mounted only with an engine,
//! behind the API-key layer, and named by the chat routes' 501.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;

use crate::models::nemotron_voicechat::streaming::VoiceChatError;
use crate::server::realtime_engine::{
    RealtimeModel, RealtimeSession, RealtimeSessionConfig, RealtimeVoiceChatEngine,
};
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

struct NoSessions;

impl RealtimeModel for NoSessions {
    fn open_session(
        &self,
        _config: &RealtimeSessionConfig,
    ) -> Result<Box<dyn RealtimeSession + '_>, VoiceChatError> {
        Err(VoiceChatError::Inference("route test".to_string()))
    }
}

fn state(config: ServerConfig, with_engine: bool) -> AppState {
    let provider = Arc::new(ModelProvider::chat_unavailable_for_route_tests());
    let batch_metrics = provider.batch_metrics().clone();
    let engine = with_engine.then(|| {
        Arc::new(
            RealtimeVoiceChatEngine::spawn_with_loader("voicechat", || Ok(NoSessions))
                .expect("fake engine"),
        )
    });
    AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_realtime_engine(engine)
}

async fn status_and_body(app: axum::Router, method: Method, path: &str) -> (StatusCode, String) {
    let body = if method == Method::POST {
        Body::from(
            json!({"model": "voicechat", "prompt": "hi",
                   "messages": [{"role": "user", "content": "hi"}]})
            .to_string(),
        )
    } else {
        Body::empty()
    };
    let response = app
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(body)
                .expect("request builds"),
        )
        .await
        .expect("route responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn realtime_server_points_chat_routes_at_the_socket() {
    for path in ["/v1/chat/completions", "/v1/completions"] {
        let app = create_app(state(ServerConfig::default(), true));
        let (status, body) = status_and_body(app, Method::POST, path).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {body}");
        assert!(body.contains("/v1/realtime"), "{path}: {body}");
    }
    let app = create_app(state(ServerConfig::default(), true));
    let (status, body) = status_and_body(app, Method::GET, "/health").await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn realtime_route_is_mounted_only_with_an_engine_and_behind_api_keys() {
    // Without an engine the path does not exist.
    let app = create_app(state(ServerConfig::default(), false));
    let (status, _) = status_and_body(app, Method::GET, "/v1/realtime").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // With one, a plain GET reaches the upgrade extractor and is refused as
    // a non-upgrade request rather than 404.
    let app = create_app(state(ServerConfig::default(), true));
    let (status, _) = status_and_body(app, Method::GET, "/v1/realtime").await;
    assert_ne!(status, StatusCode::NOT_FOUND);
    assert!(status.is_client_error(), "{status}");

    // `--api-key` guards the upgrade request.
    let config = ServerConfig {
        api_keys: crate::server::resolve_api_keys(&["realtime-key".to_string()], &[])
            .expect("valid key set"),
        ..Default::default()
    };
    let app = create_app(state(config, true));
    let (status, _) = status_and_body(app, Method::GET, "/v1/realtime").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

async fn upgrade_status(app: axum::Router, origin: Option<&str>) -> (StatusCode, String) {
    let mut request = Request::builder()
        .method(Method::GET)
        .uri("/v1/realtime")
        .header("connection", "upgrade")
        .header("upgrade", "websocket")
        .header("sec-websocket-version", "13")
        .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==");
    if let Some(origin) = origin {
        request = request.header("origin", origin);
    }
    let response = app
        .oneshot(request.body(Body::empty()).expect("request builds"))
        .await
        .expect("route responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn allow_list_config(api_key: Option<&str>) -> ServerConfig {
    let list = crate::server::cors::parse_allowed_origins(&["https://app.example.com".to_string()])
        .expect("valid origin");
    let cors_policy = crate::server::CorsPolicy::resolve("*", "GET", "*", true, Some(list))
        .expect("valid policy");
    let api_keys = api_key.map_or_else(Default::default, |key| {
        crate::server::resolve_api_keys(&[key.to_string()], &[]).expect("valid key set")
    });
    ServerConfig {
        cors_policy,
        api_keys,
        ..Default::default()
    }
}

#[tokio::test]
async fn realtime_upgrade_enforces_the_origin_policy_in_both_app_builds() {
    // #2042: the check lives in the route, so router-mode sub-apps (built
    // without the CORS layer) enforce it too.
    for with_cors in [true, false] {
        let build = |config| {
            let state = state(config, true);
            if with_cors {
                create_app(state)
            } else {
                crate::server::app::create_app_without_cors(state)
            }
        };
        let (status, body) =
            upgrade_status(build(allow_list_config(None)), Some("https://evil.example")).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "cors={with_cors}: {body}");
        assert!(
            body.contains("origin not allowed for /v1/realtime"),
            "{body}"
        );
        assert!(body.contains("invalid_request_error"), "{body}");

        // Allowed and absent origins pass the check and reach the upgrade
        // extractor, which refuses the in-process request (no hyper upgrade
        // handle) with something other than 403.
        for origin in [Some("https://app.example.com"), None] {
            let (status, body) = upgrade_status(build(allow_list_config(None)), origin).await;
            assert_ne!(status, StatusCode::FORBIDDEN, "{origin:?}: {body}");
            assert_ne!(status, StatusCode::NOT_FOUND, "{origin:?}: {body}");
        }

        // The API-key layer stays outside: failing both checks is 401.
        let (status, _) = upgrade_status(
            build(allow_list_config(Some("realtime-key"))),
            Some("https://evil.example"),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "cors={with_cors}");
    }
}
