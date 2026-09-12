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

//! Adversarial tests for the secured WebUI router mount.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderValue, Method, Request, StatusCode, header};
use tower::ServiceExt;

use super::{RouterServerState, create_router_app_with_secured_ui};
use crate::server::ServerStartupConfig;
use crate::server::config::ServerConfig;
use crate::server::router_models::{RouterPool, RouterSources};
use crate::server::router_presets::PresetCliOverrides;

const ROUTER_KEY: &str = "router-key";

fn temp_models_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel-router-app-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create models dir");
    dir
}

fn add_fake_model(root: &Path, name: &str) {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("model dir");
    std::fs::write(dir.join("config.json"), "{}").expect("config.json");
}

fn keyed_config() -> ServerConfig {
    ServerConfig {
        api_keys: crate::server::resolve_api_keys(&[ROUTER_KEY.to_string()], &[]).expect("keys"),
        ..Default::default()
    }
}

fn router_state_from(
    sources: RouterSources,
    config: ServerConfig,
    autoload: bool,
) -> RouterServerState {
    let pool = Arc::new(
        RouterPool::new(
            sources,
            ServerStartupConfig::default(),
            config.api_keys.clone(),
            PresetCliOverrides::default(),
            4,
            autoload,
        )
        .expect("pool"),
    );
    RouterServerState {
        pool,
        config: Arc::new(config),
    }
}

fn secured_router_app_with_limits(control_limit: usize, sse_limit: usize) -> Router {
    let root = temp_models_dir("secured-webui");
    add_fake_model(&root, "alpha");
    let sources = RouterSources {
        models_dir: Some(root),
        cache: None,
        presets: Default::default(),
    };
    let state = router_state_from(sources, keyed_config(), true);
    let policy = crate::server::webui::security::WebUiSecurityPolicy::with_prefixes_and_limits(
        vec!["127.0.0.1:18037".to_string()],
        vec![HeaderValue::from_static("http://127.0.0.1:18037")],
        "/webui",
        "/ui-api/v1",
        control_limit,
        sse_limit,
    )
    .expect("security policy");
    create_router_app_with_secured_ui(state, policy)
}

async fn secured_request(
    app: Router,
    method: Method,
    uri: &str,
    bearer: Option<&str>,
    origin: Option<&str>,
    fetch_site: Option<&str>,
    body: Body,
) -> axum::response::Response {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::HOST, "127.0.0.1:18037");
    if let Some(bearer) = bearer {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {bearer}"));
    }
    if let Some(origin) = origin {
        builder = builder.header(header::ORIGIN, origin);
    }
    if let Some(fetch_site) = fetch_site {
        builder = builder.header("sec-fetch-site", fetch_site);
    }
    app.oneshot(builder.body(body).expect("request builds"))
        .await
        .expect("router answers")
}

#[tokio::test]
async fn secured_webui_keeps_health_public_and_private_routes_keyed() {
    let app = secured_router_app_with_limits(32, 16);
    let health = secured_request(
        app.clone(),
        Method::GET,
        "/health",
        None,
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(health.status(), StatusCode::OK);
    assert!(
        health
            .headers()
            .get(header::CONTENT_SECURITY_POLICY)
            .is_some()
    );
    assert!(health.headers().get("permissions-policy").is_some());
    let cross_site_health = secured_request(
        app.clone(),
        Method::GET,
        "/health",
        None,
        Some("https://status.example"),
        Some("cross-site"),
        Body::empty(),
    )
    .await;
    assert_eq!(cross_site_health.status(), StatusCode::OK);

    let public_query_key = secured_request(
        app.clone(),
        Method::GET,
        "/webui?token=seeded-secret",
        None,
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(public_query_key.status(), StatusCode::BAD_REQUEST);

    let missing = secured_request(
        app.clone(),
        Method::GET,
        "/props",
        None,
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
    assert!(
        missing
            .headers()
            .get(header::CONTENT_SECURITY_POLICY)
            .is_some()
    );

    let ui_missing = secured_request(
        app.clone(),
        Method::GET,
        "/ui-api/v1/operations",
        None,
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(ui_missing.status(), StatusCode::UNAUTHORIZED);
    let ui_body = axum::body::to_bytes(ui_missing.into_body(), 4096)
        .await
        .expect("ui auth body");
    assert!(String::from_utf8_lossy(&ui_body).contains("\"code\":\"authentication\""));
    let invalid = secured_request(
        app.clone(),
        Method::GET,
        "/props",
        Some("wrong"),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(invalid.status(), StatusCode::UNAUTHORIZED);
    let ok = secured_request(
        app,
        Method::GET,
        "/props",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(ok.status(), StatusCode::OK);
}

#[tokio::test]
async fn secured_webui_rejects_host_origin_fetch_and_query_credential_attacks() {
    let app = secured_router_app_with_limits(32, 16);
    let hostile_origin = secured_request(
        app.clone(),
        Method::GET,
        "/props",
        Some(ROUTER_KEY),
        Some("https://evil.example"),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(hostile_origin.status(), StatusCode::FORBIDDEN);
    assert!(
        hostile_origin
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none()
    );

    let null_origin = secured_request(
        app.clone(),
        Method::GET,
        "/props",
        Some(ROUTER_KEY),
        Some("null"),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(null_origin.status(), StatusCode::FORBIDDEN);

    let cross_site = secured_request(
        app.clone(),
        Method::POST,
        "/ui-api/v1/model-actions",
        Some(ROUTER_KEY),
        None,
        Some("cross-site"),
        Body::empty(),
    )
    .await;
    assert_eq!(cross_site.status(), StatusCode::FORBIDDEN);

    let query_key = secured_request(
        app.clone(),
        Method::GET,
        "/props?api%5Fkey=seeded-secret",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(query_key.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(query_key.into_body(), 4096)
        .await
        .expect("body");
    assert!(!String::from_utf8_lossy(&body).contains("seeded-secret"));

    let bad_host = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/props")
                .header(header::HOST, "evil.example")
                .header(header::AUTHORIZATION, format!("Bearer {ROUTER_KEY}"))
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router answers");
    assert_eq!(bad_host.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn secured_webui_allows_configured_preflight_without_bearer() {
    let app = secured_router_app_with_limits(32, 16);
    let response = secured_request(
        app,
        Method::OPTIONS,
        "/ui-api/v1/model-actions",
        None,
        Some("http://127.0.0.1:18037"),
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
        "http://127.0.0.1:18037"
    );
}

#[tokio::test]
async fn secured_webui_protects_legacy_get_mutation_and_encoded_private_paths() {
    let app = secured_router_app_with_limits(32, 16);
    let reload = secured_request(
        app.clone(),
        Method::GET,
        "/models?reload=1",
        None,
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(reload.status(), StatusCode::UNAUTHORIZED);
    let encoded = secured_request(
        app,
        Method::GET,
        "/ui-api%2fv1/model-actions",
        None,
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(encoded.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn secured_webui_bounds_sse_connections_until_response_drop() {
    let app = secured_router_app_with_limits(32, 1);
    let first = secured_request(
        app.clone(),
        Method::GET,
        "/ui-api/v1/events",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    let second = secured_request(
        app.clone(),
        Method::GET,
        "/ui-api/v1/events",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    drop(first);
    let third = secured_request(
        app,
        Method::GET,
        "/ui-api/v1/events",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(third.status(), StatusCode::OK);
}

#[tokio::test]
async fn secured_webui_bounds_control_capacity_and_declared_body_size() {
    let app = secured_router_app_with_limits(0, 16);
    let limited = secured_request(
        app.clone(),
        Method::POST,
        "/ui-api/v1/model-actions",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(limited.status(), StatusCode::TOO_MANY_REQUESTS);

    let app = secured_router_app_with_limits(32, 16);
    let oversized = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/ui-api/v1/model-actions")
                .header(header::HOST, "127.0.0.1:18037")
                .header(header::AUTHORIZATION, format!("Bearer {ROUTER_KEY}"))
                .header(
                    header::CONTENT_LENGTH,
                    (crate::server::webui::security::WEBUI_CONTROL_BODY_LIMIT_BYTES + 1)
                        .to_string(),
                )
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router answers");
    assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);

    let chunked = secured_request(
        app,
        Method::POST,
        "/ui-api/v1/model-actions",
        Some(ROUTER_KEY),
        None,
        None,
        Body::from(vec![
            b'x';
            crate::server::webui::security::WEBUI_CONTROL_BODY_LIMIT_BYTES
                + 1
        ]),
    )
    .await;
    assert_eq!(chunked.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
