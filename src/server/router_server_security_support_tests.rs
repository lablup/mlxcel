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

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderValue, Method, Request, header};
use tower::ServiceExt;

use super::{RouterServerState, create_router_app_with_secured_ui};
use crate::server::ServerStartupConfig;
use crate::server::config::ServerConfig;
use crate::server::router_models::{RouterPool, RouterSources};
use crate::server::router_presets::PresetCliOverrides;

pub(super) const ROUTER_KEY: &str = "router-key";

#[derive(Clone)]
pub(super) struct SharedBufWriter(pub(super) Arc<Mutex<Vec<u8>>>);

pub(super) struct SharedBufGuard(Arc<Mutex<Vec<u8>>>);

impl io::Write for SharedBufGuard {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer lock")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedBufWriter {
    type Writer = SharedBufGuard;

    fn make_writer(&'a self) -> Self::Writer {
        SharedBufGuard(self.0.clone())
    }
}

fn temp_models_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mlxcel-router-app-{tag}-{}", uuid::Uuid::new_v4()));
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

fn secured_router_app(control_limit: usize, sse_limit: usize, control_rate_limit: usize) -> Router {
    let root = temp_models_dir("secured-webui");
    add_fake_model(&root, "alpha");
    let sources = RouterSources {
        models_dir: Some(root),
        cache: None,
        presets: Default::default(),
    };
    let state = router_state_from(sources, keyed_config(), true);
    let policy =
        crate::server::webui::security::WebUiSecurityPolicy::with_prefixes_limits_and_rate(
            vec!["127.0.0.1:18037".to_string()],
            vec![HeaderValue::from_static("http://127.0.0.1:18037")],
            "/webui",
            "/",
            control_limit,
            sse_limit,
            control_rate_limit,
        )
        .expect("security policy");
    create_router_app_with_secured_ui(state, policy)
}

pub(super) fn secured_router_app_with_limits(control_limit: usize, sse_limit: usize) -> Router {
    secured_router_app(control_limit, sse_limit, 120)
}

pub(super) fn secured_router_app_with_rate_limit(control_rate_limit: usize) -> Router {
    secured_router_app(32, 16, control_rate_limit)
}

pub(super) async fn secured_request(
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

// Each expected value is independently validated by verify-webui-contract.
// Normalize ONLY the dynamic request ID after validating its complete format.
fn matches_error_fixture(actual: &serde_json::Value, expected: &serde_json::Value) -> bool {
    let mut normalized = actual.clone();
    let Some(request_id) = normalized.get_mut("request_id") else {
        return false;
    };
    if !request_id.as_str().is_some_and(|id| {
        !id.is_empty()
            && id.len() <= 128
            && id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'~' | b'-'))
    }) {
        return false;
    }
    *request_id = expected["request_id"].clone();
    normalized == *expected
}

pub(super) fn assert_webui_error_fixture(json: &serde_json::Value, fixture: &str) {
    let expected: serde_json::Value = serde_json::from_str(fixture).expect("canonical fixture");
    assert!(
        matches_error_fixture(json, &expected),
        "whole error contract mismatch: {json}, expected: {expected}"
    );
}

#[test]
fn error_fixture_comparison_rejects_missing_extra_static_and_dynamic_drift() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/webui/examples/error.security-unauthorized.json"
    ))
    .unwrap();
    assert!(matches_error_fixture(&expected, &expected));
    for (parent, key) in [
        ("", "error"),
        ("", "request_id"),
        ("/error", "code"),
        ("/error", "message"),
        ("/error", "retryable"),
    ] {
        let mut changed = expected.clone();
        changed
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(key);
        assert!(
            !matches_error_fixture(&changed, &expected),
            "accepted missing {parent}/{key}"
        );
    }
    for (parent, key) in [
        ("", "unexpected"),
        ("/error", "unexpected"),
        ("/error", "field_errors"),
        ("/error", "operation_id"),
    ] {
        let mut changed = expected.clone();
        changed
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(key.into(), serde_json::Value::Null);
        assert!(
            !matches_error_fixture(&changed, &expected),
            "accepted extra/null {parent}/{key}"
        );
    }
    for (path, invalid) in [
        ("/error/code", serde_json::json!("invented_error")),
        ("/error/message", serde_json::json!("different message")),
        ("/error/retryable", serde_json::json!(true)),
        ("/request_id", serde_json::json!("")),
        ("/request_id", serde_json::json!("x".repeat(129))),
        ("/request_id", serde_json::json!("token\n")),
        ("/request_id", serde_json::json!("bad/token")),
        ("/request_id", serde_json::Value::Null),
    ] {
        let mut changed = expected.clone();
        *changed.pointer_mut(path).unwrap() = invalid;
        assert!(
            !matches_error_fixture(&changed, &expected),
            "accepted invalid {path}"
        );
    }
}

#[tokio::test]
async fn every_admin_family_enforces_rate_capacity_and_body_limits() {
    use axum::http::StatusCode;
    let paths = [
        (Method::POST, "/props"),
        (Method::PATCH, "/v1/settings"),
        (Method::POST, "/slots/0?action=save"),
        (Method::POST, "/lora-adapters"),
        (Method::POST, "/v1/cache/reset"),
        (Method::POST, "/ui-api/v1/catalog/refresh"),
        (Method::POST, "/ui-api/v1/downloads"),
        (Method::DELETE, "/ui-api/v1/model-removals"),
        (Method::PATCH, "/ui-api/v1/future-admin-action"),
    ];
    for (method, path) in paths {
        for (app, body, status, fixture) in [
            (
                secured_router_app_with_rate_limit(0),
                Body::empty(),
                StatusCode::TOO_MANY_REQUESTS,
                include_str!("../../tests/fixtures/webui/examples/error.security-rate.json"),
            ),
            (
                secured_router_app_with_limits(0, 16),
                Body::empty(),
                StatusCode::TOO_MANY_REQUESTS,
                include_str!("../../tests/fixtures/webui/examples/error.security-capacity.json"),
            ),
            (
                secured_router_app_with_limits(32, 16),
                Body::from(vec![
                    b'x';
                    crate::server::webui::security::WEBUI_CONTROL_BODY_LIMIT_BYTES
                        + 1
                ]),
                StatusCode::PAYLOAD_TOO_LARGE,
                include_str!("../../tests/fixtures/webui/examples/error.security-body.json"),
            ),
        ] {
            let response = secured_request(
                app,
                method.clone(),
                path,
                Some(ROUTER_KEY),
                None,
                None,
                body,
            )
            .await;
            assert_eq!(response.status(), status, "{method} {path}");
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            assert_webui_error_fixture(&serde_json::from_slice(&bytes).unwrap(), fixture);
        }
    }
}
