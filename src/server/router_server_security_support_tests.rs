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

pub(super) fn assert_webui_error_schema(json: &serde_json::Value, code: &str, retryable: bool) {
    const ALLOWED: &[&str] = &[
        "invalid_request",
        "unauthorized",
        "forbidden",
        "not_found",
        "stale_revision",
        "conflict",
        "unsupported",
        "rate_limited",
        "unavailable",
        "payload_too_large",
        "server_restarted",
        "event_gap",
        "partial_success",
    ];
    assert!(
        ALLOWED.contains(&code),
        "test expected invalid schema code {code}"
    );
    let envelope: crate::server::router_lifecycle::ErrorEnvelope =
        serde_json::from_value(json.clone()).expect("producer error envelope DTO");
    assert_eq!(serde_json::to_value(&envelope).expect("serialize"), *json);
    assert!(
        ALLOWED.contains(&envelope.error.code.as_str()),
        "producer emitted invalid schema code {}",
        envelope.error.code
    );
    assert_eq!(envelope.error.code, code);
    assert_eq!(envelope.error.retryable, retryable);
    assert!(!envelope.error.message.is_empty());
    assert!(envelope.error.message.len() <= 512);
    assert!(!envelope.request_id.is_empty());
}
