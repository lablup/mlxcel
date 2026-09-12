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
    router_state_with_startup(sources, config, ServerStartupConfig::default(), autoload)
}

fn router_state_with_startup(
    sources: RouterSources,
    config: ServerConfig,
    startup: ServerStartupConfig,
    autoload: bool,
) -> RouterServerState {
    let pool = Arc::new(
        RouterPool::new(
            sources,
            startup.clone(),
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
        #[cfg(feature = "webui")]
        startup: Arc::new(startup),
        #[cfg(feature = "webui")]
        catalog_cache: Arc::new(crate::server::webui::catalog::CatalogProjectionCache::new()),
    }
}

fn secured_router_app_from_state(state: RouterServerState) -> Router {
    let policy =
        crate::server::webui::security::WebUiSecurityPolicy::with_prefixes_limits_and_rate(
            vec!["127.0.0.1:18037".to_string()],
            vec![HeaderValue::from_static("http://127.0.0.1:18037")],
            "/webui",
            "/",
            32,
            16,
            120,
        )
        .expect("security policy");
    create_router_app_with_secured_ui(state, policy)
}

fn normalize_bootstrap_fixture(value: &mut serde_json::Value, expected: &serde_json::Value) {
    let server_id = value
        .pointer("/server/server_instance_id")
        .and_then(|v| v.as_str())
        .expect("server_instance_id string");
    assert!(
        server_id.starts_with("srv_") && server_id.len() <= 128,
        "unexpected server_instance_id: {server_id}"
    );
    *value.pointer_mut("/server/server_instance_id").unwrap() =
        expected["server"]["server_instance_id"].clone();

    let build = value
        .pointer_mut("/server/build")
        .and_then(|v| v.as_object_mut())
        .expect("build object");
    let version = build
        .get("version")
        .and_then(|v| v.as_str())
        .expect("build.version string");
    assert!(!version.is_empty(), "build.version must not be empty");
    build.insert(
        "version".to_string(),
        expected["server"]["build"]["version"].clone(),
    );

    let git_commit = build.get("git_commit").expect("build.git_commit present");
    assert!(
        git_commit.is_null()
            || git_commit.as_str().is_some_and(|commit| {
                commit.len() <= 64 && commit.bytes().all(|byte| byte.is_ascii_hexdigit())
            }),
        "build.git_commit must be null or a hex commit"
    );
    build.insert(
        "git_commit".to_string(),
        expected["server"]["build"]["git_commit"].clone(),
    );

    let target = build
        .get("target")
        .and_then(|v| v.as_str())
        .expect("build.target string");
    assert!(target.contains('-'), "build.target must describe arch-os");
    build.insert(
        "target".to_string(),
        expected["server"]["build"]["target"].clone(),
    );
}

struct NoBootstrapDownload;

impl crate::server::router_cache::RouterDownloader for NoBootstrapDownload {
    fn validate(&self, _: &str) -> anyhow::Result<()> {
        panic!("bootstrap must not probe the network");
    }

    fn download(
        &self,
        _: &str,
        _: &Path,
        _: crate::downloader::DownloadHooks,
    ) -> anyhow::Result<()> {
        panic!("bootstrap must not download a model");
    }
}

async fn assert_mounted_bootstrap_fixture(with_cache: bool) {
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    let cache_root = tempfile::tempdir().expect("bootstrap fixture cache");
    let mut startup = ServerStartupConfig {
        model_store_root: Some(cache_root.path().to_path_buf()),
        ..Default::default()
    };
    startup.webui_enabled = true;
    let config = ServerConfig {
        enable_settings_endpoint: true,
        ..keyed_config()
    };
    let state = router_state_with_startup(
        RouterSources {
            models_dir: None,
            cache: with_cache.then(|| {
                crate::server::router_cache::CacheSource::new(
                    cache_root.path().to_path_buf(),
                    Arc::new(NoBootstrapDownload),
                )
            }),
            presets: Default::default(),
        },
        config,
        startup,
        false,
    );
    let response = secured_request(
        secured_router_app_from_state(state),
        Method::GET,
        "/ui-api/v1/bootstrap",
        Some(ROUTER_KEY),
        None,
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 32 * 1024).await.unwrap();
    let mut actual: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/webui/examples/bootstrap.model-free.json"
    ))
    .unwrap();
    if !with_cache {
        // The no-cache case intentionally keeps a startup hint, but its pool
        // has no cache source. Change only expected cache-dependent fields;
        // compare every producer field without normalizing away this state.
        expected["roots"] = serde_json::json!([]);
        for action in ["download", "cache_delete"] {
            expected["actions"][action]["reason"] =
                serde_json::json!("no writable managed cache route is available in this mode");
        }
    }
    normalize_bootstrap_fixture(&mut actual, &expected);
    assert_eq!(
        actual, expected,
        "mounted bootstrap producer drifted from the schema-validated fixture"
    );
}

#[tokio::test]
async fn mounted_bootstrap_matches_full_model_free_fixture() {
    assert_mounted_bootstrap_fixture(true).await;
}

#[tokio::test]
async fn mounted_bootstrap_without_pool_cache_ignores_startup_cache_hint() {
    assert_mounted_bootstrap_fixture(false).await;
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

#[tokio::test]
async fn secured_router_mounts_static_bootstrap_catalog_runtime_and_events() {
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    let shell = secured_request(
        secured_router_app_with_limits(32, 16),
        Method::GET,
        "/webui/",
        None,
        None,
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(shell.status(), StatusCode::OK);

    let unauthenticated_bootstrap = secured_request(
        secured_router_app_with_limits(32, 16),
        Method::GET,
        "/ui-api/v1/bootstrap",
        None,
        None,
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(unauthenticated_bootstrap.status(), StatusCode::UNAUTHORIZED);

    let bootstrap = secured_request(
        secured_router_app_with_limits(32, 16),
        Method::GET,
        "/ui-api/v1/bootstrap",
        Some(ROUTER_KEY),
        None,
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(bootstrap.status(), StatusCode::OK);
    let bytes = to_bytes(bootstrap.into_body(), 32 * 1024).await.unwrap();
    let bootstrap_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(bootstrap_json["schema_version"], "webui.ui-api.v1");
    assert_eq!(bootstrap_json["server"]["mode"], "router_pool");
    assert_eq!(bootstrap_json["actions"]["download"]["state"], "read_only");

    let catalog = secured_request(
        secured_router_app_with_limits(32, 16),
        Method::GET,
        "/ui-api/v1/catalog?autoload=false",
        Some(ROUTER_KEY),
        None,
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(catalog.status(), StatusCode::OK);
    let bytes = to_bytes(catalog.into_body(), 64 * 1024).await.unwrap();
    let catalog_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let model_id = catalog_json["items"][0]["identity"]["id"]
        .as_str()
        .expect("catalog id")
        .to_string();

    let runtime = secured_request(
        secured_router_app_with_limits(32, 16),
        Method::GET,
        &format!("/ui-api/v1/runtime?model_id={model_id}&autoload=false"),
        Some(ROUTER_KEY),
        None,
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(runtime.status(), StatusCode::OK);
    let bytes = to_bytes(runtime.into_body(), 32 * 1024).await.unwrap();
    let runtime_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(runtime_json["model_id"], model_id);
    assert!(runtime_json["measurements"]["gpu_utilization"]["value"].is_null());

    let unknown = secured_request(
        secured_router_app_with_limits(32, 16),
        Method::GET,
        &format!(
            "/ui-api/v1/runtime?model_id=mdl_{}&autoload=false",
            "z".repeat(43)
        ),
        Some(ROUTER_KEY),
        None,
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);

    let events = secured_request(
        secured_router_app_with_limits(32, 16),
        Method::GET,
        "/ui-api/v1/events",
        Some(ROUTER_KEY),
        None,
        Some("same-origin"),
        Body::empty(),
    )
    .await;
    assert_eq!(events.status(), StatusCode::OK);
    assert!(
        events
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream"))
    );
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

async fn assembled_router_boundary(ui_enabled: bool) {
    use axum::body::{Bytes, to_bytes};
    use axum::http::StatusCode;
    let contract: serde_json::Value =
        serde_json::from_str(include_str!("../../docs/webui/api.yaml")).unwrap();
    let limit = contract
        .pointer("/components/schemas/LimitSummary/properties/json_body_bytes/const")
        .unwrap()
        .as_u64()
        .unwrap() as usize;
    assert_eq!(limit, 2_097_152);
    let (app, path, value) = if ui_enabled {
        (
            secured_router_app_with_limits(32, 16),
            "/ui-api/v1/model-actions",
            serde_json::json!({
                "model_id": format!("mdl_{}", "a".repeat(43)), "action": "load",
                "expected_revision": 1, "idempotency_key": "boundary-action"
            }),
        )
    } else {
        let state = router_state_from(
            RouterSources {
                models_dir: None,
                cache: None,
                presets: Default::default(),
            },
            keyed_config(),
            false,
        );
        (
            super::create_router_app(state),
            "/models/load",
            serde_json::json!({"model":"missing-boundary-model"}),
        )
    };
    for declared in [true, false] {
        for (size, expected) in [
            (limit, StatusCode::NOT_FOUND),
            (limit + 1, StatusCode::PAYLOAD_TOO_LARGE),
        ] {
            let mut value = value.clone();
            if ui_enabled {
                // Each probe must reach admission, not replay a previous
                // missing-model operation through its idempotency key.
                value["idempotency_key"] = serde_json::json!(format!("boundary-{declared}-{size}"));
            }
            let mut payload = serde_json::to_vec(&value).unwrap();
            payload.resize(size, b' ');
            assert_eq!(payload.len(), size);
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&payload).unwrap(),
                value
            );
            let mut request = Request::builder()
                .method(Method::POST)
                .uri(path)
                .header(header::HOST, "127.0.0.1:18037")
                .header(header::AUTHORIZATION, format!("Bearer {ROUTER_KEY}"))
                .header(header::CONTENT_TYPE, "application/json");
            let body = if declared {
                request = request.header(header::CONTENT_LENGTH, size.to_string());
                Body::from(payload)
            } else {
                request = request.header(header::TRANSFER_ENCODING, "chunked");
                let chunks: Vec<Result<Bytes, std::io::Error>> = payload
                    .chunks(65_536)
                    .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                    .collect();
                Body::from_stream(futures::stream::iter(chunks))
            };
            let response = app
                .clone()
                .oneshot(request.body(body).unwrap())
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                expected,
                "ui={ui_enabled}, declared={declared}, size={size}"
            );
            if size == limit {
                let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
                let actual: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                if ui_enabled {
                    assert_eq!(actual["error"]["code"], "not_found");
                    assert_eq!(
                        actual["error"]["message"],
                        "model was not found; refresh the catalog before retrying"
                    );
                } else {
                    assert_eq!(actual["error"]["type"], "not_found_error");
                    assert_eq!(actual["error"]["message"], "model is not found");
                }
            }
        }
    }
}

#[tokio::test]
async fn assembled_ui_router_body_boundary_reaches_actual_action_handler() {
    assembled_router_boundary(true).await;
}

#[tokio::test]
async fn assembled_ui_off_router_body_boundary_preserves_legacy_load_handler() {
    assembled_router_boundary(false).await;
}
