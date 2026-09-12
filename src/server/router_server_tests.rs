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

//! Route-level tests for the router-mode surface (issue #1438): the router
//! model inventory, load/unload refusals, add/delete refusals, the dispatch
//! contract (missing / unknown / not-loaded model), the router `/props`
//! block, and authorization on the management routes.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use futures::StreamExt;
use tower::ServiceExt;

use super::{
    RouterServerState, buffer_dispatch_body, create_router_app,
    create_router_app_with_authenticated_ui, dispatch_model_name, parse_query,
};
use crate::downloader::DownloadHooks;
use crate::server::ServerStartupConfig;
use crate::server::config::ServerConfig;
use crate::server::router_cache::{CacheSource, RouterDownloader};
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

fn add_fake_model(root: &std::path::Path, name: &str) {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("model dir");
    std::fs::write(dir.join("config.json"), "{}").expect("config.json");
}

fn add_catalog_model(root: &std::path::Path, name: &str, model_type: &str) {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("model dir");
    std::fs::write(
        dir.join("config.json"),
        format!(r#"{{"model_type":"{model_type}","quantization_config":{{"bits":4}}}}"#),
    )
    .expect("config.json");
    std::fs::write(dir.join("model.safetensors"), b"weights").expect("weights");
}

/// Instant local "downloader" for route tests: materializes the snapshot and
/// reports one terminal progress tick.
struct InstantDownloader;

impl RouterDownloader for InstantDownloader {
    fn validate(&self, _repo_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    fn download(
        &self,
        repo_id: &str,
        dest_root: &Path,
        hooks: DownloadHooks,
    ) -> anyhow::Result<()> {
        let dest = dest_root.join(repo_id);
        std::fs::create_dir_all(&dest)?;
        std::fs::write(dest.join("config.json"), "{}")?;
        if let Some(progress) = &hooks.progress {
            progress(&format!("https://example.invalid/{repo_id}"), 1, 1);
        }
        Ok(())
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

fn keyed_config() -> ServerConfig {
    ServerConfig {
        api_keys: crate::server::resolve_api_keys(&[ROUTER_KEY.to_string()], &[]).expect("keys"),
        ..Default::default()
    }
}

fn router_app_with(root: PathBuf, config: ServerConfig, autoload: bool) -> Router {
    let sources = RouterSources {
        models_dir: Some(root),
        cache: None,
        presets: Default::default(),
    };
    create_router_app(router_state_from(sources, config, autoload))
}

fn router_app_with_cache(
    root: PathBuf,
    cache_root: PathBuf,
    config: ServerConfig,
) -> (Router, RouterServerState) {
    let sources = RouterSources {
        models_dir: Some(root),
        cache: Some(CacheSource::new(cache_root, Arc::new(InstantDownloader))),
        presets: Default::default(),
    };
    let state = router_state_from(sources, config, true);
    (create_router_app(state.clone()), state)
}

/// Status without reading the body, for endpoints whose body never ends
/// (the SSE stream).
async fn status_only(
    app: Router,
    method: Method,
    uri: &str,
    body: &str,
    bearer: Option<&str>,
) -> StatusCode {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(key) = bearer {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let request = builder.body(Body::from(body.to_string())).expect("request");
    app.oneshot(request).await.expect("response").status()
}

async fn send(
    app: Router,
    method: Method,
    uri: &str,
    body: &str,
    bearer: Option<&str>,
) -> (StatusCode, serde_json::Value) {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(key) = bearer {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let request = builder.body(Body::from(body.to_string())).expect("request");
    let response = app.oneshot(request).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[path = "router_contract_test_support.rs"]
mod contract;
#[path = "router_catalog_route_tests.rs"]
mod router_catalog_route_tests;

#[tokio::test]
async fn the_router_inventory_carries_the_b10621_model_object() {
    let root = temp_models_dir("inventory");
    add_fake_model(&root, "alpha");
    add_fake_model(&root, "beta");
    let app = router_app_with(root, ServerConfig::default(), true);
    let (status, body) = send(app, Method::GET, "/models", "", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data");
    assert_eq!(data.len(), 2);
    let entry = &data[0];
    for key in [
        "id",
        "aliases",
        "tags",
        "object",
        "owned_by",
        "created",
        "status",
        "architecture",
        "source",
        "can_remove",
    ] {
        assert!(entry.get(key).is_some(), "missing {key}: {entry}");
    }
    assert_eq!(entry["id"], "alpha");
    assert_eq!(entry["owned_by"], "llamacpp");
    assert_eq!(entry["status"]["value"], "unloaded");
    // The `by_design` divergence on the manifest's `--models-dir` entry:
    // router mode serves models in-process, not as child llama-server
    // processes, so there is no child argv to report and `status.args` is
    // deliberately the empty list where b10621 reports the child's argv.
    assert_eq!(entry["status"]["args"], serde_json::json!([]));
    assert_eq!(entry["source"], "models_dir");
    assert_eq!(entry["can_remove"], false);
    assert_eq!(
        entry["architecture"]["input_modalities"],
        serde_json::json!(["text"])
    );
}

#[tokio::test]
async fn v1_models_serves_the_same_router_inventory() {
    let root = temp_models_dir("v1-inventory");
    add_fake_model(&root, "alpha");
    let app = router_app_with(root, ServerConfig::default(), true);
    let (status, body) = send(app, Method::GET, "/v1/models", "", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"][0]["id"], "alpha");
    assert!(body["data"][0].get("status").is_some());
}

#[tokio::test]
async fn reload_rescans_the_directory() {
    let root = temp_models_dir("reload");
    add_fake_model(&root, "alpha");
    let app = router_app_with(root.clone(), ServerConfig::default(), true);
    let (_, before) = send(app.clone(), Method::GET, "/models", "", None).await;
    assert_eq!(before["data"].as_array().unwrap().len(), 1);
    add_fake_model(&root, "beta");
    let (_, after) = send(app, Method::GET, "/models?reload=1", "", None).await;
    assert_eq!(after["data"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn load_and_unload_refusals_match_b10621() {
    let root = temp_models_dir("load-unload");
    add_fake_model(&root, "alpha");
    let app = router_app_with(root, ServerConfig::default(), true);

    // Unknown model on load is upstream's one 404.
    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/models/load",
        r#"{"model":"ghost"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["message"], "model is not found");

    // Unload of a model that is not running is a 400.
    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/models/unload",
        r#"{"model":"alpha"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "model is not running");

    // Unknown model on unload is a 400 too.
    let (status, body) = send(
        app,
        Method::POST,
        "/models/unload",
        r#"{"model":"ghost"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "model is not found");
}

#[tokio::test]
async fn add_and_delete_answer_their_refusal_surface() {
    let root = temp_models_dir("add-delete");
    add_fake_model(&root, "alpha");
    let app = router_app_with(root, ServerConfig::default(), true);

    let (status, body) = send(app.clone(), Method::POST, "/models", r#"{}"#, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "model must be a non-empty string");

    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/models",
        r#"{"model":"alpha"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "model 'alpha' already exists");

    // Without a configured cache, a download request is a server error.
    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/models",
        r#"{"model":"owner/new-model"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("requires a model cache"),
        "{body}"
    );

    let (status, body) = send(app.clone(), Method::DELETE, "/models", "", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "model must be a non-empty string");

    let (status, body) = send(app.clone(), Method::DELETE, "/models?model=ghost", "", None).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body["error"]["message"], "model name=ghost is not found");

    let (status, body) = send(app, Method::DELETE, "/models?model=alpha", "", None).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        body["error"]["message"],
        "model name=alpha is not removable (not from cache)"
    );
}

#[tokio::test]
async fn dispatch_refusals_match_the_proxy_contract() {
    let root = temp_models_dir("dispatch");
    add_fake_model(&root, "alpha");
    let app = router_app_with(root, ServerConfig::default(), false);

    // POST without a model field.
    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/v1/chat/completions",
        r#"{"messages":[]}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body["error"]["message"],
        "model name is missing from the request"
    );

    // Unknown model.
    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/v1/chat/completions",
        r#"{"model":"ghost"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "model 'ghost' not found");

    // Known model, not loaded, autoload off (server-wide default).
    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/v1/chat/completions",
        r#"{"model":"alpha"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "model is not loaded");

    // The same rule on a GET proxy path with ?model=.
    let (status, body) = send(app, Method::GET, "/slots?model=alpha", "", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "model is not loaded");
}

#[tokio::test]
async fn dispatch_get_patch_model_free_requests_require_a_query_model() {
    let root = temp_models_dir("dispatch-model-free");
    add_fake_model(&root, "alpha");
    let app = router_app_with(root, ServerConfig::default(), false);

    for (method, path, payload) in [
        (Method::GET, "/slots", ""),
        (Method::PATCH, "/v1/settings", r#"{"temperature":0.25}"#),
    ] {
        let (status, body) = send(app.clone(), method.clone(), path, payload, None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{method} {path}: {body}");
        assert_eq!(
            body["error"]["message"], "model name is missing from the request",
            "{method} {path}"
        );
    }
}

#[tokio::test]
async fn dispatch_get_patch_api_key_guards_query_selected_models() {
    let root = temp_models_dir("dispatch-auth");
    add_fake_model(&root, "alpha");
    let config = ServerConfig {
        api_keys: crate::server::resolve_api_keys(&[ROUTER_KEY.to_string()], &[]).expect("keys"),
        ..Default::default()
    };
    let app = router_app_with(root, config, false);

    for (method, path, payload) in [
        (Method::GET, "/slots?model=alpha", ""),
        (
            Method::PATCH,
            "/v1/settings?model=alpha",
            r#"{"temperature":0.25}"#,
        ),
    ] {
        let (status, _) = send(app.clone(), method.clone(), path, payload, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{method} {path}");

        let (status, body) =
            send(app.clone(), method.clone(), path, payload, Some(ROUTER_KEY)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{method} {path}: {body}");
        assert_eq!(
            body["error"]["message"], "model is not loaded",
            "{method} {path}"
        );
    }
}

#[tokio::test]
async fn dispatch_get_patch_patch_forwards_the_body_and_uses_the_query_model() {
    let query = parse_query(Some("model=alpha"));
    let payload = br#"{"model":"body-model","temperature":0.25}"#;
    let body = buffer_dispatch_body(&Method::PATCH, Body::from(payload.as_slice()))
        .await
        .expect("PATCH body must fit within the dispatch cap");

    assert_eq!(body.as_ref(), payload);
    assert_eq!(
        dispatch_model_name(&Method::PATCH, &query, &body),
        "alpha",
        "PATCH model selection must come from the query, not the JSON body"
    );

    let get_body = buffer_dispatch_body(&Method::GET, Body::from("not forwarded"))
        .await
        .expect("GET has no buffered dispatch body");
    assert!(get_body.is_empty());
    assert_eq!(
        dispatch_model_name(&Method::GET, &query, &get_body),
        "alpha"
    );
}

#[tokio::test]
async fn the_router_props_block_matches_b10621s_shape() {
    let root = temp_models_dir("props");
    let app = router_app_with(root, ServerConfig::default(), true);
    let (status, body) = send(app, Method::GET, "/props", "", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["role"], "router");
    assert_eq!(body["max_instances"], 4);
    assert_eq!(body["models_autoload"], true);
    assert_eq!(body["default_generation_settings"]["n_ctx"], 0);
    assert!(body.get("build_info").is_some());
}

#[tokio::test]
async fn health_is_public_and_management_routes_are_keyed() {
    let root = temp_models_dir("auth");
    add_fake_model(&root, "alpha");
    let config = ServerConfig {
        api_keys: crate::server::resolve_api_keys(&[ROUTER_KEY.to_string()], &[]).expect("keys"),
        ..Default::default()
    };
    let app = router_app_with(root, config, true);

    let (status, body) = send(app.clone(), Method::GET, "/health", "", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");

    for (method, path, payload) in [
        (Method::GET, "/models", ""),
        (Method::GET, "/v1/models", ""),
        (Method::GET, "/props", ""),
        (Method::GET, "/models/sse", ""),
        (Method::POST, "/models/load", r#"{"model":"alpha"}"#),
        (Method::POST, "/models/unload", r#"{"model":"alpha"}"#),
        (Method::POST, "/models", r#"{"model":"x"}"#),
        (Method::DELETE, "/models?model=alpha", ""),
        (Method::POST, "/v1/chat/completions", r#"{"model":"alpha"}"#),
    ] {
        let status = status_only(app.clone(), method.clone(), path, payload, None).await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "{method} {path} must require a key"
        );
        let status =
            status_only(app.clone(), method.clone(), path, payload, Some(ROUTER_KEY)).await;
        assert_ne!(
            status,
            StatusCode::UNAUTHORIZED,
            "{method} {path} must accept the configured key"
        );
    }
}

#[tokio::test]
async fn base_router_does_not_mount_webui_adapters() {
    let root = temp_models_dir("ui-off");
    add_fake_model(&root, "alpha");
    let app = router_app_with(root, ServerConfig::default(), true);
    let (status, body) = send(app, Method::GET, "/ui-api/v1/operations", "", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body["error"]["message"],
        "model name is missing from the request"
    );
}

#[tokio::test]
async fn authenticated_webui_catalog_lists_reads_and_refreshes_router_pool() {
    let root = temp_models_dir("ui-catalog");
    add_catalog_model(&root, "alpha", "qwen3");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let app = create_router_app_with_authenticated_ui(state);

    let status = status_only(app.clone(), Method::GET, "/ui-api/v1/catalog", "", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let (status, body) = send(
        app.clone(),
        Method::GET,
        "/ui-api/v1/catalog?limit=1&q=alpha",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["schema_version"], "webui.ui-api.v1");
    assert_eq!(body["items"].as_array().unwrap().len(), 1);
    assert_eq!(body["items"][0]["identity"]["inference_id"], "alpha");
    assert_eq!(body["items"][0]["metadata"]["model_type"], "qwen3");
    assert_eq!(body["items"][0]["removal"]["eligible"], false);

    let id = body["items"][0]["identity"]["id"].as_str().unwrap();
    let (status, one) = send(
        app.clone(),
        Method::GET,
        &format!("/ui-api/v1/catalog/{id}"),
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(one["identity"]["id"], id);

    let (status, invalid) = send(
        app.clone(),
        Method::GET,
        "/ui-api/v1/catalog?lifecycle=started",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(invalid["error"]["field_errors"][0]["field"], "lifecycle");

    let (status, accepted) = send(
        app,
        Method::POST,
        "/ui-api/v1/catalog/refresh",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(accepted["state"], "queued");
    assert!(
        accepted["operation_id"]
            .as_str()
            .unwrap()
            .starts_with("op_catalog_refresh_")
    );
}

#[tokio::test]
async fn explicit_webui_router_requires_configured_api_key() {
    let root = temp_models_dir("ui-auth-required");
    add_fake_model(&root, "alpha");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        ServerConfig::default(),
        true,
    );
    let app = create_router_app_with_authenticated_ui(state);
    let status = status_only(app, Method::GET, "/ui-api/v1/operations", "", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

/// A request cannot smuggle a filesystem path through the model field: names
/// resolve against the discovered registry only.
#[tokio::test]
async fn model_names_cannot_be_paths() {
    let root = temp_models_dir("traversal");
    add_fake_model(&root, "alpha");
    let app = router_app_with(root, ServerConfig::default(), true);
    for name in ["../alpha", "/etc/passwd", "alpha/../alpha"] {
        let (status, body) = send(
            app.clone(),
            Method::POST,
            "/v1/chat/completions",
            &format!(r#"{{"model":"{name}"}}"#),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{name}");
        assert_eq!(
            body["error"]["message"],
            format!("model '{name}' not found"),
            "{name}"
        );
    }
}

// ── Cache-backed routes (#1438: POST /models, DELETE /models, SSE) ──────────

#[tokio::test]
async fn post_models_downloads_into_the_cache_and_lists_it_removable() {
    let root = temp_models_dir("dl-route");
    let cache_root = temp_models_dir("dl-route-cache");
    let (app, state) = router_app_with_cache(root, cache_root.clone(), ServerConfig::default());
    let mut events = state.pool.subscribe();

    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/models",
        r#"{"model":"mlx-community/fresh"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["success"], true);

    // Wait for the background download to finish (the fake is instant, but
    // it still crosses the spawned task).
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let event = events.recv().await.expect("events open");
            if event["event"] == "download_finished" {
                assert_eq!(event["model"], "mlx-community/fresh");
                break;
            }
        }
    })
    .await
    .expect("download_finished");

    assert!(cache_root.join("mlx-community/fresh/config.json").is_file());
    let (status, body) = send(app, Method::GET, "/models", "", None).await;
    assert_eq!(status, StatusCode::OK);
    let entry = body["data"]
        .as_array()
        .expect("data")
        .iter()
        .find(|m| m["id"] == "mlx-community/fresh")
        .cloned()
        .expect("downloaded model listed");
    assert_eq!(entry["source"], "cache");
    assert_eq!(entry["can_remove"], true);
}

#[tokio::test]
async fn delete_models_removes_a_cache_entry_from_disk() {
    let root = temp_models_dir("rm-route");
    let cache_root = temp_models_dir("rm-route-cache");
    add_fake_model(&cache_root.join("mlx-community"), "doomed");
    let (app, state) = router_app_with_cache(root, cache_root.clone(), ServerConfig::default());
    let mut events = state.pool.subscribe();

    let (status, body) = send(
        app.clone(),
        Method::DELETE,
        "/models?model=mlx-community%2Fdoomed",
        "",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["success"], true);
    assert!(!cache_root.join("mlx-community/doomed").exists());

    let mut saw_remove = false;
    while let Ok(event) = events.try_recv() {
        if event["event"] == "model_remove" && event["model"] == "mlx-community/doomed" {
            saw_remove = true;
        }
    }
    assert!(saw_remove, "model_remove reached the SSE stream");

    // Removing it again is upstream's not-found 500.
    let (status, body) = send(
        app,
        Method::DELETE,
        "/models?model=mlx-community%2Fdoomed",
        "",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        body["error"]["message"],
        "model name=mlx-community/doomed is not found"
    );
}

/// Pins the by_design policy divergence on `POST /models`: a bare,
/// owner-less name expands to mlxcel's default organization (the same
/// expansion `-m <name>` and `mlxcel download <name>` apply), so the cache
/// entry lists under the expanded repo id where b10621 would list the
/// verbatim name.
#[tokio::test]
async fn post_models_expands_bare_names_to_the_default_org() {
    let root = temp_models_dir("dl-bare");
    let cache_root = temp_models_dir("dl-bare-cache");
    let (app, state) = router_app_with_cache(root, cache_root, ServerConfig::default());
    let mut events = state.pool.subscribe();

    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/models",
        r#"{"model":"Bare-Name-4bit"}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let event = events.recv().await.expect("events open");
            if event["event"] == "download_finished" {
                assert_eq!(event["model"], "mlx-community/Bare-Name-4bit");
                break;
            }
        }
    })
    .await
    .expect("download_finished");
    assert!(state.pool.get("mlx-community/Bare-Name-4bit").is_some());
}

/// Dispatch accepts preset aliases end-to-end: the alias resolves to the
/// entry and the load path addresses the entry by its real name (a broken
/// checkpoint therefore answers a load failure, never "not found").
#[tokio::test]
async fn dispatch_reaches_the_load_path_through_an_alias() {
    let checkpoint_root = temp_models_dir("alias-dispatch");
    add_fake_model(&checkpoint_root, "ckpt");
    let ini = format!(
        "[aliased-model]\nmodel = {}\nalias = nickname\n",
        checkpoint_root.join("ckpt").display()
    );
    let presets = crate::server::router_presets::parse_preset_text(&ini).expect("parse");
    let state = router_state_from(
        RouterSources {
            models_dir: None,
            cache: None,
            presets,
        },
        ServerConfig::default(),
        true,
    );
    let app = create_router_app(state);
    let (status, body) = send(
        app,
        Method::POST,
        "/v1/chat/completions",
        r#"{"model":"nickname","messages":[]}"#,
        None,
    )
    .await;
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "alias must resolve: {body}"
    );
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
}

async fn first_sse_chunk(app: Router, last_event_id: Option<&str>) -> String {
    let mut builder = Request::builder()
        .method(Method::GET)
        .uri("/ui-api/v1/events")
        .header(header::ACCEPT, "text/event-stream")
        .header(header::AUTHORIZATION, format!("Bearer {ROUTER_KEY}"));
    if let Some(event_id) = last_event_id {
        builder = builder.header("Last-Event-ID", event_id);
    }
    let response = app
        .oneshot(builder.body(Body::empty()).expect("request"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("sse chunk timeout")
        .expect("sse chunk")
        .expect("sse body ok");
    String::from_utf8(chunk.to_vec()).expect("utf8 sse")
}

#[tokio::test]
async fn ui_operations_routes_list_get_and_report_cancel_unsupported() {
    let root = temp_models_dir("ui-ops");
    add_fake_model(&root, "alpha");
    add_fake_model(&root, "beta");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let entry = state.pool.get("alpha").expect("entry");
    let eviction_entry = state.pool.get("beta").expect("eviction entry");
    // This fixture seeds the coordinator, not a real inference worker. Real
    // worker ownership and exit are covered by the separate lifecycle gate.
    let loaded_lifecycle = crate::server::router_lifecycle::ModelLifecycle::new(
        crate::server::router_lifecycle::DownloadState::Complete,
    );
    loaded_lifecycle.mark_loading();
    loaded_lifecycle.mark_ready();
    let accepted = state
        .pool
        .lifecycle_coordinator()
        .begin_operation(
            crate::server::router_lifecycle::OperationKind::ModelLoad,
            crate::server::router_lifecycle::OperationTarget::Model {
                model_id: entry.ui_model_id.clone(),
                requested_revision: Some(entry.lifecycle_revision()),
                eviction_target_id: Some(eviction_entry.ui_model_id.clone()),
            },
            Some("route-ops-0001"),
            "route:ops:1".to_string(),
        )
        .expect("operation");
    state.pool.lifecycle_coordinator().update_operation(
        &accepted.operation_id,
        crate::server::router_lifecycle::OperationState::Succeeded,
        Some(
            crate::server::router_lifecycle::OperationResult::ModelLoad {
                model_id: entry.ui_model_id.clone(),
                revision: entry.lifecycle_revision(),
                lifecycle: loaded_lifecycle.snapshot(),
                eviction: Some(crate::server::router_lifecycle::ModelEvictionReport {
                    requested_target_id: Some(eviction_entry.ui_model_id.clone()),
                    displaced_model_id: Some(eviction_entry.ui_model_id.clone()),
                    outcome: crate::server::router_lifecycle::ModelEvictionOutcome::Displaced,
                    rollbackable: false,
                }),
            },
        ),
        None,
    );
    let app = create_router_app_with_authenticated_ui(state);

    let (status, body) = send(
        app.clone(),
        Method::GET,
        "/ui-api/v1/operations?kind=model_load&limit=10",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    contract::assert_operation_list(&body, &entry.ui_model_id, &eviction_entry.ui_model_id);
    assert_eq!(body["items"][0]["operation_id"], accepted.operation_id);

    let (status, body) = send(
        app.clone(),
        Method::GET,
        &format!("/ui-api/v1/operations/{}", accepted.operation_id),
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["operation_id"], accepted.operation_id);
    contract::assert_operation(&body, &entry.ui_model_id, &eviction_entry.ui_model_id);

    let (status, body) = send(
        app,
        Method::POST,
        &format!("/ui-api/v1/operations/{}/cancel", accepted.operation_id),
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "unsupported");
    assert_eq!(body["error"]["operation_id"], accepted.operation_id);
}

#[tokio::test]
async fn ui_model_action_route_validates_profile_fields_and_idempotency() {
    let root = temp_models_dir("ui-action-profile");
    add_fake_model(&root, "alpha");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let entry = state.pool.get("alpha").expect("entry");
    let app = create_router_app_with_authenticated_ui(state);
    let body = serde_json::json!({
        "model_id": entry.ui_model_id,
        "action": "load",
        "expected_revision": entry.lifecycle_revision(),
        "idempotency_key": "profile-load-0001",
        "load_profile": {
            "ctx_size": 8192,
            "n_parallel": 4,
            "kv_cache_mode": "fp16+turbo4"
        },
        "eviction_target_id": entry.ui_model_id
    });
    let (status, response) = send(
        app,
        Method::POST,
        "/ui-api/v1/model-actions",
        &body.to_string(),
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{response}");
    assert_eq!(
        response["error"]["field_errors"][0]["field"],
        "load_profile"
    );
}

#[tokio::test]
async fn ui_model_action_route_rejects_contract_invalid_fields() {
    let root = temp_models_dir("ui-action-contract-invalid");
    add_fake_model(&root, "alpha");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let entry = state.pool.get("alpha").expect("entry");
    let model_id = entry.ui_model_id.clone();
    let revision = entry.lifecycle_revision();
    let app = create_router_app_with_authenticated_ui(state);

    let cases = vec![
        (
            serde_json::json!({
                "model_id": "alpha",
                "action": "load",
                "expected_revision": revision,
                "idempotency_key": "valid-key-0001"
            }),
            "model_id",
        ),
        (
            serde_json::json!({
                "model_id": model_id.clone(),
                "action": "load",
                "expected_revision": 0,
                "idempotency_key": "valid-key-0002"
            }),
            "expected_revision",
        ),
        (
            serde_json::json!({
                "model_id": model_id.clone(),
                "action": "load",
                "expected_revision": revision,
                "idempotency_key": "short"
            }),
            "idempotency_key",
        ),
        (
            serde_json::json!({
                "model_id": model_id.clone(),
                "action": "load",
                "expected_revision": revision,
                "idempotency_key": "bad/slash-key"
            }),
            "idempotency_key",
        ),
        (
            serde_json::json!({
                "model_id": model_id.clone(),
                "action": "load",
                "expected_revision": revision,
                "idempotency_key": "a".repeat(129)
            }),
            "idempotency_key",
        ),
        (
            serde_json::json!({
                "model_id": model_id.clone(),
                "action": "load",
                "expected_revision": revision,
                "idempotency_key": "valid-key-0003",
                "eviction_target_id": "mdl_short"
            }),
            "eviction_target_id",
        ),
        (
            serde_json::json!({
                "model_id": model_id.clone(),
                "action": "load",
                "expected_revision": revision,
                "idempotency_key": "valid-key-0004",
                "load_profile": { "ctx_size": 0 }
            }),
            "load_profile.ctx_size",
        ),
        (
            serde_json::json!({
                "model_id": model_id.clone(),
                "action": "load",
                "expected_revision": revision,
                "idempotency_key": "valid-key-0005",
                "load_profile": { "n_parallel": 33 }
            }),
            "load_profile.n_parallel",
        ),
        (
            serde_json::json!({
                "model_id": model_id.clone(),
                "action": "load",
                "expected_revision": revision,
                "idempotency_key": "valid-key-0006",
                "load_profile": { "kv_cache_mode": "q8_0" }
            }),
            "load_profile.kv_cache_mode",
        ),
    ];

    for (body, field) in cases {
        let (status, response) = send(
            app.clone(),
            Method::POST,
            "/ui-api/v1/model-actions",
            &body.to_string(),
            Some(ROUTER_KEY),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{field}: {response}");
        assert_eq!(response["error"]["code"], "invalid_request");
        assert_eq!(response["error"]["field_errors"][0]["field"], field);
    }
}

#[tokio::test]
async fn ui_operations_routes_reject_invalid_filters_and_ids() {
    let root = temp_models_dir("ui-ops-contract-invalid");
    add_fake_model(&root, "alpha");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let app = create_router_app_with_authenticated_ui(state);

    let list_cases = vec![
        ("/ui-api/v1/operations?state=unknown", "state"),
        ("/ui-api/v1/operations?kind=unknown", "kind"),
        ("/ui-api/v1/operations?cursor=not-from-server", "cursor"),
        ("/ui-api/v1/operations?cursor=ops_bad", "cursor"),
        ("/ui-api/v1/operations?target=bad/path", "target"),
    ];
    for (uri, field) in list_cases {
        let (status, response) = send(app.clone(), Method::GET, uri, "", Some(ROUTER_KEY)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {response}");
        assert_eq!(response["error"]["field_errors"][0]["field"], field);
    }

    let (status, response) = send(
        app.clone(),
        Method::GET,
        "/ui-api/v1/operations/bad%20id",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["error"]["field_errors"][0]["field"], "id");

    let (status, response) = send(
        app,
        Method::POST,
        "/ui-api/v1/operations/bad%20id/cancel",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["error"]["field_errors"][0]["field"], "id");
}

#[tokio::test]
async fn ui_events_emit_snapshot_and_gap_reset_with_sse_ids() {
    let root = temp_models_dir("ui-events");
    add_fake_model(&root, "alpha");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let server_instance = state
        .pool
        .lifecycle_coordinator()
        .server_instance_id()
        .to_string();
    let app = create_router_app_with_authenticated_ui(state);

    let snapshot = first_sse_chunk(app.clone(), None).await;
    assert!(snapshot.contains("event: snapshot"), "{snapshot}");
    assert!(snapshot.contains("\"type\":\"snapshot\""), "{snapshot}");
    assert!(snapshot.contains("id: evt_"), "{snapshot}");

    let missing_same_instance = format!("evt_{server_instance}_99999999");
    let gap = first_sse_chunk(app, Some(&missing_same_instance)).await;
    assert!(gap.contains("event: gap"), "{gap}");
    assert!(gap.contains("\"reason\":\"gap\""), "{gap}");
    contract::assert_gap_event(&gap, &server_instance);
}

#[tokio::test]
async fn catalog_refresh_singleflights_and_reports_same_size_changes() {
    use std::sync::{Mutex, mpsc};

    let root = temp_models_dir("ui-refresh-singleflight");
    add_catalog_model(&root, "alpha", "qwen3");
    let state = router_state_from(
        RouterSources {
            models_dir: Some(root.clone()),
            cache: None,
            presets: Default::default(),
        },
        keyed_config(),
        true,
    );
    let app = create_router_app_with_authenticated_ui(state.clone());

    std::fs::remove_dir_all(root.join("alpha")).expect("remove alpha");
    add_catalog_model(&root, "beta", "qwen3");

    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    state.pool.set_rescan_after_snapshot_hook(Some(Arc::new({
        let release_rx = release_rx.clone();
        move || {
            let _ = started_tx.send(());
            let _ = release_rx.lock().expect("release lock").recv();
        }
    })));

    let (status, first) = send(
        app.clone(),
        Method::POST,
        "/ui-api/v1/catalog/refresh",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let op = first["operation_id"].as_str().unwrap().to_string();
    tokio::task::spawn_blocking(move || started_rx.recv())
        .await
        .expect("started join")
        .expect("refresh started");

    let (status, replay) = send(
        app.clone(),
        Method::POST,
        "/ui-api/v1/catalog/refresh",
        "",
        Some(ROUTER_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(replay["operation_id"], op);
    assert_eq!(replay["idempotent_replay"], true);

    release_tx.send(()).expect("release refresh");
    state.pool.set_rescan_after_snapshot_hook(None);

    let mut terminal = None;
    for _ in 0..50 {
        let (status, body) = send(
            app.clone(),
            Method::GET,
            &format!("/ui-api/v1/operations/{op}"),
            "",
            Some(ROUTER_KEY),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        if body["state"] == "succeeded" {
            terminal = Some(body);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let body = terminal.expect("refresh terminal");
    assert_eq!(body["result"]["scanned_entries"], 1);
    assert_eq!(body["result"]["changed_entries"], 2);
}
