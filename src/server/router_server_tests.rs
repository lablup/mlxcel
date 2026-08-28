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

//! Route-level tests for the router-mode surface (issue #1438): the router
//! model inventory, load/unload refusals, add/delete refusals, the dispatch
//! contract (missing / unknown / not-loaded model), the router `/props`
//! block, and authorization on the management routes.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use super::{RouterServerState, create_router_app};
use crate::server::config::ServerConfig;
use crate::server::router_models::RouterPool;

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

fn router_app_with(root: PathBuf, config: ServerConfig, autoload: bool) -> Router {
    let pool = Arc::new(RouterPool::new(root, config.clone(), 4, autoload).expect("pool"));
    create_router_app(RouterServerState {
        pool,
        config: Arc::new(config),
    })
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

    let (status, _) = send(
        app.clone(),
        Method::POST,
        "/models",
        r#"{"model":"owner/new-model"}"#,
        None,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "download flow is deferred"
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
        api_keys: crate::server::resolve_api_keys(&["router-key".to_string()], &[]).expect("keys"),
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
        let status = status_only(
            app.clone(),
            method.clone(),
            path,
            payload,
            Some("router-key"),
        )
        .await;
        assert_ne!(
            status,
            StatusCode::UNAUTHORIZED,
            "{method} {path} must accept the configured key"
        );
    }
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
