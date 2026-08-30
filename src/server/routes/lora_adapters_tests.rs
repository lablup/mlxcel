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

//! Route tests for the b10621 `/lora-adapters` surface (issue #1439).

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

use crate::lora::LoraAdapterSpec;
use crate::server::config::ServerConfig;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn app_with(adapters: Vec<LoraAdapterSpec>) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig {
            lora_adapters: adapters,
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("lora-test-model"),
        batch_metrics,
    );
    create_app(state)
}

fn two_adapters() -> Vec<LoraAdapterSpec> {
    vec![
        LoraAdapterSpec {
            path: PathBuf::from("/adapters/alpha"),
            scale: 1.0,
            apply: true,
        },
        LoraAdapterSpec {
            path: PathBuf::from("/adapters/beta"),
            scale: 0.5,
            apply: true,
        },
    ]
}

async fn send(app: Router, method: Method, body: &str) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(method)
        .uri("/lora-adapters")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

#[tokio::test]
async fn inventory_carries_the_b10621_entry_shape() {
    let (status, body) = send(app_with(two_adapters()), Method::GET, "").await;
    assert_eq!(status, StatusCode::OK);
    let entries = body.as_array().expect("array");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["id"], 0);
    assert_eq!(entries[0]["path"], "/adapters/alpha");
    assert_eq!(entries[0]["scale"], 1.0);
    assert_eq!(entries[0]["task_name"], "");
    assert_eq!(entries[0]["prompt_prefix"], "");
    assert_eq!(entries[1]["id"], 1);
    assert_eq!(entries[1]["scale"], 0.5);
}

#[tokio::test]
async fn an_unapplied_adapter_reports_scale_zero() {
    let adapters = vec![LoraAdapterSpec {
        path: PathBuf::from("/adapters/alpha"),
        scale: 1.0,
        apply: false,
    }];
    let (_, body) = send(app_with(adapters), Method::GET, "").await;
    assert_eq!(body[0]["scale"], 0.0);
}

#[tokio::test]
async fn a_server_without_adapters_reports_an_empty_list() {
    let (status, body) = send(app_with(Vec::new()), Method::GET, "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, serde_json::json!([]));
}

#[tokio::test]
async fn post_with_the_current_configuration_is_acknowledged() {
    let app = app_with(two_adapters());
    let (status, body) = send(
        app,
        Method::POST,
        r#"[{"id":0,"scale":1.0},{"id":1,"scale":0.5}]"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, serde_json::json!({ "success": true }));
}

#[tokio::test]
async fn post_changing_a_scale_is_refused_with_the_migration_diagnostic() {
    let app = app_with(two_adapters());
    // Dropping beta (unlisted -> 0.0, upstream's rule) is a change.
    let (status, body) = send(app, Method::POST, r#"[{"id":0,"scale":1.0}]"#).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("--lora-scaled"), "{message}");
}

#[tokio::test]
async fn post_non_array_body_uses_upstreams_wording() {
    let (status, body) = send(app_with(two_adapters()), Method::POST, r#"{"id":0}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "Request body must be an array");
}

#[tokio::test]
async fn unknown_ids_are_ignored_like_upstreams_construct_lora_list() {
    // [{id: 99}] leaves no adapter listed, so every adapter drops to 0.0,
    // which differs from the fused configuration and is refused.
    let (status, _) = send(
        app_with(two_adapters()),
        Method::POST,
        r#"[{"id":99,"scale":1}]"#,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);

    // On a server with no adapters, any id list resolves to the empty
    // configuration, which is what is in force: acknowledged.
    let (status, body) = send(
        app_with(Vec::new()),
        Method::POST,
        r#"[{"id":99,"scale":1}]"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

// ── Runtime (unfused) path (#1439) ──────────────────────────────────────────

fn app_with_runtime(scales: &[f32]) -> (Router, Arc<crate::lora::RuntimeLoraSet>) {
    let set = Arc::new(crate::lora::RuntimeLoraSet::stub(scales));
    let adapters = set.adapters.iter().map(|a| a.spec.clone()).collect();
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig {
            lora_adapters: adapters,
            lora_runtime: Some(set.clone()),
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("lora-test-model"),
        batch_metrics,
    );
    (create_app(state), set)
}

/// The runtime path accepts a scale change: the resolved scales become the
/// server default, the handles the layers read follow at the next batch
/// application, and GET reports the new configuration (b10621's
/// `SERVER_TASK_TYPE_SET_LORA` semantics).
#[tokio::test]
async fn post_applies_a_scale_change_on_the_runtime_path() {
    let (app, set) = app_with_runtime(&[1.0, 0.5]);

    let (status, body) = send(app.clone(), Method::POST, r#"[{"id":0,"scale":0.25}]"#).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["success"], true);
    // Upstream's construct_lora_list: listed ids set their scale, unlisted
    // drop to 0.0.
    assert_eq!(set.server_scales(), vec![0.25, 0.0]);

    let (status, body) = send(app, Method::GET, "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body[0]["scale"], 0.25);
    assert_eq!(body[1]["scale"], 0.0);
}

/// `--lora-init-without-apply` end state: adapters at 0.0 can be activated
/// later through POST, the flag's entire purpose upstream.
#[tokio::test]
async fn init_without_apply_adapters_activate_through_post() {
    let (app, set) = app_with_runtime(&[0.0]);
    let (status, body) = send(app.clone(), Method::POST, r#"[{"id":0,"scale":1.0}]"#).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(set.server_scales(), vec![1.0]);
    let (_, body) = send(app, Method::GET, "").await;
    assert_eq!(body[0]["scale"], 1.0);
}

/// A non-array body keeps upstream's exact 400 on the runtime path too.
#[tokio::test]
async fn runtime_path_still_refuses_a_non_array_body() {
    let (app, set) = app_with_runtime(&[1.0]);
    let (status, body) = send(app, Method::POST, r#"{"id":0}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "Request body must be an array");
    assert_eq!(set.server_scales(), vec![1.0], "scales unchanged");
}

/// A POST swap changes the server default without touching the applied
/// handles: those are written per batch from each request's snapshot, so an
/// in-flight generation keeps its configuration.
#[tokio::test]
async fn post_changes_the_default_not_the_live_handles() {
    let (app, set) = app_with_runtime(&[1.0]);
    let (status, _) = send(app, Method::POST, r#"[{"id":0,"scale":0.5}]"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(set.server_scales(), vec![0.5]);
    assert_eq!(
        set.adapters[0].handle.get(),
        1.0,
        "handles change only when the worker applies a batch snapshot"
    );
    set.apply_scales(&set.server_scales());
    assert_eq!(set.adapters[0].handle.get(), 0.5);
}
