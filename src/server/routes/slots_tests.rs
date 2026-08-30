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

//! Route-level tests for the b10621 slots surface (issue #1440): the
//! `GET /slots` response shape and gating diagnostic, `fail_on_no_slot`,
//! and the `POST /slots/:id_slot` save/restore/erase actions with their
//! confinement and identity checks.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn state_with(config: ServerConfig) -> AppState {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("slots-test-model"),
        batch_metrics,
    )
}

async fn send(
    app: Router,
    method: Method,
    uri: &str,
    body: &str,
) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, json)
}

fn temp_save_root(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel-slots-route-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create save root");
    dir
}

#[tokio::test]
async fn get_slots_reports_the_b10621_slot_shape() {
    let state = state_with(ServerConfig {
        n_parallel: 2,
        context_size: 4096,
        ..Default::default()
    });
    let (status, body) = send(create_app(state), Method::GET, "/slots", "").await;
    assert_eq!(status, StatusCode::OK);
    let slots = body.as_array().expect("array of slots");
    assert_eq!(slots.len(), 2);
    for (i, slot) in slots.iter().enumerate() {
        assert_eq!(slot["id"], i);
        assert_eq!(slot["n_ctx"], 4096);
        assert_eq!(slot["is_processing"], false);
        assert_eq!(slot["speculative"], false);
    }
}

#[tokio::test]
async fn disabled_slots_endpoint_answers_the_b10621_diagnostic_not_404() {
    let state = state_with(ServerConfig {
        enable_slots_endpoint: false,
        ..Default::default()
    });
    let (status, body) = send(create_app(state), Method::GET, "/slots", "").await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["error"]["code"], 501);
    assert_eq!(body["error"]["type"], "not_supported_error");
    assert_eq!(
        body["error"]["message"],
        "This server does not support slots endpoint. Start it with `--slots`"
    );
}

#[tokio::test]
async fn fail_on_no_slot_distinguishes_saturation_from_idle() {
    let state = state_with(ServerConfig {
        n_parallel: 1,
        ..Default::default()
    });
    let app = create_app(state.clone());

    // Idle: the switch changes nothing.
    let (status, _) = send(app.clone(), Method::GET, "/slots?fail_on_no_slot=1", "").await;
    assert_eq!(status, StatusCode::OK);

    // Saturated: b10621's 503 "no slot available" envelope, while the plain
    // query still answers 200 with the busy slot visible.
    let handle = state.slots.begin("busy", serde_json::json!({}), None);
    // Since #1440 a handle binds on its first progress signal rather than at
    // route entry, so saturation means "every slot is serving a request",
    // not "every slot has been claimed by a request that may still be queued".
    handle.on_prefill(4, 0);
    let (status, body) = send(app.clone(), Method::GET, "/slots?fail_on_no_slot=1", "").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], 503);
    assert_eq!(body["error"]["message"], "no slot available");
    assert_eq!(body["error"]["type"], "unavailable_error");

    let (status, body) = send(app, Method::GET, "/slots", "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body[0]["is_processing"], true);
    drop(handle);
}

#[tokio::test]
async fn slot_actions_without_save_path_answer_the_b10621_diagnostic() {
    let state = state_with(ServerConfig::default());
    let (status, body) = send(
        create_app(state),
        Method::POST,
        "/slots/0?action=save",
        r#"{"filename":"x.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(
        body["error"]["message"],
        "This server does not support slots action. Start it with `--slot-save-path`"
    );
}

#[tokio::test]
async fn save_restore_erase_roundtrip_through_the_route() {
    let root = temp_save_root("roundtrip");
    let state = state_with(ServerConfig {
        n_parallel: 2,
        slot_save_path: Some(root.clone()),
        ..Default::default()
    });
    let app = create_app(state.clone());

    // Give slot 0 a finished task so it has cache state to save.
    {
        let handle = state
            .slots
            .begin("hello world", serde_json::json!({}), Some(4));
        handle.finish(2, 0, 1, "out");
    }

    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/slots/0?action=save",
        r#"{"filename":"state.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "save failed: {body}");
    assert_eq!(body["id_slot"], 0);
    assert_eq!(body["filename"], "state.bin");
    assert!(body["n_written"].as_u64().expect("n_written") > 0);
    assert!(body["timings"]["save_ms"].is_number());
    assert!(root.join("state.bin").is_file(), "save file must exist");

    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/slots/1?action=restore",
        r#"{"filename":"state.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "restore failed: {body}");
    assert_eq!(body["id_slot"], 1);
    assert!(body["n_restored"].is_number());
    assert!(body["n_read"].as_u64().expect("n_read") > 0);
    assert!(body["timings"]["restore_ms"].is_number());

    let (status, body) = send(app, Method::POST, "/slots/1?action=erase", "").await;
    assert_eq!(status, StatusCode::OK, "erase failed: {body}");
    assert_eq!(body["id_slot"], 1);
    assert!(body["n_erased"].is_number());
}

#[tokio::test]
async fn restore_survives_a_simulated_restart() {
    let root = temp_save_root("restart");
    let make = || {
        state_with(ServerConfig {
            slot_save_path: Some(root.clone()),
            ..Default::default()
        })
    };

    // First server lifetime: save.
    let (status, _) = send(
        create_app(make()),
        Method::POST,
        "/slots/0?action=save",
        r#"{"filename":"persist.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Second lifetime (fresh state, same root): the file restores.
    let (status, body) = send(
        create_app(make()),
        Method::POST,
        "/slots/0?action=restore",
        r#"{"filename":"persist.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "restore failed: {body}");
}

#[tokio::test]
async fn restore_rejects_a_file_saved_under_a_different_model() {
    let root = temp_save_root("model-mismatch");
    // A file written by "another model" (different identity, same format).
    crate::server::slot_persist::save(&root, "other.bin", "another-model", "fp", &[1, 2])
        .expect("plant file");
    let state = state_with(ServerConfig {
        slot_save_path: Some(root),
        ..Default::default()
    });
    let (status, body) = send(
        create_app(state),
        Method::POST,
        "/slots/0?action=restore",
        r#"{"filename":"other.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = body["error"]["message"].as_str().expect("message");
    assert!(
        message.contains("another-model"),
        "the refusal must name the mismatched model: {message}"
    );
}

#[tokio::test]
async fn invalid_slot_ids_and_actions_answer_b10621s_400s() {
    let root = temp_save_root("invalid");
    let state = state_with(ServerConfig {
        slot_save_path: Some(root),
        n_parallel: 2,
        ..Default::default()
    });
    let app = create_app(state);

    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/slots/abc?action=save",
        r#"{"filename":"x.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "Invalid slot ID");

    // Out of range is the same refusal as non-numeric, b10621's
    // `get_slot_by_id == nullptr` arm.
    let (status, body) = send(
        app.clone(),
        Method::POST,
        "/slots/7?action=save",
        r#"{"filename":"x.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "Invalid slot ID");

    let (status, body) = send(
        app,
        Method::POST,
        "/slots/0?action=defrag",
        r#"{"filename":"x.bin"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["message"], "Invalid action");
}

#[tokio::test]
async fn traversal_filenames_are_refused_with_invalid_filename() {
    let root = temp_save_root("traversal");
    let state = state_with(ServerConfig {
        slot_save_path: Some(root),
        ..Default::default()
    });
    let app = create_app(state);
    for filename in ["../escape.bin", "a/b.bin", "a\\\\b.bin", ".."] {
        let (status, body) = send(
            app.clone(),
            Method::POST,
            "/slots/0?action=save",
            &format!(r#"{{"filename":"{filename}"}}"#),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{filename}");
        assert_eq!(body["error"]["message"], "Invalid filename", "{filename}");
    }
}

/// Reproduces upstream's stoi prefix parsing: `/slots/1abc` addresses slot 1.
#[tokio::test]
async fn stoi_prefix_parsing_matches_b10621() {
    let root = temp_save_root("stoi");
    let state = state_with(ServerConfig {
        slot_save_path: Some(root),
        n_parallel: 2,
        ..Default::default()
    });
    let (status, body) = send(
        create_app(state),
        Method::POST,
        "/slots/1abc?action=erase",
        "",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id_slot"], 1);
}
