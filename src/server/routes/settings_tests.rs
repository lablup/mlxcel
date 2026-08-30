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

//! HTTP contract tests for the opt-in live-settings routes (#1312).

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::server::{
    AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, ServerGenerateOptions, create_app,
};
use crate::tokenizer::MlxcelTokenizer;

const API_KEY: &str = "settings-route-test-key";

fn app_with_recording(
    mut config: ServerConfig,
    api_key: Option<&str>,
) -> (Router, mpsc::Receiver<ServerGenerateOptions>) {
    if let Some(api_key) = api_key {
        config.api_keys = crate::server::resolve_api_keys(&[api_key.to_string()], &[])
            .expect("the route-test key should resolve");
    }
    let (options_tx, options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("settings-route-test-model"),
        batch_metrics,
    );
    (create_app(state), options_rx)
}

fn app_with(config: ServerConfig, api_key: Option<&str>) -> Router {
    app_with_recording(config, api_key).0
}

async fn send_json(
    app: Router,
    method: Method,
    path: &str,
    body: Option<Value>,
    api_key: Option<&str>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(path);
    if body.is_some() {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
    }
    if let Some(api_key) = api_key {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {api_key}"));
    }
    let body = body
        .map(|value| Body::from(value.to_string()))
        .unwrap_or_else(Body::empty);
    let response = app
        .oneshot(builder.body(body).expect("settings request should build"))
        .await
        .expect("settings router should respond");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("settings response body should collect");
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("settings response should be JSON")
    };
    (status, body)
}

fn enabled_config() -> ServerConfig {
    ServerConfig {
        enable_settings_endpoint: true,
        ..ServerConfig::default()
    }
}

fn assert_number_close(value: &Value, expected: f64) {
    let actual = value
        .as_f64()
        .unwrap_or_else(|| panic!("expected a JSON number, got {value}"));
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected {expected}, got {actual}"
    );
}

#[tokio::test]
async fn settings_route_is_absent_when_disabled() {
    let app = app_with(ServerConfig::default(), None);
    for path in ["/v1/settings", "/settings"] {
        let (status, _) = send_json(app.clone(), Method::GET, path, None, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "GET {path}");

        let (status, _) = send_json(app.clone(), Method::PATCH, path, Some(json!({})), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "PATCH {path}");
    }
}

#[tokio::test]
async fn settings_route_aliases_require_api_key_when_configured() {
    let app = app_with(enabled_config(), Some(API_KEY));
    for path in ["/v1/settings", "/settings"] {
        let (status, _) = send_json(app.clone(), Method::GET, path, None, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "GET {path} without key");

        let (status, _) = send_json(app.clone(), Method::PATCH, path, Some(json!({})), None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "PATCH {path} without key");

        let (status, _) = send_json(app.clone(), Method::GET, path, None, Some(API_KEY)).await;
        assert_eq!(status, StatusCode::OK, "GET {path} with key");

        let (status, _) = send_json(
            app.clone(),
            Method::PATCH,
            path,
            Some(json!({})),
            Some(API_KEY),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "PATCH {path} with key");
    }
}

#[tokio::test]
async fn settings_route_get_returns_schema_current_and_fingerprint() {
    let app = app_with(
        ServerConfig {
            enable_settings_endpoint: true,
            default_temperature: 0.42,
            default_seed: Some(11),
            ..ServerConfig::default()
        },
        Some(API_KEY),
    );
    let (status, body) = send_json(app, Method::GET, "/v1/settings", None, Some(API_KEY)).await;
    assert_eq!(status, StatusCode::OK);

    let keys: BTreeSet<_> = body
        .as_object()
        .expect("GET settings should return an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, BTreeSet::from(["current", "fingerprint", "schema"]));

    let schema = body["schema"]
        .as_array()
        .expect("schema should be an array");
    let temperature = schema
        .iter()
        .find(|entry| entry["name"] == "default_temperature")
        .expect("temperature should be classified");
    assert_eq!(temperature["type"], "float");
    assert_eq!(temperature["mutable"], true);
    assert_number_close(&temperature["default"], 0.42);
    assert_number_close(&body["current"]["default_temperature"], 0.42);
    assert_eq!(body["current"]["default_seed"], 11);

    let fingerprint = body["fingerprint"]
        .as_str()
        .expect("fingerprint should be a string");
    assert_eq!(fingerprint.len(), 64);
    assert!(
        fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{fingerprint}"
    );
}

#[tokio::test]
async fn settings_route_patch_partially_applies_and_publishes_to_both_aliases() {
    let app = app_with(
        ServerConfig {
            enable_settings_endpoint: true,
            context_size: 128,
            decode_timeout_seconds: 77,
            default_temperature: 0.8,
            default_top_p: 0.91,
            default_seed: None,
            ..ServerConfig::default()
        },
        Some(API_KEY),
    );

    let (status, before) =
        send_json(app.clone(), Method::GET, "/settings", None, Some(API_KEY)).await;
    assert_eq!(status, StatusCode::OK);
    let before_fingerprint = before["fingerprint"].clone();

    let (status, patched) = send_json(
        app.clone(),
        Method::PATCH,
        "/v1/settings",
        Some(json!({
            "default_seed": 7,
            "default_temperature": 0.25,
            "default_top_p": "wide",
            "mystery_knob": true,
            "reasoning_alias_field": "none",
            "timeout_seconds": 0
        })),
        Some(API_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(patched["applied"]["default_seed"], 7);
    assert_number_close(&patched["applied"]["default_temperature"], 0.25);
    assert_eq!(
        patched["applied"]
            .as_object()
            .expect("applied should be an object")
            .len(),
        2
    );

    let rejected: BTreeSet<_> = patched["rejected"]
        .as_array()
        .expect("rejected should be an array")
        .iter()
        .map(|entry| {
            entry["name"]
                .as_str()
                .expect("each rejection should name its setting")
        })
        .collect();
    assert_eq!(
        rejected,
        BTreeSet::from([
            "default_top_p",
            "mystery_knob",
            "reasoning_alias_field",
            "timeout_seconds"
        ])
    );
    assert_eq!(patched["current"]["default_seed"], 7);
    assert_number_close(&patched["current"]["default_temperature"], 0.25);
    assert_number_close(&patched["current"]["default_top_p"], 0.91);
    assert_eq!(patched["current"]["timeout_seconds"], 77);
    assert_ne!(patched["fingerprint"], before_fingerprint);

    let (status, published) = send_json(app, Method::GET, "/settings", None, Some(API_KEY)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(published["current"], patched["current"]);
    assert_eq!(published["fingerprint"], patched["fingerprint"]);
}

#[tokio::test]
async fn patched_reasoning_budget_is_carried_by_the_next_generation() {
    let (app, options_rx) = app_with_recording(enabled_config(), Some(API_KEY));
    let (status, _) = send_json(
        app.clone(),
        Method::PATCH,
        "/v1/settings",
        Some(json!({"reasoning_budget": 5})),
        Some(API_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, _) = send_json(
        app,
        Method::POST,
        "/v1/completions",
        Some(json!({"model": "route-test-model", "prompt": "hello"})),
        Some(API_KEY),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let options = options_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("generation options");
    assert_eq!(
        options.reasoning_budget,
        crate::server::config::ReasoningBudgetOverride::Explicit(
            crate::server::thinking_budget::ThinkingBudget::from_raw_i32(5)
                .expect("valid live budget")
        )
    );
}
