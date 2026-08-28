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

//! Tests for the Vertex AI (GCP) predict adapter (b10621, #1456).

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use super::*;
use crate::server::model_provider::ModelProvider;
use crate::server::{AppState, ChatTemplateProcessor, ServerConfig, create_app};
use crate::test_support::env_lock::env_lock;
use crate::tokenizer::MlxcelTokenizer;

// -- path_to_gcp_format --

#[test]
fn alias_format_matches_upstream_examples() {
    assert_eq!(
        path_to_gcp_format("/v1/chat/completions"),
        "chatCompletions"
    );
    assert_eq!(path_to_gcp_format("/apply-template"), "applyTemplate");
    assert_eq!(path_to_gcp_format("/v1/embeddings"), "embeddings");
    assert_eq!(path_to_gcp_format("/v1/messages"), "messages");
    assert_eq!(path_to_gcp_format("/tokenize"), "tokenize");
    assert_eq!(path_to_gcp_format("/v1/streams/lookup"), "streamsLookup");
    // Path parameters stop the conversion.
    assert_eq!(path_to_gcp_format("/v1/responses/:id"), "responses");
    assert_eq!(path_to_gcp_format("/v1/responses/:id/cancel"), "responses");
    // The root produces the empty alias, which the table skips.
    assert_eq!(path_to_gcp_format("/"), "");
    // Underscore boundaries capitalise too.
    assert_eq!(
        path_to_gcp_format("/v1/messages/count_tokens"),
        "messagesCountTokens"
    );
}

// -- resolve_from_env --

#[test]
fn without_prediction_mode_every_aip_variable_is_ignored() {
    let _guard = env_lock();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::remove_var("AIP_MODE");
        std::env::set_var("AIP_HTTP_PORT", "not-a-port");
        std::env::set_var("AIP_PREDICT_ROUTE", "colliding");
        std::env::set_var("AIP_HEALTH_ROUTE", "hp");
    }
    let resolved = resolve_from_env().expect("no error while disabled");
    assert!(resolved.is_none());
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("AIP_MODE", "prediction"); // case-sensitive upstream
    }
    assert!(resolve_from_env().expect("still disabled").is_none());
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::remove_var("AIP_MODE");
        std::env::remove_var("AIP_HTTP_PORT");
        std::env::remove_var("AIP_PREDICT_ROUTE");
        std::env::remove_var("AIP_HEALTH_ROUTE");
    }
}

#[test]
fn prediction_mode_resolves_defaults_and_leading_slashes() {
    let _guard = env_lock();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("AIP_MODE", "PREDICTION");
        std::env::remove_var("AIP_HTTP_PORT");
        std::env::remove_var("AIP_PREDICT_ROUTE");
        std::env::remove_var("AIP_HEALTH_ROUTE");
    }
    let resolved = resolve_from_env().expect("resolves").expect("enabled");
    assert_eq!(resolved.port, 8080);
    assert_eq!(resolved.routes.path_predict, "/predict");
    assert_eq!(resolved.routes.path_health, None);

    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("AIP_HTTP_PORT", "9090");
        std::env::set_var("AIP_PREDICT_ROUTE", "custom-predict");
        std::env::set_var("AIP_HEALTH_ROUTE", "hp");
    }
    let resolved = resolve_from_env().expect("resolves").expect("enabled");
    assert_eq!(resolved.port, 9090);
    // Leading slashes are ensured, as upstream's getenv helper does.
    assert_eq!(resolved.routes.path_predict, "/custom-predict");
    assert_eq!(resolved.routes.path_health.as_deref(), Some("/hp"));

    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("AIP_HTTP_PORT", "not-a-port");
    }
    let err = resolve_from_env().expect_err("bad port fails startup");
    assert!(err.to_string().contains("AIP_HTTP_PORT"));

    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::remove_var("AIP_MODE");
        std::env::remove_var("AIP_HTTP_PORT");
        std::env::remove_var("AIP_PREDICT_ROUTE");
        std::env::remove_var("AIP_HEALTH_ROUTE");
    }
}

// -- dispatch table --

#[test]
fn alias_collisions_resolve_deterministically_to_the_v1_routes() {
    let config = ServerConfig::default();
    let (aliases, direct) = dispatch_table(&config);
    // `/v1/completions` (OpenAI) and `/completions` (native) both produce
    // `completions`; registration order makes the OpenAI route win. Same
    // for `embeddings`.
    assert_eq!(aliases["completions"].path, "/v1/completions");
    assert_eq!(aliases["embeddings"].path, "/v1/embeddings");
    assert_eq!(aliases["chatCompletions"].path, "/v1/chat/completions");
    assert_eq!(aliases["messages"].path, "/v1/messages");
    assert_eq!(aliases["rerank"].path, "/v1/rerank");
    assert_eq!(aliases["tokenize"].path, "/tokenize");
    assert_eq!(aliases["health"].path, "/health");
    // The native spellings keep their distinct aliases.
    assert_eq!(aliases["completion"].path, "/completion");
    assert_eq!(aliases["embedding"].path, "/embedding");
    // Direct paths are dispatchable as-is, with their registered method.
    assert_eq!(direct["/completions"], axum::http::Method::POST);
    assert_eq!(direct["/v1/models"], axum::http::Method::GET);
    // The root's empty alias is skipped.
    assert!(!aliases.contains_key(""));
}

#[test]
fn observability_routes_are_always_aliased() {
    // Since #1440 the /props, /slots and /metrics routes mount
    // unconditionally (their gates live in the handlers), so their aliases
    // are always present; /slots/:id_slot collapses onto the earlier
    // /slots alias by the first-wins rule.
    let (aliases, _) = dispatch_table(&ServerConfig::default());
    assert_eq!(aliases["props"].path, "/props");
    assert_eq!(aliases["slots"].path, "/slots");
    assert_eq!(aliases["slots"].method, axum::http::Method::GET);
    assert_eq!(aliases["metrics"].path, "/metrics");
}

// -- collision check --

#[test]
fn predict_collision_is_refused_and_free_paths_pass() {
    let mut config = ServerConfig {
        gcp: Some(GcpRoutes {
            path_health: None,
            path_predict: "/predict".to_string(),
        }),
        ..Default::default()
    };
    check_predict_collision(&config).expect("/predict is free");

    config.gcp = Some(GcpRoutes {
        path_health: None,
        path_predict: "/v1/chat/completions".to_string(),
    });
    let err = check_predict_collision(&config).expect_err("collision refused");
    assert!(err.to_string().contains("AIP_PREDICT_ROUTE"));

    config.gcp = None;
    check_predict_collision(&config).expect("disabled adapter never collides");
}

// -- route integration --

fn gcp_config(path_health: Option<&str>, path_predict: &str) -> ServerConfig {
    ServerConfig {
        gcp: Some(GcpRoutes {
            path_health: path_health.map(str::to_string),
            path_predict: path_predict.to_string(),
        }),
        ..Default::default()
    }
}

fn app_with(config: ServerConfig) -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    );
    create_app(state)
}

fn predict_request(uri: &str, body: serde_json::Value, auth: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(key) = auth {
        builder = builder.header("authorization", format!("Bearer {key}"));
    }
    builder
        .body(Body::from(body.to_string()))
        .expect("request builds")
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .expect("body collects");
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| panic!("non-JSON body: {}", String::from_utf8_lossy(&bytes)))
}

fn chat_instance() -> serde_json::Value {
    serde_json::json!({
        "@requestFormat": "chatCompletions",
        "model": "route-test-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
}

#[tokio::test]
async fn predict_is_not_mounted_without_the_gcp_config() {
    let app = app_with(ServerConfig::default());
    let response = app
        .oneshot(predict_request(
            "/predict",
            serde_json::json!({"instances": []}),
            None,
        ))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn predict_fans_out_and_returns_predictions_in_request_order() {
    let app = app_with(gcp_config(None, "/predict"));
    let body = serde_json::json!({
        "instances": [
            chat_instance(),
            {"@requestFormat": "health"},
            {"@requestFormat": "definitely-not-a-route"},
        ]
    });
    let response = app
        .oneshot(predict_request("/predict", body, None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    let parsed = body_json(response).await;
    let predictions = parsed["predictions"].as_array().expect("predictions array");
    assert_eq!(predictions.len(), 3);
    // Slot 0: a real chat completion object from the recording provider.
    assert!(
        predictions[0]["choices"].is_array(),
        "chat prediction: {parsed}"
    );
    // Slot 1: the health handler's body, dispatched as GET.
    assert!(
        predictions[1]["status"].is_string(),
        "health prediction: {parsed}"
    );
    // Slot 2: a per-instance error object, not a whole-batch failure.
    assert_eq!(
        predictions[2]["error"]["message"],
        "no handler registered for @requestFormat: definitely-not-a-route"
    );
}

#[tokio::test]
async fn predict_accepts_direct_paths_and_forces_stream_off() {
    let app = app_with(gcp_config(None, "/predict"));
    let mut streaming_chat = chat_instance();
    streaming_chat["@requestFormat"] = serde_json::json!("/v1/chat/completions");
    streaming_chat["stream"] = serde_json::json!(true);
    let body = serde_json::json!({ "instances": [streaming_chat] });
    let response = app
        .oneshot(predict_request("/predict", body, None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    let parsed = body_json(response).await;
    // The stream field was forced off, so the slot carries a full JSON chat
    // completion, not an SSE stream and not the streaming refusal.
    assert!(
        parsed["predictions"][0]["choices"].is_array(),
        "got: {parsed}"
    );
}

#[tokio::test]
async fn predict_envelope_validation_matches_upstream() {
    let app = app_with(gcp_config(None, "/predict"));

    // Not a JSON object.
    let response = app
        .clone()
        .oneshot(predict_request("/predict", serde_json::json!([1, 2]), None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let parsed = body_json(response).await;
    assert_eq!(
        parsed["error"]["message"],
        "request body must be a JSON object"
    );

    // Missing / non-array instances.
    for body in [
        serde_json::json!({}),
        serde_json::json!({"instances": "nope"}),
    ] {
        let response = app
            .clone()
            .oneshot(predict_request("/predict", body, None))
            .await
            .expect("request");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let parsed = body_json(response).await;
        assert_eq!(
            parsed["error"]["message"],
            "request body must include an array field named instances"
        );
    }

    // Cap at 128 instances.
    let too_many: Vec<serde_json::Value> = (0..129)
        .map(|_| serde_json::json!({"@requestFormat": "health"}))
        .collect();
    let response = app
        .clone()
        .oneshot(predict_request(
            "/predict",
            serde_json::json!({"instances": too_many}),
            None,
        ))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let parsed = body_json(response).await;
    assert_eq!(
        parsed["error"]["message"],
        "instances array exceeds maximum size of 128"
    );

    // Per-instance shape errors land in their slots.
    let body = serde_json::json!({
        "instances": [42, {"no_format": true}, {"@requestFormat": 7}]
    });
    let response = app
        .oneshot(predict_request("/predict", body, None))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    let parsed = body_json(response).await;
    let predictions = parsed["predictions"].as_array().expect("array");
    assert_eq!(
        predictions[0]["error"]["message"],
        "each instance must be a JSON object"
    );
    for p in &predictions[1..] {
        assert_eq!(
            p["error"]["message"],
            "each instance must include a string @requestFormat"
        );
    }
}

#[tokio::test]
async fn predict_and_health_alias_require_the_api_key_like_every_route() {
    let config = ServerConfig {
        api_keys: crate::server::ApiKeys::from_vec(vec!["key-a".into()]),
        ..gcp_config(Some("/hp"), "/predict")
    };
    let app = app_with(config);

    // Without a key: both answer b10621's 401; neither is public.
    let response = app
        .clone()
        .oneshot(predict_request(
            "/predict",
            serde_json::json!({"instances": []}),
            None,
        ))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/hp")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    // With the key: the batch runs, and the internal dispatch is authorised
    // by the forwarded credential.
    let response = app
        .clone()
        .oneshot(predict_request(
            "/predict",
            serde_json::json!({"instances": [chat_instance()]}),
            Some("key-a"),
        ))
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    let parsed = body_json(response).await;
    assert!(parsed["predictions"][0]["choices"].is_array());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/hp")
                .header("authorization", "Bearer key-a")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn health_alias_mounts_at_a_custom_path_and_skips_collisions() {
    // A fresh alias answers the health payload.
    let app = app_with(gcp_config(Some("/hp"), "/predict"));
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/hp")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);

    // An alias naming an existing route is skipped rather than panicking in
    // axum's duplicate-route registration; the existing route still answers.
    let app = app_with(gcp_config(Some("/health"), "/predict"));
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
}
