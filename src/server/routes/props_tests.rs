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

//! Tests for the b10621 `/props` surface (issue #1440): the GET response
//! carries the upstream key set, GET is ungated, and `--props` gates POST.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

use super::{default_generation_settings, default_generation_settings_with_live, geometry_block};
use crate::server::config::ServerConfig;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn app_with(config: ServerConfig) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("props-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn send(app: Router, method: Method, uri: &str) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from("{}"))
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

#[test]
fn props_reports_the_resolved_dry_sequence_breakers() {
    let config = ServerConfig {
        default_dry_sequence_breakers: vec![198, 271],
        ..Default::default()
    };

    let settings = default_generation_settings(&config);

    assert_eq!(
        settings["dry_sequence_breakers"],
        serde_json::json!([198, 271]),
        "an operator must be able to read back what --dry-sequence-breaker resolved to"
    );
}

#[test]
fn props_reports_an_unset_breaker_list_as_empty_rather_than_omitting_it() {
    let settings = default_generation_settings(&ServerConfig::default());

    // Present-but-empty and absent are different answers to "is this flag
    // doing anything". The gap #1103 closed was that the field was absent, so
    // there was no way to tell the flag was inert.
    assert_eq!(
        settings["dry_sequence_breakers"],
        serde_json::json!([]),
        "the key must exist even when no breakers are configured"
    );
}

#[test]
fn props_reports_the_same_patched_defaults_as_live_settings() {
    let config = ServerConfig::default();
    let mut live = config.live_settings();
    live.default_max_tokens = 73;
    live.default_temperature = 0.27;
    live.default_top_k = 17;
    live.default_top_p = 0.81;
    live.default_min_p = 0.04;
    live.default_seed = Some(42);
    live.default_repetition_penalty = 1.12;
    live.default_repetition_context_size = 256;
    live.default_frequency_penalty = 0.13;
    live.default_presence_penalty = 0.29;
    live.default_dry_multiplier = 0.37;
    live.default_dry_base = 1.61;
    live.default_dry_allowed_length = 5;
    live.default_dry_penalty_last_n = 211;
    live.default_dry_sequence_breakers = vec![3, 5];

    let settings = default_generation_settings_with_live(&config, &live);

    assert_eq!(settings["n_predict"], serde_json::json!(73));
    assert_eq!(settings["max_tokens"], serde_json::json!(73));
    assert_eq!(settings["temperature"], serde_json::json!(0.27_f32));
    assert_eq!(settings["top_k"], serde_json::json!(17));
    assert_eq!(settings["top_p"], serde_json::json!(0.81_f32));
    assert_eq!(settings["min_p"], serde_json::json!(0.04_f32));
    assert_eq!(settings["seed"], serde_json::json!(42));
    assert_eq!(settings["repeat_penalty"], serde_json::json!(1.12_f32));
    assert_eq!(settings["repeat_last_n"], serde_json::json!(256));
    assert_eq!(settings["frequency_penalty"], serde_json::json!(0.13_f32));
    assert_eq!(settings["presence_penalty"], serde_json::json!(0.29_f32));
    assert_eq!(settings["dry_multiplier"], serde_json::json!(0.37_f32));
    assert_eq!(settings["dry_base"], serde_json::json!(1.61_f32));
    assert_eq!(settings["dry_allowed_length"], serde_json::json!(5));
    assert_eq!(settings["dry_penalty_last_n"], serde_json::json!(211));
    assert_eq!(settings["dry_sequence_breakers"], serde_json::json!([3, 5]));
}

/// The reported key set is the contract this function was extracted to make
/// assertable, so assert it. Without this, a regression that dropped `top_k`
/// or `seed` from the payload would pass every other test in this file.
#[test]
fn params_reports_exactly_the_documented_key_set() {
    let settings = default_generation_settings(&ServerConfig::default());
    let mut keys: Vec<&str> = settings
        .as_object()
        .expect("default_generation_settings is a JSON object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();

    assert_eq!(
        keys,
        vec![
            "dry_allowed_length",
            "dry_base",
            "dry_multiplier",
            "dry_penalty_last_n",
            "dry_sequence_breakers",
            "frequency_penalty",
            "ignore_eos",
            "max_tokens",
            "min_p",
            "n_predict",
            "presence_penalty",
            "repeat_last_n",
            "repeat_penalty",
            "seed",
            "temperature",
            "top_k",
            "top_n_sigma",
            "top_p",
            "typical_p",
            "xtc_probability",
            "xtc_threshold",
        ],
    );
}

/// The context/batch geometry that used to live inside
/// `default_generation_settings` stays reported, under the `geometry`
/// extension block (#1450).
#[test]
fn geometry_block_reports_batch_and_kv_bounds() {
    let config = ServerConfig {
        prefill_chunk_size: 512,
        max_batch_size: 4,
        max_kv_size: Some(4096),
        ..Default::default()
    };
    let block = geometry_block(&config);
    assert_eq!(block["n_batch"], 512);
    assert_eq!(block["n_ubatch"], 512);
    assert_eq!(block["n_batch_decode"], 4);
    assert_eq!(block["n_kv_max"], 4096);
}

/// GET /props answers the b10621 key set. This is the golden-schema gate for
/// the manifest's `GET /props` entry: a key disappearing from this list is a
/// compatibility regression, not a refactor detail.
#[tokio::test]
async fn get_props_carries_the_b10621_key_set() {
    let (status, body) = send(
        app_with(ServerConfig {
            n_parallel: 3,
            context_size: 2048,
            ..Default::default()
        }),
        Method::GET,
        "/props",
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    for key in [
        "default_generation_settings",
        "total_slots",
        "model_alias",
        "model_ftype",
        "model_path",
        "modalities",
        "media_marker",
        "endpoint_slots",
        "endpoint_props",
        "endpoint_metrics",
        "ui",
        "ui_settings",
        "chat_template",
        "chat_template_caps",
        "bos_token",
        "eos_token",
        "build_info",
        "is_sleeping",
        "cors_proxy_enabled",
    ] {
        assert!(body.get(key).is_some(), "missing b10621 /props key {key}");
    }

    // b10621 shape details a schema-driven client depends on.
    assert_eq!(body["total_slots"], 3);
    assert_eq!(body["default_generation_settings"]["n_ctx"], 2048);
    assert!(body["default_generation_settings"]["params"].is_object());
    let modalities = body["modalities"].as_object().expect("modalities object");
    let mut modality_keys: Vec<&str> = modalities.keys().map(String::as_str).collect();
    modality_keys.sort_unstable();
    assert_eq!(
        modality_keys,
        vec!["audio", "video", "vision"],
        "modalities carries exactly b10621's three flags"
    );
    assert_eq!(body["is_sleeping"], false);
    // The `by_design` divergence on the manifest's `GET /props` entry:
    // mlxcel takes media as content parts and has no textual marker, so
    // `media_marker` is deliberately null where b10621 reports mtmd's
    // `<__media__>`. Null, not absent, and never an invented marker.
    assert!(
        body["media_marker"].is_null(),
        "media_marker must be null: {:?}",
        body["media_marker"]
    );
    let caps = body["chat_template_caps"].as_object().expect("caps object");
    assert_eq!(caps.len(), 9, "the nine jinja::caps keys: {caps:?}");
}

/// GET /props is ungated in b10621; `--props` gates POST only.
#[tokio::test]
async fn get_props_is_served_without_the_props_flag() {
    let (status, _) = send(
        app_with(ServerConfig {
            enable_props_endpoint: false,
            ..Default::default()
        }),
        Method::GET,
        "/props",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn post_props_without_the_flag_answers_the_b10621_diagnostic() {
    let (status, body) = send(
        app_with(ServerConfig {
            enable_props_endpoint: false,
            ..Default::default()
        }),
        Method::POST,
        "/props",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["error"]["code"], 501);
    assert_eq!(
        body["error"]["message"],
        "This server does not support changing global properties. Start it with `--props`"
    );
}

#[tokio::test]
async fn post_props_with_the_flag_acknowledges_like_b10621() {
    let (status, body) = send(
        app_with(ServerConfig {
            enable_props_endpoint: true,
            ..Default::default()
        }),
        Method::POST,
        "/props",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, serde_json::json!({ "success": true }));
}

/// The endpoint toggle report follows the flags.
#[tokio::test]
async fn endpoint_toggles_reflect_the_flags() {
    let (_, body) = send(
        app_with(ServerConfig {
            enable_props_endpoint: true,
            enable_metrics_endpoint: true,
            enable_slots_endpoint: false,
            ..Default::default()
        }),
        Method::GET,
        "/props",
    )
    .await;
    assert_eq!(body["endpoint_props"], true);
    assert_eq!(body["endpoint_metrics"], true);
    assert_eq!(body["endpoint_slots"], false);
}
