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

//! Tests for the b10621 `llamacpp:` metric families on `GET /metrics`
//! (issue #1440): name parity, the enablement gate's diagnostic, the
//! `Process-Start-Time-Unix` header, value semantics, and the bounded label
//! cardinality of the whole exposition.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

use crate::server::config::ServerConfig;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn metrics_state(enabled: bool) -> AppState {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    AppState::new(
        provider,
        ServerConfig {
            enable_metrics_endpoint: enabled,
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("metrics-test-model"),
        batch_metrics,
    )
}

async fn scrape(state: AppState) -> (StatusCode, Option<String>, String) {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .expect("request");
    let response = create_app(state).oneshot(request).await.expect("response");
    let status = response.status();
    let start_time = response
        .headers()
        .get("Process-Start-Time-Unix")
        .map(|v| v.to_str().expect("header utf8").to_string());
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 22)
        .await
        .expect("body");
    (
        status,
        start_time,
        String::from_utf8(bytes.to_vec()).expect("utf8 body"),
    )
}

/// The exact family names b10621's `to_metrics` exports; a llama-server
/// Prometheus scrape config matches these and nothing else.
const B10621_COUNTERS: [&str; 10] = [
    "llamacpp:prompt_tokens_total",
    "llamacpp:prompt_tokens_cached_total",
    "llamacpp:prompt_seconds_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total",
    "llamacpp:n_decode_total",
    "llamacpp:n_tokens_max",
    "llamacpp:spec_decode_num_draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total",
    "llamacpp:spec_decode_num_drafts_total",
];
const B10621_GAUGES: [&str; 5] = [
    "llamacpp:prompt_tokens_seconds",
    "llamacpp:predicted_tokens_seconds",
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
    "llamacpp:n_busy_slots_per_decode",
];

#[tokio::test]
async fn the_b10621_families_are_exported_with_help_and_type_lines() {
    let (status, start_time, body) = scrape(metrics_state(true)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        start_time.is_some_and(|t| t.parse::<i64>().is_ok()),
        "Process-Start-Time-Unix must carry a unix timestamp"
    );
    for name in B10621_COUNTERS {
        assert!(body.contains(&format!("# TYPE {name} counter")), "{name}");
        assert!(body.contains(&format!("\n{name} ")), "{name} sample line");
    }
    for name in B10621_GAUGES {
        assert!(body.contains(&format!("# TYPE {name} gauge")), "{name}");
        assert!(body.contains(&format!("\n{name} ")), "{name} sample line");
    }
}

#[tokio::test]
async fn disabled_metrics_endpoint_answers_the_b10621_diagnostic_not_404() {
    let (status, _, body) = scrape(metrics_state(false)).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let json: serde_json::Value = serde_json::from_str(&body).expect("error envelope json");
    assert_eq!(json["error"]["code"], 501);
    assert_eq!(json["error"]["type"], "not_supported_error");
    assert_eq!(
        json["error"]["message"],
        "This server does not support metrics endpoint. Start it with `--metrics`"
    );
}

/// The llama counter values reflect the observability accumulators: a
/// completed request's prompt split (processed vs cached) and generation
/// count land in the families a b10621 dashboard graphs.
#[tokio::test]
async fn request_completion_feeds_the_llama_counters() {
    let state = metrics_state(true);
    state
        .batch_observability
        .record_request_completion(100, 30, 400, 2000, 50);
    state.batch_observability.record_decode_step(2);
    state.batch_observability.record_decode_step(0);
    let (_, _, body) = scrape(state).await;

    let value = |name: &str| -> f64 {
        body.lines()
            .find(|l| l.starts_with(&format!("{name} ")))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN)
    };
    assert_eq!(value("llamacpp:prompt_tokens_total"), 70.0);
    assert_eq!(value("llamacpp:prompt_tokens_cached_total"), 30.0);
    assert_eq!(value("llamacpp:tokens_predicted_total"), 50.0);
    assert_eq!(value("llamacpp:prompt_seconds_total"), 0.4);
    assert_eq!(value("llamacpp:tokens_predicted_seconds_total"), 2.0);
    assert_eq!(value("llamacpp:n_tokens_max"), 150.0);
    // The zero-sized guard step is not a llama_decode call.
    assert_eq!(value("llamacpp:n_decode_total"), 1.0);
    assert_eq!(value("llamacpp:n_busy_slots_per_decode"), 2.0);
}

/// The throughput gauges average over the window between two scrapes, like
/// b10621's `reset_bucket`: after a scrape drains the bucket, a second
/// scrape with no new work reports zero, not the lifetime average.
#[tokio::test]
async fn throughput_gauges_reset_between_scrapes() {
    let state = metrics_state(true);
    state
        .batch_observability
        .record_request_completion(10, 0, 1000, 1000, 100);
    let (_, _, first) = scrape(state.clone()).await;
    let (_, _, second) = scrape(state).await;
    let value = |body: &str, name: &str| -> f64 {
        body.lines()
            .find(|l| l.starts_with(&format!("{name} ")))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN)
    };
    assert_eq!(value(&first, "llamacpp:predicted_tokens_seconds"), 100.0);
    assert_eq!(value(&second, "llamacpp:predicted_tokens_seconds"), 0.0);
    // The cumulative counter is unaffected by the bucket reset.
    assert_eq!(value(&second, "llamacpp:tokens_predicted_total"), 100.0);
}

/// Label cardinality is bounded: every label key/value pair in the whole
/// exposition comes from a fixed, code-defined set. Nothing derived from
/// request content (model ids, prompts, filenames, client input) may appear
/// as a label value, so a scrape target cannot be blown up by traffic.
#[tokio::test]
async fn label_cardinality_is_bounded_and_code_defined() {
    let allowed_label_keys: HashSet<&str> = [
        "reason", "path", "le", "stage", "from", "to", "position", "quantile",
    ]
    .into_iter()
    .collect();
    let (_, _, body) = scrape(metrics_state(true)).await;
    let mut labeled_series = 0usize;
    for line in body.lines() {
        if line.starts_with('#') {
            continue;
        }
        let Some(open) = line.find('{') else { continue };
        let close = line.find('}').expect("closing brace");
        labeled_series += 1;
        for pair in line[open + 1..close].split(',') {
            let key = pair.split('=').next().expect("label key").trim();
            assert!(
                allowed_label_keys.contains(key),
                "unexpected label key {key} in {line}"
            );
        }
    }
    // The exposition is a fixed set of series; a run of this size stays well
    // under a bound that a per-request or per-model label would blow through
    // immediately.
    assert!(
        labeled_series < 200,
        "labeled series count {labeled_series} suggests unbounded cardinality"
    );
}
