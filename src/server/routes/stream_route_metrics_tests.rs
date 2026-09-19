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

//! `Metrics` recording for streamed chat completions (#1911).
//!
//! The WebUI chat always streams, and its Activity tiles read
//! `Metrics.requests_total` and `completion_tokens_total`, so a streamed
//! completion has to be recorded like a non-streamed one. The scripted
//! provider's result reports `prompt_tokens: 1`, `generation_time_ms: 1` and
//! one completion token per emitted step.

use std::sync::atomic::Ordering;

use axum::http::StatusCode;
use tower::ServiceExt;

use super::{
    chat_request, collect_body, get_request, read_until, scripted_app_with_state, wait_until,
};
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::{AppState, ServerConfig};

/// The tokens a completed stream emits. Distinctive enough that reading one
/// back cannot match the completion id or the role chunk.
const TOKENS: [&str; 3] = ["alpha", "beta", "gamma"];

/// Drive one plain (no `X-Conversation-Id`) streaming chat to completion:
/// emit [`TOKENS`], finish, and read the body through `[DONE]`.
///
/// The route records the completion on the generation thread before it sends
/// `[DONE]`, and the SSE channel orders the two, so the counters are final
/// once this returns.
async fn complete_plain_stream(app: &axum::Router, handle: &ScriptedStreamHandle) {
    let response = app
        .clone()
        .oneshot(chat_request(None, None, None))
        .await
        .expect("stream request");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    for token in TOKENS {
        handle.token(token);
    }
    handle.finish();
    let mut bytes = Vec::new();
    read_until(&mut body, &mut bytes, "[DONE]").await;
}

/// Wait until the streaming generation task has returned.
///
/// The task holds its completion-control registration until it returns,
/// which is after the point where it decides whether to record, so an empty
/// registry means that decision is made. Asserting a counter is still zero
/// any earlier would pass whether or not the route skips the record.
async fn wait_for_generation_task_exit(state: &AppState) {
    wait_until("the streaming generation task exits", || {
        state.completion_controls.len_for_tests() == 0
    })
    .await;
}

/// The acceptance case: a completed stream advances every counter by the
/// result's own numbers. The request sets no `stream_options`, so the record
/// does not depend on `include_usage`.
#[tokio::test]
async fn completed_stream_records_one_request_with_its_counts() {
    let (app, handle, _options_rx, state) = scripted_app_with_state(ServerConfig::default());

    complete_plain_stream(&app, &handle).await;

    let m = &state.metrics;
    assert_eq!(m.requests_total.load(Ordering::Relaxed), 1);
    assert_eq!(
        m.completion_tokens_total.load(Ordering::Relaxed),
        TOKENS.len() as u64
    );
    assert_eq!(m.prompt_tokens_total.load(Ordering::Relaxed), 1);
    assert_eq!(m.generation_time_ms_total.load(Ordering::Relaxed), 1);

    // Recorded once: nothing after `[DONE]` adds a second record.
    wait_for_generation_task_exit(&state).await;
    assert_eq!(m.requests_total.load(Ordering::Relaxed), 1);
}

/// A plain stream the client disconnects from is cancelled, and a cancelled
/// stream is not a completion. The worker still answers `Ok` with the tokens
/// it emitted before the abort, so only the cancellation flag keeps this
/// request out of the counters.
#[tokio::test]
async fn client_cancelled_plain_stream_is_not_recorded() {
    let (app, handle, _options_rx, state) = scripted_app_with_state(ServerConfig::default());

    let response = app
        .clone()
        .oneshot(chat_request(None, None, None))
        .await
        .expect("stream request");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    handle.token(TOKENS[0]);
    read_until(&mut body, &mut bytes, TOKENS[0]).await;
    drop(body);

    let flag = handle.cancellation_flag(0).expect("one generation");
    wait_until("disconnect cancels the plain stream", || {
        handle.token(TOKENS[1]);
        flag.load(Ordering::Acquire)
    })
    .await;
    handle.finish();
    wait_for_generation_task_exit(&state).await;

    let m = &state.metrics;
    assert_eq!(
        m.requests_total.load(Ordering::Relaxed),
        0,
        "a client-cancelled stream must not count as a completed request"
    );
    assert_eq!(m.completion_tokens_total.load(Ordering::Relaxed), 0);
}

/// A resumable stream survives a client disconnect: the generation runs to
/// the end into its session, the cancellation flag stays clear, and the
/// finished completion counts with every token it produced.
#[tokio::test]
async fn resumable_stream_is_recorded_after_a_client_disconnect() {
    let (app, handle, _options_rx, state) = scripted_app_with_state(ServerConfig::default());

    let response = app
        .clone()
        .oneshot(chat_request(Some("conv-metrics"), None, None))
        .await
        .expect("stream request");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    handle.token(TOKENS[0]);
    read_until(&mut body, &mut bytes, TOKENS[0]).await;
    drop(body);

    for token in &TOKENS[1..] {
        handle.token(token);
    }
    handle.finish();
    wait_for_generation_task_exit(&state).await;

    let flag = handle.cancellation_flag(0).expect("one generation");
    assert!(
        !flag.load(Ordering::Acquire),
        "a client disconnect must not cancel a resumable stream"
    );
    let m = &state.metrics;
    assert_eq!(m.requests_total.load(Ordering::Relaxed), 1);
    assert_eq!(
        m.completion_tokens_total.load(Ordering::Relaxed),
        TOKENS.len() as u64
    );
}

/// `/metrics` reports a completed stream. The lines are matched whole, so
/// `mlxcel_requests_total 11` cannot satisfy the first expectation.
#[tokio::test]
async fn metrics_endpoint_reports_a_completed_stream() {
    let config = ServerConfig {
        enable_metrics_endpoint: true,
        ..Default::default()
    };
    let (app, handle, _options_rx, _state) = scripted_app_with_state(config);

    complete_plain_stream(&app, &handle).await;

    let response = app
        .clone()
        .oneshot(get_request("/metrics", None))
        .await
        .expect("scrape");
    assert_eq!(response.status(), StatusCode::OK);
    let text = String::from_utf8(collect_body(response).await).expect("utf8 body");
    let expected = [
        "mlxcel_requests_total 1".to_string(),
        format!("mlxcel_completion_tokens_total {}", TOKENS.len()),
    ];
    for line in &expected {
        assert!(
            text.lines().any(|l| l == line.as_str()),
            "/metrics must print {line:?}:\n{text}"
        );
    }
    let help = text
        .lines()
        .find(|l| l.starts_with("# HELP mlxcel_requests_total "))
        .expect("requests_total HELP line");
    assert!(
        help.contains("cancelled by the client are excluded"),
        "the HELP line must state the cancelled-stream exclusion: {help}"
    );
}

/// The WebUI Activity tiles read `completed_requests_total` and
/// `completion_tokens_total` from the runtime projection; a completed stream
/// moves both.
///
/// `runtime_snapshot` lives behind the `webui` feature (`server::webui` is
/// `#[cfg(feature = "webui")]` in `server/mod.rs`), so this test needs the
/// same gate; without it, `--no-default-features --features xla-diagnostics`
/// fails to compile this file.
#[cfg(feature = "webui")]
#[tokio::test]
async fn activity_projection_counts_a_completed_stream() {
    let config = ServerConfig {
        enable_metrics_endpoint: true,
        ..Default::default()
    };
    let (app, handle, _options_rx, state) = scripted_app_with_state(config);

    complete_plain_stream(&app, &handle).await;

    let snapshot = crate::server::webui::runtime::runtime_snapshot(
        "srv-route-test".into(),
        "mdl-route-test".into(),
        1,
        1,
        &state.config,
        Some(&state),
    );
    let value = |key: &str| snapshot.measurements.get(key).and_then(|m| m.value);
    assert_eq!(value("completed_requests_total"), Some(1.0));
    assert_eq!(value("completion_tokens_total"), Some(TOKENS.len() as f64));
}
