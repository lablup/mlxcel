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

//! Health check endpoint (llama-server b10621 compatible, issue #1440).
//!
//! b10621's `/health` has exactly two answers: `503` with the
//! `{"error": {"code": 503, "message": "Loading model", "type":
//! "unavailable_error"}}` envelope while the server is not ready (its
//! state middleware refuses every route that way during load), and `200
//! {"status": "ok"}` from then on. Load does not change it: slot
//! saturation is reported by `GET /slots?fail_on_no_slot=1`, never by
//! `/health`, so orchestrator liveness probes do not restart a busy server.
//! mlxcel used to answer `503 {"status": "no slot available"}` when the
//! queue was full and a different loading body; both were recorded
//! divergences of this route and are aligned here. The former rich health
//! payload (batch gauges, observability counters) moved to where b10621
//! reports the same data: `GET /slots` and `GET /metrics`.

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse, response::Response};

use super::slots::llama_error_response;
use crate::server::AppState;

/// The b10621 not-ready answer: its server-state middleware's exact envelope.
fn loading_model_response() -> Response {
    llama_error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "unavailable_error",
        "Loading model",
    )
}

/// Readiness of a server restricted to its side-model routes (#1452).
///
/// Ready once the worker the mode selected exists. It cannot be absent in
/// practice, because `check_serving_mode` refuses to start a server whose mode
/// resolved no worker, but the check is made here too rather than assumed:
/// this is the endpoint an operator reads when something did not come up.
fn side_model_ready(state: &AppState) -> bool {
    use crate::server::config::EmbeddingServingMode;
    match state.config.embedding_serving_mode {
        EmbeddingServingMode::RerankOnly => state.rerank_model.is_some(),
        EmbeddingServingMode::EmbeddingOnly => state.embedding_model.is_some(),
        EmbeddingServingMode::Any => true,
    }
}

/// GET /health, GET /v1/health
///
/// `200 {"status": "ok"}` once the serving worker is up; the b10621 loading
/// envelope before that. Never reports saturation (see module docs).
pub async fn health_check(State(state): State<AppState>) -> Response {
    // A server started with b10621's `--embeddings` or `--reranking` has no
    // chat worker to be "loaded", and reporting it unhealthy forever would
    // make the mode unusable behind any container probe (#1452). Readiness
    // there is whether the worker the mode selected came up.
    let ready = if state.config.embedding_serving_mode.blocks_generation() {
        side_model_ready(&state)
    } else {
        state.model_provider.is_loaded()
    };
    if !ready {
        return loading_model_response();
    }
    Json(serde_json::json!({ "status": "ok" })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ready body is exactly b10621's: `{"status": "ok"}` and nothing
    /// else. Extra keys here would be a schema divergence for clients that
    /// mirror upstream's health object.
    #[test]
    fn ready_body_is_exactly_status_ok() {
        let body = serde_json::json!({ "status": "ok" });
        assert_eq!(body.as_object().map(|o| o.len()), Some(1));
        assert_eq!(body["status"], "ok");
    }

    /// The not-ready answer carries b10621's error envelope with the numeric
    /// 503 code and upstream's exact wording.
    #[tokio::test]
    async fn loading_response_matches_b10621_envelope() {
        let response = loading_model_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(body["error"]["code"], 503);
        assert_eq!(body["error"]["message"], "Loading model");
        assert_eq!(body["error"]["type"], "unavailable_error");
    }
}
