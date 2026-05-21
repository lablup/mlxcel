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

//! Health check endpoint (llama-server compatible).
//!
//! Reports server liveness, model loading status, batch scheduler metrics,
//! and detailed observability counters.

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

use crate::server::AppState;
use crate::server::types::{BatchStatusInfo, HealthResponse};

/// Build model-level health fields once the model is confirmed loaded.
///
/// Returns `(context_size, tool_call_parser)`:
/// - `context_size`: the effective per-slot context window (0 = model default).
/// - `tool_call_parser`: `Some("mlxcel")` when the chat template supports
///   tool calls; `None` when the template does not expose the `tools`
///   variable and tool-call parsing will therefore never activate.
fn model_health_fields(state: &AppState) -> (usize, Option<String>) {
    let context_size = state.config.context_size;
    let tool_call_parser = if state.chat_template.supports_tools_hint() {
        Some("mlxcel".to_string())
    } else {
        None
    };
    (context_size, tool_call_parser)
}

/// GET /health
///
/// Returns status with batch metrics and observability counters:
/// - `{"status": "ok", "batch": {...}, "observability": {...}, "context_size": N, "tool_call_parser": "mlxcel"|null}` when model is loaded
/// - `{"status": "no slot available", ...}` when all slots are busy and queue is full
/// - `{"status": "loading model"}` when model is still loading
pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    if !state.model_provider.is_loaded() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "loading model".to_string(),
                model: None,
                batch: None,
                observability: None,
                context_size: None,
                tool_call_parser: None,
            }),
        );
    }

    let batch_info = BatchStatusInfo {
        active_sequences: state.batch_metrics.active_count(),
        queue_depth: state.batch_metrics.queue_depth(),
        max_batch_size: state.config.max_batch_size,
    };

    let obs_snapshot = state.batch_observability.snapshot();
    let (context_size, tool_call_parser) = model_health_fields(&state);

    let has_capacity = state.can_accept_request();
    if !has_capacity {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "no slot available".to_string(),
                model: Some(state.display_model_id().to_string()),
                batch: Some(batch_info),
                observability: Some(obs_snapshot),
                context_size: Some(context_size),
                tool_call_parser,
            }),
        );
    }

    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok".to_string(),
            model: Some(state.display_model_id().to_string()),
            batch: Some(batch_info),
            observability: Some(obs_snapshot),
            context_size: Some(context_size),
            tool_call_parser,
        }),
    )
}

#[cfg(test)]
mod tests {
    use crate::server::chat_template::ChatTemplateProcessor;
    use crate::server::types::HealthResponse;

    // -----------------------------------------------------------------------
    // Unit tests for model_health_fields logic
    // -----------------------------------------------------------------------

    /// A chat template that references the `tools` variable should produce
    /// `tool_call_parser = Some("mlxcel")`.
    #[test]
    fn tool_call_parser_is_mlxcel_when_template_supports_tools() {
        // Template using `for tool in tools` — the most common pattern.
        let tpl = r#"{% for tool in tools %}{{ tool.function.name }}{% endfor %}"#;
        let processor = ChatTemplateProcessor::with_template(tpl.to_string());
        assert!(
            processor.supports_tools_hint(),
            "template with 'for tool in tools' must be recognized as tools-capable"
        );
    }

    /// A chat template that does NOT reference the `tools` variable should
    /// produce `tool_call_parser = None`.
    #[test]
    fn tool_call_parser_is_none_when_template_does_not_support_tools() {
        let tpl = r#"{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let processor = ChatTemplateProcessor::with_template(tpl.to_string());
        assert!(
            !processor.supports_tools_hint(),
            "plain template without tools must not be recognized as tools-capable"
        );
    }

    /// A template using `tools | tojson` (common in Qwen / Hermes families)
    /// must also be recognized.
    #[test]
    fn tool_call_parser_recognized_for_tools_tojson() {
        let tpl = r#"{{ tools | tojson }}"#;
        let processor = ChatTemplateProcessor::with_template(tpl.to_string());
        assert!(processor.supports_tools_hint());
    }

    /// The `context_size` field mirrors `ServerConfig::context_size`.
    /// When the operator sets `--ctx-size`, that value is reflected; when the
    /// field is 0, it means "use model default".
    #[test]
    fn context_size_field_matches_config_value() {
        // The HealthResponse struct carries context_size as Option<usize>.
        // Presence once loaded, absent while loading — verify the type contract.
        let loaded_resp = HealthResponse {
            status: "ok".to_string(),
            model: Some("test-model".to_string()),
            batch: None,
            observability: None,
            context_size: Some(4096),
            tool_call_parser: Some("mlxcel".to_string()),
        };
        assert_eq!(loaded_resp.context_size, Some(4096));

        // Model default (operator did not set --ctx-size)
        let default_resp = HealthResponse {
            status: "ok".to_string(),
            model: Some("test-model".to_string()),
            batch: None,
            observability: None,
            context_size: Some(0),
            tool_call_parser: None,
        };
        assert_eq!(default_resp.context_size, Some(0));

        // Loading state — fields must be absent
        let loading_resp = HealthResponse {
            status: "loading model".to_string(),
            model: None,
            batch: None,
            observability: None,
            context_size: None,
            tool_call_parser: None,
        };
        assert!(loading_resp.context_size.is_none());
    }

    /// When `tool_call_parser` is `None` the serialized JSON must not be missing
    /// the key — it must appear as `"tool_call_parser": null`.
    #[test]
    fn tool_call_parser_null_serializes_as_explicit_null() {
        let resp = HealthResponse {
            status: "ok".to_string(),
            model: Some("m".to_string()),
            batch: None,
            observability: None,
            context_size: Some(2048),
            tool_call_parser: None,
        };
        let json = serde_json::to_string(&resp).expect("serialization must succeed");
        assert!(
            json.contains("\"tool_call_parser\":null"),
            "tool_call_parser must serialize as explicit null, got: {json}"
        );
    }

    /// When `tool_call_parser` is `Some("mlxcel")` the serialized JSON must
    /// contain the string value.
    #[test]
    fn tool_call_parser_mlxcel_serializes_correctly() {
        let resp = HealthResponse {
            status: "ok".to_string(),
            model: Some("m".to_string()),
            batch: None,
            observability: None,
            context_size: Some(2048),
            tool_call_parser: Some("mlxcel".to_string()),
        };
        let json = serde_json::to_string(&resp).expect("serialization must succeed");
        assert!(
            json.contains("\"tool_call_parser\":\"mlxcel\""),
            "tool_call_parser must serialize as string 'mlxcel', got: {json}"
        );
    }

    /// `context_size` is skipped entirely while the model is loading (`None`)
    /// but present once loaded.
    #[test]
    fn context_size_absent_while_loading_present_when_loaded() {
        let loading = HealthResponse {
            status: "loading model".to_string(),
            model: None,
            batch: None,
            observability: None,
            context_size: None,
            tool_call_parser: None,
        };
        let json = serde_json::to_string(&loading).expect("serialize loading");
        assert!(
            !json.contains("context_size"),
            "context_size must be absent while loading, got: {json}"
        );

        let loaded = HealthResponse {
            status: "ok".to_string(),
            model: Some("m".to_string()),
            batch: None,
            observability: None,
            context_size: Some(8192),
            tool_call_parser: None,
        };
        let json2 = serde_json::to_string(&loaded).expect("serialize loaded");
        assert!(
            json2.contains("\"context_size\":8192"),
            "context_size must be present once loaded, got: {json2}"
        );
    }

    /// While the model is loading, `tool_call_parser` is `None` but lacks a
    /// `skip_serializing_if` attribute, so it serializes as the JSON literal
    /// `null` (not as an absent key).  `context_size`, by contrast, *does* have
    /// `skip_serializing_if = "Option::is_none"` and is therefore absent during
    /// loading.  This test asserts the former contract: monitoring clients that
    /// receive `"tool_call_parser":null` during startup know the model has not
    /// configured a parser yet.
    #[test]
    fn tool_call_parser_serializes_null_while_loading() {
        let loading = HealthResponse {
            status: "loading model".to_string(),
            model: None,
            batch: None,
            observability: None,
            context_size: None,
            tool_call_parser: None,
        };
        let json = serde_json::to_string(&loading).expect("serialize loading");
        // tool_call_parser is always serialized (no skip_serializing_if).
        assert!(
            json.contains("\"tool_call_parser\":null"),
            "tool_call_parser must be null while loading, got: {json}"
        );
    }
}
