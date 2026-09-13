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

//! WebUI library management routes.
//!
//! These routes expose safe model-library operations to the browser without
//! changing router startup state: they operate only on the existing
//! [`RouterServerState`] and delegate all authority to [`RouterPool`].

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::post;

use super::super::router_lifecycle::{ErrorBody, ErrorEnvelope, FieldError};
use super::super::router_models::RouterPoolError;
use super::super::router_server::RouterServerState;

const MODEL_ID_PREFIX: &str = "mdl_";
const MODEL_ID_SUFFIX_LEN: usize = 43;
const IDEMPOTENCY_KEY_MIN: usize = 8;
const IDEMPOTENCY_KEY_MAX: usize = 128;
const REVISION_REF_MAX: usize = 128;

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DownloadRequest {
    repo_id: String,
    revision: Option<String>,
    idempotency_key: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RemovalRequest {
    model_id: String,
    expected_revision: u64,
    idempotency_key: String,
}

/// Routes mounted by the startup/WebUI owner under `/ui-api/v1`.
pub(crate) fn routes() -> Router<RouterServerState> {
    Router::new()
        .route("/downloads", post(ui_downloads))
        .route("/model-removals", post(ui_model_removals))
}

async fn ui_downloads(State(state): State<RouterServerState>, body: axum::body::Bytes) -> Response {
    let request: DownloadRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            return webui_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("invalid download request: {err}"),
                true,
            );
        }
    };
    if let Some(response) = validate_repo_id(&request.repo_id) {
        return response;
    }
    if let Some(revision) = request.revision.as_deref()
        && let Some(response) = validate_revision_ref(revision)
    {
        return response;
    }
    if let Some(response) = validate_idempotency_key(&request.idempotency_key) {
        return response;
    }
    if crate::downloader::offline_mode() {
        return webui_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported",
            "offline mode is enabled; download the model out of band into the configured cache",
            false,
        );
    }
    if !state.pool.has_cache() {
        return webui_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported",
            "no model cache is configured; set --model-store-root, MLXCEL_MODELS_DIR, or MLXCEL_CACHE_DIR",
            false,
        );
    }
    match state.pool.submit_download(
        &request.repo_id,
        request.revision.as_deref(),
        Some(&request.idempotency_key),
    ) {
        Ok(accepted) => (StatusCode::ACCEPTED, Json(accepted)).into_response(),
        Err(err) => webui_pool_error_response(err),
    }
}

async fn ui_model_removals(
    State(state): State<RouterServerState>,
    body: axum::body::Bytes,
) -> Response {
    let request: RemovalRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            return webui_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("invalid model removal request: {err}"),
                true,
            );
        }
    };
    if let Some(response) = validate_model_id(&request.model_id, "model_id") {
        return response;
    }
    if request.expected_revision == 0 {
        return invalid_webui_field(
            "expected_revision",
            "out_of_range",
            "expected_revision must be at least 1",
        );
    }
    if let Some(response) = validate_idempotency_key(&request.idempotency_key) {
        return response;
    }
    match state.pool.submit_cache_removal_by_model_id(
        &request.model_id,
        request.expected_revision,
        &request.idempotency_key,
    ) {
        Ok(accepted) => (StatusCode::ACCEPTED, Json(accepted)).into_response(),
        Err(err) => webui_pool_error_response(err),
    }
}

fn request_id() -> String {
    format!("req_{}", chrono::Utc::now().timestamp_micros())
}

fn webui_error(
    status: StatusCode,
    code: &str,
    message: impl Into<String>,
    retryable: bool,
) -> Response {
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody {
                code: code.to_string(),
                message: message.into(),
                retryable,
                field_errors: None,
                operation_id: None,
            },
            request_id: request_id(),
        }),
    )
        .into_response()
}

fn webui_field_error(
    status: StatusCode,
    code: &str,
    message: impl Into<String>,
    field: &str,
    field_code: &str,
    field_message: impl Into<String>,
) -> Response {
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody {
                code: code.to_string(),
                message: message.into(),
                retryable: true,
                field_errors: Some(vec![FieldError {
                    field: field.to_string(),
                    code: field_code.to_string(),
                    message: field_message.into(),
                }]),
                operation_id: None,
            },
            request_id: request_id(),
        }),
    )
        .into_response()
}

fn invalid_webui_field(
    field: &str,
    field_code: &str,
    field_message: impl Into<String>,
) -> Response {
    webui_field_error(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        "request does not match the WebUI contract",
        field,
        field_code,
        field_message,
    )
}

fn webui_pool_error_response(err: RouterPoolError) -> Response {
    match err {
        RouterPoolError::OperationRejected(error) => operation_error_response(error),
        RouterPoolError::NotFound(_) => webui_error(
            StatusCode::NOT_FOUND,
            "not_found",
            "model was not found; refresh the catalog before retrying",
            true,
        ),
        RouterPoolError::NotRemovable(_) => webui_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported",
            "model is not removable",
            false,
        ),
        RouterPoolError::AlreadyExists(_) => webui_error(
            StatusCode::CONFLICT,
            "conflict",
            "model already exists",
            true,
        ),
        RouterPoolError::Capacity(_) => webui_error(
            StatusCode::CONFLICT,
            "conflict",
            "model lifecycle capacity is unavailable; refresh and retry",
            true,
        ),
        RouterPoolError::MissingName => webui_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "model name is required",
            true,
        ),
        RouterPoolError::NotLoaded => webui_error(
            StatusCode::CONFLICT,
            "conflict",
            "model is not loaded",
            true,
        ),
        RouterPoolError::LoadFailed(_) | RouterPoolError::LoadFailedWithEviction { .. } => {
            webui_error(
                StatusCode::CONFLICT,
                "conflict",
                "model load failed; see server logs",
                true,
            )
        }
    }
}

fn operation_error_response(error: ErrorBody) -> Response {
    let status = match error.code.as_str() {
        "not_found" => StatusCode::NOT_FOUND,
        "unsupported" | "auth_required" => StatusCode::UNPROCESSABLE_ENTITY,
        "rate_limited" => StatusCode::TOO_MANY_REQUESTS,
        "invalid_request" => StatusCode::BAD_REQUEST,
        _ => StatusCode::CONFLICT,
    };
    (
        status,
        Json(ErrorEnvelope {
            error,
            request_id: request_id(),
        }),
    )
        .into_response()
}

fn validate_idempotency_key(value: &str) -> Option<Response> {
    if value.len() < IDEMPOTENCY_KEY_MIN {
        return Some(invalid_webui_field(
            "idempotency_key",
            "too_short",
            "idempotency_key must contain at least 8 characters",
        ));
    }
    if value.len() > IDEMPOTENCY_KEY_MAX {
        return Some(invalid_webui_field(
            "idempotency_key",
            "too_long",
            "idempotency_key must contain at most 128 characters",
        ));
    }
    if !is_webui_token(value, IDEMPOTENCY_KEY_MIN, IDEMPOTENCY_KEY_MAX) {
        return Some(invalid_webui_field(
            "idempotency_key",
            "invalid_format",
            "idempotency_key may only contain A-Z, a-z, 0-9, dot, underscore, tilde and dash",
        ));
    }
    None
}

fn validate_model_id(value: &str, field: &str) -> Option<Response> {
    let Some(suffix) = value.strip_prefix(MODEL_ID_PREFIX) else {
        return Some(invalid_webui_field(
            field,
            "invalid_format",
            "model_id must match ^mdl_[A-Za-z0-9_-]{43}$",
        ));
    };
    if suffix.len() != MODEL_ID_SUFFIX_LEN
        || !suffix
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Some(invalid_webui_field(
            field,
            "invalid_format",
            "model_id must match ^mdl_[A-Za-z0-9_-]{43}$",
        ));
    }
    None
}

fn validate_repo_id(value: &str) -> Option<Response> {
    let mut parts = value.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if parts.next().is_some() || !valid_hf_repo_segment(owner) || !valid_hf_repo_segment(name) {
        return Some(invalid_webui_field(
            "repo_id",
            "invalid_format",
            "repo_id must be a public HuggingFace owner/name id using A-Z, a-z, 0-9, dot, underscore and dash",
        ));
    }
    None
}

fn validate_revision_ref(value: &str) -> Option<Response> {
    if value.len() > REVISION_REF_MAX || value.is_empty() || !valid_revision_ref(value) {
        return Some(invalid_webui_field(
            "revision",
            "invalid_format",
            "revision must be 1-128 characters and may only contain A-Z, a-z, 0-9, dot, underscore, tilde, plus, slash and dash; dot-only path segments are rejected",
        ));
    }
    None
}

fn is_webui_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'~' | b'-')
}

fn is_webui_token(value: &str, min: usize, max: usize) -> bool {
    let bytes = value.as_bytes();
    (min..=max).contains(&bytes.len()) && bytes.iter().copied().all(is_webui_token_byte)
}

fn valid_hf_repo_segment(segment: &str) -> bool {
    let bytes = segment.as_bytes();
    (1..=96).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && segment != "."
        && segment != ".."
        && bytes
            .iter()
            .copied()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn valid_revision_ref(value: &str) -> bool {
    value.split('/').all(|segment| {
        !segment.is_empty()
            && segment != "."
            && segment != ".."
            && segment
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'~' | b'+' | b'-'))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_id_validation_rejects_path_and_url_shapes() {
        assert!(validate_repo_id("mlx-community/Qwen3-4B-4bit").is_none());
        assert!(validate_repo_id("mlx-community/../secret").is_some());
        assert!(validate_repo_id("https://huggingface.co/mlx-community/Qwen3").is_some());
        assert!(validate_repo_id("mlx-community").is_some());
    }

    #[test]
    fn revision_validation_rejects_dot_only_segments() {
        assert!(validate_revision_ref("main").is_none());
        assert!(validate_revision_ref("refs/pr/1").is_none());
        assert!(validate_revision_ref("refs/../main").is_some());
        assert!(validate_revision_ref("").is_some());
    }
}
