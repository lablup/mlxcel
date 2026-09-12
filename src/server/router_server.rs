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

//! Router-mode HTTP surface (llama-server b10621 compatible, issue #1438).
//!
//! The top-level app owns the router routes (`GET/POST/DELETE /models`,
//! `POST /models/load`, `POST /models/unload`, `GET /models/sse`, the router
//! `GET /props`, `/health`) and dispatches every other request into the pool
//! entry named by the request's `model` (JSON body field on POST, `?model=`
//! query on GET/PATCH), exactly upstream's `proxy_post` / `proxy_get` contract:
//! a missing name, an unknown name, and a not-loaded model with autoload off
//! answer upstream's own 400s. CORS and API-key authentication run once at
//! this level; the dispatched sub-apps run without a CORS layer so the
//! response carries each header exactly once.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::HeaderMap;
use axum::http::{Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};

use super::config::ServerConfig;
use super::router_lifecycle::{
    CancelError, ErrorBody, ErrorEnvelope, FieldError, OperationKind, OperationState,
    ResetEventKind, UiEvent,
};
use super::router_models::{
    ROUTER_SHUTDOWN_TIMEOUT, RouterModelAction, RouterPool, RouterPoolError,
};
use super::routes::slots::{llama_error_response, llama_invalid_request};

/// How long an autoload dispatch waits for the model to become ready before
/// answering an error. b10621 waits unboundedly on the client socket; a
/// bounded wait fails the request instead of pinning a connection forever.
const AUTOLOAD_WAIT: std::time::Duration = std::time::Duration::from_secs(600);

/// Largest request body the dispatcher buffers while resolving `model`.
/// Matches the most permissive sub-app limit (the 25 MiB audio uploads) with
/// headroom.
const DISPATCH_BODY_CAP: usize = 64 * 1024 * 1024;

/// Shared state of the router-mode top level.
#[derive(Clone)]
pub struct RouterServerState {
    pub pool: Arc<RouterPool>,
    pub config: Arc<ServerConfig>,
}

fn llama_server_error(message: &str) -> Response {
    llama_error_response(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
}

fn llama_not_found(message: &str) -> Response {
    llama_error_response(StatusCode::NOT_FOUND, "not_found_error", message)
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
    let request_id = request_id();
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
            request_id,
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
    let request_id = request_id();
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
            request_id,
        }),
    )
        .into_response()
}

fn pool_error_response(err: RouterPoolError) -> Response {
    match err {
        RouterPoolError::MissingName => {
            llama_invalid_request("model name is missing from the request")
        }
        RouterPoolError::NotFound(name) => {
            llama_invalid_request(&format!("model '{name}' not found"))
        }
        RouterPoolError::NotLoaded => llama_invalid_request("model is not loaded"),
        RouterPoolError::LoadFailed(message) => llama_server_error(&message),
        RouterPoolError::LoadFailedWithEviction { message, .. } => llama_server_error(&message),
        RouterPoolError::Capacity(message) => llama_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable_error",
            &message,
        ),
        // b10621 raises these two as runtime errors, which its handler
        // wrapper answers as 500s with the exception text.
        RouterPoolError::NotRemovable(name) => llama_server_error(&format!(
            "model name={name} is not removable (not from cache)"
        )),
        RouterPoolError::AlreadyExists(name) => {
            llama_invalid_request(&format!("model '{name}' already exists"))
        }
        RouterPoolError::OperationRejected(error) => {
            llama_error_response(StatusCode::CONFLICT, &error.code, &error.message)
        }
    }
}

fn webui_pool_error_response(err: RouterPoolError) -> Response {
    match err {
        RouterPoolError::OperationRejected(error) => (
            StatusCode::CONFLICT,
            Json(ErrorEnvelope {
                error,
                request_id: request_id(),
            }),
        )
            .into_response(),
        RouterPoolError::NotFound(_) => webui_error(
            StatusCode::NOT_FOUND,
            "not_found",
            "model was not found; refresh the catalog before retrying",
            true,
        ),
        RouterPoolError::NotLoaded => webui_error(
            StatusCode::CONFLICT,
            "conflict",
            "model is not loaded",
            true,
        ),
        RouterPoolError::Capacity(_) => webui_error(
            StatusCode::CONFLICT,
            "conflict",
            "model lifecycle capacity is unavailable; refresh and retry",
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
        RouterPoolError::MissingName => webui_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "model name is required",
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
    }
}

/// b10621 `is_autoload`: the per-request `?autoload=` query overrides the
/// server-wide `--models-autoload` default.
fn is_autoload(state: &RouterServerState, query: &HashMap<String, String>) -> bool {
    match query.get("autoload").map(String::as_str) {
        None | Some("") => state.pool.autoload_default,
        Some(value) => value == "true" || value == "1",
    }
}

/// GET /health, GET /v1/health: the router itself is ready once listening.
async fn router_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET /props: the router's own identity block without `?model=`, the named
/// model's `/props` with it (b10621 `get_router_props`).
async fn router_props(
    State(state): State<RouterServerState>,
    Query(query): Query<HashMap<String, String>>,
    request: Request<Body>,
) -> Response {
    if query.get("model").is_none_or(|m| m.is_empty()) {
        return Json(serde_json::json!({
            "role": "router",
            "max_instances": state.pool.models_max,
            "models_autoload": state.pool.autoload_default,
            // b10621 sends a dummy alias/path pair so UIs do not break;
            // mlxcel's dummy names itself rather than llama-server.
            "model_alias": "mlxcel-server",
            "model_path": "none",
            "default_generation_settings": {
                "params": {},
                "n_ctx": 0,
            },
            "ui_settings": {},
            "build_info": concat!("mlxcel-", env!("CARGO_PKG_VERSION")),
            "cors_proxy_enabled": false,
        }))
        .into_response();
    }
    dispatch(state, request).await
}

/// The b10621 router model object (`get_router_models`).
fn router_model_json(
    snapshot: &super::router_models::RouterModelSnapshot,
    created: i64,
) -> serde_json::Value {
    let mut status = serde_json::json!({
        "value": snapshot.status.as_str(),
        // b10621 reports the child process argv; the in-process pool has
        // none (recorded as a by_design divergence on --models-dir).
        "args": [],
    });
    if let Some(preset_ini) = &snapshot.preset_ini {
        status["preset"] = serde_json::Value::String(preset_ini.clone());
    }
    if snapshot.failed {
        status["failed"] = true.into();
    }
    let mut input_modalities = vec!["text"];
    if snapshot.vision {
        input_modalities.push("image");
    }
    if snapshot.audio {
        input_modalities.push("audio");
    }
    let mut model = serde_json::json!({
        "id": snapshot.name,
        "aliases": snapshot.aliases,
        "tags": snapshot.tags,
        "object": "model",
        "owned_by": "llamacpp",
        "created": created,
        "status": status,
        "architecture": {
            "input_modalities": input_modalities,
            "output_modalities": ["text"],
        },
        "source": snapshot.source.as_str(),
        // Only cache-sourced models are removable, b10621's own rule.
        "can_remove": snapshot.source == super::router_models::RouterModelSource::Cache,
    });
    // b10621 reflects download progress through the entry's loaded_info.
    if let Some(info) = &snapshot.download_info {
        model["progress"] = info
            .get("progress")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
    }
    model
}

/// GET /models, GET /v1/models (router list). `?reload=1` rescans the
/// directory first, upstream's `reload` switch.
async fn router_models_list(
    State(state): State<RouterServerState>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    if query.get("reload").is_some_and(|v| !v.is_empty())
        && let Err(err) = state.pool.rescan()
    {
        return llama_server_error(&err.to_string());
    }
    let created = chrono::Utc::now().timestamp();
    let data: Vec<serde_json::Value> = state
        .pool
        .snapshot()
        .iter()
        // b10621 skips cache models a preset deduplicated.
        .filter(|snapshot| !snapshot.hidden)
        .map(|snapshot| router_model_json(snapshot, created))
        .collect();
    Json(serde_json::json!({ "data": data, "object": "list" })).into_response()
}

#[derive(serde::Deserialize, Default)]
struct ModelActionBody {
    model: Option<String>,
}

fn body_model_name(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<ModelActionBody>(body)
        .ok()
        .and_then(|b| b.model)
}

/// POST /models/load (b10621 `post_router_models_load`).
async fn router_models_load(
    State(state): State<RouterServerState>,
    body: axum::body::Bytes,
) -> Response {
    let name = body_model_name(&body).unwrap_or_default();
    let Some(entry) = state.pool.get(&name) else {
        // Upstream's load handler is the one place an unknown model is a 404.
        return llama_not_found("model is not found");
    };
    if entry.is_running() {
        return llama_invalid_request("model is already running");
    }
    if let Err(err) = state.pool.begin_load(&name).await {
        return pool_error_response(err);
    }
    Json(serde_json::json!({ "success": true })).into_response()
}

/// POST /models/unload (b10621 `post_router_models_unload`).
async fn router_models_unload(
    State(state): State<RouterServerState>,
    body: axum::body::Bytes,
) -> Response {
    let name = body_model_name(&body).unwrap_or_default();
    let Some(entry) = state.pool.get(&name) else {
        return llama_invalid_request("model is not found");
    };
    // b10621 accepts unload for running AND downloading models (unloading a
    // downloading model cancels the download).
    if !entry.is_running() && !entry.is_downloading() && !entry.reserves_capacity() {
        return llama_invalid_request("model is not running");
    }
    match state.pool.unload(&name).await {
        Ok(()) => Json(serde_json::json!({ "success": true })).into_response(),
        Err(err) => pool_error_response(err),
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum UiModelActionKind {
    Load,
    Unload,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UiLoadProfile {
    ctx_size: Option<u64>,
    n_parallel: Option<u64>,
    kv_cache_mode: Option<String>,
}

impl UiLoadProfile {
    fn has_overrides(&self) -> bool {
        self.ctx_size.is_some() || self.n_parallel.is_some() || self.kv_cache_mode.is_some()
    }
}

const MODEL_ID_PREFIX: &str = "mdl_";
const MODEL_ID_SUFFIX_LEN: usize = 43;
const IDEMPOTENCY_KEY_MIN: usize = 8;
const IDEMPOTENCY_KEY_MAX: usize = 128;
const OPERATION_ID_MAX: usize = 128;
const CURSOR_TOKEN_MAX: usize = 512;
const OPERATION_TARGET_TOKEN_MAX: usize = 128;
const KV_CACHE_MODE_NAMES: &[&str] = &[
    "fp16",
    "float16",
    "int8",
    "i8",
    "turbo4-asym",
    "fp16+turbo4",
    "turbo3-asym",
    "fp16+turbo3",
    "turbo3",
    "turbo4",
    "turbo4-sym",
    "turbo4-delegated",
    "fp16+turbo4-delegated",
];

fn is_webui_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'~' | b'-')
}

fn is_webui_token(value: &str, min: usize, max: usize) -> bool {
    let bytes = value.as_bytes();
    (min..=max).contains(&bytes.len()) && bytes.iter().copied().all(is_webui_token_byte)
}

fn valid_model_id(value: &str) -> bool {
    let Some(suffix) = value.strip_prefix(MODEL_ID_PREFIX) else {
        return false;
    };
    suffix.len() == MODEL_ID_SUFFIX_LEN
        && suffix
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
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
    if !valid_model_id(value) {
        return Some(invalid_webui_field(
            field,
            "invalid_format",
            "model_id must match ^mdl_[A-Za-z0-9_-]{43}$",
        ));
    }
    None
}

fn validate_operation_id(value: &str, field: &str) -> Option<Response> {
    if !is_webui_token(value, 1, OPERATION_ID_MAX) {
        return Some(invalid_webui_field(
            field,
            "invalid_format",
            "operation id must be a printable token of at most 128 characters",
        ));
    }
    None
}

fn validate_cursor(value: &str) -> Option<Response> {
    if !is_webui_token(value, 1, CURSOR_TOKEN_MAX) {
        return Some(invalid_webui_field(
            "cursor",
            "invalid_format",
            "cursor must be a printable cursor token",
        ));
    }
    let Some(offset) = value.strip_prefix("ops_") else {
        return Some(invalid_webui_field(
            "cursor",
            "invalid_cursor",
            "cursor must be returned by a previous operations response",
        ));
    };
    if offset.is_empty() || offset.parse::<usize>().is_err() {
        return Some(invalid_webui_field(
            "cursor",
            "invalid_cursor",
            "cursor must be returned by a previous operations response",
        ));
    }
    None
}

fn validate_target_filter(value: &str) -> Option<Response> {
    if !is_webui_token(value, 1, OPERATION_TARGET_TOKEN_MAX) {
        return Some(invalid_webui_field(
            "target",
            "invalid_format",
            "target must be a printable token of at most 128 characters",
        ));
    }
    None
}

fn validate_load_profile(profile: &UiLoadProfile) -> Option<Response> {
    if let Some(ctx_size) = profile.ctx_size
        && !(1..=262_144).contains(&ctx_size)
    {
        return Some(invalid_webui_field(
            "load_profile.ctx_size",
            "out_of_range",
            "ctx_size must be between 1 and 262144",
        ));
    }
    if let Some(n_parallel) = profile.n_parallel
        && !(1..=32).contains(&n_parallel)
    {
        return Some(invalid_webui_field(
            "load_profile.n_parallel",
            "out_of_range",
            "n_parallel must be between 1 and 32",
        ));
    }
    if let Some(mode) = profile.kv_cache_mode.as_deref()
        && !KV_CACHE_MODE_NAMES.contains(&mode)
    {
        return Some(invalid_webui_field(
            "load_profile.kv_cache_mode",
            "invalid_enum",
            "kv_cache_mode must be one of the WebUI contract KV cache mode names",
        ));
    }
    None
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UiModelActionRequest {
    model_id: String,
    action: UiModelActionKind,
    expected_revision: u64,
    idempotency_key: String,
    load_profile: Option<UiLoadProfile>,
    eviction_target_id: Option<String>,
}

fn ui_action_to_router(action: &UiModelActionKind) -> RouterModelAction {
    match action {
        UiModelActionKind::Load => RouterModelAction::Load,
        UiModelActionKind::Unload => RouterModelAction::Unload,
    }
}

/// POST /ui-api/v1/model-actions.
async fn ui_model_actions(
    State(state): State<RouterServerState>,
    body: axum::body::Bytes,
) -> Response {
    let request: UiModelActionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            return webui_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("invalid model action request: {err}"),
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
    if let Some(target_id) = request.eviction_target_id.as_deref()
        && let Some(response) = validate_model_id(target_id, "eviction_target_id")
    {
        return response;
    }
    if let Some(profile) = request.load_profile.as_ref()
        && let Some(response) = validate_load_profile(profile)
    {
        return response;
    }
    if request
        .load_profile
        .as_ref()
        .is_some_and(UiLoadProfile::has_overrides)
    {
        return webui_field_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported",
            "next-load profile overrides are not implemented in this runtime path yet",
            "load_profile",
            "unsupported",
            "ctx_size, n_parallel and kv_cache_mode are parsed but not applied by issue #1839",
        );
    }
    match state.pool.submit_model_action(
        &request.model_id,
        ui_action_to_router(&request.action),
        request.expected_revision,
        &request.idempotency_key,
        request.eviction_target_id.as_deref(),
    ) {
        Ok(accepted) => (StatusCode::ACCEPTED, Json(accepted)).into_response(),
        Err(err) => {
            tracing::warn!(error = ?err, "router: rejected WebUI model action request");
            webui_pool_error_response(err)
        }
    }
}

#[derive(Default, serde::Deserialize)]
struct OperationsQuery {
    limit: Option<usize>,
    cursor: Option<String>,
    state: Option<String>,
    kind: Option<String>,
    target: Option<String>,
}

fn parse_operation_state(value: Option<&str>) -> Option<OperationState> {
    match value? {
        "queued" => Some(OperationState::Queued),
        "running" => Some(OperationState::Running),
        "cancelling" => Some(OperationState::Cancelling),
        "succeeded" => Some(OperationState::Succeeded),
        "failed" => Some(OperationState::Failed),
        "cancelled" => Some(OperationState::Cancelled),
        _ => None,
    }
}

fn parse_operation_kind(value: Option<&str>) -> Option<OperationKind> {
    match value? {
        "catalog_refresh" => Some(OperationKind::CatalogRefresh),
        "model_load" => Some(OperationKind::ModelLoad),
        "model_unload" => Some(OperationKind::ModelUnload),
        "download" => Some(OperationKind::Download),
        "model_removal" => Some(OperationKind::ModelRemoval),
        "settings_patch" => Some(OperationKind::SettingsPatch),
        _ => None,
    }
}

async fn ui_operations_list(
    State(state): State<RouterServerState>,
    Query(query): Query<OperationsQuery>,
) -> Response {
    let limit = query.limit.unwrap_or(50);
    if !(1..=200).contains(&limit) {
        return webui_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "limit must be between 1 and 200",
            true,
        );
    }
    if let Some(cursor) = query.cursor.as_deref()
        && let Some(response) = validate_cursor(cursor)
    {
        return response;
    }
    let state_filter = match query.state.as_deref() {
        Some(value) => match parse_operation_state(Some(value)) {
            Some(state) => Some(state),
            None => {
                return invalid_webui_field(
                    "state",
                    "invalid_enum",
                    "state must be one of queued, running, cancelling, succeeded, failed or cancelled",
                );
            }
        },
        None => None,
    };
    let kind_filter = match query.kind.as_deref() {
        Some(value) => match parse_operation_kind(Some(value)) {
            Some(kind) => Some(kind),
            None => {
                return invalid_webui_field(
                    "kind",
                    "invalid_enum",
                    "kind must be one of catalog_refresh, model_load, model_unload, download, model_removal or settings_patch",
                );
            }
        },
        None => None,
    };
    if let Some(target) = query.target.as_deref()
        && let Some(response) = validate_target_filter(target)
    {
        return response;
    }
    Json(state.pool.lifecycle_coordinator().list_operations(
        limit,
        query.cursor.as_deref(),
        state_filter,
        kind_filter,
        query.target.as_deref(),
    ))
    .into_response()
}

async fn ui_operation_get(
    State(state): State<RouterServerState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = validate_operation_id(&id, "id") {
        return response;
    }
    match state.pool.lifecycle_coordinator().get_operation(&id) {
        Some(operation) => Json(operation).into_response(),
        None => webui_error(
            StatusCode::NOT_FOUND,
            "not_found",
            "operation not found",
            true,
        ),
    }
}

async fn ui_operation_cancel(
    State(state): State<RouterServerState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = validate_operation_id(&id, "id") {
        return response;
    }
    match state.pool.lifecycle_coordinator().cancel_operation(&id) {
        Ok(accepted) => (StatusCode::ACCEPTED, Json(accepted)).into_response(),
        Err(CancelError::NotFound) => webui_error(
            StatusCode::NOT_FOUND,
            "not_found",
            "operation not found",
            true,
        ),
        Err(CancelError::Unsupported { operation_id }) => {
            let request_id = request_id();
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorEnvelope {
                    error: ErrorBody {
                        code: "unsupported".to_string(),
                        message: "operation cancellation is not supported by the current worker"
                            .to_string(),
                        retryable: false,
                        field_errors: None,
                        operation_id: Some(operation_id),
                    },
                    request_id,
                }),
            )
                .into_response()
        }
    }
}

fn event_to_sse(event: UiEvent) -> Event {
    let id = event.event_id.clone();
    let name = event.event_type.clone();
    let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
    Event::default().id(id).event(name).data(data)
}

async fn ui_events(State(state): State<RouterServerState>, headers: HeaderMap) -> Response {
    let coordinator = state.pool.lifecycle_coordinator();
    let last_event_id = headers
        .get("last-event-id")
        .or_else(|| headers.get("Last-Event-ID"))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty());
    let (receiver, replay) =
        coordinator.subscribe_for_ui(last_event_id, state.pool.runtime_model_ids());
    let stream = futures::stream::unfold(
        (replay.into_iter(), receiver, coordinator),
        |state| async move {
            let (mut replay, mut receiver, coordinator) = state;
            if let Some(event) = replay.next() {
                return Some((
                    Ok::<Event, std::convert::Infallible>(event_to_sse(event)),
                    (replay, receiver, coordinator),
                ));
            }
            match receiver.recv().await {
                Ok(event) => Some((Ok(event_to_sse(event)), (replay, receiver, coordinator))),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    let event = coordinator.local_reset_event("gap", ResetEventKind::Gap);
                    Some((Ok(event_to_sse(event)), (replay, receiver, coordinator)))
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => None,
            }
        },
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// POST /models (b10621 `post_router_models`): validate the name as a
/// fetchable HuggingFace repository, then download it into the model cache
/// in the background, reporting progress through `GET /models/sse`
/// (issue #1438).
async fn router_models_add(
    State(state): State<RouterServerState>,
    body: axum::body::Bytes,
) -> Response {
    let name = body_model_name(&body).unwrap_or_default();
    if name.is_empty() {
        return llama_invalid_request("model must be a non-empty string");
    }
    if state.pool.lookup(&name).is_some() {
        return llama_invalid_request(&format!("model '{name}' already exists"));
    }
    if !state.pool.has_cache() {
        return llama_server_error(
            "adding a model by download requires a model cache; set --model-store-root, \
             MLXCEL_MODELS_DIR, or MLXCEL_CACHE_DIR",
        );
    }
    // Normalize into the `<owner>/<name>` repo id the cache stores (a bare
    // name expands to the default organization, like `-m <name>`). The
    // normalized spelling is the entry name the SSE events and the model
    // list will carry.
    let repo_id = match state.pool.normalize_cache_name(&name) {
        Ok(repo_id) => repo_id,
        Err(err) => return llama_invalid_request(&err.to_string()),
    };
    if repo_id != name && state.pool.lookup(&repo_id).is_some() {
        return llama_invalid_request(&format!("model '{repo_id}' already exists"));
    }
    // b10621 validates by fetching repository metadata before answering; a
    // failed probe is a 500 from its handler wrapper. The probe blocks on
    // the network, so it runs on the blocking pool.
    let probe_pool = state.pool.clone();
    let probe_repo = repo_id.clone();
    let probed = tokio::task::spawn_blocking(move || probe_pool.validate_cache_repo(&probe_repo))
        .await
        .unwrap_or_else(|join_err| Err(anyhow::anyhow!(join_err.to_string())));
    if let Err(err) = probed {
        return llama_server_error(&format!("model validation failed: {err:#}"));
    }
    match state.pool.start_download(&repo_id) {
        Ok(()) => Json(serde_json::json!({ "success": true })).into_response(),
        Err(err) => pool_error_response(err),
    }
}

/// DELETE /models (b10621 `del_router_models`): only cache-sourced models
/// are removable. Cancels an in-flight download or stops a running
/// instance, deletes the snapshot from disk, and emits `model_remove`
/// (issue #1438).
async fn router_models_delete(
    State(state): State<RouterServerState>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let name = query.get("model").cloned().unwrap_or_default();
    if name.is_empty() {
        return llama_invalid_request("model must be a non-empty string");
    }
    if state.pool.get(&name).is_none() {
        return llama_server_error(&format!("model name={name} is not found"));
    }
    match state.pool.remove(&name).await {
        Ok(()) => Json(serde_json::json!({ "success": true })).into_response(),
        Err(err) => pool_error_response(err),
    }
}

/// GET /models/sse (b10621 `get_router_models_sse`): the model-event stream.
async fn router_models_sse(State(state): State<RouterServerState>) -> Response {
    let receiver = state.pool.subscribe();
    let stream = futures::stream::unfold(receiver, |mut receiver| async move {
        loop {
            match receiver.recv().await {
                Ok(event) => {
                    let payload = Event::default().data(event.to_string());
                    return Some((Ok::<Event, std::convert::Infallible>(payload), receiver));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Minimal query parsing for the dispatcher: keys and values are split on
/// `&`/`=` and percent-decoded. Model names are directory basenames, so this
/// covers the realistic value space without a URL crate.
fn parse_query(query: Option<&str>) -> HashMap<String, String> {
    fn decode(value: &str) -> String {
        let bytes = value.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'%' if i + 2 < bytes.len() + 1 && i + 2 < bytes.len() => {
                    let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                    match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                        Some(byte) => {
                            out.push(byte);
                            i += 3;
                        }
                        None => {
                            out.push(bytes[i]);
                            i += 1;
                        }
                    }
                }
                b'+' => {
                    out.push(b' ');
                    i += 1;
                }
                byte => {
                    out.push(byte);
                    i += 1;
                }
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
    query
        .unwrap_or("")
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((k, v)) => (decode(k), decode(v)),
            None => (decode(pair), String::new()),
        })
        .collect()
}

/// The fallback dispatcher: b10621 `proxy_post` / `proxy_get` in-process.
fn dispatch_forwards_body(method: &Method) -> bool {
    method == Method::POST || method == Method::PATCH
}

async fn buffer_dispatch_body(method: &Method, body: Body) -> Result<axum::body::Bytes, Response> {
    if !dispatch_forwards_body(method) {
        return Ok(axum::body::Bytes::new());
    }
    axum::body::to_bytes(body, DISPATCH_BODY_CAP)
        .await
        .map_err(|_| llama_invalid_request("request body too large"))
}

fn dispatch_model_name(method: &Method, query: &HashMap<String, String>, body: &[u8]) -> String {
    if method == Method::POST {
        body_model_name(body).unwrap_or_default()
    } else {
        query.get("model").cloned().unwrap_or_default()
    }
}

async fn dispatch(state: RouterServerState, request: Request<Body>) -> Response {
    let (parts, body) = request.into_parts();
    let query = parse_query(parts.uri.query());
    let autoload = is_autoload(&state, &query);

    let forwards_body = dispatch_forwards_body(&parts.method);
    let body_bytes = match buffer_dispatch_body(&parts.method, body).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };
    let name = dispatch_model_name(&parts.method, &query, &body_bytes);

    let entry = match state.pool.resolve(&name, autoload) {
        Ok(entry) => entry,
        Err(err) => return pool_error_response(err),
    };
    // `resolve` accepts preset aliases; the load path addresses the entry by
    // its real name.
    if autoload && let Err(err) = state.pool.ensure_ready(&entry.name, AUTOLOAD_WAIT).await {
        return pool_error_response(err);
    }

    let rebuilt = Request::from_parts(parts, Body::from(body_bytes));
    state.pool.dispatch(&entry, rebuilt, forwards_body).await
}

async fn dispatch_fallback(
    State(state): State<RouterServerState>,
    request: Request<Body>,
) -> Response {
    dispatch(state, request).await
}

/// Router-level API-key middleware: same key set and same public-path rule as
/// the single-model server ([`super::app::is_public_endpoint`]).
async fn router_api_key_auth(
    State(state): State<RouterServerState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if state.config.api_keys.is_empty() || super::app::is_public_endpoint(request.uri().path()) {
        return next.run(request).await;
    }
    match super::auth::presented_credential(request.headers()) {
        Some(presented) if state.config.api_keys.accepts(presented) => next.run(request).await,
        _ => super::auth::unauthorized_response(),
    }
}

/// UI management routes are never implicitly public: the explicit WebUI
/// router accessor must be paired with a configured API key until #1837/#1838
/// define the production mount and browser auth story.
async fn router_ui_api_key_auth(
    State(state): State<RouterServerState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if state.config.api_keys.is_empty() {
        return super::auth::unauthorized_response();
    }
    match super::auth::presented_credential(request.headers()) {
        Some(presented) if state.config.api_keys.accepts(presented) => next.run(request).await,
        _ => super::auth::unauthorized_response(),
    }
}

async fn router_cors_middleware(
    State(state): State<RouterServerState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    super::cors::apply_cors_policy(&state.config.cors_policy, request, next).await
}

fn router_base_routes() -> axum::Router<RouterServerState> {
    axum::Router::new()
        .route("/health", get(router_health))
        .route("/v1/health", get(router_health))
        .route("/", get(router_health))
        .route("/props", get(router_props))
        .route(
            "/models",
            get(router_models_list)
                .post(router_models_add)
                .delete(router_models_delete),
        )
        .route("/v1/models", get(router_models_list))
        .route("/models/load", post(router_models_load))
        .route("/models/unload", post(router_models_unload))
        .route("/models/sse", get(router_models_sse))
}

fn router_ui_routes(state: RouterServerState) -> axum::Router<RouterServerState> {
    axum::Router::new()
        .route("/ui-api/v1/model-actions", post(ui_model_actions))
        .route("/ui-api/v1/operations", get(ui_operations_list))
        .route("/ui-api/v1/operations/:id", get(ui_operation_get))
        .route(
            "/ui-api/v1/operations/:id/cancel",
            post(ui_operation_cancel),
        )
        .route("/ui-api/v1/events", get(ui_events))
        .layer(middleware::from_fn_with_state(
            state,
            router_ui_api_key_auth,
        ))
}

fn finish_router_layers(
    routes: axum::Router<RouterServerState>,
    state: &RouterServerState,
) -> axum::Router<RouterServerState> {
    routes
        .fallback(dispatch_fallback)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            router_api_key_auth,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            router_cors_middleware,
        ))
        .layer(tower_http::trace::TraceLayer::new_for_http())
}

fn finish_router_app(
    routes: axum::Router<RouterServerState>,
    state: RouterServerState,
) -> axum::Router {
    finish_router_layers(routes, &state).with_state(state)
}

#[cfg(feature = "webui")]
fn finish_router_app_with_security(
    routes: axum::Router<RouterServerState>,
    state: RouterServerState,
    policy: super::webui::security::WebUiSecurityPolicy,
) -> axum::Router {
    let api_keys = state.config.api_keys.clone();
    super::webui::security::secure_webui_router(
        finish_router_layers(routes, &state),
        api_keys,
        policy,
    )
    .with_state(state)
}

/// Assemble the router-mode llama-compatible application. WebUI management
/// routes stay unmounted here until #1838 wires startup-time opt-in and #1837
/// wires browser-safe authentication.
pub fn create_router_app(state: RouterServerState) -> axum::Router {
    finish_router_app(router_base_routes(), state)
}

/// Assemble the router app with the issue #1839 WebUI adapters mounted behind
/// mandatory API-key authentication for internal handler tests and the future
/// secure startup mount.
pub fn create_router_app_with_authenticated_ui(state: RouterServerState) -> axum::Router {
    let routes = router_base_routes().merge(router_ui_routes(state.clone()));
    finish_router_app(routes, state)
}

/// Assemble the WebUI-enabled router with the browser security policy outside Trace/CORS/auth.
#[cfg(feature = "webui")]
#[allow(dead_code)]
pub(crate) fn create_router_app_with_secured_ui(
    state: RouterServerState,
    policy: super::webui::security::WebUiSecurityPolicy,
) -> axum::Router {
    let routes = router_base_routes().merge(router_ui_routes(state.clone()));
    finish_router_app_with_security(routes, state, policy)
}

#[cfg(test)]
#[path = "router_server_tests.rs"]
mod router_server_tests;

#[cfg(test)]
#[path = "router_server_security_support_tests.rs"]
mod router_server_security_support_tests;

#[cfg(test)]
#[path = "router_server_security_prefix_tests.rs"]
mod router_server_security_prefix_tests;

#[cfg(test)]
#[path = "router_server_security_tests.rs"]
mod router_server_security_tests;

/// Run the router server: discover models (cache, `--models-dir`, presets),
/// build the pool, and serve the b10621 router surface (issue #1438).
/// Reached from [`super::startup::start_server`] when `--models-dir` or
/// `--models-preset` is set and no model argument was given.
pub async fn run_router_server(
    startup: super::ServerStartupConfig,
    api_keys: super::ApiKeys,
) -> anyhow::Result<()> {
    let models_dir = startup.router_models_dir.clone();

    // b10621 loads INI presets per model; a parse error or untranslatable
    // key fails startup rather than serving un-preset models (#1438).
    let presets = match startup.models_preset.as_ref() {
        Some(path) => super::router_presets::parse_preset_file(path)?,
        None => super::router_presets::RouterPresets::default(),
    };

    // The router's model cache is the mlxcel model store (the b10621 cache
    // equivalent): removable entries, POST /models downloads into it.
    let cache = crate::downloader::models_root(startup.model_store_root.as_deref()).map(|root| {
        super::router_cache::CacheSource::new(
            root,
            Arc::new(super::router_cache::HfRouterDownloader),
        )
    });

    let base_config = super::startup::build_server_config(&startup, api_keys.clone());
    let sources = super::router_models::RouterSources {
        models_dir: models_dir.clone(),
        cache,
        presets,
    };
    let cli_overrides = super::router_presets::PresetCliOverrides::detect();
    let models_max = startup.models_max;
    let models_autoload = startup.models_autoload;
    let pool = Arc::new(RouterPool::new(
        sources,
        startup.clone(),
        api_keys,
        cli_overrides,
        models_max,
        models_autoload,
    )?);
    let discovered = pool.snapshot().len();
    tracing::info!(
        "Router mode: {} models discovered (models_dir {}, models_max {}, autoload {})",
        discovered,
        models_dir
            .as_ref()
            .map(|d| d.display().to_string())
            .unwrap_or_else(|| "disabled".to_string()),
        models_max,
        models_autoload,
    );
    // Preset-only `load-on-startup`: begin those loads before serving.
    for name in pool.load_on_startup_names() {
        tracing::info!("router: load-on-startup '{name}'");
        if let Err(err) = pool.begin_load(&name).await {
            tracing::warn!("router: load-on-startup '{name}' failed: {err:?}");
        }
    }
    let shutdown_pool = pool.clone();
    let state = RouterServerState {
        pool,
        config: Arc::new(base_config),
    };
    let app = create_router_app(state);
    tokio::select! {
        served = super::startup::serve_http(&startup, app) => served,
        signal = tokio::signal::ctrl_c() => {
            signal?;
            tracing::info!(
                timeout_secs = ROUTER_SHUTDOWN_TIMEOUT.as_secs(),
                "router: shutdown signal received; draining lifecycle operations"
            );
            let report = shutdown_pool.shutdown_all(ROUTER_SHUTDOWN_TIMEOUT).await;
            if report.timed_out || !report.remaining.is_empty() {
                tracing::warn!(
                    attempted = report.attempted,
                    completed = report.completed.len(),
                    remaining = ?report.remaining,
                    timed_out = report.timed_out,
                    "router: lifecycle shutdown finished with pending resources"
                );
            } else {
                tracing::info!(
                    attempted = report.attempted,
                    completed = report.completed.len(),
                    "router: lifecycle shutdown completed"
                );
            }
            Ok(())
        }
    }
}
