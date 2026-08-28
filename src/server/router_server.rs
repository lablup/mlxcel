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

//! Router-mode HTTP surface (llama-server b10621 compatible, issue #1438).
//!
//! The top-level app owns the router routes (`GET/POST/DELETE /models`,
//! `POST /models/load`, `POST /models/unload`, `GET /models/sse`, the router
//! `GET /props`, `/health`) and dispatches every other request into the pool
//! entry named by the request's `model` (JSON body field on POST, `?model=`
//! query on GET), exactly upstream's `proxy_post` / `proxy_get` contract:
//! a missing name, an unknown name, and a not-loaded model with autoload off
//! answer upstream's own 400s. CORS and API-key authentication run once at
//! this level; the dispatched sub-apps run without a CORS layer so the
//! response carries each header exactly once.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};

use super::config::ServerConfig;
use super::router_models::{RouterPool, RouterPoolError};
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
        RouterPoolError::Capacity(message) => llama_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable_error",
            &message,
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
        // none.
        "args": [],
    });
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
    serde_json::json!({
        "id": snapshot.name,
        "aliases": [],
        "tags": [],
        "object": "model",
        "owned_by": "llamacpp",
        "created": created,
        "status": status,
        "architecture": {
            "input_modalities": input_modalities,
            "output_modalities": ["text"],
        },
        "source": "models_dir",
        // Only cache-sourced models are removable in b10621; a models-dir
        // pool has none.
        "can_remove": false,
    })
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
    if !entry.is_running() {
        return llama_invalid_request("model is not running");
    }
    match state.pool.unload(&name) {
        Ok(()) => Json(serde_json::json!({ "success": true })).into_response(),
        Err(err) => pool_error_response(err),
    }
}

/// POST /models (b10621 downloads the named HF repo into its cache; mlxcel
/// does not yet, and says so instead of pretending).
async fn router_models_add(
    State(state): State<RouterServerState>,
    body: axum::body::Bytes,
) -> Response {
    let name = body_model_name(&body).unwrap_or_default();
    if name.is_empty() {
        return llama_invalid_request("model must be a non-empty string");
    }
    if state.pool.get(&name).is_some() {
        return llama_invalid_request(&format!("model '{name}' already exists"));
    }
    llama_server_error(
        "adding a model by download is not supported yet; place the checkpoint directory under \
         --models-dir and call GET /models?reload=1",
    )
}

/// DELETE /models (b10621 `del_router_models`): only cache-sourced models are
/// removable, and a models-dir pool has none, so the refusals are the whole
/// reachable surface.
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
    llama_server_error(&format!(
        "model name={name} is not removable (not from cache)"
    ))
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
async fn dispatch(state: RouterServerState, request: Request<Body>) -> Response {
    let (parts, body) = request.into_parts();
    let query = parse_query(parts.uri.query());
    let autoload = is_autoload(&state, &query);

    let is_post = parts.method == Method::POST;
    let (name, body_bytes) = if is_post {
        let bytes = match axum::body::to_bytes(body, DISPATCH_BODY_CAP).await {
            Ok(bytes) => bytes,
            Err(_) => return llama_invalid_request("request body too large"),
        };
        (body_model_name(&bytes).unwrap_or_default(), bytes)
    } else {
        (
            query.get("model").cloned().unwrap_or_default(),
            axum::body::Bytes::new(),
        )
    };

    let entry = match state.pool.resolve(&name, autoload) {
        Ok(entry) => entry,
        Err(err) => return pool_error_response(err),
    };
    if autoload && let Err(err) = state.pool.ensure_ready(&name, AUTOLOAD_WAIT).await {
        return pool_error_response(err);
    }

    let rebuilt = Request::from_parts(parts, Body::from(body_bytes));
    state.pool.dispatch(&entry, rebuilt, is_post).await
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

async fn router_cors_middleware(
    State(state): State<RouterServerState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    super::cors::apply_cors_policy(&state.config.cors_policy, request, next).await
}

/// Assemble the router-mode application.
pub fn create_router_app(state: RouterServerState) -> axum::Router {
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
        .with_state(state)
}

#[cfg(test)]
#[path = "router_server_tests.rs"]
mod router_server_tests;

/// Run the router server: discover models, build the pool, and serve the
/// b10621 router surface (issue #1438). Reached from
/// [`super::startup::start_server`] when `--models-dir` is set and no model
/// argument was given, exactly b10621's `is_router_server` condition.
pub async fn run_router_server(startup: super::ServerStartupConfig) -> anyhow::Result<()> {
    let models_dir = startup
        .router_models_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("router mode requires --models-dir"))?;
    let api_keys = super::resolve_api_keys(&startup.api_keys, &startup.api_key_files)?;
    if !api_keys.is_empty() {
        tracing::info!("API-key authentication enabled ({} keys)", api_keys.len());
    }
    let base_config = super::startup::build_server_config(&startup, api_keys);
    let pool = Arc::new(RouterPool::new(
        models_dir.clone(),
        base_config.clone(),
        startup.models_max,
        startup.models_autoload,
    )?);
    let discovered = pool.snapshot().len();
    tracing::info!(
        "Router mode: {} models discovered under {} (models_max {}, autoload {})",
        discovered,
        models_dir.display(),
        startup.models_max,
        startup.models_autoload,
    );
    let state = RouterServerState {
        pool,
        config: Arc::new(base_config),
    };
    let app = create_router_app(state);
    super::startup::serve_http(&startup, app).await
}
