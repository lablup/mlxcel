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

//! Axum application configuration

use axum::{
    Router,
    body::Body,
    extract::{DefaultBodyLimit, Path as AxumPath, Query, State},
    http::{HeaderMap, Method, Request, StatusCode, Uri},
    middleware::{self, Next},
    response::Json,
    response::{IntoResponse, Response},
    routing::{MethodRouter, get, post},
};
use tower_http::trace::TraceLayer;

use super::AppState;
use super::auth;
use super::cors::cors_middleware;
use super::routes;
use super::types::ErrorResponse;

/// API-key authentication middleware, following b10621's
/// `middleware_validate_api_key` (#1437).
///
/// An empty key set disables authentication. Otherwise the request must
/// present one of the configured keys, unless its path is public. A missing
/// credential and an unknown one produce the same 401 and the same body, so a
/// probe cannot tell them apart, and no configured key is echoed.
///
/// This runs INSIDE the CORS middleware, so an `OPTIONS` preflight is answered
/// before it: browsers do not send `Authorization` on a preflight.
async fn api_key_auth(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if state.config.api_keys.is_empty() {
        return next.run(request).await;
    }

    if is_public_endpoint(request.uri().path()) {
        return next.run(request).await;
    }

    match auth::presented_credential(request.headers()) {
        Some(presented) if state.config.api_keys.accepts(presented) => next.run(request).await,
        _ => auth::unauthorized_response(),
    }
}

/// Paths served without authentication when API keys are configured.
///
/// This is b10621's `get_public_endpoints` set: `/health`, `/v1/health`, and
/// the Web UI front-end paths, of which mlxcel has only `/` (it ships no
/// embedded UI assets, so upstream's per-asset entries have no counterpart).
/// Everything else, `/props`, `/slots`, `/metrics`, `/models`, `/v1/models`
/// and every inference route included, requires a key on both servers.
///
/// The comparison is against the request path as the client sent it, prefix
/// included, which is what upstream's `req.path` carries. With `--api-prefix`
/// set, `<prefix>/health` is therefore NOT public on either server; startup
/// warns when both are configured (#1432).
pub(crate) fn is_public_endpoint(path: &str) -> bool {
    matches!(path, "/" | "/health" | "/v1/health")
}

/// Maximum request body size for audio upload endpoints. Overrides the Axum
/// 2 MiB default because real audio uploads commonly exceed that threshold.
const AUDIO_MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

/// Axum's default extractor body limit. Kept explicit so tests can assert that
/// zero-image configurations do not accidentally lower ordinary JSON capacity.
const AXUM_DEFAULT_BODY_LIMIT_BYTES: usize = 2_097_152;

/// Fixed slack for the non-image JSON fields that share the request body.
const MAIN_JSON_BODY_LIMIT_FIXED_OVERHEAD_BYTES: usize = 1024 * 1024;

/// Per-image slack for data-URL prefixes and surrounding JSON syntax.
const MAIN_JSON_BODY_LIMIT_PER_IMAGE_OVERHEAD_BYTES: usize = 512;

/// Hard cap for the buffered JSON extractor. The image knobs can be set to
/// extreme `usize` values by tests or misconfiguration; clamp those instead of
/// turning the JSON extractor into an effectively unbounded allocation target.
const MAIN_JSON_BODY_LIMIT_MAX_BYTES: usize = 2 * 1024 * 1024 * 1024;

fn checked_base64_encoded_len(decoded_len: usize) -> Option<usize> {
    decoded_len.checked_add(2)?.checked_div(3)?.checked_mul(4)
}

fn configured_image_json_body_budget_bytes(max_payload_bytes: usize, max_images: usize) -> usize {
    if max_payload_bytes == 0 || max_images == 0 {
        return 0;
    }

    let Some(encoded_per_image) = checked_base64_encoded_len(max_payload_bytes) else {
        return usize::MAX;
    };
    let Some(encoded_images) = encoded_per_image.checked_mul(max_images) else {
        return usize::MAX;
    };
    let Some(per_image_overhead) =
        MAIN_JSON_BODY_LIMIT_PER_IMAGE_OVERHEAD_BYTES.checked_mul(max_images)
    else {
        return usize::MAX;
    };
    let Some(with_fixed_overhead) =
        encoded_images.checked_add(MAIN_JSON_BODY_LIMIT_FIXED_OVERHEAD_BYTES)
    else {
        return usize::MAX;
    };
    with_fixed_overhead.saturating_add(per_image_overhead)
}

fn main_json_body_limit_bytes_for_limits(limits: super::media::ImageInputLimits) -> usize {
    configured_image_json_body_budget_bytes(limits.max_payload_bytes, limits.max_images_per_request)
        .clamp(
            AXUM_DEFAULT_BODY_LIMIT_BYTES,
            MAIN_JSON_BODY_LIMIT_MAX_BYTES,
        )
}

pub(crate) fn main_json_body_limit_bytes() -> usize {
    main_json_body_limit_bytes_for_limits(super::media::current_image_input_limits())
}

async fn openai_payload_too_large(request: Request<Body>, next: Next) -> Response {
    let response = next.run(request).await;
    if response.status() != axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        return response;
    }

    let mut error = ErrorResponse::new("Request body too large.", "invalid_request_error");
    error.status = axum::http::StatusCode::PAYLOAD_TOO_LARGE;
    error.into_response()
}

/// Create the Axum application router.
///
/// Layer order matters and mirrors b10621's pre-routing handler (#1432): the
/// CORS middleware sits OUTSIDE the API-key middleware, so a browser preflight
/// is answered without credentials, and both sit outside the route table.
///
/// `--api-prefix` nests the whole route set under the configured path. The
/// authentication middleware stays outside the nest so it sees the request
/// path as the client sent it, which is what upstream's `req.path` carries.
pub fn create_app(state: AppState) -> Router {
    create_app_impl(state, true)
}

/// [`create_app`] without the CORS layer, for router-mode sub-apps
/// (issue #1438): the router's top level answers preflights and stamps the
/// CORS headers exactly once, so a dispatched sub-app must not add a second
/// `Access-Control-Allow-Origin` to the same response.
pub fn create_app_without_cors(state: AppState) -> Router {
    create_app_impl(state, false)
}

#[cfg(feature = "webui")]
pub(crate) fn create_app_with_secured_ui(
    state: AppState,
    policy: super::webui::security::WebUiSecurityPolicy,
) -> Router {
    let api_keys = state.config.api_keys.clone();
    let api_prefix = state.config.api_prefix.clone();
    let ui_api = single_webui_api_routes().layer(DefaultBodyLimit::max(
        super::webui::api::WEBUI_JSON_BODY_BYTES as usize,
    ));
    let ui_api = if api_prefix.is_empty() {
        ui_api
    } else {
        Router::new().nest(&api_prefix, ui_api)
    };
    let static_routes = if api_prefix.is_empty() {
        super::webui::assets::router()
    } else {
        Router::new().nest(&api_prefix, super::webui::assets::router())
    };
    super::webui::security::secure_webui_router(
        create_app_impl(state.clone(), true)
            .merge(ui_api.with_state(state.clone()))
            .merge(static_routes.with_state(state.clone())),
        api_keys,
        policy,
    )
}

fn create_app_impl(state: AppState, with_cors: bool) -> Router {
    // Start the resumable-stream GC once per session manager (#1444); a
    // completed session is retained for replay for a bounded TTL even when
    // no request ever touches the manager again.
    state.stream_sessions.ensure_gc_spawned();
    let api_prefix = state.config.api_prefix.clone();
    let routes = build_routes(&state).layer(DefaultBodyLimit::max(main_json_body_limit_bytes()));
    let routes = if api_prefix.is_empty() {
        routes
    } else {
        Router::new().nest(&api_prefix, routes)
    };

    let gcp_enabled = state.config.gcp.is_some();
    let dispatch_cell = state.gcp_dispatch.clone();
    // Middleware, innermost first.
    let routes = routes.layer(middleware::from_fn(openai_payload_too_large));
    let routes = routes.layer(middleware::from_fn_with_state(state.clone(), api_key_auth));
    let routes = if with_cors {
        routes.layer(middleware::from_fn_with_state(
            state.clone(),
            cors_middleware,
        ))
    } else {
        routes
    };
    let app = routes.layer(TraceLayer::new_for_http()).with_state(state);
    // Vertex AI predict adapter (#1456): hand the predict handler the same
    // composed router the socket serves, so per-instance dispatch runs the
    // full middleware stack (auth, CORS, tracing) in-process.
    if gcp_enabled {
        let _ = dispatch_cell.set(app.clone());
    }
    app
}

/// One entry of the route inventory: the path, the `MethodRouter` mounted
/// there, and the method the Vertex AI predict adapter dispatches with
/// (#1456). The inventory is the single source [`build_routes`] mounts from
/// and [`crate::server::gcp_compat::dispatch_table`] derives the camelCase
/// alias table from, so a route added here is automatically served, aliased,
/// and considered by the `AIP_PREDICT_ROUTE` collision check. Registration
/// order is alias priority: the first path producing a given alias wins,
/// which is why the OpenAI `/v1` routes come first.
pub(crate) struct RouteRegistration {
    pub(crate) path: &'static str,
    /// Method the predict adapter uses when dispatching to this path (the
    /// registered method; POST preferred when a path serves several).
    pub(crate) dispatch_method: Method,
    pub(crate) handler: MethodRouter<AppState>,
}

fn reg(
    path: &'static str,
    dispatch_method: Method,
    handler: MethodRouter<AppState>,
) -> RouteRegistration {
    RouteRegistration {
        path,
        dispatch_method,
        handler,
    }
}

/// The full route table, in registration (= alias priority) order.
///
/// Keep every `.route()` of the server in here: [`build_routes`] mounts
/// exactly this list, so a route bypassing the inventory would not exist.
pub(crate) fn route_inventory(config: &crate::server::ServerConfig) -> Vec<RouteRegistration> {
    let audio_limit = DefaultBodyLimit::max(AUDIO_MAX_UPLOAD_BYTES);
    let mut inventory = vec![
        // OpenAI API endpoints
        reg(
            "/v1/chat/completions",
            Method::POST,
            post(routes::chat_completions),
        ),
        // b10621 realtime control of a live completion (#1444).
        reg(
            "/v1/chat/completions/control",
            Method::POST,
            post(routes::chat_completions_control),
        ),
        // b10621 resumable-stream lifecycle (#1444): replay, discovery, and
        // stop for streaming completions that carried `X-Conversation-Id`.
        reg(
            "/v1/stream",
            Method::GET,
            get(routes::stream_get).delete(routes::stream_delete),
        ),
        reg(
            "/v1/streams/lookup",
            Method::POST,
            post(routes::streams_lookup),
        ),
        reg("/v1/completions", Method::POST, post(routes::completions)),
        reg("/v1/models", Method::GET, get(routes::list_models)),
        // Embeddings (OpenAI /v1/embeddings surface), served by the embedding
        // worker when one is loaded; a structured 501 otherwise.
        reg(
            "/v1/embeddings",
            Method::POST,
            post(routes::create_embeddings),
        ),
        // Reranking (Cohere / Jina compatible surface), served by the rerank
        // worker when one is loaded; a structured 501 otherwise.
        reg("/v1/rerank", Method::POST, post(routes::create_rerank)),
        reg("/v1/reranking", Method::POST, post(routes::create_rerank)),
        // Responses API (OpenAI /v1/responses surface).
        reg("/v1/responses", Method::POST, post(routes::create_response)),
        reg(
            "/v1/responses/:id",
            Method::GET,
            get(routes::retrieve_response).delete(routes::delete_response),
        ),
        reg(
            "/v1/responses/:id/cancel",
            Method::POST,
            post(routes::cancel_response),
        ),
        // Anthropic Messages API (/v1/messages surface).
        reg(
            "/v1/messages",
            Method::POST,
            post(routes::anthropic_messages),
        ),
        reg(
            "/v1/messages/count_tokens",
            Method::POST,
            post(routes::anthropic_count_tokens),
        ),
        // prompt-cache observability endpoints (always mounted; the handlers
        // return a stable "disabled" payload when the cache is off so
        // monitoring clients can poll without conditional logic).
        reg("/v1/cache/stats", Method::GET, get(routes::cache_stats)),
        reg("/v1/cache/reset", Method::POST, post(routes::cache_reset)),
        // Adaptive B=1 MTP policy state (issue #1257). Always mounted, and
        // returns a well-formed "unavailable" payload when no policy is
        // running: a consumer must be able to tell "nothing to report" from
        // "this server does not answer". It is the supported replacement for
        // reading the private hint files under the mlxcel cache root.
        reg(
            "/v1/internal/mtp-policy",
            Method::GET,
            get(routes::mtp_policy),
        ),
        // Audio endpoints carry a larger per-route body limit because real
        // audio uploads commonly exceed the Axum 2 MiB default.
        reg(
            "/v1/audio/speech",
            Method::POST,
            post(routes::audio_speech).layer(audio_limit),
        ),
        reg(
            "/v1/audio/transcriptions",
            Method::POST,
            post(routes::audio_transcriptions).layer(audio_limit),
        ),
        reg(
            "/v1/audio/translations",
            Method::POST,
            post(routes::audio_translations).layer(audio_limit),
        ),
        reg(
            "/audio/speech",
            Method::POST,
            post(routes::audio_speech).layer(audio_limit),
        ),
        reg(
            "/audio/transcriptions",
            Method::POST,
            post(routes::audio_transcriptions).layer(audio_limit),
        ),
        reg(
            "/audio/translations",
            Method::POST,
            post(routes::audio_translations).layer(audio_limit),
        ),
        // Aliases (some clients use these)
        reg(
            "/chat/completions",
            Method::POST,
            post(routes::chat_completions),
        ),
        // BREAKING (#1441): `/completions` and `/embeddings` are llama-server
        // NATIVE routes, not OpenAI aliases. b10621 sends `/completion` and
        // `/completions` to one handler and `/v1/completions` to a different
        // one, and does the same for `/embedding` / `/embeddings` against
        // `/v1/embeddings`. mlxcel used to answer the OpenAI shape on all of
        // them, so a llama-server client reading the native schema got an
        // object it could not parse.
        reg(
            "/completions",
            Method::POST,
            post(routes::native_completion),
        ),
        reg("/models", Method::GET, get(routes::list_models)),
        reg("/embedding", Method::POST, post(routes::native_embeddings)),
        reg("/embeddings", Method::POST, post(routes::native_embeddings)),
        reg("/rerank", Method::POST, post(routes::create_rerank)),
        reg("/reranking", Method::POST, post(routes::create_rerank)),
        reg("/responses", Method::POST, post(routes::create_response)),
        reg(
            "/responses/:id",
            Method::GET,
            get(routes::retrieve_response).delete(routes::delete_response),
        ),
        reg(
            "/responses/:id/cancel",
            Method::POST,
            post(routes::cancel_response),
        ),
        reg("/messages", Method::POST, post(routes::anthropic_messages)),
        reg(
            "/messages/count_tokens",
            Method::POST,
            post(routes::anthropic_count_tokens),
        ),
        // llama-server compatible endpoints
        reg("/completion", Method::POST, post(routes::native_completion)),
        reg("/tokenize", Method::POST, post(routes::tokenize)),
        reg("/detokenize", Method::POST, post(routes::detokenize)),
        // Fill-in-the-middle. Mounted unconditionally, like every other route:
        // whether the loaded model can serve it is a property of its
        // vocabulary, and the handler answers 501 naming the missing FIM
        // tokens rather than 404, so a client can tell "this server does not
        // implement infill" from "this model cannot do it" (#1442).
        reg("/infill", Method::POST, post(routes::infill)),
        // Prompt inspection: render or count a prompt without generating from
        // it (#1442).
        reg(
            "/apply-template",
            Method::POST,
            post(routes::apply_template),
        ),
        reg(
            "/chat/completions/input_tokens",
            Method::POST,
            post(routes::chat_input_tokens),
        ),
        reg(
            "/v1/chat/completions/input_tokens",
            Method::POST,
            post(routes::chat_input_tokens),
        ),
        reg(
            "/responses/input_tokens",
            Method::POST,
            post(routes::responses_input_tokens),
        ),
        reg(
            "/v1/responses/input_tokens",
            Method::POST,
            post(routes::responses_input_tokens),
        ),
        // b10621 disabled-feature stubs (#1435): server tools, MCP, and the
        // UI's CORS proxy are never implemented here (the enabling flags
        // fail startup in `cli::ui_compat_args`), so these four routes
        // always answer upstream's 403 `feature_disabled` envelope, exactly
        // as a llama-server with the features off does.
        reg(
            "/tools",
            Method::POST,
            get(routes::feature_disabled).post(routes::feature_disabled),
        ),
        reg(
            "/cors-proxy",
            Method::POST,
            get(routes::feature_disabled).post(routes::feature_disabled),
        ),
    ];

    // Opt-in management API (#1312). It remains inside the ordinary API-key
    // middleware and is absent, rather than returning a disabled stub, when
    // the operator did not enable it.
    if config.enable_settings_endpoint {
        inventory.push(reg(
            "/v1/settings",
            Method::PATCH,
            get(routes::get_settings).patch(routes::patch_settings),
        ));
        inventory.push(reg(
            "/settings",
            Method::PATCH,
            get(routes::get_settings).patch(routes::patch_settings),
        ));
    }

    // b10621 mounts /props, /slots, /metrics and the slot actions
    // unconditionally and answers its own diagnostics when a gate is off
    // (issue #1440): GET /props is ungated, --props gates POST /props,
    // --slots gates GET /slots, --metrics gates GET /metrics, and
    // --slot-save-path gates POST /slots/:id_slot. The handlers own those
    // gates so a disabled surface answers upstream's 501 instead of a 404.
    inventory.push(reg(
        "/props",
        Method::POST,
        get(routes::props).post(routes::post_props),
    ));
    inventory.push(reg("/slots", Method::GET, get(routes::slots)));
    inventory.push(reg(
        "/slots/:id_slot",
        Method::POST,
        post(routes::slot_action),
    ));
    inventory.push(reg("/metrics", Method::GET, get(routes::metrics)));
    // b10621 LoRA adapter inventory and hot-swap surface (issue #1439).
    inventory.push(reg(
        "/lora-adapters",
        Method::POST,
        get(routes::get_lora_adapters).post(routes::post_lora_adapters),
    ));

    // Health check
    inventory.push(reg("/health", Method::GET, get(routes::health_check)));
    inventory.push(reg("/v1/health", Method::GET, get(routes::health_check)));
    inventory.push(reg("/", Method::GET, get(routes::health_check)));
    inventory
}

/// Register every route, without middleware and without the API prefix.
fn build_routes(state: &AppState) -> Router<AppState> {
    let mut app = Router::new();
    let mut mounted: Vec<&'static str> = Vec::new();
    for entry in route_inventory(&state.config) {
        mounted.push(entry.path);
        app = app.route(entry.path, entry.handler);
    }

    // Vertex AI (GCP) compat routes (#1456): resolved once at startup from
    // the AIP_* variables into `config.gcp`; `None` (the default) mounts
    // nothing. Registered inside the middleware stack, so the predict route
    // and the health alias require an API key exactly as upstream's do
    // (neither is in b10621's public endpoint set).
    if let Some(gcp) = state.config.gcp.as_ref() {
        if let Some(health_alias) = gcp
            .path_health
            .as_deref()
            .filter(|alias| !mounted.contains(alias))
        {
            // A health alias naming an already-registered path is skipped:
            // upstream registers a duplicate httplib handler that never
            // matches, so the observable behavior (the existing route
            // answers) is the same.
            app = app.route(health_alias, get(routes::health_check));
        }
        // Startup refused a colliding AIP_PREDICT_ROUTE before the model
        // load (`gcp_compat::check_predict_collision`); a collision here
        // would panic in axum's route registration.
        app = app.route(&gcp.path_predict, post(crate::server::gcp_compat::predict));
    }

    app
}

#[cfg(feature = "webui")]
fn single_webui_api_routes() -> Router<AppState> {
    Router::new()
        .route("/ui-api/v1/bootstrap", get(single_ui_bootstrap))
        .route("/ui-api/v1/catalog", get(single_ui_catalog_list))
        .route("/ui-api/v1/catalog/:id", get(single_ui_catalog_get))
        .route("/ui-api/v1/runtime", get(single_ui_runtime))
        .route("/ui-api/v1/operations", get(single_ui_operations_list))
        .route("/ui-api/v1/events", get(single_ui_events))
}

#[cfg(feature = "webui")]
fn single_webui_error(
    status: StatusCode,
    code: &str,
    message: impl Into<String>,
    retryable: bool,
) -> Response {
    (
        status,
        Json(super::router_lifecycle::ErrorEnvelope {
            error: super::router_lifecycle::ErrorBody {
                code: code.to_string(),
                message: message.into(),
                retryable,
                field_errors: None,
                operation_id: None,
            },
            request_id: format!("req_{}", chrono::Utc::now().timestamp_micros()),
        }),
    )
        .into_response()
}

#[cfg(feature = "webui")]
fn single_invalid_field(field: &'static str, message: &'static str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(super::router_lifecycle::ErrorEnvelope {
            error: super::router_lifecycle::ErrorBody {
                code: "invalid_request".to_string(),
                message: "request does not match the WebUI contract".to_string(),
                retryable: true,
                field_errors: Some(vec![super::router_lifecycle::FieldError {
                    field: field.to_string(),
                    code: "invalid_format".to_string(),
                    message: message.to_string(),
                }]),
                operation_id: None,
            },
            request_id: format!("req_{}", chrono::Utc::now().timestamp_micros()),
        }),
    )
        .into_response()
}

#[cfg(feature = "webui")]
fn valid_model_id(value: &str) -> bool {
    let Some(suffix) = value.strip_prefix("mdl_") else {
        return false;
    };
    suffix.len() == 43
        && suffix
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

#[cfg(feature = "webui")]
async fn single_catalog_entry_blocking(
    state: AppState,
) -> Result<super::webui::catalog::CatalogEntry, tokio::task::JoinError> {
    let cache = state.webui_catalog_cache.clone();
    tokio::task::spawn_blocking(move || {
        super::webui::catalog::single_model_entry_from_state_with_cache(&cache, &state)
    })
    .await
}

#[cfg(feature = "webui")]
async fn single_ui_bootstrap(State(state): State<AppState>) -> Response {
    Json(super::webui::api::bootstrap_response(
        &state.webui_startup,
        &state.config,
        state.webui_lifecycle.server_instance_id().to_string(),
        super::webui::api::WebUiServerMode::SingleModel,
        false,
    ))
    .into_response()
}

#[cfg(feature = "webui")]
async fn single_ui_catalog_list(
    State(state): State<AppState>,
    Query(query): Query<super::webui::catalog::CatalogQuery>,
) -> Response {
    let limit = query
        .limit
        .unwrap_or(super::webui::api::CATALOG_DEFAULT_PAGE_SIZE as usize);
    if !(1..=super::webui::api::CATALOG_MAX_PAGE_SIZE as usize).contains(&limit) {
        return single_invalid_field("limit", "limit must be between 1 and 200");
    }
    if query
        .cursor
        .as_ref()
        .is_some_and(|cursor| cursor.len() > super::webui::api::CURSOR_BYTES as usize)
    {
        return single_invalid_field("cursor", "cursor is too long");
    }
    let server_instance_id = state.webui_lifecycle.server_instance_id().to_string();
    let snapshot_sequence = state.webui_lifecycle.snapshot_sequence();
    match single_catalog_entry_blocking(state).await {
        Ok(entry) => {
            let items = if query.cursor.is_some() {
                Vec::new()
            } else {
                vec![entry]
            };
            Json(super::webui::catalog::CatalogListResponse {
                schema_version: super::router_lifecycle::SCHEMA_VERSION.to_string(),
                pagination: super::webui::catalog::Pagination {
                    limit,
                    next_cursor: None,
                    total_known: Some(items.len()),
                },
                items,
                server_instance_id,
                snapshot_sequence,
            })
            .into_response()
        }
        Err(err) => single_webui_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            format!("catalog projection task failed: {err}"),
            true,
        ),
    }
}

#[cfg(feature = "webui")]
async fn single_ui_catalog_get(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if !valid_model_id(&id) {
        return single_invalid_field("id", "model_id must match ^mdl_[A-Za-z0-9_-]{43}$");
    }
    match single_catalog_entry_blocking(state).await {
        Ok(entry) if entry.identity.id == id => Json(entry).into_response(),
        Ok(_) => single_webui_error(
            StatusCode::NOT_FOUND,
            "not_found",
            "catalog entry not found",
            true,
        ),
        Err(err) => single_webui_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            format!("catalog projection task failed: {err}"),
            true,
        ),
    }
}

#[cfg(feature = "webui")]
#[derive(Default, serde::Deserialize)]
struct SingleRuntimeQuery {
    model_id: Option<String>,
    autoload: Option<bool>,
}

#[cfg(feature = "webui")]
async fn single_ui_runtime(
    State(state): State<AppState>,
    Query(query): Query<SingleRuntimeQuery>,
) -> Response {
    if query.autoload.unwrap_or(false) {
        return single_invalid_field("autoload", "runtime observations must use autoload=false");
    }
    let Some(model_id) = query.model_id else {
        return single_invalid_field("model_id", "model_id is required");
    };
    if !valid_model_id(&model_id) {
        return single_invalid_field("model_id", "model_id must match ^mdl_[A-Za-z0-9_-]{43}$");
    }
    match single_catalog_entry_blocking(state.clone()).await {
        Ok(entry) if entry.identity.id == model_id => Json(super::webui::api::runtime_snapshot(
            state.webui_lifecycle.server_instance_id().to_string(),
            model_id,
            entry.identity.revision,
            state.webui_lifecycle.snapshot_sequence(),
            &state.config,
        ))
        .into_response(),
        Ok(_) => single_webui_error(StatusCode::NOT_FOUND, "not_found", "model not found", true),
        Err(err) => single_webui_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            format!("runtime projection task failed: {err}"),
            true,
        ),
    }
}

#[cfg(feature = "webui")]
#[derive(Default, serde::Deserialize)]
struct SingleOperationsQuery {
    limit: Option<usize>,
    cursor: Option<String>,
}

#[cfg(feature = "webui")]
async fn single_ui_operations_list(
    State(state): State<AppState>,
    Query(query): Query<SingleOperationsQuery>,
) -> Response {
    let limit = query.limit.unwrap_or(50);
    if !(1..=200).contains(&limit) {
        return single_invalid_field("limit", "limit must be between 1 and 200");
    }
    Json(
        state
            .webui_lifecycle
            .list_operations(limit, query.cursor.as_deref(), None, None, None),
    )
    .into_response()
}

#[cfg(feature = "webui")]
async fn single_ui_events(State(state): State<AppState>, headers: HeaderMap, uri: Uri) -> Response {
    let runtime_model_ids = match single_catalog_entry_blocking(state.clone()).await {
        Ok(entry) => vec![entry.identity.id],
        Err(err) => {
            return single_webui_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable",
                format!("event model identity projection task failed: {err}"),
                true,
            );
        }
    };
    super::webui::events::ui_events_response(
        state.webui_lifecycle.clone(),
        runtime_model_ids,
        headers,
        uri,
    )
}

#[cfg(test)]
#[path = "api_prefix_tests.rs"]
mod api_prefix_tests;

#[cfg(test)]
#[path = "auth_route_tests.rs"]
mod auth_route_tests;

#[cfg(test)]
mod tests {
    use super::{
        AUDIO_MAX_UPLOAD_BYTES, AXUM_DEFAULT_BODY_LIMIT_BYTES,
        MAIN_JSON_BODY_LIMIT_FIXED_OVERHEAD_BYTES, MAIN_JSON_BODY_LIMIT_MAX_BYTES,
        MAIN_JSON_BODY_LIMIT_PER_IMAGE_OVERHEAD_BYTES, configured_image_json_body_budget_bytes,
        is_public_endpoint, main_json_body_limit_bytes_for_limits, openai_payload_too_large,
    };
    use axum::{
        Json, Router,
        body::{Body, Bytes, to_bytes},
        extract::DefaultBodyLimit,
        http::{Method, Request, StatusCode},
        middleware,
        routing::post,
    };
    use base64::Engine;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    fn body_limit_test_router(limit: usize) -> Router {
        Router::new()
            .route(
                "/json",
                post(|Json(_): Json<Value>| async move { StatusCode::NO_CONTENT }),
            )
            .route(
                "/bytes",
                post(|_body: Bytes| async move { StatusCode::NO_CONTENT }),
            )
            .layer(DefaultBodyLimit::max(limit))
            .layer(middleware::from_fn(openai_payload_too_large))
    }

    fn encoded_image_body(decoded_sizes: &[usize]) -> Vec<u8> {
        let content: Vec<Value> = std::iter::once(json!({"type": "text", "text": "look"}))
            .chain(decoded_sizes.iter().map(|size| {
                let encoded = base64::engine::general_purpose::STANDARD.encode(vec![0u8; *size]);
                json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:image/png;base64,{encoded}")}
                })
            }))
            .collect();
        serde_json::to_vec(&json!({
            "model": "m",
            "messages": [{"role": "user", "content": content}]
        }))
        .expect("serialize image request")
    }

    async fn post_body(path: &str, body: Vec<u8>, limit: usize) -> axum::response::Response {
        body_limit_test_router(limit)
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn response_json(response: axum::response::Response) -> Value {
        let bytes = to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read response body");
        serde_json::from_slice(&bytes).expect("json response")
    }

    /// Build a minimal audio sub-router using stub handlers and the same
    /// `DefaultBodyLimit` layer applied in `create_app`. Tests can call this
    /// without constructing a real `AppState`.
    fn audio_test_router() -> Router {
        Router::new()
            .route(
                "/v1/audio/speech",
                post(|| async { StatusCode::NO_CONTENT }),
            )
            .route(
                "/v1/audio/transcriptions",
                post(|| async { StatusCode::NO_CONTENT }),
            )
            .route(
                "/v1/audio/translations",
                post(|| async { StatusCode::NO_CONTENT }),
            )
            .route("/audio/speech", post(|| async { StatusCode::NO_CONTENT }))
            .route(
                "/audio/transcriptions",
                post(|| async { StatusCode::NO_CONTENT }),
            )
            .route(
                "/audio/translations",
                post(|| async { StatusCode::NO_CONTENT }),
            )
            .layer(DefaultBodyLimit::max(AUDIO_MAX_UPLOAD_BYTES))
    }

    #[cfg(feature = "webui")]
    fn single_webui_state() -> crate::server::AppState {
        use std::path::PathBuf;
        use std::sync::{Arc, mpsc};

        let (options_tx, _options_rx) = mpsc::channel();
        let provider = Arc::new(crate::server::ModelProvider::recording_for_route_tests(
            options_tx,
        ));
        let batch_metrics = provider.batch_metrics().clone();
        crate::server::AppState::new(
            provider,
            crate::server::ServerConfig::default(),
            crate::server::ChatTemplateProcessor::with_template("ok".to_string()),
            crate::tokenizer::MlxcelTokenizer::stub(),
            PathBuf::from("single-webui-route-test-model"),
            batch_metrics,
        )
    }

    #[cfg(feature = "webui")]
    fn assert_single_model_revision_event(chunk: &str, server_instance: &str, model_id: &str) {
        let data = chunk
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("SSE JSON data");
        let event_id = chunk
            .lines()
            .find_map(|line| line.strip_prefix("id: "))
            .expect("SSE id");
        let mut actual: Value = serde_json::from_str(data).expect("producer event JSON");
        assert_eq!(actual["server_instance_id"], server_instance);
        assert_eq!(actual["type"], "model_revision");
        assert_eq!(actual["payload"]["model_id"], model_id);
        assert_eq!(actual["event_id"], event_id);
        assert!(super::valid_model_id(model_id));
        assert!(event_id.starts_with("evt_"));
        assert!(
            actual["payload"]["revision"]
                .as_u64()
                .is_some_and(|n| n > 0)
        );
        assert!(actual["sequence"].as_u64().is_some());
        assert!(
            actual["emitted_at"]
                .as_str()
                .is_some_and(|s| chrono::DateTime::parse_from_rfc3339(s).is_ok())
        );
        let mut expected: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/webui/examples/event.1.json"
        ))
        .expect("event fixture");
        for path in [
            "/server_instance_id",
            "/event_id",
            "/sequence",
            "/emitted_at",
            "/payload/model_id",
            "/payload/revision",
        ] {
            *actual.pointer_mut(path).expect("actual dynamic") =
                expected.pointer(path).expect("expected dynamic").clone();
        }
        expected
            .as_object_mut()
            .expect("fixture object")
            .remove("$schemaName");
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "webui")]
    async fn first_single_sse_chunk(app: Router, uri: &str, last_event_id: Option<&str>) -> String {
        use axum::http::header;
        use futures::StreamExt;

        let mut builder = Request::builder()
            .method(Method::GET)
            .uri(uri)
            .header(header::ACCEPT, "text/event-stream");
        if let Some(event_id) = last_event_id {
            builder = builder.header("Last-Event-ID", event_id);
        }
        let response = app
            .oneshot(builder.body(Body::empty()).expect("request"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let mut stream = response.into_body().into_data_stream();
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
            .await
            .expect("sse chunk timeout")
            .expect("sse chunk")
            .expect("sse body ok");
        String::from_utf8(chunk.to_vec()).expect("utf8 sse")
    }

    #[cfg(feature = "webui")]
    async fn single_events_status(
        app: Router,
        uri: &str,
        last_event_id: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut builder = Request::builder().method(Method::GET).uri(uri);
        if let Some(event_id) = last_event_id {
            builder = builder.header("Last-Event-ID", event_id);
        }
        let response = app
            .oneshot(builder.body(Body::empty()).expect("request"))
            .await
            .expect("response");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20).await.expect("body");
        let json = serde_json::from_slice(&bytes).expect("json body");
        (status, json)
    }

    #[test]
    fn audio_upload_limit_is_25_mib() {
        assert_eq!(
            AUDIO_MAX_UPLOAD_BYTES,
            25 * 1024 * 1024,
            "audio upload limit must be 25 MiB"
        );
    }

    #[test]
    fn all_public_endpoints_are_unauthenticated() {
        for path in ["/", "/health", "/v1/health"] {
            assert!(is_public_endpoint(path), "{path}");
        }
        assert!(!is_public_endpoint("/v1/models"));
    }

    #[test]
    fn main_json_limit_derives_from_simultaneous_image_budget() {
        assert_eq!(
            configured_image_json_body_budget_bytes(3, 2),
            MAIN_JSON_BODY_LIMIT_FIXED_OVERHEAD_BYTES
                + (2 * MAIN_JSON_BODY_LIMIT_PER_IMAGE_OVERHEAD_BYTES)
                + 8
        );

        let mut limits = crate::server::media::ImageInputLimits {
            max_payload_bytes: 3,
            max_images_per_request: 2,
            ..crate::server::media::ImageInputLimits::default()
        };
        assert_eq!(
            main_json_body_limit_bytes_for_limits(limits),
            AXUM_DEFAULT_BODY_LIMIT_BYTES
        );

        limits.max_payload_bytes = 2 * 1024 * 1024;
        limits.max_images_per_request = 2;
        assert!(main_json_body_limit_bytes_for_limits(limits) > AXUM_DEFAULT_BODY_LIMIT_BYTES);
    }

    #[test]
    fn main_json_limit_handles_zero_and_extreme_config() {
        let zero = crate::server::media::ImageInputLimits {
            max_payload_bytes: 0,
            max_images_per_request: 0,
            ..crate::server::media::ImageInputLimits::default()
        };
        assert_eq!(
            main_json_body_limit_bytes_for_limits(zero),
            AXUM_DEFAULT_BODY_LIMIT_BYTES
        );

        let extreme = crate::server::media::ImageInputLimits {
            max_payload_bytes: usize::MAX,
            max_images_per_request: usize::MAX,
            ..crate::server::media::ImageInputLimits::default()
        };
        assert_eq!(
            main_json_body_limit_bytes_for_limits(extreme),
            MAIN_JSON_BODY_LIMIT_MAX_BYTES
        );
    }

    #[tokio::test]
    async fn main_json_limit_accepts_base64_body_above_axum_default() {
        let limits = crate::server::media::ImageInputLimits {
            max_payload_bytes: 2 * 1024 * 1024,
            max_images_per_request: 1,
            ..crate::server::media::ImageInputLimits::default()
        };
        let limit = main_json_body_limit_bytes_for_limits(limits);
        let body = encoded_image_body(&[2 * 1024 * 1024]);
        assert!(body.len() > AXUM_DEFAULT_BODY_LIMIT_BYTES);
        assert!(body.len() <= limit);

        let response = post_body("/json", body, limit).await;
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn main_json_limit_accounts_for_simultaneous_image_inputs() {
        let limits = crate::server::media::ImageInputLimits {
            max_payload_bytes: 1024,
            max_images_per_request: 2,
            ..crate::server::media::ImageInputLimits::default()
        };
        let limit = main_json_body_limit_bytes_for_limits(limits);
        let body = encoded_image_body(&[1024, 1024]);
        assert!(body.len() <= limit);

        let response = post_body("/json", body, limit).await;
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn body_limit_accepts_exact_boundary_and_rejects_one_byte_over() {
        const TEST_LIMIT: usize = 64;
        let exact = post_body("/bytes", vec![b'x'; TEST_LIMIT], TEST_LIMIT).await;
        assert_eq!(exact.status(), StatusCode::NO_CONTENT);

        let oversized = post_body("/bytes", vec![b'x'; TEST_LIMIT + 1], TEST_LIMIT).await;
        assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn oversized_body_limit_errors_are_openai_shaped_for_audio_and_json() {
        const TEST_LIMIT: usize = 16;
        let json_response = post_body("/json", vec![b' '; TEST_LIMIT + 1], TEST_LIMIT).await;
        assert_eq!(json_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let json_body = response_json(json_response).await;
        assert_eq!(json_body["error"]["type"], "invalid_request_error");
        assert_eq!(json_body["error"]["message"], "Request body too large.");
        assert!(json_body["error"]["code"].is_null());

        let audio_app = Router::new()
            .route(
                "/v1/audio/transcriptions",
                post(|_body: Bytes| async move { StatusCode::NO_CONTENT }),
            )
            .layer(DefaultBodyLimit::max(TEST_LIMIT))
            .layer(middleware::from_fn(openai_payload_too_large));
        let audio_response = audio_app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header("content-type", "multipart/form-data; boundary=x")
                    .body(Body::from(vec![0u8; TEST_LIMIT + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(audio_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let audio_body = response_json(audio_response).await;
        assert_eq!(audio_body, json_body);
    }

    #[tokio::test]
    async fn audio_speech_is_reachable_at_v1_path() {
        let response = audio_test_router()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/speech")
                    .header("content-type", "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(
            response.status(),
            StatusCode::NOT_FOUND,
            "/v1/audio/speech must be mounted"
        );
    }

    #[tokio::test]
    async fn audio_transcriptions_is_reachable_at_v1_path() {
        let response = audio_test_router()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header("content-type", "multipart/form-data; boundary=x")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(
            response.status(),
            StatusCode::NOT_FOUND,
            "/v1/audio/transcriptions must be mounted"
        );
    }

    #[tokio::test]
    async fn audio_translations_is_reachable_at_v1_path() {
        let response = audio_test_router()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/translations")
                    .header("content-type", "multipart/form-data; boundary=x")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(
            response.status(),
            StatusCode::NOT_FOUND,
            "/v1/audio/translations must be mounted"
        );
    }

    #[tokio::test]
    async fn get_to_audio_speech_returns_method_not_allowed() {
        // The route exists but only accepts POST. A 405 (not 404) confirms the
        // path is registered.
        let response = audio_test_router()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/audio/speech")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn audio_alias_paths_are_reachable_without_v1_prefix() {
        for path in [
            "/audio/speech",
            "/audio/transcriptions",
            "/audio/translations",
        ] {
            let response = audio_test_router()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_ne!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{path} alias must be mounted"
            );
        }
    }

    #[cfg(feature = "webui")]
    #[tokio::test]
    async fn single_webui_events_use_shared_replay_cursor_and_fixtures() {
        let state = single_webui_state();
        let entry = crate::server::webui::catalog::single_model_entry_from_state_with_cache(
            &state.webui_catalog_cache,
            &state,
        );
        let server_instance = state.webui_lifecycle.server_instance_id().to_string();
        let lifecycle = crate::server::router_lifecycle::ModelLifecycle::new(
            crate::server::router_lifecycle::DownloadState::Complete,
        );
        lifecycle.mark_loading();
        state.webui_lifecycle.publish_model_revision(
            &entry.identity.id,
            entry.identity.revision,
            lifecycle.snapshot(),
        );
        let app = super::single_webui_api_routes().with_state(state);

        let replay = first_single_sse_chunk(
            app.clone(),
            &format!("/ui-api/v1/events?server_instance_id={server_instance}&after_sequence=0"),
            None,
        )
        .await;
        assert!(replay.contains("event: model_revision"), "{replay}");
        assert_single_model_revision_event(&replay, &server_instance, &entry.identity.id);

        let (status, response) = single_events_status(
            app,
            &format!("/ui-api/v1/events?server_instance_id={server_instance}&after_sequence=1"),
            Some("evt_conflict"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert_eq!(response["error"]["field_errors"][0]["code"], "conflict");
    }

    #[tokio::test]
    async fn body_limit_layer_enforces_upload_cap() {
        // Use a small limit so the test does not allocate the full 25 MiB. The
        // goal is confirming DefaultBodyLimit is wired onto the audio sub-router
        // and that an over-limit body produces 413; the constant test covers the
        // 25 MiB value separately.
        const TEST_LIMIT: usize = 16;
        let app = Router::new()
            .route(
                "/v1/audio/transcriptions",
                post(|_body: axum::body::Bytes| async move { StatusCode::NO_CONTENT }),
            )
            .layer(DefaultBodyLimit::max(TEST_LIMIT));

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/transcriptions")
                    .header("content-type", "multipart/form-data; boundary=x")
                    .body(Body::from(vec![0u8; TEST_LIMIT + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
