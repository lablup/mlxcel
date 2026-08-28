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

//! Axum application configuration

use axum::{
    Router,
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{Method, Request},
    middleware::{self, Next},
    response::Response,
    routing::{MethodRouter, get, post},
};
use tower_http::trace::TraceLayer;

use super::AppState;
use super::auth;
use super::cors::cors_middleware;
use super::routes;

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
fn is_public_endpoint(path: &str) -> bool {
    matches!(path, "/" | "/health" | "/v1/health")
}

/// Maximum request body size for audio upload endpoints. Overrides the Axum
/// 2 MiB default because real audio uploads commonly exceed that threshold.
const AUDIO_MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

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
    // Start the resumable-stream GC once per session manager (#1444); a
    // completed session is retained for replay for a bounded TTL even when
    // no request ever touches the manager again.
    state.stream_sessions.ensure_gc_spawned();
    let api_prefix = state.config.api_prefix.clone();
    let routes = build_routes(&state);
    let routes = if api_prefix.is_empty() {
        routes
    } else {
        Router::new().nest(&api_prefix, routes)
    };

    let gcp_enabled = state.config.gcp.is_some();
    let dispatch_cell = state.gcp_dispatch.clone();
    let app = routes
        // Middleware, innermost first.
        .layer(middleware::from_fn_with_state(state.clone(), api_key_auth))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            cors_middleware,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
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
pub(crate) fn route_inventory(_config: &crate::server::ServerConfig) -> Vec<RouteRegistration> {
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

#[cfg(test)]
#[path = "api_prefix_tests.rs"]
mod api_prefix_tests;

#[cfg(test)]
#[path = "auth_route_tests.rs"]
mod auth_route_tests;

#[cfg(test)]
mod tests {
    use super::{AUDIO_MAX_UPLOAD_BYTES, is_public_endpoint};
    use axum::{
        Router,
        body::Body,
        extract::DefaultBodyLimit,
        http::{Method, Request, StatusCode},
        routing::post,
    };
    use tower::ServiceExt;

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
