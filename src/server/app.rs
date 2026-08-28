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
    http::Request,
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
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

    routes
        // Middleware, innermost first.
        .layer(middleware::from_fn_with_state(state.clone(), api_key_auth))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            cors_middleware,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Register every route, without middleware and without the API prefix.
fn build_routes(state: &AppState) -> Router<AppState> {
    let enable_slots = state.config.enable_slots_endpoint;
    let enable_props = state.config.enable_props_endpoint;
    let enable_metrics = state.config.enable_metrics_endpoint;

    // Audio upload endpoints carry a larger body limit via a sub-router.
    // Merging keeps the outer auth, CORS, and trace layers applying normally.
    let audio_routes: Router<AppState> = Router::new()
        .route("/v1/audio/speech", post(routes::audio_speech))
        .route(
            "/v1/audio/transcriptions",
            post(routes::audio_transcriptions),
        )
        .route("/v1/audio/translations", post(routes::audio_translations))
        .route("/audio/speech", post(routes::audio_speech))
        .route("/audio/transcriptions", post(routes::audio_transcriptions))
        .route("/audio/translations", post(routes::audio_translations))
        .layer(DefaultBodyLimit::max(AUDIO_MAX_UPLOAD_BYTES));

    let mut app = Router::new()
        // OpenAI API endpoints
        .route("/v1/chat/completions", post(routes::chat_completions))
        // b10621 realtime control of a live completion (#1444).
        .route(
            "/v1/chat/completions/control",
            post(routes::chat_completions_control),
        )
        // b10621 resumable-stream lifecycle (#1444): replay, discovery, and
        // stop for streaming completions that carried `X-Conversation-Id`.
        .route(
            "/v1/stream",
            get(routes::stream_get).delete(routes::stream_delete),
        )
        .route("/v1/streams/lookup", post(routes::streams_lookup))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/models", get(routes::list_models))
        // Embeddings (OpenAI /v1/embeddings surface), served by the embedding
        // worker when one is loaded; a structured 501 otherwise.
        .route("/v1/embeddings", post(routes::create_embeddings))
        // Reranking (Cohere / Jina compatible surface), served by the rerank
        // worker when one is loaded; a structured 501 otherwise.
        .route("/v1/rerank", post(routes::create_rerank))
        .route("/v1/reranking", post(routes::create_rerank))
        // Responses API (OpenAI /v1/responses surface).
        .route("/v1/responses", post(routes::create_response))
        .route(
            "/v1/responses/:id",
            get(routes::retrieve_response).delete(routes::delete_response),
        )
        .route("/v1/responses/:id/cancel", post(routes::cancel_response))
        // Anthropic Messages API (/v1/messages surface).
        .route("/v1/messages", post(routes::anthropic_messages))
        .route(
            "/v1/messages/count_tokens",
            post(routes::anthropic_count_tokens),
        )
        // prompt-cache observability endpoints (always mounted
        // the handlers return a stable "disabled" payload when the cache is
        // off so monitoring clients can poll without conditional logic).
        .route("/v1/cache/stats", get(routes::cache_stats))
        .route("/v1/cache/reset", post(routes::cache_reset))
        // Adaptive B=1 MTP policy state (issue #1257). Always mounted, and
        // returns a well-formed "unavailable" payload when no policy is
        // running, for the same reason the cache endpoints do: a consumer must
        // be able to tell "nothing to report" from "this server does not
        // answer". It is the supported replacement for reading the private
        // hint files under the mlxcel cache root.
        .route("/v1/internal/mtp-policy", get(routes::mtp_policy))
        // Audio routes (speech, transcriptions, translations) come from the
        // sub-router that carries the larger body-limit layer.
        .merge(audio_routes)
        // Aliases (some clients use these)
        .route("/chat/completions", post(routes::chat_completions))
        // BREAKING (#1441): `/completions` and `/embeddings` are llama-server
        // NATIVE routes, not OpenAI aliases. b10621 sends `/completion` and
        // `/completions` to one handler and `/v1/completions` to a different
        // one, and does the same for `/embedding` / `/embeddings` against
        // `/v1/embeddings`. mlxcel used to answer the OpenAI shape on all of
        // them, so a llama-server client reading the native schema got an
        // object it could not parse.
        .route("/completions", post(routes::native_completion))
        .route("/models", get(routes::list_models))
        .route("/embedding", post(routes::native_embeddings))
        .route("/embeddings", post(routes::native_embeddings))
        .route("/rerank", post(routes::create_rerank))
        .route("/reranking", post(routes::create_rerank))
        .route("/responses", post(routes::create_response))
        .route(
            "/responses/:id",
            get(routes::retrieve_response).delete(routes::delete_response),
        )
        .route("/responses/:id/cancel", post(routes::cancel_response))
        .route("/messages", post(routes::anthropic_messages))
        .route(
            "/messages/count_tokens",
            post(routes::anthropic_count_tokens),
        )
        // llama-server compatible endpoints
        .route("/completion", post(routes::native_completion))
        .route("/tokenize", post(routes::tokenize))
        .route("/detokenize", post(routes::detokenize))
        // Fill-in-the-middle. Mounted unconditionally, like every other route:
        // whether the loaded model can serve it is a property of its
        // vocabulary, and the handler answers 501 naming the missing FIM
        // tokens rather than 404, so a client can tell "this server does not
        // implement infill" from "this model cannot do it" (#1442).
        .route("/infill", post(routes::infill))
        // Prompt inspection: render or count a prompt without generating from
        // it (#1442).
        .route("/apply-template", post(routes::apply_template))
        .route(
            "/chat/completions/input_tokens",
            post(routes::chat_input_tokens),
        )
        .route(
            "/v1/chat/completions/input_tokens",
            post(routes::chat_input_tokens),
        )
        .route(
            "/responses/input_tokens",
            post(routes::responses_input_tokens),
        )
        .route(
            "/v1/responses/input_tokens",
            post(routes::responses_input_tokens),
        );

    // Conditionally enable /props endpoint
    if enable_props {
        app = app.route("/props", get(routes::props));
    }

    // Conditionally enable /slots endpoint
    if enable_slots {
        app = app.route("/slots", get(routes::slots));
    }

    // Conditionally enable /metrics endpoint
    if enable_metrics {
        app = app.route("/metrics", get(routes::metrics));
    }

    app
        // Health check
        .route("/health", get(routes::health_check))
        .route("/v1/health", get(routes::health_check))
        .route("/", get(routes::health_check))
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
