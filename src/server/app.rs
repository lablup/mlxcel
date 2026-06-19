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
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{Request, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use tower_http::trace::TraceLayer;

use super::AppState;
use super::cors::build_cors_layer;
use super::routes;
use super::types::ErrorResponse;

/// API key authentication middleware
async fn api_key_auth(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // If no API key is configured, skip authentication
    let Some(expected_key) = state.config.api_key.as_ref() else {
        return next.run(request).await;
    };

    // Skip auth for health check endpoints
    let path = request.uri().path();
    if path == "/health" || path == "/" {
        return next.run(request).await;
    }

    // Check for Authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());

    match auth_header {
        Some(auth) => {
            // Support "Bearer <token>" format
            let token = if let Some(stripped) = auth.strip_prefix("Bearer ") {
                stripped
            } else {
                auth
            };

            if token == expected_key {
                next.run(request).await
            } else {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse::new(
                        "Invalid API key".to_string(),
                        "invalid_api_key",
                    )),
                )
                    .into_response()
            }
        }
        None => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse::new(
                "Missing API key. Use 'Authorization: Bearer <api-key>' header.".to_string(),
                "missing_api_key",
            )),
        )
            .into_response(),
    }
}

/// Maximum request body size for audio upload endpoints. Overrides the Axum
/// 2 MiB default because real audio uploads commonly exceed that threshold.
const AUDIO_MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

/// Create the Axum application router
pub fn create_app(state: AppState) -> Router {
    let enable_slots = state.config.enable_slots_endpoint;
    let enable_props = state.config.enable_props_endpoint;
    let enable_metrics = state.config.enable_metrics_endpoint;
    // CORS policy (#244): restrict to the configured allow-list when set,
    // otherwise keep the historical permissive default.
    let cors = build_cors_layer(state.config.cors_allowed_origins.as_deref());

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
        .route("/v1/completions", post(routes::completions))
        .route("/v1/models", get(routes::list_models))
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
        // Audio routes (speech, transcriptions, translations) come from the
        // sub-router that carries the larger body-limit layer.
        .merge(audio_routes)
        // Aliases (some clients use these)
        .route("/chat/completions", post(routes::chat_completions))
        .route("/completions", post(routes::completions))
        .route("/models", get(routes::list_models))
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
        .route("/detokenize", post(routes::detokenize));

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
        .route("/", get(routes::health_check))
        // Middleware
        .layer(middleware::from_fn_with_state(state.clone(), api_key_auth))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::AUDIO_MAX_UPLOAD_BYTES;
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
