use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{
    AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, ServerGenerateOptions, create_app,
};
use crate::server::state::ModelMediaSupport;
use crate::tokenizer::MlxcelTokenizer;

fn route_test_app(config: ServerConfig) -> (axum::Router, mpsc::Receiver<ServerGenerateOptions>) {
    let (options_tx, options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    );
    (create_app(state), options_rx)
}

fn route_test_app_with_provider(
    config: ServerConfig,
    provider: Arc<ModelProvider>,
) -> axum::Router {
    create_app(route_test_state(config, provider))
}

/// Same app, but the loaded model declares it can consume images.
///
/// `AppState::new` defaults `media_support` to the all-refusing set, which is
/// correct for a text-only checkpoint: since #1451 the modality gate refuses an
/// `image_url` part with 501 `not_supported_error` *before* anything reads the
/// referenced URL or file. A test that wants to reach the per-request image
/// checks behind that gate has to say the model could process an image in the
/// first place, otherwise it only ever exercises the gate.
fn route_test_app_with_image_support(
    config: ServerConfig,
    provider: Arc<ModelProvider>,
) -> axum::Router {
    let state = route_test_state(config, provider).with_media_support(ModelMediaSupport {
        image: true,
        ..ModelMediaSupport::default()
    });
    create_app(state)
}

fn route_test_state(config: ServerConfig, provider: Arc<ModelProvider>) -> AppState {
    let batch_metrics = provider.batch_metrics().clone();
    AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
}

async fn post_json(app: axum::Router, path: &str, body: Value) -> StatusCode {
    app.oneshot(
        Request::builder()
            .method(Method::POST)
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("route test request"),
    )
    .await
    .expect("route test response")
    .status()
}

fn capped_config() -> ServerConfig {
    ServerConfig {
        context_size: 64,
        default_max_tokens: 32,
        ..Default::default()
    }
}

#[tokio::test]
async fn explicit_over_cap_budget_is_clamped_on_all_generation_routes() {
    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": "route-test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1024
            }),
        ),
        (
            "/v1/completions",
            json!({
                "model": "route-test-model",
                "prompt": "hello",
                "max_tokens": 1024
            }),
        ),
        (
            "/completion",
            json!({
                "prompt": "hello",
                "n_predict": 1024
            }),
        ),
    ];

    for (path, body) in cases {
        let (app, options_rx) = route_test_app(capped_config());
        assert_eq!(post_json(app, path, body).await, StatusCode::OK, "{path}");
        assert_eq!(
            options_rx
                .recv()
                .expect("route dispatched generation options")
                .max_tokens,
            64,
            "{path}"
        );
    }
}

#[tokio::test]
async fn streaming_generation_routes_queue_full_after_snapshot_return_http_503() {
    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": "route-test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "max_tokens": 1
            }),
        ),
        (
            "/v1/responses",
            json!({
                "model": "route-test-model",
                "input": "hello",
                "stream": true,
                "max_output_tokens": 1
            }),
        ),
        (
            "/v1/messages",
            json!({
                "model": "route-test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "max_tokens": 1
            }),
        ),
        (
            "/v1/completions",
            json!({
                "model": "route-test-model",
                "prompt": "hello",
                "stream": true,
                "max_tokens": 1
            }),
        ),
        (
            "/completion",
            json!({
                "prompt": "hello",
                "stream": true,
                "n_predict": 1
            }),
        ),
    ];

    for (path, body) in cases {
        let (options_tx, options_rx) = mpsc::channel();
        let provider = Arc::new(ModelProvider::recording_for_route_tests_with_admission(
            options_tx, true, 0,
        ));
        let mut config = capped_config();
        config.max_queue_depth = 1;
        let app = route_test_app_with_provider(config, provider);

        let status = post_json(app, path, body).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{path}");
        assert!(options_rx.try_recv().is_err(), "{path}");
    }
}

#[tokio::test]
async fn non_stream_generation_routes_queue_full_after_snapshot_return_http_503() {
    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": "route-test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1
            }),
        ),
        (
            "/v1/responses",
            json!({
                "model": "route-test-model",
                "input": "hello",
                "max_output_tokens": 1
            }),
        ),
        (
            "/v1/messages",
            json!({
                "model": "route-test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1
            }),
        ),
        (
            "/v1/completions",
            json!({
                "model": "route-test-model",
                "prompt": "hello",
                "max_tokens": 1
            }),
        ),
        (
            "/completion",
            json!({
                "prompt": "hello",
                "n_predict": 1
            }),
        ),
    ];

    for (path, body) in cases {
        let (options_tx, options_rx) = mpsc::channel();
        let provider = Arc::new(ModelProvider::recording_for_route_tests_with_admission(
            options_tx, true, 0,
        ));
        let mut config = capped_config();
        config.max_queue_depth = 1;
        let app = route_test_app_with_provider(config, provider);

        let status = post_json(app, path, body).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{path}");
        assert!(options_rx.try_recv().is_err(), "{path}");
    }
}

#[test]
fn route_image_cardinality_rejection_does_not_poison_same_worker() {
    std::thread::Builder::new()
        .name("route-image-cardinality-recovery".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(async {
                    let (options_tx, options_rx) = mpsc::channel();
                    let provider =
                        Arc::new(ModelProvider::recording_for_route_tests(options_tx));
                    let app = route_test_app_with_image_support(capped_config(), provider);

                    let bad = json!({
                        "model": "route-test-model",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "look"},
                                {"type": "image_url", "image_url": {"url": "missing-cardinality-route-test.png"}}
                            ]
                        }],
                        "max_tokens": 1
                    });
                    assert_eq!(
                        post_json(app.clone(), "/v1/chat/completions", bad).await,
                        StatusCode::BAD_REQUEST
                    );
                    assert!(
                        options_rx.try_recv().is_err(),
                        "rejected image request must not dispatch"
                    );

                    let good = json!({
                        "model": "route-test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1
                    });
                    assert_eq!(
                        post_json(app, "/v1/chat/completions", good).await,
                        StatusCode::OK
                    );
                    assert_eq!(
                        options_rx
                            .recv_timeout(std::time::Duration::from_secs(1))
                            .expect("same worker must receive the next valid request")
                            .max_tokens,
                        1
                    );
                });
        })
        .expect("spawn recovery test thread")
        .join()
        .expect("recovery test thread");
}

#[tokio::test]
async fn below_cap_budget_is_unchanged_on_all_generation_routes() {
    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": "route-test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 17
            }),
        ),
        (
            "/v1/completions",
            json!({
                "model": "route-test-model",
                "prompt": "hello",
                "max_tokens": 17
            }),
        ),
        (
            "/completion",
            json!({
                "prompt": "hello",
                "n_predict": 17
            }),
        ),
    ];

    for (path, body) in cases {
        let (app, options_rx) = route_test_app(capped_config());
        assert_eq!(post_json(app, path, body).await, StatusCode::OK, "{path}");
        assert_eq!(
            options_rx
                .recv()
                .expect("route dispatched generation options")
                .max_tokens,
            17,
            "{path}"
        );
    }
}

#[tokio::test]
async fn absent_native_n_predict_uses_resolved_server_default() {
    let (app, options_rx) = route_test_app(capped_config());

    assert_eq!(
        post_json(app, "/completion", json!({"prompt": "hello"})).await,
        StatusCode::OK
    );
    assert_eq!(
        options_rx
            .recv()
            .expect("native route dispatched generation options")
            .max_tokens,
        32
    );
}

#[tokio::test]
async fn reasoning_budget_is_validated_against_clamped_route_budget() {
    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": "route-test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1024,
                "thinking_budget_tokens": 65
            }),
        ),
        (
            "/v1/completions",
            json!({
                "model": "route-test-model",
                "prompt": "hello",
                "max_tokens": 1024,
                "thinking_budget_tokens": 65
            }),
        ),
        (
            "/completion",
            json!({
                "prompt": "hello",
                "n_predict": 1024,
                "thinking_budget_tokens": 65
            }),
        ),
    ];

    for (path, body) in cases {
        let (app, options_rx) = route_test_app(capped_config());
        assert_eq!(
            post_json(app, path, body).await,
            StatusCode::BAD_REQUEST,
            "{path}"
        );
        assert!(
            options_rx.try_recv().is_err(),
            "{path} must reject before model dispatch"
        );
    }
}
