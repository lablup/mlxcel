use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{
    AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, ServerGenerateOptions, create_app,
};
use crate::tokenizer::MlxcelTokenizer;

fn route_test_app(config: ServerConfig) -> (axum::Router, mpsc::Receiver<ServerGenerateOptions>) {
    let (options_tx, options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        ),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    );
    (create_app(state), options_rx)
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
