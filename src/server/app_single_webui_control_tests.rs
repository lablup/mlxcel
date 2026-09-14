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

use axum::{
    body::Body,
    http::{HeaderValue, Request, StatusCode},
};
use tower::ServiceExt;

fn app() -> axum::Router {
    let (tx, _rx) = std::sync::mpsc::channel();
    let provider = std::sync::Arc::new(crate::server::ModelProvider::recording_for_route_tests(tx));
    let metrics = provider.batch_metrics().clone();
    let config = crate::server::ServerConfig {
        api_keys: crate::server::resolve_api_keys(&["readonly-test-key".into()], &[]).unwrap(),
        api_prefix: "/admin".into(),
        ..Default::default()
    };
    let state = crate::server::AppState::new(
        provider,
        config,
        crate::server::ChatTemplateProcessor::with_template("ok".into()),
        crate::tokenizer::MlxcelTokenizer::stub(),
        "single-webui-route-test-model".into(),
        metrics,
    );
    let policy = crate::server::webui::security::WebUiSecurityPolicy::with_prefixes_and_limits(
        vec!["127.0.0.1:18038".into()],
        vec![HeaderValue::from_static("http://127.0.0.1:18038")],
        "/admin/webui",
        "/admin",
        8,
        8,
    )
    .unwrap();
    super::create_app_with_secured_ui(state, policy)
}

const CONTROLS: &[&str] = &[
    "model-actions",
    "downloads",
    "model-removals",
    "catalog/refresh",
    "operations/op_aaaaaaaaaaaaaaaa/cancel",
];

fn request(method: &str, path: &str, body: Body, authenticated: bool) -> Request<Body> {
    let mut builder = Request::builder()
        .method(method)
        .uri(format!("/admin/ui-api/v1/{path}"))
        .header("host", "127.0.0.1:18038")
        .header("content-type", "application/json");
    if authenticated {
        builder = builder.header("authorization", "Bearer readonly-test-key");
    }
    builder.body(body).unwrap()
}

async fn json(response: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap(),
    )
    .unwrap()
}

async fn assert_error(response: axum::response::Response, status: StatusCode, fixture: &str) {
    assert_eq!(response.status(), status);
    assert_eq!(response.headers()["content-type"], "application/json");
    let mut actual = json(response).await;
    let request_id = actual["request_id"].as_str().unwrap();
    assert!(request_id.starts_with("req_") && request_id.len() > 4 && request_id.len() <= 128);
    let mut expected: serde_json::Value = serde_json::from_str(fixture).unwrap();
    expected.as_object_mut().unwrap().remove("$schemaName");
    actual["request_id"] = expected["request_id"].clone();
    assert_eq!(actual, expected, "entire canonical error envelope");
}

const READ_ONLY: &str =
    include_str!("../../tests/fixtures/webui/examples/error.single-readonly.json");
const NOT_FOUND: &str =
    include_str!("../../tests/fixtures/webui/examples/error.single-operation-not-found.json");

#[tokio::test]
async fn mounted_single_control_is_canonical_read_only() {
    let app = app();
    let before = json(
        app.clone()
            .oneshot(request("GET", "catalog", Body::empty(), true))
            .await
            .unwrap(),
    )
    .await;
    let operations = json(
        app.clone()
            .oneshot(request("GET", "operations", Body::empty(), true))
            .await
            .unwrap(),
    )
    .await;
    let entry = &before["items"][0];
    assert!(super::valid_model_id(
        entry["identity"]["id"].as_str().unwrap()
    ));
    assert!(entry["identity"]["revision"].as_u64().unwrap() > 0);
    for path in CONTROLS {
        let payload = match *path {
            "model-actions" => {
                serde_json::json!({"action":"unload", "model_id":entry["identity"]["id"],
                "expected_revision":entry["identity"]["revision"], "idempotency_key":"readonly-fixture-key"})
            }
            "model-removals" => serde_json::json!({"model_id":entry["identity"]["id"],
                "expected_revision":entry["identity"]["revision"], "idempotency_key":"readonly-fixture-key"}),
            "downloads" => serde_json::json!({"repo_id":"mlx-community/Qwen3-4B-4bit",
                "revision":"main", "idempotency_key":"readonly-fixture-key"}),
            _ => serde_json::json!({}),
        };
        let response = app
            .clone()
            .oneshot(request("POST", path, Body::from(payload.to_string()), true))
            .await
            .unwrap();
        assert_error(response, StatusCode::UNPROCESSABLE_ENTITY, READ_ONLY).await;
    }
    let response = app
        .clone()
        .oneshot(request(
            "GET",
            "operations/op_aaaaaaaaaaaaaaaa",
            Body::empty(),
            true,
        ))
        .await
        .unwrap();
    assert_error(response, StatusCode::NOT_FOUND, NOT_FOUND).await;
    assert_eq!(
        json(
            app.clone()
                .oneshot(request("GET", "catalog", Body::empty(), true))
                .await
                .unwrap()
        )
        .await,
        before
    );
    assert_eq!(
        json(
            app.oneshot(request("GET", "operations", Body::empty(), true))
                .await
                .unwrap()
        )
        .await,
        operations
    );
}

#[tokio::test]
async fn mounted_single_controls_keep_auth_and_origin_envelopes() {
    for path in CONTROLS
        .iter()
        .copied()
        .chain(["operations/op_aaaaaaaaaaaaaaaa"])
    {
        let method = if path == "operations/op_aaaaaaaaaaaaaaaa" {
            "GET"
        } else {
            "POST"
        };
        let response = app()
            .oneshot(request(method, path, Body::from("{}"), false))
            .await
            .unwrap();
        assert_error(
            response,
            StatusCode::UNAUTHORIZED,
            include_str!("../../tests/fixtures/webui/examples/error.security-unauthorized.json"),
        )
        .await;
        let mut req = request(method, path, Body::from("{}"), true);
        req.headers_mut().insert(
            "origin",
            HeaderValue::from_static("https://untrusted.example"),
        );
        let response = app().oneshot(req).await.unwrap();
        assert_error(
            response,
            StatusCode::FORBIDDEN,
            include_str!(
                "../../tests/fixtures/webui/examples/error.security-forbidden-origin.json"
            ),
        )
        .await;
    }
}

#[tokio::test]
async fn mounted_single_controls_keep_declared_and_chunked_two_mib_limits() {
    let limit = crate::server::webui::api::WEBUI_JSON_BODY_BYTES as usize;
    assert_eq!(limit, 2_097_152);
    for path in CONTROLS {
        for declared in [true, false] {
            for size in [limit, limit + 1] {
                let payload = serde_json::json!({"payload": "x".repeat(size - 14)}).to_string();
                assert_eq!(payload.len(), size);
                let body = if declared {
                    Body::from(payload)
                } else {
                    let chunks: Vec<Result<axum::body::Bytes, std::io::Error>> = payload
                        .as_bytes()
                        .chunks(65_536)
                        .map(|chunk| Ok(axum::body::Bytes::copy_from_slice(chunk)))
                        .collect();
                    Body::from_stream(futures::stream::iter(chunks))
                };
                let mut req = request("POST", path, body, true);
                if declared {
                    req.headers_mut()
                        .insert("content-length", size.to_string().parse().unwrap());
                } else {
                    req.headers_mut()
                        .insert("transfer-encoding", HeaderValue::from_static("chunked"));
                }
                let response = app().oneshot(req).await.unwrap();
                if size == limit {
                    assert_error(response, StatusCode::UNPROCESSABLE_ENTITY, READ_ONLY).await;
                } else {
                    assert_error(
                        response,
                        StatusCode::PAYLOAD_TOO_LARGE,
                        include_str!(
                            "../../tests/fixtures/webui/examples/error.security-body.json"
                        ),
                    )
                    .await;
                }
            }
        }
    }
}
