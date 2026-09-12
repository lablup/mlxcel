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

//! Literal boundary probes cross-checked against the frozen API contract.

use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::http::{HeaderValue, Method, Request, StatusCode, header};
use axum::routing::post;
use tower::ServiceExt;

use super::{WebUiSecurityPolicy, secure_webui_router};

fn contract_body_limit() -> usize {
    let contract: serde_json::Value =
        serde_json::from_str(include_str!("../../../docs/webui/api.yaml")).unwrap();
    let limit = contract
        .pointer("/components/schemas/LimitSummary/properties/json_body_bytes/const")
        .unwrap()
        .as_u64()
        .unwrap() as usize;
    assert_eq!(limit, 2_097_152, "frozen public JSON body limit changed");
    limit
}

fn app() -> Router {
    let router = Router::new().route(
        "/ui-api/v1/boundary-probe",
        post(|request: Request<Body>| async {
            // Consume directly so no extractor body limit can mask the middleware.
            let bytes = to_bytes(request.into_body(), usize::MAX).await.unwrap();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert!(value["payload"].as_str().is_some());
            StatusCode::NO_CONTENT
        }),
    );
    let keys = crate::server::resolve_api_keys(&["boundary-test-key".into()], &[]).unwrap();
    let policy = WebUiSecurityPolicy::new(
        vec!["127.0.0.1:18037".into()],
        vec![HeaderValue::from_static("http://127.0.0.1:18037")],
    )
    .unwrap();
    secure_webui_router(router, keys, policy)
}

async fn check_boundary(declared: bool) {
    let limit = contract_body_limit();
    for (size, expected) in [
        (limit, StatusCode::NO_CONTENT),
        (limit + 1, StatusCode::PAYLOAD_TOO_LARGE),
    ] {
        let payload = serde_json::to_vec(&serde_json::json!({
            "payload": "x".repeat(size - 14)
        }))
        .unwrap();
        assert_eq!(payload.len(), size, "probe must have the exact byte count");
        let mut request = Request::builder()
            .method(Method::POST)
            .uri("/ui-api/v1/boundary-probe")
            .header(header::HOST, "127.0.0.1:18037")
            .header(header::AUTHORIZATION, "Bearer boundary-test-key")
            .header(header::CONTENT_TYPE, "application/json");
        let body = if declared {
            request = request.header(header::CONTENT_LENGTH, size.to_string());
            Body::from(payload)
        } else {
            request = request.header(header::TRANSFER_ENCODING, "chunked");
            let chunks: Vec<Result<Bytes, std::io::Error>> = payload
                .chunks(65_536)
                .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                .collect();
            Body::from_stream(futures::stream::iter(chunks))
        };
        let response = app().oneshot(request.body(body).unwrap()).await.unwrap();
        assert_eq!(
            response.status(),
            expected,
            "declared={declared}, bytes={size}"
        );
    }
}

#[tokio::test]
async fn declared_json_body_honors_frozen_two_mib_boundary() {
    check_boundary(true).await;
}

#[tokio::test]
async fn chunked_json_body_honors_frozen_two_mib_boundary() {
    check_boundary(false).await;
}
