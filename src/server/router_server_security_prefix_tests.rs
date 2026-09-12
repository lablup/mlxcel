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

//! Prefix-specific tests for the generic WebUI security wrapper.

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::routing::get;

use super::router_server_security_support_tests::{
    ROUTER_KEY, assert_webui_error_schema, secured_request,
};

#[tokio::test]
async fn generic_security_wrapper_handles_prefixed_legacy_sse_typed_auth_and_encoded_reload() {
    let policy =
        crate::server::webui::security::WebUiSecurityPolicy::with_prefixes_limits_and_rate(
            vec!["127.0.0.1:18037".to_string()],
            vec![HeaderValue::from_static("http://127.0.0.1:18037")],
            "/admin/webui",
            "/admin",
            32,
            1,
            1,
        )
        .expect("security policy");
    let app = crate::server::webui::security::secure_webui_router(
        Router::new()
            .route("/admin/models", get(|| async { StatusCode::OK }))
            .route(
                "/admin/models/sse",
                get(|| async { Body::from("event: snapshot\n\n") }),
            )
            .route(
                "/admin/ui-api/v1/operations",
                get(|| async { StatusCode::OK }),
            ),
        crate::server::resolve_api_keys(&[ROUTER_KEY.to_string()], &[]).expect("keys"),
        policy,
    );

    let missing = secured_request(
        app.clone(),
        Method::GET,
        "/admin/ui-api/v1/operations",
        None,
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
    let body = axum::body::to_bytes(missing.into_body(), 4096)
        .await
        .expect("typed auth body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("typed auth json");
    assert_webui_error_schema(&json, "unauthorized", false);

    let first_sse = secured_request(
        app.clone(),
        Method::GET,
        "/admin/models/sse",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(first_sse.status(), StatusCode::OK);
    let second_sse = secured_request(
        app.clone(),
        Method::GET,
        "/admin/models/sse",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(second_sse.status(), StatusCode::TOO_MANY_REQUESTS);
    drop(first_sse);

    let first_reload = secured_request(
        app.clone(),
        Method::GET,
        "/admin/models?re%6coad=1",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(first_reload.status(), StatusCode::OK);
    let second_reload = secured_request(
        app,
        Method::GET,
        "/admin/models?reload=1",
        Some(ROUTER_KEY),
        None,
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(second_reload.status(), StatusCode::TOO_MANY_REQUESTS);
}
