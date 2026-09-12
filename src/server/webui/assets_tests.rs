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

use super::*;
use axum::{
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode, header},
    routing::get,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tower::ServiceExt;

async fn request(path: &str, method: Method) -> axum::response::Response {
    router::<()>()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router answers")
}

#[test]
fn embedded_manifest_is_valid_and_budgeted() {
    let manifest: Value = serde_json::from_str(manifest_json()).expect("manifest is valid JSON");
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["package_manager"], "pnpm@11.18.0");
    let files = manifest["files"].as_array().expect("files list");
    assert!(files.iter().any(|file| file["path"] == "index.html"));
    assert!(
        files
            .iter()
            .any(|file| file["path"] == "third-party-licenses.txt")
    );
    let budgets = manifest["budgets"].as_object().expect("budgets object");
    assert!(budgets["initial_js_gzip_bytes"].as_u64().unwrap() <= 200 * 1024);
    assert!(budgets["total_js_gzip_bytes"].as_u64().unwrap() <= 700 * 1024);
    assert!(budgets["embedded_asset_bytes"].as_u64().unwrap() <= 5 * 1024 * 1024);
}

#[tokio::test]
async fn index_and_assets_are_served_with_expected_headers() {
    let response = request("/webui/", Method::GET).await;
    assert_eq!(response.status(), StatusCode::OK);
    let headers = response.headers();
    assert_eq!(headers[header::CACHE_CONTROL], "no-cache");
    assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
    assert!(
        headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/html")
    );
    assert!(headers.get(header::ETAG).is_some());
}

#[tokio::test]
async fn head_uses_the_same_metadata_without_a_body() {
    let response = request("/webui/", Method::HEAD).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers().get(header::ETAG).is_some());
    let body = to_bytes(response.into_body(), 32).await.expect("body read");
    assert!(body.is_empty());
}

#[tokio::test]
async fn conditional_get_returns_304_with_security_headers() {
    let first = request("/webui/", Method::GET).await;
    let etag = first.headers()[header::ETAG]
        .to_str()
        .expect("etag is text")
        .to_string();
    let response = router::<()>()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/webui/")
                .header(header::IF_NONE_MATCH, format!("W/{etag}, \"other\""))
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router answers");
    assert_eq!(response.status(), StatusCode::NOT_MODIFIED);
    assert_eq!(
        response.headers()[header::X_CONTENT_TYPE_OPTIONS],
        "nosniff"
    );
    let body = to_bytes(response.into_body(), 32).await.expect("body read");
    assert!(body.is_empty());
}

#[tokio::test]
async fn head_errors_suppress_body() {
    let response = request("/webui/assets/missing.js", Method::HEAD).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(response.into_body(), 32).await.expect("body read");
    assert!(body.is_empty());
}

#[tokio::test]
async fn static_assets_use_immutable_cache_and_correct_mime() {
    let manifest: Value = serde_json::from_str(manifest_json()).expect("manifest is valid JSON");
    let asset_path = manifest["files"]
        .as_array()
        .expect("files")
        .iter()
        .filter_map(|file| file["path"].as_str())
        .find(|path| path.ends_with(".js"))
        .expect("javascript asset");
    let response = request(&format!("/webui/{asset_path}"), Method::GET).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CACHE_CONTROL],
        "public, max-age=31536000, immutable"
    );
    assert!(
        response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .contains("javascript")
    );
}

#[tokio::test]
async fn missing_assets_and_unsupported_methods_do_not_fall_back_to_shell() {
    let missing = request("/webui/assets/missing.js", Method::GET).await;
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        missing.headers()[header::CONTENT_SECURITY_POLICY],
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'"
    );

    let disallowed = request("/webui/assets/missing.js", Method::POST).await;
    assert_eq!(disallowed.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(disallowed.headers()[header::ALLOW], "GET, HEAD");
    assert_eq!(
        disallowed.headers()[header::CONTENT_SECURITY_POLICY],
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'"
    );
}

#[tokio::test]
async fn path_attacks_are_rejected() {
    for path in [
        "/webui/assets/../index.html",
        "/webui/assets/%2e%2e/index.html",
        "/webui/assets/%00.js",
        "/webui/assets/%5cindex.js",
        "/webui/assets/%zz.js",
        "/webui/assets\\index.js",
    ] {
        assert_eq!(
            request(path, Method::GET).await.status(),
            StatusCode::BAD_REQUEST,
            "{path}"
        );
    }
}

#[tokio::test]
async fn trailing_slash_redirect_preserves_nested_api_prefix() {
    let app = axum::Router::new().nest("/private", router::<()>());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/private/webui")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router answers");
    assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
    assert_eq!(response.headers()[header::LOCATION], "/private/webui/");
}

#[tokio::test]
async fn nested_api_prefix_serves_assets() {
    let manifest: Value = serde_json::from_str(manifest_json()).expect("manifest is valid JSON");
    let asset_path = manifest["files"]
        .as_array()
        .expect("files")
        .iter()
        .filter_map(|file| file["path"].as_str())
        .find(|path| path.ends_with(".css"))
        .expect("css asset");
    let app = axum::Router::new().nest("/private", router::<()>());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri(format!("/private/webui/{asset_path}"))
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router answers");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn webui_router_does_not_swallow_root_or_api_routes() {
    let app = axum::Router::new()
        .route("/", get(|| async { "health" }))
        .route("/v1/models", get(|| async { "models" }))
        .merge(router::<()>());
    assert_eq!(
        app.clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        app.clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        app.oneshot(
            Request::builder()
                .uri("/webui/v1/models")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap()
        .status(),
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn manifest_hashes_match_embedded_assets() {
    let manifest: Value = serde_json::from_str(manifest_json()).expect("manifest is valid JSON");
    for file in manifest["files"].as_array().expect("files") {
        let path = file["path"].as_str().expect("path");
        let asset =
            WebUiAssets::get(path).unwrap_or_else(|| panic!("missing embedded asset {path}"));
        let data = asset.data.as_ref();
        let digest = format!("{:x}", Sha256::digest(data));
        assert_eq!(file["sha256"], digest, "{path}");
        assert_eq!(file["bytes"], data.len(), "{path}");
    }
}
