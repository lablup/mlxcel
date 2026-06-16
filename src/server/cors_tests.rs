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

//! Unit tests for CORS origin parsing and layer selection (#244).

use super::{build_cors_layer, parse_allowed_origins};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderValue, Request, StatusCode, header};
use axum::routing::get;
use tower::ServiceExt;

fn origins(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_string()).collect()
}

#[test]
fn parses_single_valid_origin() {
    let parsed = parse_allowed_origins(&origins(&["https://app.example.com"])).unwrap();
    assert_eq!(
        parsed,
        vec![HeaderValue::from_static("https://app.example.com")]
    );
}

#[test]
fn parses_multiple_valid_origins() {
    let parsed = parse_allowed_origins(&origins(&[
        "https://app.example.com",
        "http://localhost:5173",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        vec![
            HeaderValue::from_static("https://app.example.com"),
            HeaderValue::from_static("http://localhost:5173"),
        ]
    );
}

#[test]
fn trims_whitespace_and_skips_blank_inner_entries() {
    // A trailing comma yields a blank entry that should be ignored, while the
    // surrounding values are trimmed and kept.
    let parsed = parse_allowed_origins(&origins(&[
        "  https://app.example.com  ",
        "",
        "https://admin.example.com",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        vec![
            HeaderValue::from_static("https://app.example.com"),
            HeaderValue::from_static("https://admin.example.com"),
        ]
    );
}

#[test]
fn empty_input_is_unset_not_error() {
    // No flag at all maps to an empty vec, which the caller treats as permissive.
    let parsed = parse_allowed_origins(&[]).unwrap();
    assert!(parsed.is_empty());
}

#[test]
fn rejects_value_that_is_only_blank() {
    assert!(parse_allowed_origins(&origins(&[""])).is_err());
    assert!(parse_allowed_origins(&origins(&["   "])).is_err());
}

#[test]
fn rejects_origin_without_scheme() {
    let err = parse_allowed_origins(&origins(&["app.example.com"])).unwrap_err();
    assert!(err.to_string().contains("app.example.com"));
}

#[test]
fn rejects_origin_with_path() {
    assert!(parse_allowed_origins(&origins(&["https://x.com/foo"])).is_err());
}

#[test]
fn rejects_origin_with_query() {
    assert!(parse_allowed_origins(&origins(&["https://x.com?a=1"])).is_err());
}

#[test]
fn rejects_non_http_scheme() {
    assert!(parse_allowed_origins(&origins(&["ftp://x.com"])).is_err());
}

#[test]
fn rejects_control_characters() {
    assert!(parse_allowed_origins(&origins(&["https://x.com\nevil"])).is_err());
}

#[test]
fn rejects_origin_with_trailing_slash() {
    // A trailing slash is a path; the browser `Origin` header carries none, so
    // `https://app.example.com/` could only ever silently never match. Reject
    // it at startup so the misconfiguration surfaces instead of failing closed
    // but confusing. `http::Uri::path()` reports the empty path as `/`, so this
    // must be caught on the raw string, not via the parsed path.
    let err = parse_allowed_origins(&origins(&["https://app.example.com/"])).unwrap_err();
    assert!(err.to_string().contains("app.example.com"));
}

#[test]
fn rejects_origin_with_userinfo() {
    // A browser `Origin` never includes `user[:pass]@`; an authority carrying
    // userinfo can never match, so reject it rather than silently dropping it.
    assert!(parse_allowed_origins(&origins(&["http://user@host"])).is_err());
    assert!(parse_allowed_origins(&origins(&["https://user:pass@app.example.com"])).is_err());
}

fn test_router(origins: Option<&[HeaderValue]>) -> Router {
    Router::new()
        .route("/", get(|| async { "ok" }))
        .layer(build_cors_layer(origins))
}

#[tokio::test]
async fn restrictive_layer_reflects_allowed_origin() {
    let allowed = parse_allowed_origins(&origins(&["https://allowed.example.com"])).unwrap();
    let app = test_router(Some(&allowed));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(header::ORIGIN, "https://allowed.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let acao = response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .map(|v| v.to_str().unwrap().to_string());
    assert_eq!(acao, Some("https://allowed.example.com".to_string()));
}

#[tokio::test]
async fn restrictive_layer_does_not_reflect_disallowed_origin() {
    let allowed = parse_allowed_origins(&origins(&["https://allowed.example.com"])).unwrap();
    let app = test_router(Some(&allowed));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(header::ORIGIN, "https://evil.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // A disallowed origin gets no Access-Control-Allow-Origin header at all,
    // so the browser blocks the cross-origin read.
    let acao = response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN);
    assert!(
        acao.is_none(),
        "disallowed origin must not be reflected, got {acao:?}"
    );
}

#[tokio::test]
async fn permissive_default_reflects_any_origin() {
    // None (unset) keeps the historical permissive behavior: ACAO is `*`.
    let app = test_router(None);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(header::ORIGIN, "https://anything.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let acao = response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .map(|v| v.to_str().unwrap().to_string());
    assert_eq!(acao, Some("*".to_string()));
}
