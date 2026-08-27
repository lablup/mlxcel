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

//! Unit tests for CORS origin parsing (#244) and the b10621 CORS policy
//! (#1432).

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderValue, Method, Request, StatusCode, header};
use tower::ServiceExt;

use super::{CorsPolicy, OriginPolicy, parse_allowed_origins};
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn origins(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_string()).collect()
}

// ---------------------------------------------------------------------------
// --allowed-origins parsing (#244)
// ---------------------------------------------------------------------------

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
        "http://localhost:3000",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        vec![
            HeaderValue::from_static("https://app.example.com"),
            HeaderValue::from_static("http://localhost:3000"),
        ]
    );
}

#[test]
fn trims_whitespace_and_skips_blank_inner_entries() {
    let parsed = parse_allowed_origins(&origins(&[
        "  https://app.example.com  ",
        "",
        "   ",
        "http://localhost:3000",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        vec![
            HeaderValue::from_static("https://app.example.com"),
            HeaderValue::from_static("http://localhost:3000"),
        ]
    );
}

#[test]
fn empty_input_is_unset_not_error() {
    assert!(parse_allowed_origins(&[]).unwrap().is_empty());
}

#[test]
fn rejects_value_that_is_only_blank() {
    assert!(parse_allowed_origins(&origins(&["   "])).is_err());
}

#[test]
fn rejects_origin_without_scheme() {
    assert!(parse_allowed_origins(&origins(&["app.example.com"])).is_err());
}

#[test]
fn rejects_origin_with_path() {
    assert!(parse_allowed_origins(&origins(&["https://app.example.com/api"])).is_err());
}

#[test]
fn rejects_origin_with_query() {
    assert!(parse_allowed_origins(&origins(&["https://app.example.com?x=1"])).is_err());
}

#[test]
fn rejects_non_http_scheme() {
    assert!(parse_allowed_origins(&origins(&["ftp://app.example.com"])).is_err());
}

#[test]
fn rejects_control_characters() {
    assert!(parse_allowed_origins(&origins(&["https://x.com\nevil"])).is_err());
}

#[test]
fn rejects_origin_with_trailing_slash() {
    assert!(parse_allowed_origins(&origins(&["https://app.example.com/"])).is_err());
}

#[test]
fn rejects_origin_with_userinfo() {
    assert!(parse_allowed_origins(&origins(&["https://user:pass@app.example.com"])).is_err());
}

// ---------------------------------------------------------------------------
// b10621 policy resolution (#1432)
// ---------------------------------------------------------------------------

fn policy(cors_origins: &str, credentials: bool) -> CorsPolicy {
    CorsPolicy::resolve(
        cors_origins,
        "GET, POST, DELETE, OPTIONS",
        "*",
        credentials,
        None,
    )
    .expect("valid policy")
}

fn origin(value: &str) -> HeaderValue {
    HeaderValue::from_str(value).expect("valid origin")
}

#[test]
fn the_default_policy_is_the_b10621_default() {
    let default = CorsPolicy::default();
    assert_eq!(default.origins, OriginPolicy::Wildcard);
    assert_eq!(default.methods, "GET, POST, DELETE, OPTIONS");
    assert_eq!(default.headers, "*");
    assert!(default.credentials, "b10621 enables credentials by default");
}

#[test]
fn star_with_credentials_echoes_the_request_origin() {
    let p = policy("*", true);
    assert_eq!(
        p.allow_origin(Some(&origin("https://app.example.com"))),
        Some(origin("https://app.example.com"))
    );
}

#[test]
fn star_with_credentials_and_no_origin_header_echoes_an_empty_value() {
    // Upstream sets the header to `req.get_header_value("Origin")`, which is
    // the empty string when the request carries no Origin. Reproducing the
    // empty header keeps a non-browser client's view identical.
    let p = policy("*", true);
    assert_eq!(p.allow_origin(None), Some(HeaderValue::from_static("")));
}

#[test]
fn star_without_credentials_emits_the_literal_star() {
    let p = policy("*", false);
    assert_eq!(
        p.allow_origin(Some(&origin("https://app.example.com"))),
        Some(HeaderValue::from_static("*"))
    );
}

#[test]
fn localhost_reflects_only_localhost_origins() {
    let p = policy("localhost", true);
    for allowed in [
        "http://localhost:3000",
        "https://localhost",
        "http://127.0.0.1:8080",
        "http://[::1]:5173",
    ] {
        assert_eq!(
            p.allow_origin(Some(&origin(allowed))),
            Some(origin(allowed)),
            "{allowed} must be reflected"
        );
    }
}

#[test]
fn localhost_does_not_reflect_a_lookalike_host() {
    let p = policy("localhost", true);
    for denied in [
        "https://localhost.evil.com",
        "https://evil.com",
        "https://127.0.0.1.evil.com",
        "https://user@localhost.evil.com",
    ] {
        assert_eq!(
            p.allow_origin(Some(&origin(denied))),
            None,
            "{denied} must not be reflected"
        );
    }
    assert_eq!(p.allow_origin(None), None, "no Origin, no header");
}

#[test]
fn an_explicit_origin_string_is_emitted_verbatim() {
    // b10621 does no matching here: whatever the operator configured is what
    // the header carries, request Origin included or not.
    let p = policy("https://app.example.com", true);
    assert_eq!(
        p.allow_origin(Some(&origin("https://evil.example.com"))),
        Some(origin("https://app.example.com"))
    );
    assert_eq!(
        p.allow_origin(None),
        Some(origin("https://app.example.com"))
    );
}

#[test]
fn the_mlxcel_allow_list_reflects_only_a_matching_origin() {
    let list = parse_allowed_origins(&origins(&["https://allowed.example.com"])).unwrap();
    let p = CorsPolicy::resolve("*", "GET", "*", true, Some(list)).expect("valid");
    assert_eq!(p.origins.clone(), {
        let expect = parse_allowed_origins(&origins(&["https://allowed.example.com"])).unwrap();
        OriginPolicy::AllowList(expect)
    });
    assert_eq!(
        p.allow_origin(Some(&origin("https://allowed.example.com"))),
        Some(origin("https://allowed.example.com"))
    );
    assert_eq!(
        p.allow_origin(Some(&origin("https://evil.example.com"))),
        None,
        "a disallowed origin gets no header at all"
    );
    assert_eq!(p.allow_origin(None), None);
}

#[test]
fn preflight_headers_carry_the_configured_methods_and_credentials() {
    let p = CorsPolicy::resolve("*", "GET, POST", "X-Custom", false, None).expect("valid");
    let headers = p.preflight_headers();
    assert_eq!(headers[0].1, "false");
    assert_eq!(headers[1].1, "GET, POST");
    assert_eq!(headers[2].1, "X-Custom");

    let enabled = CorsPolicy::resolve("*", "GET", "*", true, None).expect("valid");
    assert_eq!(enabled.preflight_headers()[0].1, "true");
}

#[test]
fn a_header_value_with_a_newline_is_rejected_at_startup() {
    for (o, m, h) in [
        ("https://a\nevil", "GET", "*"),
        ("*", "GET\r\nX: y", "*"),
        ("*", "GET", "X\ny"),
    ] {
        assert!(
            CorsPolicy::resolve(o, m, h, true, None).is_err(),
            "{o:?} {m:?} {h:?} must be rejected"
        );
    }
}

// ---------------------------------------------------------------------------
// Middleware behavior on the real router (#1432)
// ---------------------------------------------------------------------------

fn app_with_policy(policy: CorsPolicy) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig {
            cors_policy: policy,
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("cors-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn send(app: Router, method: Method, path: &str, origin_header: Option<&str>) -> Response {
    let mut builder = Request::builder().method(method).uri(path);
    if let Some(value) = origin_header {
        builder = builder.header(header::ORIGIN, value);
    }
    app.oneshot(builder.body(Body::empty()).expect("request"))
        .await
        .expect("router responds")
}

type Response = axum::http::Response<Body>;

fn acao(response: &Response) -> Option<String> {
    response
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .map(|v| v.to_str().expect("utf-8 header").to_string())
}

#[tokio::test]
async fn a_preflight_is_answered_by_the_middleware_without_authentication() {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig {
            api_keys: crate::server::resolve_api_keys(&["secret".to_string()], &[])
                .expect("valid key set"),
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("cors-test-model"),
        batch_metrics,
    );
    let response = send(
        create_app(state),
        Method::OPTIONS,
        "/v1/chat/completions",
        Some("https://app.example.com"),
    )
    .await;

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "browsers do not send Authorization on a preflight, so it must not 401"
    );
    assert_eq!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_METHODS)
            .map(|v| v.to_str().unwrap().to_string()),
        Some("GET, POST, DELETE, OPTIONS".to_string())
    );
    assert_eq!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_CREDENTIALS)
            .map(|v| v.to_str().unwrap().to_string()),
        Some("true".to_string())
    );
    assert_eq!(acao(&response), Some("https://app.example.com".to_string()));
}

#[tokio::test]
async fn a_preflight_to_an_unrouted_path_is_still_answered() {
    // Upstream answers OPTIONS in the pre-routing handler, before the route
    // table is consulted, so an unknown path preflights successfully there too.
    let response = send(
        app_with_policy(CorsPolicy::default()),
        Method::OPTIONS,
        "/no/such/route",
        Some("https://app.example.com"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn a_normal_response_carries_allow_origin_but_not_the_preflight_headers() {
    let response = send(
        app_with_policy(CorsPolicy::default()),
        Method::GET,
        "/health",
        Some("https://app.example.com"),
    )
    .await;
    assert_eq!(acao(&response), Some("https://app.example.com".to_string()));
    assert!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_METHODS)
            .is_none(),
        "b10621 sets the method/header lists on preflights only"
    );
    assert!(
        response
            .headers()
            .get(header::ACCESS_CONTROL_EXPOSE_HEADERS)
            .is_none(),
        "b10621 never emits Access-Control-Expose-Headers"
    );
}

#[tokio::test]
async fn a_disallowed_allow_list_origin_gets_no_header_on_a_real_route() {
    let list = parse_allowed_origins(&origins(&["https://allowed.example.com"])).unwrap();
    let policy = CorsPolicy::resolve("*", "GET", "*", true, Some(list)).expect("valid");
    let response = send(
        app_with_policy(policy.clone()),
        Method::GET,
        "/health",
        Some("https://evil.example.com"),
    )
    .await;
    assert_eq!(acao(&response), None);

    let allowed = send(
        app_with_policy(policy),
        Method::GET,
        "/health",
        Some("https://allowed.example.com"),
    )
    .await;
    assert_eq!(
        acao(&allowed),
        Some("https://allowed.example.com".to_string())
    );
}
