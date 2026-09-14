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
use axum::http::StatusCode;

fn prefixed_library(root: &Path, cache: bool) -> Router {
    let state = router_state_with_startup(
        RouterSources {
            cache: cache.then(|| {
                crate::server::router_cache::CacheSource::new(
                    root.into(),
                    Arc::new(NoBootstrapDownload),
                )
            }),
            ..Default::default()
        },
        ServerConfig {
            api_prefix: "/lab".into(),
            enable_settings_endpoint: true,
            ..keyed_config()
        },
        ServerStartupConfig {
            webui_enabled: true,
            model_store_root: Some(root.into()),
            ..Default::default()
        },
        false,
    );
    let policy =
        crate::server::webui::security::WebUiSecurityPolicy::with_prefixes_limits_and_rate(
            vec!["127.0.0.1:18037".into()],
            vec![HeaderValue::from_static("http://127.0.0.1:18037")],
            "/lab/webui",
            "/lab",
            32,
            16,
            120,
        )
        .unwrap();
    create_router_app_with_secured_ui(state, policy)
}

#[allow(clippy::too_many_arguments)]
async fn request(
    app: Router,
    method: &str,
    path: &str,
    host: &str,
    origin: Option<&str>,
    fetch: Option<&str>,
    authenticated: bool,
    body: &str,
) -> axum::response::Response {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("host", host)
        .header("content-type", "application/json");
    if authenticated {
        request = request.header("authorization", format!("Bearer {ROUTER_KEY}"));
    }
    if let Some(origin) = origin {
        request = request.header("origin", origin);
    }
    if let Some(fetch) = fetch {
        request = request.header("sec-fetch-site", fetch);
    }
    app.oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn prefixed_library_is_secured_once_and_observation_does_not_create_cache() {
    let temp = tempfile::tempdir().unwrap();
    let absent = temp.path().join("absent-store");
    let app = prefixed_library(&absent, true);
    let response = request(
        app.clone(),
        "GET",
        "/lab/ui-api/v1/bootstrap",
        "127.0.0.1:18037",
        None,
        None,
        true,
        "",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 32 * 1024)
        .await
        .unwrap();
    let mut actual: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/webui/examples/bootstrap.model-free.json"
    ))
    .unwrap();
    expected["server"]["api_base"] = serde_json::json!("/lab");
    normalize_bootstrap_fixture(&mut actual, &expected);
    assert_eq!(
        actual, expected,
        "entire prefixed actual bootstrap producer"
    );
    let response = request(
        app.clone(),
        "GET",
        "/lab/ui-api/v1/catalog",
        "127.0.0.1:18037",
        None,
        None,
        true,
        "",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    for suffix in ["downloads", "model-removals"] {
        let path = format!("/lab/ui-api/v1/{suffix}");
        for (host, origin, fetch, auth, expected) in [
            (
                "127.0.0.1:18037",
                None,
                None,
                false,
                StatusCode::UNAUTHORIZED,
            ),
            ("foreign.invalid", None, None, true, StatusCode::FORBIDDEN),
            (
                "127.0.0.1:18037",
                Some("http://foreign.invalid"),
                None,
                true,
                StatusCode::FORBIDDEN,
            ),
            (
                "127.0.0.1:18037",
                None,
                Some("cross-site"),
                true,
                StatusCode::FORBIDDEN,
            ),
            (
                "127.0.0.1:18037",
                None,
                Some("same-origin"),
                true,
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let response =
                request(app.clone(), "POST", &path, host, origin, fetch, auth, "{}").await;
            assert_eq!(
                response.status(),
                expected,
                "{path}: host={host} origin={origin:?} fetch={fetch:?}"
            );
        }
        for path in [
            format!("/ui-api/v1/{suffix}"),
            format!("/lab/lab/ui-api/v1/{suffix}"),
        ] {
            let response = request(
                app.clone(),
                "POST",
                &path,
                "127.0.0.1:18037",
                None,
                None,
                true,
                "{}",
            )
            .await;
            // Unknown routes deliberately retain the legacy dispatcher, whose
            // missing-model error differs from the mounted library JSON parser.
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body["error"]["message"],
                "model name is missing from the request"
            );
        }
    }
    assert!(
        !absent.exists(),
        "bootstrap/catalog/refused mutations must not create cache/staging or probe downloader"
    );
}

#[tokio::test]
async fn no_cache_library_admission_is_canonical_unsupported_without_mutation() {
    let temp = tempfile::tempdir().unwrap();
    let absent = temp.path().join("absent-store");
    let app = prefixed_library(&absent, false);
    for (suffix, body) in [
        (
            "downloads",
            serde_json::json!({"repo_id":"test-owner/model","idempotency_key":"no-cache-download"}),
        ),
        (
            "model-removals",
            serde_json::json!({"model_id":format!("mdl_{}", "a".repeat(43)),"expected_revision":1,"idempotency_key":"no-cache-removal"}),
        ),
    ] {
        let response = request(
            app.clone(),
            "POST",
            &format!("/lab/ui-api/v1/{suffix}"),
            "127.0.0.1:18037",
            None,
            None,
            true,
            &body.to_string(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let mut actual: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(actual["request_id"].as_str().unwrap().starts_with("req_"));
        actual["request_id"] = serde_json::json!("<request>");
        assert_eq!(
            actual,
            serde_json::json!({"request_id":"<request>","error":{"code":"unsupported","message":"no managed model cache is configured","retryable":false}})
        );
    }
    assert!(!absent.exists());
}
