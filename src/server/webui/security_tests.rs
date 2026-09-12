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

use axum::http::HeaderValue;

use super::startup::{WebUiListenKind, WebUiSecurityConfig, resolve_webui_security};
use super::*;

fn origin(value: &'static str) -> HeaderValue {
    HeaderValue::from_static(value)
}

fn loopback_config(
    configured_key_present: bool,
    interactive_terminal: bool,
) -> WebUiSecurityConfig {
    WebUiSecurityConfig {
        enabled: true,
        listen: WebUiListenKind::LoopbackTcp,
        tls_enabled: false,
        configured_key_present,
        interactive_terminal,
        allowed_hosts: vec!["127.0.0.1:18037".to_string()],
        allowed_origins: vec![origin("http://127.0.0.1:18037")],
    }
}

#[test]
fn ui_disabled_resolves_to_no_policy() {
    let mut config = loopback_config(false, false);
    config.enabled = false;
    assert!(resolve_webui_security(config).unwrap().is_none());
}

#[test]
fn loopback_interactive_generates_redacted_session_key() {
    let resolved = resolve_webui_security(loopback_config(false, true))
        .unwrap()
        .expect("policy");
    let credential = resolved.generated_credential.expect("generated key");
    let debug = format!("{credential:?}");
    let secret = credential.into_terminal_secret();
    assert!(secret.starts_with("mlxcel_webui_"));
    assert!(secret.len() >= 64);
    assert_eq!(debug, "WebUiSessionCredential(<redacted>)");
}

#[test]
fn loopback_noninteractive_requires_configured_key() {
    let err = resolve_webui_security(loopback_config(false, false)).unwrap_err();
    assert!(err.to_string().contains("interactive loopback terminal"));
}

#[test]
fn nonloopback_requires_key_and_tls() {
    let mut config = loopback_config(true, false);
    config.listen = WebUiListenKind::NonLoopbackTcp;
    config.tls_enabled = false;
    let err = resolve_webui_security(config).unwrap_err();
    assert!(err.to_string().contains("explicit API key and TLS"));
}

#[test]
fn unix_socket_is_rejected_for_webui() {
    let mut config = loopback_config(true, false);
    config.listen = WebUiListenKind::UnixSocket;
    let err = resolve_webui_security(config).unwrap_err();
    assert!(err.to_string().contains("TCP listener"));
}

#[test]
fn host_authorities_are_canonicalized_and_validated() {
    assert_eq!(
        canonical_authority("LOCALHOST:18037").as_deref(),
        Some("localhost:18037")
    );
    assert_eq!(
        canonical_authority("[::1]:18037").as_deref(),
        Some("[::1]:18037")
    );
    assert!(canonical_authority("localhost,evil.example").is_none());
    assert!(canonical_authority("http://localhost:18037").is_none());
    assert!(canonical_authority("localhost:99999").is_none());
}

#[test]
fn query_credentials_are_detected_case_insensitively() {
    assert!(query_carries_credential(Some("model=a&API_KEY=secret")));
    assert!(query_carries_credential(Some("api%5Fkey=secret")));
    assert!(query_carries_credential(Some("access_token=secret")));
    assert!(!query_carries_credential(Some("model=tokenizer")));
}

#[test]
fn custom_prefixes_classify_private_api_and_public_shell() {
    let policy = WebUiSecurityPolicy::with_prefixes_and_limits(
        vec!["127.0.0.1:18037".to_string()],
        vec![origin("http://127.0.0.1:18037")],
        "/admin/webui",
        "/admin/api",
        1,
        1,
    )
    .unwrap();
    let public = classify_request(
        &policy,
        &axum::http::Method::GET,
        &"/admin/webui/".parse().unwrap(),
    );
    assert_eq!(public.kind, RequestKind::Public);
    let events = classify_request(
        &policy,
        &axum::http::Method::GET,
        &"/admin/api/events".parse().unwrap(),
    );
    assert!(events.sse);
    let action = classify_request(
        &policy,
        &axum::http::Method::POST,
        &"/admin/api/model-actions".parse().unwrap(),
    );
    assert!(action.control);
}

#[test]
fn allowed_origins_are_strict_origins() {
    let err = WebUiSecurityPolicy::new(
        vec!["127.0.0.1:18037".to_string()],
        vec![origin("http://127.0.0.1:18037/path")],
    )
    .unwrap_err();
    assert!(err.to_string().contains("path or query"));
}
