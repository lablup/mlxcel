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

#![allow(dead_code)]
//! Security policy and middleware for the bundled WebUI administrative surface.

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode, header};
use axum::middleware;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use percent_encoding::percent_decode_str;
use tokio::sync::OwnedSemaphorePermit;

use crate::server::ApiKeys;
use crate::server::auth;
use crate::server::router_lifecycle::{ErrorBody, ErrorEnvelope};

const QUERY_CREDENTIAL_NAMES: &[&str] = &["api_key", "key", "token", "access_token", "auth"];

mod policy;
mod request;
use request::{Decision, RequestKind, classify_request};
pub(crate) mod startup;

pub(crate) use policy::{
    CONTENT_SECURITY_POLICY, PERMISSIONS_POLICY, WEBUI_CONTROL_BODY_LIMIT_BYTES,
    WebUiSecurityPolicy, apply_security_headers,
};
#[derive(Clone, Debug)]
pub(crate) struct WebUiSecurityState {
    pub(crate) api_keys: ApiKeys,
    pub(crate) policy: WebUiSecurityPolicy,
}

impl WebUiSecurityState {
    pub(crate) fn new(api_keys: ApiKeys, policy: WebUiSecurityPolicy) -> Self {
        Self { api_keys, policy }
    }
}

pub(crate) fn secure_webui_router<S>(
    router: Router<S>,
    api_keys: ApiKeys,
    policy: WebUiSecurityPolicy,
) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router.layer(middleware::from_fn_with_state(
        WebUiSecurityState::new(api_keys, policy),
        webui_security_middleware,
    ))
}

pub(crate) async fn webui_security_middleware(
    State(security): State<WebUiSecurityState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let decision = classify_request(&security.policy, request.method(), request.uri());
    if let Err(rejection) = validate_browser_metadata(
        &security.policy,
        &request,
        decision.kind == RequestKind::Public,
    ) {
        return rejection.into_response();
    }
    if query_carries_credential(request.uri().query()) {
        return webui_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "credentials must be sent in Authorization or X-Api-Key headers, not query parameters",
            false,
        );
    }
    let request = request;
    let permit = match decision.kind {
        RequestKind::Public => None,
        RequestKind::PrivatePreflight => None,
        RequestKind::Private => match auth::presented_credential(request.headers()) {
            Some(presented) if security.api_keys.accepts(presented) => {
                match acquire_permit(&security.policy, &decision) {
                    Ok(permit) => permit,
                    Err(rejection) => return rejection.into_response(),
                }
            }
            _ => return unauthorized_response(&security.policy, request.uri().path()),
        },
    };
    if decision.control && request_body_too_large(request.headers()) {
        return webui_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "WebUI control request body is too large",
            true,
        );
    }
    if decision.control && !security.policy.try_record_control_request() {
        return webui_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limited",
            "WebUI administrative request rate is exhausted",
            true,
        );
    }
    let request = if decision.control {
        match buffer_limited_control_body(request).await {
            Ok(request) => request,
            Err(response) => return response,
        }
    } else {
        request
    };
    let mut response = next.run(request).await;
    apply_security_headers(response.headers_mut());
    if let Some(permit) = permit {
        response = hold_permit_until_body_drop(response, permit);
    }
    response
}

fn validate_browser_metadata(
    policy: &WebUiSecurityPolicy,
    request: &Request<Body>,
    public: bool,
) -> std::result::Result<(), SecurityRejection> {
    validate_host(policy, request)?;
    validate_origin(policy, request.headers(), public)?;
    validate_fetch_metadata(request.headers(), public)?;
    Ok(())
}

fn validate_host(
    policy: &WebUiSecurityPolicy,
    request: &Request<Body>,
) -> std::result::Result<(), SecurityRejection> {
    let hosts: Vec<_> = request.headers().get_all(header::HOST).iter().collect();
    if hosts.len() != 1 {
        return Err(forbidden(
            "invalid_host",
            "exactly one Host header is required",
        ));
    }
    let host = header_to_str(hosts[0])
        .and_then(canonical_authority)
        .ok_or_else(|| forbidden("invalid_host", "Host header is malformed"))?;
    if !policy.allowed_hosts.iter().any(|allowed| allowed == &host) {
        return Err(forbidden("invalid_host", "Host header is not allowed"));
    }
    if let Some(authority) = request.uri().authority()
        && canonical_authority(authority.as_str()).as_deref() != Some(host.as_str())
    {
        return Err(forbidden(
            "invalid_host",
            "request authority conflicts with Host",
        ));
    }
    Ok(())
}

fn validate_origin(
    policy: &WebUiSecurityPolicy,
    headers: &HeaderMap,
    public: bool,
) -> std::result::Result<Option<HeaderValue>, SecurityRejection> {
    let origins: Vec<_> = headers.get_all(header::ORIGIN).iter().collect();
    if origins.len() > 1 {
        return Err(forbidden(
            "invalid_origin",
            "multiple Origin headers are not allowed",
        ));
    }
    let Some(origin) = origins.first() else {
        return Ok(None);
    };
    if public {
        return Ok(Some((*origin).clone()));
    }
    let origin_text = header_to_str(origin)
        .ok_or_else(|| forbidden("invalid_origin", "Origin header is malformed"))?;
    if origin_text == "null"
        || !policy
            .allowed_origins
            .iter()
            .any(|allowed| allowed == *origin)
    {
        return Err(forbidden("invalid_origin", "Origin header is not allowed"));
    }
    Ok(Some((*origin).clone()))
}

fn validate_fetch_metadata(
    headers: &HeaderMap,
    public: bool,
) -> std::result::Result<(), SecurityRejection> {
    let site = headers.get("sec-fetch-site").and_then(header_to_str);
    if !public && matches!(site, Some("cross-site" | "same-site")) {
        return Err(forbidden(
            "forbidden_fetch_metadata",
            "cross-site browser requests are not allowed",
        ));
    }
    Ok(())
}

fn acquire_permit(
    policy: &WebUiSecurityPolicy,
    decision: &Decision,
) -> std::result::Result<Option<OwnedSemaphorePermit>, SecurityRejection> {
    let semaphore = if decision.control {
        &policy.control_permits
    } else if decision.sse {
        &policy.sse_permits
    } else {
        return Ok(None);
    };
    semaphore
        .clone()
        .try_acquire_owned()
        .map(Some)
        .map_err(|_| {
            rejection(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "WebUI administrative capacity is exhausted",
                true,
            )
        })
}

async fn buffer_limited_control_body(
    request: Request<Body>,
) -> std::result::Result<Request<Body>, Response> {
    let (parts, body) = request.into_parts();
    to_bytes(body, WEBUI_CONTROL_BODY_LIMIT_BYTES)
        .await
        .map(|bytes| Request::from_parts(parts, Body::from(bytes)))
        .map_err(|_| {
            webui_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "WebUI control request body is too large",
                true,
            )
        })
}

fn request_body_too_large(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|len| len > WEBUI_CONTROL_BODY_LIMIT_BYTES)
}

fn hold_permit_until_body_drop(response: Response, permit: OwnedSemaphorePermit) -> Response {
    response.map(|body| {
        let stream = body.into_data_stream().map(move |chunk| {
            let _keep_alive = &permit;
            chunk
        });
        Body::from_stream(stream)
    })
}

fn query_carries_credential(query: Option<&str>) -> bool {
    query.is_some_and(|query| {
        query.split('&').any(|part| {
            let name = part.split_once('=').map_or(part, |(name, _)| name);
            let decoded_name = percent_decode_str(name).decode_utf8_lossy();
            QUERY_CREDENTIAL_NAMES
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(decoded_name.as_ref()))
        })
    })
}

fn query_has_parameter(query: Option<&str>, expected: &str) -> bool {
    query.is_some_and(|query| {
        query.split('&').any(|part| {
            let name = part.split_once('=').map_or(part, |(name, _)| name);
            percent_decode_str(name)
                .decode_utf8_lossy()
                .eq_ignore_ascii_case(expected)
        })
    })
}

fn api_relative_path<'a>(policy: &WebUiSecurityPolicy, path: &'a str) -> Option<&'a str> {
    if policy.api_prefix.as_ref() == "/" {
        return Some(path);
    }
    path.strip_prefix(policy.api_prefix.as_ref())
        .and_then(|tail| {
            if tail.is_empty() {
                Some("/")
            } else if tail.starts_with('/') {
                Some(tail)
            } else {
                None
            }
        })
}

fn is_ui_api_path(policy: &WebUiSecurityPolicy, path: &str) -> bool {
    api_relative_path(policy, path)
        .is_some_and(|relative| path_matches_prefix(relative, "/ui-api/v1"))
}

fn path_matches_prefix(path: &str, prefix: &str) -> bool {
    path == prefix
        || (prefix != "/"
            && path
                .strip_prefix(prefix)
                .is_some_and(|tail| tail.starts_with('/')))
}

fn canonical_authority(raw: &str) -> Option<String> {
    let value = raw.trim();
    if value.is_empty()
        || value.contains(',')
        || value.contains('/')
        || value.contains('\\')
        || value.contains('@')
        || value
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace())
    {
        return None;
    }
    if let Some(end) = value.find(']') {
        if !value.starts_with('[') {
            return None;
        }
        let host = value[..=end].to_ascii_lowercase();
        let tail = &value[end + 1..];
        return if tail.is_empty() || valid_port_tail(tail) {
            Some(format!("{host}{tail}"))
        } else {
            None
        };
    }
    let (host, port) = value
        .rsplit_once(':')
        .map_or((value, ""), |(host, port)| (host, port));
    if host.is_empty() || host.contains(':') || (!port.is_empty() && port.parse::<u16>().is_err()) {
        return None;
    }
    Some(if port.is_empty() {
        host.to_ascii_lowercase()
    } else {
        format!("{}:{port}", host.to_ascii_lowercase())
    })
}

fn valid_port_tail(tail: &str) -> bool {
    tail.strip_prefix(':')
        .is_some_and(|port| !port.is_empty() && port.parse::<u16>().is_ok())
}

fn header_to_str(value: &HeaderValue) -> Option<&str> {
    value.to_str().ok().filter(|text| !text.is_empty())
}

fn unauthorized_response(policy: &WebUiSecurityPolicy, path: &str) -> Response {
    if is_ui_api_path(policy, path) {
        return webui_error(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "authentication is required",
            false,
        );
    }
    let mut response = auth::unauthorized_response();
    apply_security_headers(response.headers_mut());
    response
}

#[derive(Debug, Clone, Copy)]
struct SecurityRejection {
    status: StatusCode,
    code: &'static str,
    message: &'static str,
    retryable: bool,
}

impl SecurityRejection {
    fn into_response(self) -> Response {
        webui_error(self.status, self.code, self.message, self.retryable)
    }
}

fn forbidden(_code: &'static str, message: &'static str) -> SecurityRejection {
    rejection(StatusCode::FORBIDDEN, "forbidden", message, false)
}

fn rejection(
    status: StatusCode,
    code: &'static str,
    message: &'static str,
    retryable: bool,
) -> SecurityRejection {
    SecurityRejection {
        status,
        code,
        message,
        retryable,
    }
}

fn webui_error(
    status: StatusCode,
    code: &str,
    message: impl Into<String>,
    retryable: bool,
) -> Response {
    let mut response = (
        status,
        axum::Json(ErrorEnvelope {
            error: ErrorBody {
                code: code.to_string(),
                message: message.into(),
                retryable,
                field_errors: None,
                operation_id: None,
            },
            request_id: format!("req_{}", chrono::Utc::now().timestamp_micros()),
        }),
    )
        .into_response();
    apply_security_headers(response.headers_mut());
    response
}

#[cfg(test)]
#[path = "security_tests.rs"]
mod security_tests;

#[cfg(test)]
#[path = "security_body_boundary_tests.rs"]
mod security_body_boundary_tests;
