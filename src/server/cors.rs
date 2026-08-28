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

//! CORS policy for the HTTP server (#244, realigned onto b10621 in #1432).
//!
//! `llama-server` b10621 implements CORS in one pre-routing handler
//! (upstream `tools/server/server-http.cpp`, `set_pre_routing_handler`):
//!
//! - `Access-Control-Allow-Origin` is set on **every** response. With
//!   `--cors-origins *` (the default) and credentials enabled it echoes the
//!   request `Origin`; with the special value `localhost` it echoes the
//!   `Origin` only when its host is `localhost`, `127.0.0.1` or `::1`;
//!   otherwise it emits the configured string verbatim.
//! - `OPTIONS` is answered by the middleware itself, before authentication and
//!   before routing, with `Access-Control-Allow-Credentials`,
//!   `-Allow-Methods` and `-Allow-Headers` and an empty `text/html` body.
//! - No other CORS response header is emitted, in particular no
//!   `Access-Control-Expose-Headers`.
//!
//! [`CorsPolicy`] reproduces that, and adds one mlxcel-native origin mode:
//! [`OriginPolicy::AllowList`], reached through `--allowed-origins`
//! (`MLXCEL_ALLOWED_ORIGINS`, #244). b10621's `--cors-origins` echoes its
//! configured string unchanged, so a comma-separated list there produces an
//! `Access-Control-Allow-Origin` no browser accepts; `--allowed-origins`
//! instead reflects the request `Origin` when it matches one of a validated
//! set. The two spellings are mutually exclusive at the CLI, so exactly one
//! origin rule is ever in force.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, Method, Request, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use super::AppState;
use super::transport::{DEFAULT_CORS_HEADERS, DEFAULT_CORS_METHODS, DEFAULT_CORS_ORIGINS};

/// b10621's special `--cors-origins` value that reflects only localhost
/// origins.
pub(crate) const CORS_ORIGINS_LOCALHOST: &str = "localhost";

/// How the `Access-Control-Allow-Origin` value is decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OriginPolicy {
    /// b10621 `--cors-origins *`. With credentials enabled the request
    /// `Origin` is echoed back; with credentials disabled the literal `*` is
    /// emitted, which is what upstream's `else` branch produces.
    Wildcard,
    /// b10621 `--cors-origins localhost`: reflect the request `Origin` only
    /// when its host is `localhost`, `127.0.0.1` or `::1`, at any port.
    Localhost,
    /// b10621 `--cors-origins <anything else>`: emit the configured string
    /// verbatim, whatever the request `Origin` is.
    Literal(HeaderValue),
    /// mlxcel `--allowed-origins` (#244): reflect the request `Origin` only
    /// when it is one of these validated origins, and emit nothing otherwise.
    AllowList(Vec<HeaderValue>),
}

/// The resolved CORS policy, built once at startup and consumed per request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorsPolicy {
    pub origins: OriginPolicy,
    /// `Access-Control-Allow-Methods` sent on a preflight.
    pub methods: HeaderValue,
    /// `Access-Control-Allow-Headers` sent on a preflight.
    pub headers: HeaderValue,
    /// `Access-Control-Allow-Credentials` sent on a preflight, and the switch
    /// that turns `*` into an echoed `Origin`.
    pub credentials: bool,
}

impl Default for CorsPolicy {
    fn default() -> Self {
        Self {
            origins: OriginPolicy::Wildcard,
            methods: HeaderValue::from_static(DEFAULT_CORS_METHODS),
            headers: HeaderValue::from_static(DEFAULT_CORS_HEADERS),
            credentials: true,
        }
    }
}

impl CorsPolicy {
    /// Build the policy from the resolved CLI values.
    ///
    /// `allow_list` is the validated `--allowed-origins` set; when it is
    /// non-empty it selects [`OriginPolicy::AllowList`] and `cors_origins` is
    /// ignored, which is safe because the CLI rejects setting both.
    pub fn resolve(
        cors_origins: &str,
        cors_methods: &str,
        cors_headers: &str,
        credentials: bool,
        allow_list: Option<Vec<HeaderValue>>,
    ) -> anyhow::Result<Self> {
        let header_value = |flag: &str, raw: &str| -> anyhow::Result<HeaderValue> {
            HeaderValue::from_str(raw).map_err(|_| {
                anyhow::anyhow!(
                    "invalid {flag} value {raw:?}: it must be a valid HTTP header value (no \
                     control characters and no newlines)"
                )
            })
        };

        let origins = match allow_list {
            Some(list) if !list.is_empty() => OriginPolicy::AllowList(list),
            _ => match cors_origins.trim() {
                DEFAULT_CORS_ORIGINS => OriginPolicy::Wildcard,
                CORS_ORIGINS_LOCALHOST => OriginPolicy::Localhost,
                other => OriginPolicy::Literal(header_value("--cors-origins", other)?),
            },
        };

        Ok(Self {
            origins,
            methods: header_value("--cors-methods", cors_methods)?,
            headers: header_value("--cors-headers", cors_headers)?,
            credentials,
        })
    }

    /// The `Access-Control-Allow-Origin` value for a request carrying
    /// `origin`, or `None` when the header is omitted entirely.
    ///
    /// Under [`OriginPolicy::Wildcard`] with credentials enabled the value is
    /// the request `Origin` verbatim, including the empty string when the
    /// request carries no `Origin`. That empty header is what b10621 emits
    /// (`res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"))`
    /// on a request with no `Origin` yields an empty value), and reproducing
    /// it keeps a non-browser client's view of the two servers identical.
    pub(crate) fn allow_origin(&self, origin: Option<&HeaderValue>) -> Option<HeaderValue> {
        match &self.origins {
            OriginPolicy::Wildcard if self.credentials => {
                Some(origin.cloned().unwrap_or_else(empty_header_value))
            }
            OriginPolicy::Wildcard => Some(HeaderValue::from_static(DEFAULT_CORS_ORIGINS)),
            OriginPolicy::Localhost => match origin {
                Some(value) if origin_is_localhost(value) => Some(value.clone()),
                Some(value) => {
                    tracing::warn!(
                        "(CORS) skip non-localhost origin: {}",
                        value.to_str().unwrap_or("<non-utf8>")
                    );
                    None
                }
                None => None,
            },
            OriginPolicy::Literal(value) => Some(value.clone()),
            OriginPolicy::AllowList(list) => match origin {
                Some(value) if list.contains(value) => Some(value.clone()),
                _ => None,
            },
        }
    }

    /// The three headers b10621 adds to a preflight response.
    pub(crate) fn preflight_headers(&self) -> [(HeaderName, HeaderValue); 3] {
        [
            (
                header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
                HeaderValue::from_static(if self.credentials { "true" } else { "false" }),
            ),
            (header::ACCESS_CONTROL_ALLOW_METHODS, self.methods.clone()),
            (header::ACCESS_CONTROL_ALLOW_HEADERS, self.headers.clone()),
        ]
    }
}

fn empty_header_value() -> HeaderValue {
    HeaderValue::from_static("")
}

/// True when `origin`'s host is `localhost`, `127.0.0.1` or `::1`, at any
/// port, matching upstream `origin_is_localhost`.
///
/// Upstream parses the origin as a URL and compares the host component, so a
/// value such as `https://localhost.evil.com` does not match and neither does
/// a bare `localhost` with no scheme.
fn origin_is_localhost(origin: &HeaderValue) -> bool {
    let Ok(text) = origin.to_str() else {
        return false;
    };
    let Some((_scheme, rest)) = text.split_once("://") else {
        return false;
    };
    // Strip userinfo, then any path/query/fragment, then the port. An IPv6
    // literal is bracketed, so the port split has to happen after the bracket.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = if let Some(end) = host_port.find(']') {
        host_port
            .get(1..end)
            .filter(|_| host_port.starts_with('['))
            .unwrap_or("")
    } else {
        host_port.split_once(':').map_or(host_port, |(h, _)| h)
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Axum middleware implementing the b10621 pre-routing CORS handler.
///
/// Mounted OUTSIDE the API-key middleware so a preflight is answered without
/// credentials, exactly as upstream does ("browsers don't include
/// Authorization header"). Because it runs before routing, an `OPTIONS` to an
/// unrouted path is answered 200 here rather than 404, which is also what the
/// upstream pre-routing handler does.
///
/// Used by: `crate::server::create_app`.
pub(crate) async fn cors_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // The policy itself stays behind the shared `Arc<ServerConfig>`; only the
    // handful of header values a response actually carries is cloned.
    apply_cors_policy(&state.config.cors_policy, request, next).await
}

/// The policy application shared by the single-model middleware above and the
/// router-mode top level (issue #1438), which carries its own state type.
pub(crate) async fn apply_cors_policy(
    policy: &CorsPolicy,
    request: Request<Body>,
    next: Next,
) -> Response {
    let allow_origin = policy.allow_origin(request.headers().get(header::ORIGIN));

    if request.method() == Method::OPTIONS {
        let preflight = policy.preflight_headers();
        let mut response = (
            StatusCode::OK,
            [(header::CONTENT_TYPE, HeaderValue::from_static("text/html"))],
            "",
        )
            .into_response();
        for (name, value) in preflight {
            response.headers_mut().insert(name, value);
        }
        if let Some(value) = allow_origin {
            response
                .headers_mut()
                .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
        }
        return response;
    }

    let mut response = next.run(request).await;
    if let Some(value) = allow_origin {
        response
            .headers_mut()
            .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
    }
    response
}

/// Parse and validate comma-split origin strings into header values.
///
/// Each entry is trimmed. Blank entries (for example a stray trailing comma)
/// are skipped, but a value that contains nothing but blank entries is a
/// configuration error rather than a silent "no origins" result, because the
/// operator clearly intended to set a policy. Every surviving entry must be a
/// bare origin: a `scheme://host[:port]` with an `http`/`https` scheme and an
/// authority, and nothing else: no path (not even a bare trailing slash), no
/// query, no fragment, no userinfo, and no control characters. A browser
/// `Origin` header never carries any of those, so a value that included one
/// could only ever silently never match; rejecting it at startup surfaces the
/// misconfiguration instead. Valid values are preserved verbatim (only
/// trimmed) so they match the browser-sent `Origin` header exactly.
///
/// Returns an [`Err`] naming the offending value on the first invalid entry,
/// so the failure is surfaced clearly at startup instead of being dropped.
pub(crate) fn parse_allowed_origins(raw: &[String]) -> anyhow::Result<Vec<HeaderValue>> {
    let mut origins = Vec::new();
    for entry in raw {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        validate_origin(trimmed)?;
        let value = HeaderValue::from_str(trimmed).map_err(|_| {
            anyhow::anyhow!(
                "invalid --allowed-origins value '{trimmed}': origins must be a bare \
                 scheme://host[:port] with no path, query, or userinfo, e.g. \
                 https://app.example.com"
            )
        })?;
        origins.push(value);
    }

    if !raw.is_empty() && origins.is_empty() {
        anyhow::bail!(
            "--allowed-origins was set but contained no usable origin (every entry was \
             blank); remove the flag to keep the default permissive policy, or provide at \
             least one origin like https://app.example.com"
        );
    }

    Ok(origins)
}

/// Validate that `value` is a bare origin (`scheme://host[:port]`).
///
/// An origin has a scheme, an authority, and nothing else: no path, query,
/// fragment, or userinfo. `http::Uri` normalizes an empty path to `/`, so it
/// cannot tell `https://host` apart from `https://host/`; we therefore inspect
/// the raw authority substring directly to reject a trailing slash, path,
/// query, fragment, or `user@` userinfo (none of which a browser `Origin` ever
/// carries, so any of them could only ever silently never match), then parse
/// with [`axum::http::Uri`] to confirm a known scheme, a well-formed authority,
/// and the absence of control characters.
fn validate_origin(value: &str) -> anyhow::Result<()> {
    let bad = || {
        anyhow::anyhow!(
            "invalid --allowed-origins value '{value}': origins must be a bare \
             scheme://host[:port] with no path, query, or userinfo, e.g. \
             https://app.example.com"
        )
    };

    // Structural check on the raw string. The authority is everything after the
    // first `://`; a browser `Origin` is exactly `scheme://host[:port]`, so the
    // authority must not contain a path separator (`/`, which also catches a
    // bare trailing slash), a query (`?`), a fragment (`#`), or userinfo (`@`).
    // This is done on the raw text because `http::Uri::path()` reports an empty
    // path as `/`, hiding a configured trailing slash from a parsed-path check.
    let authority_raw = value
        .split_once("://")
        .map(|(_, rest)| rest)
        .ok_or_else(bad)?;
    if authority_raw.is_empty()
        || authority_raw.contains('/')
        || authority_raw.contains('?')
        || authority_raw.contains('#')
        || authority_raw.contains('@')
    {
        return Err(bad());
    }

    // Structural parse: confirm an http/https scheme, a well-formed authority,
    // and reject control characters or otherwise malformed authorities.
    let uri: axum::http::Uri = value.parse().map_err(|_| bad())?;
    let scheme_ok = matches!(uri.scheme_str(), Some("http") | Some("https"));
    let has_authority = uri.authority().is_some();

    if scheme_ok && has_authority {
        Ok(())
    } else {
        Err(bad())
    }
}

#[cfg(test)]
#[path = "cors_tests.rs"]
mod cors_tests;
