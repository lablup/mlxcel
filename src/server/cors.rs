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

//! CORS policy construction for the HTTP server (#244).
//!
//! The server historically applied [`CorsLayer::permissive`] unconditionally,
//! which reflects any `Origin` and is a CSRF / origin-spoofing surface for any
//! browser-reachable deployment. The `--allowed-origins` flag lets an operator
//! pin the server to a known set of front-end origins. When the flag is unset
//! the permissive default is retained so existing deployments are unaffected.
//!
//! tower-http's CORS middleware never rejects a request; it only governs the
//! `Access-Control-Allow-Origin` response header and preflight handling. A
//! request with no `Origin` header (curl, Unix-domain-socket clients) is
//! unaffected by the restrictive layer, so the same layer is safe on every
//! transport.

use axum::http::HeaderValue;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

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

/// Build the CORS layer for the HTTP router.
///
/// When a non-empty validated origin list is supplied the layer restricts
/// cross-origin requests to exactly those origins while preserving the methods
/// and headers that [`CorsLayer::permissive`] allows; otherwise it falls back
/// to the permissive default. `CorsLayer::permissive()` is exactly
/// `new().allow_headers(Any).allow_methods(Any).allow_origin(Any).expose_headers(Any)`,
/// so the restrictive branch narrows only the origin.
pub(crate) fn build_cors_layer(origins: Option<&[HeaderValue]>) -> CorsLayer {
    match origins {
        Some(list) if !list.is_empty() => CorsLayer::new()
            .allow_headers(Any)
            .allow_methods(Any)
            .allow_origin(AllowOrigin::list(list.iter().cloned()))
            .expose_headers(Any),
        _ => CorsLayer::permissive(),
    }
}

#[cfg(test)]
#[path = "cors_tests.rs"]
mod cors_tests;
