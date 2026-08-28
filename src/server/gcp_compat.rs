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

//! Google Cloud Vertex AI custom-container compatibility (b10621, #1456).
//!
//! b10621 registers this surface purely from environment variables (upstream
//! <https://github.com/ggml-org/llama.cpp/blob/main/tools/server/server-http.cpp>,
//! `gcp_params` and `register_gcp_compat`): with `AIP_MODE=PREDICTION`, the
//! server serves `AIP_HTTP_PORT` (default 8080, overriding `--port` with a
//! warning), optionally registers `AIP_HEALTH_ROUTE` as a GET alias of the
//! health handler, and mounts `AIP_PREDICT_ROUTE` (default `/predict`), which
//! fans a `{"instances": [{"@requestFormat": "chatCompletions", ...}]}` batch
//! out over the camelCase alias of every registered route and answers
//! `{"predictions": [...]}` in request order. Without `AIP_MODE=PREDICTION`
//! nothing is registered and the other variables are ignored.
//!
//! mlxcel implements the same adapter over its real route table:
//!
//! - The alias table derives from [`crate::server::app::route_inventory`],
//!   the single source the router itself is built from, so a route added
//!   there is aliased automatically. First registration wins on an alias
//!   collision, which in mlxcel's deterministic registration order resolves
//!   `completions` and `embeddings` to the OpenAI `/v1` handlers (upstream
//!   iterates an unordered map, so its winner on those collisions is
//!   unspecified).
//! - Each instance is dispatched through the composed router in-process
//!   (`tower::ServiceExt::oneshot` against the same app the socket serves),
//!   with the predict request's own headers, so API-key authentication,
//!   validation, and queue admission apply exactly as they do to a direct
//!   call of the aliased route.
//! - Instance execution is bounded to [`GCP_PREDICT_MAX_CONCURRENT`] at a
//!   time (upstream launches all of them at once); results keep request
//!   order either way, and the bound keeps one predict batch from flooding
//!   the decode queue.
//!
//! As in b10621, neither the predict route nor the health alias is in the
//! public (unauthenticated) endpoint set; with API keys configured both
//! require a key. The Vertex AI variables are documented in
//! `docs/environment-variables.md`.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use tower::ServiceExt;

use super::AppState;
use super::types::ErrorResponse;

/// b10621 caps one predict request at this many instances.
pub(crate) const GCP_MAX_INSTANCES: usize = 128;

/// mlxcel-side bound on concurrently executing instances of one predict
/// batch. Upstream runs every instance on its own thread at once; bounding
/// keeps a full 128-instance batch from monopolising queue admission while
/// still preserving result order.
pub(crate) const GCP_PREDICT_MAX_CONCURRENT: usize = 8;

/// Upper bound when collecting an internal handler's response body.
const GCP_INTERNAL_BODY_LIMIT: usize = 64 * 1024 * 1024;

/// The route-level part of the resolved Vertex AI configuration, carried on
/// [`crate::server::ServerConfig::gcp`]. `None` there means the adapter is
/// off (the b10621 default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcpRoutes {
    /// `AIP_HEALTH_ROUTE`, leading slash ensured; `None` when unset or empty
    /// (upstream then relies on the ordinary `/health`).
    pub path_health: Option<String>,
    /// `AIP_PREDICT_ROUTE`, leading slash ensured; defaults to `/predict`.
    pub path_predict: String,
}

/// Everything `AIP_*` resolves to at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcpResolution {
    pub routes: GcpRoutes,
    /// `AIP_HTTP_PORT` (default 8080). Overrides `--port`; the caller logs
    /// the b10621 warning when the two differ.
    pub port: u16,
}

fn env_with_leading_slash(name: &str, default_value: &str) -> String {
    let value = std::env::var(name).unwrap_or_default();
    let value = if value.is_empty() {
        default_value.to_string()
    } else {
        value
    };
    if !value.is_empty() && !value.starts_with('/') {
        format!("/{value}")
    } else {
        value
    }
}

/// Resolve the `AIP_*` variables, exactly as upstream's `gcp_params` does:
/// only `AIP_MODE=PREDICTION` activates anything; otherwise every other
/// variable is ignored and `Ok(None)` is returned.
///
/// One deliberate improvement over upstream: an unparsable `AIP_HTTP_PORT`
/// fails startup with a diagnostic instead of upstream's uncaught `std::stoi`
/// exception.
pub fn resolve_from_env() -> Result<Option<GcpResolution>> {
    if std::env::var("AIP_MODE").unwrap_or_default() != "PREDICTION" {
        return Ok(None);
    }
    let path_health = {
        let v = env_with_leading_slash("AIP_HEALTH_ROUTE", "");
        if v.is_empty() { None } else { Some(v) }
    };
    let path_predict = env_with_leading_slash("AIP_PREDICT_ROUTE", "/predict");
    let port_raw = std::env::var("AIP_HTTP_PORT").unwrap_or_default();
    let port: u16 = if port_raw.is_empty() {
        8080
    } else {
        port_raw.trim().parse().with_context(|| {
            format!(
                "AIP_HTTP_PORT={port_raw:?} is not a valid TCP port; Vertex AI \
                 custom containers must listen on the port this variable names"
            )
        })?
    };
    Ok(Some(GcpResolution {
        routes: GcpRoutes {
            path_health,
            path_predict,
        },
        port,
    }))
}

/// Refuse an `AIP_PREDICT_ROUTE` that collides with a registered route,
/// before the model loads, as upstream's `register_gcp_compat` exits on the
/// same condition.
pub(crate) fn check_predict_collision(config: &crate::server::ServerConfig) -> Result<()> {
    let Some(gcp) = config.gcp.as_ref() else {
        return Ok(());
    };
    let taken = super::app::route_inventory(config)
        .into_iter()
        .any(|reg| reg.path == gcp.path_predict);
    if taken {
        bail!(
            "AIP_PREDICT_ROUTE={} conflicts with an existing mlxcel-server \
             route; pick a path no API route uses (the b10621 default is \
             /predict)",
            gcp.path_predict
        );
    }
    Ok(())
}

/// Derive the camelCase `@requestFormat` alias for a registered path, the
/// port of upstream's `path_to_gcp_format`: strip a leading `/v1`, strip the
/// leading `/`, stop before the first `:` path parameter, and capitalise
/// after `/`, `-`, `_` boundaries (`/v1/chat/completions` becomes
/// `chatCompletions`, `/apply-template` becomes `applyTemplate`).
pub(crate) fn path_to_gcp_format(path: &str) -> String {
    let mut s = path;
    if s.len() > 3 && s.starts_with("/v1") {
        s = &s[3..];
    }
    s = s.strip_prefix('/').unwrap_or(s);
    let mut result = String::with_capacity(s.len());
    let mut cap = false;
    for c in s.chars() {
        if c == ':' {
            break;
        }
        if c == '/' || c == '-' || c == '_' {
            cap = true;
        } else {
            if cap {
                result.extend(c.to_uppercase());
            } else {
                result.push(c);
            }
            cap = false;
        }
    }
    result
}

/// One dispatchable target: the canonical path plus the method the route was
/// registered to serve (POST preferred when a path serves several).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DispatchTarget {
    pub(crate) path: String,
    pub(crate) method: Method,
}

/// Alias table plus the direct-path set, both derived from the route
/// inventory in registration order. First registration wins on an alias
/// collision, as upstream's `emplace` does; empty aliases (the `/` root) are
/// skipped.
pub(crate) fn dispatch_table(
    config: &crate::server::ServerConfig,
) -> (HashMap<String, DispatchTarget>, HashMap<String, Method>) {
    let mut aliases: HashMap<String, DispatchTarget> = HashMap::new();
    let mut direct: HashMap<String, Method> = HashMap::new();
    for reg in super::app::route_inventory(config) {
        let alias = path_to_gcp_format(reg.path);
        if !alias.is_empty() {
            aliases.entry(alias).or_insert_with(|| DispatchTarget {
                path: reg.path.to_string(),
                method: reg.dispatch_method.clone(),
            });
        }
        direct
            .entry(reg.path.to_string())
            .or_insert(reg.dispatch_method);
    }
    (aliases, direct)
}

fn envelope_error(message: impl Into<String>) -> Response {
    ErrorResponse::new(message, "invalid_request_error").into_response()
}

/// Per-instance error object, upstream's `{"error": format_error_response(..)}`
/// shape with mlxcel's error detail body.
fn instance_error(message: impl Into<String>, error_type: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message.into(),
            "type": error_type,
            "code": null,
        }
    })
}

/// `POST ${AIP_PREDICT_ROUTE:-/predict}` (b10621 `register_gcp_compat`).
pub(crate) async fn predict(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let data: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return envelope_error(e.to_string()),
    };
    if !data.is_object() {
        return envelope_error("request body must be a JSON object");
    }
    let Some(instances) = data.get("instances").filter(|v| v.is_array()) else {
        return envelope_error("request body must include an array field named instances");
    };
    let instances = instances.as_array().expect("checked is_array");
    if instances.len() > GCP_MAX_INSTANCES {
        return envelope_error(format!(
            "instances array exceeds maximum size of {GCP_MAX_INSTANCES}"
        ));
    }

    // The composed router the socket serves; set by `create_app` before the
    // first request can reach this handler.
    let Some(app) = state.gcp_dispatch.get().cloned() else {
        let mut err =
            ErrorResponse::new("predict dispatch router is not initialised", "server_error");
        err.status = StatusCode::INTERNAL_SERVER_ERROR;
        return err.into_response();
    };
    let (aliases, direct) = dispatch_table(&state.config);
    let api_prefix = state.config.api_prefix.clone();

    let futures = instances.iter().cloned().map(|instance| {
        let app = app.clone();
        let aliases = &aliases;
        let direct = &direct;
        let headers = &headers;
        let api_prefix = api_prefix.as_str();
        async move { dispatch_instance(instance, app, aliases, direct, headers, api_prefix).await }
    });
    let predictions: Vec<serde_json::Value> = futures::stream::iter(futures)
        .buffered(GCP_PREDICT_MAX_CONCURRENT)
        .collect()
        .await;

    axum::Json(serde_json::json!({ "predictions": predictions })).into_response()
}

/// Execute one instance; any failure becomes an error object in its slot,
/// never a whole-batch failure, matching upstream.
async fn dispatch_instance(
    instance: serde_json::Value,
    app: axum::Router,
    aliases: &HashMap<String, DispatchTarget>,
    direct: &HashMap<String, Method>,
    headers: &HeaderMap,
    api_prefix: &str,
) -> serde_json::Value {
    let Some(obj) = instance.as_object() else {
        return instance_error(
            "each instance must be a JSON object",
            "invalid_request_error",
        );
    };
    let Some(format) = obj.get("@requestFormat").and_then(|v| v.as_str()) else {
        return instance_error(
            "each instance must include a string @requestFormat",
            "invalid_request_error",
        );
    };
    let format = format.to_string();
    let mut payload = obj.clone();
    payload.remove("@requestFormat");
    if payload.contains_key("stream") {
        tracing::warn!(
            "ignoring client-provided stream field in instance, streaming is \
             not supported in predict route"
        );
        payload.insert("stream".to_string(), serde_json::Value::Bool(false));
    }

    // Accept both camelCase aliases and direct registered paths, upstream's
    // two lookups in that order.
    let target = match aliases.get(&format) {
        Some(target) => target.clone(),
        None => match direct.get(&format) {
            Some(method) => DispatchTarget {
                path: format.clone(),
                method: method.clone(),
            },
            None => {
                return instance_error(
                    format!("no handler registered for @requestFormat: {format}"),
                    "invalid_request_error",
                );
            }
        },
    };

    let body = match serde_json::to_vec(&payload) {
        Ok(b) => b,
        Err(e) => return instance_error(e.to_string(), "server_error"),
    };
    let uri = format!("{api_prefix}{}", target.path);
    let mut builder = axum::http::Request::builder()
        .method(target.method.clone())
        .uri(&uri)
        .header(header::CONTENT_TYPE, "application/json");
    // Forward the predict request's own headers (auth included) so the
    // internal dispatch is authorised exactly as a direct call would be.
    for (name, value) in headers {
        if name == header::CONTENT_LENGTH || name == header::CONTENT_TYPE || name == header::HOST {
            continue;
        }
        builder = builder.header(name, value);
    }
    let request = match builder.body(axum::body::Body::from(body)) {
        Ok(r) => r,
        Err(e) => return instance_error(e.to_string(), "server_error"),
    };
    let response = match app.oneshot(request).await {
        Ok(r) => r,
        Err(never) => match never {},
    };

    // A streaming response cannot be represented in a predictions slot;
    // upstream throws the same refusal from its response parser.
    let is_stream = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));
    if is_stream {
        return instance_error(
            "predict route does not support streaming responses",
            "invalid_request_error",
        );
    }
    let bytes = match axum::body::to_bytes(response.into_body(), GCP_INTERNAL_BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => return instance_error(e.to_string(), "server_error"),
    };
    if bytes.is_empty() {
        return serde_json::Value::Null;
    }
    match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        // Not JSON: return the raw text, as upstream's parser fallback does.
        Err(_) => serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned()),
    }
}

#[cfg(test)]
#[path = "gcp_compat_tests.rs"]
mod tests;
