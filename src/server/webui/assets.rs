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

//! Embedded WebUI static asset router.

use axum::{
    Router,
    body::Body,
    extract::OriginalUri,
    http::{HeaderMap, Method, StatusCode, Uri, header},
    response::{IntoResponse, Response},
    routing::any,
};
use rust_embed::RustEmbed;
use sha2::{Digest, Sha256};

const INDEX_HTML_SENTINEL: &str = include_str!("../../webui/assets/index.html");
const LICENSE_SENTINEL: &str = include_str!("../../webui/assets/third-party-licenses.txt");

pub const WEBUI_PREFIX: &str = "/webui";
const INDEX_PATH: &str = "index.html";
const MANIFEST_PATH: &str = "mlxcel-webui-manifest.json";

const WEBUI_ASSET_MANIFEST: &str = include_str!("../../webui/assets/mlxcel-webui-manifest.json");

#[derive(RustEmbed)]
#[folder = "src/webui/assets/"]
struct WebUiAssets;

pub fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    let _ = (INDEX_HTML_SENTINEL, LICENSE_SENTINEL);
    Router::new()
        .route(WEBUI_PREFIX, any(redirect_to_trailing_slash))
        .route(&format!("{WEBUI_PREFIX}/"), any(serve_index))
        .route(&format!("{WEBUI_PREFIX}/*path"), any(serve_asset))
}

pub fn manifest_json() -> &'static str {
    WEBUI_ASSET_MANIFEST
}

async fn redirect_to_trailing_slash(method: Method, OriginalUri(uri): OriginalUri) -> Response {
    if !is_read_method(&method) {
        return method_not_allowed();
    }
    let location = format_uri_with_trailing_slash(&uri);
    response_builder(StatusCode::PERMANENT_REDIRECT)
        .header(header::LOCATION, location)
        .body(Body::empty())
        .unwrap_or_else(internal_error)
}

async fn serve_index(method: Method, uri: Uri, headers: HeaderMap) -> Response {
    if !is_read_method(&method) {
        return method_not_allowed();
    }
    serve_embedded_path(INDEX_PATH, &method, &uri, &headers)
}

async fn serve_asset(
    method: Method,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
) -> Response {
    if !is_read_method(&method) {
        return method_not_allowed();
    }
    if reject_raw_path(uri.path()) {
        return empty_response(StatusCode::BAD_REQUEST);
    }
    let Some(path) = asset_path_from_uri(&uri) else {
        return empty_response(StatusCode::NOT_FOUND);
    };
    if reject_decoded_path(path) {
        return empty_response(StatusCode::BAD_REQUEST);
    }
    serve_embedded_path(path, &method, &uri, &headers)
}

fn serve_embedded_path(path: &str, method: &Method, uri: &Uri, headers: &HeaderMap) -> Response {
    if reject_raw_path(uri.path()) || reject_decoded_path(path) {
        return empty_response(StatusCode::BAD_REQUEST);
    }
    let Some(asset) = WebUiAssets::get(path) else {
        return empty_response(StatusCode::NOT_FOUND);
    };
    let bytes = asset.data.into_owned();
    if path == INDEX_PATH && !index_html_is_bootable(&bytes) {
        return empty_response(StatusCode::INTERNAL_SERVER_ERROR);
    }
    let etag = etag_for(&bytes);
    let cache_control = cache_control_for(path);
    if request_matches_etag(headers, &etag) {
        return response_builder(StatusCode::NOT_MODIFIED)
            .header(header::ETAG, etag)
            .header(header::CACHE_CONTROL, cache_control)
            .body(Body::empty())
            .unwrap_or_else(internal_error);
    }
    let content_type = mime_guess::from_path(path).first_or_octet_stream();
    let body = if method == Method::HEAD {
        Body::empty()
    } else {
        Body::from(bytes)
    };
    response_builder(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type.as_ref())
        .header(header::ETAG, etag)
        .header(header::CACHE_CONTROL, cache_control)
        .body(body)
        .unwrap_or_else(internal_error)
}

fn is_read_method(method: &Method) -> bool {
    method == Method::GET || method == Method::HEAD
}

fn method_not_allowed() -> Response {
    response_builder(StatusCode::METHOD_NOT_ALLOWED)
        .header(header::ALLOW, "GET, HEAD")
        .body(Body::empty())
        .unwrap_or_else(internal_error)
}

fn asset_path_from_uri(uri: &Uri) -> Option<&str> {
    uri.path()
        .rsplit_once(&format!("{WEBUI_PREFIX}/"))
        .map(|(_, path)| path)
}

fn index_html_is_bootable(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_ok_and(|html| {
        html.contains("id=\"root\"")
            && html.contains("type=\"module\"")
            && html.contains("./assets/")
    })
}

fn request_matches_etag(headers: &HeaderMap, etag: &str) -> bool {
    headers
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value.split(',').any(|candidate| {
                let candidate = candidate.trim();
                candidate == "*"
                    || candidate == etag
                    || candidate
                        .strip_prefix("W/")
                        .is_some_and(|weak| weak == etag)
            })
        })
}

fn response_builder(status: StatusCode) -> axum::http::response::Builder {
    Response::builder()
        .status(status)
        .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff")
        .header(header::REFERRER_POLICY, "no-referrer")
        .header(
            header::CONTENT_SECURITY_POLICY,
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'",
        )
}

fn empty_response(status: StatusCode) -> Response {
    response_builder(status)
        .body(Body::empty())
        .unwrap_or_else(internal_error)
}

fn internal_error(_: axum::http::Error) -> Response {
    StatusCode::INTERNAL_SERVER_ERROR.into_response()
}

fn etag_for(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("\"{digest:x}\"")
}

fn cache_control_for(path: &str) -> &'static str {
    if path == INDEX_PATH || path == MANIFEST_PATH {
        "no-cache"
    } else if has_vite_content_hash(path) {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    }
}

fn has_vite_content_hash(path: &str) -> bool {
    let Some(file_name) = path.rsplit('/').next() else {
        return false;
    };
    let Some((stem, _extension)) = file_name.rsplit_once('.') else {
        return false;
    };
    let Some((_name, hash)) = stem.rsplit_once('-') else {
        return false;
    };
    hash.len() >= 8
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn format_uri_with_trailing_slash(uri: &Uri) -> String {
    let path = uri.path();
    if let Some(query) = uri.query() {
        format!("{path}/?{query}")
    } else {
        format!("{path}/")
    }
}

fn reject_raw_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.contains('%') || lower.contains('\\') || lower.contains('\0')
}

fn reject_decoded_path(path: &str) -> bool {
    path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains('\0')
        || path
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
}

#[cfg(test)]
#[path = "assets_tests.rs"]
mod assets_tests;
