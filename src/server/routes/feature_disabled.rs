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

//! b10621's disabled-feature stub (#1435).
//!
//! Upstream mounts `GET`/`POST /tools` and `GET`/`POST /cors-proxy` with a
//! `res_403` handler whenever server tools, MCP servers, or the UI's CORS
//! proxy are off (upstream
//! <https://github.com/ggml-org/llama.cpp/blob/main/tools/server/server.cpp>).
//! mlxcel never implements those surfaces (issue #1435 classifies them
//! `not_applicable`; the flags that would enable them fail startup in
//! [`crate::cli::ui_compat_args`]), so the four routes always answer this
//! stub: a client of a llama-server deployment that had tools off sees the
//! identical 403 envelope here, rather than a 404 it would read as a
//! missing route.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// `{"error":{"message":"this feature is disabled","type":"feature_disabled"}}`
/// with status 403, byte-shaped like upstream's `res_403` body (which
/// carries no `code` field).
pub async fn feature_disabled() -> Response {
    (
        StatusCode::FORBIDDEN,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        r#"{"error":{"message":"this feature is disabled","type":"feature_disabled"}}"#,
    )
        .into_response()
}
