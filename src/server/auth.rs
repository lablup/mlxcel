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

//! API-key authentication aligned with `llama-server` b10621 (#1437).
//!
//! b10621 accepts a **set** of keys, not one key. `--api-key` takes a
//! comma-separated list, `--api-key-file` reads one key per line, and every
//! occurrence of either flag, plus their environment bindings, appends to the
//! same set (upstream processes all environment variables first and then the
//! command line, calling the same appending handler for each). mlxcel used to
//! compare a single exact CLI value, or the entire trimmed file as one key, so
//! the accepted syntax and the resulting authorization behavior both diverged.
//!
//! The request side follows upstream's `middleware_validate_api_key`
//! (<https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-http.cpp>):
//! read `Authorization`, fall back to `X-Api-Key` when it is absent or empty,
//! strip a literal `Bearer ` prefix, and compare the remainder against the
//! configured set. A missing credential and an unknown one produce the same
//! 401 and the same body, so a probe cannot tell them apart.
//!
//! Key material never leaves this module in a printable form: [`ApiKeys`]
//! implements `Debug` as a count, so a `{:?}` of `ServerConfig` (or of
//! anything holding it) cannot put a secret in a log.

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};

/// The `Authorization` scheme prefix b10621 strips before comparing.
const BEARER_PREFIX: &str = "Bearer ";

/// The Anthropic-style header b10621 falls back to when `Authorization` is
/// absent or empty.
const X_API_KEY: &str = "X-Api-Key";

/// The configured API keys.
///
/// Empty means authentication is disabled, which is b10621's behavior for an
/// empty `api_keys` vector.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct ApiKeys(Vec<String>);

impl ApiKeys {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when `presented` is one of the configured keys.
    ///
    /// The comparison is exact, as upstream's `std::find` is: no trimming, no
    /// case folding, no prefix matching.
    pub(crate) fn accepts(&self, presented: &str) -> bool {
        self.0.iter().any(|key| key == presented)
    }

    /// Build from an already-parsed key list. Kept private to the crate so
    /// every production path goes through [`resolve_api_keys`].
    pub(crate) fn from_vec(keys: Vec<String>) -> Self {
        Self(keys)
    }
}

/// Never prints key material.
///
/// `ServerConfig` derives `Debug` and is reachable from several tracing call
/// sites; without this, one `{:?}` would put every configured key in the log.
impl fmt::Debug for ApiKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ApiKeys({} configured)", self.0.len())
    }
}

/// Split one `--api-key` value the way b10621's `parse_csv_row` does.
///
/// This is a faithful port of upstream
/// <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>
/// (`parse_csv_row`), including the parts that are surprising:
///
/// - whitespace is **not** trimmed, so `a, b` yields `a` and `" b"`;
/// - a field that starts with `"` is quoted and may contain commas;
/// - `""` inside a quoted field is a literal quote;
/// - a `"` in the middle of an unquoted field is a literal quote.
///
/// Trimming here would be an improvement that silently accepts a key
/// b10621 rejects, so it is deliberately not done.
pub(crate) fn parse_csv_row(input: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if !in_quotes => {
                if field.is_empty() {
                    in_quotes = true;
                } else {
                    // A quote in the middle of an unquoted field is literal.
                    field.push('"');
                }
            }
            '"' => {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            ',' if in_quotes => field.push(','),
            ',' => fields.push(std::mem::take(&mut field)),
            other => field.push(other),
        }
    }
    fields.push(field);
    fields
}

/// Parse one `--api-key-file` body: one key per line, `#` comments.
///
/// b10621 reads the file with `std::getline` and keeps a line when it is
/// non-empty and its first character is not `#`. That means:
///
/// - the line is **not** trimmed, so leading and trailing spaces are part of
///   the key;
/// - a comment marker only counts in column one, so `  # note` is a key;
/// - `std::getline` splits on `\n` alone, so a CRLF file leaves a carriage
///   return at the end of every key.
///
/// The CRLF case is reproduced rather than fixed, because a key file that
/// works on one server and not the other is worse than one that fails the
/// same way on both. `resolve_api_keys` warns when it sees one, so the
/// operator gets told instead of guessing.
pub(crate) fn parse_api_key_file(contents: &str) -> Vec<String> {
    contents
        .split('\n')
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Build the key set from every source, in b10621's order.
///
/// Upstream applies environment variables first and the command line second,
/// and both handlers append, so the caller passes the environment-derived
/// values ahead of the command-line ones in each list. Order does not affect
/// membership; it is preserved so a log line about "the first key" means the
/// same thing on both servers.
///
/// A key file that cannot be read is a startup error, as upstream's throw is.
/// A configuration that names key sources but produces no key is a startup
/// error too: upstream would start with authentication silently disabled on a
/// server the operator meant to protect, which is the one failure mode worth
/// diverging over.
pub fn resolve_api_keys(cli_keys: &[String], key_files: &[PathBuf]) -> Result<ApiKeys> {
    let mut keys: Vec<String> = Vec::new();
    let mut saw_carriage_return = false;

    for raw in cli_keys {
        keys.extend(parse_csv_row(raw).into_iter().filter(|k| !k.is_empty()));
    }

    for path in key_files {
        let contents = read_key_file(path)?;
        let parsed = parse_api_key_file(&contents);
        saw_carriage_return |= parsed.iter().any(|key| key.ends_with('\r'));
        keys.extend(parsed);
    }

    if saw_carriage_return {
        tracing::warn!(
            "an --api-key-file has CRLF line endings, so every key it contributes ends in a \
             carriage return and clients must send that byte too. llama-server b10621 reads the \
             file the same way; convert the file to LF endings to avoid it"
        );
    }

    if keys.is_empty() && !(cli_keys.is_empty() && key_files.is_empty()) {
        bail!(
            "--api-key / --api-key-file were given but contributed no key, so the server would \
             start with authentication disabled. Check the value for empty fields, and the key \
             file for a body that is entirely blank lines or '#' comments"
        );
    }

    Ok(ApiKeys::from_vec(keys))
}

fn read_key_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path)
        .with_context(|| format!("--api-key-file {}: cannot read", path.display()))
}

/// The credential a request presents, following upstream's header order.
///
/// `Authorization` wins; an absent or empty one falls back to `X-Api-Key`. A
/// literal `Bearer ` prefix is stripped from whichever was used, so
/// `X-Api-Key: Bearer k` presents `k` exactly as upstream does. A header that
/// is not valid UTF-8 presents nothing, which fails closed.
pub(crate) fn presented_credential(headers: &HeaderMap) -> Option<&str> {
    let authorization = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty());

    let raw = match authorization {
        Some(value) => value,
        None => headers
            .get(X_API_KEY)
            .and_then(|value| value.to_str().ok())?,
    };

    Some(raw.strip_prefix(BEARER_PREFIX).unwrap_or(raw))
}

/// b10621's 401 body, byte for byte.
///
/// The same response answers a missing credential and an unknown one, so a
/// probe cannot use the status or the body to learn whether a key exists. No
/// configured key is echoed.
pub(crate) fn unauthorized_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(
            header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json; charset=utf-8"),
        )],
        serde_json::json!({
            "error": {
                "message": "Invalid API Key",
                "type": "authentication_error",
                "code": 401,
            }
        })
        .to_string(),
    )
        .into_response()
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod auth_tests;
