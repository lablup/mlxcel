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

//! Translate raw `hf-hub` errors into actionable messages for end users.

use anyhow::{Error as AnyError, anyhow};
use hf_hub::api::sync::ApiError;

/// Map a raw `hf-hub` [`ApiError`] into a user-facing [`anyhow::Error`].
///
/// `repo_id`, `revision`, and the optional `filename` add the context the
/// downloader knows but `hf-hub` does not. The status-code heuristics handle
/// the most common HTTP failure modes (auth, not-found) so that the user gets
/// a one-line action item instead of a stack of inner errors.
pub fn map_hf_error(
    err: ApiError,
    repo_id: &str,
    revision: Option<&str>,
    filename: Option<&str>,
) -> AnyError {
    let rev = revision.unwrap_or("main");
    let context_prefix = match filename {
        Some(f) => format!("repo '{repo_id}' (rev '{rev}'), file '{f}'"),
        None => format!("repo '{repo_id}' (rev '{rev}')"),
    };

    if let Some(status) = http_status_from(&err) {
        match status {
            401 | 403 => {
                return anyhow!(
                    "{context_prefix}: authentication failed (HTTP {status}). \
                     Pass --token or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN if the repo is gated."
                );
            }
            404 => {
                return anyhow!(
                    "{context_prefix}: not found (HTTP 404). \
                     Verify the repository id and that revision '{rev}' exists."
                );
            }
            429 => {
                return anyhow!(
                    "{context_prefix}: rate-limited by Hugging Face (HTTP 429). \
                     Retry after a short delay or set HF_TOKEN to use an authenticated quota."
                );
            }
            500..=599 => {
                return anyhow!(
                    "{context_prefix}: Hugging Face server error (HTTP {status}). \
                     This is usually transient — retry shortly."
                );
            }
            _ => {}
        }
    }

    anyhow!("{context_prefix}: {err}")
}

/// Best-effort extraction of an HTTP status code from an [`ApiError`].
///
/// `hf-hub` 0.5 wraps `ureq::Error` inside `RequestError(Box<ureq::Error>)`.
/// We do not depend on `ureq` directly, so we recover the status code from
/// the error's `Display` representation. `ureq::Error::StatusCode(code)`
/// formats as `"http status: <code>"` (see ureq 3.x), and the legacy 2.x
/// variants format with `"status code <code>"`. Both are matched here so
/// the helper survives a future hf-hub minor bump.
fn http_status_from(err: &ApiError) -> Option<u16> {
    let s = err.to_string();
    parse_status_from_message(&s)
}

fn parse_status_from_message(message: &str) -> Option<u16> {
    for needle in [
        "http status: ",
        "http status ",
        "status code ",
        "status code: ",
        "status: ",
    ] {
        if let Some(idx) = message.to_ascii_lowercase().find(needle) {
            let tail = &message[idx + needle.len()..];
            if let Some(num) = tail.split(|c: char| !c.is_ascii_digit()).next()
                && let Ok(parsed) = num.parse::<u16>()
                && (100..=599).contains(&parsed)
            {
                return Some(parsed);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::parse_status_from_message;

    #[test]
    fn parses_ureq_3x_format() {
        assert_eq!(
            parse_status_from_message("request error: http status: 404"),
            Some(404)
        );
    }

    #[test]
    fn parses_legacy_status_code_format() {
        assert_eq!(
            parse_status_from_message("request error: status code 401"),
            Some(401)
        );
    }

    #[test]
    fn parses_authentication_failure() {
        assert_eq!(
            parse_status_from_message("HTTP status: 403 Forbidden"),
            Some(403)
        );
    }

    #[test]
    fn returns_none_for_non_http_errors() {
        assert_eq!(
            parse_status_from_message("I/O error: connection refused"),
            None
        );
    }

    #[test]
    fn rejects_out_of_range_codes() {
        assert_eq!(parse_status_from_message("status: 999"), None);
        assert_eq!(parse_status_from_message("status: 99"), None);
    }
}
