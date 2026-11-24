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

//! The CLI `--reasoning-effort` flag (issue #3378).
//!
//! The server takes a reasoning level per request (`reasoning_effort`) and
//! resolves it into chat-template kwargs in
//! [`crate::server::chat_request::resolve_effective_kwargs`]. The CLI reuses
//! that resolution for its one-shot render ([`cli_template_kwargs`]) and for
//! the in-process chat body ([`apply_to_chat_body`]) instead of keeping a
//! second effort mapping.
//!
//! The server forwards the request field as a JSON string. Templates that
//! accept a number (Inkling 2: a level name or a float in `[0.0, 0.99]`)
//! reject the string `"0.5"`, and the server never forwards a disabled level
//! such as `none` at all. The CLI therefore also sets the explicit
//! `reasoning_effort` template kwarg, typed ([`level_value`]), which the
//! server's per-key precedence lets win over the derived one.

use serde_json::{Map, Value, json};

use crate::server::chat_request::resolve_effective_kwargs;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::server::in_process::chat::chat_request_from_json;

/// Validate and normalize a `--reasoning-effort` value: trimmed and lowercase.
///
/// Only an empty value is refused here. Whether the level is acceptable is the
/// chat template's call, and its own message is surfaced unchanged.
///
/// # Errors
/// Returns the message for an empty or whitespace-only value.
pub fn parse_reasoning_effort(raw: &str) -> Result<String, String> {
    let level = raw.trim().to_ascii_lowercase();
    if level.is_empty() {
        return Err("reasoning effort must not be empty".to_string());
    }
    Ok(level)
}

/// The JSON value the template kwarg carries: a number when `level` parses as
/// a finite number, else the level string.
#[must_use]
pub fn level_value(level: &str) -> Value {
    level
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
        .and_then(serde_json::Number::from_f64)
        .map_or_else(|| Value::String(level.to_string()), Value::Number)
}

/// Set `level` on a chat request body: the portable `reasoning_effort` field
/// (so the server's own resolution runs, including `reasoning_strength`
/// templates) and the typed explicit template kwarg.
pub fn apply_to_chat_body(body: &mut Value, level: &str) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    object.insert("reasoning_effort".to_string(), json!(level));
    let kwargs = object
        .entry("chat_template_kwargs")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(kwargs) = kwargs.as_object_mut() {
        kwargs
            .entry("reasoning_effort")
            .or_insert_with(|| level_value(level));
    }
}

/// The template kwargs a one-shot CLI render passes for `level`, resolved by
/// the server's own request-to-kwargs rules.
///
/// # Errors
/// Returns an error when the synthetic request does not deserialize.
pub fn cli_template_kwargs(
    processor: &ChatTemplateProcessor,
    level: &str,
) -> anyhow::Result<ChatTemplateKwargs> {
    let mut body = json!({
        "model": "mlxcel-cli",
        "messages": [{"role": "user", "content": ""}],
    });
    apply_to_chat_body(&mut body, level);
    let request = chat_request_from_json(body)?;
    Ok(resolve_effective_kwargs(
        processor,
        &request,
        None,
        &request.merged_extra_body(),
    ))
}

#[cfg(test)]
#[path = "reasoning_effort_tests.rs"]
mod tests;
