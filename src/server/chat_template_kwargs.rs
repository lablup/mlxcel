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

//! Shared chat-template kwargs plumbing.
//!
//! Provides the generic `chat_template_kwargs` dictionary pass-through that
//! matches llama.cpp's `--chat-template-kwargs` flag and vLLM's
//! `extra_body.chat_template_kwargs` request field. Today the primary consumer
//! is `preserve_thinking` (Qwen3.6 multi-turn reasoning retention), but the
//! mechanism is intentionally generic so future keys (`enable_thinking`,
//! model-specific Jinja hints, …) can reuse the same plumbing without
//! duplicating CLI/env/request parsing or merge semantics.
//!
//! ## Resolution precedence (per-key)
//!
//! 1. Per-request `chat_template_kwargs` (top-level, llama.cpp shape).
//! 2. Per-request `extra_body.chat_template_kwargs` (nested or flattened
//!    OpenAI-SDK/vLLM shape).
//! 3. Per-request `preserve_thinking` alias carried via nested/flattened
//!    `extra_body` — secondary DashScope/OpenAI-SDK flat shape. **Only**
//!    recognized for the `preserve_thinking` key, not a general-purpose
//!    fallback for other kwargs.
//! 4. Server-wide default from `--chat-template-kwargs` CLI flag.
//! 5. Server-wide default from `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` env var (CLI
//!    wins on conflict).
//!
//! The merge rule is "per-request wins per-key, unrelated server-default keys
//! persist." A request that sets only `preserve_thinking` will not erase a
//! server default for, say, `enable_thinking`.
//!
//! ## Rolling-checkpoint default for `preserve_thinking`
//!
//! When `preserve_thinking` is absent or `false`, [`strip_rolling_checkpoint`]
//! implements the Qwen3/Qwen3.5 "rolling checkpoint" behavior in Rust: walk
//! messages in reverse, find the most recent non-tool-call user turn, and
//! strip `<think>...</think>` blocks from every assistant message before that
//! point. When `preserve_thinking` is `true`, no stripping is applied.
//!
//! Used by: server/chat_request, server/chat_template, server/cli_input,
//! server/startup, server/routes/chat

use std::borrow::Cow;

use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// The kwargs container
// ---------------------------------------------------------------------------

/// A validated collection of chat-template keyword arguments.
///
/// Invariant: every stored entry is a plain JSON value that serializes cleanly
/// into a minijinja template context. We deliberately keep the type a thin
/// wrapper over `serde_json::Map<String, Value>` rather than inventing a
/// bespoke kwarg enum, because we cannot anticipate every future Jinja
/// template's expectations and the llama.cpp surface accepts arbitrary JSON.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatTemplateKwargs {
    values: Map<String, Value>,
}

impl ChatTemplateKwargs {
    /// Empty kwargs (no keys set). Identical to [`Self::default`].
    pub fn new() -> Self {
        Self { values: Map::new() }
    }

    /// Construct from a JSON object literal.
    ///
    /// The input must be a JSON object; any other shape (array, string,
    /// null, …) is rejected to match llama.cpp behavior.
    pub fn from_json_object(obj: Map<String, Value>) -> Self {
        Self { values: obj }
    }

    /// Parse from a JSON string (used by CLI flag and env-var parsers).
    ///
    /// Empty / whitespace-only strings yield an empty `ChatTemplateKwargs`
    /// (not an error) so operators can toggle the flag off with
    /// `--chat-template-kwargs ''` without tripping.
    pub fn from_json_str(raw: &str) -> Result<Self, ChatTemplateKwargsError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(Self::new());
        }
        let parsed: Value = serde_json::from_str(trimmed)
            .map_err(|e| ChatTemplateKwargsError::InvalidJson(e.to_string()))?;
        match parsed {
            Value::Object(obj) => Ok(Self::from_json_object(obj)),
            other => Err(ChatTemplateKwargsError::NotAnObject(value_type_name(
                &other,
            ))),
        }
    }

    /// `true` when no keys are set.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Number of stored keys.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Read-only access to the underlying map. Intended for template callers
    /// that need to pass the kwargs through to a Jinja context.
    pub fn as_map(&self) -> &Map<String, Value> {
        &self.values
    }

    /// Look up a single key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Convenience accessor for the canonical `preserve_thinking` key.
    ///
    /// Non-bool JSON values (numbers, strings, `null`) are treated as "not
    /// set" and fall back to `false` so an operator typo does not silently
    /// change semantics — in that case [`Self::preserve_thinking`] returns
    /// `false` just like the absence of the key.
    pub fn preserve_thinking(&self) -> bool {
        self.values
            .get("preserve_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Explicitly set the canonical `preserve_thinking` key.
    ///
    /// Used by the prompt-cache defaulting logic in
    /// [`crate::server::chat_request::prepare_chat_request_with_cache`] to
    /// flip the flag on when prefix stability requires it and the caller
    /// hasn't specified a preference. Callers outside that path should
    /// generally let the client + CLI precedence chain decide.
    pub fn set_preserve_thinking(&mut self, value: bool) {
        self.values
            .insert("preserve_thinking".to_string(), Value::Bool(value));
    }

    /// Merge `other` into `self` with "other wins per-key".
    ///
    /// Unrelated keys in `self` are preserved; keys present in both are
    /// overwritten by `other`. Used by [`merge_server_and_request`] to
    /// combine server-default kwargs with per-request kwargs.
    pub fn merge_with_overrides(mut self, other: Self) -> Self {
        for (k, v) in other.values {
            self.values.insert(k, v);
        }
        self
    }
}

impl From<Map<String, Value>> for ChatTemplateKwargs {
    fn from(values: Map<String, Value>) -> Self {
        Self { values }
    }
}

/// Validation errors when parsing CLI/env/request kwargs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateKwargsError {
    /// The input string was not valid JSON.
    InvalidJson(String),
    /// The JSON parsed, but the top-level value was not an object.
    NotAnObject(&'static str),
}

impl std::fmt::Display for ChatTemplateKwargsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidJson(msg) => write!(f, "chat_template_kwargs must be valid JSON: {msg}"),
            Self::NotAnObject(kind) => {
                write!(f, "chat_template_kwargs must be a JSON object, got {kind}")
            }
        }
    }
}

impl std::error::Error for ChatTemplateKwargsError {}

fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// CLI / env resolution
// ---------------------------------------------------------------------------

/// Environment variable name for the server-wide kwargs default.
pub const LLAMA_ARG_CHAT_TEMPLATE_KWARGS: &str = "LLAMA_ARG_CHAT_TEMPLATE_KWARGS";

/// Apply `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` env-var fallback to the CLI-provided
/// raw JSON string.
///
/// Precedence rule (matches the existing `LLAMA_ARG_*` helpers in
/// [`super::cli_input`]):
/// - CLI wins over env var.
/// - If CLI is `None` and env var is set, fill from env.
/// - If CLI is `Some` and env var is also set, keep CLI and log an INFO-level
///   notice about the override.
///
/// Unparseable JSON is left for [`ChatTemplateKwargs::from_json_str`] to
/// surface later with a clear error at startup; this helper only handles
/// precedence, not validation.
pub fn env_fallback_chat_template_kwargs(cli_value: &mut Option<String>) {
    if cli_value.is_none() {
        if let Ok(v) = std::env::var(LLAMA_ARG_CHAT_TEMPLATE_KWARGS) {
            *cli_value = Some(v);
        }
    } else if std::env::var_os(LLAMA_ARG_CHAT_TEMPLATE_KWARGS).is_some() {
        tracing::info!(
            "{LLAMA_ARG_CHAT_TEMPLATE_KWARGS} env var is set but --chat-template-kwargs \
             CLI flag takes precedence; ignoring {LLAMA_ARG_CHAT_TEMPLATE_KWARGS}"
        );
    }
}

/// Resolve a server-wide kwargs default from an optional raw JSON string.
///
/// Returns `Ok(None)` when the input is `None` or an empty string — the
/// baseline "no server-default kwargs" case. Otherwise parses and validates
/// via [`ChatTemplateKwargs::from_json_str`].
pub fn resolve_server_default_kwargs(
    raw: Option<&str>,
) -> Result<Option<ChatTemplateKwargs>, ChatTemplateKwargsError> {
    match raw {
        None => Ok(None),
        Some(s) => {
            let parsed = ChatTemplateKwargs::from_json_str(s)?;
            if parsed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(parsed))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-request extraction and merge
// ---------------------------------------------------------------------------

/// Extract per-request `chat_template_kwargs` from a chat completions request
/// body following the documented precedence order (primary wins):
///
/// 1. Top-level `chat_template_kwargs` (llama.cpp shape).
/// 2. `extra_body.chat_template_kwargs` (nested or flattened OpenAI-SDK/vLLM
///    shape).
/// 3. `extra_body.preserve_thinking` (nested or flattened DashScope/OpenAI-SDK
///    secondary shape; **only** materializes a `preserve_thinking` entry, not
///    a general fallback).
///
/// Returns an empty `ChatTemplateKwargs` when no override is supplied.
pub fn extract_request_kwargs(
    top_level: Option<&Map<String, Value>>,
    extra_body: Option<&Map<String, Value>>,
) -> ChatTemplateKwargs {
    // (1) Top-level wins outright — all keys come from here.
    if let Some(obj) = top_level {
        return ChatTemplateKwargs::from_json_object(obj.clone());
    }
    // (2) Nested extra_body.chat_template_kwargs.
    if let Some(body) = extra_body
        && let Some(Value::Object(nested)) = body.get("chat_template_kwargs")
    {
        return ChatTemplateKwargs::from_json_object(nested.clone());
    }
    // (3) DashScope flat shape — synthesize a single-key map.
    if let Some(body) = extra_body
        && let Some(v) = body.get("preserve_thinking")
        && v.is_boolean()
    {
        let mut m = Map::new();
        m.insert("preserve_thinking".to_string(), v.clone());
        return ChatTemplateKwargs::from_json_object(m);
    }
    ChatTemplateKwargs::new()
}

/// Merge the server-default kwargs with the per-request kwargs.
///
/// Per-request keys win on conflict; server-default keys not present in the
/// request still apply. This is the single point where the "future-proof for
/// `enable_thinking` etc." plumbing is implemented — any new kwarg inherits
/// this merge automatically.
pub fn merge_server_and_request(
    server_default: Option<&ChatTemplateKwargs>,
    per_request: &ChatTemplateKwargs,
) -> ChatTemplateKwargs {
    let base: Cow<ChatTemplateKwargs> = match server_default {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(ChatTemplateKwargs::new()),
    };
    if per_request.is_empty() {
        return base.into_owned();
    }
    base.into_owned().merge_with_overrides(per_request.clone())
}

// ---------------------------------------------------------------------------
// Rolling-checkpoint stripper (preserve_thinking=false path)
// ---------------------------------------------------------------------------

/// Strip `<think>...</think>` blocks from assistant messages **before** the
/// most recent non-tool-call user turn.
///
/// This implements the Qwen3 / Qwen3.5 "rolling checkpoint" convention as a
/// Rust-side fallback for chat-template paths that bypass Jinja's native
/// `preserve_thinking` support (e.g., the default fallback template, or a
/// custom template that doesn't understand the kwarg). When
/// `preserve_thinking=true`, callers must skip this helper entirely.
///
/// Algorithm:
/// 1. Walk the message list in reverse to find the index of the latest real
///    user turn: role is literally `"user"` and the content is not wrapped in
///    `<tool_response>...</tool_response>` (tool-call turns use role `"tool"`,
///    so they don't count either).
/// 2. For every assistant message at an index **strictly less** than that
///    threshold, strip its thinking block. (Messages at or after the
///    threshold are retained as-is, including the most recent assistant
///    reply with its reasoning.)
///
/// The threshold is exclusive so that an assistant reply `A_i` that sits
/// between two user turns `U_{j-1}` and `U_j` (where `U_j` is the latest
/// user turn, i.e. the final message) gets stripped, matching the upstream
/// Qwen template's behavior.
///
/// Edge cases:
/// - No user turns present → nothing is stripped (threshold doesn't exist;
///   we conservatively keep all content so behavior is stable for system-only
///   or assistant-only prompts).
/// - Synthetic pseudo-user tool responses (`<tool_response>...</tool_response>`)
///   do not anchor the checkpoint, matching upstream Qwen templates.
/// - Final message is a user turn → strip every assistant message before it.
///
/// The helper operates on a slice of `(role, content)` tuples rather than
/// `ChatMessage` so it is callable from both the typed `ChatMessage` path and
/// the raw JSON `apply_raw` path without depending on types from
/// `chat_template`.
pub fn strip_rolling_checkpoint<'a, M, R, C>(
    messages: &'a [M],
    role_of: fn(&'a M) -> R,
    content_of: fn(&'a M) -> C,
) -> Vec<usize>
where
    M: 'a,
    R: AsRef<str>,
    C: AsRef<str>,
{
    let Some(threshold) = latest_user_turn_index(messages, role_of, content_of) else {
        return Vec::new();
    };
    messages
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| {
            if idx < threshold && role_of(m).as_ref() == "assistant" {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

/// Find the index of the latest real user message.
///
/// `"tool"` role messages are *not* user turns; likewise, a `role == "user"`
/// message whose content is wrapped in `<tool_response>...</tool_response>` is
/// treated as a synthesized tool-response carrier rather than a genuine user
/// instruction. This matches the upstream Qwen chat-template checkpoint
/// anchor.
fn latest_user_turn_index<'a, M, R, C>(
    messages: &'a [M],
    role_of: fn(&'a M) -> R,
    content_of: fn(&'a M) -> C,
) -> Option<usize>
where
    M: 'a,
    R: AsRef<str>,
    C: AsRef<str>,
{
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| {
            let role = role_of(m);
            let content = content_of(m);
            role.as_ref() == "user" && !is_tool_response_pseudo_user(content.as_ref())
        })
        .map(|(idx, _)| idx)
}

fn is_tool_response_pseudo_user(content: &str) -> bool {
    content.starts_with("<tool_response>") && content.ends_with("</tool_response>")
}

/// Strip the first balanced `<think>...</think>` block (greedy match) from a
/// rendered string, along with any surrounding blank lines.
///
/// The regex is replaced by a hand-rolled scanner so we don't pull in a new
/// dependency for a one-shot helper. The scanner:
/// 1. Finds the first `<think>` substring.
/// 2. Finds the matching `</think>` after it.
/// 3. Removes the block plus up to two surrounding newlines on each side so
///    we don't leave a dangling paragraph break.
///
/// Messages that don't contain a block pass through unchanged (zero allocation
/// in the common case — we only allocate when a block is actually removed).
pub fn strip_think_block(content: &str) -> Cow<'_, str> {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let Some(open_idx) = content.find(OPEN) else {
        return Cow::Borrowed(content);
    };
    // Search for the matching close *after* the open. If the tag is malformed
    // (no closing), leave the content alone — we don't want to silently drop
    // everything to end-of-string.
    let search_from = open_idx + OPEN.len();
    let Some(rel_close) = content[search_from..].find(CLOSE) else {
        return Cow::Borrowed(content);
    };
    let close_end = search_from + rel_close + CLOSE.len();

    // Expand leftwards over up to two consecutive `\n` characters so we
    // remove the paragraph break that typically precedes the block.
    let mut left = open_idx;
    let bytes = content.as_bytes();
    let mut newline_budget = 2;
    while left > 0 && newline_budget > 0 && bytes[left - 1] == b'\n' {
        left -= 1;
        newline_budget -= 1;
    }

    // Expand rightwards similarly.
    let mut right = close_end;
    let mut newline_budget = 2;
    while right < bytes.len() && newline_budget > 0 && bytes[right] == b'\n' {
        right += 1;
        newline_budget -= 1;
    }

    // If everything before `left` and after `right` is also empty, just
    // return an empty string rather than preserving stray whitespace.
    let mut stripped = String::with_capacity(content.len().saturating_sub(right - left));
    stripped.push_str(&content[..left]);
    stripped.push_str(&content[right..]);
    Cow::Owned(stripped)
}

#[cfg(test)]
#[path = "chat_template_kwargs_tests.rs"]
mod tests;
