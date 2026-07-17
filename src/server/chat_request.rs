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

//! Shared chat-request preparation helpers.
//!
//! Both streaming and non-streaming chat routes should apply the same message
//! flattening, template rendering, and image extraction rules.
//!
//! Used by: routes/chat
//!
//! # Prefix stability guarantees
//!
//! When the prompt prefix cache is enabled, [`prepare_chat_request`]
//! guarantees that rendering the same conversation across turns yields a
//! prompt that is a prefix of the next turn's prompt, so the KV cache is
//! reusable:
//!
//! 1. **`preserve_thinking` defaulting.** With the prompt cache on, unset
//!    `preserve_thinking` defaults to `true`. This disables the rolling
//!    checkpoint stripper (see [`super::chat_template_kwargs::
//!    strip_rolling_checkpoint`]) whose "strip every thinking block before
//!    the latest user turn" rule is the primary source of prefix drift. Opt
//!    out explicitly via `chat_template_kwargs.preserve_thinking = false`
//!    when prefix instability is acceptable.
//! 2. **Template signature in the cache key.** The
//!    [`super::prompt_cache::key::template_sig`] hash is derived from
//!    `(chat_template_source, chat_template_kwargs, tool_choice,
//!    tools_digest)`. Any change in rendering inputs — including a
//!    non-deterministic template tweak or a tool-list reorder — drops the
//!    affected entries cleanly by producing a new `template_sig`.
//! 3. **Known non-determinism sources and how they are handled.**
//!    * `strftime_now` in Jinja (templates like Llama 3.x emit "today's
//!      date"). This is an intentional per-render value; the cache keys the
//!      prompt tokens rather than the template output, so a date-of-day
//!      difference surfaces as a different token prefix and creates a new
//!      bucket that naturally ages out. Documented, not silently masked.
//!    * `<think>` block stripping across turns — invalidated by the
//!      `preserve_thinking=true` default.
//!    * Tool-schema hashing: [`super::prompt_cache::key::tools_digest`] is
//!      order-preserving, so reordering tools invalidates the cache. This
//!      is intentional: HuggingFace templates iterate tools in order and
//!      some models key their protocol to the index.
//!    * Kwargs-key reordering: kwargs are canonicalized with sorted object
//!      keys before hashing, so map insertion-order drift is absorbed.
//!    * `enable_thinking` / future `chat_template_kwargs` additions: any
//!      new kwarg automatically propagates into `template_sig` because it
//!      participates in the canonicalized map hash.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;

use super::chat_template::{ChatMessage, ChatTemplateProcessor};
use super::chat_template_kwargs::{
    ChatTemplateKwargs, extract_request_kwargs, merge_server_and_request, strip_rolling_checkpoint,
    strip_think_block,
};
use super::media::{
    ResolvedVideo, extract_chat_audio_data, extract_chat_video_paths, try_extract_chat_image_data,
};
use super::prompt_cache::key::resolve_session_key;
use super::types::ChatCompletionRequest;
use super::types::request::{ContentPart, Message, MessageContent, Tool};

pub(crate) struct PreparedChatRequest {
    pub(crate) prompt: String,
    pub(crate) image_data: Vec<Vec<u8>>,
    /// Request-scoped Gemma 4 image soft-token budget, resolved from the
    /// `detail` / `max_soft_tokens` fields on the `image_url` content parts
    /// (see [`crate::server::types::request::ImageUrl`]).
    ///
    /// `None` means "no override" and leaves the checkpoint's configured
    /// budget in place, which is the behavior for every request that does not
    /// set either field. Validation happens here, at the request boundary, so
    /// an unsupported value fails the whole request with a 400 before any
    /// image is decoded.
    pub(crate) image_soft_tokens: Option<usize>,
    pub(crate) audio_data: Vec<Vec<u8>>,
    /// Resolved video items (hardened).
    ///
    /// Each entry holds:
    /// * a [`crate::multimodal::video::VideoSource`] handle that the
    ///   model worker passes to
    ///   [`crate::multimodal::video::load_video_source`]. On Unix this is
    ///   the fd-backed variant — the resolver opened the file once after
    ///   canonicalise + allowlist + regular-file + extension checks, and
    ///   ffmpeg reads from that open file description (via `/dev/fd/N`),
    ///   so the canonicalise → ffmpeg-open TOCTOU race is closed at the
    ///   kernel level.
    /// * the optional per-video FPS override from
    ///   [`crate::server::types::request::VideoUrl::fps`].
    /// * an internal RAII guard for any server-owned temp file (data-URI
    ///   decode / HTTP fetch). The guard fires `fs::remove_file` when the
    ///   `PreparedChatRequest` drops, so the temp file lifecycle equals
    ///   the request handler lifecycle.
    pub(crate) videos: Vec<ResolvedVideo>,
}

/// Dedup set for the "defaulted `preserve_thinking=true`" info log.
///
/// Keyed by the resolved `session_key` (see [`resolve_session_key`]) so we
/// log exactly once per logical session per server lifetime. Process-wide
/// state — the dedup tracks the *runtime identity* of the server, not a
/// persistent record. A restart resets the set, which is fine: operators
/// should see the log after each restart confirming the defaulting is live.
pub(super) fn log_once_sessions() -> &'static Mutex<HashSet<String>> {
    static LOGGED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    LOGGED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Legacy wrapper preserved for tests and any callers outside the hot
/// route path. Delegates to [`prepare_chat_request_with_cache`] with
/// the cache-enabled flag set to `false`, matching earlier behavior.
///
/// Production HTTP handlers (see `src/server/routes/chat.rs`) use the
/// `_with_cache` variant directly so they can honor the installed
/// [`super::prompt_cache::PromptCacheStore`].
#[allow(dead_code)]
pub(crate) async fn prepare_chat_request(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
    server_default_kwargs: Option<&ChatTemplateKwargs>,
) -> Result<PreparedChatRequest> {
    prepare_chat_request_with_cache(processor, request, server_default_kwargs, false).await
}

/// Full variant of [`prepare_chat_request`] with explicit prompt-cache
/// awareness.
///
/// When `prompt_cache_enabled` is `true` and the caller has not explicitly
/// set `preserve_thinking` anywhere in the request, this function defaults
/// it to `true` before running the rendering pipeline. Explicit overrides
/// (`chat_template_kwargs.preserve_thinking = false`, flattened OpenAI-SDK
/// field, nested `extra_body`, or server-default kwargs) are respected
/// unchanged.
///
/// When `prompt_cache_enabled` is `false` this is identical to the legacy
/// [`prepare_chat_request`].
pub(crate) async fn prepare_chat_request_with_cache(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
    server_default_kwargs: Option<&ChatTemplateKwargs>,
    prompt_cache_enabled: bool,
) -> Result<PreparedChatRequest> {
    // Determine effective tools based on tool_choice
    let effective_tools = effective_tools(request);
    let merged_extra_body = request.merged_extra_body();

    // resolve merged kwargs once up-front.
    //
    // Precedence: top-level `chat_template_kwargs` >
    // nested/flattened `extra_body.chat_template_kwargs` >
    // nested/flattened DashScope/OpenAI-SDK `preserve_thinking` aliases. The
    // merge with server-default kwargs follows the "per-request wins per-key,
    // unrelated server-default keys persist" rule so every future kwarg
    // inherits the same plumbing.
    let per_request_kwargs = extract_request_kwargs(
        request.chat_template_kwargs.as_ref(),
        merged_extra_body.as_ref(),
    );
    let mut merged_kwargs = merge_server_and_request(server_default_kwargs, &per_request_kwargs);

    // default preserve_thinking=true when the prompt cache is on
    // and no layer of the precedence chain set it. The per-request-kwargs
    // object already reflects top-level + flattened-SDK + nested-extra_body
    // + DashScope flat shape, and `merged_kwargs` folds in the server
    // default. So "not set anywhere" is precisely "the merged map lacks the
    // key."
    if prompt_cache_enabled && !merged_kwargs.as_map().contains_key("preserve_thinking") {
        merged_kwargs.set_preserve_thinking(true);
        maybe_log_defaulting_once(request);
    }

    let preserve_thinking = merged_kwargs.preserve_thinking();

    let prompt = if has_tool_fields(request) || has_reasoning_fields(request) {
        // When messages contain tool_calls / tool_call_id, or a parallel
        // `reasoning` field (issue #362), use raw JSON rendering so the Jinja2
        // template can access all fields. The typed `ChatMessage` path only
        // carries role + content, so reasoning would otherwise be dropped
        // before reaching templates that render `message.get('reasoning')`.
        let raw_messages = build_raw_json_messages_with_thinking(request, preserve_thinking);
        // Build the stripped ChatMessages in parallel so the fallback path can
        // use them without re-running strip_rolling_checkpoint.
        let stripped = build_chat_messages_with_thinking(request, preserve_thinking);
        processor
            .apply_raw_with_kwargs(&raw_messages, effective_tools, &merged_kwargs)
            .unwrap_or_else(|err| {
                tracing::warn!(
                    "Chat template render (raw) failed, using fallback: {:#}",
                    err
                );
                // Security (H-1): use pre-stripped messages so that a
                // template-breaking payload cannot bypass rolling-checkpoint
                // stripping and leak prior <think> blocks to the model prompt.
                render_simple_fallback(&stripped)
            })
    } else {
        let messages = build_chat_messages_with_thinking(request, preserve_thinking);
        processor
            .apply_with_kwargs(&messages, effective_tools, &merged_kwargs)
            .unwrap_or_else(|err| {
                tracing::warn!("Chat template render failed, using fallback: {:#}", err);
                // Security (H-1): use pre-stripped messages for the same reason.
                render_simple_fallback(&messages)
            })
    };

    // Validate the per-request soft-token budget before any image is fetched or
    // decoded: an unsupported `detail` / `max_soft_tokens` is a client error, so
    // there is no reason to spend a download or a decode on it first.
    let image_soft_tokens = request
        .image_soft_tokens()
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    let (image_data, audio_data, videos) = tokio::join!(
        try_extract_chat_image_data(request),
        extract_chat_audio_data(request),
        extract_chat_video_paths(request),
    );

    Ok(PreparedChatRequest {
        prompt,
        image_data: image_data?,
        image_soft_tokens,
        audio_data,
        videos,
    })
}

/// Emit an INFO log exactly once per resolved session when the
/// prompt-cache-driven `preserve_thinking=true` default kicks in.
///
/// The dedup key is the same `session_key` the cache uses, so distinct
/// users (OpenAI-standard `user` field or `prompt_cache_key`) each get
/// their own one-shot log line; anonymous traffic shares the log entry
/// for [`super::prompt_cache::key::ANONYMOUS_SESSION_SENTINEL`].
///
/// The log is purely informational; the defaulting decision itself runs
/// regardless.
fn maybe_log_defaulting_once(request: &ChatCompletionRequest) {
    let pck = request.resolve_prompt_cache_key();
    let user = request.resolve_user();
    let session = resolve_session_key(pck, user).to_string();
    let Ok(mut set) = log_once_sessions().lock() else {
        return;
    };
    if set.insert(session.clone()) {
        tracing::info!(
            session = %session,
            "prompt cache on + preserve_thinking unset: defaulting preserve_thinking=true \
             for prefix stability (override via chat_template_kwargs.preserve_thinking=false)"
        );
    }
}

/// Determine the effective tools slice to pass to the template.
///
/// Returns `None` when tool_choice is "none" or no tools are provided.
fn effective_tools(request: &ChatCompletionRequest) -> Option<&[Tool]> {
    // If tool_choice is "none", do not pass tools to template
    if let Some(ref tc) = request.tool_choice
        && tc.is_none()
    {
        return None;
    }
    request.tools.as_deref()
}

/// Check if any message in the request has tool-related fields that
/// require raw JSON rendering (tool_calls, tool_call_id).
fn has_tool_fields(request: &ChatCompletionRequest) -> bool {
    request
        .messages
        .iter()
        .any(|m| m.tool_call_id.is_some() || m.tool_calls.is_some())
}

/// Check if any message carries a non-empty `reasoning` field (issue #362).
///
/// Such requests must take the raw-JSON render path so the parallel reasoning
/// reaches templates that read `message.get('reasoning')`; the typed
/// [`ChatMessage`] path drops everything except role and content.
fn has_reasoning_fields(request: &ChatCompletionRequest) -> bool {
    request
        .messages
        .iter()
        .any(|m| m.reasoning.as_ref().is_some_and(|r| !r.is_empty()))
}

/// Returns `true` when the request carries at least one message with
/// user-meaningful ("effective") input, per issue #773.
///
/// A message counts as effective input when any of the following holds:
///
/// * its flattened text content (string form, or a content-list `text`
///   part) is non-empty after trimming whitespace
/// * a content-list `image_url` part carries a non-empty `url`
/// * a content-list `video_url` part carries a non-empty `url` (mlxcel's
///   media-input surface extends the OpenAI shape with video; treated the
///   same as an image for this check)
/// * a content-list `input_audio` part carries non-empty `data`
/// * the message carries a non-empty `tool_calls` array
/// * the message carries a non-empty `reasoning` field
///
/// An empty `messages` array (or a request where every message fails all
/// of the above) has no effective input and should be rejected with a 400
/// before any model dispatch — see [`super::routes::chat::chat_completions`]
/// and [`super::routes::responses::create_response`].
pub(crate) fn request_has_effective_input(request: &ChatCompletionRequest) -> bool {
    request.messages.iter().any(message_has_effective_input)
}

/// Single-message half of [`request_has_effective_input`].
fn message_has_effective_input(message: &Message) -> bool {
    if !message.content.text().trim().is_empty() {
        return true;
    }

    if let MessageContent::Parts(parts) = &message.content {
        for part in parts {
            match part {
                // Text parts are already covered by `content.text()` above.
                ContentPart::Text { .. } => {}
                ContentPart::ImageUrl { image_url } => {
                    if !image_url.url.trim().is_empty() {
                        return true;
                    }
                }
                ContentPart::VideoUrl { video_url } => {
                    if !video_url.url.trim().is_empty() {
                        return true;
                    }
                }
                ContentPart::InputAudio { input_audio } => {
                    if !input_audio.data.trim().is_empty() {
                        return true;
                    }
                }
            }
        }
    }

    if message.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty()) {
        return true;
    }

    if message
        .reasoning
        .as_ref()
        .is_some_and(|r| !r.trim().is_empty())
    {
        return true;
    }

    false
}

/// Build raw JSON messages for template rendering, preserving all fields
/// (including tool_calls, tool_call_id) so Jinja2 templates can iterate over
/// multi-turn tool-use conversations.
///
/// Thin wrapper with `preserve_thinking=true` — used by tests that predate
/// and by any caller that does not want rolling-checkpoint
/// stripping.
#[cfg(test)]
pub(super) fn build_raw_json_messages(request: &ChatCompletionRequest) -> serde_json::Value {
    build_raw_json_messages_with_thinking(request, true)
}

/// build raw JSON messages with optional rolling-checkpoint
/// stripping of `<think>` blocks.
///
/// When `preserve_thinking` is `true`, all `<think>...</think>` blocks reach
/// the template unchanged (Qwen3.6 multi-turn retention). When `false` (the
/// default), the rolling-checkpoint rule strips thinking from every assistant
/// message **before** the most recent non-tool-call user turn — matching the
/// Qwen3/Qwen3.5 convention. The most recent assistant reply keeps its
/// reasoning regardless.
///
/// This Rust-side stripping is the fallback for templates that don't
/// understand the `preserve_thinking` kwarg. Templates that do understand it
/// (like the official Qwen3.6 chat template) will still see the stripped
/// strings; because the stripped text contains no `<think>` markers, the
/// template's own preserve-logic is a no-op there — we reach the same
/// effective prompt either way.
fn build_raw_json_messages_with_thinking(
    request: &ChatCompletionRequest,
    preserve_thinking: bool,
) -> serde_json::Value {
    // Decide which assistant messages (by index) need their think blocks
    // stripped. Empty set means "keep everything."
    let strip_indices: std::collections::HashSet<usize> = if preserve_thinking {
        std::collections::HashSet::new()
    } else {
        strip_rolling_checkpoint(&request.messages, |m| m.role.as_str(), |m| m.content.text())
            .into_iter()
            .collect()
    };

    let messages: Vec<serde_json::Value> = request
        .messages
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            // Strip think blocks from assistant messages before the checkpoint.
            let stripped = strip_indices.contains(&idx);
            let raw_content = m.content.text();
            let content = if stripped {
                strip_think_block(&raw_content).into_owned()
            } else {
                raw_content
            };

            let mut msg = serde_json::json!({
                "role": m.role.as_str(),
                "content": content,
            });

            if let Some(ref name) = m.name {
                msg["name"] = serde_json::Value::String(name.clone());
            }
            if let Some(ref tool_call_id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
            }
            if let Some(ref tool_calls) = m.tool_calls {
                let mut tc_value =
                    serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null);
                normalize_tool_call_arguments(&mut tc_value);
                msg["tool_calls"] = tc_value;
            }

            // Forward the parallel `reasoning` field (issue #362) so templates
            // that render `message.get('reasoning')` (e.g. Gemma 4) see prior
            // assistant thinking across turns. The decision mirrors the inline
            // `<think>` handling exactly so the two channels stay consistent:
            //
            // - When this message is being stripped (preserve_thinking=false and
            //   it sits before the rolling checkpoint), drop the reasoning field
            //   too. Stripping the inline block while leaking the parallel field
            //   would still feed prior thinking back into the prompt.
            // - When preserve_thinking=true (or this is the retained latest
            //   reply), forward the reasoning field, unless the content already
            //   carries an inline `<think>` block. Forwarding it on top of an
            //   inline block would double-inject the same reasoning into
            //   templates that render both channels.
            if !stripped
                && let Some(reasoning) = m.reasoning.as_ref()
                && !reasoning.is_empty()
                && !content.contains("<think>")
            {
                msg["reasoning"] = serde_json::Value::String(reasoning.clone());
            }

            msg
        })
        .collect();

    serde_json::Value::Array(messages)
}

/// Normalize each tool call's `function.arguments` from a JSON-encoded string
/// into a parsed object so dict-iterating chat templates can consume it.
///
/// The OpenAI wire format carries `arguments` as a JSON *string* (e.g.
/// `"{\"path\":\"/foo\"}"`), and that's what agentic clients echo back on
/// later turns. But the Qwen3-Coder / Qwen3.5 / Qwen3.6 chat templates iterate
/// it with `tool_call.arguments|items`, which requires a mapping. Passing the
/// raw string makes minijinja's `|items` fail ("cannot convert value into
/// pairs"), the whole render falls back to a default template, and the
/// resulting prompt diverges from the prior turn from the first token,
/// silently degrading the model's multi-turn context and destroying
/// prompt-cache prefix reuse (every tool turn re-prefills cold).
///
/// We replace `arguments` only when the string parses to a JSON **object**;
/// templates that serialize arguments via `tojson` then emit the original
/// object shape, while malformed/scalar arguments stay strings for templates
/// that treat them as text. Mirrors mlx-serve's `chat.zig` workaround.
fn normalize_tool_call_arguments(tool_calls: &mut serde_json::Value) {
    let serde_json::Value::Array(calls) = tool_calls else {
        return;
    };
    for call in calls {
        let Some(args) = call.pointer_mut("/function/arguments") else {
            continue;
        };
        if let serde_json::Value::String(s) = args
            && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
            && parsed.is_object()
        {
            *args = parsed;
        }
    }
}

/// Flatten request messages into [`ChatMessage`], preserving all `<think>`
/// blocks.
///
/// Thin wrapper around [`build_chat_messages_with_thinking`] with
/// `preserve_thinking=true`. Only exercised by tests today — the production
/// code path in [`prepare_chat_request`] always calls
/// `build_chat_messages_with_thinking` directly so it can honor the merged
/// kwargs.
#[cfg(test)]
pub(super) fn build_chat_messages(request: &ChatCompletionRequest) -> Vec<ChatMessage> {
    build_chat_messages_with_thinking(request, true)
}

/// Security (H-1): produce the same "System: … User: … Assistant: …" fallback
/// prompt that `ChatCompletionRequest::to_prompt()` emits, but operating on
/// messages that have **already been stripped** by either
/// [`build_chat_messages_with_thinking`] or equivalent pre-processing.
///
/// This is the single fallback renderer used by both the raw-JSON path and the
/// typed-message path when Jinja template rendering fails (parse error, `raise`
/// in template, minijinja internal error).  Centralising the fallback here
/// ensures the `preserve_thinking` stripping decision made before the Jinja
/// call is never bypassed by a deliberately template-breaking request payload.
fn render_simple_fallback(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => prompt.push_str(&format!("System: {}\n\n", msg.content)),
            "user" => prompt.push_str(&format!("User: {}\n\n", msg.content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n\n", msg.content)),
            "tool" => prompt.push_str(&format!("Tool: {}\n\n", msg.content)),
            other => prompt.push_str(&format!("{}: {}\n\n", other, msg.content)),
        }
    }
    prompt.push_str("Assistant: ");
    prompt
}

/// flatten request messages into [`ChatMessage`] with optional
/// rolling-checkpoint stripping.
///
/// See [`build_raw_json_messages_with_thinking`] for the stripping rules. The
/// `ChatMessage` path is used for the common non-tool-call case; the typed
/// struct doesn't carry `tool_calls`/`tool_call_id`, which is fine because
/// `has_tool_fields` routes those cases to the raw-JSON path.
fn build_chat_messages_with_thinking(
    request: &ChatCompletionRequest,
    preserve_thinking: bool,
) -> Vec<ChatMessage> {
    let strip_indices: std::collections::HashSet<usize> = if preserve_thinking {
        std::collections::HashSet::new()
    } else {
        strip_rolling_checkpoint(&request.messages, |m| m.role.as_str(), |m| m.content.text())
            .into_iter()
            .collect()
    };

    request
        .messages
        .iter()
        .enumerate()
        .map(|(idx, message)| {
            let raw = message.content.text();
            let content = if strip_indices.contains(&idx) {
                strip_think_block(&raw).into_owned()
            } else {
                raw
            };
            ChatMessage {
                role: message.role.as_str().to_string(),
                content,
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "chat_request_tests.rs"]
mod tests;
