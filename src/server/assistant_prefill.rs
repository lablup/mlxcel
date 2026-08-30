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

//! b10621 `--prefill-assistant` (#1470).
//!
//! When the last message of a chat request is an assistant message, b10621
//! treats it as a prefix the model continues rather than a complete turn:
//! `oaicompat_chat_params_parse` sets `continue_final_message` to
//! `COMMON_CHAT_CONTINUATION_AUTO` and `add_generation_prompt` to `false`, and
//! `common_chat_templates_apply` then renders `messages[:-1]`, appends the
//! family's own generation prompt, and appends the continuation message's text
//! with no closing tag.
//!
//! mlxcel reaches the same prompt through the template rather than through
//! per-family C++: it renders `messages[:-1]` with `add_generation_prompt =
//! true`, which is exactly the family's own assistant-turn opener, and then
//! appends the continuation text. That is template-driven, so it works for
//! every family mlxcel loads rather than only the ones with a hand-written
//! handler.
//!
//! Reference:
//! <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-common.cpp>
//! and
//! <https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp>
//!
//! Used by: server::chat_request, server::routes::chat

use crate::server::types::request::{ChatCompletionRequest, Role};

/// b10621's own diagnostic for two or more trailing assistant messages.
pub(crate) const TWO_TRAILING_ASSISTANTS: &str =
    "Cannot have 2 or more assistant messages at the end of the list.";

/// mlxcel's refusal for the reasoning-only continuation b10621 calls
/// `COMMON_CHAT_CONTINUATION_REASONING`.
pub(crate) const REASONING_ONLY_UNSUPPORTED: &str = "A trailing assistant message carrying only reasoning and no content cannot be prefilled \
     unless the chat template primes an open thinking block; send `content` to continue the \
     turn, or pass --no-prefill-assistant to answer it instead";

/// What a resolved prefill contributes to the prompt and to the response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AssistantPrefill {
    /// The continuation text, appended after the rendered generation prompt
    /// with no closing tag.
    pub(crate) text: String,
    /// `true` when the text is reasoning rather than content, so the caller
    /// appends it inside the thinking block the generation prompt opened.
    pub(crate) is_reasoning: bool,
}

/// Decide whether this request's last message is an assistant prefill.
///
/// Mirrors b10621's guard exactly: prefill applies only when it is enabled,
/// the message list is non-empty, and the last role is `assistant`; two or
/// more trailing assistant messages are an error with upstream's wording.
pub(crate) fn resolve(
    request: &ChatCompletionRequest,
    enabled: bool,
) -> Result<Option<AssistantPrefill>, String> {
    if !enabled {
        return Ok(None);
    }
    let messages = &request.messages;
    let Some(last) = messages.last() else {
        return Ok(None);
    };
    if last.role != Role::Assistant {
        return Ok(None);
    }
    if messages.len() >= 2 && messages[messages.len() - 2].role == Role::Assistant {
        return Err(TWO_TRAILING_ASSISTANTS.to_string());
    }
    // A trailing assistant message that carries tool calls is a completed turn
    // the next message answers, not a prefix: continuing it would append the
    // call syntax to the prompt as if the model had started emitting it.
    if last
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
    {
        return Ok(None);
    }

    let content = last.content.text();
    if !content.is_empty() {
        return Ok(Some(AssistantPrefill {
            text: content,
            is_reasoning: false,
        }));
    }
    match last.reasoning.as_deref() {
        Some(reasoning) if !reasoning.is_empty() => Ok(Some(AssistantPrefill {
            text: reasoning.to_string(),
            is_reasoning: true,
        })),
        // An empty trailing assistant message is upstream's degenerate
        // continuation: nothing to append, and the generation prompt alone is
        // already what the model continues from.
        _ => Ok(Some(AssistantPrefill {
            text: String::new(),
            is_reasoning: false,
        })),
    }
}

/// Append a resolved prefill to a rendered prompt.
///
/// A content continuation is appended verbatim after the generation prompt.
/// A reasoning continuation is only representable when the generation prompt
/// already opened a thinking block (`enable_thinking` on the Qwen-style and
/// Gemma 4 templates), because the open marker is family-specific and lives in
/// the template rather than here; otherwise the request is refused rather than
/// answered with reasoning text that would read as content.
pub(crate) fn append_to_prompt(
    prompt: &str,
    prefill: &AssistantPrefill,
    primed_thinking_close: Option<&str>,
) -> Result<String, String> {
    if prefill.is_reasoning {
        // A reasoning continuation stays inside the block the prompt opened.
        let Some(_) = primed_thinking_close else {
            return Err(REASONING_ONLY_UNSUPPORTED.to_string());
        };
        return Ok(format!("{prompt}{}", prefill.text));
    }
    // A CONTENT continuation must not land inside a primed thinking block:
    // b10621 closes the (empty) block first, emitting `<think>\n` + reasoning +
    // `\n</think>\n\n` before the content. The prompt already ends with the open
    // marker, so only the closing half is appended here.
    let closed = match primed_thinking_close {
        Some(close) => format!("{prompt}\n{close}\n\n"),
        None => prompt.to_string(),
    };
    if prefill.text.is_empty() {
        return Ok(closed);
    }
    Ok(format!("{closed}{}", prefill.text))
}

#[cfg(test)]
#[path = "assistant_prefill_tests.rs"]
mod tests;
