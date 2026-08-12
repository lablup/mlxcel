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

//! ATEM tool-call parser used by Muse/Onyx chat templates.

use super::types::{ParsedToolCall, ToolCallFormat, ToolCallParseResult};

const FUNCTION_CALLS_OPEN: &str = "<atem:function_calls>";
const FUNCTION_CALLS_CLOSE: &str = "</atem:function_calls>";
const INVOKE_OPEN: &str = "<atem:invoke";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const PARAM_OPEN: &str = "<atem:parameter";
const PARAM_CLOSE: &str = "</atem:parameter>";

const ATEM_MAX_INPUT_BYTES: usize = 1 << 20;
const ATEM_MAX_CALLS: usize = 128;
const ATEM_MAX_PARAMS_PER_CALL: usize = 128;
const ATEM_MAX_PARAM_NAME_BYTES: usize = 256;
const ATEM_MAX_ARGUMENT_BYTES: usize = 64 * 1024;

/// Parse Muse/Onyx ATEM tool calls.
///
/// Grammar:
///
/// ```text
/// <atem:function_calls>
/// <atem:invoke name="tool.name">
/// <atem:parameter name="arg">value</atem:parameter>
/// </atem:invoke>
/// </atem:function_calls>
/// ```
///
/// Parameter values follow the template contract: JSON objects/arrays and JSON
/// scalars are parsed as JSON-compatible values; otherwise the exact raw string
/// between the parameter tags is preserved.
pub fn try_atem(text: &str) -> Option<ToolCallParseResult> {
    let text = capped_prefix(text, ATEM_MAX_INPUT_BYTES);
    let first = text.find(FUNCTION_CALLS_OPEN)?;
    let mut calls = Vec::new();
    let mut content = String::with_capacity(text.len().min(4096));
    content.push_str(&text[..first]);

    let mut remaining = &text[first..];
    while let Some(start) = remaining.find(FUNCTION_CALLS_OPEN) {
        content.push_str(&remaining[..start]);
        let body_start = start + FUNCTION_CALLS_OPEN.len();
        let (body, next) = match remaining[body_start..].find(FUNCTION_CALLS_CLOSE) {
            Some(end) => (
                &remaining[body_start..body_start + end],
                body_start + end + FUNCTION_CALLS_CLOSE.len(),
            ),
            None => (&remaining[body_start..], remaining.len()),
        };

        parse_invokes(body, &mut calls);
        if calls.len() >= ATEM_MAX_CALLS {
            remaining = "";
            break;
        }
        remaining = &remaining[next..];
    }
    content.push_str(remaining);

    if calls.is_empty() {
        return None;
    }
    let (content, reasoning_content) = split_muse_channels(&content);
    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::Atem),
        tool_calls: calls,
        content: clean_atem_channel_text(&content),
        reasoning_content,
    })
}

/// Split the recipient-oriented message envelope used by Muse/Onyx.
///
/// The chat template primes `<|start|>assistant`, so the first generated
/// message commonly starts with only ` to=self<|message|>`. Subsequent
/// messages include the full `<|start|>assistant to=...<|message|>` header.
/// Reasoning addressed to `self` must not leak into user-visible content, and
/// a tool-recipient header must remain hidden after its ATEM payload has been
/// extracted.
pub(super) fn split_muse_channels(text: &str) -> (String, Option<String>) {
    const START: &str = "<|start|>assistant";
    const MESSAGE: &str = "<|message|>";
    const TERMINATORS: &[&str] = &["<|eom|>", "<|eot|>", START];

    let trimmed = text.trim();
    if !trimmed.contains(MESSAGE)
        || !(trimmed.starts_with("to=") || trimmed.starts_with(" to=") || trimmed.contains(START))
    {
        return (text.to_string(), None);
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut rest = trimmed;

    while !rest.is_empty() {
        let (header_start, header_len) = if rest.starts_with("to=") {
            (0, 0)
        } else if rest.starts_with(" to=") {
            (1, 0)
        } else if let Some(start) = rest.find(START) {
            let prefix = rest[..start].trim();
            if !prefix.is_empty() {
                append_channel_text(&mut content, prefix);
            }
            (start, START.len())
        } else {
            append_channel_text(&mut content, rest);
            break;
        };

        let after_start = &rest[header_start + header_len..];
        let Some(message_pos) = after_start.find(MESSAGE) else {
            break;
        };
        let header = after_start[..message_pos].trim();
        let body_region = &after_start[message_pos + MESSAGE.len()..];
        let body_end = TERMINATORS
            .iter()
            .filter_map(|marker| body_region.find(marker))
            .min()
            .unwrap_or(body_region.len());
        let body = body_region[..body_end].trim();

        let recipient = header
            .strip_prefix("to=")
            .and_then(|value| value.split_whitespace().next())
            .unwrap_or("user");
        match recipient {
            "self" => append_channel_text(&mut reasoning, body),
            "user" => append_channel_text(&mut content, body),
            _ => {}
        }

        rest = &body_region[body_end..];
        if let Some(tail) = rest.strip_prefix("<|eom|>") {
            rest = tail;
        } else if let Some(tail) = rest.strip_prefix("<|eot|>") {
            rest = tail;
        }
        rest = rest.trim_start();
    }

    let reasoning = reasoning.trim();
    (
        content.trim().to_string(),
        (!reasoning.is_empty()).then(|| reasoning.to_string()),
    )
}

/// Render Muse/Onyx recipient-oriented channels for terminal display.
///
/// Returns `None` when `text` is not a Muse envelope so callers can fall back
/// to the regular reasoning-marker filter. Reasoning addressed to `self` is
/// hidden by default and shown before user-visible content when requested.
pub fn render_muse_channels_for_display(
    text: &str,
    show_reasoning: bool,
    dim_reasoning: bool,
) -> Option<String> {
    let trimmed = text.trim();
    if !trimmed.contains("<|message|>")
        || !(trimmed.starts_with("to=")
            || trimmed.starts_with(" to=")
            || trimmed.contains("<|start|>assistant"))
    {
        return None;
    }

    let (content, reasoning) = split_muse_channels(text);
    let mut rendered = String::new();
    if show_reasoning && let Some(reasoning) = reasoning {
        if dim_reasoning {
            rendered.push_str("\x1b[2m");
            rendered.push_str(&reasoning);
            rendered.push_str("\x1b[0m");
        } else {
            rendered.push_str(&reasoning);
        }
        if !content.is_empty() {
            rendered.push('\n');
        }
    }
    rendered.push_str(&content);
    Some(rendered)
}

fn append_channel_text(target: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !target.is_empty() {
        target.push('\n');
    }
    target.push_str(text);
}

/// Remove ATEM control markup from user-visible content.
pub fn strip_atem_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find(FUNCTION_CALLS_OPEN) {
        out.push_str(&remaining[..start]);
        let body_start = start + FUNCTION_CALLS_OPEN.len();
        match remaining[body_start..].find(FUNCTION_CALLS_CLOSE) {
            Some(end) => {
                remaining = &remaining[body_start + end + FUNCTION_CALLS_CLOSE.len()..];
            }
            None => {
                remaining = "";
                break;
            }
        }
    }
    out.push_str(remaining);
    strip_stray_atem_tags(&out)
}

fn parse_invokes(body: &str, calls: &mut Vec<ParsedToolCall>) {
    let mut remaining = body;
    while let Some(start) = remaining.find(INVOKE_OPEN) {
        let tag_start = start + INVOKE_OPEN.len();
        let Some(tag_end) = remaining[tag_start..].find('>') else {
            break;
        };
        let attrs = &remaining[tag_start..tag_start + tag_end];
        let body_start = tag_start + tag_end + 1;
        let (invoke_body, next) = match remaining[body_start..].find(INVOKE_CLOSE) {
            Some(end) => (
                &remaining[body_start..body_start + end],
                body_start + end + INVOKE_CLOSE.len(),
            ),
            None => (&remaining[body_start..], remaining.len()),
        };

        if let Some(name) = extract_attr(attrs, "name")
            && !name.is_empty()
        {
            let arguments = parse_parameters(invoke_body);
            calls.push(ParsedToolCall { name, arguments });
            if calls.len() >= ATEM_MAX_CALLS {
                break;
            }
        }
        remaining = &remaining[next..];
    }
}

fn parse_parameters(body: &str) -> String {
    let mut map = serde_json::Map::new();
    let mut remaining = body;
    let mut scanned = 0usize;
    let mut arg_bytes = 0usize;

    while let Some(start) = remaining.find(PARAM_OPEN) {
        let tag_start = start + PARAM_OPEN.len();
        let Some(tag_end) = remaining[tag_start..].find('>') else {
            break;
        };
        let attrs = &remaining[tag_start..tag_start + tag_end];
        let value_start = tag_start + tag_end + 1;
        let (raw_value, next) = match remaining[value_start..].find(PARAM_CLOSE) {
            Some(end) => (
                &remaining[value_start..value_start + end],
                value_start + end + PARAM_CLOSE.len(),
            ),
            None => (&remaining[value_start..], remaining.len()),
        };
        scanned += 1;

        if let Some(name) = extract_attr(attrs, "name")
            && !name.is_empty()
            && name.len() <= ATEM_MAX_PARAM_NAME_BYTES
        {
            let added = name.len().saturating_add(raw_value.len());
            if arg_bytes.saturating_add(added) > ATEM_MAX_ARGUMENT_BYTES {
                break;
            }
            arg_bytes += added;
            map.insert(name, coerce_atem_value(raw_value));
        }

        if scanned >= ATEM_MAX_PARAMS_PER_CALL {
            break;
        }
        remaining = &remaining[next..];
    }

    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".to_string())
}

fn coerce_atem_value(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if should_try_json(trimmed)
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
    {
        return value;
    }
    serde_json::Value::String(raw.to_string())
}

fn should_try_json(value: &str) -> bool {
    value.starts_with('{')
        || value.starts_with('[')
        || value.starts_with('"')
        || value.starts_with('-')
        || value.chars().next().is_some_and(|c| c.is_ascii_digit())
        || matches!(value, "true" | "false" | "null")
}

fn extract_attr(attrs: &str, name: &str) -> Option<String> {
    let bytes = attrs.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let key_start = pos;
        while pos < bytes.len() && bytes[pos] != b'=' && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let key = &attrs[key_start..pos];
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] != b'=' {
            while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            continue;
        }
        pos += 1;
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }

        let value = &attrs[pos..];
        let parsed = if let Some(quote) = value.chars().next().filter(|c| *c == '"' || *c == '\'') {
            let after_quote = quote.len_utf8();
            let end = value[after_quote..].find(quote)?;
            pos += after_quote + end + quote.len_utf8();
            value[after_quote..after_quote + end].to_string()
        } else {
            let end = value.find(char::is_whitespace).unwrap_or(value.len());
            pos += end;
            value[..end].trim_end_matches('>').to_string()
        };
        if key == name {
            return Some(parsed);
        }
    }
    None
}

fn capped_prefix(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn clean_atem_channel_text(text: &str) -> String {
    strip_stray_atem_tags(text).trim().to_string()
}

fn strip_stray_atem_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;
    loop {
        let Some(start) = remaining.find("<atem:") else {
            out.push_str(remaining);
            break;
        };
        out.push_str(&remaining[..start]);
        let Some(end) = remaining[start..].find('>') else {
            break;
        };
        remaining = &remaining[start + end + 1..];
    }
    out.replace(FUNCTION_CALLS_CLOSE, "")
        .replace(INVOKE_CLOSE, "")
        .replace(PARAM_CLOSE, "")
}

#[cfg(test)]
#[path = "atem_tests.rs"]
mod tests;
