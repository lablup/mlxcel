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

//! Inkling tool calls (#3228): every output segment is a message,
//! `<|message_model|>[recipient]<|content_KIND|>body<|end_message|>`, and a
//! call is the `invoke_tool_json` kind addressed to the function, with a
//! `{"name": NAME, "args": {...}}` body.
//!
//! The markers are special tokens of the Inkling tokenizer (Inkling 1 and
//! Inkling 2 share them), so they are claimed by presence like Harmony's and
//! Kimi K3's. An invoke message whose body is not a valid call is tool markup,
//! not content: it yields no call and is dropped from the visible text, which
//! is what the streaming filter does with it (the #1743 rule).
//!
//! Derived from mlx-vlm `mlx_vlm/tools/parsers/inkling.py` (MIT).

use std::ops::Range;

use super::types::ParsedToolCall;

pub(super) const MESSAGE: &str = "<|message_model|>";
pub(super) const END_MESSAGE: &str = "<|end_message|>";
pub(super) const INVOKE: &str = "<|content_invoke_tool_json|>";
/// Longest recipient (function name) accepted between `<|message_model|>`
/// and the kind marker; the streaming filter holds a header this long.
pub(super) const MAX_RECIPIENT_BYTES: usize = 256;

/// One invoke message: its byte span (header through `<|end_message|>`, or
/// through the end of the text when unclosed) and its JSON body.
struct InvokeMessage<'a> {
    span: Range<usize>,
    body: &'a str,
}

/// Whether `recipient` can sit between `<|message_model|>` and a kind marker.
pub(super) fn is_recipient(recipient: &str) -> bool {
    recipient.len() <= MAX_RECIPIENT_BYTES && !recipient.contains("<|")
}

fn invoke_messages(text: &str) -> Vec<InvokeMessage<'_>> {
    let mut messages = Vec::new();
    let mut from = 0;
    while let Some(rel) = text[from..].find(INVOKE) {
        let invoke = from + rel;
        let start = text[from..invoke]
            .rfind(MESSAGE)
            .map(|pos| from + pos)
            .filter(|&pos| is_recipient(&text[pos + MESSAGE.len()..invoke]))
            .unwrap_or(invoke);
        let body_start = invoke + INVOKE.len();
        let (body_end, end) = match text[body_start..].find(END_MESSAGE) {
            Some(rel) => (body_start + rel, body_start + rel + END_MESSAGE.len()),
            None => (text.len(), text.len()),
        };
        messages.push(InvokeMessage {
            span: start..end,
            body: &text[body_start..body_end],
        });
        from = end;
    }
    messages
}

/// `{"name": str, "args": object}`, with `args` defaulting to `{}`.
fn parse_call(body: &str) -> Option<ParsedToolCall> {
    let value: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    let object = value.as_object()?;
    let name = object.get("name")?.as_str()?;
    let arguments = match object.get("args") {
        None => serde_json::Value::Object(serde_json::Map::new()),
        Some(args) if args.is_object() => args.clone(),
        Some(_) => return None,
    };
    Some(ParsedToolCall {
        name: name.to_string(),
        arguments: arguments.to_string(),
        id: None,
    })
}

/// The text with every invoke message removed.
pub(super) fn strip_invoke_messages(text: &str) -> String {
    if !text.contains(INVOKE) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for message in invoke_messages(text) {
        out.push_str(&text[last..message.span.start]);
        last = message.span.end;
    }
    out.push_str(&text[last..]);
    out
}

/// The valid calls in `text`, in order, plus the text with every invoke
/// message removed; `None` when the text carries no invoke marker.
pub(super) fn extract_calls(text: &str) -> Option<(Vec<ParsedToolCall>, String)> {
    if !text.contains(INVOKE) {
        return None;
    }
    let calls = invoke_messages(text)
        .iter()
        .filter_map(|message| parse_call(message.body))
        .collect();
    Some((calls, strip_invoke_messages(text)))
}
