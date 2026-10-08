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

//! Conservative, context-sensitive adaptation for typed-content-only templates.
//!
//! Probes establish a limited compatibility heuristic, not a proof about every
//! possible conversation. A successful user-only probe never licenses changing
//! an already string-compatible system or assistant branch. Decisions are keyed
//! by the effective render context and the roles being converted, not messages.

use std::borrow::Cow;
use std::collections::{BTreeSet, VecDeque};
use std::io::Write;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use minijinja::Value;
use serde_json::{Value as JsonValue, json};

const CACHE_ENTRIES: usize = 16;
const MAX_CACHE_KEY_BYTES: usize = 16 * 1024;
// Different lengths, initial/final characters and case catch more transformations
// than equal-length markers sharing a prefix (uppercasing, slicing, length, …).
const PROBE_A: &str = "a7!mlx_probe_Z";
const PROBE_B: &str = "Q92_DIFFERENT_text_payload?end6";
const USER_BEFORE: &str = "u3_CONTEXT_before!M";
const USER_AFTER: &str = "v84_CONTEXT_after?N";

type DecisionCache = VecDeque<(Vec<u8>, bool)>;

#[derive(Clone, Default)]
pub(super) struct ContentNormalizer {
    // Clones share bounded decisions keyed by effective context and roles.
    // Keys may retain kwargs/tools, but not message objects or rendered output.
    // No template code runs while this mutex is held.
    cache: Arc<Mutex<DecisionCache>>,
}

impl ContentNormalizer {
    /// Normalize string content only, preserving every other message field.
    ///
    /// The context must be the effective context with messages undefined.
    /// The render callback is unchecked, never another normalization entry.
    /// Used by: checkpoint-backed CLI/server generation and history renders,
    /// including their raw multimodal/tool-message counterparts.
    pub(super) fn normalize<'a>(
        &self,
        messages: &'a JsonValue,
        context: &Value,
        render: impl Fn(Value) -> Result<String>,
    ) -> Cow<'a, JsonValue> {
        let Some(roles) = string_roles(messages) else {
            return Cow::Borrowed(messages);
        };
        if roles.is_empty() {
            return Cow::Borrowed(messages);
        }

        // Metadata can select a different content schema even for the same
        // role. Never retain it in keys or reuse a plain-message decision.
        let has_metadata = has_message_metadata(messages);
        let key = if has_metadata {
            None
        } else {
            cache_key(context, &roles)
        };
        let cached = key.as_ref().and_then(|key| {
            self.cache.lock().ok().and_then(|cache| {
                cache
                    .iter()
                    .find_map(|(entry, decision)| (entry == key).then_some(*decision))
            })
        });
        let should_wrap = cached.unwrap_or_else(|| {
            let decision = infer(context, &roles, messages, has_metadata, &render);
            if let Some(key) = key
                && let Ok(mut cache) = self.cache.lock()
            {
                // Another clone may have populated this key while we rendered.
                if !cache.iter().any(|(entry, _)| entry == &key) {
                    if cache.len() == CACHE_ENTRIES {
                        cache.pop_front();
                    }
                    cache.push_back((key, decision));
                }
            }
            decision
        });
        if !should_wrap {
            return Cow::Borrowed(messages);
        }

        let mut normalized = messages.clone();
        // The array and role/content shape were checked by string_roles.
        if let Some(messages) = normalized.as_array_mut() {
            for message in messages {
                if let Some(content) = message.get_mut("content")
                    && let JsonValue::String(text) = content
                {
                    *content = typed_text(std::mem::take(text));
                }
            }
        }
        Cow::Owned(normalized)
    }
}

fn string_roles(messages: &JsonValue) -> Option<BTreeSet<String>> {
    let mut roles = BTreeSet::new();
    for message in messages.as_array()? {
        if message.get("content").is_some_and(JsonValue::is_string) {
            roles.insert(message.get("role")?.as_str()?.to_owned());
        }
    }
    Some(roles)
}

fn has_message_metadata(messages: &JsonValue) -> bool {
    messages.as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message
                .as_object()
                .is_some_and(|fields| fields.keys().any(|key| key != "role" && key != "content"))
        })
    })
}

/// Replace only the messages slot without rebuilding or logging kwargs.
pub(super) fn with_messages(context: &Value, messages: &JsonValue) -> Value {
    minijinja::context! {
        messages => Value::from_serialize(messages),
        ..context.clone()
    }
}

fn infer(
    context: &Value,
    roles: &BTreeSet<String>,
    messages: &JsonValue,
    has_metadata: bool,
    render: &impl Fn(Value) -> Result<String>,
) -> bool {
    let render_messages = |messages: &JsonValue| render(with_messages(context, messages));

    // A transformation is not a dropped message: require two dissimilar
    // string payloads to produce the same successful output. In particular,
    // rejecting strings must not itself enable normalization.
    let string_a = render_messages(&JsonValue::Array(vec![message("user", PROBE_A, false)]));
    let string_b = render_messages(&JsonValue::Array(vec![message("user", PROBE_B, false)]));
    let (Ok(string_a), Ok(string_b)) = (string_a, string_b) else {
        return false;
    };
    if string_a != string_b {
        return false;
    }
    for (marker, other) in [(PROBE_A, PROBE_B), (PROBE_B, PROBE_A)] {
        let Ok(typed) = render_messages(&JsonValue::Array(vec![message("user", marker, true)]))
        else {
            return false;
        };
        if !typed.contains(marker) || typed.contains(other) {
            return false;
        }
    }

    if has_metadata {
        // Inspect every string-message prototype, including user messages:
        // name, tool calls, reasoning and other values can select a schema.
        // Only the content changes in these probes; actual metadata survives.
        for prototype in messages.as_array().into_iter().flatten() {
            if prototype.get("content").is_some_and(JsonValue::is_string)
                && !prototype_can_wrap(prototype, &render_messages)
            {
                return false;
            }
        }
    } else {
        // Absent roles do not block an otherwise valid user-only template.
        for role in roles.iter().filter(|role| role.as_str() != "user") {
            if !prototype_can_wrap(&message(role, "", false), &render_messages) {
                return false;
            }
        }
    }
    true
}

fn prototype_can_wrap(
    prototype: &JsonValue,
    render: &impl Fn(&JsonValue) -> Result<String>,
) -> bool {
    for (marker, other) in [(PROBE_A, PROBE_B), (PROBE_B, PROBE_A)] {
        let (messages, required) = prototype_scaffold(prototype, marker, true);
        let Ok(typed) = render(&messages) else {
            return false;
        };
        if typed.contains(other) || required.iter().any(|text| !typed.contains(text.as_str())) {
            return false;
        }
    }

    let string_a = render(&prototype_scaffold(prototype, PROBE_A, false).0);
    let string_b = render(&prototype_scaffold(prototype, PROBE_B, false).0);
    match (string_a, string_b) {
        (Ok(a), Ok(b)) => a == b,
        // Granite Vision 3.2 indexes system/assistant content[0].text, so a
        // string in those roles is an engine error. Accept this only AFTER
        // the successful user silent-drop gate and both typed-role renders
        // preserved all markers. Explicit rejections still veto adaptation,
        // as do inconsistent success/error outcomes.
        (Err(a), Err(b)) => {
            super::template_rejection_message(&a).is_none()
                && super::template_rejection_message(&b).is_none()
        }
        _ => false,
    }
}

// Used by: checkpoint-backed string normalization and compatibility probes.
fn typed_text(text: String) -> JsonValue {
    // Build the part from owned values: json! would serialize and clone text.
    let part = serde_json::Map::from_iter([
        ("type".to_owned(), JsonValue::String("text".to_owned())),
        ("text".to_owned(), JsonValue::String(text)),
    ]);
    JsonValue::Array(vec![JsonValue::Object(part)])
}

fn message(role: &str, text: &str, typed: bool) -> JsonValue {
    let content = if typed {
        typed_text(text.to_owned())
    } else {
        json!(text)
    };
    json!({"role": role, "content": content})
}

fn prototype_scaffold(
    prototype: &JsonValue,
    marker: &str,
    typed: bool,
) -> (JsonValue, Vec<String>) {
    // Used by: checkpoint-backed text/raw generation and history probes.
    // string_roles has already validated each converted message's role.
    // Preserve metadata without cloning the potentially large, discarded body.
    let target = JsonValue::Object(
        prototype
            .as_object()
            .expect("string-content messages have validated roles")
            .iter()
            .map(|(key, value)| {
                let value = if key == "content" {
                    if typed {
                        typed_text(marker.to_owned())
                    } else {
                        JsonValue::String(marker.to_owned())
                    }
                } else {
                    value.clone()
                };
                (key.clone(), value)
            })
            .collect(),
    );
    match prototype.get("role").and_then(JsonValue::as_str) {
        Some("user") => (JsonValue::Array(vec![target]), vec![marker.to_owned()]),
        Some("system") => (
            JsonValue::Array(vec![target, message("user", USER_AFTER, true)]),
            vec![marker.to_owned(), USER_AFTER.to_owned()],
        ),
        _ => (
            JsonValue::Array(vec![
                message("user", USER_BEFORE, true),
                target,
                message("user", USER_AFTER, true),
            ]),
            vec![
                USER_BEFORE.to_owned(),
                marker.to_owned(),
                USER_AFTER.to_owned(),
            ],
        ),
    }
}

fn cache_key(context: &Value, roles: &BTreeSet<String>) -> Option<Vec<u8>> {
    // Bound serialized key bytes, not allocator/container overhead. Oversized
    // contexts are still checked; their decisions are simply not cached.
    let mut writer = KeyWriter(Vec::new());
    serde_json::to_writer(&mut writer, &(context, roles)).ok()?;
    Some(writer.0)
}

struct KeyWriter(Vec<u8>);

impl Write for KeyWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_CACHE_KEY_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other(
                "content-adaptation cache key too large",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "chat_template_content_cache_tests.rs"]
mod cache_tests;
