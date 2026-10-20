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

//! Byte-level JSON structure scanning shared by the final tool-call parsers
//! and the incremental streamer (#2396).
//!
//! The scanners do not validate JSON. The final parsers validate a body with
//! `serde_json` first and then use [`object_members`] / [`array_elements`] to
//! slice the raw text of a member; the streamer feeds the same
//! [`ObjectWalker`] one byte at a time. Both sides therefore agree on where a
//! value starts and ends by construction.
//!
//! Every structural byte is ASCII, and an ASCII byte never occurs inside a
//! multi-byte UTF-8 sequence, so every boundary these scanners report is a
//! `char` boundary.
//!
//! Used by: tool_calls::formats, tool_calls::incremental

/// Longest member key [`ObjectWalker`] keeps. Longer keys are reported as
/// overflowed and never match a key the callers look for.
const MAX_KEY_BYTES: usize = 256;

/// The shape of a value, decided by its first byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueKind {
    Object,
    Array,
    String,
    /// Number, `true`, `false` or `null`.
    Scalar,
}

impl ValueKind {
    fn of(first: u8) -> Self {
        match first {
            b'{' => Self::Object,
            b'[' => Self::Array,
            b'"' => Self::String,
            _ => Self::Scalar,
        }
    }
}

/// Result of pushing one byte into a [`ValueScanner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValuePush {
    /// The byte belongs to the value and the value continues.
    Continue,
    /// The byte is the last byte of the value.
    EndIncl,
    /// The value (a scalar) ended before this byte; the byte is not part of it.
    EndBefore,
}

/// Tracks where one JSON value ends, starting from its first byte.
#[derive(Debug, Clone)]
pub(super) struct ValueScanner {
    kind: ValueKind,
    depth: u32,
    in_string: bool,
    escaped: bool,
}

impl ValueScanner {
    /// Start a value whose first byte is `first`. The caller has already
    /// consumed `first`; call [`ValueScanner::first_push`] to learn whether
    /// that byte alone completed the value.
    pub(super) fn start(first: u8) -> Self {
        let kind = ValueKind::of(first);
        Self {
            kind,
            depth: u32::from(matches!(kind, ValueKind::Object | ValueKind::Array)),
            in_string: kind == ValueKind::String,
            escaped: false,
        }
    }

    pub(super) fn kind(&self) -> ValueKind {
        self.kind
    }

    /// Push the byte after the first one.
    pub(super) fn push(&mut self, b: u8) -> ValuePush {
        match self.kind {
            ValueKind::Scalar => {
                if matches!(b, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    ValuePush::EndBefore
                } else {
                    ValuePush::Continue
                }
            }
            ValueKind::String => {
                if self.escaped {
                    self.escaped = false;
                } else if b == b'\\' {
                    self.escaped = true;
                } else if b == b'"' {
                    return ValuePush::EndIncl;
                }
                ValuePush::Continue
            }
            ValueKind::Object | ValueKind::Array => {
                if self.in_string {
                    if self.escaped {
                        self.escaped = false;
                    } else if b == b'\\' {
                        self.escaped = true;
                    } else if b == b'"' {
                        self.in_string = false;
                    }
                    return ValuePush::Continue;
                }
                match b {
                    b'"' => self.in_string = true,
                    b'{' | b'[' => self.depth += 1,
                    b'}' | b']' => {
                        self.depth = self.depth.saturating_sub(1);
                        if self.depth == 0 {
                            return ValuePush::EndIncl;
                        }
                    }
                    _ => {}
                }
                ValuePush::Continue
            }
        }
    }
}

/// What one byte meant to an [`ObjectWalker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ObjectStep {
    /// Structure, whitespace or key bytes: nothing for the caller to do.
    Skip,
    /// A member key just closed; read it with [`ObjectWalker::key`].
    Key,
    /// This byte is the first byte of the current member's value.
    ValueStart(ValueKind),
    /// This byte continues the current member's value.
    ValueByte,
    /// This byte is the last byte of the current member's value.
    ValueEnd,
    /// The current (scalar) value ended before this byte. The byte was not
    /// consumed: push it again.
    ValueEndBefore,
    /// The object's closing `}`.
    ObjectEnd,
    /// The bytes cannot be a JSON object.
    Invalid,
}

#[derive(Debug, Clone)]
enum WalkState {
    /// Before the opening `{`.
    Open,
    /// After `{` or `,`: a key or (after `{` only) the closing `}`.
    KeyOrEnd {
        first: bool,
    },
    Key {
        escaped: bool,
    },
    Colon,
    ValueStart,
    Value(ValueScanner),
    AfterValue,
    Done,
    Invalid,
}

/// Walks one JSON object's top-level members a byte at a time.
#[derive(Debug, Clone)]
pub(super) struct ObjectWalker {
    state: WalkState,
    key: String,
    key_overflow: bool,
}

impl Default for ObjectWalker {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectWalker {
    pub(super) fn new() -> Self {
        Self {
            state: WalkState::Open,
            key: String::new(),
            key_overflow: false,
        }
    }

    /// The decoded key of the current member, or `None` when it overflowed
    /// [`MAX_KEY_BYTES`] or is not a valid JSON string.
    pub(super) fn key(&self) -> Option<String> {
        if self.key_overflow {
            return None;
        }
        if !self.key.contains('\\') {
            return Some(self.key.clone());
        }
        serde_json::from_str::<String>(&format!("\"{}\"", self.key)).ok()
    }

    /// Push one byte of the object text.
    pub(super) fn push(&mut self, b: u8) -> ObjectStep {
        match &mut self.state {
            WalkState::Open => {
                if is_ws(b) {
                    ObjectStep::Skip
                } else if b == b'{' {
                    self.state = WalkState::KeyOrEnd { first: true };
                    ObjectStep::Skip
                } else {
                    self.fail()
                }
            }
            WalkState::KeyOrEnd { first } => {
                let first = *first;
                if is_ws(b) {
                    ObjectStep::Skip
                } else if b == b'"' {
                    self.key.clear();
                    self.key_overflow = false;
                    self.state = WalkState::Key { escaped: false };
                    ObjectStep::Skip
                } else if b == b'}' && first {
                    self.state = WalkState::Done;
                    ObjectStep::ObjectEnd
                } else {
                    self.fail()
                }
            }
            WalkState::Key { escaped } => {
                if !*escaped && b == b'"' {
                    self.state = WalkState::Colon;
                    return ObjectStep::Key;
                }
                *escaped = !*escaped && b == b'\\';
                if self.key.len() < MAX_KEY_BYTES {
                    push_key_byte(&mut self.key, b);
                } else {
                    self.key_overflow = true;
                }
                ObjectStep::Skip
            }
            WalkState::Colon => {
                if is_ws(b) {
                    ObjectStep::Skip
                } else if b == b':' {
                    self.state = WalkState::ValueStart;
                    ObjectStep::Skip
                } else {
                    self.fail()
                }
            }
            WalkState::ValueStart => {
                if is_ws(b) {
                    return ObjectStep::Skip;
                }
                if matches!(b, b'}' | b']' | b',' | b':') {
                    return self.fail();
                }
                let scanner = ValueScanner::start(b);
                let kind = scanner.kind();
                self.state = WalkState::Value(scanner);
                ObjectStep::ValueStart(kind)
            }
            WalkState::Value(scanner) => match scanner.push(b) {
                ValuePush::Continue => ObjectStep::ValueByte,
                ValuePush::EndIncl => {
                    self.state = WalkState::AfterValue;
                    ObjectStep::ValueEnd
                }
                ValuePush::EndBefore => {
                    self.state = WalkState::AfterValue;
                    ObjectStep::ValueEndBefore
                }
            },
            WalkState::AfterValue => {
                if is_ws(b) {
                    ObjectStep::Skip
                } else if b == b',' {
                    self.state = WalkState::KeyOrEnd { first: false };
                    ObjectStep::Skip
                } else if b == b'}' {
                    self.state = WalkState::Done;
                    ObjectStep::ObjectEnd
                } else {
                    self.fail()
                }
            }
            WalkState::Done | WalkState::Invalid => self.fail(),
        }
    }

    fn fail(&mut self) -> ObjectStep {
        self.state = WalkState::Invalid;
        ObjectStep::Invalid
    }
}

/// Append one raw key byte. Bytes arrive one at a time, so a multi-byte
/// character would be split across calls: ASCII bytes are kept verbatim and
/// each non-ASCII byte becomes a `U+0001` placeholder. The keys callers match
/// (`name`, `arguments`, `parameters`, `id`) are ASCII, and `\u` escapes are
/// decoded by `key()`, so a placeholder only turns a non-ASCII key into
/// another key that matches nothing.
fn push_key_byte(key: &mut String, b: u8) {
    if b.is_ascii() {
        key.push(b as char);
    } else {
        key.push('\u{1}');
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// The members of the JSON object `text` (surrounding whitespace allowed), as
/// `(decoded key, raw value text)` pairs in source order. `None` when `text`
/// is not a well-formed object at the structural level. Callers validate the
/// text with `serde_json` first.
pub(super) fn object_members(text: &str) -> Option<Vec<(Option<String>, &str)>> {
    let bytes = text.as_bytes();
    let mut walker = ObjectWalker::new();
    let mut members = Vec::new();
    let mut key: Option<String> = None;
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match walker.push(bytes[i]) {
            ObjectStep::Key => key = walker.key(),
            ObjectStep::ValueStart(_) => start = i,
            ObjectStep::ValueEnd => members.push((key.take(), &text[start..=i])),
            ObjectStep::ValueEndBefore => {
                members.push((key.take(), &text[start..i]));
                // Not consumed: push the same byte again.
                continue;
            }
            ObjectStep::ObjectEnd => {
                return text[i + 1..].trim().is_empty().then_some(members);
            }
            ObjectStep::Invalid => return None,
            ObjectStep::Skip | ObjectStep::ValueByte => {}
        }
        i += 1;
    }
    None
}

/// The raw text of the last member named `key` in the JSON object `text`, as
/// `serde_json::Value::get` resolves a repeated key (the last one wins).
pub(super) fn raw_member<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    object_members(text)?
        .into_iter()
        .rev()
        .find(|(k, _)| k.as_deref() == Some(key))
        .map(|(_, raw)| raw)
}

/// The raw text of each element of the JSON array `text` (surrounding
/// whitespace allowed). `None` when `text` is not a well-formed array at the
/// structural level.
pub(super) fn array_elements(text: &str) -> Option<Vec<&str>> {
    let trimmed = text.trim();
    let offset = text.len() - text.trim_start().len();
    let bytes = trimmed.as_bytes();
    if bytes.first() != Some(&b'[') || bytes.last() != Some(&b']') {
        return None;
    }
    let mut elements = Vec::new();
    let mut i = 1usize;
    let end = bytes.len() - 1;
    loop {
        while i < end && is_ws(bytes[i]) {
            i += 1;
        }
        if i >= end {
            break;
        }
        let start = i;
        let mut scanner = ValueScanner::start(bytes[i]);
        i += 1;
        let mut value_end = None;
        if scanner.kind() == ValueKind::Scalar {
            while i < end {
                if scanner.push(bytes[i]) == ValuePush::EndBefore {
                    break;
                }
                i += 1;
            }
            value_end = Some(i);
        } else {
            while i < end {
                let push = scanner.push(bytes[i]);
                i += 1;
                if push == ValuePush::EndIncl {
                    value_end = Some(i);
                    break;
                }
            }
        }
        let value_end = value_end?;
        elements.push(&text[offset + start..offset + value_end]);
        while i < end && is_ws(bytes[i]) {
            i += 1;
        }
        if i >= end {
            break;
        }
        if bytes[i] != b',' {
            return None;
        }
        i += 1;
    }
    Some(elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_member_keeps_the_model_spacing() {
        let text = r#" {"name": "f", "arguments": {"city": "Paris",  "n": 2}} "#;
        assert_eq!(raw_member(text, "name"), Some(r#""f""#));
        assert_eq!(
            raw_member(text, "arguments"),
            Some(r#"{"city": "Paris",  "n": 2}"#)
        );
    }

    #[test]
    fn raw_member_resolves_scalars_strings_and_repeats() {
        let text = r#"{"a":1,"b":"x\"}","a":{"k":[1,{"z":"}"}]}}"#;
        assert_eq!(raw_member(text, "b"), Some(r#""x\"}""#));
        assert_eq!(raw_member(text, "a"), Some(r#"{"k":[1,{"z":"}"}]}"#));
        let scalar = r#"{"n": 12 , "t":true}"#;
        assert_eq!(raw_member(scalar, "n"), Some("12"));
        assert_eq!(raw_member(scalar, "t"), Some("true"));
    }

    #[test]
    fn escaped_keys_are_decoded() {
        let text = r#"{"arguments": {"x": 1}}"#;
        assert_eq!(raw_member(text, "arguments"), Some(r#"{"x": 1}"#));
    }

    #[test]
    fn array_elements_slices_each_value() {
        let text = r#" [ {"a": [1, 2]}, "s,]", 3 ,null] "#;
        assert_eq!(
            array_elements(text),
            Some(vec![r#"{"a": [1, 2]}"#, r#""s,]""#, "3", "null"])
        );
        assert_eq!(array_elements("[]"), Some(vec![]));
        assert_eq!(array_elements("{}"), None);
    }

    #[test]
    fn malformed_objects_are_rejected() {
        assert_eq!(object_members(r#"{"a" 1}"#), None);
        assert_eq!(object_members(r#"{"a": 1} trailing"#), None);
        assert_eq!(object_members(r#"{"a": 1"#), None);
        assert_eq!(object_members("{}"), Some(vec![]));
    }
}
