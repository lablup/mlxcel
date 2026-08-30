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

//! Lazy-grammar trigger gate.
//!
//! While a lazy grammar is awaiting its trigger, b10621 applies no mask at all
//! and instead buffers each emitted token's piece. A trigger token activates
//! immediately; a trigger pattern activates at the byte offset the match starts
//! at, and every buffered token from that offset onward is replayed into the
//! grammar so the constraint sees the text that triggered it.
//!
//! Reference:
//! <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp>
//! (`llama_grammar_accept_impl`, `llama_grammar_trigger_pattern::find`)
//!
//! Used by: server::structured

use fancy_regex::Regex;

use super::{GrammarRequestError, GrammarTrigger};

/// One compiled trigger pattern.
struct CompiledPattern {
    /// Anchored patterns are matched against the entire buffer, unanchored
    /// ones are searched. b10621 decides this by inspecting the pattern text
    /// itself (`pattern.front() == '^' && pattern.back() == '$'`), not by how
    /// the trigger was declared, so a `PATTERN` trigger written with explicit
    /// anchors takes the full-match path too.
    anchored: bool,
    regex: Regex,
}

impl CompiledPattern {
    /// `llama_grammar_trigger_pattern::find`: the byte offset the constrained
    /// text starts at, taken from the first non-empty capture group when the
    /// pattern has one and from the whole match otherwise.
    fn find(&self, input: &str) -> Option<usize> {
        let caps = self.regex.captures(input).ok().flatten()?;
        let whole = caps.get(0)?;
        if self.anchored && (whole.start() != 0 || whole.end() != input.len()) {
            return None;
        }
        for i in 1..caps.len() {
            if let Some(m) = caps.get(i)
                && m.end() > m.start()
            {
                return Some(m.start());
            }
        }
        Some(whole.start())
    }
}

/// What the gate decided about the token just emitted.
pub enum LazyOutcome {
    /// No trigger yet; the grammar stays inert.
    StillWaiting,
    /// A trigger token fired. Feed exactly these token ids into the matcher.
    ActivateTokens(Vec<u32>),
    /// A trigger pattern fired. Feed these bytes into the matcher.
    ActivateBytes(Vec<u8>),
}

/// Buffering gate in front of a lazy grammar.
pub struct LazyGate {
    awaiting: bool,
    buffer: Vec<u8>,
    trigger_tokens: Vec<u32>,
    patterns: Vec<CompiledPattern>,
}

impl std::fmt::Debug for LazyGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyGate")
            .field("awaiting", &self.awaiting)
            .field("buffered_bytes", &self.buffer.len())
            .field("trigger_tokens", &self.trigger_tokens.len())
            .field("patterns", &self.patterns.len())
            .finish()
    }
}

/// Wrap a pattern the way b10621's `PATTERN_FULL` case does: anchor both ends
/// unless the caller already did, and turn an empty pattern into `^$`.
fn anchor_full(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }
    let mut out = String::new();
    if !pattern.starts_with('^') {
        out.push('^');
    }
    out.push_str(pattern);
    if !pattern.ends_with('$') {
        out.push('$');
    }
    out
}

impl LazyGate {
    /// Compile a gate from resolved triggers.
    ///
    /// b10621 matches with `std::regex` (ECMAScript). mlxcel uses
    /// `fancy-regex`, which carries the lookaround and backreference support
    /// that the plain `regex` crate lacks; a pattern it still cannot compile is
    /// refused here rather than silently never firing.
    pub fn new(triggers: &[GrammarTrigger]) -> Result<Self, GrammarRequestError> {
        let mut trigger_tokens = Vec::new();
        let mut patterns = Vec::new();
        for trigger in triggers {
            let raw = match trigger {
                GrammarTrigger::Token(id) => {
                    trigger_tokens.push(*id);
                    continue;
                }
                GrammarTrigger::Word(word) => fancy_regex::escape(word).into_owned(),
                GrammarTrigger::Pattern(pattern) => pattern.clone(),
                GrammarTrigger::PatternFull(pattern) => anchor_full(pattern),
            };
            let anchored = raw.starts_with('^') && raw.ends_with('$');
            let regex = Regex::new(&raw).map_err(|e| {
                GrammarRequestError(format!("invalid grammar trigger pattern {raw:?}: {e}"))
            })?;
            patterns.push(CompiledPattern { anchored, regex });
        }
        Ok(Self {
            awaiting: true,
            buffer: Vec::new(),
            trigger_tokens,
            patterns,
        })
    }

    /// `true` while the grammar must not constrain anything.
    pub fn awaiting(&self) -> bool {
        self.awaiting
    }

    /// Record one emitted token and report whether it activated the grammar.
    pub fn observe(&mut self, token: u32, piece: &[u8]) -> LazyOutcome {
        if !self.awaiting {
            return LazyOutcome::StillWaiting;
        }
        if self.trigger_tokens.contains(&token) {
            self.activate();
            return LazyOutcome::ActivateTokens(vec![token]);
        }

        self.buffer.extend_from_slice(piece);

        // A trailing multi-byte character can be split across two tokens, so
        // match against the valid UTF-8 prefix. Byte offsets in that prefix are
        // byte offsets in the buffer.
        let valid = match std::str::from_utf8(&self.buffer) {
            Ok(_) => self.buffer.len(),
            Err(e) => e.valid_up_to(),
        };
        let Ok(text) = std::str::from_utf8(&self.buffer[..valid]) else {
            return LazyOutcome::StillWaiting;
        };

        for pattern in &self.patterns {
            if let Some(offset) = pattern.find(text) {
                let constrained = self.buffer[offset..].to_vec();
                self.activate();
                return LazyOutcome::ActivateBytes(constrained);
            }
        }
        LazyOutcome::StillWaiting
    }

    fn activate(&mut self) {
        self.awaiting = false;
        self.buffer.clear();
    }
}
