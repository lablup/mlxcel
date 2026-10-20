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

//! Incremental tool-call argument streaming (#2396).
//!
//! [`IncrementalToolCallStreamer`] is fed the same raw decoded text the
//! streaming routes accumulate and reports [`ToolCallStreamEvent`]s as a call's
//! name and argument text arrive, so Chat Completions and Responses can stream
//! `arguments` fragments instead of one burst at the end. The end-of-generation
//! [`parse_tool_calls`](super::parse_tool_calls) stays authoritative:
//! [`IncrementalToolCallStreamer::plan_final`] reconciles what was streamed with
//! the parsed result.
//!
//! Formats streamed: Hermes `<tool_call>{json}</tool_call>`, Mistral
//! `[TOOL_CALLS]NAME[ARGS]{json}` and the `[TOOL_CALLS] [{..}]` list, Llama 3
//! JSON (bare, or after `<|python_tag|>`), and Qwen3-Coder
//! `<tool_call><function=..><parameter=..>` XML. Every other format produces no
//! events, and the routes emit its calls in one shot as before.
//!
//! The streamer mirrors the final parsers' gates: thinking blocks are skipped,
//! Mistral and bare JSON are claimed only at the start of the visible text,
//! names pass the same tool filter and name rules (`match_qwen_name`, the
//! Mistral header regex), and JSON argument deltas are byte slices of the
//! object the final parser keeps verbatim. Input is processed once; only a
//! bounded tail (a partial marker or an undecided name) is held.
//!
//! Used by: routes/chat, routes/responses

mod json_call;
mod mistral;
mod qwen;
mod reconcile;

pub use self::reconcile::{FinalCallPlan, FinalPlan};

use std::collections::HashSet;

use self::json_call::{JsonCallState, JsonContainer, JsonFamily};
use self::mistral::MistralState;
use self::qwen::QwenState;
use super::parser::{THINKING_MARKER_PAIRS, resolve_tool_name};
use crate::server::types::request::Tool;

/// Longest held argument text: a string-encoded `arguments` value, arguments
/// that arrive before the call's name, or a non-string Qwen3-Coder value. A
/// call whose held text would grow past this stops streaming; the final
/// emission sends the rest.
const MAX_HELD_BYTES: usize = 1 << 20;

/// Longest name or header text held while it is still undecided.
const MAX_NAME_BYTES: usize = 256;

/// One incremental tool-call event.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolCallStreamEvent {
    /// A call's name is known. `index` is the call's position among the
    /// tool-known calls, which is the position the final parse reports it at.
    /// `id` is the call id every later emission for this call must reuse.
    Start {
        index: usize,
        name: String,
        id: String,
    },
    /// The next fragment of the call's `arguments` string.
    ArgsDelta { index: usize, delta: String },
}

/// A call the streamer started, with everything it sent for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedToolCall {
    pub index: usize,
    pub name: String,
    pub id: String,
    /// Concatenation of every `ArgsDelta` sent for this call.
    pub sent: String,
}

/// The format family a stream committed to with its first tool-known call.
/// Later calls of another family are not streamed, as the final dispatcher
/// keeps only one format's calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    HermesJson,
    Qwen3Coder,
    Mistral,
    Llama3,
}

/// Per-call bookkeeping shared by every format.
#[derive(Debug, Clone, Default)]
struct CallCtx {
    /// Index among tool-known calls, once the name resolved.
    ordinal: Option<usize>,
    /// Position in `calls` once started.
    started: Option<usize>,
    /// Resolved (tool-filtered) name.
    name: Option<String>,
    /// Never streamed: unknown tool, named `tool_choice` mismatch.
    suppressed: bool,
    /// Stopped streaming arguments; the final emission sends the rest.
    bailed: bool,
    /// Extra gate a format sets before `Start` may fire.
    gate_open: bool,
    /// Argument text produced before `Start`.
    held: String,
    /// Model-supplied id seen before `Start` (Mistral list form).
    model_id: Option<String>,
}

#[derive(Debug)]
enum Mode {
    Outer,
    Thinking { closes: Vec<&'static str> },
    HermesBody,
    SkipUntil { marker: &'static str },
    Json(Box<JsonCallState>),
    List(json_call::ListState),
    Mistral(Box<MistralState>),
    Qwen(Box<QwenState>),
    Done,
}

/// Outcome of one processing step over the unprocessed input.
enum Progress {
    /// Consumed this many bytes and possibly switched mode.
    Consumed(usize),
    /// The rest cannot be decided yet; keep it for the next fragment.
    NeedMore,
}

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const TOOL_CALLS_MARKER: &str = "[TOOL_CALLS]";
const PYTHON_TAG: &str = "<|python_tag|>";

/// Streams tool-call names and argument fragments as raw text arrives.
#[derive(Debug)]
pub struct IncrementalToolCallStreamer {
    pending: String,
    mode: Mode,
    /// No visible (non-thinking, non-whitespace) text seen yet.
    at_start: bool,
    family: Option<Family>,
    /// Tool-known calls seen so far (the next call's index).
    ordinals: usize,
    calls: Vec<StreamedToolCall>,
    tool_names: HashSet<String>,
    tools: Vec<Tool>,
    specific_function: Option<String>,
    /// Every valid model-supplied id seen, so a repeat is not reused.
    seen_model_ids: HashSet<String>,
    /// Qwen3-Coder calls read, against `QWEN3_CODER_MAX_CALLS`.
    qwen_calls: usize,
    events: Vec<ToolCallStreamEvent>,
}

impl IncrementalToolCallStreamer {
    /// A streamer for a request declaring `tools`. `specific_function` is the
    /// named `tool_choice` function, if any: other calls keep their index but
    /// are never streamed. `primed_open_thinking` starts inside a thinking
    /// block, as the stream filter does for a prompt that primed one.
    pub fn new(
        tools: &[Tool],
        specific_function: Option<&str>,
        primed_open_thinking: bool,
    ) -> Self {
        let mode = if primed_open_thinking {
            Mode::Thinking {
                closes: THINKING_MARKER_PAIRS.iter().map(|(_, c)| *c).collect(),
            }
        } else {
            Mode::Outer
        };
        Self {
            pending: String::new(),
            mode,
            at_start: true,
            family: None,
            ordinals: 0,
            calls: Vec::new(),
            tool_names: tools.iter().map(|t| t.function.name.clone()).collect(),
            tools: tools.to_vec(),
            specific_function: specific_function.map(str::to_string),
            seen_model_ids: HashSet::new(),
            qwen_calls: 0,
            events: Vec::new(),
        }
    }

    /// The calls started so far, in start order.
    pub fn calls(&self) -> &[StreamedToolCall] {
        &self.calls
    }

    /// Feed the next decoded fragment; returns the events it produced.
    ///
    /// Model output is untrusted and this runs inside the routes' per-token
    /// callbacks (under the Chat callback's state mutex), so a panic in the
    /// scanner is contained here: the fragment's events are dropped, the
    /// streamer stops, and the end-of-generation emission sends every call
    /// from the authoritative parse.
    pub fn feed(&mut self, fragment: &str) -> Vec<ToolCallStreamEvent> {
        self.contained(|st| st.feed_inner(fragment))
    }

    /// Run `step` and contain a panic it raises (see [`Self::feed`]).
    fn contained(
        &mut self,
        step: impl FnOnce(&mut Self) -> Vec<ToolCallStreamEvent>,
    ) -> Vec<ToolCallStreamEvent> {
        if matches!(self.mode, Mode::Done) {
            return Vec::new();
        }
        // Only the newest call can receive argument text: a format's call
        // context ends before the next call starts.
        let calls_before = self.calls.len();
        let last_sent = self.calls.last().map(|c| c.sent.len());
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| step(self))) {
            Ok(events) => events,
            Err(_) => {
                self.calls.truncate(calls_before);
                if let (Some(call), Some(len)) = (self.calls.last_mut(), last_sent) {
                    call.sent.truncate(len);
                }
                self.events.clear();
                self.pending.clear();
                self.mode = Mode::Done;
                tracing::error!(
                    target: "mlxcel::tool_calls",
                    streamed_calls = calls_before,
                    "incremental tool-call streamer panicked; streaming disabled for \
                     this request, the final parse emits the remaining calls"
                );
                Vec::new()
            }
        }
    }

    fn feed_inner(&mut self, fragment: &str) -> Vec<ToolCallStreamEvent> {
        self.pending.push_str(fragment);
        let input = std::mem::take(&mut self.pending);
        let mut pos = 0usize;
        while pos < input.len() {
            let mode = std::mem::replace(&mut self.mode, Mode::Done);
            let (mode, progress) = self.step(mode, &input[pos..]);
            self.mode = mode;
            match progress {
                Progress::Consumed(n) => pos += n,
                Progress::NeedMore => break,
            }
            if matches!(self.mode, Mode::Done) {
                pos = input.len();
            }
        }
        if pos < input.len() {
            self.pending.push_str(&input[pos..]);
        }
        std::mem::take(&mut self.events)
    }

    fn step(&mut self, mode: Mode, s: &str) -> (Mode, Progress) {
        match mode {
            Mode::Outer => self.step_outer(s),
            Mode::Thinking { closes } => {
                step_skip_to_any(s, &closes, Mode::Outer, |closes| Mode::Thinking { closes })
            }
            Mode::SkipUntil { marker } => {
                step_skip_to_any(s, &[marker], Mode::Outer, |_| Mode::SkipUntil { marker })
            }
            Mode::HermesBody => self.step_hermes_body(s),
            Mode::Json(state) => json_call::step_json(self, state, s),
            Mode::List(state) => json_call::step_list(self, state, s),
            Mode::Mistral(state) => mistral::step_mistral(self, state, s),
            Mode::Qwen(state) => qwen::step_qwen(self, state, s),
            Mode::Done => (Mode::Done, Progress::Consumed(s.len())),
        }
    }

    /// Visible text: find thinking blocks and tool-call openers.
    fn step_outer(&mut self, s: &str) -> (Mode, Progress) {
        let bytes = s.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            let b = bytes[i];
            if b.is_ascii_whitespace() {
                i += 1;
                continue;
            }
            let rest = &s[i..];
            let mut partial = false;
            for (open, close) in THINKING_MARKER_PAIRS {
                match prefix_state(rest, open) {
                    Prefix::Full => {
                        return (
                            Mode::Thinking {
                                closes: vec![close],
                            },
                            Progress::Consumed(i + open.len()),
                        );
                    }
                    Prefix::Partial => partial = true,
                    Prefix::No => {}
                }
            }
            // Only the family the stream committed to can add calls: the
            // final dispatcher keeps one format's calls.
            let open_to = |f: Family| self.family.is_none_or(|locked| locked == f);
            let mut openers: Vec<(&'static str, Mode)> = Vec::with_capacity(3);
            openers.push((TOOL_CALL_OPEN, Mode::HermesBody));
            if open_to(Family::Llama3) {
                openers.push((PYTHON_TAG, Mode::List(json_call::ListState::python_tag())));
            }
            if self.at_start {
                openers.push((TOOL_CALLS_MARKER, Mode::Mistral(Box::default())));
            }
            for (marker, next) in openers {
                match prefix_state(rest, marker) {
                    Prefix::Full => {
                        self.at_start = false;
                        return (next, Progress::Consumed(i + marker.len()));
                    }
                    Prefix::Partial => partial = true,
                    Prefix::No => {}
                }
            }
            if partial {
                return (Mode::Outer, consumed_or_more(i));
            }
            if self.at_start && (b == b'{' || b == b'[') {
                self.at_start = false;
                let mode = if b == b'{' {
                    Mode::Json(Box::new(JsonCallState::new(
                        JsonFamily::Llama3,
                        JsonContainer::Single,
                        true,
                    )))
                } else {
                    Mode::List(json_call::ListState::bare())
                };
                return (mode, Progress::Consumed(i));
            }
            self.at_start = false;
            // Step over the whole character: visible text may be non-ASCII,
            // and `&s[i..]` above needs a char boundary. Every marker starts
            // with an ASCII byte, so no marker can begin mid-character.
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
        (Mode::Outer, Progress::Consumed(bytes.len()))
    }

    /// After `<tool_call>`: a JSON body is Hermes, `<function=` is Qwen3-Coder,
    /// anything else (GLM-4.7 key/value grammar, ...) is not streamed.
    fn step_hermes_body(&mut self, s: &str) -> (Mode, Progress) {
        let lead = s.len() - s.trim_start().len();
        let rest = &s[lead..];
        if rest.is_empty() {
            return (Mode::HermesBody, consumed_or_more(lead));
        }
        let skip = Mode::SkipUntil {
            marker: TOOL_CALL_CLOSE,
        };
        if rest.starts_with('{') {
            if self.family.is_some_and(|f| f != Family::HermesJson) {
                return (skip, Progress::Consumed(lead));
            }
            let state = JsonCallState::new(JsonFamily::Hermes, JsonContainer::Hermes, false);
            return (Mode::Json(Box::new(state)), Progress::Consumed(lead));
        }
        match prefix_state(rest, qwen::FUNCTION_OPEN) {
            Prefix::Full if self.family.is_none_or(|f| f == Family::Qwen3Coder) => (
                Mode::Qwen(Box::default()),
                Progress::Consumed(lead + qwen::FUNCTION_OPEN.len()),
            ),
            Prefix::Partial => (Mode::HermesBody, consumed_or_more(lead)),
            _ => (skip, Progress::Consumed(lead)),
        }
    }

    /// Resolve a parsed name through the tool filter and assign the call its
    /// index. Unknown tools get no index (the final parse drops them).
    fn resolve_call(&mut self, ctx: &mut CallCtx, raw_name: &str, family: Family) {
        let names: HashSet<&str> = self.tool_names.iter().map(String::as_str).collect();
        let Some(name) = resolve_tool_name(raw_name, &names).map(str::to_string) else {
            ctx.suppressed = true;
            return;
        };
        self.family.get_or_insert(family);
        ctx.ordinal = Some(self.ordinals);
        self.ordinals += 1;
        if self
            .specific_function
            .as_deref()
            .is_some_and(|chosen| chosen != name)
        {
            ctx.suppressed = true;
        }
        ctx.name = Some(name);
    }

    /// Record a model-supplied id. It is used only when it arrives before the
    /// call starts and no earlier call carried it (the `try_mistral_nemo`
    /// uniqueness rule); an id that arrives after `Start` is ignored because
    /// the client already holds the minted one.
    fn note_model_id(&mut self, ctx: &mut CallCtx, id: &str) {
        if !super::formats::is_valid_model_call_id(id) {
            return;
        }
        let fresh = self.seen_model_ids.insert(id.to_string());
        if fresh && ctx.started.is_none() && ctx.model_id.is_none() {
            ctx.model_id = Some(id.to_string());
        }
    }

    /// Emit `Start` (and any held argument text) once the call is allowed to.
    fn try_start(&mut self, ctx: &mut CallCtx) {
        if ctx.started.is_some() || ctx.suppressed || !ctx.gate_open {
            return;
        }
        let (Some(index), Some(name)) = (ctx.ordinal, ctx.name.clone()) else {
            return;
        };
        let id = ctx
            .model_id
            .clone()
            .unwrap_or_else(super::generate_tool_call_id);
        self.events.push(ToolCallStreamEvent::Start {
            index,
            name: name.clone(),
            id: id.clone(),
        });
        self.calls.push(StreamedToolCall {
            index,
            name,
            id,
            sent: String::new(),
        });
        ctx.started = Some(self.calls.len() - 1);
        let held = std::mem::take(&mut ctx.held);
        if !ctx.bailed && !held.is_empty() {
            self.send(ctx, &held);
        }
    }

    /// Argument text for the current call: sent when started, held otherwise.
    fn emit_args(&mut self, ctx: &mut CallCtx, text: &str) {
        if text.is_empty() || ctx.bailed || ctx.suppressed {
            return;
        }
        if ctx.started.is_some() {
            self.send(ctx, text);
        } else if ctx.held.len() + text.len() > MAX_HELD_BYTES {
            ctx.held.clear();
            ctx.bailed = true;
        } else {
            ctx.held.push_str(text);
        }
    }

    fn send(&mut self, ctx: &CallCtx, text: &str) {
        let Some(pos) = ctx.started else {
            return;
        };
        // `contained` relies on this to roll back a panicked fragment.
        debug_assert_eq!(pos + 1, self.calls.len(), "only the newest call streams");
        let call = &mut self.calls[pos];
        call.sent.push_str(text);
        self.events.push(ToolCallStreamEvent::ArgsDelta {
            index: call.index,
            delta: text.to_string(),
        });
    }
}

/// `Consumed(at)`, or `NeedMore` when nothing before `at` can be consumed.
fn consumed_or_more(at: usize) -> Progress {
    if at == 0 {
        Progress::NeedMore
    } else {
        Progress::Consumed(at)
    }
}

/// How `s` relates to `marker`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Prefix {
    /// `s` starts with `marker`.
    Full,
    /// `s` is a proper prefix of `marker`: more input decides.
    Partial,
    No,
}

fn prefix_state(s: &str, marker: &str) -> Prefix {
    if s.starts_with(marker) {
        Prefix::Full
    } else if s.len() < marker.len() && marker.starts_with(s) {
        Prefix::Partial
    } else {
        Prefix::No
    }
}

/// Skip to the earliest of `markers` and switch to `found`; otherwise consume
/// everything except a tail that could still start a marker.
fn step_skip_to_any(
    s: &str,
    markers: &[&'static str],
    found: Mode,
    stay: impl FnOnce(Vec<&'static str>) -> Mode,
) -> (Mode, Progress) {
    let earliest = markers
        .iter()
        .filter_map(|m| s.find(m).map(|at| at + m.len()))
        .min();
    if let Some(end) = earliest {
        return (found, Progress::Consumed(end));
    }
    let longest = markers.iter().map(|m| m.len()).max().unwrap_or(1);
    let mut keep_from = s.len().saturating_sub(longest - 1);
    while !s.is_char_boundary(keep_from) {
        keep_from -= 1;
    }
    let keep_from = (keep_from..s.len())
        .filter(|&k| s.is_char_boundary(k))
        .find(|&k| markers.iter().any(|m| m.starts_with(&s[k..])))
        .unwrap_or(s.len());
    (stay(markers.to_vec()), consumed_or_more(keep_from))
}

#[cfg(test)]
#[path = "incremental_tests.rs"]
mod incremental_tests;
