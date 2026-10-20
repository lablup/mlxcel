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

//! JSON-bodied calls for the incremental streamer: Hermes, Llama 3 (single
//! object or list) and the Mistral `[TOOL_CALLS] [{..}]` list.
//!
//! An object-valued `arguments` member streams as the raw bytes of the member,
//! which is exactly the text the final parsers keep (`formats::arguments_text`
//! slices the same member with the same scanner). A string-valued one is held
//! until it closes and sent decoded, as the final parsers unwrap it. Any other
//! value is not streamed; the final emission sends it.
//!
//! Used by: tool_calls::incremental

use super::super::json_scan::{ObjectStep, ObjectWalker, ValueKind};
use super::{
    CallCtx, Family, IncrementalToolCallStreamer, MAX_HELD_BYTES, MAX_NAME_BYTES, Mode, Progress,
    TOOL_CALL_CLOSE,
};

/// Longest model-supplied id held; valid ids are at most 64 bytes.
const MAX_ID_BYTES: usize = 128;

/// Which keys a call object uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum JsonFamily {
    /// `{"name", "arguments"}` inside `<tool_call>`.
    Hermes,
    /// `{"name", "parameters" | "arguments"}`, `parameters` preferred.
    Llama3,
    /// `{"name", "arguments", "id"}` inside `[TOOL_CALLS] [..]`.
    MistralList,
}

impl JsonFamily {
    fn family(self) -> Family {
        match self {
            Self::Hermes => Family::HermesJson,
            Self::Llama3 => Family::Llama3,
            Self::MistralList => Family::Mistral,
        }
    }
}

/// Where control returns once the call object closes.
#[derive(Debug)]
pub(super) enum JsonContainer {
    /// Inside `<tool_call>`: skip to `</tool_call>`.
    Hermes,
    /// The only call (bare Llama 3 JSON or `<|python_tag|>{..}`).
    Single,
    /// An element of a call list.
    ListElement(ListState),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Member {
    Other,
    Name,
    Args,
    Id,
}

/// What happens to the bytes of the member value being read.
#[derive(Debug)]
enum Sink {
    None,
    /// Held (name, id, string arguments); `start` is the offset in the
    /// current slice, `carry` the text from earlier slices.
    Capture {
        carry: String,
        start: Option<usize>,
        limit: usize,
        overflow: bool,
    },
    /// Object arguments, sent as they arrive; `start` as above.
    Stream {
        start: Option<usize>,
    },
}

/// One call object being read.
#[derive(Debug)]
pub(super) struct JsonCallState {
    walker: ObjectWalker,
    family: JsonFamily,
    container: JsonContainer,
    ctx: CallCtx,
    member: Member,
    sink: Sink,
    args_key: Option<&'static str>,
    name_seen: bool,
    /// Bare JSON: `Start` waits for the arguments member, so a JSON answer
    /// that merely has a `name` key is never announced as a call.
    gate_on_args: bool,
}

impl JsonCallState {
    pub(super) fn new(family: JsonFamily, container: JsonContainer, gate_on_args: bool) -> Self {
        Self {
            walker: ObjectWalker::new(),
            family,
            container,
            ctx: CallCtx {
                gate_open: !gate_on_args,
                ..CallCtx::default()
            },
            member: Member::Other,
            sink: Sink::None,
            args_key: None,
            name_seen: false,
            gate_on_args,
        }
    }

    /// Classify a member key, mirroring which value the final parser reads.
    fn classify(&mut self, key: Option<String>) -> Member {
        let Some(key) = key else {
            return Member::Other;
        };
        match (self.family, key.as_str()) {
            (_, "name") => {
                if self.name_seen {
                    // A repeated `name`: the final parse reads the last one.
                    self.ctx.bailed = true;
                    return Member::Other;
                }
                Member::Name
            }
            (JsonFamily::Llama3, "parameters") => match self.args_key {
                None => {
                    self.args_key = Some("parameters");
                    Member::Args
                }
                Some(_) => {
                    // `parameters` wins over an `arguments` already streamed,
                    // and a repeat replaces the first value.
                    self.ctx.bailed = true;
                    Member::Other
                }
            },
            (_, "arguments") => match self.args_key {
                None => {
                    self.args_key = Some("arguments");
                    Member::Args
                }
                Some("arguments") => {
                    self.ctx.bailed = true;
                    Member::Other
                }
                Some(_) => Member::Other,
            },
            (JsonFamily::MistralList, "id") => Member::Id,
            _ => Member::Other,
        }
    }

    fn capture(limit: usize, start: usize) -> Sink {
        Sink::Capture {
            carry: String::new(),
            start: Some(start),
            limit,
            overflow: false,
        }
    }
}

/// Read a call object; returns the next mode once it closes.
pub(super) fn step_json(
    st: &mut IncrementalToolCallStreamer,
    mut state: Box<JsonCallState>,
    s: &str,
) -> (Mode, Progress) {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match state.walker.push(bytes[i]) {
            ObjectStep::Skip | ObjectStep::ValueByte => {}
            ObjectStep::Key => {
                let key = state.walker.key();
                state.member = state.classify(key);
            }
            ObjectStep::ValueStart(kind) => value_start(st, &mut state, kind, i),
            ObjectStep::ValueEnd => value_end(st, &mut state, s, i + 1),
            ObjectStep::ValueEndBefore => {
                value_end(st, &mut state, s, i);
                // The byte was not consumed; push it again.
                continue;
            }
            ObjectStep::ObjectEnd => {
                let next = after_object(*state);
                return (next, Progress::Consumed(i + 1));
            }
            ObjectStep::Invalid => {
                let next = match state.container {
                    JsonContainer::Hermes => Mode::SkipUntil {
                        marker: TOOL_CALL_CLOSE,
                    },
                    _ => Mode::Done,
                };
                return (next, Progress::Consumed(i + 1));
            }
        }
        i += 1;
    }
    // Carry the open value over to the next fragment.
    match &mut state.sink {
        Sink::Stream { start } => {
            if let Some(from) = start.take() {
                st.emit_args(&mut state.ctx, &s[from..]);
            }
            *start = Some(0);
        }
        Sink::Capture {
            carry,
            start,
            limit,
            overflow,
        } => {
            let from = start.take().unwrap_or(0);
            if carry.len() + (s.len() - from) > *limit {
                *overflow = true;
                carry.clear();
            } else if !*overflow {
                carry.push_str(&s[from..]);
            }
            *start = Some(0);
        }
        Sink::None => {}
    }
    (Mode::Json(state), Progress::Consumed(s.len()))
}

fn value_start(
    st: &mut IncrementalToolCallStreamer,
    state: &mut JsonCallState,
    kind: ValueKind,
    at: usize,
) {
    state.sink = match (state.member, kind) {
        (Member::Name, ValueKind::String) => JsonCallState::capture(MAX_NAME_BYTES * 2, at),
        (Member::Id, ValueKind::String) => JsonCallState::capture(MAX_ID_BYTES, at),
        (Member::Args, _) => {
            if state.gate_on_args {
                state.ctx.gate_open = true;
                st.try_start(&mut state.ctx);
            }
            match kind {
                ValueKind::Object => Sink::Stream { start: Some(at) },
                ValueKind::String => JsonCallState::capture(MAX_HELD_BYTES, at),
                // An array or scalar is re-serialized by the final parser.
                ValueKind::Array | ValueKind::Scalar => {
                    state.ctx.bailed = true;
                    Sink::None
                }
            }
        }
        (Member::Name, _) => {
            // A non-string name: the final parser drops the call.
            state.name_seen = true;
            state.ctx.suppressed = true;
            Sink::None
        }
        _ => Sink::None,
    };
}

/// The member value ended just before byte `end` of `s`.
fn value_end(st: &mut IncrementalToolCallStreamer, state: &mut JsonCallState, s: &str, end: usize) {
    let sink = std::mem::replace(&mut state.sink, Sink::None);
    match sink {
        Sink::None => {}
        Sink::Stream { start } => {
            let from = start.unwrap_or(0);
            st.emit_args(&mut state.ctx, &s[from..end]);
        }
        Sink::Capture {
            mut carry,
            start,
            limit,
            overflow,
        } => {
            let from = start.unwrap_or(0);
            let overflow = overflow || carry.len() + (end - from) > limit;
            if !overflow {
                carry.push_str(&s[from..end]);
            }
            let decoded = if overflow {
                None
            } else {
                serde_json::from_str::<String>(&carry).ok()
            };
            finish_capture(st, state, decoded);
        }
    }
    state.member = Member::Other;
}

fn finish_capture(
    st: &mut IncrementalToolCallStreamer,
    state: &mut JsonCallState,
    decoded: Option<String>,
) {
    match state.member {
        Member::Name => {
            state.name_seen = true;
            match decoded {
                Some(name) => {
                    st.resolve_call(&mut state.ctx, &name, state.family.family());
                    st.try_start(&mut state.ctx);
                }
                None => state.ctx.suppressed = true,
            }
        }
        Member::Id => {
            if let Some(id) = decoded {
                st.note_model_id(&mut state.ctx, &id);
            }
        }
        Member::Args => match decoded {
            Some(text) => st.emit_args(&mut state.ctx, &text),
            None => state.ctx.bailed = true,
        },
        Member::Other => {}
    }
}

fn after_object(state: JsonCallState) -> Mode {
    match state.container {
        JsonContainer::Hermes => Mode::SkipUntil {
            marker: TOOL_CALL_CLOSE,
        },
        JsonContainer::Single => Mode::Done,
        JsonContainer::ListElement(mut list) => {
            list.phase = ListPhase::After;
            Mode::List(list)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ListPhase {
    /// After `<|python_tag|>`: one object, or a list.
    Lead,
    /// Expect the opening `[`.
    Open,
    /// Expect an element or `]`.
    Element,
    /// Expect `,` or `]`.
    After,
}

/// A list of call objects (or the value after `<|python_tag|>`).
#[derive(Debug)]
pub(super) struct ListState {
    family: JsonFamily,
    gate_on_args: bool,
    phase: ListPhase,
}

impl ListState {
    pub(super) fn python_tag() -> Self {
        Self {
            family: JsonFamily::Llama3,
            gate_on_args: false,
            phase: ListPhase::Lead,
        }
    }

    pub(super) fn bare() -> Self {
        Self {
            family: JsonFamily::Llama3,
            gate_on_args: true,
            phase: ListPhase::Open,
        }
    }

    pub(super) fn mistral() -> Self {
        Self {
            family: JsonFamily::MistralList,
            gate_on_args: false,
            phase: ListPhase::Open,
        }
    }
}

pub(super) fn step_list(
    _st: &mut IncrementalToolCallStreamer,
    mut list: ListState,
    s: &str,
) -> (Mode, Progress) {
    let lead = s.len() - s.trim_start().len();
    let Some(&b) = s.as_bytes().get(lead) else {
        return (Mode::List(list), Progress::Consumed(lead));
    };
    match (list.phase, b) {
        (ListPhase::Lead, b'{') => {
            let state = JsonCallState::new(list.family, JsonContainer::Single, list.gate_on_args);
            (Mode::Json(Box::new(state)), Progress::Consumed(lead))
        }
        (ListPhase::Lead | ListPhase::Open, b'[') | (ListPhase::After, b',') => {
            list.phase = ListPhase::Element;
            (Mode::List(list), Progress::Consumed(lead + 1))
        }
        (ListPhase::Element, b'{') => {
            let state = JsonCallState::new(
                list.family,
                JsonContainer::ListElement(ListState {
                    family: list.family,
                    gate_on_args: list.gate_on_args,
                    phase: ListPhase::After,
                }),
                list.gate_on_args,
            );
            (Mode::Json(Box::new(state)), Progress::Consumed(lead))
        }
        _ => (Mode::Done, Progress::Consumed(s.len())),
    }
}
