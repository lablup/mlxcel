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

//! The Mistral bracket form `[TOOL_CALLS]NAME[ARGS]{json}` for the
//! incremental streamer. The header is read with the final parser's own regex
//! (`formats::MISTRAL_HEADER_PATTERN`); `[TOOL_CALLS] [` hands over to the
//! JSON list reader.
//!
//! Used by: tool_calls::incremental

use std::sync::LazyLock;

use fancy_regex::Regex;

use super::super::formats::{MISTRAL_HEADER_PATTERN, MISTRAL_MAX_CALLS};
use super::super::json_scan::{ValueKind, ValuePush, ValueScanner};
use super::json_call::ListState;
use super::{
    CallCtx, Family, IncrementalToolCallStreamer, MAX_HELD_BYTES, MAX_NAME_BYTES, Mode, Progress,
};

static MISTRAL_HEADER: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(MISTRAL_HEADER_PATTERN).ok());

#[derive(Debug, Default)]
enum MistralPhase {
    /// Right after `[TOOL_CALLS]`: a `[` opens the list form.
    #[default]
    Lead,
    /// Waiting for a complete `NAME[ARGS]` header.
    Header,
    /// Before the value's first byte.
    ValueLead,
    /// Inside the value.
    Value(ValueScanner, ValueSink),
}

#[derive(Debug)]
enum ValueSink {
    Stream,
    Capture(String, bool),
    Skip,
}

/// The bracket form after `[TOOL_CALLS]`.
#[derive(Debug, Default)]
pub(super) struct MistralState {
    phase: MistralPhase,
    ctx: CallCtx,
    calls_seen: usize,
}

pub(super) fn step_mistral(
    st: &mut IncrementalToolCallStreamer,
    mut state: Box<MistralState>,
    s: &str,
) -> (Mode, Progress) {
    match std::mem::take(&mut state.phase) {
        MistralPhase::Lead => {
            let lead = s.len() - s.trim_start().len();
            match s.as_bytes().get(lead) {
                None => (Mode::Mistral(state), Progress::Consumed(lead)),
                Some(b'[') => (Mode::List(ListState::mistral()), Progress::Consumed(lead)),
                Some(_) => {
                    state.phase = MistralPhase::Header;
                    (Mode::Mistral(state), Progress::Consumed(0))
                }
            }
        }
        MistralPhase::Header => step_header(st, state, s),
        MistralPhase::ValueLead => {
            let lead = s.len() - s.trim_start().len();
            let Some(&b) = s.as_bytes().get(lead) else {
                state.phase = MistralPhase::ValueLead;
                return (Mode::Mistral(state), Progress::Consumed(lead));
            };
            let scanner = ValueScanner::start(b);
            let sink = match scanner.kind() {
                ValueKind::Object => {
                    st.emit_args(&mut state.ctx, "{");
                    ValueSink::Stream
                }
                ValueKind::String => ValueSink::Capture("\"".to_string(), false),
                ValueKind::Array | ValueKind::Scalar => {
                    state.ctx.bailed = true;
                    ValueSink::Skip
                }
            };
            state.phase = MistralPhase::Value(scanner, sink);
            // A scalar may start with a multi-byte character: consume all of
            // it so the next step starts on a boundary. Its continuation
            // bytes cannot end a scalar, so the scanner need not see them.
            let first_len = s[lead..].chars().next().map_or(1, char::len_utf8);
            (Mode::Mistral(state), Progress::Consumed(lead + first_len))
        }
        MistralPhase::Value(mut scanner, mut sink) => {
            let bytes = s.as_bytes();
            let mut end = None;
            for (i, &b) in bytes.iter().enumerate() {
                match scanner.push(b) {
                    ValuePush::Continue => {}
                    ValuePush::EndIncl => {
                        end = Some(i + 1);
                        break;
                    }
                    ValuePush::EndBefore => {
                        end = Some(i);
                        break;
                    }
                }
            }
            let upto = end.unwrap_or(s.len());
            match &mut sink {
                ValueSink::Stream => st.emit_args(&mut state.ctx, &s[..upto]),
                ValueSink::Capture(buf, overflow) => {
                    if buf.len() + upto > MAX_HELD_BYTES {
                        *overflow = true;
                        buf.clear();
                    } else if !*overflow {
                        buf.push_str(&s[..upto]);
                    }
                }
                ValueSink::Skip => {}
            }
            if end.is_none() {
                state.phase = MistralPhase::Value(scanner, sink);
                return (Mode::Mistral(state), Progress::Consumed(s.len()));
            }
            if let ValueSink::Capture(buf, overflow) = sink {
                match (!overflow).then(|| serde_json::from_str::<String>(&buf).ok()) {
                    Some(Some(text)) => st.emit_args(&mut state.ctx, &text),
                    _ => state.ctx.bailed = true,
                }
            }
            if state.calls_seen >= MISTRAL_MAX_CALLS {
                return (Mode::Done, Progress::Consumed(upto));
            }
            state.ctx = CallCtx::default();
            state.phase = MistralPhase::Header;
            (Mode::Mistral(state), Progress::Consumed(upto))
        }
    }
}

/// Read one `NAME[ARGS]` header with the final parser's own regex: the match
/// must follow only whitespace and end at `[ARGS]`.
fn step_header(
    st: &mut IncrementalToolCallStreamer,
    mut state: Box<MistralState>,
    s: &str,
) -> (Mode, Progress) {
    const ARGS: &str = "[ARGS]";
    let Some(at) = s.find(ARGS) else {
        if s.len() > MAX_NAME_BYTES {
            return (Mode::Done, Progress::Consumed(s.len()));
        }
        state.phase = MistralPhase::Header;
        return (Mode::Mistral(state), Progress::NeedMore);
    };
    let header = &s[..at + ARGS.len()];
    let name = MISTRAL_HEADER.as_ref().and_then(|re| {
        let caps = re.captures(header).ok()??;
        let whole = caps.get(0)?;
        let ok = header[..whole.start()].trim().is_empty() && whole.end() == header.len();
        ok.then(|| caps.get(1).map(|m| m.as_str().to_string()))?
    });
    let Some(name) = name else {
        return (Mode::Done, Progress::Consumed(s.len()));
    };
    state.calls_seen += 1;
    state.ctx = CallCtx {
        gate_open: true,
        ..CallCtx::default()
    };
    st.resolve_call(&mut state.ctx, &name, Family::Mistral);
    st.try_start(&mut state.ctx);
    state.phase = MistralPhase::ValueLead;
    (Mode::Mistral(state), Progress::Consumed(header.len()))
}
