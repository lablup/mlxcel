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

//! Qwen3-Coder XML calls for the incremental streamer.
//!
//! The arguments stream as the compact JSON object `extract_qwen_parameters`
//! serializes: keys in generation order, `{` sent when the call starts, `}`
//! when `</function>` closes it. A parameter the request schema types as a
//! plain string streams as JSON-escaped fragments: leading whitespace is
//! dropped and trailing whitespace is held until more text follows, which is
//! the final parser's `trim()`. Any other value is held until it closes and
//! sent after `coerce_xml_param`. Names are read with `match_qwen_name` once
//! enough text has arrived for its answer to be final.
//!
//! Used by: tool_calls::incremental

use std::collections::HashSet;

use super::super::formats::{
    QWEN3_CODER_MAX_CALLS, QWEN3_CODER_MAX_PARAMS_PER_CALL, coerce_xml_param, kv_param_schema,
    match_qwen_name, minimax_m3_function_schema, xml_param_keeps_raw_string,
};
use super::{
    CallCtx, Family, IncrementalToolCallStreamer, MAX_HELD_BYTES, MAX_NAME_BYTES, Mode, Prefix,
    Progress, consumed_or_more, prefix_state,
};

pub(super) const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

#[derive(Debug, Default)]
enum Phase {
    /// After `<function=`.
    #[default]
    FunctionName,
    /// After the name: `{` here is a Functionary v3.1 body, not Qwen XML.
    BodyPeek,
    /// Between parameters.
    Body,
    /// After `<parameter=`.
    ParamName,
    /// Inside a string-typed value.
    StrValue { started: bool, held_ws: String },
    /// Inside any other value.
    RawValue {
        key: String,
        schema: Option<serde_json::Value>,
        buf: String,
        overflow: bool,
    },
    /// Past the parameter cap: only `</function>` matters.
    SkipToClose,
}

/// One `<function=..>` call being read.
#[derive(Debug, Default)]
pub(super) struct QwenState {
    phase: Phase,
    ctx: CallCtx,
    fn_schema: Option<serde_json::Value>,
    keys: HashSet<String>,
    params_seen: usize,
    params_emitted: usize,
}

pub(super) fn step_qwen(
    st: &mut IncrementalToolCallStreamer,
    mut state: Box<QwenState>,
    s: &str,
) -> (Mode, Progress) {
    match std::mem::take(&mut state.phase) {
        Phase::FunctionName => {
            let Some(decided) = decide_name(s) else {
                return undecided(state, Phase::FunctionName, s);
            };
            let Some((name, consumed)) = decided else {
                // `try_qwen3_coder` stops at an opener with no name.
                return (Mode::Done, Progress::Consumed(s.len()));
            };
            st.qwen_calls += 1;
            state.fn_schema = minimax_m3_function_schema(Some(&st.tools), &name).cloned();
            st.resolve_call(&mut state.ctx, &name, Family::Qwen3Coder);
            state.phase = Phase::BodyPeek;
            (Mode::Qwen(state), Progress::Consumed(consumed))
        }
        Phase::BodyPeek => {
            let lead = s.len() - s.trim_start().len();
            match s.as_bytes().get(lead) {
                None => {
                    state.phase = Phase::BodyPeek;
                    (Mode::Qwen(state), Progress::Consumed(lead))
                }
                // Functionary v3.1 (`<function=f>{json}`) claims this call
                // ahead of Qwen3-Coder in the final dispatcher.
                Some(b'{') => (Mode::Done, Progress::Consumed(s.len())),
                Some(_) => {
                    state.ctx.gate_open = true;
                    st.try_start(&mut state.ctx);
                    st.emit_args(&mut state.ctx, "{");
                    state.phase = Phase::Body;
                    (Mode::Qwen(state), Progress::Consumed(lead))
                }
            }
        }
        Phase::Body => step_body(st, state, s),
        Phase::ParamName => {
            let Some(decided) = decide_name(s) else {
                return undecided(state, Phase::ParamName, s);
            };
            let Some((key, consumed)) = decided else {
                // An unnamed opener is skipped; scanning resumes after it.
                state.phase = Phase::Body;
                return (Mode::Qwen(state), Progress::Consumed(0));
            };
            state.params_seen += 1;
            if !state.keys.insert(key.clone()) {
                // A repeated key replaces the earlier value in place.
                state.ctx.bailed = true;
            }
            let schema = kv_param_schema(state.fn_schema.as_ref(), &key).cloned();
            state.phase = if xml_param_keeps_raw_string(schema.as_ref()) {
                let open = format!("{}{}:\"", separator(&mut state), json_string(&key));
                st.emit_args(&mut state.ctx, &open);
                Phase::StrValue {
                    started: false,
                    held_ws: String::new(),
                }
            } else {
                Phase::RawValue {
                    key,
                    schema,
                    buf: String::new(),
                    overflow: false,
                }
            };
            (Mode::Qwen(state), Progress::Consumed(consumed))
        }
        phase @ (Phase::StrValue { .. } | Phase::RawValue { .. }) => {
            step_value(st, state, phase, s)
        }
        Phase::SkipToClose => match s.find(FUNCTION_CLOSE) {
            Some(at) => finish_call(st, state, at + FUNCTION_CLOSE.len()),
            None => {
                let keep = partial_tail(s, &[FUNCTION_CLOSE]);
                state.phase = Phase::SkipToClose;
                (Mode::Qwen(state), consumed_or_more(keep))
            }
        },
    }
}

/// `match_qwen_name` on `s` once its answer can no longer change: `None`
/// while more text could still extend or terminate the name, otherwise the
/// final parser's own result.
fn decide_name(s: &str) -> Option<Option<(String, usize)>> {
    let rest = s.trim_start();
    if rest.is_empty() {
        return None;
    }
    let decidable = if rest.starts_with('"') || rest.starts_with('\'') {
        rest.contains(['<', '>'])
    } else {
        rest.contains(|c: char| c.is_whitespace() || c == '<' || c == '>')
    };
    decidable.then(|| match_qwen_name(s))
}

fn undecided(mut state: Box<QwenState>, phase: Phase, s: &str) -> (Mode, Progress) {
    if s.len() > MAX_NAME_BYTES {
        return (Mode::Done, Progress::Consumed(s.len()));
    }
    state.phase = phase;
    (Mode::Qwen(state), Progress::NeedMore)
}

fn step_body(
    st: &mut IncrementalToolCallStreamer,
    mut state: Box<QwenState>,
    s: &str,
) -> (Mode, Progress) {
    let mut from = 0usize;
    while let Some(off) = s[from..].find('<') {
        let at = from + off;
        let rest = &s[at..];
        if prefix_state(rest, FUNCTION_CLOSE) == Prefix::Full {
            return finish_call(st, state, at + FUNCTION_CLOSE.len());
        }
        if prefix_state(rest, PARAM_OPEN) == Prefix::Full {
            state.phase = if state.params_seen >= QWEN3_CODER_MAX_PARAMS_PER_CALL {
                Phase::SkipToClose
            } else {
                Phase::ParamName
            };
            return (Mode::Qwen(state), Progress::Consumed(at + PARAM_OPEN.len()));
        }
        if [FUNCTION_CLOSE, PARAM_OPEN]
            .iter()
            .any(|m| prefix_state(rest, m) == Prefix::Partial)
        {
            state.phase = Phase::Body;
            return (Mode::Qwen(state), consumed_or_more(at));
        }
        from = at + 1;
    }
    state.phase = Phase::Body;
    (Mode::Qwen(state), Progress::Consumed(s.len()))
}

/// Read value text up to `</parameter>` or `</function>`.
fn step_value(
    st: &mut IncrementalToolCallStreamer,
    mut state: Box<QwenState>,
    mut phase: Phase,
    s: &str,
) -> (Mode, Progress) {
    let mut from = 0usize;
    let mut text_end = s.len();
    let mut close: Option<(&str, usize)> = None;
    let mut hold_from = None;
    while let Some(off) = s[from..].find('<') {
        let at = from + off;
        let rest = &s[at..];
        if let Some(marker) = [PARAM_CLOSE, FUNCTION_CLOSE]
            .into_iter()
            .find(|m| prefix_state(rest, m) == Prefix::Full)
        {
            text_end = at;
            close = Some((marker, at + marker.len()));
            break;
        }
        if [PARAM_CLOSE, FUNCTION_CLOSE]
            .iter()
            .any(|m| prefix_state(rest, m) == Prefix::Partial)
        {
            text_end = at;
            hold_from = Some(at);
            break;
        }
        from = at + 1;
    }
    absorb_value_text(st, &mut state, &mut phase, &s[..text_end]);
    let Some((marker, end)) = close else {
        state.phase = phase;
        let progress = match hold_from {
            Some(at) => consumed_or_more(at),
            None => Progress::Consumed(s.len()),
        };
        return (Mode::Qwen(state), progress);
    };
    close_value(st, &mut state, phase);
    if marker == FUNCTION_CLOSE {
        return finish_call(st, state, end);
    }
    state.phase = Phase::Body;
    (Mode::Qwen(state), Progress::Consumed(end))
}

fn absorb_value_text(
    st: &mut IncrementalToolCallStreamer,
    state: &mut QwenState,
    phase: &mut Phase,
    text: &str,
) {
    match phase {
        Phase::StrValue {
            started, held_ws, ..
        } => {
            let mut out = String::new();
            for ch in text.chars() {
                if !*started {
                    if ch.is_whitespace() {
                        continue;
                    }
                    *started = true;
                }
                if ch.is_whitespace() {
                    held_ws.push(ch);
                } else {
                    out.push_str(held_ws);
                    held_ws.clear();
                    out.push(ch);
                }
            }
            if held_ws.len() > MAX_HELD_BYTES {
                held_ws.clear();
                state.ctx.bailed = true;
            }
            if !out.is_empty() {
                st.emit_args(&mut state.ctx, &json_string_body(&out));
            }
        }
        Phase::RawValue { buf, overflow, .. } => {
            if buf.len() + text.len() > MAX_HELD_BYTES {
                buf.clear();
                *overflow = true;
                state.ctx.bailed = true;
            } else if !*overflow {
                buf.push_str(text);
            }
        }
        _ => {}
    }
}

fn close_value(st: &mut IncrementalToolCallStreamer, state: &mut QwenState, phase: Phase) {
    match phase {
        Phase::StrValue { .. } => st.emit_args(&mut state.ctx, "\""),
        Phase::RawValue {
            key,
            schema,
            buf,
            overflow,
        } => {
            if overflow {
                return;
            }
            let value = coerce_xml_param(buf.trim(), schema.as_ref());
            let value = serde_json::to_string(&value).unwrap_or_else(|_| "null".to_string());
            let member = format!("{}{}:{value}", separator(state), json_string(&key));
            st.emit_args(&mut state.ctx, &member);
        }
        _ => {}
    }
}

fn finish_call(
    st: &mut IncrementalToolCallStreamer,
    mut state: Box<QwenState>,
    consumed: usize,
) -> (Mode, Progress) {
    st.emit_args(&mut state.ctx, "}");
    let next = if st.qwen_calls >= QWEN3_CODER_MAX_CALLS {
        Mode::Done
    } else {
        Mode::Outer
    };
    (next, Progress::Consumed(consumed))
}

/// `,` before every member but the first; counts the member.
fn separator(state: &mut QwenState) -> &'static str {
    let first = state.params_emitted == 0;
    state.params_emitted += 1;
    if first { "" } else { "," }
}

fn json_string(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string())
}

/// `text` JSON-escaped, without the surrounding quotes. Escaping is per
/// character, so fragments escaped one by one concatenate to the escape of
/// the whole value.
fn json_string_body(text: &str) -> String {
    let quoted = json_string(text);
    quoted[1..quoted.len() - 1].to_string()
}

/// Where the tail of `s` that could still start one of `markers` begins.
fn partial_tail(s: &str, markers: &[&str]) -> usize {
    (s.len().saturating_sub(16)..s.len())
        .filter(|&k| s.is_char_boundary(k))
        .find(|&k| markers.iter().any(|m| m.starts_with(&s[k..])))
        .unwrap_or(s.len())
}
