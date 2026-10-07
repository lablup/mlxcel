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

//! Reasoning-channel helpers shared by the CLI display and the server's chat
//! routes, over the server's
//! [`StreamFilter`](crate::server::tool_calls::stream_filter::StreamFilter).
//!
//! Until epic #2166 Phase 5 (issue #2173) the CLI split the reasoning channel
//! with a splitter of its own (`src/reasoning_stream.rs`). The chat REPL and `mlxcel run` now receive
//! the split from the server, and the one-shot `mlxcel generate` display runs
//! its decoded reply through the same `StreamFilter` ([`render_full`]), so
//! one splitter decides what is reasoning everywhere.

use crate::server::tool_calls::stream_filter::{FilterOutput, StreamFilter};
use crate::tokenizer::ThinkingMarkers;

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Whether a rendered prompt primed an open reasoning channel that the model
/// is expected to close in its generation: the tokenizer exposes a marker
/// pair and `prompt` ends with the open marker, ignoring trailing ASCII
/// whitespace (Qwen-style `<think>\n`, a thinking-on Gemma 4
/// `<|channel>thought\n`).
#[must_use]
pub fn prompt_primed_open_thinking(markers: &ThinkingMarkers, prompt: &str) -> bool {
    match (markers.think_start.as_deref(), markers.think_end.as_deref()) {
        (Some(start), Some(end)) if !start.is_empty() && !end.is_empty() => prompt
            .trim_end_matches([' ', '\t', '\r', '\n'])
            .ends_with(start),
        _ => false,
    }
}

/// The close marker of the thinking block `prompt` primed open, or `None`
/// when it primed none. Shares [`prompt_primed_open_thinking`]'s rule so the
/// two never disagree (issue #1554).
#[must_use]
pub fn prompt_primed_open_close_marker(markers: &ThinkingMarkers, prompt: &str) -> Option<String> {
    if !prompt_primed_open_thinking(markers, prompt) {
        return None;
    }
    markers.think_end.clone()
}

/// Whether a generation produced text but none of it reached the visible
/// output, because every token went to the hidden reasoning channel.
#[must_use]
pub fn is_reasoning_only(
    generated_text: &str,
    saw_visible_text: bool,
    show_reasoning: bool,
) -> bool {
    !show_reasoning && !saw_visible_text && !generated_text.trim().is_empty()
}

fn render_output(out: &FilterOutput, show_reasoning: bool, dim: bool, into: &mut String) {
    if show_reasoning
        && let Some(reasoning) = out.reasoning.as_deref()
        && !reasoning.is_empty()
    {
        if dim {
            into.push_str(DIM);
            into.push_str(reasoning);
            into.push_str(RESET);
        } else {
            into.push_str(reasoning);
        }
    }
    if let Some(content) = out.content.as_deref() {
        into.push_str(content);
    }
}

/// Split a complete decoded generation through `StreamFilter` and return the
/// terminal-visible text: content always, reasoning (dimmed when `dim`) only
/// with `show_reasoning`. `primed_open` starts the filter inside the
/// reasoning channel, for a prompt that primed the open marker.
#[must_use]
pub fn render_full(text: &str, primed_open: bool, show_reasoning: bool, dim: bool) -> String {
    let mut filter = if primed_open {
        StreamFilter::new_primed_open_thinking()
    } else {
        StreamFilter::new()
    };
    let mut out = String::new();
    render_output(&filter.feed(text), show_reasoning, dim, &mut out);
    render_output(&filter.flush(), show_reasoning, dim, &mut out);
    out
}

#[cfg(test)]
#[path = "reasoning_display_tests.rs"]
mod tests;
