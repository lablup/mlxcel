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

//! Streaming reasoning-channel splitter for the CLI output path.
//!
//! Autoregressive checkpoints with a reasoning channel (Gemma 4's
//! `<|channel>thought` / `<channel|>` pair, or the Qwen-style `<think>` /
//! `</think>` pair) emit their chain-of-thought inline with the answer. The
//! CLI detokenizer decodes with `skip_special_tokens = false`, so those raw
//! markers and the whole thought body would otherwise print to the terminal
//! as if they were the answer (issue #884).
//!
//! This filter is the CLI counterpart to the server's
//! [`crate::server::tool_calls::stream_filter::StreamFilter`]. It is
//! deliberately narrower: it splits **only** the reasoning channel and leaves
//! every other byte untouched, so an answer that opens no channel is passed
//! through unchanged (no tool-call suppression, no turn-marker stripping). The
//! open/close markers come from the tokenizer's
//! [`ThinkingMarkers`](crate::tokenizer::ThinkingMarkers), so the full
//! `<|channel>thought` open marker is consumed as a unit and the `thought`
//! channel argument never surfaces in the reasoning text.
//!
//! Both CLI entry points share this one implementation: the interactive REPL
//! feeds it decoded fragments token-by-token, and the one-shot `generate` flow
//! feeds it the whole decoded generation at once.
//!
//! ## State machine
//!
//! Two states, `Content` and `Thinking`. In `Content` the filter scans for the
//! open marker; text before it is content, and the marker flips the state to
//! `Thinking`. In `Thinking` it scans for the close marker; text before it is
//! reasoning, and the marker flips back to `Content`. A marker can straddle two
//! decoded fragments, so a trailing byte run that is a proper prefix of the
//! marker it is currently looking for is held back until the next fragment
//! resolves it (mirroring the server filter's `safe_emit_length`). At
//! end-of-stream [`flush`](ReasoningFilter::flush) releases whatever is still
//! buffered under the current state, so a thought block the model never closed
//! stays in `reasoning` (hidden by default) and no text is lost or duplicated.

use crate::tokenizer::ThinkingMarkers;

/// Visible-output split produced by [`ReasoningFilter::feed`] / `flush`.
///
/// `content` is text outside the reasoning channel and is always printed.
/// `reasoning` is text inside the channel; the CLI prints it only when the
/// user passes `--show-reasoning`. Neither field ever contains an open/close
/// marker or the `<|channel>thought` channel argument. An empty string means
/// "nothing for this side in this call".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningSplit {
    pub content: String,
    pub reasoning: String,
}

/// Filter state machine position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Outside the reasoning channel — text is answer content.
    Content,
    /// Inside the reasoning channel — text is chain-of-thought.
    Thinking,
}

/// Streaming reasoning-channel splitter.
///
/// Construct once per generation with [`ReasoningFilter::new`], feed decoded
/// fragments through [`feed`](Self::feed), and call [`flush`](Self::flush) at
/// end-of-stream. When the tokenizer exposes no reasoning markers the filter is
/// an inert passthrough: every fragment is returned verbatim as content with no
/// buffering, so a non-thinking model's output is byte-identical to feeding the
/// text straight to the terminal.
#[derive(Debug, Clone)]
pub struct ReasoningFilter {
    /// Open marker (e.g. `<|channel>thought`, `<think>`). Empty when inactive.
    think_start: String,
    /// Close marker (e.g. `<channel|>`, `</think>`). Empty when inactive.
    think_end: String,
    state: State,
    /// Bytes held back across fragment boundaries pending a delimiter decision.
    buffer: String,
    /// `false` for non-thinking models: `feed` returns its input unchanged.
    active: bool,
}

impl ReasoningFilter {
    /// Build a filter from a tokenizer's resolved reasoning markers.
    ///
    /// When both the open and close marker strings are present and non-empty
    /// the filter is active; otherwise it is an inert passthrough.
    pub fn new(markers: &ThinkingMarkers) -> Self {
        match (markers.think_start.as_deref(), markers.think_end.as_deref()) {
            (Some(start), Some(end)) if !start.is_empty() && !end.is_empty() => Self {
                think_start: start.to_string(),
                think_end: end.to_string(),
                state: State::Content,
                buffer: String::new(),
                active: true,
            },
            _ => Self::inactive(),
        }
    }

    /// Like [`new`](Self::new) but starts *inside* the reasoning channel, for a
    /// generation whose rendered prompt already primed an open thinking marker.
    ///
    /// The CLI enables thinking for Qwen-style models by rendering a prompt that
    /// ends with an open marker (`<think>\n`, or `<|channel>thought\n` for a
    /// thinking-on Gemma-4 channel). That open marker lives in the *prompt*, not
    /// in the generated tokens, so the first generated bytes are already
    /// reasoning and the close marker (`</think>` / `<channel|>`) is the first
    /// marker the model emits. A filter started in `Content` would never match
    /// the absent open marker and would print the whole chain-of-thought plus
    /// the raw close marker (issue #884, the same leak this module suppresses for
    /// a model-opened channel). Starting in `Thinking` keeps the primed reasoning
    /// hidden until the close marker flips back to content. Mirrors the server's
    /// [`StreamFilter::new_primed_open_thinking`](crate::server::tool_calls::stream_filter::StreamFilter::new_primed_open_thinking).
    /// An inactive (non-thinking) filter stays an inert passthrough.
    pub fn new_primed_open_thinking(markers: &ThinkingMarkers) -> Self {
        let mut filter = Self::new(markers);
        if filter.active {
            filter.state = State::Thinking;
        }
        filter
    }

    /// An inert passthrough filter (no reasoning markers).
    fn inactive() -> Self {
        Self {
            think_start: String::new(),
            think_end: String::new(),
            state: State::Content,
            buffer: String::new(),
            active: false,
        }
    }

    /// `true` when the filter actually splits a reasoning channel.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Feed one decoded text fragment. Returns the content/reasoning split
    /// resolvable so far; a partial trailing marker is held back for the next
    /// call.
    pub fn feed(&mut self, fragment: &str) -> ReasoningSplit {
        if !self.active {
            return ReasoningSplit {
                content: fragment.to_string(),
                reasoning: String::new(),
            };
        }
        self.buffer.push_str(fragment);
        self.drain()
    }

    /// Release any buffered tail at end-of-stream under the current state:
    /// content in `Content`, reasoning in `Thinking`. Idempotent afterwards.
    pub fn flush(&mut self) -> ReasoningSplit {
        let remaining = std::mem::take(&mut self.buffer);
        if remaining.is_empty() {
            return ReasoningSplit::default();
        }
        match self.state {
            State::Content => ReasoningSplit {
                content: remaining,
                reasoning: String::new(),
            },
            State::Thinking => ReasoningSplit {
                content: String::new(),
                reasoning: remaining,
            },
        }
    }

    /// Drain the buffer: emit resolvable content/reasoning and flip state on
    /// each complete marker, holding back a trailing partial marker.
    fn drain(&mut self) -> ReasoningSplit {
        let mut content = String::new();
        let mut reasoning = String::new();

        loop {
            if self.buffer.is_empty() {
                break;
            }
            match self.state {
                State::Content => {
                    if let Some(pos) = self.buffer.find(&self.think_start) {
                        content.push_str(&self.buffer[..pos]);
                        self.buffer.drain(..pos + self.think_start.len());
                        self.state = State::Thinking;
                    } else {
                        let safe = self.safe_emit_len(&self.think_start);
                        content.push_str(&self.buffer[..safe]);
                        self.buffer.drain(..safe);
                        break;
                    }
                }
                State::Thinking => {
                    if let Some(pos) = self.buffer.find(&self.think_end) {
                        reasoning.push_str(&self.buffer[..pos]);
                        self.buffer.drain(..pos + self.think_end.len());
                        self.state = State::Content;
                    } else {
                        let safe = self.safe_emit_len(&self.think_end);
                        reasoning.push_str(&self.buffer[..safe]);
                        self.buffer.drain(..safe);
                        break;
                    }
                }
            }
        }

        ReasoningSplit { content, reasoning }
    }

    /// Byte length of the buffer prefix that cannot be part of a `marker` that
    /// straddles into the next fragment. Any trailing suffix that is a proper
    /// prefix of `marker` is held back. The returned index always lands on a
    /// UTF-8 char boundary.
    fn safe_emit_len(&self, marker: &str) -> usize {
        let buf = &self.buffer;
        let len = buf.len();
        if len == 0 {
            return 0;
        }
        // A straddling marker can only begin within the last `marker.len() - 1`
        // bytes, so only those tail positions need checking.
        let check_from = len.saturating_sub(marker.len().saturating_sub(1));
        for (i, _) in buf.char_indices() {
            if i < check_from {
                continue;
            }
            let suffix = &buf[i..];
            if suffix.len() < marker.len() && marker.starts_with(suffix) {
                return i;
            }
        }
        len
    }
}

/// SGR "dim/faint" style; reset back to default afterwards.
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Render a split into terminal-visible text.
///
/// `content` is always emitted. `reasoning` is emitted only when
/// `show_reasoning` is set, wrapped in a dim SGR style when `dim` is true (a
/// TTY). Reasoning precedes content so a channel that closes mid-fragment keeps
/// chronological order (thinking, then answer).
pub fn render_visible(split: &ReasoningSplit, show_reasoning: bool, dim: bool) -> String {
    let mut out = String::new();
    if show_reasoning && !split.reasoning.is_empty() {
        if dim {
            out.push_str(DIM);
            out.push_str(&split.reasoning);
            out.push_str(RESET);
        } else {
            out.push_str(&split.reasoning);
        }
    }
    out.push_str(&split.content);
    out
}

/// Filter a complete decoded generation (fed at once) into terminal-visible
/// text. Used by the one-shot `generate` flow, which decodes the whole reply
/// before printing.
pub fn render_full(
    markers: &ThinkingMarkers,
    text: &str,
    primed_open: bool,
    show_reasoning: bool,
    dim: bool,
) -> String {
    let mut filter = if primed_open {
        ReasoningFilter::new_primed_open_thinking(markers)
    } else {
        ReasoningFilter::new(markers)
    };
    let mut out = render_visible(&filter.feed(text), show_reasoning, dim);
    out.push_str(&render_visible(&filter.flush(), show_reasoning, dim));
    out
}

/// Whether a rendered CLI prompt primed an open reasoning channel that the model
/// is expected to close in its generation.
///
/// True when the tokenizer exposes reasoning markers and `prompt` ends with the
/// open marker (ignoring trailing ASCII whitespace): the Qwen-style `<think>\n`
/// or Gemma-4 `<|channel>thought\n` generation-prompt suffix the CLI appends when
/// it enables thinking. The open marker is then part of the prompt, not the
/// generated text, so the caller starts the filter with
/// [`ReasoningFilter::new_primed_open_thinking`] to keep the primed thought body
/// and its raw close marker off the terminal. Mirrors the suffix check in the
/// server's `routes::chat::is_prompt_primed_open_thinking`.
pub fn prompt_primed_open_thinking(markers: &ThinkingMarkers, prompt: &str) -> bool {
    let filter = ReasoningFilter::new(markers);
    if !filter.active {
        return false;
    }
    prompt
        .trim_end_matches([' ', '\t', '\r', '\n'])
        .ends_with(filter.think_start.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 4 style markers: the open marker carries the `thought` channel
    /// argument, so consuming the whole marker keeps `thought` out of the
    /// captured reasoning.
    fn gemma_markers() -> ThinkingMarkers {
        ThinkingMarkers {
            think_start: Some("<|channel>thought".to_string()),
            think_end: Some("<channel|>".to_string()),
            think_start_tokens: Some(vec![100, 200]),
            think_end_tokens: Some(vec![101]),
            ..ThinkingMarkers::default()
        }
    }

    /// Qwen style single-token markers with no channel argument.
    fn qwen_markers() -> ThinkingMarkers {
        ThinkingMarkers {
            think_start: Some("<think>".to_string()),
            think_end: Some("</think>".to_string()),
            think_start_tokens: Some(vec![151]),
            think_end_tokens: Some(vec![152]),
            ..ThinkingMarkers::default()
        }
    }

    /// Feed the whole input to a fresh filter and aggregate feed + flush.
    fn split_once(markers: &ThinkingMarkers, input: &str) -> ReasoningSplit {
        let mut f = ReasoningFilter::new(markers);
        let a = f.feed(input);
        let b = f.flush();
        ReasoningSplit {
            content: a.content + &b.content,
            reasoning: a.reasoning + &b.reasoning,
        }
    }

    fn assert_no_markers(text: &str) {
        assert!(!text.contains("<|channel>"), "leaked open marker: {text:?}");
        assert!(
            !text.contains("<channel|>"),
            "leaked close marker: {text:?}"
        );
        assert!(!text.contains("<think>"), "leaked open marker: {text:?}");
        assert!(!text.contains("</think>"), "leaked close marker: {text:?}");
    }

    // -- (a) thought then answer: only the answer is visible, reasoning kept --

    #[test]
    fn thought_then_answer_splits() {
        let out = split_once(
            &gemma_markers(),
            "<|channel>thought\nHere is my reasoning.<channel|>The sky is blue.",
        );
        assert_eq!(out.content, "The sky is blue.");
        assert_eq!(out.reasoning, "\nHere is my reasoning.");
        assert_no_markers(&out.content);
        assert_no_markers(&out.reasoning);
        // The `thought` channel argument must not survive in the reasoning.
        assert!(
            !out.reasoning.contains("thought"),
            "channel argument leaked: {:?}",
            out.reasoning
        );
    }

    #[test]
    fn qwen_think_then_answer_splits() {
        let out = split_once(&qwen_markers(), "<think>reasoning here</think>content");
        assert_eq!(out.content, "content");
        assert_eq!(out.reasoning, "reasoning here");
        assert_no_markers(&out.content);
    }

    // -- (b) answer only, no channel: byte-identical passthrough --

    #[test]
    fn answer_only_passthrough() {
        let out = split_once(&gemma_markers(), "The answer is 4.");
        assert_eq!(out.content, "The answer is 4.");
        assert_eq!(out.reasoning, "");
    }

    #[test]
    fn answer_with_angle_brackets_passthrough() {
        // A direct answer containing `<` / `<|` must survive untouched: the
        // partial-marker holdback releases it verbatim.
        let out = split_once(&gemma_markers(), "if a < b and c <| d then stop");
        assert_eq!(out.content, "if a < b and c <| d then stop");
        assert_eq!(out.reasoning, "");
    }

    #[test]
    fn non_thinking_model_is_inert_passthrough() {
        let markers = ThinkingMarkers::default();
        let mut f = ReasoningFilter::new(&markers);
        assert!(!f.is_active());
        // Even text that looks like a marker is passed through unchanged when
        // the model has no reasoning channel.
        let out = f.feed("<|channel>thought not real<channel|>");
        assert_eq!(out.content, "<|channel>thought not real<channel|>");
        assert_eq!(out.reasoning, "");
        assert_eq!(f.flush(), ReasoningSplit::default());
    }

    // -- (c) thought opened but never closed: reasoning hidden, no marker leak --

    #[test]
    fn unclosed_thought_hides_reasoning() {
        let out = split_once(
            &gemma_markers(),
            "<|channel>thought\nStill thinking and cut off",
        );
        assert_eq!(out.content, "");
        assert_eq!(out.reasoning, "\nStill thinking and cut off");
        assert_no_markers(&out.content);
        // Nothing visible leaks the raw marker.
        assert!(out.content.is_empty());
    }

    #[test]
    fn unclosed_thought_with_partial_close_no_content_leak() {
        // Stream ends mid close-marker: the partial `<chan` stays inside the
        // hidden reasoning, never in content.
        let out = split_once(&gemma_markers(), "<|channel>thought\nreason<chan");
        assert_eq!(out.content, "");
        assert_no_markers(&out.content);
        assert!(out.reasoning.contains("reason"));
    }

    // -- (d) token-by-token, marker split across pieces --

    #[test]
    fn token_by_token_marker_split_across_pieces() {
        let mut f = ReasoningFilter::new(&gemma_markers());
        // Each marker is deliberately split across fragment boundaries.
        let pieces = [
            "<|chan",
            "nel>tho",
            "ught\n",
            "step one ",
            "step two",
            "<chan",
            "nel|>",
            "Final ",
            "answer.",
        ];
        let mut content = String::new();
        let mut reasoning = String::new();
        for p in pieces {
            let out = f.feed(p);
            // No fragment may ever surface a raw marker in content.
            assert_no_markers(&out.content);
            content.push_str(&out.content);
            reasoning.push_str(&out.reasoning);
        }
        let tail = f.flush();
        content.push_str(&tail.content);
        reasoning.push_str(&tail.reasoning);

        assert_eq!(content, "Final answer.");
        assert_eq!(reasoning, "\nstep one step two");
        assert!(!reasoning.contains("thought"));
        assert_no_markers(&content);
    }

    #[test]
    fn token_by_token_no_channel_reassembles_exactly() {
        let mut f = ReasoningFilter::new(&gemma_markers());
        let pieces = ["Hel", "lo, ", "wor", "ld <", "| ok"];
        let mut content = String::new();
        for p in pieces {
            content.push_str(&f.feed(p).content);
        }
        content.push_str(&f.flush().content);
        assert_eq!(content, "Hello, world <| ok");
    }

    #[test]
    fn multiple_reasoning_blocks_in_one_stream_all_hidden() {
        // The state machine must re-enter Thinking on a second open marker, not
        // just once per filter lifetime.
        let out = split_once(
            &gemma_markers(),
            "<|channel>thought\nfirst<channel|>mid<|channel>thought\nsecond<channel|>end",
        );
        assert_eq!(out.content, "midend");
        assert_eq!(out.reasoning, "\nfirst\nsecond");
        assert_no_markers(&out.content);
        assert_no_markers(&out.reasoning);
    }

    #[test]
    fn flush_is_idempotent_after_holding_back_a_partial_marker() {
        // "<chan" is a proper prefix of the close marker "<channel|>", so
        // `feed` holds it back in the buffer instead of releasing it as
        // reasoning; only `flush` releases it, and a second `flush` call must
        // be a no-op rather than replaying the same text.
        let mut f = ReasoningFilter::new(&gemma_markers());
        let a = f.feed("<|channel>thought\nreason<chan");
        assert_eq!(a.reasoning, "\nreason");
        assert_eq!(a.content, "");
        let first_flush = f.flush();
        let second_flush = f.flush();
        assert_eq!(first_flush.reasoning, "<chan");
        assert_eq!(first_flush.content, "");
        assert_eq!(second_flush, ReasoningSplit::default());
    }

    // -- (e) --show-reasoning surfaces reasoning without raw markers --

    #[test]
    fn show_reasoning_surfaces_dimmed_body_without_markers() {
        let out = split_once(
            &gemma_markers(),
            "<|channel>thought\nBecause of Rayleigh scattering.<channel|>The sky is blue.",
        );
        let rendered = render_visible(&out, true, true);
        // Reasoning body is present, dimmed, and precedes the answer.
        assert!(rendered.contains("Because of Rayleigh scattering."));
        assert!(rendered.contains("The sky is blue."));
        assert!(rendered.contains(DIM));
        assert!(rendered.contains(RESET));
        assert!(!rendered.contains("thought"));
        assert_no_markers(&rendered);
        let dim_at = rendered.find("Because").unwrap();
        let answer_at = rendered.find("The sky").unwrap();
        assert!(dim_at < answer_at, "reasoning must precede the answer");
    }

    #[test]
    fn show_reasoning_off_hides_reasoning_entirely() {
        let out = split_once(
            &gemma_markers(),
            "<|channel>thought\nhidden<channel|>visible",
        );
        assert_eq!(render_visible(&out, false, true), "visible");
    }

    #[test]
    fn render_visible_without_tty_omits_dim_codes() {
        let out = split_once(&gemma_markers(), "<|channel>thought\nr<channel|>a");
        let rendered = render_visible(&out, true, false);
        assert!(!rendered.contains(DIM));
        assert!(!rendered.contains(RESET));
        assert!(rendered.contains('r'));
        assert!(rendered.contains('a'));
    }

    // -- render_full convenience (one-shot generate path) --

    #[test]
    fn render_full_passthrough_is_byte_identical_when_no_channel() {
        let text = "Just a direct answer with a < b comparison.";
        assert_eq!(
            render_full(&gemma_markers(), text, false, false, true),
            text
        );
        // A non-thinking model is also byte-identical.
        assert_eq!(
            render_full(&ThinkingMarkers::default(), text, false, true, true),
            text
        );
    }

    #[test]
    fn render_full_hides_channel_by_default() {
        let text = "<|channel>thought\nreasoning<channel|>answer";
        assert_eq!(
            render_full(&gemma_markers(), text, false, false, true),
            "answer"
        );
        assert_eq!(
            render_full(&gemma_markers(), text, false, false, false),
            "answer"
        );
    }

    // -- (f) prompt-primed open thinking: the open marker lives in the prompt --

    #[test]
    fn primed_open_qwen_hides_reasoning_and_close_marker() {
        // Prompt ended with `<think>\n`, so generation starts inside the channel
        // and the first marker the model emits is the close.
        let mut f = ReasoningFilter::new_primed_open_thinking(&qwen_markers());
        let a = f.feed("reasoning body</think>The answer.");
        let b = f.flush();
        let content = a.content + &b.content;
        let reasoning = a.reasoning + &b.reasoning;
        assert_eq!(content, "The answer.");
        assert_eq!(reasoning, "reasoning body");
        assert_no_markers(&content);
    }

    #[test]
    fn primed_open_unclosed_hides_everything() {
        // No close marker before end-of-stream: the whole generation is reasoning
        // and stays hidden; nothing leaks to content.
        let mut f = ReasoningFilter::new_primed_open_thinking(&qwen_markers());
        let a = f.feed("still thinking, cut off");
        let b = f.flush();
        assert_eq!(a.content + &b.content, "");
        assert_eq!(a.reasoning + &b.reasoning, "still thinking, cut off");
    }

    #[test]
    fn primed_open_token_by_token_close_split_across_pieces() {
        let mut f = ReasoningFilter::new_primed_open_thinking(&qwen_markers());
        let pieces = ["think ", "more</", "think", ">Ans", "wer"];
        let mut content = String::new();
        let mut reasoning = String::new();
        for p in pieces {
            let out = f.feed(p);
            assert_no_markers(&out.content);
            content.push_str(&out.content);
            reasoning.push_str(&out.reasoning);
        }
        let tail = f.flush();
        content.push_str(&tail.content);
        reasoning.push_str(&tail.reasoning);
        assert_eq!(content, "Answer");
        assert_eq!(reasoning, "think more");
        assert_no_markers(&content);
    }

    #[test]
    fn primed_open_gemma_channel_hides_reasoning() {
        // Gemma-4 thinking-on primes `<|channel>thought\n`; close is `<channel|>`.
        let mut f = ReasoningFilter::new_primed_open_thinking(&gemma_markers());
        let a = f.feed("\nreasoning<channel|>Sky is blue.");
        let b = f.flush();
        assert_eq!(a.content + &b.content, "Sky is blue.");
        assert_eq!(a.reasoning + &b.reasoning, "\nreasoning");
    }

    #[test]
    fn primed_open_inactive_for_non_thinking_model() {
        // A non-thinking model has no channel, so priming is a no-op passthrough.
        let mut f = ReasoningFilter::new_primed_open_thinking(&ThinkingMarkers::default());
        assert!(!f.is_active());
        let out = f.feed("plain text</think> stays");
        assert_eq!(out.content, "plain text</think> stays");
        assert_eq!(out.reasoning, "");
    }

    #[test]
    fn prompt_primed_open_thinking_detects_open_suffix() {
        let q = qwen_markers();
        let g = gemma_markers();
        assert!(prompt_primed_open_thinking(
            &q,
            "<|im_start|>assistant\n<think>\n"
        ));
        assert!(prompt_primed_open_thinking(&q, "prefix<think>"));
        assert!(prompt_primed_open_thinking(
            &g,
            "<start_of_turn>model\n<|channel>thought\n"
        ));
        // A closed Gemma-4 scaffold ends with the close marker, not the open one.
        assert!(!prompt_primed_open_thinking(
            &g,
            "<|channel>thought\n<channel|>"
        ));
        // A plain generation prompt with no primed channel.
        assert!(!prompt_primed_open_thinking(&q, "<|im_start|>assistant\n"));
        // Non-thinking model never counts as primed.
        assert!(!prompt_primed_open_thinking(
            &ThinkingMarkers::default(),
            "anything<think>\n"
        ));
    }

    #[test]
    fn render_full_primed_open_hides_reasoning_by_default() {
        // Whole decoded generation begins inside the channel (prompt-primed).
        let text = "reasoning body</think>answer";
        assert_eq!(
            render_full(&qwen_markers(), text, true, false, false),
            "answer"
        );
        // With --show-reasoning the body surfaces, still without the raw marker.
        let shown = render_full(&qwen_markers(), text, true, true, false);
        assert!(shown.contains("reasoning body"));
        assert!(shown.contains("answer"));
        assert_no_markers(&shown);
    }
}
