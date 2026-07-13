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

//! Streaming content filter for model-specific structural tokens.
//!
//! Prevents model-internal structural tokens (`<think>`, `</think>`,
//! `<|channel>`, `<|tool_call>`, `<tool_call>`, `<turn|>`, etc.) from leaking
//! into SSE content deltas during streaming generation.
//!
//! The filter operates as a state machine on decoded text fragments (after
//! `StreamingDecodeState` has resolved UTF-8 boundaries).  It buffers text
//! at delimiter boundaries to handle the case where a delimiter straddles
//! two decode fragments.
//!
//! **Supported families:**
//! - Qwen-style reasoning: `<think>` / `</think>` — Qwen3.x, Exaone4, Hunyuan, GLM4, etc.
//! - Hermes-style tool calls: `<tool_call>` / `</tool_call>` — Qwen/DeepSeek tool call format
//! - Mistral Nemo: `[TOOL_CALLS]` — one-shot start marker (rest of output is tool call JSON)
//! - Gemma 4: `<|channel>` / `<channel|>`, `<|tool_call>` / `<tool_call|>`,
//!   `<|think|>`, `<|turn>` / `<turn|>`
//!
//! **Tool-call suppression behavior:** when the filter enters `ToolCall` state,
//! all subsequent tokens are suppressed from `delta.content`. The tool-call
//! payload is accumulated by the caller (routes/chat) and materialized as
//! `tool_calls` fields on the final SSE chunk with `finish_reason=tool_calls`.
//! Partial marker matches at token boundaries are buffered until the full
//! marker can be confirmed (no premature suppression or leakage).
//!
//! The non-streaming counterpart (`tool_calls::parser::strip_thinking`) already
//! handles these families; this filter keeps the streaming path consistent.
//!
//! Used by: routes/chat (streaming path)

/// Split output from [`StreamFilter::feed`] / [`StreamFilter::flush`].
///
/// `content` is text that should be emitted to the client as
/// `delta.content`; `reasoning` is text that belongs inside a scratchpad /
/// thinking block (e.g. `<think>…</think>` for Qwen-style models or
/// `<|channel>thought\n…<channel|>` for Gemma 4) and should be emitted as
/// `delta.reasoning_content` so downstream routers and UIs can display a
/// "thinking" status without parsing model-specific markers. Both are `None`
/// when the fragment was fully buffered or fell inside a tool-call block
/// (tool calls are materialized by the non-streaming parser path, not here).
///
/// `suppressed_positions` counts how many input **tokens** (i.e. `feed()`
/// calls) were consumed by delimiter matching in this call. When a
/// control-token sequence fires (e.g. `<tool_call>`, `</tool_call>`), the
/// matched bytes are drained from the buffer without producing any text
/// output. If the caller is tracking per-token logprobs or other
/// position-sensitive bookkeeping, it must emit an empty-text placeholder
/// event for each suppressed position so that the downstream position index
/// stays aligned with the actual token stream.
///
/// This mirrors the upstream mlx-lm fix (PR #1170, commit `aa4f880`):
///
/// ```python
/// if tok.match is not None:
///     popped = [buffered_stream.pop() for _ in tok.match]
///     for t in reversed(popped):
///         buffered_stream.append(replace(t, text=""))
/// ```
///
/// In that code each element of `tok.match` is one token ID (one token
/// position). The equivalent here counts how many `feed()` calls contributed
/// text to the matched delimiter span — **not** just the number of complete
/// delimiter matches. For single-token delimiters the two counts are
/// identical; for multi-token delimiters (e.g. a Gemma 4 `<|channel>thought`
/// marker spread across two tokens, or any delimiter that straddles two
/// consecutive `feed()` calls), this correctly counts each contributing token.
/// Callers that do not track per-token positions can ignore this field.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterOutput {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    /// Number of input token positions (i.e. `feed()` calls) consumed by
    /// delimiter matching in this call, with no text emitted. Callers that
    /// track per-token positions (e.g. logprobs alignment for streaming tool
    /// calls) should emit one empty-text placeholder event per suppressed
    /// position so the downstream position index stays in sync with the
    /// actual token stream.
    ///
    /// For most calls this is `0`. It is non-zero only when a complete
    /// delimiter (e.g. `<tool_call>`, `</tool_call>`) was matched and
    /// consumed during this `feed()` invocation, **counting each `feed()`
    /// call that contributed bytes to the matched span**.
    pub suppressed_positions: usize,
    /// Total number of input token positions (i.e. `feed()` calls) consumed
    /// from the internal buffer during this call — both text that was emitted
    /// to `content` / `reasoning` **and** delimiter matches (suppressed text).
    ///
    /// `consumed_positions >= suppressed_positions` always holds.
    /// `consumed_positions - suppressed_positions` gives the number of token
    /// positions whose text was emitted (not suppressed) in this call.
    ///
    /// Callers that maintain a per-token side-buffer (e.g. a logprob queue
    /// parallel to the text buffer) can use this to drain the corresponding
    /// entries in lockstep: drop `consumed_positions - suppressed_positions`
    /// emitted entries, then pop `suppressed_positions` entries for placeholder
    /// chunks.
    pub consumed_positions: usize,
}

impl FilterOutput {
    /// Convenience constructor for a content-only emit.
    pub fn content(text: String) -> Self {
        Self {
            content: Some(text),
            reasoning: None,
            suppressed_positions: 0,
            consumed_positions: 0,
        }
    }

    /// Convenience constructor for a reasoning-only emit.
    pub fn reasoning(text: String) -> Self {
        Self {
            content: None,
            reasoning: Some(text),
            suppressed_positions: 0,
            consumed_positions: 0,
        }
    }

    /// Convenience constructor for a suppressed-positions-only emit.
    ///
    /// Used when a delimiter match consumed one or more token positions but
    /// produced no text output. Callers that track per-token positions should
    /// emit `n` empty-text placeholder events to preserve position alignment.
    pub fn suppressed(n: usize) -> Self {
        Self {
            content: None,
            reasoning: None,
            suppressed_positions: n,
            consumed_positions: n,
        }
    }

    /// `true` when the filter produced no output at all (no text and no
    /// suppressed positions).
    pub fn is_empty(&self) -> bool {
        self.content.is_none() && self.reasoning.is_none() && self.suppressed_positions == 0
    }
}

/// Action to take when a delimiter is matched.
#[derive(Debug, Clone, Copy)]
enum DelimiterAction {
    /// Enter thinking state — suppress content until exit.
    EnterThinking,
    /// Exit thinking state — resume content emission.
    ExitThinking,
    /// Enter tool call state — suppress content until exit.
    EnterToolCall,
    /// Exit tool call state — resume content emission.
    ExitToolCall,
    /// Strip the delimiter but don't change state.
    Strip,
}

/// Delimiter table shared across reasoning model families.
///
/// Ordering notes:
/// `find_earliest_delimiter` selects by byte position first, then by longest
/// delimiter on ties, so the array order is NOT load-bearing for correctness.
/// The entries below follow a documentation convention that mirrors the probe
/// ordering in `thinking_budget::resolve_thinking_token_ids` and makes the
/// intent of each group clear:
/// - Qwen-style markers (`<think>` / `</think>`) appear before Gemma 4
///   markers to mirror the probe ordering in
///   `thinking_budget::resolve_thinking_token_ids`.  No Qwen delimiter
///   prefix-matches any Gemma 4 delimiter, so the order has no runtime effect.
/// - `</tool_call>` (Hermes exit) appears before `<tool_call>` (Hermes enter)
///   as a documentation convention: the closer is listed before the opener.
///   Because matching selects by earliest byte position, ordering cannot cause
///   the enter tag to win over the exit tag when both appear at the same offset
///   (they cannot — they are different strings at different positions).
/// - Gemma 4 `<|tool_call>` / `<tool_call|>` appear before the plain Hermes
///   `<tool_call>` / `</tool_call>` entries because the Gemma 4 open tag
///   (`<|tool_call>`) is a prefix-extension of the Hermes open tag
///   (`<tool_call>` inside `<|tool_call>`).  Probing Gemma 4 first avoids
///   a spurious Hermes match inside Gemma 4 output.  This IS a correctness
///   guard: both tags match at the same position, so the longest-delimiter
///   tiebreak selects `<|tool_call>` (12 bytes) over `<tool_call>` (11 bytes).
/// - `[TOOL_CALLS]` (Mistral Nemo) is a one-shot entry marker — no exit marker
///   exists; all subsequent content (the JSON array) is tool-call payload.
///   It is placed after the `<tag>` families so angle-bracket markers still
///   benefit from the tighter (shorter) partial-match window that `<` gives.
/// - Kimi K2 `<|tool_calls_section_end|>` / `<|tool_calls_section_begin|>` —
///   exit before enter, matching the Hermes convention above. Neither prefix-
///   collides with any other entry: `<|tool_call>` (Gemma 4) requires a `>`
///   immediately after `tool_call`, whereas Kimi's markers continue with
///   `s_section_...`, so the two families never partial-match each other.
/// - Pythonic `<|tool_call_start|>` is **enter-only on purpose** — see the
///   comment on its entry below for why an `<|tool_call_end|>` exit row must
///   NOT be added. It does not prefix-collide with Gemma 4's `<|tool_call>`
///   either: they diverge at `_start` vs `>` immediately after `tool_call`.
///
/// TODO: Consider extracting this into a startup-time configurable
/// owned by the model worker so new reasoning families don't require a rebuild.
const CHAT_DELIMITERS: &[(&str, DelimiterAction)] = &[
    // Qwen-style reasoning (Qwen3.x, Exaone4, Hunyuan, GLM4, Nemotron-H, SmolLM3, …)
    // Probe these first to mirror resolve_thinking_token_ids ordering.
    ("</think>", DelimiterAction::ExitThinking),
    ("<think>", DelimiterAction::EnterThinking),
    // Gemma 4 tool-call delimiters — must precede Hermes `<tool_call>` to avoid
    // a spurious Hermes hit on the Gemma 4 open tag.
    ("<|tool_call>", DelimiterAction::EnterToolCall),
    ("<tool_call|>", DelimiterAction::ExitToolCall),
    // Gemma 4 reasoning channel and structural strip markers.
    ("<|channel>", DelimiterAction::EnterThinking),
    ("<channel|>", DelimiterAction::ExitThinking),
    ("<|think|>", DelimiterAction::Strip),
    ("<|turn>", DelimiterAction::Strip),
    ("<turn|>", DelimiterAction::Strip),
    // Hermes / Qwen / DeepSeek tool call markers.
    // Exit before enter so the closer is matched when both appear in one fragment.
    ("</tool_call>", DelimiterAction::ExitToolCall),
    ("<tool_call>", DelimiterAction::EnterToolCall),
    // Mistral Nemo: `[TOOL_CALLS] [...]` — no exit; the rest of output is JSON.
    ("[TOOL_CALLS]", DelimiterAction::EnterToolCall),
    // Kimi K2: `<|tool_calls_section_begin|>...<|tool_calls_section_end|>`.
    // Exit before enter, mirroring the Hermes convention above.
    ("<|tool_calls_section_end|>", DelimiterAction::ExitToolCall),
    (
        "<|tool_calls_section_begin|>",
        DelimiterAction::EnterToolCall,
    ),
    // Pythonic: `<|tool_call_start|>[func(arg=value)]<|tool_call_end|>`.
    // Enter-only, deliberately with NO matching `<|tool_call_end|>` exit row:
    // that marker is already registered as Kimi K2's PER-CALL terminator
    // inside a `<|tool_calls_section_begin|>...<|tool_calls_section_end|>`
    // section, whose streaming suppression relies on the section begin/end
    // pair and treats `<|tool_call_end|>` as ordinary suppressed content. If
    // `<|tool_call_end|>` were also registered as a global `ExitToolCall`
    // here, it would fire on every Kimi K2 per-call terminator and
    // prematurely exit tool-call suppression mid-section, leaking the
    // second and later Kimi calls into visible `delta.content` (a
    // regression of #766). Pythonic calls are terminal in practice, so
    // enter-only (mirroring Mistral Nemo's one-shot `[TOOL_CALLS]` above)
    // fully suppresses the payload without needing an exit marker.
    ("<|tool_call_start|>", DelimiterAction::EnterToolCall),
];

/// The state of the streaming filter state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterState {
    /// Emitting content to the client.
    Content,
    /// Inside a thinking/channel block — suppress all text.
    Thinking,
    /// Inside a tool call block — suppress all text.
    ToolCall,
}

/// Streaming content filter that strips model-specific structural tokens from
/// content deltas, routing reasoning-block text to `delta.reasoning_content`
/// and regular response text to `delta.content`.
///
/// Covers Qwen-style (`<think>` / `</think>`), Hermes-style
/// (`<tool_call>` / `</tool_call>`), Mistral Nemo (`[TOOL_CALLS]`), and
/// Gemma 4 (`<|channel>` / `<channel|>`) reasoning families, plus Gemma 4
/// tool-call and turn markers.
///
/// Feed decoded text fragments via [`feed()`](StreamFilter::feed).  The
/// filter returns only the content that should be emitted to the client.
/// Call [`flush()`](StreamFilter::flush) at the end of generation to emit
/// any remaining buffered content.
pub struct StreamFilter {
    state: FilterState,
    buffer: String,
    /// Per-feed byte-length queue used to count how many token positions
    /// (i.e. `feed()` calls) contributed to the bytes currently in `buffer`.
    ///
    /// Each `feed(fragment)` call appends `fragment.len()` to this queue.
    /// When `drain_buffer()` consumes bytes from the front of `buffer` (via
    /// `buffer.drain(..N)`), it pops fragment-length entries from the front
    /// of this queue until `N` bytes are accounted for. The number of entries
    /// popped is the number of token positions whose text spanned that byte
    /// range — used as `suppressed_positions` for delimiter matches.
    ///
    /// Invariant: `fragment_lengths.iter().sum::<usize>() == buffer.len()`
    /// before and after every `feed()` / `drain_buffer()` call.
    fragment_lengths: std::collections::VecDeque<usize>,
    /// Length of the longest delimiter (for partial-match buffering).
    max_delim_len: usize,
}

impl Default for StreamFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamFilter {
    /// Create a new stream filter covering Qwen-style, Hermes-style, Mistral Nemo, and Gemma 4 delimiters.
    pub fn new() -> Self {
        let max_delim_len = CHAT_DELIMITERS
            .iter()
            .map(|(s, _)| s.len())
            .max()
            .unwrap_or(0);

        Self {
            state: FilterState::Content,
            buffer: String::new(),
            fragment_lengths: std::collections::VecDeque::new(),
            max_delim_len,
        }
    }

    /// Create a filter that starts inside a `Thinking` block.
    ///
    /// Use when the chat template's generation prompt primed an open thinking
    /// marker — either `<|channel>thought\n` (Gemma 4 `enable_thinking=true`)
    /// or `<think>\n` (Qwen-style `enable_thinking=true`): the first emitted
    /// token is already reasoning content, not a regular response, so it is
    /// routed to `reasoning` until the model emits the matching close marker
    /// (`<channel|>` or `</think>`) to transition back to `Content`.  If the
    /// model never closes the block, the entire generation is suppressed —
    /// matching the non-streaming post-processor behavior in
    /// `routes::chat::strip_unclosed_primed_thinking`.
    pub fn new_primed_open_thinking() -> Self {
        let mut s = Self::new();
        s.state = FilterState::Thinking;
        s
    }

    /// Feed a decoded text fragment. Returns split output: `content` holds
    /// text that should be emitted as `delta.content`, `reasoning` holds
    /// text that should be emitted as `delta.reasoning_content`. Either (or
    /// both) may be `None` when the fragment is entirely buffered / suppressed.
    ///
    /// Each call corresponds to exactly one decoded token. The byte length of
    /// `fragment` is recorded in `fragment_lengths` so that `drain_buffer()`
    /// can count how many token positions (i.e. `feed()` calls) contributed
    /// to a given byte range when a delimiter is matched.
    pub fn feed(&mut self, fragment: &str) -> FilterOutput {
        // Empty decoded fragments do not add bytes to the delimiter buffer,
        // but they still represent one generated token position. Do not push
        // a zero-length entry into `fragment_lengths`: it cannot satisfy the
        // byte-sum invariant and would keep the caller's parallel lp_data
        // queue from draining until some later non-empty fragment arrives.
        if fragment.is_empty() {
            return FilterOutput {
                consumed_positions: 1,
                ..FilterOutput::default()
            };
        }

        // Record the number of bytes this token contributes to the buffer.
        self.fragment_lengths.push_back(fragment.len());
        self.buffer.push_str(fragment);
        self.drain_buffer()
    }

    /// Flush any remaining buffered content at the end of generation.
    ///
    /// Emits the tail under the channel that matches the filter's current
    /// state: anything still buffered inside `Thinking` surfaces as
    /// `reasoning`, anything inside `Content` surfaces as `content`, and
    /// anything inside `ToolCall` is discarded (tool-call deltas are emitted
    /// via the parser path, not here).
    pub fn flush(&mut self) -> FilterOutput {
        if self.buffer.is_empty() {
            self.fragment_lengths.clear();
            return FilterOutput::default();
        }
        let remaining = std::mem::take(&mut self.buffer);
        self.fragment_lengths.clear();
        match self.state {
            FilterState::Content if !remaining.is_empty() => FilterOutput::content(remaining),
            FilterState::Thinking if !remaining.is_empty() => FilterOutput::reasoning(remaining),
            _ => FilterOutput::default(),
        }
    }

    /// Drain `n` bytes from the front of `fragment_lengths`, returning the
    /// number of `feed()` calls (token positions) that were fully or partially
    /// consumed.
    ///
    /// Each entry in `fragment_lengths` records how many bytes one `feed()`
    /// call contributed to `buffer`. This method walks the queue from the
    /// front, subtracting each entry from `n` until all `n` bytes are
    /// accounted for. The count of entries consumed is returned.
    ///
    /// A fragment entry is considered consumed if its bytes fall entirely
    /// within the `n`-byte window **or** if its bytes straddle the boundary
    /// (the last entry covers bytes on both sides). Straddling means the
    /// delimiter ended in the middle of what was originally one token's text —
    /// that token still counts as a suppressed position because it contributed
    /// bytes to the matched span.
    ///
    /// After this call the sum of remaining entries in `fragment_lengths`
    /// equals `buffer.len() - n` (since the caller drains those bytes from
    /// `buffer` immediately after).
    fn drain_fragment_lengths(&mut self, mut n: usize) -> usize {
        let mut count = 0usize;
        while n > 0 {
            match self.fragment_lengths.pop_front() {
                Some(frag_len) => {
                    count += 1;
                    if frag_len <= n {
                        n -= frag_len;
                    } else {
                        // This fragment straddles the boundary: `n` bytes were
                        // consumed from it but `frag_len - n` bytes remain.
                        // Push back the remainder so the invariant holds.
                        let remainder = frag_len - n;
                        self.fragment_lengths.push_front(remainder);
                        n = 0;
                    }
                }
                None => break,
            }
        }
        count
    }

    /// Process the internal buffer: find delimiters, emit content/reasoning,
    /// and transition states.
    ///
    /// When a complete delimiter is consumed (e.g. `<tool_call>`,
    /// `</tool_call>`), the matched bytes are drained without producing text
    /// output. `suppressed_positions` in the returned [`FilterOutput`] is
    /// incremented by the number of `feed()` calls whose text spanned the
    /// matched byte range — **not** merely by one per delimiter hit. This
    /// correctly handles multi-token delimiters (e.g. a delimiter whose bytes
    /// were delivered across two or more consecutive `feed()` calls).
    ///
    /// This mirrors the upstream mlx-lm fix (PR #1170, commit `aa4f880`):
    /// `popped = [buffered_stream.pop() for _ in tok.match]` followed by
    /// `buffered_stream.append(replace(t, text=""))` — each element of
    /// `tok.match` is one token ID (one position); here we count how many
    /// `feed()` positions contributed bytes to the matched span.
    fn drain_buffer(&mut self) -> FilterOutput {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut suppressed_positions: usize = 0;
        let mut consumed_positions: usize = 0;

        loop {
            if self.buffer.is_empty() {
                break;
            }

            match self.find_earliest_delimiter() {
                Some((pos, delim_len, action)) => {
                    // Text before the delimiter is attributed to the current
                    // state so thinking fragments surface as reasoning and
                    // regular fragments surface as content.
                    if pos > 0 {
                        match self.state {
                            FilterState::Content => content.push_str(&self.buffer[..pos]),
                            FilterState::Thinking => reasoning.push_str(&self.buffer[..pos]),
                            FilterState::ToolCall => {}
                        }
                        // These bytes are emitted (not suppressed). Drain them
                        // from fragment_lengths and count as consumed (emitted).
                        consumed_positions += self.drain_fragment_lengths(pos);
                        self.buffer.drain(..pos);
                    }

                    // Consume the delimiter bytes and count how many token
                    // positions they spanned. Each position that contributed
                    // bytes to the delimiter range is one suppressed position:
                    // callers emit empty-text placeholder events for each.
                    let delim_positions = self.drain_fragment_lengths(delim_len);
                    suppressed_positions += delim_positions;
                    consumed_positions += delim_positions;
                    self.buffer.drain(..delim_len);
                    self.apply_action(action);
                }
                None => {
                    // No complete delimiter — check for a partial match
                    // at the tail and hold it back.
                    let safe_len = self.safe_emit_length();

                    if safe_len > 0 {
                        match self.state {
                            FilterState::Content => content.push_str(&self.buffer[..safe_len]),
                            FilterState::Thinking => reasoning.push_str(&self.buffer[..safe_len]),
                            FilterState::ToolCall => {}
                        }
                        consumed_positions += self.drain_fragment_lengths(safe_len);
                        self.buffer.drain(..safe_len);
                    }
                    break;
                }
            }
        }

        FilterOutput {
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            reasoning: if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            },
            suppressed_positions,
            consumed_positions,
        }
    }

    /// Find the earliest complete delimiter in the buffer.
    ///
    /// Returns `(byte_position, delimiter_len, action)`.
    fn find_earliest_delimiter(&self) -> Option<(usize, usize, DelimiterAction)> {
        let mut earliest: Option<(usize, usize, DelimiterAction)> = None;

        for &(delim, action) in CHAT_DELIMITERS {
            if let Some(pos) = self.buffer.find(delim) {
                match earliest {
                    Some((best_pos, best_len, _)) => {
                        // Pick earliest position; on tie, pick longest delimiter
                        if pos < best_pos || (pos == best_pos && delim.len() > best_len) {
                            earliest = Some((pos, delim.len(), action));
                        }
                    }
                    None => {
                        earliest = Some((pos, delim.len(), action));
                    }
                }
            }
        }

        earliest
    }

    /// Find how many bytes at the start of the buffer are safe to emit,
    /// i.e. no suffix of the buffer could be the start of a delimiter.
    fn safe_emit_length(&self) -> usize {
        let buf = &self.buffer;
        let buf_len = buf.len();
        if buf_len == 0 {
            return 0;
        }

        // Only check positions near the end (within max_delim_len - 1 bytes)
        let check_from = buf_len.saturating_sub(self.max_delim_len.saturating_sub(1));

        for (i, _) in buf.char_indices() {
            if i < check_from {
                continue;
            }
            let suffix = &buf[i..];
            for &(delim, _) in CHAT_DELIMITERS {
                // A suffix is a partial match if it's a proper prefix of a
                // delimiter (shorter, and the delimiter starts with it).
                if suffix.len() < delim.len() && delim.starts_with(suffix) {
                    return i;
                }
            }
        }

        buf_len
    }

    /// Apply a delimiter action to transition the state machine.
    fn apply_action(&mut self, action: DelimiterAction) {
        match action {
            DelimiterAction::EnterThinking => self.state = FilterState::Thinking,
            DelimiterAction::ExitThinking => self.state = FilterState::Content,
            DelimiterAction::EnterToolCall => self.state = FilterState::ToolCall,
            DelimiterAction::ExitToolCall => self.state = FilterState::Content,
            DelimiterAction::Strip => { /* consume delimiter, no state change */ }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Accepted design constraint: any model (including non-thinking models)
    // that emits literal `<think>…</think>` text in its response will have
    // that text routed to `reasoning_content` rather than `content`.  This is
    // intentional and matches the non-streaming `strip_thinking` behavior in
    // `tool_calls::parser` — both paths treat the markers as structural
    // regardless of context.  A model that wants to discuss the `<think>` tag
    // as prose should escape or paraphrase it.

    // -- Basic passthrough --

    #[test]
    fn no_special_tokens_passthrough() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("Hello world").content,
            Some("Hello world".to_string())
        );
    }

    #[test]
    fn empty_fragment() {
        let mut f = StreamFilter::new();
        let out = f.feed("");
        assert_eq!(out.content, None);
        assert_eq!(out.reasoning, None);
        assert_eq!(out.suppressed_positions, 0);
        assert_eq!(
            out.consumed_positions, 1,
            "empty decoded text still consumes one generated token position"
        );
        assert_eq!(f.flush(), FilterOutput::default());
    }

    // -- Thinking blocks --

    #[test]
    fn thinking_only() {
        let mut f = StreamFilter::new();
        let result = f.feed("<|channel>thought\nI should search.<channel|><turn|>");
        assert_eq!(result.content, None);
    }

    #[test]
    fn thinking_then_content() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("<|channel>thought\nPlanning.<channel|>Here is the answer.")
                .content,
            Some("Here is the answer.".to_string())
        );
    }

    #[test]
    fn thinking_then_content_with_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("<|channel>thought\nPlanning.<channel|>Here is the answer.<turn|>")
                .content,
            Some("Here is the answer.".to_string())
        );
    }

    #[test]
    fn thinking_then_tool_call() {
        let mut f = StreamFilter::new();
        let input = "<|channel>thought\nI need to search.<channel|>\
                      <|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|><turn|>";
        assert_eq!(f.feed(input).content, None);
    }

    // -- Tool call blocks --

    #[test]
    fn content_then_tool_call() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("Let me search.<|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|>")
                .content,
            Some("Let me search.".to_string())
        );
    }

    #[test]
    fn tool_call_only() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|tool_call>call:fn{}<tool_call|>").content, None);
    }

    // -- Turn markers --

    #[test]
    fn strips_trailing_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Answer").content, Some("Answer".to_string()));
        assert_eq!(f.feed("<turn|>").content, None);
    }

    #[test]
    fn strips_leading_turn() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|turn>Hello").content, Some("Hello".to_string()));
    }

    #[test]
    fn strips_think_marker() {
        let mut f = StreamFilter::new();
        assert_eq!(
            f.feed("Before<|think|>After").content,
            Some("BeforeAfter".to_string())
        );
    }

    // -- Partial delimiter buffering --

    #[test]
    fn partial_delimiter_at_end() {
        let mut f = StreamFilter::new();
        // Fragment ends mid-delimiter "<|to" — could be "<|tool_call>"
        assert_eq!(f.feed("Hello <|to").content, Some("Hello ".to_string()));
        // Next fragment completes the delimiter
        assert_eq!(f.feed("ol_call>rest<tool_call|>").content, None);
    }

    #[test]
    fn partial_delimiter_resolves_to_content() {
        let mut f = StreamFilter::new();
        // "<|" at the end could start a delimiter
        assert_eq!(f.feed("test <|").content, Some("test ".to_string()));
        // Next fragment shows it's not a delimiter
        assert_eq!(
            f.feed("not_a_delim").content,
            Some("<|not_a_delim".to_string())
        );
    }

    #[test]
    fn partial_channel_delimiter() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Hello <|cha").content, Some("Hello ".to_string()));
        // Completes the channel delimiter
        assert_eq!(
            f.feed("nnel>thought\nthinking<channel|>World").content,
            Some("World".to_string())
        );
    }

    #[test]
    fn angle_bracket_mid_text_passes_through() {
        let mut f = StreamFilter::new();
        // "< 3" does NOT match any delimiter prefix (none start with "< ")
        // so the entire text passes through immediately.
        assert_eq!(f.feed("2 < 3").content, Some("2 < 3".to_string()));
    }

    #[test]
    fn trailing_angle_bracket_buffered() {
        let mut f = StreamFilter::new();
        // A lone '<' at the very end IS held back (could start any delimiter)
        assert_eq!(f.feed("end<").content, Some("end".to_string()));
        // Next fragment resolves it — not a delimiter
        assert_eq!(f.feed(" more").content, Some("< more".to_string()));
    }

    // -- Multi-fragment streaming --

    #[test]
    fn token_by_token_content() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("Hello").content, Some("Hello".to_string()));
        assert_eq!(f.feed(" ").content, Some(" ".to_string()));
        assert_eq!(f.feed("world").content, Some("world".to_string()));
    }

    #[test]
    fn token_by_token_thinking_then_content() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|channel>").content, None);
        assert_eq!(f.feed("thought\n").content, None);
        assert_eq!(f.feed("I am thinking.").content, None);
        assert_eq!(f.feed("<channel|>").content, None);
        assert_eq!(
            f.feed("Here is my answer.").content,
            Some("Here is my answer.".to_string())
        );
    }

    #[test]
    fn token_by_token_tool_call() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|tool_call>").content, None);
        assert_eq!(f.feed("call:search{").content, None);
        assert_eq!(f.feed("query:<|\"|>test<|\"|>}").content, None);
        assert_eq!(f.feed("<tool_call|>").content, None);
        // Back in content state
        assert_eq!(f.flush().content, None);
    }

    // -- Flush --

    #[test]
    fn flush_emits_buffered_content() {
        let mut f = StreamFilter::new();
        // '<' is held back as potential delimiter start
        assert_eq!(f.feed("end<").content, Some("end".to_string()));
        // Flush at end of generation: emit the held-back '<'
        assert_eq!(f.flush().content, Some("<".to_string()));
    }

    #[test]
    fn flush_in_thinking_suppresses() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("<|channel>unclosed thinking").content, None);
        // Still in thinking state at flush — suppress
        assert_eq!(f.flush().content, None);
    }

    // -- No false positives --

    #[test]
    fn angle_brackets_in_normal_text() {
        let mut f = StreamFilter::new();
        // HTML-like content should pass through
        assert_eq!(
            f.feed("<div>hello</div>").content,
            Some("<div>hello</div>".to_string())
        );
    }

    #[test]
    fn pipe_in_normal_text() {
        let mut f = StreamFilter::new();
        assert_eq!(f.feed("a | b | c").content, Some("a | b | c".to_string()));
    }

    // -- Reasoning split (FilterOutput.content vs .reasoning) --

    #[test]
    fn thinking_block_surfaces_as_reasoning_not_content() {
        // Inside a `<|channel>thought…<channel|>` block, fragments belong to
        // `reasoning` so the server can emit them as `delta.reasoning_content`
        // SSE chunks. Content after the close stays in `content`.
        let mut f = StreamFilter::new();
        let out = f.feed("<|channel>thought\nReasoning text<channel|>Answer text");
        assert_eq!(out.reasoning.as_deref(), Some("thought\nReasoning text"));
        assert_eq!(out.content.as_deref(), Some("Answer text"));
    }

    #[test]
    fn primed_open_thinking_starts_in_reasoning_state() {
        // When the chat template primed an open `<|channel>thought\n` at the
        // end of the generation prompt, the model's first emitted tokens are
        // already inside the thinking channel. Starting the filter in
        // `Thinking` state routes them to `reasoning`.
        let mut f = StreamFilter::new_primed_open_thinking();
        let out = f.feed("I am thinking<channel|>the answer");
        assert_eq!(out.reasoning.as_deref(), Some("I am thinking"));
        assert_eq!(out.content.as_deref(), Some("the answer"));
    }

    #[test]
    fn primed_open_thinking_unclosed_suppresses_content_entirely() {
        // Model ran out of budget inside the primed channel — never emits
        // `<channel|>`. The filter keeps every token in `reasoning` and
        // leaves `content` empty, including on flush.
        let mut f = StreamFilter::new_primed_open_thinking();
        let out = f.feed("reasoning continues");
        assert_eq!(out.content, None);
        assert_eq!(out.reasoning.as_deref(), Some("reasoning continues"));

        let flushed = f.flush();
        assert_eq!(flushed.content, None);
        // Buffer was already drained inside feed, so flush has nothing left
        // to surface. Either None or empty is acceptable.
        assert!(
            flushed.reasoning.as_deref().unwrap_or("").is_empty(),
            "flush must not leak buffered reasoning as content"
        );
    }

    #[test]
    fn primed_open_thinking_token_by_token() {
        // Fragment-by-fragment streaming: every pre-close token is reasoning,
        // every post-close token is content.
        let mut f = StreamFilter::new_primed_open_thinking();
        assert_eq!(f.feed("Let me").reasoning.as_deref(), Some("Let me"));
        assert_eq!(f.feed(" think.").reasoning.as_deref(), Some(" think."));
        // Close marker transitions state; nothing emitted on the marker itself.
        let closing = f.feed("<channel|>");
        assert!(closing.reasoning.is_none());
        assert!(closing.content.is_none());
        // After the close, fragments are content.
        assert_eq!(f.feed("Done.").content.as_deref(), Some("Done."));
    }

    #[test]
    fn flush_in_content_state_emits_content_only() {
        // Pre-existing behavior: when the filter is in the Content state at
        // flush, any buffered tail surfaces as content (not reasoning).
        let mut f = StreamFilter::new();
        // '<' is held back as a potential delimiter start.
        assert_eq!(f.feed("end<").content.as_deref(), Some("end"));
        let flushed = f.flush();
        assert_eq!(flushed.content.as_deref(), Some("<"));
        assert!(flushed.reasoning.is_none());
    }

    // -- Qwen-style <think> / </think> reasoning --

    #[test]
    fn qwen_think_full_chunk() {
        // Full chunk: reasoning inside <think>…</think> surfaces as `reasoning`,
        // content after the close surfaces as `content`, no raw markers leak.
        let mut f = StreamFilter::new();
        let out = f.feed("<think>reasoning here</think>content");
        assert_eq!(out.reasoning.as_deref(), Some("reasoning here"));
        assert_eq!(out.content.as_deref(), Some("content"));
        // Verify markers themselves don't appear in either field
        assert!(!out.reasoning.as_deref().unwrap_or("").contains("<think>"));
        assert!(!out.reasoning.as_deref().unwrap_or("").contains("</think>"));
        assert!(!out.content.as_deref().unwrap_or("").contains("<think>"));
        assert!(!out.content.as_deref().unwrap_or("").contains("</think>"));
    }

    #[test]
    fn qwen_think_prompt_primed() {
        // Prompt-primed case: generation prompt appended `<think>\n` so the
        // model's first token is already reasoning.  Start the filter in
        // Thinking state; the first `</think>` closes the block.
        let mut f = StreamFilter::new_primed_open_thinking();
        let out = f.feed("reasoning here</think>content");
        assert_eq!(out.reasoning.as_deref(), Some("reasoning here"));
        assert_eq!(out.content.as_deref(), Some("content"));
        assert!(!out.content.as_deref().unwrap_or("").contains("</think>"));
    }

    #[test]
    fn qwen_think_token_by_token_full_chunk() {
        // Token-by-token streaming of the full-chunk case: feed one short
        // fragment at a time and aggregate the deltas.
        let mut f = StreamFilter::new();
        let fragments = [
            "<thi", "nk>", "rea", "son", "ing", " he", "re<", "/th", "ink", ">con", "tent",
        ];
        let mut total_reasoning = String::new();
        let mut total_content = String::new();
        for frag in &fragments {
            let out = f.feed(frag);
            if let Some(r) = out.reasoning {
                total_reasoning.push_str(&r);
            }
            if let Some(c) = out.content {
                total_content.push_str(&c);
            }
        }
        // Flush picks up anything still buffered
        let flushed = f.flush();
        if let Some(r) = flushed.reasoning {
            total_reasoning.push_str(&r);
        }
        if let Some(c) = flushed.content {
            total_content.push_str(&c);
        }
        assert_eq!(total_reasoning, "reasoning here");
        assert_eq!(total_content, "content");
        assert!(!total_reasoning.contains("<think>"));
        assert!(!total_reasoning.contains("</think>"));
        assert!(!total_content.contains("<think>"));
        assert!(!total_content.contains("</think>"));
    }

    #[test]
    fn qwen_think_token_by_token_prompt_primed() {
        // Token-by-token streaming of the prompt-primed case.
        let mut f = StreamFilter::new_primed_open_thinking();
        let fragments = [
            "rea", "son", "ing", " he", "re<", "/th", "ink", ">con", "tent",
        ];
        let mut total_reasoning = String::new();
        let mut total_content = String::new();
        for frag in &fragments {
            let out = f.feed(frag);
            if let Some(r) = out.reasoning {
                total_reasoning.push_str(&r);
            }
            if let Some(c) = out.content {
                total_content.push_str(&c);
            }
        }
        let flushed = f.flush();
        if let Some(r) = flushed.reasoning {
            total_reasoning.push_str(&r);
        }
        if let Some(c) = flushed.content {
            total_content.push_str(&c);
        }
        assert_eq!(total_reasoning, "reasoning here");
        assert_eq!(total_content, "content");
    }

    // -- Gemma 4 regression guard --

    #[test]
    fn gemma4_regression_channel_reasoning_and_content() {
        // Regression guard: Gemma 4 `<|channel>reasoning<channel|>content`
        // must still route correctly after the Qwen entries were added to
        // CHAT_DELIMITERS. Any change to the filter that silently breaks
        // Gemma 4 will be caught here.
        let mut f = StreamFilter::new();
        let out = f.feed("<|channel>thought\nReasoning text<channel|>Answer text");
        assert_eq!(out.reasoning.as_deref(), Some("thought\nReasoning text"));
        assert_eq!(out.content.as_deref(), Some("Answer text"));
        assert!(!out.content.as_deref().unwrap_or("").contains("<|channel>"));
        assert!(!out.content.as_deref().unwrap_or("").contains("<channel|>"));
    }

    // -- Hermes / Qwen / DeepSeek tool-call markup suppression --

    #[test]
    fn hermes_tool_call_only_suppressed() {
        // A response that is entirely a Hermes-style tool call: the model emits
        // `<tool_call>{...}</tool_call>` without any preceding text.
        // delta.content must remain empty — no markup leaks.
        let mut f = StreamFilter::new();
        let out = f.feed(
            r#"<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#,
        );
        assert_eq!(
            out.content, None,
            "tool-call markup must not appear in delta.content"
        );
        assert_eq!(out.reasoning, None);
    }

    #[test]
    fn hermes_content_then_tool_call_strips_markup() {
        // Model emits normal text then transitions to a tool call. Only the
        // preamble text must appear in delta.content; the tool-call markup and
        // payload must be suppressed.
        let mut f = StreamFilter::new();
        let out = f.feed(r#"Let me check the weather.<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>"#);
        assert_eq!(
            out.content.as_deref(),
            Some("Let me check the weather."),
            "only the preamble text must appear in delta.content"
        );
        assert!(!out.content.as_deref().unwrap_or("").contains("<tool_call>"));
        assert!(!out.content.as_deref().unwrap_or("").contains("get_weather"));
        assert!(
            !out.content
                .as_deref()
                .unwrap_or("")
                .contains("</tool_call>")
        );
    }

    #[test]
    fn hermes_tool_call_token_by_token_suppressed() {
        // Feed the Hermes tool-call markers and payload one token at a time.
        // No token that is part of the markup or JSON payload should surface
        // as delta.content.
        let mut f = StreamFilter::new();
        let fragments = [
            "<tool_call>",
            r#"{"name": "search", "arguments": {"q": "rust"}}"#,
            "</tool_call>",
        ];
        let mut total_content = String::new();
        for frag in &fragments {
            if let Some(c) = f.feed(frag).content {
                total_content.push_str(&c);
            }
        }
        assert!(
            total_content.is_empty(),
            "tool-call tokens must not appear in delta.content during streaming; got: {total_content:?}"
        );
        let flushed = f.flush();
        assert_eq!(
            flushed.content, None,
            "flush must not emit buffered tool-call payload"
        );
    }

    #[test]
    fn hermes_partial_marker_at_token_boundary_buffered_and_released() {
        // When a fragment ends with `<tool_` (a prefix of `<tool_call>`), the
        // filter must hold it back until more text arrives to determine whether
        // it is really a delimiter.  If it turns out not to be a tool-call
        // boundary, the held-back bytes must be released into delta.content.
        let mut f = StreamFilter::new();
        // Buffer ends with a 7-byte prefix of `<tool_call>`.
        let out1 = f.feed("preamble <tool_");
        assert_eq!(
            out1.content.as_deref(),
            Some("preamble "),
            "text before partial marker must be emitted immediately"
        );
        // Continue: next fragment resolves the ambiguity as NOT a tool-call tag.
        let out2 = f.feed("notamarker>");
        assert_eq!(
            out2.content.as_deref(),
            Some("<tool_notamarker>"),
            "held-back partial prefix must be released to delta.content when not a marker"
        );
    }

    #[test]
    fn hermes_partial_marker_at_boundary_then_complete_tool_call() {
        // Fragment ends with partial marker prefix. Next fragment completes it
        // as a real `<tool_call>` open tag. The tag must NOT appear in content.
        let mut f = StreamFilter::new();
        // Buffer ends with partial prefix of `<tool_call>`.
        let out1 = f.feed("preamble <tool_");
        assert_eq!(
            out1.content.as_deref(),
            Some("preamble "),
            "preamble before partial prefix must be emitted"
        );
        // Next fragment completes the open tag and adds JSON payload.
        let out2 = f.feed(r#"call>{"name": "fn", "arguments": {}}</tool_call>"#);
        assert_eq!(
            out2.content, None,
            "tool-call open tag completion must suppress subsequent content"
        );
    }

    #[test]
    fn hermes_multiple_tool_calls_suppressed() {
        // Two consecutive Hermes tool calls — all markup and payloads suppressed.
        let mut f = StreamFilter::new();
        let input = r#"<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>"#;
        let out = f.feed(input);
        assert_eq!(
            out.content, None,
            "multiple tool calls must be fully suppressed"
        );
    }

    #[test]
    fn hermes_flush_in_tool_call_state_suppresses() {
        // If generation ends while still inside `<tool_call>...</tool_call>`,
        // flush must discard the buffered partial payload (not emit it).
        let mut f = StreamFilter::new();
        // Open tag seen, no closing tag yet.
        let out = f.feed(r#"<tool_call>{"name": "fn", "arguments": {}}"#);
        assert_eq!(out.content, None);
        let flushed = f.flush();
        assert_eq!(
            flushed.content, None,
            "flush inside ToolCall state must not emit suppressed payload"
        );
    }

    // -- Regression: Gemma 4 tool-call markers must not be confused with Hermes --

    #[test]
    fn gemma4_tool_call_not_confused_with_hermes() {
        // Gemma 4 uses `<|tool_call>` (with pipe) as the open tag; Hermes uses
        // `<tool_call>` (no pipe). After adding both to CHAT_DELIMITERS, the
        // filter must continue to correctly suppress Gemma 4 markers while not
        // confusing one format for the other.
        let mut f = StreamFilter::new();
        // Gemma 4 tool call (pipe-delimited)
        let out = f.feed("<|tool_call>call:fn{}<tool_call|>");
        assert_eq!(
            out.content, None,
            "Gemma 4 pipe-delimited tool call must still be suppressed"
        );
        // After exit, content resumes normally.
        assert_eq!(
            f.feed("answer").content.as_deref(),
            Some("answer"),
            "content after Gemma 4 tool call must be emitted"
        );
    }

    #[test]
    fn hermes_tool_call_does_not_enter_gemma4_state() {
        // Hermes `<tool_call>` must map to `EnterToolCall` just like Gemma 4's
        // `<|tool_call>`. Both put the filter into ToolCall state, so content
        // before the open tag and after the close tag must be emitted while the
        // inner payload is suppressed.
        let mut f = StreamFilter::new();
        let out = f.feed("before<tool_call>payload</tool_call>after");
        // drain_buffer processes the whole fragment in one pass: emits "before",
        // suppresses "payload", then resumes and emits "after" — all merged into
        // a single content string by the accumulating loop in drain_buffer.
        let content = out.content.as_deref().unwrap_or("");
        assert!(
            content.contains("before"),
            "text before <tool_call> must appear in delta.content; got: {content:?}"
        );
        assert!(
            content.contains("after"),
            "text after </tool_call> must appear in delta.content; got: {content:?}"
        );
        assert!(
            !content.contains("<tool_call>"),
            "open tag must not appear in delta.content; got: {content:?}"
        );
        assert!(
            !content.contains("payload"),
            "tool-call payload must not appear in delta.content; got: {content:?}"
        );
        assert!(
            !content.contains("</tool_call>"),
            "close tag must not appear in delta.content; got: {content:?}"
        );
    }

    #[test]
    fn hermes_content_after_tool_call_emitted_in_same_fragment() {
        // If the model emits content before AND after a tool call in one large
        // fragment, both pre and post text must appear in delta.content while
        // the tool-call block is suppressed.
        let mut f = StreamFilter::new();
        let out = f.feed(r#"Here is the result.<tool_call>{"name": "fn", "arguments": {}}</tool_call>Any follow-up."#);
        let content = out.content.as_deref().unwrap_or("");
        assert!(
            content.contains("Here is the result."),
            "preamble must appear in content; got: {content:?}"
        );
        assert!(
            content.contains("Any follow-up."),
            "text after tool call must appear in content; got: {content:?}"
        );
        assert!(
            !content.contains("<tool_call>"),
            "open tag must not appear in content; got: {content:?}"
        );
        assert!(
            !content.contains("</tool_call>"),
            "close tag must not appear in content; got: {content:?}"
        );
    }

    // -- Mistral Nemo `[TOOL_CALLS]` suppression --

    #[test]
    fn mistral_nemo_tool_calls_marker_suppressed() {
        // Mistral Nemo emits `[TOOL_CALLS] [{"name": ...}]`. Everything from
        // the `[TOOL_CALLS]` marker onwards is tool-call payload and must not
        // appear in delta.content.
        let mut f = StreamFilter::new();
        let out = f.feed(r#"[TOOL_CALLS] [{"name": "search", "arguments": {"query": "rust"}}]"#);
        assert_eq!(
            out.content, None,
            "Mistral Nemo [TOOL_CALLS] payload must not appear in delta.content"
        );
    }

    #[test]
    fn mistral_nemo_partial_marker_at_boundary_held() {
        // Fragment ends with `[TOOL_` — a prefix of `[TOOL_CALLS]`. The filter
        // must hold it back until more text arrives to resolve the ambiguity.
        let mut f = StreamFilter::new();
        let out1 = f.feed("preamble [TOOL_");
        assert_eq!(
            out1.content.as_deref(),
            Some("preamble "),
            "text before partial Mistral marker must be emitted"
        );
        // Complete the marker
        let out2 = f.feed(r#"CALLS] [{"name": "fn", "arguments": {}}]"#);
        assert_eq!(
            out2.content, None,
            "content after Mistral Nemo marker must be suppressed"
        );
    }

    #[test]
    fn mistral_nemo_partial_marker_resolves_to_content() {
        // `[TOOL_` at end of buffer that turns out NOT to be `[TOOL_CALLS]`
        // must be released back to delta.content.
        let mut f = StreamFilter::new();
        let out1 = f.feed("text [TOOL_");
        assert_eq!(
            out1.content.as_deref(),
            Some("text "),
            "prefix before potential Mistral marker must be emitted"
        );
        // Not a Mistral Nemo marker — the held-back partial must be released.
        let out2 = f.feed("something_else");
        assert_eq!(
            out2.content.as_deref(),
            Some("[TOOL_something_else"),
            "partial [TOOL_ that is not [TOOL_CALLS] must be released to content"
        );
    }

    #[test]
    fn mistral_bracket_tool_calls_marker_suppressed() {
        // Newer Mistral-family models (Ministral 2410+, Mistral Small 3,
        // Magistral, Devstral) emit `[TOOL_CALLS]NAME[ARGS]{json}`. Because
        // `[TOOL_CALLS]` is already a one-shot `EnterToolCall` marker, this
        // shape is suppressed by the same delimiter as Mistral Nemo: no
        // dedicated `[ARGS]` delimiter row is needed.
        let mut f = StreamFilter::new();
        let out = f.feed(r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#);
        assert_eq!(
            out.content, None,
            "bracketed Mistral tool-call payload must not appear in delta.content"
        );
    }

    // -- Token-position preservation for parallel tool calls --
    //
    // Upstream mlx-lm PR #1170 (commit aa4f880) fixed `_process_control_tokens`
    // so that matched tokens are re-emitted with `text=""` instead of being
    // silently dropped. Without this, parallel tool calls in streaming mode
    // lose token-position alignment (logprobs indices go out of sync).
    //
    // The equivalent in Rust: `FilterOutput::suppressed_positions` counts how
    // many delimiter matches were consumed in a single `feed()` call. Callers
    // that track per-token positions should emit empty-text placeholder events
    // for each suppressed position.

    #[test]
    fn parallel_tool_calls_delimiter_tokens_have_suppressed_positions() {
        // Two consecutive Hermes tool calls, fed token-by-token. Each delimiter
        // token (`<tool_call>`, `</tool_call>`) must produce exactly one
        // suppressed_positions event while payload tokens produce zero. This
        // verifies the upstream mlx-lm PR #1170 fix: delimiter-matched tokens
        // are re-emitted as empty-text placeholders, not silently dropped.
        //
        // In the Python fix: `popped = [buffered_stream.pop() for _ in tok.match]`
        // followed by `buffered_stream.append(replace(t, text=""))`. Each element
        // of `tok.match` is one matched token ID. Here, each delimiter `feed()`
        // call that matches a complete delimiter contributes one suppressed
        // position, matching the Python per-match-element semantics.
        let tokens: &[(&str, usize)] = &[
            // (fragment, expected_suppressed_positions)
            ("<tool_call>", 1), // EnterToolCall: 1 delimiter match
            (r#"{"name": "a", "arguments": {"x": 1}}"#, 0), // payload: no delimiter
            ("</tool_call>", 1), // ExitToolCall: 1 delimiter match
            ("<tool_call>", 1), // EnterToolCall: 1 delimiter match
            (r#"{"name": "b", "arguments": {"y": 2}}"#, 0), // payload: no delimiter
            ("</tool_call>", 1), // ExitToolCall: 1 delimiter match
        ];
        // Total delimiter matches (suppressed positions from control-token hits)
        let expected_suppressed: usize = tokens.iter().map(|(_, n)| n).sum();

        let mut f = StreamFilter::new();
        let mut actual_suppressed: usize = 0;

        for (tok, expected_sp) in tokens {
            let out = f.feed(tok);
            assert_eq!(
                out.suppressed_positions, *expected_sp,
                "token {tok:?}: expected suppressed_positions == {expected_sp}, \
                 got {}",
                out.suppressed_positions
            );
            actual_suppressed += out.suppressed_positions;
        }

        assert_eq!(
            actual_suppressed, expected_suppressed,
            "total suppressed_positions must match expected delimiter-match count"
        );
    }

    #[test]
    fn parallel_tool_calls_delimiter_matches_have_suppressed_positions() {
        // When a delimiter (`<tool_call>` or `</tool_call>`) is matched, the
        // returned `FilterOutput` must have `suppressed_positions >= 1` to
        // signal the position to callers tracking per-token logprobs.
        let mut f = StreamFilter::new();

        // Open tag: EnterToolCall should suppress 1 position.
        let out_open = f.feed("<tool_call>");
        assert_eq!(
            out_open.suppressed_positions, 1,
            "`<tool_call>` open tag must produce suppressed_positions == 1; \
             got {}",
            out_open.suppressed_positions
        );
        assert_eq!(out_open.content, None, "open tag must not emit content");

        // Payload (inside ToolCall state): no delimiter match, 0 suppressed.
        let out_payload = f.feed(r#"{"name": "fn", "arguments": {}}"#);
        assert_eq!(
            out_payload.suppressed_positions, 0,
            "tool-call payload must not produce suppressed_positions"
        );

        // Close tag: ExitToolCall should suppress 1 position.
        let out_close = f.feed("</tool_call>");
        assert_eq!(
            out_close.suppressed_positions, 1,
            "`</tool_call>` close tag must produce suppressed_positions == 1; \
             got {}",
            out_close.suppressed_positions
        );
        assert_eq!(out_close.content, None, "close tag must not emit content");
    }

    #[test]
    fn parallel_gemma4_tool_calls_delimiter_tokens_have_suppressed_positions() {
        // Same position-preservation test for Gemma 4 pipe-delimited tool calls.
        // Each `<|tool_call>` / `<tool_call|>` delimiter must produce one
        // suppressed_positions event; payload tokens produce zero.
        let tokens: &[(&str, usize)] = &[
            ("<|tool_call>", 1),
            ("call:search{q:<|\"|>rust<|\"|>}", 0),
            ("<tool_call|>", 1),
            ("<|tool_call>", 1),
            ("call:fetch{url:<|\"|>https://example.com<|\"|>}", 0),
            ("<tool_call|>", 1),
        ];
        let expected_suppressed: usize = tokens.iter().map(|(_, n)| n).sum();

        let mut f = StreamFilter::new();
        let mut actual_suppressed: usize = 0;

        for (tok, expected_sp) in tokens {
            let out = f.feed(tok);
            assert_eq!(
                out.suppressed_positions, *expected_sp,
                "token {tok:?}: expected suppressed_positions == {expected_sp}, \
                 got {}",
                out.suppressed_positions
            );
            actual_suppressed += out.suppressed_positions;
        }

        assert_eq!(
            actual_suppressed, expected_suppressed,
            "Gemma 4 parallel tool calls: total suppressed must match delimiter-match count"
        );
    }

    #[test]
    fn content_tokens_have_zero_suppressed_positions() {
        // Regular content tokens must always have suppressed_positions == 0
        // so that callers do not accidentally emit extra empty-text events.
        let mut f = StreamFilter::new();
        let out = f.feed("Hello, world!");
        assert_eq!(
            out.suppressed_positions, 0,
            "plain content must have suppressed_positions == 0"
        );
        assert_eq!(
            out.content.as_deref(),
            Some("Hello, world!"),
            "plain content must pass through unchanged"
        );
    }

    #[test]
    fn thinking_delimiter_also_increments_suppressed_positions() {
        // Thinking delimiters (`<think>`, `</think>`, `<|channel>`, etc.)
        // are also matched-and-drained, so they should also increment
        // suppressed_positions. This ensures position alignment is preserved
        // even when reasoning blocks appear before or between tool calls.
        let mut f = StreamFilter::new();
        let out = f.feed("<think>reasoning</think>");
        // The fragment contains TWO delimiters: <think> and </think>.
        assert_eq!(
            out.suppressed_positions, 2,
            "<think>...</think> must produce suppressed_positions == 2 (one per delimiter match)"
        );
        // No content emitted from the markers; reasoning text goes to `reasoning`.
        assert_eq!(out.content, None);
        assert_eq!(out.reasoning.as_deref(), Some("reasoning"));
    }

    // -- HIGH-1 regression: multi-fragment (multi-token) delimiter counting --
    //
    // When a delimiter straddles multiple `feed()` calls (i.e. the tokenizer
    // emits the delimiter bytes spread across several consecutive tokens),
    // `suppressed_positions` must equal the number of `feed()` calls whose
    // bytes contributed to the matched delimiter — **not** merely 1 per
    // delimiter match.
    //
    // This is the HIGH-1 fix review: the old code did
    // `suppressed_positions += 1` per match, which under-counted for
    // multi-token delimiters. The new code tracks per-fragment byte lengths
    // and counts how many fragments were consumed by each match.

    #[test]
    fn multi_fragment_hermes_close_tag_counts_all_token_positions() {
        // Regression test: `</tool_call>` split across 3 `feed()` calls.
        // Each call corresponds to one token. The close tag consumes 3 token
        // positions, so `suppressed_positions` must be 3 for the call that
        // completes the match (the third feed).
        let mut f = StreamFilter::new();
        // Enter ToolCall state with a single-token open tag.
        let open = f.feed("<tool_call>");
        assert_eq!(
            open.suppressed_positions, 1,
            "single-token open tag == 1 position"
        );
        // Feed payload: no delimiter, 0 suppressed.
        f.feed(r#"{"name":"fn"}"#);
        // Split close tag across 3 tokens: "</", "tool", "_call>"
        let frag1 = f.feed("</");
        assert_eq!(
            frag1.suppressed_positions, 0,
            "partial close tag must not fire yet"
        );
        let frag2 = f.feed("tool");
        assert_eq!(frag2.suppressed_positions, 0, "still partial close tag");
        let frag3 = f.feed("_call>");
        // When the 3rd fragment completes the close tag, 3 feed() calls'
        // bytes were consumed: suppressed_positions must be 3.
        assert_eq!(
            frag3.suppressed_positions, 3,
            "3-fragment `</tool_call>` must produce suppressed_positions == 3; \
             got {}",
            frag3.suppressed_positions
        );
    }

    #[test]
    fn multi_fragment_gemma4_close_tag_two_tokens_counts_two_positions() {
        // Regression test: Gemma 4 `<tool_call|>` (12 bytes) split across 2
        // `feed()` calls — mimicking a tokenizer that emits `<tool_call` and
        // `|>` as two separate token IDs.
        let mut f = StreamFilter::new();
        // Enter ToolCall state.
        f.feed("<|tool_call>");
        // Feed payload.
        f.feed("call:fn{}");
        // Split `<tool_call|>` across two tokens.
        let part1 = f.feed("<tool_call");
        assert_eq!(
            part1.suppressed_positions, 0,
            "partial `<tool_call` must not fire"
        );
        let part2 = f.feed("|>");
        assert_eq!(
            part2.suppressed_positions, 2,
            "2-fragment `<tool_call|>` must produce suppressed_positions == 2; \
             got {}",
            part2.suppressed_positions
        );
    }

    #[test]
    fn single_token_delimiter_still_counts_one_position() {
        // Sanity check: a delimiter delivered in a single `feed()` call must
        // still count as exactly 1 suppressed position (no regression for the
        // common case).
        let mut f = StreamFilter::new();
        let out = f.feed("</tool>");
        // `</tool>` is not a registered delimiter, so no suppression.
        assert_eq!(out.suppressed_positions, 0);

        let mut f2 = StreamFilter::new();
        let out2 = f2.feed("<tool_call>");
        assert_eq!(
            out2.suppressed_positions, 1,
            "single-token `<tool_call>` must produce suppressed_positions == 1"
        );
        let out3 = f2.feed("</tool_call>");
        assert_eq!(
            out3.suppressed_positions, 1,
            "single-token `</tool_call>` must produce suppressed_positions == 1"
        );
    }

    #[test]
    fn multi_fragment_hermes_close_three_feeds_suppressed_equals_three() {
        // Explicit assertion matching the PR review example:
        // feed("</") + feed("tool") + feed("_call>") → suppressed_positions == 3
        // on the final feed that completes the match.
        let mut f = StreamFilter::new();
        f.feed("<tool_call>");
        f.feed("payload");
        let r1 = f.feed("</");
        assert_eq!(r1.suppressed_positions, 0);
        let r2 = f.feed("tool");
        assert_eq!(r2.suppressed_positions, 0);
        let r3 = f.feed("_call>");
        assert_eq!(
            r3.suppressed_positions, 3,
            "feed(`</`) + feed(`tool`) + feed(`_call>`) must give suppressed_positions == 3; \
             got {}",
            r3.suppressed_positions
        );
    }

    // -- Kimi K2 sectioned tool-call markers --

    #[test]
    fn kimi_k2_tool_call_section_suppressed_in_one_fragment() {
        // A full Kimi K2 section fed as a single fragment: the section body
        // (including the per-call wrappers and JSON payload) must not leak
        // into delta.content.
        let mut f = StreamFilter::new();
        let out = f.feed(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0\
             <|tool_call_argument_begin|>{\"location\": \"Paris\"}<|tool_call_end|>\
             <|tool_calls_section_end|>",
        );
        assert_eq!(
            out.content, None,
            "Kimi K2 tool-call section markup must not appear in delta.content"
        );
        assert_eq!(out.reasoning, None);
    }

    #[test]
    fn kimi_k2_content_before_section_emitted() {
        let mut f = StreamFilter::new();
        let out = f.feed(
            "Let me check the weather.<|tool_calls_section_begin|><|tool_call_begin|>\
             functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"Tokyo\"}\
             <|tool_call_end|><|tool_calls_section_end|>",
        );
        assert_eq!(
            out.content.as_deref(),
            Some("Let me check the weather."),
            "only the preamble text must appear in delta.content"
        );
        assert!(
            !out.content
                .as_deref()
                .unwrap_or("")
                .contains("<|tool_calls_section_begin|>")
        );
    }

    #[test]
    fn kimi_k2_section_markers_split_across_fragment_boundaries() {
        // The long `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>`
        // markers arrive split across multiple decode fragments (as a real
        // tokenizer would emit them token-by-token). The filter must buffer
        // partial matches and suppress the whole tool-call body regardless of
        // how the marker bytes are chunked.
        let mut f = StreamFilter::new();
        let fragments = [
            "Sure, let me check.",
            "<|tool_calls",
            "_section_begin|>",
            "<|tool_call_begin|>functions.get_",
            "weather:0<|tool_call_argument_begin|>",
            "{\"location\": \"Se",
            "oul\"}",
            "<|tool_call_end|>",
            "<|tool_calls_section",
            "_end|>",
            "Done.",
        ];

        let mut total_content = String::new();
        for frag in &fragments {
            if let Some(c) = f.feed(frag).content {
                total_content.push_str(&c);
            }
        }
        let flushed = f.flush();
        if let Some(c) = flushed.content {
            total_content.push_str(&c);
        }

        assert_eq!(
            total_content, "Sure, let me check.Done.",
            "only the pre- and post-section text must survive; the tool-call \
             body must be fully suppressed even when the section markers \
             straddle fragment boundaries; got: {total_content:?}"
        );
        assert!(!total_content.contains("tool_calls_section"));
        assert!(!total_content.contains("Seoul"));
    }

    #[test]
    fn kimi_k2_tool_call_not_confused_with_hermes_or_gemma4() {
        // Kimi K2's `<|tool_calls_section_begin|>` must not be mistaken for
        // Gemma 4's `<|tool_call>` open tag (both share the `<|tool_cal` prefix)
        // and must not leave the filter in a state where a subsequent Hermes
        // `<tool_call>` fails to enter ToolCall state.
        let mut f = StreamFilter::new();
        let out = f.feed("<|tool_calls_section_begin|>body<|tool_calls_section_end|>");
        assert_eq!(out.content, None);
        assert_eq!(
            f.feed("after").content.as_deref(),
            Some("after"),
            "content after the Kimi K2 section must resume normally"
        );
    }

    #[test]
    fn kimi_k2_multi_call_section_all_calls_suppressed() {
        // Regression guard for #766. `<|tool_call_end|>` is Kimi K2's
        // PER-CALL terminator, appearing once per call inside the section —
        // it is intentionally NOT registered as a global `ExitToolCall`
        // delimiter (see the pythonic entry's comment on `CHAT_DELIMITERS`
        // for why). Feed a two-call section split right after the first
        // `<|tool_call_end|>` so the second call's body arrives in its own
        // fragment: if `<|tool_call_end|>` incorrectly triggered an
        // `ExitToolCall` transition, this second fragment would leak into
        // `delta.content`.
        let mut f = StreamFilter::new();
        let first = "<|tool_calls_section_begin|>\
                      <|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>\
                      {\"query\": \"rust\"}<|tool_call_end|>";
        let second = "<|tool_call_begin|>functions.calc:1<|tool_call_argument_begin|>\
                       {\"expr\": \"2+2\"}<|tool_call_end|><|tool_calls_section_end|>";

        let out1 = f.feed(first);
        assert_eq!(
            out1.content, None,
            "first Kimi K2 call body must not leak into delta.content"
        );

        let out2 = f.feed(second);
        assert_eq!(
            out2.content, None,
            "second Kimi K2 call body must not leak into delta.content after \
             the first call's <|tool_call_end|> terminator"
        );

        let flushed = f.flush();
        assert_eq!(flushed.content, None);
    }

    // -- Pythonic tool-call markers --

    #[test]
    fn pythonic_tool_call_suppressed() {
        let mut f = StreamFilter::new();
        let out = f.feed(r#"<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>"#);
        assert_eq!(
            out.content, None,
            "pythonic tool-call payload must not leak into delta.content"
        );
        let flushed = f.flush();
        assert_eq!(flushed.content, None);
    }

    #[test]
    fn pythonic_content_before_call_emitted() {
        let mut f = StreamFilter::new();
        let out = f.feed(
            r#"Let me check.<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>"#,
        );
        assert_eq!(
            out.content.as_deref(),
            Some("Let me check."),
            "only the preamble text must appear in delta.content"
        );
    }

    #[test]
    fn pythonic_start_marker_not_confused_with_gemma4_or_kimi_k2() {
        // `<|tool_call_start|>` shares the `<|tool_call` prefix with both
        // Gemma 4's `<|tool_call>` and Kimi K2's `<|tool_calls_section_begin|>`,
        // but diverges immediately after (`_start` vs `>` vs `s_section...`),
        // so none of the three should partial-match another.
        let mut f = StreamFilter::new();
        let out = f.feed("<|tool_call_start|>[fn(x=1)]<|tool_call_end|>");
        assert_eq!(out.content, None);
    }
}
