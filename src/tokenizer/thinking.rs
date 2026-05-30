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

//! Multi-token think / tool-call sequence detection.
//!
//! Mirrors the upstream `mlx_lm.tokenizer_utils._infer_thinking()` helper and
//! the `find_think_*` / `rfind_think_*` lookup methods on `TokenizerWrapper`
//! introduced in mlx-lm PRs #1114 and #1167.
//!
//! ## Why multi-token?
//!
//! Most reasoning families pair their `<think>` block with a single added
//! token in the tokenizer's vocab (`<think>` / `</think>`,
//! `<longcat_think>` / `</longcat_think>`).  Some newer families — Gemma 4
//! and any future model that adopts a `<|channel>thought` style marker —
//! split the marker across multiple subword pieces.  The marker becomes a
//! token *sequence* rather than a single id, and downstream consumers
//! (stream filter, thinking-budget tracker, chat-template defaulting logic)
//! need to look up that whole sequence.
//!
//! ## API mapping (Python → Rust)
//!
//! | Python (`mlx_lm.tokenizer_utils`) | Rust (`MlxcelTokenizer` / `ThinkingMarkers`) |
//! |-----------------------------------|----------------------------------------------|
//! | `_infer_thinking(tokenizer)`      | [`MlxcelTokenizer::infer_thinking_markers`]   |
//! | `wrapper._think_start_tokens`     | [`ThinkingMarkers::think_start_tokens`]       |
//! | `wrapper._think_end_tokens`       | [`ThinkingMarkers::think_end_tokens`]         |
//! | `wrapper._tool_call_start_tokens` | [`ThinkingMarkers::tool_call_start_tokens`]   |
//! | `wrapper._tool_call_end_tokens`   | [`ThinkingMarkers::tool_call_end_tokens`]     |
//! | `wrapper.has_thinking`            | [`ThinkingMarkers::has_thinking`]             |
//! | `wrapper.find_think_start`        | [`ThinkingMarkers::find_think_start`]         |
//! | `wrapper.rfind_think_start`       | [`ThinkingMarkers::rfind_think_start`]        |
//! | `wrapper.find_think_end`          | [`ThinkingMarkers::find_think_end`]           |
//! | `wrapper.rfind_think_end`         | [`ThinkingMarkers::rfind_think_end`]          |
//!
//! Used by: `tokenizer::MlxcelTokenizer`, `server::chat_template`,
//! `server::thinking_budget`, `server::tool_calls::stream_filter`.

/// Multi-token think and tool-call markers resolved from a tokenizer's vocab.
///
/// Each field holds the **token-id sequence** that materializes the marker in
/// the model's tokenizer.  Single-token markers are length-1 sequences; the
/// `<|channel>thought` Gemma 4 path produces a multi-token sequence.  An
/// empty sequence (or two `None` halves) means the model has no marker of
/// that kind.
///
/// The string forms (`think_start`, `think_end`, …) are kept alongside the
/// id sequences so callers that operate on raw text (e.g. the stream filter
/// or chat-template `enable_thinking` defaulting logic) do not need a second
/// round-trip through the tokenizer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThinkingMarkers {
    /// Literal text of the open think marker (`"<think>"`,
    /// `"<longcat_think>"`, `"<|channel>thought"` …).
    pub think_start: Option<String>,
    /// Literal text of the close think marker (`"</think>"`,
    /// `"</longcat_think>"`, `"<channel|>"` …).
    pub think_end: Option<String>,
    /// Token-id sequence that encodes [`Self::think_start`].
    ///
    /// `None` when the model is not a thinking model.  Length-1 for the
    /// single-token families; longer for Gemma 4 / `<|channel>thought`.
    pub think_start_tokens: Option<Vec<u32>>,
    /// Token-id sequence that encodes [`Self::think_end`]. See
    /// [`Self::think_start_tokens`] for shape semantics.
    pub think_end_tokens: Option<Vec<u32>>,
    /// Literal text of the open tool-call marker (e.g. `"<tool_call>"`,
    /// `"<|tool_call>"`).  Only populated when [`Self::tool_call_start_tokens`]
    /// could be resolved from the same tokenizer.
    pub tool_call_start: Option<String>,
    /// Literal text of the close tool-call marker.
    pub tool_call_end: Option<String>,
    /// Token-id sequence that encodes [`Self::tool_call_start`].
    pub tool_call_start_tokens: Option<Vec<u32>>,
    /// Token-id sequence that encodes [`Self::tool_call_end`].
    pub tool_call_end_tokens: Option<Vec<u32>>,
}

impl ThinkingMarkers {
    /// `true` when the tokenizer exposes a recognized think open/close pair.
    ///
    /// Mirrors upstream's `TokenizerWrapper.has_thinking` property.  The
    /// return value is the canonical default for the chat-template
    /// `enable_thinking` Jinja kwarg when the caller does not set it
    /// explicitly (matching the upstream
    /// `apply_chat_template(... enable_thinking=self.has_thinking ...)`
    /// behavior added in PR #1114).
    pub fn has_thinking(&self) -> bool {
        self.think_start.is_some()
    }

    /// `true` when the tokenizer exposes a recognized tool-call open/close
    /// pair as token sequences.
    pub fn has_tool_calling(&self) -> bool {
        self.tool_call_start.is_some()
    }

    /// Find the first occurrence of the resolved think-start token sequence.
    ///
    /// Mirrors upstream `TokenizerWrapper.find_think_start`. Returns `None`
    /// when this tokenizer has no recognized think-start marker.
    pub fn find_think_start(
        &self,
        tokens: &[u32],
        start: Option<usize>,
        end: Option<usize>,
    ) -> Option<usize> {
        find_marker(tokens, self.think_start_tokens.as_deref(), start, end)
    }

    /// Find the last occurrence of the resolved think-start token sequence.
    ///
    /// Mirrors upstream `TokenizerWrapper.rfind_think_start`. Returns `None`
    /// when this tokenizer has no recognized think-start marker.
    pub fn rfind_think_start(
        &self,
        tokens: &[u32],
        start: Option<usize>,
        end: Option<usize>,
    ) -> Option<usize> {
        rfind_marker(tokens, self.think_start_tokens.as_deref(), start, end)
    }

    /// Find the first occurrence of the resolved think-end token sequence.
    ///
    /// Mirrors upstream `TokenizerWrapper.find_think_end`. Returns `None`
    /// when this tokenizer has no recognized think-end marker.
    pub fn find_think_end(
        &self,
        tokens: &[u32],
        start: Option<usize>,
        end: Option<usize>,
    ) -> Option<usize> {
        find_marker(tokens, self.think_end_tokens.as_deref(), start, end)
    }

    /// Find the last occurrence of the resolved think-end token sequence.
    ///
    /// Mirrors upstream `TokenizerWrapper.rfind_think_end`. Returns `None`
    /// when this tokenizer has no recognized think-end marker.
    pub fn rfind_think_end(
        &self,
        tokens: &[u32],
        start: Option<usize>,
        end: Option<usize>,
    ) -> Option<usize> {
        rfind_marker(tokens, self.think_end_tokens.as_deref(), start, end)
    }

    /// Find the first occurrence of the resolved tool-call start token
    /// sequence. Returns `None` when no token sequence is registered.
    pub fn find_tool_call_start(
        &self,
        tokens: &[u32],
        start: Option<usize>,
        end: Option<usize>,
    ) -> Option<usize> {
        find_marker(tokens, self.tool_call_start_tokens.as_deref(), start, end)
    }

    /// Find the last occurrence of the resolved tool-call start token
    /// sequence. Returns `None` when no token sequence is registered.
    pub fn rfind_tool_call_start(
        &self,
        tokens: &[u32],
        start: Option<usize>,
        end: Option<usize>,
    ) -> Option<usize> {
        rfind_marker(tokens, self.tool_call_start_tokens.as_deref(), start, end)
    }

    /// Find the first occurrence of the resolved tool-call end token
    /// sequence. Returns `None` when no token sequence is registered.
    pub fn find_tool_call_end(
        &self,
        tokens: &[u32],
        start: Option<usize>,
        end: Option<usize>,
    ) -> Option<usize> {
        find_marker(tokens, self.tool_call_end_tokens.as_deref(), start, end)
    }

    /// Find the last occurrence of the resolved tool-call end token
    /// sequence. Returns `None` when no token sequence is registered.
    pub fn rfind_tool_call_end(
        &self,
        tokens: &[u32],
        start: Option<usize>,
        end: Option<usize>,
    ) -> Option<usize> {
        rfind_marker(tokens, self.tool_call_end_tokens.as_deref(), start, end)
    }
}

fn find_marker(
    haystack: &[u32],
    needle: Option<&[u32]>,
    start: Option<usize>,
    end: Option<usize>,
) -> Option<usize> {
    needle.and_then(|needle| find_subseq(haystack, needle, start, end))
}

fn rfind_marker(
    haystack: &[u32],
    needle: Option<&[u32]>,
    start: Option<usize>,
    end: Option<usize>,
) -> Option<usize> {
    needle.and_then(|needle| rfind_subseq(haystack, needle, start, end))
}

/// Find the first index `i` in `haystack` where `needle` matches as a
/// contiguous subsequence, restricted to the half-open range `[start, end)`.
///
/// `start` defaults to `0`, `end` defaults to `haystack.len()`.  Returns
/// `None` when `needle` is empty (matching the upstream `_find` helper which
/// has no callers passing an empty sequence) or when no match is found.
///
/// Mirrors the forward direction of upstream's
/// `TokenizerWrapper._find(tokens, sequence)`.
///
/// Used by: `MlxcelTokenizer::find_think_start`,
/// `MlxcelTokenizer::find_think_end`, and any future stream-filter caller
/// that needs to scan generated tokens for a multi-token marker.
pub fn find_subseq(
    haystack: &[u32],
    needle: &[u32],
    start: Option<usize>,
    end: Option<usize>,
) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    let start = start.unwrap_or(0);
    let end = end.unwrap_or(haystack.len()).min(haystack.len());
    if needle.len() > end.saturating_sub(start) {
        return None;
    }
    let last = end - needle.len();
    for i in start..=last {
        if haystack[i..i + needle.len()] == *needle {
            return Some(i);
        }
    }
    None
}

/// Same as [`find_subseq`] but searches in reverse (returns the index of the
/// last occurrence).
///
/// Mirrors `TokenizerWrapper._find(..., reverse=True)` — i.e. the back-end
/// of `rfind_think_start` / `rfind_think_end` upstream.
pub fn rfind_subseq(
    haystack: &[u32],
    needle: &[u32],
    start: Option<usize>,
    end: Option<usize>,
) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    let start = start.unwrap_or(0);
    let end = end.unwrap_or(haystack.len()).min(haystack.len());
    if needle.len() > end.saturating_sub(start) {
        return None;
    }
    let last = end - needle.len();
    let mut i = last;
    loop {
        if i < start {
            return None;
        }
        if haystack[i..i + needle.len()] == *needle {
            return Some(i);
        }
        if i == start {
            return None;
        }
        i -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- find_subseq / rfind_subseq -----------------------------------------

    #[test]
    fn find_subseq_simple_match() {
        let hay = [1, 2, 3, 4, 5];
        assert_eq!(find_subseq(&hay, &[3, 4], None, None), Some(2));
    }

    #[test]
    fn find_subseq_no_match() {
        let hay = [1, 2, 3, 4, 5];
        assert_eq!(find_subseq(&hay, &[6], None, None), None);
    }

    #[test]
    fn find_subseq_empty_needle_returns_none() {
        // Matches upstream behavior: callers never pass an empty sequence
        // (would mean "no marker"), and an empty match would degenerate the
        // streaming filter logic.
        let hay = [1, 2, 3];
        assert_eq!(find_subseq(&hay, &[], None, None), None);
    }

    #[test]
    fn find_subseq_needle_longer_than_haystack() {
        let hay = [1, 2];
        assert_eq!(find_subseq(&hay, &[1, 2, 3], None, None), None);
    }

    #[test]
    fn find_subseq_match_at_start() {
        let hay = [9, 8, 7];
        assert_eq!(find_subseq(&hay, &[9, 8], None, None), Some(0));
    }

    #[test]
    fn find_subseq_match_at_end() {
        let hay = [1, 2, 3, 4];
        assert_eq!(find_subseq(&hay, &[3, 4], None, None), Some(2));
    }

    #[test]
    fn find_subseq_first_of_multiple() {
        let hay = [1, 2, 3, 1, 2, 3];
        assert_eq!(find_subseq(&hay, &[1, 2], None, None), Some(0));
    }

    #[test]
    fn find_subseq_with_start_skip() {
        // Skipping past the first occurrence finds the second.
        let hay = [1, 2, 3, 1, 2, 3];
        assert_eq!(find_subseq(&hay, &[1, 2], Some(2), None), Some(3));
    }

    #[test]
    fn find_subseq_with_end_clamp() {
        // The end bound excludes the second occurrence.
        let hay = [1, 2, 3, 1, 2, 3];
        assert_eq!(find_subseq(&hay, &[1, 2], None, Some(3)), Some(0));
        assert_eq!(find_subseq(&hay, &[1, 2], Some(2), Some(3)), None);
    }

    #[test]
    fn rfind_subseq_returns_last_match() {
        let hay = [1, 2, 3, 1, 2, 3];
        assert_eq!(rfind_subseq(&hay, &[1, 2], None, None), Some(3));
    }

    #[test]
    fn rfind_subseq_no_match_returns_none() {
        let hay = [1, 2, 3];
        assert_eq!(rfind_subseq(&hay, &[9], None, None), None);
    }

    #[test]
    fn rfind_subseq_with_end_bound_excludes_tail() {
        let hay = [1, 2, 3, 1, 2, 3];
        // end=4 → consider haystack[..4] = [1,2,3,1]; last [1,2] at idx 0.
        assert_eq!(rfind_subseq(&hay, &[1, 2], None, Some(4)), Some(0));
    }

    #[test]
    fn rfind_subseq_multitoken_match() {
        // Multi-token needle used in Gemma 4 / <|channel>thought style.
        let hay = [10, 20, 30, 40, 50, 30, 40];
        assert_eq!(rfind_subseq(&hay, &[30, 40], None, None), Some(5));
    }

    #[test]
    fn rfind_subseq_empty_needle_returns_none() {
        let hay = [1, 2, 3];
        assert_eq!(rfind_subseq(&hay, &[], None, None), None);
    }

    #[test]
    fn rfind_subseq_with_start_clamp() {
        // start=4 forces the search to skip the earlier occurrence.
        let hay = [1, 2, 3, 1, 2, 3];
        assert_eq!(rfind_subseq(&hay, &[1, 2], Some(4), None), None);
        assert_eq!(rfind_subseq(&hay, &[1, 2], Some(3), None), Some(3));
    }

    // -- ThinkingMarkers ---------------------------------------------------

    #[test]
    fn has_thinking_reflects_start_string() {
        let mut m = ThinkingMarkers::default();
        assert!(!m.has_thinking());
        m.think_start = Some("<think>".to_string());
        assert!(m.has_thinking());
    }

    #[test]
    fn has_tool_calling_reflects_start_string() {
        let mut m = ThinkingMarkers::default();
        assert!(!m.has_tool_calling());
        m.tool_call_start = Some("<tool_call>".to_string());
        assert!(m.has_tool_calling());
    }

    #[test]
    fn marker_find_methods_delegate_to_registered_sequences() {
        let m = ThinkingMarkers {
            think_start_tokens: Some(vec![10, 11]),
            think_end_tokens: Some(vec![20]),
            tool_call_start_tokens: Some(vec![30, 31]),
            tool_call_end_tokens: Some(vec![40, 41]),
            ..ThinkingMarkers::default()
        };
        let hay = [0, 10, 11, 1, 20, 10, 11, 30, 31, 2, 40, 41, 30, 31];

        assert_eq!(m.find_think_start(&hay, None, None), Some(1));
        assert_eq!(m.rfind_think_start(&hay, None, None), Some(5));
        assert_eq!(m.find_think_end(&hay, None, None), Some(4));
        assert_eq!(m.rfind_think_end(&hay, None, None), Some(4));

        assert_eq!(m.find_tool_call_start(&hay, None, None), Some(7));
        assert_eq!(m.rfind_tool_call_start(&hay, None, None), Some(12));
        assert_eq!(m.find_tool_call_end(&hay, None, None), Some(10));
        assert_eq!(m.rfind_tool_call_end(&hay, None, None), Some(10));
    }

    #[test]
    fn marker_find_methods_return_none_for_missing_sequences() {
        let m = ThinkingMarkers::default();
        let hay = [1, 2, 3];

        assert_eq!(m.find_think_start(&hay, None, None), None);
        assert_eq!(m.rfind_think_start(&hay, None, None), None);
        assert_eq!(m.find_think_end(&hay, None, None), None);
        assert_eq!(m.rfind_think_end(&hay, None, None), None);
        assert_eq!(m.find_tool_call_start(&hay, None, None), None);
        assert_eq!(m.rfind_tool_call_start(&hay, None, None), None);
        assert_eq!(m.find_tool_call_end(&hay, None, None), None);
        assert_eq!(m.rfind_tool_call_end(&hay, None, None), None);
    }
}
