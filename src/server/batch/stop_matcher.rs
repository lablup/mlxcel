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

//! Streaming-safe stop-string matcher (issue #449 M3 Stage 2d).
//!
//! [`StopMatcher`] turns the post-hoc truncation semantics of
//! [`apply_stop_sequences`](crate::server::anthropic_translator::apply_stop_sequences)
//! into an incremental, streaming-safe form: text arrives one decoded piece at a
//! time, and the matcher decides how much is safe to emit *now* versus how much
//! must be held back because it could still turn out to be the start of a stop
//! string at the next piece.
//!
//! A stop string can straddle token boundaries (e.g. `"STOP"` arriving as
//! `"ST"` then `"OP"`), so any suffix of the text seen so far that is a proper
//! prefix of some stop string is ambiguous and is withheld until the next piece
//! resolves it. When a stop string fully matches, the matcher reports a stop and
//! the emitted text ends just before the match, so the stop string itself and
//! everything after it never reach the client. This matches `apply_stop_sequences`
//! (earliest match wins; the stop string is excluded), proven by an equivalence
//! test below.
//!
//! The matcher is pure (no IREE / device state), so it lives outside the
//! `xla-iree` cfg gate and its unit tests run in an ordinary `cargo test`. The
//! OpenXLA serve worker ([`XlaServeWorker`](super::xla_worker)) drives it; nothing
//! else does today, but it is backend-neutral by construction.

/// The result of feeding one decoded piece to a [`StopMatcher`].
pub(crate) struct StopChunk {
    /// Text that is safe to emit to the client now. May be empty when the whole
    /// piece is held back as a potential stop-string prefix.
    pub emit: String,
    /// `true` once a stop string has fully matched. The caller should emit
    /// [`StopChunk::emit`] (the text up to the match) and then finalize the
    /// request with a `stop` finish reason; the matcher will produce nothing
    /// further.
    pub stopped: bool,
}

/// Incremental stop-string matcher for one in-flight request.
///
/// Construct with the request's stop strings; feed each decoded piece through
/// [`push`](StopMatcher::push); on natural end of generation (EOS / length) call
/// [`flush`](StopMatcher::flush) to release any held-back tail (which, by
/// definition, did not complete a stop string).
pub(crate) struct StopMatcher {
    /// Non-empty stop strings. Empty stop strings are dropped at construction
    /// (they would match everywhere and carry no meaning), mirroring
    /// `apply_stop_sequences`.
    stops: Vec<String>,
    /// Decoded text received but not yet emitted: the ambiguous tail that could
    /// still become a stop-string prefix. Always empty when `stops` is empty.
    pending: String,
    /// Running total of bytes emitted to the client. The emitted text is always a
    /// prefix of the full decoded text, so the worker can truncate its decode
    /// buffer to this length to obtain the (stop-truncated) result text.
    emitted_len: usize,
}

impl StopMatcher {
    /// Build a matcher from the request's stop strings, dropping empty ones.
    pub(crate) fn new<I, S>(stops: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stops: Vec<String> = stops
            .into_iter()
            .map(Into::into)
            .filter(|s| !s.is_empty())
            .collect();
        Self {
            stops,
            pending: String::new(),
            emitted_len: 0,
        }
    }

    /// Whether any stop string is configured. When `false`, [`push`](Self::push)
    /// is a pass-through and the request behaves exactly as it did before stop
    /// strings were enforced.
    pub(crate) fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// Total bytes emitted so far. Because emitted text is a prefix of the full
    /// decoded text, the worker truncates its decode buffer to this length to get
    /// the result text after a stop-string match.
    pub(crate) fn emitted_len(&self) -> usize {
        self.emitted_len
    }

    /// Feed one newly decoded piece. Returns the text to emit now and whether a
    /// stop string matched.
    pub(crate) fn push(&mut self, piece: &str) -> StopChunk {
        // No stop strings: emit verbatim, nothing is ever held.
        if self.stops.is_empty() {
            self.emitted_len += piece.len();
            return StopChunk {
                emit: piece.to_string(),
                stopped: false,
            };
        }

        self.pending.push_str(piece);

        // Earliest full match across all stop strings wins (same rule as
        // `apply_stop_sequences`). Everything from the match onward is dropped.
        if let Some(idx) = self.earliest_full_match() {
            let emit = self.pending[..idx].to_string();
            self.emitted_len += emit.len();
            self.pending.clear();
            return StopChunk {
                emit,
                stopped: true,
            };
        }

        // No full match: hold back the longest suffix that is a proper prefix of
        // some stop string (it might complete on the next piece), emit the rest.
        let hold = self.longest_partial_suffix();
        let cut = self.pending.len() - hold;
        let emit = self.pending[..cut].to_string();
        self.emitted_len += emit.len();
        self.pending.drain(..cut);
        StopChunk {
            emit,
            stopped: false,
        }
    }

    /// Release any held-back tail at the natural end of generation. Because the
    /// tail never completed a stop string, it is real output and must be emitted.
    pub(crate) fn flush(&mut self) -> String {
        let out = std::mem::take(&mut self.pending);
        self.emitted_len += out.len();
        out
    }

    /// Byte index of the earliest full stop-string occurrence in `pending`, if
    /// any. Ties (two stops matching at the same index) resolve to that shared
    /// index, which is all the caller needs for truncation.
    fn earliest_full_match(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for s in &self.stops {
            if let Some(i) = self.pending.find(s.as_str()) {
                best = Some(best.map_or(i, |b| b.min(i)));
            }
        }
        best
    }

    /// Length (in bytes) of the longest suffix of `pending` that equals a
    /// non-empty proper prefix of some stop string. Such a suffix is ambiguous:
    /// the following piece could complete the stop string, so it must be held.
    ///
    /// Returns `0` when no suffix is a stop-string prefix. Callers invoke this
    /// only after [`earliest_full_match`](Self::earliest_full_match) returns
    /// `None`, so a full stop string is never present in `pending` and the
    /// considered prefixes are strictly shorter than their stop string.
    fn longest_partial_suffix(&self) -> usize {
        let mut max_hold = 0;
        for s in &self.stops {
            // Candidate prefix lengths are char boundaries of `s` below its full
            // length; check longest first and keep the first (largest) match.
            let upper = (s.len() - 1).min(self.pending.len());
            let mut pl = upper;
            while pl > max_hold {
                if s.is_char_boundary(pl) && self.pending.ends_with(&s[..pl]) {
                    max_hold = pl;
                    break;
                }
                pl -= 1;
            }
        }
        max_hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::anthropic_translator::apply_stop_sequences;

    /// Drive a matcher over `pieces` and return everything it emitted (including
    /// the flushed tail when generation was not stopped by a match).
    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()));
        let mut out = String::new();
        let mut stopped = false;
        for p in pieces {
            let chunk = m.push(p);
            out.push_str(&chunk.emit);
            if chunk.stopped {
                stopped = true;
                break;
            }
        }
        if !stopped {
            out.push_str(&m.flush());
        }
        assert_eq!(
            out.len(),
            m.emitted_len(),
            "emitted_len must track total emitted bytes"
        );
        (out, stopped)
    }

    #[test]
    fn no_stops_is_passthrough() {
        let (out, stopped) = run(&[], &["hello ", "world"]);
        assert_eq!(out, "hello world");
        assert!(!stopped);
        let mut m = StopMatcher::new(Vec::<String>::new());
        assert!(!m.is_active());
        assert_eq!(m.push("x").emit, "x");
    }

    #[test]
    fn empty_stop_strings_are_dropped() {
        let m = StopMatcher::new(vec![String::new(), "".to_string()]);
        assert!(!m.is_active());
    }

    #[test]
    fn stop_within_single_piece() {
        let (out, stopped) = run(&["STOP"], &["hello STOP world"]);
        assert_eq!(out, "hello ");
        assert!(stopped);
    }

    #[test]
    fn stop_split_across_pieces() {
        let (out, stopped) = run(&["STOP"], &["hel", "lo ", "ST", "OP", " trailing"]);
        assert_eq!(out, "hello ");
        assert!(stopped);
    }

    #[test]
    fn partial_false_alarm_is_flushed() {
        // "ST" looks like the start of "STOP" but resolves to "STxy".
        let (out, stopped) = run(&["STOP"], &["ST", "xy"]);
        assert_eq!(out, "STxy");
        assert!(!stopped);
    }

    #[test]
    fn held_tail_flushed_on_natural_end() {
        // Ends mid-prefix; no completion, so the tail is real output.
        let (out, stopped) = run(&["STOP"], &["hello ST"]);
        assert_eq!(out, "hello ST");
        assert!(!stopped);
    }

    #[test]
    fn earliest_of_multiple_stops_wins() {
        let (out, stopped) = run(&["world", "STOP"], &["a STOP b world"]);
        assert_eq!(out, "a ");
        assert!(stopped);
    }

    #[test]
    fn stop_at_very_start_emits_nothing() {
        let (out, stopped) = run(&["STOP"], &["STOP rest"]);
        assert_eq!(out, "");
        assert!(stopped);
    }

    #[test]
    fn overlapping_repeats_match_first_occurrence() {
        // "aa" with input "aaa" fed one char at a time matches at index 0.
        let (out, stopped) = run(&["aa"], &["a", "a", "a"]);
        assert_eq!(out, "");
        assert!(stopped);
        assert_eq!(apply_stop_sequences("aaa", Some(&["aa".to_string()])).0, "");
    }

    #[test]
    fn unicode_stop_string() {
        let (out, stopped) = run(&["café"], &["a ca", "fé b"]);
        assert_eq!(out, "a ");
        assert!(stopped);
    }

    #[test]
    fn unicode_partial_does_not_split_codepoint() {
        // Feeding a multibyte char that is a stop prefix then diverging must not
        // panic on a non-boundary slice and must flush the real text.
        let (out, stopped) = run(&["→end"], &["x→", "y"]);
        assert_eq!(out, "x→y");
        assert!(!stopped);
    }

    /// The streamed result must equal `apply_stop_sequences` on the whole text,
    /// for every chunking. This ties the incremental matcher to the established
    /// truncation semantics.
    #[test]
    fn matches_apply_stop_sequences_for_all_chunkings() {
        let cases: &[(&str, &[&str])] = &[
            ("hello STOP world", &["STOP"]),
            ("no match here", &["STOP"]),
            ("end at THE END now", &["THE END"]),
            ("pick earliest: B then A", &["A", "B"]),
            ("aaa", &["aa"]),
            ("café au lait", &["au"]),
            ("trailing prefix ST", &["STOP"]),
            ("", &["STOP"]),
        ];
        for (text, stops) in cases {
            let owned: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
            let expected = apply_stop_sequences(text, Some(&owned)).0;

            // Whole string in one piece.
            let (whole, _) = run(stops, &[text]);
            assert_eq!(whole, expected, "whole-string chunking: {text:?}");

            // Char-by-char.
            let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            let char_refs: Vec<&str> = chars.iter().map(String::as_str).collect();
            let (per_char, _) = run(stops, &char_refs);
            assert_eq!(per_char, expected, "char-by-char chunking: {text:?}");

            // Byte-pair-ish split at each char boundary.
            for split in 1..text.chars().count() {
                let idx: usize = text
                    .char_indices()
                    .nth(split)
                    .map(|(i, _)| i)
                    .unwrap_or(text.len());
                let (a, b) = text.split_at(idx);
                let (two, _) = run(stops, &[a, b]);
                assert_eq!(two, expected, "two-piece split at {split} of {text:?}");
            }
        }
    }
}
