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
//! `xla-iree` cfg gate and its unit tests run in an ordinary `cargo test`. Both
//! serving backends drive it: the MLX [`BatchScheduler`](super::BatchScheduler)
//! through [`SequenceInfo`](super::SequenceInfo) (issue #1466) and the OpenXLA
//! serve worker ([`XlaServeWorker`](super::xla_worker)).

/// The result of feeding one decoded piece to a [`StopMatcher`].
pub(crate) struct StopChunk {
    /// Text that is safe to emit to the client now. May be empty when the whole
    /// piece is held back as a potential stop-string prefix.
    pub emit: String,
    /// The stop string that matched, or `None` while generation continues.
    ///
    /// This carries both the stop signal and the identity of what stopped it, so
    /// the response layer can report b10621's `stopping_word` (issue #1466). It
    /// used to be a bare `stopped: bool`, which could say that a request had
    /// been stopped but not by what, and the string cannot be recovered from the
    /// text afterwards because the match is excluded from it.
    ///
    /// On a tie (two stop strings matching at the same index) this is the first
    /// entry in the request's stop list, the rule `apply_stop_sequences` uses.
    ///
    /// `Some` means the request is over: the caller emits
    /// [`emit`](Self::emit) (the text up to the match) and finalizes with a stop
    /// finish reason. The matcher produces nothing further.
    pub matched: Option<String>,
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
    /// The stop string that ended the request, recorded on the matcher so a
    /// caller that finalizes later (the scheduler builds the `Done` event well
    /// after the matching decode step) can still name it without threading the
    /// value through its own state.
    matched: Option<String>,
}

impl Default for StopMatcher {
    /// An inactive matcher: no stop strings, so [`push`](Self::push) is a
    /// pass-through. This is the state of every request that supplies no `stop`
    /// field, and the construction used by call sites that never had stop
    /// strings (the disaggregated handoff, test fixtures).
    fn default() -> Self {
        Self::new(Vec::<String>::new())
    }
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
            matched: None,
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

    /// The stop string that ended this request, or `None` while no stop string
    /// has matched. Sticky: once set it stays set for the life of the matcher,
    /// so the finalization path can read it long after the matching step.
    pub(crate) fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// Whether a stop string has already matched. Callers use this to classify
    /// the finish reason without re-running the match.
    pub(crate) fn has_matched(&self) -> bool {
        self.matched.is_some()
    }

    /// Drop all in-flight matching state, keeping the request's stop strings.
    ///
    /// Preemptive eviction restarts a sequence's decode from scratch, resetting
    /// its `StreamingDecodeState` and clearing its generated tokens. The
    /// matcher's held tail and emitted-byte count describe the discarded decode,
    /// so they must be reset alongside it or the re-run would be truncated
    /// against a stale offset.
    pub(crate) fn reset(&mut self) {
        self.pending.clear();
        self.emitted_len = 0;
        self.matched = None;
    }

    /// Feed one newly decoded piece. Returns the text to emit now and whether a
    /// stop string matched.
    pub(crate) fn push(&mut self, piece: &str) -> StopChunk {
        // Already stopped: the request is over, so nothing further is emitted.
        // Generation halts on the matching step, so this is defensive only.
        if self.matched.is_some() {
            return StopChunk {
                emit: String::new(),
                matched: self.matched.clone(),
            };
        }

        // No stop strings: emit verbatim, nothing is ever held.
        if self.stops.is_empty() {
            self.emitted_len += piece.len();
            return StopChunk {
                emit: piece.to_string(),
                matched: None,
            };
        }

        self.pending.push_str(piece);

        // Earliest full match across all stop strings wins (same rule as
        // `apply_stop_sequences`). Everything from the match onward is dropped.
        if let Some((idx, which)) = self.earliest_full_match() {
            let emit = self.pending[..idx].to_string();
            let matched = self.stops[which].clone();
            self.emitted_len += emit.len();
            self.pending.clear();
            self.matched = Some(matched.clone());
            return StopChunk {
                emit,
                matched: Some(matched),
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
            matched: None,
        }
    }

    /// Release any held-back tail at the natural end of generation. Because the
    /// tail never completed a stop string, it is real output and must be emitted.
    ///
    /// After a match the held text was already dropped, so this returns the
    /// empty string and the caller emits nothing further.
    pub(crate) fn flush(&mut self) -> String {
        let out = std::mem::take(&mut self.pending);
        self.emitted_len += out.len();
        out
    }

    /// Byte index of the earliest full stop-string occurrence in `pending` and
    /// the index into [`Self::stops`] of the string that produced it, if any.
    ///
    /// Ties (two stops matching at the same index) resolve to the first entry in
    /// the request's stop list, because the comparison is strictly-less-than.
    /// That is exactly what `apply_stop_sequences` does, so the reported
    /// `stopping_word` agrees with the post-hoc truncation path.
    fn earliest_full_match(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (n, s) in self.stops.iter().enumerate() {
            if let Some(i) = self.pending.find(s.as_str())
                && best.map(|(b, _)| i < b).unwrap_or(true)
            {
                best = Some((i, n));
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
    /// the flushed tail when generation was not stopped by a match) together
    /// with the stop string that ended it, if any.
    fn run(stops: &[&str], pieces: &[&str]) -> (String, Option<String>) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()));
        let mut out = String::new();
        let mut matched = None;
        for p in pieces {
            let chunk = m.push(p);
            out.push_str(&chunk.emit);
            if chunk.matched.is_some() {
                matched = chunk.matched;
                break;
            }
        }
        if matched.is_none() {
            out.push_str(&m.flush());
        }
        assert_eq!(
            out.len(),
            m.emitted_len(),
            "emitted_len must track total emitted bytes"
        );
        assert_eq!(
            m.matched().map(str::to_string),
            matched,
            "the matcher must remember the stop string it reported"
        );
        assert_eq!(m.has_matched(), matched.is_some());
        (out, matched)
    }

    #[test]
    fn no_stops_is_passthrough() {
        let (out, matched) = run(&[], &["hello ", "world"]);
        assert_eq!(out, "hello world");
        assert_eq!(matched, None);
        let mut m = StopMatcher::default();
        assert!(!m.is_active());
        assert_eq!(m.push("x").emit, "x");
        assert!(!m.has_matched());
    }

    #[test]
    fn empty_stop_strings_are_dropped() {
        let m = StopMatcher::new(vec![String::new(), "".to_string()]);
        assert!(!m.is_active());
    }

    #[test]
    fn stop_within_single_piece() {
        let (out, matched) = run(&["STOP"], &["hello STOP world"]);
        assert_eq!(out, "hello ");
        assert_eq!(matched.as_deref(), Some("STOP"));
    }

    #[test]
    fn stop_split_across_pieces() {
        let (out, matched) = run(&["STOP"], &["hel", "lo ", "ST", "OP", " trailing"]);
        assert_eq!(out, "hello ");
        assert_eq!(matched.as_deref(), Some("STOP"));
    }

    #[test]
    fn partial_false_alarm_is_flushed() {
        // "ST" looks like the start of "STOP" but resolves to "STxy".
        let (out, matched) = run(&["STOP"], &["ST", "xy"]);
        assert_eq!(out, "STxy");
        assert_eq!(matched, None);
    }

    #[test]
    fn held_tail_flushed_on_natural_end() {
        // Ends mid-prefix; no completion, so the tail is real output.
        let (out, matched) = run(&["STOP"], &["hello ST"]);
        assert_eq!(out, "hello ST");
        assert_eq!(matched, None);
    }

    #[test]
    fn earliest_of_multiple_stops_wins() {
        let (out, matched) = run(&["world", "STOP"], &["a STOP b world"]);
        assert_eq!(out, "a ");
        assert_eq!(matched.as_deref(), Some("STOP"));
    }

    #[test]
    fn stop_at_very_start_emits_nothing() {
        let (out, matched) = run(&["STOP"], &["STOP rest"]);
        assert_eq!(out, "");
        assert_eq!(matched.as_deref(), Some("STOP"));
    }

    #[test]
    fn overlapping_repeats_match_first_occurrence() {
        // "aa" with input "aaa" fed one char at a time matches at index 0.
        let (out, matched) = run(&["aa"], &["a", "a", "a"]);
        assert_eq!(out, "");
        assert_eq!(matched.as_deref(), Some("aa"));
        assert_eq!(apply_stop_sequences("aaa", Some(&["aa".to_string()])).0, "");
    }

    #[test]
    fn unicode_stop_string() {
        let (out, matched) = run(&["café"], &["a ca", "fé b"]);
        assert_eq!(out, "a ");
        assert_eq!(matched.as_deref(), Some("café"));
    }

    #[test]
    fn unicode_partial_does_not_split_codepoint() {
        // Feeding a multibyte char that is a stop prefix then diverging must not
        // panic on a non-boundary slice and must flush the real text.
        let (out, matched) = run(&["→end"], &["x→", "y"]);
        assert_eq!(out, "x→y");
        assert_eq!(matched, None);
    }

    /// The streamed result must equal `apply_stop_sequences` on the whole text,
    /// for every chunking, and so must the stop string it reports. This ties the
    /// incremental matcher to the established truncation semantics, identity
    /// included, so the `stopping_word` the response layer prints cannot drift
    /// from the text it printed.
    #[test]
    fn matches_apply_stop_sequences_for_all_chunkings() {
        let cases: &[(&str, &[&str])] = &[
            ("hello STOP world", &["STOP"]),
            ("no match here", &["STOP"]),
            ("end at THE END now", &["THE END"]),
            ("pick earliest: B then A", &["A", "B"]),
            ("tie at the same index", &["tie", "tie at"]),
            ("aaa", &["aa"]),
            ("café au lait", &["au"]),
            ("trailing prefix ST", &["STOP"]),
            ("", &["STOP"]),
        ];
        for (text, stops) in cases {
            let owned: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
            let (expected, expected_stop) = apply_stop_sequences(text, Some(&owned));

            // Whole string in one piece.
            let (whole, whole_stop) = run(stops, &[text]);
            assert_eq!(whole, expected, "whole-string chunking: {text:?}");
            assert_eq!(whole_stop, expected_stop, "whole-string stop: {text:?}");

            // Char-by-char.
            let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            let char_refs: Vec<&str> = chars.iter().map(String::as_str).collect();
            let (per_char, per_char_stop) = run(stops, &char_refs);
            assert_eq!(per_char, expected, "char-by-char chunking: {text:?}");
            assert_eq!(per_char_stop, expected_stop, "char-by-char stop: {text:?}");

            // Byte-pair-ish split at each char boundary.
            for split in 1..text.chars().count() {
                let idx: usize = text
                    .char_indices()
                    .nth(split)
                    .map(|(i, _)| i)
                    .unwrap_or(text.len());
                let (a, b) = text.split_at(idx);
                let (two, two_stop) = run(stops, &[a, b]);
                assert_eq!(two, expected, "two-piece split at {split} of {text:?}");
                assert_eq!(
                    two_stop, expected_stop,
                    "two-piece stop at {split} of {text:?}"
                );
            }
        }
    }

    /// The whole point of the matcher for the MLX path: no matter where the
    /// token boundaries fall, the client sees exactly the truncated text and
    /// never a byte of the stop string. Asserted over every chunking, against
    /// the concatenation of the emitted pieces rather than a final buffer.
    #[test]
    fn streamed_concatenation_never_leaks_a_partial_match() {
        let text = "count 1 2 3 4 5 6";
        let stops = ["5", "4 5"];
        let owned: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let (expected, expected_stop) = apply_stop_sequences(text, Some(&owned));
        // "4 5" starts before the bare "5", so the earliest match wins and the
        // reported word is the longer one, matching `apply_stop_sequences`.
        assert_eq!(expected, "count 1 2 3 ");
        assert_eq!(expected_stop.as_deref(), Some("4 5"));

        for split in 1..text.chars().count() {
            let idx = text.char_indices().nth(split).map(|(i, _)| i).unwrap();
            let (a, b) = text.split_at(idx);
            let mut m = StopMatcher::new(owned.clone());
            let mut streamed = String::new();
            for piece in [a, b] {
                let chunk = m.push(piece);
                // Every emitted piece must remain a prefix of the final text,
                // which is what "a partial match is never leaked" means.
                let candidate = format!("{streamed}{}", chunk.emit);
                assert!(
                    expected.starts_with(&candidate),
                    "leaked {candidate:?} at split {split}"
                );
                streamed = candidate;
                if chunk.matched.is_some() {
                    assert_eq!(chunk.matched, expected_stop);
                    break;
                }
            }
            assert_eq!(streamed, expected, "split {split}");
        }
    }

    /// Once a stop string matched, the matcher is inert: a caller that pushes
    /// again (a late token from an in-flight batch step) emits nothing more and
    /// keeps reporting the same stop string.
    #[test]
    fn push_after_match_is_inert() {
        let mut m = StopMatcher::new(vec!["STOP".to_string()]);
        let first = m.push("a STOP b");
        assert_eq!(first.emit, "a ");
        assert_eq!(first.matched.as_deref(), Some("STOP"));

        let after = m.push("more text");
        assert!(after.emit.is_empty());
        assert!(after.matched.is_some());
        assert_eq!(after.matched.as_deref(), Some("STOP"));
        assert_eq!(m.emitted_len(), "a ".len());
        assert!(m.flush().is_empty());
        assert_eq!(m.matched(), Some("STOP"));
    }
}
