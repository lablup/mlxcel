//! Per-request generation bounds that b10621 enforces during decode: the
//! minimum-indentation stop (`n_indent`) and the prediction-phase deadline
//! (`t_max_predict_ms`), both issue #1477.
//!
//! Both are stop rules rather than sampling parameters: they end a request that
//! is otherwise healthy, and upstream reports `stop_type: "limit"` for each
//! (`STOP_TYPE_LIMIT` at
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-context.cpp>,
//! `process_token`). They live here rather than on
//! [`StopMatcher`](super::stop_matcher::StopMatcher) because that component is
//! about string matching and is shared with the OpenXLA serve worker, while
//! these two need the request's own clock and its whole generated text.
//!
//! The indentation walk is a transcription of upstream's, including its one
//! -line-per-token cursor advance: `last_nl_pos` moves past a single `\n` per
//! decoded token, so the check reaches a dedented line a token or two after the
//! text of that line first appeared. Reproducing the cursor rather than a
//! simplified "first non-whitespace character" rule is what makes the stopping
//! token count agree with the pinned binary.

use std::time::Instant;

/// Why a generation bound ended a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundStop {
    /// `n_indent`: a generated line's indentation fell below the requested
    /// number of whitespace characters. `keep_bytes` is upstream's `pos`, the
    /// offset it erases the rest of the text from, which keeps the offending
    /// line's leading whitespace and drops everything after it.
    Indent { keep_bytes: usize },
    /// `t_max_predict_ms`: the prediction phase outran its deadline and a
    /// newline had been generated. Upstream truncates nothing here.
    Time,
}

/// The two bounds and the state their rules need.
///
/// [`Default`] is "no bounds configured", which makes
/// [`observe`](Self::observe) a cheap early return on every request that sends
/// neither field; nothing is accumulated in that case.
#[derive(Debug, Default)]
pub(crate) struct GenerationBounds {
    /// b10621 `n_indent`, 0 = disabled (its schema floor).
    n_indent: usize,
    /// b10621 `t_max_predict_ms`, `None` when disabled (its `<= 0` domain).
    t_max_predict_ms: Option<u64>,
    /// The generated text so far, accumulated only when `n_indent > 0`, because
    /// the indentation walk indexes into it the way upstream indexes into
    /// `slot.generated_text`.
    text: String,
    /// Upstream's `slot.last_nl_pos`: the byte offset just past the newline
    /// whose line is checked next.
    last_nl_pos: usize,
    /// Upstream's `slot.has_new_line`, which gates the indentation rule and
    /// arms the deadline.
    has_new_line: bool,
    /// When the first token was decoded; the deadline is measured from it, as
    /// the flag's own help text states.
    first_token_at: Option<Instant>,
    /// Sticky: once a bound has fired the request is over and further pieces
    /// are ignored.
    fired: Option<BoundStop>,
}

impl GenerationBounds {
    /// Build the bounds a request asked for. `n_indent == 0` and
    /// `t_max_predict_ms == None` is the inert configuration.
    pub(crate) fn new(n_indent: usize, t_max_predict_ms: Option<u64>) -> Self {
        Self {
            n_indent,
            t_max_predict_ms,
            ..Self::default()
        }
    }

    /// Whether either bound is configured. When `false`,
    /// [`observe`](Self::observe) does nothing at all.
    pub(crate) fn is_active(&self) -> bool {
        self.n_indent > 0 || self.t_max_predict_ms.is_some()
    }

    /// The bound that ended this request, or `None` while it continues. Sticky,
    /// so the finalization path can read it long after the deciding step.
    pub(crate) fn fired(&self) -> Option<BoundStop> {
        self.fired
    }

    /// Feed one decoded piece, in upstream's `process_token` order.
    ///
    /// `decoded` is the raw newly decoded text (upstream's `token_str`, which it
    /// appends to `generated_text` before any check). `emitted` is the part of
    /// it that actually reached the client (upstream's `result.text_to_send`),
    /// which is what arms `has_new_line`; the two differ only while a string
    /// stop sequence is being matched.
    ///
    /// Returns the bound that fired on this piece, if any.
    pub(crate) fn observe(&mut self, decoded: &str, emitted: &str) -> Option<BoundStop> {
        if !self.is_active() || self.fired.is_some() {
            return None;
        }
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        if self.n_indent > 0 {
            self.text.push_str(decoded);
        }

        // Indentation rule. Gated on a newline having been seen, and on the
        // cursor having advanced past it, exactly as upstream gates it.
        if self.n_indent > 0 && self.has_new_line {
            if self.last_nl_pos > 0
                && let Some(keep_bytes) = self.dedented_line_cut()
            {
                self.fired = Some(BoundStop::Indent { keep_bytes });
                return self.fired;
            }
            if let Some(rel) = self.text[self.last_nl_pos..].find('\n') {
                self.last_nl_pos += rel + 1;
            }
        }

        // Deadline rule: armed and checked on a newline-bearing piece only,
        // which is upstream's "but only upon another new line".
        if emitted.contains('\n') {
            self.has_new_line = true;
            if let Some(limit) = self.t_max_predict_ms
                && let Some(since) = self.first_token_at
                && since.elapsed().as_millis() as u64 > limit
            {
                self.fired = Some(BoundStop::Time);
                return self.fired;
            }
        }
        None
    }

    /// Upstream's indentation walk over the line starting at `last_nl_pos`:
    /// count the leading spaces and tabs, and report the cut offset when the
    /// line has a character after them and the count fell short.
    fn dedented_line_cut(&self) -> Option<usize> {
        let bytes = self.text.as_bytes();
        let mut pos = self.last_nl_pos;
        let mut indent = 0usize;
        while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
            indent += 1;
            pos += 1;
        }
        (pos < bytes.len() && indent < self.n_indent).then_some(pos)
    }
}

#[cfg(test)]
#[path = "generation_bounds_tests.rs"]
mod tests;
