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

//! Tiny `*`-wildcard pattern with positional capture.
//!
//! Why a custom matcher (and not `globset::GlobMatcher`)?
//!
//! The parser in [`crate::config`] uses `globset::GlobMatcher` for
//! syntax validation, but that matcher exposes a boolean
//! `is_match()` API only — it does not surface the substrings
//! matched by each `*` wildcard. The replace op needs those
//! substrings so it can substitute them into `source_key` to form
//! the donor tensor key. This module is the minimum-viable
//! capture-aware matcher that handles the patterns surgery configs
//! actually use.
//!
//! Wildcard semantics:
//!
//! - `*` matches zero or more characters of any kind. Tensor keys
//!   are dot-separated but `*` is *not* anchored to dot boundaries
//!   — that matches the existing `globset::Glob` validator's
//!   behavior for the typical `model.layers.*.self_attn.*.weight`
//!   form.
//! - When multiple wildcards appear, capture is leftmost-greedy
//!   with backtracking on the interior literal sequence. In
//!   practice tensor keys have stable structure
//!   (`model.layers.N.<subpath>`) so this produces the intuitive
//!   capture; the backtracking only matters when interior literals
//!   recur, which is easy to construct in tests but uncommon in
//!   real configs.
//!
//! Used by: [`super::ReplaceOp`]

/// Compiled `*`-wildcard pattern split into literal segments around
/// each wildcard. For a pattern with `N` `*` wildcards there are
/// `N + 1` literals; some may be empty (e.g. pattern `*.weight` has
/// literals `["", ".weight"]`).
pub(super) struct WildcardPattern {
    /// Original pattern string, kept for diagnostics.
    original: String,
    /// Literal segments between (and around) the wildcards.
    literals: Vec<String>,
}

impl WildcardPattern {
    /// Parse a pattern string.
    pub(super) fn parse(s: &str) -> Self {
        let literals = s.split('*').map(|seg| seg.to_string()).collect::<Vec<_>>();
        Self {
            original: s.to_string(),
            literals,
        }
    }

    /// Number of `*` wildcards in the pattern.
    pub(super) fn wildcard_count(&self) -> usize {
        // N+1 literals => N wildcards. saturating_sub guards the
        // theoretically-impossible empty-literals case.
        self.literals.len().saturating_sub(1)
    }

    /// Original pattern as written by the user. Used for error
    /// messages so the diagnostic shows what they actually typed.
    pub(super) fn original(&self) -> &str {
        &self.original
    }

    /// Match `text` against the pattern. Returns the captured
    /// fragments (one per `*` wildcard) on success.
    pub(super) fn match_with_captures(&self, text: &str) -> Option<Vec<String>> {
        if self.literals.is_empty() {
            // Should not happen — `split('*')` always returns at
            // least one segment — but be defensive.
            return if text.is_empty() {
                Some(Vec::new())
            } else {
                None
            };
        }

        // The first literal must be a prefix of `text`.
        let first = &self.literals[0];
        if !text.starts_with(first.as_str()) {
            return None;
        }
        let remaining = &text[first.len()..];

        // No wildcards: must equal exactly.
        if self.literals.len() == 1 {
            return if remaining.is_empty() {
                Some(Vec::new())
            } else {
                None
            };
        }

        // The last literal must be a suffix of what's left after the
        // prefix has been stripped.
        let last = self.literals.last().expect("non-empty");
        if !remaining.ends_with(last.as_str()) {
            return None;
        }
        let middle_end = remaining.len() - last.len();
        let middle = &remaining[..middle_end];

        // Special case: one wildcard total (two literals) — the
        // whole `middle` is the single capture.
        if self.literals.len() == 2 {
            return Some(vec![middle.to_string()]);
        }

        // General case: literals[1..len-1] are interior literals
        // that the captured fragments must straddle. Use recursive
        // matching to find a valid placement; first match wins.
        let interior = &self.literals[1..self.literals.len() - 1];
        let mut captures = Vec::with_capacity(self.wildcard_count());
        if match_interior(middle, interior, &mut captures) {
            Some(captures)
        } else {
            None
        }
    }

    /// Render the pattern by substituting `captures` for each `*`
    /// wildcard, in order. The caller must have already validated
    /// the count match (see [`super::ReplaceOp::new`]).
    pub(super) fn render(&self, captures: &[String]) -> String {
        debug_assert_eq!(
            captures.len(),
            self.wildcard_count(),
            "captures must match wildcard_count — validated at op construction"
        );
        let mut out = String::with_capacity(
            self.original.len() + captures.iter().map(|c| c.len()).sum::<usize>(),
        );
        for (i, lit) in self.literals.iter().enumerate() {
            out.push_str(lit);
            if i < captures.len() {
                out.push_str(&captures[i]);
            }
        }
        out
    }
}

/// Leftmost-greedy match of a sequence of interior literals against
/// `text`. On success the captures (one per gap between literals,
/// plus one before and one after) are pushed into `captures` and the
/// function returns `true`.
fn match_interior(text: &str, interior: &[String], captures: &mut Vec<String>) -> bool {
    if interior.is_empty() {
        // No interior literals — the entire `text` is the single
        // remaining capture.
        captures.push(text.to_string());
        return true;
    }
    let lit = &interior[0];
    if lit.is_empty() {
        // Adjacent wildcards `**` (or just `*`) — they collapse to
        // a single wildcard semantically; the empty literal here
        // forces an empty capture before it.
        captures.push(String::new());
        return match_interior(text, &interior[1..], captures);
    }
    // Find each occurrence of `lit` and try recursing from there.
    let mut search_from = 0;
    while let Some(idx) = text[search_from..].find(lit.as_str()) {
        let absolute = search_from + idx;
        captures.push(text[..absolute].to_string());
        let after = &text[absolute + lit.len()..];
        if match_interior(after, &interior[1..], captures) {
            return true;
        }
        captures.pop();
        // Advance one byte and retry; UTF-8 safe because `find`
        // yields valid byte boundaries.
        search_from = absolute + 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_stars_is_exact_match() {
        let p = WildcardPattern::parse("model.embed_tokens.weight");
        assert_eq!(p.wildcard_count(), 0);
        assert_eq!(
            p.match_with_captures("model.embed_tokens.weight"),
            Some(vec![])
        );
        assert!(p.match_with_captures("model.embed_tokens.bias").is_none());
        assert!(
            p.match_with_captures("xxmodel.embed_tokens.weight")
                .is_none()
        );
        assert_eq!(p.original(), "model.embed_tokens.weight");
    }

    #[test]
    fn single_star_captures_middle() {
        let p = WildcardPattern::parse("model.layers.*.self_attn.weight");
        assert_eq!(p.wildcard_count(), 1);
        let caps = p
            .match_with_captures("model.layers.12.self_attn.weight")
            .unwrap();
        assert_eq!(caps, vec!["12".to_string()]);

        let caps = p
            .match_with_captures("model.layers.0.self_attn.weight")
            .unwrap();
        assert_eq!(caps, vec!["0".to_string()]);
        assert!(p.match_with_captures("model.layers.0.mlp.weight").is_none());
    }

    #[test]
    fn multiple_stars_capture_in_order() {
        let p = WildcardPattern::parse("model.layers.*.self_attn.*.weight");
        let caps = p
            .match_with_captures("model.layers.5.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(caps, vec!["5".to_string(), "q_proj".to_string()]);
    }

    #[test]
    fn leading_star_captures_prefix() {
        let p = WildcardPattern::parse("*.weight");
        let caps = p.match_with_captures("model.embed_tokens.weight").unwrap();
        assert_eq!(caps, vec!["model.embed_tokens".to_string()]);
    }

    #[test]
    fn trailing_star_captures_suffix() {
        let p = WildcardPattern::parse("model.layers.0.*");
        let caps = p
            .match_with_captures("model.layers.0.q_proj.weight")
            .unwrap();
        assert_eq!(caps, vec!["q_proj.weight".to_string()]);
    }

    #[test]
    fn render_substitutes_captures_in_order() {
        let template = WildcardPattern::parse("donor.h.*.attn.*.weight");
        let rendered = template.render(&["7".to_string(), "kv".to_string()]);
        assert_eq!(rendered, "donor.h.7.attn.kv.weight");
    }

    #[test]
    fn render_with_zero_wildcards_is_verbatim() {
        let template = WildcardPattern::parse("plain.key");
        assert_eq!(template.render(&[]), "plain.key");
    }

    #[test]
    fn match_with_recurring_interior_literal_backtracks() {
        // Pattern `a*b*c` against `axbxc`: first attempt to place
        // the first `b` at index 1 (capture `x`), then look for `c`
        // in `xc` — succeeds with second capture `x`. This pins
        // that the matcher doesn't trip on simple recurrences.
        let p = WildcardPattern::parse("a.*.b.*.c");
        let caps = p.match_with_captures("a.foo.b.bar.c").unwrap();
        assert_eq!(caps, vec!["foo".to_string(), "bar".to_string()]);
    }

    #[test]
    fn match_with_repeated_interior_literal_picks_first_valid() {
        // The first occurrence of the interior literal that leads
        // to a successful match wins (leftmost). Useful sanity
        // check that backtracking doesn't blow up on duplicated
        // separators.
        let p = WildcardPattern::parse("*.x.*.y");
        let caps = p.match_with_captures("a.x.b.x.c.y").unwrap();
        // Leftmost greedy on the first `x` → captures `a` then
        // `b.x.c`.
        assert_eq!(caps, vec!["a".to_string(), "b.x.c".to_string()]);
    }

    #[test]
    fn double_star_is_treated_as_single_wildcard_with_empty_gap() {
        // `a**b` parses into literals ["a", "", "b"]. The implementation
        // treats the empty interior literal as forcing an empty
        // capture at that gap; the first wildcard consumes zero
        // characters and the second consumes the entire middle.
        // Real configs should not write `**` for this reason — the
        // parser-side `globset::Glob::new` already accepts it but
        // the surgery semantics collapse to a single capture.
        let p = WildcardPattern::parse("a**b");
        assert_eq!(p.wildcard_count(), 2);
        let caps = p.match_with_captures("aXYZb").unwrap();
        assert_eq!(caps, vec!["".to_string(), "XYZ".to_string()]);
    }
}
