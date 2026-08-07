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

//! Scanner behavior: the literal string work the parsers are built on.
//!
//! Kept apart from the parser tests because these functions know nothing
//! about tasks or coordinates; they are exercised directly on the strings
//! that reach them.

use super::*;

#[test]
fn blacklist_is_sorted_for_binary_search() {
    assert!(
        PHRASE_GROUNDING_BLACKLIST.windows(2).all(|w| w[0] < w[1]),
        "blacklist must be sorted and deduplicated"
    );
    assert_eq!(PHRASE_GROUNDING_BLACKLIST.len(), 72);
}

#[test]
fn location_bins_reads_every_token() {
    assert_eq!(location_bins("<loc_0><loc_999><loc_52>"), vec![0, 999, 52]);
    assert_eq!(location_bins("car<loc_1>dog<loc_2>"), vec![1, 2]);
    assert!(location_bins("no locations here").is_empty());
    // Malformed tokens must be skipped without swallowing the next one.
    assert_eq!(location_bins("<loc_><loc_7>"), vec![7]);
    assert_eq!(location_bins("<loc_12<loc_8>"), vec![8]);
    // Leading zeros are ordinary digits, not an overflow.
    assert_eq!(location_bins("<loc_0000000000000005>"), vec![5]);
}

/// A digit run too wide for `i32` saturates instead of vanishing. Upstream's
/// `int()` has no width limit, and dropping the token would shift every bin
/// after it into the wrong slot of the four-at-a-time grouping.
#[test]
fn location_bins_saturates_instead_of_dropping_a_wide_run() {
    assert_eq!(location_bins("<loc_99999999999><loc_7>"), vec![i32::MAX, 7]);
    assert_eq!(parse_bin("999"), 999);
    assert_eq!(parse_bin("99999999999"), i32::MAX);
}

#[test]
fn leading_phrase_stops_at_the_first_stop_marker() {
    assert_eq!(leading_phrase("car<loc_1>", PHRASE_STOPS), Some("car"));
    assert_eq!(leading_phrase("  car <loc_1>", PHRASE_STOPS), Some("car"));
    assert_eq!(leading_phrase("<loc_1>", PHRASE_STOPS), Some(""));
    assert_eq!(leading_phrase("a<od>b<loc_1>", PHRASE_STOPS), Some("a"));
    // No stop marker at all is upstream's `re.search` returning None.
    assert_eq!(leading_phrase("no markers", PHRASE_STOPS), None);
}

/// Upstream's `.` does not match a newline and its `^` is not multi-line, so
/// a newline before the first location token makes the whole chunk vanish.
/// Reproducing that keeps a multi-line answer parsing identically.
#[test]
fn leading_phrase_fails_across_a_newline() {
    assert_eq!(leading_phrase("a\nb<loc_1>", PHRASE_STOPS), None);
    // Leading whitespace is consumed by `\s*` first, so a newline there is fine.
    assert_eq!(leading_phrase("\n  a<loc_1>", PHRASE_STOPS), Some("a"));
}

/// `<sep>` is not in the stop list, so the phrase scan runs past it. This is
/// the one place the two stop lists visibly differ from "first `<`".
#[test]
fn leading_phrase_runs_past_a_sep_marker() {
    assert_eq!(
        leading_phrase("a<sep>b<loc_1>", PHRASE_STOPS_POLY),
        Some("a<sep>b")
    );
}

#[test]
fn bare_loc_prefix_is_stripped() {
    assert_eq!(strip_bare_loc_prefix("loc_12>rest"), "rest");
    assert_eq!(strip_bare_loc_prefix("<loc_12>rest"), "<loc_12>rest");
    assert_eq!(strip_bare_loc_prefix("loc_>rest"), "loc_>rest");
    assert_eq!(strip_bare_loc_prefix("plain"), "plain");
}
