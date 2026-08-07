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

//! Low-level text scanners shared by the Florence-2 answer parsers.
//!
//! These are the pieces of upstream's post-processing that are literal string
//! work rather than pattern matching: stripping sequence markers, finding the
//! leading phrase of a chunk, and reading `<loc_N>` runs. Two of them replace
//! an upstream regex with a hand-rolled scan, and the equivalence argument is
//! recorded on each.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py

/// Phrases dropped from `<CAPTION_TO_PHRASE_GROUNDING>` output.
///
/// Verbatim from the checkpoint (`_create_black_list_of_phrase_grounding`),
/// deduplicated and sorted so membership is a binary search. Upstream lists
/// 100 entries with 28 repeats and stores them in a `set`, so the duplicates
/// carry no meaning. `FILTER_BY_BLACK_LIST` is true in the shipped config, so
/// unlike most of upstream's post-processing knobs this one is live and really
/// does remove phrases from the output.
pub(super) const PHRASE_GROUNDING_BLACKLIST: &[&str] = &[
    "I",
    "a",
    "a group",
    "a set",
    "all",
    "an",
    "another",
    "any",
    "anybody",
    "anyone",
    "anything",
    "each",
    "each other",
    "everybody",
    "everyone",
    "everything",
    "few",
    "he",
    "her",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "image",
    "images",
    "it",
    "its",
    "itself",
    "lots",
    "many",
    "me",
    "mine",
    "myself",
    "nobody",
    "none",
    "one",
    "one another",
    "oneself",
    "other objects",
    "our",
    "ours",
    "ourselves",
    "several",
    "she",
    "some",
    "somebody",
    "someone",
    "something",
    "that",
    "the",
    "the image",
    "their",
    "theirs",
    "them",
    "themselves",
    "these",
    "they",
    "this",
    "those",
    "us",
    "we",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

/// Positions the phrase-string scan stops at, from upstream's
/// `(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)`.
pub(super) const PHRASE_STOPS: &[&str] = &[
    "<od>", "</od>", "<box>", "</box>", "<bbox>", "</bbox>", "<loc_",
];

/// Same list with `<poly>` appended, which is what the polygon parser uses.
pub(super) const PHRASE_STOPS_POLY: &[&str] = &[
    "<od>", "</od>", "<box>", "</box>", "<bbox>", "</bbox>", "<loc_", "<poly>",
];

/// Drop the sequence markers before parsing, as every upstream parser does.
pub(super) fn strip_sequence_markers(text: &str) -> String {
    text.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
}

/// Non-ASCII characters are dropped, not replaced: upstream round-trips the
/// phrase through `encode('ascii', errors='ignore')`.
pub(super) fn ascii_only(text: &str) -> String {
    text.chars().filter(char::is_ascii).collect()
}

/// The leading phrase of a chunk, or `None` when upstream's phrase regex would
/// have failed and skipped the chunk.
///
/// Upstream uses `^\s*(.*?)(?=<od>|...|<loc_)` and takes group 0, then strips
/// it. Two properties of that regex drive this scanner, and both are load
/// bearing:
///
/// - `.` does not match a newline, and `^` is not multi-line, so the search
///   fails outright if a newline sits between the end of the leading
///   whitespace run and the first stop position. The chunk is then dropped.
/// - The scan stops at the first stop *position*, not at the first `<`. A
///   `<sep>` in a polygon chunk is not in the stop list, so the phrase runs
///   past it.
///
/// The returned slice is the trimmed text before that stop position.
pub(super) fn leading_phrase<'a>(chunk: &'a str, stops: &[&str]) -> Option<&'a str> {
    let whitespace_end = chunk.len() - chunk.trim_start().len();
    for (offset, ch) in chunk.char_indices() {
        if offset < whitespace_end {
            continue;
        }
        if stops.iter().any(|stop| chunk[offset..].starts_with(stop)) {
            return Some(chunk[..offset].trim());
        }
        if ch == '\n' {
            return None;
        }
    }
    None
}

/// Read every `<loc_N>` bin in `text`, left to right.
///
/// Hand-rolled rather than a regex because the pattern is a fixed literal with
/// one digit run, and this is the innermost step of every parser. Values that
/// overflow `i32` or exceed the 1000-bin range cannot occur in real output
/// (the tokenizer has no token for them) but are accepted here rather than
/// rejected, matching upstream's plain `int()`.
pub(super) fn location_bins(text: &str) -> Vec<i32> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut index = 0usize;
    while let Some(found) = text[index..].find("<loc_") {
        let digits_start = index + found + "<loc_".len();
        let mut cursor = digits_start;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor > digits_start && bytes.get(cursor) == Some(&b'>') {
            if let Ok(bin) = text[digits_start..cursor].parse::<i32>() {
                out.push(bin);
            }
            index = cursor + 1;
        } else {
            index = digits_start;
        }
    }
    out
}

/// Every group of four consecutive location tokens, non-overlapping, matching
/// upstream's `finditer(r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>')`.
///
/// Because the tokens in a chunk are always adjacent, scanning the flat bin
/// list in groups of four is equivalent to the regex, and a trailing partial
/// group is discarded exactly as a failed regex match would be.
pub(super) fn box_bins(chunk: &str) -> Vec<[i32; 4]> {
    location_bins(chunk)
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

/// Upstream's `re.sub(r'^loc_\d+>', '', phrase_text, count=1)`, applied before
/// the polygon phrase search.
///
/// Unreachable in practice: a chunk either starts with `<` or with a plain
/// text run, and a model would have to emit the literal characters `loc_12>`
/// as prose to trigger it. Ported anyway because it costs a scan and removes a
/// divergence that would otherwise only show up on adversarial input.
pub(super) fn strip_bare_loc_prefix(chunk: &str) -> &str {
    let Some(rest) = chunk.strip_prefix("loc_") else {
        return chunk;
    };
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 && rest.as_bytes().get(digits) == Some(&b'>') {
        &rest[digits + 1..]
    } else {
        chunk
    }
}

#[cfg(test)]
#[path = "florence2_scan_tests.rs"]
mod florence2_scan_tests;
