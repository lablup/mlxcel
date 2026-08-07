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

//! Parsers that turn a decoded Florence-2 answer into typed instances.
//!
//! The model answers a spatial task with a flat string mixing plain words and
//! `<loc_*>` tokens, for example
//! `"car<loc_52><loc_332><loc_932><loc_774>person<loc_10><loc_20><loc_30><loc_40>"`.
//! Each parser here splits that into phrase chunks, reads the location runs,
//! and dequantizes them through [`super::coords`].
//!
//! Port of `Florence2PostProcesser`:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py
//!
//! Three upstream behaviors are deliberately **not** ported, because they are
//! unreachable from `post_process_generation`:
//!
//! - `parse_od_from_text_and_spans` and the `PATTERN` config indirection. No
//!   task routes to the `'od'` parse task, and its configured pattern is
//!   `r'([a-zA-Z0-9 ]+)<loc_(\\d+)>...'`, whose doubled backslash inside a raw
//!   string matches a literal `\` followed by `d` and so can never match model
//!   output. Every other parser overwrites its `pattern` argument on its first
//!   lines, so only the OCR pattern is ever read from config at all.
//! - The OCR area filter. It is gated on `area_threshold > 0` and the shipped
//!   `AREA_THRESHOLD` is `0.00`, so it never runs.
//! - `with_box_at_start` in the polygon parser. Every reachable call site
//!   leaves it false.
//!
//! Also inert upstream and skipped here: the `<ground>` / `<obj>` chunk
//! prefixes stripped before the phrase search. Neither string is in the
//! checkpoint's vocabulary, and the second assignment overwrites the first
//! (it reads `pharse_text` rather than the result of the `<ground>` strip), so
//! the pair cannot change any output.

use std::sync::LazyLock;

use fancy_regex::Regex;

use super::coords::{
    Florence2BoundingBox, Florence2ImageSize, Florence2Polygon, Florence2QuadBox, dequantize_box,
    dequantize_coordinates,
};
use super::scan::{
    PHRASE_GROUNDING_BLACKLIST, PHRASE_STOPS, PHRASE_STOPS_POLY, ascii_only, box_bins,
    leading_phrase, location_bins, parse_bin, strip_bare_loc_prefix, strip_sequence_markers,
};

/// One detected box with its open-vocabulary label.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoxInstance {
    pub bbox: Florence2BoundingBox,
    pub cat_name: String,
}

/// One phrase from a caption together with every box grounding it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroundedPhrase {
    pub boxes: Vec<Florence2BoundingBox>,
    pub cat_name: String,
}

/// One OCR text line and the quadrilateral around it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QuadInstance {
    pub quad_box: Florence2QuadBox,
    pub text: String,
}

/// One segmentation instance: a label and the outlines that make it up.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PolygonInstance {
    pub polygons: Vec<Florence2Polygon>,
    pub cat_name: String,
}

/// Every pattern below is a literal, compiled once, and covered by
/// `florence2_post_processing_patterns_compile`. `None` is therefore
/// unreachable; the parsers still treat it as "no matches" rather than
/// aborting, matching how the rest of the tree handles a static regex
/// (`src/server/tool_calls/formats.rs`).
type Pattern = LazyLock<Option<Regex>>;

/// Chunk splitter for the box tasks: a run of plain text followed by at least
/// four location tokens.
static CHUNK_WITH_PHRASE: Pattern = LazyLock::new(|| Regex::new(r"([^<]+(?:<loc_\d+>){4,})").ok());

/// Same, for `allow_empty_phrase`: bare location runs with no leading text.
static CHUNK_BARE: Pattern = LazyLock::new(|| Regex::new(r"(?:(?:<loc_\d+>){4,})").ok());

/// Chunk splitter for the polygon tasks. `<sep>`, `<poly>` and `</poly>`
/// count toward the four-unit minimum alongside location tokens.
static CHUNK_POLY_WITH_PHRASE: Pattern =
    LazyLock::new(|| Regex::new(r"([^<]+(?:<loc_\d+>|<sep>|<poly>|</poly>){4,})").ok());

static CHUNK_POLY_BARE: Pattern =
    LazyLock::new(|| Regex::new(r"(?:(?:<loc_\d+>|<sep>|<poly>|</poly>){4,})").ok());

/// One OCR record: a non-greedy text run followed by exactly eight location
/// tokens, the four corners of the quadrilateral.
static OCR_RECORD: Pattern = LazyLock::new(|| {
    Regex::new(concat!(
        r"(.+?)",
        r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
        r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
    ))
    .ok()
});

/// One polygon instance inside a chunk, delimited by `<poly>` / `</poly>`.
static POLY_INSTANCE: Pattern = LazyLock::new(|| Regex::new(r"<poly>(.*?)</poly>").ok());

/// A maximal run of adjacent location tokens, terminated by `<sep>` or by the
/// end of the instance. One run is one polygon outline.
///
/// One knowingly accepted difference from Python: `$` here matches only at the
/// very end of the string, while Python's `$` also matches just before a
/// trailing newline. Location runs are emitted without trailing newlines, so
/// the two agree on real output.
static POLY_RUN: Pattern = LazyLock::new(|| Regex::new(r"((?:<loc_\d+>)+)(?:<sep>|$)").ok());

/// Full-match iteration over a chunk pattern.
///
/// Every chunk pattern either has no capture group or wraps the whole match in
/// one, so group 0 and `re.findall`'s result are the same string.
fn chunks<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {
    re.find_iter(text)
        .filter_map(Result::ok)
        .map(|m| m.as_str())
        .collect()
}

/// `<OCR_WITH_REGION>`: text lines with a rotated quadrilateral each.
///
/// Only `<s>` is stripped here, not `</s>` or `<pad>`; that asymmetry is
/// upstream's and it is harmless because the trailing markers cannot sit
/// between a text run and its eight location tokens.
pub(crate) fn parse_ocr(text: &str, size: Florence2ImageSize) -> Vec<QuadInstance> {
    let Some(record) = OCR_RECORD.as_ref() else {
        return Vec::new();
    };
    let text = text.replace("<s>", "");
    let mut instances = Vec::new();
    for captures in record.captures_iter(&text).filter_map(Result::ok) {
        let Some(content) = captures.get(1) else {
            continue;
        };
        let mut bins = [0i32; 8];
        let mut complete = true;
        for (slot, bin) in bins.iter_mut().enumerate() {
            match captures.get(slot + 2) {
                Some(group) => *bin = parse_bin(group.as_str()),
                None => complete = false,
            }
        }
        if !complete {
            continue;
        }
        let dequantized = dequantize_coordinates(&bins, size);
        let mut points = [0.0f32; 8];
        if dequantized.len() != points.len() {
            continue;
        }
        points.copy_from_slice(&dequantized);
        instances.push(QuadInstance {
            quad_box: Florence2QuadBox { points },
            text: content.as_str().to_string(),
        });
    }
    instances
}

/// `<OD>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`: one instance per
/// box, with the chunk's phrase repeated across every box it carries.
///
/// `allow_empty_phrase` selects the `<REGION_PROPOSAL>` behavior, where the
/// answer is bare location runs with no category names and each instance gets
/// an empty label.
pub(crate) fn parse_boxes(
    text: &str,
    size: Florence2ImageSize,
    allow_empty_phrase: bool,
) -> Vec<BoxInstance> {
    let text = strip_sequence_markers(text);
    let cell = if allow_empty_phrase {
        &CHUNK_BARE
    } else {
        &CHUNK_WITH_PHRASE
    };
    let Some(pattern) = cell.as_ref() else {
        return Vec::new();
    };

    let mut instances = Vec::new();
    for chunk in chunks(pattern, &text) {
        let Some(phrase) = leading_phrase(chunk, PHRASE_STOPS) else {
            continue;
        };
        let boxes = box_bins(chunk);
        if boxes.is_empty() {
            continue;
        }
        let cat_name = ascii_only(phrase);
        for bins in boxes {
            instances.push(BoxInstance {
                bbox: dequantize_box(bins, size),
                cat_name: cat_name.clone(),
            });
        }
    }
    instances
}

/// `<CAPTION_TO_PHRASE_GROUNDING>`: one instance per phrase, keeping all of
/// that phrase's boxes together, with the stopword blacklist applied.
///
/// The blacklist is checked against the trimmed phrase *before* the non-ASCII
/// filter runs, which is upstream's order and matters for a phrase whose only
/// difference from a blacklisted entry is a non-ASCII character.
pub(crate) fn parse_phrase_grounding(text: &str, size: Florence2ImageSize) -> Vec<GroundedPhrase> {
    let text = strip_sequence_markers(text);
    let Some(pattern) = CHUNK_WITH_PHRASE.as_ref() else {
        return Vec::new();
    };

    let mut instances = Vec::new();
    for chunk in chunks(pattern, &text) {
        let Some(phrase) = leading_phrase(chunk, PHRASE_STOPS) else {
            continue;
        };
        let boxes = box_bins(chunk);
        if boxes.is_empty() {
            continue;
        }
        if PHRASE_GROUNDING_BLACKLIST.binary_search(&phrase).is_ok() {
            continue;
        }
        instances.push(GroundedPhrase {
            boxes: boxes
                .into_iter()
                .map(|bins| dequantize_box(bins, size))
                .collect(),
            cat_name: ascii_only(phrase),
        });
    }
    instances
}

/// `<REFERRING_EXPRESSION_SEGMENTATION>`, `<REGION_TO_SEGMENTATION>`, and the
/// polygon branch of `<OPEN_VOCABULARY_DETECTION>`.
///
/// A chunk holds one or more instances. When it carries both `<poly>` and
/// `</poly>` the instances are the delimited spans; otherwise the whole chunk
/// is a single instance. Within an instance, each maximal run of adjacent
/// location tokens is one outline, and `<sep>` separates runs.
pub(crate) fn parse_polygons(
    text: &str,
    size: Florence2ImageSize,
    allow_empty_phrase: bool,
) -> Vec<PolygonInstance> {
    let text = strip_sequence_markers(text);
    let cell = if allow_empty_phrase {
        &CHUNK_POLY_BARE
    } else {
        &CHUNK_POLY_WITH_PHRASE
    };
    let (Some(pattern), Some(instance_re), Some(run_re)) =
        (cell.as_ref(), POLY_INSTANCE.as_ref(), POLY_RUN.as_ref())
    else {
        return Vec::new();
    };

    let mut instances = Vec::new();
    for chunk in chunks(pattern, &text) {
        let Some(phrase) = leading_phrase(strip_bare_loc_prefix(chunk), PHRASE_STOPS_POLY) else {
            continue;
        };
        let phrase = phrase.to_string();

        let spans: Vec<&str> = if chunk.contains("<poly>") && chunk.contains("</poly>") {
            instance_re
                .captures_iter(chunk)
                .filter_map(Result::ok)
                .filter_map(|c| c.get(1).map(|g| g.as_str()))
                .collect()
        } else {
            vec![chunk]
        };

        for span in spans {
            let mut polygons = Vec::new();
            for run in run_re.captures_iter(span).filter_map(Result::ok) {
                let Some(run) = run.get(1) else { continue };
                let mut bins = location_bins(run.as_str());
                // An outline is a flat (x, y) sequence, so a trailing
                // unpaired bin is dropped rather than paired with nothing.
                if bins.len() % 2 == 1 {
                    bins.pop();
                }
                polygons.push(Florence2Polygon {
                    points: dequantize_coordinates(&bins, size),
                });
            }
            if polygons.is_empty() {
                continue;
            }
            instances.push(PolygonInstance {
                polygons,
                cat_name: phrase.clone(),
            });
        }
    }
    instances
}

#[cfg(test)]
#[path = "florence2_parse_tests.rs"]
mod florence2_parse_tests;
