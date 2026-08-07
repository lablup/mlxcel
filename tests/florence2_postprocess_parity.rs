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

//! Florence-2 answer post-processing, pinned against the upstream processor.
//!
//! Every expected value below was produced by importing the checkpoint's own
//! `processing_florence2.py` unmodified and calling `Florence2PostProcesser`
//! through the same dispatch `post_process_generation` uses, so these are real
//! upstream results rather than a reimplementation agreeing with itself. The
//! upstream file is published at
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py
//!
//! Post-processing is pure string-to-struct work, so unlike the rest of the
//! Florence-2 parity suite this file needs no checkpoint and always runs.
//!
//! To regenerate: in a virtualenv holding `transformers`, put the checkpoint
//! directory on `sys.path`, build
//! `Florence2PostProcesser(tokenizer=BartTokenizerFast.from_pretrained(dir))`,
//! and for each case call it with `parse_tasks` set to the task's entry in
//! `tasks_answer_post_processing_type`, then reshape the instance list exactly
//! as `post_process_generation` does. `AutoTokenizer` cannot be used because
//! it routes through `AutoConfig`, which does not know this checkpoint's
//! custom `florence2_language` model type; the post-processor only reads
//! `tokenizer.all_special_tokens`, so the choice does not affect the results.

use mlxcel::models::Florence2Task::{self, *};
use mlxcel::models::{Florence2ImageSize, Florence2TaskResult};

/// Coordinates are deterministic f32 arithmetic on both sides, so the observed
/// deviation is zero; the tolerance only guards against a literal in this file
/// being rounded during transcription.
const TOL: f32 = 1e-4;

fn size(width: u32, height: u32) -> Florence2ImageSize {
    Florence2ImageSize::new(width, height)
}

fn assert_coords(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length {got:?} vs {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= TOL,
            "{what}[{i}]: got {g}, reference {w} (tol {TOL})"
        );
    }
}

type Case<T> = (&'static str, &'static str, Florence2Task, (u32, u32), T);

// ---------------------------------------------------------------- pure text

#[rustfmt::skip]
const TEXT_CASES: &[Case<&'static str>] = &[
    ("caption_text", "<s>A green car on a street.</s>", Caption, (1000, 1000), "A green car on a street."),
    ("region_to_category", "<s>car</s>", RegionToCategory, (1000, 1000), "car"),
    // `<OCR>` is pure text even when the answer carries location tokens, so
    // they survive verbatim instead of being parsed into coordinates.
    ("plain_ocr", "<s>HELLO<loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><loc_7><loc_8>", Ocr, (1000, 1000), "HELLO<loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><loc_7><loc_8>"),
];

#[test]
fn pure_text_matches_upstream() {
    for (name, text, task, (w, h), want) in TEXT_CASES {
        let result = Florence2TaskResult::parse(text, *task, size(*w, *h));
        assert_eq!(
            result,
            Florence2TaskResult::Text((*want).to_string()),
            "{name}"
        );
    }
}

// -------------------------------------------------------------------- boxes

type BoxCase = Case<(&'static [[f32; 4]], &'static [&'static str])>;

#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const BOX_CASES: &[BoxCase] = &[
    ("od_two_objects", "<s>car<loc_52><loc_332><loc_932><loc_774>person<loc_10><loc_20><loc_30><loc_40></s>", ObjectDetection, (1000, 1000), (&[[52.5, 332.5, 932.5, 774.5], [10.5, 20.5, 30.5, 40.5]], &["car", "person"])),
    ("od_shared_label", "cat<loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><loc_7><loc_8>", ObjectDetection, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5]], &["cat", "cat"])),
    // Non-square: x scales by 640/1000 and y by 480/1000, independently.
    ("od_non_square", "car<loc_52><loc_332><loc_932><loc_774>", ObjectDetection, (640, 480), (&[[33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469]], &["car"])),
    // A bare location run read by the labelled parser: `[^<]+` starts one byte
    // into the first token and captures the literal text `loc_1>` as the
    // label, shifting the box by one bin. Upstream behavior, and the reason
    // `<REGION_PROPOSAL>` routes to the empty-phrase parser instead.
    ("od_bare_locs", "<s><loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><loc_7><loc_8></s>", ObjectDetection, (1000, 1000), (&[[2.5, 3.5, 4.5, 5.5]], &["loc_1>"])),
    ("od_short_run", "car<loc_1><loc_2><loc_3>", ObjectDetection, (1000, 1000), (&[], &[])),
    ("od_five_locs", "car<loc_1><loc_2><loc_3><loc_4><loc_5>", ObjectDetection, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5]], &["car"])),
    // Non-ASCII characters are dropped, not replaced.
    ("od_non_ascii", "caf\u{e9}<loc_1><loc_2><loc_3><loc_4>", ObjectDetection, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5]], &["caf"])),
    ("od_whitespace", "  a dog  <loc_1><loc_2><loc_3><loc_4>", ObjectDetection, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5]], &["a dog"])),
    // A newline before the first location token makes upstream's phrase regex
    // fail, dropping the whole chunk.
    ("od_newline", "a\nb<loc_1><loc_2><loc_3><loc_4>", ObjectDetection, (1000, 1000), (&[], &[])),
    ("od_multibyte", "\u{d55c}\u{ae00}car<loc_1><loc_2><loc_3><loc_4>", ObjectDetection, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5]], &["car"])),
    ("region_proposal", "<s><loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><loc_7><loc_8></s>", RegionProposal, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5]], &["", ""])),
    // Phrase grounding flattens to one label per box.
    ("grounding_groups", "<s>A green car<loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><loc_7><loc_8>parked next to a house<loc_9><loc_10><loc_11><loc_12></s>", CaptionToPhraseGrounding, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5], [9.5, 10.5, 11.5, 12.5]], &["A green car", "A green car", "parked next to a house"])),
    ("grounding_blacklist", "the image<loc_1><loc_2><loc_3><loc_4>a truck<loc_5><loc_6><loc_7><loc_8>", CaptionToPhraseGrounding, (1000, 1000), (&[[5.5, 6.5, 7.5, 8.5]], &["a truck"])),
    // The blacklist is case sensitive, so "The Image" survives.
    ("grounding_case", "The Image<loc_1><loc_2><loc_3><loc_4>", CaptionToPhraseGrounding, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5]], &["The Image"])),
    ("empty", "", ObjectDetection, (1000, 1000), (&[], &[])),
    ("markers_only", "<s></s>", ObjectDetection, (1000, 1000), (&[], &[])),
    ("pad_only", "<s><pad></s>", ObjectDetection, (1000, 1000), (&[], &[])),
    ("no_locs", "just a caption", ObjectDetection, (1000, 1000), (&[], &[])),
];

#[test]
fn boxes_match_upstream() {
    for (name, text, task, (w, h), (want_boxes, want_labels)) in BOX_CASES {
        let Florence2TaskResult::Boxes { boxes, labels } =
            Florence2TaskResult::parse(text, *task, size(*w, *h))
        else {
            panic!("{name}: expected the boxes variant");
        };
        assert_eq!(labels, *want_labels, "{name}: labels");
        assert_eq!(boxes.len(), want_boxes.len(), "{name}: box count");
        for (i, (got, want)) in boxes.iter().zip(*want_boxes).enumerate() {
            assert_coords(&got.to_array(), want, &format!("{name} box {i}"));
        }
    }
}

// --------------------------------------------------------------- quad boxes

type QuadCase = Case<(&'static [[f32; 8]], &'static [&'static str])>;

#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const QUAD_CASES: &[QuadCase] = &[
    ("ocr_two_lines", "<s>HELLO<loc_0><loc_0><loc_0><loc_0><loc_0><loc_0><loc_0><loc_0>WORLD<loc_10><loc_20><loc_30><loc_40><loc_50><loc_60><loc_70><loc_80>", OcrWithRegion, (1000, 1000), (&[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5]], &["HELLO", "WORLD"])),
    // The eight bins are (x, y) pairs, so alternating entries take different
    // scales on a non-square image.
    ("ocr_non_square", "A<loc_0><loc_0><loc_999><loc_0><loc_999><loc_999><loc_0><loc_999>", OcrWithRegion, (1000, 500), (&[[0.5, 0.25, 999.5, 0.25, 999.5, 499.75, 0.5, 499.75]], &["A"])),
    ("ocr_seven_locs", "A<loc_0><loc_0><loc_0><loc_0><loc_0><loc_0><loc_0>", OcrWithRegion, (1000, 1000), (&[], &[])),
];

#[test]
fn quad_boxes_match_upstream() {
    for (name, text, task, (w, h), (want_quads, want_labels)) in QUAD_CASES {
        let Florence2TaskResult::QuadBoxes { quad_boxes, labels } =
            Florence2TaskResult::parse(text, *task, size(*w, *h))
        else {
            panic!("{name}: expected the quad-boxes variant");
        };
        assert_eq!(labels, *want_labels, "{name}: labels");
        assert_eq!(quad_boxes.len(), want_quads.len(), "{name}: quad count");
        for (i, (got, want)) in quad_boxes.iter().zip(*want_quads).enumerate() {
            assert_coords(&got.points, want, &format!("{name} quad {i}"));
        }
    }
}

// ----------------------------------------------------------------- polygons

type PolygonCase = Case<(
    &'static [&'static [&'static [f32]]],
    &'static [&'static str],
)>;

#[rustfmt::skip]
const POLYGON_CASES: &[PolygonCase] = &[
    ("poly_single", "<s><loc_1><loc_2><loc_3><loc_4><loc_5><loc_6></s>", ReferringExpressionSegmentation, (1000, 1000), (&[&[&[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]]], &[""])),
    // `<sep>` splits outlines within one instance, not into two instances.
    ("poly_sep", "<loc_1><loc_2><loc_3><loc_4><sep><loc_5><loc_6><loc_7><loc_8>", ReferringExpressionSegmentation, (1000, 1000), (&[&[&[1.5, 2.5, 3.5, 4.5], &[5.5, 6.5, 7.5, 8.5]]], &[""])),
    // An odd bin count loses its unpaired tail.
    ("poly_odd", "<loc_1><loc_2><loc_3><loc_4><loc_5>", ReferringExpressionSegmentation, (1000, 1000), (&[&[&[1.5, 2.5, 3.5, 4.5]]], &[""])),
    ("poly_region_to_seg", "<s><loc_1><loc_2><loc_3><loc_4><sep><loc_5><loc_6><loc_7><loc_8></s>", RegionToSegmentation, (1000, 1000), (&[&[&[1.5, 2.5, 3.5, 4.5], &[5.5, 6.5, 7.5, 8.5]]], &[""])),
];

#[test]
fn polygons_match_upstream() {
    for (name, text, task, (w, h), (want_polys, want_labels)) in POLYGON_CASES {
        let Florence2TaskResult::Polygons { polygons, labels } =
            Florence2TaskResult::parse(text, *task, size(*w, *h))
        else {
            panic!("{name}: expected the polygons variant");
        };
        assert_eq!(labels, *want_labels, "{name}: labels");
        assert_polygons(&polygons, want_polys, name);
    }
}

// ---------------------------------------------- boxes or polygons (open vocab)

type MixedCase = Case<(
    &'static [[f32; 4]],
    &'static [&'static str],
    &'static [&'static [&'static [f32]]],
    &'static [&'static str],
)>;

#[rustfmt::skip]
const MIXED_CASES: &[MixedCase] = &[
    ("ovd_boxes", "<s>car<loc_1><loc_2><loc_3><loc_4></s>", OpenVocabularyDetection, (1000, 1000), (&[[1.5, 2.5, 3.5, 4.5]], &["car"], &[], &[])),
    ("ovd_polys", "<s>car<poly><loc_1><loc_2><loc_3><loc_4></poly></s>", OpenVocabularyDetection, (1000, 1000), (&[], &[], &[&[&[1.5, 2.5, 3.5, 4.5]]], &["car"])),
    ("ovd_two_polys", "car<poly><loc_1><loc_2><loc_3><loc_4></poly><poly><loc_5><loc_6><loc_7><loc_8></poly>", OpenVocabularyDetection, (1000, 1000), (&[], &[], &[&[&[1.5, 2.5, 3.5, 4.5]], &[&[5.5, 6.5, 7.5, 8.5]]], &["car", "car"])),
    ("ovd_poly_phrase", "a wheel<poly><loc_1><loc_2><loc_3><loc_4></poly>", OpenVocabularyDetection, (1000, 1000), (&[], &[], &[&[&[1.5, 2.5, 3.5, 4.5]]], &["a wheel"])),
];

#[test]
fn open_vocabulary_detection_matches_upstream() {
    for (name, text, task, (w, h), (want_boxes, want_box_labels, want_polys, want_poly_labels)) in
        MIXED_CASES
    {
        let Florence2TaskResult::BoxesOrPolygons {
            boxes,
            box_labels,
            polygons,
            polygon_labels,
        } = Florence2TaskResult::parse(text, *task, size(*w, *h))
        else {
            panic!("{name}: expected the mixed variant");
        };
        assert_eq!(box_labels, *want_box_labels, "{name}: box labels");
        assert_eq!(polygon_labels, *want_poly_labels, "{name}: polygon labels");
        assert_eq!(boxes.len(), want_boxes.len(), "{name}: box count");
        for (i, (got, want)) in boxes.iter().zip(*want_boxes).enumerate() {
            assert_coords(&got.to_array(), want, &format!("{name} box {i}"));
        }
        assert_polygons(&polygons, want_polys, name);
    }
}

fn assert_polygons(got: &[Vec<mlxcel::models::Florence2Polygon>], want: &[&[&[f32]]], name: &str) {
    assert_eq!(got.len(), want.len(), "{name}: instance count");
    for (i, (instance, want_instance)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            instance.len(),
            want_instance.len(),
            "{name} instance {i}: outline count"
        );
        for (j, (outline, want_outline)) in instance.iter().zip(*want_instance).enumerate() {
            assert_coords(
                &outline.points,
                want_outline,
                &format!("{name} instance {i} outline {j}"),
            );
        }
    }
}

/// Every case above must have been exercised, so a table that silently loses
/// rows during an edit fails rather than passing vacuously.
#[test]
fn the_parity_tables_cover_every_result_variant() {
    assert_eq!(TEXT_CASES.len(), 3);
    assert_eq!(BOX_CASES.len(), 18);
    assert_eq!(QUAD_CASES.len(), 3);
    assert_eq!(POLYGON_CASES.len(), 4);
    assert_eq!(MIXED_CASES.len(), 4);
}
