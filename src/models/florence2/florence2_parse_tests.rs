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

//! Parser behavior against synthetic answers. No checkpoint required: these
//! run on strings shaped like real Florence-2 output, so they cover the
//! branches a single real image never would.

use super::*;

/// 1000x1000 makes bin n dequantize to n + 0.5, so expected coordinates stay
/// readable and any axis mix-up still shows up in the ordering.
fn unit_size() -> Florence2ImageSize {
    Florence2ImageSize::new(1000, 1000)
}

fn locs(bins: &[i32]) -> String {
    bins.iter().map(|b| format!("<loc_{b}>")).collect()
}

#[test]
fn florence2_post_processing_patterns_compile() {
    // The parsers degrade to "no instances" if a pattern fails to build, so
    // this is the test that keeps that fallback unreachable.
    assert!(CHUNK_WITH_PHRASE.is_some());
    assert!(CHUNK_BARE.is_some());
    assert!(CHUNK_POLY_WITH_PHRASE.is_some());
    assert!(CHUNK_POLY_BARE.is_some());
    assert!(OCR_RECORD.is_some());
    assert!(POLY_INSTANCE.is_some());
    assert!(POLY_RUN.is_some());
}

#[test]
fn parse_boxes_reads_labelled_detections() {
    let text = format!(
        "<s>car{}person{}</s>",
        locs(&[52, 332, 932, 774]),
        locs(&[10, 20, 30, 40])
    );
    let instances = parse_boxes(&text, unit_size(), false);
    assert_eq!(instances.len(), 2);
    assert_eq!(instances[0].cat_name, "car");
    assert_eq!(
        instances[0].bbox.to_array(),
        [52.5, 332.5, 932.5, 774.5],
        "box must be dequantized from its own four bins"
    );
    assert_eq!(instances[1].cat_name, "person");
    assert_eq!(instances[1].bbox.to_array(), [10.5, 20.5, 30.5, 40.5]);
}

/// A phrase carrying several boxes becomes one instance per box, all sharing
/// the label. This is what makes `boxes` and `labels` index-aligned.
#[test]
fn parse_boxes_repeats_the_label_across_boxes() {
    let text = format!("cat{}{}", locs(&[1, 2, 3, 4]), locs(&[5, 6, 7, 8]));
    let instances = parse_boxes(&text, unit_size(), false);
    assert_eq!(instances.len(), 2);
    assert!(instances.iter().all(|i| i.cat_name == "cat"));
    assert_eq!(instances[0].bbox.to_array(), [1.5, 2.5, 3.5, 4.5]);
    assert_eq!(instances[1].bbox.to_array(), [5.5, 6.5, 7.5, 8.5]);
}

/// `<REGION_PROPOSAL>` answers carry no category names at all, so the chunker
/// has to accept a bare location run.
#[test]
fn parse_boxes_allows_empty_phrases_for_region_proposal() {
    let text = format!("<s>{}{}</s>", locs(&[1, 2, 3, 4]), locs(&[5, 6, 7, 8]));
    let instances = parse_boxes(&text, unit_size(), true);
    assert_eq!(instances.len(), 2);
    assert!(instances.iter().all(|i| i.cat_name.is_empty()));
    assert_eq!(instances[0].bbox.to_array(), [1.5, 2.5, 3.5, 4.5]);
    assert_eq!(instances[1].bbox.to_array(), [5.5, 6.5, 7.5, 8.5]);
}

/// The same bare run read *without* the flag does not come back empty, and the
/// reason is worth pinning: `[^<]+` can start one byte into `<loc_1>` and match
/// the literal text `loc_1>`, leaving that as the label and consuming the first
/// location token. So a bare run misparses into one box shifted by a token
/// rather than into nothing.
///
/// This is upstream behavior, verified against the checkpoint's own
/// `processing_florence2.py`, and it is exactly why `<REGION_PROPOSAL>` routes
/// to the `allow_empty_phrase` parser instead of the default one.
#[test]
fn parse_boxes_misreads_a_bare_run_without_the_flag() {
    let text = format!("<s>{}{}</s>", locs(&[1, 2, 3, 4]), locs(&[5, 6, 7, 8]));
    let instances = parse_boxes(&text, unit_size(), false);
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].cat_name, "loc_1>");
    assert_eq!(instances[0].bbox.to_array(), [2.5, 3.5, 4.5, 5.5]);
}

/// Fewer than four location tokens is not a box, and a fifth trailing token
/// does not create a second one.
#[test]
fn parse_boxes_ignores_short_runs() {
    assert!(parse_boxes(&format!("car{}", locs(&[1, 2, 3])), unit_size(), false).is_empty());
    let instances = parse_boxes(
        &format!("car{}", locs(&[1, 2, 3, 4, 5])),
        unit_size(),
        false,
    );
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].bbox.to_array(), [1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn parse_boxes_drops_non_ascii_characters() {
    // `encode('ascii', errors='ignore')` removes the character rather than
    // replacing it, so "café" becomes "caf" and not "caf?".
    let instances = parse_boxes(
        &format!("caf\u{e9}{}", locs(&[1, 2, 3, 4])),
        unit_size(),
        false,
    );
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].cat_name, "caf");
}

#[test]
fn parse_boxes_trims_surrounding_whitespace() {
    let instances = parse_boxes(
        &format!("  a dog  {}", locs(&[1, 2, 3, 4])),
        unit_size(),
        false,
    );
    assert_eq!(instances[0].cat_name, "a dog");
}

#[test]
fn parse_ocr_reads_quadrilaterals() {
    let text = format!(
        "<s>HELLO{}WORLD{}",
        locs(&[0; 8]),
        locs(&[10, 20, 30, 40, 50, 60, 70, 80])
    );
    let instances = parse_ocr(&text, unit_size());
    assert_eq!(instances.len(), 2);
    assert_eq!(instances[0].text, "HELLO");
    assert_eq!(instances[0].quad_box.points, [0.5; 8]);
    assert_eq!(instances[1].text, "WORLD");
    assert_eq!(
        instances[1].quad_box.points,
        [10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5]
    );
}

/// The quad's eight bins are (x, y) pairs, so a non-square image scales
/// alternating entries by different factors. Getting this wrong is invisible
/// on a square test image.
#[test]
fn parse_ocr_scales_x_and_y_separately() {
    let text = format!("A{}", locs(&[0, 0, 999, 0, 999, 999, 0, 999]));
    let instances = parse_ocr(&text, Florence2ImageSize::new(1000, 500));
    assert_eq!(instances.len(), 1);
    let p = instances[0].quad_box.points;
    assert!((p[0] - 0.5).abs() < 1e-3, "x0 {}", p[0]);
    assert!((p[1] - 0.25).abs() < 1e-3, "y0 {}", p[1]);
    assert!((p[2] - 999.5).abs() < 1e-2, "x1 {}", p[2]);
    assert!((p[5] - 499.75).abs() < 1e-2, "y2 {}", p[5]);
}

/// Exactly eight tokens make a record; seven is not an OCR line.
#[test]
fn parse_ocr_requires_eight_locations() {
    assert!(parse_ocr(&format!("A{}", locs(&[0; 7])), unit_size()).is_empty());
}

#[test]
fn parse_phrase_grounding_groups_boxes_per_phrase() {
    let text = format!(
        "<s>A green car{}{}parked next to a house{}</s>",
        locs(&[1, 2, 3, 4]),
        locs(&[5, 6, 7, 8]),
        locs(&[9, 10, 11, 12])
    );
    let instances = parse_phrase_grounding(&text, unit_size());
    assert_eq!(instances.len(), 2);
    assert_eq!(instances[0].cat_name, "A green car");
    assert_eq!(instances[0].boxes.len(), 2, "both boxes stay on one phrase");
    assert_eq!(instances[1].cat_name, "parked next to a house");
    assert_eq!(instances[1].boxes.len(), 1);
}

/// `FILTER_BY_BLACK_LIST` is true in the shipped config, so stopword phrases
/// really are dropped rather than returned with empty labels.
#[test]
fn parse_phrase_grounding_drops_blacklisted_phrases() {
    let text = format!(
        "the image{}a truck{}",
        locs(&[1, 2, 3, 4]),
        locs(&[5, 6, 7, 8])
    );
    let instances = parse_phrase_grounding(&text, unit_size());
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].cat_name, "a truck");

    // Case sensitive, matching the Python `in` test against a set of
    // lowercase strings plus the bare "I".
    let cased = format!("The Image{}", locs(&[1, 2, 3, 4]));
    assert_eq!(parse_phrase_grounding(&cased, unit_size()).len(), 1);
}

#[test]
fn parse_polygons_reads_a_single_outline() {
    let text = format!("<s>{}</s>", locs(&[1, 2, 3, 4, 5, 6]));
    let instances = parse_polygons(&text, unit_size(), true);
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].cat_name, "");
    assert_eq!(instances[0].polygons.len(), 1);
    assert_eq!(
        instances[0].polygons[0].points,
        vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    );
}

/// `<sep>` splits one instance into several outlines, which is how a
/// disconnected object is expressed.
#[test]
fn parse_polygons_splits_outlines_on_sep() {
    let text = format!("{}<sep>{}", locs(&[1, 2, 3, 4]), locs(&[5, 6, 7, 8]));
    let instances = parse_polygons(&text, unit_size(), true);
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].polygons.len(), 2);
    assert_eq!(instances[0].polygons[0].points, vec![1.5, 2.5, 3.5, 4.5]);
    assert_eq!(instances[0].polygons[1].points, vec![5.5, 6.5, 7.5, 8.5]);
}

/// `<poly>` ... `</poly>` delimits separate instances inside one chunk.
#[test]
fn parse_polygons_splits_instances_on_poly_markers() {
    let text = format!(
        "car<poly>{}</poly><poly>{}</poly>",
        locs(&[1, 2, 3, 4]),
        locs(&[5, 6, 7, 8])
    );
    let instances = parse_polygons(&text, unit_size(), false);
    assert_eq!(instances.len(), 2);
    assert!(instances.iter().all(|i| i.cat_name == "car"));
    assert_eq!(instances[0].polygons[0].points, vec![1.5, 2.5, 3.5, 4.5]);
    assert_eq!(instances[1].polygons[0].points, vec![5.5, 6.5, 7.5, 8.5]);
}

/// An outline is a flat (x, y) run, so an odd token count loses its tail
/// instead of pairing a coordinate with nothing.
#[test]
fn parse_polygons_drops_an_unpaired_trailing_bin() {
    let text = locs(&[1, 2, 3, 4, 5]);
    let instances = parse_polygons(&text, unit_size(), true);
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].polygons[0].points, vec![1.5, 2.5, 3.5, 4.5]);
}

/// The polygon phrase scan stops at `<poly>` as well as `<loc_`, unlike the
/// box scan. Without that extra stop the label would swallow the marker.
#[test]
fn parse_polygons_stops_the_phrase_at_the_poly_marker() {
    let text = format!("a wheel<poly>{}</poly>", locs(&[1, 2, 3, 4]));
    let instances = parse_polygons(&text, unit_size(), false);
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].cat_name, "a wheel");
}

#[test]
fn multibyte_phrases_do_not_split_a_character() {
    // A UTF-8 phrase must not be sliced mid-character on the way to the
    // ASCII filter.
    let text = format!("\u{d55c}\u{ae00}car{}", locs(&[1, 2, 3, 4]));
    let instances = parse_boxes(&text, unit_size(), false);
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].cat_name, "car");
}

#[test]
fn empty_and_marker_only_answers_produce_nothing() {
    for text in ["", "<s></s>", "<s><pad></s>", "just a caption"] {
        assert!(parse_boxes(text, unit_size(), false).is_empty(), "{text}");
        assert!(parse_ocr(text, unit_size()).is_empty(), "{text}");
        assert!(parse_polygons(text, unit_size(), true).is_empty(), "{text}");
        assert!(
            parse_phrase_grounding(text, unit_size()).is_empty(),
            "{text}"
        );
    }
}
