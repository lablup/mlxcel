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

//! `--layout-detections` parsing and region-planning tests.

use super::*;
use mlxcel::vision::falcon_ocr_layout::{OcrCategory, plan_layout_regions};

const PAGE_WIDTH: u32 = 1000;
const PAGE_HEIGHT: u32 = 1300;

fn page() -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::new(PAGE_WIDTH, PAGE_HEIGHT))
}

/// Plan against the standard test page, the way the driver does.
fn plan(detections: &[Detection]) -> Vec<LayoutOcrPlan> {
    plan_layout_region_boxes(
        PAGE_WIDTH,
        PAGE_HEIGHT,
        detections,
        CONTAINMENT_THRESHOLD,
        MIN_CROP_DIM,
        1024,
    )
}

fn kept(detections: &[Detection]) -> Vec<Detection> {
    filter_nested_detections(detections, CONTAINMENT_THRESHOLD)
}

/// The exact shape `mlxcel detect --format json` prints, so the two commands
/// compose without a translation step.
const DETECT_JSON: &str = r#"{
  "image": "page.png",
  "threshold": 0.4,
  "detections": [
    {"label": "title", "label_id": 10, "confidence": 0.95,
     "box": {"l": 60.0, "t": 55.0, "r": 940.0, "b": 140.0}},
    {"label": "text", "label_id": 9, "confidence": 0.9,
     "box": {"l": 60.0, "t": 295.0, "r": 940.0, "b": 450.0}},
    {"label": "picture", "label_id": 6, "confidence": 0.8,
     "box": {"l": 60.0, "t": 600.0, "r": 500.0, "b": 900.0}}
  ]
}"#;

#[test]
fn the_detect_subcommand_json_shape_parses() {
    let parsed = parse_layout_detections(DETECT_JSON).expect("detect JSON parses");
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0].class_name, "title");
    assert_eq!(parsed[0].label, 10);
    assert!((parsed[0].score - 0.95).abs() < 1e-6);
    assert_eq!(parsed[0].bbox, [60.0, 55.0, 940.0, 140.0]);
    assert_eq!(parsed[2].class_name, "picture");
}

/// mlx-vlm's `falcon_ocr/layout.py` emits `category` / `bbox` / `score`; a
/// hand-written file is easiest as a bare array. Both must land on the same
/// `Detection`.
#[test]
fn the_reference_layout_py_shape_and_a_bare_array_parse() {
    let json = r#"[
      {"category": "doc_title", "bbox": [10, 20, 300, 80], "score": 0.7},
      {"category": "table", "bbox": [10, 100, 300, 400], "score": 0.6}
    ]"#;
    let parsed = parse_layout_detections(json).expect("bare array parses");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].class_name, "doc_title");
    assert_eq!(parsed[0].bbox, [10.0, 20.0, 300.0, 80.0]);
    assert!((parsed[1].score - 0.6).abs() < 1e-6);
    // No `label_id` in this spelling: the class id defaults rather than failing.
    assert_eq!(parsed[1].label, 0);
}

/// A missing confidence is not an error: a hand-written box is still a box.
#[test]
fn a_detection_without_a_score_defaults_to_full_confidence() {
    let parsed = parse_layout_detections(r#"[{"label": "text", "bbox": [0, 0, 100, 100]}]"#)
        .expect("score is optional");
    assert_eq!(parsed[0].score, 1.0);
}

#[test]
fn alternative_box_key_spellings_parse() {
    let parsed = parse_layout_detections(
        r#"[{"label": "text", "box": {"x1": 1, "y1": 2, "x2": 30, "y2": 40}}]"#,
    )
    .expect("x1/y1/x2/y2 parses");
    assert_eq!(parsed[0].bbox, [1.0, 2.0, 30.0, 40.0]);
}

#[test]
fn malformed_json_is_rejected_with_a_parse_error() {
    let error = parse_layout_detections("{ this is not json").expect_err("must fail");
    assert!(error.to_string().contains("not valid JSON"), "got: {error}");
}

#[test]
fn a_document_without_a_detections_array_names_the_expected_shape() {
    let error = parse_layout_detections(r#"{"image": "page.png"}"#).expect_err("must fail");
    let message = error.to_string();
    assert!(message.contains("detections"), "got: {message}");
    assert!(message.contains("mlxcel detect"), "got: {message}");
}

#[test]
fn a_non_array_detections_field_is_rejected() {
    let error = parse_layout_detections(r#"{"detections": 5}"#).expect_err("must fail");
    assert!(
        error.to_string().contains("must be an array"),
        "got: {error}"
    );
}

/// Every per-entry rejection names the index: a detector dump is long enough
/// that "invalid input" alone cannot be acted on.
#[test]
fn a_detection_missing_its_class_name_names_the_index() {
    let json = r#"[{"label": "text", "bbox": [0, 0, 10, 10]}, {"bbox": [0, 0, 10, 10]}]"#;
    let error = parse_layout_detections(json).expect_err("must fail");
    let message = error.to_string();
    assert!(message.contains("detection #1"), "got: {message}");
    assert!(message.contains("class name"), "got: {message}");
}

#[test]
fn a_numeric_class_name_is_reported_as_a_type_error_not_a_missing_field() {
    let error = parse_layout_detections(r#"[{"label": 9, "bbox": [0, 0, 10, 10]}]"#)
        .expect_err("must fail");
    let message = error.to_string();
    assert!(message.contains("must be a string"), "got: {message}");
}

#[test]
fn a_detection_missing_its_box_names_the_index() {
    let error = parse_layout_detections(r#"[{"label": "text"}]"#).expect_err("must fail");
    let message = error.to_string();
    assert!(message.contains("detection #0"), "got: {message}");
    assert!(message.contains("bounding box"), "got: {message}");
}

#[test]
fn a_short_box_array_is_rejected() {
    let error = parse_layout_detections(r#"[{"label": "text", "bbox": [0, 0, 10]}]"#)
        .expect_err("must fail");
    assert!(error.to_string().contains("exactly four"), "got: {error}");
}

#[test]
fn a_non_numeric_coordinate_is_rejected() {
    let error = parse_layout_detections(r#"[{"label": "text", "bbox": [0, 0, "10", 10]}]"#)
        .expect_err("must fail");
    assert!(
        error.to_string().contains("must be a number"),
        "got: {error}"
    );
}

/// An inverted box would crop to nothing and vanish silently, so it is a hard
/// error rather than a dropped region.
#[test]
fn an_inverted_box_is_rejected() {
    let error = parse_layout_detections(r#"[{"label": "text", "bbox": [100, 0, 10, 50]}]"#)
        .expect_err("must fail");
    let message = error.to_string();
    assert!(message.contains("inverted"), "got: {message}");
}

#[test]
fn a_non_object_entry_is_rejected() {
    let error = parse_layout_detections(r#"["text"]"#).expect_err("must fail");
    assert!(
        error.to_string().contains("expected an object"),
        "got: {error}"
    );
}

#[test]
fn a_top_level_scalar_is_rejected() {
    let error = parse_layout_detections("42").expect_err("must fail");
    assert!(error.to_string().contains("top level"), "got: {error}");
}

fn detections_json(count: usize) -> String {
    let entries: Vec<String> = (0..count)
        .map(|_| r#"{"label": "text", "bbox": [0, 0, 100, 100]}"#.to_string())
        .collect();
    format!("[{}]", entries.join(","))
}

/// The nested-box suppression is quadratic, so an unbounded array turns a small
/// file into minutes of comparisons.
#[test]
fn an_array_over_the_detection_cap_is_rejected_with_both_numbers() {
    let json = detections_json(MAX_LAYOUT_DETECTIONS + 1);
    let error = parse_layout_detections(&json).expect_err("must fail");
    let message = error.to_string();
    assert!(
        message.contains(&(MAX_LAYOUT_DETECTIONS + 1).to_string()),
        "the actual count must be named, got: {message}"
    );
    assert!(
        message.contains(&MAX_LAYOUT_DETECTIONS.to_string()),
        "the cap must be named, got: {message}"
    );
}

/// The cap is inclusive; pinned so a later refactor cannot quietly turn it into
/// an off-by-one that rejects a legal file.
#[test]
fn exactly_the_detection_cap_still_parses() {
    let json = detections_json(MAX_LAYOUT_DETECTIONS);
    let parsed = parse_layout_detections(&json).expect("the cap itself is allowed");
    assert_eq!(parsed.len(), MAX_LAYOUT_DETECTIONS);
}

/// The class name reaches the operator's terminal verbatim through the
/// summary's `no text category:` list, so an escape sequence in it would repaint
/// their screen. This is the injection guard.
#[test]
fn a_class_name_with_a_control_character_is_rejected() {
    for raw in ["te\u{1b}[2Jxt", "te\u{7}xt", "te\nxt"] {
        let json = format!(
            r#"[{{"label": "ok", "bbox": [0, 0, 10, 10]}}, {{"label": {}, "bbox": [0, 0, 10, 10]}}]"#,
            serde_json::to_string(raw).expect("string encodes")
        );
        let error = parse_layout_detections(&json).expect_err("must fail: {raw:?}");
        let message = error.to_string();
        assert!(message.contains("detection #1"), "got: {message}");
        assert!(message.contains("control character"), "got: {message}");
    }
}

#[test]
fn an_over_long_class_name_is_rejected() {
    let name = "t".repeat(MAX_LAYOUT_CLASS_NAME_CHARS + 1);
    let json = format!(
        r#"[{{"label": "ok", "bbox": [0, 0, 10, 10]}}, {{"label": "{name}", "bbox": [0, 0, 10, 10]}}]"#
    );
    let error = parse_layout_detections(&json).expect_err("must fail");
    let message = error.to_string();
    assert!(message.contains("detection #1"), "got: {message}");
    assert!(message.contains("longer than"), "got: {message}");
}

/// An exponent past `f64::MAX` must not reach the planner as an infinite
/// coordinate. The test pins "rejected either way" rather than a specific
/// serde_json behavior: depending on the version, `1e400` either fails to parse
/// as a number at all or yields an infinite one that the finiteness check
/// catches.
#[test]
fn a_coordinate_that_overflows_to_infinity_is_rejected() {
    let error = parse_layout_detections(r#"[{"label": "text", "bbox": [0, 0, 1e400, 10]}]"#)
        .expect_err("must fail");
    let message = error.to_string();
    assert!(
        message.contains("not valid JSON") || message.contains("non-finite"),
        "got: {message}"
    );
}

/// The parsed file must reach the planner as OCR-ready regions: the picture is
/// dropped for carrying no text, and each survivor gets its category's
/// instruction.
#[test]
fn parsed_detections_plan_into_regions_with_their_own_instruction() {
    let detections = parse_layout_detections(DETECT_JSON).expect("parses");
    let plans = plan(&detections);
    assert_eq!(plans.len(), 2, "the picture carries no text");
    assert_eq!(plans[0].ocr_category, OcrCategory::Title);
    assert_eq!(
        plans[0].ocr_category.instruction(),
        "Extract the title content from this image."
    );
    assert_eq!(plans[1].ocr_category, OcrCategory::Text);
    assert_eq!(plans[1].crop, (60, 295, 880, 155));
}

/// The driver plans without cropping and cuts one region at a time, so the
/// lazy planner has to agree with the eager one on every decision: which boxes
/// survive, in what order, and what rectangle each one covers.
#[test]
fn the_lazy_plan_matches_the_eager_planner_region_for_region() {
    let detections = parse_layout_detections(DETECT_JSON).expect("parses");
    let plans = plan(&detections);
    let regions = plan_layout_regions(
        &page(),
        &detections,
        CONTAINMENT_THRESHOLD,
        MIN_CROP_DIM,
        1024,
    );

    assert_eq!(plans.len(), regions.len());
    for (plan, region) in plans.iter().zip(regions.iter()) {
        assert_eq!(plan.category, region.category);
        assert_eq!(plan.bbox, region.bbox);
        assert_eq!(plan.score, region.score);
        assert_eq!(plan.ocr_category, region.ocr_category);
        let (_, _, width, height) = plan.crop;
        assert_eq!(
            (width, height),
            (region.image.width(), region.image.height()),
            "the planned rect must be the rect the eager crop produced"
        );
    }
}

/// Nothing reorders the detections, so the file's order is the output order.
/// A caller that wants reading order supplies reading order.
#[test]
fn regions_follow_the_files_order_not_a_geometric_sort() {
    let json = r#"[
      {"label": "page_footer", "bbox": [60, 1205, 400, 1260]},
      {"label": "title", "bbox": [60, 55, 940, 140]},
      {"label": "text", "bbox": [60, 295, 940, 450]}
    ]"#;
    let detections = parse_layout_detections(json).expect("parses");
    let order: Vec<String> = plan(&detections)
        .into_iter()
        .map(|plan| plan.category)
        .collect();
    assert_eq!(order, ["page_footer", "title", "text"]);
}

/// A file whose every box is a figure has no text region, so the planner falls
/// back to the whole page and the summary says so instead of printing nothing.
#[test]
fn a_page_of_figures_falls_back_and_the_summary_says_so() {
    let json = r#"[{"label": "picture", "bbox": [0, 0, 500, 500]}]"#;
    let detections = parse_layout_detections(json).expect("parses");
    let plans = plan(&detections);
    assert!(is_whole_page_fallback(&plans));
    let summary = plan_summary(&detections, &kept(&detections), &plans);
    assert!(summary.contains("whole page"), "got: {summary}");
}

#[test]
fn the_summary_reports_nested_and_textless_boxes() {
    let json = r#"[
      {"label": "text", "bbox": [0, 0, 400, 200]},
      {"label": "formula", "bbox": [10, 10, 100, 60]},
      {"label": "chart", "bbox": [0, 400, 400, 800]}
    ]"#;
    let detections = parse_layout_detections(json).expect("parses");
    let plans = plan(&detections);
    assert_eq!(plans.len(), 1, "only the enclosing text box survives");
    let summary = plan_summary(&detections, &kept(&detections), &plans);
    assert!(summary.contains("3 detection(s)"), "got: {summary}");
    assert!(
        summary.contains("1 nested box(es) dropped"),
        "got: {summary}"
    );
    assert!(
        summary.contains("no text category: chart"),
        "got: {summary}"
    );
    assert!(summary.contains("1 region(s) to OCR"), "got: {summary}");
}

/// `generate_with_layout` strips the task terminators before returning a
/// region's text; leaving them in would put raw markers in the page output.
#[test]
fn region_answers_drop_the_task_terminators() {
    assert_eq!(
        clean_region_answer("  Quarterly Revenue Report<|end_of_query|><|end_of_text|>  "),
        "Quarterly Revenue Report"
    );
    assert_eq!(clean_region_answer("plain text"), "plain text");
}

/// Verbatim `mlxcel detect --format json` output for a real multi-region page:
/// a heading, a body paragraph, and a footer on a 1000x1300 portrait page, as
/// produced by the docling-layout-heron checkpoint after the readback fix in
/// issue #1089. Kept literal, not regenerated, so a change to either command's
/// contract shows up here.
const DETECT_PIPELINE_JSON: &str = r#"{
  "image": "page.png",
  "threshold": 0.30000001192092896,
  "detections": [
    {"label": "section_header", "label_id": 7, "confidence": 0.6150878667831421,
     "box": {"l": 80.078125, "t": 58.3984375, "r": 771.484375, "b": 105.37109375}},
    {"label": "title", "label_id": 10, "confidence": 0.4148988723754883,
     "box": {"l": 80.078125, "t": 58.3984375, "r": 771.484375, "b": 105.37109375}},
    {"label": "list_item", "label_id": 3, "confidence": 0.31742626428604126,
     "box": {"l": 80.078125, "t": 58.3984375, "r": 771.484375, "b": 105.37109375}},
    {"label": "text", "label_id": 9, "confidence": 0.9149009585380554,
     "box": {"l": 62.5, "t": 197.412109375, "r": 953.125, "b": 335.791015625}},
    {"label": "page_footer", "label_id": 4, "confidence": 0.5,
     "box": {"l": 78.125, "t": 1200.341796875, "r": 671.875, "b": 1237.158203125}}
  ]
}"#;

/// `mlxcel detect --format json | mlxcel generate --layout-detections` must
/// plan one region per page element, in reading order (issue #1089).
///
/// This pins the composition of the two commands at the seam they share: the
/// detector's JSON goes in unmodified, and what comes out is the region list
/// the per-region OCR loop walks, in the order it prints them. Three things
/// have to hold at once, and each was broken or unverified before: the boxes
/// must be real page geometry rather than one collapsed rectangle, the heading
/// must be OCRed once rather than once per label, and the footer must come last
/// rather than first (it outranks nothing on confidence, but it is last on the
/// page).
#[test]
fn detect_json_plans_into_reading_ordered_regions_without_repeats() {
    let detections = parse_layout_detections(DETECT_PIPELINE_JSON).expect("detect JSON parses");
    assert_eq!(detections.len(), 5, "all five entries survive parsing");

    let plans = plan(&detections);

    let categories: Vec<&str> = plans.iter().map(|p| p.category.as_str()).collect();
    assert_eq!(
        categories,
        vec!["section_header", "text", "page_footer"],
        "one region per page element, in reading order"
    );

    // Reading order: strictly increasing down the page.
    let tops: Vec<f32> = plans.iter().map(|p| p.bbox[1]).collect();
    assert!(
        tops.windows(2).all(|w| w[0] < w[1]),
        "regions must run down the page, got {tops:?}"
    );

    // Each region is a distinct rectangle: no collapse onto one box.
    for (i, a) in plans.iter().enumerate() {
        for b in plans.iter().skip(i + 1) {
            assert_ne!(a.bbox, b.bbox, "two regions share a box");
        }
    }

    // The footer reaches the bottom of a 1300-tall page. Before the fix every
    // detection shared one box that stopped near the middle of the page.
    let footer = plans.last().expect("a footer region");
    assert_eq!(footer.category, "page_footer");
    assert!(
        footer.bbox[3] > 1200.0,
        "footer must reach the bottom of the page, got {:?}",
        footer.bbox
    );

    // Every region crops to something OCRable.
    for plan in &plans {
        let (_, _, w, h) = plan.crop;
        assert!(w >= MIN_CROP_DIM && h >= MIN_CROP_DIM, "{plan:?}");
    }
}
