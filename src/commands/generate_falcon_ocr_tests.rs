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
use mlxcel::vision::falcon_ocr_layout::OcrCategory;

fn page() -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::new(1000, 1300))
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

/// The parsed file must reach the planner as OCR-ready regions: the picture is
/// dropped for carrying no text, and each survivor gets its category's
/// instruction.
#[test]
fn parsed_detections_plan_into_regions_with_their_own_instruction() {
    let detections = parse_layout_detections(DETECT_JSON).expect("parses");
    let regions = plan_layout_regions(
        &page(),
        &detections,
        CONTAINMENT_THRESHOLD,
        MIN_CROP_DIM,
        1024,
    );
    assert_eq!(regions.len(), 2, "the picture carries no text");
    assert_eq!(regions[0].ocr_category, OcrCategory::Title);
    assert_eq!(
        regions[0].ocr_category.instruction(),
        "Extract the title content from this image."
    );
    assert_eq!(regions[1].ocr_category, OcrCategory::Text);
    assert_eq!(
        (regions[1].image.width(), regions[1].image.height()),
        (880, 155)
    );
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
    let regions = plan_layout_regions(
        &page(),
        &detections,
        CONTAINMENT_THRESHOLD,
        MIN_CROP_DIM,
        1024,
    );
    let order: Vec<&str> = regions
        .iter()
        .map(|region| region.category.as_str())
        .collect();
    assert_eq!(order, ["page_footer", "title", "text"]);
}

/// A file whose every box is a figure has no text region, so the planner falls
/// back to the whole page and the summary says so instead of printing nothing.
#[test]
fn a_page_of_figures_falls_back_and_the_summary_says_so() {
    let json = r#"[{"label": "picture", "bbox": [0, 0, 500, 500]}]"#;
    let detections = parse_layout_detections(json).expect("parses");
    let regions = plan_layout_regions(
        &page(),
        &detections,
        CONTAINMENT_THRESHOLD,
        MIN_CROP_DIM,
        1024,
    );
    assert!(is_whole_page_fallback(&regions));
    let summary = plan_summary(&detections, &regions);
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
    let regions = plan_layout_regions(
        &page(),
        &detections,
        CONTAINMENT_THRESHOLD,
        MIN_CROP_DIM,
        1024,
    );
    assert_eq!(regions.len(), 1, "only the enclosing text box survives");
    let summary = plan_summary(&detections, &regions);
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
