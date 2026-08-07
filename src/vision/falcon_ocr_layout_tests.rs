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

//! Falcon-OCR layout post-processing tests.

use super::*;

fn det(class_name: &str, bbox: [f32; 4], score: f32) -> Detection {
    Detection {
        bbox,
        score,
        label: 0,
        class_name: class_name.to_string(),
    }
}

fn page() -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::new(400, 300))
}

#[test]
fn layout_classes_route_to_the_reference_ocr_categories() {
    assert_eq!(layout_to_ocr_category("table"), Some(OcrCategory::Table));
    assert_eq!(
        layout_to_ocr_category("formula"),
        Some(OcrCategory::Formula)
    );
    // Aliases that collapse onto a broader category.
    assert_eq!(layout_to_ocr_category("abstract"), Some(OcrCategory::Text));
    assert_eq!(
        layout_to_ocr_category("doc_title"),
        Some(OcrCategory::Title)
    );
    assert_eq!(
        layout_to_ocr_category("paragraph_title"),
        Some(OcrCategory::SectionHeader)
    );
    assert_eq!(
        layout_to_ocr_category("footer"),
        Some(OcrCategory::PageFooter)
    );
    assert_eq!(
        layout_to_ocr_category("figure_title"),
        Some(OcrCategory::Caption)
    );
    // Textless regions must be skipped, not defaulted to text.
    for skip in ["image", "picture", "figure", "chart", "seal"] {
        assert_eq!(layout_to_ocr_category(skip), None, "{skip} must be skipped");
    }
}

#[test]
fn every_category_keeps_its_reference_instruction() {
    assert_eq!(
        OcrCategory::Plain.instruction(),
        "Extract the text content from this image."
    );
    assert_eq!(
        OcrCategory::Table.instruction(),
        "Extract the table content from this image."
    );
    assert_eq!(
        OcrCategory::SectionHeader.instruction(),
        "Extract the section-header content from this image."
    );
    assert_eq!(OcrCategory::parse("List-Item"), Some(OcrCategory::ListItem));
    assert_eq!(OcrCategory::parse("nonsense"), None);
}

/// An inline formula detected inside a paragraph must be dropped, or the page
/// text contains it twice.
#[test]
fn a_box_mostly_inside_a_larger_box_is_dropped() {
    let dets = vec![
        det("text", [0.0, 0.0, 200.0, 100.0], 0.9),
        det("formula", [10.0, 10.0, 60.0, 40.0], 0.8),
    ];
    let kept = filter_nested_detections(&dets, 0.8);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].class_name, "text");
}

#[test]
fn partially_overlapping_boxes_both_survive() {
    let dets = vec![
        det("text", [0.0, 0.0, 100.0, 100.0], 0.9),
        det("table", [90.0, 0.0, 300.0, 100.0], 0.8),
    ];
    assert_eq!(filter_nested_detections(&dets, 0.8).len(), 2);
}

/// Equal-area boxes cannot nest each other: the reference requires a strictly
/// larger container, otherwise two duplicates would delete each other.
#[test]
fn identical_boxes_do_not_delete_each_other() {
    let dets = vec![
        det("text", [0.0, 0.0, 100.0, 100.0], 0.9),
        det("text", [0.0, 0.0, 100.0, 100.0], 0.8),
    ];
    assert_eq!(filter_nested_detections(&dets, 0.8).len(), 2);
}

#[test]
fn a_region_smaller_than_the_minimum_is_not_cropped() {
    let img = page();
    assert!(crop_region(&img, &[0.0, 0.0, 8.0, 40.0], MIN_CROP_DIM, 1024).is_none());
    let cropped = crop_region(&img, &[10.0, 10.0, 110.0, 60.0], MIN_CROP_DIM, 1024)
        .expect("large enough region crops");
    assert_eq!((cropped.width(), cropped.height()), (100, 50));
}

/// A sliver whose short side would collapse below the minimum once the long
/// side is capped is rejected up front.
#[test]
fn a_sliver_that_would_collapse_after_resize_is_rejected() {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(6000, 100));
    assert!(crop_region(&img, &[0.0, 0.0, 5000.0, 20.0], MIN_CROP_DIM, 1024).is_none());
}

#[test]
fn a_page_without_detections_falls_back_to_a_single_plain_region() {
    let img = page();
    let regions = plan_layout_regions(&img, &[], 0.8, MIN_CROP_DIM, 1024);
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].ocr_category, OcrCategory::Plain);
    assert_eq!(regions[0].bbox, [0.0, 0.0, 400.0, 300.0]);
}

/// A page whose only detection is a figure has no text region, so the reference
/// OCRs the whole page rather than returning nothing.
#[test]
fn a_page_with_only_a_figure_falls_back_to_the_whole_page() {
    let img = page();
    let dets = vec![det("image", [0.0, 0.0, 400.0, 300.0], 0.9)];
    let regions = plan_layout_regions(&img, &dets, 0.8, MIN_CROP_DIM, 1024);
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].category, "plain");
}

#[test]
fn detections_become_ordered_regions_with_their_own_instruction() {
    let img = page();
    let dets = vec![
        det("doc_title", [0.0, 0.0, 300.0, 40.0], 0.95),
        det("table", [0.0, 60.0, 300.0, 200.0], 0.8),
        det("chart", [310.0, 60.0, 390.0, 200.0], 0.7),
    ];
    let regions = plan_layout_regions(&img, &dets, 0.8, MIN_CROP_DIM, 1024);
    assert_eq!(regions.len(), 2, "the chart carries no text");
    assert_eq!(regions[0].ocr_category, OcrCategory::Title);
    assert_eq!(regions[1].ocr_category, OcrCategory::Table);
    assert_eq!(
        regions[1].ocr_category.instruction(),
        "Extract the table content from this image."
    );
    assert_eq!(
        (regions[1].image.width(), regions[1].image.height()),
        (300, 140)
    );
}
