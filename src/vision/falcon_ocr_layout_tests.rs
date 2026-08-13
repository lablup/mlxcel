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

/// DocLayNet-derived detectors (docling-layout, and so anything routed through
/// `mlxcel detect`) spell the four two-word classes with an underscore where
/// PP-DocLayoutV3 hyphenates them. Both spellings must land on the same
/// category, or those regions would be dropped as unknown with no diagnostic.
#[test]
fn the_underscore_spelling_of_a_two_word_class_maps_the_same_way() {
    for (hyphenated, underscored) in [
        ("list-item", "list_item"),
        ("page-footer", "page_footer"),
        ("page-header", "page_header"),
        ("section-header", "section_header"),
    ] {
        let expected = layout_to_ocr_category(hyphenated);
        assert!(expected.is_some(), "{hyphenated} must map");
        assert_eq!(
            layout_to_ocr_category(underscored),
            expected,
            "{underscored} must map like {hyphenated}"
        );
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

/// The lazy planner decides viability from `crop_rect` alone, so the two must
/// not disagree: a box `crop_rect` accepts and `crop_region` rejects would be
/// planned and then fail to cut, and the reverse would silently drop a region.
#[test]
fn the_geometry_predicate_agrees_with_the_actual_crop() {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(400, 300));
    let sliver = image::DynamicImage::ImageRgb8(image::RgbImage::new(6000, 100));
    let cases: [(&image::DynamicImage, [f32; 4]); 5] = [
        // Ordinary interior box.
        (&img, [10.0, 10.0, 110.0, 60.0]),
        // Too small on a side.
        (&img, [0.0, 0.0, 8.0, 40.0]),
        // Extends past the right and bottom edges, so it clamps.
        (&img, [350.0, 250.0, 900.0, 700.0]),
        // Entirely past the edge, so it clamps to nothing.
        (&img, [500.0, 400.0, 800.0, 700.0]),
        // Sliver rejected by the post-resize check rather than by its raw size.
        (&sliver, [0.0, 0.0, 5000.0, 20.0]),
    ];
    for (image, bbox) in cases {
        let rect = crop_rect(image.width(), image.height(), &bbox, MIN_CROP_DIM, 1024);
        let cropped = crop_region(image, &bbox, MIN_CROP_DIM, 1024);
        match (rect, cropped) {
            (Some((_, _, w, h)), Some(cropped)) => {
                assert_eq!((w, h), (cropped.width(), cropped.height()), "{bbox:?}");
            }
            (None, None) => {}
            (rect, cropped) => panic!(
                "{bbox:?}: crop_rect gave {rect:?} but crop_region gave {:?}",
                cropped.map(|c| (c.width(), c.height()))
            ),
        }
    }
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

/// A DETR-style head emits one detection per class above threshold, all
/// carrying the same query's box. Those are alternative labels for one region,
/// so the planner must OCR that region once, not once per label (issue #1089).
#[test]
fn a_box_repeated_under_several_labels_becomes_one_region() {
    let img = page();
    // What `mlxcel detect` prints for a heading: one box, three labels,
    // highest confidence first.
    let heading = [10.0, 10.0, 300.0, 60.0];
    let dets = vec![
        det("section_header", heading, 0.71),
        det("title", heading, 0.41),
        det("page_header", heading, 0.40),
        det("text", [10.0, 100.0, 300.0, 200.0], 0.95),
    ];
    let plans = plan_layout_regions(&img, &dets, 0.8, MIN_CROP_DIM, 1024);
    assert_eq!(
        plans.len(),
        2,
        "the heading must plan as one region, not three: {:?}",
        plans.iter().map(|p| &p.category).collect::<Vec<_>>()
    );
    // The surviving label is the first one supplied for that box.
    assert_eq!(plans[0].category, "section_header");
    assert_eq!(plans[1].category, "text");
}

/// Collapsing duplicates must not drop a region whose highest-confidence label
/// happens to be un-OCRable: the dedup runs after the category mapping, so a
/// box labelled both `picture` and `text` is still read as text.
#[test]
fn a_duplicate_box_falls_through_to_its_first_ocrable_label() {
    let img = page();
    let region = [10.0, 10.0, 300.0, 120.0];
    let dets = vec![det("picture", region, 0.80), det("text", region, 0.55)];
    let plans = plan_layout_regions(&img, &dets, 0.8, MIN_CROP_DIM, 1024);
    assert_eq!(plans.len(), 1, "{plans:?}");
    assert_eq!(plans[0].category, "text");
    assert_eq!(plans[0].ocr_category, OcrCategory::Text);
}

/// Two genuinely distinct regions that share a bounding-box *size* are not
/// duplicates and must both survive.
#[test]
fn same_sized_boxes_at_different_positions_stay_separate() {
    let img = page();
    let dets = vec![
        det("text", [10.0, 10.0, 210.0, 110.0], 0.9),
        det("text", [10.0, 150.0, 210.0, 250.0], 0.9),
    ];
    let plans = plan_layout_regions(&img, &dets, 0.8, MIN_CROP_DIM, 1024);
    assert_eq!(plans.len(), 2, "{plans:?}");
}
