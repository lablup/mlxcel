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

//! Task-to-result routing: the shape each task produces, and the reshaping
//! from parser instances into parallel arrays.

use super::*;
use crate::models::florence2::tasks::Florence2Task::*;

fn unit_size() -> Florence2ImageSize {
    Florence2ImageSize::new(1000, 1000)
}

fn locs(bins: &[i32]) -> String {
    bins.iter().map(|b| format!("<loc_{b}>")).collect()
}

#[test]
fn pure_text_strips_only_the_sequence_markers() {
    let result = post_process("<s>A green car on a street.</s>", Caption, unit_size());
    assert_eq!(
        result,
        Florence2TaskResult::Text("A green car on a street.".to_string())
    );
}

/// `<REGION_TO_CATEGORY>` answers a region question with a plain word, so it
/// must not be routed through a coordinate parser.
#[test]
fn region_questions_return_text() {
    let result = post_process("<s>car</s>", RegionToCategory, unit_size());
    assert_eq!(result, Florence2TaskResult::Text("car".to_string()));
}

#[test]
fn object_detection_returns_aligned_boxes_and_labels() {
    let text = format!(
        "<s>car{}person{}</s>",
        locs(&[52, 332, 932, 774]),
        locs(&[10, 20, 30, 40])
    );
    let Florence2TaskResult::Boxes { boxes, labels } =
        post_process(&text, ObjectDetection, unit_size())
    else {
        panic!("expected boxes");
    };
    assert_eq!(labels, vec!["car".to_string(), "person".to_string()]);
    assert_eq!(boxes.len(), labels.len());
    assert_eq!(boxes[0].to_array(), [52.5, 332.5, 932.5, 774.5]);
    assert_eq!(boxes[1].to_array(), [10.5, 20.5, 30.5, 40.5]);
}

#[test]
fn region_proposal_returns_unlabelled_boxes() {
    let text = format!("<s>{}{}</s>", locs(&[1, 2, 3, 4]), locs(&[5, 6, 7, 8]));
    let Florence2TaskResult::Boxes { boxes, labels } =
        post_process(&text, RegionProposal, unit_size())
    else {
        panic!("expected boxes");
    };
    assert_eq!(boxes.len(), 2);
    assert_eq!(labels, vec![String::new(), String::new()]);
}

/// Phrase grounding is flattened one label per box, so a phrase grounded by
/// two boxes contributes its label twice and the arrays stay index-aligned.
#[test]
fn phrase_grounding_flattens_one_label_per_box() {
    let text = format!(
        "<s>A green car{}{}a house{}</s>",
        locs(&[1, 2, 3, 4]),
        locs(&[5, 6, 7, 8]),
        locs(&[9, 10, 11, 12])
    );
    let Florence2TaskResult::Boxes { boxes, labels } =
        post_process(&text, CaptionToPhraseGrounding, unit_size())
    else {
        panic!("expected boxes");
    };
    assert_eq!(boxes.len(), 3);
    assert_eq!(
        labels,
        vec![
            "A green car".to_string(),
            "A green car".to_string(),
            "a house".to_string()
        ]
    );
}

#[test]
fn ocr_with_region_returns_quad_boxes() {
    let text = format!("<s>HELLO{}", locs(&[1, 2, 3, 4, 5, 6, 7, 8]));
    let Florence2TaskResult::QuadBoxes { quad_boxes, labels } =
        post_process(&text, OcrWithRegion, unit_size())
    else {
        panic!("expected quad boxes");
    };
    assert_eq!(labels, vec!["HELLO".to_string()]);
    assert_eq!(
        quad_boxes[0].points,
        [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    );
}

/// Plain `<OCR>` is text, `<OCR_WITH_REGION>` is quads. Routing these to the
/// same parser would silently drop every coordinate.
#[test]
fn plain_ocr_and_ocr_with_region_diverge() {
    let text = format!("<s>HELLO{}", locs(&[1, 2, 3, 4, 5, 6, 7, 8]));
    assert!(matches!(
        post_process(&text, Ocr, unit_size()),
        Florence2TaskResult::Text(_)
    ));
    assert!(matches!(
        post_process(&text, OcrWithRegion, unit_size()),
        Florence2TaskResult::QuadBoxes { .. }
    ));
}

#[test]
fn segmentation_returns_polygons_grouped_per_instance() {
    let text = format!("<s>{}<sep>{}</s>", locs(&[1, 2, 3, 4]), locs(&[5, 6, 7, 8]));
    let Florence2TaskResult::Polygons { polygons, labels } =
        post_process(&text, ReferringExpressionSegmentation, unit_size())
    else {
        panic!("expected polygons");
    };
    assert_eq!(labels.len(), 1);
    assert_eq!(polygons.len(), 1);
    assert_eq!(polygons[0].len(), 2, "both outlines stay on one instance");
}

/// `<OPEN_VOCABULARY_DETECTION>` picks its branch on whether `<poly>` appears
/// anywhere in the answer, so both arms have to be exercised.
#[test]
fn open_vocabulary_detection_switches_on_the_poly_marker() {
    let boxed = format!("<s>car{}</s>", locs(&[1, 2, 3, 4]));
    let Florence2TaskResult::BoxesOrPolygons {
        boxes,
        box_labels,
        polygons,
        polygon_labels,
    } = post_process(&boxed, OpenVocabularyDetection, unit_size())
    else {
        panic!("expected the mixed variant");
    };
    assert_eq!(box_labels, vec!["car".to_string()]);
    assert_eq!(boxes.len(), 1);
    assert!(polygons.is_empty() && polygon_labels.is_empty());

    let polyed = format!("<s>car<poly>{}</poly></s>", locs(&[1, 2, 3, 4]));
    let Florence2TaskResult::BoxesOrPolygons {
        boxes,
        polygons,
        polygon_labels,
        ..
    } = post_process(&polyed, OpenVocabularyDetection, unit_size())
    else {
        panic!("expected the mixed variant");
    };
    assert!(boxes.is_empty());
    assert_eq!(polygon_labels, vec!["car".to_string()]);
    assert_eq!(polygons.len(), 1);
}

/// Every task must produce the variant its post-processing type names, and
/// none may panic on an answer that carries no coordinates at all.
#[test]
fn every_task_routes_to_its_declared_variant() {
    use Florence2PostProcessingType as P;
    for task in Florence2Task::ALL {
        let result = post_process("<s>nothing here</s>", task, unit_size());
        let matches = matches!(
            (task.post_processing_type(), &result),
            (P::PureText, Florence2TaskResult::Text(_))
                | (P::Ocr, Florence2TaskResult::QuadBoxes { .. })
                | (
                    P::DescriptionWithBoxes | P::Boxes | P::PhraseGrounding,
                    Florence2TaskResult::Boxes { .. }
                )
                | (P::Polygons, Florence2TaskResult::Polygons { .. })
                | (
                    P::DescriptionWithBoxesOrPolygons,
                    Florence2TaskResult::BoxesOrPolygons { .. }
                )
        );
        assert!(matches, "{task} produced {result:?}");
    }
}

/// The image size is what turns bins into pixels, so passing the resized
/// 768x768 extent instead of the original must visibly change the answer.
/// This is the failure the named-field `Florence2ImageSize` exists to prevent.
#[test]
fn coordinates_scale_with_the_declared_image_size() {
    let text = format!("car{}", locs(&[500, 500, 999, 999]));
    let Florence2TaskResult::Boxes { boxes: small, .. } =
        post_process(&text, ObjectDetection, Florence2ImageSize::new(100, 200))
    else {
        panic!("expected boxes");
    };
    let Florence2TaskResult::Boxes { boxes: large, .. } =
        post_process(&text, ObjectDetection, Florence2ImageSize::new(1000, 2000))
    else {
        panic!("expected boxes");
    };
    assert!((small[0].xmin - 50.05).abs() < 1e-3, "{:?}", small[0]);
    assert!((small[0].ymin - 100.1).abs() < 1e-2, "{:?}", small[0]);
    assert!((large[0].xmin - 500.5).abs() < 1e-2, "{:?}", large[0]);
    assert!((large[0].ymin - 1001.0).abs() < 1e-1, "{:?}", large[0]);
}
