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

//! Turning a decoded Florence-2 answer into a typed result.
//!
//! This is the dispatch half of upstream's `post_process_generation`: it maps
//! a task to its parse behavior, runs the matching parser from
//! [`super::parse`], and reshapes the parser's instance list into the
//! parallel-array form callers consume.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py
//! (`Florence2Processor.post_process_generation`).

use super::coords::{Florence2BoundingBox, Florence2ImageSize, Florence2Polygon, Florence2QuadBox};
use super::parse;
use super::tasks::Florence2Task;

/// How a task's answer is parsed.
///
/// Upstream keys this off a string in `tasks_answer_post_processing_type`;
/// here it is an enum so [`Florence2Task::post_processing_type`] is total.
/// The `'od'` and `'description_with_polygons'` parse tasks upstream declares
/// are absent because no task routes to either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Florence2PostProcessingType {
    /// The answer is prose. Only the sequence markers are removed.
    PureText,
    /// `<OCR_WITH_REGION>`: text lines with quadrilaterals.
    Ocr,
    /// `<OD>`, `<DENSE_REGION_CAPTION>`: labelled boxes.
    DescriptionWithBoxes,
    /// `<REGION_PROPOSAL>`: boxes with no labels.
    Boxes,
    /// `<CAPTION_TO_PHRASE_GROUNDING>`: caption phrases with their boxes.
    PhraseGrounding,
    /// `<REFERRING_EXPRESSION_SEGMENTATION>`, `<REGION_TO_SEGMENTATION>`.
    Polygons,
    /// `<OPEN_VOCABULARY_DETECTION>`: boxes or polygons, whichever the model
    /// chose to emit.
    DescriptionWithBoxesOrPolygons,
}

/// A parsed Florence-2 answer.
///
/// `#[non_exhaustive]` because the variant set follows the task set, and
/// upstream has grown parse tasks (`description_with_polygons`) that no
/// released task routes to yet.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Florence2TaskResult {
    /// Prose answer, sequence markers removed.
    Text(String),
    /// Boxes with one label each. Labels are empty strings for
    /// `<REGION_PROPOSAL>`, which predicts regions without naming them. For
    /// `<CAPTION_TO_PHRASE_GROUNDING>` a phrase grounded by several boxes
    /// appears once per box, so `boxes` and `labels` stay index-aligned.
    Boxes {
        boxes: Vec<Florence2BoundingBox>,
        labels: Vec<String>,
    },
    /// OCR lines: one quadrilateral and one text string each.
    QuadBoxes {
        quad_boxes: Vec<Florence2QuadBox>,
        labels: Vec<String>,
    },
    /// Segmentation: one label per instance, and one or more outlines per
    /// instance (a disconnected object needs several).
    Polygons {
        polygons: Vec<Vec<Florence2Polygon>>,
        labels: Vec<String>,
    },
    /// `<OPEN_VOCABULARY_DETECTION>`. Upstream returns all four arrays
    /// unconditionally and leaves the unused pair empty, so both are kept
    /// rather than collapsing to an either/or.
    BoxesOrPolygons {
        boxes: Vec<Florence2BoundingBox>,
        box_labels: Vec<String>,
        polygons: Vec<Vec<Florence2Polygon>>,
        polygon_labels: Vec<String>,
    },
}

impl Florence2TaskResult {
    /// Parse `text`, the answer decoded with special tokens kept, into the
    /// structured form for `task`.
    ///
    /// `size` is the size of the image **as it was handed to the processor**,
    /// not the 768x768 resized tensor: location bins are relative to the
    /// original extent, so passing the resized size silently scales every
    /// coordinate.
    ///
    /// Free of any model or tokenizer state, so a caller holding an answer
    /// from elsewhere can parse it without loading a checkpoint.
    /// [`super::Florence2Processor::post_process`] is the same call reached
    /// through a loaded processor.
    pub fn parse(text: &str, task: Florence2Task, size: Florence2ImageSize) -> Self {
        post_process(text, task, size)
    }
}

pub(crate) fn post_process(
    text: &str,
    task: Florence2Task,
    size: Florence2ImageSize,
) -> Florence2TaskResult {
    use Florence2PostProcessingType as P;
    match task.post_processing_type() {
        P::PureText => Florence2TaskResult::Text(strip_sequence_markers(text)),
        P::Ocr => {
            let instances = parse::parse_ocr(text, size);
            let mut quad_boxes = Vec::with_capacity(instances.len());
            let mut labels = Vec::with_capacity(instances.len());
            for instance in instances {
                quad_boxes.push(instance.quad_box);
                labels.push(instance.text);
            }
            Florence2TaskResult::QuadBoxes { quad_boxes, labels }
        }
        P::DescriptionWithBoxes => boxes_result(parse::parse_boxes(text, size, false)),
        P::Boxes => boxes_result(parse::parse_boxes(text, size, true)),
        P::PhraseGrounding => {
            let mut boxes = Vec::new();
            let mut labels = Vec::new();
            for phrase in parse::parse_phrase_grounding(text, size) {
                for bbox in phrase.boxes {
                    boxes.push(bbox);
                    labels.push(phrase.cat_name.clone());
                }
            }
            Florence2TaskResult::Boxes { boxes, labels }
        }
        P::Polygons => {
            let instances = parse::parse_polygons(text, size, true);
            let mut polygons = Vec::with_capacity(instances.len());
            let mut labels = Vec::with_capacity(instances.len());
            for instance in instances {
                polygons.push(instance.polygons);
                labels.push(instance.cat_name);
            }
            Florence2TaskResult::Polygons { polygons, labels }
        }
        // Upstream picks the branch on a bare `'<poly>' in text` check rather
        // than on anything structural, so a single answer is read as entirely
        // polygons or entirely boxes.
        P::DescriptionWithBoxesOrPolygons => {
            if text.contains("<poly>") {
                let instances = parse::parse_polygons(text, size, false);
                let mut polygons = Vec::with_capacity(instances.len());
                let mut polygon_labels = Vec::with_capacity(instances.len());
                for instance in instances {
                    polygons.push(instance.polygons);
                    polygon_labels.push(instance.cat_name);
                }
                Florence2TaskResult::BoxesOrPolygons {
                    boxes: Vec::new(),
                    box_labels: Vec::new(),
                    polygons,
                    polygon_labels,
                }
            } else {
                let instances = parse::parse_boxes(text, size, false);
                let mut boxes = Vec::with_capacity(instances.len());
                let mut box_labels = Vec::with_capacity(instances.len());
                for instance in instances {
                    boxes.push(instance.bbox);
                    box_labels.push(instance.cat_name);
                }
                Florence2TaskResult::BoxesOrPolygons {
                    boxes,
                    box_labels,
                    polygons: Vec::new(),
                    polygon_labels: Vec::new(),
                }
            }
        }
    }
}

fn boxes_result(instances: Vec<parse::BoxInstance>) -> Florence2TaskResult {
    let mut boxes = Vec::with_capacity(instances.len());
    let mut labels = Vec::with_capacity(instances.len());
    for instance in instances {
        boxes.push(instance.bbox);
        labels.push(instance.cat_name);
    }
    Florence2TaskResult::Boxes { boxes, labels }
}

/// `pure_text` strips only `<s>` and `</s>`, not `<pad>`, which is what
/// upstream's `final_answer.replace('<s>', '').replace('</s>', '')` does. The
/// box parsers strip `<pad>` as well; the asymmetry is upstream's.
fn strip_sequence_markers(text: &str) -> String {
    text.replace("<s>", "").replace("</s>", "")
}

#[cfg(test)]
#[path = "florence2_postprocess_tests.rs"]
mod florence2_postprocess_tests;
