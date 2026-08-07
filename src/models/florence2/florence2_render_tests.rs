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

//! Tests for the Florence-2 answer presentation layer: the CLI/server text
//! renderer (moved here with the renderer when the server seq2seq worker
//! landed, issue #1073) and the structured JSON mapping.

use super::*;
use crate::models::florence2::{
    Florence2BoundingBox, Florence2Polygon, Florence2QuadBox, Florence2Task, Florence2TaskResult,
};

// ---------------------------------------------------------------------------
// render_task_result (text form)
// ---------------------------------------------------------------------------

#[test]
fn renders_text_result() {
    let result = Florence2TaskResult::Text("A car parked on the street.".to_string());
    assert_eq!(
        render_task_result(&result, ""),
        "A car parked on the street."
    );
}

#[test]
fn renders_boxes_one_instance_per_line() {
    let result = Florence2TaskResult::Boxes {
        boxes: vec![
            Florence2BoundingBox {
                xmin: 1.0,
                ymin: 2.0,
                xmax: 3.0,
                ymax: 4.0,
            },
            Florence2BoundingBox {
                xmin: 5.0,
                ymin: 6.0,
                xmax: 7.0,
                ymax: 8.0,
            },
        ],
        labels: vec!["car".to_string(), String::new()],
    };
    let rendered = render_task_result(&result, "");
    assert_eq!(
        rendered,
        "car: [1.0, 2.0, 3.0, 4.0]\n(region): [5.0, 6.0, 7.0, 8.0]"
    );
}

#[test]
fn renders_quad_boxes_and_polygons() {
    let quads = Florence2TaskResult::QuadBoxes {
        quad_boxes: vec![Florence2QuadBox {
            points: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }],
        labels: vec!["HELLO".to_string()],
    };
    assert_eq!(
        render_task_result(&quads, ""),
        "HELLO: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]"
    );

    let polygons = Florence2TaskResult::Polygons {
        polygons: vec![vec![Florence2Polygon {
            points: vec![1.0, 2.0, 3.0, 4.0],
        }]],
        labels: vec!["dog".to_string()],
    };
    assert_eq!(
        render_task_result(&polygons, ""),
        "dog: [1.0, 2.0, 3.0, 4.0]"
    );
}

#[test]
fn renders_boxes_or_polygons_box_branch() {
    let result = Florence2TaskResult::BoxesOrPolygons {
        boxes: vec![Florence2BoundingBox {
            xmin: 1.0,
            ymin: 2.0,
            xmax: 3.0,
            ymax: 4.0,
        }],
        box_labels: vec!["cat".to_string()],
        polygons: vec![],
        polygon_labels: vec![],
    };
    assert_eq!(render_task_result(&result, ""), "cat: [1.0, 2.0, 3.0, 4.0]");
}

#[test]
fn empty_result_falls_back_to_raw_text() {
    let result = Florence2TaskResult::Boxes {
        boxes: vec![],
        labels: vec![],
    };
    let rendered = render_task_result(&result, "<s></s>");
    assert!(rendered.contains("raw answer"), "got: {rendered}");
}

// ---------------------------------------------------------------------------
// structured_task_json (JSON form)
// ---------------------------------------------------------------------------

#[test]
fn json_text_result_carries_trimmed_text() {
    let result = Florence2TaskResult::Text("  Two cats on a couch.  ".to_string());
    let value = structured_task_json(Florence2Task::Caption, &result);
    assert_eq!(value["task"], "<CAPTION>");
    assert_eq!(value["kind"], "text");
    assert_eq!(value["text"], "Two cats on a couch.");
}

#[test]
fn json_boxes_use_upstream_bboxes_labels_keys() {
    let result = Florence2TaskResult::Boxes {
        boxes: vec![Florence2BoundingBox {
            xmin: 9.5,
            ymin: 54.0,
            xmax: 316.0,
            ymax: 474.0,
        }],
        labels: vec!["cat".to_string()],
    };
    let value = structured_task_json(Florence2Task::ObjectDetection, &result);
    assert_eq!(value["task"], "<OD>");
    assert_eq!(value["kind"], "bboxes");
    assert_eq!(value["bboxes"][0][0], 9.5);
    assert_eq!(value["bboxes"][0][3], 474.0);
    assert_eq!(value["labels"][0], "cat");
    assert_eq!(value["bboxes"].as_array().unwrap().len(), 1);
}

#[test]
fn json_quad_boxes_flatten_eight_points() {
    let result = Florence2TaskResult::QuadBoxes {
        quad_boxes: vec![Florence2QuadBox {
            points: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }],
        labels: vec!["HELLO".to_string()],
    };
    let value = structured_task_json(Florence2Task::OcrWithRegion, &result);
    assert_eq!(value["kind"], "quad_boxes");
    assert_eq!(value["quad_boxes"][0].as_array().unwrap().len(), 8);
    assert_eq!(value["quad_boxes"][0][7], 8.0);
    assert_eq!(value["labels"][0], "HELLO");
}

#[test]
fn json_polygons_keep_per_instance_outline_nesting() {
    let result = Florence2TaskResult::Polygons {
        polygons: vec![vec![
            Florence2Polygon {
                points: vec![1.0, 2.0, 3.0, 4.0],
            },
            Florence2Polygon {
                points: vec![5.0, 6.0, 7.0, 8.0],
            },
        ]],
        labels: vec!["dog".to_string()],
    };
    let value = structured_task_json(Florence2Task::ReferringExpressionSegmentation, &result);
    assert_eq!(value["kind"], "polygons");
    // one instance, two disconnected outlines
    assert_eq!(value["polygons"].as_array().unwrap().len(), 1);
    assert_eq!(value["polygons"][0].as_array().unwrap().len(), 2);
    assert_eq!(value["polygons"][0][1][0], 5.0);
    assert_eq!(value["labels"][0], "dog");
}

#[test]
fn json_boxes_or_polygons_carries_all_four_arrays() {
    let result = Florence2TaskResult::BoxesOrPolygons {
        boxes: vec![Florence2BoundingBox {
            xmin: 1.0,
            ymin: 2.0,
            xmax: 3.0,
            ymax: 4.0,
        }],
        box_labels: vec!["cat".to_string()],
        polygons: vec![],
        polygon_labels: vec![],
    };
    let value = structured_task_json(Florence2Task::OpenVocabularyDetection, &result);
    assert_eq!(value["kind"], "bboxes_or_polygons");
    assert_eq!(value["bboxes"][0][2], 3.0);
    assert_eq!(value["bboxes_labels"][0], "cat");
    assert_eq!(value["polygons"].as_array().unwrap().len(), 0);
    assert_eq!(value["polygons_labels"].as_array().unwrap().len(), 0);
}
