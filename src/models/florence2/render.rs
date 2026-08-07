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

//! Presentation of a parsed Florence-2 answer: the human-readable text form
//! and the structured JSON form.
//!
//! Both the CLI (`mlxcel generate`) and the server's seq2seq worker
//! (issue #1073) surface the same parsed [`Florence2TaskResult`], and the
//! text they print/return must be byte-identical so an HTTP answer can be
//! validated against the CLI answer for the same model, image, and task.
//! Keeping the renderer here, next to the parser it consumes, is what
//! guarantees that: `src/commands/generate_florence2.rs` prints
//! [`render_task_result`] and the server puts the same string into
//! `message.content`.
//!
//! [`structured_task_json`] is the machine-readable companion the server
//! attaches as the `florence2_result` extension field on the assistant
//! message. Its key names (`bboxes`, `quad_boxes`, `polygons`, `labels`,
//! `bboxes_labels`, `polygons_labels`) mirror upstream
//! `Florence2Processor.post_process_generation`
//! (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py)
//! so code written against the HF / mlx-vlm dict shape ports directly.

use serde_json::{Value, json};

use super::coords::Florence2Polygon;
use super::postprocess::Florence2TaskResult;
use super::tasks::Florence2Task;

/// Render a parsed task result for the terminal / `message.content`.
///
/// Spatial results are printed one instance per line as
/// `label: [coordinates]` in original-image pixels. An answer that parsed to
/// nothing falls back to the raw decoded text, which is the only way to tell
/// "the model found nothing" from "the parser rejected the answer".
///
/// Used by: `commands/generate_florence2.rs` (CLI print),
/// `server/florence2_worker.rs` (chat `message.content`).
pub fn render_task_result(result: &Florence2TaskResult, raw_text: &str) -> String {
    let rendered = match result {
        Florence2TaskResult::Text(text) => text.trim().to_string(),
        Florence2TaskResult::Boxes { boxes, labels } => boxes
            .iter()
            .zip(labels)
            .map(|(bbox, label)| {
                format!(
                    "{}: [{:.1}, {:.1}, {:.1}, {:.1}]",
                    display_label(label),
                    bbox.xmin,
                    bbox.ymin,
                    bbox.xmax,
                    bbox.ymax
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Florence2TaskResult::QuadBoxes { quad_boxes, labels } => quad_boxes
            .iter()
            .zip(labels)
            .map(|(quad, label)| {
                format!("{}: [{}]", display_label(label), join_points(&quad.points))
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Florence2TaskResult::Polygons { polygons, labels } => {
            render_polygon_instances(polygons, labels)
        }
        Florence2TaskResult::BoxesOrPolygons {
            boxes,
            box_labels,
            polygons,
            polygon_labels,
        } => {
            let mut lines: Vec<String> = boxes
                .iter()
                .zip(box_labels)
                .map(|(bbox, label)| {
                    format!(
                        "{}: [{:.1}, {:.1}, {:.1}, {:.1}]",
                        display_label(label),
                        bbox.xmin,
                        bbox.ymin,
                        bbox.xmax,
                        bbox.ymax
                    )
                })
                .collect();
            let polygon_block = render_polygon_instances(polygons, polygon_labels);
            if !polygon_block.is_empty() {
                lines.push(polygon_block);
            }
            lines.join("\n")
        }
    };
    if rendered.is_empty() {
        format!("[no parsed instances] raw answer: {}", raw_text.trim())
    } else {
        rendered
    }
}

fn render_polygon_instances(polygons: &[Vec<Florence2Polygon>], labels: &[String]) -> String {
    polygons
        .iter()
        .zip(labels)
        .map(|(outlines, label)| {
            let rendered_outlines = outlines
                .iter()
                .map(|polygon| format!("[{}]", join_points(&polygon.points)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}: {}", display_label(label), rendered_outlines)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn join_points(points: &[f32]) -> String {
    points
        .iter()
        .map(|point| format!("{point:.1}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `<REGION_PROPOSAL>` predicts unnamed regions; keep the line shape stable.
fn display_label(label: &str) -> &str {
    if label.is_empty() { "(region)" } else { label }
}

/// Serialize a parsed task result into the JSON object the server returns as
/// the `florence2_result` extension field.
///
/// Shape, discriminated by `kind`:
///
/// - `{"task": "<CAPTION>", "kind": "text", "text": "..."}`
/// - `{"task": "<OD>", "kind": "bboxes", "bboxes": [[x1,y1,x2,y2], ...],
///   "labels": ["cat", ...]}`
/// - `{"task": "<OCR_WITH_REGION>", "kind": "quad_boxes",
///   "quad_boxes": [[x0,y0, ... x3,y3], ...], "labels": ["line", ...]}`
/// - `{"task": "<REFERRING_EXPRESSION_SEGMENTATION>", "kind": "polygons",
///   "polygons": [[[x0,y0,x1,y1, ...], ...], ...], "labels": [...]}` (one
///   entry per instance, one flat outline list per disconnected part)
/// - `{"task": "<OPEN_VOCABULARY_DETECTION>", "kind": "bboxes_or_polygons",
///   "bboxes": ..., "bboxes_labels": ..., "polygons": ...,
///   "polygons_labels": ...}` (the unused pair is empty, mirroring upstream)
///
/// Coordinates are f32 pixels in the original image extent, exactly as
/// [`Florence2TaskResult`] carries them.
///
/// Used by: `server/florence2_worker.rs` (chat response extension field).
pub fn structured_task_json(task: Florence2Task, result: &Florence2TaskResult) -> Value {
    let task_token = task.token();
    match result {
        Florence2TaskResult::Text(text) => json!({
            "task": task_token,
            "kind": "text",
            "text": text.trim(),
        }),
        Florence2TaskResult::Boxes { boxes, labels } => json!({
            "task": task_token,
            "kind": "bboxes",
            "bboxes": boxes.iter().map(|b| b.to_array().to_vec()).collect::<Vec<_>>(),
            "labels": labels,
        }),
        Florence2TaskResult::QuadBoxes { quad_boxes, labels } => json!({
            "task": task_token,
            "kind": "quad_boxes",
            "quad_boxes": quad_boxes.iter().map(|q| q.points.to_vec()).collect::<Vec<_>>(),
            "labels": labels,
        }),
        Florence2TaskResult::Polygons { polygons, labels } => json!({
            "task": task_token,
            "kind": "polygons",
            "polygons": polygons_json(polygons),
            "labels": labels,
        }),
        Florence2TaskResult::BoxesOrPolygons {
            boxes,
            box_labels,
            polygons,
            polygon_labels,
        } => json!({
            "task": task_token,
            "kind": "bboxes_or_polygons",
            "bboxes": boxes.iter().map(|b| b.to_array().to_vec()).collect::<Vec<_>>(),
            "bboxes_labels": box_labels,
            "polygons": polygons_json(polygons),
            "polygons_labels": polygon_labels,
        }),
    }
}

fn polygons_json(polygons: &[Vec<Florence2Polygon>]) -> Vec<Vec<Vec<f32>>> {
    polygons
        .iter()
        .map(|instance| {
            instance
                .iter()
                .map(|outline| outline.points.clone())
                .collect()
        })
        .collect()
}

#[cfg(test)]
#[path = "florence2_render_tests.rs"]
mod florence2_render_tests;
