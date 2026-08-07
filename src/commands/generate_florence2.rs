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

//! CLI driver for Florence-2 task generation (issue #856).
//!
//! Florence-2 is an encoder-decoder (seq2seq) VLM: the encoder consumes the
//! fused image-plus-prompt sequence and the decoder cross-attends to it, so
//! it cannot run on the autoregressive loop `run_generation_mode` drives.
//! `run_generate_once` routes `LoadedModel::Florence2VLM` here before that
//! loop, mirroring the DiffusionGemma / LLaDA-2 early exits.
//!
//! The `-p/--prompt` string selects one of the fifteen task modes
//! (`<CAPTION>`, `<OCR>`, `<OD>`, ...), optionally followed by the input
//! text the task interpolates. Image decoding routes through the shared
//! [`mlxcel::ImageInputLimits`] admission bounds so an oversized or
//! decompression-bomb payload is rejected before any pixel work.

use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};

use mlxcel::models::florence2::{Florence2TaskResult, Florence2VlmModel, parse_task_prompt};

use super::generate::print_generation_preamble;
use crate::GenerateArgs;

/// Run one Florence-2 task from the CLI flag surface and print the parsed
/// answer plus a generation-stats line.
pub(crate) fn run_florence2_generation(
    model: &Florence2VlmModel,
    args: &GenerateArgs,
    user_prompt: &str,
) -> Result<()> {
    ensure!(
        args.generation.audio.is_none(),
        "Florence-2 does not take --audio input"
    );
    ensure!(
        args.generation.video.is_empty(),
        "Florence-2 does not take --video input"
    );
    ensure!(
        !args.generation.image.is_empty(),
        "Florence-2 is an image-task model: pass --image <path> together with a task prompt \
         such as -p '<CAPTION>', -p '<OCR>', or -p '<OD>'"
    );
    ensure!(
        args.generation.image.len() == 1,
        "Florence-2 processes one image per request; got {} --image paths",
        args.generation.image.len()
    );

    let (task, input) = parse_task_prompt(user_prompt).map_err(|e| anyhow!("-p/--prompt: {e}"))?;

    // Decode through the shared admission limits (decompression-bomb
    // defense, issue #855 handoff): `preprocess_with_sizes` takes an
    // already-decoded image, so the bound has to hold at this boundary.
    let image_path = &args.generation.image[0];
    let bytes = std::fs::read(image_path)
        .with_context(|| format!("Failed to read image {image_path:?}"))?;
    let mut images =
        mlxcel::decode_image_payloads_with_limits(&[bytes], mlxcel::current_image_input_limits())
            .with_context(|| format!("Failed to decode image {image_path:?}"))?;
    let image = images
        .pop()
        .ok_or_else(|| anyhow!("image decoding returned no image for {image_path:?}"))?;

    print_generation_preamble(user_prompt)?;
    println!();

    let started = Instant::now();
    let run = model.run_task(task, input.as_deref(), &image, args.generation.max_tokens)?;
    let elapsed = started.elapsed().as_secs_f64();

    println!(
        "{}",
        render_task_result(&run.output.result, &run.output.raw_text)
    );
    println!();
    let tps = if elapsed > 0.0 {
        run.generated_tokens as f64 / elapsed
    } else {
        0.0
    };
    println!(
        "[Generated {} tokens in {:.2}s = {:.2} tok/s]",
        run.generated_tokens, elapsed, tps
    );
    if args.generation.profile {
        println!("[Raw answer] {}", run.output.raw_text);
    }

    mlxcel_core::clear_memory_cache();
    Ok(())
}

/// Render a parsed task result for the terminal.
///
/// Spatial results are printed one instance per line as
/// `label: [coordinates]` in original-image pixels. An answer that parsed to
/// nothing falls back to the raw decoded text, which is the only way to tell
/// "the model found nothing" from "the parser rejected the answer".
fn render_task_result(result: &Florence2TaskResult, raw_text: &str) -> String {
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
        // `Florence2TaskResult` is `#[non_exhaustive]`: render future
        // variants through the raw-text fallback below.
        _ => String::new(),
    };
    if rendered.is_empty() {
        format!("[no parsed instances] raw answer: {}", raw_text.trim())
    } else {
        rendered
    }
}

fn render_polygon_instances(
    polygons: &[Vec<mlxcel::models::Florence2Polygon>],
    labels: &[String],
) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel::models::{Florence2BoundingBox, Florence2Polygon, Florence2QuadBox};

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
    fn empty_result_falls_back_to_raw_text() {
        let result = Florence2TaskResult::Boxes {
            boxes: vec![],
            labels: vec![],
        };
        let rendered = render_task_result(&result, "<s></s>");
        assert!(rendered.contains("raw answer"), "got: {rendered}");
    }
}
