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

//! CLI driver for layout-aware Falcon-OCR (issue #848).
//!
//! `--layout-detections FILE` turns one page into a sequence of per-region OCR
//! runs: the file's boxes go through
//! [`mlxcel::vision::falcon_ocr_layout`] (nested-box suppression, layout-class
//! to OCR-category routing, cropping), and each surviving region is OCRed with
//! the instruction its category was trained on, mirroring
//! `modeling_falcon_ocr.py::generate_with_layout`.
//!
//! The half this does **not** do is detection. The reference loads
//! PP-DocLayoutV3 through `transformers.AutoModelForObjectDetection`; mlxcel
//! ships no document-layout detector, so the boxes are an input rather than
//! something the command produces. The accepted JSON is the shape `mlxcel
//! detect --format json` prints, so a detector that mlxcel does gain later
//! feeds this path without a format change.
//!
//! Region order is the file's order. The reference detector emits reading order
//! (it sorts by its `order_logits` head), and nothing here reorders, so the
//! output preserves whatever order the caller supplied.

use std::io::Read;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};

use mlxcel::vision::detection::Detection;
use mlxcel::vision::falcon_ocr_layout::{
    LayoutOcrPlan, MIN_CROP_DIM, filter_nested_detections, layout_to_ocr_category,
    plan_layout_region_boxes,
};
use mlxcel::{LoadedModel, SamplingConfig};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::sampling::TokenBiasMap;

use super::generate::{decode_generated_text, run_generation_mode};
use crate::{GenerateArgs, MlxcelTokenizer};

/// Fraction of a box that must lie inside a larger box before it is treated as
/// nested and dropped. `containment_threshold` in the reference.
const CONTAINMENT_THRESHOLD: f32 = 0.8;

/// Trailing markers `generate_with_layout` strips from each region's answer.
const ANSWER_TERMINATORS: [&str; 2] = ["<|end_of_query|>", "<|end_of_text|>"];

/// Largest `--layout-detections` document that is read at all.
///
/// A real detector dump for one page is kilobytes, so this is three orders of
/// magnitude of headroom. The ceiling exists because the alternative is reading
/// whatever the path happens to be into memory unbounded: a truncated or
/// hostile file, or a fifo that never ends.
const MAX_LAYOUT_DETECTIONS_BYTES: usize = 32 * 1024 * 1024;

/// Largest number of detections one page may carry.
///
/// A dense page yields low hundreds of boxes. The cap matters because the
/// nested-box suppression is quadratic in the entry count, so an array that
/// parses in a moment can still cost minutes of comparisons.
const MAX_LAYOUT_DETECTIONS: usize = 4096;

/// Longest layout class name accepted, in characters.
///
/// Real class names are under 20 characters. The bound is on the class name in
/// particular because it is the one attacker-controlled string this command
/// echoes back to the terminal, in the summary's `no text category:` list.
const MAX_LAYOUT_CLASS_NAME_CHARS: usize = 64;

/// Read and validate a `--layout-detections` file.
///
/// Called before the model is loaded so a malformed file costs nothing.
///
/// The read is bounded rather than a plain `read_to_string`, and the bound is
/// enforced on the bytes actually read rather than on `metadata().len()`, which
/// reports zero for a fifo and for anything else that is not a regular file.
pub(crate) fn load_layout_detections(path: &Path) -> Result<Vec<Detection>> {
    let read_bounded = || -> std::io::Result<String> {
        let file = std::fs::File::open(path)?;
        let mut text = String::new();
        file.take(MAX_LAYOUT_DETECTIONS_BYTES as u64 + 1)
            .read_to_string(&mut text)?;
        Ok(text)
    };
    let text = read_bounded()
        .with_context(|| format!("--layout-detections: failed to read {}", path.display()))?;
    if text.len() > MAX_LAYOUT_DETECTIONS_BYTES {
        bail!(
            "--layout-detections {}: the file is larger than the {} MiB limit; a detector \
             dump for one page is kilobytes",
            path.display(),
            MAX_LAYOUT_DETECTIONS_BYTES / (1024 * 1024)
        );
    }
    parse_layout_detections(&text)
        .map_err(|error| anyhow!("--layout-detections {}: {error}", path.display()))
}

/// Parse the detections array out of a `--layout-detections` document.
///
/// Three spellings are accepted, all producing the same [`Detection`] list:
///
/// - `{"detections": [...]}`, what `mlxcel detect --format json` prints;
/// - a bare top-level `[...]` array, convenient for a hand-written file;
/// - per entry, either `{"label", "confidence", "box": {"l","t","r","b"}}` or
///   the `{"category", "score", "bbox": [l, t, r, b]}` spelling mlx-vlm's
///   `falcon_ocr/layout.py` emits.
///
/// Every rejection names the offending entry by index, because a detector
/// dump is long enough that "invalid input" alone is not actionable.
fn parse_layout_detections(text: &str) -> Result<Vec<Detection>> {
    let root: Value =
        serde_json::from_str(text).map_err(|error| anyhow!("not valid JSON: {error}"))?;

    let items: &[Value] = match &root {
        Value::Array(items) => items,
        Value::Object(map) => match map.get("detections") {
            Some(Value::Array(items)) => items,
            Some(other) => bail!("`detections` must be an array, got {}", json_kind(other)),
            None => bail!(
                "expected an object with a `detections` array (the shape `mlxcel detect \
                 --format json` prints) or a bare array of detections"
            ),
        },
        other => bail!(
            "expected a JSON object or array at the top level, got {}",
            json_kind(other)
        ),
    };

    // Checked before the loop so an enormous array is rejected without first
    // building the `Detection` vector it describes.
    if items.len() > MAX_LAYOUT_DETECTIONS {
        bail!(
            "{} detections is more than the {MAX_LAYOUT_DETECTIONS} this command accepts \
             for one page",
            items.len()
        );
    }

    let mut detections = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let object = item.as_object().ok_or_else(|| {
            anyhow!(
                "detection #{index}: expected an object, got {}",
                json_kind(item)
            )
        })?;
        detections.push(parse_detection(object, index)?);
    }
    Ok(detections)
}

fn parse_detection(object: &Map<String, Value>, index: usize) -> Result<Detection> {
    const CLASS_KEYS: [&str; 4] = ["label", "category", "class_name", "class"];

    let class_value = CLASS_KEYS
        .iter()
        .filter_map(|key| object.get(*key))
        .find(|value| value.is_string());
    let class_name = match class_value {
        Some(Value::String(name)) => name.trim(),
        // A key is there but holds a number (some dumps put the class id under
        // `label`). Say which key was wrong instead of "missing".
        _ if CLASS_KEYS.iter().any(|key| object.contains_key(*key)) => bail!(
            "detection #{index}: the layout class name must be a string (found a \
             non-string under `label` / `category` / `class_name`)"
        ),
        _ => bail!(
            "detection #{index}: missing the layout class name (expected `label`, \
             `category`, or `class_name`)"
        ),
    };
    if class_name.is_empty() {
        bail!("detection #{index}: the layout class name is empty");
    }
    // The class name is the one piece of this file that is printed back to the
    // operator verbatim (the summary's `no text category:` list), so a control
    // character in it is an escape-sequence injection into their terminal.
    if class_name.chars().any(char::is_control) {
        bail!("detection #{index}: the layout class name contains a control character");
    }
    if class_name.chars().count() > MAX_LAYOUT_CLASS_NAME_CHARS {
        bail!(
            "detection #{index}: the layout class name is longer than \
             {MAX_LAYOUT_CLASS_NAME_CHARS} characters"
        );
    }

    let bbox = parse_bbox(object, index)?;

    let score = match ["confidence", "score"]
        .iter()
        .find_map(|key| object.get(*key))
    {
        None => 1.0,
        Some(value) => {
            let score = as_f32(value).ok_or_else(|| {
                anyhow!(
                    "detection #{index}: `confidence` must be a number, got {}",
                    json_kind(value)
                )
            })?;
            if !score.is_finite() {
                bail!("detection #{index}: `confidence` is not a finite number");
            }
            score
        }
    };

    let label = object
        .get("label_id")
        .and_then(Value::as_u64)
        .unwrap_or_default() as usize;

    Ok(Detection {
        bbox,
        score,
        label,
        class_name: class_name.to_string(),
    })
}

fn parse_bbox(object: &Map<String, Value>, index: usize) -> Result<[f32; 4]> {
    let raw = ["box", "bbox", "bounding_box"]
        .iter()
        .find_map(|key| object.get(*key))
        .ok_or_else(|| {
            anyhow!(
                "detection #{index}: missing the bounding box (expected a `box` object \
                 with `l` / `t` / `r` / `b`, or a `bbox` array of four numbers)"
            )
        })?;

    let bbox = match raw {
        Value::Array(values) => {
            if values.len() != 4 {
                bail!(
                    "detection #{index}: the bounding-box array must hold exactly four \
                     numbers (left, top, right, bottom), got {}",
                    values.len()
                );
            }
            let mut bbox = [0.0f32; 4];
            for (axis, value) in values.iter().enumerate() {
                bbox[axis] = as_f32(value).ok_or_else(|| {
                    anyhow!(
                        "detection #{index}: bounding-box entry {axis} must be a number, got {}",
                        json_kind(value)
                    )
                })?;
            }
            bbox
        }
        Value::Object(map) => {
            // `l/t/r/b` is what `mlxcel detect` prints; the other two spellings
            // are what detector dumps in the wild use.
            const AXES: [[&str; 3]; 4] = [
                ["l", "x1", "left"],
                ["t", "y1", "top"],
                ["r", "x2", "right"],
                ["b", "y2", "bottom"],
            ];
            let mut bbox = [0.0f32; 4];
            for (axis, aliases) in AXES.iter().enumerate() {
                let value = aliases
                    .iter()
                    .find_map(|key| map.get(*key))
                    .ok_or_else(|| {
                        anyhow!(
                            "detection #{index}: the bounding box has no `{}` (or `{}` / `{}`)",
                            aliases[0],
                            aliases[1],
                            aliases[2]
                        )
                    })?;
                bbox[axis] = as_f32(value).ok_or_else(|| {
                    anyhow!(
                        "detection #{index}: bounding-box `{}` must be a number, got {}",
                        aliases[0],
                        json_kind(value)
                    )
                })?;
            }
            bbox
        }
        other => bail!(
            "detection #{index}: the bounding box must be an object or an array, got {}",
            json_kind(other)
        ),
    };

    if bbox.iter().any(|coord| !coord.is_finite()) {
        bail!("detection #{index}: the bounding box has a non-finite coordinate");
    }
    if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
        bail!(
            "detection #{index}: the bounding box is empty or inverted \
             (left={:.2}, top={:.2}, right={:.2}, bottom={:.2}); expected right > left \
             and bottom > top",
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3]
        );
    }
    Ok(bbox)
}

fn as_f32(value: &Value) -> Option<f32> {
    value.as_f64().map(|number| number as f32)
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// One-line account of what the layout stage kept and what it threw away.
///
/// Regions that silently vanish are the failure mode of this pipeline (a page
/// OCRs to half its text and nothing says why), so the counts are printed
/// whether or not anything was dropped.
///
/// `kept` is the nested-box survivors, passed in rather than recomputed: the
/// suppression is quadratic in the detection count and the planner has already
/// paid for it once.
fn plan_summary(detections: &[Detection], kept: &[Detection], plans: &[LayoutOcrPlan]) -> String {
    let nested = detections.len() - kept.len();
    let mut textless: Vec<&str> = kept
        .iter()
        .filter(|detection| layout_to_ocr_category(&detection.class_name).is_none())
        .map(|detection| detection.class_name.as_str())
        .collect();
    textless.sort_unstable();
    textless.dedup();

    let mut parts = vec![format!("{} detection(s)", detections.len())];
    if nested > 0 {
        parts.push(format!("{nested} nested box(es) dropped"));
    }
    if !textless.is_empty() {
        parts.push(format!("no text category: {}", textless.join(", ")));
    }
    if is_whole_page_fallback(plans) {
        parts.push("no usable region, OCRing the whole page as `plain`".to_string());
    } else {
        parts.push(format!("{} region(s) to OCR", plans.len()));
    }
    parts.join("; ")
}

/// The planner returns a single `plain` region when nothing usable survived.
/// No layout class maps onto `plain`, so the category is unambiguous.
fn is_whole_page_fallback(plans: &[LayoutOcrPlan]) -> bool {
    plans.len() == 1 && plans[0].category == "plain"
}

/// Strip the reference's trailing task markers from one region's answer.
fn clean_region_answer(text: &str) -> String {
    let mut cleaned = text.to_string();
    for terminator in ANSWER_TERMINATORS {
        cleaned = cleaned.replace(terminator, "");
    }
    cleaned.trim().to_string()
}

/// Run layout-aware OCR over one page and print each region in file order.
pub(crate) fn run_falcon_ocr_layout_generation(
    model: &LoadedModel,
    args: &GenerateArgs,
    tokenizer: &MlxcelTokenizer,
    sampling_config: &SamplingConfig,
    kv_cache_mode: KVCacheMode,
    token_bias: &TokenBiasMap,
    detections: &[Detection],
) -> Result<()> {
    let LoadedModel::FalconOcrVL(falcon) = model else {
        bail!(
            "--layout-detections is a Falcon-OCR path (per-region prompts come from the \
             checkpoint's OCR category instructions); this model is not a Falcon-OCR \
             checkpoint"
        );
    };

    // Decode through the shared admission limits, same as the Florence-2 task
    // path: a decompression bomb must be rejected before any pixel work.
    let image_path = &args.generation.image[0];
    let bytes = std::fs::read(image_path)
        .with_context(|| format!("Failed to read image {image_path:?}"))?;
    let mut images =
        mlxcel::decode_image_payloads_with_limits(&[bytes], mlxcel::current_image_input_limits())
            .with_context(|| format!("Failed to decode image {image_path:?}"))?;
    let page = images
        .pop()
        .ok_or_else(|| anyhow!("image decoding returned no image for {image_path:?}"))?;

    // Planned without cropping, then cropped one region at a time below. A crop
    // copies, so materializing the whole plan up front would hold one full-size
    // image per detection before a single OCR token is generated, which for a
    // page-sized box is megabytes each.
    let kept = filter_nested_detections(detections, CONTAINMENT_THRESHOLD);
    let plans = plan_layout_region_boxes(
        page.width(),
        page.height(),
        detections,
        CONTAINMENT_THRESHOLD,
        MIN_CROP_DIM,
        falcon.processor.max_dimension,
    );
    println!(
        "Falcon-OCR layout: {}",
        plan_summary(detections, &kept, &plans)
    );
    println!();

    let started = Instant::now();
    let mut total_generated = 0usize;
    for (position, plan) in plans.iter().enumerate() {
        let (x, y, width, height) = plan.crop;
        let crop = page.crop_imm(x, y, width, height);
        let (answer, generated) = ocr_region(
            model,
            args,
            tokenizer,
            sampling_config,
            kv_cache_mode,
            token_bias,
            plan,
            &crop,
        )?;
        total_generated += generated;
        println!(
            "[{}] {} (score {:.3}) bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
            position + 1,
            plan.category,
            plan.score,
            plan.bbox[0],
            plan.bbox[1],
            plan.bbox[2],
            plan.bbox[3]
        );
        println!("{answer}");
        println!();
    }

    let elapsed = started.elapsed().as_secs_f64();
    let tokens_per_second = if elapsed > 0.0 {
        total_generated as f64 / elapsed
    } else {
        0.0
    };
    println!(
        "[Layout OCR: {} region(s), {} tokens in {:.2}s = {:.2} tok/s]",
        plans.len(),
        total_generated,
        elapsed,
        tokens_per_second
    );

    mlxcel_core::clear_memory_cache();
    Ok(())
}

/// OCR one planned region and return its cleaned answer plus the token count.
///
/// The prompt is the reference's `f"{instruction}\n"` followed by the OCR task
/// token, which the Falcon-OCR prompt builder appends itself along with the
/// image block for the crop.
///
/// `crop` is the caller's freshly made cut of `plan.crop`; it is a parameter
/// rather than a field of the plan so it lives only as long as this one region
/// is being generated.
fn ocr_region(
    model: &LoadedModel,
    args: &GenerateArgs,
    tokenizer: &MlxcelTokenizer,
    sampling_config: &SamplingConfig,
    kv_cache_mode: KVCacheMode,
    token_bias: &TokenBiasMap,
    plan: &LayoutOcrPlan,
    crop: &image::DynamicImage,
) -> Result<(String, usize)> {
    let prompt = format!("{}\n", plan.ocr_category.instruction());
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode(&prompt, true)
        .map_err(|error| anyhow!("Tokenization failed: {error}"))?
        .iter()
        .map(|&token| token as i32)
        .collect();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings_with_budget(
        model,
        &mut prompt_tokens,
        &prompt,
        std::slice::from_ref(crop),
        None,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&token| token as i32)
                .collect()
        },
    )?;
    let embeddings = prepared.map(|prepared| prepared.embeddings);

    let (generated_tokens, _stats) = run_generation_mode(
        model,
        args,
        &prompt_tokens,
        sampling_config,
        embeddings.as_ref(),
        kv_cache_mode,
        token_bias.clone(),
    )?;
    let answer = decode_generated_text(tokenizer, &prompt_tokens, &generated_tokens);
    Ok((clean_region_answer(&answer), generated_tokens.len()))
}

#[cfg(test)]
#[path = "generate_falcon_ocr_tests.rs"]
mod tests;
