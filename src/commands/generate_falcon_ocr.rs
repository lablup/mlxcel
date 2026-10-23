// Copyright 2025-2026 Lablup Inc.
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
//! Detection remains a separate command. Native PP-DocLayoutV3 writes the
//! accepted JSON through `mlxcel detect --format json`, and this command
//! consumes that file without changing its model-predicted reading order.
//!
//! Region order is the file's order. The reference detector emits reading order
//! (it sorts by its `order_logits` head), and nothing here reorders, so the
//! output preserves whatever order the caller supplied.

use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
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

#[path = "generate_falcon_ocr_input.rs"]
mod input;

pub(crate) use input::load_layout_detections;
#[cfg(test)]
use input::{MAX_LAYOUT_CLASS_NAME_CHARS, MAX_LAYOUT_DETECTIONS, parse_layout_detections};

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
        .ok_or_else(|| anyhow!("Image decoding returned no image for {image_path:?}"))?;

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

    let prepared = mlxcel::vlm_runtime::prepare_vlm_embeddings_with_budget(
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
