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

//! Prompted image detection and instance segmentation.

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use super::super::interpolate::resize_bilinear_nhwc;
use super::super::{DenseOutput, Instance};
use super::model::{DetectorFeatures, DetectorModel, DetectorOutput};
use super::options::PredictionOptions;
use super::postprocess::{clip_xyxy, confidence_scores, nms_indices, normalized_cxcywh};
use super::processing::Sam3Processor;

pub(crate) fn predict_image<M: DetectorModel>(
    model: &M,
    processor: &Sam3Processor,
    image: &DynamicImage,
    options: &PredictionOptions,
) -> Result<DenseOutput> {
    let width = image.width();
    let height = image.height();
    let pixels = processor
        .preprocess_image(image)
        .context("preprocess SAM 3 image")?;
    let backbone = model.backbone(&pixels);
    let features = model
        .detector_features(&backbone)
        .map_err(anyhow::Error::msg)
        .context("encode SAM 3 image")?;
    let normalized = normalized_cxcywh(&options.boxes, width, height)?;
    let box_prompts = (!normalized.is_empty())
        .then(|| mlxcel_core::from_slice_f32(&normalized, &[1, options.boxes.len() as i32, 4]));

    let mut instances = Vec::new();
    let mut mask_parts = Vec::new();
    for prompt in &options.prompts {
        let tokenized = processor.tokenize(prompt)?;
        let text = model.prompt_features(&tokenized.input_ids, &tokenized.attention_mask);
        let prediction = model
            .detect(
                &features,
                &text,
                &tokenized.attention_mask,
                box_prompts.as_deref(),
                options.task == "segment",
            )
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("run SAM 3 detector for prompt {prompt:?}"))?;
        append_prompt(
            prediction,
            &features,
            prompt,
            width,
            height,
            options,
            &mut instances,
            &mut mask_parts,
        )?;
    }

    let mut output = DenseOutput {
        instances,
        ..DenseOutput::default()
    };
    let boxes: Vec<f32> = output.instances.iter().flat_map(|row| row.bbox).collect();
    let scores: Vec<f32> = output.instances.iter().map(|row| row.score).collect();
    output.tensors.insert(
        "boxes".into(),
        mlxcel_core::from_slice_f32(&boxes, &[output.instances.len() as i32, 4]),
    );
    output.tensors.insert(
        "scores".into(),
        mlxcel_core::from_slice_f32(&scores, &[output.instances.len() as i32]),
    );
    if options.task == "segment" {
        let masks = if mask_parts.is_empty() {
            mlxcel_core::from_bytes(
                &[],
                &[0, height as i32, width as i32],
                mlxcel_core::dtype::UINT8,
            )
        } else {
            let refs: Vec<&MlxArray> = mask_parts.iter().map(|value| &**value).collect();
            mlxcel_core::concatenate_many(&refs, 0)
        };
        output.tensors.insert("masks".into(), masks);
    }
    output
        .meta
        .insert("threshold".into(), json!(options.threshold));
    output.meta.insert("nms".into(), json!(options.nms));
    output.meta.insert("prompts".into(), json!(options.prompts));
    Ok(output)
}

pub(crate) struct SelectedDetections {
    pub instances: Vec<Instance>,
    pub queries: Vec<i32>,
}

pub(crate) fn ensure_tensor_finite(value: &MlxArray, label: &str) -> Result<()> {
    let finite = mlxcel_core::all_all(&mlxcel_core::isfinite(value));
    mlxcel_core::eval(&finite);
    ensure!(
        mlxcel_core::array_to_raw_bytes(&finite) == [1],
        "{label} contains non-finite values"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn select_detections(
    prediction: &DetectorOutput,
    prompt: &str,
    width: u32,
    height: u32,
    threshold: f32,
    nms: f32,
    with_masks: bool,
    first_mask_index: usize,
) -> Result<SelectedDetections> {
    let logits_shape = mlxcel_core::array_shape(&prediction.logits);
    let boxes_shape = mlxcel_core::array_shape(&prediction.boxes);
    let presence_shape = mlxcel_core::array_shape(&prediction.presence);
    ensure!(
        logits_shape.len() == 2 && logits_shape[0] == 1 && logits_shape[1] > 0,
        "SAM 3 detector logits must have [1,Q], got {logits_shape:?}"
    );
    ensure!(
        boxes_shape == [1, logits_shape[1], 4],
        "SAM 3 detector boxes must have [1,Q,4], got {boxes_shape:?} for logits {logits_shape:?}"
    );
    ensure!(
        presence_shape.iter().product::<i32>() == 1,
        "SAM 3 detector presence must contain one value, got {presence_shape:?}"
    );
    ensure_tensor_finite(&prediction.logits, "SAM 3 detector logits")?;
    ensure_tensor_finite(&prediction.boxes, "SAM 3 detector boxes")?;
    ensure_tensor_finite(&prediction.presence, "SAM 3 detector presence")?;
    if with_masks {
        let masks = prediction
            .masks
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("SAM 3 detector omitted segmentation masks"))?;
        let masks_shape = mlxcel_core::array_shape(masks);
        ensure!(
            masks_shape.len() == 4
                && masks_shape[0] == 1
                && masks_shape[1] == logits_shape[1]
                && masks_shape[2] > 0
                && masks_shape[3] > 0,
            "SAM 3 detector masks must have [1,Q,H,W] matching logits, got {masks_shape:?} for {logits_shape:?}"
        );
    }
    let logits = mlxcel_core::astype(&prediction.logits, mlxcel_core::dtype::FLOAT32);
    let boxes = mlxcel_core::astype(&prediction.boxes, mlxcel_core::dtype::FLOAT32);
    let presence = mlxcel_core::astype(&prediction.presence, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&logits);
    mlxcel_core::eval(&boxes);
    mlxcel_core::eval(&presence);
    let logits = mlxcel_core::utils::array_to_vec_f32(&logits);
    let raw_boxes = mlxcel_core::utils::array_to_vec_f32(&boxes);
    let presence = mlxcel_core::utils::array_to_vec_f32(&presence);
    ensure!(
        raw_boxes.iter().all(|value| value.is_finite())
            && presence.iter().all(|value| value.is_finite()),
        "SAM 3 detector returned non-finite boxes or presence"
    );
    let scores = confidence_scores(
        &logits,
        presence.first().copied().unwrap_or(f32::NEG_INFINITY),
    );
    ensure!(
        scores.iter().all(|value| value.is_finite()),
        "SAM 3 detector returned non-finite confidence scores"
    );

    let mut candidate_queries = Vec::new();
    let mut candidate_scores = Vec::new();
    let mut candidate_boxes = Vec::new();
    for (query, score) in scores.iter().copied().enumerate() {
        if score <= threshold {
            continue;
        }
        let offset = query * 4;
        let bbox = clip_xyxy(
            [
                raw_boxes[offset] * width as f32,
                raw_boxes[offset + 1] * height as f32,
                raw_boxes[offset + 2] * width as f32,
                raw_boxes[offset + 3] * height as f32,
            ],
            width,
            height,
        );
        candidate_queries.push(query);
        candidate_scores.push(score);
        candidate_boxes.push(bbox);
    }
    let keep = nms_indices(&candidate_boxes, &candidate_scores, nms)?;
    let queries = keep
        .iter()
        .map(|&index| candidate_queries[index] as i32)
        .collect();
    let instances = keep
        .iter()
        .enumerate()
        .map(|(output_index, &candidate)| Instance {
            bbox: candidate_boxes[candidate],
            score: candidate_scores[candidate],
            label: prompt.to_owned(),
            mask_index: with_masks.then_some(first_mask_index + output_index),
            object_id: None,
            frame: None,
        })
        .collect();
    Ok(SelectedDetections { instances, queries })
}

#[allow(clippy::too_many_arguments)]
fn append_prompt(
    prediction: DetectorOutput,
    _features: &DetectorFeatures,
    prompt: &str,
    width: u32,
    height: u32,
    options: &PredictionOptions,
    instances: &mut Vec<Instance>,
    mask_parts: &mut Vec<UniquePtr<MlxArray>>,
) -> Result<()> {
    let first_mask_index = instances
        .iter()
        .filter(|row| row.mask_index.is_some())
        .count();
    let selected = select_detections(
        &prediction,
        prompt,
        width,
        height,
        options.threshold,
        options.nms,
        options.task == "segment",
        first_mask_index,
    )?;
    if selected.queries.is_empty() {
        return Ok(());
    }
    instances.extend(selected.instances);
    if options.task == "segment" {
        let prediction_masks = prediction
            .masks
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("SAM 3 segment output omitted masks"))?;
        mask_parts.push(select_and_resize_masks(
            prediction_masks,
            &selected.queries,
            width,
            height,
        )?);
    }
    Ok(())
}

pub(crate) fn select_and_resize_masks(
    masks: &MlxArray,
    indices: &[i32],
    width: u32,
    height: u32,
) -> Result<UniquePtr<MlxArray>> {
    let input_shape = mlxcel_core::array_shape(masks);
    ensure!(
        input_shape.len() == 4
            && input_shape[0] == 1
            && input_shape[1] > 0
            && input_shape[2] > 0
            && input_shape[3] > 0,
        "SAM 3 detector masks must have [1,Q,H,W], got {input_shape:?}"
    );
    ensure!(
        indices
            .iter()
            .all(|index| *index >= 0 && *index < input_shape[1]),
        "SAM 3 mask query indices exceed detector mask shape {input_shape:?}"
    );
    let masks = mlxcel_core::squeeze_axis(masks, 0);
    let indices = mlxcel_core::from_slice_i32(indices, &[indices.len() as i32]);
    let selected = mlxcel_core::take(&masks, &indices, 0);
    ensure_tensor_finite(&selected, "SAM 3 selected masks")?;
    let shape = mlxcel_core::array_shape(&selected);
    let selected = mlxcel_core::astype(&selected, mlxcel_core::dtype::FLOAT32);
    let selected = mlxcel_core::reshape(&selected, &[shape[0], shape[1], shape[2], 1]);
    let resized = resize_bilinear_nhwc(&selected, (height as usize, width as usize), false, true)?;
    let binary = mlxcel_core::greater(&resized, &mlxcel_core::zeros_like(&resized));
    let binary = mlxcel_core::astype(&binary, mlxcel_core::dtype::UINT8);
    let output = mlxcel_core::reshape(&binary, &[shape[0], height as i32, width as i32]);
    // Break the lazy interpolation graph before callers retain multiple mask
    // chunks and concatenate them into the final output.
    mlxcel_core::eval(&output);
    Ok(output)
}

#[cfg(test)]
#[path = "image_tests.rs"]
mod tests;
