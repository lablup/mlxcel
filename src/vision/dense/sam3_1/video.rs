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

//! SAM 3.1 detect-once, multiplex-propagate video inference.

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use super::super::interpolate::resize_bilinear_nhwc;
use super::super::sam3::image::{ensure_tensor_finite, select_and_resize_masks, select_detections};
use super::super::sam3::options::PredictionOptions;
use super::super::sam3::processing::Sam3Processor;
use super::super::{DenseOutput, Instance};
use super::model::Sam31Model;
use super::video_budget::validate_video_mask_budget;

struct InitialObject {
    instance: Instance,
    mask: UniquePtr<MlxArray>,
}

pub(crate) fn track_video(
    model: &Sam31Model,
    processor: &Sam3Processor,
    frames: &[DynamicImage],
    options: &PredictionOptions,
) -> Result<DenseOutput> {
    ensure!(!frames.is_empty(), "SAM 3.1 track received no video frames");
    ensure!(
        !options.prompts.is_empty(),
        "SAM 3.1 region boxes filter text detections and require at least one --prompt"
    );
    let (width, height) = (frames[0].width(), frames[0].height());
    ensure!(
        frames
            .iter()
            .all(|frame| frame.width() == width && frame.height() == height),
        "SAM 3.1 track requires every decoded frame to have the same dimensions"
    );

    let pixels = processor
        .preprocess_image(&frames[0])
        .context("preprocess SAM 3.1 video frame 0")?;
    let backbone = model.backbone(&pixels);
    let detector_features = model
        .detector_features(&backbone)
        .map_err(anyhow::Error::msg)
        .context("encode SAM 3.1 video frame 0")?;
    let mut initial = Vec::new();
    for prompt in &options.prompts {
        let tokens = processor.tokenize(prompt)?;
        let text = model.prompt_features(&tokens.input_ids, &tokens.attention_mask);
        let prediction = model
            .detect(
                &detector_features,
                &text,
                &tokens.attention_mask,
                None,
                true,
            )
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("detect SAM 3.1 video prompt {prompt:?}"))?;
        let selected = select_detections(
            &prediction,
            prompt,
            width,
            height,
            options.threshold,
            options.nms,
            true,
            initial.len(),
        )?;
        let retained: Vec<_> = selected
            .instances
            .into_iter()
            .zip(selected.queries)
            .filter(|(instance, _)| {
                options.boxes.is_empty()
                    || options
                        .boxes
                        .iter()
                        .any(|region| box_iou(instance.bbox, *region) > 0.1)
            })
            .collect();
        if retained.is_empty() {
            continue;
        }
        ensure!(
            initial.len() + retained.len() <= model.config.max_num_objects,
            "SAM 3.1 detected {} objects, exceeding max_num_objects={}",
            initial.len() + retained.len(),
            model.config.max_num_objects
        );
        validate_video_mask_budget(
            width,
            height,
            frames.len(),
            initial.len() + retained.len(),
            &model.config.tracker_config,
        )?;
        let (retained_instances, retained_queries): (Vec<_>, Vec<_>) = retained.into_iter().unzip();
        let prediction_masks = prediction
            .masks
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("SAM 3.1 detector omitted masks"))?;
        let source_masks =
            select_and_resize_masks(prediction_masks, &retained_queries, width, height)?;
        for (index, mut instance) in retained_instances.into_iter().enumerate() {
            let mask = mlxcel_core::slice(
                &source_masks,
                &[index as i32, 0, 0],
                &[index as i32 + 1, height as i32, width as i32],
            );
            instance.mask_index = Some(initial.len());
            initial.push(InitialObject { instance, mask });
        }
    }

    if initial.is_empty() {
        return empty_output(frames.len(), width, height, options);
    }
    ensure!(
        initial.len() <= model.config.max_num_objects,
        "SAM 3.1 detected {} objects, exceeding max_num_objects={}",
        initial.len(),
        model.config.max_num_objects
    );
    validate_video_mask_budget(
        width,
        height,
        frames.len(),
        initial.len(),
        &model.config.tracker_config,
    )?;

    let source_masks: Vec<&MlxArray> = initial.iter().map(|object| &*object.mask).collect();
    let source_masks = mlxcel_core::concatenate_many(&source_masks, 0);
    let tracker_masks = resize_float_masks(&source_masks, model.config.tracker_config.image_size)?;
    let tracker_features = model
        .tracker_features(&backbone, true)
        .map_err(anyhow::Error::msg)?;
    let object_ids: Vec<i64> = (1..=initial.len() as i64).collect();
    let mut state = model
        .tracker
        .init_state(initial.len(), Some(object_ids))
        .map_err(anyhow::Error::msg)?;
    model
        .tracker
        .track_init(&mut state, &tracker_features, &tracker_masks)
        .map_err(anyhow::Error::msg)
        .context("initialize SAM 3.1 multiplex tracker")?;

    let mut instances = Vec::with_capacity(initial.len() * frames.len());
    let mut masks = Vec::with_capacity(initial.len() * frames.len());
    let labels: Vec<String> = initial
        .iter()
        .map(|object| object.instance.label.clone())
        .collect();
    for (index, mut object) in initial.into_iter().enumerate() {
        object.instance.object_id = Some(index as u32 + 1);
        object.instance.frame = Some(0);
        object.instance.mask_index = Some(masks.len());
        masks.push(object.mask);
        instances.push(object.instance);
    }

    let mut track_step_calls = 0usize;
    for (frame_index, frame) in frames.iter().enumerate().skip(1) {
        let pixels = processor
            .preprocess_image(frame)
            .with_context(|| format!("preprocess SAM 3.1 video frame {frame_index}"))?;
        let backbone = model.backbone(&pixels);
        let tracker_features = model
            .tracker_features(&backbone, false)
            .map_err(anyhow::Error::msg)?;
        let prediction = model
            .tracker
            .track_step(&mut state, frame_index, frames.len(), &tracker_features)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("propagate SAM 3.1 video frame {frame_index}"))?;
        track_step_calls += 1;
        let prediction_masks = prediction
            .masks
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SAM 3.1 tracker returned a null mask tensor"))?;
        let prediction_scores = prediction.object_score_logits.as_ref().ok_or_else(|| {
            anyhow::anyhow!("SAM 3.1 tracker returned a null object-score tensor")
        })?;
        append_prediction(
            prediction_masks,
            prediction_scores,
            frame_index as u32,
            &labels,
            width,
            height,
            &mut instances,
            &mut masks,
        )?;
    }

    let mut output = output_from_rows(instances, masks, width, height);
    output
        .meta
        .insert("threshold".into(), json!(options.threshold));
    output.meta.insert("nms".into(), json!(options.nms));
    output.meta.insert("prompts".into(), json!(options.prompts));
    output
        .meta
        .insert("box_filters".into(), json!(options.boxes.len()));
    output
        .meta
        .insert("track_step_calls".into(), json!(track_step_calls));
    output.meta.insert("frames".into(), json!(frames.len()));
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn append_prediction(
    mask_logits: &MlxArray,
    object_logits: &MlxArray,
    frame: u32,
    labels: &[String],
    width: u32,
    height: u32,
    instances: &mut Vec<Instance>,
    masks: &mut Vec<UniquePtr<MlxArray>>,
) -> Result<()> {
    let mask_shape = mlxcel_core::array_shape(mask_logits);
    let score_shape = mlxcel_core::array_shape(object_logits);
    ensure!(
        mask_shape.len() == 4
            && mask_shape[0] == labels.len() as i32
            && mask_shape[1] == 1
            && mask_shape[2] > 0
            && mask_shape[3] > 0,
        "SAM 3.1 tracker masks must have [N,1,H,W] matching {0} labels, got {mask_shape:?}",
        labels.len()
    );
    ensure!(
        score_shape == [labels.len() as i32, 1],
        "SAM 3.1 tracker scores must have [N,1] matching {} labels, got {score_shape:?}",
        labels.len()
    );
    ensure_tensor_finite(mask_logits, "SAM 3.1 tracker mask logits")?;
    ensure_tensor_finite(object_logits, "SAM 3.1 tracker object logits")?;
    let mask_logits = mlxcel_core::squeeze_axis(mask_logits, 1);
    let resized = nearest_resize(&mask_logits, width as i32, height as i32);
    let binary = mlxcel_core::astype(
        &mlxcel_core::greater(&resized, &mlxcel_core::zeros_like(&resized)),
        mlxcel_core::dtype::UINT8,
    );
    let scores = mlxcel_core::sigmoid(&mlxcel_core::astype(
        object_logits,
        mlxcel_core::dtype::FLOAT32,
    ));
    mlxcel_core::eval(&binary);
    mlxcel_core::eval(&scores);
    let score_values = mlxcel_core::utils::array_to_vec_f32(&scores);
    ensure!(
        score_values.iter().all(|score| score.is_finite()),
        "SAM 3.1 tracker returned non-finite object scores"
    );
    for (index, label) in labels.iter().enumerate() {
        let mask = mlxcel_core::slice(
            &binary,
            &[index as i32, 0, 0],
            &[index as i32 + 1, height as i32, width as i32],
        );
        let bbox = mask_bbox(&mask, width, height);
        instances.push(Instance {
            bbox,
            score: score_values[index],
            label: label.clone(),
            mask_index: Some(masks.len()),
            object_id: Some(index as u32 + 1),
            frame: Some(frame),
        });
        masks.push(mask);
    }
    Ok(())
}

fn resize_float_masks(masks: &MlxArray, target: usize) -> Result<UniquePtr<MlxArray>> {
    // Upstream constructs Pillow mode-F images from float32 masks before its
    // bilinear resize. The shared NHWC helper has the same half-pixel float32
    // sampling semantics and avoids the byte rounding used by RGB image input.
    let masks =
        mlxcel_core::expand_dims(&mlxcel_core::astype(masks, mlxcel_core::dtype::FLOAT32), -1);
    let resized = resize_bilinear_nhwc(&masks, (target, target), false, true)?;
    Ok(mlxcel_core::squeeze_axis(&resized, -1))
}

fn nearest_resize(masks: &MlxArray, width: i32, height: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(masks);
    if shape[1] == height && shape[2] == width {
        return mlxcel_core::copy(masks);
    }
    let rows: Vec<i32> = (0..height).map(|row| row * shape[1] / height).collect();
    let columns: Vec<i32> = (0..width).map(|column| column * shape[2] / width).collect();
    let rows = mlxcel_core::from_slice_i32(&rows, &[height]);
    let columns = mlxcel_core::from_slice_i32(&columns, &[width]);
    mlxcel_core::take(&mlxcel_core::take(masks, &rows, 1), &columns, 2)
}

fn output_from_rows(
    instances: Vec<Instance>,
    masks: Vec<UniquePtr<MlxArray>>,
    width: u32,
    height: u32,
) -> DenseOutput {
    let boxes: Vec<f32> = instances.iter().flat_map(|row| row.bbox).collect();
    let scores: Vec<f32> = instances.iter().map(|row| row.score).collect();
    let mask_refs: Vec<&MlxArray> = masks.iter().map(|value| &**value).collect();
    let mut output = DenseOutput {
        instances,
        ..DenseOutput::default()
    };
    output.tensors.insert(
        "boxes".into(),
        mlxcel_core::from_slice_f32(&boxes, &[output.instances.len() as i32, 4]),
    );
    output.tensors.insert(
        "scores".into(),
        mlxcel_core::from_slice_f32(&scores, &[output.instances.len() as i32]),
    );
    output.tensors.insert(
        "masks".into(),
        if mask_refs.is_empty() {
            mlxcel_core::from_bytes(
                &[],
                &[0, height as i32, width as i32],
                mlxcel_core::dtype::UINT8,
            )
        } else {
            mlxcel_core::concatenate_many(&mask_refs, 0)
        },
    );
    output
}

fn empty_output(
    frames: usize,
    width: u32,
    height: u32,
    options: &PredictionOptions,
) -> Result<DenseOutput> {
    let mut output = output_from_rows(Vec::new(), Vec::new(), width, height);
    output
        .meta
        .insert("threshold".into(), json!(options.threshold));
    output.meta.insert("nms".into(), json!(options.nms));
    output.meta.insert("prompts".into(), json!(options.prompts));
    output.meta.insert("frames".into(), json!(frames));
    output.meta.insert("track_step_calls".into(), json!(0));
    Ok(output)
}

fn mask_bbox(mask: &MlxArray, width: u32, height: u32) -> [f32; 4] {
    let values = mlxcel_core::array_to_raw_bytes(mask);
    let mut bounds = [width, height, 0, 0];
    let mut found = false;
    for (index, value) in values.into_iter().enumerate() {
        if value == 0 {
            continue;
        }
        let x = index as u32 % width;
        let y = index as u32 / width;
        bounds[0] = bounds[0].min(x);
        bounds[1] = bounds[1].min(y);
        bounds[2] = bounds[2].max(x);
        bounds[3] = bounds[3].max(y);
        found = true;
    }
    if found {
        bounds.map(|value| value as f32)
    } else {
        [0.0; 4]
    }
}

fn box_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let width = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let height = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let intersection = width * height;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "video_tests.rs"]
mod tests;
