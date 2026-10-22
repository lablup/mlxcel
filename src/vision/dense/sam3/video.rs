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

//! Text/box initialization and memory-based video mask propagation.

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use super::super::{DenseOutput, Instance};
use super::image::select_and_resize_masks;
use super::model::{DetectorFeatures, DetectorOutput, Sam3Model, TrackerBundle};
use super::options::PredictionOptions;
use super::postprocess::{clip_xyxy, confidence_scores};
use super::processing::Sam3Processor;

struct ObjectState {
    id: u32,
    label: String,
    memory: Vec<UniquePtr<MlxArray>>,
}

struct InitialMask {
    mask: UniquePtr<MlxArray>,
    score: f32,
    bbox: Option<[f32; 4]>,
}

pub(crate) fn track_video(
    model: &Sam3Model,
    processor: &Sam3Processor,
    frames: &[DynamicImage],
    options: &PredictionOptions,
) -> Result<DenseOutput> {
    ensure!(!frames.is_empty(), "SAM 3 track received no video frames");
    let width = frames[0].width();
    let height = frames[0].height();
    ensure!(
        frames
            .iter()
            .all(|frame| frame.width() == width && frame.height() == height),
        "SAM 3 track requires every decoded frame to have the same dimensions"
    );
    let tracker = model
        .tracker
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SAM 3 checkpoint has no tracker weights"))?;

    let first_pixels = processor
        .preprocess_image(&frames[0])
        .context("preprocess SAM 3 video frame 0")?;
    let first_backbone = model.backbone(&first_pixels);
    mlxcel_core::eval(&first_backbone);

    let first_detector = model
        .detector_features(&first_backbone)
        .map_err(anyhow::Error::msg)?;
    let first_tracker_fpn = tracker.neck.forward(&first_backbone);
    let track_feature = tracker_feature(&first_tracker_fpn)?;
    let mut states = Vec::with_capacity(options.prompts.len() + options.boxes.len());
    let mut initial = Vec::with_capacity(states.capacity());

    for prompt in &options.prompts {
        let Some(mask) = initialize_text(
            model,
            processor,
            &first_detector,
            prompt,
            width,
            height,
            options.threshold,
        )?
        else {
            continue;
        };
        states.push(ObjectState {
            id: states.len() as u32 + 1,
            label: prompt.clone(),
            memory: vec![encode_binary_memory(tracker, track_feature, &mask.mask)?],
        });
        initial.push(mask);
    }
    for (index, bbox) in options.boxes.iter().enumerate() {
        let mask = initialize_box(tracker, track_feature, *bbox, width, height)?;
        states.push(ObjectState {
            id: states.len() as u32 + 1,
            label: format!("box {}", index + 1),
            memory: vec![encode_binary_memory(tracker, track_feature, &mask.mask)?],
        });
        initial.push(mask);
    }
    drop(first_detector);
    drop(first_tracker_fpn);
    drop(first_backbone);

    let mut instances = Vec::with_capacity(frames.len() * states.len());
    let mut masks = Vec::with_capacity(instances.capacity());
    append_frame(
        0,
        &states,
        initial,
        width,
        height,
        &mut instances,
        &mut masks,
    )?;

    for (frame, source_frame) in frames.iter().enumerate().skip(1) {
        if states.is_empty() {
            break;
        }
        let pixels = processor
            .preprocess_image(source_frame)
            .with_context(|| format!("preprocess SAM 3 video frame {frame}"))?;
        let backbone = model.backbone(&pixels);
        let fpn = tracker.neck.forward(&backbone);
        let current = tracker_feature(&fpn)?;
        // Preserve the pinned mlx-vlm propagation order. Its mask decoder's
        // shape guards skip these high-resolution additions because it passes
        // 288 then 144 where the learned skip convolutions expect 144 then
        // 288. Reordering changes reference masks and belongs with an upstream
        // fix plus regenerated parity fixtures.
        let high_res = [&*fpn[0], &*fpn[1]];
        let mut results = Vec::with_capacity(states.len());
        for state in &mut states {
            let memory: Vec<&MlxArray> = state.memory.iter().map(|value| &**value).collect();
            let step = tracker
                .model
                .track_step_upstream(current, &memory, None, None, None, false, Some(&high_res))
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("track object {} on frame {frame}", state.id))?;
            let score = first_f32(&step.iou_scores)?;
            let mask = binary_low_mask(&step.pred_masks);
            state
                .memory
                .push(encode_binary_memory(tracker, current, &mask)?);
            if state.memory.len() > model.config.tracker_config.num_maskmem {
                state.memory.remove(0);
            }
            results.push(InitialMask {
                mask,
                score,
                bbox: None,
            });
        }
        append_frame(
            frame as u32,
            &states,
            results,
            width,
            height,
            &mut instances,
            &mut masks,
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
    let refs: Vec<&MlxArray> = masks.iter().map(|value| &**value).collect();
    output.tensors.insert(
        "masks".into(),
        if refs.is_empty() {
            mlxcel_core::from_bytes(
                &[],
                &[0, height as i32, width as i32],
                mlxcel_core::dtype::UINT8,
            )
        } else {
            mlxcel_core::concatenate_many(&refs, 0)
        },
    );
    output
        .meta
        .insert("threshold".into(), json!(options.threshold));
    output.meta.insert("prompts".into(), json!(options.prompts));
    output
        .meta
        .insert("box_prompts".into(), json!(options.boxes.len()));
    Ok(output)
}

fn initialize_text(
    model: &Sam3Model,
    processor: &Sam3Processor,
    features: &DetectorFeatures,
    prompt: &str,
    width: u32,
    height: u32,
    threshold: f32,
) -> Result<Option<InitialMask>> {
    let tokens = processor.tokenize(prompt)?;
    let text = model.prompt_features(&tokens.input_ids, &tokens.attention_mask);
    let output = model
        .detect(features, &text, &tokens.attention_mask, None, true)
        .map_err(anyhow::Error::msg)?;
    best_detection(output, width, height, threshold)
}

fn best_detection(
    output: DetectorOutput,
    width: u32,
    height: u32,
    threshold: f32,
) -> Result<Option<InitialMask>> {
    let logits = mlxcel_core::astype(&output.logits, mlxcel_core::dtype::FLOAT32);
    let presence = mlxcel_core::astype(&output.presence, mlxcel_core::dtype::FLOAT32);
    let boxes = mlxcel_core::astype(&output.boxes, mlxcel_core::dtype::FLOAT32);
    let logits = mlxcel_core::utils::array_to_vec_f32(&logits);
    let presence = mlxcel_core::utils::array_to_vec_f32(&presence);
    let boxes = mlxcel_core::utils::array_to_vec_f32(&boxes);
    let scores = confidence_scores(&logits, presence[0]);
    let Some(query) = best_query_at_threshold(&scores, threshold) else {
        return Ok(None);
    };
    let offset = query * 4;
    let bbox = clip_xyxy(
        [
            boxes[offset] * width as f32,
            boxes[offset + 1] * height as f32,
            boxes[offset + 2] * width as f32,
            boxes[offset + 3] * height as f32,
        ],
        width,
        height,
    );
    let masks = output
        .masks
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("SAM 3 text initialization omitted masks"))?;
    let masks = mlxcel_core::squeeze_axis(masks, 0);
    let index = mlxcel_core::from_slice_i32(&[query as i32], &[1]);
    let mask = mlxcel_core::expand_dims(&mlxcel_core::take(&masks, &index, 0), 0);
    Ok(Some(InitialMask {
        mask: binary_low_mask(&mask),
        score: scores[query],
        bbox: Some(bbox),
    }))
}

fn best_query_at_threshold(scores: &[f32], threshold: f32) -> Option<usize> {
    let query = scores
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)?;
    (scores[query] > threshold).then_some(query)
}

fn initialize_box(
    tracker: &TrackerBundle,
    feature: &MlxArray,
    bbox: [f32; 4],
    width: u32,
    height: u32,
) -> Result<InitialMask> {
    let scaled = [
        (bbox[0] / width as f32) + 0.5 / 1008.0,
        (bbox[1] / height as f32) + 0.5 / 1008.0,
        (bbox[2] / width as f32) + 0.5 / 1008.0,
        (bbox[3] / height as f32) + 0.5 / 1008.0,
    ];
    let boxes = mlxcel_core::from_slice_f32(&scaled, &[1, 1, 4]);
    let step = tracker
        .model
        .track_step_upstream(feature, &[], None, Some(&boxes), None, false, None)
        .map_err(anyhow::Error::msg)?;
    Ok(InitialMask {
        mask: binary_low_mask(&step.pred_masks),
        score: first_f32(&step.iou_scores)?,
        bbox: Some(clip_xyxy(bbox, width, height)),
    })
}

fn append_frame(
    frame: u32,
    states: &[ObjectState],
    results: Vec<InitialMask>,
    width: u32,
    height: u32,
    instances: &mut Vec<Instance>,
    masks: &mut Vec<UniquePtr<MlxArray>>,
) -> Result<()> {
    for (state, result) in states.iter().zip(results) {
        let resized = select_and_resize_masks(&result.mask, &[0], width, height)?;
        let bbox = result
            .bbox
            .unwrap_or_else(|| mask_bbox(&resized, width, height));
        let mask_index = masks.len();
        masks.push(resized);
        instances.push(Instance {
            bbox,
            score: result.score,
            label: state.label.clone(),
            mask_index: Some(mask_index),
            object_id: Some(state.id),
            frame: Some(frame),
        });
    }
    Ok(())
}

fn tracker_feature(fpn: &[UniquePtr<MlxArray>]) -> Result<&MlxArray> {
    fpn.get(2)
        .map(|value| &**value)
        .ok_or_else(|| anyhow::anyhow!("SAM 3 tracker neck returned fewer than three levels"))
}

fn binary_low_mask(mask: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(
        &mlxcel_core::greater(mask, &mlxcel_core::zeros_like(mask)),
        mlxcel_core::dtype::UINT8,
    )
}

fn encode_binary_memory(
    tracker: &TrackerBundle,
    feature: &MlxArray,
    mask: &MlxArray,
) -> Result<UniquePtr<MlxArray>> {
    let mask = mlxcel_core::astype(mask, mlxcel_core::dtype::FLOAT32);
    let mask = mlxcel_core::transpose_axes(&mask, &[0, 2, 3, 1]);
    let mask = nearest_resize_nhwc(&mask, 1152, 1152);
    let memory = tracker
        .model
        .memory_encoder
        .forward(feature, &mask)
        .map_err(anyhow::Error::msg)?;
    let shape = mlxcel_core::array_shape(&memory);
    Ok(mlxcel_core::reshape(
        &memory,
        &[shape[0], shape[1] * shape[2], shape[3]],
    ))
}

fn nearest_resize_nhwc(x: &MlxArray, height: i32, width: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let rows: Vec<i32> = (0..height).map(|row| row * shape[1] / height).collect();
    let columns: Vec<i32> = (0..width).map(|column| column * shape[2] / width).collect();
    let rows = mlxcel_core::from_slice_i32(&rows, &[height]);
    let columns = mlxcel_core::from_slice_i32(&columns, &[width]);
    mlxcel_core::take(&mlxcel_core::take(x, &rows, 1), &columns, 2)
}

fn first_f32(value: &MlxArray) -> Result<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(value, mlxcel_core::dtype::FLOAT32))
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("SAM 3 tracker returned no score"))
}

fn mask_bbox(mask: &MlxArray, width: u32, height: u32) -> [f32; 4] {
    let values = mlxcel_core::array_to_raw_bytes(mask);
    let mut bbox = [width, height, 0, 0];
    let mut found = false;
    for (index, value) in values.into_iter().enumerate() {
        if value == 0 {
            continue;
        }
        let x = (index as u32) % width;
        let y = (index as u32) / width;
        bbox[0] = bbox[0].min(x);
        bbox[1] = bbox[1].min(y);
        bbox[2] = bbox[2].max(x + 1);
        bbox[3] = bbox[3].max(y + 1);
        found = true;
    }
    if found {
        bbox.map(|value| value as f32)
    } else {
        [0.0; 4]
    }
}

#[cfg(test)]
mod tests {
    use super::best_query_at_threshold;

    #[test]
    fn video_threshold_only_initializes_a_passing_query() {
        assert_eq!(best_query_at_threshold(&[0.1, 0.6, 0.4], 0.5), Some(1));
        assert_eq!(best_query_at_threshold(&[0.1, 0.6, 0.4], 0.6), None);
        assert_eq!(best_query_at_threshold(&[0.1, 0.6, 0.4], 0.7), None);
        assert_eq!(best_query_at_threshold(&[], 0.0), None);
    }
}
