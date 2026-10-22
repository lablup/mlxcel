// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Sapiens2 human-centric pose and dense prediction.

mod backbone;
mod config;
mod deconv;
#[cfg(test)]
mod diagnostic;
mod heads;
mod model;
mod pose;
mod processor;
mod weights;

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

use self::config::Sapiens2Task;
use self::model::{ModelOutput, Sapiens2Model};
use self::processor::{image_to_nhwc_255, normalize_sapiens2, preprocess_image, remove_padding};
use super::interpolate::resize_bilinear_nhwc;
use super::{DenseInput, DenseOutput, DensePredictor, PersonKeypoints};

const BACKBONE_TASK: &[&str] = &["backbone"];
const POSE_TASK: &[&str] = &["pose"];
const SEG_TASK: &[&str] = &["seg"];
const NORMAL_TASK: &[&str] = &["normal"];
const POINTMAP_TASK: &[&str] = &["pointmap"];
const MATTING_TASK: &[&str] = &["matting"];
const MAX_POSE_BOXES: usize = 16;

pub struct Sapiens2Predictor {
    model: Sapiens2Model,
}

pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
    Ok(Box::new(Sapiens2Predictor {
        model: Sapiens2Model::load(path)?,
    }))
}

impl DensePredictor for Sapiens2Predictor {
    fn family(&self) -> &'static str {
        "sapiens2"
    }

    fn tasks(&self) -> &[&'static str] {
        match self.model.config.task {
            Sapiens2Task::Backbone => BACKBONE_TASK,
            Sapiens2Task::Pose => POSE_TASK,
            Sapiens2Task::Segmentation => SEG_TASK,
            Sapiens2Task::Normal => NORMAL_TASK,
            Sapiens2Task::Pointmap => POINTMAP_TASK,
            Sapiens2Task::Matting => MATTING_TASK,
        }
    }

    fn predict(&self, input: &DenseInput, opts: &Map<String, Value>) -> Result<DenseOutput> {
        let DenseInput::Image(image) = input else {
            bail!("Sapiens2 supports still images; video and webcam streaming are out of scope")
        };
        if self.model.config.task == Sapiens2Task::Pose {
            return predict_pose(&self.model, image, opts);
        }
        predict_dense(&self.model, image)
    }
}

fn predict_pose(
    model: &Sapiens2Model,
    image: &image::DynamicImage,
    opts: &Map<String, Value>,
) -> Result<DenseOutput> {
    let boxes = parse_pose_boxes(opts, image.width(), image.height())?;
    let flip_test = match opts.get("flip_test") {
        None => true,
        Some(Value::Bool(value)) => *value,
        Some(_) => bail!("Sapiens2 pose option `flip_test` must be a boolean"),
    };
    if boxes.is_empty() {
        return Ok(empty_pose_output(model.config.num_labels));
    }

    let source = image_to_nhwc_255(image);
    let input_size = (model.config.image_size.1, model.config.image_size.0);
    let mut keypoint_batches = Vec::with_capacity(boxes.len());
    let mut score_batches = Vec::with_capacity(boxes.len());
    for &bbox in &boxes {
        let (crop, center, scale) =
            pose::warp_to_input(&source, bbox, input_size).context("Sapiens2 pose crop")?;
        let (keypoints, scores) = infer_pose_crop(
            model,
            normalize_sapiens2(&crop)?,
            input_size,
            center,
            scale,
            flip_test,
        )?;
        keypoint_batches.push(keypoints);
        score_batches.push(scores);
    }
    let keypoints = concatenate_batch(&keypoint_batches);
    let scores = concatenate_batch(&score_batches);
    build_pose_output(model, boxes, keypoints, scores)
}

fn infer_pose_crop(
    model: &Sapiens2Model,
    crop: mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    input_size: (usize, usize),
    center: [f32; 2],
    scale: [f32; 2],
    flip_test: bool,
) -> Result<(
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
)> {
    let mut batch = crop;
    if flip_test {
        let width_indices: Vec<i32> = (0..input_size.0 as i32).rev().collect();
        let width_indices = mlxcel_core::from_slice_i32(&width_indices, &[input_size.0 as i32]);
        let flipped = mlxcel_core::take(&batch, &width_indices, 2);
        batch = mlxcel_core::concatenate(&batch, &flipped, 0);
    }

    let ModelOutput::Task {
        prediction: heatmaps,
        ..
    } = model.forward(&batch)?
    else {
        bail!("Sapiens2 pose checkpoint returned backbone features")
    };
    let heatmap_shape = mlxcel_core::array_shape(&heatmaps);
    ensure!(
        heatmap_shape.len() == 4
            && heatmap_shape[0] == if flip_test { 2 } else { 1 }
            && heatmap_shape[3] == model.config.num_labels as i32,
        "Sapiens2 pose head returned unexpected shape {heatmap_shape:?}"
    );
    let original = mlxcel_core::slice(
        &heatmaps,
        &[0, 0, 0, 0],
        &[1, heatmap_shape[1], heatmap_shape[2], heatmap_shape[3]],
    );
    let heatmaps = if flip_test {
        let flipped = mlxcel_core::slice(
            &heatmaps,
            &[1, 0, 0, 0],
            &[2, heatmap_shape[1], heatmap_shape[2], heatmap_shape[3]],
        );
        let width_indices: Vec<i32> = (0..heatmap_shape[2]).rev().collect();
        let width_indices = mlxcel_core::from_slice_i32(&width_indices, &[heatmap_shape[2]]);
        let flipped = mlxcel_core::take(&flipped, &width_indices, 2);
        let pairs: Vec<[usize; 2]> = model
            .config
            .flip_pairs
            .iter()
            .map(|&(left, right)| [left, right])
            .collect();
        let keypoint_indices = pose::flip_indices_from_pairs(model.config.num_labels, &pairs)?;
        let keypoint_indices =
            mlxcel_core::from_slice_i32(&keypoint_indices, &[model.config.num_labels as i32]);
        let flipped = mlxcel_core::take(&flipped, &keypoint_indices, 3);
        let original = mlxcel_core::astype(&original, mlxcel_core::dtype::FLOAT32);
        let flipped = mlxcel_core::astype(&flipped, mlxcel_core::dtype::FLOAT32);
        let half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::multiply(&mlxcel_core::add(&original, &flipped), &half)
    } else {
        original
    };
    let (crop_keypoints, scores) = pose::udp_decode_batch(&heatmaps, input_size, 11)?;
    let keypoints = pose::keypoints_to_image(&crop_keypoints, input_size, &[center], &[scale])?;
    Ok((keypoints, scores))
}

fn parse_pose_boxes(opts: &Map<String, Value>, width: u32, height: u32) -> Result<Vec<[f32; 4]>> {
    ensure!(
        width > 0 && height > 0,
        "Sapiens2 pose image must not be empty"
    );
    let Some(value) = opts.get("boxes") else {
        return pose::clip_boxes(
            &[[0.0, 0.0, (width - 1) as f32, (height - 1) as f32]],
            width,
            height,
        );
    };
    let rows = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Sapiens2 pose option `boxes` must be an array"))?;
    ensure!(
        rows.len() <= MAX_POSE_BOXES,
        "Sapiens2 pose accepts at most {MAX_POSE_BOXES} boxes per call"
    );
    let mut boxes = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let values = row.as_array().ok_or_else(|| {
            anyhow::anyhow!("Sapiens2 pose box {index} must be an array of four numbers")
        })?;
        ensure!(
            values.len() == 4,
            "Sapiens2 pose box {index} must contain exactly four numbers"
        );
        let mut bbox = [0.0f32; 4];
        for (axis, value) in values.iter().enumerate() {
            let value = value.as_f64().ok_or_else(|| {
                anyhow::anyhow!("Sapiens2 pose box {index} coordinate {axis} must be a number")
            })?;
            ensure!(
                value.is_finite() && value >= f32::MIN as f64 && value <= f32::MAX as f64,
                "Sapiens2 pose box {index} coordinate {axis} must be finite"
            );
            bbox[axis] = value as f32;
        }
        boxes.push(bbox);
    }
    pose::clip_boxes(&boxes, width, height)
}

fn concatenate_batch(
    arrays: &[mlxcel_core::UniquePtr<mlxcel_core::MlxArray>],
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut batch = mlxcel_core::copy(&arrays[0]);
    for array in &arrays[1..] {
        batch = mlxcel_core::concatenate(&batch, array, 0);
    }
    batch
}

fn empty_pose_output(keypoints: usize) -> DenseOutput {
    let mut output = DenseOutput::default();
    output.tensors.insert(
        "keypoints".into(),
        mlxcel_core::zeros(&[0, keypoints as i32, 2], mlxcel_core::dtype::FLOAT32),
    );
    output.tensors.insert(
        "scores".into(),
        mlxcel_core::zeros(&[0, keypoints as i32], mlxcel_core::dtype::FLOAT32),
    );
    output.tensors.insert(
        "boxes".into(),
        mlxcel_core::zeros(&[0, 4], mlxcel_core::dtype::FLOAT32),
    );
    output
}

fn build_pose_output(
    model: &Sapiens2Model,
    boxes: Vec<[f32; 4]>,
    keypoints: mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    scores: mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
) -> Result<DenseOutput> {
    let coordinates = mlxcel_core::utils::array_to_vec_f32(&keypoints);
    let confidences = mlxcel_core::utils::array_to_vec_f32(&scores);
    let count = model.config.num_labels;
    ensure!(
        coordinates.len() == boxes.len() * count * 2 && confidences.len() == boxes.len() * count,
        "Sapiens2 pose decode returned inconsistent result lengths"
    );
    let people = boxes
        .iter()
        .enumerate()
        .map(|(person, &bbox)| PersonKeypoints {
            bbox,
            points: (0..count)
                .map(|index| {
                    let coordinate = (person * count + index) * 2;
                    [
                        coordinates[coordinate],
                        coordinates[coordinate + 1],
                        confidences[person * count + index],
                    ]
                })
                .collect(),
            names: model.config.keypoint_names.clone(),
        })
        .collect();
    let flat_boxes: Vec<f32> = boxes.iter().flatten().copied().collect();
    let mut output = DenseOutput {
        keypoints: people,
        ..DenseOutput::default()
    };
    output.tensors.insert("keypoints".into(), keypoints);
    output.tensors.insert("scores".into(), scores);
    output.tensors.insert(
        "boxes".into(),
        mlxcel_core::from_slice_f32(&flat_boxes, &[boxes.len() as i32, 4]),
    );
    Ok(output)
}

fn predict_dense(model: &Sapiens2Model, image: &image::DynamicImage) -> Result<DenseOutput> {
    let original = (image.height() as usize, image.width() as usize);
    let processed = preprocess_image(image, &model.config)?;
    match model.forward(&processed.pixels)? {
        ModelOutput::Backbone {
            last_hidden_state,
            pooler_output,
        } => {
            let mut output = DenseOutput::default();
            output.tensors.insert(
                "last_hidden_state".into(),
                mlxcel_core::astype(&last_hidden_state, mlxcel_core::dtype::FLOAT32),
            );
            output.tensors.insert(
                "pooler_output".into(),
                mlxcel_core::astype(&pooler_output, mlxcel_core::dtype::FLOAT32),
            );
            Ok(output)
        }
        ModelOutput::Task { prediction, scale } => {
            let mut output = DenseOutput::default();
            match model.config.task {
                Sapiens2Task::Segmentation => {
                    let prediction = resize_prediction(&prediction, processed.padding, original)?;
                    let labels = mlxcel_core::argmax(&prediction, 3, false);
                    output.tensors.insert(
                        "segmentation".into(),
                        squeeze_batch(&mlxcel_core::astype(&labels, mlxcel_core::dtype::INT32)),
                    );
                }
                Sapiens2Task::Normal => {
                    let prediction = mlxcel_core::astype(&prediction, mlxcel_core::dtype::FLOAT32);
                    let norm = mlxcel_core::linalg_norm(&prediction, 3, true);
                    let floor = mlxcel_core::full_f32(&[1], 1e-8, mlxcel_core::dtype::FLOAT32);
                    let normalized =
                        mlxcel_core::divide(&prediction, &mlxcel_core::maximum(&norm, &floor));
                    let normalized = resize_prediction(&normalized, processed.padding, original)?;
                    output
                        .tensors
                        .insert("normals".into(), squeeze_batch(&normalized));
                }
                Sapiens2Task::Pointmap => {
                    let prediction = resize_prediction(
                        &mlxcel_core::astype(&prediction, mlxcel_core::dtype::FLOAT32),
                        processed.padding,
                        original,
                    )?;
                    output
                        .tensors
                        .insert("pointmaps".into(), squeeze_batch(&prediction));
                    let scale = scale.ok_or_else(|| {
                        anyhow::anyhow!("Sapiens2 pointmap head returned no scale")
                    })?;
                    output.tensors.insert(
                        "scales".into(),
                        squeeze_batch(&mlxcel_core::astype(&scale, mlxcel_core::dtype::FLOAT32)),
                    );
                }
                Sapiens2Task::Matting => {
                    let rgba = mlxcel_core::sigmoid(&prediction);
                    let rgba = resize_bilinear_nhwc(
                        &mlxcel_core::astype(&rgba, mlxcel_core::dtype::FLOAT32),
                        original,
                        false,
                        false,
                    )?;
                    let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
                    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
                    let rgba = mlxcel_core::clip(&rgba, &zero, &one);
                    let shape = mlxcel_core::array_shape(&rgba);
                    let foreground =
                        mlxcel_core::slice(&rgba, &[0, 0, 0, 0], &[1, shape[1], shape[2], 3]);
                    let alpha =
                        mlxcel_core::slice(&rgba, &[0, 0, 0, 3], &[1, shape[1], shape[2], 4]);
                    output
                        .tensors
                        .insert("foregrounds".into(), squeeze_batch(&foreground));
                    output.tensors.insert(
                        "alphas".into(),
                        mlxcel_core::reshape(&alpha, &[shape[1], shape[2]]),
                    );
                }
                Sapiens2Task::Backbone | Sapiens2Task::Pose => {
                    unreachable!("handled outside dense task output")
                }
            }
            Ok(output)
        }
    }
}

fn resize_prediction(
    prediction: &mlxcel_core::MlxArray,
    padding: Option<processor::Padding>,
    target: (usize, usize),
) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> {
    let unpadded = if let Some(padding) = padding {
        remove_padding(prediction, padding)
    } else {
        mlxcel_core::copy(prediction)
    };
    resize_bilinear_nhwc(&unpadded, target, false, false)
}

fn squeeze_batch(x: &mlxcel_core::MlxArray) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let output = &shape[1..];
    mlxcel_core::reshape(x, output)
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
