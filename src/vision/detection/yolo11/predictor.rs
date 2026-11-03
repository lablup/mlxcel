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

//! YOLO11 image preparation, NMS and the `Detector` implementation.
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/yolo11/inference.py:
//! centred letterbox to a multiple of 32 on a `(114, 114, 114)` canvas, then
//! class-aware greedy NMS and the inverse letterbox transform.

use std::path::Path;

use image::DynamicImage;
use mlxcel_core::MlxArray;

use crate::vision::detection::{Detection, DetectionResult, Detector, DetectorOptions};

use super::model::Yolo11Model;

pub const DEFAULT_THRESHOLD: f32 = 0.05;
pub const DEFAULT_IOU: f32 = 0.1;
const LETTERBOX_STRIDE: u32 = 32;
const LETTERBOX_FILL: u8 = 114;
const MAX_DETECTIONS: usize = 300;
const MAX_NMS_CANDIDATES: usize = 30_000;

/// One post-NMS box in input pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct YoloDetection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub label: usize,
}

/// Letterboxed detector input and the transform needed to undo it.
pub struct Letterbox {
    /// `(1, padded_height, padded_width, 3)` NHWC in `[0, 1]`.
    pub pixel_values: mlxcel_core::UniquePtr<MlxArray>,
    pub original_size: (u32, u32),
    pub gain: f32,
    pub left: i32,
    pub top: i32,
}

pub struct Yolo11Predictor {
    model: Yolo11Model,
    threshold: f32,
    iou: f32,
    size: Option<usize>,
}

impl Yolo11Predictor {
    pub fn from_pretrained<P: AsRef<Path>>(directory: P, threshold: f32) -> Result<Self, String> {
        validate_threshold(threshold)?;
        Ok(Self {
            model: Yolo11Model::load(directory)?,
            threshold,
            iou: DEFAULT_IOU,
            size: None,
        })
    }

    pub fn model(&self) -> &Yolo11Model {
        &self.model
    }

    /// Detect in an image file. `size` is the longest-side target (no resize
    /// when unset); `iou` is the NMS IoU threshold.
    pub fn predict_path_with_size(
        &self,
        path: &Path,
        size: Option<usize>,
        iou: Option<f32>,
    ) -> Result<DetectionResult, String> {
        let iou = iou.unwrap_or(self.iou);
        if !iou.is_finite() || !(0.0..=1.0).contains(&iou) {
            return Err(format!(
                "YOLO11 NMS IoU threshold must be finite and in [0, 1], got {iou}"
            ));
        }
        if size == Some(0) {
            return Err("YOLO11 --size must be a positive pixel count".into());
        }
        let image = image::open(path)
            .map_err(|error| format!("failed to open image {}: {error}", path.display()))?;
        self.predict_image(&image, size, iou)
    }

    /// Detect in an already decoded image.
    pub fn predict_image(
        &self,
        image: &DynamicImage,
        size: Option<usize>,
        iou: f32,
    ) -> Result<DetectionResult, String> {
        let letterbox = prepare_image(image, size);
        let output = self.model.forward(&letterbox.pixel_values)?;
        let shape = mlxcel_core::array_shape(&output);
        let channels = 4 + self.model.config().nc;
        if shape.len() != 3 || shape[0] != 1 || shape[1] as usize != channels || shape[2] <= 0 {
            return Err(format!(
                "YOLO11 prediction must be (1, {channels}, N), got {shape:?}"
            ));
        }
        let values = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
            &output,
            mlxcel_core::dtype::FLOAT32,
        ));
        let detections = non_max_suppression(
            &values,
            self.model.config().nc,
            shape[2] as usize,
            self.threshold,
            iou,
            MAX_DETECTIONS,
            MAX_NMS_CANDIDATES,
        )?;
        let (width, height) = (
            letterbox.original_size.0 as f32,
            letterbox.original_size.1 as f32,
        );
        let mapped = detections
            .into_iter()
            .map(|detection| {
                let inverse = |value: f32, offset: i32, limit: f32| {
                    ((value - offset as f32) / letterbox.gain).clamp(0.0, limit)
                };
                Detection {
                    bbox: [
                        inverse(detection.bbox[0], letterbox.left, width),
                        inverse(detection.bbox[1], letterbox.top, height),
                        inverse(detection.bbox[2], letterbox.left, width),
                        inverse(detection.bbox[3], letterbox.top, height),
                    ],
                    score: detection.score,
                    label: detection.label,
                    class_name: self.model.config().class_name(detection.label),
                    reading_order: None,
                    mask: None,
                }
            })
            .collect();
        Ok(DetectionResult { detections: mapped })
    }
}

impl Detector for Yolo11Predictor {
    fn predict_path(&self, path: &Path) -> Result<DetectionResult, String> {
        self.predict_path_with_size(path, self.size, Some(self.iou))
    }

    fn predict_path_with_options(
        &self,
        path: &Path,
        options: DetectorOptions,
    ) -> Result<DetectionResult, String> {
        self.predict_path_with_size(path, options.size.or(self.size), options.iou)
    }

    fn name(&self) -> &'static str {
        "yolo11"
    }
}

fn validate_threshold(threshold: f32) -> Result<(), String> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(format!(
            "YOLO11 confidence threshold must be finite and in [0, 1], got {threshold}"
        ));
    }
    Ok(())
}

/// Centred letterbox to a multiple of 32, mirroring `prepare_image`.
pub fn prepare_image(image: &DynamicImage, size: Option<usize>) -> Letterbox {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    let gain = match size {
        Some(size) => (size as f64 / width as f64).min(size as f64 / height as f64) as f32,
        None => 1.0,
    };
    let resized_width = python_round(width as f64 * gain as f64).max(1.0) as u32;
    let resized_height = python_round(height as f64 * gain as f64).max(1.0) as u32;
    let resized = if (resized_width, resized_height) != (width, height) {
        image::imageops::resize(
            &rgb,
            resized_width,
            resized_height,
            image::imageops::FilterType::Triangle,
        )
    } else {
        rgb
    };
    let padded_width = resized_width.div_ceil(LETTERBOX_STRIDE) * LETTERBOX_STRIDE;
    let padded_height = resized_height.div_ceil(LETTERBOX_STRIDE) * LETTERBOX_STRIDE;
    let left = ((padded_width - resized_width) / 2) as i32;
    let top = ((padded_height - resized_height) / 2) as i32;

    let fill = f32::from(LETTERBOX_FILL) / 255.0;
    let mut data = vec![fill; (padded_width * padded_height * 3) as usize];
    let row_width = (resized_width * 3) as usize;
    for row in 0..resized_height {
        let source_start = (row * resized_width * 3) as usize;
        let target_start = (((top as u32 + row) * padded_width) as usize + left as usize) * 3;
        for column in 0..row_width {
            data[target_start + column] =
                f32::from(resized.as_raw()[source_start + column]) / 255.0;
        }
    }
    Letterbox {
        pixel_values: mlxcel_core::from_slice_f32(
            &data,
            &[1, padded_height as i32, padded_width as i32, 3],
        ),
        original_size: (width, height),
        gain,
        left,
        top,
    }
}

/// Python's `round`: nearest integer with ties to even.
fn python_round(value: f64) -> f64 {
    value.round_ties_even()
}

/// Class-aware greedy NMS over a `(1, 4 + classes, anchors)` prediction.
#[allow(clippy::too_many_arguments)]
pub fn non_max_suppression(
    prediction: &[f32],
    classes: usize,
    anchors: usize,
    confidence: f32,
    iou_threshold: f32,
    max_detections: usize,
    max_candidates: usize,
) -> Result<Vec<YoloDetection>, String> {
    if classes == 0 {
        return Err("YOLO11 NMS needs at least one class".into());
    }
    if anchors == 0 {
        return Ok(Vec::new());
    }
    let channels = 4 + classes;
    if prediction.len() != channels * anchors {
        return Err(format!(
            "YOLO11 prediction has {} values; expected ({channels}, {anchors})",
            prediction.len()
        ));
    }
    if let Some((index, value)) = prediction
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("YOLO11 prediction[{index}] is not finite: {value}"));
    }
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err(format!(
            "YOLO11 confidence threshold must be finite and in [0, 1], got {confidence}"
        ));
    }
    if !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
        return Err(format!(
            "YOLO11 NMS IoU threshold must be finite and in [0, 1], got {iou_threshold}"
        ));
    }

    let value = |channel: usize, anchor: usize| prediction[channel * anchors + anchor];
    let mut candidates: Vec<(usize, f32, usize)> = Vec::new();
    for anchor in 0..anchors {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_label = 0usize;
        for label in 0..classes {
            let score = value(4 + label, anchor);
            if score > best_score {
                best_score = score;
                best_label = label;
            }
        }
        if best_score > confidence {
            candidates.push((anchor, best_score, best_label));
        }
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    // Descending score, later anchor first on an exact tie, matching a stable
    // ascending argsort that is then reversed.
    candidates.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| right.0.cmp(&left.0))
    });
    candidates.truncate(max_candidates.min(candidates.len()));

    let mut boxes = Vec::with_capacity(candidates.len());
    for (anchor, _, _) in &candidates {
        let (cx, cy, w, h) = (
            value(0, *anchor),
            value(1, *anchor),
            value(2, *anchor),
            value(3, *anchor),
        );
        boxes.push([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]);
    }
    let extent = boxes
        .iter()
        .flat_map(|bbox| bbox.iter())
        .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
    let offset_scale = extent + 1.0;
    let class_boxes: Vec<[f32; 4]> = boxes
        .iter()
        .zip(&candidates)
        .map(|(bbox, (_, _, label))| {
            let offset = *label as f32 * offset_scale;
            [
                bbox[0] + offset,
                bbox[1] + offset,
                bbox[2] + offset,
                bbox[3] + offset,
            ]
        })
        .collect();

    let mut keep: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();
    while !remaining.is_empty() && keep.len() < max_detections {
        let current = remaining[0];
        keep.push(current);
        if remaining.len() == 1 {
            break;
        }
        let mut survivors = Vec::with_capacity(remaining.len() - 1);
        for &candidate in &remaining[1..] {
            if box_iou(class_boxes[current], class_boxes[candidate]) < iou_threshold {
                survivors.push(candidate);
            }
        }
        remaining = survivors;
    }

    Ok(keep
        .into_iter()
        .map(|index| {
            let (anchor, score, label) = candidates[index];
            let _ = anchor;
            YoloDetection {
                bbox: boxes[index],
                score,
                label,
            }
        })
        .collect())
}

fn box_iou(first: [f32; 4], second: [f32; 4]) -> f32 {
    let intersection_width = (first[2].min(second[2]) - first[0].max(second[0])).max(0.0);
    let intersection_height = (first[3].min(second[3]) - first[1].max(second[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    let first_area = (first[2] - first[0]) * (first[3] - first[1]);
    let second_area = (second[2] - second[0]) * (second[3] - second[1]);
    let union = first_area + second_area - intersection;
    intersection / union.max(1e-7)
}
