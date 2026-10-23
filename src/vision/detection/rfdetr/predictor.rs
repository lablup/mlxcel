// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! RF-DETR predictor and checked device-to-host output seam.

use std::path::Path;

use image::{DynamicImage, GrayImage};
use mlxcel_core::MlxArray;

use crate::vision::dense::interpolate::resize_bicubic_nhwc;
use crate::vision::detection::{Detection, DetectionResult, Detector};

use super::model::{RfDetrModel, RfDetrOutput};
use super::postprocess::{DecodedDetection, postprocess};
use super::processor::{ProcessedImage, RfDetrProcessor};

pub const DEFAULT_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.5;
const MAX_OUTPUT_MASK_ELEMENTS: usize = 32 * 1024 * 1024;

pub struct RfDetrPredictor {
    model: RfDetrModel,
    processor: RfDetrProcessor,
    threshold: f32,
}

impl RfDetrPredictor {
    pub fn from_pretrained<P: AsRef<Path>>(directory: P, threshold: f32) -> Result<Self, String> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(format!(
                "RF-DETR confidence threshold must be finite and in [0, 1], got {threshold}"
            ));
        }
        let directory = directory.as_ref();
        let model = RfDetrModel::load(directory)?;
        let processor = RfDetrProcessor::from_pretrained(directory, model.config())?;
        Ok(Self {
            model,
            processor,
            threshold,
        })
    }

    pub fn model(&self) -> &RfDetrModel {
        &self.model
    }

    pub fn processor(&self) -> &RfDetrProcessor {
        &self.processor
    }

    pub fn predict_path<P: AsRef<Path>>(&self, path: P) -> Result<DetectionResult, String> {
        self.predict_path_with_masks(path, true)
    }

    pub fn predict_path_with_masks<P: AsRef<Path>>(
        &self,
        path: P,
        include_masks: bool,
    ) -> Result<DetectionResult, String> {
        let path = path.as_ref();
        let processed = self.processor.preprocess_path(path)?;
        self.predict_processed(processed, include_masks)
    }

    pub fn predict_image(&self, image: &DynamicImage) -> Result<DetectionResult, String> {
        let processed = self.processor.preprocess(image)?;
        self.predict_processed(processed, true)
    }

    fn predict_processed(
        &self,
        processed: ProcessedImage,
        include_masks: bool,
    ) -> Result<DetectionResult, String> {
        let output = self
            .model
            .forward_with_masks(&processed.pixel_values, include_masks)?;
        self.decode_output(&output, processed.original_size)
    }

    fn decode_output(
        &self,
        output: &RfDetrOutput,
        original_size: (u32, u32),
    ) -> Result<DetectionResult, String> {
        let queries = self.model.config().num_queries;
        let classes = self.model.config().logits_classes();
        check_shape("pred_logits", &output.pred_logits, &[1, queries, classes])?;
        check_shape("pred_boxes", &output.pred_boxes, &[1, queries, 4])?;
        let logits = read_output_f32(&output.pred_logits);
        let boxes = read_output_f32(&output.pred_boxes);
        let (height, width) = original_size;
        let class_names = self.model.config().class_names();
        let decoded = postprocess(
            &logits,
            &boxes,
            queries,
            classes,
            width,
            height,
            self.threshold,
            self.processor.num_select(),
            NMS_THRESHOLD,
            class_names.as_deref(),
        )?;
        let masks = match &output.pred_masks {
            Some(pred_masks) => decode_masks(pred_masks, &decoded, queries, width, height)?,
            None => vec![None; decoded.len()],
        };
        let detections = decoded
            .into_iter()
            .zip(masks)
            .map(|(decoded, mask)| Detection {
                bbox: decoded.bbox,
                score: decoded.score,
                label: decoded.label,
                class_name: decoded.class_name,
                reading_order: None,
                mask,
            })
            .collect();
        Ok(DetectionResult { detections })
    }
}

impl Detector for RfDetrPredictor {
    fn predict_path(&self, path: &Path) -> Result<DetectionResult, String> {
        RfDetrPredictor::predict_path(self, path)
    }

    fn predict_path_with_masks(
        &self,
        path: &Path,
        include_masks: bool,
    ) -> Result<DetectionResult, String> {
        RfDetrPredictor::predict_path_with_masks(self, path, include_masks)
    }

    fn name(&self) -> &'static str {
        "rf-detr"
    }

    fn supports_masks(&self) -> bool {
        self.model.config().supports_masks()
    }
}

fn check_shape(name: &str, value: &MlxArray, expected: &[usize]) -> Result<(), String> {
    let expected: Vec<i32> = expected
        .iter()
        .copied()
        .map(|dimension| {
            i32::try_from(dimension)
                .map_err(|_| format!("RF-DETR {name} expected shape exceeds i32"))
        })
        .collect::<Result<_, _>>()?;
    let actual = mlxcel_core::array_shape(value);
    if actual != expected {
        return Err(format!(
            "RF-DETR {name} shape mismatch: got {actual:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

fn read_output_f32(output: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(output, mlxcel_core::dtype::FLOAT32))
}

pub(crate) fn decode_masks(
    pred_masks: &MlxArray,
    detections: &[DecodedDetection],
    num_queries: usize,
    width: u32,
    height: u32,
) -> Result<Vec<Option<GrayImage>>, String> {
    let shape = mlxcel_core::array_shape(pred_masks);
    let expected_queries =
        i32::try_from(num_queries).map_err(|_| "RF-DETR query count exceeds i32".to_string())?;
    if shape.len() != 4
        || shape[0] != 1
        || shape[1] != expected_queries
        || shape[2] <= 0
        || shape[3] <= 0
    {
        return Err(format!(
            "RF-DETR mask logits must be [1,{num_queries},H,W], got {shape:?}"
        ));
    }
    if detections.is_empty() {
        return Ok(Vec::new());
    }
    let pixels = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| "RF-DETR output mask size overflows usize".to_string())?;
    let total = pixels
        .checked_mul(detections.len())
        .ok_or_else(|| "RF-DETR output mask workset overflows usize".to_string())?;
    if total > MAX_OUTPUT_MASK_ELEMENTS {
        return Err(format!(
            "RF-DETR output masks require {total} elements; limit is {MAX_OUTPUT_MASK_ELEMENTS}"
        ));
    }
    let indices: Vec<i32> = detections
        .iter()
        .map(|detection| {
            i32::try_from(detection.query_index)
                .map_err(|_| "RF-DETR selected query index exceeds i32".to_string())
        })
        .collect::<Result<_, _>>()?;
    let indices = mlxcel_core::from_slice_i32(&indices, &[indices.len() as i32]);
    let selected = mlxcel_core::take(pred_masks, &indices, 1);
    let nhwc = mlxcel_core::transpose_axes(&selected, &[0, 2, 3, 1]);
    let resized = resize_bicubic_nhwc(
        &mlxcel_core::astype(&nhwc, mlxcel_core::dtype::FLOAT32),
        Some((height as usize, width as usize)),
        None,
        false,
    )
    .map_err(|error| format!("RF-DETR output mask resize failed: {error}"))?;
    let values = mlxcel_core::utils::array_to_vec_f32(&resized);
    if values.len() != total || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "RF-DETR resized mask readback is invalid: got {} values, expected {total}",
            values.len()
        ));
    }
    let count = detections.len();
    let mut planes = vec![Vec::with_capacity(pixels); count];
    for pixel in values.chunks_exact(count) {
        for (plane, value) in planes.iter_mut().zip(pixel) {
            plane.push(if *value > 0.0 { 255 } else { 0 });
        }
    }
    planes
        .into_iter()
        .map(|plane| {
            GrayImage::from_raw(width, height, plane)
                .map(Some)
                .ok_or_else(|| "RF-DETR failed to construct output mask image".to_string())
        })
        .collect()
}
