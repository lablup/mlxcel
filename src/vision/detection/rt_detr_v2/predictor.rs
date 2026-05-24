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

//! RT-DETRv2 postprocessing and predictor.
//!
//! Port of `references/mlx-vlm/mlx_vlm/models/rt_detr_v2/generate.py`. The
//! predictor runs the model on one image and decodes detections: it does a flat
//! top-K over the `(queries x labels)` score space (so one query can yield
//! multiple detections — matching `RTDetrImageProcessor.post_process_object_
//! detection` for `use_focal_loss=True` models), thresholds, and converts the
//! normalized `(cx, cy, w, h)` boxes to `(l, t, r, b)` pixel coordinates in the
//! original image.

use std::path::Path;

use mlxcel_core::MlxArray;

use super::model::RtDetrV2Model;
use super::processor::RtDetrV2Processor;

/// Default confidence threshold (matches the upstream predictor).
pub const DEFAULT_THRESHOLD: f32 = 0.3;

/// One detected box.
#[derive(Debug, Clone)]
pub struct Detection {
    /// `(left, top, right, bottom)` in original-image pixel coordinates.
    pub bbox: [f32; 4],
    /// Confidence in `[0, 1]`.
    pub score: f32,
    /// Integer class id.
    pub label: usize,
    /// Human-readable class name (resolved via `config.id2label`, else the
    /// stringified id).
    pub class_name: String,
}

/// All detections for one image.
#[derive(Debug, Clone, Default)]
pub struct DetectionResult {
    pub detections: Vec<Detection>,
}

/// Inference wrapper over [`RtDetrV2Model`] + [`RtDetrV2Processor`].
pub struct RtDetrV2Predictor {
    model: RtDetrV2Model,
    processor: RtDetrV2Processor,
    threshold: f32,
    labels: Option<Vec<String>>,
}

impl RtDetrV2Predictor {
    /// Build a predictor from a model directory.
    pub fn from_pretrained<P: AsRef<Path>>(dir: P, threshold: f32) -> Result<Self, String> {
        let dir = dir.as_ref();
        let model = RtDetrV2Model::load(dir)?;
        let processor = RtDetrV2Processor::from_pretrained(dir)?;
        let labels = model.config().class_names();
        Ok(Self {
            model,
            processor,
            threshold,
            labels,
        })
    }

    pub fn model(&self) -> &RtDetrV2Model {
        &self.model
    }

    /// Detect objects in one image file.
    pub fn predict_path<P: AsRef<Path>>(&self, path: P) -> Result<DetectionResult, String> {
        let processed = self.processor.process_path(path)?;
        let out = self.model.forward(&processed.pixel_values);
        let (img_w, img_h) = processed.original_size;

        let logits_shape = mlxcel_core::array_shape(&out.pred_logits);
        // (B, Q, num_labels) — batch is 1.
        let q = logits_shape[1] as usize;
        let num_labels = logits_shape[2] as usize;

        let logits = to_f32_vec(&out.pred_logits); // length B*Q*num_labels
        let boxes = to_f32_vec(&out.pred_boxes); // length B*Q*4

        let detections =
            self.decode_one(&logits, &boxes, q, num_labels, img_w as f32, img_h as f32);
        Ok(DetectionResult { detections })
    }

    /// Top-K extraction across the flat `(queries x labels)` score space.
    fn decode_one(
        &self,
        logits: &[f32],
        boxes: &[f32],
        q: usize,
        num_labels: usize,
        img_w: f32,
        img_h: f32,
    ) -> Vec<Detection> {
        // scores = sigmoid(logits), flattened to q*num_labels.
        let flat_len = q * num_labels;
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(flat_len);
        for (i, &l) in logits.iter().take(flat_len).enumerate() {
            scored.push((i, sigmoid(l)));
        }
        // Top-K = top-`q` by score (upstream takes k = min(Q, flat.size) = Q).
        let k = q.min(flat_len);
        // Partial sort: select the k highest scores, then sort descending.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let mut detections = Vec::new();
        for (flat_idx, score) in scored {
            if score < self.threshold {
                continue;
            }
            let query = flat_idx / num_labels;
            let label = flat_idx % num_labels;

            // box = boxes[query] = (cx, cy, w, h) normalized.
            let bx = query * 4;
            let cx = boxes[bx] * img_w;
            let cy = boxes[bx + 1] * img_h;
            let bw = boxes[bx + 2] * img_w;
            let bh = boxes[bx + 3] * img_h;
            let x1 = (cx - bw / 2.0).clamp(0.0, img_w);
            let y1 = (cy - bh / 2.0).clamp(0.0, img_h);
            let x2 = (cx + bw / 2.0).clamp(0.0, img_w);
            let y2 = (cy + bh / 2.0).clamp(0.0, img_h);

            let class_name = match &self.labels {
                Some(names) if label < names.len() => names[label].clone(),
                _ => label.to_string(),
            };

            detections.push(Detection {
                bbox: [x1, y1, x2, y2],
                score,
                label,
                class_name,
            });
        }
        detections
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Materialize an MLX array and read it back as a row-major `Vec<f32>`.
///
/// The forward path produces f32 outputs, so a direct 4-byte LE parse is
/// correct. `contiguous` guarantees row-major before extraction.
fn to_f32_vec(arr: &MlxArray) -> Vec<f32> {
    let contiguous = mlxcel_core::contiguous(arr, false);
    let c = contiguous.as_ref().expect("contiguous returned null");
    mlxcel_core::eval(c);
    let bytes = mlxcel_core::array_to_raw_bytes(c);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
