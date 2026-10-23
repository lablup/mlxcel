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

//! PP-DocLayoutV3 predictor and checked device-to-host output seam.

use std::path::Path;

use image::DynamicImage;
use mlxcel_core::MlxArray;

use crate::vision::detection::{DetectionResult, Detector};

use super::model::{PpDocLayoutV3Model, PpDocLayoutV3Output};
use super::postprocess::decode_detections;
use super::processor::{PpDocLayoutV3Processor, ProcessedImage};

pub const DEFAULT_THRESHOLD: f32 = 0.3;

pub struct PpDocLayoutV3Predictor {
    model: PpDocLayoutV3Model,
    processor: PpDocLayoutV3Processor,
    threshold: f32,
    labels: Option<Vec<String>>,
}

impl PpDocLayoutV3Predictor {
    pub fn from_pretrained<P: AsRef<Path>>(directory: P, threshold: f32) -> Result<Self, String> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(format!(
                "PP-DocLayoutV3 confidence threshold must be finite and in [0, 1], got {threshold}"
            ));
        }
        let directory = directory.as_ref();
        let model = PpDocLayoutV3Model::load(directory)?;
        let processor = PpDocLayoutV3Processor::from_pretrained(directory)?;
        let expected = model.config().image_size;
        let actual = processor.config();
        if actual.height != expected || actual.width != expected {
            return Err(format!(
                "PP-DocLayoutV3 processor is {}x{} but model image_size is {expected}",
                actual.height, actual.width
            ));
        }
        let labels = model.config().class_names();
        Ok(Self {
            model,
            processor,
            threshold,
            labels,
        })
    }

    pub fn model(&self) -> &PpDocLayoutV3Model {
        &self.model
    }

    pub fn processor(&self) -> &PpDocLayoutV3Processor {
        &self.processor
    }

    pub fn predict_path<P: AsRef<Path>>(&self, path: P) -> Result<DetectionResult, String> {
        let processed = self.processor.process_path(path)?;
        self.predict_processed(processed)
    }

    pub fn predict_image(&self, image: &DynamicImage) -> Result<DetectionResult, String> {
        let processed = self.processor.process_image(image)?;
        self.predict_processed(processed)
    }

    fn predict_processed(&self, processed: ProcessedImage) -> Result<DetectionResult, String> {
        let output = self.model.forward(&processed.pixel_values)?;
        self.decode_output(&output, processed.original_size)
    }

    fn decode_output(
        &self,
        output: &PpDocLayoutV3Output,
        original_size: (u32, u32),
    ) -> Result<DetectionResult, String> {
        let logits_shape = output_shape("logits", &output.logits, 3)?;
        let boxes_shape = output_shape("pred_boxes", &output.pred_boxes, 3)?;
        let order_shape = output_shape("order_logits", &output.order_logits, 3)?;
        let queries = self.model.config().num_queries;
        let labels = self.model.config().num_labels;
        if logits_shape != [1, queries as i32, labels as i32]
            || boxes_shape != [1, queries as i32, 4]
            || order_shape != [1, queries as i32, queries as i32]
        {
            return Err(format!(
                "PP-DocLayoutV3 output shapes differ from config: logits={logits_shape:?}, \
                 pred_boxes={boxes_shape:?}, order_logits={order_shape:?}; expected [1,{queries},{labels}], \
                 [1,{queries},4], [1,{queries},{queries}]"
            ));
        }
        let logits = read_output_f32(&output.logits);
        let boxes = read_output_f32(&output.pred_boxes);
        let order = read_output_f32(&output.order_logits);
        let expected_logits = queries * labels;
        let expected_boxes = queries * 4;
        let expected_order = queries * queries;
        if logits.len() != expected_logits
            || boxes.len() != expected_boxes
            || order.len() != expected_order
        {
            return Err(format!(
                "PP-DocLayoutV3 output readback mismatch: logits={}, boxes={}, order_logits={}; \
                 expected {expected_logits}, {expected_boxes}, {expected_order}",
                logits.len(),
                boxes.len(),
                order.len()
            ));
        }
        let detections = decode_detections(
            &logits,
            &boxes,
            &order,
            queries,
            labels,
            original_size.0 as f32,
            original_size.1 as f32,
            self.threshold,
            self.labels.as_deref(),
        )?;
        Ok(DetectionResult { detections })
    }
}

impl Detector for PpDocLayoutV3Predictor {
    fn predict_path(&self, path: &Path) -> Result<DetectionResult, String> {
        PpDocLayoutV3Predictor::predict_path(self, path)
    }

    fn name(&self) -> &'static str {
        "pp_doclayout_v3"
    }
}

fn output_shape(name: &str, output: &MlxArray, rank: usize) -> Result<Vec<i32>, String> {
    let shape = mlxcel_core::array_shape(output);
    if shape.len() != rank || shape.iter().any(|&dimension| dimension <= 0) {
        return Err(format!(
            "PP-DocLayoutV3 {name} must be a nonempty rank-{rank} tensor, got {shape:?}"
        ));
    }
    Ok(shape)
}

pub(crate) fn read_output_f32(output: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(output, mlxcel_core::dtype::FLOAT32))
}
