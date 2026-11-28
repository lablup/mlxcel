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
use super::postprocess::{decode_detections, decode_mlx_vlm_detections};
use super::processor::{PpDocLayoutV3Processor, ProcessedImage};

pub const DEFAULT_THRESHOLD: f32 = 0.3;
/// Default confidence for [`PpDocLayoutDecode::MlxVlmDetect`].
pub const MLX_VLM_DEFAULT_CONF: f32 = 0.5;
/// Default square input side for [`PpDocLayoutDecode::MlxVlmDetect`].
pub const MLX_VLM_DEFAULT_IMG_SIZE: usize = 1024;

/// How to turn the detector's raw head into boxes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum PpDocLayoutDecode {
    /// Hugging Face post-processing: 800 px input, flat top-k over the
    /// query-by-class grid, and the page-wide GlobalPointer order. This is
    /// what `mlxcel detect` uses.
    #[default]
    HuggingFace,
    /// mlx-vlm `detect()` semantics: resize to `img_size` square, one argmax
    /// label per query, keep `score > conf`, and decode the order matrix over
    /// the retained sub-block only. IndicOCR's layout stage was trained on
    /// this path.
    MlxVlmDetect { conf: f32, img_size: usize },
}

impl PpDocLayoutDecode {
    fn validate(self) -> Result<(), String> {
        match self {
            Self::HuggingFace => Ok(()),
            Self::MlxVlmDetect { conf, img_size } => {
                if !conf.is_finite() || !(0.0..=1.0).contains(&conf) {
                    return Err(format!(
                        "PP-DocLayoutV3 mlx-vlm confidence must be finite and in [0, 1], got {conf}"
                    ));
                }
                if img_size == 0 || img_size > 4096 {
                    return Err(format!(
                        "PP-DocLayoutV3 mlx-vlm image size must be in 1..=4096, got {img_size}"
                    ));
                }
                Ok(())
            }
        }
    }
}

pub struct PpDocLayoutV3Predictor {
    model: PpDocLayoutV3Model,
    processor: PpDocLayoutV3Processor,
    threshold: f32,
    labels: Option<Vec<String>>,
    decode: PpDocLayoutDecode,
}

impl PpDocLayoutV3Predictor {
    pub fn from_pretrained<P: AsRef<Path>>(directory: P, threshold: f32) -> Result<Self, String> {
        Self::from_pretrained_with_decode(directory, threshold, PpDocLayoutDecode::default())
    }

    pub fn from_pretrained_with_decode<P: AsRef<Path>>(
        directory: P,
        threshold: f32,
        decode: PpDocLayoutDecode,
    ) -> Result<Self, String> {
        decode.validate()?;
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
            decode,
        })
    }

    pub fn decode(&self) -> PpDocLayoutDecode {
        self.decode
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

    /// Run the detector with mlx-vlm's `detect()` preprocessing and decoding
    /// (IndicOCR's layout stage). The image is stretched to `img_size` square,
    /// so boxes are scaled back to the original page.
    pub fn detect_mlx_vlm_image(
        &self,
        image: &DynamicImage,
        conf: f32,
        img_size: usize,
    ) -> Result<DetectionResult, String> {
        PpDocLayoutDecode::MlxVlmDetect { conf, img_size }.validate()?;
        let (width, height) = (image.width(), image.height());
        if width == 0 || height == 0 {
            return Err("PP-DocLayoutV3 mlx-vlm path received an empty image".into());
        }
        let side = img_size as u32;
        let rgb = image.to_rgb8();
        // Pillow's `Image.resize` defaults to bicubic, which is what mlx-vlm's
        // `detect()` uses.
        let resized =
            image::imageops::resize(&rgb, side, side, image::imageops::FilterType::CatmullRom);
        let mut data = vec![0.0_f32; (side * side * 3) as usize];
        for (index, pixel) in resized.pixels().enumerate() {
            let base = index * 3;
            data[base] = f32::from(pixel[0]) / 255.0;
            data[base + 1] = f32::from(pixel[1]) / 255.0;
            data[base + 2] = f32::from(pixel[2]) / 255.0;
        }
        let pixel_values = mlxcel_core::from_slice_f32(&data, &[1, side as i32, side as i32, 3]);
        let output = self.model.forward(&pixel_values)?;
        let queries = self.model.config().num_queries;
        let labels = self.model.config().num_labels;
        let logits_shape = output_shape("logits", &output.logits, 3)?;
        let boxes_shape = output_shape("pred_boxes", &output.pred_boxes, 3)?;
        let order_shape = output_shape("order_logits", &output.order_logits, 3)?;
        if logits_shape != [1, queries as i32, labels as i32]
            || boxes_shape != [1, queries as i32, 4]
            || order_shape != [1, queries as i32, queries as i32]
        {
            return Err(format!(
                "PP-DocLayoutV3 output shapes differ from config: logits={logits_shape:?}, \
                 pred_boxes={boxes_shape:?}, order_logits={order_shape:?}"
            ));
        }
        let detections = decode_mlx_vlm_detections(
            &read_output_f32(&output.logits),
            &read_output_f32(&output.pred_boxes),
            &read_output_f32(&output.order_logits),
            queries,
            labels,
            width as f32,
            height as f32,
            conf,
            self.labels.as_deref(),
        )?;
        Ok(DetectionResult { detections })
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
