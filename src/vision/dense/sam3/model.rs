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

//! Top-level SAM 3 detector graph and shared image backbone.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::Sam3Config;
use super::decoder::DetrDecoder;
use super::encoder::DetrEncoder;
use super::geometry::GeometryEncoder;
use super::position::PositionEmbeddingSine;
use super::segmentation::{DotProductScoring, MaskDecoder};
use super::text_encoder::TextEncoder;
use super::tracker::TrackerModel;
use super::vision::{FpnNeck, VisionEncoder};

pub(crate) struct DetectorFeatures {
    pub fpn: Vec<UniquePtr<MlxArray>>,
    pub src: UniquePtr<MlxArray>,
    pub pos: UniquePtr<MlxArray>,
    pub spatial: (i32, i32),
}

pub(crate) struct DetectorOutput {
    pub logits: UniquePtr<MlxArray>,
    pub boxes: UniquePtr<MlxArray>,
    pub masks: Option<UniquePtr<MlxArray>>,
    pub presence: UniquePtr<MlxArray>,
}

pub(crate) struct TrackerBundle {
    pub neck: FpnNeck,
    pub model: TrackerModel,
}

pub(crate) struct Sam3Model {
    pub config: Sam3Config,
    vision: VisionEncoder,
    text: TextEncoder,
    detr_encoder: DetrEncoder,
    detr_decoder: DetrDecoder,
    geometry: GeometryEncoder,
    mask_decoder: MaskDecoder,
    scoring: DotProductScoring,
    position: PositionEmbeddingSine,
    pub tracker: Option<TrackerBundle>,
}

impl Sam3Model {
    pub fn from_weights(weights: WeightMap, config: Sam3Config) -> Result<Self, String> {
        let quantization = config.quantization_params();
        let detector = &config.detector_config;
        let detector_root = if weights
            .keys()
            .any(|name| name.starts_with("detector_model."))
        {
            "detector_model."
        } else {
            ""
        };
        let prefix = |component: &str| format!("{detector_root}{component}");
        let vision = VisionEncoder::from_weights(
            &weights,
            &detector.vision_config,
            &prefix("vision_encoder"),
            quantization.0,
            quantization.1,
        )?;
        let text = TextEncoder::from_weights(
            &weights,
            detector_root.trim_end_matches('.'),
            &detector.text_config,
            quantization,
        )?;
        let detr_encoder = DetrEncoder::from_weights(
            &weights,
            &prefix("detr_encoder"),
            &detector.detr_encoder_config,
            quantization,
        )?;
        let detr_decoder = DetrDecoder::from_weights(
            &weights,
            &detector.detr_decoder_config,
            &prefix("detr_decoder"),
            quantization.0,
            quantization.1,
        )?;
        let geometry = GeometryEncoder::from_weights(
            &weights,
            &prefix("geometry_encoder"),
            &detector.geometry_encoder_config,
            quantization,
        )?;
        let mask_decoder = MaskDecoder::from_weights(
            &weights,
            &detector.mask_decoder_config,
            &prefix("mask_decoder"),
            quantization.0,
            quantization.1,
        )?;
        let scoring = DotProductScoring::from_weights(
            &weights,
            detector.detr_encoder_config.hidden_size,
            &prefix("dot_product_scoring"),
            quantization.0,
            quantization.1,
        )?;
        let position = PositionEmbeddingSine::new(detector.detr_encoder_config.hidden_size / 2);
        let has_tracker = weights
            .keys()
            .any(|name| name.starts_with("tracker_model."));
        let tracker = if has_tracker {
            Some(TrackerBundle {
                neck: FpnNeck::from_weights(
                    &weights,
                    &config.tracker_config.vision_config,
                    "tracker_neck",
                )?,
                model: TrackerModel::from_weights(
                    &weights,
                    &config.tracker_config,
                    quantization.0,
                    quantization.1,
                )?,
            })
        } else {
            None
        };
        Ok(Self {
            config,
            vision,
            text,
            detr_encoder,
            detr_decoder,
            geometry,
            mask_decoder,
            scoring,
            position,
            tracker,
        })
    }

    #[must_use]
    pub fn backbone(&self, image: &MlxArray) -> UniquePtr<MlxArray> {
        self.vision.forward_backbone(image)
    }

    pub fn detector_features(&self, backbone: &MlxArray) -> Result<DetectorFeatures, String> {
        let fpn = self.vision.forward_neck(backbone);
        if fpn.len() < 3 {
            return Err(format!(
                "SAM 3 detector needs at least three FPN levels, got {}",
                fpn.len()
            ));
        }
        let encoder_feature = &fpn[fpn.len() - 2];
        let shape = mlxcel_core::array_shape(encoder_feature);
        if shape.len() != 4 {
            return Err(format!(
                "SAM 3 detector expected NHWC FPN feature, got {shape:?}"
            ));
        }
        let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
        let position = self.position.forward(encoder_feature);
        Ok(DetectorFeatures {
            src: mlxcel_core::reshape(encoder_feature, &[batch, height * width, channels]),
            pos: mlxcel_core::reshape(&position, &[batch, height * width, channels]),
            fpn,
            spatial: (height, width),
        })
    }

    #[must_use]
    pub fn prompt_features(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        self.text
            .project(&self.text.forward(input_ids, Some(attention_mask)))
    }

    #[cfg(test)]
    pub(crate) fn prompt_hidden(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        self.text.forward(input_ids, Some(attention_mask))
    }

    #[cfg(test)]
    pub(crate) fn prompt_projection_dtype(&self) -> Option<i32> {
        self.text.detector_projection_dtype()
    }

    #[cfg(test)]
    pub(crate) fn prompt_diagnostic_stages(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> Vec<(String, UniquePtr<MlxArray>)> {
        self.text.diagnostic_stages(input_ids, attention_mask)
    }

    pub fn detect(
        &self,
        features: &DetectorFeatures,
        text: &MlxArray,
        attention_mask: &MlxArray,
        boxes: Option<&MlxArray>,
        with_masks: bool,
    ) -> Result<DetectorOutput, String> {
        let mut prompt = mlxcel_core::copy(text);
        let mut prompt_mask = mlxcel_core::copy(attention_mask);
        if let Some(boxes) = boxes {
            let shape = mlxcel_core::array_shape(boxes);
            if shape.len() != 3 || shape[2] != 4 {
                return Err(format!(
                    "SAM 3 box prompts must have shape (B,N,4), got {shape:?}"
                ));
            }
            if shape[1] > 0 {
                let labels = mlxcel_core::ones(&[shape[0], shape[1]], mlxcel_core::dtype::INT32);
                let mask = mlxcel_core::ones(&[shape[0], shape[1]], mlxcel_core::dtype::INT32);
                let encoder_feature = &features.fpn[features.fpn.len() - 2];
                let position = self.position.forward(encoder_feature);
                let (geometry, geometry_mask) =
                    self.geometry
                        .forward(boxes, &labels, &mask, encoder_feature, &position);
                prompt = mlxcel_core::concatenate(&prompt, &geometry, 1);
                prompt_mask = mlxcel_core::concatenate(&prompt_mask, &geometry_mask, 1);
            }
        }
        let encoded =
            self.detr_encoder
                .forward(&features.src, &features.pos, &prompt, Some(&prompt_mask));
        let decoded = self.detr_decoder.forward(
            &encoded,
            &prompt,
            &features.pos,
            Some(&prompt_mask),
            Some(features.spatial),
        );
        let layers = mlxcel_core::array_shape(&decoded.hidden_states)[0];
        let hidden = take_last_layer(&decoded.hidden_states, layers);
        let boxes = take_last_layer(&decoded.boxes, layers);
        let presence = take_last_layer(&decoded.presence_logits, layers);
        let scores = self
            .scoring
            .forward(&decoded.hidden_states, &prompt, Some(&prompt_mask));
        let logits = mlxcel_core::squeeze_axis(&take_last_layer(&scores, layers), -1);
        let shape = mlxcel_core::array_shape(&boxes);
        let coord =
            |index| mlxcel_core::slice(&boxes, &[0, 0, index], &[shape[0], shape[1], index + 1]);
        let cx = coord(0);
        let cy = coord(1);
        let half_w = mlxcel_core::multiply_scalar(&coord(2), 0.5);
        let half_h = mlxcel_core::multiply_scalar(&coord(3), 0.5);
        let boxes = mlxcel_core::concatenate_many(
            &[
                &mlxcel_core::subtract(&cx, &half_w),
                &mlxcel_core::subtract(&cy, &half_h),
                &mlxcel_core::add(&cx, &half_w),
                &mlxcel_core::add(&cy, &half_h),
            ],
            -1,
        );
        let masks = if with_masks {
            let fpn: Vec<UniquePtr<MlxArray>> = features
                .fpn
                .iter()
                .take(features.fpn.len() - 1)
                .map(|value| mlxcel_core::copy(value))
                .collect();
            Some(
                self.mask_decoder
                    .forward(
                        &hidden,
                        &fpn,
                        Some(&encoded),
                        Some(&prompt),
                        Some(&prompt_mask),
                    )?
                    .pred_masks,
            )
        } else {
            None
        };
        Ok(DetectorOutput {
            logits,
            boxes,
            masks,
            presence,
        })
    }
}

fn take_last_layer(value: &MlxArray, layers: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(value);
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    starts[0] = layers - 1;
    stops[0] = layers;
    mlxcel_core::squeeze_axis(&mlxcel_core::slice(value, &starts, &stops), 0)
}
