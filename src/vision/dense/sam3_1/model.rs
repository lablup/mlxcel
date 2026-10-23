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

//! Shared-detector SAM 3.1 model wrapper.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::sam3::model::{DetectorCore, DetectorFeatures, DetectorModel, DetectorOutput};
use super::super::sam3::vision::ViTBackbone;
use super::config::Sam31Config;
use super::neck::TriFpnNeck;
use super::state::FrameFeatures;
use super::tracker::MultiplexTracker;

pub(crate) struct Sam31Model {
    pub config: Sam31Config,
    backbone: ViTBackbone,
    neck: TriFpnNeck,
    detector: DetectorCore,
    pub tracker: MultiplexTracker,
}

impl Sam31Model {
    pub(crate) fn from_weights(weights: WeightMap, config: Sam31Config) -> Result<Self, String> {
        let quantization = config.quantization_params();
        let detector = &config.detector_config;
        Ok(Self {
            backbone: ViTBackbone::from_weights(
                &weights,
                &detector.vision_config.backbone_config,
                "detector_model.vision_encoder.backbone",
                quantization.0,
                quantization.1,
            )?,
            neck: TriFpnNeck::from_weights(&weights, &detector.vision_config)?,
            detector: DetectorCore::from_weights(
                &weights,
                detector,
                "detector_model.",
                quantization,
            )?,
            tracker: MultiplexTracker::from_weights(
                &weights,
                config.tracker_config.clone(),
                quantization.0,
                quantization.1,
            )?,
            config,
        })
    }

    #[must_use]
    pub(crate) fn backbone(&self, image: &MlxArray) -> UniquePtr<MlxArray> {
        self.backbone.forward(image)
    }

    pub(crate) fn detector_features(
        &self,
        backbone: &MlxArray,
    ) -> Result<DetectorFeatures, String> {
        self.detector
            .features_from_fpn(self.neck.detection(backbone))
    }

    pub(crate) fn tracker_features(
        &self,
        backbone: &MlxArray,
        need_interactive: bool,
    ) -> Result<FrameFeatures, String> {
        let interactive = need_interactive.then(|| self.neck.interactive(backbone));
        self.tracker
            .prepare_features(interactive, self.neck.propagation(backbone))
    }

    #[must_use]
    pub(crate) fn prompt_features(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        self.detector.prompt_features(input_ids, attention_mask)
    }

    pub(crate) fn detect(
        &self,
        features: &DetectorFeatures,
        text: &MlxArray,
        attention_mask: &MlxArray,
        boxes: Option<&MlxArray>,
        with_masks: bool,
    ) -> Result<DetectorOutput, String> {
        self.detector
            .detect(features, text, attention_mask, boxes, with_masks)
    }
}

impl DetectorModel for Sam31Model {
    fn backbone(&self, image: &MlxArray) -> UniquePtr<MlxArray> {
        Self::backbone(self, image)
    }

    fn detector_features(&self, backbone: &MlxArray) -> Result<DetectorFeatures, String> {
        Self::detector_features(self, backbone)
    }

    fn prompt_features(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        Self::prompt_features(self, input_ids, attention_mask)
    }

    fn detect(
        &self,
        features: &DetectorFeatures,
        text: &MlxArray,
        attention_mask: &MlxArray,
        boxes: Option<&MlxArray>,
        with_masks: bool,
    ) -> Result<DetectorOutput, String> {
        Self::detect(self, features, text, attention_mask, boxes, with_masks)
    }
}
