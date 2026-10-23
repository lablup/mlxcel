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

//! SAM 3.1's three parallel FPN branches.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::sam3::config::VisionEncoderConfig;
use super::super::sam3::vision::FpnNeck;

pub(crate) struct TriFpnNeck {
    detection: FpnNeck,
    interactive: FpnNeck,
    propagation: FpnNeck,
}

impl TriFpnNeck {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &VisionEncoderConfig,
    ) -> Result<Self, String> {
        let root = "detector_model.vision_encoder.neck";
        Ok(Self {
            detection: FpnNeck::from_weight_root(weights, config, &format!("{root}.convs"))?,
            interactive: FpnNeck::from_weight_root(
                weights,
                config,
                &format!("{root}.interactive_convs"),
            )?,
            propagation: FpnNeck::from_weight_root(
                weights,
                config,
                &format!("{root}.propagation_convs"),
            )?,
        })
    }

    pub(crate) fn detection(&self, backbone: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        self.detection.forward(backbone)
    }

    pub(crate) fn interactive(&self, backbone: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        self.interactive.forward(backbone)
    }

    pub(crate) fn propagation(&self, backbone: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        self.propagation.forward(backbone)
    }
}
