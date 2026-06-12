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

//! Linear Multi-Modal Projector (single linear layer, no activation, no bias)
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llama4/vision.py (Llama4MultiModalProjector)
//!
//! Architecture: Linear(vision_output_dim → text_hidden_size, bias=False)
//!
//! Used by: Llama 4 Vision

use super::MultiModalConnector;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct LinearProjector {
    linear_1: UnifiedLinear,
}

impl LinearProjector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let linear_1 = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_1", prefix),
            group_size,
            bits,
        )?;
        Ok(Self { linear_1 })
    }
}

impl MultiModalConnector for LinearProjector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        self.linear_1.forward(vision_features)
    }
}
