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

//! LLaVA MLP Multi-Modal Projector
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llava/llava.py#L13-L28
//!
//! Architecture: Linear(vision_hidden → text_hidden) → GELU → Linear(text_hidden → text_hidden)
//!
//! Used by: LLaVA

use super::MultiModalConnector;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct MLPProjector {
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
}

impl MLPProjector {
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
        let linear_2 = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_2", prefix),
            group_size,
            bits,
        )?;
        Ok(Self { linear_1, linear_2 })
    }
}

impl MultiModalConnector for MLPProjector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.linear_1.forward(vision_features);
        let x = mlxcel_core::gelu(&x);
        self.linear_2.forward(&x)
    }
}
