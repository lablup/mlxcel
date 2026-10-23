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

//! DINOv2 encoder reuse plus MoGe3 output projections.

use anyhow::{Result, ensure};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::EncoderConfig;
use super::weights::Conv2d;
use crate::vision::encoders::dinov2::{Dinov2, Dinov2Encoder};

pub struct Moge3Encoder {
    encoder: Dinov2Encoder,
    projections: Vec<Conv2d>,
}
impl Moge3Encoder {
    pub(super) fn with_objects_policy(mut self) -> Self {
        self.encoder.backbone = self.encoder.backbone.with_objects_policy();
        self
    }

    pub fn from_weights(weights: &WeightMap, config: &EncoderConfig) -> Result<Self> {
        let dino_config = config.dinov2()?;
        let backbone = Dinov2::from_hf_weights(weights, dino_config, "encoder.backbone")
            .map_err(anyhow::Error::msg)?;
        let encoder = Dinov2Encoder::new(backbone);
        let mut projections = Vec::with_capacity(config.intermediate_layers.len());
        for i in 0..config.intermediate_layers.len() {
            projections.push(Conv2d::from_weights(
                weights,
                &format!("encoder.output_projections.{i}"),
                1,
                config.dim_out,
                config.dim_out,
            )?)
        }
        Ok(Self {
            encoder,
            projections,
        })
    }
    pub fn forward(
        &self,
        image: &MlxArray,
        rows: usize,
        cols: usize,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        #[cfg(test)]
        super::diagnostics::encoder_input(
            image,
            rows,
            cols,
            self.encoder.config.patch_size,
            if self.encoder.backbone.uses_objects_policy() {
                super::policy::Policy::Objects
            } else {
                super::policy::Policy::Standard
            },
        )?;
        let features = self.encoder.call(image, rows, cols)?;
        #[cfg(test)]
        for (layer, (grid, cls)) in self
            .encoder
            .config
            .intermediate_layers
            .iter()
            .zip(&features)
        {
            super::diagnostics::emit(&format!("moge.encoder.layer{layer}.grid"), grid)?;
            super::diagnostics::emit(&format!("moge.encoder.layer{layer}.cls"), cls)?;
        }
        ensure!(
            features.len() == self.projections.len(),
            "MoGe3 encoder feature count mismatch"
        );
        let mut projected: Option<UniquePtr<MlxArray>> = None;
        for (projection, (grid, _)) in self.projections.iter().zip(&features) {
            let value = projection.forward(grid, 0);
            projected = Some(match projected {
                Some(sum) => mlxcel_core::add(&sum, &value),
                None => value,
            })
        }
        let cls = mlxcel_core::copy(&features.last().expect("four features").1);
        let projected = projected.expect("four projections");
        #[cfg(test)]
        super::diagnostics::emit("moge.encoder.projected", &projected)?;
        Ok((projected, cls))
    }
}
