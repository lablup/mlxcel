// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::config::VisionEncoderConfig;
use super::{Conv2d, ConvTranspose2d};

#[rustfmt::skip]
enum ScaleLayers { None, One(ConvTranspose2d), Two(ConvTranspose2d, ConvTranspose2d) }

#[rustfmt::skip]
struct FpnLayer { scale: ScaleLayers, downsample: bool, proj1: Conv2d, proj2: Conv2d }

impl FpnLayer {
    fn load(weights: &WeightMap, prefix: &str, factor: f32, stride: i32) -> Result<Self, String> {
        let scale = if factor >= 4.0 {
            ScaleLayers::Two(
                ConvTranspose2d::load(weights, &format!("{prefix}.scale_layers.0"), stride)?,
                ConvTranspose2d::load(weights, &format!("{prefix}.scale_layers.2"), stride)?,
            )
        } else if factor >= 2.0 {
            ScaleLayers::One(ConvTranspose2d::load(
                weights,
                &format!("{prefix}.scale_layers.0"),
                stride,
            )?)
        } else {
            ScaleLayers::None
        };
        Ok(Self {
            scale,
            downsample: factor <= 0.5,
            proj1: Conv2d::load(weights, &format!("{prefix}.proj1"), 1, 0)?,
            proj2: Conv2d::load(weights, &format!("{prefix}.proj2"), 1, 1)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut y = match &self.scale {
            ScaleLayers::None => mlxcel_core::copy(x),
            ScaleLayers::One(a) => a.forward(x),
            ScaleLayers::Two(a, b) => b.forward(&super::super::activation::gelu(&a.forward(x))),
        };
        if self.downsample {
            let shape = mlxcel_core::array_shape(&y);
            y = mlxcel_core::reshape(&y, &[shape[0], shape[1] / 2, 2, shape[2] / 2, 2, shape[3]]);
            y = mlxcel_core::max_axis(&y, 4, false);
            y = mlxcel_core::max_axis(&y, 2, false);
        }
        self.proj2.forward(&self.proj1.forward(&y))
    }
}

#[rustfmt::skip]
pub struct FpnNeck { layers: Vec<FpnLayer> }

impl FpnNeck {
    pub fn from_weights(
        weights: &WeightMap,
        config: &VisionEncoderConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Self::from_weight_root(weights, config, &format!("{prefix}.fpn_layers"))
    }

    /// Load an FPN whose numbered layers live directly below `root`.
    ///
    /// SAM 3 stores layers below `neck.fpn_layers`, while SAM 3.1 has three
    /// parallel neck roots. The shared layer implementation keeps resize and
    /// projection math identical between the two model families.
    pub fn from_weight_root(
        weights: &WeightMap,
        config: &VisionEncoderConfig,
        root: &str,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.scale_factors.len());
        for (index, factor) in config.scale_factors.iter().copied().enumerate() {
            layers.push(FpnLayer::load(
                weights,
                &format!("{root}.{index}"),
                factor,
                config.fpn_stride as i32,
            )?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, input: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        self.layers
            .iter()
            .map(|layer| layer.forward(input))
            .collect()
    }
}
