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

//! Sparse 3D U-Net depth refiner for MoGe3.

use anyhow::{Result, ensure};
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{MAX_SPARSE_WORKING_BYTES, RefinerConfig};
use super::sparse::{
    MAX_SPARSE_NEIGHBOR_BYTES, SparseShape, SubmanifoldConv3d, neighbor_construction_peak_bytes,
    neighbor_map_bytes, sparse_pool2x_mean, sparse_upsample2x_nearest,
};
use super::weights::{self, Linear};

struct SparseResidualBlock {
    norm: LayerNorm,
    conv1: SubmanifoldConv3d,
    conv2: SubmanifoldConv3d,
    skip: Option<Linear>,
}
impl SparseResidualBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let norm_weight = weights::array(weights, &format!("{prefix}.norm1.weight"))?;
        let norm_bias = weights::array(weights, &format!("{prefix}.norm1.bias"))?;
        ensure!(
            mlxcel_core::array_shape(&norm_weight) == [channels as i32]
                && mlxcel_core::array_shape(&norm_bias) == [channels as i32],
            "MoGe3 refiner norm {prefix} must have [{channels}] weight and bias"
        );
        Ok(Self {
            norm: LayerNorm::new(norm_weight, Some(norm_bias), 1e-6),
            conv1: SubmanifoldConv3d::from_weights(
                weights,
                &format!("{prefix}.conv1"),
                channels,
                out_channels,
            )?,
            conv2: SubmanifoldConv3d::from_weights(
                weights,
                &format!("{prefix}.conv2"),
                out_channels,
                out_channels,
            )?,
            skip: (channels != out_channels)
                .then(|| {
                    Linear::from_weights(
                        weights,
                        &format!("{prefix}.skip_connection"),
                        channels,
                        out_channels,
                    )
                })
                .transpose()?,
        })
    }
    fn forward(&self, feats: &MlxArray, neighbor: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let normalized = self.norm.forward(feats);
        let activated = mlxcel_core::utils::silu(&normalized);
        mlxcel_core::eval(&activated);
        drop(normalized);
        let first = self.conv1.forward_owned(activated, neighbor)?;
        let activated = mlxcel_core::utils::silu(&first);
        mlxcel_core::eval(&activated);
        drop(first);
        let second = self.conv2.forward_owned(activated, neighbor)?;
        let output = match &self.skip {
            Some(v) => mlxcel_core::add(&second, &v.forward(feats)),
            None => mlxcel_core::add(&second, feats),
        };
        mlxcel_core::eval(&output);
        Ok(output)
    }
}
struct PoolDown {
    linear: Linear,
}
struct NearestUp {
    linear: Linear,
}
struct Skip {
    features: UniquePtr<MlxArray>,
    coords: UniquePtr<MlxArray>,
    shape: SparseShape,
}

pub struct Sparse3dUnet {
    encoder_downsample: usize,
    encoder_channels: usize,
    input_channels: usize,
    channels: Vec<usize>,
    input: Linear,
    encoder_fuse: Linear,
    fuse1: Linear,
    fuse2: Linear,
    down_stages: Vec<Vec<SparseResidualBlock>>,
    downsample: Vec<PoolDown>,
    bottleneck: Vec<SparseResidualBlock>,
    upsample: Vec<NearestUp>,
    up_stages: Vec<Vec<SparseResidualBlock>>,
    output: Linear,
}
impl Sparse3dUnet {
    pub fn from_weights(weights: &WeightMap, config: &RefinerConfig) -> Result<Self> {
        config.validate()?;
        let levels = config.model_channels.len();
        let enc_counts = config
            .encoder_blocks_per_level
            .resolve(levels, "encoder_blocks_per_level")?;
        let dec_counts = config
            .decoder_blocks_per_level
            .resolve(levels - 1, "decoder_blocks_per_level")?;
        let mut down_stages = Vec::with_capacity(levels);
        for (level, (&ch, &count)) in config.model_channels.iter().zip(&enc_counts).enumerate() {
            let mut stage = Vec::with_capacity(count);
            for block in 0..count {
                stage.push(SparseResidualBlock::from_weights(
                    weights,
                    &format!("refiner.down_stages.{level}.{block}"),
                    ch,
                    ch,
                )?)
            }
            down_stages.push(stage)
        }
        let mut downsample = Vec::new();
        for i in 0..levels - 1 {
            downsample.push(PoolDown {
                linear: Linear::from_weights(
                    weights,
                    &format!("refiner.downsample_blocks.{i}.linear"),
                    config.model_channels[i],
                    config.model_channels[i + 1],
                )?,
            })
        }
        let last = *config.model_channels.last().expect("levels validated");
        let mut bottleneck = Vec::new();
        for i in 0..config.bottleneck_blocks {
            bottleneck.push(SparseResidualBlock::from_weights(
                weights,
                &format!("refiner.bottleneck_stage.{i}"),
                last,
                last,
            )?)
        }
        let mut upsample = Vec::new();
        let mut up_stages = Vec::new();
        for (i, level) in (1..levels).rev().enumerate() {
            upsample.push(NearestUp {
                linear: Linear::from_weights(
                    weights,
                    &format!("refiner.upsample_blocks.{i}.linear"),
                    config.model_channels[level],
                    config.model_channels[level - 1],
                )?,
            });
            let mut stage = Vec::new();
            for j in 0..dec_counts[i] {
                let ch = config.model_channels[level - 1];
                stage.push(SparseResidualBlock::from_weights(
                    weights,
                    &format!("refiner.up_stages.{i}.{j}"),
                    ch,
                    ch,
                )?)
            }
            up_stages.push(stage)
        }
        Ok(Self {
            encoder_downsample: config.encoder_downsample,
            encoder_channels: config.encoder_channels,
            input_channels: config.in_channels,
            channels: config.model_channels.clone(),
            input: Linear::from_weights(
                weights,
                "refiner.input_proj",
                config.in_channels,
                config.model_channels[0],
            )?,
            encoder_fuse: Linear::from_weights(
                weights,
                "refiner.encoder_fuse",
                config.encoder_channels,
                last,
            )?,
            fuse1: Linear::from_weights(weights, "refiner.fuse_proj.0", last * 2, last)?,
            fuse2: Linear::from_weights(weights, "refiner.fuse_proj.2", last, last)?,
            down_stages,
            downsample,
            bottleneck,
            upsample,
            up_stages,
            output: Linear::from_weights(
                weights,
                "refiner.out_proj",
                config.model_channels[0],
                config.out_channels,
            )?,
        })
    }
}

#[path = "refiner_budget.rs"]
mod budget;
#[path = "refiner_forward.rs"]
mod forward;
#[path = "refiner_policy.rs"]
mod policy;
#[cfg(test)]
#[path = "refiner_tests.rs"]
mod tests;
