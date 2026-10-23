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

//! Convolution-stack and sparse-refiner sub-configuration.

use anyhow::{Result, ensure};
use serde::Deserialize;

pub(super) const MAX_CHANNELS: usize = 4096;
const MAX_LEVELS: usize = 8;
const MAX_RES_BLOCKS: usize = 16;

fn default_in_channels() -> usize {
    3
}
fn default_out_channels() -> usize {
    1
}
fn default_encoder_channels() -> usize {
    1026
}
fn default_model_channels() -> Vec<usize> {
    vec![32, 64, 128, 256, 512]
}
fn default_one_or_many() -> OneOrManyUsize {
    OneOrManyUsize::One(1)
}
fn default_bottleneck_blocks() -> usize {
    1
}
fn default_downsample_factors() -> Vec<usize> {
    vec![2, 2, 2, 2]
}
fn default_encoder_downsample() -> usize {
    16
}
fn default_activation() -> String {
    "relu".into()
}
fn default_norm() -> String {
    "none".into()
}
fn default_hidden_multiplier() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OneOrManyUsize {
    One(usize),
    Many(Vec<usize>),
}
impl OneOrManyUsize {
    pub fn resolve(&self, count: usize, name: &str) -> Result<Vec<usize>> {
        let values = match self {
            Self::One(v) => vec![*v; count],
            Self::Many(v) => v.clone(),
        };
        ensure!(
            values.len() == count,
            "MoGe3 {name} must have {count} entries"
        );
        Ok(values)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OneOrManyString {
    One(String),
    Many(Vec<String>),
}
impl OneOrManyString {
    pub fn resolve(&self, count: usize, name: &str) -> Result<Vec<String>> {
        let values = match self {
            Self::One(v) => vec![v.clone(); count],
            Self::Many(v) => v.clone(),
        };
        ensure!(
            values.len() == count,
            "MoGe3 {name} must have {count} entries"
        );
        Ok(values)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ConvStackConfig {
    pub dim_in: Vec<Option<usize>>,
    pub dim_res_blocks: Vec<usize>,
    pub dim_out: Option<Vec<Option<usize>>>,
    pub resamplers: OneOrManyString,
    #[serde(default = "default_hidden_multiplier")]
    pub dim_times_res_block_hidden: usize,
    pub num_res_blocks: OneOrManyUsize,
    #[serde(default = "default_norm")]
    pub res_block_in_norm: String,
    #[serde(default = "default_norm")]
    pub res_block_hidden_norm: String,
    #[serde(default = "default_activation")]
    pub activation: String,
}
impl Default for ConvStackConfig {
    fn default() -> Self {
        default_neck(1024)
    }
}
impl ConvStackConfig {
    pub fn validate(&self, name: &str) -> Result<()> {
        let levels = self.dim_res_blocks.len();
        ensure!(
            (2..=MAX_LEVELS).contains(&levels),
            "MoGe3 {name} must have 2..={MAX_LEVELS} levels"
        );
        ensure!(
            self.dim_in.len() == levels,
            "MoGe3 {name}.dim_in length mismatch"
        );
        if let Some(v) = &self.dim_out {
            ensure!(v.len() == levels, "MoGe3 {name}.dim_out length mismatch");
        }
        ensure!(
            self.dim_res_blocks
                .iter()
                .all(|&v| v > 0 && v <= MAX_CHANNELS),
            "MoGe3 {name} channels out of range"
        );
        ensure!(
            self.dim_in
                .iter()
                .flatten()
                .all(|&v| v > 0 && v <= MAX_CHANNELS),
            "MoGe3 {name} input channels out of range"
        );
        ensure!(
            self.dim_out
                .iter()
                .flatten()
                .flatten()
                .all(|&v| v > 0 && v <= MAX_CHANNELS),
            "MoGe3 {name} output channels out of range"
        );
        ensure!(
            self.dim_times_res_block_hidden > 0 && self.dim_times_res_block_hidden <= 8,
            "MoGe3 {name} hidden multiplier out of range"
        );
        for &channels in &self.dim_res_blocks {
            let hidden = channels
                .checked_mul(self.dim_times_res_block_hidden)
                .ok_or_else(|| anyhow::anyhow!("MoGe3 {name} hidden channel overflow"))?;
            ensure!(
                hidden <= MAX_CHANNELS,
                "MoGe3 {name} hidden channels out of range"
            );
            if self.res_block_in_norm == "group_norm" {
                let groups = (channels / 32).max(1);
                ensure!(
                    channels % groups == 0,
                    "MoGe3 {name} input channels are not divisible by group count"
                );
            }
            if self.res_block_hidden_norm == "group_norm" {
                let groups = (hidden / 32).max(1);
                ensure!(
                    hidden % groups == 0,
                    "MoGe3 {name} hidden channels are not divisible by group count"
                );
            }
        }
        ensure!(
            self.num_res_blocks
                .resolve(levels, "num_res_blocks")?
                .iter()
                .all(|&v| v <= MAX_RES_BLOCKS),
            "MoGe3 {name} has too many residual blocks"
        );
        let resamplers = self.resamplers.resolve(levels - 1, "resamplers")?;
        ensure!(
            resamplers
                .iter()
                .all(|v| matches!(v.as_str(), "conv_transpose" | "bilinear" | "nearest")),
            "MoGe3 {name} has unsupported resampler"
        );
        ensure!(
            matches!(
                self.activation.as_str(),
                "relu" | "leaky_relu" | "silu" | "elu"
            ),
            "MoGe3 {name} has unsupported activation"
        );
        for norm in [&self.res_block_in_norm, &self.res_block_hidden_norm] {
            ensure!(
                matches!(
                    norm.as_str(),
                    "none" | "layer_norm" | "group_norm" | "instance_norm"
                ),
                "MoGe3 {name} has unsupported normalization"
            );
        }
        Ok(())
    }
    pub fn output_dims(&self) -> Vec<Option<usize>> {
        self.dim_out
            .clone()
            .unwrap_or_else(|| vec![None; self.dim_res_blocks.len()])
    }
}

fn default_dims() -> Vec<usize> {
    vec![1024, 1024, 1024, 1]
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ScaleHeadConfig {
    #[serde(default = "default_dims")]
    pub dims: Vec<usize>,
}
impl Default for ScaleHeadConfig {
    fn default() -> Self {
        Self {
            dims: default_dims(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RefinerConfig {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_encoder_channels")]
    pub encoder_channels: usize,
    #[serde(default = "default_model_channels")]
    pub model_channels: Vec<usize>,
    #[serde(default = "default_one_or_many")]
    pub encoder_blocks_per_level: OneOrManyUsize,
    #[serde(default = "default_one_or_many")]
    pub decoder_blocks_per_level: OneOrManyUsize,
    #[serde(default = "default_bottleneck_blocks")]
    pub bottleneck_blocks: usize,
    #[serde(default = "default_downsample_factors")]
    pub downsample_factors: Vec<usize>,
    #[serde(default = "default_encoder_downsample")]
    pub encoder_downsample: usize,
}
impl Default for RefinerConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 1,
            encoder_channels: 1026,
            model_channels: default_model_channels(),
            encoder_blocks_per_level: default_one_or_many(),
            decoder_blocks_per_level: default_one_or_many(),
            bottleneck_blocks: 1,
            downsample_factors: default_downsample_factors(),
            encoder_downsample: 16,
        }
    }
}
impl RefinerConfig {
    pub fn validate(&self) -> Result<()> {
        let levels = self.model_channels.len();
        ensure!(
            (2..=MAX_LEVELS).contains(&levels),
            "MoGe3 refiner must have 2..={MAX_LEVELS} levels"
        );
        ensure!(
            self.model_channels
                .iter()
                .all(|&v| v > 0 && v <= MAX_CHANNELS),
            "MoGe3 refiner channels out of range"
        );
        ensure!(
            self.in_channels > 0 && self.out_channels > 0 && self.encoder_channels > 0,
            "MoGe3 refiner channels must be positive"
        );
        ensure!(
            self.downsample_factors.len() == levels - 1
                && self.downsample_factors.iter().all(|&v| v == 2),
            "MoGe3 refiner supports only 2x downsampling"
        );
        let expected = 1usize
            .checked_shl((levels - 1) as u32)
            .ok_or_else(|| anyhow::anyhow!("MoGe3 refiner downsample overflow"))?;
        ensure!(
            self.encoder_downsample == expected,
            "MoGe3 refiner encoder_downsample must equal product of factors"
        );
        ensure!(
            self.encoder_blocks_per_level
                .resolve(levels, "encoder_blocks_per_level")?
                .iter()
                .all(|&v| v <= MAX_RES_BLOCKS),
            "too many refiner encoder blocks"
        );
        ensure!(
            self.decoder_blocks_per_level
                .resolve(levels - 1, "decoder_blocks_per_level")?
                .iter()
                .all(|&v| v <= MAX_RES_BLOCKS),
            "too many refiner decoder blocks"
        );
        ensure!(
            self.bottleneck_blocks <= MAX_RES_BLOCKS,
            "too many refiner bottleneck blocks"
        );
        Ok(())
    }
}

pub(super) fn default_neck(dim: usize) -> ConvStackConfig {
    let dims = if dim == 1536 {
        vec![1024, 512, 256, 128, 64]
    } else {
        vec![1024, 256, 128, 64, 32]
    };
    ConvStackConfig {
        dim_in: vec![Some(dim + 2), Some(2), Some(2), Some(2), Some(2)],
        dim_res_blocks: dims,
        dim_out: None,
        resamplers: OneOrManyString::Many(vec![
            "conv_transpose".into(),
            "conv_transpose".into(),
            "conv_transpose".into(),
            "bilinear".into(),
        ]),
        dim_times_res_block_hidden: 1,
        num_res_blocks: OneOrManyUsize::Many(vec![0, 2, 2, 2, 0]),
        res_block_in_norm: "none".into(),
        res_block_hidden_norm: "none".into(),
        activation: "relu".into(),
    }
}
pub(super) fn default_head(last: usize) -> ConvStackConfig {
    let dims = vec![1024, 256, 128, 64, 32];
    ConvStackConfig {
        dim_in: dims.iter().copied().map(Some).collect(),
        dim_res_blocks: dims,
        dim_out: Some(vec![None, None, None, None, Some(last)]),
        resamplers: OneOrManyString::Many(vec![
            "conv_transpose".into(),
            "conv_transpose".into(),
            "conv_transpose".into(),
            "bilinear".into(),
        ]),
        dim_times_res_block_hidden: 1,
        num_res_blocks: OneOrManyUsize::Many(vec![0, 1, 1, 1, 0]),
        res_block_in_norm: "none".into(),
        res_block_hidden_norm: "none".into(),
        activation: "relu".into(),
    }
}
