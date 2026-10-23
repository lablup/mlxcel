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

//! MoGe3 convolutional neck and decoder heads.

use anyhow::{Result, ensure};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{ConvStackConfig, ScaleHeadConfig};
use super::weights::{self, Conv2d, Linear};
use crate::vision::dense::interpolate::{resize_bilinear_nhwc, resize_nearest_nhwc};

#[derive(Clone, Copy)]
enum Activation {
    Relu,
    LeakyRelu,
    Silu,
    Elu,
}
impl Activation {
    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "relu" => Self::Relu,
            "leaky_relu" => Self::LeakyRelu,
            "silu" => Self::Silu,
            "elu" => Self::Elu,
            other => anyhow::bail!("unsupported MoGe3 activation {other}"),
        })
    }
    fn forward(self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Relu => mlxcel_core::relu(x),
            Self::LeakyRelu => mlxcel_core::leaky_relu(x, 0.2),
            Self::Silu => mlxcel_core::utils::silu(x),
            Self::Elu => {
                let positive = mlxcel_core::greater(x, &mlxcel_core::from_slice_f32(&[0.0], &[1]));
                let negative = mlxcel_core::expm1(x);
                mlxcel_core::where_cond(&positive, x, &negative)
            }
        }
    }
}

enum Norm {
    None,
    Group {
        weight: Option<UniquePtr<MlxArray>>,
        bias: Option<UniquePtr<MlxArray>>,
        groups: i32,
    },
}
impl Norm {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        kind: &str,
        channels: usize,
    ) -> Result<Self> {
        if kind == "none" {
            return Ok(Self::None);
        }
        let groups = match kind {
            "group_norm" => (channels / 32).max(1),
            "layer_norm" => 1,
            "instance_norm" => channels,
            other => anyhow::bail!("unsupported MoGe3 norm {other}"),
        };
        let (weight, bias) = if kind == "instance_norm" {
            (None, None)
        } else {
            let weight = weights::array(weights, &format!("{prefix}.weight"))?;
            let bias = weights::array(weights, &format!("{prefix}.bias"))?;
            ensure!(
                mlxcel_core::array_shape(&weight) == [channels as i32]
                    && mlxcel_core::array_shape(&bias) == [channels as i32],
                "MoGe3 normalization {prefix} must have [{channels}] weight and bias"
            );
            (Some(weight), Some(bias))
        };
        Ok(Self::Group {
            weight,
            bias,
            groups: groups as i32,
        })
    }
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::None => mlxcel_core::copy(x),
            Self::Group {
                weight,
                bias,
                groups,
            } => {
                let dtype = mlxcel_core::array_dtype(x);
                let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
                let s = mlxcel_core::array_shape(&x);
                let b = s[0];
                let c = *s.last().expect("rank");
                let spatial: i32 = s[1..s.len() - 1].iter().product();
                let width = c / groups;
                let x = mlxcel_core::reshape(&x, &[b, spatial, *groups, width]);
                let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
                let x = mlxcel_core::reshape(&x, &[b, *groups, spatial * width]);
                let x = unsafe {
                    mlxcel_core::fast_layer_norm(&x, std::ptr::null(), std::ptr::null(), 1e-5)
                };
                let x = mlxcel_core::reshape(&x, &[b, *groups, spatial, width]);
                let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
                let x = mlxcel_core::reshape(&x, &s);
                match (weight, bias) {
                    (Some(weight), Some(bias)) => mlxcel_core::astype(
                        &mlxcel_core::add(&mlxcel_core::multiply(&x, weight), bias),
                        dtype,
                    ),
                    (None, None) => mlxcel_core::astype(&x, dtype),
                    _ => unreachable!("normalization affine parameters are paired"),
                }
            }
        }
    }
}

fn replicate_pad(x: &MlxArray, padding: usize) -> UniquePtr<MlxArray> {
    if padding == 0 {
        return mlxcel_core::copy(x);
    }
    let s = mlxcel_core::array_shape(x);
    let (h, w, c) = (s[1] as usize, s[2] as usize, s[3]);
    let mut indices = Vec::with_capacity((h + 2 * padding) * (w + 2 * padding));
    for i in 0..h + 2 * padding {
        let ii = i.saturating_sub(padding).min(h - 1);
        for j in 0..w + 2 * padding {
            let jj = j.saturating_sub(padding).min(w - 1);
            indices.push((ii * w + jj) as i32)
        }
    }
    let indices = mlxcel_core::from_slice_i32(&indices, &[indices.len() as i32]);
    let flat = mlxcel_core::reshape(x, &[s[0], (h * w) as i32, c]);
    let padded = mlxcel_core::take(&flat, &indices, 1);
    mlxcel_core::reshape(
        &padded,
        &[s[0], (h + 2 * padding) as i32, (w + 2 * padding) as i32, c],
    )
}

struct ReplicateConv {
    conv: Conv2d,
}
impl ReplicateConv {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        kernel: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::from_weights(weights, prefix, kernel, in_channels, out_channels)?,
        })
    }
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.conv
            .forward(&replicate_pad(x, self.conv.kernel / 2), 0)
    }
}

struct ResidualBlock {
    in_norm: Norm,
    activation: Activation,
    conv1: ReplicateConv,
    hidden_norm: Norm,
    conv2: ReplicateConv,
    skip: Option<Conv2d>,
}
impl ResidualBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        hidden: usize,
        activation: Activation,
        in_norm: &str,
        hidden_norm: &str,
    ) -> Result<Self> {
        Ok(Self {
            in_norm: Norm::from_weights(weights, &format!("{prefix}.layers.0"), in_norm, in_ch)?,
            activation,
            conv1: ReplicateConv::from_weights(
                weights,
                &format!("{prefix}.layers.2"),
                3,
                in_ch,
                hidden,
            )?,
            hidden_norm: Norm::from_weights(
                weights,
                &format!("{prefix}.layers.3"),
                hidden_norm,
                hidden,
            )?,
            conv2: ReplicateConv::from_weights(
                weights,
                &format!("{prefix}.layers.5"),
                3,
                hidden,
                out_ch,
            )?,
            skip: (in_ch != out_ch)
                .then(|| {
                    Conv2d::from_weights(
                        weights,
                        &format!("{prefix}.skip_connection"),
                        1,
                        in_ch,
                        out_ch,
                    )
                })
                .transpose()?,
        })
    }
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.in_norm.forward(x);
        let h = self.activation.forward(&h);
        let h = self.conv1.forward(&h);
        let h = self.hidden_norm.forward(&h);
        let h = self.activation.forward(&h);
        let h = self.conv2.forward(&h);
        let skip = match &self.skip {
            Some(v) => v.forward(x, 0),
            None => mlxcel_core::copy(x),
        };
        mlxcel_core::add(&h, &skip)
    }
}

enum Resampler {
    Deconv {
        weight: UniquePtr<MlxArray>,
        bias: UniquePtr<MlxArray>,
        conv: ReplicateConv,
    },
    Bilinear(ReplicateConv),
    Nearest(ReplicateConv),
}
impl Resampler {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        kind: &str,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        Ok(match kind {
            "conv_transpose" => {
                let weight = weights::array(weights, &format!("{prefix}.0.weight"))?;
                let bias = weights::array(weights, &format!("{prefix}.0.bias"))?;
                ensure!(
                    mlxcel_core::array_shape(&weight)
                        == [out_channels as i32, 2, 2, in_channels as i32],
                    "MoGe3 deconvolution {prefix} weight shape mismatch"
                );
                ensure!(
                    mlxcel_core::array_shape(&bias) == [out_channels as i32],
                    "MoGe3 deconvolution {prefix} bias shape mismatch"
                );
                Self::Deconv {
                    weight,
                    bias,
                    conv: ReplicateConv::from_weights(
                        weights,
                        &format!("{prefix}.1"),
                        3,
                        out_channels,
                        out_channels,
                    )?,
                }
            }
            "bilinear" => Self::Bilinear(ReplicateConv::from_weights(
                weights,
                &format!("{prefix}.1"),
                3,
                in_channels,
                out_channels,
            )?),
            "nearest" => Self::Nearest(ReplicateConv::from_weights(
                weights,
                &format!("{prefix}.1"),
                3,
                in_channels,
                out_channels,
            )?),
            _ => anyhow::bail!("unsupported MoGe3 resampler {kind}"),
        })
    }
    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let s = mlxcel_core::array_shape(x);
        Ok(match self {
            Self::Deconv { weight, bias, conv } => {
                let x = mlxcel_core::add(
                    &mlxcel_core::conv_transpose2d(x, weight, 2, 2, 0, 0, 1, 1, 0, 0, 1),
                    bias,
                );
                conv.forward(&x)
            }
            Self::Bilinear(conv) => {
                let resized =
                    resize_bilinear_nhwc(x, (s[1] as usize * 2, s[2] as usize * 2), false, false)?;
                conv.forward(&resized)
            }
            Self::Nearest(conv) => {
                let resized = resize_nearest_nhwc(x, (s[1] as usize * 2, s[2] as usize * 2))?;
                conv.forward(&resized)
            }
        })
    }
}

pub struct ConvStack {
    input: Vec<Option<Conv2d>>,
    resamplers: Vec<Resampler>,
    blocks: Vec<Vec<ResidualBlock>>,
    output: Vec<Option<Conv2d>>,
}
impl ConvStack {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &ConvStackConfig,
    ) -> Result<Self> {
        config.validate(prefix)?;
        let levels = config.dim_res_blocks.len();
        let counts = config.num_res_blocks.resolve(levels, "num_res_blocks")?;
        let resampler_kinds = config.resamplers.resolve(levels - 1, "resamplers")?;
        let output_dims = config.output_dims();
        let activation = Activation::parse(&config.activation)?;
        let mut input = Vec::with_capacity(levels);
        let mut output = Vec::with_capacity(levels);
        let mut blocks = Vec::with_capacity(levels);
        for i in 0..levels {
            input.push(
                config.dim_in[i]
                    .map(|in_channels| {
                        Conv2d::from_weights(
                            weights,
                            &format!("{prefix}.input_blocks.{i}"),
                            1,
                            in_channels,
                            config.dim_res_blocks[i],
                        )
                    })
                    .transpose()?,
            );
            output.push(
                output_dims[i]
                    .map(|out_channels| {
                        Conv2d::from_weights(
                            weights,
                            &format!("{prefix}.output_blocks.{i}"),
                            1,
                            config.dim_res_blocks[i],
                            out_channels,
                        )
                    })
                    .transpose()?,
            );
            let mut stage = Vec::with_capacity(counts[i]);
            for j in 0..counts[i] {
                let ch = config.dim_res_blocks[i];
                stage.push(ResidualBlock::from_weights(
                    weights,
                    &format!("{prefix}.res_blocks.{i}.{j}"),
                    ch,
                    ch,
                    ch * config.dim_times_res_block_hidden,
                    activation,
                    &config.res_block_in_norm,
                    &config.res_block_hidden_norm,
                )?)
            }
            blocks.push(stage)
        }
        let mut resamplers = Vec::with_capacity(levels - 1);
        for (i, kind) in resampler_kinds.iter().enumerate() {
            resamplers.push(Resampler::from_weights(
                weights,
                &format!("{prefix}.resamplers.{i}"),
                kind,
                config.dim_res_blocks[i],
                config.dim_res_blocks[i + 1],
            )?)
        }
        Ok(Self {
            input,
            resamplers,
            blocks,
            output,
        })
    }
    pub fn levels(&self) -> usize {
        self.blocks.len()
    }
    pub fn forward(&self, features: &[UniquePtr<MlxArray>]) -> Result<Vec<UniquePtr<MlxArray>>> {
        ensure!(
            features.len() == self.blocks.len(),
            "MoGe3 ConvStack feature count mismatch"
        );
        let mut x: Option<UniquePtr<MlxArray>> = None;
        let mut outputs = Vec::with_capacity(features.len());
        for i in 0..features.len() {
            let feature = match &self.input[i] {
                Some(v) => v.forward(&features[i], 0),
                None => mlxcel_core::copy(&features[i]),
            };
            let mut current = match x.take() {
                Some(previous) => mlxcel_core::add(&previous, &feature),
                None => feature,
            };
            for block in &self.blocks[i] {
                current = block.forward(&current)
            }
            outputs.push(match &self.output[i] {
                Some(v) => v.forward(&current, 0),
                None => mlxcel_core::copy(&current),
            });
            if i < self.resamplers.len() {
                current = self.resamplers[i].forward(&current)?;
            }
            x = Some(current)
        }
        Ok(outputs)
    }
}

pub struct ScaleHead {
    layers: Vec<Linear>,
}
impl ScaleHead {
    pub fn from_weights(weights: &WeightMap, config: &ScaleHeadConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..config.dims.len() - 1 {
            layers.push(Linear::from_weights(
                weights,
                &format!("scale_head.{}", i * 2),
                config.dims[i],
                config.dims[i + 1],
            )?)
        }
        Ok(Self { layers })
    }
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut x = mlxcel_core::copy(x);
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i + 1 < self.layers.len() {
                x = mlxcel_core::relu(&x)
            }
        }
        x
    }
}

pub fn sanitize_conv_transpose_weight(
    weight: &MlxArray,
    in_channels: usize,
    out_channels: usize,
) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(weight);
    let target = [out_channels as i32, 2, 2, in_channels as i32];
    if shape == target {
        return Ok(mlxcel_core::copy(weight));
    }
    let torch = [in_channels as i32, out_channels as i32, 2, 2];
    ensure!(
        shape == torch,
        "MoGe3 ConvTranspose weight shape {shape:?} is neither MLX {target:?} nor Torch {torch:?}"
    );
    Ok(mlxcel_core::transpose_axes(weight, &[1, 2, 3, 0]))
}
