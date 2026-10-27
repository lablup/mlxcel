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

//! Building blocks of the FLUX.2 VAE.
//!
//! A port of `mlx_vlm/models/flux2/vae/common/*` and the encoder/decoder block
//! files in <https://github.com/Blaizzy/mlx-vlm>. Upstream stores activations
//! NCHW and transposes around every convolution; this port keeps activations
//! NHWC end to end and transposes once at the VAE boundary, which is the same
//! arithmetic with fewer copies. Convolution weights are MLX layout
//! `(out, kH, kW, in)`.
//!
//! Every norm runs in f32 and casts back to the activation dtype, as upstream
//! does, so a f16 or bf16 VAE does not lose its statistics to rounding.

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Copy a weight by exact key, naming the key when it is missing.
pub(super) fn weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .with_context(|| format!("FLUX.2 VAE weight not found: {key}"))
}

fn weight_opt(weights: &WeightMap, key: &str) -> Option<UniquePtr<MlxArray>> {
    weights.get(key).map(|w| mlxcel_core::copy(w))
}

/// `[B, H, W, C]` -> `[B, C, H, W]`.
pub(super) fn nhwc_to_nchw(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::transpose_axes(x, &[0, 3, 1, 2])
}

/// `[B, C, H, W]` -> `[B, H, W, C]`.
pub(super) fn nchw_to_nhwc(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::transpose_axes(x, &[0, 2, 3, 1])
}

fn to_f32(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(x, dtype::FLOAT32)
}

/// 2-D convolution over NHWC with optional bias.
pub(super) struct Conv2d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
}

impl Conv2d {
    pub(super) fn load(
        weights: &WeightMap,
        prefix: &str,
        stride: i32,
        padding: i32,
    ) -> Result<Self> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: weight_opt(weights, &format!("{prefix}.bias")),
            stride,
            padding,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv2d(
            x,
            &self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            1,
            1,
            1,
        );
        match &self.bias {
            Some(b) => mlxcel_core::add(&y, b),
            None => y,
        }
    }
}

/// `nn.Linear` with bias, over the last axis.
struct Linear {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl Linear {
    fn load(weights: &WeightMap, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: weight_opt(weights, &format!("{prefix}.bias")),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let bias = self
            .bias
            .as_ref()
            .map_or(std::ptr::null(), |b| b.as_ref().unwrap() as *const MlxArray);
        // SAFETY: `bias` is null or points at an array owned by `self`.
        unsafe { mlxcel_core::linear_forward(x, &self.weight, bias) }
    }
}

/// `nn.GroupNorm(pytorch_compatible=True)` over the channel (last) axis.
///
/// Statistics are taken per group over all spatial positions and the group's
/// channels, then the per-channel affine is applied.
pub(super) struct GroupNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    groups: i32,
    eps: f32,
}

impl GroupNorm {
    pub(super) fn load(weights: &WeightMap, prefix: &str, groups: i32, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: weight(weights, &format!("{prefix}.bias"))?,
            groups,
            eps,
        })
    }

    /// Normalize in f32 and return the input's dtype.
    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let out_dtype = mlxcel_core::array_dtype(x);
        let x = to_f32(x);
        let shape = mlxcel_core::array_shape(&x);
        let batch = shape[0];
        let channels = shape[shape.len() - 1];
        let group_size = channels / self.groups;
        let spatial: i32 = shape[1..shape.len() - 1].iter().product();

        let grouped = mlxcel_core::reshape(&x, &[batch, spatial, self.groups, group_size]);
        let grouped = mlxcel_core::transpose_axes(&grouped, &[0, 2, 1, 3]);
        let grouped = mlxcel_core::reshape(&grouped, &[batch, self.groups, spatial * group_size]);
        // SAFETY: null weight and bias select the unscaled normalization.
        let normed = unsafe {
            mlxcel_core::fast_layer_norm(&grouped, std::ptr::null(), std::ptr::null(), self.eps)
        };
        let normed = mlxcel_core::reshape(&normed, &[batch, self.groups, spatial, group_size]);
        let normed = mlxcel_core::transpose_axes(&normed, &[0, 2, 1, 3]);
        let normed = mlxcel_core::reshape(&normed, &shape);

        let scaled = mlxcel_core::add(&mlxcel_core::multiply(&normed, &self.weight), &self.bias);
        mlxcel_core::astype(&scaled, out_dtype)
    }
}

/// `Flux2ResnetBlock2D`: two norm-silu-conv stages and a residual, with a 1x1
/// shortcut when the channel count changes.
pub(super) struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    pub(super) fn load(
        weights: &WeightMap,
        prefix: &str,
        in_channels: i32,
        out_channels: i32,
        groups: i32,
        eps: f32,
    ) -> Result<Self> {
        let shortcut = if in_channels != out_channels {
            Some(Conv2d::load(
                weights,
                &format!("{prefix}.conv_shortcut"),
                1,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: GroupNorm::load(weights, &format!("{prefix}.norm1"), groups, eps)?,
            conv1: Conv2d::load(weights, &format!("{prefix}.conv1"), 1, 1)?,
            norm2: GroupNorm::load(weights, &format!("{prefix}.norm2"), groups, eps)?,
            conv2: Conv2d::load(weights, &format!("{prefix}.conv2"), 1, 1)?,
            shortcut,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = mlxcel_core::silu(&self.norm1.forward(x));
        let h = self.conv1.forward(&h);
        let h = mlxcel_core::silu(&self.norm2.forward(&h));
        let h = self.conv2.forward(&h);
        let residual = match &self.shortcut {
            Some(conv) => conv.forward(x),
            None => mlxcel_core::copy(x),
        };
        mlxcel_core::add(&h, &residual)
    }
}

/// `Flux2AttentionBlock`: single-head self-attention over all spatial
/// positions, with a group norm in front and a residual around it.
pub(super) struct AttentionBlock {
    group_norm: GroupNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl AttentionBlock {
    pub(super) fn load(weights: &WeightMap, prefix: &str, groups: i32, eps: f32) -> Result<Self> {
        // Diffusers names the output projection `to_out.0`; upstream renames it.
        let to_out = if weights.contains_key(&format!("{prefix}.to_out.0.weight")) {
            format!("{prefix}.to_out.0")
        } else {
            format!("{prefix}.to_out")
        };
        Ok(Self {
            group_norm: GroupNorm::load(weights, &format!("{prefix}.group_norm"), groups, eps)?,
            to_q: Linear::load(weights, &format!("{prefix}.to_q"))?,
            to_k: Linear::load(weights, &format!("{prefix}.to_k"))?,
            to_v: Linear::load(weights, &format!("{prefix}.to_v"))?,
            to_out: Linear::load(weights, &to_out)?,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
        let normed = self.group_norm.forward(x);

        // One head: [B, 1, H*W, C].
        let heads = |lin: &Linear| {
            let y = lin.forward(&normed);
            mlxcel_core::reshape(&y, &[batch, 1, height * width, channels])
        };
        let (q, k, v) = (heads(&self.to_q), heads(&self.to_k), heads(&self.to_v));
        let scale = 1.0 / (channels as f32).sqrt();
        // Head dim `C` (512) is never flash-eligible on CUDA, so the shared
        // dispatch query-chunks the `[B, 1, H*W, H*W]` score matrix to
        // `MLXCEL_ATTENTION_CHUNK_BUDGET_MB`; below the budget it is the same
        // single unmasked SDPA call.
        let attended = mlxcel_core::layers::attention(&q, &k, &v, scale, None, 0.0, 0);
        let attended = mlxcel_core::reshape(&attended, &[batch, height, width, channels]);
        let attended = self.to_out.forward(&attended);
        mlxcel_core::add(x, &attended)
    }
}

/// `Flux2UNetMidBlock2D`: resnet, optional attention, resnet.
pub(super) struct MidBlock {
    resnets: [ResnetBlock; 2],
    attention: Option<AttentionBlock>,
}

impl MidBlock {
    pub(super) fn load(
        weights: &WeightMap,
        prefix: &str,
        channels: i32,
        groups: i32,
        eps: f32,
        add_attention: bool,
    ) -> Result<Self> {
        let resnet = |i: usize| {
            ResnetBlock::load(
                weights,
                &format!("{prefix}.resnets.{i}"),
                channels,
                channels,
                groups,
                eps,
            )
        };
        Ok(Self {
            resnets: [resnet(0)?, resnet(1)?],
            attention: if add_attention {
                Some(AttentionBlock::load(
                    weights,
                    &format!("{prefix}.attentions.0"),
                    groups,
                    eps,
                )?)
            } else {
                None
            },
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = self.resnets[0].forward(x);
        if let Some(attention) = &self.attention {
            h = attention.forward(&h);
        }
        self.resnets[1].forward(&h)
    }
}

/// `Flux2UpDecoderBlock2D`: resnets then an optional 2x nearest upsample + conv.
pub(super) struct UpDecoderBlock {
    resnets: Vec<ResnetBlock>,
    upsample: Option<Conv2d>,
}

impl UpDecoderBlock {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        weights: &WeightMap,
        prefix: &str,
        in_channels: i32,
        out_channels: i32,
        num_layers: i32,
        groups: i32,
        eps: f32,
        add_upsample: bool,
    ) -> Result<Self> {
        let resnets = (0..num_layers)
            .map(|i| {
                ResnetBlock::load(
                    weights,
                    &format!("{prefix}.resnets.{i}"),
                    if i == 0 { in_channels } else { out_channels },
                    out_channels,
                    groups,
                    eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let upsample = if add_upsample {
            Some(Conv2d::load(
                weights,
                &format!("{prefix}.upsamplers.0.conv"),
                1,
                1,
            )?)
        } else {
            None
        };
        Ok(Self { resnets, upsample })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = mlxcel_core::copy(x);
        for resnet in &self.resnets {
            h = resnet.forward(&h);
        }
        if let Some(conv) = &self.upsample {
            // Nearest-neighbour 2x on the spatial axes (1 and 2 in NHWC).
            let h2 = mlxcel_core::repeat(&h, 2, 1);
            let h2 = mlxcel_core::repeat(&h2, 2, 2);
            h = conv.forward(&h2);
        }
        h
    }
}

/// `Flux2DownEncoderBlock2D`: resnets then an optional stride-2 conv with
/// bottom/right zero padding.
pub(super) struct DownEncoderBlock {
    resnets: Vec<ResnetBlock>,
    downsample: Option<Conv2d>,
}

impl DownEncoderBlock {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        weights: &WeightMap,
        prefix: &str,
        in_channels: i32,
        out_channels: i32,
        num_layers: i32,
        groups: i32,
        eps: f32,
        add_downsample: bool,
    ) -> Result<Self> {
        let resnets = (0..num_layers)
            .map(|i| {
                ResnetBlock::load(
                    weights,
                    &format!("{prefix}.resnets.{i}"),
                    if i == 0 { in_channels } else { out_channels },
                    out_channels,
                    groups,
                    eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let downsample = if add_downsample {
            Some(Conv2d::load(
                weights,
                &format!("{prefix}.downsamplers.0.conv"),
                2,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsample,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = mlxcel_core::copy(x);
        for resnet in &self.resnets {
            h = resnet.forward(&h);
        }
        if let Some(conv) = &self.downsample {
            // (before, after) per axis: only H and W, one row/column at the end.
            let padded = mlxcel_core::pad(&h, &[0, 0, 0, 1, 0, 1, 0, 0], 0.0);
            h = conv.forward(&padded);
        }
        h
    }
}

/// Reject a block layout the weights cannot satisfy before any forward pass.
pub(super) fn check_block_channels(channels: &[i32], groups: i32) -> Result<()> {
    if channels.is_empty() {
        bail!("FLUX.2 VAE block_out_channels must not be empty");
    }
    if groups < 1 {
        bail!("FLUX.2 VAE norm_num_groups must be positive, got {groups}");
    }
    for c in channels {
        if *c < 1 || c % groups != 0 {
            bail!("FLUX.2 VAE channel count {c} is not divisible by norm_num_groups {groups}");
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "layers_tests.rs"]
mod tests;
