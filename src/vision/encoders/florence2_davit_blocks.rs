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

//! Building blocks of the Florence-2 DaViT vision backbone: patch
//! embedding, depthwise positional convolutions, the feed-forward net, and
//! the block wrappers that assemble them around the two attention flavors in
//! `florence2_davit_attention`.
//!
//! Every block runs a windowed spatial attention half and a channel
//! attention half. Both are wrapped in a pre-norm residual, and both are
//! sandwiched between two depthwise positional convolutions.
//!
//! Activations are token-major `(B, N, C)` with the spatial extent carried
//! alongside as `(H, W)`; the convolutional paths reshape to channels-last
//! `(B, H, W, C)` and back. MLX `conv2d` takes input `(B, H, W, C_in)` with
//! weight `(C_out, kH, kW, C_in / groups)`.
//!
//! Reference: mlx-vlm `mlx_vlm/models/florence2/vision.py`
//! (<https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/vision.py>).

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::florence2_davit_attention::{ChannelAttention, WindowAttention};

/// `nn.LayerNorm` default epsilon in MLX, which the reference relies on.
pub(crate) const LAYER_NORM_EPS: f32 = 1e-5;

fn get_weight(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Florence-2 DaViT weight missing: {name}"))
}

pub(crate) fn layer_norm_from_weights(
    weights: &WeightMap,
    prefix: &str,
) -> Result<LayerNorm, String> {
    Ok(LayerNorm::new(
        get_weight(weights, &format!("{prefix}.weight"))?,
        weights
            .get(&format!("{prefix}.bias"))
            .map(|w| mlxcel_core::copy(w)),
        LAYER_NORM_EPS,
    ))
}

/// Per-stage patch embedding: a strided convolution plus one LayerNorm,
/// applied either before the projection (on the incoming stage's channel
/// width) or after it (on the outgoing width).
pub(crate) struct ConvEmbed {
    proj_weight: UniquePtr<MlxArray>,
    proj_bias: Option<UniquePtr<MlxArray>>,
    norm: LayerNorm,
    pre_norm: bool,
    stride: i32,
    padding: i32,
}

impl ConvEmbed {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_channels: i32,
        out_channels: i32,
        stride: i32,
        padding: i32,
        pre_norm: bool,
    ) -> Result<Self, String> {
        let proj_weight = get_weight(weights, &format!("{prefix}.proj.weight"))?;
        let shape = mlxcel_core::array_shape(&proj_weight);
        if shape.len() != 4 || shape[0] != out_channels || shape[3] != in_channels {
            return Err(format!(
                "Florence-2 DaViT {prefix}.proj.weight has shape {shape:?}, expected \
                 channels-last [{out_channels}, kH, kW, {in_channels}]; pass the weight map \
                 through `sanitize` first (a PyTorch export is [out, in, kH, kW])"
            ));
        }
        Ok(Self {
            proj_weight,
            proj_bias: weights
                .get(&format!("{prefix}.proj.bias"))
                .map(|w| mlxcel_core::copy(w)),
            norm: layer_norm_from_weights(weights, &format!("{prefix}.norm"))?,
            pre_norm,
            stride,
            padding,
        })
    }

    /// `x` is either NCHW `(B, 3, H, W)` (stage 0, straight off the pixel
    /// tensor) or token-major `(B, N, C)` (every later stage). Returns the
    /// embedded tokens and the new spatial extent.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        size: (i32, i32),
    ) -> (UniquePtr<MlxArray>, (i32, i32)) {
        let shape = mlxcel_core::array_shape(x);
        let nhwc = if shape.len() == 3 {
            let normed = if self.pre_norm {
                self.norm.forward(x)
            } else {
                mlxcel_core::copy(x)
            };
            mlxcel_core::reshape(&normed, &[-1, size.0, size.1, shape[2]])
        } else {
            mlxcel_core::transpose_axes(x, &[0, 2, 3, 1])
        };

        let projected = mlxcel_core::conv2d(
            &nhwc,
            &self.proj_weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            1,
            1,
            1,
        );
        let projected = match &self.proj_bias {
            Some(bias) => mlxcel_core::add(&projected, bias),
            None => projected,
        };

        let out_shape = mlxcel_core::array_shape(&projected);
        let (b, h, w, c) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
        let tokens = mlxcel_core::reshape(&projected, &[b, h * w, c]);
        let tokens = if self.pre_norm {
            tokens
        } else {
            self.norm.forward(&tokens)
        };
        (tokens, (h, w))
    }
}

/// Depthwise 3x3 positional convolution used as a residual branch around
/// both attention flavors (`conv_at_attn` / `conv_at_ffn`).
pub(crate) struct DepthWiseConv2d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    channels: i32,
}

impl DepthWiseConv2d {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: i32,
    ) -> Result<Self, String> {
        let weight = get_weight(weights, &format!("{prefix}.dw.weight"))?;
        let shape = mlxcel_core::array_shape(&weight);
        if shape.len() != 4 || shape[0] != channels || shape[3] != 1 {
            return Err(format!(
                "Florence-2 DaViT {prefix}.dw.weight has shape {shape:?}, expected \
                 channels-last [{channels}, kH, kW, 1]; pass the weight map through \
                 `sanitize` first (a PyTorch export is [C, 1, kH, kW])"
            ));
        }
        Ok(Self {
            weight,
            bias: weights
                .get(&format!("{prefix}.dw.bias"))
                .map(|w| mlxcel_core::copy(w)),
            channels,
        })
    }

    /// `(B, N, C)` -> `(B, N, C)` with `N == H * W`. Stride 1 and padding 1
    /// keep the spatial extent, so the caller's `size` is unchanged.
    pub(crate) fn forward(&self, x: &MlxArray, size: (i32, i32)) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let grid = mlxcel_core::reshape(x, &[b, size.0, size.1, c]);
        let y = mlxcel_core::conv2d(&grid, &self.weight, 1, 1, 1, 1, 1, 1, self.channels);
        let y = match &self.bias {
            Some(bias) => mlxcel_core::add(&y, bias),
            None => y,
        };
        mlxcel_core::reshape(&y, &[b, n, c])
    }
}

/// Two-layer feed-forward with the exact (erf-based) GELU that
/// `mlx.nn.GELU()` defaults to. Weight path is `ffn.fn.net.fc{1,2}`.
pub(crate) struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub(crate) fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            fc1: Linear::from_weights(weights, &format!("{prefix}.net.fc1"))?,
            fc2: Linear::from_weights(weights, &format!("{prefix}.net.fc2"))?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(x);
        let h = mlxcel_core::gelu(&h);
        self.fc2.forward(&h)
    }
}

/// `conv1 -> window_attn -> conv2 -> ffn`, each a residual branch.
pub(crate) struct SpatialBlock {
    conv1: Option<DepthWiseConv2d>,
    attn_norm: LayerNorm,
    window_attn: WindowAttention,
    conv2: Option<DepthWiseConv2d>,
    ffn_norm: LayerNorm,
    ffn: Mlp,
}

/// `conv1 -> channel_attn -> conv2 -> ffn`, each a residual branch.
pub(crate) struct ChannelBlock {
    conv1: Option<DepthWiseConv2d>,
    attn_norm: LayerNorm,
    channel_attn: ChannelAttention,
    conv2: Option<DepthWiseConv2d>,
    ffn_norm: LayerNorm,
    ffn: Mlp,
}

/// Static per-stage block geometry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockParams {
    pub dim: i32,
    pub num_heads: i32,
    pub num_groups: i32,
    pub window_size: i32,
    pub conv_at_attn: bool,
    pub conv_at_ffn: bool,
}

fn optional_dw(
    weights: &WeightMap,
    prefix: &str,
    enabled: bool,
    dim: i32,
) -> Result<Option<DepthWiseConv2d>, String> {
    if enabled {
        Ok(Some(DepthWiseConv2d::from_weights(weights, prefix, dim)?))
    } else {
        Ok(None)
    }
}

impl SpatialBlock {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        params: BlockParams,
    ) -> Result<Self, String> {
        Ok(Self {
            conv1: optional_dw(
                weights,
                &format!("{prefix}.conv1.fn"),
                params.conv_at_attn,
                params.dim,
            )?,
            attn_norm: layer_norm_from_weights(weights, &format!("{prefix}.window_attn.norm"))?,
            window_attn: WindowAttention::from_weights(
                weights,
                &format!("{prefix}.window_attn.fn"),
                params.dim,
                params.num_heads,
                params.window_size,
            )?,
            conv2: optional_dw(
                weights,
                &format!("{prefix}.conv2.fn"),
                params.conv_at_ffn,
                params.dim,
            )?,
            ffn_norm: layer_norm_from_weights(weights, &format!("{prefix}.ffn.norm"))?,
            ffn: Mlp::from_weights(weights, &format!("{prefix}.ffn.fn"))?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray, size: (i32, i32)) -> UniquePtr<MlxArray> {
        let mut cur = match &self.conv1 {
            Some(conv) => mlxcel_core::add(x, &conv.forward(x, size)),
            None => mlxcel_core::copy(x),
        };
        let branch = self
            .window_attn
            .forward(&self.attn_norm.forward(&cur), size);
        cur = mlxcel_core::add(&cur, &branch);
        if let Some(conv) = &self.conv2 {
            cur = mlxcel_core::add(&cur, &conv.forward(&cur, size));
        }
        let branch = self.ffn.forward(&self.ffn_norm.forward(&cur));
        mlxcel_core::add(&cur, &branch)
    }
}

impl ChannelBlock {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        params: BlockParams,
    ) -> Result<Self, String> {
        Ok(Self {
            conv1: optional_dw(
                weights,
                &format!("{prefix}.conv1.fn"),
                params.conv_at_attn,
                params.dim,
            )?,
            attn_norm: layer_norm_from_weights(weights, &format!("{prefix}.channel_attn.norm"))?,
            channel_attn: ChannelAttention::from_weights(
                weights,
                &format!("{prefix}.channel_attn.fn"),
                params.num_groups,
            )?,
            conv2: optional_dw(
                weights,
                &format!("{prefix}.conv2.fn"),
                params.conv_at_ffn,
                params.dim,
            )?,
            ffn_norm: layer_norm_from_weights(weights, &format!("{prefix}.ffn.norm"))?,
            ffn: Mlp::from_weights(weights, &format!("{prefix}.ffn.fn"))?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray, size: (i32, i32)) -> UniquePtr<MlxArray> {
        let mut cur = match &self.conv1 {
            Some(conv) => mlxcel_core::add(x, &conv.forward(x, size)),
            None => mlxcel_core::copy(x),
        };
        let branch = self.channel_attn.forward(&self.attn_norm.forward(&cur));
        cur = mlxcel_core::add(&cur, &branch);
        if let Some(conv) = &self.conv2 {
            cur = mlxcel_core::add(&cur, &conv.forward(&cur, size));
        }
        let branch = self.ffn.forward(&self.ffn_norm.forward(&cur));
        mlxcel_core::add(&cur, &branch)
    }
}

/// One DaViT block: the spatial half followed by the channel half.
///
/// Stochastic depth (`DropPath`) is a training-only regularizer and is the
/// identity at inference, so `drop_path_rate` is parsed but has no effect
/// here.
pub(crate) struct Block {
    spatial_block: SpatialBlock,
    channel_block: ChannelBlock,
}

impl Block {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        params: BlockParams,
    ) -> Result<Self, String> {
        Ok(Self {
            spatial_block: SpatialBlock::from_weights(
                weights,
                &format!("{prefix}.spatial_block"),
                params,
            )?,
            channel_block: ChannelBlock::from_weights(
                weights,
                &format!("{prefix}.channel_block"),
                params,
            )?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray, size: (i32, i32)) -> UniquePtr<MlxArray> {
        let x = self.spatial_block.forward(x, size);
        self.channel_block.forward(&x, size)
    }
}
