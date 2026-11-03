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

//! YOLO11 building blocks.
//!
//! All tensors are NHWC and BatchNorm uses running statistics, matching
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/yolo11/yolo11.py.
//! Convolutions are depth-separable through the `groups` argument, which the
//! detection head's DWConv branches need.

use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::detection::rt_detr_v2::layers::{max_pool2d, upsample_nearest_2x};

use super::weights::Loader;

/// YOLO11 BatchNorm epsilon (`nn.BatchNorm(c2, eps=1e-3)`).
const BATCH_NORM_EPS: f32 = 1e-3;

/// Convolution parameters over NHWC with optional groups and bias.
struct ConvWeights {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl ConvWeights {
    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let convolved = mlxcel_core::conv2d(
            x,
            &self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            1,
            1,
            self.groups,
        );
        match &self.bias {
            Some(bias) => mlxcel_core::add(&convolved, bias),
            None => convolved,
        }
    }
}

/// Inference BatchNorm over the channel axis of an NHWC tensor.
struct BatchNormWeights {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    mean: UniquePtr<MlxArray>,
    variance: UniquePtr<MlxArray>,
    eps: UniquePtr<MlxArray>,
}

impl BatchNormWeights {
    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let centered = mlxcel_core::subtract(x, &self.mean);
        let variance = mlxcel_core::add(&self.variance, &self.eps);
        let scaled = mlxcel_core::multiply(&centered, &mlxcel_core::rsqrt(&variance));
        mlxcel_core::add(&mlxcel_core::multiply(&scaled, &self.weight), &self.bias)
    }
}

/// Conv (no bias) + BatchNorm + SiLU. The `Conv` module of Ultralytics.
pub(super) struct Conv {
    weights: ConvWeights,
    norm: BatchNormWeights,
    activate: bool,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        activate: bool,
    ) -> Result<Self, String> {
        if groups == 0 || !input.is_multiple_of(groups) || !output.is_multiple_of(groups) {
            return Err(format!(
                "YOLO11 {prefix}: {input} -> {output} channels cannot split into {groups} groups"
            ));
        }
        let weight = loader.take(
            &format!("{prefix}.conv.weight"),
            &[output, kernel, kernel, input / groups],
        )?;
        let norm = BatchNormWeights {
            weight: loader.take(&format!("{prefix}.bn.weight"), &[output])?,
            bias: loader.take(&format!("{prefix}.bn.bias"), &[output])?,
            mean: loader.take(&format!("{prefix}.bn.running_mean"), &[output])?,
            variance: loader.take(&format!("{prefix}.bn.running_var"), &[output])?,
            eps: mlxcel_core::full_f32(&[1], BATCH_NORM_EPS, loader.dtype()),
        };
        Ok(Self {
            weights: ConvWeights {
                weight,
                bias: None,
                stride: stride as i32,
                padding: (kernel / 2) as i32,
                groups: groups as i32,
            },
            norm,
            activate,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = self.norm.forward(&self.weights.forward(x));
        if self.activate {
            mlxcel_core::silu(&y)
        } else {
            y
        }
    }
}

/// Plain Conv2d with bias, used at the end of both detection branches.
pub(super) struct BiasConv {
    weights: ConvWeights,
}

impl BiasConv {
    pub(super) fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weights: ConvWeights {
                weight: loader.take(
                    &format!("{prefix}.weight"),
                    &[output, kernel, kernel, input],
                )?,
                bias: Some(loader.take(&format!("{prefix}.bias"), &[output])?),
                stride: 1,
                padding: (kernel / 2) as i32,
                groups: 1,
            },
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.weights.forward(x)
    }
}

/// Standard bottleneck with an optional residual connection.
pub(super) struct Bottleneck {
    cv1: Conv,
    cv2: Conv,
    add: bool,
}

impl Bottleneck {
    pub(super) fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        output: usize,
        shortcut: bool,
        kernel: usize,
        groups: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            cv1: Conv::load(
                loader,
                &format!("{prefix}.cv1"),
                input,
                output,
                kernel,
                1,
                groups,
                true,
            )?,
            cv2: Conv::load(
                loader,
                &format!("{prefix}.cv2"),
                output,
                output,
                kernel,
                1,
                groups,
                true,
            )?,
            add: shortcut && input == output,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let out = self.cv2.forward(&self.cv1.forward(x));
        if self.add {
            mlxcel_core::add(x, &out)
        } else {
            out
        }
    }
}

/// CSP bottleneck with three convolutions (`C3k` in Ultralytics).
pub(super) struct C3k {
    cv1: Conv,
    cv2: Conv,
    cv3: Conv,
    blocks: Vec<Bottleneck>,
}

impl C3k {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        output: usize,
        repeats: usize,
        shortcut: bool,
        groups: usize,
        kernel: usize,
        expand: f32,
    ) -> Result<Self, String> {
        let hidden = scaled_channels(output, expand);
        let mut blocks = Vec::with_capacity(repeats);
        for index in 0..repeats {
            blocks.push(Bottleneck::load(
                loader,
                &format!("{prefix}.m.{index}"),
                hidden,
                hidden,
                shortcut,
                kernel,
                groups,
            )?);
        }
        Ok(Self {
            cv1: Conv::load(
                loader,
                &format!("{prefix}.cv1"),
                input,
                hidden,
                1,
                1,
                1,
                true,
            )?,
            cv2: Conv::load(
                loader,
                &format!("{prefix}.cv2"),
                input,
                hidden,
                1,
                1,
                1,
                true,
            )?,
            cv3: Conv::load(
                loader,
                &format!("{prefix}.cv3"),
                2 * hidden,
                output,
                1,
                1,
                1,
                true,
            )?,
            blocks,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut feature = self.blocks[0].forward(&self.cv1.forward(x));
        for block in &self.blocks[1..] {
            feature = block.forward(&feature);
        }
        let joined = mlxcel_core::concatenate(&feature, &self.cv2.forward(x), 3);
        self.cv3.forward(&joined)
    }
}

/// `C2f`, aliased to `C3k2` by the OmniParser topology.
pub(super) struct C3k2 {
    hidden: usize,
    cv1: Conv,
    cv2: Conv,
    blocks: Vec<C3k>,
}

impl C3k2 {
    pub(super) fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        output: usize,
        repeats: usize,
        expand: f32,
        shortcut: bool,
        groups: usize,
    ) -> Result<Self, String> {
        let hidden = scaled_channels(output, expand);
        let mut blocks = Vec::with_capacity(repeats);
        for index in 0..repeats {
            blocks.push(C3k::load(
                loader,
                &format!("{prefix}.m.{index}"),
                hidden,
                hidden,
                2,
                shortcut,
                groups,
                3,
                // Ultralytics `C3k` keeps its default expansion of 0.5.
                0.5,
            )?);
        }
        Ok(Self {
            hidden,
            cv1: Conv::load(
                loader,
                &format!("{prefix}.cv1"),
                input,
                2 * hidden,
                1,
                1,
                1,
                true,
            )?,
            cv2: Conv::load(
                loader,
                &format!("{prefix}.cv2"),
                (2 + repeats) * hidden,
                output,
                1,
                1,
                1,
                true,
            )?,
            blocks,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let split = self.cv1.forward(x);
        let mut pieces: Vec<UniquePtr<MlxArray>> = vec![
            slice_axis(&split, 3, 0, self.hidden as i32),
            slice_axis(&split, 3, self.hidden as i32, (2 * self.hidden) as i32),
        ];
        for block in &self.blocks {
            let next = block.forward(&pieces[pieces.len() - 1]);
            pieces.push(next);
        }
        let refs: Vec<&MlxArray> = pieces.iter().map(|piece| &**piece).collect();
        let joined = mlxcel_core::concatenate_many(&refs, 3);
        self.cv2.forward(&joined)
    }
}

/// Spatial Pyramid Pooling - Fast.
pub(super) struct Sppf {
    cv1: Conv,
    cv2: Conv,
    kernel: i32,
}

impl Sppf {
    pub(super) fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        let hidden = input / 2;
        Ok(Self {
            cv1: Conv::load(
                loader,
                &format!("{prefix}.cv1"),
                input,
                hidden,
                1,
                1,
                1,
                true,
            )?,
            cv2: Conv::load(
                loader,
                &format!("{prefix}.cv2"),
                hidden * 4,
                output,
                1,
                1,
                1,
                true,
            )?,
            kernel: kernel as i32,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let pooled = self.cv1.forward(x);
        let y1 = max_pool2d(&pooled, self.kernel, 1, self.kernel / 2);
        let y2 = max_pool2d(&y1, self.kernel, 1, self.kernel / 2);
        let y3 = max_pool2d(&y2, self.kernel, 1, self.kernel / 2);
        let joined = mlxcel_core::concatenate_many(&[&*pooled, &*y1, &*y2, &*y3], 3);
        self.cv2.forward(&joined)
    }
}

/// Position-sensitive multi-head attention over HxW tokens (NHWC).
struct Attention {
    heads: i32,
    head_dim: i32,
    key_dim: i32,
    scale: f32,
    qkv: Conv,
    proj: Conv,
    position: Conv,
}

impl Attention {
    fn load(
        loader: &mut Loader,
        prefix: &str,
        dim: usize,
        heads: usize,
        ratio: f32,
    ) -> Result<Self, String> {
        if heads == 0 || !dim.is_multiple_of(heads) {
            return Err(format!(
                "YOLO11 {prefix}: {dim} channels cannot split into {heads} attention heads"
            ));
        }
        let head_dim = dim / heads;
        let key_dim = (head_dim as f32 * ratio) as usize;
        let query_size = dim + key_dim * heads * 2;
        Ok(Self {
            heads: heads as i32,
            head_dim: head_dim as i32,
            key_dim: key_dim as i32,
            scale: (key_dim as f32).powf(-0.5),
            qkv: Conv::load(
                loader,
                &format!("{prefix}.qkv"),
                dim,
                query_size,
                1,
                1,
                1,
                false,
            )?,
            proj: Conv::load(loader, &format!("{prefix}.proj"), dim, dim, 1, 1, 1, false)?,
            position: Conv::load(loader, &format!("{prefix}.pe"), dim, dim, 3, 1, dim, false)?,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
        let tokens = height * width;
        let packed = self.qkv.forward(x);
        let packed = mlxcel_core::reshape(
            &packed,
            &[batch, tokens, self.heads, 2 * self.key_dim + self.head_dim],
        );
        let query = slice_axis(&packed, 3, 0, self.key_dim);
        let key = slice_axis(&packed, 3, self.key_dim, 2 * self.key_dim);
        let value = slice_axis(
            &packed,
            3,
            2 * self.key_dim,
            2 * self.key_dim + self.head_dim,
        );
        // (B, heads, N, d)
        let query = mlxcel_core::transpose_axes(&query, &[0, 2, 1, 3]);
        let key = mlxcel_core::transpose_axes(&key, &[0, 2, 1, 3]);
        let value = mlxcel_core::transpose_axes(&value, &[0, 2, 1, 3]);

        let scaled_query = mlxcel_core::multiply_scalar(&query, self.scale);
        let scores = mlxcel_core::matmul(&scaled_query, &swap_last_two(&key));
        let weights = mlxcel_core::softmax(&scores, -1);
        let attended = mlxcel_core::matmul(&swap_last_two(&value), &swap_last_two(&weights));
        let attended = swap_last_two(&attended);
        let attended = mlxcel_core::reshape(&attended, &[batch, tokens, self.heads, self.head_dim]);
        let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
        let attended = mlxcel_core::reshape(&attended, &[batch, height, width, channels]);

        let spatial_value = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&value, &[0, 2, 1, 3]),
            &[batch, height, width, channels],
        );
        let combined = mlxcel_core::add(&attended, &self.position.forward(&spatial_value));
        self.proj.forward(&combined)
    }
}

/// Attention + feed-forward block with residual connections.
struct PsaBlock {
    attention: Attention,
    ffn0: Conv,
    ffn1: Conv,
    add: bool,
}

impl PsaBlock {
    fn load(
        loader: &mut Loader,
        prefix: &str,
        channels: usize,
        heads: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            attention: Attention::load(loader, &format!("{prefix}.attn"), channels, heads, 0.5)?,
            ffn0: Conv::load(
                loader,
                &format!("{prefix}.ffn.0"),
                channels,
                channels * 2,
                1,
                1,
                1,
                true,
            )?,
            ffn1: Conv::load(
                loader,
                &format!("{prefix}.ffn.1"),
                channels * 2,
                channels,
                1,
                1,
                1,
                false,
            )?,
            add: true,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let attended = self.attention.forward(x);
        let attended = if self.add {
            mlxcel_core::add(x, &attended)
        } else {
            attended
        };
        let expanded = self.ffn1.forward(&self.ffn0.forward(&attended));
        if self.add {
            mlxcel_core::add(&attended, &expanded)
        } else {
            expanded
        }
    }
}

/// CSP module with position-sensitive attention (`C2PSA`).
pub(super) struct C2Psa {
    hidden: usize,
    cv1: Conv,
    cv2: Conv,
    blocks: Vec<PsaBlock>,
}

impl C2Psa {
    pub(super) fn load(
        loader: &mut Loader,
        prefix: &str,
        channels: usize,
        repeats: usize,
        expand: f32,
    ) -> Result<Self, String> {
        let hidden = scaled_channels(channels, expand);
        // Ultralytics: max(c // 64, 1) with c = channels / 2.
        let heads = (hidden / 64).max(1);
        let mut blocks = Vec::with_capacity(repeats);
        for index in 0..repeats {
            blocks.push(PsaBlock::load(
                loader,
                &format!("{prefix}.m.{index}"),
                hidden,
                heads,
            )?);
        }
        Ok(Self {
            hidden,
            cv1: Conv::load(
                loader,
                &format!("{prefix}.cv1"),
                channels,
                2 * hidden,
                1,
                1,
                1,
                true,
            )?,
            cv2: Conv::load(
                loader,
                &format!("{prefix}.cv2"),
                2 * hidden,
                channels,
                1,
                1,
                1,
                true,
            )?,
            blocks,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let split = self.cv1.forward(x);
        let left = slice_axis(&split, 3, 0, self.hidden as i32);
        let mut feature = slice_axis(&split, 3, self.hidden as i32, (2 * self.hidden) as i32);
        for block in &self.blocks {
            feature = block.forward(&feature);
        }
        let joined = mlxcel_core::concatenate_many(&[&*left, &*feature], 3);
        self.cv2.forward(&joined)
    }
}

/// Nearest-neighbour 2x upsample.
pub(super) fn upsample2x(x: &MlxArray) -> UniquePtr<MlxArray> {
    upsample_nearest_2x(x)
}

fn scaled_channels(channels: usize, expand: f32) -> usize {
    ((channels as f32) * expand) as usize
}

fn swap_last_two(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::transpose_axes(x, &[0, 1, 3, 2])
}

#[cfg(test)]
mod tests {
    use super::scaled_channels;

    #[test]
    fn expand_scales_like_python_int_truncation() {
        assert_eq!(scaled_channels(256, 0.25), 64);
        assert_eq!(scaled_channels(512, 0.25), 128);
        assert_eq!(scaled_channels(512, 0.5), 256);
        assert_eq!(scaled_channels(1024, 0.5), 512);
    }
}
