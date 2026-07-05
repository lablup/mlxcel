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

//! FastViTHD vision tower for FastVLM (`vision_tower.vision_model.*`).
//!
//! A hybrid convolutional / transformer encoder (Apple MobileCLIP-L 1024). It
//! runs entirely on channels-last `(B, H, W, C)` maps: a 3-block conv stem, a
//! flat `network` list of RepMixer stages, attention stages, inter-stage
//! PatchEmbed downsamples and RepCPE position encoders, then a `conv_exp`
//! MobileOneBlock with a squeeze-excite head. All weights are the inference-time
//! reparameterized form (checkpoints ship no training-time branches). The final
//! `(B, 16, 16, 3072)` map is flattened to `(B, 256, 3072)` for the projector.
//!
//! Reference: mlx-vlm `mlx_vlm/models/fastvlm/` (FastViTHD `MCi` tower).
//! Reused primitives: `Conv2dLayer` (channels-last grouped conv) from
//! [`super::gemma3n`] and inference `BatchNorm` from
//! [`crate::vision::detection::rt_detr_v2::layers`].

use super::{VisionEncoder, VisionEncoderOutput};
use crate::vision::detection::rt_detr_v2::layers::BatchNorm;
use crate::vision::encoders::gemma3n::Conv2dLayer;
use mlxcel_core::layers::Linear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const BN_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-5;
const HEAD_DIM: i32 = 32;

/// FastViTHD tower config; defaults are the released MobileCLIP-L 1024 geometry
/// (genuine checkpoints ship no `vision_config`, so every field has a default).
#[derive(Debug, Clone)]
pub struct FastvlmVisionConfig {
    pub image_size: i32,
    pub embed_dims: Vec<i32>,
    pub layers: Vec<usize>,
    pub token_mixers: Vec<String>,
    /// Per stage: whether a RepCPE position encoder precedes the blocks
    /// (`pos_embs_shapes[i]` non-null).
    pub pos_emb: Vec<bool>,
    pub mlp_ratios: Vec<i32>,
    pub down_patch_size: i32,
    pub down_stride: i32,
    pub repmixer_kernel_size: i32,
    pub cls_ratio: f32,
}

impl Default for FastvlmVisionConfig {
    fn default() -> Self {
        Self {
            image_size: 1024,
            embed_dims: vec![96, 192, 384, 768, 1536],
            layers: vec![2, 12, 24, 4, 2],
            token_mixers: vec![
                "repmixer".into(),
                "repmixer".into(),
                "repmixer".into(),
                "attention".into(),
                "attention".into(),
            ],
            pos_emb: vec![false, false, false, true, true],
            mlp_ratios: vec![4, 4, 4, 4, 4],
            down_patch_size: 7,
            down_stride: 2,
            repmixer_kernel_size: 3,
            cls_ratio: 2.0,
        }
    }
}

fn get(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("FastVLM vision weight missing: {name}"))
}

/// Load a channels-last conv (`{prefix}.weight` + optional `{prefix}.bias`).
#[allow(clippy::too_many_arguments)]
fn conv(
    weights: &WeightMap,
    prefix: &str,
    stride: i32,
    padding: i32,
    groups: i32,
    bias: bool,
) -> Result<Conv2dLayer, String> {
    let weight = get(weights, &format!("{prefix}.weight"))?;
    let bias = if bias {
        Some(get(weights, &format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(Conv2dLayer {
        weight,
        bias,
        stride_h: stride,
        stride_w: stride,
        padding_h: padding,
        padding_w: padding,
        dilation_h: 1,
        dilation_w: 1,
        groups,
    })
}

fn linear(weights: &WeightMap, prefix: &str, bias: bool) -> Result<Linear, String> {
    let weight = get(weights, &format!("{prefix}.weight"))?;
    let b = if bias {
        Some(get(weights, &format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, b))
}

/// Squeeze-excite head used only inside `conv_exp`.
struct SEBlock {
    reduce: Conv2dLayer, // 1x1 conv, C -> C/r, bias
    expand: Conv2dLayer, // 1x1 conv, C/r -> C, bias
}

impl SEBlock {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            reduce: conv(weights, &format!("{prefix}.reduce"), 1, 0, 1, true)?,
            expand: conv(weights, &format!("{prefix}.expand"), 1, 0, 1, true)?,
        })
    }

    /// `x`: `(B, H, W, C)` -> `x * sigmoid(expand(relu(reduce(mean_hw(x)))))`.
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::mean_axis(x, 1, true); // (B, 1, W, C)
        let s = mlxcel_core::mean_axis(&s, 2, true); // (B, 1, 1, C)
        let s = self.reduce.forward(&s);
        let s = mlxcel_core::relu(&s);
        let s = self.expand.forward(&s);
        let s = mlxcel_core::sigmoid(&s);
        mlxcel_core::multiply(x, &s)
    }
}

/// `GELU(SE(reparam_conv(x)))`; SE is present only in `conv_exp`.
struct MobileOneBlock {
    reparam_conv: Conv2dLayer,
    se: Option<SEBlock>,
}

impl MobileOneBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        stride: i32,
        padding: i32,
        groups: i32,
        with_se: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            reparam_conv: conv(
                weights,
                &format!("{prefix}.reparam_conv"),
                stride,
                padding,
                groups,
                true,
            )?,
            se: if with_se {
                Some(SEBlock::from_weights(weights, &format!("{prefix}.se"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.reparam_conv.forward(x);
        let x = match &self.se {
            Some(se) => se.forward(&x),
            None => x,
        };
        mlxcel_core::gelu(&x)
    }
}

/// `GELU(lkb_reparam(x))`, a strided large-kernel downsample conv.
struct ReparamLargeKernelConv {
    conv: Conv2dLayer,
}

impl ReparamLargeKernelConv {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::gelu(&self.conv.forward(x))
    }
}

/// `fc2(GELU(fc1(BN(dwconv(x)))))`; spatial mixing inside the FFN.
struct ConvFFN {
    dwconv: Conv2dLayer, // depthwise 7x7, no bias
    bn: BatchNorm,
    fc1: Conv2dLayer, // 1x1, bias
    fc2: Conv2dLayer, // 1x1, bias
}

impl ConvFFN {
    fn from_weights(weights: &WeightMap, prefix: &str, channels: i32) -> Result<Self, String> {
        Ok(Self {
            dwconv: conv(
                weights,
                &format!("{prefix}.conv.conv"),
                1,
                3,
                channels,
                false,
            )?,
            bn: BatchNorm::from_weights(weights, &format!("{prefix}.conv.bn"), BN_EPS)?,
            fc1: conv(weights, &format!("{prefix}.fc1"), 1, 0, 1, true)?,
            fc2: conv(weights, &format!("{prefix}.fc2"), 1, 0, 1, true)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.dwconv.forward(x);
        let x = self.bn.forward(&x);
        let x = self.fc1.forward(&x);
        let x = mlxcel_core::gelu(&x);
        self.fc2.forward(&x)
    }
}

/// Per-channel affine layer scale of shape `(1, 1, C)`, broadcast over `(B, H, W, C)`.
fn apply_layer_scale(scale: &MlxArray, x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::multiply(scale, x)
}

/// RepMixer: `t = token_mixer(x); y = t + layer_scale * convffn(t)`.
struct RepMixerBlock {
    token_mixer: Conv2dLayer, // depthwise 3x3, replaces x (identity folded in)
    convffn: ConvFFN,
    layer_scale: UniquePtr<MlxArray>, // (1, 1, C)
}

impl RepMixerBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: i32,
        kernel: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            token_mixer: conv(
                weights,
                &format!("{prefix}.token_mixer.reparam_conv"),
                1,
                kernel / 2,
                channels,
                true,
            )?,
            convffn: ConvFFN::from_weights(weights, &format!("{prefix}.convffn"), channels)?,
            layer_scale: get(weights, &format!("{prefix}.layer_scale"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let t = self.token_mixer.forward(x);
        let ff = self.convffn.forward(&t);
        mlxcel_core::add(&t, &apply_layer_scale(&self.layer_scale, &ff))
    }
}

/// Channel LayerNorm over axis 3 (C) of `(B, H, W, C)`.
struct ChannelLayerNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
}

impl ChannelLayerNorm {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            weight: get(weights, &format!("{prefix}.weight"))?,
            bias: get(weights, &format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let u = mlxcel_core::mean_axis(x, 3, true); // (B, H, W, 1)
        let centered = mlxcel_core::subtract(x, &u);
        let sq = mlxcel_core::multiply(&centered, &centered);
        let var = mlxcel_core::mean_axis(&sq, 3, true);
        let eps = mlxcel_core::full_f32(&[1], LN_EPS, mlxcel_core::array_dtype(x));
        let inv_std = mlxcel_core::rsqrt(&mlxcel_core::add(&var, &eps));
        let normed = mlxcel_core::multiply(&centered, &inv_std);
        let scaled = mlxcel_core::multiply(&self.weight, &normed);
        mlxcel_core::add(&scaled, &self.bias)
    }
}

/// Multi-head self-attention over the flattened `(B, N, C)` map, `head_dim = 32`.
struct MHSA {
    qkv: Linear,  // C -> 3C, no bias
    proj: Linear, // C -> C, bias
    num_heads: i32,
    scale: f32,
}

impl MHSA {
    fn from_weights(weights: &WeightMap, prefix: &str, channels: i32) -> Result<Self, String> {
        Ok(Self {
            qkv: linear(weights, &format!("{prefix}.qkv"), false)?,
            proj: linear(weights, &format!("{prefix}.proj"), true)?,
            num_heads: channels / HEAD_DIM,
            scale: (HEAD_DIM as f32).powf(-0.5),
        })
    }

    /// `x`: `(B, H, W, C)` -> `(B, H, W, C)`.
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
        let n = h * w;
        let heads = self.num_heads;

        let x2 = mlxcel_core::reshape(x, &[b, n, c]);
        let qkv = self.qkv.forward(&x2); // (B, N, 3C)
        let qkv = mlxcel_core::reshape(&qkv, &[b, n, 3, heads, HEAD_DIM]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[2, 0, 3, 1, 4]); // (3, B, heads, N, hd)
        let q = mlxcel_core::slice(&qkv, &[0, 0, 0, 0, 0], &[1, b, heads, n, HEAD_DIM]);
        let k = mlxcel_core::slice(&qkv, &[1, 0, 0, 0, 0], &[2, b, heads, n, HEAD_DIM]);
        let v = mlxcel_core::slice(&qkv, &[2, 0, 0, 0, 0], &[3, b, heads, n, HEAD_DIM]);
        let q = mlxcel_core::reshape(&q, &[b, heads, n, HEAD_DIM]);
        let k = mlxcel_core::reshape(&k, &[b, heads, n, HEAD_DIM]);
        let v = mlxcel_core::reshape(&v, &[b, heads, n, HEAD_DIM]);

        // SAFETY: q/k/v valid; null mask (bidirectional attention over the map).
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]); // (B, N, heads, hd)
        let out = mlxcel_core::reshape(&out, &[b, n, c]);
        let out = self.proj.forward(&out);
        mlxcel_core::reshape(&out, &[b, h, w, c])
    }
}

/// Attention block: `x1 = x + ls1 * MHSA(LNc(x)); y = x1 + ls2 * convffn(x1)`.
struct AttentionBlock {
    norm: ChannelLayerNorm,
    token_mixer: MHSA,
    ls1: UniquePtr<MlxArray>,
    convffn: ConvFFN,
    ls2: UniquePtr<MlxArray>,
}

impl AttentionBlock {
    fn from_weights(weights: &WeightMap, prefix: &str, channels: i32) -> Result<Self, String> {
        Ok(Self {
            norm: ChannelLayerNorm::from_weights(weights, &format!("{prefix}.norm"))?,
            token_mixer: MHSA::from_weights(weights, &format!("{prefix}.token_mixer"), channels)?,
            ls1: get(weights, &format!("{prefix}.layer_scale_1"))?,
            convffn: ConvFFN::from_weights(weights, &format!("{prefix}.convffn"), channels)?,
            ls2: get(weights, &format!("{prefix}.layer_scale_2"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let attn = self.token_mixer.forward(&self.norm.forward(x));
        let x1 = mlxcel_core::add(x, &apply_layer_scale(&self.ls1, &attn));
        let ff = self.convffn.forward(&x1);
        mlxcel_core::add(&x1, &apply_layer_scale(&self.ls2, &ff))
    }
}

/// RepCPE conditional position encoding: `y = reparam_conv(x)` (identity folded in).
struct RepCPE {
    conv: Conv2dLayer, // depthwise 7x7
}

impl RepCPE {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.conv.forward(x)
    }
}

/// Inter-stage downsample: ReparamLargeKernelConv then a 1x1 MobileOneBlock.
struct PatchEmbed {
    lkb: ReparamLargeKernelConv,
    pw: MobileOneBlock,
}

impl PatchEmbed {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.pw.forward(&self.lkb.forward(x))
    }
}

/// One entry of the flat `network` list.
enum NetworkEntry {
    RepMixerStage(Vec<RepMixerBlock>),
    AttentionStage(Vec<AttentionBlock>),
    PatchEmbed(PatchEmbed),
    RepCPE(RepCPE),
}

impl NetworkEntry {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            NetworkEntry::RepMixerStage(blocks) => {
                let mut h = mlxcel_core::copy(x);
                for b in blocks {
                    h = b.forward(&h);
                }
                h
            }
            NetworkEntry::AttentionStage(blocks) => {
                let mut h = mlxcel_core::copy(x);
                for b in blocks {
                    h = b.forward(&h);
                }
                h
            }
            NetworkEntry::PatchEmbed(pe) => pe.forward(x),
            NetworkEntry::RepCPE(rc) => rc.forward(x),
        }
    }
}

pub struct FastvlmVisionEncoder {
    stem: Vec<MobileOneBlock>, // patch_embed.blocks.{0,1,2}
    network: Vec<NetworkEntry>,
    conv_exp: MobileOneBlock,
}

impl FastvlmVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &FastvlmVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let vp = format!("{prefix}.vision_model");
        let dims = &config.embed_dims;
        let c0 = dims[0];

        // Stem: 3x3 s2 (3->c0, groups 1), 3x3 s2 dw (c0->c0), 1x1 s1 (c0->c0).
        let sp = format!("{vp}.patch_embed.blocks");
        let stem = vec![
            MobileOneBlock::from_weights(weights, &format!("{sp}.0"), 2, 1, 1, false)?,
            MobileOneBlock::from_weights(weights, &format!("{sp}.1"), 2, 1, c0, false)?,
            MobileOneBlock::from_weights(weights, &format!("{sp}.2"), 1, 0, 1, false)?,
        ];

        // Flat network list, generic stage rule. `idx` tracks the load-bearing
        // `network.<idx>` key segment as entries are appended.
        let mut network: Vec<NetworkEntry> = Vec::new();
        let mut idx = 0usize;
        let num_stages = dims.len();
        // `i` indexes several parallel config arrays (dims, layers, token_mixers,
        // pos_emb) plus `dims[i + 1]` for the downsample, so a range loop is clearer
        // than zipping five iterators.
        #[allow(clippy::needless_range_loop)]
        for i in 0..num_stages {
            let c = dims[i];
            let np = |ix: usize| format!("{vp}.network.{ix}");

            if config.pos_emb.get(i).copied().unwrap_or(false) {
                // RepCPE: depthwise 7x7 (kernel = down_patch_size), pad 3.
                let rc = RepCPE {
                    conv: conv(weights, &format!("{}.reparam_conv", np(idx)), 1, 3, c, true)?,
                };
                network.push(NetworkEntry::RepCPE(rc));
                idx += 1;
            }

            let n_blocks = config.layers[i];
            let is_attn = config.token_mixers.get(i).map(String::as_str) == Some("attention");
            if is_attn {
                let mut blocks = Vec::with_capacity(n_blocks);
                for j in 0..n_blocks {
                    blocks.push(AttentionBlock::from_weights(
                        weights,
                        &format!("{}.{j}", np(idx)),
                        c,
                    )?);
                }
                network.push(NetworkEntry::AttentionStage(blocks));
            } else {
                let mut blocks = Vec::with_capacity(n_blocks);
                for j in 0..n_blocks {
                    blocks.push(RepMixerBlock::from_weights(
                        weights,
                        &format!("{}.{j}", np(idx)),
                        c,
                        config.repmixer_kernel_size,
                    )?);
                }
                network.push(NetworkEntry::RepMixerStage(blocks));
            }
            idx += 1;

            // Inter-stage downsample (PatchEmbed) for every non-final stage.
            // The PatchEmbed output width comes from the checkpoint weights, not
            // config, so only the input channel count `c` is needed here.
            if i + 1 < num_stages {
                let pe = PatchEmbed {
                    lkb: ReparamLargeKernelConv {
                        conv: conv(
                            weights,
                            &format!("{}.proj.0.lkb_reparam", np(idx)),
                            config.down_stride,
                            config.down_patch_size / 2,
                            c, // depthwise over the input channels
                            true,
                        )?,
                    },
                    pw: MobileOneBlock::from_weights(
                        weights,
                        &format!("{}.proj.1", np(idx)),
                        1,
                        0,
                        1,
                        false,
                    )?,
                };
                network.push(NetworkEntry::PatchEmbed(pe));
                idx += 1;
            }
        }

        // conv_exp: depthwise 3x3 with channel multiplier cls_ratio, plus SE.
        let last = *dims.last().unwrap();
        let conv_exp = MobileOneBlock::from_weights(
            weights,
            &format!("{vp}.conv_exp"),
            1,
            1,
            last, // depthwise; out = last * cls_ratio
            true,
        )?;

        Ok(Self {
            stem,
            network,
            conv_exp,
        })
    }
}

impl VisionEncoder for FastvlmVisionEncoder {
    /// `pixel_values`: channels-last `(B, 1024, 1024, 3)`. Returns
    /// `hidden_states (B, 256, 3072)`.
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput {
        let mut h = mlxcel_core::copy(pixel_values);
        for block in &self.stem {
            h = block.forward(&h);
        }
        for entry in &self.network {
            h = entry.forward(&h);
        }
        h = self.conv_exp.forward(&h);

        let s = mlxcel_core::array_shape(&h);
        let (b, hh, ww, c) = (s[0], s[1], s[2], s[3]);
        let hidden_states = mlxcel_core::reshape(&h, &[b, hh * ww, c]);
        VisionEncoderOutput { hidden_states }
    }
}
