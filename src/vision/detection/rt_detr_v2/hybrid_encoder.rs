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

//! RT-DETRv2 hybrid encoder: AIFI transformer on the deepest level, top-down
//! FPN, bottom-up PAN.
//!
//! Port of the hybrid-encoder half of
//! `references/mlx-vlm/mlx_vlm/models/rt_detr_v2/vision.py`. All feature maps
//! are NHWC. Output is `num_levels` feature maps at the original strides, all
//! with `encoder_hidden_dim` channels.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::common::{copy_weight, copy_weight_opt};
use super::config::RtDetrV2Config;
use super::layers::{Activation, ConvNorm, upsample_nearest_2x};

/// Multi-head self-attention with position embeddings added to q,k (not v).
/// Shared by AIFI here and by the decoder's `SelfAttention` (same field names
/// `q_proj`/`k_proj`/`v_proj`/`out_proj`).
pub struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl SelfAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        d_model: usize,
        n_heads: usize,
    ) -> Result<Self, String> {
        let head_dim = d_model / n_heads;
        Ok(Self {
            q_proj: Linear::from_weights(weights, &format!("{prefix}.q_proj"))?,
            k_proj: Linear::from_weights(weights, &format!("{prefix}.k_proj"))?,
            v_proj: Linear::from_weights(weights, &format!("{prefix}.v_proj"))?,
            out_proj: Linear::from_weights(weights, &format!("{prefix}.out_proj"))?,
            n_heads: n_heads as i32,
            head_dim: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: (B, N, D). `pos`: optional (·, N, D) added to q,k.
    pub fn forward(&self, x: &MlxArray, pos: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, n, d) = (shape[0], shape[1], shape[2]);
        let qk = match pos {
            Some(p) => mlxcel_core::add(x, p),
            None => mlxcel_core::copy(x),
        };
        let split = |proj: &Linear, src: &MlxArray| {
            let t = proj.forward(src);
            let t = mlxcel_core::reshape(&t, &[b, n, self.n_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = split(&self.q_proj, &qk);
        let k = split(&self.k_proj, &qk);
        let v = split(&self.v_proj, x);
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, n, d]);
        self.out_proj.forward(&out)
    }
}

/// One AIFI encoder layer: pre/post-norm MHSA + FFN.
struct EncoderLayer {
    self_attn: SelfAttention,
    self_attn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_ln: LayerNorm,
    act: Activation,
    normalize_before: bool,
}

impl EncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &RtDetrV2Config,
    ) -> Result<Self, String> {
        let d = cfg.encoder_hidden_dim;
        Ok(Self {
            self_attn: SelfAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                d,
                cfg.encoder_attention_heads,
            )?,
            self_attn_ln: load_layer_norm(
                weights,
                &format!("{prefix}.self_attn_layer_norm"),
                cfg.layer_norm_eps,
            )?,
            fc1: Linear::from_weights(weights, &format!("{prefix}.fc1"))?,
            fc2: Linear::from_weights(weights, &format!("{prefix}.fc2"))?,
            final_ln: load_layer_norm(
                weights,
                &format!("{prefix}.final_layer_norm"),
                cfg.layer_norm_eps,
            )?,
            act: Activation::parse(&cfg.encoder_activation_function)?,
            normalize_before: cfg.normalize_before,
        })
    }

    fn forward(&self, x: &MlxArray, pos: &MlxArray) -> UniquePtr<MlxArray> {
        // Self-attention sub-block.
        let residual = mlxcel_core::copy(x);
        let h = if self.normalize_before {
            self.self_attn_ln.forward(x)
        } else {
            mlxcel_core::copy(x)
        };
        let h = self.self_attn.forward(&h, Some(pos));
        let h = mlxcel_core::add(&residual, &h);
        let h = if self.normalize_before {
            h
        } else {
            self.self_attn_ln.forward(&h)
        };

        // FFN sub-block.
        let residual = mlxcel_core::copy(&h);
        let f = if self.normalize_before {
            self.final_ln.forward(&h)
        } else {
            mlxcel_core::copy(&h)
        };
        let f = self.fc2.forward(&self.act.apply(&self.fc1.forward(&f)));
        let f = mlxcel_core::add(&residual, &f);
        if self.normalize_before {
            f
        } else {
            self.final_ln.forward(&f)
        }
    }
}

/// Attention-based Intra-scale Feature Interaction: a stack of `EncoderLayer`s
/// over a single flattened feature map, plus a 2D sine position embedding.
struct Aifi {
    layers: Vec<EncoderLayer>,
    embed_dim: usize,
    temperature: f32,
}

impl Aifi {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &RtDetrV2Config,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(EncoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                cfg,
            )?);
        }
        Ok(Self {
            layers,
            embed_dim: cfg.encoder_hidden_dim,
            temperature: cfg.positional_encoding_temperature,
        })
    }

    /// `x`: (B, H, W, C). Returns the same shape.
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let mut flat = mlxcel_core::reshape(x, &[b, h * w, c]);
        let pos = sine_position_embedding(h, w, self.embed_dim, self.temperature);
        for layer in &self.layers {
            flat = layer.forward(&flat, &pos);
        }
        mlxcel_core::reshape(&flat, &[b, h, w, c])
    }
}

/// 2D sinusoidal position embedding, `(1, H*W, embed_dim)`.
///
/// `embed_dim` is split into `[sin(h), cos(h), sin(w), cos(w)]` quarters. Built
/// in f32 from `arange` + broadcast to mirror
/// `SinePositionEmbedding` in the reference (`meshgrid` with default "xy"
/// indexing, then `out_w/out_h = grid.flatten()[:, None] * omega[None, :]`).
fn sine_position_embedding(
    height: i32,
    width: i32,
    embed_dim: usize,
    temperature: f32,
) -> UniquePtr<MlxArray> {
    let pos_dim = (embed_dim / 4) as i32;
    let dt = mlxcel_core::dtype::FLOAT32;

    // omega[j] = 1 / temperature^(j / pos_dim), shape (pos_dim,).
    let omega: Vec<f32> = (0..pos_dim)
        .map(|j| 1.0 / temperature.powf(j as f32 / pos_dim as f32))
        .collect();
    let omega = mlxcel_core::from_slice_f32(&omega, &[pos_dim]);

    // meshgrid(grid_w, grid_h) with "xy" indexing produces gw, gh each of shape
    // (height, width); flatten row-major. Build the flattened coordinate
    // vectors directly: for index k = r*width + col (r in [0,height),
    // col in [0,width)) gw=col, gh=r.
    let n = (height * width) as usize;
    let mut gw_flat = vec![0f32; n];
    let mut gh_flat = vec![0f32; n];
    for r in 0..height {
        for col in 0..width {
            let k = (r * width + col) as usize;
            gw_flat[k] = col as f32;
            gh_flat[k] = r as f32;
        }
    }
    let gw = mlxcel_core::from_slice_f32(&gw_flat, &[height * width, 1]);
    let gh = mlxcel_core::from_slice_f32(&gh_flat, &[height * width, 1]);
    let omega_row = mlxcel_core::reshape(&omega, &[1, pos_dim]);

    // out_* = grid[:, None] * omega[None, :] -> (H*W, pos_dim).
    let out_w = mlxcel_core::multiply(&gw, &omega_row);
    let out_h = mlxcel_core::multiply(&gh, &omega_row);

    let parts = [
        mlxcel_core::sin(&out_h),
        mlxcel_core::cos(&out_h),
        mlxcel_core::sin(&out_w),
        mlxcel_core::cos(&out_w),
    ];
    // Concatenate along the feature axis -> (H*W, embed_dim).
    let mut pe = mlxcel_core::concatenate(&parts[0], &parts[1], 1);
    pe = mlxcel_core::concatenate(&pe, &parts[2], 1);
    pe = mlxcel_core::concatenate(&pe, &parts[3], 1);
    let pe = mlxcel_core::astype(&pe, dt);
    // (1, H*W, embed_dim).
    mlxcel_core::expand_dims(&pe, 0)
}

/// RepVGG block: 3x3 conv + 1x1 conv branches summed and activated.
struct RepVggBlock {
    conv1: ConvNorm, // 3x3, pad 1
    conv2: ConvNorm, // 1x1, pad 0
    act: Activation,
}

impl RepVggBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        act: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            conv1: ConvNorm::from_weights(
                weights,
                &format!("{prefix}.conv1"),
                1,
                1,
                Activation::None,
                eps,
            )?,
            conv2: ConvNorm::from_weights(
                weights,
                &format!("{prefix}.conv2"),
                1,
                0,
                Activation::None,
                eps,
            )?,
            act,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::add(&self.conv1.forward(x), &self.conv2.forward(x));
        self.act.apply(&y)
    }
}

/// CSPNet block built from RepVGG blocks. `conv3` collapses to identity when
/// `hidden_channels == out_channels` (no `conv3.*` keys present).
struct CspRepLayer {
    conv1: ConvNorm,
    conv2: ConvNorm,
    bottlenecks: Vec<RepVggBlock>,
    conv3: Option<ConvNorm>,
}

impl CspRepLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_blocks: usize,
        act: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        let conv1 = ConvNorm::from_weights(weights, &format!("{prefix}.conv1"), 1, 0, act, eps)?;
        let conv2 = ConvNorm::from_weights(weights, &format!("{prefix}.conv2"), 1, 0, act, eps)?;
        let mut bottlenecks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            bottlenecks.push(RepVggBlock::from_weights(
                weights,
                &format!("{prefix}.bottlenecks.{i}"),
                act,
                eps,
            )?);
        }
        // conv3 only exists when hidden_channels != out_channels; detect by key.
        let conv3 = if copy_weight_opt(weights, &format!("{prefix}.conv3.conv.weight")).is_some() {
            Some(ConvNorm::from_weights(
                weights,
                &format!("{prefix}.conv3"),
                1,
                0,
                act,
                eps,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv2,
            bottlenecks,
            conv3,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut a = self.conv1.forward(x);
        for b in &self.bottlenecks {
            a = b.forward(&a);
        }
        let b = self.conv2.forward(x);
        let sum = mlxcel_core::add(&a, &b);
        match &self.conv3 {
            Some(c) => c.forward(&sum),
            None => sum,
        }
    }
}

/// The hybrid encoder.
pub struct HybridEncoder {
    aifi: Vec<Aifi>,
    encode_proj_layers: Vec<usize>,
    lateral_convs: Vec<ConvNorm>,
    fpn_blocks: Vec<CspRepLayer>,
    downsample_convs: Vec<ConvNorm>,
    pan_blocks: Vec<CspRepLayer>,
}

impl HybridEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &RtDetrV2Config,
    ) -> Result<Self, String> {
        let act = Activation::parse(&cfg.activation_function)?;
        let eps = cfg.batch_norm_eps;
        let num_levels = cfg.num_levels();
        let num_fpn = num_levels - 1;

        let mut aifi = Vec::with_capacity(cfg.encode_proj_layers.len());
        for i in 0..cfg.encode_proj_layers.len() {
            aifi.push(Aifi::from_weights(
                weights,
                &format!("{prefix}.aifi.{i}"),
                cfg,
            )?);
        }

        let mut lateral_convs = Vec::with_capacity(num_fpn);
        let mut fpn_blocks = Vec::with_capacity(num_fpn);
        for i in 0..num_fpn {
            lateral_convs.push(ConvNorm::from_weights(
                weights,
                &format!("{prefix}.lateral_convs.{i}"),
                1,
                0,
                act,
                eps,
            )?);
            fpn_blocks.push(CspRepLayer::from_weights(
                weights,
                &format!("{prefix}.fpn_blocks.{i}"),
                3,
                act,
                eps,
            )?);
        }

        let mut downsample_convs = Vec::with_capacity(num_fpn);
        let mut pan_blocks = Vec::with_capacity(num_fpn);
        for i in 0..num_fpn {
            downsample_convs.push(ConvNorm::from_weights(
                weights,
                &format!("{prefix}.downsample_convs.{i}"),
                2,
                1,
                act,
                eps,
            )?);
            pan_blocks.push(CspRepLayer::from_weights(
                weights,
                &format!("{prefix}.pan_blocks.{i}"),
                3,
                act,
                eps,
            )?);
        }

        Ok(Self {
            aifi,
            encode_proj_layers: cfg.encode_proj_layers.clone(),
            lateral_convs,
            fpn_blocks,
            downsample_convs,
            pan_blocks,
        })
    }

    /// `features`: one NHWC map per level. Returns `num_levels` fused maps.
    pub fn forward(&self, features: Vec<UniquePtr<MlxArray>>) -> Vec<UniquePtr<MlxArray>> {
        let mut feats = features;

        // AIFI on each level in encode_proj_layers.
        for (i, &lvl) in self.encode_proj_layers.iter().enumerate() {
            feats[lvl] = self.aifi[i].forward(&feats[lvl]);
        }

        // Top-down FPN.
        let num_fpn = self.lateral_convs.len();
        let mut fpn: Vec<UniquePtr<MlxArray>> = vec![mlxcel_core::copy(&feats[feats.len() - 1])];
        for idx in 0..num_fpn {
            let backbone_feat = &feats[num_fpn - idx - 1];
            let top_feat = self.lateral_convs[idx].forward(&fpn[fpn.len() - 1]);
            // Overwrite the last FPN entry with the lateral-projected feature.
            let last = fpn.len() - 1;
            fpn[last] = mlxcel_core::copy(&top_feat);
            let up = upsample_nearest_2x(&top_feat);
            let fused = mlxcel_core::concatenate(&up, backbone_feat, 3);
            fpn.push(self.fpn_blocks[idx].forward(&fused));
        }
        fpn.reverse();

        // Bottom-up PAN.
        let num_pan = self.downsample_convs.len();
        let mut pan: Vec<UniquePtr<MlxArray>> = vec![mlxcel_core::copy(&fpn[0])];
        for idx in 0..num_pan {
            let down = self.downsample_convs[idx].forward(&pan[pan.len() - 1]);
            let up = &fpn[idx + 1];
            let fused = mlxcel_core::concatenate(&down, up, 3);
            pan.push(self.pan_blocks[idx].forward(&fused));
        }
        pan
    }
}

/// Load a `LayerNorm` (weight + bias) from `{prefix}.weight` / `{prefix}.bias`.
pub fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = copy_weight(weights, &format!("{prefix}.weight"))?;
    let bias = copy_weight_opt(weights, &format!("{prefix}.bias"));
    Ok(LayerNorm::new(weight, bias, eps))
}
