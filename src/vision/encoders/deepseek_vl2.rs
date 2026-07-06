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

//! SigLIP-style ViT vision tower for DeepSeek-VL2 (`vision.vision_tower.*`).
//!
//! A no-CLS ViT run per tile (batch = tiles): a `patch_size` Conv2d patch embed,
//! a learned absolute `pos_embed`, `layers` pre-norm blocks (fused-qkv full
//! attention with no mask, GELU-tanh MLP), and a trailing LayerNorm. The output
//! `(tiles, N, width)` feeds the downsample projector. The checkpoint's
//! `attn_pool.*` head is unused and dropped at load.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseek_vl_v2/` (SigLIP `VisionTransformer`).
//! Layout: activations channels-last; linear weights `(out, in)`; the patch conv
//! weight is normalized to a 2-D `(width, C*p*p)` linear at load.

use mlxcel_core::layers::{LayerNorm, Linear, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

fn default_layers() -> usize {
    27
}
fn default_width() -> i32 {
    1152
}
fn default_heads() -> i32 {
    16
}
fn default_intermediate() -> i32 {
    4304
}
fn default_patch_size() -> i32 {
    14
}
fn default_image_size() -> i32 {
    384
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekVl2VisionConfig {
    #[serde(default = "default_layers")]
    pub layers: usize,
    #[serde(default = "default_width")]
    pub width: i32,
    #[serde(default = "default_heads", alias = "heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: i32,
    #[serde(default = "default_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_image_size")]
    pub image_size: i32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default)]
    pub quant_group_size: i32,
    #[serde(default)]
    pub quant_bits: i32,
}

fn get(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("DeepSeek-VL2 vision weight missing: {name}"))
}

fn load_ln(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    Ok(LayerNorm::new(
        get(weights, &format!("{prefix}.weight"))?,
        weights
            .get(&format!("{prefix}.bias"))
            .map(|w| mlxcel_core::copy(w)),
        eps,
    ))
}

struct Attention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    /// `x`: `(tiles, N, width)` -> `(tiles, N, width)`, full attention (no mask).
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let (b, n) = (s[0], s[1]);
        let (heads, hd) = (self.num_heads, self.head_dim);
        let qkv = self.qkv.forward(x); // (b, N, 3*width)
        let split = |i: i32| {
            let sl =
                mlxcel_core::slice(&qkv, &[0, 0, i * heads * hd], &[b, n, (i + 1) * heads * hd]);
            let sl = mlxcel_core::reshape(&sl, &[b, n, heads, hd]);
            mlxcel_core::transpose_axes(&sl, &[0, 2, 1, 3]) // (b, heads, N, hd)
        };
        let q = split(0);
        let k = split(1);
        let v = split(2);
        // SAFETY: q/k/v valid; null mask (bidirectional).
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, n, heads * hd]);
        self.proj.forward(&out)
    }
}

struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl Block {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = self.attn.forward(&self.norm1.forward(x));
        let x = mlxcel_core::add(x, &y);
        let y = self.fc1.forward(&self.norm2.forward(&x));
        let y = mlxcel_core::gelu_approx(&y); // GELU tanh approximation
        let y = self.fc2.forward(&y);
        mlxcel_core::add(&x, &y)
    }
}

pub struct DeepSeekVl2VisionEncoder {
    patch_embed: Linear,            // (width, C*p*p) with bias
    pos_embed: UniquePtr<MlxArray>, // (1, N, width)
    blocks: Vec<Block>,
    norm: LayerNorm,
    patch_size: i32,
    /// Weight dtype code; pixel activations are cast to it before the patch
    /// linear so a bf16 pixel batch meets f16 (or bf16) tower weights.
    dtype: i32,
}

impl DeepSeekVl2VisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &DeepSeekVl2VisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let (gs, bits) = (config.quant_group_size, config.quant_bits);
        let head_dim = config.width / config.num_attention_heads;

        // Patch conv normalized to a 2-D (width, C*p*p) linear with column order
        // (c, dy, dx): the 4-bit export ships OHWI (width, p, p, C); OIHW and
        // pre-flattened layouts are also accepted.
        let raw = get(weights, &format!("{prefix}.patch_embed.proj.weight"))?;
        let shape = mlxcel_core::array_shape(&raw);
        let (c, p, out) = (3, config.patch_size, config.width);
        let patch_w = match shape.as_slice() {
            [_, _] => raw,
            [_, a, _, _] if *a == c => mlxcel_core::reshape(&raw, &[out, c * p * p]),
            [_, _, _, a] if *a == c => {
                let t = mlxcel_core::transpose_axes(&raw, &[0, 3, 1, 2]);
                mlxcel_core::reshape(&t, &[out, c * p * p])
            }
            _ => return Err(format!("unexpected patch proj shape {shape:?}")),
        };
        let patch_b = get(weights, &format!("{prefix}.patch_embed.proj.bias"))?;
        let dtype = mlxcel_core::array_dtype(&patch_w);
        let patch_embed = Linear::new(patch_w, Some(patch_b));
        let pos_embed = get(weights, &format!("{prefix}.pos_embed"))?;

        let mut blocks = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let bp = format!("{prefix}.blocks.{i}");
            blocks.push(Block {
                norm1: load_ln(weights, &format!("{bp}.norm1"), config.layer_norm_eps)?,
                attn: Attention {
                    qkv: UnifiedLinear::from_weights(weights, &format!("{bp}.attn.qkv"), gs, bits)?,
                    proj: UnifiedLinear::from_weights(
                        weights,
                        &format!("{bp}.attn.proj"),
                        gs,
                        bits,
                    )?,
                    num_heads: config.num_attention_heads,
                    head_dim,
                    scale: (head_dim as f32).powf(-0.5),
                },
                norm2: load_ln(weights, &format!("{bp}.norm2"), config.layer_norm_eps)?,
                fc1: UnifiedLinear::from_weights(weights, &format!("{bp}.mlp.fc1"), gs, bits)?,
                fc2: UnifiedLinear::from_weights(weights, &format!("{bp}.mlp.fc2"), gs, bits)?,
            });
        }

        // The final norm uses the standard 1e-5 LayerNorm epsilon.
        let norm = load_ln(weights, &format!("{prefix}.norm"), 1e-5)?;

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            norm,
            patch_size: config.patch_size,
            dtype,
        })
    }

    /// `pixel_values`: channels-last `(tiles, img, img, 3)`. Returns
    /// `(tiles, N, width)` where `N` is the patch count.
    pub fn forward(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(pixel_values);
        let (tiles, img) = (s[0], s[1]);
        let p = self.patch_size;
        let g = (img - p) / p + 1; // conv stride p, no padding
        let n = g * g;

        // Patchify: extract (c, dy, dx)-ordered patch vectors, then linear.
        let feat = 3 * p * p;
        // Reshape image into non-overlapping patches:
        // (tiles, g, p, g, p, 3) -> (tiles, g, g, 3, p, p) -> (tiles*n, feat).
        let crop = mlxcel_core::slice(pixel_values, &[0, 0, 0, 0], &[tiles, g * p, g * p, 3]);
        let x = mlxcel_core::reshape(&crop, &[tiles, g, p, g, p, 3]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 5, 2, 4]); // (tiles, g, g, 3, p, p)
        let x = mlxcel_core::reshape(&x, &[tiles * n, feat]);
        // Match the tower weight dtype (pixels arrive bf16; weights may be f16).
        let x = mlxcel_core::astype(&x, self.dtype);
        let embeds = self.patch_embed.forward(&x);
        let mut h = mlxcel_core::reshape(&embeds, &[tiles, n, -1]);

        let pos = mlxcel_core::astype(&self.pos_embed, mlxcel_core::array_dtype(&h));
        h = mlxcel_core::add(&h, &pos);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.norm.forward(&h)
    }
}
