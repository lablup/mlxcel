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

//! InternViT Vision Encoder (InternVL family).
//!
//! Faithful port of
//! `references/mlx-vlm/mlx_vlm/models/internvl_chat/vision.py`.
//!
//! Architecture (per `vision_model.*` in `internvl3-1b`):
//! - `VisionEmbeddings`: Conv2d patch embedding (3 -> hidden, k=14, s=14) +
//!   learned `class_embedding` (CLS token at position 0) + learned
//!   `position_embedding`.
//! - `EncoderLayer` x depth:
//!   `x + ls1 * attn(norm1(x))`, then `x + ls2 * mlp(norm2(x))`.
//!   - `attn`: fused `qkv` (3*hidden, with bias) + `proj` (with bias),
//!     standard scaled-dot-product attention. `qk_normalization=false`
//!     for InternVL3 so q/k norms are skipped.
//!   - `mlp`: `fc1` -> GELU(precise) -> `fc2` (all with bias).
//!   - `norm1`/`norm2`: LayerNorm (weight + bias).
//!   - `ls1`/`ls2`: per-channel layer-scale vectors multiplied into the
//!     attention / MLP residual branches.
//!
//! The encoder returns the **last** hidden state with the CLS token still
//! attached (`[B, 1 + num_patches, hidden]`). The caller (`InternVLChatVLM`)
//! strips the CLS token before the connector, matching upstream
//! `hidden_states[:, 1:, :]` with `select_layer = -1`.
//!
//! Precision: the InternViT tower is non-quantized bf16 in the released
//! checkpoint. On Apple Silicon the loader converts these tensors to f16
//! (no Apple GPU has a native bf16 ALU; M5 additionally JIT-crashes on
//! bf16). The conversion happens at weight-load time, so this module treats
//! the weights as plain dense `Linear`/`LayerNorm` tensors.
//!
//! Used by: InternVL (internvl_chat) VLM.

use super::{VisionEncoder, VisionEncoderOutput};
use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// InternViT vision encoder configuration (parsed from `vision_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct InternVitConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// `false` for InternVL3 — q/k RMSNorm is skipped. Parsed so future
    /// InternViT-6B checkpoints (which enable it) can branch on it.
    #[serde(default)]
    pub qk_normalization: bool,
}

fn default_hidden_size() -> usize {
    1024
}
fn default_intermediate_size() -> usize {
    4096
}
fn default_num_hidden_layers() -> usize {
    24
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_patch_size() -> usize {
    14
}
fn default_image_size() -> usize {
    448
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}

impl InternVitConfig {
    /// Number of patches per 448x448 tile (`(image_size / patch_size)^2`).
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }
}

// Helper: load a LayerNorm (weight + optional bias) from the weight map.
fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, eps))
}

// InternViT fused-QKV attention (standard SDPA, no vision RoPE).
struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &InternVitConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let qkv = Linear::from_weights(weights, &format!("{prefix}.attn.qkv"))?;
        let proj = Linear::from_weights(weights, &format!("{prefix}.attn.proj"))?;
        let head_dim = (config.hidden_size / config.num_attention_heads) as i32;
        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[B, L, C]` -> `[B, L, C]`.
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // qkv: [B, L, 3*C] -> [B, L, 3, num_heads, head_dim] -> [3, B, num_heads, L, head_dim]
        let qkv = self.qkv.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[b, l, 3, self.num_heads, self.head_dim]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[2, 0, 3, 1, 4]);

        let q = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, 0, 0],
            &[1, b, self.num_heads, l, self.head_dim],
        );
        let k = mlxcel_core::slice(
            &qkv,
            &[1, 0, 0, 0, 0],
            &[2, b, self.num_heads, l, self.head_dim],
        );
        let v = mlxcel_core::slice(
            &qkv,
            &[2, 0, 0, 0, 0],
            &[3, b, self.num_heads, l, self.head_dim],
        );
        let q = mlxcel_core::squeeze_axis(&q, 0);
        let k = mlxcel_core::squeeze_axis(&k, 0);
        let v = mlxcel_core::squeeze_axis(&v, 0);

        // Standard scaled-dot-product attention (no mask for a vision tower).
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q,
                &k,
                &v,
                self.scale,
                std::ptr::null(),
                0.0,
                0,
            )
        };

        // [B, num_heads, L, head_dim] -> [B, L, C]
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, self.num_heads * self.head_dim]);
        self.proj.forward(&output)
    }
}

// InternViT MLP: fc1 -> GELU(precise) -> fc2.
struct VisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMLP {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            fc1: Linear::from_weights(weights, &format!("{prefix}.mlp.fc1"))?,
            fc2: Linear::from_weights(weights, &format!("{prefix}.mlp.fc2"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(x);
        // GELU(approx="precise") in the Python reference; `gelu_approx` is
        // mlxcel's tanh-free precise GELU (matches the SigLIP encoder).
        let h = mlxcel_core::gelu_approx(&h);
        self.fc2.forward(&h)
    }
}

// InternViT encoder layer with layer-scale (ls1/ls2) on each residual branch.
struct EncoderLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMLP,
    ls1: UniquePtr<MlxArray>,
    ls2: UniquePtr<MlxArray>,
}

impl EncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &InternVitConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let ls1 = weights
            .get(&format!("{prefix}.ls1"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.ls1"))?;
        let ls2 = weights
            .get(&format!("{prefix}.ls2"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.ls2"))?;

        Ok(Self {
            norm1: load_layer_norm(weights, &format!("{prefix}.norm1"), config.layer_norm_eps)?,
            norm2: load_layer_norm(weights, &format!("{prefix}.norm2"), config.layer_norm_eps)?,
            attn: VisionAttention::from_weights(weights, config, prefix)?,
            mlp: VisionMLP::from_weights(weights, prefix)?,
            ls1,
            ls2,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // x = x + ls1 * attn(norm1(x))
        let normed = self.norm1.forward(x);
        let attn_out = self.attn.forward(&normed);
        let scaled = mlxcel_core::multiply(&attn_out, &self.ls1);
        let x = mlxcel_core::add(x, &scaled);

        // x = x + ls2 * mlp(norm2(x))
        let normed = self.norm2.forward(&x);
        let mlp_out = self.mlp.forward(&normed);
        let scaled = mlxcel_core::multiply(&mlp_out, &self.ls2);
        mlxcel_core::add(&x, &scaled)
    }
}

// InternViT embeddings: Conv2d patch embed + CLS token + position embedding.
struct VisionEmbeddings {
    patch_weight: UniquePtr<MlxArray>,
    patch_bias: Option<UniquePtr<MlxArray>>,
    class_embedding: UniquePtr<MlxArray>,
    position_embedding: UniquePtr<MlxArray>,
    num_patches: usize,
    patch_size: usize,
    hidden_size: usize,
}

impl VisionEmbeddings {
    fn from_weights(
        weights: &WeightMap,
        config: &InternVitConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight_key = format!("{prefix}.patch_embedding.weight");
        let mut patch_weight = weights
            .get(&weight_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {weight_key}"))?;

        // PyTorch conv2d weight: [out, in, kH, kW] -> MLX conv2d expects
        // [out, kH, kW, in]. Mirror the reference `sanitize`'s shape check
        // (the released checkpoint stores the PyTorch layout) and only
        // transpose when the layout is not already MLX-shaped.
        let w_shape = mlxcel_core::array_shape(&patch_weight);
        if w_shape.len() == 4 {
            let (out_ch, dim1, dim2, _dim3) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
            if !(out_ch >= dim1 && out_ch >= dim2 && dim1 == dim2) {
                patch_weight = mlxcel_core::transpose_axes(&patch_weight, &[0, 2, 3, 1]);
            }
        }

        let patch_bias = weights
            .get(&format!("{prefix}.patch_embedding.bias"))
            .map(|b| mlxcel_core::copy(b));

        let class_embedding = weights
            .get(&format!("{prefix}.class_embedding"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.class_embedding"))?;
        let position_embedding = weights
            .get(&format!("{prefix}.position_embedding"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.position_embedding"))?;

        Ok(Self {
            patch_weight,
            patch_bias,
            class_embedding,
            position_embedding,
            num_patches: config.num_patches(),
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
        })
    }

    /// `x`: channels-last `[B, H, W, C]` -> `[B, 1 + num_patches, hidden]`.
    ///
    /// Every InternVL3 tile is exactly `image_size x image_size`, so the
    /// patch grid matches the trained `position_embedding` grid and the
    /// upstream `interpolate` is an identity — we add `position_embedding`
    /// directly and skip bicubic interpolation.
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let patch = mlxcel_core::conv2d(
            x,
            &self.patch_weight,
            self.patch_size as i32,
            self.patch_size as i32,
            0,
            0,
            1,
            1,
            1,
        );
        let patch = match &self.patch_bias {
            Some(bias) => mlxcel_core::add(&patch, bias),
            None => patch,
        };

        // [B, H/P, W/P, hidden] -> [B, num_patches, hidden]
        let shape = mlxcel_core::array_shape(&patch);
        let b = shape[0];
        let hidden = self.hidden_size as i32;
        let patch = mlxcel_core::reshape(&patch, &[b, self.num_patches as i32, hidden]);

        // Prepend CLS token: class_embedding is [1, 1, hidden].
        let cls = mlxcel_core::reshape(&self.class_embedding, &[1, 1, hidden]);
        let cls = mlxcel_core::broadcast_to(&cls, &[b, 1, hidden]);
        let embeddings = mlxcel_core::concatenate(&cls, &patch, 1);

        // Add position embedding ([1, 1 + num_patches, hidden], broadcasts on B).
        mlxcel_core::add(&embeddings, &self.position_embedding)
    }
}

/// InternViT vision tower.
pub struct InternVitVisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<EncoderLayer>,
}

impl InternVitVisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &InternVitConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let embeddings =
            VisionEmbeddings::from_weights(weights, config, &format!("{prefix}.embeddings"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(EncoderLayer::from_weights(
                weights,
                config,
                &format!("{prefix}.encoder.layers.{i}"),
            )?);
        }

        Ok(Self { embeddings, layers })
    }
}

impl VisionEncoder for InternVitVisionModel {
    /// `pixel_values`: channels-last `[num_tiles, H, W, C]`.
    ///
    /// Returns the **last** hidden state with the CLS token still attached
    /// (`select_layer = -1`, no post-layernorm in InternViT). The caller
    /// strips the CLS token (`[:, 1:, :]`) before the connector.
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput {
        let mut h = self.embeddings.forward(pixel_values);
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        VisionEncoderOutput { hidden_states: h }
    }
}
