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

//! Pixtral Vision Encoder
//!
//! Custom ViT with 2D RoPE and SwiGLU MLP for Mistral's Pixtral VLM.
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/pixtral/vision.py
//!
//! Architecture:
//!   patch_conv (Conv2d, stride=patch_size) → ln_pre (RMSNorm)
//!   → N × EncoderLayer (RMSNorm + Attention w/ 2D RoPE + RMSNorm + SwiGLU MLP)
//!
//! Key differences from SigLIP:
//! - Uses RMSNorm instead of LayerNorm
//! - Uses 2D RoPE instead of learned position embeddings
//! - Uses SwiGLU MLP instead of GELU MLP
//! - Attention projections have no bias
//!
//! Used by: Pixtral VLM

use super::{VisionEncoder, VisionEncoderOutput};
use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// 2D Rotary Position Embedding.
/// Precomputed 2D RoPE frequencies for vision patches
///
/// Computes separate frequency components for height and width dimensions,
/// then tiles and concatenates to create 2D position embeddings.
struct PixtralRoPE {
    /// Precomputed inverse frequencies [max_patches², head_dim]
    inv_freq: Vec<f32>,
    #[allow(dead_code)]
    max_patches_per_side: usize,
    head_dim: usize,
}

impl PixtralRoPE {
    fn new(head_dim: usize, rope_theta: f32, max_patches_per_side: usize) -> Self {
        let half_dim = head_dim / 2;
        let quarter_dim = half_dim / 2;

        // freqs = 1.0 / (base ^ (arange(0, dim, 2) / dim))  → [half_dim]
        let mut freqs = vec![0.0f32; half_dim];
        for (i, freq) in freqs.iter_mut().enumerate() {
            *freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
        }

        // freqs_h = outer(arange(max_patches), freqs[::2]) → [max_patches, quarter_dim]
        // freqs_w = outer(arange(max_patches), freqs[1::2]) → [max_patches, quarter_dim]
        let n = max_patches_per_side;
        let mut freqs_h = vec![0.0f32; n * quarter_dim];
        let mut freqs_w = vec![0.0f32; n * quarter_dim];
        for h in 0..n {
            for j in 0..quarter_dim {
                freqs_h[h * quarter_dim + j] = h as f32 * freqs[2 * j]; // even indices
                freqs_w[h * quarter_dim + j] = h as f32 * freqs[2 * j + 1]; // odd indices
            }
        }

        // tile(freqs_h[:, None, :], (1, n, 1)) → [n, n, quarter_dim]
        // tile(freqs_w[None, :, :], (n, 1, 1)) → [n, n, quarter_dim]
        // concat along last axis → [n, n, half_dim]
        // reshape → [n², half_dim]
        let total_patches = n * n;
        let mut inv_freq_half = vec![0.0f32; total_patches * half_dim];
        for h in 0..n {
            for w in 0..n {
                let idx = h * n + w;
                for j in 0..quarter_dim {
                    // freqs_h tiled: freqs_h[h, j] repeated for all w
                    inv_freq_half[idx * half_dim + j] = freqs_h[h * quarter_dim + j];
                    // freqs_w tiled: freqs_w[w, j] repeated for all h
                    inv_freq_half[idx * half_dim + quarter_dim + j] = freqs_w[w * quarter_dim + j];
                }
            }
        }

        // Duplicate: concat(inv_freq_half, inv_freq_half) → [n², head_dim]
        let mut inv_freq = vec![0.0f32; total_patches * head_dim];
        for i in 0..total_patches {
            for j in 0..half_dim {
                inv_freq[i * head_dim + j] = inv_freq_half[i * half_dim + j];
                inv_freq[i * head_dim + half_dim + j] = inv_freq_half[i * half_dim + j];
            }
        }

        Self {
            inv_freq,
            max_patches_per_side,
            head_dim,
        }
    }

    /// Get cos/sin embeddings for given position IDs
    /// Returns (cos, sin) each shaped [num_positions, head_dim]
    fn forward(&self, position_ids: &[i32]) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let num_pos = position_ids.len();
        let dim = self.head_dim;

        let mut cos_data = vec![0.0f32; num_pos * dim];
        let mut sin_data = vec![0.0f32; num_pos * dim];

        for (i, &pos_id) in position_ids.iter().enumerate() {
            let pos = pos_id as usize;
            for j in 0..dim {
                let freq = self.inv_freq[pos * dim + j];
                cos_data[i * dim + j] = freq.cos();
                sin_data[i * dim + j] = freq.sin();
            }
        }

        let cos = mlxcel_core::from_slice_f32(&cos_data, &[num_pos as i32, dim as i32]);
        let sin = mlxcel_core::from_slice_f32(&sin_data, &[num_pos as i32, dim as i32]);
        (cos, sin)
    }
}

/// Compute meshgrid position IDs for patches
/// For image of size (h_patches, w_patches), returns flattened IDs:
///   id[h][w] = h * max_width + w
fn position_ids_in_meshgrid(
    num_patches_h: usize,
    num_patches_w: usize,
    max_width: usize,
) -> Vec<i32> {
    let mut ids = Vec::with_capacity(num_patches_h * num_patches_w);
    for h in 0..num_patches_h {
        for w in 0..num_patches_w {
            ids.push((h * max_width + w) as i32);
        }
    }
    ids
}

// RoPE application (rotate_half pattern).
/// Apply rotary position embeddings to queries and keys
/// q, k: [B, num_heads, L, head_dim]
/// cos, sin: [L, head_dim] → expanded to [1, L, head_dim] for broadcasting
fn apply_rotary_pos_emb(
    q: &MlxArray,
    k: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Expand cos/sin: [L, D] → [1, L, D] for broadcasting with [B, num_heads, L, D]
    let cos_shape = mlxcel_core::array_shape(cos);
    let cos = mlxcel_core::reshape(cos, &[1, cos_shape[0], cos_shape[1]]);
    let sin = mlxcel_core::reshape(sin, &[1, cos_shape[0], cos_shape[1]]);

    // q_embed = q * cos + rotate_half(q) * sin
    let q_cos = mlxcel_core::multiply(q, &cos);
    let q_rot = rotate_half(q);
    let q_sin = mlxcel_core::multiply(&q_rot, &sin);
    let q_embed = mlxcel_core::add(&q_cos, &q_sin);

    let k_cos = mlxcel_core::multiply(k, &cos);
    let k_rot = rotate_half(k);
    let k_sin = mlxcel_core::multiply(&k_rot, &sin);
    let k_embed = mlxcel_core::add(&k_cos, &k_sin);

    (q_embed, k_embed)
}

/// rotate_half: [-x2, x1] where x1 = x[..., :D/2], x2 = x[..., D/2:]
fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len();
    let last_dim = shape[ndim - 1];
    let half = last_dim / 2;

    // Build slice starts/ends for x1 = x[..., :half] and x2 = x[..., half:]
    let start1 = vec![0i32; ndim];
    let mut end1: Vec<i32> = shape.clone();
    end1[ndim - 1] = half;

    let mut start2 = vec![0i32; ndim];
    start2[ndim - 1] = half;
    let end2: Vec<i32> = shape.clone();

    let x1 = mlxcel_core::slice(x, &start1, &end1);
    let x2 = mlxcel_core::slice(x, &start2, &end2);

    // -x2
    let neg_x2 = mlxcel_core::multiply_scalar(&x2, -1.0);

    // concat(-x2, x1, axis=-1)
    mlxcel_core::concatenate(&neg_x2, &x1, ndim as i32 - 1)
}

// Vision Attention (with 2D RoPE).
struct PixtralAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl PixtralAttention {
    fn forward(
        &self,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let queries = self.q_proj.forward(x);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);

        // Reshape [B, L, D] → [B, L, num_heads, head_dim] → [B, num_heads, L, head_dim]
        let queries = mlxcel_core::reshape(&queries, &[b, l, self.num_heads, self.head_dim]);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = mlxcel_core::reshape(&keys, &[b, l, self.num_heads, self.head_dim]);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::reshape(&values, &[b, l, self.num_heads, self.head_dim]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        // Apply 2D RoPE to queries and keys
        let (queries, keys) = apply_rotary_pos_emb(&queries, &keys, cos, sin);

        // Scaled dot product attention
        let mask_ptr = mask
            .map(|m| m as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &queries, &keys, &values, self.scale, mask_ptr, 0.0, 0,
            )
        };

        // Transpose back and reshape: [B, num_heads, L, head_dim] → [B, L, D]
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&output)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: usize,
        dims: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = dims / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads as i32,
            head_dim: head_dim as i32,
            scale,
        })
    }
}

// Vision MLP (SwiGLU).
struct PixtralMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl PixtralMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = self.gate_proj.forward(x);
        let gate = mlxcel_core::silu(&gate);
        let up = self.up_proj.forward(x);
        let hidden = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&hidden)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let gate_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?;
        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

// Encoder Layer.
struct PixtralEncoderLayer {
    attention: PixtralAttention,
    feed_forward: PixtralMLP,
    attention_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

impl PixtralEncoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // x += attention(attention_norm(x), cos, sin, mask)
        let r = self
            .attention
            .forward(&self.attention_norm.forward(x), cos, sin, mask);
        let h = mlxcel_core::add(x, &r);

        // x += feed_forward(ffn_norm(x))
        let r = self.feed_forward.forward(&self.ffn_norm.forward(&h));
        mlxcel_core::add(&h, &r)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
        rms_norm_eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let attention = PixtralAttention::from_weights(
            weights,
            &format!("{}.attention", prefix),
            num_heads,
            hidden_size,
            group_size,
            bits,
        )?;

        let feed_forward = PixtralMLP::from_weights(
            weights,
            &format!("{}.feed_forward", prefix),
            group_size,
            bits,
        )?;

        let attention_norm =
            load_rms_norm(weights, &format!("{}.attention_norm", prefix), rms_norm_eps)?;
        let ffn_norm = load_rms_norm(weights, &format!("{}.ffn_norm", prefix), rms_norm_eps)?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }
}

// Pixtral Vision Model.
/// Pixtral vision encoder config (parsed from vision_config in config.json)
pub struct PixtralVisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl PixtralVisionConfig {
    pub fn from_json(v: &serde_json::Value) -> Self {
        Self {
            hidden_size: v
                .get("hidden_size")
                .and_then(|x| x.as_u64())
                .unwrap_or(1024) as usize,
            num_hidden_layers: v
                .get("num_hidden_layers")
                .and_then(|x| x.as_u64())
                .unwrap_or(24) as usize,
            num_attention_heads: v
                .get("num_attention_heads")
                .and_then(|x| x.as_u64())
                .unwrap_or(16) as usize,
            head_dim: v.get("head_dim").and_then(|x| x.as_u64()).unwrap_or(64) as usize,
            image_size: v.get("image_size").and_then(|x| x.as_u64()).unwrap_or(336) as usize,
            patch_size: v.get("patch_size").and_then(|x| x.as_u64()).unwrap_or(14) as usize,
            rope_theta: v
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .unwrap_or(10000.0) as f32,
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-5) as f32,
        }
    }
}

/// Pixtral Vision Encoder (ViT with 2D RoPE + SwiGLU)
pub struct PixtralVisionModel {
    patch_conv_weight: UniquePtr<MlxArray>,
    patch_size: usize,
    ln_pre: RMSNorm,
    layers: Vec<PixtralEncoderLayer>,
    rope: PixtralRoPE,
    max_patches_per_side: usize,
    /// Which layer's hidden states to return (-1 = last, -2 = second-to-last)
    vision_feature_layer: i32,
}

impl PixtralVisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &PixtralVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = 64;
        let bits = 4;

        // Load patch_conv weight
        let conv_key = format!("{}.patch_conv.weight", prefix);
        let mut patch_conv_weight = weights
            .get(&conv_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", conv_key))?;

        // Sanitize conv2d weights: PyTorch [out, in, kH, kW] → MLX [out, kH, kW, in]
        let w_shape = mlxcel_core::array_shape(&patch_conv_weight);
        if w_shape.len() == 4 {
            let (out_ch, dim1, dim2, _dim3) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
            if !(out_ch >= dim1 && out_ch >= dim2 && dim1 == dim2) {
                patch_conv_weight = mlxcel_core::transpose_axes(&patch_conv_weight, &[0, 2, 3, 1]);
            }
        }

        // Load ln_pre (RMSNorm)
        let ln_pre = load_rms_norm(weights, &format!("{}.ln_pre", prefix), config.rms_norm_eps)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("{}.transformer.layers.{}", prefix, i);
            let layer = PixtralEncoderLayer::from_weights(
                weights,
                &layer_prefix,
                config.hidden_size,
                config.num_attention_heads,
                config.rms_norm_eps,
                group_size,
                bits,
            )?;
            layers.push(layer);
        }

        // Build 2D RoPE
        let max_patches_per_side = config.image_size / config.patch_size;
        let rope = PixtralRoPE::new(config.head_dim, config.rope_theta, max_patches_per_side);

        Ok(Self {
            patch_conv_weight,
            patch_size: config.patch_size,
            ln_pre,
            layers,
            rope,
            max_patches_per_side,
            vision_feature_layer: -1, // default: use last layer
        })
    }

    /// Configure which layer's hidden states to return
    pub fn with_feature_layer(mut self, layer: i32) -> Self {
        self.vision_feature_layer = layer;
        self
    }
}

impl VisionEncoder for PixtralVisionModel {
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput {
        // pixel_values: [B, H, W, C] (channels-last, already transposed by VisionModule)
        let pv_shape = mlxcel_core::array_shape(pixel_values);
        let batch_size = pv_shape[0] as usize;
        let img_h = pv_shape[1] as usize;
        let img_w = pv_shape[2] as usize;

        // Conv2d patch embedding: [B, H, W, C] → [B, H/P, W/P, hidden]
        let patch_emb = mlxcel_core::conv2d(
            pixel_values,
            &self.patch_conv_weight,
            self.patch_size as i32,
            self.patch_size as i32,
            0,
            0,
            1,
            1,
            1,
        );

        // Compute actual patch dimensions
        let patches_h = img_h / self.patch_size;
        let patches_w = img_w / self.patch_size;
        let num_patches = patches_h * patches_w;

        // Flatten spatial dims and concatenate across batch:
        // [B, pH, pW, hidden] → [1, B*num_patches, hidden]
        let emb_shape = mlxcel_core::array_shape(&patch_emb);
        let hidden = emb_shape[3];
        let total_patches = batch_size * num_patches;
        let patch_emb = mlxcel_core::reshape(&patch_emb, &[1, total_patches as i32, hidden]);

        // Pre-norm
        let patch_emb = self.ln_pre.forward(&patch_emb);

        // Compute position IDs via meshgrid (per-image, then concatenated)
        let mut all_position_ids = Vec::with_capacity(total_patches);
        for _ in 0..batch_size {
            let ids = position_ids_in_meshgrid(patches_h, patches_w, self.max_patches_per_side);
            all_position_ids.extend(ids);
        }

        // Get cos/sin from 2D RoPE
        let (cos, sin) = self.rope.forward(&all_position_ids);

        // Generate block attention mask for multi-image batches
        let mask = if batch_size > 1 {
            Some(generate_block_attention_mask(
                &vec![num_patches; batch_size],
                total_patches,
            ))
        } else {
            None
        };

        // Run through encoder layers, collecting hidden states for feature selection
        let num_layers = self.layers.len() as i32;
        let target_layer = if self.vision_feature_layer < 0 {
            (num_layers + self.vision_feature_layer) as usize
        } else {
            self.vision_feature_layer as usize
        };

        let mut h = patch_emb;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos, &sin, mask.as_deref());
            if i == target_layer && target_layer < self.layers.len() - 1 {
                // Return hidden states from this intermediate layer
                return VisionEncoderOutput { hidden_states: h };
            }
        }

        // Default: return last layer output
        VisionEncoderOutput { hidden_states: h }
    }
}

// Block attention mask for multi-image batches.
/// Generate block-diagonal attention mask for multi-image batches
/// Each image only attends to patches from the same image.
/// Returns [1, 1, total_patches, total_patches]
fn generate_block_attention_mask(
    patch_counts: &[usize],
    total_patches: usize,
) -> UniquePtr<MlxArray> {
    let d_min = -1e9f32;
    let mut mask_data = vec![d_min; total_patches * total_patches];

    let mut offset = 0;
    for &count in patch_counts {
        for i in 0..count {
            for j in 0..count {
                mask_data[(offset + i) * total_patches + (offset + j)] = 0.0;
            }
        }
        offset += count;
    }

    mlxcel_core::from_slice_f32(
        &mask_data,
        &[1, 1, total_patches as i32, total_patches as i32],
    )
}

// Helper: load RMSNorm from weights.
fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    let weight_key = format!("{}.weight", prefix);
    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", weight_key))?;
    Ok(RMSNorm::new(weight, eps))
}

/// Sanitize weight keys for Pixtral models.
/// Handles alternative naming conventions from different model sources:
/// - `vision_tower.transformer.*` → `vision_tower.vision_model.transformer.*`
/// - `vision_tower.patch_conv.*` → `vision_tower.vision_model.patch_conv.*`
/// - `vision_tower.ln_pre.*` → `vision_tower.vision_model.ln_pre.*`
/// - `model.vision_encoder.*` → `vision_tower.vision_model.*`
pub fn sanitize_pixtral_weights(weights: &mut WeightMap) {
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        let new_key = if key.contains("vision_tower") && !key.contains("vision_model") {
            if key.contains("transformer") || key.contains("patch_conv") || key.contains("ln_pre") {
                key.replace("vision_tower", "vision_tower.vision_model")
            } else {
                continue;
            }
        } else if key.contains("vision_encoder") && !key.contains("vision_tower") {
            if key.contains("transformer") || key.contains("patch_conv") || key.contains("ln_pre") {
                key.replace("model.vision_encoder", "vision_tower.vision_model")
            } else {
                continue;
            }
        } else {
            continue;
        };

        if let Some(value) = weights.remove(&key) {
            weights.insert(new_key, value);
        }
    }
}
