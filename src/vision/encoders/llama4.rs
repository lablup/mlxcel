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

//! Llama 4 Vision Encoder
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llama4/vision.py
//!
//! Architecture:
//! - UnfoldConvolution (im2col patch embedding, not Conv2d)
//! - 2D Vision RoPE (coordinate-based cos/sin rotation)
//! - VisionAttention with RoPE (full attention, no masking)
//! - VisionMLP (GELU fast)
//! - VisionEncoderLayer (pre-norm with LayerNorm)
//! - PixelShuffleMLP (vision adapter: 6400→1600 patches)
//! - VisionModel: full pipeline with CLS token and positional embeddings
//!
//! Used by: Llama 4 Vision

use super::{VisionEncoder, VisionEncoderOutput};
use mlxcel_core::layers::{LayerNorm, Linear, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Llama4VisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_pixel_shuffle_ratio")]
    pub pixel_shuffle_ratio: f32,
    #[serde(default)]
    pub projector_input_dim: usize,
    #[serde(default)]
    pub projector_output_dim: usize,
    #[serde(default)]
    pub vision_output_dim: usize,
}

fn default_num_channels() -> usize {
    3
}
fn default_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_pixel_shuffle_ratio() -> f32 {
    0.5
}

// UnfoldConvolution (im2col patch embedding).
/// im2col-based patch embedding: extract non-overlapping patches + linear projection
///
/// Input [B, C, H, W] -> extract patches -> [B, num_patches, C*P*P] -> linear -> [B, num_patches, hidden]
struct UnfoldConvolution {
    linear: Linear,
    patch_size: usize,
}

impl UnfoldConvolution {
    fn from_weights(weights: &WeightMap, prefix: &str, patch_size: usize) -> Result<Self, String> {
        let linear = Linear::from_weights(weights, &format!("{}.linear", prefix))?;
        Ok(Self { linear, patch_size })
    }

    /// Extract non-overlapping patches using efficient reshape
    /// [B, C, H, W] -> [B, num_patches, C*P*P] -> linear -> [B, num_patches, hidden]
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let p = self.patch_size as i32;
        let h_patches = h / p;
        let w_patches = w / p;

        // [B, C, H, W] -> [B, C, H/P, P, W/P, P]
        let x = mlxcel_core::reshape(x, &[b, c, h_patches, p, w_patches, p]);
        // -> [B, H/P, W/P, C, P, P]
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 4, 1, 3, 5]);
        // -> [B, num_patches, C*P*P]
        let num_patches = h_patches * w_patches;
        let patch_dim = c * p * p;
        let x = mlxcel_core::reshape(&x, &[b, num_patches, patch_dim]);

        // Linear projection
        self.linear.forward(&x)
    }
}

// Vision RoPE (2D coordinate-based).
/// Precomputed 2D rotary position embeddings for vision encoder
///
/// Creates cos/sin frequency pairs from 2D (x, y) patch coordinates.
/// CLS token gets zero frequencies.
struct VisionRotaryEmbedding {
    /// Complex frequency tensor as real pairs: [num_patches+1, 1, head_dim/2] complex
    /// Stored as two arrays: cos and sin components
    freqs_cos: UniquePtr<MlxArray>,
    freqs_sin: UniquePtr<MlxArray>,
}

impl VisionRotaryEmbedding {
    fn new(config: &Llama4VisionConfig) -> Self {
        let idx = (config.image_size / config.patch_size) as i32; // e.g., 80
        let num_patches = idx * idx; // 6400
        let num_positions = num_patches + 1; // +1 for CLS

        // freq_dim = head_dim / 2 = (hidden_size / num_heads) / 2
        let head_dim = config.hidden_size / config.num_attention_heads;
        let freq_dim = head_dim / 2; // 20 for head_dim=80... wait

        // From Python: freq_dim = hidden_size // num_attention_heads // 2
        // For hidden=1280, heads=16: head_dim=80, freq_dim=40
        // Then: rope_freq = 1.0 / (theta ** (arange(0, freq_dim, 2)[:freq_dim//2] / freq_dim))
        // So we take freq_dim//2 = 20 frequency values
        let half_freq_dim = freq_dim / 2; // 20

        // Build 2D coordinates: patch indices -> (x, y)
        // img_idx: [0, 1, ..., num_patches-1], CLS gets -2
        let mut frequencies_x = Vec::with_capacity(num_positions as usize);
        let mut frequencies_y = Vec::with_capacity(num_positions as usize);
        for i in 0..num_patches {
            frequencies_x.push((i % idx + 1) as f32); // +1 as in Python
            frequencies_y.push((i / idx + 1) as f32);
        }
        // CLS token: gets zero frequency (masked to 0 later)
        frequencies_x.push(0.0);
        frequencies_y.push(0.0);

        // Compute rope frequencies: 1.0 / (theta ** (i / freq_dim))
        // where i = 0, 2, 4, ..., up to half_freq_dim values
        let mut rope_freq = Vec::with_capacity(half_freq_dim);
        for i in 0..half_freq_dim {
            let exp = (2 * i) as f32 / freq_dim as f32;
            rope_freq.push(1.0 / config.rope_theta.powf(exp));
        }

        // Compute freqs for each position
        // freqs_x[pos, :] = frequencies_x[pos] * rope_freq[:]  -> interleaved double
        // freqs_y[pos, :] = frequencies_y[pos] * rope_freq[:]  -> interleaved double
        // Then concatenate x and y, take every other -> freqs

        // Python does: freqs_x = repeat_interleave(freqs_x_expanded, 2) -> doubles each freq
        // freqs_y = repeat_interleave(freqs_y_expanded, 2)
        // freqs = concat([freqs_x, freqs_y], -1)[..., ::2]  -> takes every other
        // This is equivalent to: freqs = concat([freqs_x_expanded, freqs_y_expanded], -1)
        // Because repeat_interleave(x,2) then ::2 = x

        // So final freqs shape: [num_positions, 1, half_freq_dim*2] = [num_positions, 1, freq_dim]
        // freq_dim = head_dim/2 = 40

        // Mask: CLS token (last position) gets zero
        // Then cos/sin of freqs

        let mut cos_data = Vec::with_capacity(num_positions as usize * freq_dim);
        let mut sin_data = Vec::with_capacity(num_positions as usize * freq_dim);

        for pos in 0..num_positions as usize {
            let is_cls = pos == (num_positions - 1) as usize;
            // First half: x frequencies
            for &freq in &rope_freq[..half_freq_dim] {
                let val = if is_cls {
                    0.0
                } else {
                    frequencies_x[pos] * freq
                };
                cos_data.push(val.cos());
                sin_data.push(val.sin());
            }
            // Second half: y frequencies
            for &freq in &rope_freq[..half_freq_dim] {
                let val = if is_cls {
                    0.0
                } else {
                    frequencies_y[pos] * freq
                };
                cos_data.push(val.cos());
                sin_data.push(val.sin());
            }
        }

        // Shape: [num_positions, 1, freq_dim]
        let freqs_cos =
            mlxcel_core::from_slice_f32(&cos_data, &[num_positions, 1, freq_dim as i32]);
        let freqs_sin =
            mlxcel_core::from_slice_f32(&sin_data, &[num_positions, 1, freq_dim as i32]);

        Self {
            freqs_cos,
            freqs_sin,
        }
    }
}

/// Apply vision rotary embeddings to query and key tensors
///
/// q/k: [B, seq, num_heads, head_dim]
/// Rotation uses even/odd pair splitting (complex multiply pattern)
fn apply_vision_rope(
    q: &MlxArray,
    k: &MlxArray,
    freqs_cos: &MlxArray,
    freqs_sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // q shape: [B, seq, num_heads, head_dim]
    let q_shape = mlxcel_core::array_shape(q);
    let head_dim = q_shape[3];

    // Reshape to pairs: [B, seq, num_heads, head_dim/2, 2]
    let q_reshaped =
        mlxcel_core::reshape(q, &[q_shape[0], q_shape[1], q_shape[2], head_dim / 2, 2]);
    let k_shape = mlxcel_core::array_shape(k);
    let k_reshaped =
        mlxcel_core::reshape(k, &[k_shape[0], k_shape[1], k_shape[2], head_dim / 2, 2]);

    // Cast to float32 for precision
    let q_f32 = mlxcel_core::astype(&q_reshaped, mlxcel_core::dtype::FLOAT32);
    let k_f32 = mlxcel_core::astype(&k_reshaped, mlxcel_core::dtype::FLOAT32);

    // Extract even/odd: [..., 0] and [..., 1]
    let q_even = mlxcel_core::slice(
        &q_f32,
        &[0, 0, 0, 0, 0],
        &[q_shape[0], q_shape[1], q_shape[2], head_dim / 2, 1],
    );
    let q_odd = mlxcel_core::slice(
        &q_f32,
        &[0, 0, 0, 0, 1],
        &[q_shape[0], q_shape[1], q_shape[2], head_dim / 2, 2],
    );

    let k_even = mlxcel_core::slice(
        &k_f32,
        &[0, 0, 0, 0, 0],
        &[k_shape[0], k_shape[1], k_shape[2], head_dim / 2, 1],
    );
    let k_odd = mlxcel_core::slice(
        &k_f32,
        &[0, 0, 0, 0, 1],
        &[k_shape[0], k_shape[1], k_shape[2], head_dim / 2, 2],
    );

    // Squeeze last dim for broadcasting
    let q_even = mlxcel_core::reshape(&q_even, &[q_shape[0], q_shape[1], q_shape[2], head_dim / 2]);
    let q_odd = mlxcel_core::reshape(&q_odd, &[q_shape[0], q_shape[1], q_shape[2], head_dim / 2]);
    let k_even = mlxcel_core::reshape(&k_even, &[k_shape[0], k_shape[1], k_shape[2], head_dim / 2]);
    let k_odd = mlxcel_core::reshape(&k_odd, &[k_shape[0], k_shape[1], k_shape[2], head_dim / 2]);

    // freqs_cos/sin: [seq, 1, head_dim/2] - broadcast to [B, seq, num_heads, head_dim/2]
    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // q_rotated_even = q_even * cos - q_odd * sin
    // q_rotated_odd = q_even * sin + q_odd * cos
    let q_rot_even = mlxcel_core::subtract(
        &mlxcel_core::multiply(&q_even, freqs_cos),
        &mlxcel_core::multiply(&q_odd, freqs_sin),
    );
    let q_rot_odd = mlxcel_core::add(
        &mlxcel_core::multiply(&q_even, freqs_sin),
        &mlxcel_core::multiply(&q_odd, freqs_cos),
    );

    let k_rot_even = mlxcel_core::subtract(
        &mlxcel_core::multiply(&k_even, freqs_cos),
        &mlxcel_core::multiply(&k_odd, freqs_sin),
    );
    let k_rot_odd = mlxcel_core::add(
        &mlxcel_core::multiply(&k_even, freqs_sin),
        &mlxcel_core::multiply(&k_odd, freqs_cos),
    );

    // Interleave back: stack [even, odd] on last dim -> [B, seq, heads, head_dim/2, 2]
    // Then flatten to [B, seq, heads, head_dim]
    let q_rot_even = mlxcel_core::reshape(
        &q_rot_even,
        &[q_shape[0], q_shape[1], q_shape[2], head_dim / 2, 1],
    );
    let q_rot_odd = mlxcel_core::reshape(
        &q_rot_odd,
        &[q_shape[0], q_shape[1], q_shape[2], head_dim / 2, 1],
    );
    let q_stacked = mlxcel_core::concatenate(&q_rot_even, &q_rot_odd, 4);
    let q_out = mlxcel_core::reshape(&q_stacked, &[q_shape[0], q_shape[1], q_shape[2], head_dim]);
    let q_out = mlxcel_core::astype(&q_out, mlxcel_core::array_dtype(q));

    let k_rot_even = mlxcel_core::reshape(
        &k_rot_even,
        &[k_shape[0], k_shape[1], k_shape[2], head_dim / 2, 1],
    );
    let k_rot_odd = mlxcel_core::reshape(
        &k_rot_odd,
        &[k_shape[0], k_shape[1], k_shape[2], head_dim / 2, 1],
    );
    let k_stacked = mlxcel_core::concatenate(&k_rot_even, &k_rot_odd, 4);
    let k_out = mlxcel_core::reshape(&k_stacked, &[k_shape[0], k_shape[1], k_shape[2], head_dim]);
    let k_out = mlxcel_core::astype(&k_out, mlxcel_core::array_dtype(k));

    (q_out, k_out)
}

// Vision Attention.
/// Multi-head attention for Llama4 vision encoder
/// Uses biases on all projections and vision RoPE
struct VisionAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Llama4VisionConfig,
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

        let head_dim = config.hidden_size / config.num_attention_heads;
        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            head_dim,
            scale,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        freqs_cos: &MlxArray,
        freqs_sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let num_heads = self.num_heads as i32;
        let head_dim = self.head_dim as i32;

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [B, L, num_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, num_heads, head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, num_heads, head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, num_heads, head_dim]);

        // Apply vision RoPE
        let (q, k) = apply_vision_rope(&q, &k, freqs_cos, freqs_sin);

        // Transpose to [B, num_heads, L, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Scaled dot-product attention (no mask - full attention)
        let attn_output = mlxcel_core::layers::attention(&q, &k, &v, self.scale, None, 0.0, 0);

        // Transpose back and reshape: [B, num_heads, L, head_dim] -> [B, L, hidden]
        let attn_output = mlxcel_core::transpose_axes(&attn_output, &[0, 2, 1, 3]);
        let attn_output = mlxcel_core::reshape(&attn_output, &[b, l, num_heads * head_dim]);

        // Output projection
        self.o_proj.forward(&attn_output)
    }
}

// Vision MLP.
/// Vision MLP: fc1 -> GELU(fast) -> fc2
/// When is_projector=true, applies double GELU: fc1 -> GELU -> fc2 -> GELU
struct VisionMLP {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    is_projector: bool,
}

impl VisionMLP {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let fc1 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc1", prefix), group_size, bits)?;
        let fc2 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc2", prefix), group_size, bits)?;
        Ok(Self {
            fc1,
            fc2,
            is_projector: false,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.fc1.forward(x);
        let x = mlxcel_core::gelu_approx(&x);
        if self.is_projector {
            let x = self.fc2.forward(&x);
            mlxcel_core::gelu_approx(&x)
        } else {
            self.fc2.forward(&x)
        }
    }
}

// Vision Encoder Layer.
/// Pre-norm transformer layer for vision encoder
struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl VisionEncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Llama4VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let self_attn = VisionAttention::from_weights(
            weights,
            &format!("{}.self_attn", prefix),
            config,
            group_size,
            bits,
        )?;
        let mlp = VisionMLP::from_weights(weights, &format!("{}.mlp", prefix), group_size, bits)?;

        // LayerNorm with bias
        let ln_prefix = format!("{}.input_layernorm", prefix);
        let ln_weight = weights
            .get(&format!("{}.weight", ln_prefix))
            .ok_or_else(|| format!("Missing {}.weight", ln_prefix))?;
        let ln_bias = weights
            .get(&format!("{}.bias", ln_prefix))
            .map(|b| mlxcel_core::copy(b));
        let input_layernorm =
            LayerNorm::new(mlxcel_core::copy(ln_weight), ln_bias, config.norm_eps);

        let pln_prefix = format!("{}.post_attention_layernorm", prefix);
        let pln_weight = weights
            .get(&format!("{}.weight", pln_prefix))
            .ok_or_else(|| format!("Missing {}.weight", pln_prefix))?;
        let pln_bias = weights
            .get(&format!("{}.bias", pln_prefix))
            .map(|b| mlxcel_core::copy(b));
        let post_attention_layernorm =
            LayerNorm::new(mlxcel_core::copy(pln_weight), pln_bias, config.norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        freqs_cos: &MlxArray,
        freqs_sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        // Self attention with pre-norm
        let residual = mlxcel_core::copy(x);
        let h = self.input_layernorm.forward(x);
        let h = self.self_attn.forward(&h, freqs_cos, freqs_sin);
        let h = mlxcel_core::add(&residual, &h);

        // Feed-forward with pre-norm
        let residual = mlxcel_core::copy(&h);
        let h2 = self.post_attention_layernorm.forward(&h);
        let h2 = self.mlp.forward(&h2);
        mlxcel_core::add(&residual, &h2)
    }
}

// PixelShuffleMLP (Vision Adapter).
/// Pixel shuffle + MLP adapter
///
/// 1. pixel_shuffle: [B, 6400, 1280] -> [B, 1600, 5120]
/// 2. MLP: fc1 -> GELU -> fc2 -> GELU (double GELU, no bias)
struct PixelShuffleMLP {
    mlp: VisionMLP,
    pixel_shuffle_ratio: f32,
}

impl PixelShuffleMLP {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        _config: &Llama4VisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut mlp =
            VisionMLP::from_weights(weights, &format!("{}.mlp", prefix), group_size, bits)?;
        mlp.is_projector = true; // enables double GELU

        Ok(Self {
            mlp,
            pixel_shuffle_ratio: 0.5,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shuffled = pixel_shuffle(x, self.pixel_shuffle_ratio);
        self.mlp.forward(&shuffled)
    }
}

/// Pixel shuffle: reduces spatial patches by factor of ratio^2, increases channels
///
/// [B, num_patches, channels] -> [B, num_patches * ratio^2, channels / ratio^2]
/// For ratio=0.5: 6400 patches -> 1600 patches, 1280 channels -> 5120 channels
fn pixel_shuffle(x: &MlxArray, ratio: f32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let batch_size = shape[0];
    let num_patches = shape[1];
    let channels = shape[2];

    let patch_size = (num_patches as f32).sqrt() as i32; // e.g., 80

    // Reshape to spatial grid: [B, H, W, C]
    let x = mlxcel_core::reshape(x, &[batch_size, patch_size, patch_size, channels]);

    let width = patch_size;
    let height = patch_size;

    // First pass: reshape + transpose
    let new_w = (width as f32 * ratio) as i32;
    let new_c1 = (channels as f32 / ratio) as i32;
    let x = mlxcel_core::reshape(&x, &[batch_size, height, new_w, new_c1]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);

    // Second pass: reshape + transpose
    let new_h = (height as f32 * ratio) as i32;
    let new_c2 = (new_c1 as f32 / ratio) as i32;
    let x = mlxcel_core::reshape(&x, &[batch_size, new_h, new_w, new_c2]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);

    // Flatten spatial dims: [B, new_patches, final_channels]
    let final_patches = new_h * new_w;
    mlxcel_core::reshape(&x, &[batch_size, final_patches, new_c2])
}

// VisionModel.
/// Llama 4 Vision Model (full encoder pipeline)
pub struct Llama4VisionModel {
    patch_embedding: UnfoldConvolution,
    class_embedding: UniquePtr<MlxArray>,
    positional_embedding_vlm: UniquePtr<MlxArray>,
    layernorm_pre: LayerNorm,
    layernorm_post: LayerNorm,
    layers: Vec<VisionEncoderLayer>,
    vision_adapter: PixelShuffleMLP,
    // Precomputed RoPE
    freqs_cos: UniquePtr<MlxArray>,
    freqs_sin: UniquePtr<MlxArray>,
}

impl Llama4VisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Llama4VisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        // Patch embedding (UnfoldConvolution)
        let patch_embedding = UnfoldConvolution::from_weights(
            weights,
            &format!("{}.patch_embedding", prefix),
            config.patch_size,
        )?;

        // Class embedding: [hidden_size]
        let class_embedding = weights
            .get(&format!("{}.class_embedding", prefix))
            .ok_or_else(|| format!("Missing {}.class_embedding", prefix))?;
        let class_embedding = mlxcel_core::copy(class_embedding);

        // Positional embedding: [num_patches + 1, hidden_size]
        let positional_embedding_vlm = weights
            .get(&format!("{}.positional_embedding_vlm", prefix))
            .ok_or_else(|| format!("Missing {}.positional_embedding_vlm", prefix))?;
        let positional_embedding_vlm = mlxcel_core::copy(positional_embedding_vlm);

        // Layer norms
        let lnpre_w = weights
            .get(&format!("{}.layernorm_pre.weight", prefix))
            .ok_or_else(|| format!("Missing {}.layernorm_pre.weight", prefix))?;
        let lnpre_b = weights
            .get(&format!("{}.layernorm_pre.bias", prefix))
            .map(|b| mlxcel_core::copy(b));
        let layernorm_pre = LayerNorm::new(mlxcel_core::copy(lnpre_w), lnpre_b, config.norm_eps);

        let lnpost_w = weights
            .get(&format!("{}.layernorm_post.weight", prefix))
            .ok_or_else(|| format!("Missing {}.layernorm_post.weight", prefix))?;
        let lnpost_b = weights
            .get(&format!("{}.layernorm_post.bias", prefix))
            .map(|b| mlxcel_core::copy(b));
        let layernorm_post = LayerNorm::new(mlxcel_core::copy(lnpost_w), lnpost_b, config.norm_eps);

        // Encoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = VisionEncoderLayer::from_weights(
                weights,
                &format!("{}.model.layers.{}", prefix, i),
                config,
                group_size,
                bits,
            )?;
            layers.push(layer);
        }

        // Vision adapter (PixelShuffleMLP)
        let vision_adapter = PixelShuffleMLP::from_weights(
            weights,
            &format!("{}.vision_adapter", prefix),
            config,
            group_size,
            bits,
        )?;

        // Precompute RoPE
        let rope = VisionRotaryEmbedding::new(config);

        Ok(Self {
            patch_embedding,
            class_embedding,
            positional_embedding_vlm,
            layernorm_pre,
            layernorm_post,
            layers,
            vision_adapter,
            freqs_cos: rope.freqs_cos,
            freqs_sin: rope.freqs_sin,
        })
    }
}

impl VisionEncoder for Llama4VisionModel {
    /// Forward pass through the full vision encoder
    ///
    /// Input: [B, C, H, W] (channels-first from SigLipProcessor)
    /// Note: The VisionModule wrapper transposes to [B, H, W, C] before calling this,
    /// but Llama4 vision needs [B, C, H, W] for UnfoldConvolution.
    /// We handle this by transposing back.
    fn forward(&self, pixel_values: &MlxArray) -> VisionEncoderOutput {
        let pv_shape = mlxcel_core::array_shape(pixel_values);

        // The generic VisionModule transposes [B,C,H,W] -> [B,H,W,C]
        // We need [B,C,H,W] for our unfold convolution, so transpose back if needed
        let pv = if pv_shape.len() == 4 && pv_shape[3] <= 4 {
            // [B, H, W, C] -> [B, C, H, W]
            mlxcel_core::transpose_axes(pixel_values, &[0, 3, 1, 2])
        } else {
            mlxcel_core::copy(pixel_values)
        };

        let pv_shape = mlxcel_core::array_shape(&pv);
        let b = pv_shape[0];

        // 1. Patch embedding: [B, C, H, W] -> [B, num_patches, hidden]
        let mut hidden_state = self.patch_embedding.forward(&pv);

        let h_shape = mlxcel_core::array_shape(&hidden_state);
        let num_patches = h_shape[1];
        let hidden_dim = h_shape[2];

        // 2. Append class embedding: [B, num_patches+1, hidden]
        let class_emb = mlxcel_core::broadcast_to(&self.class_embedding, &[b, 1, hidden_dim]);
        hidden_state = mlxcel_core::concatenate(&hidden_state, &class_emb, 1);

        // 3. Add positional embedding
        hidden_state = mlxcel_core::add(&hidden_state, &self.positional_embedding_vlm);

        // 4. Pre-LayerNorm
        hidden_state = self.layernorm_pre.forward(&hidden_state);

        // 5. Encoder layers
        for layer in &self.layers {
            hidden_state = layer.forward(&hidden_state, &self.freqs_cos, &self.freqs_sin);
        }

        // 6. Post-LayerNorm
        hidden_state = self.layernorm_post.forward(&hidden_state);

        // 7. Remove CLS token (last token): [:, :-1, :]
        hidden_state = mlxcel_core::slice(&hidden_state, &[0, 0, 0], &[b, num_patches, hidden_dim]);

        // 8. Vision adapter (pixel shuffle + MLP)
        let output = self.vision_adapter.forward(&hidden_state);

        VisionEncoderOutput {
            hidden_states: output,
        }
    }
}
