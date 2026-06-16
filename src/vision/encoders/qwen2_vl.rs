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

//! Qwen2-VL Vision Encoder
//!
//! Custom ViT with:
//! - 3D patch embedding (temporal + spatial) implemented as Linear
//! - 2D Rotary Position Embeddings for vision
//! - Fused QKV attention with packed variable-length sequences (cu_seqlens)
//! - PatchMerger for spatial downsampling + projection to text hidden size
//!
//! Used by: Qwen2-VL
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen2_vl/vision.py

use super::VisionEncoderOutput;
use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// Concatenate multiple arrays along an axis
/// Used by: Qwen2-VL, Qwen2.5-VL
pub fn concat_many(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    assert!(!arrays.is_empty());
    let mut result = mlxcel_core::copy(arrays[0].as_ref().unwrap());
    for arr in &arrays[1..] {
        result = mlxcel_core::concatenate(result.as_ref().unwrap(), arr.as_ref().unwrap(), axis);
    }
    result
}

/// Qwen2-VL vision encoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2VLVisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,
    pub hidden_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(alias = "in_chans", default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f32,
    /// Quantization group_size (inherited from top-level config)
    #[serde(default)]
    pub quant_group_size: i32,
    /// Quantization bits (inherited from top-level config)
    #[serde(default)]
    pub quant_bits: i32,
}

fn default_depth() -> usize {
    32
}
fn default_embed_dim() -> usize {
    1280
}
fn default_num_heads() -> usize {
    16
}
fn default_patch_size() -> usize {
    14
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_in_channels() -> usize {
    3
}
fn default_mlp_ratio() -> f32 {
    4.0
}

// Helper: load LayerNorm from weights.
fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight_key = format!("{}.weight", prefix);
    let bias_key = format!("{}.bias", prefix);

    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", weight_key))?;
    let bias = weights.get(&bias_key).map(|b| mlxcel_core::copy(b));

    Ok(LayerNorm::new(weight, bias, eps))
}

// PatchEmbed - Conv3d degenerated to Linear (kernel == stride).
struct PatchEmbed {
    proj_weight: UniquePtr<MlxArray>,
    proj_bias: Option<UniquePtr<MlxArray>>,
    in_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
}

impl PatchEmbed {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen2VLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight_key = format!("{}.proj.weight", prefix);
        let w = weights
            .get(&weight_key)
            .ok_or_else(|| format!("Missing {}", weight_key))?;

        let shape = mlxcel_core::array_shape(w);
        let out_features = config.embed_dim as i32;
        let in_features = (config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size) as i32;

        // Handle Conv3d weight shape -> 2D Linear weight
        let w_reshaped = if shape.len() == 5 {
            // MLX Conv3d weight: [out, kT, kH, kW, in_channels]
            // Input data is in TCHW order (temporal, channel, height, width)
            // Reorder weight to [out, T, C, H, W] to match input layout
            let w_reordered = mlxcel_core::transpose_axes(w, &[0, 1, 4, 2, 3]);
            mlxcel_core::reshape(&w_reordered, &[out_features, in_features])
        } else if shape.len() == 2 {
            mlxcel_core::copy(w)
        } else {
            return Err(format!("Unexpected patch_embed weight shape: {:?}", shape));
        };

        let bias_key = format!("{}.proj.bias", prefix);
        let proj_bias = weights.get(&bias_key).map(|b| mlxcel_core::copy(b));

        Ok(Self {
            proj_weight: w_reshaped,
            proj_bias,
            in_channels: config.in_channels,
            temporal_patch_size: config.temporal_patch_size,
            patch_size: config.patch_size,
        })
    }

    fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        // Input: [total_patches * temporal_patch_size, in_channels * patch_size * patch_size]
        let shape = mlxcel_core::array_shape(hidden_states);
        let total_elements = shape[0];
        let n = total_elements / self.temporal_patch_size as i32;
        let in_features =
            (self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size)
                as i32;

        // Reshape: [total*temporal, C*P*P] -> [N, temporal, C*P*P] -> [N, temporal*C*P*P]
        let h = mlxcel_core::reshape(
            hidden_states,
            &[n, self.temporal_patch_size as i32, shape[1]],
        );
        let h = mlxcel_core::reshape(&h, &[n, in_features]);

        // Linear: h @ W.T + bias
        let wt = mlxcel_core::transpose(&self.proj_weight);
        let result = mlxcel_core::matmul(&h, &wt);
        match &self.proj_bias {
            Some(b) => mlxcel_core::add(&result, b),
            None => result,
        }
    }
}

// VisionRotaryEmbedding - 2D spatial position encoding.
/// Used by: Qwen2-VL, Qwen2.5-VL
pub struct VisionRotaryEmbedding {
    dim: usize,
    theta: f32,
}

impl VisionRotaryEmbedding {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            theta: 10000.0,
        }
    }

    /// Compute frequency table: [seqlen, dim/2]
    pub fn forward(&self, seqlen: i32) -> UniquePtr<MlxArray> {
        let half_dim = self.dim / 2;
        let dim_f = self.dim as f32;
        let mut inv_freq_data = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq_data.push(1.0 / self.theta.powf((2 * i) as f32 / dim_f));
        }
        let inv_freq = mlxcel_core::from_slice_f32(&inv_freq_data, &[half_dim as i32]);
        let seq = mlxcel_core::arange_i32(0, seqlen, 1);
        let seq = mlxcel_core::astype(&seq, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::outer(&seq, &inv_freq)
    }
}

// Vision Attention - Fused QKV with packed sequences.
struct VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen2VLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let qkv = UnifiedLinear::from_weights(weights, &format!("{}.attn.qkv", prefix), gs, bits)?;
        let proj =
            UnifiedLinear::from_weights(weights, &format!("{}.attn.proj", prefix), gs, bits)?;
        let head_dim = (config.embed_dim / config.num_heads) as i32;

        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// Forward with packed sequences (no batch dimension)
    /// x: [total_tokens, dim], cu_seqlens: [num_segments+1], rotary_pos_emb: [total_tokens, head_dim]
    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let seq_length = shape[0];

        // QKV: [seq_len, dim] -> [seq_len, dim*3]
        let qkv = self.qkv.forward(x);

        // Reshape: [seq_len, 3, num_heads, head_dim] -> [3, seq_len, num_heads, head_dim]
        let qkv = mlxcel_core::reshape(&qkv, &[seq_length, 3, self.num_heads, self.head_dim]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[1, 0, 2, 3]);

        // Split q, k, v
        let q = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, 0],
            &[1, seq_length, self.num_heads, self.head_dim],
        );
        let k = mlxcel_core::slice(
            &qkv,
            &[1, 0, 0, 0],
            &[2, seq_length, self.num_heads, self.head_dim],
        );
        let v = mlxcel_core::slice(
            &qkv,
            &[2, 0, 0, 0],
            &[3, seq_length, self.num_heads, self.head_dim],
        );
        let q = mlxcel_core::squeeze_axis(&q, 0);
        let k = mlxcel_core::squeeze_axis(&k, 0);
        let v = mlxcel_core::squeeze_axis(&v, 0);

        // Apply vision RoPE
        let q = apply_rotary_pos_emb_vision(&q, rotary_pos_emb);
        let k = apply_rotary_pos_emb_vision(&k, rotary_pos_emb);

        // Transpose for attention: [seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[1, 0, 2]);
        let k = mlxcel_core::transpose_axes(&k, &[1, 0, 2]);
        let v = mlxcel_core::transpose_axes(&v, &[1, 0, 2]);
        let q = mlxcel_core::expand_dims(&q, 0);
        let k = mlxcel_core::expand_dims(&k, 0);
        let v = mlxcel_core::expand_dims(&v, 0);

        // Per-image attention using cu_seqlens
        let num_segments = cu_seqlens.len() - 1;
        let mut attn_outputs = Vec::with_capacity(num_segments);

        for seg in 0..num_segments {
            let start = cu_seqlens[seg];
            let end = cu_seqlens[seg + 1];

            let q_seg = mlxcel_core::slice(
                &q,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            let k_seg = mlxcel_core::slice(
                &k,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            let v_seg = mlxcel_core::slice(
                &v,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );

            let attn = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_seg,
                    &k_seg,
                    &v_seg,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            };
            attn_outputs.push(attn);
        }

        // Concatenate along seq dimension
        let output = if attn_outputs.len() == 1 {
            attn_outputs.into_iter().next().unwrap()
        } else {
            concat_many(&attn_outputs, 2)
        };

        // [1, heads, seq, head_dim] -> [seq, heads, head_dim] -> [seq, dim]
        let output = mlxcel_core::squeeze_axis(&output, 0);
        let output = mlxcel_core::transpose_axes(&output, &[1, 0, 2]);
        let output = mlxcel_core::reshape(&output, &[seq_length, -1]);

        self.proj.forward(&output)
    }
}

/// Apply rotary position embedding to vision tensor
/// tensor: [seq_len, num_heads, head_dim]
/// freqs: [seq_len, head_dim] (position-dependent frequencies)
/// Used by: Qwen2-VL, Qwen2.5-VL
pub fn apply_rotary_pos_emb_vision(tensor: &MlxArray, freqs: &MlxArray) -> UniquePtr<MlxArray> {
    let orig_dtype = mlxcel_core::array_dtype(tensor);
    let tensor_f32 = mlxcel_core::astype(tensor, mlxcel_core::dtype::FLOAT32);

    let cos_vals = mlxcel_core::cos(freqs);
    let sin_vals = mlxcel_core::sin(freqs);

    // [seq_len, head_dim/2] -> [seq_len, 1, head_dim/2] -> tile to [seq_len, 1, head_dim]
    let cos_vals = mlxcel_core::expand_dims(&cos_vals, 1);
    let sin_vals = mlxcel_core::expand_dims(&sin_vals, 1);
    let cos_vals = mlxcel_core::tile(&cos_vals, &[1, 1, 2]);
    let sin_vals = mlxcel_core::tile(&sin_vals, &[1, 1, 2]);

    // output = tensor * cos + rotate_half(tensor) * sin
    let rotated = rotate_half(&tensor_f32);
    let term1 = mlxcel_core::multiply(&tensor_f32, &cos_vals);
    let term2 = mlxcel_core::multiply(&rotated, &sin_vals);
    let output = mlxcel_core::add(&term1, &term2);

    mlxcel_core::astype(&output, orig_dtype)
}

/// Rotate half: [-x2, x1] where x1 = x[..., :half], x2 = x[..., half:]
/// Used by: Qwen2-VL, Qwen2.5-VL
pub fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let half = shape[shape.len() - 1] / 2;
    let ndim = shape.len();

    let mut starts = vec![0i32; ndim];
    let mut stops = shape.clone();

    stops[ndim - 1] = half;
    let x1 = mlxcel_core::slice(x, &starts, &stops);

    starts[ndim - 1] = half;
    stops[ndim - 1] = shape[ndim - 1];
    let x2 = mlxcel_core::slice(x, &starts, &stops);

    let neg_x2 = mlxcel_core::negative(&x2);
    mlxcel_core::concatenate(&neg_x2, &x1, ndim as i32 - 1)
}

// Vision MLP.
struct VisionMLP {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl VisionMLP {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{}.mlp.fc1", prefix), gs, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{}.mlp.fc2", prefix), gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(x);
        let h = mlxcel_core::gelu_approx(&h);
        self.fc2.forward(&h)
    }
}

// VisionBlock.
struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMLP,
}

impl VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen2VLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: load_layer_norm(weights, &format!("{}.norm1", prefix), 1e-6)?,
            norm2: load_layer_norm(weights, &format!("{}.norm2", prefix), 1e-6)?,
            attn: VisionAttention::from_weights(weights, config, prefix, gs, bits)?,
            mlp: VisionMLP::from_weights(weights, prefix, gs, bits)?,
        })
    }

    fn forward(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = self.norm1.forward(hidden_states);
        let attn_out = self.attn.forward(&normed, cu_seqlens, rotary_pos_emb);
        let h = mlxcel_core::add(hidden_states, &attn_out);
        let normed = self.norm2.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// PatchMerger - spatial downsampling + projection to text hidden size.
struct PatchMerger {
    ln_q: LayerNorm,
    mlp_0: UnifiedLinear,
    mlp_2: UnifiedLinear,
    hidden_size: usize,
}

impl PatchMerger {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        context_dim: usize,
        spatial_merge_size: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let hidden_size = context_dim * spatial_merge_size * spatial_merge_size;
        Ok(Self {
            ln_q: load_layer_norm(weights, &format!("{}.ln_q", prefix), 1e-6)?,
            mlp_0: UnifiedLinear::from_weights(weights, &format!("{}.mlp.0", prefix), gs, bits)?,
            mlp_2: UnifiedLinear::from_weights(weights, &format!("{}.mlp.2", prefix), gs, bits)?,
            hidden_size,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.ln_q.forward(x);
        let h = mlxcel_core::reshape(&h, &[-1, self.hidden_size as i32]);
        let h = self.mlp_0.forward(&h);
        let h = mlxcel_core::gelu(&h);
        self.mlp_2.forward(&h)
    }
}

// Qwen2VLVisionEncoder.
/// Qwen2-VL Vision Model
///
/// Used by: Qwen2-VL
pub struct Qwen2VLVisionEncoder {
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    spatial_merge_size: usize,
}

impl Qwen2VLVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Qwen2VLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        let patch_embed =
            PatchEmbed::from_weights(weights, config, &format!("{}.patch_embed", prefix))?;
        let head_dim = config.embed_dim / config.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2);

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            blocks.push(VisionBlock::from_weights(
                weights,
                config,
                &format!("{}.blocks.{}", prefix, i),
                gs,
                bits,
            )?);
        }

        let merger = PatchMerger::from_weights(
            weights,
            &format!("{}.merger", prefix),
            config.embed_dim,
            config.spatial_merge_size,
            gs,
            bits,
        )?;

        Ok(Self {
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            spatial_merge_size: config.spatial_merge_size,
        })
    }

    /// Compute 2D rotary position embeddings from grid_thw
    /// grid_thw: Vec of (temporal, height, width) per image
    fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> UniquePtr<MlxArray> {
        let mut all_pos_ids: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut max_grid_dim: i32 = 0;

        for &(t, h, w) in grid_thw {
            if h > max_grid_dim {
                max_grid_dim = h;
            }
            if w > max_grid_dim {
                max_grid_dim = w;
            }
            let merge = self.spatial_merge_size as i32;

            // Build h position IDs with spatial merge grouping
            let h_arange = mlxcel_core::arange_i32(0, h, 1);
            let h_col = mlxcel_core::reshape(&h_arange, &[h, 1]);
            let hpos = mlxcel_core::repeat(&h_col, w, 1);
            let hpos = mlxcel_core::reshape(&hpos, &[h / merge, merge, w / merge, merge]);
            let hpos = mlxcel_core::transpose_axes(&hpos, &[0, 2, 1, 3]);
            let hpos = mlxcel_core::flatten(&hpos);

            // Build w position IDs
            let w_arange = mlxcel_core::arange_i32(0, w, 1);
            let w_row = mlxcel_core::reshape(&w_arange, &[1, w]);
            let wpos = mlxcel_core::repeat(&w_row, h, 0);
            let wpos = mlxcel_core::reshape(&wpos, &[h / merge, merge, w / merge, merge]);
            let wpos = mlxcel_core::transpose_axes(&wpos, &[0, 2, 1, 3]);
            let wpos = mlxcel_core::flatten(&wpos);

            // Stack [hpos, wpos] -> [h*w, 2], tile by t -> [t*h*w, 2]
            let stacked = mlxcel_core::stack_owned(&[hpos, wpos], -1);
            let tiled = mlxcel_core::tile(&stacked, &[t, 1]);
            all_pos_ids.push(tiled);
        }

        // Concatenate all position IDs: [total_tokens, 2]
        let pos_ids = if all_pos_ids.len() == 1 {
            all_pos_ids.into_iter().next().unwrap()
        } else {
            concat_many(&all_pos_ids, 0)
        };

        // Compute rotary embedding table for max grid dimension
        let rotary_table = self.rotary_pos_emb.forward(max_grid_dim);

        // Look up: pos_ids[i] = [h_idx, w_idx] -> rotary_table[h_idx] ++ rotary_table[w_idx]
        let pos_ids_flat = mlxcel_core::flatten(&pos_ids);
        let all_freqs = mlxcel_core::take(&rotary_table, &pos_ids_flat, 0);
        // [total*2, dim/2] -> [total, 2*dim/2] = [total, dim]
        let total_shape = mlxcel_core::array_shape(&pos_ids);
        let total_tokens = total_shape[0];
        let freq_shape = mlxcel_core::array_shape(&all_freqs);
        let half_dim = freq_shape[1];
        let all_freqs = mlxcel_core::reshape(&all_freqs, &[total_tokens, 2, half_dim]);
        mlxcel_core::reshape(&all_freqs, &[total_tokens, 2 * half_dim])
    }

    /// Compute cu_seqlens from grid_thw (on host)
    fn compute_cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
        let mut cu_seqlens = vec![0i32];
        let mut cumulative = 0i32;
        for &(t, h, w) in grid_thw {
            let tokens_per_frame = h * w;
            for _ in 0..t {
                cumulative += tokens_per_frame;
                cu_seqlens.push(cumulative);
            }
        }
        cu_seqlens
    }

    /// Forward pass
    /// hidden_states: [total_patches * temporal_patch_size, C * patch_size * patch_size]
    /// grid_thw: Vec of (temporal, height, width) for each image
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        let mut h = self.patch_embed.forward(hidden_states);
        let rotary_pos_emb = self.rot_pos_emb(grid_thw);
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw);

        for block in &self.blocks {
            h = block.forward(&h, &cu_seqlens, &rotary_pos_emb);
        }

        h = self.merger.forward(&h);

        VisionEncoderOutput { hidden_states: h }
    }
}

/// VisionEncoder trait - panics for Qwen2-VL since grid_thw is required
impl super::VisionEncoder for Qwen2VLVisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("Qwen2-VL vision encoder requires grid_thw; use forward_with_grid() instead");
    }
}
