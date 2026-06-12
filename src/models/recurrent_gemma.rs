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

// Griffin (RecurrentGemma): Hybrid Mamba + Local Attention architecture for mlxcel-core
// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/recurrent_gemma.py
//
// Key features:
// - Hybrid architecture with both recurrent (RGLRU) and local attention blocks
// - RGLRU (Real-Gated Linear Recurrent Unit) for recurrent blocks
// - Sliding window local attention for attention blocks
// - GemmaRMSNorm with (1 + weight) pattern
// - Conv1d for temporal mixing in recurrent blocks
// - logits_soft_cap for output softcapping
// - embeddings_scale_by_sqrt_dim option

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{
    GemmaRMSNorm, KVCache, RotatingKVCache, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::{create_causal_mask, repeat_kv, slice_axis, softcap};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GriffinConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_conv1d_width")]
    pub conv1d_width: usize,

    #[serde(default = "default_attention_window_size")]
    pub attention_window_size: usize,

    #[serde(default = "default_true")]
    pub embeddings_scale_by_sqrt_dim: bool,

    #[serde(default)]
    pub logits_soft_cap: Option<f32>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(alias = "_block_types", default)]
    pub block_types: Option<Vec<String>>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_conv1d_width() -> usize {
    4
}

fn default_attention_window_size() -> usize {
    2048
}

fn default_true() -> bool {
    true
}

impl GriffinConfig {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn get_block_types(&self) -> Vec<String> {
        self.block_types
            .clone()
            .unwrap_or_else(|| vec!["recurrent".to_string(), "attention".to_string()])
    }

    pub fn get_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// Griffin Cache Types.
/// Cache for RGLRU (recurrent) blocks
pub struct RGLRUCache {
    pub conv_cache: Option<UniquePtr<MlxArray>>,
    pub state_cache: Option<UniquePtr<MlxArray>>,
}

impl RGLRUCache {
    pub fn new() -> Self {
        Self {
            conv_cache: None,
            state_cache: None,
        }
    }
}

impl Default for RGLRUCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Enum for mixed cache types (RGLRU vs LocalAttention)
pub enum GriffinLayerCache {
    Attention(RotatingKVCache),
    Recurrent(RGLRUCache),
}

// RNN Scan Operation.
/// RNN scan operation for recurrent computation.
/// Implements the scan: y_t = a_t * y_{t-1} + x_t
fn rnn_scan(
    x: &MlxArray,
    a: &MlxArray,
    h0: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(x);
    let seq_len = shape[1] as usize;

    if seq_len == 1 {
        // Single token mode - simple update
        if let Some(h) = h0 {
            let h_exp = mlxcel_core::expand_dims(h, 1);
            let ah = mlxcel_core::multiply(a, &h_exp);
            let y = mlxcel_core::add(&ah, x);
            let last_h = slice_axis(&y, 1, -1, -1);
            let last_h = mlxcel_core::squeeze_axis(&last_h, 1);
            (y, last_h)
        } else {
            let last_h = mlxcel_core::squeeze_axis(x, 1);
            (mlxcel_core::copy(x), last_h)
        }
    } else {
        // Sequence mode - iterate through time
        let batch = shape[0];
        let d = shape[2];

        let mut h_t = if let Some(h) = h0 {
            mlxcel_core::copy(h)
        } else {
            mlxcel_core::zeros(&[batch, d], mlxcel_core::array_dtype(x))
        };

        let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::new();

        for t in 0..seq_len {
            let x_t = slice_axis(x, 1, t as i32, (t + 1) as i32);
            let x_t = mlxcel_core::squeeze_axis(&x_t, 1);
            let a_t = slice_axis(a, 1, t as i32, (t + 1) as i32);
            let a_t = mlxcel_core::squeeze_axis(&a_t, 1);

            let ah = mlxcel_core::multiply(&a_t, &h_t);
            h_t = mlxcel_core::add(&ah, &x_t);

            outputs.push(mlxcel_core::expand_dims(&h_t, 1));
        }

        // Concatenate outputs along time dimension
        let mut y = mlxcel_core::copy(&outputs[0]);
        for out in outputs.iter().skip(1) {
            y = concatenate(&y, out, 1);
        }

        (y, h_t)
    }
}

// RGLRU (Real-Gated Linear Recurrent Unit).
struct RGLRU {
    width: usize,
    num_heads: usize,
    head_dim: usize,

    recurrent_param: UniquePtr<MlxArray>,
    input_gate_weight: UniquePtr<MlxArray>,
    input_gate_bias: UniquePtr<MlxArray>,
    recurrent_gate_weight: UniquePtr<MlxArray>,
    recurrent_gate_bias: UniquePtr<MlxArray>,
}

impl RGLRU {
    fn apply_block_linear(&self, h: &MlxArray, w: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(h);
        let batch = shape[0];
        let seq_len = shape[1];

        // Reshape to [B, L, num_heads, head_dim]
        let h_reshaped = mlxcel_core::reshape(
            h,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );

        // Swap to [B, num_heads, L, head_dim]
        let h_swapped = mlxcel_core::transpose_axes(&h_reshaped, &[0, 2, 1, 3]);

        // Matmul with weight [num_heads, head_dim, head_dim]
        let out = mlxcel_core::matmul(&h_swapped, w);

        // Swap back to [B, L, num_heads, head_dim]
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);

        // Add bias
        let out = mlxcel_core::add(&out, b);

        // Flatten to [B, L, width]
        let out = mlxcel_core::reshape(&out, &[batch, seq_len, self.width as i32]);

        // Apply sigmoid
        mlxcel_core::sigmoid(&out)
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // Compute gates
        let gate_x = self.apply_block_linear(x, &self.input_gate_weight, &self.input_gate_bias);
        let gate_a =
            self.apply_block_linear(x, &self.recurrent_gate_weight, &self.recurrent_gate_bias);

        // Compute parameter A of the recurrence
        // log_a = -8.0 * gate_a * softplus(recurrent_param)
        let softplus_param = mlxcel_core::softplus(&self.recurrent_param);
        let neg_eight = mlxcel_core::full_f32(&[1], -8.0, mlxcel_core::array_dtype(&gate_a));
        let log_a =
            mlxcel_core::multiply(&mlxcel_core::multiply(&neg_eight, &gate_a), &softplus_param);
        let a = mlxcel_core::exp(&log_a);

        // Compute a_square = exp(2 * log_a)
        let two = mlxcel_core::full_f32(&[1], 2.0, mlxcel_core::array_dtype(&log_a));
        let log_a_2 = mlxcel_core::multiply(&two, &log_a);
        let a_square = mlxcel_core::exp(&log_a_2);

        // Gate the input
        let gated_x = mlxcel_core::multiply(x, &gate_x);

        // Apply gamma normalization: multiplier = sqrt(1 - a_square)
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&a_square));
        let one_minus_a2 = mlxcel_core::subtract(&one, &a_square);
        let mut multiplier = mlxcel_core::sqrt(&one_minus_a2);

        // If no cache, set first position multiplier to 1
        if cache.is_none() {
            let shape = mlxcel_core::array_shape(&multiplier);
            let ones_first = mlxcel_core::ones(
                &[shape[0], 1, shape[2]],
                mlxcel_core::array_dtype(&multiplier),
            );
            let rest = slice_axis(&multiplier, 1, 1, -1);
            multiplier = concatenate(&ones_first, &rest, 1);
        }

        // Cast multiplier to match x dtype
        let multiplier = mlxcel_core::astype(&multiplier, mlxcel_core::array_dtype(x));
        let normalized_x = mlxcel_core::multiply(&gated_x, &multiplier);

        // RNN scan
        let (y, last_h) = rnn_scan(&normalized_x, &a, cache);

        (y, last_h)
    }
}

// Recurrent Block (RGLRU-based).
struct RecurrentBlock {
    linear_y: UnifiedLinear,
    linear_x: UnifiedLinear,
    linear_out: UnifiedLinear,
    conv_weight: UniquePtr<MlxArray>,
    conv_bias: UniquePtr<MlxArray>,
    rg_lru: RGLRU,
    lru_width: usize,
    conv_kernel_size: usize,
}

impl RecurrentBlock {
    fn forward(&self, x: &MlxArray, cache: Option<&mut RGLRUCache>) -> UniquePtr<MlxArray> {
        // Y branch: linear + GELU
        let y = self.linear_y.forward(x);
        let y = mlxcel_core::gelu_approx(&y);

        // X branch: linear -> conv -> RGLRU
        let h = self.linear_x.forward(x);

        // Extract cache states
        let conv_cache = cache.as_ref().and_then(|c| {
            c.conv_cache
                .as_ref()
                .and_then(|s| s.as_ref().map(mlxcel_core::copy))
        });
        let state_cache = cache.as_ref().and_then(|c| {
            c.state_cache
                .as_ref()
                .and_then(|s| s.as_ref().map(mlxcel_core::copy))
        });

        // Conv1d
        let shape = mlxcel_core::array_shape(&h);
        let k = self.conv_kernel_size;

        // Pad input with cache or zeros
        let padded_input = if let Some(ref conv_st) = conv_cache {
            concatenate(conv_st, &h, 1)
        } else {
            let pad_arr = mlxcel_core::zeros(
                &[shape[0], (k - 1) as i32, shape[2]],
                mlxcel_core::array_dtype(&h),
            );
            concatenate(&pad_arr, &h, 1)
        };

        // Depthwise conv1d: transpose to [B, C, L], apply conv, transpose back
        let h_t = mlxcel_core::transpose_axes(&padded_input, &[0, 2, 1]);
        let conv_out = mlxcel_core::conv1d(&h_t, &self.conv_weight, 1, 0, 1, self.lru_width as i32);
        let conv_out = mlxcel_core::transpose_axes(&conv_out, &[0, 2, 1]);

        // Add bias
        let bias_reshaped = mlxcel_core::reshape(&self.conv_bias, &[1, 1, -1]);
        let h_conv = mlxcel_core::add(&conv_out, &bias_reshaped);

        // Update conv cache
        let new_conv_cache = {
            let padded_shape = mlxcel_core::array_shape(&padded_input);
            let len = padded_shape[1] as usize;
            slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32)
        };

        // RGLRU
        let (h_lru, new_state_cache) = self.rg_lru.forward(&h_conv, state_cache.as_deref());

        // Update cache
        if let Some(c) = cache {
            c.conv_cache = Some(new_conv_cache);
            c.state_cache = Some(new_state_cache);
        }

        // Combine branches: h * y
        let combined = mlxcel_core::multiply(&h_lru, &y);

        // Output projection
        self.linear_out.forward(&combined)
    }
}

// Local Attention Block (sliding window).
struct LocalAttentionBlock {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_dims: usize,
    rope_theta: f32,
}

impl LocalAttentionBlock {
    fn forward(
        &self,
        x: &MlxArray,
        cache: Option<&mut RotatingKVCache>,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let queries = self.q_proj.forward(x);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let queries = mlxcel_core::reshape(
            &queries,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );
        let keys = mlxcel_core::reshape(
            &keys,
            &[
                batch,
                seq_len,
                self.num_kv_heads as i32,
                self.head_dim as i32,
            ],
        );
        let values = mlxcel_core::reshape(
            &values,
            &[
                batch,
                seq_len,
                self.num_kv_heads as i32,
                self.head_dim as i32,
            ],
        );

        // Apply RoPE
        let offset = cache.as_ref().map(|c| c.get_offset()).unwrap_or(0);
        let queries = mlxcel_core::fast_rope(
            &queries,
            self.rope_dims as i32,
            false,
            self.rope_theta,
            1.0,
            offset,
        );
        let keys = mlxcel_core::fast_rope(
            &keys,
            self.rope_dims as i32,
            false,
            self.rope_theta,
            1.0,
            offset,
        );

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        // Update KV cache
        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(keys, values)
        } else {
            (keys, values)
        };

        // Repeat KV for GQA if needed
        let n_rep = self.num_heads / self.num_kv_heads;
        let (keys, values) = if n_rep > 1 {
            (
                repeat_kv(&keys, n_rep as i32),
                repeat_kv(&values, n_rep as i32),
            )
        } else {
            (keys, values)
        };

        // Scaled dot-product attention
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let attn_out = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &queries, &keys, &values, self.scale, mask_ptr, 0.0, 0,
            )
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(
            &attn_out,
            &[batch, seq_len, (self.num_heads * self.head_dim) as i32],
        );

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}

// MLP Block (Gated).
struct MLPBlock {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MLPBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // GELU(gate) * up
        let activated = mlxcel_core::multiply(&mlxcel_core::gelu_approx(&gate), &up);

        self.down_proj.forward(&activated)
    }
}

// Temporal Block Enum.
enum TemporalBlock {
    Recurrent(RecurrentBlock),
    Attention(LocalAttentionBlock),
}

// Residual Block.
struct ResidualBlock {
    temporal_block: TemporalBlock,
    mlp_block: MLPBlock,
    temporal_pre_norm: GemmaRMSNorm,
    channel_pre_norm: GemmaRMSNorm,
}

impl ResidualBlock {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut GriffinLayerCache,
    ) -> UniquePtr<MlxArray> {
        // Temporal block with pre-norm
        let h_norm = self.temporal_pre_norm.forward(x);

        let h = match (&self.temporal_block, cache) {
            (TemporalBlock::Recurrent(block), GriffinLayerCache::Recurrent(rcache)) => {
                block.forward(&h_norm, Some(rcache))
            }
            (TemporalBlock::Recurrent(block), _) => block.forward(&h_norm, None),
            (TemporalBlock::Attention(block), GriffinLayerCache::Attention(kvcache)) => {
                block.forward(&h_norm, Some(kvcache), mask)
            }
            (TemporalBlock::Attention(block), _) => block.forward(&h_norm, None, mask),
        };

        // Residual connection
        let r = mlxcel_core::add(x, &h);

        // MLP block with pre-norm
        let ff_norm = self.channel_pre_norm.forward(&r);
        let ff_out = self.mlp_block.forward(&ff_norm);

        // Final residual
        mlxcel_core::add(&r, &ff_out)
    }
}

// Griffin Model Backbone.
struct GriffinBackbone {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<ResidualBlock>,
    final_norm: GemmaRMSNorm,
    block_types: Vec<String>,
    window_size: usize,
    scale_by_sqrt_dim: bool,
    attn_idx: usize,
}

impl GriffinBackbone {
    fn forward(
        &self,
        inputs: &MlxArray,
        caches: Option<&mut [GriffinLayerCache]>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(inputs);

        // Scale embeddings if configured
        if self.scale_by_sqrt_dim {
            let dim = mlxcel_core::array_shape(&h)[2] as f32;
            let scale_val = dim.sqrt();
            let scale = mlxcel_core::full_f32(&[1], scale_val, mlxcel_core::array_dtype(&h));
            h = mlxcel_core::multiply(&h, &scale);
        }

        // Create attention mask for sliding window
        let mask = {
            let shape = mlxcel_core::array_shape(&h);
            let seq_len = shape[1];
            if seq_len > 1 {
                let offset = caches
                    .as_ref()
                    .map(|c| {
                        if self.attn_idx < c.len() {
                            match &c[self.attn_idx] {
                                GriffinLayerCache::Attention(kv) => kv.get_offset(),
                                _ => 0,
                            }
                        } else {
                            0
                        }
                    })
                    .unwrap_or(0);
                Some(create_causal_mask(seq_len, offset))
            } else {
                None
            }
        };

        // Forward through layers
        if let Some(cache_slice) = caches {
            for (layer, cache) in self.layers.iter().zip(cache_slice.iter_mut()) {
                h = layer.forward(&h, mask.as_deref(), cache);
            }
        } else {
            // No cache - create temporary caches
            let mut temp_caches: Vec<GriffinLayerCache> = self
                .block_types
                .iter()
                .map(|t| {
                    if t == "attention" {
                        GriffinLayerCache::Attention(RotatingKVCache::new(self.window_size as i32))
                    } else {
                        GriffinLayerCache::Recurrent(RGLRUCache::new())
                    }
                })
                .collect();
            for (layer, cache) in self.layers.iter().zip(temp_caches.iter_mut()) {
                h = layer.forward(&h, mask.as_deref(), cache);
            }
        }

        self.final_norm.forward(&h)
    }
}

// Full Griffin Model.
pub struct GriffinModel {
    config: GriffinConfig,
    model: GriffinBackbone,
    lm_head: Option<UnifiedLinear>,
}

impl GriffinModel {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_griffin_caches(&self) -> Vec<GriffinLayerCache> {
        self.model
            .block_types
            .iter()
            .map(|t| {
                if t == "attention" {
                    GriffinLayerCache::Attention(RotatingKVCache::new(
                        self.config.attention_window_size as i32,
                    ))
                } else {
                    GriffinLayerCache::Recurrent(RGLRUCache::new())
                }
            })
            .collect()
    }

    pub fn forward_with_caches(
        &self,
        inputs: &MlxArray,
        caches: Option<&mut [GriffinLayerCache]>,
    ) -> UniquePtr<MlxArray> {
        let out = self.model.forward(inputs, caches);

        // Apply lm_head
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&out)
        } else {
            self.model.embed_tokens.as_linear(&out)
        };

        // Apply logits soft cap if configured
        if let Some(cap) = self.config.logits_soft_cap {
            softcap(&logits, cap)
        } else {
            logits
        }
    }

    /// Load model from safetensors files
    pub fn load(model_path: &str) -> Result<(Self, GriffinConfig), Box<dyn std::error::Error>> {
        let path = Path::new(model_path);

        // Load config
        println!("[Griffin] Loading config...");
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_str = super::sanitize_config_json(&config_str);
        let config: GriffinConfig = serde_json::from_str(&config_str)?;
        println!(
            "[Griffin] Config loaded: {} layers ({} recurrent, {} attention)",
            config.num_hidden_layers,
            config
                .get_block_types()
                .iter()
                .filter(|t| *t == "recurrent")
                .count(),
            config
                .get_block_types()
                .iter()
                .filter(|t| *t == "attention")
                .count()
        );

        // Load weights
        println!("[Griffin] Loading weights from safetensors...");
        let weights = crate::models::load_text_weights(path, None)?;

        // Process weights
        let weights = Self::sanitize_weights(weights, &config);

        // Build model
        println!("[Griffin] Building model...");
        let model = Self::from_weights(config.clone(), weights)?;

        println!("[Griffin] Model loaded successfully");
        Ok((model, config))
    }

    fn sanitize_weights(mut weights: WeightMap, _config: &GriffinConfig) -> WeightMap {
        // Handle conv1d weight transpose if needed
        let keys: Vec<String> = weights.keys().cloned().collect();
        for k in keys {
            if k.contains("conv_1d.weight")
                && let Some(v) = weights.get(&k)
            {
                let shape = mlxcel_core::array_shape(v);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    let transposed = mlxcel_core::swap_axes(v, -1, -2);
                    weights.insert(k, transposed);
                }
            }
        }

        // Remove lm_head if tied embeddings
        if !weights.contains_key("lm_head.weight") {
            // Model uses tied embeddings
        }

        weights
    }

    pub fn from_weights(
        config: GriffinConfig,
        mut weights: WeightMap,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let block_types = config.get_block_types();

        // Find first attention layer index
        let attn_idx = block_types
            .iter()
            .position(|t| t == "attention")
            .unwrap_or(0);

        // Get quantization parameters
        let group_size = config.group_size();
        let bits = config.bits();

        // Build embeddings
        let embed_tokens =
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", group_size, bits)?;

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (i, layer_type) in block_types.iter().enumerate() {
            let prefix = format!("model.layers.{}", i);

            // Build temporal block
            let temporal_block = if layer_type == "recurrent" {
                // Recurrent block components
                let linear_y = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.temporal_block.linear_y", prefix),
                    group_size,
                    bits,
                )?;
                let linear_x = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.temporal_block.linear_x", prefix),
                    group_size,
                    bits,
                )?;
                let linear_out = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.temporal_block.linear_out", prefix),
                    group_size,
                    bits,
                )?;

                // Conv1d
                let conv_weight = weights
                    .remove(&format!("{}.temporal_block.conv_1d.weight", prefix))
                    .ok_or(format!("Missing conv_1d weight for layer {}", i))?;
                let conv_bias = weights
                    .remove(&format!("{}.temporal_block.conv_1d.bias", prefix))
                    .ok_or(format!("Missing conv_1d bias for layer {}", i))?;

                // RGLRU parameters
                let recurrent_param = weights
                    .remove(&format!("{}.temporal_block.rg_lru.recurrent_param", prefix))
                    .ok_or(format!("Missing recurrent_param for layer {}", i))?;
                let input_gate_weight = weights
                    .remove(&format!(
                        "{}.temporal_block.rg_lru.input_gate_weight",
                        prefix
                    ))
                    .ok_or(format!("Missing input_gate_weight for layer {}", i))?;
                let input_gate_bias = weights
                    .remove(&format!("{}.temporal_block.rg_lru.input_gate_bias", prefix))
                    .ok_or(format!("Missing input_gate_bias for layer {}", i))?;
                let recurrent_gate_weight = weights
                    .remove(&format!(
                        "{}.temporal_block.rg_lru.recurrent_gate_weight",
                        prefix
                    ))
                    .ok_or(format!("Missing recurrent_gate_weight for layer {}", i))?;
                let recurrent_gate_bias = weights
                    .remove(&format!(
                        "{}.temporal_block.rg_lru.recurrent_gate_bias",
                        prefix
                    ))
                    .ok_or(format!("Missing recurrent_gate_bias for layer {}", i))?;

                let lru_width = config.hidden_size; // Default LRU width = hidden_size
                let rg_lru = RGLRU {
                    width: lru_width,
                    num_heads: config.num_attention_heads,
                    head_dim: lru_width / config.num_attention_heads,
                    recurrent_param,
                    input_gate_weight,
                    input_gate_bias,
                    recurrent_gate_weight,
                    recurrent_gate_bias,
                };

                TemporalBlock::Recurrent(RecurrentBlock {
                    linear_y,
                    linear_x,
                    linear_out,
                    conv_weight,
                    conv_bias,
                    rg_lru,
                    lru_width,
                    conv_kernel_size: config.conv1d_width,
                })
            } else {
                // Attention block
                let q_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.temporal_block.q_proj", prefix),
                    group_size,
                    bits,
                )?;
                let k_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.temporal_block.k_proj", prefix),
                    group_size,
                    bits,
                )?;
                let v_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.temporal_block.v_proj", prefix),
                    group_size,
                    bits,
                )?;
                let o_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.temporal_block.o_proj", prefix),
                    group_size,
                    bits,
                )?;

                let head_dim = config.get_head_dim();

                TemporalBlock::Attention(LocalAttentionBlock {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim,
                    scale: (head_dim as f32).powf(-0.5),
                    rope_dims: head_dim / 2,
                    rope_theta: config.rope_theta,
                })
            };

            // Build MLP block
            let _half_expanded = config.intermediate_size / 2;
            let gate_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mlp_block.gate_proj", prefix),
                group_size,
                bits,
            )?;
            let up_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mlp_block.up_proj", prefix),
                group_size,
                bits,
            )?;
            let down_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mlp_block.down_proj", prefix),
                group_size,
                bits,
            )?;

            let mlp_block = MLPBlock {
                gate_proj,
                up_proj,
                down_proj,
            };

            // Build norms (Gemma-style with 1 + weight)
            let temporal_norm_weight = weights
                .remove(&format!("{}.temporal_pre_norm.weight", prefix))
                .ok_or(format!("Missing temporal_pre_norm for layer {}", i))?;
            let channel_norm_weight = weights
                .remove(&format!("{}.channel_pre_norm.weight", prefix))
                .ok_or(format!("Missing channel_pre_norm for layer {}", i))?;

            layers.push(ResidualBlock {
                temporal_block,
                mlp_block,
                temporal_pre_norm: GemmaRMSNorm::new(temporal_norm_weight, config.rms_norm_eps),
                channel_pre_norm: GemmaRMSNorm::new(channel_norm_weight, config.rms_norm_eps),
            });
        }

        // Final norm
        let final_norm_weight = weights
            .remove("model.final_norm.weight")
            .ok_or("Missing final_norm weight")?;
        let final_norm = GemmaRMSNorm::new(final_norm_weight, config.rms_norm_eps);

        // LM head (quantized if not tied)
        let lm_head = if weights.contains_key("lm_head.weight") {
            Some(UnifiedLinear::from_weights(
                &weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        let model = GriffinBackbone {
            embed_tokens,
            layers,
            final_norm,
            block_types,
            window_size: config.attention_window_size,
            scale_by_sqrt_dim: config.embeddings_scale_by_sqrt_dim,
            attn_idx,
        };

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for GriffinModel {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Note: Griffin uses mixed cache types (KVCache + RGLRUCache)
        // For LanguageModel trait compatibility, we use internal caching
        self.forward_with_caches(input, None)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Return KV caches for compatibility - actual usage should use make_griffin_caches()
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt RGLRU recurrent state
    }

    fn supports_batching(&self) -> bool {
        false // RecurrentGemma uses internal RGLRUCache state, not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![1] // Gemma EOS token
    }
}
