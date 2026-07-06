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

//! DeepSeek V2 model implementation using mlxcel-core
//!
//! Key features:
//! - MLA (Multi-head Latent Attention) with compressed KV cache
//! - Q/K with LoRA-like compression (q_lora_rank, kv_lora_rank)
//! - Separate rope and non-rope portions (qk_rope_head_dim, qk_nope_head_dim)
//! - Yarn RoPE for extended context
//! - MoE with group-limited greedy routing
//! - Shared experts plus routed experts

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{repeat_kv, silu, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,

    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub n_routed_experts: Option<usize>,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    #[serde(default = "default_kv_lora_rank")]
    pub kv_lora_rank: usize,

    #[serde(default)]
    pub q_lora_rank: Option<usize>,

    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,

    #[serde(default = "default_v_head_dim")]
    pub v_head_dim: usize,

    #[serde(default = "default_qk_nope_head_dim")]
    pub qk_nope_head_dim: usize,

    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,

    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,

    #[serde(default)]
    pub first_k_dense_replace: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "deepseek_v2".to_string()
}
fn default_vocab_size() -> usize {
    102400
}
fn default_hidden_size() -> usize {
    4096
}
fn default_intermediate_size() -> usize {
    11008
}
fn default_moe_intermediate_size() -> usize {
    1407
}
fn default_num_hidden_layers() -> usize {
    30
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_kv_lora_rank() -> usize {
    512
}
fn default_qk_rope_head_dim() -> usize {
    64
}
fn default_v_head_dim() -> usize {
    128
}
fn default_qk_nope_head_dim() -> usize {
    128
}
fn default_moe_layer_freq() -> usize {
    1
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.n_routed_experts.is_some()
            && layer_idx >= self.first_k_dense_replace
            && (layer_idx - self.first_k_dense_replace).is_multiple_of(self.moe_layer_freq)
    }

    // Yarn RoPE parameters
    pub fn scaling_factor(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32
    }

    pub fn original_max_pos(&self) -> usize {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("original_max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize
    }

    pub fn beta_fast(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("beta_fast"))
            .and_then(|v| v.as_f64())
            .unwrap_or(32.0) as f32
    }

    pub fn beta_slow(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("beta_slow"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32
    }

    pub fn mscale(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("mscale"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32
    }

    pub fn mscale_all_dim(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|m| m.get("mscale_all_dim"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32
    }
}

// Yarn RoPE.
fn yarn_find_correction_dim(num_rotations: f32, dim: usize, base: f32, max_pos: usize) -> f32 {
    let dim_f = dim as f32;
    let max_pos_f = max_pos as f32;
    (dim_f * (max_pos_f / (num_rotations * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * base.ln())
}

fn yarn_find_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_pos: usize,
) -> (usize, usize) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_pos).floor() as i32;
    let high = yarn_find_correction_dim(high_rot, dim, base, max_pos).ceil() as i32;
    (low.max(0) as usize, (high as usize).min(dim - 1))
}

fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

struct YarnRoPE {
    freqs: UniquePtr<MlxArray>,
    mscale: f32,
    dim: i32,
}

impl YarnRoPE {
    fn new(
        dim: usize,
        base: f32,
        scaling_factor: f32,
        original_max_pos: usize,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all_dim: f32,
    ) -> Self {
        let computed_mscale = yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim);

        let half_dim = dim / 2;
        let mut freq_extra = Vec::with_capacity(half_dim);
        let mut freq_inter = Vec::with_capacity(half_dim);

        for i in 0..half_dim {
            let t = (2 * i) as f32 / dim as f32;
            freq_extra.push(base.powf(t));
            freq_inter.push(scaling_factor * base.powf(t));
        }

        let (low, high) =
            yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max_pos);

        let mut freqs_vec = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq_mask = if i >= low && i <= high {
                let t = (i - low) as f32 / (high - low).max(1) as f32;
                1.0 - t.clamp(0.0, 1.0)
            } else if i < low {
                1.0
            } else {
                0.0
            };

            let f = (freq_inter[i] * freq_extra[i])
                / (freq_inter[i] * freq_mask + freq_extra[i] * (1.0 - freq_mask));
            freqs_vec.push(f);
        }

        let freqs = mlxcel_core::from_slice_f32(&freqs_vec, &[half_dim as i32]);

        Self {
            freqs,
            mscale: computed_mscale,
            dim: dim as i32,
        }
    }

    fn forward(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let x = if self.mscale != 1.0 {
            mlxcel_core::multiply_scalar(x, self.mscale)
        } else {
            mlxcel_core::copy(x)
        };

        mlxcel_core::fast_rope_with_freqs(&x, self.dim, true, 1.0, offset, &self.freqs)
    }
}

// MLA Attention.
#[allow(dead_code)]
struct MLAAttention {
    // Q projection: either direct or LoRA-style
    q_proj: Option<UnifiedLinear>,
    q_a_proj: Option<UnifiedLinear>,
    q_a_layernorm: Option<RMSNorm>,
    q_b_proj: Option<UnifiedLinear>,

    // KV projection (always LoRA-style)
    kv_a_proj_with_mqa: UnifiedLinear,
    kv_a_layernorm: RMSNorm,
    kv_b_proj: UnifiedLinear,

    o_proj: UnifiedLinear,

    num_heads: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    q_head_dim: usize,
    scale: f32,

    rope: YarnRoPE,
}

impl MLAAttention {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Compute Q (direct or LoRA-style)
        let q = if let Some(ref q_proj) = self.q_proj {
            q_proj.forward(x)
        } else {
            let q_a = self.q_a_proj.as_ref().unwrap().forward(x);
            let q_a_norm = self.q_a_layernorm.as_ref().unwrap().forward(&q_a);
            self.q_b_proj.as_ref().unwrap().forward(&q_a_norm)
        };

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads as i32, self.q_head_dim as i32]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Split Q into nope and pe parts
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim as i32);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim as i32, -1);

        // Compute KV with compression
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);
        let compressed = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank as i32);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank as i32, -1);

        // Reshape k_pe: [B, L, rope_dim] -> [B, 1, L, rope_dim]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim as i32]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // Decompress KV
        let kv = self.kv_a_layernorm.forward(&compressed);
        let kv = self.kv_b_proj.forward(&kv);
        let kv = mlxcel_core::reshape(&kv, &[b, l, self.num_heads as i32, -1]);
        let kv = mlxcel_core::transpose_axes(&kv, &[0, 2, 1, 3]);

        let k_nope = slice_axis(&kv, -1, 0, self.qk_nope_head_dim as i32);
        let values = slice_axis(&kv, -1, self.qk_nope_head_dim as i32, -1);

        // Apply Yarn RoPE
        let offset = cache.seq_len();
        let q_pe = self.rope.forward(&q_pe, offset);
        let k_pe = self.rope.forward(&k_pe, offset);

        // Broadcast k_pe to all heads
        let k_pe = repeat_kv(&k_pe, self.num_heads as i32);

        // Concatenate
        let queries = concatenate(&q_nope, &q_pe, -1);
        let keys = concatenate(&k_nope, &k_pe, -1);

        // Update cache
        let (keys, values) = cache.update_and_fetch(keys, values);

        // Scaled dot product attention
        let output = if l > 1 {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &queries, &keys, &values, self.scale, mask_ptr, 0.0, 0,
                )
            }
        } else {
            mlxcel_core::causal_attention(&queries, &keys, &values, self.scale, 0.0, 0)
        };

        // Reshape output
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);

        self.o_proj.forward(&output)
    }
}

// MLP (Dense and MoE).
struct DenseMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl DenseMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        let h = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&h)
    }
}

struct MoEGate {
    weight: UniquePtr<MlxArray>,
    top_k: usize,
    routed_scaling_factor: f32,
}

impl MoEGate {
    fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // gates = x @ weight.T
        let weight_t = mlxcel_core::transpose(&self.weight);
        let gates = mlxcel_core::matmul(x, &weight_t);
        let scores = mlxcel_core::softmax(&gates, -1);

        // Top-k selection
        let k = self.top_k as i32;
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);
        let topk_indices = slice_axis(&indices, -1, 0, k);

        let topk_scores = mlxcel_core::take_along_axis(&scores, &topk_indices, -1);
        let topk_scores = mlxcel_core::multiply_scalar(&topk_scores, self.routed_scaling_factor);

        (topk_indices, topk_scores)
    }
}

// Simplified MoE - no sorted_indices optimization (hardcoded false)
// Kept local: forward() uses sorted_indices=false, different from shared SwitchLinear
enum SwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    },
    Regular {
        weight: UniquePtr<MlxArray>,
    },
}

impl SwitchLinear {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => unsafe {
                mlxcel_core::gather_qmm(
                    x,
                    weight,
                    scales,
                    biases
                        .as_ref()
                        .map(|b| b as *const _)
                        .unwrap_or(std::ptr::null()),
                    std::ptr::null(),
                    indices as *const _,
                    true,
                    *group_size,
                    *bits,
                    false,
                    "affine",
                )
            },
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(x, &wt, std::ptr::null(), indices as *const _, false)
                }
            }
        }
    }
}

struct SwitchGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGLU {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        // Expand x for MoE: [B, L, hidden] -> [B, L, 1, 1, hidden]
        let x_expanded = mlxcel_core::expand_dims(x, -2);
        let x_expanded = mlxcel_core::expand_dims(&x_expanded, -3);

        let gate = self.gate_proj.forward(&x_expanded, indices);
        let gate = silu(&gate);
        let up = self.up_proj.forward(&x_expanded, indices);
        let h = mlxcel_core::multiply(&gate, &up);
        let out = self.down_proj.forward(&h, indices);

        // Squeeze the second-to-last dim: [B, L, k, 1, D] -> [B, L, k, D]
        mlxcel_core::squeeze_axis(&out, -2)
    }
}

struct MoEBlock {
    gate: MoEGate,
    experts: SwitchGLU,
    shared_experts: Option<DenseMLP>,
}

impl MoEBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (indices, scores) = self.gate.forward(x);
        let y = self.experts.forward(x, &indices);

        // Weighted sum over experts: [B, L, k, D] * [B, L, k, 1] -> sum over k -> [B, L, D]
        let scores_expanded = mlxcel_core::expand_dims(&scores, -1);
        let weighted = mlxcel_core::multiply(&y, &scores_expanded);
        let mut result = mlxcel_core::sum_axis(&weighted, -2, false);

        // Add shared experts if present
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x);
            result = mlxcel_core::add(&result, &shared_out);
        }

        result
    }
}

enum MLPType {
    Dense(DenseMLP),
    MoE(MoEBlock),
}

impl MLPType {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MLPType::Dense(mlp) => mlp.forward(x),
            MLPType::MoE(moe) => moe.forward(x),
        }
    }
}

// Decoder Layer.
struct DecoderLayer {
    self_attn: MLAAttention,
    mlp: MLPType,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let r = self.self_attn.forward(&normed, mask, cache);
        let h = mlxcel_core::add(x, &r);

        let normed = self.post_attention_layernorm.forward(&h);
        let r = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &r)
    }
}

// DeepSeek V2 Model.
pub struct DeepSeekV2Model {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
}

impl DeepSeekV2Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        let mask = mask.map(mlxcel_core::copy);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask.as_deref(), cache);
        }

        let h = self.norm.forward(&h);

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward from pre-computed input embeddings (VLM feature splicing), the
    /// same path as [`Self::forward`] with the token-embedding lookup skipped.
    pub fn forward_from_embeds(
        &self,
        inputs_embeds: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = mlxcel_core::copy(inputs_embeds);
        let mask = mask.map(mlxcel_core::copy);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask.as_deref(), cache);
        }
        let h = self.norm.forward(&h);
        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    /// Raw token-embedding lookup, exposed for VLM feature splicing.
    pub fn embed_tokens_forward(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = load_decoder_layer(weights, args, i)?;
            layers.push(layer);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        let lm_head = if !args.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}

// Weight Loading Helpers.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

fn load_decoder_layer(
    weights: &WeightMap,
    args: &ModelArgs,
    layer_idx: usize,
) -> Result<DecoderLayer, String> {
    let prefix = format!("model.layers.{}", layer_idx);
    let group_size = args.group_size();
    let bits = args.bits();

    let self_attn = load_mla_attention(weights, args, &format!("{}.self_attn", prefix))?;

    let mlp = if args.is_moe_layer(layer_idx) {
        MLPType::MoE(load_moe_block(weights, args, &format!("{}.mlp", prefix))?)
    } else {
        MLPType::Dense(load_dense_mlp(
            weights,
            &format!("{}.mlp", prefix),
            group_size,
            bits,
        )?)
    };

    let input_layernorm_weight =
        get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
    let post_attention_layernorm_weight = get_weight_copy(
        weights,
        &format!("{}.post_attention_layernorm.weight", prefix),
    )?;

    Ok(DecoderLayer {
        self_attn,
        mlp,
        input_layernorm: RMSNorm::new(input_layernorm_weight, args.rms_norm_eps),
        post_attention_layernorm: RMSNorm::new(post_attention_layernorm_weight, args.rms_norm_eps),
    })
}

fn load_mla_attention(
    weights: &WeightMap,
    args: &ModelArgs,
    prefix: &str,
) -> Result<MLAAttention, String> {
    let group_size = args.group_size();
    let bits = args.bits();
    let q_head_dim = args.q_head_dim();

    // Determine Q projection type
    let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = if args.q_lora_rank.is_none() {
        (
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{}.q_proj", prefix),
                group_size,
                bits,
            )?),
            None,
            None,
            None,
        )
    } else {
        (
            None,
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{}.q_a_proj", prefix),
                group_size,
                bits,
            )?),
            Some(RMSNorm::new(
                get_weight_copy(weights, &format!("{}.q_a_layernorm.weight", prefix))?,
                1e-6,
            )),
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{}.q_b_proj", prefix),
                group_size,
                bits,
            )?),
        )
    };

    // Compute scale with mscale adjustment
    let mut scale = (q_head_dim as f32).powf(-0.5);
    if args.mscale_all_dim() > 0.0 {
        let ms = yarn_get_mscale(args.scaling_factor(), args.mscale_all_dim());
        scale *= ms * ms;
    }

    Ok(MLAAttention {
        q_proj,
        q_a_proj,
        q_a_layernorm,
        q_b_proj,
        kv_a_proj_with_mqa: UnifiedLinear::from_weights(
            weights,
            &format!("{}.kv_a_proj_with_mqa", prefix),
            group_size,
            bits,
        )?,
        kv_a_layernorm: RMSNorm::new(
            get_weight_copy(weights, &format!("{}.kv_a_layernorm.weight", prefix))?,
            1e-6,
        ),
        kv_b_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.kv_b_proj", prefix),
            group_size,
            bits,
        )?,
        o_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.o_proj", prefix),
            group_size,
            bits,
        )?,
        num_heads: args.num_attention_heads,
        kv_lora_rank: args.kv_lora_rank,
        qk_nope_head_dim: args.qk_nope_head_dim,
        qk_rope_head_dim: args.qk_rope_head_dim,
        v_head_dim: args.v_head_dim,
        q_head_dim,
        scale,
        rope: YarnRoPE::new(
            args.qk_rope_head_dim,
            args.rope_theta,
            args.scaling_factor(),
            args.original_max_pos(),
            args.beta_fast(),
            args.beta_slow(),
            args.mscale(),
            args.mscale_all_dim(),
        ),
    })
}

fn load_dense_mlp(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<DenseMLP, String> {
    Ok(DenseMLP {
        gate_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?,
        up_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.up_proj", prefix),
            group_size,
            bits,
        )?,
        down_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?,
    })
}

fn load_moe_block(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<MoEBlock, String> {
    let group_size = args.group_size();
    let bits = args.bits();
    let n_routed = args.n_routed_experts.unwrap_or(1);

    let gate_weight = get_weight_copy(weights, &format!("{}.gate.weight", prefix))?;
    let gate = MoEGate {
        weight: gate_weight,
        top_k: args.num_experts_per_tok.unwrap_or(1),
        routed_scaling_factor: args.routed_scaling_factor,
    };

    let experts = load_switch_glu(
        weights,
        &format!("{}.switch_mlp", prefix),
        n_routed,
        group_size,
        bits,
    )?;

    let shared_experts = if let Some(n_shared) = args.n_shared_experts {
        let _intermediate_size = args.moe_intermediate_size * n_shared;
        Some(DenseMLP {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.down_proj", prefix),
                group_size,
                bits,
            )?,
        })
    } else {
        None
    };

    Ok(MoEBlock {
        gate,
        experts,
        shared_experts,
    })
}

fn load_switch_glu(
    weights: &WeightMap,
    prefix: &str,
    num_experts: usize,
    group_size: i32,
    bits: i32,
) -> Result<SwitchGLU, String> {
    Ok(SwitchGLU {
        gate_proj: load_switch_linear(weights, num_experts, prefix, "gate_proj", group_size, bits)?,
        up_proj: load_switch_linear(weights, num_experts, prefix, "up_proj", group_size, bits)?,
        down_proj: load_switch_linear(weights, num_experts, prefix, "down_proj", group_size, bits)?,
    })
}

fn load_switch_linear(
    weights: &WeightMap,
    _num_experts: usize,
    prefix: &str,
    weight_name: &str,
    group_size: i32,
    bits: i32,
) -> Result<SwitchLinear, String> {
    // Load fused 3D tensor directly: [num_experts, out_features, in_features/pack_factor]
    let weight = get_weight_copy(weights, &format!("{}.{}.weight", prefix, weight_name))?;
    let scales_key = format!("{}.{}.scales", prefix, weight_name);
    if weights.contains_key(&scales_key) {
        let scales = mlxcel_core::copy(weights.get(&scales_key).unwrap());
        let biases = get_weight_copy(weights, &format!("{}.{}.biases", prefix, weight_name))?;
        Ok(SwitchLinear::Quantized {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    } else {
        Ok(SwitchLinear::Regular { weight })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for DeepSeekV2Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        DeepSeekV2Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        DeepSeekV2Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![100001] // <|end▁of▁sentence|>
    }
}
