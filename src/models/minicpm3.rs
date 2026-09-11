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

//! MiniCPM3 model implementation using mlxcel-core
//!
//! Key features:
//! - MLA (Multi-head Latent Attention) with q_lora_rank and kv_lora_rank
//! - Split head dimensions: qk_nope_head_dim + qk_rope_head_dim
//! - SuScaledRoPE for position encoding
//! - Embedding, residual, and output scaling (like MiniCPM)

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, repeat_kv, silu, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub dim_model_base: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,

    // MLA parameters
    pub q_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub kv_lora_rank: usize,

    // Scaling factors
    pub scale_depth: f32,
    pub scale_emb: f32,
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

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

fn default_rope_theta() -> f32 {
    1000000.0
}

impl ModelArgs {
    pub fn v_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn residual_scale(&self) -> f32 {
        self.scale_depth / (self.num_hidden_layers as f32).sqrt()
    }

    pub fn output_scale(&self) -> f32 {
        self.hidden_size as f32 / self.dim_model_base as f32
    }

    pub fn original_max_pos(&self) -> usize {
        self.rope_scaling
            .as_ref()
            .and_then(|s| s.get("original_max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4096)
    }

    pub fn long_factor(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|s| s.get("long_factor"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1.0)
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
}

// SuScaledRoPE - Su-scaled rotary position embedding.
struct SuScaledRoPE {
    scaled_base: f32,
    mscale: f32,
    dims: i32,
}

impl SuScaledRoPE {
    fn new(
        dims: usize,
        base: f32,
        max_pos: usize,
        original_max_pos: usize,
        long_factor: f32,
    ) -> Self {
        let factor = max_pos as f32 / original_max_pos as f32;
        let scaled_base = base * long_factor;

        let mscale = if factor <= 1.0 {
            1.0
        } else {
            (1.0 + (factor.ln() / (original_max_pos as f32).ln())).sqrt()
        };

        Self {
            scaled_base,
            mscale,
            dims: dims as i32,
        }
    }

    fn forward(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        mlxcel_core::fast_rope(x, self.dims, false, self.scaled_base, self.mscale, offset)
    }
}

// MLA Attention.
#[allow(dead_code)]
struct MLAAttention {
    // Query path: x -> q_a_proj -> q_a_layernorm -> q_b_proj
    q_a_proj: UnifiedLinear,
    q_a_layernorm: RMSNorm,
    q_b_proj: UnifiedLinear,

    // KV path: x -> kv_a_proj_with_mqa -> split (compressed_kv, k_pe)
    //              -> kv_a_layernorm(compressed_kv) -> kv_b_proj
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

    rope: SuScaledRoPE,
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

        // Query path: x -> q_a_proj -> q_a_layernorm -> q_b_proj -> reshape
        let q = self.q_a_proj.forward(x);
        let q = self.q_a_layernorm.forward(&q);
        let q = self.q_b_proj.forward(&q);
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads as i32, self.q_head_dim as i32]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Split query into nope and pe parts
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim as i32);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim as i32, -1);

        // KV path: x -> kv_a_proj_with_mqa -> split (compressed_kv, k_pe)
        let kv_compressed = self.kv_a_proj_with_mqa.forward(x);
        let compressed_kv = slice_axis(&kv_compressed, -1, 0, self.kv_lora_rank as i32);
        let k_pe = slice_axis(&kv_compressed, -1, self.kv_lora_rank as i32, -1);

        // Reshape k_pe: [B, L, rope_dim] -> [B, 1, L, rope_dim]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim as i32]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // kv_a_layernorm(compressed_kv) -> kv_b_proj -> reshape -> split
        let kv = self.kv_a_layernorm.forward(&compressed_kv);
        let kv = self.kv_b_proj.forward(&kv);
        let kv = mlxcel_core::reshape(&kv, &[b, l, self.num_heads as i32, -1]);
        let kv = mlxcel_core::transpose_axes(&kv, &[0, 2, 1, 3]);

        // Split kv into k_nope and values
        let k_nope = slice_axis(&kv, -1, 0, self.qk_nope_head_dim as i32);
        let values = slice_axis(&kv, -1, self.qk_nope_head_dim as i32, -1);

        // Apply RoPE to q_pe and k_pe
        let offset = cache.seq_len();
        let q_pe = self.rope.forward(&q_pe, offset);
        let k_pe = self.rope.forward(&k_pe, offset);

        // Broadcast k_pe to all heads: [B, 1, L, rope_dim] -> [B, H, L, rope_dim]
        let k_pe = repeat_kv(&k_pe, self.num_heads as i32);

        // Concatenate: queries = [q_nope, q_pe], keys = [k_nope, k_pe]
        let queries = concatenate(&q_nope, &q_pe, -1);
        let keys = concatenate(&k_nope, &k_pe, -1);

        // Update cache
        let (keys, values) = cache.update_and_fetch(keys, values);

        // Scaled dot product attention
        let output = if l > 1 {
            // Prefill: use mask
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &queries, &keys, &values, self.scale, mask_ptr, 0.0, 0,
                )
            }
        } else {
            // Single token: use causal SDPA (no mask needed)
            mlxcel_core::causal_attention(&queries, &keys, &values, self.scale, 0.0, 0)
        };

        // Reshape output: [B, H, L, D] -> [B, L, H*D]
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);

        self.o_proj.forward(&output)
    }
}

// MLP.
struct MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        let h = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&h)
    }
}

// Decoder Layer.
struct DecoderLayer {
    self_attn: MLAAttention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    residual_scale: f32,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        // r = self_attn(input_layernorm(x))
        // h = x + r * residual_scale
        let normed = self.input_layernorm.forward(x);
        let r = self.self_attn.forward(&normed, mask, cache);
        let r_scaled = mlxcel_core::multiply_scalar(&r, self.residual_scale);
        let h = mlxcel_core::add(x, &r_scaled);

        // r = mlp(post_attention_layernorm(h))
        // out = h + r * residual_scale
        let normed = self.post_attention_layernorm.forward(&h);
        let r = self.mlp.forward(&normed);
        let r_scaled = mlxcel_core::multiply_scalar(&r, self.residual_scale);
        mlxcel_core::add(&h, &r_scaled)
    }
}

// MiniCPM3 Model.
pub struct MiniCPM3Model {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    scale_emb: f32,
    output_scale: f32,
}

impl MiniCPM3Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed with scaling
        let h = self.embed_tokens.forward(input_ids);
        let mut h = mlxcel_core::multiply_scalar(&h, self.scale_emb);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Create causal mask if needed
        let mask = if seq_len > 1 {
            let offset = caches.first().map(|c| c.seq_len()).unwrap_or(0);
            Some(create_causal_mask(seq_len, offset))
        } else {
            mask.map(mlxcel_core::copy)
        };

        // Forward through layers
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask.as_deref(), cache);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // Apply output scaling (inverse)
        let h = if self.output_scale != 1.0 {
            mlxcel_core::divide_scalar(&h, self.output_scale)
        } else {
            h
        };

        // LM head
        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = load_decoder_layer(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
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
            scale_emb: args.scale_emb,
            output_scale: args.output_scale(),
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

    // Load attention
    let self_attn = load_mla_attention(weights, args, &format!("{}.self_attn", prefix))?;

    // Load MLP
    let mlp = MLP {
        gate_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.mlp.gate_proj", prefix),
            group_size,
            bits,
        )?,
        up_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.mlp.up_proj", prefix),
            group_size,
            bits,
        )?,
        down_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.mlp.down_proj", prefix),
            group_size,
            bits,
        )?,
    };

    // Load norms
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
        residual_scale: args.residual_scale(),
    })
}

fn load_mla_attention(
    weights: &WeightMap,
    args: &ModelArgs,
    prefix: &str,
) -> Result<MLAAttention, String> {
    let group_size = args.group_size();
    let bits = args.bits();

    Ok(MLAAttention {
        q_a_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.q_a_proj", prefix),
            group_size,
            bits,
        )?,
        q_a_layernorm: RMSNorm::new(
            get_weight_copy(weights, &format!("{}.q_a_layernorm.weight", prefix))?,
            1e-6,
        ),
        q_b_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.q_b_proj", prefix),
            group_size,
            bits,
        )?,
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
        v_head_dim: args.v_head_dim(),
        q_head_dim: args.q_head_dim(),
        scale: (args.q_head_dim() as f32).powf(-0.5),
        rope: SuScaledRoPE::new(
            args.qk_rope_head_dim,
            args.rope_theta,
            args.max_position_embeddings,
            args.original_max_pos(),
            args.long_factor(),
        ),
    })
}

// LanguageModel trait implementation.
impl LanguageModel for MiniCPM3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        MiniCPM3Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        MiniCPM3Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![2] // </s>
    }
}
