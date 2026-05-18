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

//! Gemma v1 model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Uses GELU activation instead of SiLU in MLP
//! - RMSNorm with (1.0 + weight) instead of just weight
//! - Embedding scale: h = h * sqrt(hidden_size)
//! - Always uses tied word embeddings

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{FusedQKVLinear, KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::pipeline_hint;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_traditional: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
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

// Gemma RMSNorm (uses 1.0 + weight).
pub struct GemmaRMSNorm {
    pub weight: UniquePtr<MlxArray>,
    /// Pre-computed (1 + weight) to avoid per-forward allocation
    adjusted_weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl GemmaRMSNorm {
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&weight));
        let adjusted_weight = mlxcel_core::add(&one, &weight);
        Self {
            weight,
            adjusted_weight,
            eps,
        }
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::fast_rms_norm(x, &self.adjusted_weight, self.eps)
    }
}

// Attention (standard, same as Llama).
// Uses FusedQKVLinear: Q, K, V weights are concatenated at load time
// into a single [q_dim+k_dim+v_dim, hidden_dim] weight matrix,
// enabling a single matmul instead of 3 separate ones per forward pass.
pub struct Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Fused QKV projection: single matmul → split into Q, K, V
        let (q, k, v) = self.qkv_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE
        let q = mlxcel_core::fast_rope(
            &q,
            self.head_dim,
            false, // not traditional
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        // Update KV cache
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = args.head_dim as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_key_value_heads as i32;

        // Fuse Q/K/V into a single projection: concatenate along output dim at load time.
        let qkv_proj = FusedQKVLinear::from_weights_separate(
            weights,
            prefix,
            group_size,
            bits,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base: args.rope_theta,
        })
    }
}

// MLP (uses tanh-approx GELU instead of SiLU).
pub struct GemmaMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl GemmaMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // GeGLU: gelu_pytorch_tanh(gate_proj(x)) * up_proj(x), then down_proj.
        // Gemma v1 configs use `hidden_activation = gelu_pytorch_tanh`.
        // Quantized path: fused compiled quantized MLP
        if let (Some(gate_qw), Some(up_qw), Some(down_qw)) = (
            self.gate_proj.quantized_weight(),
            self.up_proj.quantized_weight(),
            self.down_proj.quantized_weight(),
        ) {
            return unsafe {
                mlxcel_core::compiled_gelu_approx_mlp_forward(
                    x,
                    &gate_qw.weight,
                    &gate_qw.scales,
                    gate_qw.biases_ptr(),
                    &up_qw.weight,
                    &up_qw.scales,
                    up_qw.biases_ptr(),
                    &down_qw.weight,
                    &down_qw.scales,
                    down_qw.biases_ptr(),
                    gate_qw.group_size,
                    gate_qw.bits,
                    &gate_qw.mode,
                )
            };
        }

        // Non-quantized path: fused compiled FP MLP
        if let Some(result) = mlxcel_core::layers::compiled_gelu_mlp_fp16(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return result;
        }

        // Fallback: separate operations with compiled activation
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

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

// Transformer Block.
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: GemmaMLP,
    pub input_layernorm: GemmaRMSNorm,
    pub post_attention_layernorm: GemmaRMSNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm FFN
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = GemmaMLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = GemmaRMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = GemmaRMSNorm::new(post_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Gemma Model.
pub struct GemmaModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: GemmaRMSNorm,
    pub hidden_size: usize,
}

impl GemmaModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Gemma scales embeddings by sqrt(hidden_size)
        h = mlxcel_core::multiply_scalar(&h, (self.hidden_size as f32).sqrt());

        // Pass through transformer layers
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            pipeline_hint(&h, i, n);
        }

        // Final norm
        self.norm.forward(&h)
    }

    /// Get token embeddings with Gemma scaling (for VLM merge)
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        mlxcel_core::multiply_scalar(&h, (self.hidden_size as f32).sqrt())
    }

    /// Forward with pre-computed embeddings (for VLM prefill)
    pub fn forward_with_embeddings_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.get_embed_tokens(input_ids)
        };

        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            pipeline_hint(&h, i, n);
        }

        let h = self.norm.forward(&h);
        self.embed_tokens.as_linear(&h)
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
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = GemmaRMSNorm::new(norm_weight, args.rms_norm_eps);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            hidden_size: args.hidden_size,
        })
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for GemmaModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = GemmaModel::forward(self, input_ids, caches, mask);

        // Gemma always uses tied embeddings (embed_tokens as lm_head)
        self.embed_tokens.as_linear(&h)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        GemmaModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Gemma EOS token
        vec![1, 107] // <eos>, <end_of_turn>
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        true
    }
}
