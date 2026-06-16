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

//! Qwen2 MoE model implementation using mlxcel-core
//!
//! Qwen2 MoE features:
//! - MoE architecture with SwitchGLU experts
//! - Shared expert with gating (sigmoid)
//! - attention_bias=true for Q, K, V projections (not O)
//! - RMSNorm for layer normalization
//! - Standard RoPE with high theta (1M)

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
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
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_traditional: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub group_size: Option<i32>,

    #[serde(default)]
    pub bits: Option<i32>,

    #[serde(default)]
    pub head_dim: Option<usize>,
}

fn default_rope_theta() -> f32 {
    1000000.0
}

impl ModelArgs {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn group_size(&self) -> i32 {
        self.group_size.unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.bits.unwrap_or(4)
    }
}

// Attention with Q, K, V bias.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub rope_traditional: bool,
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

        // Project Q, K, V (with bias support)
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

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
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0, // scale
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0, // scale
            offset,
        );

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            // Single token or explicit mask
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

        // Output projection (no bias)
        self.o_proj.forward(&attn_out)
    }
}

// Shared Expert MLP.
/// Standard MLP with SwiGLU activation for shared expert
pub struct SharedExpertMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl SharedExpertMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // SwiGLU: silu(gate) * up
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }
}

// Sparse MoE Block with Shared Expert.
/// Qwen2 MoE layer with sparse experts and shared expert
pub struct SparseMoeBlock {
    pub router: UnifiedLinear,
    pub experts: crate::models::switch_layers::SwitchGLU,
    pub shared_expert: SharedExpertMLP,
    pub shared_expert_gate: UnifiedLinear,
    pub num_experts: usize,
    pub top_k: usize,
}

impl SparseMoeBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        // Flatten to [n_tokens, hidden]
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Get router logits and apply softmax
        let logits = self.router.forward(&x_flat);
        let gates = mlxcel_core::softmax(&logits, -1);

        // Top-k selection
        let k = self.top_k as i32;

        // Get top-k expert indices using argpartition
        let neg_gates = mlxcel_core::negative(&gates);
        let indices = mlxcel_core::argpartition(&neg_gates, k - 1, -1);

        // Slice to get top-k: indices[..., :k]
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices = mlxcel_core::slice(&indices, &[0, 0], &[indices_shape[0], k]);

        // Get corresponding scores
        let scores = mlxcel_core::take_along_axis(&gates, &topk_indices, -1);

        // Apply routed experts. Fused single-token decode kernel (#268) on by
        // default; MLXCEL_FUSED_MOE=0 forces the proven SwitchGLU +
        // moe_weighted_sum path (also the automatic fallback when the kernel does
        // not support the config). The shared-expert path below is unchanged.
        let expert_out_sum = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1
                && crate::models::switch_layers::fused_moe_enabled()
            {
                self.experts
                    .forward_fused_kernel(&x_flat, &topk_indices, &scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.experts.forward(&x_flat, &topk_indices);
                    crate::models::switch_layers::moe_weighted_sum(
                        &expert_out,
                        &scores,
                        mlxcel_core::array_dtype(&x_flat),
                    )
                }
            }
        };

        // Compute shared expert output
        let shared_output = self.shared_expert.forward(&x_flat);

        // Compute shared expert gate (sigmoid activation)
        let shared_gate_logits = self.shared_expert_gate.forward(&x_flat);
        let shared_gate = mlxcel_core::sigmoid(&shared_gate_logits);

        // Weighted shared expert: sigmoid(gate) * shared_expert(x)
        let weighted_shared = mlxcel_core::multiply(&shared_gate, &shared_output);

        // Final output: expert_out + shared_out
        let result = mlxcel_core::add(&expert_out_sum, &weighted_shared);

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// Transformer Block.
/// Qwen2 MoE transformer block
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: SparseMoeBlock,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
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

        // Pre-norm MoE
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// Full Model.
/// Qwen2 MoE model
pub struct Qwen2MoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl Qwen2MoeModel {
    /// Forward pass through the entire model
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        self.lm_head.forward(&h)
    }

    /// Load model from a directory containing safetensors files and config.json
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

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        // Load quantized embedding
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            "model.embed_tokens",
            args.group_size(),
            args.bits(),
        )?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
        let lm_head = load_quantized_linear(weights, "lm_head", args)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<KVCache> {
        self.layers.iter().map(|_| KVCache::new()).collect()
    }
}

impl TransformerBlock {
    /// Load transformer block from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Load attention
        let self_attn = Attention::from_weights(weights, args, &prefix)?;

        // Load MoE
        let mlp = SparseMoeBlock::from_weights(weights, args, &prefix)?;

        // Load norms
        let input_layernorm = RMSNorm::new(
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?,
            args.rms_norm_eps,
        );
        let post_attention_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{}.post_attention_layernorm.weight", prefix),
            )?,
            args.rms_norm_eps,
        );

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl Attention {
    /// Load attention from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let attn_prefix = format!("{}.self_attn", prefix);

        // Q, K, V projections have bias
        let q_proj = load_quantized_linear(weights, &format!("{}.q_proj", attn_prefix), args)?;
        let k_proj = load_quantized_linear(weights, &format!("{}.k_proj", attn_prefix), args)?;
        let v_proj = load_quantized_linear(weights, &format!("{}.v_proj", attn_prefix), args)?;
        // O projection has no bias
        let o_proj = load_quantized_linear(weights, &format!("{}.o_proj", attn_prefix), args)?;

        let head_dim = args.head_dim();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim: head_dim as i32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim as i32,
            rope_base: args.rope_theta,
            rope_traditional: args.rope_traditional,
        })
    }
}

impl SharedExpertMLP {
    /// Load shared expert MLP from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: load_quantized_linear(weights, &format!("{}.gate_proj", prefix), args)?,
            up_proj: load_quantized_linear(weights, &format!("{}.up_proj", prefix), args)?,
            down_proj: load_quantized_linear(weights, &format!("{}.down_proj", prefix), args)?,
        })
    }
}

impl SparseMoeBlock {
    /// Load MoE block from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let moe_prefix = format!("{}.mlp", prefix);

        let router = load_quantized_linear(weights, &format!("{}.gate", moe_prefix), args)?;
        let experts = crate::models::switch_layers::SwitchGLU::from_weights(
            weights,
            &format!("{}.switch_mlp", moe_prefix),
            args.group_size(),
            args.bits(),
        )?;

        // Load shared expert
        let shared_expert =
            SharedExpertMLP::from_weights(weights, args, &format!("{}.shared_expert", moe_prefix))?;

        // Load shared expert gate (single output, sigmoid activation)
        let shared_expert_gate =
            load_quantized_linear(weights, &format!("{}.shared_expert_gate", moe_prefix), args)?;

        Ok(Self {
            router,
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts: args.num_experts,
            top_k: args.num_experts_per_tok,
        })
    }
}

// Helper functions for weight loading.
/// Get a copy of a weight from the weight map
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

/// Load a quantized linear layer from weights (falls back to Linear for non-quantized)
fn load_quantized_linear(
    weights: &WeightMap,
    prefix: &str,
    args: &ModelArgs,
) -> Result<UnifiedLinear, String> {
    UnifiedLinear::from_weights(weights, prefix, args.group_size(), args.bits())
}

// LanguageModel trait implementation.
impl LanguageModel for Qwen2MoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Qwen2MoeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Qwen2MoeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Qwen2 EOS token
        vec![151643]
    }
}
