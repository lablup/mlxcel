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

//! OLMoE model implementation using mlxcel-core
//!
//! Key features:
//! - Q/K normalization (RMSNorm after projection, before RoPE)
//! - Sparse MoE with SwitchGLU and top-k routing
//! - Full softmax over all experts, then gather the probabilities at the top-k
//! - Optional norm_topk_prob renormalization of the gathered top-k scores
//! - Standard RoPE positional embeddings

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
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,

    #[serde(default)]
    pub norm_topk_prob: bool,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default)]
    pub rope_traditional: bool,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_tie_word_embeddings() -> bool {
    true
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
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }
}

// Sparse MoE Block.
/// OLMoE sparse mixture of experts layer
pub struct SparseMoeBlock {
    pub router: UnifiedLinear,
    pub experts: crate::models::switch_layers::SwitchGLU,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
}

/// Compute the top-k expert indices and their router scores from raw router
/// logits, matching ml-explore/mlx-lm OlmoeSparseMoeBlock
/// (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/olmoe.py).
///
/// Order matters. mlx-lm softmaxes over ALL experts FIRST (precise f32
/// accumulation), then gathers those full-softmax probabilities at the top-k
/// experts, then renormalizes only when `norm_topk_prob` is set. Softmaxing
/// over only the top-k logits instead always sums to 1, i.e. it silently
/// behaves as if `norm_topk_prob` were always true. OLMoE-1B-7B-0125 ships
/// `norm_topk_prob = false`, so its top-k probabilities must sum to < 1 and stay
/// un-normalized; the top-k-only softmax over-weights every MoE block and the
/// error compounds with depth and generation length (issue #318).
///
/// Expert selection is by argpartition on the raw logits. softmax is monotonic,
/// so this yields the same expert set as mlx-lm's
/// `argpartition(-routing_weights)[..., :k]`; the indices are unchanged from the
/// previous implementation and only the scores differ. Returns
/// `(topk_indices, scores)`, both shaped `[n_tokens, k]` and aligned so that
/// `scores[t, j]` is the weight for expert `topk_indices[t, j]`.
fn router_topk_scores(
    logits: &MlxArray,
    k: i32,
    norm_topk_prob: bool,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let n_experts = mlxcel_core::array_shape(logits)[1];
    let kth = n_experts - k;

    // Top-k selection: indices[..., kth:] after argpartition on the logits.
    let indices = mlxcel_core::argpartition(logits, kth, -1);
    let indices_shape = mlxcel_core::array_shape(&indices);
    let topk_indices =
        mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

    // Full softmax over ALL experts first (precise/f32 accumulation), then gather
    // the probabilities at the selected experts. This is NOT a fresh softmax over
    // only the top-k logits.
    let routing_weights = mlxcel_core::softmax_precise(logits, -1);
    let mut scores = mlxcel_core::take_along_axis(&routing_weights, &topk_indices, -1);

    // Only renormalize when the config requests it (false for OLMoE-1B-7B-0125).
    if norm_topk_prob {
        let sum = mlxcel_core::sum_axis(&scores, -1, true);
        scores = mlxcel_core::divide(&scores, &sum);
    }

    (topk_indices, scores)
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

        // Router logits and top-k expert routing scores. The scoring matches
        // ml-explore/mlx-lm OlmoeSparseMoeBlock exactly: softmax over all experts
        // first, then gather the full-softmax probabilities at the top-k, then
        // renormalize only when norm_topk_prob is set. See router_topk_scores for
        // why this differs from a top-k-only softmax when norm_topk_prob is false
        // (issue #318). Both the fused-kernel path and the SwitchGLU +
        // moe_weighted_sum fallback below consume the same indices and scores.
        let logits = self.router.forward(&x_flat);
        let k = self.num_experts_per_tok as i32;
        let (topk_indices, scores) = router_topk_scores(&logits, k, self.norm_topk_prob);

        // Apply experts and weighted-sum. Fused single-token decode kernel
        // on by default; MLXCEL_FUSED_MOE=0 forces the proven SwitchGLU +
        // moe_weighted_sum path (also the automatic fallback when the kernel
        // does not support the config).
        let result = {
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

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// Attention with Q/K Normalization.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm, // Q normalization
    pub k_norm: RMSNorm, // K normalization
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

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Apply Q/K normalization on the full projection BEFORE the head reshape.
        // OLMoE normalizes over the whole num_heads*head_dim projection (the norm
        // weight is hidden-sized), matching the upstream order; normalizing after
        // the reshape would size-mismatch the weight against head_dim.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE AFTER normalization
        q = mlxcel_core::fast_rope(
            &q,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0, // scale
            offset,
        );
        k = mlxcel_core::fast_rope(
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

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        // Load Q/K normalization weights
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_weight, args.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, args.rms_norm_eps),
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
            rope_traditional: args.rope_traditional,
        })
    }
}

// Transformer Block.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: SparseMoeBlock,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
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
        let moe_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &moe_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = SparseMoeBlock::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// OLMoE Model.
pub struct OlmoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub tie_word_embeddings: bool,
}

impl OlmoeModel {
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
        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Load model from directory
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

        // Sanitize weights (join expert weights if separate)
        let weights = sanitize_weights(weights, &args);

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = DecoderLayer::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head (or use tied embeddings)
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tie_word_embeddings: args.tie_word_embeddings,
        })
    }
}

// MoE Implementation Details.
impl SparseMoeBlock {
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate", prefix),
            args.group_size(),
            args.bits(),
        )?;

        let experts = crate::models::switch_layers::SwitchGLU::from_weights(
            weights,
            &format!("{}.switch_mlp", prefix),
            args.group_size(),
            args.bits(),
        )?;

        Ok(Self {
            router,
            experts,
            num_experts_per_tok: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
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

/// Sanitize weights by joining separate expert weights into stacked format
fn sanitize_weights(mut weights: WeightMap, args: &ModelArgs) -> WeightMap {
    // Check if weights have separate expert format (e.g., mlp.experts.0.up_proj.weight)
    let check_key = "model.layers.0.mlp.experts.0.up_proj.weight";
    if !weights.contains_key(check_key) {
        return weights;
    }

    // Join expert weights into stacked format
    for l in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{}", l);
        for proj_name in &["up_proj", "down_proj", "gate_proj"] {
            for weight_name in &["weight", "scales", "biases"] {
                let first_key = format!("{}.mlp.experts.0.{}.{}", prefix, proj_name, weight_name);
                if weights.contains_key(&first_key) {
                    let mut expert_arrays = Vec::new();
                    for e in 0..args.num_experts {
                        let key =
                            format!("{}.mlp.experts.{}.{}.{}", prefix, e, proj_name, weight_name);
                        if let Some(w) = weights.remove(&key) {
                            expert_arrays.push(w);
                        }
                    }
                    if !expert_arrays.is_empty() {
                        let expert_ptrs: Vec<*const MlxArray> = expert_arrays
                            .iter()
                            .map(|a| a.as_ref().unwrap() as *const _)
                            .collect();
                        let stacked = mlxcel_core::stack(&expert_ptrs, 0);
                        let target_key =
                            format!("{}.mlp.switch_mlp.{}.{}", prefix, proj_name, weight_name);
                        weights.insert(target_key, stacked);
                    }
                }
            }
        }
    }

    weights
}

// LanguageModel trait implementation.
impl LanguageModel for OlmoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        OlmoeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        OlmoeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // OLMoE uses GPT-2/OLMo tokenizer with EOS token id 50279
        vec![50279]
    }
}

#[cfg(test)]
#[path = "olmoe_tests.rs"]
mod tests;
