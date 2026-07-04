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

//! DeepSeek v1 MoE model implementation using mlxcel-core
//!
//! DeepSeek v1 MoE features:
//! - Standard transformer attention with RoPE
//! - MoE layers with optional shared experts
//! - Dense MLP layers for first_k_dense_replace layers
//! - MoE layers at moe_layer_freq interval
//! - routed_scaling_factor for expert outputs (default 1.0)
//! - Top-k routing with softmax scoring

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
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    // DeepSeek-OCR's `language_config` omits both; supply the family defaults.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    // MoE parameters
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub group_size: Option<i32>,

    #[serde(default)]
    pub bits: Option<i32>,
}

fn default_moe_layer_freq() -> usize {
    1
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn group_size(&self) -> i32 {
        self.group_size.unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.bits.unwrap_or(4)
    }
}

// SwitchLinear: Stacked expert weights for MoE.
/// Stacked linear layers for MoE experts
/// Weights shape: [num_experts, output_dim, input_dim_packed]
/// Supports both quantized (gather_qmm) and non-quantized (gather_mm) forward paths.
pub enum SwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
        num_experts: usize,
    },
    Regular {
        weight: UniquePtr<MlxArray>,
        num_experts: usize,
    },
}

impl SwitchLinear {
    /// Return the number of experts this layer holds.
    pub fn num_experts(&self) -> usize {
        match self {
            Self::Quantized { num_experts, .. } => *num_experts,
            Self::Regular { num_experts, .. } => *num_experts,
        }
    }

    /// Forward pass: gather_qmm for quantized weights, gather_mm for regular weights.
    /// x: [n_tokens, 1, hidden] or [n_sorted, 1, hidden]
    /// indices: [n_tokens, top_k] or [n_sorted] (flattened when sorted)
    pub fn forward(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        sorted_indices: bool,
    ) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                ..
            } => {
                // SAFETY: weight/scales/biases are valid UniquePtr-owned MlxArray values.
                unsafe {
                    mlxcel_core::gather_qmm(
                        x,
                        weight,
                        scales,
                        &**biases as *const MlxArray,
                        std::ptr::null(), // lhs_indices
                        indices as *const _,
                        true, // transpose
                        *group_size,
                        *bits,
                        sorted_indices,
                        "affine",
                    )
                }
            }
            Self::Regular { weight, .. } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                // SAFETY: wt and indices are valid MlxArray values in scope.
                unsafe {
                    mlxcel_core::gather_mm(
                        x,
                        &wt,
                        std::ptr::null(),
                        indices as *const _,
                        sorted_indices,
                    )
                }
            }
        }
    }
}

// SwitchGLU: SwiGLU with stacked expert weights.
/// SwitchGLU: SwiGLU activation with stacked expert weights for MoE
pub struct SwitchGLU {
    pub gate_proj: SwitchLinear,
    pub up_proj: SwitchLinear,
    pub down_proj: SwitchLinear,
}

impl SwitchGLU {
    /// Forward pass with kernel-fused SwiGLU activation
    /// x: [n_tokens, hidden]
    /// indices: [n_tokens, top_k]
    /// Returns: [n_tokens, top_k, hidden]
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];

        // Check if we should use sorted_indices optimization (>= 64 tokens)
        let total_elements = n_tokens * top_k;
        let do_sort = total_elements >= 64;

        // Expand x for broadcasting: [n_tokens, hidden] -> [n_tokens, 1, 1, hidden]
        let x_expanded = mlxcel_core::expand_dims(x, -2);
        let x_expanded = mlxcel_core::expand_dims(&x_expanded, -3);

        if do_sort {
            // Sort tokens by expert for better memory access
            let (sorted_x, sorted_idx, inv_order) = self.gather_sort(&x_expanded, indices);

            // Apply projections with sorted_indices=true
            let x_gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            let x_up = self.up_proj.forward(&sorted_x, &sorted_idx, true);

            // Kernel-fused SwiGLU: silu(gate) * up
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);

            // Down projection
            let output = self.down_proj.forward(&activated, &sorted_idx, true);

            // Restore original order
            self.scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            // Direct path without sorting
            let x_gate = self.gate_proj.forward(&x_expanded, indices, false);
            let x_up = self.up_proj.forward(&x_expanded, indices, false);

            // Kernel-fused SwiGLU: silu(gate) * up
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);

            // Down projection
            let output = self.down_proj.forward(&activated, indices, false);

            // Squeeze: [n_tokens, top_k, 1, hidden] -> [n_tokens, top_k, hidden]
            mlxcel_core::squeeze_axis(&output, -2)
        }
    }

    /// Sort tokens by expert index for better memory access
    fn gather_sort(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let indices_shape = mlxcel_core::array_shape(indices);
        let top_k = indices_shape[indices_shape.len() - 1];

        // Flatten indices: [n_tokens, top_k] -> [n_tokens * top_k]
        let flat_indices = mlxcel_core::reshape(indices, &[-1]);

        // Sort indices by expert
        let order = mlxcel_core::argsort(&flat_indices, -1);
        let inv_order = mlxcel_core::argsort(&order, -1);

        // x is [n_tokens, 1, 1, hidden]
        // Flatten: [n_tokens, 1, hidden]
        let x_shape = mlxcel_core::array_shape(x);
        let x_flat = mlxcel_core::reshape(x, &[x_shape[0], 1, x_shape[3]]);

        // Divide order by top_k to get token indices
        let top_k_arr = mlxcel_core::from_slice_i32(&[top_k], &[1]);
        let token_indices = mlxcel_core::divide(&order, &top_k_arr);
        let token_indices = mlxcel_core::astype(&token_indices, mlxcel_core::dtype::INT32);

        // Take x rows in sorted order
        let sorted_x = mlxcel_core::take(&x_flat, &token_indices, 0);

        // Get sorted expert indices
        let sorted_indices = mlxcel_core::take(&flat_indices, &order, 0);

        (sorted_x, sorted_indices, inv_order)
    }

    /// Restore original order after sorted expert computation
    fn scatter_unsort(
        &self,
        x: &MlxArray,
        inv_order: &MlxArray,
        orig_shape: &[i32],
    ) -> UniquePtr<MlxArray> {
        // x has shape [n_sorted, 1, hidden]
        // Reorder by inv_order
        let unsorted = mlxcel_core::take(x, inv_order, 0);

        // Unflatten and squeeze
        let x_shape = mlxcel_core::array_shape(&unsorted);
        let n_tokens = orig_shape[0];
        let top_k = orig_shape[1];

        let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
        mlxcel_core::squeeze_axis(&reshaped, 2)
    }
}

// MLP for dense layers.
/// Standard MLP with SwiGLU activation
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // SwiGLU: silu(gate) * up
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }
}

// MoE Layer.
/// MoE Gate for routing
pub struct MoEGate {
    pub weight: UniquePtr<MlxArray>,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
}

impl MoEGate {
    pub fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // Gate logits: x @ weight.T
        let weight_t = mlxcel_core::transpose(&self.weight);
        let gates = mlxcel_core::matmul(x, &weight_t);

        // Softmax scores
        let scores = mlxcel_core::softmax(&gates, -1);

        let k = self.num_experts_per_tok as i32;

        // Top-k selection using argpartition
        // Negate scores to get top-k (largest values)
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);

        // Slice to get top-k indices: [..., :k]
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices = mlxcel_core::slice(&indices, &[0, 0], &[indices_shape[0], k]);

        // Get corresponding scores
        let topk_scores = mlxcel_core::take_along_axis(&scores, &topk_indices, -1);

        // Apply routed_scaling_factor
        if (self.routed_scaling_factor - 1.0).abs() > 1e-6 {
            let scale_arr = mlxcel_core::full_f32(
                &[1],
                self.routed_scaling_factor,
                mlxcel_core::dtype::FLOAT32,
            );
            let scaled_scores = mlxcel_core::multiply(&topk_scores, &scale_arr);
            (topk_indices, scaled_scores)
        } else {
            (topk_indices, topk_scores)
        }
    }
}

/// Mixture of Experts layer with optional shared expert
pub struct MoE {
    pub gate: MoEGate,
    pub experts: SwitchGLU,
    pub shared_experts: Option<MLP>,
}

impl MoE {
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

        // Get router indices and scores
        let (indices, scores) = self.gate.forward(&x_flat);

        // Apply experts
        let expert_out = self.experts.forward(&x_flat, &indices);

        let y = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Add shared experts if present
        let result = if let Some(shared) = &self.shared_experts {
            let shared_out = shared.forward(&x_flat);
            mlxcel_core::add(&y, &shared_out)
        } else {
            y
        };

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// Attention.
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
            false,
            self.rope_base,
            1.0, // scale
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.rope_dims,
            false,
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
}

// Transformer Block.
pub enum FeedForward {
    Dense(MLP),
    MoE(MoE),
}

impl FeedForward {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::MoE(moe) => moe.forward(x),
        }
    }
}

/// Transformer block with attention and feed-forward
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub feed_forward: FeedForward,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub layer_idx: usize,
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

        // Pre-norm feed-forward
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }
}

// Full Model.
/// DeepSeek v1 MoE language model
pub struct DeepSeekModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl DeepSeekModel {
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

    /// Forward pass from precomputed input embeddings (VLM prefill), skipping the
    /// token-embedding lookup. `inputs_embeds` is `(batch, seq, hidden)`.
    pub fn forward_from_embeds(
        &self,
        inputs_embeds: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = mlxcel_core::copy(inputs_embeds);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }
        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    /// Raw token-embedding lookup, exposed for VLM feature splicing.
    pub fn embed_tokens_forward(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
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

        // Determine if this is a MoE layer
        let is_moe_layer = args.n_routed_experts.is_some()
            && layer_idx >= args.first_k_dense_replace
            && layer_idx.is_multiple_of(args.moe_layer_freq);

        // Load feed-forward (MLP or MoE based on layer index)
        let feed_forward = if is_moe_layer {
            FeedForward::MoE(MoE::from_weights(weights, args, &prefix)?)
        } else {
            FeedForward::Dense(MLP::from_weights(weights, args, &prefix)?)
        };

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
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
            layer_idx,
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

        let q_proj = load_quantized_linear(weights, &format!("{}.q_proj", attn_prefix), args)?;
        let k_proj = load_quantized_linear(weights, &format!("{}.k_proj", attn_prefix), args)?;
        let v_proj = load_quantized_linear(weights, &format!("{}.v_proj", attn_prefix), args)?;
        let o_proj = load_quantized_linear(weights, &format!("{}.o_proj", attn_prefix), args)?;

        let head_dim = args.head_dim();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim: head_dim as i32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim as i32,
            rope_base: args.rope_theta,
        })
    }
}

impl MLP {
    /// Load MLP from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let mlp_prefix = format!("{}.mlp", prefix);

        Ok(Self {
            gate_proj: load_quantized_linear(weights, &format!("{}.gate_proj", mlp_prefix), args)?,
            up_proj: load_quantized_linear(weights, &format!("{}.up_proj", mlp_prefix), args)?,
            down_proj: load_quantized_linear(weights, &format!("{}.down_proj", mlp_prefix), args)?,
        })
    }
}

impl MoE {
    /// Load MoE from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let moe_prefix = format!("{}.mlp", prefix);

        // Load gate
        let gate_weight = get_weight_copy(weights, &format!("{}.gate.weight", moe_prefix))?;
        let gate = MoEGate {
            weight: gate_weight,
            num_experts_per_tok: args.num_experts_per_tok.unwrap_or(1),
            routed_scaling_factor: args.routed_scaling_factor,
        };

        // Load experts
        let experts =
            SwitchGLU::from_weights(weights, args, &format!("{}.switch_mlp", moe_prefix))?;

        // Load shared experts if present
        let shared_experts = if args.n_shared_experts.is_some() {
            Some(MLP::from_shared_weights(
                weights,
                args,
                &format!("{}.shared_experts", moe_prefix),
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_experts,
        })
    }
}

impl MLP {
    /// Load shared experts MLP (no separate prefix structure)
    fn from_shared_weights(
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

impl SwitchGLU {
    /// Load SwitchGLU from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(weights, args, &format!("{}.gate_proj", prefix))?,
            up_proj: SwitchLinear::from_weights(weights, args, &format!("{}.up_proj", prefix))?,
            down_proj: SwitchLinear::from_weights(weights, args, &format!("{}.down_proj", prefix))?,
        })
    }
}

impl SwitchLinear {
    /// Load SwitchLinear from weights, falling back to non-quantized when scales are absent.
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
        let scales_key = format!("{}.scales", prefix);
        if weights.contains_key(&scales_key) {
            let scales = mlxcel_core::copy(weights.get(&scales_key).unwrap());
            let biases = get_weight_copy(weights, &format!("{}.biases", prefix))?;
            let shape = mlxcel_core::array_shape(&weight);
            let num_experts = shape[0] as usize;
            Ok(Self::Quantized {
                weight,
                scales,
                biases,
                group_size: args.group_size(),
                bits: args.bits(),
                num_experts,
            })
        } else {
            let shape = mlxcel_core::array_shape(&weight);
            let num_experts = shape[0] as usize;
            Ok(Self::Regular {
                weight,
                num_experts,
            })
        }
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
impl LanguageModel for DeepSeekModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        DeepSeekModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        DeepSeekModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // DeepSeek v1 typically uses standard EOS token
        vec![2] // </s>
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        match input_embeddings {
            Some(embeds) => self.forward_from_embeds(embeds, caches, mask),
            None => DeepSeekModel::forward(self, input_ids, caches, mask),
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens_forward(input_ids))
    }
}
