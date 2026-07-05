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

//! ERNIE 4.5 MoE model implementation using mlxcel-core
//!
//! Key features:
//! - MoE with shared experts (num_shared_experts)
//! - Shared experts run in parallel with routed experts
//! - Traditional RoPE (rope_traditional=true)
//! - RMSNorm layer normalization
//! - Softmax routing for expert selection

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
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub moe_num_experts: usize,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub use_bias: bool,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub moe_layer_start_index: usize,

    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    #[serde(default = "default_moe_k")]
    pub moe_k: usize,

    #[serde(default = "default_moe_layer_interval")]
    pub moe_layer_interval: usize,

    #[serde(default)]
    pub moe_num_shared_experts: usize,

    #[serde(default)]
    pub moe_layer_end_index: Option<usize>,

    #[serde(default = "default_moe_gate_act")]
    pub moe_gate_act: String,

    #[serde(default)]
    pub group_size: Option<i32>,

    #[serde(default)]
    pub bits: Option<i32>,
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_max_position_embeddings() -> usize {
    4096
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_moe_k() -> usize {
    1
}

fn default_moe_layer_interval() -> usize {
    1
}

fn default_moe_gate_act() -> String {
    "softmax".to_string()
}

impl ModelArgs {
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

    pub fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    pub fn moe_layer_end_index(&self) -> usize {
        self.moe_layer_end_index
            .unwrap_or(self.num_hidden_layers - 1)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1).is_multiple_of(self.moe_layer_interval)
            && layer_idx >= self.moe_layer_start_index
            && layer_idx <= self.moe_layer_end_index()
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

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE (traditional=true for ERNIE)
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

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}

// SwitchLinear: Stacked expert weights for MoE.
// Used by: Ernie45Moe, Ernie45MoeVL (dual expert banks)
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
                        biases.as_ref().unwrap() as *const _,
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
// Used by: Ernie45Moe, Ernie45MoeVL (dual expert banks)
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

// Dense MLP (for non-MoE layers and shared experts).
// Used by: Ernie45Moe, Ernie45MoeVL (dense layers + fused shared experts)
/// Standard MLP with SwiGLU activation
pub struct DenseMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl DenseMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // SwiGLU: silu(gate) * up
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }
}

// Shared Experts for MoE.
/// Shared experts that always process input in parallel with routed experts
pub struct SharedExperts {
    pub experts: Vec<DenseMLP>,
}

impl SharedExperts {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Sum outputs from all shared experts
        let mut result: Option<UniquePtr<MlxArray>> = None;
        for expert in &self.experts {
            let output = expert.forward(x);
            result = match result {
                None => Some(output),
                Some(acc) => Some(mlxcel_core::add(&acc, &output)),
            };
        }
        result.expect("No experts in SharedExperts")
    }
}

// MoE Block.
/// MoE layer with sparse routed experts and shared experts
pub struct MoEBlock {
    pub gate: UnifiedLinear,
    pub switch_mlp: SwitchGLU,
    pub shared_experts: Option<SharedExperts>,
    pub top_k: usize,
    pub use_softmax: bool,
}

impl MoEBlock {
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

        // Get router logits
        let gates = self.gate.forward(&x_flat);

        // Apply gate activation (softmax or sigmoid)
        let gates = if self.use_softmax {
            mlxcel_core::softmax(&gates, -1)
        } else {
            mlxcel_core::sigmoid(&gates)
        };

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

        // Normalize scores
        let score_sum = mlxcel_core::sum_axis(&scores, -1, true);
        let eps = mlxcel_core::full_f32(&[1], 1e-12, mlxcel_core::array_dtype(&score_sum));
        let score_sum = mlxcel_core::maximum(&score_sum, &eps);
        let scores = mlxcel_core::divide(&scores, &score_sum);

        // Apply routed experts
        let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices);

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Add shared experts if present
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(&x_flat);
            result = mlxcel_core::add(&result, &shared_out);
        }

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// MLP Variant (Dense or MoE).
pub enum MLPVariant {
    Dense(DenseMLP),
    MoE(MoEBlock),
}

impl MLPVariant {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MLPVariant::Dense(mlp) => mlp.forward(x),
            MLPVariant::MoE(moe) => moe.forward(x),
        }
    }
}

// Transformer Block.
/// ERNIE 4.5 MoE transformer block
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLPVariant,
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

        // Pre-norm MLP
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// Full Model.
/// ERNIE 4.5 MoE model
pub struct Ernie45MoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub tie_word_embeddings: bool,
}

impl Ernie45MoeModel {
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
        if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&h)
        } else {
            self.lm_head.as_ref().unwrap().forward(&h)
        }
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
        let lm_head = if !args.tie_word_embeddings {
            Some(load_quantized_linear(weights, "lm_head", args)?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tie_word_embeddings: args.tie_word_embeddings,
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

        // Load MLP (dense or MoE based on layer index)
        let mlp = if args.is_moe_layer(layer_idx) {
            MLPVariant::MoE(MoEBlock::from_weights(weights, args, &prefix)?)
        } else {
            MLPVariant::Dense(DenseMLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
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
            rope_traditional: true, // ERNIE uses traditional RoPE
        })
    }
}

impl DenseMLP {
    /// Load dense MLP from weights
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

impl MoEBlock {
    /// Load MoE block from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let moe_prefix = format!("{}.mlp", prefix);

        let gate = load_quantized_linear(weights, &format!("{}.gate", moe_prefix), args)?;
        let switch_mlp =
            SwitchGLU::from_weights(weights, args, &format!("{}.switch_mlp", moe_prefix))?;

        // Load shared experts if present
        let shared_experts = if args.moe_num_shared_experts > 0 {
            let mut experts = Vec::with_capacity(args.moe_num_shared_experts);
            for i in 0..args.moe_num_shared_experts {
                let expert_prefix = format!("{}.shared_experts.{}", moe_prefix, i);
                let expert = DenseMLP::from_weights(weights, args, &expert_prefix)?;
                experts.push(expert);
            }
            Some(SharedExperts { experts })
        } else {
            None
        };

        Ok(Self {
            gate,
            switch_mlp,
            shared_experts,
            top_k: args.moe_k,
            use_softmax: args.moe_gate_act == "softmax",
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
impl LanguageModel for Ernie45MoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Ernie45MoeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Ernie45MoeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // ERNIE EOS token
        vec![2]
    }
}
