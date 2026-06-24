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

//! ExaOne MoE model implementation using mlxcel-core (LG AI MoE model)
//!
//! Key features:
//! - MoE with grouped expert selection (simplified to standard top-k routing)
//! - Q/K norm with RMSNorm
//! - Sliding window attention with layer_types per layer
//! - Sigmoid scoring for expert routing
//! - Shared experts alongside routed experts
//! - RotatingKVCache for sliding window layers

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_shared_experts: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub is_moe_layer: Vec<bool>,

    #[serde(default = "default_n_group")]
    pub n_group: usize,

    #[serde(default = "default_topk_group")]
    pub topk_group: usize,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,

    #[serde(default = "default_topk_method")]
    pub topk_method: String,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

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

fn default_n_group() -> usize {
    1
}
fn default_topk_group() -> usize {
    1
}
fn default_routed_scaling_factor() -> f32 {
    2.5
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_scoring_func() -> String {
    "sigmoid".to_string()
}
fn default_topk_method() -> String {
    "noaux_tc".to_string()
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}

impl ModelArgs {
    pub fn get_rope_theta(&self) -> f32 {
        // Check rope_parameters for rope_theta override
        if let Some(ref params) = self.rope_parameters
            && let Some(v) = params.get("rope_theta")
            && let Some(f) = v.as_f64()
        {
            return f as f32;
        }
        self.rope_theta
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

// MoE SwitchLinear and SwitchGLU.
/// Stacked linear layers for MoE experts
pub enum SwitchLinear {
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
                    sorted_indices,
                    "affine",
                )
            },
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
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

/// SwitchGLU: SwiGLU activation with stacked expert weights
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

        // Check if we should use sorted_indices optimization
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

        // Flatten indices
        let flat_indices = mlxcel_core::reshape(indices, &[-1]);

        // Sort indices by expert
        let order = mlxcel_core::argsort(&flat_indices, -1);
        let inv_order = mlxcel_core::argsort(&order, -1);

        // Flatten x and reorder
        let x_shape = mlxcel_core::array_shape(x);
        let x_flat = mlxcel_core::reshape(x, &[x_shape[0], 1, x_shape[3]]);

        let top_k_arr = mlxcel_core::from_slice_i32(&[top_k], &[1]);
        let token_indices = mlxcel_core::divide(&order, &top_k_arr);
        let token_indices = mlxcel_core::astype(&token_indices, dtype::INT32);

        let sorted_x = mlxcel_core::take(&x_flat, &token_indices, 0);
        let sorted_indices = mlxcel_core::take(&flat_indices, &order, 0);

        (sorted_x, sorted_indices, inv_order)
    }

    fn scatter_unsort(
        &self,
        x: &MlxArray,
        inv_order: &MlxArray,
        orig_shape: &[i32],
    ) -> UniquePtr<MlxArray> {
        let unsorted = mlxcel_core::take(x, inv_order, 0);
        let x_shape = mlxcel_core::array_shape(&unsorted);
        let n_tokens = orig_shape[0];
        let top_k = orig_shape[1];

        let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
        mlxcel_core::squeeze_axis(&reshaped, 2)
    }
}

// MoE Gate with Sigmoid Scoring.
/// MoE Gate with grouped expert selection (simplified to standard top-k)
pub struct MoEGate {
    pub router: UnifiedLinear,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
}

impl MoEGate {
    /// Forward pass with sigmoid scoring
    /// Returns (topk_indices, topk_scores)
    pub fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // Get router logits
        let logits = self.router.forward(x);

        // Sigmoid scoring (unlike Mixtral which uses softmax)
        let scores = mlxcel_core::sigmoid(&logits);
        let orig_scores = mlxcel_core::copy(&scores);

        // Add correction bias
        let scores = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Group-based expert masking (zero out non-selected groups)
        let scores = if self.n_group > 1 {
            super::switch_layers::group_mask_scores(
                &scores,
                self.n_group as i32,
                self.topk_group as i32,
            )
        } else {
            scores
        };

        // Top-k selection using argpartition
        let k = self.top_k as i32;
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);
        let topk_indices = utils::slice_axis(&indices, -1, 0, k);

        // Get scores for top-k experts from original (un-biased) scores
        let topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        // Normalize if needed
        let topk_scores = if self.top_k > 1 && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
            let denom = mlxcel_core::add(
                &sum,
                &mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::array_dtype(&topk_scores)),
            );
            mlxcel_core::divide(&topk_scores, &denom)
        } else {
            topk_scores
        };

        // Apply scaling factor
        let topk_scores = mlxcel_core::multiply_scalar(&topk_scores, self.routed_scaling_factor);

        (topk_indices, topk_scores)
    }
}

// Dense MLP (for non-MoE layers).
pub struct ExaoneMoeMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl ExaoneMoeMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // Kernel-fused SwiGLU: silu(gate) * up
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }
}

// MoE Block with Shared Experts.
pub struct ExaoneMoE {
    pub switch_mlp: SwitchGLU,
    pub gate: MoEGate,
    pub shared_experts: Option<ExaoneMoeMLP>,
}

impl ExaoneMoE {
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

        // Get top-k expert indices and scores
        let (indices, scores) = self.gate.forward(&x_flat);

        // Apply experts - returns [n_tokens, k, hidden]
        let expert_out = self.switch_mlp.forward(&x_flat, &indices);

        let mut output = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Add shared experts output if present
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(&x_flat);
            output = mlxcel_core::add(&output, &shared_out);
        }

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&output, &orig_shape)
        } else {
            output
        }
    }
}

// FFN Enum for Dense or MoE.
pub enum ExaoneMoeFFN {
    Dense(ExaoneMoeMLP),
    Moe(ExaoneMoE),
}

impl ExaoneMoeFFN {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Moe(moe) => moe.forward(x),
        }
    }
}

// Attention with Q/K Normalization.
pub struct ExaoneMoeAttention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub rope_traditional: bool,
    pub is_sliding_window: bool,
    pub window_size: i32,
}

impl ExaoneMoeAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn KVCacheTrait,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape for per-head normalization
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);

        // Apply Q/K norms
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset();

        // Apply RoPE
        let q = mlxcel_core::fast_rope(
            &q,
            self.head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, self.window_size)
        } else {
            // Single token or explicit mask
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.window_size,
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}

// Decoder Layer.
pub struct ExaoneMoeDecoderLayer {
    pub self_attn: ExaoneMoeAttention,
    pub mlp: ExaoneMoeFFN,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub is_sliding_window: bool,
}

impl ExaoneMoeDecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn KVCacheTrait,
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

// KVCache Trait for Dynamic Dispatch.
pub trait KVCacheTrait {
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);

    fn offset(&self) -> i32;
}

impl KVCacheTrait for KVCache {
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        KVCache::update_and_fetch(self, k, v)
    }

    fn offset(&self) -> i32 {
        self.offset
    }
}

impl KVCacheTrait for RotatingKVCache {
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        RotatingKVCache::update_and_fetch(self, k, v)
    }

    fn offset(&self) -> i32 {
        self.offset
    }
}

pub enum AnyKVCache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl AnyKVCache {
    /// Number of live keys the cache will actually return on the next
    /// `update_and_fetch`. For a `Standard` (`KVCache`) this is
    /// `offset - live_start` (`live_len()`), which shrinks below the
    /// monotonic `offset` after a `--max-kv-size` `trim_front`; for a
    /// `Rotating` cache `seq_len()` already reports the live window. Prefill
    /// masks size from this so the mask key axis matches the returned K/V,
    /// while `offset()` stays for RoPE. See issue #430.
    fn live_len(&self) -> i32 {
        match self {
            Self::Standard(c) => c.live_len(),
            Self::Rotating(c) => c.seq_len(),
        }
    }
}

impl KVCacheTrait for AnyKVCache {
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self {
            Self::Standard(c) => c.update_and_fetch(k, v),
            Self::Rotating(c) => c.update_and_fetch(k, v),
        }
    }

    fn offset(&self) -> i32 {
        match self {
            Self::Standard(c) => c.offset,
            Self::Rotating(c) => c.offset,
        }
    }
}

// Model.
pub struct ExaoneMoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<ExaoneMoeDecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub window_size: usize,
}

impl ExaoneMoeModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [AnyKVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        let seq_len = mlxcel_core::array_shape(&h)[1];

        // Create masks if needed.
        //
        // Size both prefill masks from the cache's live window
        // (`live_len()`), not the monotonic `offset`. Under `--max-kv-size`,
        // the scheduler trims the global (full-attention) `KVCache` via
        // `trim_front`, which advances `live_start` while `offset` keeps
        // growing to preserve the RoPE relative positions, so
        // `update_and_fetch` returns only `live_len = offset - live_start`
        // keys. A mask sized from `offset` would be wider than the returned
        // K/V and trip `broadcast_shapes`. With no trim (`live_start == 0`),
        // `live_len == offset`, so this is byte-identical to the untrimmed
        // path; RoPE keeps reading the monotonic `offset` inside the
        // attention layer. See issue #430 (mirrors #419/#420, #421/#422).
        let (global_mask, swa_mask) = if seq_len > 1 && mask.is_none() {
            // Create global causal mask
            let ga_live_len = caches
                .iter()
                .find(|c| matches!(c, AnyKVCache::Standard(_)))
                .map(|c| c.live_len())
                .unwrap_or(0);
            let global_mask = Some(utils::create_causal_mask(seq_len, ga_live_len));

            // Create sliding window mask if applicable
            let swa_live_len = caches
                .iter()
                .find(|c| matches!(c, AnyKVCache::Rotating(_)))
                .map(|c| c.live_len())
                .unwrap_or(0);
            // Full-width windowed mask for a fresh single-pass prefill that
            // exceeds the window (RotatingKVCache keeps all prefill keys),
            // clamped mask otherwise. See issue #408.
            let max_cache = self.window_size as i32;
            let swa_mask = Some(utils::create_sliding_window_prefill_mask(
                seq_len,
                swa_live_len,
                max_cache,
            ));

            (global_mask, swa_mask)
        } else {
            (None, None)
        };

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_mask = if layer.is_sliding_window {
                swa_mask.as_ref().map(|m| m.as_ref().unwrap())
            } else {
                global_mask.as_ref().map(|m| m.as_ref().unwrap())
            };
            h = layer.forward(&h, &mut caches[i], layer_mask);
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

    pub fn make_caches(&self) -> Vec<AnyKVCache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.is_sliding_window {
                    AnyKVCache::Rotating(RotatingKVCache::new(self.window_size as i32))
                } else {
                    AnyKVCache::Standard(KVCache::new())
                }
            })
            .collect()
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

        // Load LM head if not tied
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
            window_size: args.sliding_window,
        })
    }
}

// Weight Loading Helpers.
fn load_decoder_layer(
    weights: &WeightMap,
    args: &ModelArgs,
    layer_idx: usize,
) -> Result<ExaoneMoeDecoderLayer, String> {
    let prefix = format!("model.layers.{}", layer_idx);
    let group_size = args.group_size();
    let bits = args.bits();

    // Load attention
    let attn_prefix = format!("{}.self_attn", prefix);
    let q_proj = UnifiedLinear::from_weights(
        weights,
        &format!("{}.q_proj", attn_prefix),
        group_size,
        bits,
    )?;
    let k_proj = UnifiedLinear::from_weights(
        weights,
        &format!("{}.k_proj", attn_prefix),
        group_size,
        bits,
    )?;
    let v_proj = UnifiedLinear::from_weights(
        weights,
        &format!("{}.v_proj", attn_prefix),
        group_size,
        bits,
    )?;
    let o_proj = UnifiedLinear::from_weights(
        weights,
        &format!("{}.o_proj", attn_prefix),
        group_size,
        bits,
    )?;

    let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", attn_prefix))?;
    let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", attn_prefix))?;

    let is_sliding_window = args.layer_types[layer_idx] == "sliding_attention";

    let self_attn = ExaoneMoeAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm: RMSNorm::new(q_norm_weight, args.rms_norm_eps),
        k_norm: RMSNorm::new(k_norm_weight, args.rms_norm_eps),
        n_heads: args.num_attention_heads as i32,
        n_kv_heads: args.num_key_value_heads as i32,
        head_dim: args.head_dim as i32,
        scale: (args.head_dim as f32).powf(-0.5),
        rope_base: args.get_rope_theta(),
        rope_traditional: args.rope_traditional,
        is_sliding_window,
        window_size: if is_sliding_window {
            args.sliding_window as i32
        } else {
            0
        },
    };

    // Load MLP (either Dense or MoE)
    let mlp_prefix = format!("{}.mlp", prefix);
    let mlp = if args.is_moe_layer[layer_idx] {
        // MoE layer
        let switch_mlp = load_switch_glu(weights, &format!("{}.switch_mlp", mlp_prefix), args)?;

        // Load gate
        let gate_router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate.weight", mlp_prefix),
            group_size,
            bits,
        )?;

        let e_score_bias = get_weight_copy(
            weights,
            &format!("{}.gate.e_score_correction_bias", mlp_prefix),
        )?;

        let gate = MoEGate {
            router: gate_router,
            e_score_correction_bias: e_score_bias,
            top_k: args.num_experts_per_tok,
            n_group: args.n_group,
            topk_group: args.topk_group,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
        };

        // Load shared experts if present
        let shared_experts = if args.num_shared_experts > 0 {
            let gate_proj = UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.gate_proj", mlp_prefix),
                group_size,
                bits,
            )?;
            let up_proj = UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.up_proj", mlp_prefix),
                group_size,
                bits,
            )?;
            let down_proj = UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.down_proj", mlp_prefix),
                group_size,
                bits,
            )?;

            Some(ExaoneMoeMLP {
                gate_proj,
                up_proj,
                down_proj,
            })
        } else {
            None
        };

        ExaoneMoeFFN::Moe(ExaoneMoE {
            switch_mlp,
            gate,
            shared_experts,
        })
    } else {
        // Dense layer
        let gate_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", mlp_prefix),
            group_size,
            bits,
        )?;
        let up_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.up_proj", mlp_prefix),
            group_size,
            bits,
        )?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", mlp_prefix),
            group_size,
            bits,
        )?;

        ExaoneMoeFFN::Dense(ExaoneMoeMLP {
            gate_proj,
            up_proj,
            down_proj,
        })
    };

    // Load norms
    let input_norm_weight =
        get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
    let post_norm_weight = get_weight_copy(
        weights,
        &format!("{}.post_attention_layernorm.weight", prefix),
    )?;

    Ok(ExaoneMoeDecoderLayer {
        self_attn,
        mlp,
        input_layernorm: RMSNorm::new(input_norm_weight, args.rms_norm_eps),
        post_attention_layernorm: RMSNorm::new(post_norm_weight, args.rms_norm_eps),
        is_sliding_window,
    })
}

fn load_switch_glu(
    weights: &WeightMap,
    prefix: &str,
    args: &ModelArgs,
) -> Result<SwitchGLU, String> {
    let group_size = args.group_size();
    let bits = args.bits();

    let gate_proj =
        load_switch_linear(weights, &format!("{}.gate_proj", prefix), group_size, bits)?;
    let up_proj = load_switch_linear(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
    let down_proj =
        load_switch_linear(weights, &format!("{}.down_proj", prefix), group_size, bits)?;

    Ok(SwitchGLU {
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn load_switch_linear(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<SwitchLinear, String> {
    let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
    let scales_key = format!("{}.scales", prefix);
    if weights.contains_key(&scales_key) {
        let scales = mlxcel_core::copy(weights.get(&scales_key).unwrap());
        let biases = get_weight_copy(weights, &format!("{}.biases", prefix))?;
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

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for ExaoneMoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Convert KVCache to AnyKVCache
        let mut any_caches: Vec<AnyKVCache> = caches
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if self.layers[i].is_sliding_window {
                    AnyKVCache::Rotating(RotatingKVCache::new(self.window_size as i32))
                } else {
                    AnyKVCache::Standard(KVCache::new())
                }
            })
            .collect();

        let result = self.forward(input_ids, &mut any_caches, mask);

        // Copy offsets back to original caches
        for (i, cache) in caches.iter_mut().enumerate() {
            cache.offset = any_caches[i].offset();
        }

        result
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // For LanguageModel trait, we return standard caches
        // The actual model uses AnyKVCache internally
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // ExaOne MoE EOS token
        vec![2] // </s>
    }
}

#[cfg(test)]
mod exaone_moe_mask_tests {
    // `KVCacheTrait` (defined in this module) brings `update_and_fetch` /
    // `offset` into scope for `AnyKVCache`; `live_len` is an inherent method.
    use super::*;

    // [1, H, n, D] K/V tensor; only the key axis (n) matters for the shape
    // invariant exercised here.
    fn make_kv(h: i32, n: i32, d: i32, base: f32) -> UniquePtr<MlxArray> {
        let count = (h * n * d) as usize;
        let data: Vec<f32> = (0..count).map(|i| base + i as f32).collect();
        mlxcel_core::from_slice_f32(&data, &[1, h, n, d])
    }

    /// Regression for #430: the global prefill mask must size from
    /// `AnyKVCache::live_len()` (the `Standard`/`KVCache` live window), not the
    /// monotonic `offset()`. After a `--max-kv-size` `trim_front`
    /// (`live_start > 0`), `live_len < offset`; a continuation chunk's
    /// `update_and_fetch` returns only `live_len + m` keys, so a `live_len`-
    /// sized causal mask matches the returned K/V while an `offset`-sized mask
    /// is strictly wider (the crash this fix prevents).
    #[test]
    fn global_mask_sizes_from_live_len_not_offset_after_trim() {
        const H: i32 = 2;
        const D: i32 = 4;
        let (n1, trimmed, m) = (8, 3, 4);

        let mut cache = AnyKVCache::Standard(KVCache::new());
        let _ = cache.update_and_fetch(make_kv(H, n1, D, 0.0), make_kv(H, n1, D, 100.0));
        let AnyKVCache::Standard(ref mut std_cache) = cache else {
            unreachable!("constructed as Standard")
        };
        assert_eq!(std_cache.trim_front(trimmed), trimmed);

        let offset = cache.offset();
        let live_len = cache.live_len();
        assert_eq!(offset, n1, "offset stays monotonic after trim");
        assert_eq!(live_len, n1 - trimmed, "live_len == offset - live_start");
        assert!(live_len < offset, "trim must make live_len < offset");

        let (cont_k, _) = cache.update_and_fetch(make_kv(H, m, D, 200.0), make_kv(H, m, D, 300.0));
        let returned_klen = mlxcel_core::array_shape(&cont_k)[2];
        assert_eq!(
            returned_klen,
            live_len + m,
            "cache returns live_len + m keys"
        );

        let live_mask = utils::create_causal_mask(m, live_len);
        mlxcel_core::eval(&live_mask);
        assert_eq!(
            *mlxcel_core::array_shape(&live_mask).last().unwrap(),
            returned_klen,
            "live_len-sized mask key axis must match the returned K/V"
        );
        let offset_mask = utils::create_causal_mask(m, offset);
        mlxcel_core::eval(&offset_mask);
        assert!(
            *mlxcel_core::array_shape(&offset_mask).last().unwrap() > returned_klen,
            "offset-sized mask must be wider than the returned K/V (the bug)"
        );
    }

    /// The sliding lookup uses `RotatingKVCache::seq_len()` via
    /// `AnyKVCache::live_len()`, which equals the keys `update_and_fetch`
    /// returns.
    #[test]
    fn sliding_cache_live_len_matches_returned_keys() {
        const H: i32 = 2;
        const D: i32 = 4;
        let window = 6;
        let m = 4;

        let mut cache = AnyKVCache::Rotating(RotatingKVCache::new(window));
        let (k, _) = cache.update_and_fetch(make_kv(H, m, D, 0.0), make_kv(H, m, D, 100.0));
        let returned_klen = mlxcel_core::array_shape(&k)[2];
        assert_eq!(
            cache.live_len(),
            returned_klen,
            "AnyKVCache::live_len() for Rotating (== seq_len) must equal the returned key axis"
        );
    }
}
