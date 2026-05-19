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

//! Llama 4 model implementation using mlxcel-core for maximum MoE performance
//!
//! This is a parallel implementation of Llama 4 using direct C++ bindings
//! to leverage kernel fusion for the MoE layers.

use crate::models::llama4_helpers::{
    create_chunked_attention_mask, get_weight_copy, load_quantized_linear,
};
use crate::models::model_owned::{
    ModelOwnedSequenceState, dispatch_paged_decode_from_backing_caches,
};
use mlxcel_core::cache::{CachePool, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{ChunkedKVCache, KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Llama4 Cache Types.
/// Cache enum for Llama4's iGQA (Interleaved GQA) pattern
/// MoE layers use ChunkedKVCache, dense layers use regular KVCache
pub enum Llama4Cache {
    /// Chunked cache for MoE layers (every non-4th layer)
    Chunked(ChunkedKVCache),
    /// Regular cache for dense layers (every 4th layer: 3, 7, 11, ...)
    Regular(KVCache),
}

impl Llama4Cache {
    /// Get the current offset (total tokens processed).
    ///
    /// In Python mlx-vlm, `BatchKVCache` can return an `mx.array`-valued offset
    /// when doing batch inference, requiring a `.max().item()` guard before using
    /// it in `mx.arange`. In Rust, offset is always `i32` (scalar), so no such
    /// guard is needed. Batched decode now threads per-sequence offsets
    /// explicitly through vectorized metadata instead of exposing array-valued
    /// offsets from the cache type itself.
    pub fn offset(&self) -> i32 {
        match self {
            Llama4Cache::Chunked(c) => c.get_offset(),
            Llama4Cache::Regular(c) => c.offset,
        }
    }

    /// Get the start position (for chunked cache).
    ///
    /// Returns `0` for `Regular` variant. In Python, `getattr(cache[0],
    /// "start_position", 0)` is used defensively for caches that may not have
    /// this attribute (e.g. `BatchKVCache`). In Rust the enum always has a
    /// well-defined answer, so no fallback is needed.
    pub fn start_position(&self) -> i32 {
        match self {
            Llama4Cache::Chunked(c) => c.get_start_position(),
            Llama4Cache::Regular(_) => 0,
        }
    }

    /// Maybe trim the front of chunked caches.
    ///
    /// In Python, this is guarded with `hasattr(c, "maybe_trim_front")` because
    /// `BatchKVCache` does not implement that method. In Rust the enum handles
    /// both variants explicitly: `Chunked` delegates to the inner cache and
    /// `Regular` is a no-op, so no runtime guard is required.
    pub fn maybe_trim_front(&mut self) {
        if let Llama4Cache::Chunked(c) = self {
            c.maybe_trim_front();
        }
    }

    /// Update cache and fetch keys/values
    pub fn update_and_fetch(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self {
            Llama4Cache::Chunked(c) => c.update_and_fetch(keys, values),
            Llama4Cache::Regular(c) => c.update_and_fetch(keys, values),
        }
    }

    pub fn visible_len(&self) -> usize {
        match self {
            Llama4Cache::Chunked(c) => c
                .keys
                .as_ref()
                .map(|keys| mlxcel_core::array_shape(keys)[2].max(0) as usize)
                .unwrap_or(0),
            Llama4Cache::Regular(c) => c.seq_len().max(0) as usize,
        }
    }

    pub fn keys_ptr(&self) -> Option<*const MlxArray> {
        match self {
            Llama4Cache::Chunked(c) => c
                .keys
                .as_ref()
                .and_then(|keys| keys.as_ref().map(|arr| arr as *const _)),
            Llama4Cache::Regular(c) => c
                .keys
                .as_ref()
                .and_then(|keys| keys.as_ref().map(|arr| arr as *const _)),
        }
    }

    pub fn values_ptr(&self) -> Option<*const MlxArray> {
        match self {
            Llama4Cache::Chunked(c) => c
                .values
                .as_ref()
                .and_then(|values| values.as_ref().map(|arr| arr as *const _)),
            Llama4Cache::Regular(c) => c
                .values
                .as_ref()
                .and_then(|values| values.as_ref().map(|arr| arr as *const _)),
        }
    }

    pub fn is_chunked(&self) -> bool {
        matches!(self, Llama4Cache::Chunked(_))
    }
}

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TextArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub intermediate_size_mlp: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub attention_chunk_size: usize,
    pub interleave_moe_layer_step: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub use_qk_norm: bool,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_attn_temperature_tuning")]
    pub attn_temperature_tuning: i32,

    #[serde(default = "default_floor_scale")]
    pub floor_scale: usize,

    #[serde(default = "default_attn_scale")]
    pub attn_scale: f32,

    #[serde(default)]
    pub group_size: Option<i32>,

    #[serde(default)]
    pub bits: Option<i32>,
}

fn default_rope_theta() -> f32 {
    500000.0
}

fn default_attn_temperature_tuning() -> i32 {
    4
}

fn default_floor_scale() -> usize {
    8192
}

fn default_attn_scale() -> f32 {
    0.1
}

impl TextArgs {
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
                        biases
                            .as_ref()
                            .map(|b| b as *const _)
                            .unwrap_or(std::ptr::null()),
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

// MoE Layer.
/// Mixture of Experts layer with shared expert
pub struct MoE {
    pub router: UnifiedLinear,
    pub experts: SwitchGLU,
    pub shared_expert: MLP,
    pub num_experts: usize,
    pub top_k: usize,
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

        // Get router logits
        let logits = self.router.forward(&x_flat);

        // Top-k selection
        let k = self.top_k as i32;

        // Get top-k expert indices using argpartition
        let neg_logits = mlxcel_core::negative(&logits);
        let indices = mlxcel_core::argpartition(&neg_logits, k - 1, -1);

        // Slice to get top-k: indices[..., :k]
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices = mlxcel_core::slice(&indices, &[0, 0], &[indices_shape[0], k]);

        // Get corresponding logits and apply sigmoid for scores
        let topk_logits = mlxcel_core::take_along_axis(&logits, &topk_indices, -1);
        let scores = mlxcel_core::sigmoid(&topk_logits);

        // Pre-multiply input by scores (Llama 4 style)
        let x_weighted = mlxcel_core::multiply(&x_flat, &scores);

        // Apply experts
        let expert_out = self.experts.forward(&x_weighted, &topk_indices);

        // Squeeze out top_k dimension (since top_k=1)
        let expert_out = mlxcel_core::squeeze_axis(&expert_out, 1);

        // Add shared expert output
        let shared_out = self.shared_expert.forward(&x_flat);
        let result = mlxcel_core::add(&expert_out, &shared_out);

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
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

// Attention.
/// Multi-head attention with RoPE and optional QK normalization
pub struct CxxAttention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub use_rope: bool,
    pub use_qk_norm: bool,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub attn_temperature_tuning: i32,
    pub floor_scale: usize,
    pub attn_scale: f32,
}

impl CxxAttention {
    /// Forward pass with Llama4Cache (supports both Chunked and Regular caches)
    pub fn forward_llama4(
        &self,
        x: &MlxArray,
        cache: &mut Llama4Cache,
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
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset();

        // Apply RoPE if enabled
        if self.use_rope {
            q = mlxcel_core::fast_rope(
                &q,
                self.rope_dims,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            );
            k = mlxcel_core::fast_rope(
                &k,
                self.rope_dims,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            );
        }

        // Apply QK normalization if enabled
        if self.use_qk_norm {
            let q_dtype = mlxcel_core::array_dtype(&q);
            let norm_weight = mlxcel_core::ones(&[self.head_dim], q_dtype);
            q = mlxcel_core::fast_rms_norm(&q, &norm_weight, 1e-6);
            k = mlxcel_core::fast_rms_norm(&k, &norm_weight, 1e-6);
        }

        // Temperature tuning for dense layers (no RoPE)
        if self.attn_temperature_tuning != 0 && !self.use_rope {
            let q_dtype = mlxcel_core::array_dtype(&q);
            let positions =
                mlxcel_core::arange_f32((offset + 1) as f32, (offset + l + 1) as f32, 1.0);
            let floor_scale_arr =
                mlxcel_core::full_f32(&[1], self.floor_scale as f32, mlxcel_core::dtype::FLOAT32);
            let floored = mlxcel_core::floor_divide(&positions, &floor_scale_arr);
            let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
            let attn_scale_arr =
                mlxcel_core::full_f32(&[1], self.attn_scale, mlxcel_core::dtype::FLOAT32);
            let floored_plus_one = mlxcel_core::add(&floored, &one);
            let log_val = mlxcel_core::log(&floored_plus_one);
            let scaled = mlxcel_core::multiply(&log_val, &attn_scale_arr);
            let attn_scales = mlxcel_core::add(&scaled, &one);
            let attn_scales = mlxcel_core::reshape(&attn_scales, &[1, 1, l, 1]);
            let attn_scales = mlxcel_core::astype(&attn_scales, q_dtype);
            q = mlxcel_core::multiply(&q, &attn_scales);
        }

        // Update KV cache and get cached keys/values
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention using cached K,V
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill with no mask: use causal masking
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else if let Some(m) = mask {
            // Explicit mask provided
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    m as *const _,
                    0.0,
                    0,
                )
            }
        } else {
            // Single token, no mask needed
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    fn forward_batched_decode_chunked(
        &self,
        x: &MlxArray,
        caches: &mut [&mut Llama4Cache],
        context: &DecodeBatchContext,
    ) -> Result<Option<UniquePtr<MlxArray>>, String> {
        let shape = mlxcel_core::array_shape(x);
        if shape[1] != 1 || !self.use_rope || !caches.iter().all(|cache| cache.is_chunked()) {
            return Ok(None);
        }

        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim]);

        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offsets: Vec<i32> = caches.iter().map(|cache| cache.offset()).collect();
        q = mlxcel_core::fast_rope_batched(
            &q,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            &offsets,
        );
        k = mlxcel_core::fast_rope_batched(
            &k,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            &offsets,
        );

        if self.use_qk_norm {
            let q_dtype = mlxcel_core::array_dtype(&q);
            let norm_weight = mlxcel_core::ones(&[self.head_dim], q_dtype);
            q = mlxcel_core::fast_rms_norm(&q, &norm_weight, 1e-6);
            k = mlxcel_core::fast_rms_norm(&k, &norm_weight, 1e-6);
        }

        let Some(attn_out) = dispatch_paged_decode_from_backing_caches(
            &q,
            &k,
            &v,
            caches,
            self.scale,
            context,
            |cache, k_i, v_i| {
                let _ = cache.update_and_fetch(k_i, v_i);
                Ok(())
            },
            |cache| {
                cache
                    .keys_ptr()
                    .ok_or_else(|| "llama4 chunked cache missing key backing array".to_string())
            },
            |cache| {
                cache
                    .values_ptr()
                    .ok_or_else(|| "llama4 chunked cache missing value backing array".to_string())
            },
            |cache| Ok(cache.visible_len() as i32),
        )?
        else {
            return Ok(None);
        };

        tracing::debug!(
            batch_size = batch,
            block_size = context.paged_block_size,
            native_kernel = context.use_native_paged_kernel,
            "Llama4 chunked paged decode attention dispatch"
        );

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out =
            mlxcel_core::reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim]);
        Ok(Some(self.o_proj.forward(&attn_out)))
    }

    /// Legacy forward pass with regular KVCache (kept for compatibility)
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        if self.use_rope {
            q = mlxcel_core::fast_rope(
                &q,
                self.rope_dims,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            );
            k = mlxcel_core::fast_rope(
                &k,
                self.rope_dims,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            );
        }

        if self.use_qk_norm {
            let q_dtype = mlxcel_core::array_dtype(&q);
            let norm_weight = mlxcel_core::ones(&[self.head_dim], q_dtype);
            q = mlxcel_core::fast_rms_norm(&q, &norm_weight, 1e-6);
            k = mlxcel_core::fast_rms_norm(&k, &norm_weight, 1e-6);
        }

        if self.attn_temperature_tuning != 0 && !self.use_rope {
            let q_dtype = mlxcel_core::array_dtype(&q);
            let positions =
                mlxcel_core::arange_f32((offset + 1) as f32, (offset + l + 1) as f32, 1.0);
            let floor_scale_arr =
                mlxcel_core::full_f32(&[1], self.floor_scale as f32, mlxcel_core::dtype::FLOAT32);
            let floored = mlxcel_core::floor_divide(&positions, &floor_scale_arr);
            let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
            let attn_scale_arr =
                mlxcel_core::full_f32(&[1], self.attn_scale, mlxcel_core::dtype::FLOAT32);
            let floored_plus_one = mlxcel_core::add(&floored, &one);
            let log_val = mlxcel_core::log(&floored_plus_one);
            let scaled = mlxcel_core::multiply(&log_val, &attn_scale_arr);
            let attn_scales = mlxcel_core::add(&scaled, &one);
            let attn_scales = mlxcel_core::reshape(&attn_scales, &[1, 1, l, 1]);
            let attn_scales = mlxcel_core::astype(&attn_scales, q_dtype);
            q = mlxcel_core::multiply(&q, &attn_scales);
        }

        cache.update(mlxcel_core::copy(&k), mlxcel_core::copy(&v));

        let cache_k = cache
            .keys
            .as_ref()
            .expect("Cache keys should exist after update");
        let cache_v = cache
            .values
            .as_ref()
            .expect("Cache values should exist after update");

        let attn_out = if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, cache_k, cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, cache_k, cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&attn_out)
    }

    pub fn forward_debug(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        fn check_nan(name: &str, arr: &MlxArray) -> bool {
            mlxcel_core::eval(arr);
            let sum = mlxcel_core::sum_all(arr);
            mlxcel_core::eval(&sum);
            let val = mlxcel_core::item_f32(&sum);
            println!("    {} sum: {}", name, val);
            val.is_nan()
        }

        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        println!("    Input shape: {:?}", shape);

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        if check_nan("q_proj", &q) {
            return q;
        }

        let k = self.k_proj.forward(x);
        if check_nan("k_proj", &k) {
            return k;
        }

        let v = self.v_proj.forward(x);
        if check_nan("v_proj", &v) {
            return v;
        }

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        println!(
            "    use_rope: {}, use_qk_norm: {}, offset: {}",
            self.use_rope, self.use_qk_norm, offset
        );

        // Apply RoPE if enabled
        if self.use_rope {
            q = mlxcel_core::fast_rope(
                &q,
                self.rope_dims,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            );
            if check_nan("q_after_rope", &q) {
                return q;
            }

            k = mlxcel_core::fast_rope(
                &k,
                self.rope_dims,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            );
            if check_nan("k_after_rope", &k) {
                return k;
            }
        }

        // Apply QK normalization if enabled
        if self.use_qk_norm {
            let q_dtype = mlxcel_core::array_dtype(&q);
            let norm_weight = mlxcel_core::ones(&[self.head_dim], q_dtype);
            q = mlxcel_core::fast_rms_norm(&q, &norm_weight, 1e-6);
            if check_nan("q_after_qknorm", &q) {
                return q;
            }

            k = mlxcel_core::fast_rms_norm(&k, &norm_weight, 1e-6);
            if check_nan("k_after_qknorm", &k) {
                return k;
            }
        }

        // Skip temperature tuning for debug
        // Update KV cache
        cache.update(mlxcel_core::copy(&k), mlxcel_core::copy(&v));

        // Get full K, V from cache
        let cache_k = cache
            .keys
            .as_ref()
            .expect("Cache keys should exist after update");
        let cache_v = cache
            .values
            .as_ref()
            .expect("Cache values should exist after update");

        // SDPA with causal masking for prefill
        let attn_out = if l > 1 {
            mlxcel_core::causal_attention(&q, cache_k, cache_v, self.scale, 0.0, 0)
        } else {
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    cache_k,
                    cache_v,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            }
        };
        if check_nan("sdpa_output", &attn_out) {
            return attn_out;
        }

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        let out = self.o_proj.forward(&attn_out);
        check_nan("o_proj", &out);
        out
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
    pub self_attn: CxxAttention,
    pub feed_forward: FeedForward,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub layer_idx: usize,
}

impl TransformerBlock {
    /// Forward pass with Llama4Cache for iGQA support
    pub fn forward_llama4(
        &self,
        x: &MlxArray,
        cache: &mut Llama4Cache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward_llama4(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm feed-forward
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    fn forward_batched_decode_llama4(
        &self,
        x: &MlxArray,
        caches: &mut [&mut Llama4Cache],
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        if let Some(context) = decode_context {
            let normed = self.input_layernorm.forward(x);
            if let Some(attn_out) = self
                .self_attn
                .forward_batched_decode_chunked(&normed, caches, context)
                .expect("valid llama4 chunked paged decode inputs")
            {
                let h = mlxcel_core::add(x, &attn_out);
                let normed = self.post_attention_layernorm.forward(&h);
                let ff_out = self.feed_forward.forward(&normed);
                return mlxcel_core::add(&h, &ff_out);
            }
        }

        let shape = mlxcel_core::array_shape(x);
        let mut outputs = Vec::with_capacity(caches.len());
        for (batch_idx, cache) in caches.iter_mut().enumerate() {
            let x_i = mlxcel_core::slice(
                x,
                &[batch_idx as i32, 0, 0],
                &[batch_idx as i32 + 1, shape[1], shape[2]],
            );
            outputs.push(self.forward_llama4(&x_i, cache, None));
        }

        let mut out = outputs.remove(0);
        for output in outputs {
            out = mlxcel_core::concatenate(&out, &output, 0);
        }
        out
    }

    /// Legacy forward pass with regular KVCache
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn forward_debug(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        fn check_nan(name: &str, arr: &MlxArray) -> bool {
            mlxcel_core::eval(arr);
            let sum = mlxcel_core::sum_all(arr);
            mlxcel_core::eval(&sum);
            let val = mlxcel_core::item_f32(&sum);
            println!("  {} sum: {}", name, val);
            val.is_nan()
        }

        // Input layernorm
        let normed = self.input_layernorm.forward(x);
        if check_nan("input_layernorm", &normed) {
            return normed;
        }

        // Attention (debug version)
        println!("  Attention debug:");
        let attn_out = self.self_attn.forward_debug(&normed, cache, mask);
        if check_nan("attention", &attn_out) {
            return attn_out;
        }

        // Residual
        let h = mlxcel_core::add(x, &attn_out);
        if check_nan("after_attn_residual", &h) {
            return h;
        }

        // Post-attention layernorm
        let normed = self.post_attention_layernorm.forward(&h);
        if check_nan("post_attn_layernorm", &normed) {
            return normed;
        }

        // Feed-forward
        let ff_out = self.feed_forward.forward(&normed);
        if check_nan("feed_forward", &ff_out) {
            return ff_out;
        }

        // Final residual
        let out = mlxcel_core::add(&h, &ff_out);
        check_nan("final_residual", &out);
        out
    }
}

// Full Model.
/// Llama 4 language model using mlx-cxx for MoE optimization
pub struct Llama4CxxModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
    pub attention_chunk_size: usize,
    pub interleave_moe_layer_step: usize,
}

impl Llama4CxxModel {
    /// Forward pass with iGQA (Interleaved GQA) pattern
    pub fn forward_igqa(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Llama4Cache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Trim chunked caches before processing (like Python's maybe_trim_front)
        for (idx, cache) in caches.iter_mut().enumerate() {
            if (idx + 1) % 4 != 0 {
                cache.maybe_trim_front();
            }
        }

        // Get start_position and offset from first cache for mask creation
        let start_position = caches[0].start_position();
        let offset = caches[0].offset();

        // Create chunk mask for MoE layers (iGQA pattern)
        let chunk_mask = if seq_len > 1 {
            Some(create_chunked_attention_mask(
                seq_len,
                start_position,
                offset,
                self.attention_chunk_size,
            ))
        } else {
            None
        };

        // Create global mask for dense layers
        let global_mask = if seq_len > 1 {
            Some(mlxcel_core::utils::create_causal_mask(seq_len, offset))
        } else {
            None
        };

        // Pass through transformer layers with appropriate masks
        for (idx, layer) in self.layers.iter().enumerate() {
            // MoE layers use chunked attention, dense layers use global attention
            let use_chunked = (idx + 1) % 4 != 0;
            let mask = if use_chunked {
                chunk_mask.as_deref()
            } else {
                global_mask.as_deref()
            };
            h = layer.forward_llama4(&h, &mut caches[idx], mask);
        }

        // Final norm and LM head
        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    /// Forward pass with iGQA that accepts optional pre-computed embeddings (for VLM)
    pub fn forward_igqa_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Llama4Cache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Use pre-computed embeddings if provided, otherwise embed tokens
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embed_tokens.forward(input_ids),
        };

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Trim chunked caches before processing
        for (idx, cache) in caches.iter_mut().enumerate() {
            if (idx + 1) % 4 != 0 {
                cache.maybe_trim_front();
            }
        }

        let start_position = caches[0].start_position();
        let offset = caches[0].offset();

        let chunk_mask = if seq_len > 1 {
            Some(create_chunked_attention_mask(
                seq_len,
                start_position,
                offset,
                self.attention_chunk_size,
            ))
        } else {
            None
        };

        let global_mask = if seq_len > 1 {
            Some(mlxcel_core::utils::create_causal_mask(seq_len, offset))
        } else {
            None
        };

        for (idx, layer) in self.layers.iter().enumerate() {
            let use_chunked = (idx + 1) % 4 != 0;
            let mask = if use_chunked {
                chunk_mask.as_deref()
            } else {
                global_mask.as_deref()
            };
            h = layer.forward_llama4(&h, &mut caches[idx], mask);
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    /// Legacy forward pass with regular KVCache (for LanguageModel trait compatibility)
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], None);
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [Vec<Llama4Cache>],
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        for caches in batch_caches.iter_mut() {
            for (idx, cache) in caches.iter_mut().enumerate() {
                if (idx + 1) % 4 != 0 {
                    cache.maybe_trim_front();
                }
            }
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut layer_caches: Vec<&mut Llama4Cache> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches[layer_idx])
                .collect();
            h = layer.forward_batched_decode_llama4(&h, &mut layer_caches, decode_context);
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    /// Debug forward pass to identify where NaN appears
    pub fn forward_debug(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens (quantized)
        let mut h = self.embed_tokens.forward(input_ids);
        mlxcel_core::eval(&h);

        // Check embedding output
        let h_shape = mlxcel_core::array_shape(&h);
        let h_dtype = mlxcel_core::array_dtype(&h);
        println!("Embedding output shape: {:?}, dtype: {}", h_shape, h_dtype);

        // Convert to float32 before computing statistics (float16 sum has precision issues)
        let h_f32 = mlxcel_core::astype(&h, mlxcel_core::dtype::FLOAT32);
        let h_sum = mlxcel_core::sum_all(&h_f32);
        mlxcel_core::eval(&h_sum);
        let sum_val = mlxcel_core::item_f32(&h_sum);
        let h_max = mlxcel_core::max_all(&h_f32);
        let h_min = mlxcel_core::min_all(&h_f32);
        mlxcel_core::eval(&h_max);
        mlxcel_core::eval(&h_min);
        println!(
            "After embedding - sum: {}, max: {}, min: {}",
            sum_val,
            mlxcel_core::item_f32(&h_max),
            mlxcel_core::item_f32(&h_min)
        );
        if sum_val.is_nan() {
            println!("NaN detected after embedding!");
            return h;
        }

        // Pass through first 1 transformer layer with detailed debug
        println!("Layer 0 detailed debug:");
        h = self.layers[0].forward_debug(&h, &mut caches[0], None);
        mlxcel_core::eval(&h);

        h
    }

    /// Load model from a directory containing safetensors files and config.json
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, TextArgs), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Parse text_config or use root config
        let text_config = config.get("text_config").unwrap_or(&config);
        let args: TextArgs = serde_json::from_value(text_config.clone())
            .map_err(|e| format!("Failed to parse text config: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &TextArgs) -> Result<Self, String> {
        // Load quantized embedding
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            "language_model.model.embed_tokens",
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
        let norm_weight = get_weight_copy(weights, "language_model.model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
        let lm_head = load_quantized_linear(weights, "language_model.lm_head", args)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            attention_chunk_size: args.attention_chunk_size,
            interleave_moe_layer_step: args.interleave_moe_layer_step,
        })
    }

    /// Create KV caches for all layers
    /// Create KV caches for all layers (legacy, for LanguageModel trait)
    pub fn make_caches(&self) -> Vec<KVCache> {
        self.layers.iter().map(|_| KVCache::new()).collect()
    }

    /// Create Llama4 caches for iGQA pattern
    /// MoE layers (non-4th) use ChunkedKVCache, dense layers (4th) use regular KVCache
    pub fn make_llama4_caches(&self) -> Vec<Llama4Cache> {
        let chunk_size = self.attention_chunk_size as i32;
        self.layers
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                if (idx + 1) % 4 != 0 {
                    // MoE layer: use chunked cache
                    Llama4Cache::Chunked(ChunkedKVCache::new(chunk_size))
                } else {
                    // Dense layer: use regular cache
                    Llama4Cache::Regular(KVCache::new())
                }
            })
            .collect()
    }
}

impl TransformerBlock {
    /// Load transformer block from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &TextArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("language_model.model.layers.{}", layer_idx);

        // Load attention
        let self_attn = CxxAttention::from_weights(weights, args, &prefix, layer_idx)?;

        // Load feed-forward (MLP or MoE based on layer index)
        // Python: is_moe_layer = (layer_idx % step) == (step - 1)
        // For step=1: all layers are MoE
        // For step=4: layers 3, 7, 11, ... are MoE
        let is_moe_layer =
            (layer_idx % args.interleave_moe_layer_step) == (args.interleave_moe_layer_step - 1);
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

impl CxxAttention {
    /// Load attention from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &TextArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let attn_prefix = format!("{}.self_attn", prefix);

        let q_proj = load_quantized_linear(weights, &format!("{}.q_proj", attn_prefix), args)?;
        let k_proj = load_quantized_linear(weights, &format!("{}.k_proj", attn_prefix), args)?;
        let v_proj = load_quantized_linear(weights, &format!("{}.v_proj", attn_prefix), args)?;
        let o_proj = load_quantized_linear(weights, &format!("{}.o_proj", attn_prefix), args)?;

        // RoPE unused for dense layers (every 4th layer)
        let use_rope = !(layer_idx + 1).is_multiple_of(4);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim: args.head_dim as i32,
            scale: 1.0 / (args.head_dim as f32).sqrt(),
            use_rope,
            use_qk_norm: args.use_qk_norm && use_rope,
            rope_dims: args.head_dim as i32,
            rope_base: args.rope_theta,
            rope_scale: 1.0,
            attn_temperature_tuning: args.attn_temperature_tuning,
            floor_scale: args.floor_scale,
            attn_scale: args.attn_scale,
        })
    }
}

impl MLP {
    /// Load MLP from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &TextArgs,
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
        args: &TextArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let moe_prefix = format!("{}.feed_forward", prefix);

        let router = load_quantized_linear(weights, &format!("{}.router", moe_prefix), args)?;
        let experts = SwitchGLU::from_weights(weights, args, &format!("{}.experts", moe_prefix))?;

        // Load shared_expert directly (no .mlp suffix in Llama 4 weight names)
        let shared_prefix = format!("{}.shared_expert", moe_prefix);
        let shared_expert = MLP {
            gate_proj: load_quantized_linear(
                weights,
                &format!("{}.gate_proj", shared_prefix),
                args,
            )?,
            up_proj: load_quantized_linear(weights, &format!("{}.up_proj", shared_prefix), args)?,
            down_proj: load_quantized_linear(
                weights,
                &format!("{}.down_proj", shared_prefix),
                args,
            )?,
        };

        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts: args.num_local_experts,
            top_k: args.num_experts_per_tok,
        })
    }
}

impl SwitchGLU {
    /// Load SwitchGLU from weights
    pub fn from_weights(
        weights: &WeightMap,
        args: &TextArgs,
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
        args: &TextArgs,
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

// Helper functions for attention masks.
/// Create chunked attention mask for iGQA (interleaved GQA)
/// This mask limits attention to within chunk_size blocks while maintaining causality.
///
/// From Python mlx-lm llama4.py:
/// ```python
/// linds = mx.arange(start, end)           # Key positions (visible window)
/// rinds = mx.arange(offset, end)[:, None] # Query positions
/// block_pos = mx.abs((linds // chunk_size) - (rinds // chunk_size))
/// token_pos = linds <= rinds
/// chunk_mask = (block_pos == 0) & token_pos
/// ```
///
/// Args:
/// - seq_len: Current sequence length (h.shape[1])
/// - start_position: Start of visible window (from ChunkedKVCache)
/// - offset: Global position offset (total tokens processed before this batch)
/// - chunk_size: Size of each attention chunk
///
/// Returns: Boolean mask where True allows attention
// LanguageModel trait implementation.
impl LanguageModel for Llama4CxxModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Llama4CxxModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Llama4CxxModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Llama 4 EOS tokens: <|end_of_text|>, <|eom_id|>, <|eot_id|>
        vec![200001, 200007, 200008]
    }
}

/// Wrapper for Llama4CxxModel that implements LanguageModel trait
/// Uses internal Llama4Cache management for iGQA (interleaved GQA) attention pattern
pub struct Llama4Wrapper {
    model: Llama4CxxModel,
    sequence_state: ModelOwnedSequenceState<Llama4Cache>,
}

impl Llama4Wrapper {
    pub fn new(model: Llama4CxxModel) -> Self {
        let caches = model.make_llama4_caches();
        Self {
            model,
            sequence_state: ModelOwnedSequenceState::new(caches),
        }
    }

    pub fn reset_caches(&self) {
        self.sequence_state
            .replace_internal(self.model.make_llama4_caches());
    }
}

impl LanguageModel for Llama4Wrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_sequence_state(None, |sequence_caches| {
                self.model.forward_igqa(input_ids, sequence_caches, None)
            })
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_sequence_state(None, |sequence_caches| {
                self.model.forward_igqa_with_embeddings(
                    input_ids,
                    input_embeddings,
                    sequence_caches,
                    None,
                )
            })
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.model.embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.model.layers.len())
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn supports_paged_decode_backend(&self) -> bool {
        true
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.model.make_llama4_caches());
    }

    fn reset_runtime_state(&self) {
        // Used by: CxxGenerator single-row generation paths. Llama 4 owns
        // iGQA cache state inside `ModelOwnedSequenceState`; reset only the
        // legacy fallback slot, leaving scheduler-owned per-sequence entries
        // untouched.
        self.reset_caches();
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_llama4_caches(),
            |sequence_caches| self.model.forward_igqa(input_ids, sequence_caches, None),
        )
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_llama4_caches(),
            |sequence_caches| {
                self.model.forward_igqa_with_embeddings(
                    input_ids,
                    input_embeddings,
                    sequence_caches,
                    None,
                )
            },
        )
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_sequence_state(Some(seq_id), |sequence_caches| {
                let visible_lens: Vec<usize> = sequence_caches
                    .iter()
                    .map(Llama4Cache::visible_len)
                    .collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &visible_lens)
            })
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        if shape[1] != 1 || mask.is_some() {
            let input_0 = mlxcel_core::slice(input_ids, &[0, 0], &[1, shape[1]]);
            let mut result = self.forward_with_sequence_id(
                &input_0,
                seq_ids.and_then(|ids| ids.first().copied()),
                batch_caches[0],
                mask,
            );
            for (batch_idx, caches) in batch_caches.iter_mut().enumerate().skip(1) {
                let input_i = mlxcel_core::slice(
                    input_ids,
                    &[batch_idx as i32, 0],
                    &[batch_idx as i32 + 1, shape[1]],
                );
                let logits_i = self.forward_with_sequence_id(
                    &input_i,
                    seq_ids.and_then(|ids| ids.get(batch_idx).copied()),
                    caches,
                    mask,
                );
                result = mlxcel_core::concatenate(&result, &logits_i, 0);
            }
            return result;
        }

        tracing::debug!(
            batch_size = batch_caches.len(),
            seq_len = shape[1],
            "Llama4 model-owned batched decode dispatch"
        );
        self.sequence_state
            .with_batched_sequence_states(
                seq_ids.expect("llama4 batched decode requires sequence ids"),
                |sequence_caches| {
                    self.model.forward_batched_decode_with_caches(
                        input_ids,
                        sequence_caches,
                        context,
                    )
                },
            )
            .expect("llama4 batched decode requires sequence-local cache state")
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Llama 4 EOS tokens: <|end_of_text|>, <|eom_id|>, <|eot_id|>
        vec![200001, 200007, 200008]
    }
}
