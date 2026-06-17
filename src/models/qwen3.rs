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

//! Qwen3 model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Q/K normalization (RMSNorm after projection, before RoPE)
//! - Explicit head_dim in config

use mlxcel_core::cache::{BatchedAttentionMetadata, PagedDecodeMetadata};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{
    FusedQKVLinear, KVCache, KVCacheMode, RMSNorm, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::pipeline_hint;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

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

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_tie_word_embeddings() -> bool {
    true
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

// Attention with Q/K Normalization.
pub struct Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm, // Q normalization
    pub k_norm: RMSNorm, // K normalization
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
        let offset = cache.offset;

        // On decode (l == 1) collapse the QKV projection, split, Q/K RMSNorm and
        // RoPE into one fused C++ kernel to cut per-token op count (#326). The
        // norm reduces over head_dim, which the head transpose leaves untouched,
        // so the fused result matches the graph path below. Prefill (l > 1),
        // non-quantized weights (the kernel returns None), and
        // MLXCEL_FUSED_QK_NORM=0 all take the graph path.
        let fused = if l == 1 && mlxcel_core::layers::fused_qk_norm_enabled() {
            self.qkv_proj.forward_split_norm_rope_quantized(
                x,
                &self.q_norm,
                &self.k_norm,
                self.rope_dims,
                self.rope_base,
                offset,
            )
        } else {
            None
        };

        let (q, k, v) = if let Some((q, k, v)) = fused {
            (q, k, v)
        } else {
            // Fused QKV projection: single matmul → split into Q, K, V
            let (q, k, v) = self.qkv_proj.forward(x);

            // Reshape to [batch, seq_len, n_heads, head_dim]
            let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
            let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
            let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

            // Apply Q/K normalization BEFORE transpose
            let q = self.q_norm.forward(&q);
            let k = self.k_norm.forward(&k);

            // Transpose to [batch, n_heads, seq_len, head_dim]
            let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
            let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
            let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

            // Apply RoPE AFTER normalization
            let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
            let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);
            (q, k, v)
        };

        // Try the fused sparse-V SDPA path for the decode case
        // (l == 1) when the cache is in Turbo4Asym mode
        // and `MLXCEL_SPARSE_V_THRESHOLD > 0`. Skipped on prefill (l > 1)
        // because the per-token skip only wins at long context — and
        // prefill builds the cache from scratch, so there's nothing to
        // skip yet.
        // Turbo4 compressed attention mirrors mlx-swift-lm's current decode
        // policy: prefer dequant-first native SDPA by default. Delegated mode
        // keeps the custom packed-V Metal kernels available as a forced
        // fallback.
        // The gate is parsed once and cached in a `OnceLock<bool>` — see
        // `mlxcel_core::cache::turbo::sparse_v::turbo4_delegated_compressed_attention_enabled`.
        let use_delegated_compressed =
            mlxcel_core::cache::turbo::sparse_v::turbo4_delegated_compressed_attention_enabled();
        let use_turbo4_dequant_sdpa =
            mlxcel_core::cache::turbo::sparse_v::turbo4_dequant_sdpa_enabled();
        let attn_out = if l == 1 && cache.sparse_v_available() {
            // The helper consumes k/v, fills the packed cache, and runs
            // the fused kernel (or graph fallback). When `sparse_v_available`
            // is true the helper always returns Some.
            cache
                .update_and_sparse_v_attention(&q, k, v, self.scale, mask)
                .expect("update_and_sparse_v_attention returned None despite sparse_v_available")
        } else if l == 1 && use_turbo4_dequant_sdpa && cache.turbo4_dequant_sdpa_available() {
            cache.update_and_turbo4_dequant_sdpa_attention(&q, k, v, self.scale, mask)
        } else if l == 1 && use_delegated_compressed && cache.turbo4_delegated_available() {
            // The helper always produces an attention output: it routes
            // through the fused Metal kernel when available and falls
            // through to the graph-only reference path otherwise.
            cache.update_and_turbo4_delegated_attention(&q, k, v, self.scale, mask)
        } else if l > 1 && mask.is_none() {
            let (cache_k, cache_v) = cache.update_and_fetch(k, v);
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let (cache_k, cache_v) = cache.update_and_fetch(k, v);
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

    /// Split-attention forward for batched decode.
    ///
    /// Receives pre-projected Q/K/V tensors of shape `[B, T, proj_dim]`,
    /// applies Q/K normalization and RoPE using batched positional metadata,
    /// then runs per-sequence cache updates and attention before concatenating
    /// the results back into `[B, T, hidden_dim]`.
    ///
    /// Key difference from Llama3: Q/K normalization (RMSNorm) still happens
    /// before RoPE, but it now stays batched instead of forcing a per-sequence
    /// loop for positional handling.
    ///
    /// Used by: Qwen3 batched decode (TransformerBlock::forward_batched)
    pub fn forward_split_attention(
        &self,
        q_batched: &MlxArray,
        k_batched: &MlxArray,
        v_batched: &MlxArray,
        caches: &mut [&mut KVCache],
        metadata: &BatchedAttentionMetadata,
        mask: Option<&MlxArray>,
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let b = caches.len();
        let seq_len = mlxcel_core::array_shape(q_batched)[1];
        debug_assert_eq!(metadata.len(), b);
        let mut attn_outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(b);

        let q_batched = mlxcel_core::reshape(
            q_batched,
            &[b as i32, seq_len, self.num_heads, self.head_dim],
        );
        let k_batched = mlxcel_core::reshape(
            k_batched,
            &[b as i32, seq_len, self.num_kv_heads, self.head_dim],
        );
        let v_batched = mlxcel_core::reshape(
            v_batched,
            &[b as i32, seq_len, self.num_kv_heads, self.head_dim],
        );

        let q_batched = self.q_norm.forward(&q_batched);
        let k_batched = self.k_norm.forward(&k_batched);

        let q_batched = mlxcel_core::transpose_axes(&q_batched, &[0, 2, 1, 3]);
        let k_batched = mlxcel_core::transpose_axes(&k_batched, &[0, 2, 1, 3]);
        let v_batched = mlxcel_core::transpose_axes(&v_batched, &[0, 2, 1, 3]);

        let q_batched = mlxcel_core::fast_rope_batched(
            &q_batched,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            &metadata.rope_offsets,
        );
        let k_batched = mlxcel_core::fast_rope_batched(
            &k_batched,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            &metadata.rope_offsets,
        );

        let paged_decode = decode_context.and_then(|context| {
            if seq_len != 1 || mask.is_some() || !context.is_paged_decode() {
                return None;
            }
            if caches.iter().any(|cache| cache.mode != KVCacheMode::Fp16) {
                return None;
            }
            // Pool-backed caches (scheduler paged decode, #121) keep no dense
            // `keys`/`values` buffers for the native paged kernel to read.
            // Route them through the per-sequence `update_and_fetch` loop below,
            // whose transparent pool intercept writes new K/V into the shared
            // `PagedBlockPool` (`write_prefill`) and gathers the visible window
            // back (`gather_visible`) — the #152-validated single-stream path.
            if caches.iter().any(|cache| cache.is_paged_backed()) {
                return None;
            }
            let metadata =
                PagedDecodeMetadata::from_attention_metadata(metadata, context.paged_block_size)
                    .ok()?;
            Some((context.use_native_paged_kernel, metadata))
        });

        if let Some((use_native_kernel, paged_metadata)) = paged_decode {
            tracing::debug!(
                batch_size = b,
                block_size = paged_metadata.block_size,
                native_kernel = use_native_kernel,
                "Qwen3 paged decode attention dispatch"
            );
            let mut cache_keys: Vec<*const MlxArray> = Vec::with_capacity(b);
            let mut cache_values: Vec<*const MlxArray> = Vec::with_capacity(b);

            for (i, cache) in caches.iter_mut().enumerate() {
                let k_i = mlxcel_core::slice(
                    &k_batched,
                    &[i as i32, 0, 0, 0],
                    &[i as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
                );
                let v_i = mlxcel_core::slice(
                    &v_batched,
                    &[i as i32, 0, 0, 0],
                    &[i as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
                );
                cache.update(k_i, v_i);
                cache_keys.push(cache.keys.as_ref().unwrap().as_ref().unwrap() as *const MlxArray);
                cache_values
                    .push(cache.values.as_ref().unwrap().as_ref().unwrap() as *const MlxArray);
            }

            let attn_out = if use_native_kernel {
                mlxcel_core::layers::paged_decode_attention_dense_compat(
                    &q_batched,
                    &cache_keys,
                    &cache_values,
                    &paged_metadata,
                    self.scale,
                )
            } else {
                mlxcel_core::layers::paged_decode_attention_dense_fallback(
                    &q_batched,
                    &cache_keys,
                    &cache_values,
                    &paged_metadata,
                    self.scale,
                )
            }
            .expect("valid qwen3 paged decode attention inputs");

            let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
            return mlxcel_core::reshape(
                &attn_out,
                &[b as i32, seq_len, self.num_heads * self.head_dim],
            );
        }

        for (i, cache) in caches.iter_mut().enumerate() {
            // Slice [B, heads, T, dim] -> [1, heads, T, dim] for sequence i.
            let q_i = mlxcel_core::slice(
                &q_batched,
                &[i as i32, 0, 0, 0],
                &[i as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            let k_i = mlxcel_core::slice(
                &k_batched,
                &[i as i32, 0, 0, 0],
                &[i as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            let v_i = mlxcel_core::slice(
                &v_batched,
                &[i as i32, 0, 0, 0],
                &[i as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );

            // Update KV cache
            let (cache_k, cache_v) = cache.update_and_fetch(k_i, v_i);

            let mask_i = mask.map(|m| {
                let sliced =
                    mlxcel_core::slice(m, &[i as i32, 0, 0], &[i as i32 + 1, seq_len, i32::MAX]);
                mlxcel_core::squeeze_axis(&sliced, 0)
            });

            let attn_out = if seq_len > 1 && mask_i.is_none() {
                mlxcel_core::causal_attention(&q_i, &cache_k, &cache_v, self.scale, 0.0, 0)
            } else {
                let mask_ptr = mask_i
                    .as_ref()
                    .map(|m| m.as_ref().unwrap() as *const _)
                    .unwrap_or(std::ptr::null());
                unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        &q_i, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                    )
                }
            };

            // Transpose back: [1, n_heads, T, head_dim] -> [1, T, n_heads * head_dim]
            let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
            let attn_out =
                mlxcel_core::reshape(&attn_out, &[1, seq_len, self.num_heads * self.head_dim]);

            attn_outputs.push(attn_out);
        }

        // Concatenate along batch dim: B * [1, T, hidden] -> [B, T, hidden]
        let mut result = attn_outputs.remove(0);
        for attn_out in attn_outputs {
            result = mlxcel_core::concatenate(&result, &attn_out, 0);
        }
        result
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

        // Load Q/K normalization weights
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let head_dim = args.head_dim as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_key_value_heads as i32;

        // Fused QKV: concatenate q/k/v weights into one projection at load time
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
            q_norm: RMSNorm::new(q_norm_weight, args.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, args.rms_norm_eps),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
        })
    }
}

// MLP (SwiGLU).
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Non-quantized path: fused compiled FP MLP (single compiled graph)
        if let Some(result) = mlxcel_core::layers::compiled_swiglu_mlp_fp16(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return result;
        }

        // Quantized path: separate projections + compiled SwiGLU activation
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
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
    pub mlp: MLP,
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

        // Pre-norm FFN
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    /// Batched forward: batch norms + projections + FFN, per-sequence attention.
    ///
    /// `x` has shape `[B, T, hidden_dim]`, `caches[i]` is the KVCache for
    /// the i-th sequence. Returns `[B, T, hidden_dim]`.
    ///
    /// Used by: Qwen3Model::forward_batched_impl
    pub fn forward_batched(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        mask: Option<&MlxArray>,
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        // Batched pre-attention norm
        let normed = self.input_layernorm.forward(x);

        // Batched Q/K/V projection (fused single matmul)
        let (q, k, v) = self.self_attn.qkv_proj.forward(&normed);
        let seq_len = mlxcel_core::array_shape(&q)[1];
        let metadata = BatchedAttentionMetadata::uniform_kv_caches(caches, seq_len, 0)
            .expect("valid qwen3 batched attention metadata");

        // Per-sequence attention still owns cache mutation, but positional
        // metadata and RoPE now stay on a batched path.
        let attn_concat = self.self_attn.forward_split_attention(
            &q,
            &k,
            &v,
            caches,
            &metadata,
            mask,
            decode_context,
        );

        // Batched output projection
        let attn_out = self.self_attn.o_proj.forward(&attn_concat);

        // Residual connection
        let h = mlxcel_core::add(x, &attn_out);

        // Batched post-attention norm + FFN
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
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

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

// Qwen3 Model.
pub struct Qwen3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub tie_word_embeddings: bool,
}

impl Qwen3Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, None, caches, mask)
    }

    /// Forward with optional pre-computed embeddings (for VLM prefill).
    /// Used by: MiniCPM-o VLM
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeddings) = input_embeddings {
            mlxcel_core::copy(embeddings)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        // Pass through transformer layers
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            pipeline_hint(&h, i, n);
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

    /// Batched forward pass: batch compute-bound layers, per-sequence attention.
    ///
    /// `input_ids` has shape `[B, T]`. `batch_caches[i]` is the per-layer
    /// KV cache slice for the i-th sequence. Returns `[B, T, vocab_size]`.
    ///
    /// This is the explicit batched implementation that amortizes weight-loading
    /// bandwidth for embedding, normalization, linear projections, and FFN/MLP
    /// across all B sequences, while running attention per-sequence to handle
    /// different KV cache lengths and RoPE offsets.
    ///
    /// Used by: LanguageModel::forward_batched (overrides the loop-based default)
    pub fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let b = batch_caches.len();

        // Batched embedding lookup: [B, 1] -> [B, 1, hidden_dim]
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers with split-attention
        for layer_idx in 0..self.layers.len() {
            // Collect per-sequence caches for this layer
            let mut layer_caches: Vec<&mut KVCache> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches[layer_idx])
                .collect();

            h = self.layers[layer_idx].forward_batched(&h, &mut layer_caches, mask, decode_context);
        }

        // Batched final norm: [B, 1, hidden_dim]
        let h = self.norm.forward(&h);

        // Batched lm_head: [B, 1, vocab_size]
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        };

        // Sanity check in debug builds
        debug_assert_eq!(mlxcel_core::array_shape(&logits)[0], b as i32);

        logits
    }

    /// Get raw token embeddings (for VLM embedding merge).
    /// Used by: MiniCPM-o VLM
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
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

        // Load quantized embedding
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

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for Qwen3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Qwen3Model::forward(self, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Qwen3Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![151643, 151645] // Qwen3 EOS tokens
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_impl(input_ids, batch_caches, mask, None)
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_impl(input_ids, batch_caches, mask, context)
    }

    fn supports_batched_prefill(&self) -> bool {
        true
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        true
    }

    fn supports_paged_decode_backend(&self) -> bool {
        true
    }
}
