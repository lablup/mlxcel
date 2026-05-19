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

//! Gemma 3 model implementation using mlxcel-core
//!
//! Key features:
//! - sliding_window_pattern (int): layer i is global if (i+1) % pattern == 0
//! - Local: RotatingKVCache + sliding window mask + local RoPE base
//! - Global: KVCache + full mask + global RoPE base
//! - Q/K norm with (1+weight) GemmaRMSNorm
//! - 4 RMSNorm per layer
//! - GELU activation
//! - clip_residual_f16 for float16 safety
//! - Embedding scaling: h *= sqrt(hidden_size)

use crate::distributed::pipeline::LayerFilter;
use crate::distributed::pipeline::StageExecutionOutput;
use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models::model_owned::{
    ModelOwnedSequenceState, dispatch_paged_decode_from_visible_caches,
};
use mlxcel_core::cache::{CachePool, RotatingPagedDecodeMetadata, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::layers::{
    FusedQKVLinear, GemmaRMSNorm, KVCache, RotatingKVCache, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::{create_causal_mask, create_causal_mask_with_window, pipeline_hint};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
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
    /// Global RoPE base frequency (used for non-sliding-window layers)
    #[serde(alias = "rope_global_base_freq")]
    pub rope_theta: f32,
    pub rope_local_base_freq: f32,
    pub query_pre_attn_scalar: f32,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,

    pub rope_scaling: Option<std::collections::HashMap<String, serde_json::Value>>,

    pub quantization: Option<Quantization>,
}

impl Default for ModelArgs {
    fn default() -> Self {
        // Defaults match Python mlx-vlm TextConfig for Gemma3
        Self {
            model_type: "gemma3_text".to_string(),
            hidden_size: 2048,
            num_hidden_layers: 26,
            intermediate_size: 16384,
            num_attention_heads: 8,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            vocab_size: 262208,
            num_key_value_heads: 4,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            query_pre_attn_scalar: 256.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            max_position_embeddings: 4096,
            rope_scaling: None,
            quantization: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
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

// Attention.
pub struct Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub is_sliding: bool,
    pub window_size: i32,
    pub rope_base: f32,
    pub q_norm: GemmaRMSNorm,
    pub k_norm: GemmaRMSNorm,
}

impl Attention {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Fused QKV projection: single matmul → split into Q, K, V
        let (q, k, v) = self.qkv_proj.forward(x);

        // Reshape and transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Apply Q/K normalization AFTER transpose (matches Python mlx-lm)
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let offset = cache.offset();

        // Apply RoPE
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Use fused scaled dot-product attention (handles GQA internally)
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let attn_out = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q,
                &cache_k,
                &cache_v,
                self.scale,
                mask_ptr,
                0.0,
                self.window_size,
            )
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut Cache],
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0] as usize;
        let seq_len = shape[1];

        let (q, k, v) = self.qkv_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[shape[0], seq_len, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[shape[0], seq_len, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[shape[0], seq_len, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let offsets: Vec<i32> = caches.iter().map(|cache| cache.offset()).collect();
        let q =
            mlxcel_core::fast_rope_batched(&q, self.head_dim, false, self.rope_base, 1.0, &offsets);
        let k =
            mlxcel_core::fast_rope_batched(&k, self.head_dim, false, self.rope_base, 1.0, &offsets);

        if let Some(context) = decode_context {
            let paged_attn = if self.is_sliding && context.is_paged_decode() {
                let mut cache_keys = Vec::with_capacity(caches.len());
                let mut cache_values = Vec::with_capacity(caches.len());
                let mut kv_lens = Vec::with_capacity(caches.len());
                let mut logical_starts = Vec::with_capacity(caches.len());

                for (batch_idx, cache) in caches.iter_mut().enumerate() {
                    let k_i = mlxcel_core::slice(
                        &k,
                        &[batch_idx as i32, 0, 0, 0],
                        &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
                    );
                    let v_i = mlxcel_core::slice(
                        &v,
                        &[batch_idx as i32, 0, 0, 0],
                        &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
                    );
                    let _ = cache.update_and_fetch(k_i, v_i);
                    kv_lens.push(cache.visible_len() as i32);
                    logical_starts.push(cache.rotating_logical_start().unwrap_or_default());
                    cache_keys.push(
                        cache
                            .keys_ptr()
                            .expect("gemma3 rotating cache should expose key buffer"),
                    );
                    cache_values.push(
                        cache
                            .values_ptr()
                            .expect("gemma3 rotating cache should expose value buffer"),
                    );
                }

                let metadata = RotatingPagedDecodeMetadata::from_parts(
                    &kv_lens,
                    &logical_starts,
                    context.paged_block_size,
                )
                .expect("valid gemma3 rotating paged decode metadata");
                let attn = if context.use_native_paged_kernel {
                    mlxcel_core::layers::paged_decode_attention_rotating_compat(
                        &q,
                        &cache_keys,
                        &cache_values,
                        &metadata,
                        self.scale,
                    )
                } else {
                    mlxcel_core::layers::paged_decode_attention_rotating_fallback(
                        &q,
                        &cache_keys,
                        &cache_values,
                        &metadata,
                        self.scale,
                    )
                }
                .expect("valid gemma3 rotating paged decode inputs");
                Some(attn)
            } else {
                dispatch_paged_decode_from_visible_caches(
                    &q,
                    &k,
                    &v,
                    caches,
                    self.scale,
                    context,
                    |cache, k_i, v_i| Ok(cache.update_and_fetch(k_i, v_i)),
                )
                .expect("valid gemma3 paged decode inputs")
            };

            if let Some(attn_out) = paged_attn {
                tracing::debug!(
                    batch_size = batch,
                    block_size = context.paged_block_size,
                    native_kernel = context.use_native_paged_kernel,
                    sliding = self.is_sliding,
                    "Gemma3 paged decode attention dispatch"
                );
                let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
                let attn_out = mlxcel_core::reshape(
                    &attn_out,
                    &[batch as i32, seq_len, self.num_heads * self.head_dim],
                );
                return self.o_proj.forward(&attn_out);
            }
        }

        let mut outputs = Vec::with_capacity(batch);
        for (batch_idx, cache) in caches.iter_mut().enumerate() {
            let q_i = mlxcel_core::slice(
                &q,
                &[batch_idx as i32, 0, 0, 0],
                &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            let k_i = mlxcel_core::slice(
                &k,
                &[batch_idx as i32, 0, 0, 0],
                &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            let v_i = mlxcel_core::slice(
                &v,
                &[batch_idx as i32, 0, 0, 0],
                &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            let (cache_k, cache_v) = cache.update_and_fetch(k_i, v_i);
            outputs.push(unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_i,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    self.window_size,
                )
            });
        }

        let mut attn_out = outputs.remove(0);
        for output in outputs {
            attn_out = mlxcel_core::concatenate(&attn_out, &output, 0);
        }
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(
            &attn_out,
            &[batch as i32, seq_len, self.num_heads * self.head_dim],
        );
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = args.head_dim as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_key_value_heads as i32;
        let scale = 1.0 / args.query_pre_attn_scalar.sqrt();

        // Determine if this is a sliding window layer
        let is_sliding = !(layer_idx + 1).is_multiple_of(args.sliding_window_pattern);

        // Choose RoPE base based on layer type
        let rope_base = if is_sliding {
            args.rope_local_base_freq
        } else {
            args.rope_theta
        };

        // Load Q/K normalization
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let q_norm = GemmaRMSNorm::new(q_norm_weight, args.rms_norm_eps);
        let k_norm = GemmaRMSNorm::new(k_norm_weight, args.rms_norm_eps);

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
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            is_sliding,
            window_size: if is_sliding {
                args.sliding_window as i32
            } else {
                0
            },
            rope_base,
            q_norm,
            k_norm,
        })
    }
}

// MLP (GELU activation).
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // GeGLU: gelu(gate_proj(x)) * up_proj(x), then down_proj
        // Quantized path: fused compiled quantized MLP
        if let (Some(gate_qw), Some(up_qw), Some(down_qw)) = (
            self.gate_proj.quantized_weight(),
            self.up_proj.quantized_weight(),
            self.down_proj.quantized_weight(),
        ) {
            return unsafe {
                mlxcel_core::compiled_gelu_mlp_forward(
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
        let activated = mlxcel_core::compiled_geglu_activation(&gate, &up);
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

// Transformer Block (4 RMSNorm per layer).
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: GemmaRMSNorm,
    pub post_attention_layernorm: GemmaRMSNorm,
    pub pre_feedforward_layernorm: GemmaRMSNorm,
    pub post_feedforward_layernorm: GemmaRMSNorm,
}

impl TransformerBlock {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let post_attn_normed = self.post_attention_layernorm.forward(&attn_out);
        let h = mlxcel_core::compiled_clip_residual(x, &post_attn_normed);

        // Pre-norm FFN
        let normed = self.pre_feedforward_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        let post_ff_normed = self.post_feedforward_layernorm.forward(&ff_out);
        mlxcel_core::compiled_clip_residual(&h, &post_ff_normed)
    }

    fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut Cache],
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self
            .self_attn
            .forward_batched_decode(&normed, caches, decode_context);
        let post_attn_normed = self.post_attention_layernorm.forward(&attn_out);
        let h = mlxcel_core::compiled_clip_residual(x, &post_attn_normed);

        let normed = self.pre_feedforward_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        let post_ff_normed = self.post_feedforward_layernorm.forward(&ff_out);
        mlxcel_core::compiled_clip_residual(&h, &post_ff_normed)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn =
            Attention::from_weights(weights, args, &format!("{}.self_attn", prefix), layer_idx)?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let pre_ff_norm_weight = get_weight_copy(
            weights,
            &format!("{}.pre_feedforward_layernorm.weight", prefix),
        )?;
        let post_ff_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_feedforward_layernorm.weight", prefix),
        )?;

        let input_layernorm = GemmaRMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = GemmaRMSNorm::new(post_attn_norm_weight, args.rms_norm_eps);
        let pre_feedforward_layernorm = GemmaRMSNorm::new(pre_ff_norm_weight, args.rms_norm_eps);
        let post_feedforward_layernorm = GemmaRMSNorm::new(post_ff_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }
}

// Cache Interface.
pub(crate) trait CacheInterface {
    fn offset(&self) -> i32;
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);
}

impl CacheInterface for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }
}

impl CacheInterface for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }
}

pub(crate) enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    pub(crate) fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Cache::Standard(c) => c,
            Cache::Rotating(c) => c,
        }
    }

    pub(crate) fn offset(&self) -> i32 {
        match self {
            Cache::Standard(c) => c.offset,
            Cache::Rotating(c) => c.offset,
        }
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self {
            Cache::Standard(c) => c.update_and_fetch(k, v),
            Cache::Rotating(c) => c.update_and_fetch(k, v),
        }
    }

    fn visible_len(&self) -> usize {
        match self {
            Cache::Standard(c) => c.seq_len().max(0) as usize,
            Cache::Rotating(c) => c.visible_len().max(0) as usize,
        }
    }

    fn rotating_logical_start(&self) -> Option<i32> {
        match self {
            Cache::Standard(_) => None,
            Cache::Rotating(c) => Some(c.logical_start()),
        }
    }

    fn keys_ptr(&self) -> Option<*const MlxArray> {
        match self {
            Cache::Standard(c) => c
                .keys
                .as_ref()
                .map(|keys| keys.as_ref().unwrap() as *const _),
            Cache::Rotating(c) => c
                .keys
                .as_ref()
                .map(|keys| keys.as_ref().unwrap() as *const _),
        }
    }

    fn values_ptr(&self) -> Option<*const MlxArray> {
        match self {
            Cache::Standard(c) => c
                .values
                .as_ref()
                .map(|values| values.as_ref().unwrap() as *const _),
            Cache::Rotating(c) => c
                .values
                .as_ref()
                .map(|values| values.as_ref().unwrap() as *const _),
        }
    }
}

// Gemma3 Model.
pub struct Gemma3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: GemmaRMSNorm,
    pub lm_head: UnifiedLinear,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub hidden_size: usize,
}

impl Gemma3Model {
    /// Get token embeddings scaled by sqrt(hidden_size)
    /// Used by: VisionModule for merging vision and text embeddings
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        let scale = (self.hidden_size as f32).sqrt();
        mlxcel_core::multiply_scalar(&h, scale)
    }

    /// Forward pass with pre-computed embeddings (for VLM)
    /// If input_embeddings is Some, skip embed_tokens and use provided embeddings
    pub(crate) fn forward_with_caches_and_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        external_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[1];

        // Use pre-computed embeddings if provided, otherwise embed tokens.
        // In both cases, apply Gemma's sqrt(hidden_size) normalizer.
        // Python: h *= mx.array(self.args.hidden_size**0.5, mx.bfloat16)
        let mut h = if let Some(embeddings) = input_embeddings {
            mlxcel_core::copy(embeddings)
        } else {
            self.embed_tokens.forward(input_ids)
        };
        // Gemma scaling: h *= sqrt(hidden_size)
        let scale = (self.hidden_size as f32).sqrt();
        h = mlxcel_core::multiply_scalar(&h, scale);

        let n = self.layers.len();
        // If external 4D mask is provided (from VLM), use it for all layers.
        // Python mlx-vlm passes a 0/1 INT32 mask from prepare_inputs_for_multimodal
        // directly to all layers, bypassing the causal mask creation.
        // When attention_mask is all-ones (no padding), this creates a bidirectional
        // mask that lets all tokens attend to all others during prefill.
        if let Some(ext_mask) = external_mask {
            // VLM provides a 4D attention mask -- apply it to all layers
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(&h, caches[i].as_interface(), Some(ext_mask));
                pipeline_hint(&h, i, n);
            }
        } else if seq_len == 1 {
            // Decode path (seq_len=1): no mask needed — matches Python mlx-lm
            // which returns None from create_attention_mask when N=1.
            // The fused SDPA handles single-token attention without explicit masks.
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(&h, caches[i].as_interface(), None);
                pipeline_hint(&h, i, n);
            }
        } else {
            // Prefill path (seq_len > 1): create causal masks
            let global_idx = self.sliding_window_pattern - 1;
            let global_offset = caches[global_idx].as_interface().offset();
            let global_mask = Some(create_causal_mask(seq_len, global_offset));

            let sliding_mask = if self.sliding_window_pattern > 1 {
                let sliding_offset = caches[0].as_interface().offset();
                // Clamp offset so mask shape matches RotatingKVCache output.
                // The cache returns at most max_size tokens, so the mask's
                // total_len (= seq_len + offset) must not exceed max_size.
                let max_cache = self.sliding_window as i32;
                let effective_offset = sliding_offset.min((max_cache - seq_len).max(0));
                Some(create_causal_mask_with_window(
                    seq_len,
                    effective_offset,
                    Some(max_cache),
                ))
            } else {
                None
            };

            for (i, layer) in self.layers.iter().enumerate() {
                let is_global =
                    (i % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                let mask = if is_global {
                    global_mask.as_ref().map(|m| m.as_ref().unwrap())
                } else {
                    sliding_mask.as_ref().map(|m| m.as_ref().unwrap())
                };
                h = layer.forward(&h, caches[i].as_interface(), mask);
                pipeline_hint(&h, i, n);
            }
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        self.lm_head.forward(&h)
    }

    /// Forward pass through the entire model
    pub(crate) fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        self.forward_with_caches_and_embeddings(input_ids, None, caches, None)
    }

    /// Create KV caches for all layers
    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        (0..self.layers.len())
            .map(|i| {
                let is_global =
                    (i % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                if is_global {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                }
            })
            .collect()
    }

    fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [Vec<Cache>],
        decode_context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        let scale = (self.hidden_size as f32).sqrt();
        h = mlxcel_core::multiply_scalar(&h, scale);

        let n = self.layers.len();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut layer_caches: Vec<&mut Cache> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches[layer_idx])
                .collect();
            h = layer.forward_batched_decode(&h, &mut layer_caches, decode_context);
            pipeline_hint(&h, layer_idx, n);
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
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

        // Load weights (with tied-embedding sanitization)
        let weights = crate::models::load_text_weights(model_dir, None)?;

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
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = GemmaRMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: args.sliding_window,
            sliding_window_pattern: args.sliding_window_pattern,
            hidden_size: args.hidden_size,
        })
    }
}

pub(crate) struct Gemma3StageModel {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    layers: Vec<TransformerBlock>,
    norm: Option<GemmaRMSNorm>,
    lm_head: Option<UnifiedLinear>,
    sliding_window: usize,
    sliding_window_pattern: usize,
    hidden_size: usize,
}

impl Gemma3StageModel {
    pub(crate) fn load(
        model_dir: &Path,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read {}: {}", config_path.display(), e))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse {}: {}", config_path.display(), e))?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        filter_weight_map(&mut weights, filter);
        Self::from_filtered_weights(&weights, &args, filter, stage_index)
    }

    fn from_filtered_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let embed_tokens = if filter.has_embedding {
            Some(UnifiedEmbedding::from_weights(
                weights,
                "model.embed_tokens",
                group_size,
                bits,
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            layers.push(TransformerBlock::from_weights(weights, args, layer_idx)?);
        }

        if layers.is_empty() {
            return Err(format!(
                "stage {} did not load any layers from range {}..{}",
                stage_index, filter.layer_range.start, filter.layer_range.end
            ));
        }

        let (norm, lm_head) = if filter.has_lm_head {
            let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
            let norm = GemmaRMSNorm::new(norm_weight, args.rms_norm_eps);
            let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;
            (Some(norm), Some(lm_head))
        } else {
            (None, None)
        };

        Ok(Self {
            filter: filter.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: args.sliding_window,
            sliding_window_pattern: args.sliding_window_pattern,
            hidden_size: args.hidden_size,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        (0..self.layers.len())
            .map(|layer_idx| {
                let global_idx = self.global_layer_index(layer_idx);
                let is_global =
                    (global_idx % self.sliding_window_pattern) == (self.sliding_window_pattern - 1);
                if is_global {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                }
            })
            .collect()
    }

    pub(crate) fn execute_from_token_ids(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        let mut hidden = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| {
                "stage does not host embeddings; hidden-state input required".to_string()
            })?
            .forward(input_ids);
        hidden = mlxcel_core::multiply_scalar(&hidden, (self.hidden_size as f32).sqrt());
        self.execute_hidden(hidden, caches)
    }

    pub(crate) fn execute_from_hidden_states(
        &self,
        hidden: UniquePtr<MlxArray>,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        if self.filter.has_embedding {
            return Err("entry stage expects token IDs, not hidden states".to_string());
        }
        self.execute_hidden(hidden, caches)
    }

    fn execute_hidden(
        &self,
        mut hidden: UniquePtr<MlxArray>,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        if caches.len() != self.layers.len() {
            return Err(format!(
                "stage cache count mismatch: expected {}, got {}",
                self.layers.len(),
                caches.len()
            ));
        }

        let seq_len = mlxcel_core::array_shape(hidden.as_ref().unwrap())[1];
        if seq_len > 1 {
            let global_offset = self
                .first_global_cache_index()
                .map(|idx| caches[idx].offset())
                .unwrap_or(0);
            let global_mask = create_causal_mask(seq_len, global_offset);

            let sliding_mask = if self.first_sliding_cache_index().is_some() {
                let sliding_offset = caches[self.first_sliding_cache_index().unwrap()].offset();
                let max_cache = self.sliding_window as i32;
                let effective_offset = sliding_offset.min((max_cache - seq_len).max(0));
                Some(create_causal_mask_with_window(
                    seq_len,
                    effective_offset,
                    Some(max_cache),
                ))
            } else {
                None
            };

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mask = if self.is_global_layer(layer_idx) {
                    Some(global_mask.as_ref().unwrap() as &MlxArray)
                } else {
                    sliding_mask.as_deref()
                };
                hidden = layer.forward(
                    hidden.as_ref().unwrap(),
                    caches[layer_idx].as_interface(),
                    mask,
                );
            }
        } else {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward(
                    hidden.as_ref().unwrap(),
                    caches[layer_idx].as_interface(),
                    None,
                );
            }
        }

        match (&self.norm, &self.lm_head) {
            (Some(norm), Some(lm_head)) => {
                let hidden = norm.forward(hidden.as_ref().unwrap());
                Ok(StageExecutionOutput::Logits(lm_head.forward(&hidden)))
            }
            _ => Ok(StageExecutionOutput::HiddenStates(hidden)),
        }
    }

    fn global_layer_index(&self, local_idx: usize) -> usize {
        self.filter.layer_range.start + local_idx
    }

    fn is_global_layer(&self, local_idx: usize) -> bool {
        let global_idx = self.global_layer_index(local_idx);
        (global_idx % self.sliding_window_pattern) == (self.sliding_window_pattern - 1)
    }

    fn first_global_cache_index(&self) -> Option<usize> {
        (0..self.layers.len()).find(|&idx| self.is_global_layer(idx))
    }

    fn first_sliding_cache_index(&self) -> Option<usize> {
        (0..self.layers.len()).find(|&idx| !self.is_global_layer(idx))
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

/// Wrapper for Gemma3Model that implements LanguageModel trait
/// Uses internal cache management for sliding window attention
pub struct Gemma3Wrapper {
    model: Gemma3Model,
    sequence_state: ModelOwnedSequenceState<Cache>,
}

impl Gemma3Wrapper {
    pub fn new(model: Gemma3Model) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            sequence_state: ModelOwnedSequenceState::new(caches),
        }
    }

    pub fn reset_caches(&self) {
        self.sequence_state
            .replace_internal(self.model.make_caches());
    }
}

impl mlxcel_core::generate::LanguageModel for Gemma3Wrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_sequence_id(input_ids, None, caches, None)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            None,
            caches,
            mask,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        // Return RAW embeddings (without sqrt(hidden_size) scaling).
        // Python mlx-vlm calls embed_tokens() to get unscaled embeddings for VLM merge,
        // then the language model forward applies sqrt(hidden_size) to ALL embeddings
        // (both text and merged image features). Returning scaled embeddings here would
        // cause double-scaling for text tokens while leaving image features unscaled.
        Some(self.model.embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
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
            .prepare_sequence_state(seq_id, self.model.make_caches());
    }

    fn reset_runtime_state(&self) {
        // Used by: CxxGenerator single-row generation paths. Gemma 3 owns
        // its fallback sliding/global KV cache state inside
        // `ModelOwnedSequenceState`; the caller-provided `KVCache` slice is
        // intentionally empty/ignored. Reset the fallback slot before each
        // fresh CLI / benchmark generation so a second VLM prefill does not
        // reuse offsets from a previous run with a stale 4D attention mask
        // (issue #731).
        self.reset_caches();
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| self.model.forward_with_caches(input_ids, sequence_caches),
        )
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [mlxcel_core::layers::KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| {
                self.model.forward_with_caches_and_embeddings(
                    input_ids,
                    input_embeddings,
                    sequence_caches,
                    mask,
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
                let visible_lens: Vec<usize> =
                    sequence_caches.iter().map(Cache::visible_len).collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &visible_lens)
            })
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [mlxcel_core::layers::KVCache]],
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

        self.sequence_state
            .with_batched_sequence_states(
                seq_ids.expect("gemma3 batched decode requires sequence ids"),
                |sequence_caches| {
                    self.model.forward_batched_decode_with_caches(
                        input_ids,
                        sequence_caches,
                        context,
                    )
                },
            )
            .expect("gemma3 batched decode requires sequence-local cache state")
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0, 1, 106] // Gemma3: <pad> (0), <eos> (1), <end_of_turn> (106)
    }
}
