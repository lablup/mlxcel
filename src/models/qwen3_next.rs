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

//! Qwen3-Next: Hybrid Transformer + Linear Attention (GatedDeltaNet) + MoE
//!
//! Key Features:
//! - Hybrid architecture alternating between full attention and linear attention (GatedDeltaNet)
//! - GatedDeltaNet for efficient linear attention with gated delta rule
//! - Sparse MoE layers with shared experts
//! - Conv1d preprocessing for linear attention layers
//! - Mixed cache types: KVCache for attention, MambaCache for linear attention
//! - Gated output in attention blocks (sigmoid gating)
//! - Q/K normalization
//!
//! Reference: mlx-lm/mlx_lm/models/qwen3_next.py

#[path = "qwen3_next_helpers.rs"]
mod helpers;

#[cfg(test)]
#[path = "qwen3_next_helpers_tests.rs"]
mod helper_tests;

use crate::models::gated_delta::{
    GatedDeltaCache, RMSNormGated, gated_delta_update, scaled_fast_rms_norm_no_weight,
};
use mlxcel_core::dtype;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, silu, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

use self::helpers::{build_projection_layout, split_conv_output_ranges};

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3NextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    // Linear attention parameters
    pub linear_num_value_heads: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,

    // MoE parameters
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,

    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    pub vocab_size: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default)]
    pub norm_topk_prob: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_partial_rotary_factor() -> f32 {
    0.5
}
fn default_full_attention_interval() -> usize {
    4
}

impl Qwen3NextConfig {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn rope_dims(&self) -> i32 {
        (self.head_dim as f32 * self.partial_rotary_factor) as i32
    }

    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        !(layer_idx + 1).is_multiple_of(self.full_attention_interval)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

// Cache Types.
/// Mixed cache type for Qwen3Next layers
pub enum Qwen3NextCache {
    Attention(KVCache),
    Linear(GatedDeltaCache),
}

impl Qwen3NextCache {
    pub fn offset(&self) -> i32 {
        match self {
            Qwen3NextCache::Attention(kv) => kv.offset,
            Qwen3NextCache::Linear(gd) => gd.offset,
        }
    }
}

// GatedDeltaNet - Linear Attention Layer.
/// GatedDeltaNet layer
#[allow(dead_code)]
pub(crate) struct GatedDeltaNet {
    hidden_size: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_dim: usize,

    conv1d_weight: UniquePtr<MlxArray>,
    in_proj_qkvz: UnifiedLinear,
    in_proj_ba: UnifiedLinear,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    norm: RMSNormGated,
    out_proj: UnifiedLinear,
}

impl GatedDeltaNet {
    fn fix_query_key_value_ordering(
        &self,
        mixed_qkvz: &MlxArray,
        mixed_ba: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let shape = mlxcel_core::array_shape(mixed_qkvz);
        let batch_dims = &shape[..shape.len() - 1];
        let layout = build_projection_layout(
            batch_dims,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        );

        let mixed_qkvz = mlxcel_core::reshape(mixed_qkvz, &layout.mixed_qkvz_shape);
        let mixed_ba = mlxcel_core::reshape(mixed_ba, &layout.mixed_ba_shape);

        // Split qkvz into q, k, v, z
        let mut starts = vec![0i32; layout.mixed_qkvz_shape.len()];
        let mut stops = layout.mixed_qkvz_shape.clone();
        let last_axis = layout.mixed_qkvz_shape.len() - 1;

        starts[last_axis] = layout.q_range.0;
        stops[last_axis] = layout.q_range.1;
        let q = mlxcel_core::slice(&mixed_qkvz, &starts, &stops);

        starts[last_axis] = layout.k_range.0;
        stops[last_axis] = layout.k_range.1;
        let k = mlxcel_core::slice(&mixed_qkvz, &starts, &stops);

        starts[last_axis] = layout.v_range.0;
        stops[last_axis] = layout.v_range.1;
        let v = mlxcel_core::slice(&mixed_qkvz, &starts, &stops);

        starts[last_axis] = layout.z_range.0;
        stops[last_axis] = layout.z_range.1;
        let z = mlxcel_core::slice(&mixed_qkvz, &starts, &stops);

        // Split ba into b, a
        let ba_last_axis = layout.mixed_ba_shape.len() - 1;
        let mut ba_starts = vec![0i32; layout.mixed_ba_shape.len()];
        let mut ba_stops = layout.mixed_ba_shape.clone();

        ba_starts[ba_last_axis] = layout.b_range.0;
        ba_stops[ba_last_axis] = layout.b_range.1;
        let b = mlxcel_core::slice(&mixed_ba, &ba_starts, &ba_stops);

        ba_starts[ba_last_axis] = layout.a_range.0;
        ba_stops[ba_last_axis] = layout.a_range.1;
        let a = mlxcel_core::slice(&mixed_ba, &ba_starts, &ba_stops);

        // Reshape v, z, b, a
        let v = mlxcel_core::reshape(&v, &layout.v_shape);
        let z = mlxcel_core::reshape(&z, &layout.v_shape);
        let b = mlxcel_core::reshape(&b, &layout.ba_final_shape);
        let a = mlxcel_core::reshape(&a, &layout.ba_final_shape);

        (q, k, v, z, b, a)
    }

    fn forward(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        mut cache: Option<&mut GatedDeltaCache>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        let b = shape[0];
        let s = shape[1];

        // Pre-compute projections
        let qkvz_proj = self.in_proj_qkvz.forward(inputs);
        let ba_proj = self.in_proj_ba.forward(inputs);
        let (q, k, v, z, b_proj, a) = self.fix_query_key_value_ordering(&qkvz_proj, &ba_proj);

        // Get conv state from cache
        let input_dtype = mlxcel_core::array_dtype(&q);
        let conv_state = if let Some(ref c) = cache {
            c.conv_state
                .as_ref()
                .map(|s| mlxcel_core::copy(s.as_ref().unwrap()))
                .unwrap_or_else(|| {
                    mlxcel_core::zeros(
                        &[b, (self.conv_kernel_size - 1) as i32, self.conv_dim as i32],
                        input_dtype,
                    )
                })
        } else {
            mlxcel_core::zeros(
                &[b, (self.conv_kernel_size - 1) as i32, self.conv_dim as i32],
                input_dtype,
            )
        };

        // Concatenate q, k, v for conv
        let q_flat = mlxcel_core::reshape(&q, &[b, s, -1]);
        let k_flat = mlxcel_core::reshape(&k, &[b, s, -1]);
        let v_flat = mlxcel_core::reshape(&v, &[b, s, -1]);
        let qk = concatenate(&q_flat, &k_flat, -1);
        let mixed_qkv = concatenate(&qk, &v_flat, -1);

        // Apply mask if present
        let mixed_qkv = if let Some(m) = mask {
            let m_exp = mlxcel_core::expand_dims(m, -1);
            let zero = mlxcel_core::full_f32(&[1], 0.0, input_dtype);
            mlxcel_core::where_cond(&m_exp, &mixed_qkv, &zero)
        } else {
            mixed_qkv
        };

        // Concatenate with conv state
        let conv_input = concatenate(&conv_state, &mixed_qkv, 1);

        // Update cache with new conv state.
        // Wrap slice in contiguous() to force MLX to materialize a fresh,
        // independent buffer. Without this, the slice is a lazy view that
        // retains a reference to the full conv_input allocation, causing a
        // memory leak proportional to the sequence length.
        if let Some(c) = cache.as_deref_mut() {
            let n_keep = (self.conv_kernel_size - 1) as i32;
            let conv_shape = mlxcel_core::array_shape(&conv_input);
            let conv_len = conv_shape[1];
            let tail = mlxcel_core::slice(
                &conv_input,
                &[0, conv_len - n_keep, 0],
                &[b, conv_len, self.conv_dim as i32],
            );
            c.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        }

        // Apply conv1d with SiLU activation
        let conv_out = mlxcel_core::conv1d(
            &conv_input,
            &self.conv1d_weight,
            1,
            0,
            1,
            self.conv_dim as i32,
        );
        let conv_out = silu(&conv_out);

        // Split back into q, k, v
        // Note: MLX slice with stop=-1 means dim_size-1 (excludes last), not "to end"
        // Use actual conv_out seq length for correct slicing
        let conv_out_shape = mlxcel_core::array_shape(&conv_out);
        let conv_seq = conv_out_shape[1];
        let [q_range, k_range, v_range] = split_conv_output_ranges(self.key_dim, self.conv_dim);
        let q_out = mlxcel_core::slice(&conv_out, &[0, 0, q_range.0], &[b, conv_seq, q_range.1]);
        let k_out = mlxcel_core::slice(&conv_out, &[0, 0, k_range.0], &[b, conv_seq, k_range.1]);
        let v_out = mlxcel_core::slice(&conv_out, &[0, 0, v_range.0], &[b, conv_seq, v_range.1]);

        // Reshape to heads
        let q = mlxcel_core::reshape(
            &q_out,
            &[b, s, self.num_k_heads as i32, self.head_k_dim as i32],
        );
        let k = mlxcel_core::reshape(
            &k_out,
            &[b, s, self.num_k_heads as i32, self.head_k_dim as i32],
        );
        let v = mlxcel_core::reshape(
            &v_out,
            &[b, s, self.num_v_heads as i32, self.head_v_dim as i32],
        );

        // Get recurrent state from cache
        let state = cache.as_ref().and_then(|c| {
            c.state_cache
                .as_ref()
                .map(|s| mlxcel_core::copy(s.as_ref().unwrap()))
        });

        // Apply RMS norm with scaling. Reference mlx-lm keeps this on
        // mx.fast.rms_norm rather than expanding it into primitive ops.
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q = scaled_fast_rms_norm_no_weight(&q, inv_scale * inv_scale, 1e-6);
        let k = scaled_fast_rms_norm_no_weight(&k, inv_scale, 1e-6);

        // Run gated delta update
        let (out, new_state) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b_proj,
            &self.a_log,
            &self.dt_bias,
            state.as_deref(),
            mask,
        );

        // Update cache state
        if let Some(c) = cache {
            c.state_cache = Some(new_state);
            c.advance(s);
        }

        // Apply norm with gating
        let out = self.norm.forward(&out, Some(&z));
        let out = mlxcel_core::reshape(&out, &[b, s, -1]);

        self.out_proj.forward(&out)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let group_size = config.group_size();
        let bits = config.bits();

        let conv1d_weight = weights
            .get(&format!("{}.conv1d.weight", prefix))
            .map(|w| {
                let shape = mlxcel_core::array_shape(w);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    mlxcel_core::swap_axes(w, -1, -2)
                } else {
                    mlxcel_core::copy(w)
                }
            })
            .ok_or_else(|| format!("Missing conv1d weight: {}", prefix))?;

        let in_proj_qkvz = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_qkvz", prefix),
            group_size,
            bits,
        )?;

        let in_proj_ba = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_ba", prefix),
            group_size,
            bits,
        )?;

        let dt_bias = weights
            .get(&format!("{}.dt_bias", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing dt_bias: {}", prefix))?;

        let a_log = weights
            .get(&format!("{}.A_log", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing A_log: {}", prefix))?;

        let norm_weight = weights
            .get(&format!("{}.norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing norm weight: {}", prefix))?;

        let out_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.out_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_dim,
            conv1d_weight,
            in_proj_qkvz,
            in_proj_ba,
            dt_bias,
            a_log,
            norm: RMSNormGated::new(norm_weight, config.rms_norm_eps),
            out_proj,
        })
    }
}

// Attention with Gated Output.
pub(crate) struct Qwen3NextAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_dims: i32,
    rope_base: f32,
    /// Optional interleaved MRoPE for VLM (Qwen3.5 VLM)
    pub(crate) mrope: Option<super::qwen3_vl::InterleavedMRoPE>,
}

impl Qwen3NextAttention {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_position_ids(x, cache, mask, None)
    }

    /// Forward with optional MRoPE position_ids [3, batch, seq_len]
    /// Used by: Qwen3Next (text-only, position_ids=None), Qwen3.5 VLM (with MRoPE position_ids)
    pub(crate) fn forward_with_position_ids(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_position_ids_verify(x, cache, mask, position_ids, false)
    }

    /// Variant of [`Self::forward_with_position_ids`] that opts into the
    /// MTP target-verify attention path when `target_verify` is set.
    ///
    /// `target_verify == false` is byte-identical to
    /// [`Self::forward_with_position_ids`] (the standard decode / prefill
    /// path). `target_verify == true` with a multi-token verify block routes
    /// scaled-dot-product attention through the per-query-position causal
    /// loop in [`Self::forward_hidden_with_position_ids_verify`], mirroring
    /// upstream `Qwen3_5Attention.__call__`'s `target_verify and L > 1`
    /// branch (`mlx-vlm/mlx_vlm/models/qwen3_5/language.py`). This eliminates
    /// the batched-SDPA-vs-decode logit drift that flips speculative
    /// accept/reject decisions away from the drafter-less greedy pass.
    ///
    /// Used by: `Qwen35DecoderLayer::forward_with_capture` (the Qwen 3.5 MTP
    /// verify pass).
    pub(crate) fn forward_with_position_ids_verify(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
        target_verify: bool,
    ) -> UniquePtr<MlxArray> {
        let output = self.forward_hidden_with_position_ids_verify(
            x,
            cache,
            mask,
            position_ids,
            target_verify,
        );
        self.o_proj.forward(&output)
    }

    pub(crate) fn forward_hidden_with_position_ids(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_hidden_with_position_ids_verify(x, cache, mask, position_ids, false)
    }

    /// Hidden-state attention forward with an opt-in MTP target-verify path.
    ///
    /// When `target_verify` is `true` and the query block spans more than one
    /// position (`l > 1`), scaled-dot-product attention is computed **per
    /// query position**: query `i` attends only to keys/values
    /// `[.. prefix_len + i + 1]`, exactly as it would during single-token
    /// decode. This is a faithful port of upstream's
    /// `target_verify and L > 1` attention branch and guarantees the verify
    /// logits match the per-token decode logits bit-for-bit modulo the
    /// kernel's own reduction order, so the speculative walk's greedy argmax
    /// agrees with the drafter-less greedy decode.
    ///
    /// When `target_verify` is `false`, the body is byte-identical to the
    /// historical [`Self::forward_hidden_with_position_ids`] decode path.
    pub(crate) fn forward_hidden_with_position_ids_verify(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: Option<&MlxArray>,
        target_verify: bool,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Q projection with gating: [B, L, D] -> [B, L, 2 * num_heads * head_dim]
        let q_proj_output = self.q_proj.forward(x);
        let q_proj_reshaped = mlxcel_core::reshape(&q_proj_output, &[b, l, self.num_heads, -1]);

        // Split into queries and gate
        let queries = mlxcel_core::slice(
            &q_proj_reshaped,
            &[0, 0, 0, 0],
            &[b, l, self.num_heads, self.head_dim],
        );
        // Note: MLX slice stop=-1 means dim_size-1 (excludes last), not "to end"
        let q_last_dim = mlxcel_core::array_shape(&q_proj_reshaped)[3];
        let gate = mlxcel_core::slice(
            &q_proj_reshaped,
            &[0, 0, 0, self.head_dim],
            &[b, l, self.num_heads, q_last_dim],
        );
        let gate = mlxcel_core::reshape(&gate, &[b, l, -1]);

        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);

        // Reshape and apply Q/K norms
        let queries = mlxcel_core::reshape(&queries, &[b, l, self.num_heads, self.head_dim]);
        let keys = mlxcel_core::reshape(&keys, &[b, l, self.num_kv_heads, self.head_dim]);
        let values = mlxcel_core::reshape(&values, &[b, l, self.num_kv_heads, self.head_dim]);

        let queries = self.q_norm.forward(&queries);
        let keys = self.k_norm.forward(&keys);

        // Transpose to [B, H, L, D]
        let mut queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let mut keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE (standard or MRoPE with position_ids)
        if let (Some(mrope), Some(pos_ids)) = (&self.mrope, position_ids) {
            // Interleaved MRoPE for VLM
            let (cos, sin) = mrope.forward(pos_ids);
            let embed_dtype = mlxcel_core::array_dtype(&queries);
            let cos = mlxcel_core::astype(&cos, embed_dtype);
            let sin = mlxcel_core::astype(&sin, embed_dtype);
            // Partial rotary: only apply to first rope_dims dimensions
            let rotary_dim = self.rope_dims;
            let q_rot =
                mlxcel_core::slice(&queries, &[0, 0, 0, 0], &[b, self.num_heads, l, rotary_dim]);
            let q_pass = mlxcel_core::slice(
                &queries,
                &[0, 0, 0, rotary_dim],
                &[b, self.num_heads, l, self.head_dim],
            );
            let k_rot =
                mlxcel_core::slice(&keys, &[0, 0, 0, 0], &[b, self.num_kv_heads, l, rotary_dim]);
            let k_pass = mlxcel_core::slice(
                &keys,
                &[0, 0, 0, rotary_dim],
                &[b, self.num_kv_heads, l, self.head_dim],
            );
            let (q_embed, k_embed) =
                super::qwen3_vl::apply_multimodal_rotary_pos_emb(&q_rot, &k_rot, &cos, &sin);
            queries = mlxcel_core::concatenate(&q_embed, &q_pass, -1);
            keys = mlxcel_core::concatenate(&k_embed, &k_pass, -1);
        } else {
            // Standard RoPE with offset
            queries = mlxcel_core::fast_rope(
                &queries,
                self.rope_dims,
                false,
                self.rope_base,
                1.0,
                offset,
            );
            keys =
                mlxcel_core::fast_rope(&keys, self.rope_dims, false, self.rope_base, 1.0, offset);
        }

        // Update KV cache
        let (cache_k, cache_v) = cache.update_and_fetch(keys, values);

        // Scaled dot-product attention.
        //
        // MTP target-verify path: when verifying a multi-token
        // draft block, attention is computed per query position so each
        // position sees exactly the prefix it would see during single-token
        // decode. This mirrors upstream `Qwen3_5Attention.__call__`'s
        // `target_verify and L > 1` branch and keeps the verify logits from
        // drifting away from the drafter-less greedy decode.
        let attn_out = if target_verify && l > 1 {
            self.attend_per_position(&queries, &cache_k, &cache_v)
        } else if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&queries, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &queries, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        // Transpose back and reshape
        let output = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);

        // Apply sigmoid gating to output
        let gate_sigmoid = mlxcel_core::sigmoid(&gate);
        mlxcel_core::multiply(&output, &gate_sigmoid)
    }

    /// Per-query-position causal attention for the MTP target-verify pass.
    ///
    /// `queries`/`keys`/`values` are `[B, H, L_q, D]` / `[B, H_kv, L_kv, D]`
    /// after the KV cache update, so `L_kv = prefix_len + L_q`. For each
    /// query position `i` we attend `queries[:, :, i:i+1, :]` to the
    /// causal prefix `keys[:, :, .. prefix_len + i + 1, :]` (and the matching
    /// `values` slice) with no mask — the slice itself enforces causality, so
    /// each position computes exactly the attention it would compute during
    /// single-token decode. Results are concatenated back along the query
    /// axis into `[B, H, L_q, D]`.
    ///
    /// Faithful port of upstream
    /// `mlx-vlm/mlx_vlm/models/qwen3_5/language.py::Qwen3_5Attention.__call__`
    /// (`target_verify and L > 1` branch). This is the load-bearing fix for
    /// Qwen 3.5 MTP verification drift / sampling parity.
    fn attend_per_position(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let q_shape = mlxcel_core::array_shape(queries);
        let k_shape = mlxcel_core::array_shape(keys);
        let v_shape = mlxcel_core::array_shape(values);
        let b = q_shape[0];
        let n_q_heads = q_shape[1];
        let l_q = q_shape[2];
        let head_dim = q_shape[3];
        let n_kv_heads = k_shape[1];
        let l_kv = k_shape[2];
        let prefix_len = l_kv - l_q;

        let mut out: Option<UniquePtr<MlxArray>> = None;
        for i in 0..l_q {
            // queries[:, :, i:i+1, :]
            let q_i = mlxcel_core::slice(queries, &[0, 0, i, 0], &[b, n_q_heads, i + 1, head_dim]);
            // keys/values[:, :, : prefix_len + i + 1, :]
            let kv_len = prefix_len + i + 1;
            let k_i = mlxcel_core::slice(keys, &[0, 0, 0, 0], &[b, n_kv_heads, kv_len, head_dim]);
            let v_i =
                mlxcel_core::slice(values, &[0, 0, 0, 0], &[b, n_kv_heads, kv_len, v_shape[3]]);
            // Single-query attention, no mask: the K/V slice is already the
            // exact causal prefix, matching the single-token decode call.
            let attn_i = mlxcel_core::layers::attention(&q_i, &k_i, &v_i, self.scale, None, 0.0, 0);
            out = Some(match out {
                None => attn_i,
                Some(prev) => mlxcel_core::concatenate(&prev, &attn_i, 2),
            });
        }
        out.expect("attend_per_position requires l_q >= 1")
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let q_norm_weight = weights
            .get(&format!("{}.q_norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing q_norm weight: {}", prefix))?;
        let k_norm_weight = weights
            .get(&format!("{}.k_norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing k_norm weight: {}", prefix))?;

        let head_dim = config.head_dim as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_weight, config.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, config.rms_norm_eps),
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: config.rope_dims(),
            rope_base: config.rope_theta,
            mrope: None,
        })
    }
}

// MLP and MoE Layers.
/// Dense MLP layer
pub(crate) struct MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MLP {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gated = self.forward_hidden(x);
        self.down_proj.forward(&gated)
    }

    pub(crate) fn forward_hidden(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        mlxcel_core::multiply(&gate, &up)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }
}

/// SwitchLinear for MoE experts (falls back to gather_mm for non-quantized)
pub(crate) enum SwitchLinear {
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
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        sorted: bool,
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
                    sorted,
                    "affine",
                )
            },
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(x, &wt, std::ptr::null(), indices as *const _, sorted)
                }
            }
        }
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing weight: {}", prefix))?;
        let scales_key = format!("{}.scales", prefix);
        if weights.contains_key(&scales_key) {
            let scales = mlxcel_core::copy(weights.get(&scales_key).unwrap());
            let biases = weights
                .get(&format!("{}.biases", prefix))
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Missing biases: {}", prefix))?;
            Ok(Self::Quantized {
                weight,
                scales,
                biases,
                group_size: config.group_size(),
                bits: config.bits(),
            })
        } else {
            Ok(Self::Regular { weight })
        }
    }
}

/// SwitchGLU: SwiGLU with stacked expert weights
pub(crate) struct SwitchGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGLU {
    pub(crate) fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];
        let total = n_tokens * top_k;
        let do_sort = total >= 64;

        // Expand x: [n_tokens, hidden] -> [n_tokens, 1, 1, hidden]
        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);

        if do_sort {
            // Sort for better memory access
            let (sorted_x, sorted_idx, inv_order) = self.gather_sort(&x_exp, indices);
            let x_gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            let x_up = self.up_proj.forward(&sorted_x, &sorted_idx, true);
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);
            let output = self.down_proj.forward(&activated, &sorted_idx, true);
            self.scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            let x_gate = self.gate_proj.forward(&x_exp, indices, false);
            let x_up = self.up_proj.forward(&x_exp, indices, false);
            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);
            let output = self.down_proj.forward(&activated, indices, false);
            mlxcel_core::squeeze_axis(&output, -2)
        }
    }

    pub(crate) fn gather_sort(
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

        let flat_indices = mlxcel_core::reshape(indices, &[-1]);
        let order = mlxcel_core::argsort(&flat_indices, -1);
        let inv_order = mlxcel_core::argsort(&order, -1);

        let x_shape = mlxcel_core::array_shape(x);
        let x_flat = mlxcel_core::reshape(x, &[x_shape[0], 1, x_shape[3]]);

        let top_k_arr = mlxcel_core::from_slice_i32(&[top_k], &[1]);
        let token_indices = mlxcel_core::divide(&order, &top_k_arr);
        let token_indices = mlxcel_core::astype(&token_indices, dtype::INT32);

        let sorted_x = mlxcel_core::take(&x_flat, &token_indices, 0);
        let sorted_indices = mlxcel_core::take(&flat_indices, &order, 0);

        (sorted_x, sorted_indices, inv_order)
    }

    pub(crate) fn scatter_unsort(
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

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(weights, config, &format!("{}.w1", prefix))?,
            up_proj: SwitchLinear::from_weights(weights, config, &format!("{}.w3", prefix))?,
            down_proj: SwitchLinear::from_weights(weights, config, &format!("{}.w2", prefix))?,
        })
    }
}

/// Sparse MoE Block with shared expert
pub(crate) struct SparseMoeBlock {
    router: UnifiedLinear,
    experts: SwitchGLU,
    shared_expert: MLP,
    shared_expert_gate: UnifiedLinear,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        // Flatten to [n_tokens, hidden]
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Router logits
        let logits = self.router.forward(&x_flat);
        let gates = mlxcel_core::softmax(&logits, -1);

        // Top-k selection
        let k = self.num_experts_per_tok as i32;
        let logits_shape = mlxcel_core::array_shape(&logits);
        let n_experts = logits_shape[1];
        let kth = n_experts - k;

        let indices = mlxcel_core::argpartition(&logits, kth, -1);
        let topk_indices = mlxcel_core::slice(&indices, &[0, kth], &[logits_shape[0], n_experts]);

        // Get scores for top-k
        let mut scores = mlxcel_core::take_along_axis(&gates, &topk_indices, -1);

        if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&scores, -1, true);
            scores = mlxcel_core::divide(&scores, &sum);
        }

        // Expert computation
        let expert_out = self.experts.forward(&x_flat, &topk_indices);
        let y = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Shared expert
        let shared_y = self.shared_expert.forward(&x_flat);
        let shared_gate = mlxcel_core::sigmoid(&self.shared_expert_gate.forward(&x_flat));
        let shared_y = mlxcel_core::multiply(&shared_y, &shared_gate);

        let result = mlxcel_core::add(&y, &shared_y);

        // Reshape back
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        Ok(Self {
            router: UnifiedLinear::from_weights(
                weights,
                &format!("{}.gate", prefix),
                group_size,
                bits,
            )?,
            experts: SwitchGLU::from_weights(weights, config, &format!("{}.switch_mlp", prefix))?,
            shared_expert: MLP::from_weights(
                weights,
                config,
                &format!("{}.shared_expert", prefix),
            )?,
            shared_expert_gate: UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_expert_gate", prefix),
                group_size,
                bits,
            )?,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        })
    }
}

/// MLP variant enum
enum MLPVariant {
    Dense(MLP),
    MoE(SparseMoeBlock),
}

impl MLPVariant {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MLPVariant::Dense(mlp) => mlp.forward(x),
            MLPVariant::MoE(moe) => moe.forward(x),
        }
    }
}

/// Attention variant enum
enum AttentionVariant {
    FullAttention(Qwen3NextAttention),
    LinearAttention(GatedDeltaNet),
}

// Decoder Layer.
pub(crate) struct DecoderLayer {
    is_linear: bool,
    attention: AttentionVariant,
    mlp: MLPVariant,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen3NextCache,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);

        let r = match (&self.attention, cache) {
            (AttentionVariant::LinearAttention(attn), Qwen3NextCache::Linear(c)) => {
                attn.forward(&normed, mask, Some(c))
            }
            (AttentionVariant::LinearAttention(attn), _) => attn.forward(&normed, mask, None),
            (AttentionVariant::FullAttention(attn), Qwen3NextCache::Attention(c)) => {
                attn.forward(&normed, c, mask)
            }
            (AttentionVariant::FullAttention(attn), _) => {
                let mut temp_cache = KVCache::new();
                attn.forward(&normed, &mut temp_cache, mask)
            }
        };

        let h = mlxcel_core::add(x, &r);

        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &mlp_out)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen3NextConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);
        let is_linear = config.is_linear_layer(layer_idx);

        let attention = if is_linear {
            AttentionVariant::LinearAttention(GatedDeltaNet::from_weights(
                weights,
                config,
                &format!("{}.linear_attn", prefix),
            )?)
        } else {
            AttentionVariant::FullAttention(Qwen3NextAttention::from_weights(
                weights,
                config,
                &format!("{}.self_attn", prefix),
            )?)
        };

        let mlp = if config.is_moe_layer(layer_idx) {
            MLPVariant::MoE(SparseMoeBlock::from_weights(
                weights,
                config,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            MLPVariant::Dense(MLP::from_weights(
                weights,
                config,
                &format!("{}.mlp", prefix),
            )?)
        };

        let input_norm_weight = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing input_layernorm: {}", prefix))?;

        let post_norm_weight = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing post_attention_layernorm: {}", prefix))?;

        Ok(Self {
            is_linear,
            attention,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_weight, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm_weight, config.rms_norm_eps),
        })
    }
}

// Qwen3Next Model.
pub struct Qwen3NextModel {
    pub embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub tie_word_embeddings: bool,
    pub full_attention_interval: usize,
}

impl Qwen3NextModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Qwen3NextCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask, cache);
        }

        let h = self.norm.forward(&h);

        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<Qwen3NextCache> {
        self.layers
            .iter()
            .map(|l| {
                if l.is_linear {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    Qwen3NextCache::Attention(KVCache::new())
                }
            })
            .collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Qwen3NextConfig), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        println!("[Qwen3Next] Loading config...");
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: Qwen3NextConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        println!(
            "[Qwen3Next] Config loaded: {} layers ({} full attention, {} linear attention, {} MoE)",
            config.num_hidden_layers,
            (0..config.num_hidden_layers)
                .filter(|&i| !config.is_linear_layer(i))
                .count(),
            (0..config.num_hidden_layers)
                .filter(|&i| config.is_linear_layer(i))
                .count(),
            (0..config.num_hidden_layers)
                .filter(|&i| config.is_moe_layer(i))
                .count()
        );

        // Load weights
        println!("[Qwen3Next] Loading weights...");
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Sanitize weights
        let weights = Self::sanitize_weights(weights, &config);

        // Build model
        println!("[Qwen3Next] Building model...");
        let model = Self::from_weights(&weights, &config)?;

        println!("[Qwen3Next] Model loaded successfully");
        Ok((model, config))
    }

    fn sanitize_weights(mut weights: WeightMap, config: &Qwen3NextConfig) -> WeightMap {
        // Remove lm_head if tied
        if config.tie_word_embeddings {
            weights.remove("lm_head.weight");
        }

        // Handle MoE weight stacking
        for l in 0..config.num_hidden_layers {
            if !config.is_moe_layer(l) {
                continue;
            }

            let base = format!("model.layers.{}.mlp.switch_mlp", l);
            for proj in ["w1", "w2", "w3"] {
                let mut expert_weights: Vec<UniquePtr<MlxArray>> = Vec::new();
                let mut expert_scales: Vec<UniquePtr<MlxArray>> = Vec::new();
                let mut expert_biases: Vec<UniquePtr<MlxArray>> = Vec::new();

                let mut e = 0;
                while let Some(w) = weights.remove(&format!(
                    "model.layers.{}.mlp.experts.{}.{}.weight",
                    l, e, proj
                )) {
                    expert_weights.push(w);
                    if let Some(s) = weights.remove(&format!(
                        "model.layers.{}.mlp.experts.{}.{}.scales",
                        l, e, proj
                    )) {
                        expert_scales.push(s);
                    }
                    if let Some(b) = weights.remove(&format!(
                        "model.layers.{}.mlp.experts.{}.{}.biases",
                        l, e, proj
                    )) {
                        expert_biases.push(b);
                    }
                    e += 1;
                }

                if !expert_weights.is_empty() {
                    let stacked = stack_arrays(&expert_weights, 0);
                    weights.insert(format!("{}.{}.weight", base, proj), stacked);

                    if !expert_scales.is_empty() {
                        let stacked = stack_arrays(&expert_scales, 0);
                        weights.insert(format!("{}.{}.scales", base, proj), stacked);
                    }

                    if !expert_biases.is_empty() {
                        let stacked = stack_arrays(&expert_biases, 0);
                        weights.insert(format!("{}.{}.biases", base, proj), stacked);
                    }
                }
            }
        }

        weights
    }

    pub fn from_weights(weights: &WeightMap, config: &Qwen3NextConfig) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DecoderLayer::from_weights(weights, config, i)?;
            layers.push(layer);
        }

        let norm_weight = weights
            .get("model.norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Missing model.norm.weight".to_string())?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(norm_weight, config.rms_norm_eps),
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            full_attention_interval: config.full_attention_interval,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for Qwen3NextModel {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Note: Qwen3Next uses mixed cache types (KVCache + GatedDeltaCache)
        // For LanguageModel trait compatibility, we use internal caching
        let mut caches = self.make_caches();
        let mask = {
            let shape = mlxcel_core::array_shape(input);
            let seq_len = shape[1];
            if seq_len > 1 {
                let fa_idx = self.full_attention_interval - 1;
                let offset = if fa_idx < caches.len() {
                    caches[fa_idx].offset()
                } else {
                    0
                };
                Some(create_causal_mask(seq_len, offset))
            } else {
                None
            }
        };
        self.forward(input, &mut caches, mask.as_deref())
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Return KV caches for compatibility
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_batching(&self) -> bool {
        false // Qwen3Next uses internal mixed cache types, not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![151645] // Qwen3 EOS token
    }
}
