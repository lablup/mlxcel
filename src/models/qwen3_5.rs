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

//! Qwen3.5: Hybrid Transformer + GatedDeltaNet (Linear Attention) + MoE
//!
//! Key differences from Qwen3Next:
//! - GatedDeltaNet uses 4 separate projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a)
//!   instead of 2 combined (in_proj_qkvz, in_proj_ba)
//! - Config uses rope_parameters dict instead of flat rope_theta/partial_rotary_factor
//! - Weight sanitization handles MTP weights and norm weight shifting
//! - MoE variant (qwen3_5_moe) uses text_config indirection and gate_up_proj split
//!
//! Reuses from qwen3_next: Qwen3NextAttention, MLP, SparseMoeBlock, SwitchGLU, SwitchLinear
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py

use crate::distributed::pipeline::LayerFilter;
use crate::distributed::pipeline::StageExecutionOutput;
use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models::gated_delta::{
    GatedDeltaCache, RMSNormGated, gated_delta_update, scaled_fast_rms_norm_no_weight,
};
use crate::models::model_owned::ModelOwnedSequenceState;
use crate::models::qwen_mrope_state::MRopeState;
use crate::models::qwen3_next::{
    MLP, Quantization, Qwen3NextAttention, Qwen3NextCache, Qwen3NextConfig, SparseMoeBlock,
};
use mlxcel_core::cache::{CachePool, SequenceId, SequenceStateLayout};
use mlxcel_core::dtype;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, silu, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,

    // Linear attention parameters
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,

    // MoE parameters (0 = dense)
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,

    // Rope parameters (dict format)
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,

    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_full_attention_interval() -> usize {
    4
}
fn default_linear_num_value_heads() -> usize {
    64
}
fn default_linear_num_key_heads() -> usize {
    16
}
fn default_linear_key_head_dim() -> usize {
    192
}
fn default_linear_value_head_dim() -> usize {
    128
}
fn default_linear_conv_kernel_dim() -> usize {
    4
}
fn default_decoder_sparse_step() -> usize {
    1
}
fn default_true() -> bool {
    true
}

impl Qwen35Config {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(100000.0)
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.25)
    }

    fn head_dim_resolved(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_dims(&self) -> i32 {
        (self.head_dim_resolved() as f32 * self.partial_rotary_factor()) as i32
    }

    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        !(layer_idx + 1).is_multiple_of(self.full_attention_interval)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    /// Convert to Qwen3NextConfig for reusing shared components
    pub fn to_qwen3next_config(&self) -> Qwen3NextConfig {
        Qwen3NextConfig {
            model_type: self.model_type.clone(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim_resolved(),
            linear_num_value_heads: self.linear_num_value_heads,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            decoder_sparse_step: self.decoder_sparse_step,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            mlp_only_layers: self.mlp_only_layers.clone(),
            full_attention_interval: self.full_attention_interval,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            rope_theta: self.rope_theta(),
            partial_rotary_factor: self.partial_rotary_factor(),
            max_position_embeddings: None,
            norm_topk_prob: self.norm_topk_prob,
            tie_word_embeddings: self.tie_word_embeddings,
            attention_bias: self.attention_bias,
            quantization: self.quantization.clone(),
        }
    }
}

// GatedDeltaNet - Qwen3.5 variant with separate projections.
/// GatedDeltaNet for Qwen3.5 with separate in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
#[allow(dead_code)]
pub(crate) struct Qwen35GatedDeltaNet {
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
    in_proj_qkv: UnifiedLinear,
    in_proj_z: UnifiedLinear,
    in_proj_b: UnifiedLinear,
    in_proj_a: UnifiedLinear,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    norm: RMSNormGated,
    out_proj: UnifiedLinear,
}

#[cfg(test)]
pub(crate) struct Qwen35LinearDebugTensors {
    pub qkv: UniquePtr<MlxArray>,
    pub z: UniquePtr<MlxArray>,
    pub b_proj: UniquePtr<MlxArray>,
    pub a: UniquePtr<MlxArray>,
    pub conv_out: UniquePtr<MlxArray>,
    pub q: UniquePtr<MlxArray>,
    pub k: UniquePtr<MlxArray>,
    pub v: UniquePtr<MlxArray>,
    pub beta: UniquePtr<MlxArray>,
    pub g: UniquePtr<MlxArray>,
    pub gated_out: UniquePtr<MlxArray>,
    pub normed_out: UniquePtr<MlxArray>,
    pub projected: UniquePtr<MlxArray>,
}

impl Qwen35GatedDeltaNet {
    pub(crate) fn forward(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<&mut GatedDeltaCache>,
    ) -> UniquePtr<MlxArray> {
        // Standalone Qwen3.5 follows mlx-lm create_ssm_mask(): no mask when
        // all tokens are valid. The tensor-parallel path keeps the forced mask
        // in forward_hidden_tp() for CUDA parity.
        let out = self.forward_hidden_internal(inputs, mask, cache, false);
        self.out_proj.forward(&out)
    }

    pub(crate) fn forward_hidden_tp(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<&mut GatedDeltaCache>,
    ) -> UniquePtr<MlxArray> {
        self.forward_hidden_internal(inputs, mask, cache, true)
    }

    fn forward_hidden_internal(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        mut cache: Option<&mut GatedDeltaCache>,
        force_ops_path: bool,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        let b = shape[0];
        let s = shape[1];

        let forced_mask = if force_ops_path && mask.is_none() {
            Some(mlxcel_core::ones(&[b, s], dtype::BOOL))
        } else {
            None
        };
        let effective_mask = forced_mask.as_deref().or(mask);

        // Separate projections (different from Qwen3Next's combined projections)
        let qkv = self.in_proj_qkv.forward(inputs);
        let z = self.in_proj_z.forward(inputs);
        let z = mlxcel_core::reshape(&z, &[b, s, self.num_v_heads as i32, self.head_v_dim as i32]);
        let b_proj = self.in_proj_b.forward(inputs);
        let a = self.in_proj_a.forward(inputs);

        // Get conv state from cache
        let input_dtype = mlxcel_core::array_dtype(&qkv);
        let conv_state = if let Some(ref c) = cache {
            c.conv_state
                .as_ref()
                .and_then(|s| {
                    let s = s.as_ref().unwrap();
                    let state_shape = mlxcel_core::array_shape(s);
                    // Guard: reinitialize if batch dimension doesn't match (continuous batching)
                    if state_shape[0] != b {
                        None
                    } else {
                        Some(mlxcel_core::copy(s))
                    }
                })
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

        // Guard: discard mask if batch dimension doesn't match (continuous batching).
        // Uses guarded_mask consistently for both the conv masking and gated_delta_update.
        let guarded_mask = effective_mask.filter(|m| {
            let mask_shape = mlxcel_core::array_shape(m);
            mask_shape[0] == b
        });

        // Apply mask if present (mask qkv before conv)
        let qkv = if let Some(m) = guarded_mask {
            let m_exp = mlxcel_core::expand_dims(m, -1);
            let zero = mlxcel_core::full_f32(&[1], 0.0, input_dtype);
            mlxcel_core::where_cond(&m_exp, &qkv, &zero)
        } else {
            qkv
        };

        // Concatenate with conv state
        let conv_input = concatenate(&conv_state, &qkv, 1);

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

        // Split conv output into q, k, v
        // Note: MLX slice with stop=-1 means dim_size-1 (excludes last), not "to end"
        // Use actual conv_out seq length for correct slicing
        let conv_out_shape = mlxcel_core::array_shape(&conv_out);
        let conv_seq = conv_out_shape[1];
        let q_out = mlxcel_core::slice(&conv_out, &[0, 0, 0], &[b, conv_seq, self.key_dim as i32]);
        let k_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, self.key_dim as i32],
            &[b, conv_seq, (2 * self.key_dim) as i32],
        );
        let v_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, (2 * self.key_dim) as i32],
            &[b, conv_seq, self.conv_dim as i32],
        );

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
        // Guard: discard cached state if batch dimension doesn't match (continuous batching)
        let state = cache.as_ref().and_then(|c| {
            c.state_cache.as_ref().and_then(|s| {
                let s = s.as_ref().unwrap();
                let state_shape = mlxcel_core::array_shape(s);
                if state_shape[0] != b {
                    None
                } else {
                    Some(mlxcel_core::copy(s))
                }
            })
        });

        // Apply RMS norm with scaling (same as Qwen3Next). Reference mlx-lm
        // keeps this on mx.fast.rms_norm rather than expanding it into
        // primitive ops.
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q = scaled_fast_rms_norm_no_weight(&q, inv_scale * inv_scale, 1e-6);
        let k = scaled_fast_rms_norm_no_weight(&k, inv_scale, 1e-6);

        // Run gated delta update (use guarded_mask which is None if batch dims mismatch)
        let (out, new_state) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b_proj,
            &self.a_log,
            &self.dt_bias,
            state.as_deref(),
            guarded_mask,
        );

        // Update cache state
        if let Some(c) = cache {
            c.state_cache = Some(new_state);
            c.advance(s);
        }

        // Apply norm with gating
        let out = self.norm.forward(&out, Some(&z));
        mlxcel_core::reshape(&out, &[b, s, -1])
    }

    /// Variant of [`forward_hidden_internal`] that also captures a
    /// per-layer snapshot suitable for `rollback_speculative_cache`.
    ///
    /// Mirrors the `gdn_sink` parameter in upstream
    /// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/language.py (Qwen3_5GatedDeltaNet)
    /// The snapshot is captured *before* `gated_delta_update`
    /// runs, holding the same per-step inputs and the pre-block recurrent
    /// state — exactly what is needed to replay the block over a truncated
    /// `[B, n]` slice during DFlash rollback.
    ///
    /// Apple Silicon precision (`docs/apple-silicon-precision.md`):
    /// captured `q/k/v/a/b/conv_input` retain their bf16/f16 dtype from the
    /// projections; `init_state`, when present, stays float32. The rollback
    /// path must preserve these dtypes to avoid numerical drift from the
    /// cold-pass reference the drafter was trained against.
    ///
    /// Used by: `Qwen35Model::forward_speculative`
    fn forward_hidden_internal_with_capture(
        &self,
        layer_idx: usize,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
        mut cache: Option<&mut GatedDeltaCache>,
        snapshot_sink: &mut Vec<GdnRollbackSnapshot>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        let b = shape[0];
        let s = shape[1];

        // Projections — keep bf16/f16 throughout.
        let qkv = self.in_proj_qkv.forward(inputs);
        let z = self.in_proj_z.forward(inputs);
        let z = mlxcel_core::reshape(&z, &[b, s, self.num_v_heads as i32, self.head_v_dim as i32]);
        let b_proj = self.in_proj_b.forward(inputs);
        let a = self.in_proj_a.forward(inputs);

        let input_dtype = mlxcel_core::array_dtype(&qkv);

        // Reproduce the conv-state handling from `forward_hidden_internal`.
        let conv_state = if let Some(ref c) = cache {
            c.conv_state
                .as_ref()
                .and_then(|s| {
                    let s_ref = s.as_ref().unwrap();
                    let state_shape = mlxcel_core::array_shape(s_ref);
                    if state_shape[0] != b {
                        None
                    } else {
                        Some(mlxcel_core::copy(s_ref))
                    }
                })
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

        let guarded_mask = mask.filter(|m| {
            let mask_shape = mlxcel_core::array_shape(m);
            mask_shape[0] == b
        });

        // Apply mask to qkv (matches non-capture path).
        let qkv = if let Some(m) = guarded_mask {
            let m_exp = mlxcel_core::expand_dims(m, -1);
            let zero = mlxcel_core::full_f32(&[1], 0.0, input_dtype);
            mlxcel_core::where_cond(&m_exp, &qkv, &zero)
        } else {
            qkv
        };

        // Build the full conv input. The snapshot stores this in its entirety
        // so rollback can re-derive `conv_state` from any acceptance window.
        let conv_input = concatenate(&conv_state, &qkv, 1);

        // Update the cache's `conv_state` from the verify-pass tail — same as
        // the non-capture path; if the drafter later rejects this block, the
        // rollback path will overwrite this with the per-row trimmed window.
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

        // Conv1d + SiLU.
        let conv_out = mlxcel_core::conv1d(
            &conv_input,
            &self.conv1d_weight,
            1,
            0,
            1,
            self.conv_dim as i32,
        );
        let conv_out = silu(&conv_out);

        let conv_out_shape = mlxcel_core::array_shape(&conv_out);
        let conv_seq = conv_out_shape[1];
        let q_out = mlxcel_core::slice(&conv_out, &[0, 0, 0], &[b, conv_seq, self.key_dim as i32]);
        let k_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, self.key_dim as i32],
            &[b, conv_seq, (2 * self.key_dim) as i32],
        );
        let v_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, (2 * self.key_dim) as i32],
            &[b, conv_seq, self.conv_dim as i32],
        );

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

        // Pre-block recurrent state (float32 when present), guarded against
        // continuous-batching dim mismatch — identical to the non-capture path.
        let init_state = cache.as_ref().and_then(|c| {
            c.state_cache.as_ref().and_then(|s| {
                let s_ref = s.as_ref().unwrap();
                let state_shape = mlxcel_core::array_shape(s_ref);
                if state_shape[0] != b {
                    None
                } else {
                    Some(mlxcel_core::copy(s_ref))
                }
            })
        });

        // RMSNorm scaling for q and k (preserves the verify-pass dtype).
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q = scaled_fast_rms_norm_no_weight(&q, inv_scale * inv_scale, 1e-6);
        let k = scaled_fast_rms_norm_no_weight(&k, inv_scale, 1e-6);

        // Capture the snapshot BEFORE the gated_delta_update consumes/mutates
        // the recurrent state. The drafter uses these to replay over the
        // accepted prefix.
        let snapshot_init_state = init_state.as_ref().map(|s| mlxcel_core::copy(s));
        snapshot_sink.push(GdnRollbackSnapshot {
            layer_idx,
            q: mlxcel_core::copy(&q),
            k: mlxcel_core::copy(&k),
            v: mlxcel_core::copy(&v),
            a: mlxcel_core::copy(&a),
            b: mlxcel_core::copy(&b_proj),
            init_state: snapshot_init_state,
            conv_input: mlxcel_core::copy(&conv_input),
        });

        // Run gated_delta_update for the full verify block.
        let (out, new_state) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b_proj,
            &self.a_log,
            &self.dt_bias,
            init_state.as_deref(),
            guarded_mask,
        );

        if let Some(c) = cache {
            c.state_cache = Some(new_state);
            c.advance(s);
        }

        let out = self.norm.forward(&out, Some(&z));
        let reshaped = mlxcel_core::reshape(&out, &[b, s, -1]);
        self.out_proj.forward(&reshaped)
    }

    #[cfg(test)]
    pub(crate) fn debug_prefill_no_cache(
        &self,
        inputs: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Qwen35LinearDebugTensors {
        let shape = mlxcel_core::array_shape(inputs);
        let b = shape[0];
        let s = shape[1];

        let qkv = self.in_proj_qkv.forward(inputs);
        let z = self.in_proj_z.forward(inputs);
        let z_reshaped =
            mlxcel_core::reshape(&z, &[b, s, self.num_v_heads as i32, self.head_v_dim as i32]);
        let b_proj = self.in_proj_b.forward(inputs);
        let a = self.in_proj_a.forward(inputs);

        let input_dtype = mlxcel_core::array_dtype(&qkv);
        let conv_state = mlxcel_core::zeros(
            &[b, (self.conv_kernel_size - 1) as i32, self.conv_dim as i32],
            input_dtype,
        );
        let guarded_mask = mask.filter(|m| mlxcel_core::array_shape(m)[0] == b);
        let qkv_masked = if let Some(m) = guarded_mask {
            let m_exp = mlxcel_core::expand_dims(m, -1);
            let zero = mlxcel_core::full_f32(&[1], 0.0, input_dtype);
            mlxcel_core::where_cond(&m_exp, &qkv, &zero)
        } else {
            mlxcel_core::copy(&qkv)
        };
        let conv_input = concatenate(&conv_state, &qkv_masked, 1);
        let conv_out = mlxcel_core::conv1d(
            &conv_input,
            &self.conv1d_weight,
            1,
            0,
            1,
            self.conv_dim as i32,
        );
        let conv_out = silu(&conv_out);

        let conv_seq = mlxcel_core::array_shape(&conv_out)[1];
        let q_out = mlxcel_core::slice(&conv_out, &[0, 0, 0], &[b, conv_seq, self.key_dim as i32]);
        let k_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, self.key_dim as i32],
            &[b, conv_seq, (2 * self.key_dim) as i32],
        );
        let v_out = mlxcel_core::slice(
            &conv_out,
            &[0, 0, (2 * self.key_dim) as i32],
            &[b, conv_seq, self.conv_dim as i32],
        );

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

        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q = scaled_fast_rms_norm_no_weight(&q, inv_scale * inv_scale, 1e-6);
        let k = scaled_fast_rms_norm_no_weight(&k, inv_scale, 1e-6);

        let beta = mlxcel_core::sigmoid(&b_proj);
        let g = crate::models::gated_delta::compute_g(&self.a_log, &a, &self.dt_bias);
        let (gated_out, _) =
            crate::models::gated_delta::gated_delta_ops(&q, &k, &v, &g, &beta, None, guarded_mask);
        let normed_out = self.norm.forward(&gated_out, Some(&z_reshaped));
        let normed_out = mlxcel_core::reshape(&normed_out, &[b, s, -1]);
        let projected = self.out_proj.forward(&normed_out);

        Qwen35LinearDebugTensors {
            qkv,
            z: z_reshaped,
            b_proj,
            a,
            conv_out,
            q,
            k,
            v,
            beta,
            g,
            gated_out,
            normed_out,
            projected,
        }
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
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

        // Qwen3.5 uses separate projections instead of combined
        let in_proj_qkv = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_qkv", prefix),
            group_size,
            bits,
        )?;
        let in_proj_z = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_z", prefix),
            group_size,
            bits,
        )?;
        let in_proj_b = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_b", prefix),
            group_size,
            bits,
        )?;
        let in_proj_a = UnifiedLinear::from_weights(
            weights,
            &format!("{}.in_proj_a", prefix),
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
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            dt_bias,
            a_log,
            norm: RMSNormGated::new(norm_weight, config.rms_norm_eps),
            out_proj,
        })
    }
}

// Decoder Layer.
/// Attention variant for Qwen3.5
pub(crate) enum Qwen35AttentionVariant {
    FullAttention(Qwen3NextAttention),
    Linear(Qwen35GatedDeltaNet),
}

/// MLP variant for Qwen3.5
pub(crate) enum Qwen35MLPVariant {
    Dense(MLP),
    MoE(SparseMoeBlock),
}

pub(crate) struct Qwen35DecoderLayer {
    pub(crate) is_linear: bool,
    pub(crate) attention: Qwen35AttentionVariant,
    pub(crate) mlp: Qwen35MLPVariant,
    pub(crate) input_layernorm: RMSNorm,
    pub(crate) post_attention_layernorm: RMSNorm,
}

impl Qwen35DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen3NextCache,
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);

        let r = match (&self.attention, cache) {
            (Qwen35AttentionVariant::Linear(attn), Qwen3NextCache::Linear(c)) => {
                attn.forward(&normed, mask, Some(c))
            }
            (Qwen35AttentionVariant::Linear(attn), _) => attn.forward(&normed, mask, None),
            (Qwen35AttentionVariant::FullAttention(attn), Qwen3NextCache::Attention(c)) => {
                attn.forward_with_position_ids(&normed, c, mask, position_ids)
            }
            (Qwen35AttentionVariant::FullAttention(attn), _) => {
                let mut temp_cache = KVCache::new();
                attn.forward_with_position_ids(&normed, &mut temp_cache, mask, position_ids)
            }
        };

        let h = mlxcel_core::add(x, &r);

        let mlp_out = match &self.mlp {
            Qwen35MLPVariant::Dense(mlp) => mlp.forward(&self.post_attention_layernorm.forward(&h)),
            Qwen35MLPVariant::MoE(moe) => moe.forward(&self.post_attention_layernorm.forward(&h)),
        };
        mlxcel_core::add(&h, &mlp_out)
    }

    /// Speculative-capture variant of [`Self::forward`]. When this layer is
    /// linear-attention, also pushes a [`GdnRollbackSnapshot`] into
    /// `snapshot_sink`; attention layers behave identically to `forward`.
    ///
    /// used by `Qwen35Model::forward_speculative` to drive the
    /// DFlash drafter without duplicating the prefill / decode forward path.
    fn forward_with_capture(
        &self,
        layer_idx: usize,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Qwen3NextCache,
        position_ids: Option<&MlxArray>,
        snapshot_sink: &mut Vec<GdnRollbackSnapshot>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);

        let r = match (&self.attention, cache) {
            (Qwen35AttentionVariant::Linear(attn), Qwen3NextCache::Linear(c)) => attn
                .forward_hidden_internal_with_capture(
                    layer_idx,
                    &normed,
                    mask,
                    Some(c),
                    snapshot_sink,
                ),
            (Qwen35AttentionVariant::Linear(attn), _) => attn.forward_hidden_internal_with_capture(
                layer_idx,
                &normed,
                mask,
                None,
                snapshot_sink,
            ),
            // MTP target-verify pass: full-attention layers
            // compute per-query-position causal attention so the verify
            // logits stay bit-aligned with single-token decode. Mirrors
            // upstream `Qwen3_5DecoderLayer.__call__`'s `target_verify=True`
            // propagation into `Qwen3_5Attention`.
            (Qwen35AttentionVariant::FullAttention(attn), Qwen3NextCache::Attention(c)) => {
                attn.forward_with_position_ids_verify(&normed, c, mask, position_ids, true)
            }
            (Qwen35AttentionVariant::FullAttention(attn), _) => {
                let mut temp_cache = KVCache::new();
                attn.forward_with_position_ids_verify(
                    &normed,
                    &mut temp_cache,
                    mask,
                    position_ids,
                    true,
                )
            }
        };

        let h = mlxcel_core::add(x, &r);

        let mlp_out = match &self.mlp {
            Qwen35MLPVariant::Dense(mlp) => mlp.forward(&self.post_attention_layernorm.forward(&h)),
            Qwen35MLPVariant::MoE(moe) => moe.forward(&self.post_attention_layernorm.forward(&h)),
        };
        mlxcel_core::add(&h, &mlp_out)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
        qn_config: &Qwen3NextConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);
        let is_linear = config.is_linear_layer(layer_idx);

        let attention = if is_linear {
            Qwen35AttentionVariant::Linear(Qwen35GatedDeltaNet::from_weights(
                weights,
                config,
                &format!("{}.linear_attn", prefix),
            )?)
        } else {
            Qwen35AttentionVariant::FullAttention(Qwen3NextAttention::from_weights(
                weights,
                qn_config,
                &format!("{}.self_attn", prefix),
            )?)
        };

        let mlp = if config.is_moe_layer(layer_idx) {
            Qwen35MLPVariant::MoE(SparseMoeBlock::from_weights(
                weights,
                qn_config,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            Qwen35MLPVariant::Dense(MLP::from_weights(
                weights,
                qn_config,
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

// Speculative Decoding Hooks.
/// Per-GDN-layer snapshot captured during a verify-pass forward.
///
/// Stores everything needed to replay [`gated_delta_update`] against the same
/// inputs but truncated to the accepted prefix, so the recurrent linear-attention
/// state can be rolled back to the position of the last accepted token.
///
/// Field shapes (with batch `B`, verify-pass block length `S`):
/// - `q`, `k`: `[B, S, num_k_heads, head_k_dim]` — already RMSNorm-scaled.
/// - `v`: `[B, S, num_v_heads, head_v_dim]`.
/// - `a`, `b`: `[B, S, num_v_heads]` — pre-sigmoid `b`, raw `a`.
/// - `init_state`: `[B, num_v_heads, head_v_dim, head_k_dim]` (float32) or `None`
///   when the layer entered the block with no recurrent state.
/// - `conv_input`: `[B, S + conv_kernel_size - 1, conv_dim]` — concatenated
///   prev-conv-state + qkv. Used to recover the post-rollback `conv_state` window.
/// - `layer_idx`: which decoder layer this snapshot belongs to (for replay).
///
/// Dtype policy (Apple Silicon precision rules — `docs/apple-silicon-precision.md`):
/// the captured tensors retain the dtype produced by the verify-pass kernels
/// (typically bf16 / f16 for activations; float32 for `init_state`). The rollback
/// path must NOT promote them to float32, otherwise the rewound state diverges
/// from the cold-pass result that DFlash's drafter expects.
///
/// Used by: `Qwen35Model::forward_speculative`, `Qwen35Model::rollback_speculative_cache`
pub struct GdnRollbackSnapshot {
    pub layer_idx: usize,
    pub q: UniquePtr<MlxArray>,
    pub k: UniquePtr<MlxArray>,
    pub v: UniquePtr<MlxArray>,
    pub a: UniquePtr<MlxArray>,
    pub b: UniquePtr<MlxArray>,
    pub init_state: Option<UniquePtr<MlxArray>>,
    pub conv_input: UniquePtr<MlxArray>,
}

/// Verify-pass output for the speculative path.
///
/// Returned by [`Qwen35Model::forward_speculative`]. The `hidden_states` vector
/// is ordered to match the `capture_layer_ids` argument so the DFlash drafter
/// can directly call `concat(hidden_states, axis=-1)` to obtain its
/// `5 * hidden_size`-wide projection input. The `gdn_states` vector is ordered
/// by layer index over linear-attention layers only (skipping full-attention
/// layers), matching the per-position correspondence used by upstream
/// `rollback_speculative_cache` in https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/language.py.
///
/// Used by: DFlash round loop (sub-12), `Qwen35Model::rollback_speculative_cache`
pub struct VerifyOutput {
    pub logits: UniquePtr<MlxArray>,
    pub hidden_states: Vec<UniquePtr<MlxArray>>,
    pub gdn_states: Vec<GdnRollbackSnapshot>,
}

// Qwen3.5 Model.
pub struct Qwen35Model {
    pub(crate) embed_tokens: UnifiedEmbedding,
    pub(crate) layers: Vec<Qwen35DecoderLayer>,
    pub(crate) norm: RMSNorm,
    pub(crate) lm_head: Option<UnifiedLinear>,
    pub(crate) config: Qwen35Config,
    /// Internal and per-sequence mixed cache state.
    sequence_state: ModelOwnedSequenceState<Qwen3NextCache>,
    /// Per-sequence MRoPE state (mlx-vlm PR #1095). Each row
    /// in a server batch needs its own delta — the legacy fallback slot
    /// preserves CLI/single-row behavior when no `SequenceId` is plumbed.
    mrope_state: MRopeState,
}

impl Qwen35Model {
    fn forward_internal(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen3NextCache],
        position_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Create masks
        let fa_idx = self.config.full_attention_interval - 1;
        let fa_mask = if seq_len > 1 {
            let offset = if fa_idx < caches.len() {
                caches[fa_idx].offset()
            } else {
                0
            };
            Some(create_causal_mask(seq_len, offset))
        } else {
            None
        };

        // SSM mask: for linear attention layers
        // None means all tokens are valid, which covers:
        // - Generation (L=1): single token always valid
        // - Full prefill (no prior cache): all tokens valid
        // The only case needing a non-None SSM mask is resuming prefill after
        // partial generation, which is rare and can be added later.

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let mask = if layer.is_linear {
                None
            } else {
                fa_mask.as_deref()
            };
            h = layer.forward(&h, mask, cache, position_ids);
        }

        let h = self.norm.forward(&h);

        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    fn make_internal_caches(&self) -> Vec<Qwen3NextCache> {
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

    /// Construct a fresh heterogeneous cache vec for the DFlash
    /// speculative round loop.
    ///
    /// Distinct name from the trait
    /// [`LanguageModel::make_caches`] (which returns `Vec<KVCache>` —
    /// empty for Qwen 3.5 because the model owns its caches
    /// internally) so callers outside this module can pick the right
    /// variant unambiguously. The speculative round-loop driver needs
    /// `Vec<Qwen3NextCache>` (the heterogeneous attention +
    /// linear-attention cache shape that `forward_speculative` /
    /// `rollback_speculative_cache` expect).
    ///
    /// Used by:
    /// [`crate::server::batch::speculative_burst::run_dflash_on_qwen35`]
    /// for both `Qwen35Model` and `Qwen35VLModel` text-only DFlash bursts
    pub(crate) fn make_speculative_caches(&self) -> Vec<Qwen3NextCache> {
        self.make_internal_caches()
    }

    /// Public test seam: construct the heterogeneous attention +
    /// linear-attention cache vector the verify pass expects.
    ///
    /// Functionally identical to [`Self::make_speculative_caches`], exposed
    /// so out-of-crate integration tests (notably the
    /// verify-vs-decode logit-parity test in `tests/`) can drive
    /// [`Self::forward_speculative`] directly without going through the
    /// server burst dispatch. Not intended for production callers — the
    /// server path uses `make_speculative_caches` internally.
    #[doc(hidden)]
    pub fn make_speculative_caches_for_test(&self) -> Vec<Qwen3NextCache> {
        self.make_internal_caches()
    }

    fn visible_len(cache: &Qwen3NextCache) -> usize {
        match cache {
            Qwen3NextCache::Attention(kv) => kv.seq_len().max(0) as usize,
            Qwen3NextCache::Linear(gd) => gd.offset.max(0) as usize,
        }
    }

    fn split_batched_cache(cache: &Qwen3NextCache, batch_idx: usize) -> Qwen3NextCache {
        match cache {
            Qwen3NextCache::Attention(kv) => {
                let mut split = KVCache::new();
                split.offset = kv.offset;
                split.keys = kv.keys.as_ref().map(|keys| {
                    mlxcel_core::slice(
                        keys,
                        &[batch_idx as i32, 0, 0, 0],
                        &[
                            batch_idx as i32 + 1,
                            mlxcel_core::array_shape(keys)[1],
                            kv.offset,
                            mlxcel_core::array_shape(keys)[3],
                        ],
                    )
                });
                split.values = kv.values.as_ref().map(|values| {
                    mlxcel_core::slice(
                        values,
                        &[batch_idx as i32, 0, 0, 0],
                        &[
                            batch_idx as i32 + 1,
                            mlxcel_core::array_shape(values)[1],
                            kv.offset,
                            mlxcel_core::array_shape(values)[3],
                        ],
                    )
                });
                Qwen3NextCache::Attention(split)
            }
            Qwen3NextCache::Linear(gd) => {
                let mut split = GatedDeltaCache::new();
                split.offset = gd.offset;
                split.conv_state = gd.conv_state.as_ref().map(|state| {
                    let shape = mlxcel_core::array_shape(state);
                    mlxcel_core::slice(
                        state,
                        &[batch_idx as i32, 0, 0],
                        &[batch_idx as i32 + 1, shape[1], shape[2]],
                    )
                });
                split.state_cache = gd.state_cache.as_ref().map(|state| {
                    let shape = mlxcel_core::array_shape(state);
                    mlxcel_core::slice(
                        state,
                        &[batch_idx as i32, 0, 0, 0],
                        &[batch_idx as i32 + 1, shape[1], shape[2], shape[3]],
                    )
                });
                Qwen3NextCache::Linear(split)
            }
        }
    }

    fn forward_batched_prefill(
        &self,
        input_ids: &MlxArray,
        seq_ids: &[SequenceId],
    ) -> UniquePtr<MlxArray> {
        let mut batched_caches = self.make_internal_caches();
        let logits = self.forward_internal(input_ids, None, &mut batched_caches, None);
        for (batch_idx, seq_id) in seq_ids.iter().copied().enumerate() {
            let split_caches = batched_caches
                .iter()
                .map(|cache| Self::split_batched_cache(cache, batch_idx))
                .collect();
            self.sequence_state
                .replace_sequence_state(seq_id, split_caches);
        }
        logits
    }

    fn forward_with_sequence_caches(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |sequence_caches| {
                self.forward_with_mrope_state(input_ids, input_embeddings, sequence_caches, seq_id)
            },
        )
    }

    fn forward_with_mrope_state(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Qwen3NextCache],
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let cache_offset = caches
            .iter()
            .find_map(|c| match c {
                super::qwen3_next::Qwen3NextCache::Attention(kv) => Some(kv.offset),
                _ => None,
            })
            .unwrap_or(0);

        let ids_shape = mlxcel_core::array_shape(input_ids);
        let batch = ids_shape[0];
        let seq_len = ids_shape[1];

        // Compute position_ids with sufficiency check for chunked prefill.
        // This matches upstream mlx-vlm PR #1048 (commit 1bf7742): the cached
        // _position_ids entry is reusable when shape[-1] >= cache_offset + seq_length,
        // not only when cache_offset == 0.  During chunked prefill (cache_offset > 0)
        // the stored array is sliced to the needed window rather than recomputed.
        //
        // the MRoPE entry is resolved per `SequenceId` so a row
        // that just finished a VL prefill cannot poison a subsequent
        // text-only sequence's decode delta.
        //
        // (upstream mlx-vlm PR #1040, commit 58e2435): also validate
        // pos_shape[1] == batch before reusing, matching the upstream Python check:
        //   self._position_ids.shape[1] == batch_size
        //   and self._position_ids.shape[-1] >= cache_offset + seq_length
        // Without this, a sequential request with a different batch_size would
        // silently reuse stale position IDs and crash on broadcast_shapes.
        let position_ids = self.mrope_state.with_entry(seq_id, |entry| {
            if let Some(ref stored_pos) = entry.position_ids {
                let pos_shape = mlxcel_core::array_shape(stored_pos);
                // Sufficient when the stored tensor covers [cache_offset, cache_offset+seq_len)
                // and has the same batch dimension as the current request.
                if pos_shape.len() == 3
                    && pos_shape[1] == batch
                    && pos_shape[2] >= cache_offset + seq_len
                {
                    return Some(mlxcel_core::slice(
                        stored_pos,
                        &[0, 0, cache_offset],
                        &[pos_shape[0], pos_shape[1], cache_offset + seq_len],
                    ));
                }
                // Stored ids no longer cover this window; fall back to delta-based compute.
                Self::compute_decode_position_ids_with_delta(
                    entry.rope_deltas,
                    batch,
                    seq_len,
                    cache_offset,
                )
            } else {
                // No stored position_ids; use delta-based compute when rope_deltas is set.
                Self::compute_decode_position_ids_with_delta(
                    entry.rope_deltas,
                    batch,
                    seq_len,
                    cache_offset,
                )
            }
        });

        self.forward_internal(input_ids, input_embeddings, caches, position_ids.as_deref())
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Qwen35Config), String> {
        let model_dir = model_dir.as_ref();

        println!("[Qwen3.5] Loading config...");
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let v: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Handle text_config indirection (VLM wrapper format)
        let mut text_config_val = if let Some(tc) = v.get("text_config") {
            tc.clone()
        } else {
            v.clone()
        };

        // Merge quantization from top level if text_config doesn't have it
        if text_config_val.get("quantization").is_none() && v.get("quantization").is_some() {
            text_config_val
                .as_object_mut()
                .unwrap()
                .insert("quantization".to_string(), v["quantization"].clone());
        }

        let config: Qwen35Config = serde_json::from_value(text_config_val)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        println!(
            "[Qwen3.5] Config loaded: {} layers ({} full attention, {} linear attention)",
            config.num_hidden_layers,
            (0..config.num_hidden_layers)
                .filter(|&i| !config.is_linear_layer(i))
                .count(),
            (0..config.num_hidden_layers)
                .filter(|&i| config.is_linear_layer(i))
                .count(),
        );

        println!("[Qwen3.5] Loading weights...");
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Strip language_model. prefix and sanitize
        let weights = sanitize_moe_weights(weights, &config);

        println!("[Qwen3.5] Building model...");
        let model = Self::from_weights(&weights, &config)?;

        println!("[Qwen3.5] Model loaded successfully");
        Ok((model, config))
    }

    pub fn from_weights(weights: &WeightMap, config: &Qwen35Config) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();
        let qn_config = config.to_qwen3next_config();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Qwen35DecoderLayer::from_weights(weights, config, &qn_config, i)?;
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

        let config_clone = config.clone();
        let internal_caches: Vec<Qwen3NextCache> = (0..config.num_hidden_layers)
            .map(|i| {
                if config.is_linear_layer(i) {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    Qwen3NextCache::Attention(KVCache::new())
                }
            })
            .collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(norm_weight, config.rms_norm_eps),
            lm_head,
            config: config_clone,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            mrope_state: MRopeState::new(),
        })
    }

    /// Set MRoPE state for the legacy/non-server caller. Used by the CLI
    /// generate path and by the vision wrapper when a `SequenceId` is not
    /// (yet) available.
    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        self.mrope_state.set_fallback(position_ids, rope_deltas);
    }

    /// Set MRoPE state for a specific server-side sequence so the cached
    /// per-sequence delta no longer leaks across requests.
    pub fn set_mrope_state_for_sequence(
        &self,
        seq_id: SequenceId,
        position_ids: UniquePtr<MlxArray>,
        rope_deltas: i32,
    ) {
        self.mrope_state
            .set_for_sequence(seq_id, position_ids, rope_deltas);
    }

    /// Drop a server sequence's MRoPE entry so the per-sequence map
    /// does not grow without bound across requests.
    pub fn release_mrope_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.release_sequence(seq_id);
    }

    /// Move whatever the fallback slot holds into the per-sequence map
    /// under `seq_id`. Called by the scheduler right after the vision
    /// wrapper's `get_input_embeddings` has populated the fallback slot,
    /// so subsequent decode steps for this sequence resolve the MRoPE
    /// state by id instead of by leaky scalar.
    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.bind_fallback_to_sequence(seq_id);
    }

    /// Remove and return the per-sequence MRoPE entry under `seq_id`
    /// without dropping the contained position-id tensor. Used by the
    /// server preemption path so the entry can survive an evict-and-
    /// reallocate cycle (follow-up).
    pub(crate) fn take_mrope_entry(
        &self,
        seq_id: SequenceId,
    ) -> Option<crate::models::qwen_mrope_state::MRopeEntry> {
        self.mrope_state.take_for_sequence(seq_id)
    }

    /// Re-install a previously taken MRoPE entry under a (possibly new)
    /// `seq_id`. Used to rebind state across preemption.
    pub(crate) fn install_mrope_entry(
        &self,
        seq_id: SequenceId,
        entry: crate::models::qwen_mrope_state::MRopeEntry,
    ) {
        self.mrope_state.bind_for_sequence(seq_id, entry);
    }

    /// Compute position_ids for decode steps using a per-row `rope_delta`.
    ///
    /// Returns `Some([3, batch, seq_len])` when `delta` is provided (VLM decode
    /// path), or `None` for text-only models where fast_rope handles positioning
    /// internally. makes this static so the caller passes the
    /// per-`SequenceId` delta resolved through `MRopeState::with_entry`.
    // Used by: forward_with_mrope_state
    fn compute_decode_position_ids_with_delta(
        delta: Option<i32>,
        batch: i32,
        seq_len: i32,
        cache_offset: i32,
    ) -> Option<UniquePtr<MlxArray>> {
        let delta = delta?;
        let offset = cache_offset + delta;
        let pos = mlxcel_core::arange_i32(offset, offset + seq_len, 1);
        let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
        let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
        let pos = mlxcel_core::expand_dims(&pos, 0);
        Some(mlxcel_core::broadcast_to(&pos, &[3, batch, seq_len]))
    }

    /// Get token embeddings (used by VLM wrapper)
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward pass with VLM support
    ///
    /// Position IDs handling (for MRoPE VLM):
    /// - Prefill / chunked-prefill (stored position_ids cover this chunk): slice cached ids
    /// - Decode (cache_offset + seq_len beyond stored range, has rope_deltas): compute with offset
    /// - Text-only (no rope_deltas): position_ids = None, uses fast_rope
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_sequence_caches(input_ids, input_embeddings, None)
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Set MRoPE on all full-attention layers
    pub fn set_mrope(&mut self, mrope_section: Vec<i32>, rope_theta: f32, rope_dims: usize) {
        for layer in &mut self.layers {
            if let Qwen35AttentionVariant::FullAttention(ref mut attn) = layer.attention {
                attn.mrope = Some(super::qwen3_vl::InterleavedMRoPE::new(
                    rope_dims, // dim = rope_dims (MRoPE sections sum to dim/2)
                    rope_theta,
                    mrope_section.clone(),
                ));
            }
        }
    }

    /// Forward pass that captures DFlash-target hooks: per-layer hidden
    /// states at `capture_layer_ids` and per-GDN-layer rollback snapshots.
    ///
    /// This is the verify-pass hot path consumed by the DFlash drafter
    /// round loop (sub-12). It mirrors upstream
    /// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/language.py (LanguageModel.__call__)
    /// when called with `capture_layer_ids` set, with two changes:
    ///   1. `return_hidden` is implied by `capture_layer_ids.is_some()` —
    ///      the upstream `return_hidden` flag is redundant on Qwen 3.5 since
    ///      the DFlash drafter always wants a *specific* set of layer captures.
    ///   2. The returned `gdn_states` carries enough verify-pass tensor state
    ///      that [`Self::rollback_speculative_cache`] can later rewind both
    ///      KV (attention) caches and GDN (linear-attention) caches to the
    ///      accepted position.
    ///
    /// Apple Silicon precision (`docs/apple-silicon-precision.md`):
    /// captured hidden tensors keep the verify-pass dtype (bf16/f16); GDN
    /// snapshot tensors keep their per-field dtype. Do not promote to f32.
    ///
    /// Used by: DFlash drafter round loop (sub-12).
    pub fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Qwen3NextCache],
        capture_layer_ids: &[usize],
    ) -> VerifyOutput {
        let h0 = self.embed_tokens.forward(input_ids);

        let shape = mlxcel_core::array_shape(&h0);
        let seq_len = shape[1];

        // Build the full-attention causal mask using the first full-attention
        // layer's cache offset, matching `forward_internal`.
        let fa_idx = self.config.full_attention_interval - 1;
        let fa_mask = if seq_len > 1 {
            let offset = if fa_idx < caches.len() {
                caches[fa_idx].offset()
            } else {
                0
            };
            Some(create_causal_mask(seq_len, offset))
        } else {
            None
        };

        // Order-preserving membership check for capture_layer_ids. We need the
        // *index in capture_layer_ids* so DFlash's drafter sees its layers in
        // the order it requested (the concat-axis ordering is load-bearing).
        let mut hidden_slots: Vec<Option<UniquePtr<MlxArray>>> =
            (0..capture_layer_ids.len()).map(|_| None).collect();

        let mut gdn_states: Vec<GdnRollbackSnapshot> = Vec::new();
        let mut h = h0;
        for (i, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            let mask = if layer.is_linear {
                None
            } else {
                fa_mask.as_deref()
            };
            h = layer.forward_with_capture(i, &h, mask, cache, None, &mut gdn_states);

            // Capture the post-block hidden state for any requested layer index.
            // The check is O(capture_layer_ids.len()) per layer; for DFlash's
            // typical k=5 captures that is negligible.
            for (slot_idx, &want_idx) in capture_layer_ids.iter().enumerate() {
                if want_idx == i {
                    hidden_slots[slot_idx] = Some(mlxcel_core::copy(&h));
                }
            }
        }

        // Materialize hidden states in capture_layer_ids order. Missing slots
        // (out-of-range indices) are filled with zeros to keep the resulting
        // vector length-aligned with the request, but in practice the drafter
        // configures `target_layer_ids` against the model so all entries must
        // resolve.
        let hidden_states: Vec<UniquePtr<MlxArray>> = hidden_slots
            .into_iter()
            .map(|opt| {
                opt.unwrap_or_else(|| {
                    mlxcel_core::zeros(
                        &[shape[0], seq_len, self.config.hidden_size as i32],
                        mlxcel_core::array_dtype(&h),
                    )
                })
            })
            .collect();

        // Final norm + LM head — same as `forward_internal`.
        let h = self.norm.forward(&h);
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        };

        VerifyOutput {
            logits,
            hidden_states,
            gdn_states,
        }
    }

    /// Rewind both KV (attention) and GDN (linear-attention) caches to the
    /// position of the last accepted token after a DFlash verify-pass block.
    ///
    /// Mirrors upstream
    /// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/language.py (LanguageModel.rollback_speculative_cache)
    /// Returns `max(accepted)`.
    ///
    /// Arguments:
    ///   * `caches` — the per-layer cache slice the verify pass just mutated.
    ///   * `gdn_states` — the `gdn_states` field of [`VerifyOutput`] from the
    ///     SAME verify pass. Ordered by linear-attention layer index.
    ///   * `accepted` — per-row accepted count `a_i` (i.e. the prefix
    ///     `[0..a_i]` is kept). For `B == 1` this is a single-element slice.
    ///   * `block_size` — the verify-block length (typically the drafter's
    ///     `block_size` config). Used to compute the trim amount:
    ///     `trim = block_size - (max(accepted) + 1)`.
    ///
    /// Per-row tail zeroing for `B > 1`: rows with smaller accept counts have
    /// their KV-cache tail zeroed and their GDN state re-derived from the
    /// per-row prefix length. Rows that fully accept the block (i.e.
    /// `accepted == block_size - 1`) keep both cache types as the verify-pass
    /// produced them.
    ///
    /// Apple Silicon precision (`docs/apple-silicon-precision.md`):
    /// GDN replay re-uses the captured `q/k/v/a/b` and `init_state` tensors
    /// at their original dtype (bf16/f16/float32 as captured). Do not promote
    /// to f32 — the drafter's reference forward stays in the activation dtype.
    ///
    /// Used by: DFlash drafter round loop (sub-12).
    pub fn rollback_speculative_cache(
        &self,
        caches: &mut [Qwen3NextCache],
        gdn_states: &[GdnRollbackSnapshot],
        accepted: &[i32],
        block_size: i32,
    ) -> i32 {
        if accepted.is_empty() {
            return 0;
        }
        let max_a = *accepted.iter().max().unwrap_or(&0);
        let n = max_a + 1;
        let trim = block_size - n;
        let is_batch = accepted.len() > 1;

        // Attention caches: trim by `trim`. Per-row tail zeroing for batched.
        for cache in caches.iter_mut() {
            if let Qwen3NextCache::Attention(kv) = cache {
                if trim > 0 {
                    kv.trim(trim);
                }
                if is_batch && max_a > 0 {
                    let kv_len = kv.offset;
                    let verify_start = kv_len - n;
                    for (bi, &acc) in accepted.iter().enumerate() {
                        let valid_end = acc + 1;
                        let start = verify_start + valid_end;
                        if start < kv_len {
                            // Zero the per-row tail in both K and V. We rebuild
                            // each tensor by concatenating the head + zero tail
                            // along the seq axis for this row only, then
                            // assembling the batch via copy-slice-replace.
                            zero_per_row_kv_tail(kv, bi as i32, start, kv_len);
                        }
                    }
                }
            }
        }

        // GDN caches: replay the captured block over `[:, :n]` so the
        // post-block state matches the accepted-prefix position.
        let mut snapshot_iter = gdn_states.iter();
        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            let Qwen3NextCache::Linear(linear_cache) = cache else {
                continue;
            };
            let snap = match snapshot_iter.next() {
                Some(s) => s,
                None => continue,
            };
            // Sanity check: the snapshot must come from this layer.
            debug_assert_eq!(snap.layer_idx, layer_idx);

            let layer = match &self.layers[layer_idx].attention {
                Qwen35AttentionVariant::Linear(linear) => linear,
                _ => continue,
            };

            // Re-run gated_delta_update with the first n tokens. We always
            // pass `mask = None`: mlxcel's verify-pass does not maintain an
            // SSM mask, mirroring `forward_hidden_internal`. (The upstream
            // batched-replay mask is also a no-op when `accepted` is uniform.)
            let q_n = mlxcel_core::slice(
                &snap.q,
                &[0, 0, 0, 0],
                &[
                    mlxcel_core::array_shape(&snap.q)[0],
                    n,
                    mlxcel_core::array_shape(&snap.q)[2],
                    mlxcel_core::array_shape(&snap.q)[3],
                ],
            );
            let k_n = mlxcel_core::slice(
                &snap.k,
                &[0, 0, 0, 0],
                &[
                    mlxcel_core::array_shape(&snap.k)[0],
                    n,
                    mlxcel_core::array_shape(&snap.k)[2],
                    mlxcel_core::array_shape(&snap.k)[3],
                ],
            );
            let v_n = mlxcel_core::slice(
                &snap.v,
                &[0, 0, 0, 0],
                &[
                    mlxcel_core::array_shape(&snap.v)[0],
                    n,
                    mlxcel_core::array_shape(&snap.v)[2],
                    mlxcel_core::array_shape(&snap.v)[3],
                ],
            );
            let a_n = mlxcel_core::slice(
                &snap.a,
                &[0, 0, 0],
                &[
                    mlxcel_core::array_shape(&snap.a)[0],
                    n,
                    mlxcel_core::array_shape(&snap.a)[2],
                ],
            );
            let b_n = mlxcel_core::slice(
                &snap.b,
                &[0, 0, 0],
                &[
                    mlxcel_core::array_shape(&snap.b)[0],
                    n,
                    mlxcel_core::array_shape(&snap.b)[2],
                ],
            );

            let (_y, replayed_state) = gated_delta_update(
                &q_n,
                &k_n,
                &v_n,
                &a_n,
                &b_n,
                &layer.a_log,
                &layer.dt_bias,
                snap.init_state.as_deref(),
                None,
            );

            linear_cache.state_cache = Some(replayed_state);

            // Recover conv_state from the captured conv_input.
            // For B=1: cache[0] = conv_input[:, a0+1 : a0+K]  (K = conv_kernel_size).
            // For B>1: per-row slicing because each row may have accepted a
            //          different count. We assemble the per-row conv_state
            //          window first, then concat to a [B, K-1, conv_dim] tensor.
            let k_kernel = layer.conv_kernel_size as i32;
            let conv_shape = mlxcel_core::array_shape(&snap.conv_input);
            let conv_dim = conv_shape[2];
            if is_batch {
                let mut rows: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(accepted.len());
                for (bi, &acc) in accepted.iter().enumerate() {
                    let row = mlxcel_core::slice(
                        &snap.conv_input,
                        &[bi as i32, acc + 1, 0],
                        &[bi as i32 + 1, acc + k_kernel, conv_dim],
                    );
                    rows.push(row);
                }
                // Stack along batch axis 0.
                let mut concatenated = mlxcel_core::copy(rows[0].as_ref().unwrap());
                for row in rows.iter().skip(1) {
                    concatenated = concatenate(&concatenated, row.as_ref().unwrap(), 0);
                }
                linear_cache.conv_state = Some(mlxcel_core::contiguous(&concatenated, false));
            } else {
                let a0 = accepted[0];
                let row = mlxcel_core::slice(
                    &snap.conv_input,
                    &[0, a0 + 1, 0],
                    &[conv_shape[0], a0 + k_kernel, conv_dim],
                );
                linear_cache.conv_state = Some(mlxcel_core::contiguous(&row, false));
            }

            // Roll the linear cache offset back by `trim` so subsequent decode
            // steps see the cache at the accepted position.
            if trim > 0 {
                linear_cache.offset -= trim;
            }
        }

        max_a
    }
}

/// DFlash speculative-decoding target adapter for the Qwen 3.5 text
/// model.
///
/// Bridges `Qwen35Model::forward_speculative` /
/// `Qwen35Model::rollback_speculative_cache` to the
/// drafter-side [`mlxcel_core::drafter::dflash::SpeculativeTarget`]
/// trait. This is what lets the
/// `DFlashGenerator` round loop call into the binary's concrete model
/// type without mlxcel-core having to name `Qwen3NextCache` /
/// `VerifyOutput` / `GdnRollbackSnapshot`.
///
/// The associated types are exactly the binary-side concrete types the
/// verify pass produces; the trait's hooks delegate to the existing
/// instance methods on `Qwen35Model` without any extra allocation or
/// state.
///
/// Used by: DFlash B=1 round loop; future B>1 round loop
/// will provide a peer impl with batched accept slices.
impl mlxcel_core::drafter::dflash::SpeculativeTarget for Qwen35Model {
    type Cache = super::qwen3_next::Qwen3NextCache;
    type VerifyOut = VerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        // The target itself does not own the capture layer ids — the
        // drafter does (`DFlashConfig::target_layer_ids`). The round
        // loop reads this only as documentation; the actual
        // `forward_speculative` call passes its own slice through.
        //
        // We return an empty slice here as a sentinel: callers that
        // need the real value pull it from the drafter side. The
        // round-loop driver never reads this method on the hot path
        // (it passes `capture_layer_ids` to `forward_speculative`
        // directly), but keeping the trait method here means a future
        // configuration plumbing change can attach the real list to
        // the model without breaking the contract.
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.forward_speculative(
            verify_input,
            caches,
            mlxcel_core::drafter::dflash::config::DEFAULT_TARGET_LAYER_IDS,
        )
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        let capture_layer_ids = if capture_layer_ids.is_empty() {
            // Backwards-compatible default for older tests/callers that still
            // invoke `verify_forward` semantics directly.
            mlxcel_core::drafter::dflash::config::DEFAULT_TARGET_LAYER_IDS
        } else {
            capture_layer_ids
        };
        self.forward_speculative(verify_input, caches, capture_layer_ids)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        // Delegates to the per-Qwen-3.5 rollback that combines KV
        // attention-cache trim with GDN linear-attention state replay.
        // For B = 1 the accepted slice is a single-element view; the
        // batched B > 1 path uses `rollback_partial_batched`.
        let _ = self.rollback_speculative_cache(
            caches,
            &verify_out.gdn_states,
            &[accepted],
            block_size,
        );
    }

    fn rollback_partial_batched(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: &[i32],
        block_size: i32,
    ) {
        // For B > 1 the rollback path uses the same `rollback_speculative_cache`
        // entrypoint with a multi-element `accepted` slice. The per-row
        // KV tail-zeroing and per-row GDN-state replay (with per-row
        // conv_state slicing) already lives inside that method;
        // we just thread the slice through.
        //
        // Apple Silicon precision (`docs/apple-silicon-precision.md`):
        // both the KV per-row tail zero buffer and the GDN replay tensors
        // keep their captured dtype (bf16/f16 activations + float32
        // init_state); no f32 promotion.
        let _ =
            self.rollback_speculative_cache(caches, &verify_out.gdn_states, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        // Upstream: `hidden = mx.concatenate(verify_out.hidden_states, axis=-1)`.
        // The captured hidden states each have shape `[1, bs, hidden_size]`;
        // concatenating along axis -1 (the feature axis) yields
        // `[1, bs, num_capture_layers * hidden_size]`.
        debug_assert!(
            !verify_out.hidden_states.is_empty(),
            "DFlash verify output must carry at least one captured hidden layer"
        );
        // `mlxcel_core::concatenate` takes two refs at a time; fold
        // over the captured-hidden vector along the feature axis.
        let mut acc = mlxcel_core::copy(verify_out.hidden_states[0].as_ref().unwrap());
        for slab in verify_out.hidden_states.iter().skip(1) {
            acc = concatenate(&acc, slab.as_ref().unwrap(), -1);
        }
        acc
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        verify_out
            .logits
            .as_ref()
            .expect("DFlash verify output must carry logits")
    }
}

/// Zero the per-row tail `[bi, :, start:kv_len, :]` in both K and V of a
/// `KVCache`. Used by [`Qwen35Model::rollback_speculative_cache`] for batched
/// verify-pass rollback when rows accepted different numbers of tokens.
///
/// Apple Silicon precision: the zero buffer is built with the
/// same dtype as the K/V tensors so no implicit f32 promotion happens.
///
/// Used by: `Qwen35Model::rollback_speculative_cache`
pub(crate) fn zero_per_row_kv_tail(kv: &mut KVCache, bi: i32, start: i32, kv_len: i32) {
    let Some(keys_ref) = kv.keys.as_ref().map(|k| k.as_ref().unwrap()) else {
        return;
    };
    let Some(vals_ref) = kv.values.as_ref().map(|v| v.as_ref().unwrap()) else {
        return;
    };
    let k_shape = mlxcel_core::array_shape(keys_ref);
    let v_shape = mlxcel_core::array_shape(vals_ref);
    let k_dtype = mlxcel_core::array_dtype(keys_ref);
    let v_dtype = mlxcel_core::array_dtype(vals_ref);

    // For each tensor, build a copy where the tail of row `bi` is zeroed.
    // We reconstruct the tensor by concatenating: rows-before + zeroed-row + rows-after,
    // each split per the batch axis. The zeroed-row itself is head + zero-tail per the
    // seq axis. All zero tensors keep the original dtype to honor the no-f32-promotion
    // rule from `docs/apple-silicon-precision.md`.
    kv.keys = Some(rebuild_with_zero_tail(
        keys_ref, &k_shape, bi, start, kv_len, k_dtype,
    ));
    kv.values = Some(rebuild_with_zero_tail(
        vals_ref, &v_shape, bi, start, kv_len, v_dtype,
    ));
}

/// Reconstruct a 4D KV tensor `[B, H, S, D]` where row `bi`'s
/// `[H, start:kv_len, D]` slab is replaced with zeros of the same dtype.
pub(crate) fn rebuild_with_zero_tail(
    tensor: &MlxArray,
    shape: &[i32],
    bi: i32,
    start: i32,
    kv_len: i32,
    dtype: i32,
) -> UniquePtr<MlxArray> {
    let batch = shape[0];
    let heads = shape[1];
    let head_dim = shape[3];

    // Row-bi slice with head + zero-tail along the seq axis.
    let mut head = mlxcel_core::slice(tensor, &[bi, 0, 0, 0], &[bi + 1, heads, start, head_dim]);
    let zero_tail_len = kv_len - start;
    if zero_tail_len > 0 {
        let zero_tail = mlxcel_core::zeros(&[1, heads, zero_tail_len, head_dim], dtype);
        head = concatenate(&head, &zero_tail, 2);
    }

    // Assemble the batch: rows-before, fixed row bi, rows-after.
    let mut out = if bi > 0 {
        let before = mlxcel_core::slice(tensor, &[0, 0, 0, 0], &[bi, heads, kv_len, head_dim]);
        concatenate(&before, &head, 0)
    } else {
        head
    };
    if bi + 1 < batch {
        let after = mlxcel_core::slice(
            tensor,
            &[bi + 1, 0, 0, 0],
            &[batch, heads, kv_len, head_dim],
        );
        out = concatenate(&out, &after, 0);
    }
    mlxcel_core::contiguous(&out, false)
}

pub(crate) struct Qwen35StageModel {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    layers: Vec<Qwen35DecoderLayer>,
    norm: Option<RMSNorm>,
    lm_head: Option<UnifiedLinear>,
}

impl Qwen35StageModel {
    pub(crate) fn load(
        model_dir: &Path,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let v: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let mut text_config_val = if let Some(tc) = v.get("text_config") {
            tc.clone()
        } else {
            v.clone()
        };
        if text_config_val.get("quantization").is_none() && v.get("quantization").is_some() {
            text_config_val
                .as_object_mut()
                .expect("Qwen3.5 text config must be a JSON object")
                .insert("quantization".to_string(), v["quantization"].clone());
        }
        let config: Qwen35Config = serde_json::from_value(text_config_val)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        weights = sanitize_moe_weights(weights, &config);
        let mut effective_filter = filter.clone();
        if config.tie_word_embeddings && filter.has_lm_head {
            effective_filter.has_embedding = true;
        }
        filter_weight_map(&mut weights, &effective_filter);
        Self::from_filtered_weights(&weights, &config, filter, stage_index)
    }

    fn from_filtered_weights(
        weights: &WeightMap,
        config: &Qwen35Config,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();
        let qn_config = config.to_qwen3next_config();

        let load_embeddings =
            filter.has_embedding || (config.tie_word_embeddings && filter.has_lm_head);
        let embed_tokens = if load_embeddings {
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
            layers.push(Qwen35DecoderLayer::from_weights(
                weights, config, &qn_config, layer_idx,
            )?);
        }

        if layers.is_empty() {
            return Err(format!(
                "stage {} did not load any layers from range {}..{}",
                stage_index, filter.layer_range.start, filter.layer_range.end
            ));
        }

        let norm = if filter.has_lm_head {
            Some(RMSNorm::new(
                weights
                    .get("model.norm.weight")
                    .map(|w| mlxcel_core::copy(w))
                    .ok_or_else(|| "Missing model.norm.weight".to_string())?,
                config.rms_norm_eps,
            ))
        } else {
            None
        };

        let lm_head = if filter.has_lm_head && !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            filter: filter.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn make_caches(&self) -> Vec<Qwen3NextCache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Qwen3NextCache::Linear(GatedDeltaCache::new())
                } else {
                    Qwen3NextCache::Attention(KVCache::new())
                }
            })
            .collect()
    }

    pub(crate) fn execute_from_token_ids(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Qwen3NextCache],
    ) -> Result<StageExecutionOutput, String> {
        let hidden = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| {
                "stage does not host embeddings; hidden-state input required".to_string()
            })?
            .forward(input_ids);
        self.execute_hidden(hidden, caches)
    }

    pub(crate) fn execute_from_hidden_states(
        &self,
        hidden_states: UniquePtr<MlxArray>,
        caches: &mut [Qwen3NextCache],
    ) -> Result<StageExecutionOutput, String> {
        if self.filter.has_embedding {
            return Err("entry stage expects token IDs, not hidden states".to_string());
        }
        self.execute_hidden(hidden_states, caches)
    }

    fn execute_hidden(
        &self,
        mut hidden: UniquePtr<MlxArray>,
        caches: &mut [Qwen3NextCache],
    ) -> Result<StageExecutionOutput, String> {
        if caches.len() != self.layers.len() {
            return Err(format!(
                "stage cache count mismatch: expected {}, got {}",
                self.layers.len(),
                caches.len()
            ));
        }

        let shape = mlxcel_core::array_shape(hidden.as_ref().unwrap());
        let seq_len = shape[1];
        let fa_mask = if seq_len > 1 {
            let offset = self
                .layers
                .iter()
                .zip(caches.iter())
                .find_map(|(layer, cache)| {
                    (!layer.is_linear).then_some(match cache {
                        Qwen3NextCache::Attention(kv) => kv.offset,
                        Qwen3NextCache::Linear(gd) => gd.offset,
                    })
                })
                .unwrap_or(0);
            Some(create_causal_mask(seq_len, offset))
        } else {
            None
        };

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let mask = if layer.is_linear {
                None
            } else {
                fa_mask.as_deref()
            };
            hidden = layer.forward(hidden.as_ref().unwrap(), mask, cache, None);
        }

        let hidden = if let Some(norm) = &self.norm {
            norm.forward(hidden.as_ref().unwrap())
        } else {
            hidden
        };

        if self.filter.has_lm_head {
            let logits = if let Some(lm_head) = &self.lm_head {
                lm_head.forward(&hidden)
            } else {
                self.embed_tokens
                    .as_ref()
                    .ok_or_else(|| {
                        "final tied-word-embedding stage missing embeddings".to_string()
                    })?
                    .as_linear(&hidden)
            };
            Ok(StageExecutionOutput::Logits(logits))
        } else {
            Ok(StageExecutionOutput::HiddenStates(hidden))
        }
    }
}

// Weight Sanitization.
pub fn sanitize_weights(mut weights: WeightMap, config: &Qwen35Config) -> WeightMap {
    // 1. Detect sanitization needs
    let has_mtp = weights.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv1d = weights.iter().any(|(k, v)| {
        k.contains("conv1d.weight") && {
            let shape = mlxcel_core::array_shape(v);
            shape.last() != Some(&1)
        }
    });
    let should_shift_norms = has_mtp || has_unsanitized_conv1d;

    // 2. Filter MTP weights
    weights.retain(|k, _| !k.contains("mtp."));

    // 3. Remove lm_head if tied
    if config.tie_word_embeddings {
        weights.remove("lm_head.weight");
    }

    // 4. Conv1d weight transpose and 5. Norm weight shift
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];

    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in &keys {
        // Conv1d weight: moveaxis(2, 1) when shape[-1] != 1
        if k.contains("conv1d.weight") {
            let v = weights.get(k.as_str()).unwrap();
            let shape = mlxcel_core::array_shape(v);
            if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                let transposed = mlxcel_core::swap_axes(v, -1, -2);
                weights.insert(k.clone(), transposed);
            }
        }

        // Norm weight shift (+1.0) when should_shift_norms
        if should_shift_norms && norm_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            let v = weights.get(k.as_str()).unwrap();
            let ndim = mlxcel_core::array_shape(v).len();
            if ndim == 1 {
                let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(v));
                let shifted = mlxcel_core::add(v, &one);
                weights.insert(k.clone(), shifted);
            }
        }
    }

    // 6. MoE expert stacking (same as qwen3_next)
    for l in 0..config.num_hidden_layers {
        if !config.is_moe_layer(l) {
            continue;
        }

        let base = format!("model.layers.{}.mlp.switch_mlp", l);
        // Stack into the names `SparseMoeBlock`/`SwitchGLU` actually load
        // (gate_proj/up_proj/down_proj since #588); w1->gate, w3->up, w2->down
        // (issue #670: stacking into the legacy w1/w2/w3 slots left every
        // qwen3.5-MoE checkpoint failing with "Missing weight: ...gate_proj").
        for (src_proj, dst_proj) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
            let mut expert_weights: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut expert_scales: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut expert_biases: Vec<UniquePtr<MlxArray>> = Vec::new();

            let mut e = 0;
            while let Some(w) = weights.remove(&format!(
                "model.layers.{}.mlp.experts.{}.{}.weight",
                l, e, src_proj
            )) {
                expert_weights.push(w);
                if let Some(s) = weights.remove(&format!(
                    "model.layers.{}.mlp.experts.{}.{}.scales",
                    l, e, src_proj
                )) {
                    expert_scales.push(s);
                }
                if let Some(b) = weights.remove(&format!(
                    "model.layers.{}.mlp.experts.{}.{}.biases",
                    l, e, src_proj
                )) {
                    expert_biases.push(b);
                }
                e += 1;
            }

            if !expert_weights.is_empty() {
                let stacked = stack_arrays(&expert_weights, 0);
                weights.insert(format!("{}.{}.weight", base, dst_proj), stacked);

                if !expert_scales.is_empty() {
                    let stacked = stack_arrays(&expert_scales, 0);
                    weights.insert(format!("{}.{}.scales", base, dst_proj), stacked);
                }

                if !expert_biases.is_empty() {
                    let stacked = stack_arrays(&expert_biases, 0);
                    weights.insert(format!("{}.{}.biases", base, dst_proj), stacked);
                }
            }
        }

        // Handle per-expert gate_proj/up_proj/down_proj naming variant.
        // Checkpoints that store experts under `experts.{e}.gate_proj.weight`
        // instead of `experts.{e}.w1.weight` use this layout; the names map
        // straight onto the block's own gate/up/down slots (issue #670).
        for (src_proj, dst_proj) in [
            ("gate_proj", "gate_proj"),
            ("up_proj", "up_proj"),
            ("down_proj", "down_proj"),
        ] {
            // Skip if this target slot is already populated by the w1/w2/w3 pass above.
            if weights.contains_key(format!("{}.{}.weight", base, dst_proj).as_str()) {
                continue;
            }

            let mut expert_weights: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut expert_scales: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut expert_biases: Vec<UniquePtr<MlxArray>> = Vec::new();

            let mut e = 0;
            while let Some(w) = weights.remove(&format!(
                "model.layers.{}.mlp.experts.{}.{}.weight",
                l, e, src_proj
            )) {
                expert_weights.push(w);
                if let Some(s) = weights.remove(&format!(
                    "model.layers.{}.mlp.experts.{}.{}.scales",
                    l, e, src_proj
                )) {
                    expert_scales.push(s);
                }
                if let Some(b) = weights.remove(&format!(
                    "model.layers.{}.mlp.experts.{}.{}.biases",
                    l, e, src_proj
                )) {
                    expert_biases.push(b);
                }
                e += 1;
            }

            if !expert_weights.is_empty() {
                let stacked = stack_arrays(&expert_weights, 0);
                weights.insert(format!("{}.{}.weight", base, dst_proj), stacked);

                if !expert_scales.is_empty() {
                    let stacked = stack_arrays(&expert_scales, 0);
                    weights.insert(format!("{}.{}.scales", base, dst_proj), stacked);
                }

                if !expert_biases.is_empty() {
                    let stacked = stack_arrays(&expert_biases, 0);
                    weights.insert(format!("{}.{}.biases", base, dst_proj), stacked);
                }
            }
        }
    }

    // 7. MoE gate_up_proj split (for qwen3_5_moe format)
    for l in 0..config.num_hidden_layers {
        let gate_up_key = format!("model.layers.{}.mlp.experts.gate_up_proj", l);
        if let Some(gate_up) = weights.remove(&gate_up_key) {
            let shape = mlxcel_core::array_shape(&gate_up);
            // shape: [num_experts, gate_up_size, hidden] or similar
            // mid = shape[-2] // 2
            let mid = shape[shape.len() - 2] / 2;
            let ndims = shape.len();

            // gate_proj = gate_up[..., :mid, :]
            let mut starts = vec![0i32; ndims];
            let mut stops: Vec<i32> = shape.clone();
            stops[ndims - 2] = mid;
            let gate_proj = mlxcel_core::slice(&gate_up, &starts, &stops);

            // up_proj = gate_up[..., mid:, :]
            starts[ndims - 2] = mid;
            stops[ndims - 2] = shape[ndims - 2];
            let up_proj = mlxcel_core::slice(&gate_up, &starts, &stops);

            let base = format!("model.layers.{}.mlp.switch_mlp", l);
            weights.insert(format!("{}.gate_proj.weight", base), gate_proj);
            weights.insert(format!("{}.up_proj.weight", base), up_proj);

            // Move down_proj if present
            let down_key = format!("model.layers.{}.mlp.experts.down_proj", l);
            if let Some(down) = weights.remove(&down_key) {
                weights.insert(format!("{}.down_proj.weight", base), down);
            }
        }
    }

    // 8. Rename legacy stacked switch_mlp.{w1,w3,w2} → switch_mlp.{gate_proj,
    // up_proj,down_proj}. SparseMoeBlock loads the gate/up/down names since
    // #588; pre-stacked checkpoints that already ship them pass through
    // untouched, and w1/w2/w3-style exports are renamed forward (issue #670:
    // the old direction renamed gate_proj INTO w1 and broke every
    // qwen3.5-MoE checkpoint).
    let rename_map = [
        ("switch_mlp.w1.", "switch_mlp.gate_proj."),
        ("switch_mlp.w3.", "switch_mlp.up_proj."),
        ("switch_mlp.w2.", "switch_mlp.down_proj."),
    ];
    let keys_to_rename: Vec<String> = weights
        .keys()
        .filter(|k| rename_map.iter().any(|(from, _)| k.contains(from)))
        .cloned()
        .collect();
    for key in keys_to_rename {
        for (from, to) in &rename_map {
            if key.contains(from) {
                let new_key = key.replace(from, to);
                if let Some(v) = weights.remove(&key) {
                    weights.insert(new_key, v);
                }
                break;
            }
        }
    }

    weights
}

/// Sanitize weights for MoE wrapper variant (qwen3_5_moe)
/// Handles language_model prefix stripping and gate_up_proj splitting
pub fn sanitize_moe_weights(weights: WeightMap, config: &Qwen35Config) -> WeightMap {
    let mut sanitized = WeightMap::new();

    for (key, value) in weights {
        // Skip vision tower weights
        if key.starts_with("vision_tower") || key.starts_with("model.visual") {
            continue;
        }

        let new_key = if key.starts_with("model.language_model") {
            key.replace("model.language_model", "language_model.model")
        } else if key.starts_with("language_model.") {
            key.clone()
        } else {
            format!("language_model.{}", key)
        };

        sanitized.insert(new_key, value);
    }

    // Handle gate_up_proj split for MoE
    let keys: Vec<String> = sanitized.keys().cloned().collect();
    for key in &keys {
        if key.contains("experts.gate_up_proj") && sanitized.contains_key(key.as_str()) {
            let gate_up = sanitized.remove(key).unwrap();
            let shape = mlxcel_core::array_shape(&gate_up);
            let ndims = shape.len();
            let mid = shape[ndims - 2] / 2;

            let mut starts = vec![0i32; ndims];
            let mut stops: Vec<i32> = shape.clone();
            stops[ndims - 2] = mid;
            let gate_proj = mlxcel_core::slice(&gate_up, &starts, &stops);

            starts[ndims - 2] = mid;
            stops[ndims - 2] = shape[ndims - 2];
            let up_proj = mlxcel_core::slice(&gate_up, &starts, &stops);

            let base = key.replace("experts.gate_up_proj", "switch_mlp");
            sanitized.insert(format!("{}.gate_proj.weight", base), gate_proj);
            sanitized.insert(format!("{}.up_proj.weight", base), up_proj);

            // Move down_proj
            let down_key = key.replace("experts.gate_up_proj", "experts.down_proj");
            if let Some(down) = sanitized.remove(&down_key) {
                sanitized.insert(format!("{}.down_proj.weight", base), down);
            }
        }
    }

    // Strip language_model. prefix for internal model loading
    let mut final_weights = WeightMap::new();
    for (key, value) in sanitized {
        let stripped = if let Some(rest) = key.strip_prefix("language_model.") {
            rest.to_string()
        } else {
            key
        };
        final_weights.insert(stripped, value);
    }

    // Apply standard sanitization
    sanitize_weights(final_weights, config)
}

// LanguageModel trait implementation.
impl LanguageModel for Qwen35Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_sequence_caches(input, None, None)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_sequence_caches(input_ids, input_embeddings, None)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input_ids))
    }

    /// Hand out a shared-buffer handle to the input embedding table for
    /// the DFlash drafter's lazy-bind path. The upstream
    /// `z-lab/Qwen3.5-4B-DFlash` checkpoint omits `embed_tokens.weight`
    /// and resolves it from this Qwen 3.5 target during
    /// `mlxcel_core::drafter::Drafter::bind`.
    fn embed_tokens_module(&self) -> Option<UnifiedEmbedding> {
        Some(self.embed_tokens.clone_shared())
    }

    /// Hand out the untied output projection for DFlash draft checkpoints
    /// that bind their `lm_head` from the target at runtime (for example the
    /// 27B drafter).
    fn lm_head_module(&self) -> Option<UnifiedLinear> {
        self.lm_head.as_ref().map(UnifiedLinear::clone_shared)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.layers.len())
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn supports_batched_prefill(&self) -> bool {
        true
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn supports_paged_decode_backend(&self) -> bool {
        true
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_internal_caches());
    }

    fn reset_runtime_state(&self) {
        // Used by: CxxGenerator single-row generation paths. Qwen 3.5 Next
        // owns mixed attention / GatedDelta cache state in
        // `ModelOwnedSequenceState`; reset the fallback cache slot for fresh
        // CLI / benchmark runs without touching scheduler-owned sequence
        // entries. Do not clear the MRoPE fallback here: VLM callers populate
        // it during embedding preparation immediately before generation.
        self.sequence_state
            .replace_internal(self.make_internal_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
        // drop the per-sequence MRoPE entry alongside the
        // cache state so the map cannot grow as sequences cycle through.
        self.release_mrope_sequence(seq_id);
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<mlxcel_core::generate::ModelStateSnapshot> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot =
                    mlxcel_core::generate::ModelStateSnapshot::new("qwen3_5", token_len);
                for (idx, cache) in state.iter().enumerate() {
                    cache.snapshot_into(&mut snapshot, &format!("layer{idx}"));
                }
                snapshot
            })
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != "qwen3_5" {
            return Err(format!(
                "cannot restore {} snapshot into Qwen 3.5",
                snapshot.family()
            ));
        }
        let mut state = self.make_internal_caches();
        for (idx, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{idx}"));
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_sequence_caches(input_ids, None, seq_id)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_sequence_caches(input_ids, input_embeddings, seq_id)
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_sequence_state(Some(seq_id), |sequence_caches| {
                let visible_lens: Vec<usize> =
                    sequence_caches.iter().map(Self::visible_len).collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &visible_lens)
            })
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_with_context_and_ids(input_ids, None, batch_caches, mask, None)
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        _context: Option<&mlxcel_core::generate::DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        if batch_caches.len() <= 1 || shape[1] <= 1 {
            let token_0 = mlxcel_core::slice(input_ids, &[0, 0], &[1, shape[1]]);
            if batch_caches.len() == 1 {
                return self.forward_with_sequence_id(
                    &token_0,
                    seq_ids.and_then(|ids| ids.first().copied()),
                    batch_caches[0],
                    None,
                );
            }
            let mut result = self.forward_with_sequence_id(
                &token_0,
                seq_ids.and_then(|ids| ids.first().copied()),
                batch_caches[0],
                None,
            );
            for (i, caches) in batch_caches.iter_mut().enumerate().skip(1) {
                let input_i =
                    mlxcel_core::slice(input_ids, &[i as i32, 0], &[i as i32 + 1, shape[1]]);
                let logits_i = self.forward_with_sequence_id(
                    &input_i,
                    seq_ids.and_then(|ids| ids.get(i).copied()),
                    caches,
                    None,
                );
                result = mlxcel_core::concatenate(&result, &logits_i, 0);
            }
            return result;
        }

        if mask.is_some() {
            let input_0 = mlxcel_core::slice(input_ids, &[0, 0], &[1, shape[1]]);
            let mut result = self.forward_with_sequence_id(
                &input_0,
                seq_ids.and_then(|ids| ids.first().copied()),
                batch_caches[0],
                None,
            );
            for (i, caches) in batch_caches.iter_mut().enumerate().skip(1) {
                let input_i =
                    mlxcel_core::slice(input_ids, &[i as i32, 0], &[i as i32 + 1, shape[1]]);
                let logits_i = self.forward_with_sequence_id(
                    &input_i,
                    seq_ids.and_then(|ids| ids.get(i).copied()),
                    caches,
                    None,
                );
                result = mlxcel_core::concatenate(&result, &logits_i, 0);
            }
            return result;
        }

        if let Some(seq_ids) = seq_ids {
            return self.forward_batched_prefill(input_ids, seq_ids);
        }

        let input_0 = mlxcel_core::slice(input_ids, &[0, 0], &[1, shape[1]]);
        let mut result = self.forward_with_sequence_id(&input_0, None, batch_caches[0], None);
        for (i, caches) in batch_caches.iter_mut().enumerate().skip(1) {
            let input_i = mlxcel_core::slice(input_ids, &[i as i32, 0], &[i as i32 + 1, shape[1]]);
            let logits_i = self.forward_with_sequence_id(&input_i, None, caches, None);
            result = mlxcel_core::concatenate(&result, &logits_i, 0);
        }
        result
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![248046, 248044] // Qwen 3.5 EOS tokens
    }
}

#[cfg(test)]
mod sanitize_tests {
    use super::{Qwen35Config, sanitize_weights};
    use mlxcel_core::weights::WeightMap;

    fn moe_config(num_layers: usize, num_experts: usize) -> Qwen35Config {
        serde_json::from_value(serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 16,
            "num_hidden_layers": num_layers,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 0,
            "full_attention_interval": 4,
            "vocab_size": 100
        }))
        .expect("minimal qwen3.5-moe test config")
    }

    #[test]
    fn sanitize_weights_stacks_per_expert_gate_up_down_proj_names() {
        // Per-expert layout with gate_proj/up_proj/down_proj naming (not w1/w2/w3).
        // The names map straight onto the block's gate/up/down slots (#670).
        let num_experts: usize = 3;
        let out = 4i32;
        let in_dim = 8i32;
        let config = moe_config(1, num_experts);

        let mut weights = WeightMap::new();
        for e in 0..num_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                weights.insert(
                    format!("model.layers.0.mlp.experts.{}.{}.weight", e, proj),
                    mlxcel_core::from_slice_f32(
                        &vec![e as f32; (out * in_dim) as usize],
                        &[out, in_dim],
                    ),
                );
            }
        }

        let result = sanitize_weights(weights, &config);

        // Each projection must be stacked into the slots the block loads.
        for dst in ["gate_proj", "up_proj", "down_proj"] {
            let key = format!("model.layers.0.mlp.switch_mlp.{}.weight", dst);
            let arr = result
                .get(key.as_str())
                .unwrap_or_else(|| panic!("missing stacked weight at {key}"));
            let shape = mlxcel_core::array_shape(arr);
            assert_eq!(
                shape,
                vec![num_experts as i32, out, in_dim],
                "stacked shape for {dst} should be [num_experts, out, in_dim]"
            );
        }

        // Per-expert source keys must be consumed.
        for e in 0..num_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let key = format!("model.layers.0.mlp.experts.{}.{}.weight", e, proj);
                assert!(
                    !result.contains_key(key.as_str()),
                    "source key {key} should be removed after stacking"
                );
            }
        }
    }

    #[test]
    fn sanitize_weights_preserves_per_expert_gate_up_down_stack_order() {
        let num_experts: usize = 2;
        let out = 2i32;
        let in_dim = 4i32;
        let config = moe_config(1, num_experts);

        let mut weights = WeightMap::new();
        for expert in 0..num_experts {
            for (proj, base) in [
                ("gate_proj", 10.0_f32),
                ("up_proj", 20.0_f32),
                ("down_proj", 30.0_f32),
            ] {
                let value = base + expert as f32;
                weights.insert(
                    format!("model.layers.0.mlp.experts.{}.{}.weight", expert, proj),
                    mlxcel_core::from_slice_f32(
                        &vec![value; (out * in_dim) as usize],
                        &[out, in_dim],
                    ),
                );
            }
        }

        let result = sanitize_weights(weights, &config);

        for (proj, base) in [
            ("gate_proj", 10.0_f32),
            ("up_proj", 20.0_f32),
            ("down_proj", 30.0_f32),
        ] {
            let key = format!("model.layers.0.mlp.switch_mlp.{}.weight", proj);
            let arr = result
                .get(key.as_str())
                .unwrap_or_else(|| panic!("missing stacked weight at {key}"));
            assert_eq!(
                mlxcel_core::array_shape(arr),
                vec![num_experts as i32, out, in_dim],
                "stacked shape for {proj} should be [num_experts, out, in_dim]"
            );

            for expert in 0..num_experts {
                let actual = mlxcel_core::slice(
                    arr,
                    &[expert as i32, 0, 0],
                    &[expert as i32 + 1, out, in_dim],
                );
                let expected = mlxcel_core::from_slice_f32(
                    &vec![base + expert as f32; (out * in_dim) as usize],
                    &[1, out, in_dim],
                );
                let close = mlxcel_core::allclose(&actual, &expected, 1e-6, 1e-6);
                mlxcel_core::eval(&close);
                assert!(
                    mlxcel_core::item_bool(&close),
                    "{proj} expert {expert} should preserve its source values"
                );
            }
        }
    }

    #[test]
    fn sanitize_weights_stacks_per_expert_gate_up_down_proj_with_scales_biases() {
        // Quantized per-expert gate_proj layout: weight + scales + biases per expert.
        let num_experts: usize = 2;
        let out = 4i32;
        let packed_in = 1i32; // 4-bit packed: in_dim=8 -> 8/8=1 uint32 per row
        let num_groups = 1i32;
        let config = moe_config(1, num_experts);

        let mut weights = WeightMap::new();
        for e in 0..num_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let prefix = format!("model.layers.0.mlp.experts.{}.{}", e, proj);
                weights.insert(
                    format!("{}.weight", prefix),
                    mlxcel_core::from_slice_f32(
                        &vec![0.0; (out * packed_in) as usize],
                        &[out, packed_in],
                    ),
                );
                weights.insert(
                    format!("{}.scales", prefix),
                    mlxcel_core::from_slice_f32(
                        &vec![1.0; (out * num_groups) as usize],
                        &[out, num_groups],
                    ),
                );
                weights.insert(
                    format!("{}.biases", prefix),
                    mlxcel_core::from_slice_f32(
                        &vec![0.0; (out * num_groups) as usize],
                        &[out, num_groups],
                    ),
                );
            }
        }

        let result = sanitize_weights(weights, &config);

        for (dst, suffix) in [
            ("gate_proj", "weight"),
            ("gate_proj", "scales"),
            ("gate_proj", "biases"),
            ("down_proj", "weight"),
            ("down_proj", "scales"),
            ("down_proj", "biases"),
            ("up_proj", "weight"),
            ("up_proj", "scales"),
            ("up_proj", "biases"),
        ] {
            let key = format!("model.layers.0.mlp.switch_mlp.{}.{}", dst, suffix);
            assert!(
                result.contains_key(key.as_str()),
                "stacked quantized tensor missing at {key}"
            );
        }
    }

    #[test]
    fn sanitize_weights_gate_up_down_proj_does_not_clobber_w1_w2_w3() {
        // When both naming conventions appear in the same checkpoint, the w1/w2/w3
        // pass runs first and populates the slots; the gate_proj pass must not
        // overwrite them.  Verified by checking that the gate_proj source keys
        // survive (not consumed) while the w1 stacked output is present.
        let num_experts: usize = 2;
        let out = 4i32;
        let in_dim = 8i32;
        let config = moe_config(1, num_experts);

        let mut weights = WeightMap::new();
        // Populate via the w1/w2/w3 naming.
        for e in 0..num_experts {
            for proj in ["w1", "w2", "w3"] {
                weights.insert(
                    format!("model.layers.0.mlp.experts.{}.{}.weight", e, proj),
                    mlxcel_core::from_slice_f32(
                        &vec![1.0; (out * in_dim) as usize],
                        &[out, in_dim],
                    ),
                );
            }
        }
        // Add gate_proj-named tensors for the same layer.
        for e in 0..num_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                weights.insert(
                    format!("model.layers.0.mlp.experts.{}.{}.weight", e, proj),
                    mlxcel_core::from_slice_f32(
                        &vec![2.0; (out * in_dim) as usize],
                        &[out, in_dim],
                    ),
                );
            }
        }

        let result = sanitize_weights(weights, &config);

        // The stacked gate/up/down slots must exist (from the w1/w2/w3 source,
        // which runs first and wins the slot).
        for dst in ["gate_proj", "up_proj", "down_proj"] {
            let key = format!("model.layers.0.mlp.switch_mlp.{}.weight", dst);
            assert!(
                result.contains_key(key.as_str()),
                "stacked {dst} should be present"
            );
        }
        // The gate_proj source keys should be untouched (not consumed) because
        // the slot was already filled by the w1/w2/w3 pass.
        for e in 0..num_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let key = format!("model.layers.0.mlp.experts.{}.{}.weight", e, proj);
                assert!(
                    result.contains_key(key.as_str()),
                    "source key {key} should remain when slot is already filled"
                );
            }
        }
    }

    /// The missing gate behind issue #670: sanitize output must actually LOAD
    /// through the shared `SwitchGLU` (which requests gate/up/down since
    /// #588), for every naming convention checkpoints ship. Output-key-only
    /// assertions passed against the wrong convention while every real
    /// qwen3.5-MoE checkpoint failed with "Missing weight: ...gate_proj".
    #[test]
    fn sanitize_output_loads_through_switch_glu_for_all_naming_conventions() {
        use crate::models::qwen3_next::SwitchGLU;

        let num_experts: usize = 2;
        let out = 4i32;
        let in_dim = 8i32;
        let next_config: crate::models::qwen3_next::Qwen3NextConfig =
            serde_json::from_value(serde_json::json!({
                "model_type": "qwen3_next",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "num_experts": num_experts,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 0,
                "full_attention_interval": 4,
                "vocab_size": 100
            }))
            .expect("minimal qwen3_next config");

        let mk = |vals: f32| {
            mlxcel_core::from_slice_f32(&vec![vals; (out * in_dim) as usize], &[out, in_dim])
        };

        // Convention A: per-expert w1/w2/w3. B: per-expert gate/up/down.
        // C: pre-stacked gate/up/down. D: pre-stacked (legacy) w1/w2/w3.
        let mut variants: Vec<(&str, WeightMap)> = Vec::new();

        let mut a = WeightMap::new();
        for e in 0..num_experts {
            for proj in ["w1", "w2", "w3"] {
                a.insert(
                    format!("model.layers.0.mlp.experts.{}.{}.weight", e, proj),
                    mk(1.0),
                );
            }
        }
        variants.push(("per-expert w1/w2/w3", a));

        let mut b = WeightMap::new();
        for e in 0..num_experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                b.insert(
                    format!("model.layers.0.mlp.experts.{}.{}.weight", e, proj),
                    mk(1.0),
                );
            }
        }
        variants.push(("per-expert gate/up/down", b));

        let mut c = WeightMap::new();
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            c.insert(
                format!("model.layers.0.mlp.switch_mlp.{}.weight", proj),
                mlxcel_core::from_slice_f32(
                    &vec![1.0; (num_experts as i32 * out * in_dim) as usize],
                    &[num_experts as i32, out, in_dim],
                ),
            );
        }
        variants.push(("pre-stacked gate/up/down", c));

        let mut d = WeightMap::new();
        for proj in ["w1", "w2", "w3"] {
            d.insert(
                format!("model.layers.0.mlp.switch_mlp.{}.weight", proj),
                mlxcel_core::from_slice_f32(
                    &vec![1.0; (num_experts as i32 * out * in_dim) as usize],
                    &[num_experts as i32, out, in_dim],
                ),
            );
        }
        variants.push(("pre-stacked legacy w1/w2/w3", d));

        let config = moe_config(1, num_experts);
        for (label, weights) in variants {
            let sanitized = sanitize_weights(weights, &config);
            SwitchGLU::from_weights(&sanitized, &next_config, "model.layers.0.mlp.switch_mlp")
                .unwrap_or_else(|e| panic!("SwitchGLU must load after sanitize for {label}: {e}"));
        }
    }
}
