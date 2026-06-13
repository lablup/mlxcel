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

//! IBM Granite 4.x: interleaved Mamba2 (SSM) + attention hybrid with the Granite
//! scalar multipliers.
//!
//! Each `GraniteMoeHybridLayer` is EITHER a Mamba2 mixer OR a GQA attention
//! mixer, chosen by `layer_types[i] in {"mamba", "attention"}` (interleaved,
//! like PLaMo 2 / LFM2 / Nemotron-H, NOT the parallel both-per-layer Falcon-H1
//! block). The block applies the four Granite multipliers verbatim from the
//! reference:
//!
//! ```text
//! residual = x; h = input_layernorm(x)
//! h = mamba(h, cache)  OR  self_attn(h, mask, cache)        // by layer_type
//! x = residual + h * residual_multiplier
//! residual = x; n = post_attention_layernorm(x)
//! ff = mlp(n)                                                // dense
//!   OR ff = block_sparse_moe(n) + shared_mlp(n)              // MoE
//! x = residual + ff * residual_multiplier
//! ```
//!
//! and at the model boundary: `h = embed_tokens(x) * embedding_multiplier`,
//! `attention_multiplier` is the SDPA scale (not `1/sqrt(head_dim)`), the final
//! `logits = (tied embed / lm_head)(h) / logits_scaling`.
//!
//! Mirrored from the mlx-lm reference:
//! - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/granitemoehybrid.py
//!
//! ## Mode selection (`use_moe = num_local_experts > 0`)
//!
//! For a dense checkpoint (`granite-4.0-h-350m`, `num_local_experts == 0`) the
//! per-layer feed-forward is a standard SwiGLU `GraniteMoeHybridMLP`. For an MoE
//! checkpoint the feed-forward is `block_sparse_moe(x) + shared_mlp(x)`, where
//! `block_sparse_moe` routes through [`SwitchGLU`] with a softmax-over-top-k
//! gate and `shared_mlp` is an always-on fused gate/up SwiGLU.
//!
//! ## NoPE attention
//!
//! When `position_embedding_type == "nope"` (the granite-4.x default) the
//! attention applies NO RoPE; the KV cache offset is still tracked so the causal
//! mask stays anchored. Otherwise standard GQA with RoPE.
//!
//! ## Mamba2 mixer (adapted from [`super::falcon_h1`])
//!
//! The `in_proj` carries `gate / conv_input / dt` directly (standard Mamba2
//! layout), runs the depthwise causal conv + SiLU, splits `hidden / B / C`, and
//! runs the SSD scan (the float32-hardened graph path for prefill and the fused
//! Metal `ssm_update_kernel` for single-token decode, both shared with Falcon-H1
//! and PLaMo 2). Two Granite differences from Falcon-H1: the post-scan gate is
//! ALWAYS the gated RMSNorm (`GraniteMoeHybridRMSNormGated`), which gates BEFORE
//! a PLAIN full-weight `rms_norm` (no `n_groups` reshape), and the SSM
//! `time_step_limit` is the `ssm_update` default `(0.001, 100.0)`.
//!
//! Because the Mamba2 conv/SSM state is recurrent (not per-token positional), the
//! model owns a mixed per-layer cache (`Mamba2Cache` for Mamba layers, `KVCache`
//! for attention layers) through [`ModelOwnedSequenceState`] and reports
//! `supports_batching() == false` (like Jamba / Nemotron-H / LFM2 / Falcon-H1 /
//! PLaMo 2).

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, silu, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

use super::mamba2::Mamba2Cache;
use super::model_owned::ModelOwnedSequenceState;
use super::recurrent_snapshot::{push_optional, restore_optional};
use super::switch_layers::{SwitchGLU, moe_weighted_sum};

// Configuration.

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,

    // Granite scalar multipliers.
    pub embedding_multiplier: f32,
    pub attention_multiplier: f32,
    pub logits_scaling: f32,
    pub residual_multiplier: f32,

    /// Per-layer mixer selector: `"mamba"` or `"attention"`.
    pub layer_types: Vec<String>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_position_embedding_type")]
    pub position_embedding_type: String,

    // MoE parameters (optional; dense when `num_local_experts == 0`).
    #[serde(default)]
    pub num_local_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub shared_intermediate_size: usize,

    // Mamba2 mixer dimensions.
    #[serde(default = "default_d_conv")]
    pub mamba_d_conv: usize,
    #[serde(default = "default_d_state")]
    pub mamba_d_state: usize,
    #[serde(default = "default_n_heads")]
    pub mamba_n_heads: usize,
    #[serde(default = "default_d_head")]
    pub mamba_d_head: usize,
    #[serde(default = "default_n_groups")]
    pub mamba_n_groups: usize,
    #[serde(default = "default_true")]
    pub mamba_conv_bias: bool,
    #[serde(default)]
    pub mamba_proj_bias: bool,

    #[serde(default = "default_time_step_limit")]
    pub time_step_limit: (f32, f32),

    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_position_embedding_type() -> String {
    "rope".to_string()
}
fn default_d_conv() -> usize {
    4
}
fn default_d_state() -> usize {
    128
}
fn default_n_heads() -> usize {
    128
}
fn default_d_head() -> usize {
    64
}
fn default_n_groups() -> usize {
    1
}
fn default_true() -> bool {
    true
}
fn default_time_step_limit() -> (f32, f32) {
    // The `ssm_update` default (Granite passes no explicit limit).
    (0.001, 100.0)
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

    /// Attention head dimension (`hidden_size / num_attention_heads`).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Mamba inner width: `mamba_n_heads * mamba_d_head`.
    pub fn mamba_intermediate(&self) -> usize {
        self.mamba_n_heads * self.mamba_d_head
    }

    /// `conv_dim = mamba_intermediate + 2 * n_groups * d_state` (the depthwise
    /// conv channels span the SSM hidden states plus the `B` and `C` projections).
    pub fn conv_dim(&self) -> usize {
        self.mamba_intermediate() + 2 * self.mamba_n_groups * self.mamba_d_state
    }

    /// `in_proj` output width: `gate (intermediate) + conv_input (conv_dim) + dt (n_heads)`.
    pub fn projection_size(&self) -> usize {
        self.mamba_intermediate() + self.conv_dim() + self.mamba_n_heads
    }

    /// Whether this checkpoint runs sparse-MoE feed-forward (`num_local_experts > 0`).
    pub fn use_moe(&self) -> bool {
        self.num_local_experts > 0
    }

    /// Whether attention applies RoPE. `position_embedding_type == "nope"`
    /// (the granite-4.x default) disables it.
    pub fn use_rope(&self) -> bool {
        self.position_embedding_type != "nope"
    }

    /// Whether layer `i` is a Mamba mixer (`layer_types[i] == "mamba"`).
    pub fn is_mamba_layer(&self, i: usize) -> bool {
        self.layer_types
            .get(i)
            .map(|t| t == "mamba")
            .unwrap_or(false)
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        // Granite 4.x checkpoints use the `<|end_of_text|>` id (100257).
        super::mamba::parse_eos_token_ids(&self.eos_token_id, 100257)
    }
}

// Gated RMSNorm (`GraniteMoeHybridRMSNormGated`).
//
// `rms_norm(swiglu(gate, y), weight, eps)` = gate BEFORE a PLAIN full-weight
// RMSNorm (no `n_groups` grouping, unlike the Falcon-H1 gated norm). Promotes to
// float32 for the whole computation: float16/bf16 RMS-norm (x^2 sum) and
// mixed-dtype multiply can overflow to NaN on M5 Max (Metal GPU Family 4) NAx
// kernels.
struct GraniteMoeHybridRMSNormGated {
    weight: UniquePtr<MlxArray>,
    eps: f32,
    dim: i32,
}

impl GraniteMoeHybridRMSNormGated {
    fn forward(&self, y: &MlxArray, gate: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_dtype = mlxcel_core::array_dtype(y);

        let y_f32 = mlxcel_core::astype(y, mlxcel_core::dtype::FLOAT32);
        let g_f32 = mlxcel_core::astype(gate, mlxcel_core::dtype::FLOAT32);
        // swiglu(gate, y) = silu(gate) * y (gate applied before the norm).
        let gated = mlxcel_core::multiply(&y_f32, &silu(&g_f32));

        // Full-weight RMSNorm (no grouping): normalize with a ones vector, then
        // multiply the learned weight, keeping everything in float32.
        let ones = mlxcel_core::ones(&[self.dim], mlxcel_core::dtype::FLOAT32);
        let normed = mlxcel_core::fast_rms_norm(&gated, &ones, self.eps);
        let w_f32 = mlxcel_core::astype(&self.weight, mlxcel_core::dtype::FLOAT32);
        let result = mlxcel_core::multiply(&w_f32, &normed);

        mlxcel_core::astype(&result, orig_dtype)
    }
}

// SSD-SSM helpers (adapted from `super::falcon_h1`).

/// Repeat an array along `axis` by `repeats` (broadcast then reshape).
fn repeat_axis(x: &MlxArray, repeats: i32, axis: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len() as i32;
    let axis = if axis < 0 { ndim + axis } else { axis };

    let mut new_shape: Vec<i32> = shape.iter().take(axis as usize + 1).copied().collect();
    new_shape.push(1);
    new_shape.extend(shape.iter().skip(axis as usize + 1));
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    new_shape[axis as usize + 1] = repeats;
    let x_broad = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    let mut final_shape: Vec<i32> = shape.clone();
    final_shape[axis as usize] *= repeats;
    mlxcel_core::reshape(&x_broad, &final_shape)
}

/// Segmented cumulative sum used by the SSD surrogate-attention decay.
fn segsum(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let l = shape[shape.len() - 1];

    let mut new_shape = shape.clone();
    new_shape.push(1);
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    let last_idx = new_shape.len() - 1;
    new_shape[last_idx] = l;
    let x_rep = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    let x_tril = mlxcel_core::tril(&x_rep, -1);
    mlxcel_core::cumsum(&x_tril, -2, false, true)
}

// Mamba2 mixer.

struct GraniteMoeHybridMamba2Mixer {
    num_heads: usize,
    ssm_state_size: usize,
    conv_kernel_size: usize,
    intermediate_size: usize,
    n_groups: usize,
    head_dim: usize,
    time_step_limit: (f32, f32),
    conv_dim: usize,

    conv_weight: UniquePtr<MlxArray>,
    conv_bias: Option<UniquePtr<MlxArray>>,
    in_proj: UnifiedLinear,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    d_param: UniquePtr<MlxArray>,
    norm: GraniteMoeHybridRMSNormGated,
    out_proj: UnifiedLinear,
}

impl GraniteMoeHybridMamba2Mixer {
    fn forward(
        &self,
        hidden_states: &MlxArray,
        mut cache: Option<&mut Mamba2Cache>,
    ) -> UniquePtr<MlxArray> {
        let projected = self.in_proj.forward(hidden_states);

        // Split into gate, conv_input, dt.
        let gate = slice_axis(&projected, -1, 0, self.intermediate_size as i32);
        let conv_input = slice_axis(
            &projected,
            -1,
            self.intermediate_size as i32,
            (self.intermediate_size + self.conv_dim) as i32,
        );
        let dt = slice_axis(
            &projected,
            -1,
            (self.intermediate_size + self.conv_dim) as i32,
            -1,
        );

        // Depthwise causal conv1d with left padding / cached state.
        let shape = mlxcel_core::array_shape(&conv_input);
        let batch = shape[0];
        let k = self.conv_kernel_size;

        let conv_state_ref = cache
            .as_ref()
            .and_then(|c| c.conv_state.as_ref())
            .and_then(|s| s.as_ref());

        let padded_input = if let Some(conv_st) = conv_state_ref {
            concatenate(conv_st, &conv_input, 1)
        } else {
            let conv_dtype = mlxcel_core::array_dtype(&conv_input);
            let pad_arr =
                mlxcel_core::zeros(&[batch, (k - 1) as i32, self.conv_dim as i32], conv_dtype);
            concatenate(&pad_arr, &conv_input, 1)
        };

        // Persist the trailing `k - 1` time steps as the next conv state.
        // `contiguous` forces a fresh buffer so the cached slice does not retain
        // the full `padded_input` allocation (a per-token memory leak otherwise).
        if let Some(c) = cache.as_deref_mut() {
            let padded_shape = mlxcel_core::array_shape(&padded_input);
            let len = padded_shape[1] as usize;
            let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
            c.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        }

        let conv_out = mlxcel_core::conv1d(
            &padded_input,
            &self.conv_weight,
            1,
            0,
            1,
            self.conv_dim as i32,
        );
        let conv_out = if let Some(ref b) = self.conv_bias {
            let b_reshaped = mlxcel_core::reshape(b, &[1, 1, -1]);
            mlxcel_core::add(&conv_out, &b_reshaped)
        } else {
            conv_out
        };
        let conv_output = silu(&conv_out);

        // Split conv output into hidden_states, B, C.
        let bc_size = (self.n_groups * self.ssm_state_size) as i32;
        let hidden_ssm = slice_axis(&conv_output, -1, 0, self.intermediate_size as i32);
        let b = slice_axis(
            &conv_output,
            -1,
            self.intermediate_size as i32,
            self.intermediate_size as i32 + bc_size,
        );
        let c = slice_axis(
            &conv_output,
            -1,
            self.intermediate_size as i32 + bc_size,
            -1,
        );

        // SSD-SSM scan. Fused Metal kernel for single-token decode with state;
        // graph path (float32-hardened) for prefill and the first token.
        let ssm_state_ref = cache
            .as_ref()
            .and_then(|c| c.ssm_state.as_ref())
            .and_then(|s| s.as_ref());

        let seq_len = shape[1];
        let (y, new_state) = if seq_len == 1 && mlxcel_core::ssm_kernel_available() {
            if let Some(state) = ssm_state_ref {
                self.ssm_step_kernel(&hidden_ssm, &b, &c, &dt, state)
            } else {
                self.ssm_step(&hidden_ssm, &b, &c, &dt, ssm_state_ref)
            }
        } else {
            self.ssm_step(&hidden_ssm, &b, &c, &dt, ssm_state_ref)
        };

        if let Some(c) = cache {
            c.ssm_state = Some(new_state);
        }

        let y = mlxcel_core::reshape(&y, &[batch, seq_len, self.intermediate_size as i32]);

        // Gate the scan output through the always-on gated RMSNorm.
        let y_gated = self.norm.forward(&y, &gate);

        let result = self.out_proj.forward(&y_gated);

        // Materialize at a clean dtype boundary. The lazy SSM graph mixes
        // float32/float16 nodes that can produce NaN when fused with downstream
        // layers in one Metal command buffer on M5 Max (Metal GPU Family 4).
        mlxcel_core::eval(&result);
        result
    }

    /// Float32-hardened SSD-SSM scan (prefill and first-token path). Mirrors the
    /// Falcon-H1 / PLaMo 2 scan; promotes `x`, `B`, `C`, `dt` to float32 before
    /// the matmuls to avoid float16 overflow and mixed-dtype NaN on M5 Max.
    fn ssm_step(
        &self,
        hidden_states: &MlxArray,
        b: &MlxArray,
        c: &MlxArray,
        dt: &MlxArray,
        state: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(hidden_states);
        let batch = shape[0];
        let seq_len = shape[1];

        let x = mlxcel_core::reshape(
            hidden_states,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );
        let b = mlxcel_core::reshape(
            b,
            &[
                batch,
                seq_len,
                self.n_groups as i32,
                self.ssm_state_size as i32,
            ],
        );
        let c = mlxcel_core::reshape(
            c,
            &[
                batch,
                seq_len,
                self.n_groups as i32,
                self.ssm_state_size as i32,
            ],
        );

        let num_heads = self.num_heads as i32;
        let n_groups = self.n_groups as i32;
        let state_dim = self.ssm_state_size as i32;
        let head_dim = self.head_dim as i32;
        let repeats = num_heads / n_groups;

        // dt = clip(softplus(dt + dt_bias), limit) in float32.
        let dt_f32 = mlxcel_core::astype(dt, mlxcel_core::dtype::FLOAT32);
        let dt_bias_f32 = mlxcel_core::astype(&self.dt_bias, mlxcel_core::dtype::FLOAT32);
        let dt_biased = mlxcel_core::add(&dt_f32, &dt_bias_f32);
        let dt_soft = mlxcel_core::softplus(&dt_biased);
        let min_val =
            mlxcel_core::full_f32(&[1], self.time_step_limit.0, mlxcel_core::dtype::FLOAT32);
        let max_val =
            mlxcel_core::full_f32(&[1], self.time_step_limit.1, mlxcel_core::dtype::FLOAT32);
        let dt = mlxcel_core::clip(&dt_soft, &min_val, &max_val);

        // A = -exp(A_log).
        let a = mlxcel_core::negative(&mlxcel_core::exp(&self.a_log));
        let a = mlxcel_core::astype(&a, mlxcel_core::array_dtype(&dt));
        let a_reshaped = mlxcel_core::reshape(&a, &[1, 1, num_heads]);
        let dt_a = mlxcel_core::multiply(&dt, &a_reshaped);

        let dt_exp = mlxcel_core::reshape(&dt, &[batch, seq_len, num_heads, 1]);
        let x_f32 = mlxcel_core::astype(&x, mlxcel_core::dtype::FLOAT32);
        let dtx = mlxcel_core::multiply(&dt_exp, &x_f32);

        let b_f32 = mlxcel_core::astype(&b, mlxcel_core::dtype::FLOAT32);
        let c_f32 = mlxcel_core::astype(&c, mlxcel_core::dtype::FLOAT32);

        // CB = C.swapaxes(1, 2) @ B.transpose(0, 2, 3, 1), repeated per head.
        let b_t = mlxcel_core::transpose_axes(&b_f32, &[0, 2, 3, 1]);
        let c_t = mlxcel_core::swap_axes(&c_f32, 1, 2);
        let cb = mlxcel_core::matmul(&c_t, &b_t);
        let cb = repeat_axis(&cb, repeats, 1);

        // decay = exp(segsum(dtA.swapaxes(1, 2)));  surrogate = tril(CB * decay).
        let dt_a_t = mlxcel_core::swap_axes(&dt_a, 1, 2);
        let seg = segsum(&dt_a_t);
        let decay = mlxcel_core::exp(&seg);
        let attn_matrix = mlxcel_core::multiply(&cb, &decay);
        let attn_matrix = mlxcel_core::tril(&attn_matrix, 0);

        let dtx_t = mlxcel_core::swap_axes(&dtx, 1, 2);
        let y = mlxcel_core::matmul(&attn_matrix, &dtx_t);
        let y = mlxcel_core::swap_axes(&y, 1, 2);

        // next_state from this chunk.
        let decay_shape = mlxcel_core::array_shape(&decay);
        let decay_last = slice_axis(&decay, 2, decay_shape[2] - 1, decay_shape[2]);
        let decay_t = mlxcel_core::transpose_axes(&decay_last, &[0, 3, 1, 2]);

        let b_rep = repeat_axis(&b_t, repeats, 1);
        let b_sw = mlxcel_core::swap_axes(&b_rep, 2, 3);

        let dtx_decay = mlxcel_core::multiply(&dtx, &decay_t);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay, 1, 2);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay_t, 2, 3);
        let mut next_state = mlxcel_core::matmul(&dtx_decay_t, &b_sw);

        // Carry the previous recurrent state forward.
        let y = if let Some(prev_state) = state {
            let dta_cumsum = mlxcel_core::cumsum(&dt_a, -2, false, true);
            let exp_dta_cumsum = mlxcel_core::exp(&dta_cumsum);

            let exp_shape = mlxcel_core::array_shape(&exp_dta_cumsum);
            let last_exp = slice_axis(&exp_dta_cumsum, 1, exp_shape[1] - 1, exp_shape[1]);
            let last_exp = mlxcel_core::reshape(&last_exp, &[batch, num_heads, 1, 1]);
            let state_contrib = mlxcel_core::multiply(&last_exp, prev_state);
            next_state = mlxcel_core::add(&next_state, &state_contrib);

            let c_reshaped =
                mlxcel_core::reshape(&c_f32, &[batch, seq_len, n_groups, 1, state_dim, 1]);
            let state_reshaped = mlxcel_core::reshape(
                prev_state,
                &[batch, 1, n_groups, repeats, head_dim, state_dim],
            );
            let y_prev = mlxcel_core::matmul(&state_reshaped, &c_reshaped);
            let y_prev = mlxcel_core::squeeze_axis(&y_prev, -1);
            let y_prev = mlxcel_core::reshape(&y_prev, &[batch, seq_len, num_heads, head_dim]);

            let mut exp_shape_y = mlxcel_core::array_shape(&exp_dta_cumsum);
            exp_shape_y.push(1);
            let exp_dta_exp = mlxcel_core::reshape(&exp_dta_cumsum, &exp_shape_y);
            let y_prev_scaled = mlxcel_core::multiply(&exp_dta_exp, &y_prev);
            mlxcel_core::add(&y, &y_prev_scaled)
        } else {
            y
        };

        // y = y + x * D (float32 throughout to avoid mixed-dtype NaN).
        let d_f32 = mlxcel_core::astype(&self.d_param, mlxcel_core::dtype::FLOAT32);
        let d_reshaped = mlxcel_core::reshape(&d_f32, &[1, 1, num_heads, 1]);
        let d_contrib = mlxcel_core::multiply(&x_f32, &d_reshaped);
        let y = mlxcel_core::add(&y, &d_contrib);

        let y = mlxcel_core::astype(&y, mlxcel_core::array_dtype(hidden_states));

        mlxcel_core::eval(&y);
        mlxcel_core::eval(&next_state);

        (y, next_state)
    }

    /// Fused single-token decode scan via the SSM Metal kernel. The kernel
    /// applies `compute_dt` internally, so the raw `dt` and `dt_bias` are passed
    /// through alongside the time-step limits.
    fn ssm_step_kernel(
        &self,
        hidden_states: &MlxArray,
        b: &MlxArray,
        c: &MlxArray,
        dt: &MlxArray,
        state: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(hidden_states);
        let batch = shape[0];
        let seq_len = shape[1];

        let x = mlxcel_core::reshape(
            hidden_states,
            &[batch, seq_len, self.num_heads as i32, self.head_dim as i32],
        );
        let b = mlxcel_core::reshape(
            b,
            &[
                batch,
                seq_len,
                self.n_groups as i32,
                self.ssm_state_size as i32,
            ],
        );
        let c = mlxcel_core::reshape(
            c,
            &[
                batch,
                seq_len,
                self.n_groups as i32,
                self.ssm_state_size as i32,
            ],
        );

        let mut output = mlxcel_core::UniquePtr::null();
        let mut next_state = mlxcel_core::UniquePtr::null();

        mlxcel_core::ssm_update_kernel(
            &x,
            &self.a_log,
            &b,
            &c,
            &self.d_param,
            dt,
            &self.dt_bias,
            state,
            self.time_step_limit.0,
            self.time_step_limit.1,
            &mut output,
            &mut next_state,
        );

        (output, next_state)
    }
}

// Attention (GQA + optional RoPE; `attention_multiplier` SDPA scale, no Q/K norm).

struct GraniteMoeHybridAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    /// `Some(theta)` applies RoPE; `None` is the NoPE path
    /// (`position_embedding_type == "nope"`).
    rope_base: Option<f32>,
}

impl GraniteMoeHybridAttention {
    fn forward(
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

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Apply RoPE only when not NoPE. The KV cache offset is tracked either
        // way (the causal mask is anchored on it).
        let (q, k) = if let Some(base) = self.rope_base {
            let offset = cache.offset;
            (
                mlxcel_core::fast_rope(&q, self.head_dim, false, base, 1.0, offset),
                mlxcel_core::fast_rope(&k, self.head_dim, false, base, 1.0, offset),
            )
        } else {
            (q, k)
        };

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let attn_out = if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }
}

// Dense SwiGLU MLP (`GraniteMoeHybridMLP`).

struct GraniteMoeHybridMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl GraniteMoeHybridMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

// Sparse MoE block (`block_sparse_moe`): softmax-over-top-k router + SwitchGLU.

struct GraniteMoeHybridMoE {
    router: UnifiedLinear,
    switch_mlp: SwitchGLU,
    top_k: i32,
    num_experts: i32,
}

impl GraniteMoeHybridMoE {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Router logits, then top-k selection via argpartition.
        let logits = self.router.forward(&x_flat);
        let kth = self.num_experts - self.top_k;
        let indices = mlxcel_core::argpartition(&logits, kth, -1);
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

        // softmax over ONLY the top-k logits (precise = float32). This equals
        // softmax-over-all then renormalize over the top-k (norm_topk_prob).
        let topk_logits = mlxcel_core::take_along_axis(&logits, &topk_indices, -1);
        let topk_logits = mlxcel_core::astype(&topk_logits, mlxcel_core::dtype::FLOAT32);
        let scores = mlxcel_core::softmax(&topk_logits, -1);

        let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices);
        let result = moe_weighted_sum(&expert_out, &scores, mlxcel_core::array_dtype(&x_flat));

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// Shared dense expert (`shared_mlp`): fused gate/up SwiGLU, always summed with MoE.

struct GraniteMoeHybridSharedMLP {
    input_linear: UnifiedLinear,
    output_linear: UnifiedLinear,
    shared_intermediate: i32,
}

impl GraniteMoeHybridSharedMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate_up = self.input_linear.forward(x);
        let gate = slice_axis(&gate_up, -1, 0, self.shared_intermediate);
        let up = slice_axis(
            &gate_up,
            -1,
            self.shared_intermediate,
            2 * self.shared_intermediate,
        );
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.output_linear.forward(&activated)
    }
}

// Per-layer feed-forward: dense MLP, or MoE + shared MLP.

enum FeedForward {
    Dense(GraniteMoeHybridMLP),
    Moe {
        moe: GraniteMoeHybridMoE,
        shared: GraniteMoeHybridSharedMLP,
    },
}

impl FeedForward {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::Moe { moe, shared } => {
                let moe_out = moe.forward(x);
                let shared_out = shared.forward(x);
                mlxcel_core::add(&moe_out, &shared_out)
            }
        }
    }
}

// Per-layer mixer (Mamba OR attention).

enum Mixer {
    Mamba(GraniteMoeHybridMamba2Mixer),
    Attention(GraniteMoeHybridAttention),
}

// Decoder layer: input_layernorm -> mixer -> residual * mult;
//                post_attention_layernorm -> feed_forward -> residual * mult.

struct GraniteMoeHybridDecoderLayer {
    mixer: Mixer,
    feed_forward: FeedForward,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    residual_multiplier: f32,
}

impl GraniteMoeHybridDecoderLayer {
    fn is_attention(&self) -> bool {
        matches!(self.mixer, Mixer::Attention(_))
    }

    fn make_cache(&self) -> GraniteMoeHybridLayerCache {
        match self.mixer {
            Mixer::Mamba(_) => GraniteMoeHybridLayerCache::Mamba(Mamba2Cache::new()),
            Mixer::Attention(_) => GraniteMoeHybridLayerCache::Attention(KVCache::new()),
        }
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut GraniteMoeHybridLayerCache,
        attn_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Mixer. `make_caches` pairs each cache kind with its mixer, so the cross
        // combinations never occur.
        let normed = self.input_layernorm.forward(x);
        let mixer_out = match (&self.mixer, cache) {
            (Mixer::Mamba(m), GraniteMoeHybridLayerCache::Mamba(mc)) => {
                m.forward(&normed, Some(mc))
            }
            (Mixer::Attention(a), GraniteMoeHybridLayerCache::Attention(kv)) => {
                a.forward(&normed, kv, attn_mask)
            }
            _ => unreachable!("Granite layer cache kind does not match its mixer"),
        };
        let mixer_out = mlxcel_core::multiply_scalar(&mixer_out, self.residual_multiplier);
        let x = mlxcel_core::add(x, &mixer_out);

        // Feed-forward.
        let normed = self.post_attention_layernorm.forward(&x);
        let ff_out = self.feed_forward.forward(&normed);
        let ff_out = mlxcel_core::multiply_scalar(&ff_out, self.residual_multiplier);
        mlxcel_core::add(&x, &ff_out)
    }
}

// Per-layer cache: Mamba2 conv/ssm state for Mamba layers, KV for attention.

pub enum GraniteMoeHybridLayerCache {
    Mamba(Mamba2Cache),
    Attention(KVCache),
}

impl GraniteMoeHybridLayerCache {
    fn offset(&self) -> i32 {
        match self {
            GraniteMoeHybridLayerCache::Mamba(_) => 0,
            GraniteMoeHybridLayerCache::Attention(kv) => kv.offset,
        }
    }

    fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        match self {
            GraniteMoeHybridLayerCache::Mamba(mamba) => {
                mamba.snapshot_into(snapshot, &format!("{prefix}.mamba"));
            }
            GraniteMoeHybridLayerCache::Attention(kv) => {
                push_optional(snapshot, format!("{prefix}.attn.keys"), &kv.keys);
                push_optional(snapshot, format!("{prefix}.attn.values"), &kv.values);
            }
        }
    }

    fn restore_from(&mut self, snapshot: &mlxcel_core::generate::ModelStateSnapshot, prefix: &str) {
        match self {
            GraniteMoeHybridLayerCache::Mamba(mamba) => {
                mamba.restore_from(snapshot, &format!("{prefix}.mamba"));
            }
            GraniteMoeHybridLayerCache::Attention(kv) => {
                kv.keys = restore_optional(snapshot, format!("{prefix}.attn.keys"));
                kv.values = restore_optional(snapshot, format!("{prefix}.attn.values"));
                kv.offset = snapshot.token_len() as i32;
            }
        }
    }
}

// Granite 4.x model.

pub struct GraniteMoeHybridModel {
    config: ModelArgs,
    embed_tokens: UnifiedEmbedding,
    layers: Vec<GraniteMoeHybridDecoderLayer>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    embedding_multiplier: f32,
    logits_scaling: f32,
    eos_token_ids: Vec<i32>,
    sequence_state: ModelOwnedSequenceState<GraniteMoeHybridLayerCache>,
}

impl GraniteMoeHybridModel {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<GraniteMoeHybridLayerCache> {
        self.layers.iter().map(|l| l.make_cache()).collect()
    }

    fn forward_with_caches(
        &self,
        inputs: &MlxArray,
        caches: &mut [GraniteMoeHybridLayerCache],
    ) -> UniquePtr<MlxArray> {
        // h = embed_tokens(x) * embedding_multiplier.
        let h = self.embed_tokens.forward(inputs);
        let mut h = mlxcel_core::multiply_scalar(&h, self.embedding_multiplier);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Single causal mask anchored on the first attention layer's offset.
        // Mamba layers stay causal via conv left-padding + the SSD `tril`, so
        // they need no mask on the single-sequence path.
        let attn_offset = caches
            .iter()
            .find(|c| matches!(c, GraniteMoeHybridLayerCache::Attention(_)))
            .map(|c| c.offset())
            .unwrap_or(0);
        let attn_mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, attn_offset))
        } else {
            None
        };

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let mask = if layer.is_attention() {
                attn_mask.as_deref()
            } else {
                None
            };
            h = layer.forward(&h, cache, mask);
        }

        let h = self.norm.forward(&h);

        // Tied embeddings unless an explicit lm_head exists, then logits /
        // logits_scaling (applies to both heads, matching the reference).
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        };
        mlxcel_core::divide_scalar(&logits, self.logits_scaling)
    }

    pub fn load(model_path: &str) -> Result<(Self, ModelArgs), String> {
        let model_dir = Path::new(model_path);
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(args.clone(), weights)?;
        Ok((model, args))
    }

    /// Build from owned weights. Applies the Granite sanitize internally so every
    /// load path (directory loader and the owned-weights route) produces the same
    /// canonical layout.
    pub fn from_weights(args: ModelArgs, weights: WeightMap) -> Result<Self, String> {
        let weights = sanitize_weights(weights, &args);

        let gs = args.group_size();
        let bits = args.bits();
        let eps = args.rms_norm_eps;

        let embed_tokens =
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", gs, bits)?;

        let intermediate = args.mamba_intermediate();
        let conv_dim = args.conv_dim();
        let attn_head_dim = args.head_dim() as i32;
        let rope_base = if args.use_rope() {
            Some(args.rope_theta)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let mixer = if args.is_mamba_layer(i) {
                let mamba_prefix = format!("{prefix}.mamba");
                let conv_weight =
                    get_weight_copy(&weights, &format!("{mamba_prefix}.conv1d.weight"))?;
                let conv_bias = weights
                    .get(&format!("{mamba_prefix}.conv1d.bias"))
                    .map(|w| mlxcel_core::copy(w));
                let in_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{mamba_prefix}.in_proj"),
                    gs,
                    bits,
                )?;
                let out_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{mamba_prefix}.out_proj"),
                    gs,
                    bits,
                )?;
                let norm = GraniteMoeHybridRMSNormGated {
                    weight: get_weight_copy(&weights, &format!("{mamba_prefix}.norm.weight"))?,
                    eps,
                    dim: intermediate as i32,
                };

                Mixer::Mamba(GraniteMoeHybridMamba2Mixer {
                    num_heads: args.mamba_n_heads,
                    ssm_state_size: args.mamba_d_state,
                    conv_kernel_size: args.mamba_d_conv,
                    intermediate_size: intermediate,
                    n_groups: args.mamba_n_groups,
                    head_dim: args.mamba_d_head,
                    time_step_limit: args.time_step_limit,
                    conv_dim,
                    conv_weight,
                    conv_bias,
                    in_proj,
                    dt_bias: get_weight_copy(&weights, &format!("{mamba_prefix}.dt_bias"))?,
                    a_log: get_weight_copy(&weights, &format!("{mamba_prefix}.A_log"))?,
                    d_param: get_weight_copy(&weights, &format!("{mamba_prefix}.D"))?,
                    norm,
                    out_proj,
                })
            } else {
                let attn_prefix = format!("{prefix}.self_attn");
                Mixer::Attention(GraniteMoeHybridAttention {
                    q_proj: UnifiedLinear::from_weights(
                        &weights,
                        &format!("{attn_prefix}.q_proj"),
                        gs,
                        bits,
                    )?,
                    k_proj: UnifiedLinear::from_weights(
                        &weights,
                        &format!("{attn_prefix}.k_proj"),
                        gs,
                        bits,
                    )?,
                    v_proj: UnifiedLinear::from_weights(
                        &weights,
                        &format!("{attn_prefix}.v_proj"),
                        gs,
                        bits,
                    )?,
                    o_proj: UnifiedLinear::from_weights(
                        &weights,
                        &format!("{attn_prefix}.o_proj"),
                        gs,
                        bits,
                    )?,
                    num_heads: args.num_attention_heads as i32,
                    num_kv_heads: args.num_key_value_heads as i32,
                    head_dim: attn_head_dim,
                    // Granite uses `attention_multiplier` as the SDPA scale.
                    scale: args.attention_multiplier,
                    rope_base,
                })
            };

            let feed_forward = build_feed_forward(&weights, &args, &prefix)?;

            let input_layernorm = RMSNorm::new(
                get_weight_copy(&weights, &format!("{prefix}.input_layernorm.weight"))?,
                eps,
            );
            let post_attention_layernorm = RMSNorm::new(
                get_weight_copy(
                    &weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                eps,
            );

            layers.push(GraniteMoeHybridDecoderLayer {
                mixer,
                feed_forward,
                input_layernorm,
                post_attention_layernorm,
                residual_multiplier: args.residual_multiplier,
            });
        }

        let norm = RMSNorm::new(get_weight_copy(&weights, "model.norm.weight")?, eps);

        let lm_head = if !args.tie_word_embeddings && weights.contains_key("lm_head.weight") {
            Some(UnifiedLinear::from_weights(&weights, "lm_head", gs, bits)?)
        } else {
            None
        };

        let internal_caches: Vec<GraniteMoeHybridLayerCache> =
            layers.iter().map(|l| l.make_cache()).collect();
        let embedding_multiplier = args.embedding_multiplier;
        let logits_scaling = args.logits_scaling;
        let eos_token_ids = args.eos_token_ids();

        Ok(Self {
            config: args,
            embed_tokens,
            layers,
            norm,
            lm_head,
            embedding_multiplier,
            logits_scaling,
            eos_token_ids,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
        })
    }
}

/// Build the per-layer feed-forward: dense `mlp`, or `block_sparse_moe` +
/// `shared_mlp` when `use_moe`.
fn build_feed_forward(
    weights: &WeightMap,
    args: &ModelArgs,
    prefix: &str,
) -> Result<FeedForward, String> {
    let gs = args.group_size();
    let bits = args.bits();

    if args.use_moe() {
        let moe = GraniteMoeHybridMoE {
            router: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.block_sparse_moe.router.layer"),
                gs,
                bits,
            )?,
            switch_mlp: SwitchGLU::from_weights(
                weights,
                &format!("{prefix}.block_sparse_moe.switch_mlp"),
                gs,
                bits,
            )?,
            top_k: args.num_experts_per_tok as i32,
            num_experts: args.num_local_experts as i32,
        };
        let shared = GraniteMoeHybridSharedMLP {
            input_linear: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.shared_mlp.input_linear"),
                gs,
                bits,
            )?,
            output_linear: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.shared_mlp.output_linear"),
                gs,
                bits,
            )?,
            shared_intermediate: args.shared_intermediate_size as i32,
        };
        Ok(FeedForward::Moe { moe, shared })
    } else {
        let mlp_prefix = format!("{prefix}.mlp");
        Ok(FeedForward::Dense(GraniteMoeHybridMLP {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{mlp_prefix}.gate_proj"),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{mlp_prefix}.up_proj"),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{mlp_prefix}.down_proj"),
                gs,
                bits,
            )?,
        }))
    }
}

// Weight sanitize (mirrors the reference `sanitize`).
//
// 1. Depthwise conv weight orientation: PyTorch stores `[conv_dim, 1, d_conv]`;
//    MLX/mlxcel wants `[conv_dim, d_conv, 1]`. The mlx-community checkpoints are
//    already MLX-layout (`shape[-1] == 1`), so this is a no-op there but keeps
//    raw HF exports loadable.
// 2. MoE: split `block_sparse_moe.input_linear` `[E, 2*ffn, hidden]` into
//    `switch_mlp.gate_proj` / `up_proj` (along axis 1) and rename `output_linear`
//    -> `switch_mlp.down_proj`. The mlx-community 4-bit MoE checkpoint may already
//    ship the stacked `switch_mlp.*`, so this probes and skips when present.
//    Quantized `input_linear` is never sliced (slicing a quantized tensor is
//    unsound); a pre-stacked checkpoint is required for quantized MoE.
// 3. Dense: rename `shared_mlp.input_linear` -> `mlp.gate_proj`/`up_proj` (split)
//    and `shared_mlp.output_linear` -> `mlp.down_proj`. The mlx-community dense
//    checkpoint already ships `mlp.*` (the converter ran this), so it is a no-op;
//    it only fires for a raw, non-quantized HF export.
fn sanitize_weights(mut weights: WeightMap, args: &ModelArgs) -> WeightMap {
    // 1. Conv weight orientation.
    let conv_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("conv1d.weight"))
        .cloned()
        .collect();
    for k in conv_keys {
        if let Some(v) = weights.get(&k) {
            let shape = mlxcel_core::array_shape(v);
            if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                let transposed = mlxcel_core::swap_axes(v, -1, -2);
                weights.insert(k, transposed);
            }
        }
    }

    if args.use_moe() {
        // 2. MoE input_linear split -> switch_mlp gate/up; output_linear -> down.
        for l in 0..args.num_hidden_layers {
            let moe_prefix = format!("model.layers.{l}.block_sparse_moe");
            // Already stacked (mlx-community layout): nothing to do.
            if weights.contains_key(&format!("{moe_prefix}.switch_mlp.gate_proj.weight")) {
                continue;
            }
            let input_key = format!("{moe_prefix}.input_linear.weight");
            // Only split a present, non-quantized input_linear (slicing a
            // quantized tensor is unsound; pre-stacked is required for quant MoE).
            if weights.contains_key(&input_key)
                && !weights.contains_key(&format!("{moe_prefix}.input_linear.scales"))
                && let Some(input_weight) = weights.remove(&input_key)
            {
                // input_linear is [E, 2*expert_hidden, hidden]; split dim 1 in half.
                let shape = mlxcel_core::array_shape(&input_weight);
                let half = shape[1] / 2;
                let gate = slice_axis(&input_weight, 1, 0, half);
                let up = slice_axis(&input_weight, 1, half, shape[1]);
                weights.insert(format!("{moe_prefix}.switch_mlp.gate_proj.weight"), gate);
                weights.insert(format!("{moe_prefix}.switch_mlp.up_proj.weight"), up);
            }
            if let Some(out) = weights.remove(&format!("{moe_prefix}.output_linear.weight")) {
                weights.insert(format!("{moe_prefix}.switch_mlp.down_proj.weight"), out);
            }
        }
    } else {
        // 3. Dense shared_mlp.input_linear -> mlp.gate/up; output_linear -> down.
        for l in 0..args.num_hidden_layers {
            let layer_prefix = format!("model.layers.{l}");
            // Already converted (mlx-community layout): nothing to do.
            if weights.contains_key(&format!("{layer_prefix}.mlp.gate_proj.weight")) {
                continue;
            }
            let input_key = format!("{layer_prefix}.shared_mlp.input_linear.weight");
            if weights.contains_key(&input_key)
                && !weights.contains_key(&format!("{layer_prefix}.shared_mlp.input_linear.scales"))
                && let Some(input_weight) = weights.remove(&input_key)
            {
                let shape = mlxcel_core::array_shape(&input_weight);
                let half = shape[0] / 2;
                let gate = slice_axis(&input_weight, 0, 0, half);
                let up = slice_axis(&input_weight, 0, half, 2 * half);
                weights.insert(format!("{layer_prefix}.mlp.gate_proj.weight"), gate);
                weights.insert(format!("{layer_prefix}.mlp.up_proj.weight"), up);
            }
            if let Some(out) =
                weights.remove(&format!("{layer_prefix}.shared_mlp.output_linear.weight"))
            {
                weights.insert(format!("{layer_prefix}.mlp.down_proj.weight"), out);
            }
        }
    }

    weights
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

// LanguageModel trait implementation.

impl LanguageModel for GraniteMoeHybridModel {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Granite owns its mixed cache (Mamba2 conv/ssm state + KVCache); the
        // external KV slice is unused. The fallback internal state covers
        // CLI / benchmark paths.
        self.sequence_state
            .with_sequence_state(None, |internal| self.forward_with_caches(input, internal))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state
            .replace_internal(GraniteMoeHybridModel::make_caches(self));
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn supports_batching(&self) -> bool {
        false // Recurrent Mamba2 state is not compatible with per-sequence KV isolation.
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt the Mamba2 conv/ssm recurrent state.
    }

    fn prepare_sequence_state(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, GraniteMoeHybridModel::make_caches(self));
    }

    fn release_sequence_state_by_id(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || GraniteMoeHybridModel::make_caches(self),
            |internal| self.forward_with_caches(input_ids, internal),
        )
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        token_len: usize,
    ) -> Option<mlxcel_core::generate::ModelStateSnapshot> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot =
                    mlxcel_core::generate::ModelStateSnapshot::new("granitemoehybrid", token_len);
                for (idx, cache) in state.iter().enumerate() {
                    cache.snapshot_into(&mut snapshot, &format!("layer{idx}"));
                }
                snapshot
            })
    }

    fn restore_sequence_state(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != "granitemoehybrid" {
            return Err(format!(
                "cannot restore {} snapshot into GraniteMoeHybrid",
                snapshot.family()
            ));
        }
        let mut state = GraniteMoeHybridModel::make_caches(self);
        for (idx, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{idx}"));
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    fn trim_internal_caches(&self, excess: i32) {
        if excess <= 0 {
            return;
        }
        self.sequence_state.with_sequence_state(None, |internal| {
            for cache in internal.iter_mut() {
                match cache {
                    GraniteMoeHybridLayerCache::Attention(kv) => {
                        kv.trim(excess);
                    }
                    GraniteMoeHybridLayerCache::Mamba(mamba) => {
                        // Recurrent conv/ssm state (computed from padding tokens)
                        // is reset rather than trimmed positionally.
                        mamba.conv_state = None;
                        mamba.ssm_state = None;
                    }
                }
            }
        });
    }
}
