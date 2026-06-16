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

//! TII Falcon-H1: parallel Mamba2 (SSM) + attention hybrid.
//!
//! Each [`FalconH1DecoderLayer`] runs a Mamba2 mixer and a standard GQA
//! attention block **in parallel** on the same normed input, then sums the two
//! outputs with the residual:
//!
//! ```text
//! residual = h
//! h       = input_layernorm(h)
//! h       = residual + mamba(h) + self_attn(h)   // both mixers, one normed input
//! residual = h
//! h       = residual + feed_forward(pre_ff_layernorm(h))   // SwiGLU MLP
//! ```
//!
//! The Mamba2 mixer is structurally the Nemotron-H / Mamba2 mixer
//! (`in_proj` → split `gate, conv_input, dt` → depthwise `conv1d` + SiLU →
//! split `hidden, B, C` → SSD scan → gate → `out_proj`). The one Falcon-H1
//! difference is the post-scan gate: `mamba_rms_norm` defaults to `false`, so
//! the scan output is gated by plain SwiGLU (`silu(gate) * y`) rather than the
//! gated RMSNorm. The gated-RMSNorm path is only taken when `mamba_rms_norm`
//! is set in the config.
//!
//! Mirrored from the mlx-lm reference:
//! - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/falcon_h1.py
//!
//! ## MUP multipliers are pre-folded in the MLX checkpoint
//!
//! Falcon-H1 carries many MUP channel multipliers (embedding, lm_head,
//! attention in/out, key, ssm in/out, mlp, and a per-channel `mup_vector`
//! folded into `in_proj`). Upstream `sanitize()` folds these into the weights
//! **only for raw HF exports**, and early-returns doing nothing once the
//! checkpoint is already in MLX channels-last `conv1d` layout
//! (`conv1d.weight.shape[-1] <= shape[1]`). The mlx-community checkpoints are
//! already MLX-layout, so every multiplier is already baked into the stored
//! weights during conversion. This loader therefore folds nothing; it only
//! applies the one runtime factor the reference applies for tied embeddings:
//! `logits = embed_tokens.as_linear(h) * (lm_head_multiplier / embedding_multiplier)`.
//! The embedding forward uses the pre-scaled `embed_tokens` weight directly.
//!
//! Because the Mamba2 conv/SSM state is recurrent (not per-token positional),
//! the model owns a mixed per-layer cache (Mamba2 conv+ssm state **and** a
//! `KVCache`, both present in every layer) through [`ModelOwnedSequenceState`]
//! and reports `supports_batching() == false` (like Jamba / Nemotron-H / LFM2).

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
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    // Mamba2 mixer dimensions.
    #[serde(default = "default_d_conv")]
    pub mamba_d_conv: usize,
    pub mamba_d_ssm: usize,
    #[serde(default = "default_d_state")]
    pub mamba_d_state: usize,
    #[serde(default = "default_n_heads")]
    pub mamba_n_heads: usize,
    #[serde(default = "default_n_groups")]
    pub mamba_n_groups: usize,
    /// Per-head SSM dimension. Falls back to `mamba_d_ssm / mamba_n_heads`.
    #[serde(default)]
    pub mamba_d_head: Option<usize>,
    #[serde(default = "default_true")]
    pub mamba_conv_bias: bool,
    #[serde(default)]
    pub mamba_proj_bias: bool,
    /// When `false` (the Falcon-H1 default), the scan output is gated by plain
    /// SwiGLU. When `true`, the gated RMSNorm path is used instead.
    #[serde(default)]
    pub mamba_rms_norm: bool,
    #[serde(default)]
    pub mamba_norm_before_gate: bool,

    // MUP multipliers. Pre-folded into the MLX weights; only the tied-head
    // ratio (`lm_head_multiplier / embedding_multiplier`) is applied at runtime.
    #[serde(default = "default_one")]
    pub embedding_multiplier: f32,
    #[serde(default = "default_one")]
    pub lm_head_multiplier: f32,

    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub projectors_bias: bool,

    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_head_dim() -> usize {
    64
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    100_000_000_000.0
}
fn default_d_conv() -> usize {
    4
}
fn default_d_state() -> usize {
    128
}
fn default_n_heads() -> usize {
    24
}
fn default_n_groups() -> usize {
    1
}
fn default_true() -> bool {
    true
}
fn default_one() -> f32 {
    1.0
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

    /// Per-head SSM dimension. Mirrors the reference `mamba_d_head`, falling
    /// back to `mamba_d_ssm / mamba_n_heads` when absent so that
    /// `mamba_head_dim * mamba_n_heads == mamba_d_ssm`.
    pub fn mamba_head_dim(&self) -> usize {
        self.mamba_d_head
            .unwrap_or(self.mamba_d_ssm / self.mamba_n_heads)
    }

    /// `conv_dim = d_ssm + 2 * n_groups * d_state` (the depthwise conv channels
    /// span the SSM hidden states plus the `B` and `C` projections).
    pub fn conv_dim(&self) -> usize {
        self.mamba_d_ssm + 2 * self.mamba_n_groups * self.mamba_d_state
    }

    /// `in_proj` output width: `gate (d_ssm) + conv_input (conv_dim) + dt (n_heads)`.
    pub fn projection_size(&self) -> usize {
        self.mamba_d_ssm + self.conv_dim() + self.mamba_n_heads
    }

    /// Number of trailing time steps retained as the conv state (`d_conv - 1`).
    pub fn conv_state_len(&self) -> usize {
        self.mamba_d_conv.saturating_sub(1)
    }

    /// Runtime logit scale for the tied head:
    /// `embed_tokens.as_linear(h) * (lm_head_multiplier / embedding_multiplier)`.
    /// The reference applies this only in the tied path; `1.0` is a safe no-op
    /// when both multipliers are `1.0` (or absent).
    pub fn tied_head_factor(&self) -> f32 {
        self.lm_head_multiplier / self.embedding_multiplier
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        // Falcon-H1 checkpoints list `[228, 11]`; fall back to 11 when absent.
        super::mamba::parse_eos_token_ids(&self.eos_token_id, 11)
    }
}

// Gated RMSNorm (only used when `mamba_rms_norm == true`).
//
// Promotes to float32 for the whole gated-norm computation, matching the
// Nemotron-H mixer: float16/bf16 RMS-norm (x^2 sum) and mixed-dtype multiply
// can overflow to NaN on M5 Max (Metal GPU Family 4) NAx kernels.
struct MambaRMSNormGated {
    weight: UniquePtr<MlxArray>,
    eps: f32,
    group_size: usize,
    /// When false (the Falcon-H1 default), the SwiGLU gate is applied to the
    /// input before the RMSNorm; when true, after the full norm (weight
    /// included). Mirrors the reference `FalconH1RMSNormGated.norm_before_gate`.
    norm_before_gate: bool,
}

impl MambaRMSNormGated {
    fn forward(&self, x: &MlxArray, gate: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_dtype = mlxcel_core::array_dtype(x);

        let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let g_f32 = mlxcel_core::astype(gate, mlxcel_core::dtype::FLOAT32);
        let silu_gate = silu(&g_f32);

        // norm_before_gate == false: gate the input before normalizing.
        let x = if self.norm_before_gate {
            x
        } else {
            mlxcel_core::multiply(&x, &silu_gate)
        };

        let shape = mlxcel_core::array_shape(&x);
        let ndim = shape.len();
        let last_dim = shape[ndim - 1] as usize;
        let num_groups = last_dim / self.group_size;

        let mut new_shape: Vec<i32> = shape[..ndim - 1].to_vec();
        new_shape.push(num_groups as i32);
        new_shape.push(self.group_size as i32);
        let x_grouped = mlxcel_core::reshape(&x, &new_shape);

        let group_weight =
            mlxcel_core::ones(&[self.group_size as i32], mlxcel_core::dtype::FLOAT32);
        let x_normed = mlxcel_core::fast_rms_norm(&x_grouped, &group_weight, self.eps);

        let x_flat = mlxcel_core::reshape(&x_normed, &shape);
        let w_f32 = mlxcel_core::astype(&self.weight, mlxcel_core::dtype::FLOAT32);
        let result = mlxcel_core::multiply(&w_f32, &x_flat);

        // norm_before_gate == true: gate after the full RMSNorm (weight included).
        let result = if self.norm_before_gate {
            mlxcel_core::multiply(&result, &silu_gate)
        } else {
            result
        };
        mlxcel_core::astype(&result, orig_dtype)
    }
}

// Mamba2 mixer (parallel SSM branch of the hybrid block).

struct FalconH1Mixer {
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
    /// Gated RMSNorm, present only when `mamba_rms_norm == true`. The Falcon-H1
    /// default (`false`) gates with plain SwiGLU instead.
    norm: Option<MambaRMSNormGated>,
    out_proj: UnifiedLinear,
}

impl FalconH1Mixer {
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
        let seq_len = shape[1];
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

        // Gate the scan output. Falcon-H1 default (`mamba_rms_norm == false`):
        // SwiGLU `silu(gate) * y`. Otherwise the gated RMSNorm.
        let y_gated = match &self.norm {
            Some(norm) => norm.forward(&y, &gate),
            None => mlxcel_core::compiled_swiglu_activation(&gate, &y),
        };

        let result = self.out_proj.forward(&y_gated);

        // Materialize at a clean dtype boundary ONLY on M5 Max (Metal GPU
        // Family 4): there the lazy float32/float16 SSM graph can fuse into NaN
        // within one Metal command buffer. On every other chip this per-layer
        // sync is pure decode-throughput loss (it blocks cross-layer
        // pipelining), so skip it. CLAUDE.md "Apple Silicon precision".
        if mlxcel_core::hardware::is_m5_neural_accelerator() {
            mlxcel_core::eval(&result);
        }
        result
    }

    /// Float32-hardened SSD-SSM scan (prefill and first-token path).
    ///
    /// Promotes `x`, `B`, `C`, and `dt` to float32 before the matmuls: the dot
    /// products over `ssm_state_size` can overflow float16 (65504), and mixed
    /// float32×float16 multiplies NaN on M5 Max (Metal GPU Family 4) NAx
    /// kernels. Mirrors the Nemotron-H mixer scan.
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

// Attention (standard GQA + RoPE, no Q/K norm, no bias by default).

struct FalconH1Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_base: f32,
}

impl FalconH1Attention {
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

        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let attn_out = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
            )
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }
}

// SwiGLU MLP feed-forward.

struct FalconH1MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl FalconH1MLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

// Decoder layer: parallel mamba + attention, then SwiGLU MLP.

struct FalconH1DecoderLayer {
    mamba: FalconH1Mixer,
    self_attn: FalconH1Attention,
    feed_forward: FalconH1MLP,
    input_layernorm: RMSNorm,
    pre_ff_layernorm: RMSNorm,
}

impl FalconH1DecoderLayer {
    fn forward(
        &self,
        h: &MlxArray,
        cache: &mut FalconH1LayerCache,
        attn_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Both mixers consume the SAME normed input; their outputs are summed
        // with the residual (the defining Falcon-H1 parallel-hybrid step).
        let normed = self.input_layernorm.forward(h);
        let mamba_h = self.mamba.forward(&normed, Some(&mut cache.mamba));
        let attn_h = self.self_attn.forward(&normed, &mut cache.attn, attn_mask);

        let h = mlxcel_core::add(h, &mamba_h);
        let h = mlxcel_core::add(&h, &attn_h);

        let normed = self.pre_ff_layernorm.forward(&h);
        let ff = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &ff)
    }
}

// Per-layer cache: BOTH a Mamba2 conv/ssm state and a KV cache (always paired).

pub struct FalconH1LayerCache {
    pub mamba: Mamba2Cache,
    pub attn: KVCache,
}

impl FalconH1LayerCache {
    fn new() -> Self {
        Self {
            mamba: Mamba2Cache::new(),
            attn: KVCache::new(),
        }
    }

    fn offset(&self) -> i32 {
        self.attn.offset
    }

    fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        self.mamba
            .snapshot_into(snapshot, &format!("{prefix}.mamba"));
        push_optional(snapshot, format!("{prefix}.attn.keys"), &self.attn.keys);
        push_optional(snapshot, format!("{prefix}.attn.values"), &self.attn.values);
    }

    fn restore_from(&mut self, snapshot: &mlxcel_core::generate::ModelStateSnapshot, prefix: &str) {
        self.mamba
            .restore_from(snapshot, &format!("{prefix}.mamba"));
        self.attn.keys = restore_optional(snapshot, format!("{prefix}.attn.keys"));
        self.attn.values = restore_optional(snapshot, format!("{prefix}.attn.values"));
        self.attn.offset = snapshot.token_len() as i32;
    }
}

// Falcon-H1 model.

pub struct FalconH1Model {
    config: ModelArgs,
    embed_tokens: UnifiedEmbedding,
    layers: Vec<FalconH1DecoderLayer>,
    final_layernorm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    tied_head_factor: f32,
    eos_token_ids: Vec<i32>,
    sequence_state: ModelOwnedSequenceState<FalconH1LayerCache>,
}

impl FalconH1Model {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<FalconH1LayerCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| FalconH1LayerCache::new())
            .collect()
    }

    fn forward_with_caches(
        &self,
        inputs: &MlxArray,
        caches: &mut [FalconH1LayerCache],
    ) -> UniquePtr<MlxArray> {
        // Pre-scaled embedding: the embedding_multiplier is already folded into
        // the MLX weights, so no runtime embedding scale is applied here.
        let mut h = self.embed_tokens.forward(inputs);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Single causal attention mask anchored on the KV offset. The conv
        // branch stays causal via left padding, so no separate mamba mask is
        // needed on the single-sequence (unpadded) path.
        let attn_offset = caches.first().map(|c| c.offset()).unwrap_or(0);
        let attn_mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, attn_offset))
        } else {
            None
        };

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, cache, attn_mask.as_deref());
        }

        let h = self.final_layernorm.forward(&h);

        // Tied head: embed_tokens.as_linear(h) * (lm_head_mult / embedding_mult).
        // Untied head: lm_head with no extra factor.
        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            let logits = self.embed_tokens.as_linear(&h);
            if (self.tied_head_factor - 1.0).abs() > f32::EPSILON {
                mlxcel_core::multiply_scalar(&logits, self.tied_head_factor)
            } else {
                logits
            }
        }
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

    /// Build from owned weights. Applies the Falcon-H1 sanitize internally so
    /// every load path (directory loader and the owned-weights route) produces
    /// the same canonical layout.
    pub fn from_weights(args: ModelArgs, weights: WeightMap) -> Result<Self, String> {
        let weights = sanitize_weights(weights);

        let gs = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", gs, bits)?;

        let intermediate_size = args.mamba_d_ssm;
        let conv_dim = args.conv_dim();
        let head_dim = args.head_dim as i32;
        let mamba_head_dim = args.mamba_head_dim();
        let mamba_group_size = intermediate_size / args.mamba_n_groups;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let mamba_prefix = format!("{prefix}.mamba");

            // Mamba2 mixer.
            let conv_weight = get_weight_copy(&weights, &format!("{mamba_prefix}.conv1d.weight"))?;
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
            let dt_bias = get_weight_copy(&weights, &format!("{mamba_prefix}.dt_bias"))?;
            let a_log = get_weight_copy(&weights, &format!("{mamba_prefix}.A_log"))?;
            let d_param = get_weight_copy(&weights, &format!("{mamba_prefix}.D"))?;

            let norm = if args.mamba_rms_norm {
                Some(MambaRMSNormGated {
                    weight: get_weight_copy(&weights, &format!("{mamba_prefix}.norm.weight"))?,
                    eps: args.rms_norm_eps,
                    group_size: mamba_group_size,
                    norm_before_gate: args.mamba_norm_before_gate,
                })
            } else {
                None
            };

            let mamba = FalconH1Mixer {
                num_heads: args.mamba_n_heads,
                ssm_state_size: args.mamba_d_state,
                conv_kernel_size: args.mamba_d_conv,
                intermediate_size,
                n_groups: args.mamba_n_groups,
                head_dim: mamba_head_dim,
                // Falcon-H1 fixes the SSM time-step limit to (0, +inf).
                time_step_limit: (0.0, f32::INFINITY),
                conv_dim,
                conv_weight,
                conv_bias,
                in_proj,
                dt_bias,
                a_log,
                d_param,
                norm,
                out_proj,
            };

            // Attention.
            let attn_prefix = format!("{prefix}.self_attn");
            let q_proj =
                UnifiedLinear::from_weights(&weights, &format!("{attn_prefix}.q_proj"), gs, bits)?;
            let k_proj =
                UnifiedLinear::from_weights(&weights, &format!("{attn_prefix}.k_proj"), gs, bits)?;
            let v_proj =
                UnifiedLinear::from_weights(&weights, &format!("{attn_prefix}.v_proj"), gs, bits)?;
            let o_proj =
                UnifiedLinear::from_weights(&weights, &format!("{attn_prefix}.o_proj"), gs, bits)?;
            let self_attn = FalconH1Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                num_heads: args.num_attention_heads as i32,
                num_kv_heads: args.num_key_value_heads as i32,
                head_dim,
                scale: (args.head_dim as f32).powf(-0.5),
                rope_base: args.rope_theta,
            };

            // SwiGLU MLP.
            let ff_prefix = format!("{prefix}.feed_forward");
            let feed_forward = FalconH1MLP {
                gate_proj: UnifiedLinear::from_weights(
                    &weights,
                    &format!("{ff_prefix}.gate_proj"),
                    gs,
                    bits,
                )?,
                up_proj: UnifiedLinear::from_weights(
                    &weights,
                    &format!("{ff_prefix}.up_proj"),
                    gs,
                    bits,
                )?,
                down_proj: UnifiedLinear::from_weights(
                    &weights,
                    &format!("{ff_prefix}.down_proj"),
                    gs,
                    bits,
                )?,
            };

            let input_layernorm = RMSNorm::new(
                get_weight_copy(&weights, &format!("{prefix}.input_layernorm.weight"))?,
                args.rms_norm_eps,
            );
            let pre_ff_layernorm = RMSNorm::new(
                get_weight_copy(&weights, &format!("{prefix}.pre_ff_layernorm.weight"))?,
                args.rms_norm_eps,
            );

            layers.push(FalconH1DecoderLayer {
                mamba,
                self_attn,
                feed_forward,
                input_layernorm,
                pre_ff_layernorm,
            });
        }

        let final_layernorm = RMSNorm::new(
            get_weight_copy(&weights, "model.final_layernorm.weight")?,
            args.rms_norm_eps,
        );

        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(&weights, "lm_head", gs, bits)?)
        };

        let internal_caches: Vec<FalconH1LayerCache> = (0..args.num_hidden_layers)
            .map(|_| FalconH1LayerCache::new())
            .collect();
        let tied_head_factor = args.tied_head_factor();
        let eos_token_ids = args.eos_token_ids();

        Ok(Self {
            config: args,
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
            tied_head_factor,
            eos_token_ids,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
        })
    }
}

// Weight sanitize.
//
// Depthwise conv weight orientation only. PyTorch stores `conv1d.weight` as
// `[conv_dim, 1, d_conv]`; MLX/mlxcel wants `[conv_dim, d_conv, 1]`. The
// mlx-community checkpoints are already MLX-layout (`shape[-1] == 1`), so this
// is a no-op there but keeps raw HF exports loadable.
//
// No MUP multiplier is folded here: the mlx-community conversion already baked
// every multiplier into the stored weights (the reference `sanitize()`
// early-returns once the checkpoint is in MLX `conv1d` layout). See the module
// docs. Folding into already-quantized weights would be unsound anyway.
fn sanitize_weights(mut weights: WeightMap) -> WeightMap {
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
    weights
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

// LanguageModel trait implementation.

impl LanguageModel for FalconH1Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Falcon-H1 owns its mixed cache (Mamba2 conv/ssm state + KVCache); the
        // external KV slice is unused. The fallback internal state covers
        // CLI / benchmark paths.
        self.sequence_state
            .with_sequence_state(None, |internal| self.forward_with_caches(input, internal))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state
            .replace_internal(FalconH1Model::make_caches(self));
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
            .prepare_sequence_state(seq_id, FalconH1Model::make_caches(self));
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
            || FalconH1Model::make_caches(self),
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
                    mlxcel_core::generate::ModelStateSnapshot::new("falcon_h1", token_len);
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
        if snapshot.family() != "falcon_h1" {
            return Err(format!(
                "cannot restore {} snapshot into Falcon-H1",
                snapshot.family()
            ));
        }
        let mut state = FalconH1Model::make_caches(self);
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
                // KV cache trims positionally; the Mamba2 conv/ssm state is
                // recurrent (computed from padding tokens), so reset it.
                cache.attn.trim(excess);
                cache.mamba.conv_state = None;
                cache.mamba.ssm_state = None;
            }
        });
    }
}
