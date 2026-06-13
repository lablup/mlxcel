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

//! Preferred Networks PLaMo 2: interleaved Mamba (SSM) + attention hybrid.
//!
//! PLaMo 2 is a decoder where each layer is EITHER a Mamba mixer OR a GQA
//! attention mixer, chosen by index: `is_mamba(i)` is true when
//! `mamba_enabled && (i % mamba_step) != (mamba_step / 2)` (with a small-model
//! fallback that puts attention only in the last layer). For the `plamo-2-1b`
//! checkpoint (`mamba_step == 2`) this makes the even layers Mamba and the odd
//! layers attention. The either-or layout mirrors LFM2 / Nemotron-H, not the
//! parallel both-per-layer Falcon-H1 block.
//!
//! Mirrored from the mlx-lm reference:
//! - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/plamo2.py
//!
//! ## Three PLaMo 2 specific pieces
//!
//! 1. **Offset RMSNorm and normformer-style double norms.** PLaMo 2's RMSNorm is
//!    `rms_norm(x, weight + offset, eps)`: an additive offset on the learned
//!    weight. The offset is folded into the stored weight at construction time
//!    so the standard mlxcel [`RMSNorm`] (which computes `x_normed * weight`)
//!    reproduces it. Every block applies a POST norm to both the mixer and the
//!    MLP output:
//!
//!    ```text
//!    residual = h; h = pre_mixer_norm(h); h = mixer(h); h = post_mixer_norm(h); h = residual + h
//!    residual = h; h = pre_mlp_norm(h);   h = mlp(h);   h = post_mlp_norm(h);   h = residual + h
//!    ```
//!
//!    The offsets are `pre_mixer = 1.0`, `post_mixer = 1/5`, `pre_mlp = 1.0`,
//!    `post_mlp = 1/(5**1.5)`, and the final `model.norm` offset = `1.0`.
//!
//! 2. **Mamba mixer with a post-conv B/C/dt projection.** Unlike Falcon-H1 /
//!    Mamba2 (where `in_proj` already carries `B`/`C`/`dt`), PLaMo 2 splits
//!    `in_proj` into only `z` (gate) and `x`, runs `x` through the depthwise
//!    causal conv, and then derives `B`, `C`, and `dt` from a SEPARATE
//!    `bcdt_proj(conv_x)`. Each of `B`, `C`, `dt` is RMS-normed (plain norm,
//!    no offset) before `dt` is projected to per-head deltas and the SSD scan
//!    runs. The scan output is gated by `swiglu(z, y) = silu(z) * y`.
//!
//! 3. **Attention with fused qkv and weight=None Q/K norm + per-head scale.**
//!    `qkv_proj` is a single fused projection. After the head reshape, `q` and
//!    `k` are RMS-normed WITHOUT a learned weight (`weight=None`, i.e. a ones
//!    vector) and then multiplied by a per-head learned scale (`q_weight`
//!    `[num_heads, head_dim]`, `k_weight` `[num_kv_heads, head_dim]`). RoPE
//!    (base 10000, non-traditional) follows with the cache offset.
//!
//! The SSD scan, the conv-state handling, and the fused decode kernel dispatch
//! are adapted from [`super::falcon_h1`] (the shared `mlxcel_core::ssm_update_kernel`
//! path), with two PLaMo 2 differences: the `B`/`C`/`dt` inputs come from the
//! post-conv projection described above, and the SSM time-step limit is the
//! `ssm_update` default `(0.001, 100.0)` (PLaMo 2 passes no explicit limit),
//! not Falcon-H1's open `(0, +inf)`.
//!
//! Because the Mamba conv/SSM state is recurrent (not per-token positional), the
//! model owns a mixed per-layer cache (`Mamba2Cache` for Mamba layers, `KVCache`
//! for attention layers) through [`ModelOwnedSequenceState`] and reports
//! `supports_batching() == false` (like Jamba / Nemotron-H / LFM2 / Falcon-H1).

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

// Per-norm offsets (folded into the stored weight at construction time).
// Exposed to `plamo2_tests` so the exact PLaMo 2 offset values stay pinned.
pub(crate) const PRE_MIXER_NORM_OFFSET: f32 = 1.0;
pub(crate) const PRE_MLP_NORM_OFFSET: f32 = 1.0;
pub(crate) const FINAL_NORM_OFFSET: f32 = 1.0;

/// `post_mixer_norm` offset: `1.0 / 5`.
pub(crate) fn post_mixer_norm_offset() -> f32 {
    1.0 / 5.0
}

/// `post_mlp_norm` offset: `1.0 / (5 ** 1.5)`.
pub(crate) fn post_mlp_norm_offset() -> f32 {
    1.0 / 5.0_f32.powf(1.5)
}

// Configuration.

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default = "default_hidden_size_per_head")]
    pub hidden_size_per_head: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    // Mamba mixer dimensions.
    #[serde(default = "default_d_state")]
    pub mamba_d_state: usize,
    #[serde(default = "default_d_conv")]
    pub mamba_d_conv: usize,
    #[serde(default = "default_mamba_num_heads")]
    pub mamba_num_heads: usize,
    #[serde(default = "default_mamba_step")]
    pub mamba_step: usize,
    #[serde(default = "default_true")]
    pub mamba_enabled: bool,

    // `tie_word_embeddings` is absent in the `plamo-2-1b` config; default to
    // true (and the loader falls back to tied when no `lm_head.weight` exists).
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_hidden_size_per_head() -> usize {
    128
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_d_state() -> usize {
    64
}
fn default_d_conv() -> usize {
    4
}
fn default_mamba_num_heads() -> usize {
    64
}
fn default_mamba_step() -> usize {
    2
}
fn default_true() -> bool {
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

    /// Per-head dimension shared by the Mamba head split and the attention head
    /// split (`hidden_size_per_head`).
    pub fn head_dim(&self) -> usize {
        self.hidden_size_per_head
    }

    /// Mamba inner width: `mamba_num_heads * hidden_size_per_head`.
    pub fn mamba_intermediate(&self) -> usize {
        self.mamba_num_heads * self.hidden_size_per_head
    }

    /// Width of the `dt` branch out of `bcdt_proj`: `max(64, hidden_size / 16)`.
    pub fn dt_dim(&self) -> usize {
        std::cmp::max(64, self.hidden_size / 16)
    }

    /// Whether layer `i` is a Mamba layer. Mirrors the reference `is_mamba`:
    /// `mamba_enabled && (i % mamba_step) != (mamba_step / 2)`, with the
    /// small-model fallback (`num_hidden_layers <= mamba_step / 2`) that uses
    /// attention only in the last layer.
    pub fn is_mamba(&self, i: usize) -> bool {
        if !self.mamba_enabled {
            return false;
        }
        if self.num_hidden_layers <= self.mamba_step / 2 {
            return i != self.num_hidden_layers - 1;
        }
        (i % self.mamba_step) != (self.mamba_step / 2)
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        // `plamo-2-1b` uses `<|plamo:eos|>` (id 2) as EOS.
        super::mamba::parse_eos_token_ids(&self.eos_token_id, 2)
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

// PLaMo 2 Mamba mixer.

struct Plamo2Mamba {
    num_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    dt_dim: usize,
    conv_kernel_size: usize,
    n_groups: usize,
    rms_norm_eps: f32,
    time_step_limit: (f32, f32),

    in_proj: UnifiedLinear,
    conv_weight: UniquePtr<MlxArray>,
    bcdt_proj: UnifiedLinear,
    dt_proj: UnifiedLinear,
    out_proj: UnifiedLinear,

    dt_norm_weight: UniquePtr<MlxArray>,
    b_norm_weight: UniquePtr<MlxArray>,
    c_norm_weight: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    d_param: UniquePtr<MlxArray>,
    dt_bias: UniquePtr<MlxArray>,
}

impl Plamo2Mamba {
    /// Depthwise causal conv1d over the `x` branch only (groups ==
    /// `intermediate_size`), with the per-sequence conv-state cache. No bias
    /// (PLaMo 2's `conv1d` has `bias=False`).
    fn conv(&self, conv_input: &MlxArray, cache: Option<&mut Mamba2Cache>) -> UniquePtr<MlxArray> {
        let k = self.conv_kernel_size;
        let shape = mlxcel_core::array_shape(conv_input);
        let batch = shape[0];

        let conv_state_ref = cache
            .as_ref()
            .and_then(|c| c.conv_state.as_ref())
            .and_then(|s| s.as_ref());

        let padded_input = if let Some(conv_st) = conv_state_ref {
            concatenate(conv_st, conv_input, 1)
        } else {
            let conv_dtype = mlxcel_core::array_dtype(conv_input);
            let pad_arr = mlxcel_core::zeros(
                &[batch, (k - 1) as i32, self.intermediate_size as i32],
                conv_dtype,
            );
            concatenate(&pad_arr, conv_input, 1)
        };

        // Persist the trailing `k - 1` time steps as the next conv state.
        // `contiguous` forces a fresh buffer so the cached slice does not retain
        // the full `padded_input` allocation (a per-token memory leak otherwise).
        if let Some(c) = cache {
            let padded_shape = mlxcel_core::array_shape(&padded_input);
            let len = padded_shape[1] as usize;
            let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
            c.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        }

        // Match the conv weight dtype to the (possibly bf16/f16) activations.
        let in_dtype = mlxcel_core::array_dtype(&padded_input);
        let conv_out = if mlxcel_core::array_dtype(&self.conv_weight) == in_dtype {
            mlxcel_core::conv1d(
                &padded_input,
                &self.conv_weight,
                1,
                0,
                1,
                self.intermediate_size as i32,
            )
        } else {
            let w = mlxcel_core::astype(&self.conv_weight, in_dtype);
            mlxcel_core::conv1d(&padded_input, &w, 1, 0, 1, self.intermediate_size as i32)
        };

        silu(&conv_out)
    }

    fn forward(&self, h: &MlxArray, mut cache: Option<&mut Mamba2Cache>) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(h);
        let b = shape[0];
        let l = shape[1];
        let num_heads = self.num_heads as i32;
        let head_per = self.head_dim as i32;
        let intermediate = self.intermediate_size as i32;
        let d_state = self.ssm_state_size as i32;
        let dt_dim = self.dt_dim as i32;

        // in_proj -> reshape [b, l, num_heads, 2*head_dim] -> split z, x.
        let zx = self.in_proj.forward(h);
        let zx = mlxcel_core::reshape(&zx, &[b, l, num_heads, 2 * head_per]);
        let z = slice_axis(&zx, -1, 0, head_per);
        let x = slice_axis(&zx, -1, head_per, 2 * head_per);
        let z_flat = mlxcel_core::reshape(&z, &[b, l, intermediate]);
        let x_flat = mlxcel_core::reshape(&x, &[b, l, intermediate]);

        // Depthwise causal conv over x, then SiLU.
        let conv_x = self.conv(&x_flat, cache.as_deref_mut());

        // B, C, dt come from a post-conv projection (PLaMo 2 specific).
        let bcdt = self.bcdt_proj.forward(&conv_x);
        let b_raw = slice_axis(&bcdt, -1, 0, d_state);
        let c_raw = slice_axis(&bcdt, -1, d_state, 2 * d_state);
        let dt_raw = slice_axis(&bcdt, -1, 2 * d_state, 2 * d_state + dt_dim);

        // Per-branch RMSNorm (plain, no offset; weights default to ones).
        let dt_normed =
            mlxcel_core::fast_rms_norm(&dt_raw, &self.dt_norm_weight, self.rms_norm_eps);
        let b_normed = mlxcel_core::fast_rms_norm(&b_raw, &self.b_norm_weight, self.rms_norm_eps);
        let c_normed = mlxcel_core::fast_rms_norm(&c_raw, &self.c_norm_weight, self.rms_norm_eps);

        // dt_proj maps the normed dt to per-head deltas.
        let dt = self.dt_proj.forward(&dt_normed);

        // SSD-SSM scan. Fused Metal kernel for single-token decode with state;
        // graph path (float32-hardened) for prefill and the first token.
        let ssm_state_ref = cache
            .as_ref()
            .and_then(|c| c.ssm_state.as_ref())
            .and_then(|s| s.as_ref());

        let (y, new_state) = if l == 1 && mlxcel_core::ssm_kernel_available() {
            if let Some(state) = ssm_state_ref {
                self.ssm_step_kernel(&conv_x, &b_normed, &c_normed, &dt, state)
            } else {
                self.ssm_step(&conv_x, &b_normed, &c_normed, &dt, ssm_state_ref)
            }
        } else {
            self.ssm_step(&conv_x, &b_normed, &c_normed, &dt, ssm_state_ref)
        };

        if let Some(c) = cache {
            c.ssm_state = Some(new_state);
        }

        let y = mlxcel_core::reshape(&y, &[b, l, intermediate]);

        // z gate: swiglu(z, y) = silu(z) * y, then out_proj.
        let gated = mlxcel_core::compiled_swiglu_activation(&z_flat, &y);
        let result = self.out_proj.forward(&gated);

        // Materialize at a clean dtype boundary. The lazy SSM graph mixes
        // float32/float16 nodes that can fuse into NaN downstream on M5 Max.
        mlxcel_core::eval(&result);
        result
    }

    /// Float32-hardened SSD-SSM scan (prefill and first-token path).
    ///
    /// Promotes `x`, `B`, `C`, and `dt` to float32 before the matmuls: the dot
    /// products over `ssm_state_size` can overflow float16, and mixed
    /// float32xfloat16 multiplies NaN on M5 Max NAx kernels. The `dt` here is
    /// the per-head `dt_proj` output; `compute_dt` (softplus(dt + dt_bias) then
    /// clip to `time_step_limit`) runs inside this scan, matching `ssm_update`.
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
    /// applies `compute_dt` internally, so the per-head `dt` (the `dt_proj`
    /// output) and `dt_bias` are passed through alongside the time-step limits.
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

// PLaMo 2 attention (fused qkv, weight=None Q/K norm + per-head scale, RoPE, GQA).

struct Plamo2Attention {
    qkv_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_weight: UniquePtr<MlxArray>,
    k_weight: UniquePtr<MlxArray>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    q_dim: i32,
    k_dim: i32,
    v_dim: i32,
    scale: f32,
    rope_base: f32,
}

impl Plamo2Attention {
    /// RMSNorm over `head_dim` with `weight=None` (a ones vector), then multiply
    /// by the per-head learned scale `weight[:, None]` (shape
    /// `[heads, 1, head_dim]`). Input `x` is `[batch, heads, seq, head_dim]`.
    fn qk_norm_scale(&self, x: &MlxArray, weight: &MlxArray) -> UniquePtr<MlxArray> {
        let xd = mlxcel_core::array_dtype(x);
        let ones_w = mlxcel_core::ones(&[self.head_dim], xd);
        // The reference hardcodes eps = 1e-6 for the Q/K norm.
        let normed = mlxcel_core::fast_rms_norm(x, &ones_w, 1e-6);
        let w = mlxcel_core::reshape(weight, &[-1, 1, self.head_dim]);
        let w = if mlxcel_core::array_dtype(&w) == xd {
            w
        } else {
            mlxcel_core::astype(&w, xd)
        };
        mlxcel_core::multiply(&normed, &w)
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let qkv = self.qkv_proj.forward(x);
        let q = slice_axis(&qkv, -1, 0, self.q_dim);
        let k = slice_axis(&qkv, -1, self.q_dim, self.q_dim + self.k_dim);
        let v = slice_axis(
            &qkv,
            -1,
            self.q_dim + self.k_dim,
            self.q_dim + self.k_dim + self.v_dim,
        );

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // weight=None RMSNorm over head_dim, then per-head learned scale.
        let q = self.qk_norm_scale(&q, &self.q_weight);
        let k = self.qk_norm_scale(&k, &self.k_weight);

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

// SwiGLU MLP with a fused gate_up projection.

struct Plamo2MLP {
    gate_up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    intermediate_size: i32,
}

impl Plamo2MLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate_up = self.gate_up_proj.forward(x);
        let gate = slice_axis(&gate_up, -1, 0, self.intermediate_size);
        let up = slice_axis(
            &gate_up,
            -1,
            self.intermediate_size,
            2 * self.intermediate_size,
        );
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

// Per-layer mixer (Mamba OR attention).

enum Plamo2Mixer {
    Mamba(Plamo2Mamba),
    Attention(Plamo2Attention),
}

// Decoder layer: normformer-style double norm around the mixer and the MLP.

struct Plamo2DecoderLayer {
    mixer: Plamo2Mixer,
    mlp: Plamo2MLP,
    pre_mixer_norm: RMSNorm,
    post_mixer_norm: RMSNorm,
    pre_mlp_norm: RMSNorm,
    post_mlp_norm: RMSNorm,
}

impl Plamo2DecoderLayer {
    fn is_attention(&self) -> bool {
        matches!(self.mixer, Plamo2Mixer::Attention(_))
    }

    fn make_cache(&self) -> Plamo2LayerCache {
        match self.mixer {
            Plamo2Mixer::Mamba(_) => Plamo2LayerCache::Mamba(Mamba2Cache::new()),
            Plamo2Mixer::Attention(_) => Plamo2LayerCache::Attention(KVCache::new()),
        }
    }

    fn forward(
        &self,
        h: &MlxArray,
        cache: &mut Plamo2LayerCache,
        attn_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Mixer with a pre- and a post-norm (the offsets are folded into each
        // norm weight). `make_caches` pairs each cache kind with its mixer, so
        // the cross combinations never occur.
        let normed = self.pre_mixer_norm.forward(h);
        let mixer_out = match (&self.mixer, cache) {
            (Plamo2Mixer::Mamba(m), Plamo2LayerCache::Mamba(mc)) => m.forward(&normed, Some(mc)),
            (Plamo2Mixer::Attention(a), Plamo2LayerCache::Attention(kv)) => {
                a.forward(&normed, kv, attn_mask)
            }
            _ => unreachable!("PLaMo 2 layer cache kind does not match its mixer"),
        };
        let mixer_out = self.post_mixer_norm.forward(&mixer_out);
        let h = mlxcel_core::add(h, &mixer_out);

        // MLP with a pre- and a post-norm.
        let normed = self.pre_mlp_norm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        let mlp_out = self.post_mlp_norm.forward(&mlp_out);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// Per-layer cache: Mamba2 conv/ssm state for Mamba layers, KV for attention.

pub enum Plamo2LayerCache {
    Mamba(Mamba2Cache),
    Attention(KVCache),
}

impl Plamo2LayerCache {
    fn offset(&self) -> i32 {
        match self {
            Plamo2LayerCache::Mamba(_) => 0,
            Plamo2LayerCache::Attention(kv) => kv.offset,
        }
    }

    fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        match self {
            Plamo2LayerCache::Mamba(mamba) => {
                mamba.snapshot_into(snapshot, &format!("{prefix}.mamba"));
            }
            Plamo2LayerCache::Attention(kv) => {
                push_optional(snapshot, format!("{prefix}.attn.keys"), &kv.keys);
                push_optional(snapshot, format!("{prefix}.attn.values"), &kv.values);
            }
        }
    }

    fn restore_from(&mut self, snapshot: &mlxcel_core::generate::ModelStateSnapshot, prefix: &str) {
        match self {
            Plamo2LayerCache::Mamba(mamba) => {
                mamba.restore_from(snapshot, &format!("{prefix}.mamba"));
            }
            Plamo2LayerCache::Attention(kv) => {
                kv.keys = restore_optional(snapshot, format!("{prefix}.attn.keys"));
                kv.values = restore_optional(snapshot, format!("{prefix}.attn.values"));
                kv.offset = snapshot.token_len() as i32;
            }
        }
    }
}

// PLaMo 2 model.

pub struct Plamo2Model {
    config: ModelArgs,
    embed_tokens: UnifiedEmbedding,
    layers: Vec<Plamo2DecoderLayer>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    eos_token_ids: Vec<i32>,
    sequence_state: ModelOwnedSequenceState<Plamo2LayerCache>,
}

impl Plamo2Model {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<Plamo2LayerCache> {
        self.layers.iter().map(|l| l.make_cache()).collect()
    }

    fn forward_with_caches(
        &self,
        inputs: &MlxArray,
        caches: &mut [Plamo2LayerCache],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(inputs);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Single causal mask anchored on the first attention layer's offset.
        // Mamba layers stay causal via conv left-padding + the SSD `tril`, so
        // they need no mask on the single-sequence path.
        let attn_offset = caches
            .iter()
            .find(|c| matches!(c, Plamo2LayerCache::Attention(_)))
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

        // Tied embeddings unless an explicit `lm_head` was loaded. No logit
        // multiplier (PLaMo 2 applies none).
        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
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

    /// Build from owned weights. Applies the PLaMo 2 conv-orientation sanitize
    /// internally so every load path produces the same canonical layout.
    pub fn from_weights(args: ModelArgs, weights: WeightMap) -> Result<Self, String> {
        let weights = sanitize_weights(weights);

        let gs = args.group_size();
        let bits = args.bits();
        let eps = args.rms_norm_eps;

        let embed_tokens =
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", gs, bits)?;

        let num_heads = args.mamba_num_heads;
        let head_dim = args.head_dim();
        let intermediate = args.mamba_intermediate();
        let d_state = args.mamba_d_state;
        let dt_dim = args.dt_dim();

        let attn_head_dim = args.head_dim() as i32;
        let q_dim = (args.num_attention_heads * args.head_dim()) as i32;
        let k_dim = (args.num_key_value_heads * args.head_dim()) as i32;
        let post_mixer_offset = post_mixer_norm_offset();
        let post_mlp_offset = post_mlp_norm_offset();

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            // `model.layers.layers.{i}` (PlamoModel.layers is the PlamoDecoder,
            // whose `.layers` is the list of blocks).
            let prefix = format!("model.layers.layers.{i}");
            let mixer_prefix = format!("{prefix}.mixer");

            let mixer = if args.is_mamba(i) {
                let conv_weight =
                    get_weight_copy(&weights, &format!("{mixer_prefix}.conv1d.weight"))?;
                let in_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{mixer_prefix}.in_proj"),
                    gs,
                    bits,
                )?;
                let bcdt_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{mixer_prefix}.bcdt_proj"),
                    gs,
                    bits,
                )?;
                let dt_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{mixer_prefix}.dt_proj"),
                    gs,
                    bits,
                )?;
                let out_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{mixer_prefix}.out_proj"),
                    gs,
                    bits,
                )?;

                Plamo2Mixer::Mamba(Plamo2Mamba {
                    num_heads,
                    head_dim,
                    intermediate_size: intermediate,
                    ssm_state_size: d_state,
                    dt_dim,
                    conv_kernel_size: args.mamba_d_conv,
                    n_groups: 1,
                    rms_norm_eps: eps,
                    // PLaMo 2 passes no explicit time-step limit, so the
                    // `ssm_update` default `(0.001, 100.0)` applies (not the
                    // open `(0, +inf)` used by Falcon-H1).
                    time_step_limit: (0.001, 100.0),
                    in_proj,
                    conv_weight,
                    bcdt_proj,
                    dt_proj,
                    out_proj,
                    dt_norm_weight: get_weight_copy(
                        &weights,
                        &format!("{mixer_prefix}.dt_norm_weight"),
                    )?,
                    b_norm_weight: get_weight_copy(
                        &weights,
                        &format!("{mixer_prefix}.B_norm_weight"),
                    )?,
                    c_norm_weight: get_weight_copy(
                        &weights,
                        &format!("{mixer_prefix}.C_norm_weight"),
                    )?,
                    a_log: get_weight_copy(&weights, &format!("{mixer_prefix}.A_log"))?,
                    d_param: get_weight_copy(&weights, &format!("{mixer_prefix}.D"))?,
                    dt_bias: get_weight_copy(&weights, &format!("{mixer_prefix}.dt_bias"))?,
                })
            } else {
                let qkv_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{mixer_prefix}.qkv_proj"),
                    gs,
                    bits,
                )?;
                let o_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{mixer_prefix}.o_proj"),
                    gs,
                    bits,
                )?;
                Plamo2Mixer::Attention(Plamo2Attention {
                    qkv_proj,
                    o_proj,
                    q_weight: get_weight_copy(&weights, &format!("{mixer_prefix}.q_weight"))?,
                    k_weight: get_weight_copy(&weights, &format!("{mixer_prefix}.k_weight"))?,
                    num_heads: args.num_attention_heads as i32,
                    num_kv_heads: args.num_key_value_heads as i32,
                    head_dim: attn_head_dim,
                    q_dim,
                    k_dim,
                    v_dim: k_dim,
                    scale: (args.head_dim() as f32).powf(-0.5),
                    rope_base: 10000.0,
                })
            };

            let mlp = Plamo2MLP {
                gate_up_proj: UnifiedLinear::from_weights(
                    &weights,
                    &format!("{prefix}.mlp.gate_up_proj"),
                    gs,
                    bits,
                )?,
                down_proj: UnifiedLinear::from_weights(
                    &weights,
                    &format!("{prefix}.mlp.down_proj"),
                    gs,
                    bits,
                )?,
                intermediate_size: args.intermediate_size as i32,
            };

            let pre_mixer_norm = offset_rms_norm(
                &weights,
                &format!("{prefix}.pre_mixer_norm.weight"),
                eps,
                PRE_MIXER_NORM_OFFSET,
            )?;
            let post_mixer_norm = offset_rms_norm(
                &weights,
                &format!("{prefix}.post_mixer_norm.weight"),
                eps,
                post_mixer_offset,
            )?;
            let pre_mlp_norm = offset_rms_norm(
                &weights,
                &format!("{prefix}.pre_mlp_norm.weight"),
                eps,
                PRE_MLP_NORM_OFFSET,
            )?;
            let post_mlp_norm = offset_rms_norm(
                &weights,
                &format!("{prefix}.post_mlp_norm.weight"),
                eps,
                post_mlp_offset,
            )?;

            layers.push(Plamo2DecoderLayer {
                mixer,
                mlp,
                pre_mixer_norm,
                post_mixer_norm,
                pre_mlp_norm,
                post_mlp_norm,
            });
        }

        let norm = offset_rms_norm(&weights, "model.norm.weight", eps, FINAL_NORM_OFFSET)?;

        // Tied embeddings unless the checkpoint ships an explicit `lm_head`.
        let lm_head = if !args.tie_word_embeddings && weights.contains_key("lm_head.weight") {
            Some(UnifiedLinear::from_weights(&weights, "lm_head", gs, bits)?)
        } else {
            None
        };

        let internal_caches: Vec<Plamo2LayerCache> =
            layers.iter().map(|l| l.make_cache()).collect();
        let eos_token_ids = args.eos_token_ids();

        Ok(Self {
            config: args,
            embed_tokens,
            layers,
            norm,
            lm_head,
            eos_token_ids,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
        })
    }
}

// Weight sanitize.
//
// Depthwise conv weight orientation only. PyTorch stores `conv1d.weight` as
// `[channels, 1, d_conv]`; MLX/mlxcel wants `[channels, d_conv, 1]`. The
// `plamo-2-1b` checkpoint is already MLX-layout (`shape[-1] == 1`), so this is a
// no-op there but keeps raw HF exports loadable.
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

/// Load a norm weight and fold the additive offset into it so the standard
/// [`RMSNorm`] (`x_normed * weight`) reproduces `rms_norm(x, weight + offset)`.
fn offset_rms_norm(
    weights: &WeightMap,
    name: &str,
    eps: f32,
    offset: f32,
) -> Result<RMSNorm, String> {
    let weight = get_weight_copy(weights, name)?;
    let dtype = mlxcel_core::array_dtype(&weight);
    let offset_arr = mlxcel_core::full_f32(&[1], offset, dtype);
    let folded = mlxcel_core::add(&weight, &offset_arr);
    Ok(RMSNorm::new(folded, eps))
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

// LanguageModel trait implementation.

impl LanguageModel for Plamo2Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // PLaMo 2 owns its mixed cache (Mamba2 conv/ssm state + KVCache); the
        // external KV slice is unused. The fallback internal state covers
        // CLI / benchmark paths.
        self.sequence_state
            .with_sequence_state(None, |internal| self.forward_with_caches(input, internal))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state
            .replace_internal(Plamo2Model::make_caches(self));
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
        false // Recurrent Mamba state is not compatible with per-sequence KV isolation.
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt the Mamba conv/ssm recurrent state.
    }

    fn prepare_sequence_state(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, Plamo2Model::make_caches(self));
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
            || Plamo2Model::make_caches(self),
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
                    mlxcel_core::generate::ModelStateSnapshot::new("plamo2", token_len);
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
        if snapshot.family() != "plamo2" {
            return Err(format!(
                "cannot restore {} snapshot into PLaMo 2",
                snapshot.family()
            ));
        }
        let mut state = Plamo2Model::make_caches(self);
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
                    Plamo2LayerCache::Attention(kv) => {
                        kv.trim(excess);
                    }
                    Plamo2LayerCache::Mamba(mamba) => {
                        // Recurrent conv/ssm state (computed from padding
                        // tokens) is reset rather than trimmed positionally.
                        mamba.conv_state = None;
                        mamba.ssm_state = None;
                    }
                }
            }
        });
    }
}
