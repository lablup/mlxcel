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

// Mamba2: Multi-head SSM-based architecture for mlxcel-core
// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba2.py

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{silu, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

use super::model_owned::ModelOwnedSequenceState;
use super::recurrent_snapshot::{push_optional, restore_optional};

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mamba2Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_expand")]
    pub expand: usize,
    pub state_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_epsilon: f32,
    #[serde(alias = "d_conv")]
    pub conv_kernel: usize,
    pub n_groups: usize,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_true")]
    pub use_conv_bias: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(
        default = "default_time_step_limit",
        deserialize_with = "deserialize_time_step_limit"
    )]
    pub time_step_limit: (f32, f32),
    #[serde(default, deserialize_with = "deserialize_time_step_rank")]
    pub time_step_rank: usize,
    #[serde(default)]
    pub ssm_state_size: Option<usize>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl Mamba2Config {
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

fn default_true() -> bool {
    true
}
fn default_expand() -> usize {
    2
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_time_step_limit() -> (f32, f32) {
    (0.0, f32::INFINITY)
}

fn deserialize_time_step_limit<'de, D>(deserializer: D) -> Result<(f32, f32), D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{SeqAccess, Visitor};

    struct TimeStepLimitVisitor;

    impl<'de> Visitor<'de> for TimeStepLimitVisitor {
        type Value = (f32, f32);

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a tuple of two numbers")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let first: f32 = seq.next_element()?.unwrap_or(0.0);
            let second: f32 = seq
                .next_element::<serde_json::Value>()?
                .map(|v| match v {
                    serde_json::Value::Number(n) => n.as_f64().unwrap_or(f64::INFINITY) as f32,
                    _ => f32::INFINITY,
                })
                .unwrap_or(f32::INFINITY);
            Ok((first, second))
        }
    }

    deserializer.deserialize_seq(TimeStepLimitVisitor)
}

fn deserialize_time_step_rank<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TimeStepRank {
        Number(usize),
        String(String),
    }

    match TimeStepRank::deserialize(deserializer)? {
        TimeStepRank::Number(n) => Ok(n),
        TimeStepRank::String(s) if s == "auto" => Ok(0),
        TimeStepRank::String(s) => Err(D::Error::custom(format!("invalid time_step_rank: {}", s))),
    }
}

impl Mamba2Config {
    pub fn compute_time_step_rank(&mut self) {
        if self.time_step_rank == 0 {
            self.time_step_rank = self.hidden_size.div_ceil(16);
        }
        if self.ssm_state_size.is_none() {
            self.ssm_state_size = Some(self.state_size);
        }
    }

    pub fn get_ssm_state_size(&self) -> usize {
        self.ssm_state_size.unwrap_or(self.state_size)
    }

    #[allow(dead_code)]
    pub fn get_intermediate_size(&self) -> usize {
        self.intermediate_size
            .unwrap_or(self.hidden_size * self.expand)
    }
}

// SSM Cache for Mamba2.
pub struct Mamba2Cache {
    pub conv_state: Option<UniquePtr<MlxArray>>,
    pub ssm_state: Option<UniquePtr<MlxArray>>,
}

impl Mamba2Cache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
        }
    }

    pub fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        push_optional(snapshot, format!("{prefix}.conv_state"), &self.conv_state);
        push_optional(snapshot, format!("{prefix}.ssm_state"), &self.ssm_state);
    }

    pub fn restore_from(
        &mut self,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        self.conv_state = restore_optional(snapshot, format!("{prefix}.conv_state"));
        self.ssm_state = restore_optional(snapshot, format!("{prefix}.ssm_state"));
    }
}

impl Default for Mamba2Cache {
    fn default() -> Self {
        Self::new()
    }
}

// Helper Functions.
/// Compute dt with bias and time step limits in float32 for numerical precision
/// (upstream mlx-lm casts dt to float32 before softplus)
fn compute_dt(
    dt: &MlxArray,
    dt_bias: &MlxArray,
    time_step_limit: (f32, f32),
) -> UniquePtr<MlxArray> {
    let dt_f32 = mlxcel_core::astype(dt, mlxcel_core::dtype::FLOAT32);
    let dt_bias_f32 = mlxcel_core::astype(dt_bias, mlxcel_core::dtype::FLOAT32);
    let dt_biased = mlxcel_core::add(&dt_f32, &dt_bias_f32);
    let dt_soft = mlxcel_core::softplus(&dt_biased);

    // Clip to time step limits (float32)
    let min_val = mlxcel_core::full_f32(&[1], time_step_limit.0, mlxcel_core::dtype::FLOAT32);
    let max_val = mlxcel_core::full_f32(&[1], time_step_limit.1, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::clip(&dt_soft, &min_val, &max_val)
}

/// Repeat array along axis by broadcasting
fn repeat_axis(x: &MlxArray, repeats: i32, axis: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len() as i32;
    let axis = if axis < 0 { ndim + axis } else { axis };

    // Expand dims after target axis
    let mut new_shape: Vec<i32> = shape.iter().take(axis as usize + 1).copied().collect();
    new_shape.push(1);
    new_shape.extend(shape.iter().skip(axis as usize + 1));
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    // Broadcast along the new dimension
    new_shape[axis as usize + 1] = repeats;
    let x_broad = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    // Merge the repeated dimension
    let mut final_shape: Vec<i32> = shape.clone();
    final_shape[axis as usize] *= repeats;
    mlxcel_core::reshape(&x_broad, &final_shape)
}

/// Segmented cumulative sum for SSM attention
fn segsum(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let l = shape[shape.len() - 1];

    // Repeat x along last dimension: [b, h, l] -> [b, h, l, l]
    let mut new_shape = shape.clone();
    new_shape.push(1);
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    // Broadcast to [b, h, l, l]
    let last_idx = new_shape.len() - 1;
    new_shape[last_idx] = l;
    let x_rep = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    // Apply tril with k=-1 (below diagonal)
    let x_tril = mlxcel_core::tril(&x_rep, -1);

    // Cumsum along axis -2
    mlxcel_core::cumsum(&x_tril, -2, false, true)
}

/// Chunked SSM attention implementation (SSD-SSM forward pass)
///
/// Processes the sequence in chunks of `step` tokens to avoid O(L^2) memory
/// from the [batch, heads, seq, seq] attention matrix.
/// Reference: mlx-lm ssm.py ssm_attn()
fn ssm_attn(
    x: &MlxArray,       // [batch, seq, heads, head_dim]
    a_log: &MlxArray,   // [heads]
    b: &MlxArray,       // [batch, seq, groups, state_dim]
    c: &MlxArray,       // [batch, seq, groups, state_dim]
    d: &MlxArray,       // [heads]
    dt: &MlxArray,      // [batch, seq, heads]
    dt_bias: &MlxArray, // [heads]
    state: Option<&MlxArray>,
    time_step_limit: (f32, f32),
    step: usize,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let x_shape = mlxcel_core::array_shape(x);
    let batch = x_shape[0];
    let seq_len = x_shape[1] as usize;
    let num_heads = x_shape[2];
    let head_dim = x_shape[3];

    let b_shape = mlxcel_core::array_shape(b);
    let n_groups = b_shape[2];
    let state_dim = b_shape[3];

    let repeats = num_heads / n_groups;

    // Compute dt, A, dtA, dtx for full sequence
    // dt is now float32 after compute_dt promotion; A matches dt dtype for precision
    let dt = compute_dt(dt, dt_bias, time_step_limit);
    let a = mlxcel_core::negative(&mlxcel_core::exp(a_log));
    let a = mlxcel_core::astype(&a, mlxcel_core::array_dtype(&dt));
    let a_reshaped = mlxcel_core::reshape(&a, &[1, 1, num_heads]);
    let dt_a = mlxcel_core::multiply(&dt, &a_reshaped);
    let dt_exp = mlxcel_core::reshape(&dt, &[batch, seq_len as i32, num_heads, 1]);
    let dtx = mlxcel_core::multiply(&dt_exp, x);

    // Process sequence in chunks
    let mut ys: Vec<UniquePtr<MlxArray>> = Vec::new();
    let mut current_state: Option<UniquePtr<MlxArray>> = state.map(mlxcel_core::copy);

    for chunk_start in (0..seq_len).step_by(step) {
        let chunk_end = (chunk_start + step).min(seq_len);
        let s = (chunk_end - chunk_start) as i32;

        // Slice chunks along sequence dimension (axis 1)
        let chunk_dtx = slice_axis(&dtx, 1, chunk_start as i32, chunk_end as i32);
        let chunk_dt_a = slice_axis(&dt_a, 1, chunk_start as i32, chunk_end as i32);
        let chunk_b = slice_axis(b, 1, chunk_start as i32, chunk_end as i32);
        let chunk_c = slice_axis(c, 1, chunk_start as i32, chunk_end as i32);

        // B: [batch, chunk, groups, state_dim] -> [batch, groups, state_dim, chunk]
        let b_t = mlxcel_core::transpose_axes(&chunk_b, &[0, 2, 3, 1]);

        // CB = C.swapaxes(1, 2) @ B_t
        let c_t = mlxcel_core::swap_axes(&chunk_c, 1, 2);
        let cb = mlxcel_core::matmul(&c_t, &b_t);
        let cb = repeat_axis(&cb, repeats, 1);

        // decay = exp(segsum(dtA.swapaxes(1, 2)))
        let dt_a_t = mlxcel_core::swap_axes(&chunk_dt_a, 1, 2);
        let seg = segsum(&dt_a_t);
        let decay = mlxcel_core::exp(&seg);

        // surrogate_attention_matrix = tril(CB * decay)
        let attn_matrix = mlxcel_core::multiply(&cb, &decay);
        let attn_matrix = mlxcel_core::tril(&attn_matrix, 0);

        // y = attn_matrix @ dtx.swapaxes(1, 2)
        let dtx_t = mlxcel_core::swap_axes(&chunk_dtx, 1, 2);
        let mut y_chunk = mlxcel_core::matmul(&attn_matrix, &dtx_t);
        y_chunk = mlxcel_core::swap_axes(&y_chunk, 1, 2);

        // Compute next state
        let decay_shape = mlxcel_core::array_shape(&decay);
        let decay_last = slice_axis(&decay, 2, decay_shape[2] - 1, decay_shape[2]);
        let decay_for_state = mlxcel_core::transpose_axes(&decay_last, &[0, 3, 1, 2]);

        let b_rep = repeat_axis(&b_t, repeats, 1);
        let b_sw = mlxcel_core::swap_axes(&b_rep, 2, 3);

        let dtx_decay = mlxcel_core::multiply(&chunk_dtx, &decay_for_state);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay, 1, 2);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay_t, 2, 3);

        let mut next_state = mlxcel_core::matmul(&dtx_decay_t, &b_sw);

        // Handle previous state carry-forward
        if let Some(ref prev_state) = current_state {
            let dta_cumsum = mlxcel_core::cumsum(&chunk_dt_a, -2, false, true);
            let exp_dta_cumsum = mlxcel_core::exp(&dta_cumsum);

            // Update next_state: next_state += exp_dtA_cumsum[:, -1, :, None, None] * state
            let exp_shape = mlxcel_core::array_shape(&exp_dta_cumsum);
            let last_exp = slice_axis(&exp_dta_cumsum, 1, exp_shape[1] - 1, exp_shape[1]);
            let mut exp_shape_new = mlxcel_core::array_shape(&last_exp);
            exp_shape_new.push(1);
            exp_shape_new.push(1);
            let last_exp = mlxcel_core::reshape(&last_exp, &exp_shape_new);
            let state_contrib = mlxcel_core::multiply(&last_exp, prev_state);
            next_state = mlxcel_core::add(&next_state, &state_contrib);

            // y_prev = (state @ C) contribution
            let c_reshaped = mlxcel_core::reshape(&chunk_c, &[batch, s, n_groups, 1, state_dim, 1]);
            let state_reshaped = mlxcel_core::reshape(
                prev_state,
                &[batch, 1, n_groups, repeats, head_dim, state_dim],
            );
            let y_prev = mlxcel_core::matmul(&state_reshaped, &c_reshaped);
            let y_prev = mlxcel_core::squeeze_axis(&y_prev, -1);
            let y_prev = mlxcel_core::reshape(&y_prev, &[batch, s, num_heads, head_dim]);

            let mut exp_shape_y = mlxcel_core::array_shape(&exp_dta_cumsum);
            exp_shape_y.push(1);
            let exp_dta_exp = mlxcel_core::reshape(&exp_dta_cumsum, &exp_shape_y);
            let y_prev_scaled = mlxcel_core::multiply(&exp_dta_exp, &y_prev);
            y_chunk = mlxcel_core::add(&y_chunk, &y_prev_scaled);
        }

        current_state = Some(next_state);
        ys.push(y_chunk);
    }

    // Concatenate chunk results along sequence dimension
    let mut y = ys.remove(0);
    for chunk in ys {
        let result = concatenate(y.as_ref().unwrap(), chunk.as_ref().unwrap(), 1);
        y = result;
    }

    // Add D term: y = y + x * D
    let d_reshaped = mlxcel_core::reshape(d, &[1, 1, num_heads, 1]);
    let d_contrib = mlxcel_core::multiply(x, &d_reshaped);
    y = mlxcel_core::add(&y, &d_contrib);

    // Cast y back to input dtype (dt computation was promoted to float32)
    y = mlxcel_core::astype(&y, mlxcel_core::array_dtype(x));

    (y, current_state.unwrap())
}

// Model Components.
/// Gated RMS Normalization for Mamba2
struct MambaRMSNormGated {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl MambaRMSNormGated {
    fn forward(&self, x: &MlxArray, gate: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let x = if let Some(g) = gate {
            mlxcel_core::multiply(x, &silu(g))
        } else {
            mlxcel_core::copy(x)
        };
        mlxcel_core::fast_rms_norm(&x, &self.weight, self.eps)
    }
}

/// Mamba2 SSM Block
pub struct Mamba2Block {
    num_heads: usize,
    #[allow(dead_code)]
    hidden_size: usize,
    ssm_state_size: usize,
    conv_kernel_size: usize,
    intermediate_size: usize,
    n_groups: usize,
    head_dim: usize,
    time_step_limit: (f32, f32),
    conv_dim: usize,

    // Conv1d weights
    conv_weight: UniquePtr<MlxArray>,
    conv_bias: Option<UniquePtr<MlxArray>>,

    // Projections
    in_proj: UnifiedLinear,
    out_proj: UnifiedLinear,

    // SSM parameters
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    d_param: UniquePtr<MlxArray>,

    // Normalization
    norm: MambaRMSNormGated,
}

impl Mamba2Block {
    fn conv(&self, conv_input: &MlxArray, cache: Option<&mut Mamba2Cache>) -> UniquePtr<MlxArray> {
        let k = self.conv_kernel_size;
        let shape = mlxcel_core::array_shape(conv_input);

        // Get or create conv state
        let padded_input = if let Some(c) = cache.as_ref() {
            if let Some(ref conv_state) = c.conv_state {
                // UniquePtr::as_ref() returns Option<&T>, unwrap since we know it's Some
                concatenate(conv_state.as_ref().unwrap(), conv_input, 1)
            } else {
                let pad_arr = mlxcel_core::zeros(
                    &[shape[0], (k - 1) as i32, shape[2]],
                    mlxcel_core::array_dtype(conv_input),
                );
                concatenate(&pad_arr, conv_input, 1)
            }
        } else {
            let pad_arr = mlxcel_core::zeros(
                &[shape[0], (k - 1) as i32, shape[2]],
                mlxcel_core::array_dtype(conv_input),
            );
            concatenate(&pad_arr, conv_input, 1)
        };

        // Update conv cache.
        // Wrap slice in contiguous() to force MLX to materialize a fresh,
        // independent buffer. Without this, the slice is a lazy view that
        // retains a reference to the full padded_input allocation, causing a
        // memory leak proportional to the sequence length.
        if let Some(c) = cache {
            let n_keep = k - 1;
            let padded_shape = mlxcel_core::array_shape(&padded_input);
            let len = padded_shape[1] as usize;
            let tail = slice_axis(&padded_input, 1, (len - n_keep) as i32, len as i32);
            c.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        }

        // Depthwise conv1d
        let conv_out = mlxcel_core::conv1d(
            &padded_input,
            &self.conv_weight,
            1,
            0,
            1,
            self.conv_dim as i32,
        );
        let conv_out = if let Some(ref b) = self.conv_bias {
            // Reshape bias for broadcasting
            let b_reshaped = mlxcel_core::reshape(b, &[1, 1, -1]);
            mlxcel_core::add(&conv_out, &b_reshaped)
        } else {
            conv_out
        };

        silu(&conv_out)
    }

    fn ssm(
        &self,
        hidden_states: &MlxArray,
        b: &MlxArray,
        c: &MlxArray,
        dt: &MlxArray,
        cache: Option<&mut Mamba2Cache>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let batch = shape[0];
        let seq_len = shape[1];

        // Reshape for multi-head
        let hidden_states = mlxcel_core::reshape(
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

        let state = cache
            .as_ref()
            .and_then(|c| c.ssm_state.as_ref())
            .and_then(|s| s.as_ref());

        let (y, new_state) = ssm_attn(
            &hidden_states,
            &self.a_log,
            &b,
            &c,
            &self.d_param,
            dt,
            &self.dt_bias,
            state,
            self.time_step_limit,
            256, // chunk step size matching Python mlx-lm
        );

        if let Some(c) = cache {
            c.ssm_state = Some(new_state);
        }

        mlxcel_core::reshape(&y, &[batch, seq_len, self.intermediate_size as i32])
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        mut cache: Option<&mut Mamba2Cache>,
    ) -> UniquePtr<MlxArray> {
        // in_proj -> (gate, conv_input, dt)
        let projected = self.in_proj.forward(x);

        let proj_shape = mlxcel_core::array_shape(&projected);
        let proj_dim = proj_shape[proj_shape.len() - 1];

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
            proj_dim,
        );

        // Conv
        let conv_output = self.conv(&conv_input, cache.as_deref_mut());

        // Split conv_output -> (hidden_states, B, C)
        let conv_shape = mlxcel_core::array_shape(&conv_output);
        let conv_last_dim = conv_shape[conv_shape.len() - 1];
        let bc_split = (self.intermediate_size + self.n_groups * self.ssm_state_size) as i32;
        let hidden_states = slice_axis(&conv_output, -1, 0, self.intermediate_size as i32);
        let b = slice_axis(&conv_output, -1, self.intermediate_size as i32, bc_split);
        let c_out = slice_axis(&conv_output, -1, bc_split, conv_last_dim);

        // SSM
        let y = self.ssm(&hidden_states, &b, &c_out, &dt, cache);

        // Gated norm and output
        let y = self.norm.forward(&y, Some(&gate));
        self.out_proj.forward(&y)
    }
}

/// Residual block wrapping Mamba2Block
pub struct ResidualBlock {
    mixer: Mamba2Block,
    norm: RMSNorm,
}

impl ResidualBlock {
    pub fn forward(&self, x: &MlxArray, cache: Option<&mut Mamba2Cache>) -> UniquePtr<MlxArray> {
        let normed = self.norm.forward(x);
        let out = self.mixer.forward(&normed, cache);
        mlxcel_core::add(x, &out)
    }
}

// Full Mamba2 Model.

pub struct Mamba2Model {
    config: Mamba2Config,
    embeddings: UnifiedEmbedding,
    layers: Vec<ResidualBlock>,
    norm_f: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    /// Internal caches for LanguageModel trait compatibility
    sequence_state: ModelOwnedSequenceState<Mamba2Cache>,
}

impl Mamba2Model {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<Mamba2Cache> {
        (0..self.config.num_hidden_layers)
            .map(|_| Mamba2Cache::new())
            .collect()
    }

    pub fn forward_with_caches(
        &self,
        x: &MlxArray,
        caches: &mut [Mamba2Cache],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embeddings.forward(x);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, Some(cache));
        }

        let h = self.norm_f.forward(&h);

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embeddings.as_linear(&h)
        }
    }

    /// Load model from safetensors files
    pub fn load(model_path: &str) -> Result<(Self, Mamba2Config), Box<dyn std::error::Error>> {
        let path = Path::new(model_path);

        // Load config
        println!("[Mamba2] Loading config...");
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_str = super::sanitize_config_json(&config_str);
        let mut config: Mamba2Config = serde_json::from_str(&config_str)?;
        config.compute_time_step_rank();
        println!(
            "[Mamba2] Config loaded: {} layers",
            config.num_hidden_layers
        );

        // Get quantization parameters
        let _group_size = config.group_size();
        let _bits = config.bits();

        // Load weights
        println!("[Mamba2] Loading weights from safetensors...");
        let weights = crate::models::load_text_weights(path, None)?;

        // Process weights (handle conv1d weight transpose)
        let weights = Self::sanitize_weights(weights, &config);

        // Build model
        println!("[Mamba2] Building model...");
        let model = Self::from_weights(config.clone(), weights)?;

        println!("[Mamba2] Model loaded successfully");
        Ok((model, config))
    }

    fn sanitize_weights(mut weights: WeightMap, config: &Mamba2Config) -> WeightMap {
        // Handle conv1d weight transpose
        let keys: Vec<String> = weights.keys().cloned().collect();
        for k in keys {
            if k.contains("conv1d.weight")
                && let Some(v) = weights.get(&k)
            {
                let shape = mlxcel_core::array_shape(v);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    // swap axes -1 and -2 (equivalent to moveaxis from -1 to -2)
                    let transposed = mlxcel_core::swap_axes(v, -1, -2);
                    weights.insert(k, transposed);
                }
            }
        }

        // Remove lm_head if tie_word_embeddings
        if config.tie_word_embeddings {
            weights.remove("lm_head.weight");
        }

        weights
    }

    pub fn from_weights(
        config: Mamba2Config,
        mut weights: WeightMap,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let group_size = config.group_size();
        let bits = config.bits();

        // Build embeddings (auto-detect quantization)
        let embed_weight = weights
            .remove("backbone.embeddings.weight")
            .or_else(|| weights.remove("model.embed_tokens.weight"))
            .ok_or("Missing embedding weight")?;
        let embed_scales = weights
            .remove("backbone.embeddings.scales")
            .or_else(|| weights.remove("model.embed_tokens.scales"));
        let embed_biases = weights
            .remove("backbone.embeddings.biases")
            .or_else(|| weights.remove("model.embed_tokens.biases"));

        let embeddings = if let (Some(scales), Some(biases)) = (embed_scales, embed_biases) {
            UnifiedEmbedding::Quantized(mlxcel_core::layers::QuantizedEmbedding::new(
                embed_weight,
                scales,
                biases,
                group_size,
                bits,
            ))
        } else {
            UnifiedEmbedding::Regular(mlxcel_core::layers::Embedding::new(embed_weight))
        };

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("backbone.layers.{}", i);

            // Build Mamba2Block
            let num_heads = config.num_heads;
            let head_dim = config.head_dim;
            let ssm_state_size = config.get_ssm_state_size();
            let n_groups = config.n_groups;
            let intermediate_size = num_heads * head_dim;
            let conv_dim = intermediate_size + 2 * n_groups * ssm_state_size;

            // Conv1d weights
            let conv_weight = weights
                .remove(&format!("{}.mixer.conv1d.weight", prefix))
                .ok_or(format!("Missing conv1d weight for layer {}", i))?;
            let conv_bias = weights.remove(&format!("{}.mixer.conv1d.bias", prefix));

            // in_proj (quantized)
            let in_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mixer.in_proj", prefix),
                group_size,
                bits,
            )?;

            // out_proj (quantized)
            let out_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mixer.out_proj", prefix),
                group_size,
                bits,
            )?;

            // SSM parameters
            let dt_bias = weights
                .remove(&format!("{}.mixer.dt_bias", prefix))
                .ok_or(format!("Missing dt_bias for layer {}", i))?;
            let a_log = weights
                .remove(&format!("{}.mixer.A_log", prefix))
                .ok_or(format!("Missing A_log for layer {}", i))?;
            let d_param = weights
                .remove(&format!("{}.mixer.D", prefix))
                .ok_or(format!("Missing D for layer {}", i))?;

            // Norm
            let norm_weight = weights
                .remove(&format!("{}.mixer.norm.weight", prefix))
                .ok_or(format!("Missing mixer norm weight for layer {}", i))?;
            let norm = MambaRMSNormGated {
                weight: norm_weight,
                eps: config.layer_norm_epsilon,
            };

            let mixer = Mamba2Block {
                num_heads,
                hidden_size: config.hidden_size,
                ssm_state_size,
                conv_kernel_size: config.conv_kernel,
                intermediate_size,
                n_groups,
                head_dim,
                time_step_limit: config.time_step_limit,
                conv_dim,
                conv_weight,
                conv_bias,
                in_proj,
                out_proj,
                dt_bias,
                a_log,
                d_param,
                norm,
            };

            // ResidualBlock norm
            let block_norm_weight = weights
                .remove(&format!("{}.norm.weight", prefix))
                .ok_or(format!("Missing block norm weight for layer {}", i))?;
            let block_norm = RMSNorm::new(block_norm_weight, config.layer_norm_epsilon);

            layers.push(ResidualBlock {
                mixer,
                norm: block_norm,
            });
        }

        // Final norm
        let norm_f_weight = weights
            .remove("backbone.norm_f.weight")
            .ok_or("Missing final norm weight")?;
        let norm_f = RMSNorm::new(norm_f_weight, config.layer_norm_epsilon);

        // LM head (if not tie_word_embeddings)
        let lm_head = if !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                &weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        // Create internal caches for LanguageModel trait compatibility
        let internal_caches: Vec<Mamba2Cache> = (0..config.num_hidden_layers)
            .map(|_| Mamba2Cache::new())
            .collect();

        Ok(Self {
            config,
            embeddings,
            layers,
            norm_f,
            lm_head,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
        })
    }
}

// LanguageModel trait implementation.
use mlxcel_core::layers::KVCache;

impl LanguageModel for Mamba2Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Mamba2 uses internal caching (Mamba2Cache) instead of KV cache
        // We use model-owned state to isolate scheduler sequence ids while
        // retaining a fallback slot for CLI / benchmark paths.
        self.sequence_state
            .with_sequence_state(None, |internal| self.forward_with_caches(input, internal))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset fallback internal caches
        self.sequence_state
            .replace_internal(Mamba2Model::make_caches(self));
        // Return dummy KV caches for trait compatibility
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt Mamba2 recurrent state
    }

    fn supports_batching(&self) -> bool {
        false // Mamba2 uses internal recurrent state, not compatible with per-sequence KV isolation
    }

    fn prepare_sequence_state(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, Mamba2Model::make_caches(self));
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
            || Mamba2Model::make_caches(self),
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
                    mlxcel_core::generate::ModelStateSnapshot::new("mamba2", token_len);
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
        if snapshot.family() != "mamba2" {
            return Err(format!(
                "cannot restore {} snapshot into Mamba2",
                snapshot.family()
            ));
        }
        let mut state = Mamba2Model::make_caches(self);
        for (idx, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{idx}"));
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        super::mamba::parse_eos_token_ids(&self.config.eos_token_id, 0)
    }
}

#[cfg(test)]
#[path = "mamba2_tests.rs"]
mod tests;
