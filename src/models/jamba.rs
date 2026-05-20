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

// Jamba: Hybrid Mamba + Transformer architecture for mlxcel-core
// Reference: mlx-lm/mlx_lm/models/jamba.py
//
// Key features:
// - Interleaved Mamba and Attention blocks (configurable pattern)
// - Optional Sparse MoE (SwitchGLU)
// - Mamba blocks with dt/b/c layernorms
// - Mixed cache: KVCache for attention, MambaCache for Mamba

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{
    FusedQKVLinear, KVCache, Linear, RMSNorm, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::{create_causal_mask, silu, slice_axis, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JambaConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    // Attention layer pattern
    #[serde(default)]
    pub attn_layer_offset: usize,
    #[serde(default = "default_attn_layer_period")]
    pub attn_layer_period: usize,

    // Expert (MoE) layer pattern
    #[serde(default)]
    pub expert_layer_offset: usize,
    #[serde(default = "default_expert_layer_period")]
    pub expert_layer_period: usize,

    // Mamba parameters
    #[serde(default = "default_mamba_d_conv")]
    pub mamba_d_conv: usize,
    #[serde(default = "default_mamba_d_state")]
    pub mamba_d_state: usize,
    #[serde(default = "default_mamba_expand")]
    pub mamba_expand: usize,
    #[serde(default, deserialize_with = "deserialize_mamba_dt_rank")]
    pub mamba_dt_rank: usize,
    #[serde(default)]
    pub mamba_proj_bias: bool,
    #[serde(default = "default_true")]
    pub mamba_conv_bias: bool,

    // MoE parameters
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub layers_block_type: Option<Vec<String>>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

impl JambaConfig {
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

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_attn_layer_period() -> usize {
    8
}

fn default_expert_layer_period() -> usize {
    2
}

fn default_mamba_d_conv() -> usize {
    4
}

fn default_mamba_d_state() -> usize {
    16
}

fn default_mamba_expand() -> usize {
    2
}

fn default_num_experts() -> usize {
    1
}

fn default_num_experts_per_tok() -> usize {
    1
}

fn default_true() -> bool {
    true
}

fn deserialize_mamba_dt_rank<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DtRank {
        Number(usize),
        String(String),
    }

    match DtRank::deserialize(deserializer)? {
        DtRank::Number(n) => Ok(n),
        DtRank::String(s) if s == "auto" => Ok(0), // Will be computed in post_init
        DtRank::String(s) => Err(D::Error::custom(format!("invalid mamba_dt_rank: {}", s))),
    }
}

impl JambaConfig {
    pub fn post_init(&mut self) {
        // Compute dt_rank if "auto"
        if self.mamba_dt_rank == 0 {
            self.mamba_dt_rank = self.hidden_size.div_ceil(16);
        }

        // Generate layers_block_type if not provided
        if self.layers_block_type.is_none() {
            let block_types: Vec<String> = (0..self.num_hidden_layers)
                .map(|i| {
                    if i % self.attn_layer_period == self.attn_layer_offset {
                        "attention".to_string()
                    } else {
                        "mamba".to_string()
                    }
                })
                .collect();
            self.layers_block_type = Some(block_types);
        }
    }

    pub fn get_layers_block_type(&self) -> Vec<String> {
        self.layers_block_type.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|i| {
                    if i % self.attn_layer_period == self.attn_layer_offset {
                        "attention".to_string()
                    } else {
                        "mamba".to_string()
                    }
                })
                .collect()
        })
    }

    pub fn get_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn get_mamba_intermediate_size(&self) -> usize {
        self.mamba_expand * self.hidden_size
    }

    pub fn is_expert_layer(&self, layer_idx: usize) -> bool {
        self.num_experts > 1
            && (layer_idx + self.expert_layer_offset).is_multiple_of(self.expert_layer_period)
    }
}

// Jamba Cache Types.
/// Cache for Mamba blocks (conv state + SSM state)
pub struct JambaMambaCache {
    pub conv_state: Option<UniquePtr<MlxArray>>,
    pub ssm_state: Option<UniquePtr<MlxArray>>,
}

impl JambaMambaCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
        }
    }
}

impl Default for JambaMambaCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Enum for mixed cache types
pub enum JambaLayerCache {
    Attention(KVCache),
    Mamba(JambaMambaCache),
}

impl JambaLayerCache {
    pub fn offset(&self) -> i32 {
        match self {
            JambaLayerCache::Attention(kv) => kv.offset,
            JambaLayerCache::Mamba(_) => 0,
        }
    }
}

// MLP (Non-MoE Feed Forward).
struct JambaMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl JambaMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(result) = mlxcel_core::layers::compiled_swiglu_mlp(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return result;
        }

        let gate = silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        let gated = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&gated)
    }
}

// SwitchLinear for MoE.
// Kept local: uses take+matmul (not gather_mm), different forward shape handling
#[allow(dead_code)]
enum SwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    },
    Regular {
        weight: UniquePtr<MlxArray>, // [num_experts, out_features, in_features]
    },
}

impl SwitchLinear {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                // Use gather_qmm for quantized MoE experts
                unsafe {
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
                        false,
                        "affine",
                    )
                }
            }
            Self::Regular { weight } => {
                // Gather weights for selected experts, then batched matmul
                let w = mlxcel_core::take(weight, indices, 0);
                let x_shape = mlxcel_core::array_shape(x);
                let x_reshaped = mlxcel_core::reshape(x, &[x_shape[0], x_shape[1], 1, x_shape[2]]);
                let w_transposed = mlxcel_core::transpose_axes(&w, &[0, 1, 3, 2]);
                let result = mlxcel_core::matmul(&x_reshaped, &w_transposed);
                mlxcel_core::squeeze_axis(&result, 2)
            }
        }
    }
}

// SwitchGLU (MoE FFN).
struct SwitchGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGLU {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        // Expand x for expert routing: [batch, seq, hidden] -> [batch, seq, 1, hidden]
        let x_exp = mlxcel_core::reshape(x, &[batch, seq_len, 1, shape[2]]);

        // Forward through gate and up projections
        let x_gate = self.gate_proj.forward(&x_exp, indices);
        let x_up = self.up_proj.forward(&x_exp, indices);

        // SiLU activation on gate
        let gated = mlxcel_core::multiply(&silu(&x_gate), &x_up);

        // Reshape for down projection
        let gated_shape = mlxcel_core::array_shape(&gated);
        let k = gated_shape[2]; // num_experts_per_tok

        let flat = mlxcel_core::reshape(&gated, &[batch * seq_len, k, gated_shape[3]]);

        // Flatten indices too
        let indices_flat = mlxcel_core::reshape(indices, &[-1, k]);

        // Down projection
        let out = self.down_proj.forward(&flat, &indices_flat);

        // Reshape back to [batch, seq, k, hidden]
        let out_shape = mlxcel_core::array_shape(&out);
        mlxcel_core::reshape(&out, &[batch, seq_len, k, out_shape[out_shape.len() - 1]])
    }
}

// Sparse MoE Block.
struct JambaSparseMoeBlock {
    router: Linear,
    switch_mlp: SwitchGLU,
    num_experts_per_tok: usize,
}

impl JambaSparseMoeBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gates = self.router.forward(x);
        let k = self.num_experts_per_tok as i32;

        // Get top-k expert indices using argpartition
        let neg_gates = mlxcel_core::negative(&gates);
        let inds = mlxcel_core::argpartition(&neg_gates, k - 1, -1);
        let inds = slice_axis(&inds, -1, 0, k);

        // Get scores for selected experts
        let scores = mlxcel_core::take_along_axis(&gates, &inds, -1);
        let scores = mlxcel_core::softmax(&scores, -1);

        // Apply MoE
        let y = self.switch_mlp.forward(x, &inds);

        // Weighted sum: y * scores[..., None]
        let mut scores_shape = mlxcel_core::array_shape(&scores);
        scores_shape.push(1);
        let scores_exp = mlxcel_core::reshape(&scores, &scores_shape);
        let weighted = mlxcel_core::multiply(&y, &scores_exp);

        // Sum over expert dimension
        mlxcel_core::sum_axis(&weighted, -2, false)
    }
}

// Feed Forward Enum.
enum FeedForward {
    MLP(JambaMLP),
    MoE(JambaSparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FeedForward::MLP(mlp) => mlp.forward(x),
            FeedForward::MoE(moe) => moe.forward(x),
        }
    }
}

// Jamba Attention.
struct JambaAttention {
    qkv_proj: FusedQKVLinear,
    o_proj: UnifiedLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl JambaAttention {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<&mut KVCache>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        let (queries, keys, values) = self.qkv_proj.forward(x);

        // Reshape to [batch, seq, n_heads, head_dim]
        let queries = mlxcel_core::reshape(
            &queries,
            &[batch, seq_len, self.n_heads as i32, self.head_dim as i32],
        );
        let keys = mlxcel_core::reshape(
            &keys,
            &[batch, seq_len, self.n_kv_heads as i32, self.head_dim as i32],
        );
        let values = mlxcel_core::reshape(
            &values,
            &[batch, seq_len, self.n_kv_heads as i32, self.head_dim as i32],
        );

        // Transpose to [batch, n_heads, seq, head_dim]
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        // Update cache
        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(keys, values)
        } else {
            (keys, values)
        };

        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &queries, &keys, &values, self.scale, mask_ptr, 0.0, 0,
            )
        };

        // Transpose back and reshape
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[batch, seq_len, -1]);

        self.o_proj.forward(&output)
    }
}

// Jamba Mamba Mixer.
#[allow(dead_code)]
struct JambaMambaMixer {
    hidden_size: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    conv_kernel_size: usize,
    time_step_rank: usize,

    in_proj: UnifiedLinear,
    conv_weight: UniquePtr<MlxArray>,
    conv_bias: Option<UniquePtr<MlxArray>>,
    x_proj: UnifiedLinear,
    dt_proj: Linear, // dt_proj is NOT quantized in the model weights
    out_proj: UnifiedLinear,

    dt_layernorm: RMSNorm,
    b_layernorm: RMSNorm,
    c_layernorm: RMSNorm,

    a_log: UniquePtr<MlxArray>,
    d_param: UniquePtr<MlxArray>,
}

impl JambaMambaMixer {
    fn ssm_step(
        &self,
        x: &MlxArray,
        a: &MlxArray,
        state: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        // deltaBC = x_proj(x)
        let delta_bc = self.x_proj.forward(x);

        // Split into delta, B, C
        let delta = slice_axis(&delta_bc, -1, 0, self.time_step_rank as i32);
        let b = slice_axis(
            &delta_bc,
            -1,
            self.time_step_rank as i32,
            (self.time_step_rank + self.ssm_state_size) as i32,
        );
        let c = slice_axis(
            &delta_bc,
            -1,
            (self.time_step_rank + self.ssm_state_size) as i32,
            -1,
        );

        // Apply layernorms
        let delta = self.dt_layernorm.forward(&delta);
        let b = self.b_layernorm.forward(&b);
        let c = self.c_layernorm.forward(&c);

        // delta = softplus(dt_proj(delta))
        let delta = mlxcel_core::softplus(&self.dt_proj.forward(&delta));

        // new_state = (delta * x)[..., None] * B[..., None, :]
        let delta_x = mlxcel_core::multiply(&delta, x);
        let delta_x_exp = mlxcel_core::reshape(
            &delta_x,
            &[batch, seq_len, self.intermediate_size as i32, 1],
        );
        let b_exp = mlxcel_core::reshape(&b, &[batch, seq_len, 1, self.ssm_state_size as i32]);
        let new_state = mlxcel_core::multiply(&delta_x_exp, &b_exp);

        // dtA = exp(delta[..., None] * A)
        let delta_exp =
            mlxcel_core::reshape(&delta, &[batch, seq_len, self.intermediate_size as i32, 1]);
        let dt_a = mlxcel_core::exp(&mlxcel_core::multiply(&delta_exp, a));

        if seq_len == 1 {
            let new_state_t = mlxcel_core::squeeze_axis(&new_state, 1);
            let dt_a_t = mlxcel_core::squeeze_axis(&dt_a, 1);
            let updated_t = if let Some(prev) = state {
                let prev_contrib = mlxcel_core::multiply(prev, &dt_a_t);
                mlxcel_core::add(&prev_contrib, &new_state_t)
            } else {
                new_state_t
            };

            let c_t = mlxcel_core::squeeze_axis(&c, 1);
            let c_exp = mlxcel_core::reshape(&c_t, &[batch, self.ssm_state_size as i32, 1]);
            let y = mlxcel_core::matmul(&updated_t, &c_exp);
            let y = mlxcel_core::squeeze_axis(&y, -1);
            let y = mlxcel_core::expand_dims(&y, 1);

            let d_reshaped =
                mlxcel_core::reshape(&self.d_param, &[1, 1, self.intermediate_size as i32]);
            let d_contrib = mlxcel_core::multiply(&d_reshaped, x);
            let y = mlxcel_core::add(&y, &d_contrib);

            return (y, updated_t);
        }

        // Sequential scan through timesteps (runs regardless of initial state)
        // Python: for t in range(T): if state: new_state[:,t] = state*dtA[:,t] + new_state[:,t]; state = new_state[:,t]
        let mut current_state = state.map(mlxcel_core::copy);
        let mut updated_states: Vec<UniquePtr<MlxArray>> = Vec::new();

        for t in 0..seq_len {
            let new_state_t = slice_axis(&new_state, 1, t, t + 1);
            let new_state_t = mlxcel_core::squeeze_axis(&new_state_t, 1);
            let dt_a_t = slice_axis(&dt_a, 1, t, t + 1);
            let dt_a_t = mlxcel_core::squeeze_axis(&dt_a_t, 1);

            let updated_t = if let Some(ref prev) = current_state {
                // new_state[:, t] = state * dtA[:, t] + new_state[:, t]
                let prev_contrib = mlxcel_core::multiply(prev, &dt_a_t);
                mlxcel_core::add(&prev_contrib, &new_state_t)
            } else {
                new_state_t
            };

            // state = new_state[:, t]
            current_state = Some(mlxcel_core::copy(&updated_t));
            updated_states.push(mlxcel_core::expand_dims(&updated_t, 1));
        }

        // Stack all updated states
        let final_new_state = if updated_states.len() == 1 {
            mlxcel_core::copy(updated_states[0].as_ref().unwrap())
        } else {
            let mut result = mlxcel_core::copy(updated_states[0].as_ref().unwrap());
            for st in updated_states.iter().skip(1) {
                result = concatenate(&result, st.as_ref().unwrap(), 1);
            }
            result
        };

        // y = (new_state @ C[..., None]).squeeze(-1)
        let c_exp = mlxcel_core::reshape(&c, &[batch, seq_len, self.ssm_state_size as i32, 1]);
        let y = mlxcel_core::matmul(&final_new_state, &c_exp);
        let y = mlxcel_core::squeeze_axis(&y, -1);

        // y = y + D * x
        let d_reshaped =
            mlxcel_core::reshape(&self.d_param, &[1, 1, self.intermediate_size as i32]);
        let d_contrib = mlxcel_core::multiply(&d_reshaped, x);
        let y = mlxcel_core::add(&y, &d_contrib);

        // Final state is current_state (the last timestep's state)
        let final_state = current_state.unwrap();

        (y, final_state)
    }

    fn forward(
        &self,
        x: &MlxArray,
        mut cache: Option<&mut JambaMambaCache>,
    ) -> UniquePtr<MlxArray> {
        let conv_state = cache.as_ref().and_then(|c| {
            c.conv_state
                .as_ref()
                .and_then(|s| s.as_ref().map(mlxcel_core::copy))
        });
        let ssm_state = cache.as_ref().and_then(|c| {
            c.ssm_state
                .as_ref()
                .and_then(|s| s.as_ref().map(mlxcel_core::copy))
        });

        // xz = in_proj(x)
        let xz = self.in_proj.forward(x);
        let mid = self.intermediate_size as i32;
        let x_part = slice_axis(&xz, -1, 0, mid);
        let z = slice_axis(&xz, -1, mid, mid * 2);

        // Conv with padding
        let shape = mlxcel_core::array_shape(&x_part);
        let k = self.conv_kernel_size;
        let padded_input = if let Some(ref conv_st) = conv_state {
            concatenate(conv_st, &x_part, 1)
        } else {
            let pad_arr = mlxcel_core::zeros(
                &[shape[0], (k - 1) as i32, shape[2]],
                mlxcel_core::array_dtype(&x_part),
            );
            concatenate(&pad_arr, &x_part, 1)
        };

        // Depthwise conv1d
        let conv_out = mlxcel_core::conv1d(
            &padded_input,
            &self.conv_weight,
            1,
            0,
            1,
            self.intermediate_size as i32,
        );
        let conv_out = if let Some(ref b) = self.conv_bias {
            let b_reshaped = mlxcel_core::reshape(b, &[1, 1, -1]);
            mlxcel_core::add(&conv_out, &b_reshaped)
        } else {
            conv_out
        };

        // Update conv cache.
        // Wrap slice in contiguous() to force MLX to materialize a fresh,
        // independent buffer. Without this, the slice is a lazy view that
        // retains a reference to the full padded_input allocation, causing a
        // memory leak proportional to the sequence length. (issue #336)
        if let Some(c) = cache.as_deref_mut() {
            let padded_shape = mlxcel_core::array_shape(&padded_input);
            let len = padded_shape[1] as usize;
            let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
            c.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        }

        let x_conv = silu(&conv_out);

        // SSM
        let a = mlxcel_core::negative(&mlxcel_core::exp(&self.a_log));
        let (y, new_ssm_state) = self.ssm_step(&x_conv, &a, ssm_state.as_deref());

        // Update SSM state
        if let Some(c) = cache {
            c.ssm_state = Some(new_ssm_state);
        }

        // Output: out_proj(silu(z) * y)
        let z_act = silu(&z);
        let gated = mlxcel_core::multiply(&z_act, &y);
        self.out_proj.forward(&gated)
    }
}

// Temporal Block Enum.
enum TemporalBlock {
    Attention(JambaAttention),
    Mamba(JambaMambaMixer),
}

// Decoder Layer.
#[allow(dead_code)]
struct JambaDecoderLayer {
    is_attn: bool,
    temporal: TemporalBlock,
    feed_forward: FeedForward,
    input_layernorm: RMSNorm,
    pre_ff_layernorm: RMSNorm,
}

impl JambaDecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut JambaLayerCache,
    ) -> UniquePtr<MlxArray> {
        let h_norm = self.input_layernorm.forward(x);

        let h = match (&self.temporal, cache) {
            (TemporalBlock::Attention(attn), JambaLayerCache::Attention(kv_cache)) => {
                attn.forward(&h_norm, mask, Some(kv_cache))
            }
            (TemporalBlock::Attention(attn), _) => attn.forward(&h_norm, mask, None),
            (TemporalBlock::Mamba(mamba), JambaLayerCache::Mamba(mamba_cache)) => {
                mamba.forward(&h_norm, Some(mamba_cache))
            }
            (TemporalBlock::Mamba(mamba), _) => mamba.forward(&h_norm, None),
        };

        let r = mlxcel_core::add(x, &h);
        let ff_norm = self.pre_ff_layernorm.forward(&r);
        let ff_out = self.feed_forward.forward(&ff_norm);
        mlxcel_core::add(&r, &ff_out)
    }
}

// Jamba Model Backbone.
struct JambaModelBackbone {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<JambaDecoderLayer>,
    final_layernorm: RMSNorm,
    layers_block_type: Vec<String>,
    attn_idx: usize,
}

impl JambaModelBackbone {
    fn forward(
        &self,
        inputs: &MlxArray,
        caches: Option<&mut [JambaLayerCache]>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(inputs);

        // Create attention mask
        let mask = {
            let shape = mlxcel_core::array_shape(&h);
            let seq_len = shape[1];
            if seq_len > 1 {
                let offset = caches
                    .as_ref()
                    .map(|c| {
                        if self.attn_idx < c.len() {
                            c[self.attn_idx].offset()
                        } else {
                            0
                        }
                    })
                    .unwrap_or(0);
                Some(create_causal_mask(seq_len, offset))
            } else {
                None
            }
        };

        if let Some(cache_slice) = caches {
            for (layer, cache) in self.layers.iter().zip(cache_slice.iter_mut()) {
                h = layer.forward(&h, mask.as_deref(), cache);
            }
        } else {
            // No cache - create temporary caches
            let mut temp_caches: Vec<JambaLayerCache> = self
                .layers_block_type
                .iter()
                .map(|t| {
                    if t == "attention" {
                        JambaLayerCache::Attention(KVCache::new())
                    } else {
                        JambaLayerCache::Mamba(JambaMambaCache::new())
                    }
                })
                .collect();
            for (layer, cache) in self.layers.iter().zip(temp_caches.iter_mut()) {
                h = layer.forward(&h, mask.as_deref(), cache);
            }
        }

        self.final_layernorm.forward(&h)
    }
}

// Full Jamba Model.
use std::cell::RefCell;

pub struct JambaModel {
    config: JambaConfig,
    model: JambaModelBackbone,
    lm_head: Option<UnifiedLinear>,
    /// Internal caches for LanguageModel trait compatibility
    /// Using RefCell to allow mutation through shared reference (required by trait)
    internal_caches: RefCell<Vec<JambaLayerCache>>,
}

impl JambaModel {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<JambaLayerCache> {
        self.model
            .layers_block_type
            .iter()
            .map(|t| {
                if t == "attention" {
                    JambaLayerCache::Attention(KVCache::new())
                } else {
                    JambaLayerCache::Mamba(JambaMambaCache::new())
                }
            })
            .collect()
    }

    pub fn forward_with_caches(
        &self,
        inputs: &MlxArray,
        caches: Option<&mut [JambaLayerCache]>,
    ) -> UniquePtr<MlxArray> {
        let out = self.model.forward(inputs, caches);
        if let Some(ref head) = self.lm_head {
            head.forward(&out)
        } else {
            self.model.embed_tokens.as_linear(&out)
        }
    }

    /// Return the classification of every layer in the full Jamba stack, so
    /// a pipeline stage can construct one [`JambaLayerCache`] per local
    /// layer without importing `JambaModelBackbone` internals.
    ///
    /// Used by: Jamba pipeline stage executor
    pub fn layer_block_types(&self) -> &[String] {
        &self.model.layers_block_type
    }

    /// Run a pipeline-stage forward pass that only touches layers in
    /// `layer_range`. `has_embedding` decides whether `inputs` is treated
    /// as token IDs (and embedded on the fly) or as hidden states carried
    /// in from the previous stage. `has_lm_head` decides whether the final
    /// norm and LM head are applied on the way out.
    ///
    /// Because SSM state is carried across tokens inside each Mamba block,
    /// the entire layer range (and therefore every SSM block within it)
    /// must be evaluated on the same stage — callers must never split an
    /// individual Mamba block across stages. See the module-level design
    /// note on the Jamba pipeline stage executor for details.
    ///
    /// Used by: Jamba pipeline stage executor
    pub fn forward_stage(
        &self,
        inputs: &MlxArray,
        layer_range: std::ops::Range<usize>,
        has_embedding: bool,
        has_lm_head: bool,
        caches: &mut [JambaLayerCache],
    ) -> UniquePtr<MlxArray> {
        assert_eq!(
            caches.len(),
            layer_range.end - layer_range.start,
            "jamba forward_stage: cache count must match layer range length"
        );

        // Stage input: either embed tokens or accept hidden states as-is.
        let mut h = if has_embedding {
            self.model.embed_tokens.forward(inputs)
        } else {
            mlxcel_core::copy(inputs)
        };

        // Build a mask anchored on the first attention layer inside this
        // stage. If the stage has no attention layers (pure Mamba slice),
        // no mask is required.
        let local_attn_idx = layer_range.clone().enumerate().find_map(|(local, abs)| {
            if self.model.layers_block_type.get(abs).map(|s| s.as_str()) == Some("attention") {
                Some(local)
            } else {
                None
            }
        });
        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];
        let mask = if let Some(idx) = local_attn_idx {
            if seq_len > 1 {
                let offset = caches[idx].offset();
                Some(create_causal_mask(seq_len, offset))
            } else {
                None
            }
        } else {
            None
        };

        for (local, abs) in layer_range.clone().enumerate() {
            h = self.model.layers[abs].forward(&h, mask.as_deref(), &mut caches[local]);
        }

        if has_lm_head {
            let h = self.model.final_layernorm.forward(&h);
            if let Some(ref head) = self.lm_head {
                head.forward(&h)
            } else {
                self.model.embed_tokens.as_linear(&h)
            }
        } else {
            h
        }
    }

    /// Load model from safetensors files
    pub fn load(model_path: &str) -> Result<(Self, JambaConfig), Box<dyn std::error::Error>> {
        let path = Path::new(model_path);

        // Load config
        println!("[Jamba] Loading config...");
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_str = super::sanitize_config_json(&config_str);
        let mut config: JambaConfig = serde_json::from_str(&config_str)?;
        config.post_init();
        println!(
            "[Jamba] Config loaded: {} layers ({} attention, {} mamba)",
            config.num_hidden_layers,
            config
                .get_layers_block_type()
                .iter()
                .filter(|t| *t == "attention")
                .count(),
            config
                .get_layers_block_type()
                .iter()
                .filter(|t| *t == "mamba")
                .count()
        );

        // Load weights
        println!("[Jamba] Loading weights from safetensors...");
        let weights = crate::models::load_text_weights(path, None)?;

        // Process weights
        let weights = Self::sanitize_weights(weights, &config);

        // Build model
        println!("[Jamba] Building model...");
        let model = Self::from_weights(config.clone(), weights)?;

        println!("[Jamba] Model loaded successfully");
        Ok((model, config))
    }

    fn sanitize_weights(mut weights: WeightMap, config: &JambaConfig) -> WeightMap {
        // Handle conv1d weight transpose
        let keys: Vec<String> = weights.keys().cloned().collect();
        for k in keys {
            if k.contains("conv1d.weight")
                && let Some(v) = weights.get(&k)
            {
                let shape = mlxcel_core::array_shape(v);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    let transposed = mlxcel_core::swap_axes(v, -1, -2);
                    weights.insert(k, transposed);
                }
            }
        }

        // Remove lm_head if tie_word_embeddings
        if config.tie_word_embeddings {
            weights.remove("lm_head.weight");
        }

        // Handle MoE weight conversion (experts.N -> switch_mlp)
        for l in 0..config.num_hidden_layers {
            let base = format!("model.layers.{}.feed_forward", l);
            let has_experts = weights
                .keys()
                .any(|k| k.starts_with(&format!("{}.experts.", base)));
            if !has_experts {
                continue;
            }

            for proj in ["gate_proj", "down_proj", "up_proj"] {
                let mut expert_tensors: Vec<UniquePtr<MlxArray>> = Vec::new();
                let mut e = 0;
                while let Some(w) =
                    weights.remove(&format!("{}.experts.{}.{}.weight", base, e, proj))
                {
                    expert_tensors.push(w);
                    e += 1;
                }
                if !expert_tensors.is_empty() {
                    let stacked = stack_arrays(&expert_tensors, 0);
                    weights.insert(format!("{}.switch_mlp.{}.weight", base, proj), stacked);
                }
            }
        }

        weights
    }

    pub fn from_weights(
        config: JambaConfig,
        mut weights: WeightMap,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let layers_block_type = config.get_layers_block_type();

        // Find first attention layer index
        let attn_idx = layers_block_type
            .iter()
            .position(|t| t == "attention")
            .unwrap_or(0);

        // Get quantization parameters
        let group_size = config.group_size();
        let bits = config.bits();

        // Build quantized embeddings
        let embed_tokens =
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", group_size, bits)?;

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (i, layer_type) in layers_block_type.iter().enumerate() {
            let prefix = format!("model.layers.{}", i);
            let is_attn = layer_type == "attention";

            // Build temporal block
            let temporal = if is_attn {
                let head_dim = config.get_head_dim();
                let qkv_proj = FusedQKVLinear::from_weights_separate(
                    &weights,
                    &format!("{}.self_attn", prefix),
                    group_size,
                    bits,
                    config.num_attention_heads as i32,
                    config.num_key_value_heads as i32,
                    head_dim as i32,
                )?;
                let o_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.self_attn.o_proj", prefix),
                    group_size,
                    bits,
                )?;

                TemporalBlock::Attention(JambaAttention {
                    qkv_proj,
                    o_proj,
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim,
                    scale: (head_dim as f32).powf(-0.5),
                })
            } else {
                let mamba_prefix = format!("{}.mamba", prefix);

                let in_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.in_proj", mamba_prefix),
                    group_size,
                    bits,
                )?;
                let out_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.out_proj", mamba_prefix),
                    group_size,
                    bits,
                )?;
                let x_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.x_proj", mamba_prefix),
                    group_size,
                    bits,
                )?;
                // dt_proj is NOT quantized in the model
                let dt_proj = load_linear(&mut weights, &format!("{}.dt_proj", mamba_prefix))?;

                let conv_weight = weights
                    .remove(&format!("{}.conv1d.weight", mamba_prefix))
                    .ok_or(format!("Missing conv1d weight for layer {}", i))?;
                let conv_bias = weights.remove(&format!("{}.conv1d.bias", mamba_prefix));

                let dt_norm_weight = weights
                    .remove(&format!("{}.dt_layernorm.weight", mamba_prefix))
                    .ok_or(format!("Missing dt_layernorm for layer {}", i))?;
                let b_norm_weight = weights
                    .remove(&format!("{}.b_layernorm.weight", mamba_prefix))
                    .ok_or(format!("Missing b_layernorm for layer {}", i))?;
                let c_norm_weight = weights
                    .remove(&format!("{}.c_layernorm.weight", mamba_prefix))
                    .ok_or(format!("Missing c_layernorm for layer {}", i))?;

                let a_log = weights
                    .remove(&format!("{}.A_log", mamba_prefix))
                    .ok_or(format!("Missing A_log for layer {}", i))?;
                let d_param = weights
                    .remove(&format!("{}.D", mamba_prefix))
                    .ok_or(format!("Missing D for layer {}", i))?;

                TemporalBlock::Mamba(JambaMambaMixer {
                    hidden_size: config.hidden_size,
                    intermediate_size: config.get_mamba_intermediate_size(),
                    ssm_state_size: config.mamba_d_state,
                    conv_kernel_size: config.mamba_d_conv,
                    time_step_rank: config.mamba_dt_rank,
                    in_proj,
                    conv_weight,
                    conv_bias,
                    x_proj,
                    dt_proj,
                    out_proj,
                    dt_layernorm: RMSNorm::new(dt_norm_weight, config.rms_norm_eps),
                    b_layernorm: RMSNorm::new(b_norm_weight, config.rms_norm_eps),
                    c_layernorm: RMSNorm::new(c_norm_weight, config.rms_norm_eps),
                    a_log,
                    d_param,
                })
            };

            // Build feed forward (MLP or MoE)
            let feed_forward = if config.is_expert_layer(i) {
                let router = load_linear(&mut weights, &format!("{}.feed_forward.router", prefix))?;

                let gate_weight = weights
                    .remove(&format!(
                        "{}.feed_forward.switch_mlp.gate_proj.weight",
                        prefix
                    ))
                    .ok_or(format!("Missing MoE gate_proj for layer {}", i))?;
                let up_weight = weights
                    .remove(&format!(
                        "{}.feed_forward.switch_mlp.up_proj.weight",
                        prefix
                    ))
                    .ok_or(format!("Missing MoE up_proj for layer {}", i))?;
                let down_weight = weights
                    .remove(&format!(
                        "{}.feed_forward.switch_mlp.down_proj.weight",
                        prefix
                    ))
                    .ok_or(format!("Missing MoE down_proj for layer {}", i))?;

                FeedForward::MoE(JambaSparseMoeBlock {
                    router,
                    switch_mlp: SwitchGLU {
                        gate_proj: SwitchLinear::Regular {
                            weight: gate_weight,
                        },
                        up_proj: SwitchLinear::Regular { weight: up_weight },
                        down_proj: SwitchLinear::Regular {
                            weight: down_weight,
                        },
                    },
                    num_experts_per_tok: config.num_experts_per_tok,
                })
            } else {
                let gate_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.feed_forward.gate_proj", prefix),
                    group_size,
                    bits,
                )?;
                let up_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.feed_forward.up_proj", prefix),
                    group_size,
                    bits,
                )?;
                let down_proj = UnifiedLinear::from_weights(
                    &weights,
                    &format!("{}.feed_forward.down_proj", prefix),
                    group_size,
                    bits,
                )?;

                FeedForward::MLP(JambaMLP {
                    gate_proj,
                    up_proj,
                    down_proj,
                })
            };

            // Build norms
            let input_norm_weight = weights
                .remove(&format!("{}.input_layernorm.weight", prefix))
                .ok_or(format!("Missing input_layernorm for layer {}", i))?;
            let pre_ff_norm_weight = weights
                .remove(&format!("{}.pre_ff_layernorm.weight", prefix))
                .ok_or(format!("Missing pre_ff_layernorm for layer {}", i))?;

            layers.push(JambaDecoderLayer {
                is_attn,
                temporal,
                feed_forward,
                input_layernorm: RMSNorm::new(input_norm_weight, config.rms_norm_eps),
                pre_ff_layernorm: RMSNorm::new(pre_ff_norm_weight, config.rms_norm_eps),
            });
        }

        // Final norm
        let final_norm_weight = weights
            .remove("model.final_layernorm.weight")
            .ok_or("Missing final_layernorm weight")?;
        let final_layernorm = RMSNorm::new(final_norm_weight, config.rms_norm_eps);

        // LM head (quantized if not tied)
        let lm_head = if !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                &weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        // Create internal caches for LanguageModel trait compatibility
        let internal_caches: Vec<JambaLayerCache> = layers_block_type
            .iter()
            .map(|t| {
                if t == "attention" {
                    JambaLayerCache::Attention(KVCache::new())
                } else {
                    JambaLayerCache::Mamba(JambaMambaCache::new())
                }
            })
            .collect();

        let model = JambaModelBackbone {
            embed_tokens,
            layers,
            final_layernorm,
            layers_block_type,
            attn_idx,
        };

        Ok(Self {
            config,
            model,
            lm_head,
            internal_caches: RefCell::new(internal_caches),
        })
    }
}

/// Helper to load a Linear layer from weights
fn load_linear(
    weights: &mut WeightMap,
    prefix: &str,
) -> Result<Linear, Box<dyn std::error::Error>> {
    let weight = weights
        .remove(&format!("{}.weight", prefix))
        .ok_or(format!("Missing weight for {}", prefix))?;
    let bias = weights.remove(&format!("{}.bias", prefix));
    Ok(Linear::new(weight, bias))
}

// LanguageModel trait implementation.
impl LanguageModel for JambaModel {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Jamba uses mixed cache types (KVCache + MambaCache)
        // We use internal RefCell caches to maintain state through shared reference
        let mut internal = self.internal_caches.borrow_mut();
        self.forward_with_caches(input, Some(&mut internal))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset internal caches
        *self.internal_caches.borrow_mut() = JambaModel::make_caches(self);
        // Return dummy KV caches for trait compatibility
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt Mamba recurrent state in hybrid architecture
    }

    fn supports_batching(&self) -> bool {
        false // Jamba is a hybrid Mamba+Transformer, internal MambaCache not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![519] // Jamba EOS token: <|im_end|>
    }
}

#[cfg(test)]
#[path = "jamba_tests.rs"]
mod tests;
