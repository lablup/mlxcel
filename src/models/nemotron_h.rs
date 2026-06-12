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

// Nemotron-H: NVIDIA's hybrid Mamba2+Transformer model for mlxcel-core
// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron_h.py
//
// Key features:
// - Hybrid architecture with configurable block types (M/*/−/E)
// - M = Mamba2 mixer, * = Attention, - = MLP, E = MoE
// - MambaRMSNormGated for Mamba blocks
// - Mixed cache: MambaCache for M blocks, KVCache for * blocks
// - relu^2 activation for MLP/MoE

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, Linear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, relu_squared, silu, slice_axis, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

// Configuration.
// Custom deserializer for hybrid_override_pattern which can be either:
// - A string like "MEMEM*EMEMEM*..." (each char is a block type)
// - A Vec<String> like ["M", "E", "M", ...]
// - null / absent (returns None)
fn deserialize_hybrid_pattern<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};

    struct PatternVisitor;

    impl<'de> Visitor<'de> for PatternVisitor {
        type Value = Option<Vec<String>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or array of strings for hybrid_override_pattern, or null")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // Convert string like "MEMEM*..." to Vec<String>
            Ok(Some(v.chars().map(|c| c.to_string()).collect()))
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while let Some(elem) = seq.next_element()? {
                vec.push(elem);
            }
            Ok(Some(vec))
        }
    }

    deserializer.deserialize_any(PatternVisitor)
}

fn deserialize_time_step_limit_opt<'de, D>(deserializer: D) -> Result<Option<(f32, f32)>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{SeqAccess, Visitor};

    struct TimeStepLimitOptVisitor;

    impl<'de> Visitor<'de> for TimeStepLimitOptVisitor {
        type Value = Option<(f32, f32)>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a tuple of two numbers, or null/absent")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
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
            Ok(Some((first, second)))
        }
    }

    deserializer.deserialize_any(TimeStepLimitOptVisitor)
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NemotronHConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default)]
    pub attention_bias: bool,

    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,

    #[serde(default)]
    pub mamba_proj_bias: bool,

    pub ssm_state_size: usize,
    pub conv_kernel: usize,
    pub n_groups: usize,

    #[serde(default, deserialize_with = "deserialize_time_step_limit_opt")]
    pub time_step_limit: Option<(f32, f32)>,

    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,

    #[serde(default)]
    pub use_bias: bool,

    #[serde(default = "default_true")]
    pub use_conv_bias: bool,

    #[serde(default, deserialize_with = "deserialize_hybrid_pattern")]
    pub hybrid_override_pattern: Option<Vec<String>>,

    /// Alternative to hybrid_override_pattern using word names
    /// (e.g., ["mamba", "attention", "moe", "mlp"]).
    /// Normalized to hybrid_override_pattern in post_init().
    #[serde(default)]
    pub layers_block_type: Option<Vec<String>>,

    #[serde(default)]
    pub head_dim: Option<usize>,

    // MoE parameters
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    #[serde(default)]
    pub moe_shared_expert_intermediate_size: Option<usize>,

    /// Latent size for MoE dimensionality reduction (NemotronSuper).
    /// When set, experts operate on the latent dim instead of hidden_size.
    #[serde(default)]
    pub moe_latent_size: Option<usize>,

    #[serde(default)]
    pub n_group: Option<usize>,

    #[serde(default)]
    pub n_routed_experts: Option<usize>,

    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub topk_group: Option<usize>,

    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,

    #[serde(default)]
    pub norm_topk_prob: Option<bool>,

    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,

    #[serde(default)]
    pub quantization: Option<Quantization>,

    /// Softplus initialization bounds retained for config-schema parity with
    /// upstream https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron_h.py. They do NOT participate in the
    /// `time_step_limit` default (which is always `(0.0, +inf)` when absent;
    /// see upstream PR #1026 / commit `6ddfdda`).
    #[serde(default)]
    pub time_step_min: Option<f32>,

    #[serde(default)]
    pub time_step_max: Option<f32>,
}

impl NemotronHConfig {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// Post-deserialization normalization (mirrors Python __post_init__).
    /// - Defaults time_step_limit to (0.0, +inf) when absent from the config.
    /// - Normalizes layers_block_type word names to single-char hybrid_override_pattern.
    /// - Sets num_hidden_layers from pattern length.
    pub fn post_init(&mut self) -> Result<(), String> {
        // When time_step_limit is absent from the config, default to (0.0, +inf).
        // This matches upstream mlx-lm PR #1026 (commit 6ddfdda):
        //   if time_step_limit is None: time_step_limit = (0.0, float("inf"))
        // time_step_min / time_step_max are softplus initialisation bounds used in
        // the forward path — they are NOT the dt clamp bounds.
        if self.time_step_limit.is_none() {
            self.time_step_limit = Some((0.0, f32::INFINITY));
        }

        // Normalize layers_block_type word names to single-char codes
        if self.hybrid_override_pattern.is_none()
            && let Some(ref block_types) = self.layers_block_type
        {
            let mut pattern = Vec::with_capacity(block_types.len());
            for t in block_types {
                let code = match t.as_str() {
                    "mamba" => "M",
                    "attention" => "*",
                    "moe" => "E",
                    "mlp" => "-",
                    other => {
                        return Err(format!("Unknown block type name: {other}"));
                    }
                };
                pattern.push(code.to_string());
            }
            self.hybrid_override_pattern = Some(pattern);
        }

        // Set num_hidden_layers from pattern length
        if let Some(ref pattern) = self.hybrid_override_pattern {
            self.num_hidden_layers = pattern.len();
        }

        Ok(())
    }
}

fn default_layer_norm_epsilon() -> f32 {
    1e-5
}

fn default_true() -> bool {
    true
}

impl NemotronHConfig {
    /// Returns the effective time_step_limit, falling back to (0.0, +inf) if post_init
    /// was not called (should not happen in normal usage).
    pub fn effective_time_step_limit(&self) -> (f32, f32) {
        self.time_step_limit.unwrap_or((0.0, f32::INFINITY))
    }

    pub fn get_mamba_intermediate_size(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    pub fn get_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn get_conv_dim(&self) -> usize {
        let intermediate_size = self.get_mamba_intermediate_size();
        intermediate_size + 2 * self.n_groups * self.ssm_state_size
    }
}

// Block Types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockType {
    Mamba,     // "M"
    Attention, // "*"
    MLP,       // "-"
    MoE,       // "E"
}

impl BlockType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "M" => BlockType::Mamba,
            "*" => BlockType::Attention,
            "-" => BlockType::MLP,
            "E" => BlockType::MoE,
            _ => panic!("Unknown block type: {}", s),
        }
    }

    pub fn needs_cache(&self) -> bool {
        matches!(self, BlockType::Mamba | BlockType::Attention)
    }
}

// Cache Types.
pub struct NemotronMambaCache {
    pub conv_state: Option<UniquePtr<MlxArray>>,
    pub ssm_state: Option<UniquePtr<MlxArray>>,
}

impl NemotronMambaCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
        }
    }
}

impl Default for NemotronMambaCache {
    fn default() -> Self {
        Self::new()
    }
}

pub enum NemotronLayerCache {
    Attention(KVCache),
    Mamba(NemotronMambaCache),
}

impl NemotronLayerCache {
    pub fn offset(&self) -> i32 {
        match self {
            NemotronLayerCache::Attention(kv) => kv.offset,
            NemotronLayerCache::Mamba(_) => 0,
        }
    }
}

// MambaRMSNormGated - RMS norm with gating and group structure.
struct MambaRMSNormGated {
    weight: UniquePtr<MlxArray>,
    eps: f32,
    group_size: usize,
}

impl MambaRMSNormGated {
    fn forward(&self, x: &MlxArray, gate: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let orig_dtype = mlxcel_core::array_dtype(x);

        // Promote to float32 for the entire gated norm computation to prevent
        // NaN from float16 overflow in RMS norm (x^2 sum) and mixed-dtype
        // multiply on M5 Max (Metal GPU Family 4) NAx kernels.
        let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let x = if let Some(g) = gate {
            let g_f32 = mlxcel_core::astype(g, mlxcel_core::dtype::FLOAT32);
            mlxcel_core::multiply(&x, &silu(&g_f32))
        } else {
            x
        };

        // Unflatten last dim into groups
        let shape = mlxcel_core::array_shape(&x);
        let ndim = shape.len();
        let last_dim = shape[ndim - 1] as usize;
        let num_groups = last_dim / self.group_size;

        let mut new_shape: Vec<i32> = shape[..ndim - 1].to_vec();
        new_shape.push(num_groups as i32);
        new_shape.push(self.group_size as i32);
        let x_grouped = mlxcel_core::reshape(&x, &new_shape);

        // Apply RMS norm per group in float32
        let group_weight =
            mlxcel_core::ones(&[self.group_size as i32], mlxcel_core::dtype::FLOAT32);
        let x_normed = mlxcel_core::fast_rms_norm(&x_grouped, &group_weight, self.eps);

        // Flatten back, apply weight, and cast back to original dtype
        let x_flat = mlxcel_core::reshape(&x_normed, &shape);
        let w_f32 = mlxcel_core::astype(&self.weight, mlxcel_core::dtype::FLOAT32);
        let result = mlxcel_core::multiply(&w_f32, &x_flat);
        mlxcel_core::astype(&result, orig_dtype)
    }
}

// Mamba2 Mixer for Nemotron-H.
struct NemotronHMamba2Mixer {
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

    conv_weight: UniquePtr<MlxArray>,
    conv_bias: Option<UniquePtr<MlxArray>>,
    in_proj: UnifiedLinear,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    d_param: UniquePtr<MlxArray>,
    norm: MambaRMSNormGated,
    out_proj: UnifiedLinear,
}

impl NemotronHMamba2Mixer {
    fn forward(
        &self,
        hidden_states: &MlxArray,
        mut cache: Option<&mut NemotronMambaCache>,
    ) -> UniquePtr<MlxArray> {
        // Use fused C++ forward for single-token decode with existing cache
        let shape = mlxcel_core::array_shape(hidden_states);
        let seq_len = shape[1];
        if seq_len == 1
            && mlxcel_core::ssm_kernel_available()
            && cache
                .as_ref()
                .is_some_and(|c| c.conv_state.is_some() && c.ssm_state.is_some())
        {
            return self.forward_fused(hidden_states, cache.as_deref_mut().unwrap());
        }

        let projected = self.in_proj.forward(hidden_states);

        // Split: gate, conv_input, dt
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

        // Conv with padding
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

        // Update conv cache.
        // Wrap slice in contiguous() to force MLX to materialize a fresh,
        // independent buffer. Without this, the slice is a lazy view that
        // retains a reference to the full padded_input allocation, causing a
        // memory leak proportional to the sequence length.
        if let Some(c) = cache.as_deref_mut() {
            let padded_shape = mlxcel_core::array_shape(&padded_input);
            let len = padded_shape[1] as usize;
            let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
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
            let b_reshaped = mlxcel_core::reshape(b, &[1, 1, -1]);
            mlxcel_core::add(&conv_out, &b_reshaped)
        } else {
            conv_out
        };

        let conv_output = silu(&conv_out);

        // Split conv output: hidden_states_ssm, B, C
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

        // SSM computation
        let ssm_state_ref = cache
            .as_ref()
            .and_then(|c| c.ssm_state.as_ref())
            .and_then(|s| s.as_ref());

        // Use fused Metal kernel for single-token decode when state is available
        let (y, new_state) = if seq_len == 1 && mlxcel_core::ssm_kernel_available() {
            if let Some(state) = ssm_state_ref {
                self.ssm_step_kernel(&hidden_ssm, &b, &c, &dt, state)
            } else {
                self.ssm_step(&hidden_ssm, &b, &c, &dt, ssm_state_ref)
            }
        } else {
            self.ssm_step(&hidden_ssm, &b, &c, &dt, ssm_state_ref)
        };

        // Update SSM state
        if let Some(c) = cache {
            c.ssm_state = Some(new_state);
        }

        // Reshape y back to [batch, seq, intermediate_size]
        let y = mlxcel_core::reshape(&y, &[batch, seq_len, self.intermediate_size as i32]);

        // Apply gated norm and output projection
        let y_normed = self.norm.forward(&y, Some(&gate));
        let result = self.out_proj.forward(&y_normed);

        // Force evaluation of the Mamba layer output.
        // On M5 Max (Metal GPU Family 4), the lazy computation graph from
        // the SSM step contains mixed float32×float16 intermediate nodes
        // that produce NaN when fused with downstream layers in a single
        // Metal command buffer.  Materializing here splits the graph at a
        // clean dtype boundary.
        mlxcel_core::eval(&result);

        result
    }

    /// SSM computation using the same approach as Mamba2's ssm_attn
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

        // Reshape inputs for multi-head processing
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

        // Compute dt with bias and limits in float32 for numerical precision
        // (same as Mamba2's compute_dt — upstream mlx-lm casts to float32 before softplus)
        let dt_f32 = mlxcel_core::astype(dt, mlxcel_core::dtype::FLOAT32);
        let dt_bias_f32 = mlxcel_core::astype(&self.dt_bias, mlxcel_core::dtype::FLOAT32);
        let dt_biased = mlxcel_core::add(&dt_f32, &dt_bias_f32);
        let dt_soft = mlxcel_core::softplus(&dt_biased);
        let min_val =
            mlxcel_core::full_f32(&[1], self.time_step_limit.0, mlxcel_core::dtype::FLOAT32);
        let max_val =
            mlxcel_core::full_f32(&[1], self.time_step_limit.1, mlxcel_core::dtype::FLOAT32);
        let dt = mlxcel_core::clip(&dt_soft, &min_val, &max_val);

        // A = -exp(A_log), cast to dt dtype (Python does .astype(dt.dtype);
        // since dt is now float32, A stays in float32 for precision)
        let a = mlxcel_core::negative(&mlxcel_core::exp(&self.a_log));
        let a = mlxcel_core::astype(&a, mlxcel_core::array_dtype(&dt));
        let a_reshaped = mlxcel_core::reshape(&a, &[1, 1, num_heads]);

        // dtA = dt * A
        let dt_a = mlxcel_core::multiply(&dt, &a_reshaped);

        // dtx = dt * x (expand dt for broadcasting)
        // Promote x to float32 before multiplication with dt (float32).
        // On M5 Max (Metal GPU Family 4), mixed float32×float16 multiply
        // in a lazy computation graph produces NaN via the NAx broadcast kernel.
        let dt_exp = mlxcel_core::reshape(&dt, &[batch, seq_len, num_heads, 1]);
        let x_f32 = mlxcel_core::astype(&x, mlxcel_core::dtype::FLOAT32);
        let dtx = mlxcel_core::multiply(&dt_exp, &x_f32);

        // Promote B, C to float32 before CB matmul to prevent float16 overflow
        // on M5 Max (Metal GPU Family 4) NAx GEMM kernel.  The dot products
        // over ssm_state_size elements can exceed float16 max (65504) for
        // certain weight distributions, producing NaN that propagates through
        // the entire model and causes all-<unk> output.
        let b_f32 = mlxcel_core::astype(&b, mlxcel_core::dtype::FLOAT32);
        let c_f32 = mlxcel_core::astype(&c, mlxcel_core::dtype::FLOAT32);

        // B: [batch, seq, groups, state_dim] -> [batch, groups, state_dim, seq]
        let b_t = mlxcel_core::transpose_axes(&b_f32, &[0, 2, 3, 1]);

        // CB = C.swapaxes(1, 2) @ B (in float32 to avoid overflow)
        let c_t = mlxcel_core::swap_axes(&c_f32, 1, 2);
        let cb = mlxcel_core::matmul(&c_t, &b_t);

        // Repeat CB for each head in group
        let cb = repeat_axis_nemotron(&cb, repeats, 1);

        // decay = exp(segsum(dtA.swapaxes(1, 2)))
        let dt_a_t = mlxcel_core::swap_axes(&dt_a, 1, 2);
        let seg = segsum_nemotron(&dt_a_t);
        let decay = mlxcel_core::exp(&seg);

        // surrogate_attention_matrix = tril(CB * decay)
        let attn_matrix = mlxcel_core::multiply(&cb, &decay);
        let attn_matrix = mlxcel_core::tril(&attn_matrix, 0);

        // y = attn_matrix @ dtx.swapaxes(1, 2)
        let dtx_t = mlxcel_core::swap_axes(&dtx, 1, 2);
        let y = mlxcel_core::matmul(&attn_matrix, &dtx_t);
        let y = mlxcel_core::swap_axes(&y, 1, 2);

        // Compute next state
        // decay_last = decay[:, :, -1:, :]
        let decay_shape = mlxcel_core::array_shape(&decay);
        let decay_last = slice_axis(&decay, 2, decay_shape[2] - 1, decay_shape[2]);
        let decay_t = mlxcel_core::transpose_axes(&decay_last, &[0, 3, 1, 2]);

        // B for state update (already float32 from promotion above)
        let b_rep = repeat_axis_nemotron(&b_t, repeats, 1);
        let b_sw = mlxcel_core::swap_axes(&b_rep, 2, 3);

        let dtx_decay = mlxcel_core::multiply(&dtx, &decay_t);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay, 1, 2);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay_t, 2, 3);

        let mut next_state = mlxcel_core::matmul(&dtx_decay_t, &b_sw);

        // Handle previous state if present (using Mamba2's approach)
        let y = if let Some(prev_state) = state {
            // exp_dtA_cumsum = exp(cumsum(dtA, axis=-2))
            // Python mx.cumsum defaults to inclusive; match mamba2.rs ssm_step
            let dta_cumsum = mlxcel_core::cumsum(&dt_a, -2, false, true);
            let exp_dta_cumsum = mlxcel_core::exp(&dta_cumsum);

            // Update next_state with previous state
            // Python: exp_dtA_cumsum[:, -1, :, None, None] → [batch, heads, 1, 1]
            let exp_shape = mlxcel_core::array_shape(&exp_dta_cumsum);
            let last_exp = slice_axis(&exp_dta_cumsum, 1, exp_shape[1] - 1, exp_shape[1]);
            // last_exp is [batch, 1, num_heads] — squeeze seq dim, add two trailing dims
            let last_exp = mlxcel_core::reshape(&last_exp, &[batch, num_heads, 1, 1]);

            let state_contrib = mlxcel_core::multiply(&last_exp, prev_state);
            next_state = mlxcel_core::add(&next_state, &state_contrib);

            // y_prev contribution (use float32 C for precision with float32 state)
            let c_reshaped =
                mlxcel_core::reshape(&c_f32, &[batch, seq_len, n_groups, 1, state_dim, 1]);
            let state_reshaped = mlxcel_core::reshape(
                prev_state,
                &[batch, 1, n_groups, repeats, head_dim, state_dim],
            );
            let y_prev = mlxcel_core::matmul(&state_reshaped, &c_reshaped);
            let y_prev = mlxcel_core::squeeze_axis(&y_prev, -1);
            let y_prev = mlxcel_core::reshape(&y_prev, &[batch, seq_len, num_heads, head_dim]);

            // Expand exp_dta_cumsum for y_prev multiplication
            let mut exp_shape_y = mlxcel_core::array_shape(&exp_dta_cumsum);
            exp_shape_y.push(1);
            let exp_dta_exp = mlxcel_core::reshape(&exp_dta_cumsum, &exp_shape_y);

            let y_prev_scaled = mlxcel_core::multiply(&exp_dta_exp, &y_prev);
            mlxcel_core::add(&y, &y_prev_scaled)
        } else {
            y
        };

        // Add D term: y = y + x * D (all in float32 to prevent mixed-dtype NaN
        // on M5 Max NAx kernel — keep D and x in float32 matching y's dtype)
        let d_f32 = mlxcel_core::astype(&self.d_param, mlxcel_core::dtype::FLOAT32);
        let d_reshaped = mlxcel_core::reshape(&d_f32, &[1, 1, num_heads, 1]);
        let d_contrib = mlxcel_core::multiply(&x_f32, &d_reshaped);
        let y = mlxcel_core::add(&y, &d_contrib);

        // Cast y back to input dtype (dt computation was promoted to float32)
        let y = mlxcel_core::astype(&y, mlxcel_core::array_dtype(hidden_states));

        // Force evaluation of y and next_state before returning.
        // On M5 Max (Metal GPU Family 4), the lazy computation graph from the
        // SSM step contains mixed float32×float16 intermediate nodes.  When
        // this graph is fused with downstream operations (gated norm, residual
        // add, MoE, attention) in a single Metal command buffer, the NAx
        // kernel produces NaN.  Materializing the SSM outputs here splits
        // the graph at a clean float16 boundary.
        mlxcel_core::eval(&y);
        mlxcel_core::eval(&next_state);

        (y, next_state)
    }

    /// Fused SSM step using Metal kernel (single-token decode only)
    /// Replaces ~55 individual FFI calls with a single Metal kernel invocation
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

        // Reshape inputs for multi-head processing (matching Python's shapes)
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

    /// Fully fused Mamba2 forward for single-token decode.
    /// Combines in_proj + conv + SSM kernel + gated norm + out_proj into one C++ call.
    fn forward_fused(
        &self,
        hidden_states: &MlxArray,
        cache: &mut NemotronMambaCache,
    ) -> UniquePtr<MlxArray> {
        let conv_state = cache.conv_state.as_ref().unwrap();
        let ssm_state = cache.ssm_state.as_ref().unwrap();

        let conv_bias_ptr = self
            .conv_bias
            .as_ref()
            .map(|b| b.as_ref().unwrap() as *const MlxArray)
            .unwrap_or(std::ptr::null());

        let mut output = mlxcel_core::UniquePtr::null();
        let mut new_conv_state = mlxcel_core::UniquePtr::null();
        let mut new_ssm_state = mlxcel_core::UniquePtr::null();

        // Extract weight references from UnifiedLinear quantized variants
        let (ip_w, ip_s, ip_b) = match &self.in_proj {
            UnifiedLinear::Quantized { weight, .. } => {
                (&weight.weight, &weight.scales, weight.biases_ptr())
            }
            _ => panic!("fused_mamba2_forward requires quantized in_proj"),
        };
        let (op_w, op_s, op_b) = match &self.out_proj {
            UnifiedLinear::Quantized { weight, .. } => {
                (&weight.weight, &weight.scales, weight.biases_ptr())
            }
            _ => panic!("fused_mamba2_forward requires quantized out_proj"),
        };

        unsafe {
            mlxcel_core::fused_mamba2_forward(
                hidden_states,
                // in_proj
                ip_w,
                ip_s,
                ip_b,
                // conv
                &self.conv_weight,
                conv_bias_ptr,
                // SSM params
                &self.a_log,
                &self.d_param,
                &self.dt_bias,
                // norm
                &self.norm.weight,
                // out_proj
                op_w,
                op_s,
                op_b,
                // cache
                conv_state.as_ref().unwrap(),
                ssm_state.as_ref().unwrap(),
                // config
                self.intermediate_size as i32,
                self.conv_dim as i32,
                self.conv_kernel_size as i32,
                self.num_heads as i32,
                self.head_dim as i32,
                self.n_groups as i32,
                self.ssm_state_size as i32,
                self.time_step_limit.0,
                self.time_step_limit.1,
                self.norm.eps,
                self.in_proj.group_size(),
                self.in_proj.bits(),
                // outputs
                &mut output,
                &mut new_conv_state,
                &mut new_ssm_state,
            );
        }

        // The C++ fused kernel builds new_conv_state via slice(padded_input, ...)
        // which is a lazy alias. Wrap with contiguous() to materialize a compact
        // buffer and drop the upstream padded_input graph reference.
        cache.conv_state = Some(mlxcel_core::contiguous(&new_conv_state, false));
        cache.ssm_state = Some(new_ssm_state);

        output
    }
}

/// Repeat array along axis by broadcasting (for Nemotron)
fn repeat_axis_nemotron(x: &MlxArray, repeats: i32, axis: i32) -> UniquePtr<MlxArray> {
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

/// Segmented cumulative sum for SSM attention (for Nemotron)
fn segsum_nemotron(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let l = shape[shape.len() - 1];

    let mut new_shape = shape.clone();
    new_shape.push(1);
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    let last_idx = new_shape.len() - 1;
    new_shape[last_idx] = l;
    let x_rep = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    let x_tril = mlxcel_core::tril(&x_rep, -1);
    // Inclusive cumsum matching mamba2.rs segsum and Python mx.cumsum default
    mlxcel_core::cumsum(&x_tril, -2, false, true)
}

// Attention.
struct NemotronHAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl NemotronHAttention {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<&mut KVCache>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        let queries = self.q_proj.forward(x);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);

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

        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(keys, values)
        } else {
            (keys, values)
        };

        // Scaled dot-product attention (MLX fast kernel, handles GQA internally)
        let mask_ptr = mask
            .map(|m| m as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &queries, &keys, &values, self.scale, mask_ptr, 0.0, 0,
            )
        };

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[batch, seq_len, -1]);

        self.o_proj.forward(&output)
    }
}

// MLP with relu^2 activation.
struct NemotronHMLP {
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl NemotronHMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let up = self.up_proj.forward(x);
        let relu_sq = relu_squared(&up);
        self.down_proj.forward(&relu_sq)
    }
}

/// Helper to extract raw weight/scales/biases pointers from UnifiedLinear
/// Returns null pointers for non-quantized linear layers
trait QuantizedWeightPtrs {
    fn weight_ptr(&self) -> *const MlxArray;
    fn scales_ptr(&self) -> *const MlxArray;
    fn biases_ptr(&self) -> *const MlxArray;
}

impl QuantizedWeightPtrs for UnifiedLinear {
    fn weight_ptr(&self) -> *const MlxArray {
        match self {
            UnifiedLinear::Quantized { weight, .. } => {
                weight.weight.as_ref().unwrap() as *const MlxArray
            }
            UnifiedLinear::Regular(l) => l.weight.as_ref().unwrap() as *const MlxArray,
        }
    }
    fn scales_ptr(&self) -> *const MlxArray {
        match self {
            UnifiedLinear::Quantized { weight, .. } => {
                weight.scales.as_ref().unwrap() as *const MlxArray
            }
            _ => std::ptr::null(),
        }
    }
    fn biases_ptr(&self) -> *const MlxArray {
        match self {
            UnifiedLinear::Quantized { weight, .. } => weight.biases_ptr(),
            _ => std::ptr::null(),
        }
    }
}

trait QuantConfigAccessors {
    fn group_size(&self) -> i32;
    fn bits(&self) -> i32;
}

impl QuantConfigAccessors for UnifiedLinear {
    fn group_size(&self) -> i32 {
        match self {
            UnifiedLinear::Quantized { weight, .. } => weight.group_size,
            _ => 64,
        }
    }
    fn bits(&self) -> i32 {
        match self {
            UnifiedLinear::Quantized { weight, .. } => weight.bits,
            _ => 4,
        }
    }
}

/// Extract (weight, scales, biases) raw pointers from a QuantizedSwitchLinear.
/// Returns null pointers for the absent fields in the Regular variant.
fn switch_linear_ptrs(
    sl: &QuantizedSwitchLinear,
) -> (*const MlxArray, *const MlxArray, *const MlxArray) {
    match sl {
        QuantizedSwitchLinear::Quantized {
            weight,
            scales,
            biases,
            ..
        } => (
            weight.as_ref().unwrap() as *const MlxArray,
            scales.as_ref().unwrap() as *const MlxArray,
            biases.as_ref().unwrap() as *const MlxArray,
        ),
        QuantizedSwitchLinear::Regular { weight } => (
            weight.as_ref().unwrap() as *const MlxArray,
            std::ptr::null(),
            std::ptr::null(),
        ),
    }
}

// QuantizedSwitchLinear and SwitchMLP for MoE (using gather_qmm).
// Kept local: uses weights.remove() (ownership), different naming (QuantizedSwitchLinear)
enum QuantizedSwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>, // [num_experts, out_features, packed_in_features]
        scales: UniquePtr<MlxArray>, // [num_experts, out_features, groups]
        biases: UniquePtr<MlxArray>, // [num_experts, out_features, groups]
        group_size: i32,
        bits: i32,
    },
    Regular {
        weight: UniquePtr<MlxArray>, // [num_experts, out_features, in_features]
    },
}

impl QuantizedSwitchLinear {
    fn forward(
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
            } => {
                let biases_ptr = biases.as_ref().unwrap() as *const MlxArray;
                let indices_ptr = indices as *const MlxArray;
                unsafe {
                    mlxcel_core::gather_qmm(
                        x,
                        weight,
                        scales,
                        biases_ptr,
                        std::ptr::null(),
                        indices_ptr,
                        true,
                        *group_size,
                        *bits,
                        sorted_indices,
                        "affine",
                    )
                }
            }
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
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

struct SwitchMLP {
    fc1: QuantizedSwitchLinear,
    fc2: QuantizedSwitchLinear,
}

impl SwitchMLP {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        // x: [tokens, hidden], indices: [tokens, top_k]
        // Expand x to [tokens, 1, 1, hidden] for gather_qmm
        let x_shape = mlxcel_core::array_shape(x);
        let x_exp = mlxcel_core::reshape(x, &[x_shape[0], 1, 1, x_shape[1]]);

        // Note: sorted_indices optimization skipped for simplicity
        let sorted = false;

        let h = self.fc1.forward(&x_exp, indices, sorted);
        let h = relu_squared(&h);
        let out = self.fc2.forward(&h, indices, sorted);

        // Squeeze middle dimension: [tokens, top_k, 1, hidden] -> [tokens, top_k, hidden]
        mlxcel_core::squeeze_axis(&out, -2)
    }
}

// MoE Gate and Block.
struct NemotronHMoEGate {
    weight: UniquePtr<MlxArray>,
    e_score_correction_bias: UniquePtr<MlxArray>,
    top_k: usize,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
}

impl NemotronHMoEGate {
    fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // Gate projection: x @ W^T
        let w_t = mlxcel_core::transpose(&self.weight);
        let gates = mlxcel_core::matmul(x, &w_t);

        // Use fused C++ gate function (matches Python @mx.compile group_expert_select)
        let mut indices = mlxcel_core::UniquePtr::null();
        let mut scores = mlxcel_core::UniquePtr::null();
        mlxcel_core::compiled_moe_gate(
            &gates,
            &self.e_score_correction_bias,
            self.top_k as i32,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            &mut indices,
            &mut scores,
        );

        (indices, scores)
    }
}

struct NemotronHMoE {
    gate: NemotronHMoEGate,
    switch_mlp: SwitchMLP,
    shared_experts: Option<NemotronHMLP>,
    /// Latent projection: hidden_size -> moe_latent_size (NemotronSuper)
    fc1_latent_proj: Option<UnifiedLinear>,
    /// Latent projection: moe_latent_size -> hidden_size (NemotronSuper)
    fc2_latent_proj: Option<UnifiedLinear>,
}

impl NemotronHMoE {
    /// Returns true when latent projection is active (NemotronSuper MoE).
    fn has_latent_proj(&self) -> bool {
        self.fc1_latent_proj.is_some()
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);

        // Flatten to [n_tokens, hidden]
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, orig_shape[orig_shape.len() - 1]])
        } else {
            mlxcel_core::copy(x)
        };

        let use_fused = !self.has_latent_proj();

        // Try fused MoE forward (quantized path only, no latent projection)
        let result = if use_fused {
            if let (
                QuantizedSwitchLinear::Quantized {
                    weight: fc1_w,
                    scales: fc1_s,
                    biases: fc1_b,
                    group_size,
                    bits,
                },
                QuantizedSwitchLinear::Quantized {
                    weight: fc2_w,
                    scales: fc2_s,
                    biases: fc2_b,
                    ..
                },
            ) = (&self.switch_mlp.fc1, &self.switch_mlp.fc2)
            {
                // Get shared expert weight pointers (nullable)
                let (sup_w, sup_s, sup_b, sdn_w, sdn_s, sdn_b) =
                    if let Some(ref shared) = self.shared_experts {
                        (
                            shared.up_proj.weight_ptr(),
                            shared.up_proj.scales_ptr(),
                            shared.up_proj.biases_ptr(),
                            shared.down_proj.weight_ptr(),
                            shared.down_proj.scales_ptr(),
                            shared.down_proj.biases_ptr(),
                        )
                    } else {
                        (
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                        )
                    };

                unsafe {
                    mlxcel_core::fused_moe_forward(
                        &x_flat,
                        &self.gate.weight,
                        &self.gate.e_score_correction_bias,
                        fc1_w,
                        fc1_s,
                        fc1_b,
                        fc2_w,
                        fc2_s,
                        fc2_b,
                        sup_w,
                        sup_s,
                        sup_b,
                        sdn_w,
                        sdn_s,
                        sdn_b,
                        self.gate.top_k as i32,
                        self.gate.routed_scaling_factor,
                        self.gate.norm_topk_prob,
                        *group_size,
                        *bits,
                    )
                }
            } else {
                // Non-quantized without latent projection
                self.forward_nonfused(&x_flat)
            }
        } else {
            // Latent projection path (always non-fused)
            self.forward_nonfused(&x_flat)
        };

        // Reshape back
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    /// Non-fused MoE forward with latent projection support.
    /// CRITICAL: shared expert uses original residuals (not latent-projected input).
    fn forward_nonfused(&self, x_flat: &MlxArray) -> UniquePtr<MlxArray> {
        // Gate always operates on the original hidden dim
        let (indices, scores) = self.gate.forward(x_flat);

        // Apply latent projection before routing to experts (skip copy when
        // no projection -- switch_mlp.forward takes &MlxArray)
        let projected;
        let expert_input: &MlxArray = if let Some(ref fc1) = self.fc1_latent_proj {
            projected = fc1.forward(x_flat);
            &projected
        } else {
            x_flat
        };

        let y = self.switch_mlp.forward(expert_input, &indices);
        let mut scores_shape = mlxcel_core::array_shape(&scores);
        scores_shape.push(1);
        let scores_exp = mlxcel_core::reshape(&scores, &scores_shape);
        // Cast scores to y's dtype to avoid mixed float32×float16 multiply
        // which produces NaN on M5 Max (Metal GPU Family 4) NAx kernel.
        let scores_cast = mlxcel_core::astype(&scores_exp, mlxcel_core::array_dtype(&y));
        let weighted = mlxcel_core::multiply(&y, &scores_cast);
        let summed = mlxcel_core::sum_axis(&weighted, -2, false);

        // Cast back to input dtype (matching Python: .astype(y.dtype))
        let mut result = mlxcel_core::astype(&summed, mlxcel_core::array_dtype(x_flat));

        // Project back from latent to hidden dim
        if let Some(ref fc2) = self.fc2_latent_proj {
            result = fc2.forward(&result);
        }

        // Shared expert uses original residuals (not latent-projected input)
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x_flat);
            result = mlxcel_core::add(&result, &shared_out);
        }

        result
    }
}

// Hybrid Block.
enum NemotronHMixer {
    Mamba(NemotronHMamba2Mixer),
    Attention(NemotronHAttention),
    MLP(NemotronHMLP),
    MoE(NemotronHMoE),
}

#[allow(dead_code)]
struct NemotronHBlock {
    norm: RMSNorm,
    block_type: BlockType,
    mixer: NemotronHMixer,
}

impl NemotronHBlock {
    fn forward(
        &self,
        x: &MlxArray,
        attn_mask: Option<&MlxArray>,
        mamba_cache: Option<&mut NemotronMambaCache>,
        kv_cache: Option<&mut KVCache>,
    ) -> UniquePtr<MlxArray> {
        let h = self.norm.forward(x);

        let out = match &self.mixer {
            NemotronHMixer::Mamba(m) => m.forward(&h, mamba_cache),
            NemotronHMixer::Attention(a) => a.forward(&h, attn_mask, kv_cache),
            NemotronHMixer::MLP(m) => m.forward(&h),
            NemotronHMixer::MoE(m) => m.forward(&h),
        };

        mlxcel_core::add(x, &out)
    }
}

// Full Model.
use std::cell::RefCell;

pub struct NemotronHModel {
    config: NemotronHConfig,
    embeddings: UnifiedEmbedding,
    layers: Vec<NemotronHBlock>,
    norm_f: RMSNorm,
    lm_head: UnifiedLinear,
    block_types: Vec<BlockType>,
    /// Internal caches for LanguageModel trait compatibility
    internal_caches: RefCell<Vec<NemotronLayerCache>>,
    /// Opaque C++ handle for full-model decode (0 = not registered).
    /// When non-zero, `forward_with_caches` uses `nemotron_decode_step`
    /// for single-token decode instead of the Rust per-layer loop.
    c_handle: u64,
}

impl Drop for NemotronHModel {
    fn drop(&mut self) {
        if self.c_handle != 0 {
            mlxcel_core::nemotron_free_model(self.c_handle);
            self.c_handle = 0;
        }
    }
}

impl NemotronHModel {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<NemotronLayerCache> {
        self.block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .map(|bt| match bt {
                BlockType::Mamba => NemotronLayerCache::Mamba(NemotronMambaCache::new()),
                BlockType::Attention => NemotronLayerCache::Attention(KVCache::new()),
                _ => unreachable!(),
            })
            .collect()
    }

    /// Number of layers in the model that need a persistent per-token cache
    /// (i.e. Mamba or Attention blocks). MLP blocks are stateless.
    ///
    /// Used by: Nemotron-H pipeline stage executor
    pub fn num_cache_layers(&self) -> usize {
        self.block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .count()
    }

    /// Return whether the `i`-th layer in the model stack carries a
    /// persistent cache (Mamba / Attention) or is stateless (MLP / MoE).
    ///
    /// Used by: Nemotron-H pipeline stage executor
    pub fn layer_needs_cache(&self, layer_idx: usize) -> bool {
        self.block_types
            .get(layer_idx)
            .is_some_and(|bt| bt.needs_cache())
    }

    /// Build a single [`NemotronLayerCache`] for the layer at `layer_idx`.
    /// Callers that want caches for a pipeline stage's layer range can
    /// invoke this once per stateful layer.
    ///
    /// Used by: Nemotron-H pipeline stage executor
    pub fn make_layer_cache(&self, layer_idx: usize) -> Option<NemotronLayerCache> {
        match self.block_types.get(layer_idx)? {
            BlockType::Mamba => Some(NemotronLayerCache::Mamba(NemotronMambaCache::new())),
            BlockType::Attention => Some(NemotronLayerCache::Attention(KVCache::new())),
            _ => None,
        }
    }

    /// Run a pipeline-stage forward pass that only touches layers in
    /// `layer_range`. See `JambaModel::forward_stage` for the semantics of
    /// `has_embedding` and `has_lm_head`.
    ///
    /// Caches: one entry per stateful layer inside `layer_range`, in layer
    /// order. The caller is responsible for persisting these across
    /// sequence-step boundaries (see the stage executor's
    /// `PointerOwnedCacheStore`).
    ///
    /// SSM design note — Mamba / Attention / MLP / MoE blocks are executed
    /// in full on the stage that owns them. Splitting a single Mamba block
    /// across stages is not permitted: the conv and SSM state are
    /// per-block and per-token, so the entire block must run on one
    /// device.
    ///
    /// Used by: Nemotron-H pipeline stage executor
    pub fn forward_stage(
        &self,
        inputs: &MlxArray,
        layer_range: std::ops::Range<usize>,
        has_embedding: bool,
        has_lm_head: bool,
        caches: &mut [NemotronLayerCache],
    ) -> UniquePtr<MlxArray> {
        let expected_caches: usize = layer_range
            .clone()
            .filter(|&abs| self.layer_needs_cache(abs))
            .count();
        assert_eq!(
            caches.len(),
            expected_caches,
            "nemotron_h forward_stage: cache count mismatch (expected {}, got {})",
            expected_caches,
            caches.len()
        );

        let mut h = if has_embedding {
            self.embeddings.forward(inputs)
        } else {
            mlxcel_core::copy(inputs)
        };

        // Anchor attention mask on the first attention layer in this stage.
        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];
        let first_attn_local_cache_idx = {
            let mut local_cache_idx = 0usize;
            let mut found = None;
            for &abs in &layer_range.clone().collect::<Vec<_>>() {
                if self.layer_needs_cache(abs) {
                    if self.block_types.get(abs) == Some(&BlockType::Attention) {
                        found = Some(local_cache_idx);
                        break;
                    }
                    local_cache_idx += 1;
                }
            }
            found
        };
        let attn_mask = if seq_len > 1 {
            let offset = first_attn_local_cache_idx
                .map(|idx| caches[idx].offset())
                .unwrap_or(0);
            Some(create_causal_mask(seq_len, offset))
        } else {
            None
        };

        let mut cache_idx = 0usize;
        for abs in layer_range.clone() {
            let layer = &self.layers[abs];
            let block_type = self.block_types[abs];
            if block_type.needs_cache() {
                let cache_entry = &mut caches[cache_idx];
                match (block_type, cache_entry) {
                    (BlockType::Mamba, NemotronLayerCache::Mamba(mc)) => {
                        h = layer.forward(&h, None, Some(mc), None);
                    }
                    (BlockType::Attention, NemotronLayerCache::Attention(kv)) => {
                        h = layer.forward(&h, attn_mask.as_deref(), None, Some(kv));
                    }
                    _ => {}
                }
                cache_idx += 1;
            } else {
                h = layer.forward(&h, None, None, None);
            }
        }

        if has_lm_head {
            let h = self.norm_f.forward(&h);
            self.lm_head.forward(&h)
        } else {
            h
        }
    }

    /// Look up token embeddings without running the rest of the model.
    /// The VLM path needs raw embeddings to merge image features at
    /// `<image>` token positions before running the layer stack.
    ///
    /// Used by: Nemotron H Nano Omni VLM
    pub fn input_embeddings(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embeddings.forward(input_ids)
    }

    /// Hidden size of the text backbone, useful for shape sanity checks
    /// in VLM glue code.
    ///
    /// Used by: Nemotron H Nano Omni VLM
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn forward_with_caches(
        &self,
        inputs: &MlxArray,
        caches: &mut [NemotronLayerCache],
    ) -> UniquePtr<MlxArray> {
        // Fast path: single-token decode via C++ full-forward when handle is active.
        // All Mamba layers are fused into one C++ call; attention KV is supplied from
        // (and updated back into) the Rust caches after the call.
        let shape = mlxcel_core::array_shape(inputs);
        let _seq_len = shape[1];
        // C++ full-forward decode available but disabled: hypothesis test showed
        // no speedup (19.2ms C++ vs 18.9ms Rust) — the bottleneck is GPU ops,
        // not graph build CPU overhead.
        // To re-enable: uncomment the block below.
        /*
        if seq_len == 1 && self.c_handle != 0 {
            let all_mamba_ready = caches.iter().all(|c| match c {
                NemotronLayerCache::Mamba(mc) => mc.conv_state.is_some() && mc.ssm_state.is_some(),
                NemotronLayerCache::Attention(_) => true,
            });
            if all_mamba_ready {
                return self.forward_decode_cpp(inputs, caches);
            }
        }
        */

        // Standard Rust path (prefill or fallback).
        if std::env::var("MLXCEL_PROFILE_FORWARD").is_ok() && shape[1] == 1 {
            self.forward_profiled(inputs, caches)
        } else {
            self.forward_rust(inputs, caches)
        }
    }

    /// Full-model single-token decode via `nemotron_decode_step`.
    ///
    /// The C++ kernel handles the embedding lookup, all 52 layers, and the
    /// lm_head projection in one call.  Mamba conv/ssm states are updated from
    /// the returned output arrays.  Attention KV caches are grown on the Rust
    /// side by running only the attention layers' norm + k/v projections using
    /// the post-decode hidden state from a lightweight Rust pass.
    ///
    /// # Known limitation
    ///
    /// The Rust-side KV update below requires per-attention-layer hidden states
    /// which are not returned by `nemotron_decode_step`.  Until C++ exposes KV
    /// output parameters, we run the full Rust forward after the C++ call and
    /// use *its* KV updates.  This doubles compute but keeps both paths
    /// consistent for the hypothesis / correctness test.  Once the C++ API is
    /// extended to return updated KV arrays, the Rust forward can be removed.
    // Retained for hypothesis testing; not called in production paths.
    #[allow(dead_code)]
    fn forward_decode_cpp(
        &self,
        inputs: &MlxArray,
        caches: &mut [NemotronLayerCache],
    ) -> UniquePtr<MlxArray> {
        // -----------------------------------------------------------------------
        // Hypothesis-test execution order:
        //
        //   1. Copy pre-step mamba and KV arrays into owned UniquePtr snapshots.
        //      This prevents use-after-free: the Rust forward will drop the old
        //      arrays when it writes new state into the caches.
        //   2. Run `nemotron_decode_step` using the snapshots → C++ logits.
        //   3. Run the Rust forward pass to advance all caches (authoritative).
        //      Rust logits are discarded; C++ logits are returned.
        //
        // TODO: once nemotron_decode_step exposes KV output parameters we can
        //   skip the Rust forward entirely and write C++ outputs into the caches.
        // -----------------------------------------------------------------------

        let mamba_count = caches
            .iter()
            .filter(|c| matches!(c, NemotronLayerCache::Mamba(_)))
            .count();
        let attn_count = caches
            .iter()
            .filter(|c| matches!(c, NemotronLayerCache::Attention(_)))
            .count();

        // --- 1. Own snapshots of pre-step arrays so pointers stay valid after
        //        the Rust forward (which replaces arrays in-place) ---
        //
        // MLX arrays are ref-counted on the C++ side; `mlxcel_core::copy` creates
        // a shallow copy (same backing buffer, incremented refcount), so this is
        // cheap – it does NOT duplicate tensor data.
        let mamba_conv_snap: Vec<UniquePtr<MlxArray>> = caches
            .iter()
            .filter_map(|c| {
                if let NemotronLayerCache::Mamba(mc) = c {
                    mc.conv_state
                        .as_ref()
                        .map(|s| mlxcel_core::copy(s.as_ref().unwrap()))
                } else {
                    None
                }
            })
            .collect();
        let mamba_ssm_snap: Vec<UniquePtr<MlxArray>> = caches
            .iter()
            .filter_map(|c| {
                if let NemotronLayerCache::Mamba(mc) = c {
                    mc.ssm_state
                        .as_ref()
                        .map(|s| mlxcel_core::copy(s.as_ref().unwrap()))
                } else {
                    None
                }
            })
            .collect();

        let attn_keys_snap: Vec<UniquePtr<MlxArray>> = caches
            .iter()
            .filter_map(|c| {
                if let NemotronLayerCache::Attention(kv) = c {
                    kv.keys
                        .as_ref()
                        .map(|k| mlxcel_core::copy(k.as_ref().unwrap()))
                } else {
                    None
                }
            })
            .collect();
        let attn_values_snap: Vec<UniquePtr<MlxArray>> = caches
            .iter()
            .filter_map(|c| {
                if let NemotronLayerCache::Attention(kv) = c {
                    kv.values
                        .as_ref()
                        .map(|v| mlxcel_core::copy(v.as_ref().unwrap()))
                } else {
                    None
                }
            })
            .collect();

        let attn_offsets_snap: Vec<i32> = caches
            .iter()
            .filter_map(|c| {
                if let NemotronLayerCache::Attention(kv) = c {
                    Some(kv.offset)
                } else {
                    None
                }
            })
            .collect();

        // Build raw pointer slices from the owned snapshots (safe: snapshots outlive
        // the nemotron_decode_step call below).
        let mamba_conv_ptrs: Vec<*const MlxArray> = mamba_conv_snap
            .iter()
            .map(|a| a.as_ref().unwrap() as *const _)
            .collect();
        let mamba_ssm_ptrs: Vec<*const MlxArray> = mamba_ssm_snap
            .iter()
            .map(|a| a.as_ref().unwrap() as *const _)
            .collect();

        // For attention layers with no prior context, pass null.
        // attn_keys_snap / attn_values_snap may have fewer entries than attn_count
        // if some attention caches are empty.  Build dense arrays with null-fill.
        let mut attn_key_ptrs: Vec<*const MlxArray> = Vec::with_capacity(attn_count);
        let mut attn_val_ptrs: Vec<*const MlxArray> = Vec::with_capacity(attn_count);
        {
            let mut snap_idx = 0;
            for c in caches.iter() {
                if let NemotronLayerCache::Attention(kv) = c {
                    if kv.keys.is_some() {
                        attn_key_ptrs.push(attn_keys_snap[snap_idx].as_ref().unwrap() as *const _);
                        attn_val_ptrs
                            .push(attn_values_snap[snap_idx].as_ref().unwrap() as *const _);
                        snap_idx += 1;
                    } else {
                        attn_key_ptrs.push(std::ptr::null());
                        attn_val_ptrs.push(std::ptr::null());
                    }
                }
            }
        }

        // --- 2. C++ full-model decode with pre-step snapshots ---
        let mut mamba_conv_out: Vec<UniquePtr<MlxArray>> = (0..mamba_count)
            .map(|_| mlxcel_core::UniquePtr::null())
            .collect();
        let mut mamba_ssm_out: Vec<UniquePtr<MlxArray>> = (0..mamba_count)
            .map(|_| mlxcel_core::UniquePtr::null())
            .collect();
        let mut cpp_logits = mlxcel_core::UniquePtr::null();

        // SAFETY:
        //   - handle is valid (set by register_cpp_model, freed only in Drop).
        //   - all pointer slices reference arrays owned by {mamba,attn}_*_snap which
        //     live until the end of this function.
        //   - nemotron_decode_step does not take ownership of the pointed arrays.
        unsafe {
            mlxcel_core::nemotron_decode_step(
                self.c_handle,
                inputs,
                &mamba_conv_ptrs,
                &mamba_ssm_ptrs,
                &attn_key_ptrs,
                &attn_val_ptrs,
                &attn_offsets_snap,
                &mut cpp_logits,
                &mut mamba_conv_out,
                &mut mamba_ssm_out,
            );
        }

        // Update Mamba caches from C++ output.
        // The full-decode C++ kernel calls fused_mamba2_forward internally,
        // which builds conv states via slice(padded_input, ...) — a lazy alias.
        // Wrap each conv_state with contiguous() to materialize a compact buffer
        // and drop the upstream padded_input graph reference.
        let mut mamba_idx = 0;
        for c in caches.iter_mut() {
            if let NemotronLayerCache::Mamba(mc) = c {
                let raw_conv = std::mem::replace(
                    &mut mamba_conv_out[mamba_idx],
                    mlxcel_core::UniquePtr::null(),
                );
                mc.conv_state = Some(mlxcel_core::contiguous(&raw_conv, false));
                mc.ssm_state = Some(std::mem::replace(
                    &mut mamba_ssm_out[mamba_idx],
                    mlxcel_core::UniquePtr::null(),
                ));
                mamba_idx += 1;
            }
        }

        // Note: Attention KV caches are NOT updated by C++ path.
        // This is a hypothesis test — output will be incorrect after first attention layer.
        // For production use, C++ would need to return updated KV arrays.

        cpp_logits
    }

    /// Measure graph build + GPU eval separately (debug only)
    fn forward_profiled(
        &self,
        inputs: &MlxArray,
        caches: &mut [NemotronLayerCache],
    ) -> UniquePtr<MlxArray> {
        let t0 = std::time::Instant::now();
        let logits = self.forward_rust(inputs, caches);
        let build_ms = t0.elapsed().as_nanos() as f64 / 1e6;

        let t1 = std::time::Instant::now();
        mlxcel_core::eval(&logits);
        let eval_ms = t1.elapsed().as_nanos() as f64 / 1e6;

        eprintln!(
            "[FORWARD] build: {:.2}ms, eval: {:.2}ms, total: {:.2}ms",
            build_ms,
            eval_ms,
            build_ms + eval_ms,
        );
        logits
    }

    /// Standard Rust layer-by-layer forward pass.
    fn forward_rust(
        &self,
        inputs: &MlxArray,
        caches: &mut [NemotronLayerCache],
    ) -> UniquePtr<MlxArray> {
        let h = self.embeddings.forward(inputs);
        self.forward_layer_stack(h, caches)
    }

    /// Run the full layer stack starting from a pre-computed hidden
    /// state. Used by the VLM path that produces multimodal embeddings
    /// upstream of the text backbone (Nemotron H Nano Omni).
    ///
    /// Used by: Nemotron H Nano Omni VLM
    pub fn forward_with_inputs_embeds(&self, inputs_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        let mut internal = self.internal_caches.borrow_mut();
        self.forward_layer_stack(mlxcel_core::copy(inputs_embeds), &mut internal)
    }

    fn forward_layer_stack(
        &self,
        starting_hidden: UniquePtr<MlxArray>,
        caches: &mut [NemotronLayerCache],
    ) -> UniquePtr<MlxArray> {
        let profile_blocks = std::env::var("MLXCEL_PROFILE_BLOCKS").is_ok();
        let mut mamba_ns = 0u128;
        let mut attn_ns = 0u128;
        let mut moe_ns = 0u128;

        let mut h = starting_hidden;

        // Find first attention cache offset
        let attn_offset = caches
            .iter()
            .find(|c| matches!(c, NemotronLayerCache::Attention(_)))
            .map(|c| c.offset())
            .unwrap_or(0);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];
        let attn_mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, attn_offset))
        } else {
            None
        };

        // On M5 Max (Metal GPU Family 4), the lazy evaluation graph for 52
        // hybrid SSM+MoE layers can exceed the Metal command buffer capacity,
        // causing a GPU hang.  Periodically evaluating the hidden state during
        // prefill (seq_len > 1) keeps the graph small.  During single-token
        // decode the graph is already tiny, so no intermediate eval is needed.
        let needs_chunked_eval = seq_len > 1 && !profile_blocks;
        let eval_interval = 8; // evaluate every 8 layers

        let mut cache_idx = 0;
        for (layer_num, (layer, &block_type)) in
            self.layers.iter().zip(self.block_types.iter()).enumerate()
        {
            let t0 = if profile_blocks {
                mlxcel_core::eval(&h);
                Some(std::time::Instant::now())
            } else {
                None
            };

            if block_type.needs_cache() {
                let cache_entry = &mut caches[cache_idx];
                match (block_type, cache_entry) {
                    (BlockType::Mamba, NemotronLayerCache::Mamba(mc)) => {
                        h = layer.forward(&h, None, Some(mc), None);
                    }
                    (BlockType::Attention, NemotronLayerCache::Attention(kv)) => {
                        h = layer.forward(&h, attn_mask.as_deref(), None, Some(kv));
                    }
                    _ => {}
                }
                cache_idx += 1;
            } else {
                h = layer.forward(&h, None, None, None);
            }

            if let Some(t0) = t0 {
                let layer_idx = if block_type.needs_cache() {
                    cache_idx - 1
                } else {
                    cache_idx
                };
                let bt = match block_type {
                    BlockType::Mamba => "M",
                    BlockType::Attention => "*",
                    BlockType::MoE => "E",
                    _ => "-",
                };
                eprint!("[{}{}]", bt, layer_idx);
                mlxcel_core::eval(&h);
                let e = t0.elapsed().as_nanos();
                eprintln!(" {:.1}ms", e as f64 / 1e6);
                match block_type {
                    BlockType::Mamba => mamba_ns += e,
                    BlockType::Attention => attn_ns += e,
                    BlockType::MoE => moe_ns += e,
                    _ => {}
                }
            } else if needs_chunked_eval && (layer_num + 1) % eval_interval == 0 {
                mlxcel_core::eval(&h);
                // Also evaluate Mamba cache states to prevent lazy graph issues
                // on M5 Max (Metal GPU Family 4).  The conv/ssm states are
                // separate outputs from the SSM computation that don't feed into
                // h, so eval(&h) alone doesn't materialize them.  Leaving them
                // as lazy arrays across Metal command buffer boundaries can cause
                // stale GPU buffer references on NAx architecture.
                for c in caches.iter() {
                    if let NemotronLayerCache::Mamba(mc) = c {
                        if let Some(ref cs) = mc.conv_state {
                            mlxcel_core::eval(cs);
                        }
                        if let Some(ref ss) = mc.ssm_state {
                            mlxcel_core::eval(ss);
                        }
                    }
                }
            }
        }

        if profile_blocks {
            let total = (mamba_ns + attn_ns + moe_ns).max(1);
            eprintln!(
                "[BLOCKS] M:{:.1}ms({:.0}%) A:{:.1}ms({:.0}%) E:{:.1}ms({:.0}%) T:{:.1}ms",
                mamba_ns as f64 / 1e6,
                mamba_ns as f64 * 100.0 / total as f64,
                attn_ns as f64 / 1e6,
                attn_ns as f64 * 100.0 / total as f64,
                moe_ns as f64 / 1e6,
                moe_ns as f64 * 100.0 / total as f64,
                total as f64 / 1e6,
            );
        }

        let h = self.norm_f.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn load(model_path: &str) -> Result<(Self, NemotronHConfig), Box<dyn std::error::Error>> {
        let path = Path::new(model_path);

        println!("[NemotronH] Loading config...");
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_str = super::sanitize_config_json(&config_str);
        let mut config: NemotronHConfig = serde_json::from_str(&config_str)?;
        config
            .post_init()
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        let pattern = config
            .hybrid_override_pattern
            .as_ref()
            .ok_or("hybrid_override_pattern must be set (directly or via layers_block_type)")?;
        let block_types: Vec<BlockType> = pattern.iter().map(|s| BlockType::from_str(s)).collect();

        println!(
            "[NemotronH] Config loaded: {} layers ({} mamba, {} attention, {} mlp, {} moe)",
            config.num_hidden_layers,
            block_types
                .iter()
                .filter(|t| **t == BlockType::Mamba)
                .count(),
            block_types
                .iter()
                .filter(|t| **t == BlockType::Attention)
                .count(),
            block_types.iter().filter(|t| **t == BlockType::MLP).count(),
            block_types.iter().filter(|t| **t == BlockType::MoE).count()
        );

        println!("[NemotronH] Loading weights from safetensors...");
        let weights = crate::models::load_text_weights(path, None)?;
        let weights = Self::sanitize_weights(weights, &config);

        println!("[NemotronH] Building model...");
        let model = Self::from_weights(config.clone(), weights, block_types)?;

        println!("[NemotronH] Model loaded successfully");
        Ok((model, config))
    }

    pub fn sanitize_weights(mut weights: WeightMap, config: &NemotronHConfig) -> WeightMap {
        // Filter out multi-token prediction (mtp.*) weights (NemotronSuper)
        weights.retain(|k, _| !k.starts_with("mtp."));

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

        // Stack expert weights
        let n_routed = config.n_routed_experts.unwrap_or(0);
        if n_routed > 0 {
            for l in 0..config.num_hidden_layers {
                let prefix = format!("backbone.layers.{}.mixer", l);

                for (m, n) in [("down_proj", "fc2"), ("up_proj", "fc1")] {
                    let first_expert_key = format!("{}.experts.0.{}.weight", prefix, m);
                    if weights.contains_key(&first_expert_key) {
                        let mut expert_tensors: Vec<UniquePtr<MlxArray>> = Vec::new();
                        for e in 0..n_routed {
                            if let Some(w) =
                                weights.remove(&format!("{}.experts.{}.{}.weight", prefix, e, m))
                            {
                                expert_tensors.push(w);
                            }
                        }
                        if !expert_tensors.is_empty() {
                            let stacked = stack_arrays(&expert_tensors, 0);
                            weights.insert(format!("{}.switch_mlp.{}.weight", prefix, n), stacked);
                        }
                    }
                }
            }
        }

        weights
    }

    pub fn from_weights(
        config: NemotronHConfig,
        mut weights: WeightMap,
        block_types: Vec<BlockType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Get quantization parameters
        let group_size = config.group_size();
        let bits = config.bits();

        // Quantized Embeddings
        let embeddings =
            UnifiedEmbedding::from_weights(&weights, "backbone.embeddings", group_size, bits)?;

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (i, &block_type) in block_types.iter().enumerate() {
            let prefix = format!("backbone.layers.{}", i);

            // Norm
            let norm_weight = weights
                .remove(&format!("{}.norm.weight", prefix))
                .ok_or(format!("Missing norm weight for layer {}", i))?;
            let norm = RMSNorm::new(norm_weight, config.layer_norm_epsilon);

            // Mixer based on block type
            let mixer = match block_type {
                BlockType::Mamba => {
                    let mixer_prefix = format!("{}.mixer", prefix);
                    let intermediate_size = config.get_mamba_intermediate_size();
                    let conv_dim = config.get_conv_dim();
                    let mamba_group_size = intermediate_size / config.n_groups;

                    let conv_weight = weights
                        .remove(&format!("{}.conv1d.weight", mixer_prefix))
                        .ok_or(format!("Missing conv1d weight for layer {}", i))?;
                    let conv_bias = weights.remove(&format!("{}.conv1d.bias", mixer_prefix));

                    let in_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.in_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let out_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.out_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;

                    let dt_bias = weights
                        .remove(&format!("{}.dt_bias", mixer_prefix))
                        .ok_or(format!("Missing dt_bias for layer {}", i))?;
                    let a_log = weights
                        .remove(&format!("{}.A_log", mixer_prefix))
                        .ok_or(format!("Missing A_log for layer {}", i))?;
                    let d_param = weights
                        .remove(&format!("{}.D", mixer_prefix))
                        .ok_or(format!("Missing D for layer {}", i))?;

                    let norm_weight = weights
                        .remove(&format!("{}.norm.weight", mixer_prefix))
                        .ok_or(format!("Missing mixer norm weight for layer {}", i))?;

                    NemotronHMixer::Mamba(NemotronHMamba2Mixer {
                        num_heads: config.mamba_num_heads,
                        hidden_size: config.hidden_size,
                        ssm_state_size: config.ssm_state_size,
                        conv_kernel_size: config.conv_kernel,
                        intermediate_size,
                        n_groups: config.n_groups,
                        head_dim: config.mamba_head_dim,
                        time_step_limit: config.effective_time_step_limit(),
                        conv_dim,
                        conv_weight,
                        conv_bias,
                        in_proj,
                        dt_bias,
                        a_log,
                        d_param,
                        norm: MambaRMSNormGated {
                            weight: norm_weight,
                            eps: config.layer_norm_epsilon,
                            group_size: mamba_group_size,
                        },
                        out_proj,
                    })
                }
                BlockType::Attention => {
                    let mixer_prefix = format!("{}.mixer", prefix);
                    let head_dim = config.get_head_dim();

                    let q_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.q_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let k_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.k_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let v_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.v_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let o_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.o_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;

                    NemotronHMixer::Attention(NemotronHAttention {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        n_heads: config.num_attention_heads,
                        n_kv_heads: config.num_key_value_heads,
                        head_dim,
                        scale: (head_dim as f32).powf(-0.5),
                    })
                }
                BlockType::MLP => {
                    let mixer_prefix = format!("{}.mixer", prefix);
                    let up_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.up_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;
                    let down_proj = UnifiedLinear::from_weights(
                        &weights,
                        &format!("{}.down_proj", mixer_prefix),
                        group_size,
                        bits,
                    )?;

                    NemotronHMixer::MLP(NemotronHMLP { up_proj, down_proj })
                }
                BlockType::MoE => {
                    let mixer_prefix = format!("{}.mixer", prefix);
                    let n_routed = config.n_routed_experts.unwrap_or(1);
                    let _moe_hidden = config
                        .moe_intermediate_size
                        .unwrap_or(config.intermediate_size);
                    let top_k = config.num_experts_per_tok.unwrap_or(1);

                    // Gate
                    let gate_weight = weights
                        .remove(&format!("{}.gate.weight", mixer_prefix))
                        .ok_or(format!("Missing gate weight for layer {}", i))?;
                    let e_score_bias = weights
                        .remove(&format!("{}.gate.e_score_correction_bias", mixer_prefix))
                        .unwrap_or_else(|| {
                            mlxcel_core::zeros(&[n_routed as i32], mlxcel_core::dtype::FLOAT32)
                        });

                    // Switch MLP (quantized or regular)
                    let fc1_weight = weights
                        .remove(&format!("{}.switch_mlp.fc1.weight", mixer_prefix))
                        .ok_or(format!("Missing switch_mlp.fc1.weight for layer {}", i))?;
                    let fc1_scales =
                        weights.remove(&format!("{}.switch_mlp.fc1.scales", mixer_prefix));
                    let fc1_biases =
                        weights.remove(&format!("{}.switch_mlp.fc1.biases", mixer_prefix));
                    let fc2_weight = weights
                        .remove(&format!("{}.switch_mlp.fc2.weight", mixer_prefix))
                        .ok_or(format!("Missing switch_mlp.fc2.weight for layer {}", i))?;
                    let fc2_scales =
                        weights.remove(&format!("{}.switch_mlp.fc2.scales", mixer_prefix));
                    let fc2_biases =
                        weights.remove(&format!("{}.switch_mlp.fc2.biases", mixer_prefix));

                    // Shared experts (optional, quantized)
                    let shared_experts = if config.n_shared_experts.is_some() {
                        let up_proj = UnifiedLinear::from_weights(
                            &weights,
                            &format!("{}.shared_experts.up_proj", mixer_prefix),
                            group_size,
                            bits,
                        )
                        .ok();
                        let down_proj = UnifiedLinear::from_weights(
                            &weights,
                            &format!("{}.shared_experts.down_proj", mixer_prefix),
                            group_size,
                            bits,
                        )
                        .ok();

                        if let (Some(up), Some(down)) = (up_proj, down_proj) {
                            Some(NemotronHMLP {
                                up_proj: up,
                                down_proj: down,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Latent projection layers (NemotronSuper)
                    let (fc1_latent_proj, fc2_latent_proj) = if config.moe_latent_size.is_some() {
                        let fc1_lp = UnifiedLinear::from_weights(
                            &weights,
                            &format!("{}.fc1_latent_proj", mixer_prefix),
                            group_size,
                            bits,
                        )
                        .ok();
                        let fc2_lp = UnifiedLinear::from_weights(
                            &weights,
                            &format!("{}.fc2_latent_proj", mixer_prefix),
                            group_size,
                            bits,
                        )
                        .ok();
                        (fc1_lp, fc2_lp)
                    } else {
                        (None, None)
                    };

                    NemotronHMixer::MoE(NemotronHMoE {
                        gate: NemotronHMoEGate {
                            weight: gate_weight,
                            e_score_correction_bias: e_score_bias,
                            top_k,
                            routed_scaling_factor: config.routed_scaling_factor.unwrap_or(1.0),
                            norm_topk_prob: config.norm_topk_prob.unwrap_or(false),
                        },
                        switch_mlp: SwitchMLP {
                            fc1: if let Some(scales) = fc1_scales {
                                QuantizedSwitchLinear::Quantized {
                                    weight: fc1_weight,
                                    scales,
                                    biases: fc1_biases.ok_or(format!(
                                        "Missing switch_mlp.fc1.biases for layer {}",
                                        i
                                    ))?,
                                    group_size,
                                    bits,
                                }
                            } else {
                                QuantizedSwitchLinear::Regular { weight: fc1_weight }
                            },
                            fc2: if let Some(scales) = fc2_scales {
                                QuantizedSwitchLinear::Quantized {
                                    weight: fc2_weight,
                                    scales,
                                    biases: fc2_biases.ok_or(format!(
                                        "Missing switch_mlp.fc2.biases for layer {}",
                                        i
                                    ))?,
                                    group_size,
                                    bits,
                                }
                            } else {
                                QuantizedSwitchLinear::Regular { weight: fc2_weight }
                            },
                        },
                        shared_experts,
                        fc1_latent_proj,
                        fc2_latent_proj,
                    })
                }
            };

            layers.push(NemotronHBlock {
                norm,
                block_type,
                mixer,
            });
        }

        // Final norm
        let norm_f_weight = weights
            .remove("backbone.norm_f.weight")
            .ok_or("Missing final norm weight")?;
        let norm_f = RMSNorm::new(norm_f_weight, config.layer_norm_epsilon);

        // LM head (quantized)
        let lm_head = UnifiedLinear::from_weights(&weights, "lm_head", group_size, bits)?;

        // Create internal caches for LanguageModel trait compatibility
        let internal_caches: Vec<NemotronLayerCache> = block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .map(|bt| match bt {
                BlockType::Mamba => NemotronLayerCache::Mamba(NemotronMambaCache::new()),
                BlockType::Attention => NemotronLayerCache::Attention(KVCache::new()),
                _ => unreachable!(),
            })
            .collect();

        let mut model = Self {
            config,
            embeddings,
            layers,
            norm_f,
            lm_head,
            block_types,
            internal_caches: RefCell::new(internal_caches),
            c_handle: 0,
        };

        // Register weights with the C++ full-forward engine.
        // This is a one-time cost that enables nemotron_decode_step for single-token decode.
        model.c_handle = model.register_cpp_model();

        Ok(model)
    }

    /// Collect all weight pointers and call `nemotron_register_model`.
    /// Returns the opaque handle, or 0 on failure.
    fn register_cpp_model(&self) -> u64 {
        // ---- per-layer norm weights (one per layer, in layer order) ----
        let norm_weights: Vec<*const MlxArray> = self
            .layers
            .iter()
            .map(|l| l.norm.weight.as_ref().unwrap() as *const MlxArray)
            .collect();

        // ---- block-type encoding (matches C++ enum: M=0, *=1, -=2, E=3) ----
        let block_type_ints: Vec<i32> = self
            .block_types
            .iter()
            .map(|bt| match bt {
                BlockType::Mamba => 0,
                BlockType::Attention => 1,
                BlockType::MLP => 2,
                BlockType::MoE => 3,
            })
            .collect();

        // ---- mamba weight pointers (12 per mamba layer) ----
        // Order: in_w, in_s, in_b, conv_w, conv_b, a_log, d, dt_bias,
        //        norm_w, out_w, out_s, out_b
        let mut mamba_weights: Vec<*const MlxArray> = Vec::new();
        for layer in &self.layers {
            if let NemotronHMixer::Mamba(m) = &layer.mixer {
                mamba_weights.push(m.in_proj.weight_ptr());
                mamba_weights.push(m.in_proj.scales_ptr());
                mamba_weights.push(m.in_proj.biases_ptr());
                mamba_weights.push(m.conv_weight.as_ref().unwrap() as *const MlxArray);
                mamba_weights.push(
                    m.conv_bias
                        .as_ref()
                        .map(|b| b.as_ref().unwrap() as *const MlxArray)
                        .unwrap_or(std::ptr::null()),
                );
                mamba_weights.push(m.a_log.as_ref().unwrap() as *const MlxArray);
                mamba_weights.push(m.d_param.as_ref().unwrap() as *const MlxArray);
                mamba_weights.push(m.dt_bias.as_ref().unwrap() as *const MlxArray);
                mamba_weights.push(m.norm.weight.as_ref().unwrap() as *const MlxArray);
                mamba_weights.push(m.out_proj.weight_ptr());
                mamba_weights.push(m.out_proj.scales_ptr());
                mamba_weights.push(m.out_proj.biases_ptr());
            }
        }

        // ---- MoE weight pointers (14 per MoE layer) ----
        // Order: gate_w, corr_bias,
        //        fc1_w, fc1_s, fc1_b, fc2_w, fc2_s, fc2_b,
        //        su_w, su_s, su_b, sd_w, sd_s, sd_b
        let mut moe_weights: Vec<*const MlxArray> = Vec::new();
        for layer in &self.layers {
            if let NemotronHMixer::MoE(m) = &layer.mixer {
                moe_weights.push(m.gate.weight.as_ref().unwrap() as *const MlxArray);
                moe_weights
                    .push(m.gate.e_score_correction_bias.as_ref().unwrap() as *const MlxArray);
                // fc1
                let (fc1_w, fc1_s, fc1_b) = switch_linear_ptrs(&m.switch_mlp.fc1);
                moe_weights.push(fc1_w);
                moe_weights.push(fc1_s);
                moe_weights.push(fc1_b);
                // fc2
                let (fc2_w, fc2_s, fc2_b) = switch_linear_ptrs(&m.switch_mlp.fc2);
                moe_weights.push(fc2_w);
                moe_weights.push(fc2_s);
                moe_weights.push(fc2_b);
                // shared-expert up_proj
                if let Some(ref se) = m.shared_experts {
                    moe_weights.push(se.up_proj.weight_ptr());
                    moe_weights.push(se.up_proj.scales_ptr());
                    moe_weights.push(se.up_proj.biases_ptr());
                    moe_weights.push(se.down_proj.weight_ptr());
                    moe_weights.push(se.down_proj.scales_ptr());
                    moe_weights.push(se.down_proj.biases_ptr());
                } else {
                    for _ in 0..6 {
                        moe_weights.push(std::ptr::null());
                    }
                }
            }
        }

        // ---- attention weight pointers (12 per attention layer) ----
        // Order: q_w, q_s, q_b, k_w, k_s, k_b, v_w, v_s, v_b, o_w, o_s, o_b
        let mut attn_weights: Vec<*const MlxArray> = Vec::new();
        for layer in &self.layers {
            if let NemotronHMixer::Attention(a) = &layer.mixer {
                attn_weights.push(a.q_proj.weight_ptr());
                attn_weights.push(a.q_proj.scales_ptr());
                attn_weights.push(a.q_proj.biases_ptr());
                attn_weights.push(a.k_proj.weight_ptr());
                attn_weights.push(a.k_proj.scales_ptr());
                attn_weights.push(a.k_proj.biases_ptr());
                attn_weights.push(a.v_proj.weight_ptr());
                attn_weights.push(a.v_proj.scales_ptr());
                attn_weights.push(a.v_proj.biases_ptr());
                attn_weights.push(a.o_proj.weight_ptr());
                attn_weights.push(a.o_proj.scales_ptr());
                attn_weights.push(a.o_proj.biases_ptr());
            }
        }

        // ---- embedding & lm_head pointers ----
        let (emb_w, emb_s, emb_b) = match &self.embeddings {
            UnifiedEmbedding::Quantized(q) => (
                q.weight.as_ref().unwrap() as *const MlxArray,
                q.scales.as_ref().unwrap() as *const MlxArray,
                q.biases
                    .as_ref()
                    .map(|b| b.as_ref().unwrap() as *const MlxArray)
                    .unwrap_or(std::ptr::null()),
            ),
            UnifiedEmbedding::Regular(e) => (
                e.weight.as_ref().unwrap() as *const MlxArray,
                std::ptr::null(),
                std::ptr::null(),
            ),
        };

        let final_norm_w = self.norm_f.weight.as_ref().unwrap() as *const MlxArray;

        let lm_head_b = self.lm_head.biases_ptr();

        let moe_top_k = self
            .layers
            .iter()
            .find_map(|l| {
                if let NemotronHMixer::MoE(m) = &l.mixer {
                    Some(m.gate.top_k as i32)
                } else {
                    None
                }
            })
            .unwrap_or(1);
        let moe_scaling = self
            .layers
            .iter()
            .find_map(|l| {
                if let NemotronHMixer::MoE(m) = &l.mixer {
                    Some(m.gate.routed_scaling_factor)
                } else {
                    None
                }
            })
            .unwrap_or(1.0);
        let moe_norm = self
            .layers
            .iter()
            .find_map(|l| {
                if let NemotronHMixer::MoE(m) = &l.mixer {
                    Some(m.gate.norm_topk_prob)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        let first_attn = self.layers.iter().find_map(|l| {
            if let NemotronHMixer::Attention(a) = &l.mixer {
                Some(a)
            } else {
                None
            }
        });
        let (a_heads, a_kvh, a_hdim, a_scale) = first_attn
            .map(|a| {
                (
                    a.n_heads as i32,
                    a.n_kv_heads as i32,
                    a.head_dim as i32,
                    a.scale,
                )
            })
            .unwrap_or((0, 0, 0, 1.0));

        let group_size = self.config.group_size();
        let bits = self.config.bits();

        // SAFETY invariants for the unsafe block below:
        //   - All weight pointers were obtained from UniquePtr<MlxArray> that are owned
        //     by `self` (which outlives this call).
        //   - The C++ function stores pointers but does NOT take ownership; it is the
        //     caller's responsibility to keep the model alive while the handle is valid.
        //   - For non-quantized embeddings, we pass the weight pointer for both the
        //     scales and biases stubs.  The C++ side detects non-quantized case by
        //     checking `bits == 0` (passed via the `bits` argument).
        //   - `emb_s` / `emb_b` / `lm_head_s_ptr` are only non-null for quantized weights.
        //   - `nemotron_register_model` requires non-nullable `&MlxArray` refs for embed
        //     and lm_head scales/biases; for the non-quantized path we pass the weight
        //     array as a harmless stub (C++ ignores them when bits == 0).
        unsafe {
            // Resolve potentially null scale/bias pointers to concrete refs for the FFI
            // call (cxx bridge requires &T, not *const T for these parameters).
            let emb_w_ref: &MlxArray = &*emb_w;
            let emb_s_ref: &MlxArray = if emb_s.is_null() { emb_w_ref } else { &*emb_s };
            let emb_b_ref: &MlxArray = if emb_b.is_null() { emb_w_ref } else { &*emb_b };

            let lm_w_ptr = self.lm_head.weight_ptr();
            let lm_s_ptr = self.lm_head.scales_ptr();
            let lm_w_ref: &MlxArray = &*lm_w_ptr;
            let lm_s_ref: &MlxArray = if lm_s_ptr.is_null() {
                lm_w_ref
            } else {
                &*lm_s_ptr
            };

            mlxcel_core::nemotron_register_model(
                // embedding
                emb_w_ref,
                emb_s_ref,
                emb_b_ref,
                // final norm
                &*final_norm_w,
                // lm_head (biases nullable via *const MlxArray)
                lm_w_ref,
                lm_s_ref,
                lm_head_b,
                // per-layer norms
                &norm_weights,
                &block_type_ints,
                // mixer weights
                &mamba_weights,
                &moe_weights,
                &attn_weights,
                // config scalars
                self.config.layer_norm_epsilon,
                group_size,
                bits,
                // mamba config
                self.config.get_mamba_intermediate_size() as i32,
                self.config.get_conv_dim() as i32,
                self.config.conv_kernel as i32,
                self.config.mamba_num_heads as i32,
                self.config.mamba_head_dim as i32,
                self.config.n_groups as i32,
                self.config.ssm_state_size as i32,
                self.config.effective_time_step_limit().0,
                self.config.effective_time_step_limit().1,
                self.config.layer_norm_epsilon,
                // MoE config
                moe_top_k,
                moe_scaling,
                moe_norm,
                // attention config
                a_heads,
                a_kvh,
                a_hdim,
                0.0_f32, // a_rope: NemotronH uses no RoPE (positions embedded differently)
                a_scale,
            )
        }
    }
}

#[allow(dead_code)]
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
impl LanguageModel for NemotronHModel {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // NemotronH uses mixed cache types (KVCache + MambaCache)
        // We use internal RefCell caches to maintain state through shared reference
        let mut internal = self.internal_caches.borrow_mut();
        self.forward_with_caches(input, &mut internal)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset internal caches for new generation session
        let new_internal_caches: Vec<NemotronLayerCache> = self
            .block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .map(|bt| match bt {
                BlockType::Mamba => NemotronLayerCache::Mamba(NemotronMambaCache::new()),
                BlockType::Attention => NemotronLayerCache::Attention(KVCache::new()),
                _ => unreachable!(),
            })
            .collect();
        *self.internal_caches.borrow_mut() = new_internal_caches;

        // Return empty KV caches for trait compatibility
        // Actual caching uses internal_caches
        (0..self
            .block_types
            .iter()
            .filter(|bt| bt.needs_cache())
            .count())
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn supports_batching(&self) -> bool {
        false // NemotronH is a hybrid Mamba+Transformer, internal caches not compatible with per-sequence KV isolation
    }

    fn trim_internal_caches(&self, excess: i32) {
        if excess <= 0 {
            return;
        }
        let mut internal = self.internal_caches.borrow_mut();
        for cache in internal.iter_mut() {
            match cache {
                NemotronLayerCache::Attention(kv) => {
                    kv.trim(excess);
                }
                NemotronLayerCache::Mamba(mc) => {
                    // Mamba caches are recurrent state, not positional.
                    // Reset them since the SSM/conv state was computed using
                    // padding tokens and would corrupt subsequent decode steps.
                    mc.conv_state = None;
                    mc.ssm_state = None;
                }
            }
        }
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Hybrid SSM model: padding tokens corrupt Mamba recurrent state
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![11] // <|im_end|>
    }
}

#[cfg(test)]
#[path = "nemotron_h_tests.rs"]
mod tests;
