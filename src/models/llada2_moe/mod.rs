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

//! LLaDA-2 MoE masked-diffusion language model (issue #546).
//!
//! `inclusionAI/LLaDA2.0-mini` (`model_type: llada2_moe`) is a decoder-only
//! transformer with bidirectional attention and a DeepSeek-V3-style MoE FFN.
//! It generates by iterative block-wise unmasking of `<|mask|>` tokens rather
//! than autoregressive decode, so the model owns its generation loop (see
//! [`generate`]) and reports `supports_batching() == false`, exactly like
//! [`crate::models::diffusion_gemma::DiffusionGemmaModel`].
//!
//! Structure per layer (pre-norm RMSNorm residual blocks):
//!
//! ```text
//! h   = x + Attention(RMSNorm_input(x))
//! out = h + FFN(RMSNorm_post_attention(h))
//! ```
//!
//! `FFN` is a dense SwiGLU MLP for `i < first_k_dense_replace` and the MoE
//! block otherwise. Attention uses a fused QKV projection (split at runtime),
//! per-head QK-norm, and partial RoPE (only `rotary_dim` of `head_dim` is
//! rotated). No causal mask exists at any stage: every query attends to every
//! visible key (committed prefix + current block).
//!
//! Reused building blocks: [`crate::models::switch_layers::SwitchGLU`] for the
//! routed experts and [`crate::models::switch_layers::moe_weighted_sum`] for
//! the combine. The gate is modeled on
//! [`crate::models::deepseek_v3::MoEGate`] with three deltas: the bias key is
//! `expert_bias` (not `e_score_correction_bias`), the top-k normalization adds
//! `+ 1e-20` to the denominator, and the group mask fills non-kept groups with
//! a large negative constant (not zero) because `expert_bias` may be negative.

mod generate;

pub use generate::{
    Llada2FinishReason, Llada2GenerateOptions, Llada2GenerationStats, block_num_blocks,
    block_threshold, transfer_mask, truncate_at_eos,
};

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde::Deserialize;
use std::path::Path;

use super::switch_layers::{SwitchGLU, moe_weighted_sum};

/// `pad_token_id` fallback when `config.json` omits it (`LLaDA2.0-mini` ships
/// it, but the default is load-bearing for exports that do not).
const DEFAULT_PAD_TOKEN_ID: i32 = 156_892;
/// `mask_token_id` fallback (`<|mask|>`); the real config omits it.
const DEFAULT_MASK_TOKEN_ID: i32 = 156_895;
/// Fill value for experts in non-kept routing groups. Sigmoid scores plus the
/// (possibly negative) expert bias stay far above this, so a masked expert can
/// never be selected by the top-k argpartition. A large finite negative avoids
/// the `-inf` NaN risk in later arithmetic while ordering strictly last.
const GROUP_MASK_FILL: f32 = -1.0e30;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

fn default_model_type() -> String {
    "llada2_moe".to_string()
}
fn default_vocab_size() -> usize {
    157_184
}
fn default_hidden_size() -> usize {
    2048
}
fn default_intermediate_size() -> usize {
    5120
}
fn default_num_hidden_layers() -> usize {
    20
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_num_key_value_heads() -> usize {
    4
}
fn default_partial_rotary_factor() -> f32 {
    0.5
}
fn default_rope_theta() -> f32 {
    600_000.0
}
fn default_use_qk_norm() -> bool {
    true
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_num_experts() -> usize {
    256
}
fn default_num_experts_per_tok() -> usize {
    8
}
fn default_n_group() -> usize {
    8
}
fn default_topk_group() -> usize {
    4
}
fn default_num_shared_experts() -> usize {
    1
}
fn default_moe_intermediate_size() -> usize {
    512
}
fn default_first_k_dense_replace() -> usize {
    1
}
fn default_routed_scaling_factor() -> f32 {
    2.5
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_max_position_embeddings() -> usize {
    32_768
}

/// Top-level `config.json` arguments for `model_type == "llada2_moe"`.
///
/// Every field carries the serde default from the issue spec so the real
/// config (which omits `eos_token_id`, `mask_token_id`, and `use_qk_norm`)
/// loads correctly. Derived values (`head_dim`, `rotary_dim`, `eos_token_id`)
/// are resolved by the accessor methods below.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rotary_dim: Option<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_use_qk_norm")]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub use_qkv_bias: bool,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
    #[serde(default = "default_num_shared_experts")]
    pub num_shared_experts: usize,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub pad_token_id: Option<i32>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub mask_token_id: Option<i32>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    // Quantization group_size / bits (present on pre-quantized exports).
    #[serde(default)]
    pub group_size: Option<i32>,
    #[serde(default)]
    pub bits: Option<i32>,
}

impl ModelArgs {
    /// Per-head dimension, defaulting to `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Rotated channel count, defaulting to `head_dim * partial_rotary_factor`.
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
            .unwrap_or((self.head_dim() as f32 * self.partial_rotary_factor) as usize)
    }

    /// `pad_token_id`, defaulting to 156892.
    pub fn pad_token_id(&self) -> i32 {
        self.pad_token_id.unwrap_or(DEFAULT_PAD_TOKEN_ID)
    }

    /// `mask_token_id` (`<|mask|>`), defaulting to 156895.
    pub fn mask_token_id(&self) -> i32 {
        self.mask_token_id.unwrap_or(DEFAULT_MASK_TOKEN_ID)
    }

    /// EOS ids: parse `eos_token_id` (scalar or list); when absent, fall back
    /// to `{pad_token_id}` (the real config omits `eos_token_id`).
    pub fn eos_token_ids(&self) -> Vec<i32> {
        let mut ids = parse_eos_ids(self.eos_token_id.as_ref());
        if ids.is_empty() {
            ids.push(self.pad_token_id());
        }
        ids
    }

    fn group_size(&self) -> i32 {
        self.group_size.unwrap_or(64)
    }

    fn bits(&self) -> i32 {
        self.bits.unwrap_or(4)
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_k_dense_replace
    }
}

/// Parse an `eos_token_id` JSON value into a list of ids: accepts a scalar
/// integer, an array of integers, or `null`/absent (empty list).
pub(crate) fn parse_eos_ids(value: Option<&serde_json::Value>) -> Vec<i32> {
    match value {
        Some(serde_json::Value::Number(n)) => {
            n.as_i64().map(|v| vec![v as i32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_i64().map(|x| x as i32))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Attention (fused QKV, QK-norm, partial RoPE, bidirectional)
// ---------------------------------------------------------------------------

struct Attention {
    qkv_proj: UnifiedLinear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    dense: UnifiedLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    rope_theta: f32,
    scale: f32,
}

impl Attention {
    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.head_dim() as i32;

        let qkv_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.query_key_value"),
            group_size,
            bits,
        )?;
        let dense =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.dense"), group_size, bits)?;

        let (q_norm, k_norm) = if args.use_qk_norm {
            let q = get_weight_copy(weights, &format!("{prefix}.query_layernorm.weight"))?;
            let k = get_weight_copy(weights, &format!("{prefix}.key_layernorm.weight"))?;
            (
                Some(RMSNorm::new(q, args.rms_norm_eps)),
                Some(RMSNorm::new(k, args.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            qkv_proj,
            q_norm,
            k_norm,
            dense,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            rotary_dim: args.rotary_dim() as i32,
            rope_theta: args.rope_theta,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    /// Project the fused QKV, split along the head axis, apply per-head QK-norm
    /// and partial RoPE, and return `(q, k, v)` transposed to
    /// `[1, heads, seq, head_dim]`. `offset` is the sequence position of the
    /// first token in `x`.
    fn project(
        &self,
        x: &MlxArray,
        offset: i32,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let total_heads = self.num_heads + 2 * self.num_kv_heads;

        // y = x @ W^T then reshape to [b, l, total_heads, head_dim]; splitting
        // along axis 2 at [num_heads, num_heads + num_kv_heads] recovers Q/K/V
        // because the reshape groups consecutive head_dim-sized row blocks into
        // heads in the same order as the fused weight rows.
        let qkv = self.qkv_proj.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[b, l, total_heads, self.head_dim]);
        let q = slice_axis(&qkv, 2, 0, self.num_heads);
        let k = slice_axis(&qkv, 2, self.num_heads, self.num_heads + self.num_kv_heads);
        let v = slice_axis(&qkv, 2, self.num_heads + self.num_kv_heads, total_heads);

        // Per-head QK-norm over the last axis (head_dim), before the transpose.
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

        // Transpose to [b, heads, seq, head_dim].
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Partial, non-traditional (split-halves) RoPE over the first
        // `rotary_dim` channels; the rest pass through unrotated.
        let q = mlxcel_core::fast_rope(&q, self.rotary_dim, false, self.rope_theta, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rotary_dim, false, self.rope_theta, 1.0, offset);
        (q, k, v)
    }

    /// Full-attention output over `(q, full_k, full_v)` with no mask
    /// (bidirectional), followed by the output projection.
    fn attend(&self, q: &MlxArray, full_k: &MlxArray, full_v: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(q);
        let b = shape[0];
        let l = shape[2];
        let out = mlxcel_core::layers::attention(q, full_k, full_v, self.scale, None, 0.0, 0);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.num_heads * self.head_dim]);
        self.dense.forward(&out)
    }

    /// Prefill / commit forward: append the block K/V to `cache` and attend
    /// over the full `[prefix + block]` key axis.
    fn forward_append(
        &self,
        x: &MlxArray,
        offset: i32,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let (q, k, v) = self.project(x, offset);
        let (full_k, full_v) = cache.update_and_fetch(k, v);
        self.attend(&q, &full_k, &full_v)
    }

    /// Denoising forward: read `cache` as a frozen prefix, attend over
    /// `[prefix + block]` with freshly computed block K/V, and DO NOT append.
    fn forward_readonly(&self, x: &MlxArray, offset: i32, cache: &KVCache) -> UniquePtr<MlxArray> {
        let (q, k, v) = self.project(x, offset);
        match cache.visible_state() {
            Some((pk, pv)) => {
                let full_k = mlxcel_core::concatenate(&pk, &k, 2);
                let full_v = mlxcel_core::concatenate(&pv, &v, 2);
                self.attend(&q, &full_k, &full_v)
            }
            None => self.attend(&q, &k, &v),
        }
    }
}

// ---------------------------------------------------------------------------
// Dense MLP (dense layers + shared experts)
// ---------------------------------------------------------------------------

struct DenseMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl DenseMLP {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.up_proj"),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// MoE gate (sigmoid scoring, group-limited top-k, DeepSeek-V3 style)
// ---------------------------------------------------------------------------

/// Group-limited masking that fills non-kept groups with `fill` instead of
/// zero (as [`crate::models::switch_layers::group_mask_scores`] does).
///
/// The zero fill is only equivalent to `-inf` when every biased score is
/// non-negative; `expert_bias` may be negative, so a zeroed non-kept expert
/// could outrank a kept one. Filling with a large negative constant keeps the
/// masked experts strictly last in the top-k argpartition.
fn group_limited_mask(
    biased: &MlxArray,
    n_group: i32,
    topk_group: i32,
    fill: f32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(biased);
    let n = shape[0];
    let n_experts = shape[1];
    let experts_per_group = n_experts / n_group;

    let grouped = mlxcel_core::reshape(biased, &[n, n_group, experts_per_group]);

    // group_score = sum of the top-2 expert scores within each group.
    let neg_grouped = mlxcel_core::negative(&grouped);
    let part_idx = mlxcel_core::argpartition(&neg_grouped, 1, -1);
    let top2_idx = slice_axis(&part_idx, -1, 0, 2);
    let top2_vals = mlxcel_core::take_along_axis(&grouped, &top2_idx, -1);
    let group_scores = mlxcel_core::sum_axis(&top2_vals, -1, true);

    // Fill the bottom `k = n_group - topk_group` groups with `fill`.
    let k = n_group - topk_group;
    let group_idx = mlxcel_core::argpartition(&group_scores, k - 1, -2);
    let group_idx = slice_axis(&group_idx, -2, 0, k);
    let fill_arr = mlxcel_core::full_f32(&[1], fill, mlxcel_core::array_dtype(&grouped));
    let grouped = mlxcel_core::put_along_axis(&grouped, &group_idx, &fill_arr, -2);

    mlxcel_core::reshape(&grouped, &[n, n_experts])
}

struct MoEGate {
    /// Router projection, kept in float32 (`router_dtype: fp32`).
    weight: UniquePtr<MlxArray>,
    /// Selection bias `expert_bias`, float32; affects selection only.
    expert_bias: UniquePtr<MlxArray>,
    top_k: i32,
    n_group: i32,
    topk_group: i32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
}

impl MoEGate {
    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        // Router runs in float32; keep the gate weight and bias in f32 and do
        // not quantize them.
        let weight = get_weight_f32(weights, &format!("{prefix}.weight"))?;
        let expert_bias = get_weight_f32(weights, &format!("{prefix}.expert_bias"))?;
        Ok(Self {
            weight,
            expert_bias,
            top_k: args.num_experts_per_tok as i32,
            n_group: args.n_group as i32,
            topk_group: args.topk_group as i32,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
        })
    }

    /// Returns `(expert_indices [n, top_k], expert_weights [n, top_k])`.
    fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // logits = f32(x) @ f32(gate.weight)^T, router in float32.
        let x_f32 = mlxcel_core::astype(x, dtype::FLOAT32);
        let weight_t = mlxcel_core::transpose(&self.weight);
        let logits = mlxcel_core::matmul(&x_f32, &weight_t);

        let scores = mlxcel_core::sigmoid(&logits);
        let orig_scores = mlxcel_core::copy(&scores);

        // Bias affects SELECTION only.
        let biased = mlxcel_core::add(&scores, &self.expert_bias);
        let biased = if self.n_group > 1 {
            group_limited_mask(&biased, self.n_group, self.topk_group, GROUP_MASK_FILL)
        } else {
            biased
        };

        // Top-k over the masked biased scores.
        let neg = mlxcel_core::negative(&biased);
        let indices = mlxcel_core::argpartition(&neg, self.top_k - 1, -1);
        let topk_indices = slice_axis(&indices, -1, 0, self.top_k);

        // Weights gathered from the UNBIASED sigmoid scores.
        let topk_weights = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);
        let topk_weights = if self.top_k > 1 && self.norm_topk_prob {
            // Divide by (sum + 1e-20): the +1e-20 is part of the LLaDA-2 spec.
            let sum = mlxcel_core::sum_axis(&topk_weights, -1, true);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::array_dtype(&sum));
            let denom = mlxcel_core::add(&sum, &eps);
            mlxcel_core::divide(&topk_weights, &denom)
        } else {
            topk_weights
        };
        let topk_weights = mlxcel_core::multiply_scalar(&topk_weights, self.routed_scaling_factor);

        (topk_indices, topk_weights)
    }
}

// ---------------------------------------------------------------------------
// MoE block (routed experts + shared expert)
// ---------------------------------------------------------------------------

struct MoEBlock {
    gate: MoEGate,
    switch_mlp: SwitchGLU,
    shared_experts: Option<DenseMLP>,
}

impl MoEBlock {
    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let gate = MoEGate::from_weights(weights, args, &format!("{prefix}.gate"))?;
        // `.switch_mlp` keys the shared per-expert stacker to the
        // `{prefix}.experts.{e}.{proj}.weight` layout the checkpoint ships.
        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{prefix}.switch_mlp"), group_size, bits)?;
        let shared_experts = if args.num_shared_experts > 0 {
            Some(DenseMLP::from_weights(
                weights,
                &format!("{prefix}.shared_experts"),
                group_size,
                bits,
            )?)
        } else {
            None
        };
        Ok(Self {
            gate,
            switch_mlp,
            shared_experts,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = orig_shape[orig_shape.len() - 1];
        let x_flat = if orig_shape.len() > 2 {
            mlxcel_core::reshape(x, &[-1, hidden])
        } else {
            mlxcel_core::copy(x)
        };

        let (indices, weights) = self.gate.forward(&x_flat);
        let expert_out = self.switch_mlp.forward(&x_flat, &indices);
        let mut result = moe_weighted_sum(&expert_out, &weights, mlxcel_core::array_dtype(&x_flat));

        if let Some(shared) = &self.shared_experts {
            let shared_out = shared.forward(&x_flat);
            result = mlxcel_core::add(&result, &shared_out);
        }

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

enum FFN {
    Dense(DenseMLP),
    MoE(MoEBlock),
}

impl FFN {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FFN::Dense(mlp) => mlp.forward(x),
            FFN::MoE(moe) => moe.forward(x),
        }
    }
}

struct Layer {
    input_layernorm: RMSNorm,
    attn: Attention,
    post_attention_layernorm: RMSNorm,
    ffn: FFN,
}

impl Layer {
    fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let group_size = args.group_size();
        let bits = args.bits();

        let attn = Attention::from_weights(weights, args, &format!("{prefix}.attention"))?;
        let ffn = if args.is_moe_layer(layer_idx) {
            FFN::MoE(MoEBlock::from_weights(
                weights,
                args,
                &format!("{prefix}.mlp"),
            )?)
        } else {
            FFN::Dense(DenseMLP::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                group_size,
                bits,
            )?)
        };

        let input_ln = get_weight_copy(weights, &format!("{prefix}.input_layernorm.weight"))?;
        let post_ln = get_weight_copy(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;

        Ok(Self {
            input_layernorm: RMSNorm::new(input_ln, args.rms_norm_eps),
            attn,
            post_attention_layernorm: RMSNorm::new(post_ln, args.rms_norm_eps),
            ffn,
        })
    }

    fn forward_append(
        &self,
        x: &MlxArray,
        offset: i32,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.attn.forward_append(&normed, offset, cache);
        let h = mlxcel_core::add(x, &attn_out);
        let normed = self.post_attention_layernorm.forward(&h);
        let ffn_out = self.ffn.forward(&normed);
        mlxcel_core::add(&h, &ffn_out)
    }

    fn forward_readonly(&self, x: &MlxArray, offset: i32, cache: &KVCache) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.attn.forward_readonly(&normed, offset, cache);
        let h = mlxcel_core::add(x, &attn_out);
        let normed = self.post_attention_layernorm.forward(&h);
        let ffn_out = self.ffn.forward(&normed);
        mlxcel_core::add(&h, &ffn_out)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// LLaDA-2 MoE masked-diffusion text model.
pub struct Llada2MoeModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<Layer>,
    norm: RMSNorm,
    /// Untied output projection (`tie_word_embeddings: false`). `None` for a
    /// hypothetical tied export, where the embedding's `as_linear` is used.
    lm_head: Option<UnifiedLinear>,
    pub(crate) vocab_size: i32,
    pub(crate) mask_token_id: i32,
    pub(crate) eos_token_ids: Vec<i32>,
}

impl Llada2MoeModel {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        Self::from_weights(&weights, &args)
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Embedding is `model.word_embeddings`, NOT `model.embed_tokens`.
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.word_embeddings", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(Layer::from_weights(weights, args, i)?);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Untied LM head (`tie_word_embeddings: false`): `lm_head.weight` is a
        // real tensor. A hypothetical tied export drops it; then the embedding
        // acts as the output projection.
        let lm_head =
            if weights.contains_key("lm_head.weight") || weights.contains_key("lm_head.scales") {
                Some(UnifiedLinear::from_weights(
                    weights, "lm_head", group_size, bits,
                )?)
            } else {
                None
            };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vocab_size: args.vocab_size as i32,
            mask_token_id: args.mask_token_id(),
            eos_token_ids: args.eos_token_ids(),
        })
    }

    /// Project the final hidden state to vocab logits through the untied head,
    /// or the tied embedding when no separate head is present.
    fn project_logits(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.lm_head {
            Some(head) => head.forward(hidden),
            None => self.embed_tokens.as_linear(hidden),
        }
    }

    /// Allocate one dense KV cache per layer.
    pub fn make_diffusion_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Run the transformer stack over `ids` at sequence `offset`, appending the
    /// block K/V into `caches` (prompt prefill and per-block commit). Returns
    /// the final normed hidden state `[1, L, hidden]`.
    pub(crate) fn forward_append(
        &self,
        ids: &MlxArray,
        caches: &mut [KVCache],
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(ids);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward_append(&h, offset, cache);
        }
        self.norm.forward(&h)
    }

    /// Run the transformer stack over `ids` at sequence `offset`, reading
    /// `caches` as a frozen prefix (per-denoising-step block forward). Returns
    /// softmax-ready logits `[1, L, vocab]`.
    pub(crate) fn forward_readonly_logits(
        &self,
        ids: &MlxArray,
        caches: &[KVCache],
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(ids);
        for (layer, cache) in self.layers.iter().zip(caches.iter()) {
            h = layer.forward_readonly(&h, offset, cache);
        }
        let h = self.norm.forward(&h);
        self.project_logits(&h)
    }

    pub fn eos_token_ids(&self) -> &[i32] {
        &self.eos_token_ids
    }
}

impl LanguageModel for Llada2MoeModel {
    /// Minimal trait forward: a bidirectional prefill pass (writing `caches`)
    /// followed by the final norm and untied-head logits. The CLI and server
    /// route LLaDA-2 to the block-unmasking engine before any autoregressive
    /// loop, so this exists for trait completeness (warmup, tooling).
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let offset = caches.first().map(|c| c.offset).unwrap_or(0);
        let hidden = self.forward_append(input_ids, caches, offset);
        self.project_logits(&hidden)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.make_diffusion_caches()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    /// Block-unmasking generation is a model-owned batch-1 loop; the
    /// batched/paged scheduler must never pick this model up.
    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Weight helpers
// ---------------------------------------------------------------------------

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

/// Copy a weight and cast it to float32 (used for the router gate weight and
/// `expert_bias`, which run in float32 per `router_dtype: fp32`).
fn get_weight_f32(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    let w = weights
        .get(name)
        .ok_or_else(|| format!("Weight not found: {name}"))?;
    Ok(mlxcel_core::astype(w, dtype::FLOAT32))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
