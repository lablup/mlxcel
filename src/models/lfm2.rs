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

//! LiquidAI LFM2 and LFM2-MoE (Liquid Foundation Models) implementation.
//!
//! LFM2 is a hybrid decoder. Each layer is EITHER a full-attention layer (when
//! its index is in `full_attn_idxs`, derived from `layer_types == "full_attention"`)
//! OR a gated short-convolution (`ShortConv`) layer; every layer then applies a
//! SwiGLU feed-forward that is dense (`lfm2`) or sparse/MoE (`lfm2_moe`). The
//! final norm is `embedding_norm` and the embeddings are tied
//! (`embed_tokens.as_linear`).
//!
//! Both `model_type = "lfm2"` and `model_type = "lfm2_moe"` route here; the only
//! structural difference is the per-layer feed-forward.
//!
//! Mirrored from the mlx-lm reference:
//! - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2.py
//! - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2_moe.py
//!
//! Correctness note (the load-bearing LFM2-MoE detail, upstream
//! ml-explore/mlx-lm#1354): the sparse router is **sigmoid-gated, not
//! softmax**. `routing_weights = sigmoid(gate(x))`; the optional `expert_bias`
//! participates ONLY in top-k selection (`argpartition(routing_weights + bias)`),
//! while the combine scores are gathered from the UNBIASED `routing_weights`.
//!
//! Because the short-convolution and attention layers both carry per-sequence
//! recurrent/positional state, the model owns its mixed cache through
//! [`ModelOwnedSequenceState`] (like Jamba / NemotronH) and reports
//! `supports_batching() == false`.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, slice_axis, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use super::model_owned::ModelOwnedSequenceState;
use super::recurrent_snapshot::{push_optional, restore_optional};
use super::switch_layers::{SwitchGLU, moe_weighted_sum};

// Configuration.

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    pub norm_eps: f32,
    pub conv_bias: bool,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Present in the dense `lfm2` config; the MoE config derives the indices
    /// from `layer_types` instead.
    #[serde(default)]
    pub full_attn_idxs: Option<Vec<usize>>,

    /// Present in the `lfm2_moe` config (`"conv"` / `"full_attention"`).
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    // MoE-only fields (absent for the dense `lfm2` checkpoint).
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub num_dense_layers: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    #[serde(default)]
    pub use_expert_bias: Option<bool>,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// Indices of the full-attention layers. Uses the explicit `full_attn_idxs`
    /// when present (dense `lfm2`), otherwise derives them from `layer_types`
    /// (`lfm2_moe`). Mirrors `ModelArgs.__post_init__` in the reference.
    pub fn full_attn_idxs(&self) -> Vec<usize> {
        if let Some(idxs) = &self.full_attn_idxs {
            idxs.clone()
        } else if let Some(types) = &self.layer_types {
            types
                .iter()
                .enumerate()
                .filter(|(_, t)| t.as_str() == "full_attention")
                .map(|(i, _)| i)
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        self.full_attn_idxs().contains(&layer_idx)
    }

    /// Whether this checkpoint carries sparse-MoE feed-forward layers.
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some() && self.moe_intermediate_size.is_some()
    }

    /// Layers with index `< num_dense_layers` use a dense SwiGLU MLP; the rest
    /// use the sparse-MoE block. For the dense `lfm2` checkpoint there are no
    /// MoE layers, so this defaults to `num_hidden_layers` (every layer dense).
    pub fn num_dense_layers(&self) -> usize {
        self.num_dense_layers.unwrap_or(self.num_hidden_layers)
    }

    pub fn layer_is_moe(&self, layer_idx: usize) -> bool {
        self.is_moe() && layer_idx >= self.num_dense_layers()
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        // LFM2 checkpoints use `<|im_end|>` (id 7) as EOS.
        const DEFAULT_EOS: i32 = 7;
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => n
                .as_i64()
                .map(|v| vec![v as i32])
                .unwrap_or(vec![DEFAULT_EOS]),
            Some(serde_json::Value::Array(arr)) => {
                let ids: Vec<i32> = arr
                    .iter()
                    .filter_map(|v| v.as_i64().map(|x| x as i32))
                    .collect();
                if ids.is_empty() {
                    vec![DEFAULT_EOS]
                } else {
                    ids
                }
            }
            _ => vec![DEFAULT_EOS],
        }
    }
}

// Attention (GQA with per-head Q/K RMSNorm before RoPE).

struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    q_layernorm: RMSNorm,
    k_layernorm: RMSNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_base: f32,
}

impl Attention {
    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let head_dim = args.head_dim() as i32;

        let q_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), gs, bits)?;
        let k_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), gs, bits)?;
        let v_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), gs, bits)?;
        let out_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.out_proj"), gs, bits)?;

        let q_layernorm = RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.q_layernorm.weight"))?,
            args.norm_eps,
        );
        let k_layernorm = RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.k_layernorm.weight"))?,
            args.norm_eps,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_base: args.rope_theta,
        })
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

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Per-head RMSNorm over head_dim, applied before transpose and RoPE.
        let q = self.q_layernorm.forward(&q);
        let k = self.k_layernorm.forward(&k);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

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
        self.out_proj.forward(&attn_out)
    }
}

// ShortConv (gated depthwise causal convolution mixer).

struct ShortConv {
    in_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    conv_weight: UniquePtr<MlxArray>,
    conv_bias: Option<UniquePtr<MlxArray>>,
    hidden_size: i32,
    l_cache: i32,
    /// Time-major depthwise-conv weight for the single-step (decode) fast path
    /// (issue #748), shape `[1, L_cache, hidden]` where
    /// `decode_weight[0, k, c] == conv_weight[c, k, 0]`. `None` on Metal, where
    /// MLX's `conv1d` already dispatches a fast small-conv kernel and the proven
    /// path is kept as-is.
    decode_weight: Option<UniquePtr<MlxArray>>,
}

/// Materialize the time-major weight used by the decode fast path.
/// `conv_weight` is `[hidden, L_cache, 1]` (MLX depthwise layout); transposing
/// to `[1, L_cache, hidden]` lets a single broadcast multiply-and-sum over the
/// `L_cache` axis replace `conv1d`. Materialized once at load so decode never
/// reshapes the weight.
pub(crate) fn build_conv_decode_weight(conv_weight: &MlxArray) -> UniquePtr<MlxArray> {
    // [hidden, L_cache, 1] -> [1, L_cache, hidden].
    let w = mlxcel_core::transpose_axes(conv_weight, &[2, 1, 0]);
    let w = mlxcel_core::contiguous(&w, false);
    mlxcel_core::eval(&w);
    w
}

/// Single decode step of the depthwise causal short conv, computed as a
/// broadcast weighted sum instead of `conv1d` (issue #748).
///
/// For `padded` of shape `[batch, L_cache, hidden]` and `decode_weight` of
/// shape `[1, L_cache, hidden]` this returns `[batch, 1, hidden]` where
/// `out[b, 0, c] = sum_k padded[b, k, c] * decode_weight[0, k, c]`, which is
/// exactly what a stride-1, no-pad, dilation-1, `groups == hidden` `conv1d`
/// produces for a length-1 output. The two-kernel elementwise form avoids the
/// `conv1d` CUDA dispatch (MLX 0.32.1) that sends this tiny bf16 depthwise conv
/// to cuDNN's generic `convolve_common_engine`, which launches one kernel per
/// channel (~1024 for a 350M LFM2) and dominates decode.
pub(crate) fn short_conv_decode_step(
    padded: &MlxArray,
    decode_weight: &MlxArray,
    in_dtype: i32,
) -> UniquePtr<MlxArray> {
    let prod = if mlxcel_core::array_dtype(decode_weight) == in_dtype {
        mlxcel_core::multiply(padded, decode_weight)
    } else {
        let w = mlxcel_core::astype(decode_weight, in_dtype);
        mlxcel_core::multiply(padded, &w)
    };
    mlxcel_core::sum_axis(&prod, 1, true) // [batch, 1, hidden]
}

impl ShortConv {
    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        let in_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.in_proj"), gs, bits)?;
        let out_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.out_proj"), gs, bits)?;
        let conv_weight = get_weight_copy(weights, &format!("{prefix}.conv.weight"))?;
        let conv_bias = weights
            .get(&format!("{prefix}.conv.bias"))
            .map(|w| mlxcel_core::copy(w));

        // Precompute the decode fast-path weight off Metal (issue #748). On
        // Metal `conv1d` already dispatches a fast small-conv kernel, so keep
        // its proven path and leave `decode_weight` unset there.
        let decode_weight = if mlxcel_core::metal_is_available() {
            None
        } else {
            Some(build_conv_decode_weight(&conv_weight))
        };

        Ok(Self {
            in_proj,
            out_proj,
            conv_weight,
            conv_bias,
            hidden_size: args.hidden_size as i32,
            l_cache: args.conv_l_cache as i32,
            decode_weight,
        })
    }

    /// `BCx = in_proj(x)` → split into `B, C, x` → `Bx = B * x` → depthwise
    /// causal Conv1d (kernel `L_cache`, left-padded by `L_cache - 1` on prefill
    /// or prepended with the cached `L_cache - 1` time steps on decode) →
    /// `y = C * conv_out` → `out_proj(y)`.
    fn forward(
        &self,
        x: &MlxArray,
        conv_state: &mut Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_size;
        let bcx = self.in_proj.forward(x);

        // Split along the last axis into B, C, x (each [batch, seq, hidden]).
        let b_part = slice_axis(&bcx, -1, 0, h);
        let c_part = slice_axis(&bcx, -1, h, 2 * h);
        let x_part = slice_axis(&bcx, -1, 2 * h, 3 * h);

        let bx = mlxcel_core::multiply(&b_part, &x_part);

        // Prepend the cached tail (decode) or zero-pad by L_cache - 1 (prefill),
        // so the kernel-size-`L_cache` depthwise conv stays causal.
        let n_keep = self.l_cache - 1;
        let bx_shape = mlxcel_core::array_shape(&bx);
        let padded = match conv_state.as_ref().and_then(|s| s.as_ref()) {
            Some(state) => mlxcel_core::concatenate(state, &bx, 1),
            None => {
                let pad = mlxcel_core::zeros(
                    &[bx_shape[0], n_keep, bx_shape[2]],
                    mlxcel_core::array_dtype(&bx),
                );
                mlxcel_core::concatenate(&pad, &bx, 1)
            }
        };

        // Persist the last `L_cache - 1` time steps as the next conv state.
        // `contiguous` forces a fresh buffer so the cached slice does not retain
        // the full padded allocation (mirrors the Mamba conv-state handling).
        let plen = mlxcel_core::array_shape(&padded)[1];
        let tail = slice_axis(&padded, 1, plen - n_keep, plen);
        *conv_state = Some(mlxcel_core::contiguous(&tail, false));

        // Depthwise Conv1d (groups == hidden). Match the conv weight dtype to
        // the (possibly bf16) activations; quantized checkpoints leave the
        // non-quantized conv weight at its stored precision.
        //
        // Off Metal, any single-position output step (a decode step, or a
        // 1-token prefill) computes the conv as an explicit weighted sum of
        // the L_cache taps instead of calling `conv1d`: a tiny bf16 depthwise
        // conv on CUDA (MLX 0.32.1) otherwise falls into cuDNN's generic
        // `convolve_common_engine`, which launches one kernel per channel and
        // regressed lfm2-350m-8bit decode ~10x (issue #748). Multi-position
        // outputs (prefill, speculative verify) and Metal keep `conv1d`.
        let in_dtype = mlxcel_core::array_dtype(&padded);
        let conv_out = if let (Some(decode_weight), 1) = (&self.decode_weight, bx_shape[1]) {
            short_conv_decode_step(&padded, decode_weight, in_dtype)
        } else if mlxcel_core::array_dtype(&self.conv_weight) == in_dtype {
            mlxcel_core::conv1d(&padded, &self.conv_weight, 1, 0, 1, h)
        } else {
            let w = mlxcel_core::astype(&self.conv_weight, in_dtype);
            mlxcel_core::conv1d(&padded, &w, 1, 0, 1, h)
        };
        let conv_out = match &self.conv_bias {
            Some(bias) => {
                let bias = mlxcel_core::reshape(bias, &[1, 1, -1]);
                mlxcel_core::add(&conv_out, &bias)
            }
            None => conv_out,
        };

        let y = mlxcel_core::multiply(&c_part, &conv_out);
        self.out_proj.forward(&y)
    }
}

// Dense SwiGLU MLP.

struct MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MLP {
    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.up_proj"), gs, bits)?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                gs,
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

// Sparse MoE block (sigmoid-gated routing).

struct Lfm2MoeSparseMoeBlock {
    gate: UnifiedLinear,
    switch_mlp: SwitchGLU,
    expert_bias: Option<UniquePtr<MlxArray>>,
    top_k: i32,
    num_experts: i32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
}

impl Lfm2MoeSparseMoeBlock {
    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        // The router stays a plain (possibly quantized) Linear. `UnifiedLinear`
        // reconciles the per-tensor bit width from the tensor shapes, so the
        // 8-bit gate is loaded correctly even when the checkpoint's top-level
        // quantization advertises 4-bit experts.
        let gate = UnifiedLinear::from_weights(weights, &format!("{prefix}.gate"), gs, bits)?;
        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{prefix}.switch_mlp"), gs, bits)?;
        let expert_bias = weights
            .get(&format!("{prefix}.expert_bias"))
            .map(|w| mlxcel_core::copy(w));

        Ok(Self {
            gate,
            switch_mlp,
            expert_bias,
            top_k: args.num_experts_per_tok.unwrap_or(1) as i32,
            num_experts: args.num_experts.unwrap_or(0) as i32,
            norm_topk_prob: args.norm_topk_prob.unwrap_or(false),
            routed_scaling_factor: args.routed_scaling_factor,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Sigmoid-gated routing (NOT softmax). See module note + ml-explore/mlx-lm#1354.
        let routing_weights = mlxcel_core::sigmoid(&self.gate.forward(&x_flat));

        let k = self.top_k;
        let kth = self.num_experts - k;

        // The expert_bias shifts only the SELECTION, never the gathered scores.
        let selection = match &self.expert_bias {
            Some(bias) => {
                let rw_f32 = mlxcel_core::astype(&routing_weights, mlxcel_core::dtype::FLOAT32);
                let bias_row = mlxcel_core::reshape(bias, &[1, self.num_experts]);
                mlxcel_core::add(&rw_f32, &bias_row)
            }
            None => mlxcel_core::copy(&routing_weights),
        };

        let indices = mlxcel_core::argpartition(&selection, kth, -1);
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

        // Gather scores from the UNBIASED routing weights at the selected experts.
        let mut scores = mlxcel_core::take_along_axis(&routing_weights, &topk_indices, -1);

        if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&scores, -1, true);
            let eps = mlxcel_core::full_f32(&[1], 1e-6, mlxcel_core::array_dtype(&sum));
            let denom = mlxcel_core::add(&sum, &eps);
            scores = mlxcel_core::divide(&scores, &denom);
        }
        if (self.routed_scaling_factor - 1.0).abs() > f32::EPSILON {
            scores = mlxcel_core::multiply_scalar(&scores, self.routed_scaling_factor);
        }

        let result = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1
                && crate::models::switch_layers::fused_moe_enabled()
            {
                self.switch_mlp
                    .forward_fused_kernel(&x_flat, &topk_indices, &scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices);
                    moe_weighted_sum(&expert_out, &scores, mlxcel_core::array_dtype(&x_flat))
                }
            }
        };

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// Per-layer feed-forward (dense or sparse).

enum FeedForward {
    Dense(MLP),
    Moe(Lfm2MoeSparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::Moe(moe) => moe.forward(x),
        }
    }
}

// Per-layer token mixer (attention or short-conv).

enum Mixer {
    Attention(Attention),
    Conv(ShortConv),
}

// Decoder layer: operator_norm → mixer → residual; ffn_norm → feed_forward → residual.

struct Lfm2DecoderLayer {
    mixer: Mixer,
    feed_forward: FeedForward,
    operator_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

impl Lfm2DecoderLayer {
    fn is_attention(&self) -> bool {
        matches!(self.mixer, Mixer::Attention(_))
    }

    fn make_cache(&self) -> Lfm2LayerCache {
        match self.mixer {
            Mixer::Attention(_) => Lfm2LayerCache::Attention(KVCache::new()),
            Mixer::Conv(_) => Lfm2LayerCache::Conv(None),
        }
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut Lfm2LayerCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.operator_norm.forward(x);
        // `make_caches` builds the per-layer cache in lockstep with the mixer
        // kind, so the cross combinations never occur.
        let r = match (&self.mixer, cache) {
            (Mixer::Attention(attn), Lfm2LayerCache::Attention(kv)) => {
                attn.forward(&normed, kv, mask)
            }
            (Mixer::Conv(conv), Lfm2LayerCache::Conv(state)) => conv.forward(&normed, state),
            _ => unreachable!("LFM2 layer cache kind does not match its mixer"),
        };
        let h = mlxcel_core::add(x, &r);

        let normed = self.ffn_norm.forward(&h);
        let ff = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &ff)
    }

    fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");

        let mixer = if args.is_attention_layer(layer_idx) {
            Mixer::Attention(Attention::from_weights(
                weights,
                args,
                &format!("{prefix}.self_attn"),
            )?)
        } else {
            Mixer::Conv(ShortConv::from_weights(
                weights,
                args,
                &format!("{prefix}.conv"),
            )?)
        };

        let feed_forward = if args.layer_is_moe(layer_idx) {
            FeedForward::Moe(Lfm2MoeSparseMoeBlock::from_weights(
                weights,
                args,
                &format!("{prefix}.feed_forward"),
            )?)
        } else {
            FeedForward::Dense(MLP::from_weights(
                weights,
                args,
                &format!("{prefix}.feed_forward"),
            )?)
        };

        let operator_norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.operator_norm.weight"))?,
            args.norm_eps,
        );
        let ffn_norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.ffn_norm.weight"))?,
            args.norm_eps,
        );

        Ok(Self {
            mixer,
            feed_forward,
            operator_norm,
            ffn_norm,
        })
    }
}

// Per-layer cache: KV cache for attention layers, conv state for short-conv.

pub enum Lfm2LayerCache {
    Attention(KVCache),
    /// Last `L_cache - 1` time steps of `Bx`, shape `[batch, L_cache - 1, hidden]`.
    Conv(Option<UniquePtr<MlxArray>>),
}

impl Lfm2LayerCache {
    fn offset(&self) -> i32 {
        match self {
            Lfm2LayerCache::Attention(kv) => kv.offset,
            Lfm2LayerCache::Conv(_) => 0,
        }
    }

    fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        match self {
            Lfm2LayerCache::Attention(kv) => {
                push_optional(snapshot, format!("{prefix}.attention.keys"), &kv.keys);
                push_optional(snapshot, format!("{prefix}.attention.values"), &kv.values);
            }
            Lfm2LayerCache::Conv(state) => {
                push_optional(snapshot, format!("{prefix}.conv.state"), state);
            }
        }
    }

    fn restore_from(&mut self, snapshot: &mlxcel_core::generate::ModelStateSnapshot, prefix: &str) {
        match self {
            Lfm2LayerCache::Attention(kv) => {
                kv.keys = restore_optional(snapshot, format!("{prefix}.attention.keys"));
                kv.values = restore_optional(snapshot, format!("{prefix}.attention.values"));
                kv.offset = snapshot.token_len() as i32;
            }
            Lfm2LayerCache::Conv(state) => {
                *state = restore_optional(snapshot, format!("{prefix}.conv.state"));
            }
        }
    }
}

// LFM2 model.

pub struct Lfm2Model {
    config: ModelArgs,
    embed_tokens: UnifiedEmbedding,
    layers: Vec<Lfm2DecoderLayer>,
    embedding_norm: RMSNorm,
    eos_token_ids: Vec<i32>,
    /// Model-owned mixed caches keyed by scheduler sequence id, plus a fallback
    /// slot for CLI / benchmark paths.
    sequence_state: ModelOwnedSequenceState<Lfm2LayerCache>,
}

impl Lfm2Model {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<Lfm2LayerCache> {
        self.layers.iter().map(|l| l.make_cache()).collect()
    }

    fn forward_with_caches(
        &self,
        inputs: &MlxArray,
        caches: &mut [Lfm2LayerCache],
    ) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(inputs);
        self.forward_embeds_with_caches(&h, caches)
    }

    /// Token embedding front end. VLM wrappers (e.g. LFM2-VL) call this, scatter
    /// image features into the `<image>` rows, and feed the result to
    /// [`Self::forward_embeds_with_caches`]. `forward_with_caches` is exactly this
    /// followed by the layer stack, so text-only behavior is unchanged.
    pub fn input_embeddings(&self, inputs: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(inputs)
    }

    /// Layer stack over pre-computed input embeddings (the VLM injection point).
    fn forward_embeds_with_caches(
        &self,
        input_embeds: &MlxArray,
        caches: &mut [Lfm2LayerCache],
    ) -> UniquePtr<MlxArray> {
        let mut h = mlxcel_core::copy(input_embeds);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        // Single causal mask anchored on the first attention layer's offset.
        // Short-conv layers stay causal via left-padding, so they need no mask
        // in the single-sequence path.
        let attn_offset = caches
            .iter()
            .find(|c| matches!(c, Lfm2LayerCache::Attention(_)))
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

        let h = self.embedding_norm.forward(&h);
        self.embed_tokens.as_linear(&h)
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

    /// Build from owned weights. Applies the LFM2 weight sanitize internally so
    /// every load path (directory loader and the adapter/owned-weights route)
    /// produces the same canonical layout.
    pub fn from_weights(args: ModelArgs, weights: WeightMap) -> Result<Self, String> {
        let weights = sanitize_weights(weights, &args);

        let gs = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", gs, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(Lfm2DecoderLayer::from_weights(&weights, &args, i)?);
        }

        let embedding_norm = RMSNorm::new(
            get_weight_copy(&weights, "model.embedding_norm.weight")?,
            args.norm_eps,
        );

        let internal_caches: Vec<Lfm2LayerCache> = layers.iter().map(|l| l.make_cache()).collect();
        let eos_token_ids = args.eos_token_ids();

        Ok(Self {
            config: args,
            embed_tokens,
            layers,
            embedding_norm,
            eos_token_ids,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
        })
    }
}

// Weight sanitize (mirrors the reference `sanitize`).
//
// 1. Depthwise conv weight orientation: PyTorch stores `[hidden, 1, L_cache]`;
//    MLX/mlxcel wants `[hidden, L_cache, 1]`. The mlx-community checkpoints are
//    already in MLX layout, so this is a no-op there but keeps raw HF exports
//    loadable.
// 2. Dense MLP rename `w1 → gate_proj`, `w2 → down_proj`, `w3 → up_proj`
//    (the dense `lfm2` checkpoint stores the SwiGLU MLP as w1/w2/w3).
// 3. MoE expert stacking `feed_forward.experts.{e}.{proj} → feed_forward.switch_mlp.{proj}`.
//    The 4-bit mlx-community MoE checkpoint already ships the stacked
//    `switch_mlp` tensors, so this only fires for non-pre-stacked exports.
fn sanitize_weights(mut weights: WeightMap, args: &ModelArgs) -> WeightMap {
    // 1. Conv weight orientation.
    let conv_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("conv.conv.weight"))
        .cloned()
        .collect();
    for k in conv_keys {
        if let Some(v) = weights.get(&k) {
            let shape = mlxcel_core::array_shape(v);
            if shape.len() == 3 && shape[shape.len() - 1] > shape[1] {
                let transposed = mlxcel_core::transpose_axes(v, &[0, 2, 1]);
                weights.insert(k, transposed);
            }
        }
    }

    // 2. Dense MLP w1/w2/w3 → gate/down/up rename (covers weight + quantized
    //    scales/biases by matching the `.feed_forward.wN.` key segment).
    let rename = |key: &str| -> Option<String> {
        for (old, new) in [
            (".feed_forward.w1.", ".feed_forward.gate_proj."),
            (".feed_forward.w2.", ".feed_forward.down_proj."),
            (".feed_forward.w3.", ".feed_forward.up_proj."),
        ] {
            if key.contains(old) {
                return Some(key.replace(old, new));
            }
        }
        None
    };
    let to_rename: Vec<String> = weights
        .keys()
        .filter(|k| rename(k.as_str()).is_some())
        .cloned()
        .collect();
    for key in to_rename {
        if let Some(new_key) = rename(&key)
            && let Some(v) = weights.remove(&key)
        {
            weights.insert(new_key, v);
        }
    }

    // 3. MoE expert stacking (no-op for the pre-stacked mlx checkpoint).
    if let Some(num_experts) = args.num_experts {
        for l in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{l}.feed_forward");
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let first = format!("{prefix}.experts.0.{proj}.weight");
                if !weights.contains_key(&first) {
                    continue;
                }
                for suffix in ["weight", "scales", "biases"] {
                    let probe = format!("{prefix}.experts.0.{proj}.{suffix}");
                    if !weights.contains_key(&probe) {
                        continue;
                    }
                    let mut stacked: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(num_experts);
                    let mut complete = true;
                    for e in 0..num_experts {
                        match weights.remove(&format!("{prefix}.experts.{e}.{proj}.{suffix}")) {
                            Some(w) => stacked.push(w),
                            None => {
                                complete = false;
                                break;
                            }
                        }
                    }
                    if complete && !stacked.is_empty() {
                        let joined = stack_arrays(&stacked, 0);
                        weights.insert(format!("{prefix}.switch_mlp.{proj}.{suffix}"), joined);
                    }
                }
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

impl LanguageModel for Lfm2Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // LFM2 owns its mixed cache (KVCache + conv state); the external KV
        // slice is unused. The fallback internal state covers CLI / benchmarks.
        self.sequence_state
            .with_sequence_state(None, |internal| self.forward_with_caches(input, internal))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset fallback internal caches for a fresh generation session.
        self.sequence_state
            .replace_internal(Lfm2Model::make_caches(self));
        // Return placeholder KV caches for trait compatibility; the real caches
        // are model-owned.
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
        false // Hybrid short-conv state is not compatible with per-sequence KV isolation.
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt the short-conv recurrent state.
    }

    fn prepare_sequence_state(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, Lfm2Model::make_caches(self));
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
            || Lfm2Model::make_caches(self),
            |internal| self.forward_with_caches(input_ids, internal),
        )
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_sequence_state(None, |internal| match input_embeddings {
                Some(embeds) => self.forward_embeds_with_caches(embeds, internal),
                None => self.forward_with_caches(input_ids, internal),
            })
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || Lfm2Model::make_caches(self),
            |internal| match input_embeddings {
                Some(embeds) => self.forward_embeds_with_caches(embeds, internal),
                None => self.forward_with_caches(input_ids, internal),
            },
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
                    mlxcel_core::generate::ModelStateSnapshot::new("lfm2", token_len);
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
        if snapshot.family() != "lfm2" {
            return Err(format!(
                "cannot restore {} snapshot into LFM2",
                snapshot.family()
            ));
        }
        let mut state = Lfm2Model::make_caches(self);
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
                    Lfm2LayerCache::Attention(kv) => {
                        kv.trim(excess);
                    }
                    Lfm2LayerCache::Conv(state) => {
                        // The conv state is recurrent (not positional); it was
                        // computed from padding tokens, so reset it.
                        *state = None;
                    }
                }
            }
        });
    }
}
