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

//! Kimi K3 text backbone (`model_type: "kimi_k3"`, `text_config.model_type:
//! "kimi_linear"`).
//!
//! The decoder reuses the Kimi Linear design (`src/models/kimi_linear.rs`)
//! and adds five mechanisms that module does not have:
//!
//! - **Fused-QKV KDA** with a full-rank output gate and a lower-bounded gate
//!   in the gated delta recurrence (`gate_lower_bound = -5.0`).
//! - **Gated q-LoRA NoPE MLA**: `q_a_proj -> q_a_layernorm -> q_b_proj` and a
//!   sigmoid output gate `g_proj` before `o_proj`; no rotation is applied to
//!   the `q_pe` / `k_pe` halves.
//! - **SiTU activation** everywhere a SwiGLU would be.
//! - **Latent MoE**: routed experts run on a `routed_expert_hidden_size`
//!   projection of the hidden state, followed by an RMSNorm and a projection
//!   back, with the shared experts on the full hidden size.
//! - **Attention Residuals**: every `attn_res_block_size` layers the running
//!   residual is frozen into a block list, and each sublayer input is a
//!   softmax mix of the frozen blocks and the current partial sum.
//!
//! `kimi_linear.rs` stays the reference for Kimi Linear 48B; the pieces
//! shared with it (`ShortConv1d`, `MultiLinear`, the `kv_b_proj`
//! decomposition) are borrowed from there rather than copied.
//!
//! Reference: https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py
//! (text backbone) and https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_k3.py
//! (the `language_model.` wrapper). Vision (#1342) and the tokenizer / chat
//! renderer (#1338) are separate.

use crate::models::conv_decode::{build_conv_decode_weight, short_conv_decode_step};
use crate::models::gated_delta::{
    gated_delta_update_with_lower_bound, scaled_fast_rms_norm_no_weight,
};
use crate::models::kimi_linear::{MultiLinear, Quantization, ShortConv1d};
use crate::models::switch_layers::{SwitchGLU, SwitchGluActivation, situ_activation};
use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::dtype;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, silu, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use super::model_owned::ModelOwnedSequenceState;

#[path = "kimi_k3_sanitize.rs"]
mod sanitize;

/// Epsilon of the MLA `q_a_layernorm` / `kv_a_layernorm`. The reference
/// builds both as `KimiRMSNorm(rank)` with the class default of `1e-6`
/// rather than `rms_norm_eps` (which is `1e-5` on the published config).
const MLA_LORA_NORM_EPS: f32 = 1e-6;

/// Epsilon of the FLA in-kernel l2norm applied to the KDA q/k
/// (`use_qk_l2norm_in_kernel=True`). See [`KimiK3TextConfig::qk_norm_eps`].
const KDA_L2NORM_EPS: f32 = 1e-6;

// Configuration.

#[derive(Debug, Clone, Deserialize)]
pub struct KimiK3LinearAttnConfig {
    /// 1-based indices of the KDA layers.
    pub kda_layers: Vec<usize>,
    /// 1-based indices of the MLA layers (informational; layer `i` is KDA iff
    /// `i + 1` is in `kda_layers`, otherwise MLA).
    #[serde(default)]
    pub full_attn_layers: Vec<usize>,
    pub num_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_conv_kernel")]
    pub short_conv_kernel_size: usize,
    /// Lower bound of the log-gate (published `-5.0`); `None` keeps the
    /// softplus gate of Kimi Linear.
    #[serde(default)]
    pub gate_lower_bound: Option<f32>,
    /// `g_proj` (`[P, hidden]`) instead of `g_b_proj(g_a_proj(x))`.
    #[serde(default)]
    pub use_full_rank_gate: bool,
}

fn default_conv_kernel() -> usize {
    4
}

#[derive(Debug, Clone, Deserialize)]
pub struct KimiK3TextConfig {
    #[serde(default = "default_text_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Width of the dense MLP (layer 0 on the published config).
    pub intermediate_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Must be `"situ"`.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// SiTU `beta` (published `4.0`; the reference defaults to `1.0`).
    #[serde(default)]
    pub activation_situ_beta: Option<f32>,
    /// SiTU `linear_beta` (published `25.0`); `None` means no input tanh.
    #[serde(default)]
    pub activation_situ_linear_beta: Option<f32>,
    /// Attention Residuals block size (published `12`); `None` disables them.
    #[serde(default)]
    pub attn_res_block_size: Option<usize>,
    pub linear_attn_config: KimiK3LinearAttnConfig,
    /// q-LoRA rank (published `1536`); `None` means a plain `q_proj`.
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Must be `true`.
    #[serde(default)]
    pub mla_use_nope: bool,
    #[serde(default)]
    pub mla_use_output_gate: bool,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default = "default_one_usize")]
    pub num_experts_per_token: usize,
    #[serde(default)]
    pub num_shared_experts: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    /// Latent width the routed experts run on (published `3584`); `None`
    /// means the experts run on `hidden_size` with no down / up projection.
    #[serde(default)]
    pub routed_expert_hidden_size: Option<usize>,
    #[serde(default)]
    pub latent_moe_use_norm: bool,
    /// Must be `"sigmoid"`.
    #[serde(default = "default_sigmoid")]
    pub moe_router_activation_func: String,
    #[serde(default = "default_true")]
    pub moe_renormalize: bool,
    #[serde(default = "default_one_f32")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_one_usize")]
    pub moe_layer_freq: usize,
    #[serde(default = "default_true")]
    pub use_grouped_topk: bool,
    /// Must be `1` (grouped routing is not implemented for this family).
    #[serde(default = "default_one_usize")]
    pub num_expert_group: usize,
    /// Must be `1`.
    #[serde(default = "default_one_usize")]
    pub topk_group: usize,
    /// MTP heads; out of scope, their weights are dropped at load.
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// MLX-style `quantization` block. Absent on the published checkpoint,
    /// whose only quantized planes are the compressed-tensors mxfp4 experts;
    /// honored for a future MLX affine conversion of the dense projections.
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

fn default_text_model_type() -> String {
    "kimi_linear".to_string()
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_hidden_act() -> String {
    "situ".to_string()
}
fn default_one_usize() -> usize {
    1
}
fn default_one_f32() -> f32 {
    1.0
}
fn default_sigmoid() -> String {
    "sigmoid".to_string()
}
fn default_true() -> bool {
    true
}

impl KimiK3TextConfig {
    /// Reject the config values this port does not implement, naming the
    /// field, so a checkpoint that would otherwise load and produce fluent
    /// nonsense (a softmax router, a rotated MLA, a SwiGLU MLP) fails at
    /// load instead.
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_act != "situ" {
            return Err(format!(
                "kimi_k3: text_config.hidden_act must be \"situ\", got {:?}",
                self.hidden_act
            ));
        }
        if self.moe_router_activation_func != "sigmoid" {
            return Err(format!(
                "kimi_k3: text_config.moe_router_activation_func must be \"sigmoid\", got {:?}",
                self.moe_router_activation_func
            ));
        }
        if !self.mla_use_nope {
            return Err(
                "kimi_k3: text_config.mla_use_nope must be true (the MLA path applies no rotation)"
                    .to_string(),
            );
        }
        if self.num_expert_group != 1 {
            return Err(format!(
                "kimi_k3: text_config.num_expert_group must be 1 (grouped routing is not \
                 implemented for this family), got {}",
                self.num_expert_group
            ));
        }
        if self.topk_group != 1 {
            return Err(format!(
                "kimi_k3: text_config.topk_group must be 1, got {}",
                self.topk_group
            ));
        }
        if self.attn_res_block_size == Some(0) {
            return Err(
                "kimi_k3: text_config.attn_res_block_size must be positive or null".to_string(),
            );
        }
        if self.moe_layer_freq == 0 {
            return Err("kimi_k3: text_config.moe_layer_freq must be positive".to_string());
        }
        if self.num_experts > 0 {
            // The router calls `argpartition(kth = k - 1)` and slices `[0, k)`
            // off the result, so a `k` of zero or one above `num_experts`
            // indexes past the score axis inside MLX rather than here.
            if self.num_experts_per_token == 0 {
                return Err(
                    "kimi_k3: text_config.num_experts_per_token must be at least 1 when \
                     num_experts is positive"
                        .to_string(),
                );
            }
            if self.num_experts_per_token > self.num_experts {
                return Err(format!(
                    "kimi_k3: text_config.num_experts_per_token ({}) exceeds num_experts ({})",
                    self.num_experts_per_token, self.num_experts
                ));
            }
        }
        if self.routed_expert_hidden_size == Some(0) {
            return Err(
                "kimi_k3: text_config.routed_expert_hidden_size must be positive or null"
                    .to_string(),
            );
        }
        if self.linear_attn_config.num_heads == 0
            || self.linear_attn_config.head_dim == 0
            || self.linear_attn_config.short_conv_kernel_size == 0
        {
            return Err(
                "kimi_k3: text_config.linear_attn_config num_heads, head_dim and \
                 short_conv_kernel_size must be positive"
                    .to_string(),
            );
        }
        if let Some(beta) = self.activation_situ_beta
            && !(beta.is_finite() && beta > 0.0)
        {
            return Err(format!(
                "kimi_k3: text_config.activation_situ_beta must be a positive finite number, got {beta}"
            ));
        }
        if let Some(lb) = self.activation_situ_linear_beta
            && !(lb.is_finite() && lb > 0.0)
        {
            return Err(format!(
                "kimi_k3: text_config.activation_situ_linear_beta must be a positive finite \
                 number or null, got {lb}"
            ));
        }
        if let Some(q) = &self.quantization {
            mlxcel_core::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|e| format!("kimi_k3: text_config.quantization: {e}"))?;
        }
        Ok(())
    }

    pub fn group_size(&self) -> i32 {
        self.quantization.as_ref().map_or(64, |q| q.group_size)
    }
    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map_or(4, |q| q.bits)
    }
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    pub fn is_linear_layer(&self, idx: usize) -> bool {
        self.linear_attn_config.kda_layers.contains(&(idx + 1))
    }
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        self.num_experts > 0
            && idx >= self.first_k_dense_replace
            && self.moe_layer_freq > 0
            && idx.is_multiple_of(self.moe_layer_freq)
    }
    pub fn delta_num_heads(&self) -> usize {
        self.linear_attn_config.num_heads
    }
    pub fn delta_head_dim(&self) -> usize {
        self.linear_attn_config.head_dim
    }
    pub fn delta_projection_dim(&self) -> usize {
        self.delta_num_heads() * self.delta_head_dim()
    }
    pub fn situ_beta(&self) -> f32 {
        self.activation_situ_beta.unwrap_or(1.0)
    }
    pub fn situ_activation(&self) -> SwitchGluActivation {
        SwitchGluActivation::SiTU {
            beta: self.situ_beta(),
            linear_beta: self.activation_situ_linear_beta,
        }
    }
    pub fn use_attn_residuals(&self) -> bool {
        self.attn_res_block_size.is_some()
    }

    /// Epsilon handed to `rms_norm` on the KDA q/k so that the scaled RMS
    /// norm equals FLA's l2norm.
    ///
    /// The reference runs `use_qk_l2norm_in_kernel=True`, which is
    /// `l2norm(x) = x / sqrt(sum(x^2) + 1e-6)`. With `D = head_dim`,
    /// `sum(x^2) = D * mean(x^2)`, so
    /// `l2norm(x) = D^-0.5 * x / sqrt(mean(x^2) + 1e-6 / D)
    ///            = D^-0.5 * rms_norm(x, eps = 1e-6 / D)`.
    /// The attention scale is folded in the same way as Kimi Linear does:
    /// `q = scale^2 * rms_norm(q)` is `scale * l2norm(q)` and
    /// `k = scale * rms_norm(k)` is `l2norm(k)`, with `scale = D^-0.5`.
    /// Kimi Linear in this tree uses mlx-lm's plain `1e-6` and is left as is.
    pub fn qk_norm_eps(&self) -> f32 {
        KDA_L2NORM_EPS / self.delta_head_dim() as f32
    }
}

/// Top-level `config.json` of a Kimi K3 checkpoint. `vision_config` is
/// carried as opaque JSON: the text backbone never reads it, and #1342 owns
/// the vision tower.
#[derive(Debug, Clone, Deserialize)]
pub struct KimiK3Config {
    pub model_type: String,
    pub text_config: KimiK3TextConfig,
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub media_placeholder_token_id: Option<i64>,
}

impl KimiK3Config {
    /// Parse and validate a top-level `config.json`.
    pub fn from_json_str(config_str: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(config_str)
            .map_err(|e| format!("Failed to parse kimi_k3 config.json: {e}"))?;
        config.text_config.validate()?;
        Ok(config)
    }

    /// Stop tokens from the top-level `eos_token_id` (163586 on the published
    /// config), falling back to the text config's.
    pub fn eos_token_ids(&self) -> Vec<i32> {
        let top = super::parse_optional_eos_token_ids(&self.eos_token_id);
        if !top.is_empty() {
            return top;
        }
        super::parse_optional_eos_token_ids(&self.text_config.eos_token_id)
    }
}

// Caches.

/// Recurrent state of one KDA layer: the fused conv window (`[B, K-1, 3P]`
/// in the activation dtype) and the float32 SSM state (`[B, H, Dv, Dk]`).
pub struct KimiK3DeltaCache {
    pub conv: Option<UniquePtr<MlxArray>>,
    pub ssm: Option<UniquePtr<MlxArray>>,
    pub offset: i32,
}

impl KimiK3DeltaCache {
    pub fn new() -> Self {
        Self {
            conv: None,
            ssm: None,
            offset: 0,
        }
    }

    pub fn advance(&mut self, step: i32) {
        self.offset += step;
    }
}

impl Default for KimiK3DeltaCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-layer cache: recurrent state for KDA layers, an MLA latent pair
/// (`kv_latent`, `k_pe`) in a plain `KVCache` for the attention layers.
pub enum KimiK3LayerCache {
    Delta(KimiK3DeltaCache),
    Attn(KVCache),
}

impl KimiK3LayerCache {
    pub fn offset(&self) -> i32 {
        match self {
            KimiK3LayerCache::Delta(d) => d.offset,
            KimiK3LayerCache::Attn(kv) => kv.offset,
        }
    }
}

fn make_layer_caches(layers: &[KimiK3DecoderLayer]) -> Vec<KimiK3LayerCache> {
    layers
        .iter()
        .map(|layer| {
            if layer.is_linear {
                KimiK3LayerCache::Delta(KimiK3DeltaCache::new())
            } else {
                KimiK3LayerCache::Attn(KVCache::new())
            }
        })
        .collect()
}

// Attention Residuals.

/// The frozen residual states of the current forward call, one entry per
/// block boundary crossed so far, with their float32 inverse RMS
/// precomputed once at push time.
///
/// The blocks are per-token rows (`[B, T, D]`), so a decode step's blocks are
/// computed from that step's own hidden state at the block-boundary layers,
/// exactly as in the reference forward, and nothing is carried across calls:
/// the list is rebuilt from the embeddings on every `forward`. Attention
/// Residuals mix a token only with its own frozen states, never with other
/// positions, which is why no cache entry exists for them.
pub(crate) struct ResidualBlocks {
    raw: Vec<UniquePtr<MlxArray>>,
    inv_rms: Vec<UniquePtr<MlxArray>>,
}

impl ResidualBlocks {
    pub(crate) fn new() -> Self {
        Self {
            raw: Vec::new(),
            inv_rms: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.raw.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// Freeze `x` as the next block.
    pub(crate) fn push(&mut self, x: &MlxArray, eps: f32) {
        self.inv_rms.push(inv_rms_f32(x, eps));
        self.raw.push(mlxcel_core::copy(x));
    }
}

/// `rsqrt(mean(x^2, -1) + eps)` in float32, shape `[.., 1]`.
fn inv_rms_f32(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    let xf = mlxcel_core::astype(x, dtype::FLOAT32);
    let ms = mlxcel_core::mean_axis(&mlxcel_core::square(&xf), -1, true);
    let eps_arr = mlxcel_core::full_f32(&[1], eps, dtype::FLOAT32);
    mlxcel_core::rsqrt(&mlxcel_core::add(&ms, &eps_arr))
}

/// Attention Residuals mix of the frozen `blocks` and the current `partial`
/// (`[B, T, D]`), with the effective score weight `w_eff` (`[D, 1]` float32,
/// see [`load_attn_res_weight`]).
///
/// ```text
/// logit_k = (raw_k.f32 @ w_eff) * inv_rms_k          for every stored block
/// logit_p = (partial.f32 @ w_eff) * rsqrt(mean(partial^2) + eps)
/// p = softmax([logit_0, .., logit_{K-1}, logit_p])
/// out = (sum_k p_k * raw_k + p_p * partial).astype(partial.dtype)
/// ```
///
/// Returns `partial` unchanged when no block is stored yet. Logits, softmax
/// and the weighted sum run in float32 and only the result is cast back, as
/// `_apply_attn_res` does in the reference.
pub(crate) fn attn_res_mix(
    blocks: &ResidualBlocks,
    partial: &MlxArray,
    w_eff: &MlxArray,
    eps: f32,
) -> UniquePtr<MlxArray> {
    if blocks.is_empty() {
        return mlxcel_core::copy(partial);
    }
    let out_dtype = mlxcel_core::array_dtype(partial);
    let n = blocks.len();

    // Every block enters `matmul` and `multiply` in its stored dtype rather
    // than through a float32 array this function holds. MLX promotes a
    // half-precision operand against the float32 `w_eff` (and against the
    // float32 probability column below) inside the op, so the arithmetic is
    // identical, but the promoted copy is then an op-local temporary MLX
    // frees as soon as that one op is done. A promotion shared between the
    // logit matmul and the weighted term would instead be a graph node with
    // two consumers, and every consumer of the softmax runs after every
    // logit, so all of them would have to stay resident from the first loop
    // until the fold: at D = 7168 with a full 12-layer window that is the
    // whole residual window of a 4096-token prefill live in float32 at once,
    // about 0.9 GB, which is the allocation this mix exists to avoid.
    let mut logits: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(n + 1);
    for (raw, inv_rms) in blocks.raw.iter().zip(blocks.inv_rms.iter()) {
        logits.push(mlxcel_core::multiply(
            &mlxcel_core::matmul(raw, w_eff),
            inv_rms,
        ));
    }
    let inv_p = inv_rms_f32(partial, eps);
    logits.push(mlxcel_core::multiply(
        &mlxcel_core::matmul(partial, w_eff),
        &inv_p,
    ));

    // [B, T, 1] x (n + 1) -> [B, T, n + 1, 1] -> [B, T, n + 1]
    let logits = mlxcel_core::squeeze_axis(&stack_arrays(&logits, -2), -1);
    let probs = mlxcel_core::softmax(&logits, -1);

    // Fold the weighted sum one term at a time. Stacking the values into
    // [B, T, n + 1, D] and folding with a single matmul is the same
    // arithmetic up to summation order, but it allocates a contiguous float32
    // copy of every stored block on top of the blocks themselves: at D = 7168
    // with a full 12-layer window that is about 0.9 GB of transient for a
    // 4096-token prefill, and it grows with the block count. The per-term form
    // broadcasts one [B, T, 1] probability column against one [B, T, D] block,
    // so only the running accumulator and the current term are resident.
    let probs_shape = mlxcel_core::array_shape(&probs);
    let last = probs_shape.len() - 1;
    let mut mixed: Option<UniquePtr<MlxArray>> = None;
    for i in 0..=n {
        let mut start = vec![0i32; probs_shape.len()];
        let mut stop = probs_shape.clone();
        start[last] = i as i32;
        stop[last] = i as i32 + 1;
        let weight = mlxcel_core::slice(&probs, &start, &stop);
        let value: &MlxArray = match blocks.raw.get(i) {
            Some(raw) => raw,
            None => partial,
        };
        let term = mlxcel_core::multiply(value, &weight);
        mixed = Some(match mixed {
            Some(acc) => mlxcel_core::add(&acc, &term),
            None => term,
        });
    }
    let mixed = mixed.expect("attn_res_mix: at least the partial term is always present");
    if out_dtype == dtype::FLOAT32 {
        mixed
    } else {
        mlxcel_core::astype(&mixed, out_dtype)
    }
}

/// `w_eff = res_norm.weight.f32 * res_proj.weight.reshape(D)` as a `[D, 1]`
/// float32 column, precomputed once at load so the mix is one matmul per
/// stored block.
///
/// Both tensors are cross-checked against `hidden_size` first. Neither MLX
/// call below reports anything the caller can catch: a norm and a projection
/// of different widths abort the process inside `multiply` here, and a pair
/// that agrees with each other but not with `hidden_size` survives load and
/// aborts inside [`attn_res_mix`]'s `matmul` on the first forward pass, with
/// no key to name. Both are checkpoint data (a partially converted export, a
/// hand-edited `text_config`, a checkpoint paired with the wrong config), so
/// they are named here instead.
fn load_attn_res_weight(
    weights: &WeightMap,
    proj_key: &str,
    norm_key: &str,
    hidden_size: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    check_numel(weights, proj_key, hidden_size, "hidden_size")?;
    check_numel(weights, norm_key, hidden_size, "hidden_size")?;
    let proj = weights
        .get(proj_key)
        .ok_or_else(|| format!("Missing attention-residual projection: {proj_key}"))?;
    let norm = weights
        .get(norm_key)
        .ok_or_else(|| format!("Missing attention-residual norm: {norm_key}"))?;
    let d: i32 = mlxcel_core::array_shape(proj).iter().product();
    let proj = mlxcel_core::astype(&mlxcel_core::reshape(proj, &[d]), dtype::FLOAT32);
    let norm = mlxcel_core::astype(norm, dtype::FLOAT32);
    let w = mlxcel_core::reshape(&mlxcel_core::multiply(&norm, &proj), &[d, 1]);
    mlxcel_core::eval(&w);
    Ok(w)
}

/// The activation dtype the decoder runs in, read from the embedding table
/// (its `.scales` when the table is quantized). Float32 side tensors that
/// enter half-precision ops (the conv weight) are cast to it at load so the
/// residual stream keeps one dtype (see `docs/code-guidelines.md`, "One
/// Checkpoint, One Dtype").
fn activation_dtype(weights: &WeightMap) -> i32 {
    let table = weights
        .get("model.embed_tokens.scales")
        .or_else(|| weights.get("model.embed_tokens.weight"));
    match table.map(|w| mlxcel_core::array_dtype(w)) {
        Some(d) if d == dtype::FLOAT16 || d == dtype::BFLOAT16 || d == dtype::FLOAT32 => d,
        _ => dtype::BFLOAT16,
    }
}

fn cast_to(x: &MlxArray, target: i32) -> UniquePtr<MlxArray> {
    if mlxcel_core::array_dtype(x) == target {
        mlxcel_core::copy(x)
    } else {
        mlxcel_core::astype(x, target)
    }
}

fn take_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Missing weight: {key}"))
}

/// Cross-check one checkpoint tensor axis against the config value that
/// indexes it.
///
/// Nothing else in the load path compares the two: the projections take their
/// shapes from the checkpoint while the forward pass slices and reshapes with
/// the config's head counts, ranks and expert count. When a config is edited
/// (a layer-truncated debug copy, a hand-written `text_config`) or paired with
/// the wrong checkpoint, the disagreement surfaces either as an MLX abort in
/// the middle of a forward pass, with no key to name, or -- when the
/// mismatched axis still broadcasts -- as fluent nonsense. Checking at load
/// turns both into a message naming the tensor and the field.
///
/// A missing key is not an error here; the loader that needs it reports it.
/// Quantized planes keep out-features on axis 0, so axis-0 checks hold for
/// packed weights as well.
fn check_axis(
    weights: &WeightMap,
    key: &str,
    axis: usize,
    expected: usize,
    field: &str,
) -> Result<(), String> {
    let Some(w) = weights.get(key) else {
        return Ok(());
    };
    let shape = mlxcel_core::array_shape(w);
    let Some(&got) = shape.get(axis) else {
        return Err(format!(
            "{key}: shape {shape:?} has no axis {axis} to match the config's {field} ({expected})"
        ));
    };
    if got < 0 || got as usize != expected {
        return Err(format!(
            "{key}: axis {axis} is {got}, but the config's {field} is {expected} (shape {shape:?})"
        ));
    }
    Ok(())
}

/// [`check_axis`] for a tensor the config sizes as a flat vector (a norm
/// weight, a bias), whose rank the checkpoint may write as `[N]` or `[1, N]`.
fn check_numel(weights: &WeightMap, key: &str, expected: usize, field: &str) -> Result<(), String> {
    let Some(w) = weights.get(key) else {
        return Ok(());
    };
    let shape = mlxcel_core::array_shape(w);
    let got: i64 = shape.iter().map(|&d| d as i64).product();
    if got != expected as i64 {
        return Err(format!(
            "{key}: {got} entries, but the config's {field} is {expected} (shape {shape:?})"
        ));
    }
    Ok(())
}

// KDA: KimiK3DeltaAttention.

enum DeltaGate {
    FullRank(UnifiedLinear),
    LowRank {
        g_a_proj: UnifiedLinear,
        g_b_proj: UnifiedLinear,
    },
}

/// Kimi Delta Attention with the fused `qkv_proj` / `qkv_conv`, the
/// lower-bounded gate and the (full-rank) sigmoid output gate.
///
/// Forward (`P = H * D`, `K = short_conv_kernel_size`):
///
/// ```text
/// qkv = qkv_proj(x)                                  [B, T, 3P]
/// qkv, conv = short_conv(qkv, conv)                  depthwise causal conv over 3P, kernel K, silu
/// q, k, v = split(qkv, [P, 2P])  -> [B, T, H, D]
/// q = scale^2 * rms_norm(q, eps = 1e-6 / D);  k = scale * rms_norm(k, same eps)
/// a = f_b_proj(f_a_proj(x)) -> [B, T, H, D];  b = b_proj(x) -> [B, T, H]
/// beta = sigmoid(b);  g = exp(lower_bound * sigmoid(exp(A_log[h]) * (a + dt_bias[h, :])))
/// out, ssm = gated_delta(q, k, v, g, beta, ssm)
/// gate = g_proj(x) -> [B, T, H, D]   (or g_b_proj(g_a_proj(x)))
/// y = o_proj((rms_norm(out; o_norm) * sigmoid(gate)).reshape(B, T, P))
/// ```
pub(crate) struct KimiK3DeltaAttention {
    qkv_proj: UnifiedLinear,
    qkv_conv: ShortConv1d,
    /// `[1, K, 3P]` time-major copy of the conv weight for the `T == 1` step.
    conv_decode_weight: UniquePtr<MlxArray>,
    f_a_proj: UnifiedLinear,
    f_b_proj: UnifiedLinear,
    b_proj: UnifiedLinear,
    gate: DeltaGate,
    /// `[H, 1]` float32.
    a_log: UniquePtr<MlxArray>,
    /// `[H, D]` float32.
    dt_bias: UniquePtr<MlxArray>,
    o_norm: RMSNorm,
    o_proj: UnifiedLinear,

    num_heads: i32,
    head_dim: i32,
    projection_dim: i32,
    kernel_size: i32,
    scale: f32,
    qk_eps: f32,
    gate_lower_bound: Option<f32>,
}

impl KimiK3DeltaAttention {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KimiK3DeltaCache,
    ) -> UniquePtr<MlxArray> {
        let (q, k, v, conv_state) = self.qkv_stage(x, cache.conv.as_deref());
        cache.conv = Some(conv_state);
        self.finish(x, &q, &k, &v, cache)
    }

    /// Fused projection + short conv, split into per-head q / k / v.
    ///
    /// Returns `(q, k, v, new_conv_state)` with q / k / v as `[B, T, H, D]`.
    pub(crate) fn qkv_stage(
        &self,
        x: &MlxArray,
        conv_state: Option<&MlxArray>,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let t = shape[1];
        let p = self.projection_dim;

        let qkv = self.qkv_proj.forward(x);
        let (qkv, new_state) = if t == 1 {
            self.conv_decode_step(&qkv, conv_state, b)
        } else {
            self.qkv_conv.forward(&qkv, conv_state, None)
        };

        let q = mlxcel_core::slice(&qkv, &[0, 0, 0], &[b, t, p]);
        let k = mlxcel_core::slice(&qkv, &[0, 0, p], &[b, t, 2 * p]);
        let v = mlxcel_core::slice(&qkv, &[0, 0, 2 * p], &[b, t, 3 * p]);
        let per_head = [b, t, self.num_heads, self.head_dim];
        (
            mlxcel_core::reshape(&q, &per_head),
            mlxcel_core::reshape(&k, &per_head),
            mlxcel_core::reshape(&v, &per_head),
            new_state,
        )
    }

    /// Single-token step of the fused short conv as a broadcast weighted sum
    /// (`src/models/conv_decode.rs`) instead of a length-1 `conv1d`.
    fn conv_decode_step(
        &self,
        qkv: &MlxArray,
        conv_state: Option<&MlxArray>,
        b: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let in_dtype = mlxcel_core::array_dtype(qkv);
        let channels = 3 * self.projection_dim;
        let state = match conv_state {
            Some(s) => mlxcel_core::copy(s),
            None => mlxcel_core::zeros(&[b, self.kernel_size - 1, channels], in_dtype),
        };
        // [B, K, 3P]: the previous K-1 inputs and the new one.
        let padded = mlxcel_core::concatenate(&state, qkv, 1);
        let out = silu(&short_conv_decode_step(
            &padded,
            &self.conv_decode_weight,
            in_dtype,
        ));
        let tail = mlxcel_core::slice(&padded, &[0, 1, 0], &[b, self.kernel_size, channels]);
        (out, mlxcel_core::contiguous(&tail, false))
    }

    /// Everything after the conv: q/k norm, gate, recurrence, output gate,
    /// `o_proj`. Advances `cache` by the token count.
    pub(crate) fn finish(
        &self,
        x: &MlxArray,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        cache: &mut KimiK3DeltaCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let t = shape[1];
        let x_dtype = mlxcel_core::array_dtype(x);

        // scale * l2norm(q) and l2norm(k), see `KimiK3TextConfig::qk_norm_eps`.
        let q = scaled_fast_rms_norm_no_weight(q, self.scale * self.scale, self.qk_eps);
        let k = scaled_fast_rms_norm_no_weight(k, self.scale, self.qk_eps);

        let a = self.f_b_proj.forward(&self.f_a_proj.forward(x));
        let a = mlxcel_core::reshape(&a, &[b, t, self.num_heads, self.head_dim]);
        let b_logits = mlxcel_core::reshape(&self.b_proj.forward(x), &[b, t, self.num_heads]);

        let (out, new_ssm) = gated_delta_update_with_lower_bound(
            &q,
            &k,
            v,
            &a,
            &b_logits,
            &self.a_log,
            &self.dt_bias,
            cache.ssm.as_deref(),
            None,
            self.gate_lower_bound,
        );
        cache.ssm = Some(new_ssm);
        cache.advance(t);

        // o_norm(out) * sigmoid(gate): `FusedRMSNormGated(head_dim, eps,
        // activation='sigmoid')` in the reference. The product runs in float32
        // (the float32 `o_norm.weight` already promotes the norm output) and
        // is cast back to the activation dtype once.
        let gate = match &self.gate {
            DeltaGate::FullRank(g_proj) => g_proj.forward(x),
            DeltaGate::LowRank { g_a_proj, g_b_proj } => g_b_proj.forward(&g_a_proj.forward(x)),
        };
        let gate = mlxcel_core::reshape(&gate, &[b, t, self.num_heads, self.head_dim]);
        let out = mlxcel_core::reshape(&out, &[b, t, self.num_heads, self.head_dim]);
        let normed = mlxcel_core::astype(&self.o_norm.forward(&out), dtype::FLOAT32);
        let gate_sig = mlxcel_core::sigmoid(&mlxcel_core::astype(&gate, dtype::FLOAT32));
        let gated = mlxcel_core::multiply(&normed, &gate_sig);
        let gated = cast_to(&gated, x_dtype);
        let gated = mlxcel_core::reshape(&gated, &[b, t, self.projection_dim]);

        self.o_proj.forward(&gated)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &KimiK3TextConfig,
        prefix: &str,
        act_dtype: i32,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        let num_heads = config.delta_num_heads();
        let head_dim = config.delta_head_dim();
        let projection_dim = config.delta_projection_dim();
        let kernel_size = config.linear_attn_config.short_conv_kernel_size;

        let qkv_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.qkv_proj"), gs, bits)?;

        check_axis(
            weights,
            &format!("{prefix}.qkv_proj.weight"),
            0,
            3 * projection_dim,
            "linear_attn_config num_heads * head_dim * 3",
        )?;
        check_axis(
            weights,
            &format!("{prefix}.f_b_proj.weight"),
            0,
            projection_dim,
            "linear_attn_config num_heads * head_dim",
        )?;
        check_axis(
            weights,
            &format!("{prefix}.b_proj.weight"),
            0,
            num_heads,
            "linear_attn_config.num_heads",
        )?;
        check_numel(
            weights,
            &format!("{prefix}.o_norm.weight"),
            head_dim,
            "linear_attn_config.head_dim",
        )?;

        // `[3P, K, 1]` after sanitize. The checkpoint stores it float32; cast to
        // the activation dtype so `conv1d` does not promote the q/k/v stream.
        let conv_key = format!("{prefix}.qkv_conv.conv.weight");
        let conv_weight = take_weight(weights, &conv_key)?;
        let conv_shape = mlxcel_core::array_shape(&conv_weight);
        let expected = [3 * projection_dim as i32, kernel_size as i32, 1];
        if conv_shape != expected {
            return Err(format!(
                "{conv_key}: expected shape {expected:?} (fused q/k/v channels, kernel, 1), got \
                 {conv_shape:?}"
            ));
        }
        let conv_weight = cast_to(&conv_weight, act_dtype);
        let conv_decode_weight = build_conv_decode_weight(&conv_weight);
        let qkv_conv = ShortConv1d::new(conv_weight, kernel_size, 3 * projection_dim);

        let f_a_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.f_a_proj"), gs, bits)?;
        let f_b_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.f_b_proj"), gs, bits)?;
        let b_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.b_proj"), gs, bits)?;

        let gate = if config.linear_attn_config.use_full_rank_gate {
            DeltaGate::FullRank(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.g_proj"),
                gs,
                bits,
            )?)
        } else {
            DeltaGate::LowRank {
                g_a_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{prefix}.g_a_proj"),
                    gs,
                    bits,
                )?,
                g_b_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{prefix}.g_b_proj"),
                    gs,
                    bits,
                )?,
            }
        };

        let a_log_key = format!("{prefix}.A_log");
        let a_log = take_weight(weights, &a_log_key)?;
        let a_log_len: i32 = mlxcel_core::array_shape(&a_log).iter().product();
        if a_log_len < num_heads as i32 {
            return Err(format!(
                "{a_log_key}: {a_log_len} entries for {num_heads} heads"
            ));
        }
        let a_log = mlxcel_core::reshape(&a_log, &[a_log_len]);
        let a_log = mlxcel_core::slice(&a_log, &[0], &[num_heads as i32]);
        let a_log = mlxcel_core::astype(&a_log, dtype::FLOAT32);
        let a_log = mlxcel_core::reshape(&a_log, &[num_heads as i32, 1]);

        let dt_key = format!("{prefix}.dt_bias");
        let dt_bias = take_weight(weights, &dt_key)?;
        let dt_len: i32 = mlxcel_core::array_shape(&dt_bias).iter().product();
        if dt_len != projection_dim as i32 {
            return Err(format!(
                "{dt_key}: {dt_len} entries, expected num_heads * head_dim = {projection_dim}"
            ));
        }
        let dt_bias = mlxcel_core::astype(&dt_bias, dtype::FLOAT32);
        let dt_bias = mlxcel_core::reshape(&dt_bias, &[num_heads as i32, head_dim as i32]);

        let o_norm_weight = take_weight(weights, &format!("{prefix}.o_norm.weight"))?;
        let o_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), gs, bits)?;

        Ok(Self {
            qkv_proj,
            qkv_conv,
            conv_decode_weight,
            f_a_proj,
            f_b_proj,
            b_proj,
            gate,
            a_log,
            dt_bias,
            o_norm: RMSNorm::new(o_norm_weight, config.rms_norm_eps),
            o_proj,
            num_heads: num_heads as i32,
            head_dim: head_dim as i32,
            projection_dim: projection_dim as i32,
            kernel_size: kernel_size as i32,
            scale: (head_dim as f32).powf(-0.5),
            qk_eps: config.qk_norm_eps(),
            gate_lower_bound: config.linear_attn_config.gate_lower_bound,
        })
    }
}

// MLA: KimiK3MLAAttention.

enum QueryProjection {
    Plain(UnifiedLinear),
    LoRA {
        q_a_proj: UnifiedLinear,
        q_a_layernorm: RMSNorm,
        q_b_proj: UnifiedLinear,
    },
}

/// NoPE Multi-head Latent Attention with q-LoRA and a sigmoid output gate.
///
/// Identical to Kimi Linear's absorbed MLA (`embed_q` / `unembed_out` derived
/// from `kv_b_proj` at sanitize time, the `(kv_latent, k_pe)` pair in the
/// `KVCache`) plus `q = q_b_proj(rms_norm(q_a_proj(x)))` and
/// `o = o * sigmoid(g_proj(x))` before `o_proj`. No rotation touches `q_pe`
/// or `k_pe` (`mla_use_nope`).
pub(crate) struct KimiK3MLAAttention {
    q_proj: QueryProjection,
    kv_a_proj_with_mqa: UnifiedLinear,
    kv_a_layernorm: RMSNorm,
    embed_q: MultiLinear,
    unembed_out: MultiLinear,
    g_proj: Option<UnifiedLinear>,
    o_proj: UnifiedLinear,

    num_heads: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    q_head_dim: i32,
    kv_lora_rank: i32,
    scale: f32,
}

impl KimiK3MLAAttention {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = match &self.q_proj {
            QueryProjection::Plain(q_proj) => q_proj.forward(x),
            QueryProjection::LoRA {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
            } => q_b_proj.forward(&q_a_layernorm.forward(&q_a_proj.forward(x))),
        };
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]); // [B, H, L, q_head_dim]
        let q_nope = mlxcel_core::slice(
            &q,
            &[0, 0, 0, 0],
            &[b, self.num_heads, l, self.qk_nope_head_dim],
        );
        let q_pe = mlxcel_core::slice(
            &q,
            &[0, 0, 0, self.qk_nope_head_dim],
            &[b, self.num_heads, l, self.q_head_dim],
        );

        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);
        let compressed = mlxcel_core::slice(&compressed_kv, &[0, 0, 0], &[b, l, self.kv_lora_rank]);
        let k_pe = mlxcel_core::slice(
            &compressed_kv,
            &[0, 0, self.kv_lora_rank],
            &[b, l, self.kv_lora_rank + self.qk_rope_head_dim],
        );
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]); // [B, 1, L, rope]
        let kv_latent = mlxcel_core::expand_dims(&self.kv_a_layernorm.forward(&compressed), 1); // [B, 1, L, rank]

        let (cached_latent, cached_k_pe) = cache.update_and_fetch(kv_latent, k_pe);

        // pe_scores = (q_pe * scale) @ k_pe^T -> [B, H, L, S]; the mask is
        // additive (0 / -inf) and is added in the scores dtype.
        let scale_arr = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(&q_pe));
        let q_pe_scaled = mlxcel_core::multiply(&q_pe, &scale_arr);
        let k_pe_t = mlxcel_core::swap_axes(&cached_k_pe, -1, -2);
        let mut pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);
        if let Some(m) = mask {
            let m = cast_to(m, mlxcel_core::array_dtype(&pe_scores));
            pe_scores = mlxcel_core::add(&pe_scores, &m);
        }

        let output = if l == 1 {
            // Decode in the latent space: q_lat = embed_q(q_nope) [B, H, 1, rank].
            let q_lat = self.embed_q.forward(&q_nope, true);
            let q_lat = mlxcel_core::multiply(&q_lat, &scale_arr);
            let k_t = mlxcel_core::swap_axes(&cached_latent, -1, -2);
            let scores = mlxcel_core::add(&mlxcel_core::matmul(&q_lat, &k_t), &pe_scores);
            let probs = mlxcel_core::softmax(&scores, -1);
            let attn = mlxcel_core::matmul(&probs, &cached_latent);
            self.unembed_out.forward(&attn, true) // [B, H, 1, v_head]
        } else {
            // Prefill: expand the latent to per-head k / v.
            let k = self.embed_q.forward(&cached_latent, false); // [B, H, S, nope]
            let v = self.unembed_out.forward(&cached_latent, true); // [B, H, S, v_head]
            let q_nope_scaled = mlxcel_core::multiply(&q_nope, &scale_arr);
            let k_t = mlxcel_core::swap_axes(&k, -1, -2);
            let scores = mlxcel_core::add(&mlxcel_core::matmul(&q_nope_scaled, &k_t), &pe_scores);
            let probs = mlxcel_core::softmax(&scores, -1);
            mlxcel_core::matmul(&probs, &v) // [B, H, L, v_head]
        };

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        let output = match &self.g_proj {
            Some(g_proj) => {
                let gate = mlxcel_core::sigmoid(&g_proj.forward(x));
                mlxcel_core::multiply(&output, &gate)
            }
            None => output,
        };
        self.o_proj.forward(&output)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &KimiK3TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        check_axis(
            weights,
            &format!("{prefix}.kv_a_proj_with_mqa.weight"),
            0,
            config.kv_lora_rank + config.qk_rope_head_dim,
            "kv_lora_rank + qk_rope_head_dim",
        )?;
        check_numel(
            weights,
            &format!("{prefix}.kv_a_layernorm.weight"),
            config.kv_lora_rank,
            "kv_lora_rank",
        )?;
        let q_out_key = if config.q_lora_rank.is_some() {
            format!("{prefix}.q_b_proj.weight")
        } else {
            format!("{prefix}.q_proj.weight")
        };
        check_axis(
            weights,
            &q_out_key,
            0,
            config.num_attention_heads * config.q_head_dim(),
            "num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)",
        )?;
        if let Some(rank) = config.q_lora_rank {
            check_axis(
                weights,
                &format!("{prefix}.q_a_proj.weight"),
                0,
                rank,
                "q_lora_rank",
            )?;
            check_numel(
                weights,
                &format!("{prefix}.q_a_layernorm.weight"),
                rank,
                "q_lora_rank",
            )?;
        }

        let q_proj = if config.q_lora_rank.is_some() {
            let q_a_norm = take_weight(weights, &format!("{prefix}.q_a_layernorm.weight"))?;
            QueryProjection::LoRA {
                q_a_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{prefix}.q_a_proj"),
                    gs,
                    bits,
                )?,
                q_a_layernorm: RMSNorm::new(q_a_norm, MLA_LORA_NORM_EPS),
                q_b_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{prefix}.q_b_proj"),
                    gs,
                    bits,
                )?,
            }
        } else {
            QueryProjection::Plain(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_proj"),
                gs,
                bits,
            )?)
        };
        let kv_a_proj_with_mqa = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.kv_a_proj_with_mqa"),
            gs,
            bits,
        )?;
        let kv_norm_weight = take_weight(weights, &format!("{prefix}.kv_a_layernorm.weight"))?;
        let embed_q = MultiLinear::from_weights(weights, &format!("{prefix}.embed_q"), gs, bits)?;
        let unembed_out =
            MultiLinear::from_weights(weights, &format!("{prefix}.unembed_out"), gs, bits)?;
        let g_proj = if config.mla_use_output_gate {
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.g_proj"),
                gs,
                bits,
            )?)
        } else {
            None
        };
        let o_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), gs, bits)?;

        Ok(Self {
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm: RMSNorm::new(kv_norm_weight, MLA_LORA_NORM_EPS),
            embed_q,
            unembed_out,
            g_proj,
            o_proj,
            num_heads: config.num_attention_heads as i32,
            qk_nope_head_dim: config.qk_nope_head_dim as i32,
            qk_rope_head_dim: config.qk_rope_head_dim as i32,
            q_head_dim: config.q_head_dim() as i32,
            kv_lora_rank: config.kv_lora_rank as i32,
            scale: (config.q_head_dim() as f32).powf(-0.5),
        })
    }
}

// MLP and MoE.

/// Dense SiTU MLP: `down_proj(situ(up_proj(h), gate_proj(h)))`.
pub(crate) struct KimiK3MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    beta: f32,
    linear_beta: Option<f32>,
}

impl KimiK3MLP {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = situ_activation(&gate, &up, self.beta, self.linear_beta);
        self.down_proj.forward(&activated)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &KimiK3TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
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
            beta: config.situ_beta(),
            linear_beta: config.activation_situ_linear_beta,
        })
    }
}

/// Quantization triple of the stacked expert planes under `{prefix}`.
///
/// The published checkpoint ships compressed-tensors `mxfp4-pack-quantized`
/// experts, which sanitize turns into MLX's mxfp4 layout: uint32 packed codes
/// with uint8 E8M0 block scales and no biases. A uint8 scales plane is that
/// layout's signature, and MLX's mxfp4 is fixed at `(group_size 32, bits 4)`.
/// Any other scales dtype is an MLX conversion and follows the config's pair
/// with the mode inferred from the biases plane; no scales means dense
/// experts, where the mode is inert.
fn expert_quantization(
    weights: &WeightMap,
    prefix: &str,
    config: &KimiK3TextConfig,
) -> (i32, i32, &'static str) {
    let gs = config.group_size();
    let bits = config.bits();
    match weights.get(&format!("{prefix}.gate_proj.scales")) {
        None => (gs, bits, "affine"),
        Some(scales) if mlxcel_core::array_dtype(scales) == dtype::UINT8 => (32, 4, "mxfp4"),
        Some(_) => {
            let has_biases = weights.contains_key(&format!("{prefix}.gate_proj.biases"));
            (
                gs,
                bits,
                mlxcel_core::layers::infer_quantization_mode(has_biases, gs, bits),
            )
        }
    }
}

/// Latent sparse MoE with sigmoid routing:
///
/// ```text
/// scores = sigmoid(gate(h).f32);  sel = scores + e_score_correction_bias
/// idx = top-k(sel);  w = take(scores, idx);  w /= sum(w) + 1e-20;  w *= routed_scaling_factor
/// z = routed_expert_down_proj(h)                       [N, latent]   (identity without the latent)
/// e = sum_k w_k * down_k(situ(up_k(z), gate_k(z)))
/// e = rms_norm(e; routed_expert_norm)  (latent_moe_use_norm)
/// out = routed_expert_up_proj(e) + shared_experts(h)
/// ```
pub(crate) struct KimiK3SparseMoE {
    gate: UnifiedLinear,
    e_score_correction_bias: Option<UniquePtr<MlxArray>>,
    routed_expert_down_proj: Option<UnifiedLinear>,
    routed_expert_norm: Option<RMSNorm>,
    routed_expert_up_proj: Option<UnifiedLinear>,
    switch_mlp: SwitchGLU,
    shared_experts: Option<KimiK3MLP>,
    num_experts_per_token: i32,
    routed_scaling_factor: f32,
    renormalize: bool,
}

impl KimiK3SparseMoE {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = orig_shape[orig_shape.len() - 1];
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden])
        } else {
            mlxcel_core::copy(x)
        };
        let x_dtype = mlxcel_core::array_dtype(&x_flat);

        // Router: matmul in the activation dtype, sigmoid and selection in f32.
        let logits = self.gate.forward(&x_flat);
        let scores = mlxcel_core::sigmoid(&mlxcel_core::astype(&logits, dtype::FLOAT32));
        let selection = match &self.e_score_correction_bias {
            Some(bias) => mlxcel_core::add(&scores, bias),
            None => mlxcel_core::copy(&scores),
        };
        let k = self.num_experts_per_token;
        let n_tokens = mlxcel_core::array_shape(&selection)[0];
        let indices = mlxcel_core::argpartition(&mlxcel_core::negative(&selection), k - 1, -1);
        let topk_indices = mlxcel_core::slice(&indices, &[0, 0], &[n_tokens, k]);
        let mut topk_scores = mlxcel_core::take_along_axis(&scores, &topk_indices, -1);
        if k > 1 && self.renormalize {
            let eps = mlxcel_core::full_f32(&[1], 1e-20, dtype::FLOAT32);
            let sum = mlxcel_core::add(&mlxcel_core::sum_axis(&topk_scores, -1, true), &eps);
            topk_scores = mlxcel_core::divide(&topk_scores, &sum);
        }
        if self.routed_scaling_factor != 1.0 {
            topk_scores = mlxcel_core::multiply_scalar(&topk_scores, self.routed_scaling_factor);
        }

        // Routed experts on the latent (or the hidden state itself).
        let z = match &self.routed_expert_down_proj {
            Some(down) => down.forward(&x_flat),
            None => mlxcel_core::copy(&x_flat),
        };
        let expert_out = self.switch_mlp.forward(&z, &topk_indices);
        let mut y =
            crate::models::switch_layers::moe_weighted_sum(&expert_out, &topk_scores, x_dtype);
        if let Some(norm) = &self.routed_expert_norm {
            y = norm.forward(&y);
        }
        if let Some(up) = &self.routed_expert_up_proj {
            y = up.forward(&y);
        }
        if let Some(shared) = &self.shared_experts {
            y = mlxcel_core::add(&y, &shared.forward(&x_flat));
        }

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&y, &orig_shape)
        } else {
            y
        }
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &KimiK3TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        let gate = UnifiedLinear::from_weights(weights, &format!("{prefix}.gate"), gs, bits)?;
        let e_score_correction_bias = weights
            .get(&format!("{prefix}.e_score_correction_bias"))
            .map(|w| mlxcel_core::astype(w, dtype::FLOAT32));

        let switch_prefix = format!("{prefix}.switch_mlp");
        // Sanitize counts the per-expert planes it stacks; a checkpoint that
        // already ships stacked planes never passes through that count.
        for leaf in ["gate_proj", "up_proj", "down_proj"] {
            check_axis(
                weights,
                &format!("{switch_prefix}.{leaf}.weight"),
                0,
                config.num_experts,
                "num_experts",
            )?;
        }
        check_axis(
            weights,
            &format!("{prefix}.gate.weight"),
            0,
            config.num_experts,
            "num_experts",
        )?;
        check_numel(
            weights,
            &format!("{prefix}.e_score_correction_bias"),
            config.num_experts,
            "num_experts",
        )?;
        let (expert_gs, expert_bits, expert_mode) =
            expert_quantization(weights, &switch_prefix, config);
        let switch_mlp = SwitchGLU::from_weights_with_mode(
            weights,
            &switch_prefix,
            expert_gs,
            expert_bits,
            expert_mode,
        )?
        .with_activation(config.situ_activation());

        let (routed_expert_down_proj, routed_expert_norm, routed_expert_up_proj) =
            if config.routed_expert_hidden_size.is_some() {
                let latent = config
                    .routed_expert_hidden_size
                    .unwrap_or(config.hidden_size);
                check_axis(
                    weights,
                    &format!("{prefix}.routed_expert_down_proj.weight"),
                    0,
                    latent,
                    "routed_expert_hidden_size",
                )?;
                let norm = if config.latent_moe_use_norm {
                    check_numel(
                        weights,
                        &format!("{prefix}.routed_expert_norm.weight"),
                        latent,
                        "routed_expert_hidden_size",
                    )?;
                    let w = take_weight(weights, &format!("{prefix}.routed_expert_norm.weight"))?;
                    Some(RMSNorm::new(w, config.rms_norm_eps))
                } else {
                    None
                };
                (
                    Some(UnifiedLinear::from_weights(
                        weights,
                        &format!("{prefix}.routed_expert_down_proj"),
                        gs,
                        bits,
                    )?),
                    norm,
                    Some(UnifiedLinear::from_weights(
                        weights,
                        &format!("{prefix}.routed_expert_up_proj"),
                        gs,
                        bits,
                    )?),
                )
            } else {
                (None, None, None)
            };

        let shared_experts = if config.num_shared_experts > 0 {
            Some(KimiK3MLP::from_weights(
                weights,
                config,
                &format!("{prefix}.shared_experts"),
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            e_score_correction_bias,
            routed_expert_down_proj,
            routed_expert_norm,
            routed_expert_up_proj,
            switch_mlp,
            shared_experts,
            num_experts_per_token: config.num_experts_per_token as i32,
            routed_scaling_factor: config.routed_scaling_factor,
            renormalize: config.moe_renormalize,
        })
    }
}

enum MlpVariant {
    Dense(KimiK3MLP),
    MoE(KimiK3SparseMoE),
}

impl MlpVariant {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MlpVariant::Dense(mlp) => mlp.forward(x),
            MlpVariant::MoE(moe) => moe.forward(x),
        }
    }
}

enum AttentionVariant {
    Delta(KimiK3DeltaAttention),
    Mla(KimiK3MLAAttention),
}

/// The per-layer Attention Residuals weights.
struct LayerAttnRes {
    block_size: usize,
    /// `[D, 1]` float32 for the mix in front of the attention sublayer.
    w_attn: UniquePtr<MlxArray>,
    /// `[D, 1]` float32 for the mix in front of the MLP sublayer.
    w_mlp: UniquePtr<MlxArray>,
    eps: f32,
}

// Decoder layer.

pub(crate) struct KimiK3DecoderLayer {
    layer_idx: usize,
    is_linear: bool,
    self_attn: AttentionVariant,
    mlp: MlpVariant,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    attn_res: Option<LayerAttnRes>,
}

impl KimiK3DecoderLayer {
    fn attention(
        &self,
        normed: &MlxArray,
        attn_mask: Option<&MlxArray>,
        cache: &mut KimiK3LayerCache,
    ) -> UniquePtr<MlxArray> {
        match (&self.self_attn, cache) {
            (AttentionVariant::Delta(attn), KimiK3LayerCache::Delta(c)) => attn.forward(normed, c),
            (AttentionVariant::Mla(attn), KimiK3LayerCache::Attn(c)) => {
                attn.forward(normed, attn_mask, c)
            }
            _ => panic!("kimi_k3: cache type mismatch at layer {}", self.layer_idx),
        }
    }

    /// One decoder layer. With Attention Residuals (`attn_res` set):
    ///
    /// ```text
    /// partial = x
    /// h = mix(blocks, partial, w_attn)
    /// if l % B == 0: blocks.push(partial); partial = None    (layer 0 stores the embeddings)
    /// y = attention(input_layernorm(h))
    /// partial = y if partial is None else partial + y
    /// h = mix(blocks, partial, w_mlp)
    /// return partial + mlp(post_attention_layernorm(h))
    /// ```
    ///
    /// Without them the layer is the ordinary `x + attn(norm(x))`, `+ mlp(norm(.))`.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        attn_mask: Option<&MlxArray>,
        cache: &mut KimiK3LayerCache,
        blocks: &mut ResidualBlocks,
    ) -> UniquePtr<MlxArray> {
        let Some(res) = &self.attn_res else {
            let normed = self.input_layernorm.forward(x);
            let r = self.attention(&normed, attn_mask, cache);
            let h = mlxcel_core::add(x, &r);
            let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
            return mlxcel_core::add(&h, &mlp_out);
        };

        let h = attn_res_mix(blocks, x, &res.w_attn, res.eps);
        let starts_block = self.layer_idx.is_multiple_of(res.block_size);
        if starts_block {
            blocks.push(x, res.eps);
        }
        let y = self.attention(&self.input_layernorm.forward(&h), attn_mask, cache);
        let partial = if starts_block {
            y
        } else {
            mlxcel_core::add(x, &y)
        };
        let h = attn_res_mix(blocks, &partial, &res.w_mlp, res.eps);
        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&partial, &mlp_out)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &KimiK3TextConfig,
        layer_idx: usize,
        act_dtype: i32,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let is_linear = config.is_linear_layer(layer_idx);

        let self_attn = if is_linear {
            AttentionVariant::Delta(KimiK3DeltaAttention::from_weights(
                weights,
                config,
                &format!("{prefix}.self_attn"),
                act_dtype,
            )?)
        } else {
            AttentionVariant::Mla(KimiK3MLAAttention::from_weights(
                weights,
                config,
                &format!("{prefix}.self_attn"),
            )?)
        };

        let mlp = if config.is_moe_layer(layer_idx) {
            MlpVariant::MoE(KimiK3SparseMoE::from_weights(
                weights,
                config,
                &format!("{prefix}.mlp"),
            )?)
        } else {
            MlpVariant::Dense(KimiK3MLP::from_weights(
                weights,
                config,
                &format!("{prefix}.mlp"),
            )?)
        };

        let input_norm = take_weight(weights, &format!("{prefix}.input_layernorm.weight"))?;
        let post_norm = take_weight(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;

        let attn_res = match config.attn_res_block_size {
            Some(block_size) => Some(LayerAttnRes {
                block_size,
                w_attn: load_attn_res_weight(
                    weights,
                    &format!("{prefix}.self_attention_res_proj.weight"),
                    &format!("{prefix}.self_attention_res_norm.weight"),
                    config.hidden_size,
                )?,
                w_mlp: load_attn_res_weight(
                    weights,
                    &format!("{prefix}.mlp_res_proj.weight"),
                    &format!("{prefix}.mlp_res_norm.weight"),
                    config.hidden_size,
                )?,
                eps: config.rms_norm_eps,
            }),
            None => None,
        };

        Ok(Self {
            layer_idx,
            is_linear,
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(input_norm, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm, config.rms_norm_eps),
            attn_res,
        })
    }
}

// Model.

pub struct KimiK3Model {
    pub embed_tokens: UnifiedEmbedding,
    layers: Vec<KimiK3DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub tie_word_embeddings: bool,
    /// Model-level `output_attn_res_{proj,norm}` mix applied after the last
    /// layer, `[D, 1]` float32 with its epsilon.
    output_attn_res: Option<(UniquePtr<MlxArray>, f32)>,
    sequence_state: ModelOwnedSequenceState<KimiK3LayerCache>,
    eos_token_ids: Vec<i32>,
}

impl KimiK3Model {
    /// Run the decoder stack on `h` (`[B, T, D]`), returning the final hidden
    /// state before the output mix and the residual blocks of this call.
    pub(crate) fn run_layers(
        &self,
        h: UniquePtr<MlxArray>,
        attn_mask: Option<&MlxArray>,
        caches: &mut [KimiK3LayerCache],
    ) -> (UniquePtr<MlxArray>, ResidualBlocks) {
        let mut blocks = ResidualBlocks::new();
        let mut h = h;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, attn_mask, cache, &mut blocks);
        }
        (h, blocks)
    }

    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KimiK3LayerCache],
    ) -> UniquePtr<MlxArray> {
        assert_eq!(
            caches.len(),
            self.layers.len(),
            "kimi_k3: cache cardinality must match layer count"
        );

        let h = self.embed_tokens.forward(input_ids);
        let l = mlxcel_core::array_shape(&h)[1];

        // Every cache advances by the same token count, so any layer's offset
        // is the sequence position; the additive mask is [L, L + offset].
        let attn_mask = if l > 1 {
            let offset = caches.first().map_or(0, KimiK3LayerCache::offset);
            Some(create_causal_mask(l, offset))
        } else {
            None
        };

        let (h, blocks) = self.run_layers(h, attn_mask.as_deref(), caches);
        let h = match &self.output_attn_res {
            Some((w, eps)) => attn_res_mix(&blocks, &h, w, *eps),
            None => h,
        };
        let h = self.norm.forward(&h);

        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_layer_caches(&self) -> Vec<KimiK3LayerCache> {
        make_layer_caches(&self.layers)
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, KimiK3Config), String> {
        let model_dir = model_dir.as_ref();

        println!("[KimiK3] Loading config...");
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config = KimiK3Config::from_json_str(&config_str)?;
        let text = &config.text_config;

        let n_linear = (0..text.num_hidden_layers)
            .filter(|&i| text.is_linear_layer(i))
            .count();
        let n_moe = (0..text.num_hidden_layers)
            .filter(|&i| text.is_moe_layer(i))
            .count();
        println!(
            "[KimiK3] Config loaded: {} layers ({} MLA, {} KDA, {} MoE), attn_res_block_size {:?}",
            text.num_hidden_layers,
            text.num_hidden_layers - n_linear,
            n_linear,
            n_moe,
            text.attn_res_block_size
        );

        println!("[KimiK3] Loading weights...");
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = Self::sanitize_weights(weights, text)?;

        println!("[KimiK3] Building model...");
        let mut model = Self::from_weights(&weights, text)?;
        model.set_eos_token_ids(config.eos_token_ids());
        model.set_eos_token_ids(crate::loading::read_eos_token_ids(model_dir));

        println!("[KimiK3] Model loaded successfully");
        Ok((model, config))
    }

    /// Rewrite the checkpoint's keys into the layout [`Self::from_weights`]
    /// reads. Idempotent. See `kimi_k3_sanitize.rs`.
    pub fn sanitize_weights(
        weights: WeightMap,
        config: &KimiK3TextConfig,
    ) -> Result<WeightMap, String> {
        sanitize::sanitize_weights(weights, config)
    }

    pub fn from_weights(weights: &WeightMap, config: &KimiK3TextConfig) -> Result<Self, String> {
        config.validate()?;
        let gs = config.group_size();
        let bits = config.bits();

        check_axis(
            weights,
            "model.embed_tokens.weight",
            0,
            config.vocab_size,
            "vocab_size",
        )?;
        check_numel(
            weights,
            "model.norm.weight",
            config.hidden_size,
            "hidden_size",
        )?;
        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;
        let act_dtype = activation_dtype(weights);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(KimiK3DecoderLayer::from_weights(
                weights, config, i, act_dtype,
            )?);
        }

        let norm_weight = take_weight(weights, "model.norm.weight")?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?)
        };

        let output_attn_res = if config.use_attn_residuals() {
            Some((
                load_attn_res_weight(
                    weights,
                    "model.output_attn_res_proj.weight",
                    "model.output_attn_res_norm.weight",
                    config.hidden_size,
                )?,
                config.rms_norm_eps,
            ))
        } else {
            None
        };

        let internal_caches = make_layer_caches(&layers);
        let eos_token_ids = super::parse_optional_eos_token_ids(&config.eos_token_id);

        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(norm_weight, config.rms_norm_eps),
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            output_attn_res,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            eos_token_ids,
        })
    }

    pub(crate) fn set_eos_token_ids(&mut self, eos_token_ids: Vec<i32>) {
        if !eos_token_ids.is_empty() {
            self.eos_token_ids = eos_token_ids;
        }
    }

    fn forward_for_sequence(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(input_ids)[1];
        if seq_id.is_none() && seq_len > 1 {
            self.sequence_state
                .replace_internal(self.make_layer_caches());
        }

        if let Some(seq_id) = seq_id {
            self.sequence_state
                .with_existing_sequence_state(seq_id, |internal| self.forward(input_ids, internal))
                .unwrap_or_else(|err| panic!("KimiK3 {err}"))
        } else {
            self.sequence_state
                .with_sequence_state(None, |internal| self.forward(input_ids, internal))
        }
    }
}

impl LanguageModel for KimiK3Model {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt the KDA recurrent state.
    }

    fn supports_batching(&self) -> bool {
        false // Mixed model-owned caches (MLA latent + KDA state), like KimiLinear.
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state
            .replace_internal(self.make_layer_caches());
        // Dummy KV caches for trait compatibility; the real state is model-owned.
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, None)
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.layers.len())
    }

    fn reset_runtime_state(&self) {
        self.sequence_state
            .replace_internal(self.make_layer_caches());
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_layer_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, seq_id)
    }
}

#[cfg(test)]
#[path = "kimi_k3_tests.rs"]
mod tests;
