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

//! DeepSeek-V4 (`deepseek_v4`, DeepSeek-V4-Flash).
//!
//! Ported from the in-tree reference
//! `references/mlx-vlm/mlx_vlm/models/deepseek_v4/`. V4 is a genuinely new
//! architecture, NOT a V3 variant: an earlier attempt (PR #592) modelled it
//! as a thin wrapper over `DeepSeekV3Model` and could not load or run a real
//! V4 checkpoint. Five features distinguish it, and every one is load-bearing:
//!
//! * **HyperConnections replace plain residuals** ([`hyper`]): the state
//!   carried between blocks is rank-4 `[B, L, hc_mult, D]`, collapsed per
//!   sublayer through Sinkhorn-normalised learned gates and re-expanded.
//! * **Pooled-KV `Compressor`, not V3-style MLA** ([`compress`]): per layer,
//!   `compress_ratios[layer]` selects local (0), compressed (128), or
//!   sparse-compressed (4) attention over a single shared 512-wide KV head
//!   in a 128-token sliding window plus softmax-pooled global rows.
//! * **HiSA hierarchical sparse selection** ([`indexer`]): a per-layer
//!   `Indexer` with its own compressor scores pooled rows and returns the
//!   top-`index_topk` for a split-softmax sparse attention.
//! * **Hash-routed MoE with `sqrtsoftplus` gating** ([`moe`]): the first
//!   `num_hash_layers` layers route by a `tid2eid` token-id lookup, which is
//!   why `input_ids` threads through every block; every layer is MoE and
//!   both expert paths clamp their SwiGLU at `swiglu_limit`.
//! * **Grouped `MultiLinear` output projection**: `wo_a` is an
//!   `o_groups`-way grouped low-rank projection followed by `wo_b`.
//!
//! Heterogeneous per-layer state (rotating local KV plus one or two pooling
//! caches) does not fit the trait's homogeneous `&mut [KVCache]`, so the
//! model owns its state through [`ModelOwnedSequenceState`] the way
//! [`crate::models::gemma3`] and [`crate::models::afmoe`] do;
//! `make_caches` returns placeholder `KVCache`s for trait compatibility.
//!
//! Untrusted config: [`ModelArgs::validate`] rejects every scalar that could
//! size an allocation, divide, or violate an MLX C++ precondition (an MLX
//! exception crossing the cxx bridge is an uncatchable `std::terminate`),
//! and `validate_weight_coverage` in [`sanitize`] rejects any checkpoint
//! whose tensor set does not exactly match what the config describes, so a
//! misnamed tensor fails the load instead of silently zero-initialising.
//!
//! Out of scope, deliberately: the reference's fused HC Metal kernel (the
//! ops path is the correctness baseline), MTP drafting
//! (`num_nextn_predict_layers`; `mtp.*` tensors are dropped), tensor/pipeline
//! sharding, and VLM wrapping.

#[path = "deepseek_v4_rope.rs"]
mod rope;

#[path = "deepseek_v4_hyper.rs"]
mod hyper;

#[path = "deepseek_v4_compress.rs"]
mod compress;

#[path = "deepseek_v4_indexer.rs"]
mod indexer;

#[path = "deepseek_v4_moe.rs"]
mod moe;

#[path = "deepseek_v4_attention.rs"]
mod attention;

#[path = "deepseek_v4_sanitize.rs"]
mod sanitize;

#[cfg(test)]
#[path = "deepseek_v4_tests.rs"]
mod deepseek_v4_tests;

use std::collections::HashMap;
use std::path::Path;

use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::create_sliding_window_prefill_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::models::bailing_moe::TokenIdField;
use crate::models::model_owned::ModelOwnedSequenceState;

use attention::V4Attention;
use compress::PoolingCache;
use hyper::{HyperConnection, HyperHead, hc_expand};
use moe::DeepseekV4MoE;

/// The compress ratio that switches the compressor into overlap mode and the
/// layer into sparse-compressed attention.
pub(crate) const OVERLAP_COMPRESS_RATIO: i32 = 4;

// Configuration.

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingV4 {
    #[serde(alias = "type", alias = "rope_type")]
    pub scaling_type: Option<String>,
    pub factor: Option<f32>,
    pub original_max_position_embeddings: Option<u64>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
}

/// The `quantization` / `quantization_config` block: one top-level triple
/// plus per-module-path overrides (the real DeepSeek-V4-Flash-4bit ships its
/// routed experts as mxfp4 at group size 32 under an affine/64 top level).
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationV4 {
    pub group_size: i32,
    pub bits: i32,
    #[serde(default)]
    pub mode: Option<String>,
    /// Remaining keys: `module.path -> {group_size, bits, mode}`. Kept as raw
    /// values so a stray non-object key (e.g. `quant_method`) cannot fail the
    /// parse; only object entries are treated as overrides.
    #[serde(flatten)]
    pub overrides: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantOverride {
    pub group_size: i32,
    pub bits: i32,
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScoringFunc {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_q_lora_rank")]
    pub q_lora_rank: usize,
    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,
    #[serde(default = "default_o_groups")]
    pub o_groups: usize,
    #[serde(default = "default_o_lora_rank")]
    pub o_lora_rank: usize,

    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_n_routed_experts")]
    pub n_routed_experts: usize,
    #[serde(default = "default_n_shared_experts")]
    pub n_shared_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_num_hash_layers")]
    pub num_hash_layers: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,

    #[serde(default)]
    pub compress_ratios: Vec<i64>,
    #[serde(default = "default_compress_rope_theta")]
    pub compress_rope_theta: f32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,

    #[serde(default = "default_hc_mult")]
    pub hc_mult: usize,
    #[serde(default = "default_hc_sinkhorn_iters")]
    pub hc_sinkhorn_iters: usize,
    #[serde(default = "default_hc_eps")]
    pub hc_eps: f32,

    #[serde(default = "default_index_n_heads")]
    pub index_n_heads: usize,
    #[serde(default = "default_index_head_dim")]
    pub index_head_dim: usize,
    #[serde(default = "default_index_topk")]
    pub index_topk: usize,
    /// HiSA block size. NOT in the real checkpoint's config.json; the
    /// reference dataclass default (64) applies.
    #[serde(default = "default_index_block")]
    pub index_block: usize,
    /// HiSA kept-block count. Also absent from the real config; default 16.
    #[serde(default = "default_index_keep")]
    pub index_keep: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingV4>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    /// mlx-lm-style block. The real checkpoint ships BOTH `quantization`
    /// and `quantization_config` (identical), so these are two fields with a
    /// precedence accessor rather than a serde alias: an alias makes serde
    /// reject the pair as a duplicate field.
    #[serde(default)]
    pub quantization: Option<QuantizationV4>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationV4>,
}

fn default_model_type() -> String {
    "deepseek_v4".to_string()
}
fn default_num_key_value_heads() -> usize {
    1
}
fn default_head_dim() -> usize {
    512
}
fn default_q_lora_rank() -> usize {
    1024
}
fn default_qk_rope_head_dim() -> usize {
    64
}
fn default_o_groups() -> usize {
    8
}
fn default_o_lora_rank() -> usize {
    1024
}
fn default_moe_intermediate_size() -> usize {
    2048
}
fn default_n_routed_experts() -> usize {
    256
}
fn default_n_shared_experts() -> usize {
    1
}
fn default_num_experts_per_tok() -> usize {
    6
}
fn default_num_hash_layers() -> usize {
    3
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_scoring_func() -> String {
    "sqrtsoftplus".to_string()
}
fn default_routed_scaling_factor() -> f32 {
    1.5
}
fn default_swiglu_limit() -> f32 {
    10.0
}
fn default_compress_rope_theta() -> f32 {
    160000.0
}
fn default_sliding_window() -> usize {
    128
}
fn default_hc_mult() -> usize {
    4
}
fn default_hc_sinkhorn_iters() -> usize {
    20
}
fn default_hc_eps() -> f32 {
    1e-6
}
fn default_index_n_heads() -> usize {
    64
}
fn default_index_head_dim() -> usize {
    128
}
fn default_index_topk() -> usize {
    512
}
fn default_index_block() -> usize {
    64
}
fn default_index_keep() -> usize {
    16
}
fn default_max_position_embeddings() -> usize {
    1048576
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}

const DEFAULT_EOS_TOKEN_ID: i32 = 1;

impl ModelArgs {
    /// The effective quantization block: `quantization` wins over
    /// `quantization_config` when both are present (they are identical on
    /// the real checkpoint).
    pub fn quantization_block(&self) -> Option<&QuantizationV4> {
        self.quantization
            .as_ref()
            .or(self.quantization_config.as_ref())
    }

    pub fn group_size(&self) -> i32 {
        self.quantization_block()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization_block().map(|q| q.bits).unwrap_or(4)
    }

    /// The explicit per-module-path quantization override for `path`, if the
    /// config declares one (object-valued entries only).
    pub fn quantization_override(&self, path: &str) -> Option<QuantOverride> {
        let value = self.quantization_block()?.overrides.get(path)?;
        serde_json::from_value(value.clone()).ok()
    }

    /// `(group_size, bits, mode)` for one routed-expert projection: the
    /// per-path override when declared, otherwise the top-level pair with the
    /// mode inferred from the plane's bias presence at the call site.
    pub(crate) fn expert_quantization(&self, path: &str) -> (i32, i32, String) {
        if let Some(o) = self.quantization_override(path) {
            let mode = o.mode.unwrap_or_else(|| "affine".to_string());
            return (o.group_size, o.bits, mode);
        }
        let mode = self
            .quantization_block()
            .and_then(|q| q.mode.clone())
            .unwrap_or_else(|| "affine".to_string());
        (self.group_size(), self.bits(), mode)
    }

    pub(crate) fn scoring_func_parsed(&self) -> Result<ScoringFunc, String> {
        match self.scoring_func.as_str() {
            "softmax" => Ok(ScoringFunc::Softmax),
            "sigmoid" => Ok(ScoringFunc::Sigmoid),
            "sqrtsoftplus" => Ok(ScoringFunc::SqrtSoftplus),
            other => Err(format!(
                "Unsupported DeepSeek-V4 scoring function: {other} (supported: softmax, sigmoid, \
                 sqrtsoftplus)"
            )),
        }
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(TokenIdField::Single(id)) => vec![*id],
            Some(TokenIdField::Multiple(ids)) if !ids.is_empty() => ids.clone(),
            _ => vec![DEFAULT_EOS_TOKEN_ID],
        }
    }

    /// Fill the reference `__post_init__` default `compress_ratios` when the
    /// config omits them, truncate to `num_hidden_layers`, then validate
    /// everything. Returns the normalized args ready for construction.
    pub fn normalized(mut self) -> Result<Self, String> {
        if self.compress_ratios.is_empty() {
            let n = self.num_hidden_layers;
            let mut ratios = Vec::with_capacity(n);
            if n >= 1 {
                ratios.push(0);
            }
            for i in 0..n.saturating_sub(2) {
                ratios.push(if i % 2 == 1 { 4 } else { 128 });
            }
            if n >= 2 {
                ratios.push(0);
            }
            self.compress_ratios = ratios;
        }
        self.compress_ratios.truncate(self.num_hidden_layers);
        self.validate()?;
        Ok(self)
    }

    /// Reject a config that cannot describe a real DeepSeek-V4 before any
    /// field sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        // Bounded BOTH ways. Every scalar this helper checks is later cast
        // `as i32` on its way into MLX: `DeepseekV4Block::from_weights` and
        // `DeepSeekV4Model::build` here, `V4Attention::from_weights` and its
        // output-projection reshapes, `Indexer::from_weights`, and the
        // `wo_a` reshape in `sanitize_weights`. A `usize` above `i32::MAX`
        // does not fail that cast: it truncates silently to zero or to a
        // negative dimension, which then reaches `reshape` / `broadcast_to` /
        // `zeros` / `from_slice_*` as an MLX precondition violation. An MLX
        // throw crossing the cxx bridge is an uncatchable `std::terminate`,
        // not a catchable Rust error, so the ceiling has to hold here at
        // config validation rather than at the dozens of cast sites
        // (docs/adding-models.md, Quantization Parameter Bounds, makes the
        // same argument for the quantization pair).
        fn positive(name: &str, v: usize) -> Result<(), String> {
            if v == 0 {
                return Err(format!("DeepSeek-V4 config: {name} must be > 0"));
            }
            if v > i32::MAX as usize {
                return Err(format!(
                    "DeepSeek-V4 config: {name} ({v}) exceeds i32::MAX; every architecture \
                     scalar is cast to i32 before it reaches an MLX kernel"
                ));
            }
            Ok(())
        }
        positive("vocab_size", self.vocab_size)?;
        positive("hidden_size", self.hidden_size)?;
        positive("num_hidden_layers", self.num_hidden_layers)?;
        positive("num_attention_heads", self.num_attention_heads)?;
        positive("head_dim", self.head_dim)?;
        positive("q_lora_rank", self.q_lora_rank)?;
        positive("o_lora_rank", self.o_lora_rank)?;
        positive("o_groups", self.o_groups)?;
        positive("moe_intermediate_size", self.moe_intermediate_size)?;
        positive("n_routed_experts", self.n_routed_experts)?;
        positive("n_shared_experts", self.n_shared_experts)?;
        positive("sliding_window", self.sliding_window)?;
        positive("index_n_heads", self.index_n_heads)?;
        positive("index_head_dim", self.index_head_dim)?;
        positive("index_topk", self.index_topk)?;
        // `index_block` / `index_keep` are config-readable and reach
        // `Indexer::from_weights` as `as i32` casts like the rest; they gate
        // the HiSA hierarchical path and divide `Np` inside it.
        positive("index_block", self.index_block)?;
        positive("index_keep", self.index_keep)?;
        positive("max_position_embeddings", self.max_position_embeddings)?;

        if self.num_key_value_heads != 1 {
            return Err(format!(
                "DeepSeek-V4 uses exactly one shared KV head; config declares \
                 num_key_value_heads = {}",
                self.num_key_value_heads
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.o_groups) {
            return Err(format!(
                "DeepSeek-V4 config: num_attention_heads ({}) must be divisible by o_groups ({})",
                self.num_attention_heads, self.o_groups
            ));
        }
        for (name, head_dim) in [
            ("head_dim", self.head_dim),
            ("index_head_dim", self.index_head_dim),
        ] {
            if head_dim < self.qk_rope_head_dim || !head_dim.is_multiple_of(2) {
                return Err(format!(
                    "DeepSeek-V4 config: {name} ({head_dim}) must be even and >= \
                     qk_rope_head_dim ({})",
                    self.qk_rope_head_dim
                ));
            }
        }
        if self.qk_rope_head_dim == 0 || !self.qk_rope_head_dim.is_multiple_of(2) {
            return Err(format!(
                "DeepSeek-V4 config: qk_rope_head_dim ({}) must be a positive even number",
                self.qk_rope_head_dim
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.n_routed_experts {
            return Err(format!(
                "DeepSeek-V4 config: num_experts_per_tok ({}) must be in 1..={}",
                self.num_experts_per_tok, self.n_routed_experts
            ));
        }
        if self.num_hash_layers > self.num_hidden_layers {
            return Err(format!(
                "DeepSeek-V4 config: num_hash_layers ({}) exceeds num_hidden_layers ({})",
                self.num_hash_layers, self.num_hidden_layers
            ));
        }
        if self.hc_mult == 0 || self.hc_mult > 64 {
            return Err(format!(
                "DeepSeek-V4 config: hc_mult ({}) must be in 1..=64",
                self.hc_mult
            ));
        }
        // `deepseek_v4_hyper.rs` builds the `fn` plane's expected shape as
        // `hc_mult * hidden_size` in i32. Both factors are individually
        // bounded now, but their PRODUCT still has to fit: an i32 overflow
        // wraps in release and panics in debug, and a wrapped width turns the
        // shape check into a comparison against a nonsense number.
        if self.hidden_size.saturating_mul(self.hc_mult) > i32::MAX as usize {
            return Err(format!(
                "DeepSeek-V4 config: hidden_size ({}) * hc_mult ({}) exceeds i32::MAX",
                self.hidden_size, self.hc_mult
            ));
        }
        if self.hc_sinkhorn_iters > 10_000 {
            return Err(format!(
                "DeepSeek-V4 config: hc_sinkhorn_iters ({}) is not a plausible iteration count",
                self.hc_sinkhorn_iters
            ));
        }
        for (name, eps) in [("hc_eps", self.hc_eps), ("rms_norm_eps", self.rms_norm_eps)] {
            if !(eps.is_finite() && eps > 0.0) {
                return Err(format!(
                    "DeepSeek-V4 config: {name} must be a positive finite number, got {eps}"
                ));
            }
        }
        for (name, theta) in [
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
        ] {
            if !(theta.is_finite() && theta > 1.0) {
                return Err(format!(
                    "DeepSeek-V4 config: {name} must be > 1, got {theta}"
                ));
            }
        }
        if !(self.routed_scaling_factor.is_finite() && self.routed_scaling_factor > 0.0) {
            return Err(format!(
                "DeepSeek-V4 config: routed_scaling_factor must be positive and finite, got {}",
                self.routed_scaling_factor
            ));
        }
        if !self.swiglu_limit.is_finite() || self.swiglu_limit < 0.0 {
            return Err(format!(
                "DeepSeek-V4 config: swiglu_limit must be finite and >= 0, got {}",
                self.swiglu_limit
            ));
        }
        self.scoring_func_parsed()?;

        if self.compress_ratios.len() != self.num_hidden_layers {
            return Err(format!(
                "DeepSeek-V4 config: compress_ratios must have one entry per hidden layer, got \
                 {} for {} layers",
                self.compress_ratios.len(),
                self.num_hidden_layers
            ));
        }
        let bad: Vec<i64> = self
            .compress_ratios
            .iter()
            .copied()
            .filter(|r| !matches!(r, 0 | 4 | 128))
            .collect();
        if !bad.is_empty() {
            return Err(format!(
                "Unsupported DeepSeek-V4 compress ratios: {bad:?} (supported: 0, 4, 128)"
            ));
        }

        // Rope scaling: reject unknown types now rather than at table build.
        rope::v4_rope_base_freqs(
            self.qk_rope_head_dim as i32,
            self.compress_rope_theta,
            self.rope_scaling.as_ref(),
        )?;

        // Quantization: bound the top-level pair, the mode, and EVERY
        // per-path override (a bound on the defaults says nothing about an
        // individual override; docs/adding-models.md, Quantization Parameter
        // Bounds).
        if let Some(q) = self.quantization_block() {
            mlxcel_core::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|e| format!("DeepSeek-V4 quantization: {e}"))?;
            if let Some(mode) = &q.mode {
                mlxcel_core::layers::validate_quantization_mode(mode)
                    .map_err(|e| format!("DeepSeek-V4 quantization: {e}"))?;
            }
            for (path, value) in &q.overrides {
                if !value.is_object() {
                    continue;
                }
                let o: QuantOverride = serde_json::from_value(value.clone()).map_err(|e| {
                    format!("DeepSeek-V4 quantization override `{path}` is malformed: {e}")
                })?;
                mlxcel_core::layers::validate_quantization_params(o.group_size, o.bits)
                    .map_err(|e| format!("DeepSeek-V4 quantization override `{path}`: {e}"))?;
                if let Some(mode) = &o.mode {
                    mlxcel_core::layers::validate_quantization_mode(mode)
                        .map_err(|e| format!("DeepSeek-V4 quantization override `{path}`: {e}"))?;
                }
            }
        }
        Ok(())
    }
}

// Shared helpers.

pub(crate) fn get_weight_copy(
    weights: &WeightMap,
    name: &str,
) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

/// `where(visible, scores, f32::MIN)`, the reference's
/// `mx.finfo(scores.dtype).min` masking for f32 scores.
pub(crate) fn masked_fill_min(scores: &MlxArray, visible: &MlxArray) -> UniquePtr<MlxArray> {
    let min = mlxcel_core::full_f32(&[1], f32::MIN, mlxcel_core::array_dtype(scores));
    mlxcel_core::where_cond(visible, scores, &min)
}

/// Per-layer heterogeneous state: the rotating local window plus the
/// attention and indexer pooling caches (present per [`AttnKind`]).
pub(crate) struct V4LayerCache {
    pub(crate) local: RotatingKVCache,
    pub(crate) pool: Option<PoolingCache>,
    pub(crate) idx_pool: Option<PoolingCache>,
}

impl V4LayerCache {
    /// Force this layer's pooling caches to materialise, so the graph that
    /// produced them stops pinning the hidden states that fed it.
    ///
    /// See [`PoolingCache::eval_state`] for the whole story: the indexer
    /// cache is the one that accumulates, because the `AttnKind::Sparse` arm
    /// of `V4Attention::forward` drops the top-k selection (the indexer
    /// pooled buffer's only consumer) on both of its non-sparse branches,
    /// while the attention cache is already in the logits graph on every
    /// branch and so is forced by the caller's eval anyway.
    ///
    /// `local` is deliberately absent. The rotating window is concatenated
    /// into the SDPA input on every layer of every step, so it is in the
    /// logits graph already and a barrier on it would buy nothing.
    pub(crate) fn eval_state(&self) {
        if let Some(pool) = self.pool.as_ref() {
            pool.eval_state();
        }
        if let Some(idx_pool) = self.idx_pool.as_ref() {
            idx_pool.eval_state();
        }
    }
}

// Decoder block.

struct DeepseekV4Block {
    attn: V4Attention,
    ffn: DeepseekV4MoE,
    attn_norm: RMSNorm,
    ffn_norm: RMSNorm,
    attn_hc: HyperConnection,
    ffn_hc: HyperConnection,
}

impl DeepseekV4Block {
    fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let hidden = args.hidden_size as i32;
        let hc_mult = args.hc_mult as i32;
        let iters = args.hc_sinkhorn_iters as i32;
        Ok(Self {
            attn: V4Attention::from_weights(weights, args, &format!("{prefix}.attn"), layer_idx)?,
            ffn: DeepseekV4MoE::from_weights(weights, args, &format!("{prefix}.ffn"), layer_idx)?,
            attn_norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{prefix}.attn_norm.weight"))?,
                args.rms_norm_eps,
            ),
            ffn_norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{prefix}.ffn_norm.weight"))?,
                args.rms_norm_eps,
            ),
            attn_hc: HyperConnection::from_weights(
                weights,
                &format!("{prefix}.attn_hc"),
                hidden,
                hc_mult,
                iters,
                args.hc_eps,
                args.rms_norm_eps,
            )?,
            ffn_hc: HyperConnection::from_weights(
                weights,
                &format!("{prefix}.ffn_hc"),
                hidden,
                hc_mult,
                iters,
                args.hc_eps,
                args.rms_norm_eps,
            )?,
        })
    }

    /// `h` is the widened `[B, L, hc, D]` state; `input_ids` feeds the
    /// hash-routed MoE gates.
    fn forward(
        &self,
        h: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut V4LayerCache,
        input_ids: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let (x, post, comb) = self.attn_hc.forward(h);
        let x = self.attn.forward(&self.attn_norm.forward(&x), mask, cache);
        let h = hc_expand(&x, h, &post, &comb);

        let (x, post, comb) = self.ffn_hc.forward(&h);
        let x = self.ffn.forward(&self.ffn_norm.forward(&x), input_ids);
        hc_expand(&x, &h, &post, &comb)
    }
}

// Model.

pub struct DeepSeekV4Model {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DeepseekV4Block>,
    norm: RMSNorm,
    hc_head: HyperHead,
    lm_head: Option<UnifiedLinear>,
    hc_mult: i32,
    sliding_window: i32,
    compress_ratios: Vec<i32>,
    eos_token_ids: Vec<i32>,
    /// Heterogeneous per-sequence cache state; see the module docs.
    sequence_state: ModelOwnedSequenceState<V4LayerCache>,
}

impl DeepSeekV4Model {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    /// Build from a raw (checkpoint-named) weight map. Sanitization,
    /// coverage validation, and config normalization all happen here so the
    /// directory route and the weight route cannot drift apart.
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let args = args.clone().normalized()?;
        let weights = sanitize::sanitize_weights(weights, &args)?;
        sanitize::validate_weight_coverage(&weights, &args)?;
        Self::build(&weights, &args)
    }

    fn build(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(DeepseekV4Block::from_weights(weights, args, i)?);
        }

        let norm = RMSNorm::new(
            get_weight_copy(weights, "model.norm.weight")?,
            args.rms_norm_eps,
        );
        let hc_head = HyperHead::from_weights(
            weights,
            "model.hc_head",
            args.hidden_size as i32,
            args.hc_mult as i32,
            args.hc_eps,
            args.rms_norm_eps,
        )?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        let compress_ratios: Vec<i32> = args.compress_ratios.iter().map(|&r| r as i32).collect();
        let model = Self {
            embed_tokens,
            layers,
            norm,
            hc_head,
            lm_head,
            hc_mult: args.hc_mult as i32,
            sliding_window: args.sliding_window as i32,
            compress_ratios,
            eos_token_ids: args.eos_token_ids(),
            sequence_state: ModelOwnedSequenceState::new(Vec::new()),
        };
        model
            .sequence_state
            .replace_internal(model.make_internal_caches());
        Ok(model)
    }

    fn make_internal_caches(&self) -> Vec<V4LayerCache> {
        self.compress_ratios
            .iter()
            .map(|&ratio| V4LayerCache {
                local: RotatingKVCache::new(self.sliding_window),
                pool: (ratio > 0).then(|| PoolingCache::new(ratio)),
                idx_pool: (ratio == OVERLAP_COMPRESS_RATIO).then(|| PoolingCache::new(ratio)),
            })
            .collect()
    }

    pub(crate) fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [V4LayerCache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let (b, l) = (shape[0], shape[1]);

        let h = self.embed_tokens.forward(input_ids);
        let d = *mlxcel_core::array_shape(&h).last().expect("embed output");
        let h = mlxcel_core::expand_dims(&h, 2);
        let h = mlxcel_core::broadcast_to(&h, &[b, l, self.hc_mult, d]);
        let mut h = mlxcel_core::contiguous(&h, false);

        // Every layer's local cache advances in lockstep, so the first
        // layer's rotating cache sizes the shared sliding prefill mask, the
        // same way `create_attention_mask(..., window_size=sliding_window)`
        // reads `cache[0]` in the reference. Decode needs no mask: the
        // rotating cache returns only its window.
        let mask = if l > 1 {
            let local_offset = caches.first().map(|c| c.local.offset).unwrap_or(0);
            Some(create_sliding_window_prefill_mask(
                l,
                local_offset,
                self.sliding_window,
            ))
        } else {
            None
        };

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask.as_deref(), cache, input_ids);
        }

        // Cache-state barrier: the DeepSeek-V4 equivalent of the
        // `[c.state for c in cache]` eval upstream mlx-lm runs after every
        // step, and the reason `KVCache::eval_state` exists on the ordinary
        // KV path. It sits AFTER the layer loop and BEFORE `hc_head` on
        // purpose. The pooled buffers depend only on the layer stack, so
        // forcing them here forces the stack while leaving the `hc_head` /
        // `norm` / `lm_head` chain unevaluated: the full `[B, L, vocab]`
        // logits tensor and its peak allocation are NOT forced by this, which
        // is the same trade `KVCache::eval_state` documents. Without the
        // barrier a sparse layer's indexer cache accumulates one unevaluated
        // `slice_update` node per decode step, pinning that step's hidden
        // state, for as long as the sparse selection path is not being taken
        // (see `PoolingCache::eval_state`).
        for cache in caches.iter() {
            cache.eval_state();
        }

        let h = self.hc_head.forward(&h);
        let h = self.norm.forward(&h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    fn forward_for_sequence(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| self.forward_with_caches(input_ids, caches),
        )
    }
}

// LanguageModel trait implementation (model-owned state, gemma3 pattern).

impl LanguageModel for DeepSeekV4Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, None)
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

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        _input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Text-only family: there is no image-embedding prefill path.
        self.forward_for_sequence(input_ids, seq_id)
    }

    fn reset_runtime_state(&self) {
        self.sequence_state
            .replace_internal(self.make_internal_caches());
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.layers.len())
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_internal_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Compatibility only: the real state lives in `sequence_state`.
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_batching(&self) -> bool {
        // No batched multi-sequence decode: the rotating window and pooling
        // caches have no batched path here. Concurrent server requests stay
        // isolated per sequence through the model-owned `sequence_state`.
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}
