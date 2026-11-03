// Copyright 2025-2026 Lablup Inc.
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

//! Xiaomi MiMo v2 Flash (`mimo_v2_flash`).
//!
//! Ported from mlx-lm's
//! [`mimo_v2_flash.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mimo_v2_flash.py),
//! cross-checked against the vendor `modeling_mimo_v2_flash.py` and the
//! published `mlx-community/MiMo-V2-Flash-4bit` config.
//!
//! The family looks like a DeepSeek-shaped sparse decoder, and the expert half
//! does resemble that lineage, but five details decide whether the port runs
//! the checkpoint or merely loads it.
//!
//! # Hybrid sliding / full attention, selected by a LIST
//!
//! `hybrid_layer_pattern[idx] == 1` marks a sliding layer; `0` marks a full
//! layer. The published schedule is full at layers 0, 5, 11, 17, ..., 47 and
//! sliding at every other layer (9 full, 39 sliding). Sliding layers need a
//! `RotatingKVCache` and full layers a `KVCache`, which does not fit the
//! trait's homogeneous `&mut [KVCache]`, so this model owns its heterogeneous
//! per-sequence state through [`ModelOwnedSequenceState`] and reuses
//! [`crate::models::gemma3::Cache`].
//!
//! # Q/K and V have different head widths
//!
//! Q and K project to `num_attention_heads * head_dim` and
//! `num_key_value_heads * head_dim` (64 x 192 and 4 x 192 on a full layer), but
//! V projects to `num_key_value_heads * v_head_dim` (4 x 128) and O consumes
//! `num_attention_heads * v_head_dim` (64 x 128). Sliding layers have their own
//! head counts and widths (`swa_*`; 8 KV heads instead of 4). Reusing one
//! `head_dim` for all four projections loads no tensor at the wrong shape and
//! is only caught by a forward pass, which is why [`Attention`] carries both
//! widths explicitly.
//!
//! # Sliding layers carry learned attention sinks
//!
//! `add_swa_attention_sink_bias` is true (and the full-attention flag false) on
//! the published checkpoint, so exactly the sliding layers ship
//! `self_attn.attention_sink_bias`, one learned logit per head appended to the
//! softmax denominator. [`Attention::forward`] routes those layers through
//! [`mlxcel_core::fast_scaled_dot_product_attention_with_sinks`].
//!
//! # Partial RoPE with a per-layer base
//!
//! `partial_rotary_factor` is 0.334 in the published JSON. `int(0.334 * 192)`
//! is 64, so only the first 64 of each 192-wide head rotate. The base is
//! `swa_rope_theta` (10000) on sliding layers and `rope_theta` (5000000) on full
//! ones. Both `partial_rotary_factor` and the JSON number are floats while
//! upstream types the factor as an `int`, so it is parsed as `f64` and floored.
//!
//! # The router is `noaux_tc`
//!
//! Scores are `sigmoid(gate(x))` in float32. `e_score_correction_bias` is added
//! before top-k selection, but the returned weights are gathered from the
//! UNBIASED sigmoid scores, normalized (`norm_topk_prob`), and scaled by
//! `routed_scaling_factor` (null, which means 1.0). Gathering from the biased
//! copy loads and generates fluent text while misweighting every routed
//! contribution. Group routing exists but `n_group` is 1 on this checkpoint, so
//! it is inert and only exercised by tests.
//!
//! There are no shared experts in the published checkpoint
//! (`n_shared_experts` is null and no `shared_experts.*` tensors exist). The
//! optional path is kept because upstream has it, but nothing here exercises
//! it.
//!
//! # Weight layout
//!
//! The 4-bit conversion already ships stacked `mlp.switch_mlp.{gate,up,down}_proj`
//! planes, `mlp.gate.weight`, `mlp.gate.e_score_correction_bias`, and
//! `self_attn.attention_sink_bias`. There are no per-expert `mlp.experts.E.*`
//! keys, no fp8 `weight_scale_inv` tensors, and no `model.mtp.*` tensors, so
//! [`sanitize_weights`] is a pass-through that only drops a future MTP block.

use crate::models::gemma3::{Cache, CacheInterface};
use crate::models::gpt2::validate_embedding_table;
use crate::models::model_owned::{KvCacheLayerModes, LayerKv, ModelOwnedSequenceState};
use crate::models::switch_layers::{SwitchGLU, group_mask_scores};
use mlxcel_core::cache::{CachePool, KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// A `config.json` token-id field, which may be a single int or a list of ints.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenIdField {
    Single(i64),
    Multiple(Vec<i64>),
}

impl TokenIdField {
    fn ids(&self) -> Vec<i32> {
        match self {
            Self::Single(id) => vec![*id as i32],
            Self::Multiple(ids) => ids.iter().map(|&id| id as i32).collect(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

/// MiMo v2 Flash `config.json`.
///
/// Every field carries a default so a truncated config still deserializes far
/// enough to report what is wrong; [`ModelArgs::validate`] is what refuses a
/// config that cannot describe this checkpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelArgs {
    pub model_type: String,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Width of the dense prefix MLP (`mlp.gate_proj` on layer 0).
    pub intermediate_size: usize,
    /// Width of one routed expert.
    pub moe_intermediate_size: usize,
    pub vocab_size: usize,

    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,

    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub swa_head_dim: usize,
    pub swa_v_head_dim: usize,

    /// `1` selects sliding-window attention, `0` full attention.
    pub hybrid_layer_pattern: Vec<i64>,
    /// `1` on a sparse layer, `0` on a dense one. Layer 0 is dense.
    pub moe_layer_freq: Vec<i64>,

    pub sliding_window_size: usize,
    pub max_position_embeddings: usize,

    pub layernorm_epsilon: f32,
    pub rope_theta: f32,
    pub swa_rope_theta: f32,
    /// Fraction of each head that rotates. A float in the published JSON even
    /// though upstream types it as an `int`.
    pub partial_rotary_factor: f64,

    pub add_swa_attention_sink_bias: bool,
    pub add_full_attention_sink_bias: bool,
    /// Multiplies V before the cache update. Upstream applies it on both layer
    /// kinds whenever the value is present.
    pub attention_value_scale: Option<f32>,

    pub n_routed_experts: Option<usize>,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: Option<usize>,
    pub routed_scaling_factor: Option<f32>,
    pub topk_method: String,
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub n_group: usize,
    pub topk_group: usize,

    pub tie_word_embeddings: bool,
    pub eos_token_id: Option<TokenIdField>,
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    pub quantization: Option<Quantization>,
}

impl Default for ModelArgs {
    fn default() -> Self {
        Self {
            model_type: "mimo_v2_flash".to_string(),
            hidden_size: 4096,
            num_hidden_layers: 48,
            intermediate_size: 16384,
            moe_intermediate_size: 2048,
            vocab_size: 152576,
            num_attention_heads: 64,
            num_key_value_heads: 4,
            head_dim: 192,
            v_head_dim: 128,
            swa_num_attention_heads: 64,
            swa_num_key_value_heads: 8,
            swa_head_dim: 192,
            swa_v_head_dim: 128,
            hybrid_layer_pattern: Vec::new(),
            moe_layer_freq: Vec::new(),
            sliding_window_size: 128,
            max_position_embeddings: 262_144,
            layernorm_epsilon: 1e-5,
            rope_theta: 5_000_000.0,
            swa_rope_theta: 10_000.0,
            partial_rotary_factor: 0.334,
            add_swa_attention_sink_bias: false,
            add_full_attention_sink_bias: false,
            attention_value_scale: None,
            n_routed_experts: None,
            num_experts_per_tok: 8,
            n_shared_experts: None,
            routed_scaling_factor: None,
            topk_method: "noaux_tc".to_string(),
            scoring_func: "sigmoid".to_string(),
            norm_topk_prob: true,
            n_group: 1,
            topk_group: 1,
            tie_word_embeddings: false,
            eos_token_id: None,
            rope_scaling: None,
            quantization: None,
        }
    }
}

/// Upper bounds on the architecture scalars a `config.json` may declare.
/// `config.json` arrives with a downloaded checkpoint and is untrusted input.
const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
const MAX_HIDDEN_SIZE: usize = 65_536;
const MAX_NUM_ATTENTION_HEADS: usize = 4096;
const MAX_HEAD_DIM: usize = 8192;
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
const MAX_MAX_POSITION_EMBEDDINGS: usize = 1 << 22;
const MAX_VOCAB_SIZE: usize = 1 << 24;
const MAX_NUM_EXPERTS: usize = 4096;
const MAX_NUM_SHARED_EXPERTS: usize = 1024;

impl ModelArgs {
    /// Whether layer `layer_idx` uses sliding-window attention.
    ///
    /// The schedule is the list, not a modulus. A missing entry falls back to
    /// full attention, the conservative direction: a global layer attends to
    /// every key a sliding one would and more, so a short list is slower rather
    /// than wrong.
    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.hybrid_layer_pattern
            .get(layer_idx)
            .copied()
            .unwrap_or(0)
            == 1
    }

    /// Whether layer `layer_idx` is sparse.
    ///
    /// Mirrors the vendor decoder: a layer is MoE only when the checkpoint
    /// declares routed experts AND `moe_layer_freq[idx]` is 1.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.n_routed_experts.is_some()
            && self.moe_layer_freq.get(layer_idx).copied().unwrap_or(0) == 1
    }

    /// Index of the first full-attention layer, whose cache sizes the global
    /// prefill mask.
    pub fn full_attention_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| !self.is_sliding_layer(i))
    }

    /// Index of the first sliding layer, whose cache sizes the windowed prefill
    /// mask.
    pub fn sliding_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| self.is_sliding_layer(i))
    }

    /// Number of routed experts, or 0 when the checkpoint declares none.
    pub fn num_routed_experts(&self) -> usize {
        self.n_routed_experts.unwrap_or(0)
    }

    /// Width of the always-on shared MLP, if any.
    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
            .saturating_mul(self.n_shared_experts.unwrap_or(0))
    }

    pub fn has_shared_expert(&self) -> bool {
        self.n_shared_experts.unwrap_or(0) > 0
    }

    /// `routed_scaling_factor`, where null means 1.0.
    pub fn routed_scaling_factor(&self) -> f32 {
        self.routed_scaling_factor.unwrap_or(1.0)
    }

    /// `int(partial_rotary_factor * head_dim)`, floored as upstream does.
    pub fn rope_dims_for(&self, head_dim: usize) -> Result<i32, String> {
        let dims = (self.partial_rotary_factor * head_dim as f64).floor();
        if !dims.is_finite() || dims < 2.0 {
            return Err(format!(
                "MiMo v2 Flash partial_rotary_factor ({}) times head_dim ({head_dim}) gives \
                 {dims} rotary dimensions; RoPE rotates channel pairs and needs at least 2",
                self.partial_rotary_factor
            ));
        }
        let dims = dims as usize;
        if dims > head_dim {
            return Err(format!(
                "MiMo v2 Flash partial_rotary_factor ({}) times head_dim ({head_dim}) gives \
                 {dims} rotary dimensions, more channels than the head has",
                self.partial_rotary_factor
            ));
        }
        if !dims.is_multiple_of(2) {
            return Err(format!(
                "MiMo v2 Flash rope dimensions ({dims}) must be even; RoPE rotates channel pairs"
            ));
        }
        i32::try_from(dims)
            .map_err(|_| format!("MiMo v2 Flash rope dimensions ({dims}) do not fit in i32"))
    }

    /// The RoPE base layer `layer_idx` rotates with.
    pub fn rope_base_for(&self, layer_idx: usize) -> f32 {
        if self.is_sliding_layer(layer_idx) {
            self.swa_rope_theta
        } else {
            self.rope_theta
        }
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

    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdField::ids)
            .unwrap_or_default()
    }

    /// Reject a config that cannot describe a real MiMo v2 Flash before any of
    /// its fields sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        check_range(
            "num_attention_heads",
            self.num_attention_heads,
            MAX_NUM_ATTENTION_HEADS,
        )?;
        check_range("hidden_size", self.hidden_size, MAX_HIDDEN_SIZE)?;
        check_range("head_dim", self.head_dim, MAX_HEAD_DIM)?;
        check_range("v_head_dim", self.v_head_dim, MAX_HEAD_DIM)?;
        check_range(
            "swa_num_attention_heads",
            self.swa_num_attention_heads,
            MAX_NUM_ATTENTION_HEADS,
        )?;
        check_range("swa_head_dim", self.swa_head_dim, MAX_HEAD_DIM)?;
        check_range("swa_v_head_dim", self.swa_v_head_dim, MAX_HEAD_DIM)?;
        check_range(
            "num_hidden_layers",
            self.num_hidden_layers,
            MAX_NUM_HIDDEN_LAYERS,
        )?;
        check_range(
            "intermediate_size",
            self.intermediate_size,
            MAX_INTERMEDIATE_SIZE,
        )?;
        check_range(
            "moe_intermediate_size",
            self.moe_intermediate_size,
            MAX_INTERMEDIATE_SIZE,
        )?;
        check_range("vocab_size", self.vocab_size, MAX_VOCAB_SIZE)?;
        check_range(
            "max_position_embeddings",
            self.max_position_embeddings,
            MAX_MAX_POSITION_EMBEDDINGS,
        )?;

        check_gqa(
            "num_attention_heads",
            self.num_attention_heads,
            "num_key_value_heads",
            self.num_key_value_heads,
        )?;
        check_gqa(
            "swa_num_attention_heads",
            self.swa_num_attention_heads,
            "swa_num_key_value_heads",
            self.swa_num_key_value_heads,
        )?;

        // The schedule lists are indexed directly by both implementations, so a
        // short list would silently change layer kinds.
        if self.hybrid_layer_pattern.len() < self.num_hidden_layers {
            return Err(format!(
                "MiMo v2 Flash hybrid_layer_pattern has {} entries for {} layers; it is the \
                 per-layer sliding/full schedule and a short list would silently make the \
                 uncovered tail global",
                self.hybrid_layer_pattern.len(),
                self.num_hidden_layers
            ));
        }
        if self.moe_layer_freq.len() < self.num_hidden_layers {
            return Err(format!(
                "MiMo v2 Flash moe_layer_freq has {} entries for {} layers; it is the per-layer \
                 dense/sparse schedule and a short list would silently make the uncovered tail \
                 dense",
                self.moe_layer_freq.len(),
                self.num_hidden_layers
            ));
        }
        for (idx, value) in self.hybrid_layer_pattern.iter().enumerate() {
            if !matches!(value, 0 | 1) {
                return Err(format!(
                    "MiMo v2 Flash hybrid_layer_pattern[{idx}] is {value}; only 0 (full \
                     attention) and 1 (sliding) are defined"
                ));
            }
        }
        for (idx, value) in self.moe_layer_freq.iter().enumerate() {
            if !matches!(value, 0 | 1) {
                return Err(format!(
                    "MiMo v2 Flash moe_layer_freq[{idx}] is {value}; only 0 (dense) and 1 (MoE) \
                     are defined"
                ));
            }
        }

        // Only the Q/K widths rotate, so only they need to be even; V's width
        // has no RoPE constraint.
        checks_even("head_dim", self.head_dim)?;
        checks_even("swa_head_dim", self.swa_head_dim)?;
        let rope_dims = self.rope_dims_for(self.head_dim)?;
        let swa_rope_dims = self.rope_dims_for(self.swa_head_dim)?;
        if self.sliding_index().is_some()
            && (swa_rope_dims <= 0 || !rotary_fits(swa_rope_dims, self.swa_head_dim))
        {
            return Err("MiMo v2 Flash sliding-layer rotary width is invalid".to_string());
        }
        if self.full_attention_index().is_some()
            && (rope_dims <= 0 || !rotary_fits(rope_dims, self.head_dim))
        {
            return Err("MiMo v2 Flash full-attention rotary width is invalid".to_string());
        }

        // A zero window keeps no keys; `cache_window` also bounds it to what a
        // `RotatingKVCache` can hold.
        crate::models::model_owned::cache_window(
            "MiMo v2 Flash",
            self.sliding_window_size,
            self.sliding_index().is_some(),
        )?;

        check_positive_finite("layernorm_epsilon", self.layernorm_epsilon as f64)?;
        check_positive_finite("rope_theta", self.rope_theta as f64)?;
        check_positive_finite("swa_rope_theta", self.swa_rope_theta as f64)?;
        if let Some(scale) = self.attention_value_scale {
            check_positive_finite("attention_value_scale", scale as f64)?;
        }

        self.validate_rope_scaling()?;
        self.validate_routing()?;
        self.validate_shared_expert()?;
        self.validate_quantization()
    }

    fn validate_routing(&self) -> Result<(), String> {
        if self.topk_method != "noaux_tc" {
            return Err(format!(
                "MiMo v2 Flash topk_method ({:?}) is not implemented; upstream asserts \
                 \"noaux_tc\" and this loader implements no other selection rule",
                self.topk_method
            ));
        }
        if self.scoring_func != "sigmoid" {
            return Err(format!(
                "MiMo v2 Flash scoring_func ({:?}) is not implemented; both references raise on \
                 anything but \"sigmoid\"",
                self.scoring_func
            ));
        }

        let has_moe_layer = (0..self.num_hidden_layers)
            .any(|i| self.moe_layer_freq.get(i).copied().unwrap_or(0) == 1);
        if !has_moe_layer {
            return Ok(());
        }
        let experts = self.n_routed_experts.ok_or_else(|| {
            "MiMo v2 Flash moe_layer_freq declares sparse layers but n_routed_experts is null; \
             the checkpoint would load as an all-dense model and ignore every expert tensor"
                .to_string()
        })?;
        check_range("n_routed_experts", experts, MAX_NUM_EXPERTS)?;
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > experts {
            return Err(format!(
                "MiMo v2 Flash num_experts_per_tok ({}) must be between 1 and n_routed_experts \
                 ({experts}); the router selects that many indices out of a row of expert scores",
                self.num_experts_per_tok
            ));
        }
        if self.n_group == 0 {
            return Err(
                "MiMo v2 Flash n_group must be at least 1 (1 disables grouped routing)".to_string(),
            );
        }
        if self.n_group > 1 {
            if !experts.is_multiple_of(self.n_group) {
                return Err(format!(
                    "MiMo v2 Flash n_routed_experts ({experts}) must be divisible by n_group ({}); \
                     grouped routing reshapes the score row into equal groups",
                    self.n_group
                ));
            }
            if experts / self.n_group < 2 {
                return Err(format!(
                    "MiMo v2 Flash n_group ({}) leaves fewer than 2 experts per group; grouped \
                     routing scores a group by the sum of its top two experts",
                    self.n_group
                ));
            }
            // `topk_group == n_group` keeps every group, which is the
            // identity mask both references accept.
            if self.topk_group == 0 || self.topk_group > self.n_group {
                return Err(format!(
                    "MiMo v2 Flash topk_group ({}) must be between 1 and n_group ({})",
                    self.topk_group, self.n_group
                ));
            }
        }
        check_positive_finite("routed_scaling_factor", self.routed_scaling_factor() as f64)
    }

    fn validate_shared_expert(&self) -> Result<(), String> {
        let Some(shared) = self.n_shared_experts else {
            return Ok(());
        };
        if shared > MAX_NUM_SHARED_EXPERTS {
            return Err(format!(
                "MiMo v2 Flash n_shared_experts ({shared}) must not exceed \
                 {MAX_NUM_SHARED_EXPERTS}"
            ));
        }
        if shared == 0 {
            return Ok(());
        }
        let width = self
            .moe_intermediate_size
            .checked_mul(shared)
            .ok_or_else(|| {
                format!(
                    "MiMo v2 Flash shared-expert width overflows: moe_intermediate_size ({}) * \
                 n_shared_experts ({shared})",
                    self.moe_intermediate_size
                )
            })?;
        if width == 0 || width > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "MiMo v2 Flash shared-expert width ({width}) must be between 1 and \
                 {MAX_INTERMEDIATE_SIZE}"
            ));
        }
        Ok(())
    }

    fn validate_rope_scaling(&self) -> Result<(), String> {
        let Some(scaling) = self.rope_scaling.as_ref() else {
            return Ok(());
        };
        if scaling.is_empty() {
            return Ok(());
        }
        let kind = scaling
            .get("rope_type")
            .or_else(|| scaling.get("type"))
            .and_then(serde_json::Value::as_str);
        if kind == Some("default") {
            return Ok(());
        }
        Err(format!(
            "MiMo v2 Flash rope_scaling ({scaling:?}) is not implemented; this loader always \
             builds the plain rotation, so accepting a scaled block would move every token to \
             the wrong position while the model still loaded and generated fluent text"
        ))
    }

    fn validate_quantization(&self) -> Result<(), String> {
        let Some(quantization) = self.quantization.as_ref() else {
            return Ok(());
        };
        mlxcel_core::layers::validate_quantization_params(
            quantization.group_size,
            quantization.bits,
        )
        .map_err(|e| format!("MiMo v2 Flash config.json: {e}"))
    }
}

fn check_range(name: &str, value: usize, max: usize) -> Result<(), String> {
    if value == 0 || value > max {
        return Err(format!(
            "MiMo v2 Flash {name} ({value}) must be between 1 and {max}"
        ));
    }
    Ok(())
}

fn check_gqa(heads_name: &str, heads: usize, kv_name: &str, kv_heads: usize) -> Result<(), String> {
    if kv_heads == 0 || kv_heads > heads {
        return Err(format!(
            "MiMo v2 Flash {kv_name} ({kv_heads}) must be between 1 and {heads_name} ({heads})"
        ));
    }
    if !heads.is_multiple_of(kv_heads) {
        return Err(format!(
            "MiMo v2 Flash {heads_name} ({heads}) must be divisible by {kv_name} ({kv_heads}) \
             for grouped-query attention"
        ));
    }
    Ok(())
}

fn checks_even(name: &str, value: usize) -> Result<(), String> {
    if !value.is_multiple_of(2) {
        return Err(format!(
            "MiMo v2 Flash {name} ({value}) must be even; RoPE rotates channel pairs"
        ));
    }
    Ok(())
}

fn rotary_fits(rope_dims: i32, head_dim: usize) -> bool {
    rope_dims > 0 && (rope_dims as usize) <= head_dim && (rope_dims as usize).is_multiple_of(2)
}

fn check_positive_finite(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "MiMo v2 Flash {name} ({value}) must be a finite positive number"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Weight sanitization
// ---------------------------------------------------------------------------

/// Normalize a weight map for this family.
///
/// The published 4-bit conversion already stores the stacked
/// `mlp.switch_mlp.*` expert planes and every tensor under the names this
/// loader reads, so this is a pass-through. It only drops a `model.mtp.*` block
/// if a future conversion adds one, matching upstream's sanitizer; the current
/// index has no such tensors.
pub fn sanitize_weights(weights: &mut WeightMap) {
    let mtp: Vec<String> = weights
        .keys()
        .filter(|key| key.starts_with("model.mtp"))
        .cloned()
        .collect();
    for key in mtp {
        weights.remove(&key);
    }
}

fn rms_norm_from_weights(weights: &WeightMap, key: &str, eps: f32) -> Result<RMSNorm, String> {
    let weight = weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    Ok(RMSNorm::new(weight, eps))
}

fn check_tensor_shape(weights: &WeightMap, key: &str, expected: &[i32]) -> Result<(), String> {
    let Some(array) = weights.get(key) else {
        return Ok(());
    };
    let actual = mlxcel_core::array_shape(array);
    if actual != expected {
        return Err(format!(
            "MiMo v2 Flash {key} has shape {actual:?}; expected {expected:?}"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// MiMo v2 Flash attention: GQA with a separate V head width, partial RoPE, and
/// (on sliding layers) learned per-head sinks.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    /// Q/K head width.
    pub head_dim: i32,
    /// V/O head width.
    pub v_head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    /// Applied to V before the cache update on both layer kinds.
    pub value_scale: Option<f32>,
    /// Learned per-head logit appended to the softmax denominator.
    pub sinks: Option<UniquePtr<MlxArray>>,
}

impl Attention {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Q/K use `head_dim`; V uses `v_head_dim`. One width for all four is
        // the mistake this split exists to make impossible.
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.v_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let v = match self.value_scale {
            Some(scale) => mlxcel_core::multiply_scalar(&v, scale),
            None => v,
        };

        let offset = cache.offset();
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let attn_out = match &self.sinks {
            Some(sinks) => {
                let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
                // SAFETY: `sinks` is a live per-head array owned by `self`, and
                // the mask pointer is either null or borrowed from the caller
                // for the duration of the call.
                unsafe {
                    mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
                        &q,
                        &cache_k,
                        &cache_v,
                        self.scale,
                        mask_ptr,
                        sinks.as_ref().expect("attention sinks are present") as *const _,
                    )
                }
            }
            None if l > 1 && mask.is_none() => {
                mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
            }
            None => {
                let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
                // SAFETY: the mask pointer is either null or borrowed from the
                // caller for the duration of the call.
                unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                    )
                }
            }
        };

        // O consumes `num_heads * v_head_dim`, not `num_heads * head_dim`.
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.v_head_dim]);
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        is_sliding: bool,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let load = |leaf: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{leaf}"), group_size, bits)
        };

        let (num_heads, num_kv_heads, head_dim, v_head_dim, rope_base, has_sinks) = if is_sliding {
            (
                args.swa_num_attention_heads,
                args.swa_num_key_value_heads,
                args.swa_head_dim,
                args.swa_v_head_dim,
                args.swa_rope_theta,
                args.add_swa_attention_sink_bias,
            )
        } else {
            (
                args.num_attention_heads,
                args.num_key_value_heads,
                args.head_dim,
                args.v_head_dim,
                args.rope_theta,
                args.add_full_attention_sink_bias,
            )
        };

        let num_heads_i32 = i32::try_from(num_heads)
            .map_err(|_| format!("MiMo v2 Flash {prefix}: head count does not fit in i32"))?;
        let num_kv_heads_i32 = i32::try_from(num_kv_heads)
            .map_err(|_| format!("MiMo v2 Flash {prefix}: KV head count does not fit in i32"))?;
        let head_dim_i32 = i32::try_from(head_dim)
            .map_err(|_| format!("MiMo v2 Flash {prefix}: head_dim does not fit in i32"))?;
        let v_head_dim_i32 = i32::try_from(v_head_dim)
            .map_err(|_| format!("MiMo v2 Flash {prefix}: v_head_dim does not fit in i32"))?;

        let sinks = if has_sinks {
            let key = format!("{prefix}.attention_sink_bias");
            check_tensor_shape(weights, &key, &[num_heads_i32])?;
            let sinks = weights
                .get(&key)
                .map(|w| mlxcel_core::astype(w, mlxcel_core::dtype::FLOAT32))
                .ok_or_else(|| {
                    format!(
                        "MiMo v2 Flash: {key} is required because the config enables attention \
                         sinks on this layer kind"
                    )
                })?;
            Some(sinks)
        } else {
            None
        };

        Ok(Self {
            q_proj: load("q_proj")?,
            k_proj: load("k_proj")?,
            v_proj: load("v_proj")?,
            o_proj: load("o_proj")?,
            num_heads: num_heads_i32,
            num_kv_heads: num_kv_heads_i32,
            head_dim: head_dim_i32,
            v_head_dim: v_head_dim_i32,
            scale: 1.0 / (head_dim_i32 as f32).sqrt(),
            rope_dims: args.rope_dims_for(head_dim)?,
            rope_base,
            value_scale: args.attention_value_scale,
            sinks,
        })
    }
}

// ---------------------------------------------------------------------------
// Feed-forward
// ---------------------------------------------------------------------------

/// SwiGLU MLP, used for the dense prefix layers and the optional shared expert.
pub struct DenseMlp {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl DenseMlp {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let load = |leaf: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{leaf}"), group_size, bits)
        };
        Ok(Self {
            gate_proj: load("gate_proj")?,
            up_proj: load("up_proj")?,
            down_proj: load("down_proj")?,
        })
    }
}

/// `noaux_tc` router: sigmoid scores, a selection-only correction bias,
/// top-k weights gathered from the unbiased scores, normalization, and the
/// routed scaling factor.
pub struct MoeGate {
    pub gate: UnifiedLinear,
    /// `[n_routed_experts]` in float32; selection only.
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub top_k: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
}

impl MoeGate {
    /// Returns `(expert_indices, expert_weights)` for a `[tokens, hidden]` input.
    pub fn route(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let logits = self.gate.forward(x);
        let logits = mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32);
        let scores = mlxcel_core::sigmoid(&logits);
        self.route_scores(&scores)
    }

    /// The selection and weighting step, split out so it is unit-testable
    /// without a weight map.
    fn route_scores(&self, scores: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // The bias participates only in selection.
        let selection = mlxcel_core::add(scores, &self.e_score_correction_bias);
        let selection = if self.n_group > 1 {
            group_mask_scores(&selection, self.n_group, self.topk_group)
        } else {
            selection
        };

        // `argpartition(-selection, kth = k - 1)` then the first k, which is
        // upstream's orientation. The mirrored form picks a different set when
        // selection scores tie.
        let shape = mlxcel_core::array_shape(&selection);
        let n_experts = *shape.last().unwrap_or(&0);
        let kth = (self.top_k - 1).clamp(0, (n_experts - 1).max(0));
        let order = mlxcel_core::argpartition(&mlxcel_core::negative(&selection), kth, -1);
        let indices = slice_axis(&order, -1, 0, self.top_k);

        // Weights come from the UNBIASED, unmasked sigmoid scores.
        let selected = mlxcel_core::take_along_axis(scores, &indices, -1);
        let selected = if self.top_k > 1 && self.norm_topk_prob {
            // Upstream divides by `sum + 1e-20`; the epsilon keeps a row whose
            // sigmoid scores all underflowed to zero from dividing by zero.
            let denominator = mlxcel_core::add(
                &mlxcel_core::sum_axis(&selected, -1, true),
                &mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::dtype::FLOAT32),
            );
            mlxcel_core::divide(&selected, &denominator)
        } else {
            selected
        };

        let selected = if self.routed_scaling_factor == 1.0 {
            selected
        } else {
            let scale = mlxcel_core::full_f32(
                &[1],
                self.routed_scaling_factor,
                mlxcel_core::dtype::FLOAT32,
            );
            mlxcel_core::multiply(&selected, &scale)
        };

        (indices, selected)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let experts = args.num_routed_experts();
        let experts_i32 = i32::try_from(experts)
            .map_err(|_| "MiMo v2 Flash: expert count does not fit in i32".to_string())?;

        let bias_key = format!("{prefix}.gate.e_score_correction_bias");
        check_tensor_shape(weights, &bias_key, &[experts_i32])?;
        let e_score_correction_bias = match weights.get(&bias_key) {
            Some(bias) => mlxcel_core::astype(bias, mlxcel_core::dtype::FLOAT32),
            // Upstream initializes the parameter to zeros, so a checkpoint that
            // omits it still routes with a zero correction.
            None => mlxcel_core::zeros(&[experts_i32], mlxcel_core::dtype::FLOAT32),
        };

        Ok(Self {
            gate: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate"),
                group_size,
                bits,
            )?,
            e_score_correction_bias,
            top_k: i32::try_from(args.num_experts_per_tok)
                .map_err(|_| "MiMo v2 Flash: num_experts_per_tok does not fit in i32")?,
            n_group: i32::try_from(args.n_group)
                .map_err(|_| "MiMo v2 Flash: n_group does not fit in i32")?,
            topk_group: i32::try_from(args.topk_group)
                .map_err(|_| "MiMo v2 Flash: topk_group does not fit in i32")?,
            routed_scaling_factor: args.routed_scaling_factor(),
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

/// Sparse block: router, stacked experts, and the optional shared expert.
pub struct Moe {
    pub gate: MoeGate,
    pub experts: SwitchGLU,
    pub shared_experts: Option<DenseMlp>,
}

impl Moe {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = *orig_shape.last().unwrap_or(&0);
        let x_flat = if orig_shape.len() > 2 {
            let tokens: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[tokens, hidden])
        } else {
            mlxcel_core::copy(x)
        };

        let (indices, scores) = self.gate.route(&x_flat);
        let expert_out = self.experts.forward(&x_flat, &indices);

        // Upstream multiplies in float32 and casts once at the end, the same
        // contract AFMoE documents; the shared `moe_weighted_sum` helper would
        // cast the weights down first.
        let out32 = mlxcel_core::astype(&expert_out, mlxcel_core::dtype::FLOAT32);
        let weighted = mlxcel_core::multiply(&out32, &mlxcel_core::expand_dims(&scores, -1));
        let summed = mlxcel_core::sum_axis(&weighted, -2, false);
        let routed = mlxcel_core::astype(&summed, mlxcel_core::array_dtype(&expert_out));
        let routed = if orig_shape.len() > 2 {
            mlxcel_core::reshape(&routed, &orig_shape)
        } else {
            routed
        };

        match &self.shared_experts {
            Some(shared) => mlxcel_core::add(&routed, &shared.forward(x)),
            None => routed,
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        Ok(Self {
            gate: MoeGate::from_weights(weights, args, prefix)?,
            experts: SwitchGLU::from_weights(
                weights,
                &format!("{prefix}.switch_mlp"),
                group_size,
                bits,
            )?,
            shared_experts: if args.has_shared_expert() {
                Some(DenseMlp::from_weights(
                    weights,
                    args,
                    &format!("{prefix}.shared_experts"),
                )?)
            } else {
                None
            },
        })
    }
}

/// Either the dense prefix MLP or the sparse block, per layer.
pub enum FeedForward {
    Dense(DenseMlp),
    Sparse(Box<Moe>),
}

impl FeedForward {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(moe) => moe.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder layer and model
// ---------------------------------------------------------------------------

/// One MiMo v2 Flash block: pre-norm attention and FFN with two residuals.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: FeedForward,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub uses_sliding: bool,
}

impl DecoderLayer {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let attn = self
            .self_attn
            .forward(&self.input_layernorm.forward(x), cache, mask);
        let h = mlxcel_core::add(x, &attn);
        let ffn = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &ffn)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let uses_sliding = args.is_sliding_layer(layer_idx);
        let norm = |leaf: &str| {
            rms_norm_from_weights(
                weights,
                &format!("{prefix}.{leaf}.weight"),
                args.layernorm_epsilon,
            )
        };
        Ok(Self {
            self_attn: Attention::from_weights(
                weights,
                args,
                &format!("{prefix}.self_attn"),
                uses_sliding,
            )?,
            mlp: if args.is_moe_layer(layer_idx) {
                FeedForward::Sparse(Box::new(Moe::from_weights(
                    weights,
                    args,
                    &format!("{prefix}.mlp"),
                )?))
            } else {
                FeedForward::Dense(DenseMlp::from_weights(
                    weights,
                    args,
                    &format!("{prefix}.mlp"),
                )?)
            },
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            uses_sliding,
        })
    }
}

/// Xiaomi MiMo v2 Flash.
pub struct MiMoV2FlashModel {
    pub embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    sliding_window: i32,
    full_attention_index: Option<usize>,
    sliding_index: Option<usize>,
    /// Whether any full-attention layer carries sinks. When none does, full
    /// layers take the maskless causal kernel and no dense quadratic prefill
    /// mask is built.
    full_attention_sinks: bool,
    eos_token_ids: Vec<i32>,
    /// Sliding layers hold a `RotatingKVCache` and full layers a `KVCache`,
    /// which does not fit the trait's homogeneous `&mut [KVCache]`.
    sequence_state: ModelOwnedSequenceState<Cache>,
    kv_cache_layer_modes: KvCacheLayerModes,
}

impl MiMoV2FlashModel {
    pub(crate) fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        let embeddings = self.embed_tokens.forward(input_ids);
        self.forward_prepared(&embeddings, caches)
    }

    /// Run the decoder over already-embedded rows.
    ///
    /// Multimodal wrappers scatter vision features into the text embeddings
    /// and hand the result here, so this must not re-embed the token ids.
    pub(crate) fn forward_prepared(
        &self,
        embeddings: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(embeddings)[1];
        let mut h = mlxcel_core::copy(embeddings);

        // Masks are sized from each cache's live window rather than the
        // monotonic offset, so a trimmed cache does not produce a mask wider
        // than the K/V it returns.
        let (full_mask, sliding_mask) = if seq_len > 1 {
            let full_live = self
                .full_attention_index
                .map(|i| caches[i].as_interface().live_len())
                .unwrap_or(0);
            let sliding_live = self
                .sliding_index
                .map(|i| caches[i].as_interface().live_len())
                .unwrap_or(0);
            (
                // A sink-aware full layer needs an explicit causal mask, since
                // the sink kernel has no maskless causal mode. Without sinks,
                // `Attention::forward` takes `causal_attention`, so the dense
                // `[seq, seq]` mask is never materialized.
                (self.full_attention_index.is_some() && self.full_attention_sinks)
                    .then(|| create_causal_mask(seq_len, full_live)),
                self.sliding_index.map(|_| {
                    create_sliding_window_prefill_mask(seq_len, sliding_live, self.sliding_window)
                }),
            )
        } else {
            (None, None)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.uses_sliding {
                sliding_mask.as_ref().map(|m| m.as_ref().expect("mask"))
            } else {
                full_mask.as_ref().map(|m| m.as_ref().expect("mask"))
            };
            h = layer.forward(&h, caches[i].as_interface(), mask);
        }

        let h = self.norm.forward(&h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    fn make_internal_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| {
                if layer.uses_sliding {
                    let mode = self
                        .kv_cache_layer_modes
                        .rotating_mode_for_layer(layer_idx, self.sliding_window);
                    Cache::Rotating(RotatingKVCache::new_with_mode(self.sliding_window, mode))
                } else {
                    let mode = self.kv_cache_layer_modes.mode_for_layer(layer_idx);
                    Cache::Standard(KVCache::new_with_mode(mode))
                }
            })
            .collect()
    }

    fn forward_for_sequence(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(input)[1];
        if seq_id.is_none() && seq_len > 1 {
            self.sequence_state
                .replace_internal(self.make_internal_caches());
        }
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| self.forward(input, caches),
        )
    }

    /// Token embedding table only, for a multimodal wrapper that has to
    /// scatter features before decoding.
    pub fn input_embeddings(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Decode already-embedded rows, resetting the model-owned caches when a
    /// fresh multi-token prompt arrives without a sequence id.
    pub fn forward_prepared_embeddings_with_sequence_id(
        &self,
        embeddings: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(embeddings)[1];
        if seq_id.is_none() && seq_len > 1 {
            self.sequence_state
                .replace_internal(self.make_internal_caches());
        }
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| self.forward_prepared(embeddings, caches),
        )
    }

    /// Replace the stop-token set with the one resolved from the checkpoint's
    /// sidecars (`generation_config.json`, `tokenizer_config.json`,
    /// `config.json`).
    pub fn set_eos_token_ids(&mut self, ids: Vec<i32>) {
        if !ids.is_empty() {
            self.eos_token_ids = ids;
        }
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        args.validate()?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        sanitize_weights(&mut weights);
        let mut model = Self::from_weights(&weights, &args)?;
        model.set_eos_token_ids(crate::loading::read_eos_token_ids(model_dir));

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;

        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;
        validate_embedding_table(
            &embed_tokens,
            "model.embed_tokens",
            args.vocab_size,
            "vocab_size",
            args.hidden_size,
            "hidden_size",
        )?;

        let sliding_window = crate::models::model_owned::cache_window(
            "MiMo v2 Flash",
            args.sliding_window_size,
            args.sliding_index().is_some(),
        )?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for layer_idx in 0..args.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(weights, args, layer_idx)?);
        }

        let norm = rms_norm_from_weights(weights, "model.norm.weight", args.layernorm_epsilon)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        let internal_caches: Vec<Cache> = layers
            .iter()
            .map(|layer| {
                if layer.uses_sliding {
                    Cache::Rotating(RotatingKVCache::new(sliding_window))
                } else {
                    Cache::Standard(KVCache::new())
                }
            })
            .collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window,
            full_attention_index: args.full_attention_index(),
            sliding_index: args.sliding_index(),
            full_attention_sinks: args.add_full_attention_sink_bias
                && args.full_attention_index().is_some(),
            eos_token_ids: args.eos_token_ids(),
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
        })
    }
}

impl LanguageModel for MiMoV2FlashModel {
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

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |caches| {
                let visible_lens: Vec<_> = caches.iter().map(Cache::visible_len).collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &visible_lens)
            })
            .unwrap_or_else(|| {
                Err(format!(
                    "missing MiMo v2 Flash sequence state while synchronizing sequence {seq_id}"
                ))
            })
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.kv_cache_layer_modes.set(modes);
        self.sequence_state
            .replace_internal(self.make_internal_caches());
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        self.kv_cache_layer_modes.storage_modes(|layer_idx| {
            self.layers
                .get(layer_idx)
                .is_some_and(|layer| layer.uses_sliding)
                .then_some(self.sliding_window)
                .map_or(LayerKv::Full, LayerKv::Sliding)
        })
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

    /// The model-owned heterogeneous caches cannot be batched through the
    /// scheduler's dense `KVCache` rows, so decode stays one sequence at a time.
    fn supports_batching(&self) -> bool {
        false
    }

    /// The scheduler's post-pad trim only reaches `CachePool` caches, and a
    /// model-owned family's pool entry holds none.
    fn supports_padded_prefill(&self) -> bool {
        false
    }

    /// A multi-token no-id forward starts an independent prompt by resetting
    /// the model-owned caches, so legacy chunked prefill would lose state.
    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "mimo_v2_flash_tests.rs"]
mod tests;
