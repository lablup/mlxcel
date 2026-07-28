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

//! Ant Group Ling / Bailing MoE (`bailing_moe`).
//!
//! Ported from mlx-lm's
//! [`bailing_moe.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/bailing_moe.py).
//!
//! Bailing is a DeepSeek-shaped sparse decoder: RMSNorm before attention and
//! before the FFN, grouped-query attention with RoPE, `num_experts` routed
//! SwiGLU experts selected per token, and one always-on shared expert. The
//! expert machinery is [`crate::models::switch_layers`] unchanged. What is not
//! shared with DeepSeek, and what this module exists to get right, is below.
//!
//! # The weight keys are not Llama keys
//!
//! Despite the DeepSeek-shaped block, the checkpoint uses the GPT-NeoX naming:
//! the token table is `model.word_embeddings` (not `model.embed_tokens`), the
//! output projection is `attention.dense` (not `self_attn.o_proj`), and Q, K and
//! V arrive fused in one `attention.query_key_value` matrix. On
//! `inclusionAI/Ling-lite-1.5` that matrix is `[3072, 2048]`, which is
//! `(16 + 2 * 4) * 128`: 16 query heads and 4 KV heads of width 128. The split
//! offsets are therefore `[q_size, q_size + kv_size]`, the uneven GPT-BigCode
//! form, not an even three-way split. See [`ModelArgs::qkv_split_offsets`].
//!
//! # The router rename collides with the expert projection names
//!
//! Upstream's `sanitize` renames the router `mlp.gate.weight` to
//! `mlp.gate.gate_proj.weight`, so after conversion the router lives at a key
//! ending in `gate_proj` while every routed expert also has a `gate_proj`. A
//! remap or lookup rule that matches on the `gate_proj` suffix without anchoring
//! the prefix swallows the router weight, and the model still loads and still
//! generates. This loader never pattern-matches a suffix: the router is read
//! from one of two fully anchored keys resolved by [`router_prefix`], and the
//! experts are read through the `switch_mlp` virtual prefix, whose per-expert
//! fallback in `switch_layers` anchors on `.experts.{idx}.`. The two key spaces
//! cannot overlap. `router_lookup_is_anchored_against_the_expert_gate_proj` in
//! the test module pins this with both tensors present at once.
//!
//! # Selection uses the biased scores, weights come from the unbiased ones
//!
//! `group_expert_select` saves `orig_scores` *before* adding the router
//! correction bias, selects the top-k indices from the biased copy, and then
//! gathers the returned weights from `orig_scores`. Applying the bias to the
//! weights as well leaves the output finite and plausible while misweighting
//! every routed contribution, which no shape or NaN check can see. This is the
//! same selection-only-bias contract DeepSeek-V3 and ERNIE-4.5 implement in this
//! tree; see [`BailingMoeGate::forward`].
//!
//! # `moe_router_enable_routed_scaling` is dead code upstream
//!
//! `BailingMoeGate` stores the flag and never reads it: `group_expert_select`
//! ends with an unconditional `scores * routed_scaling_factor`. This port
//! **mirrors upstream** and scales unconditionally, so that a checkpoint
//! decoded here matches the reference token for token. Because the two readings
//! diverge on any checkpoint that sets a non-unit factor together with the flag
//! false, [`ModelArgs::warn_on_ignored_routed_scaling_flag`] prints a diagnostic
//! naming both fields for exactly that combination, and
//! `routed_scaling_is_applied_even_when_the_flag_is_false` pins the behavior.
//!
//! # `norm_head` is live, `norm_softmax` is not
//!
//! `norm_head` is read by upstream's `sanitize`, which L2-normalizes
//! `lm_head.weight` along axis 0 in float32 with a `1e-7` epsilon and casts back
//! to the original dtype; the vendored `modeling_bailing_moe.py` does the same
//! once at inference. It is implemented here in [`normalize_lm_head_weight`].
//! `norm_softmax` is declared by upstream's `ModelArgs` and never read, does not
//! appear in the vendored modeling file, and is not even a named parameter of
//! the vendored config, so a checkpoint that sets it true is asking for behavior
//! no released implementation defines. That is rejected at load rather than
//! parsed and ignored.
//!
//! # Untrusted config
//!
//! `config.json` arrives from a third-party HuggingFace repo in the common
//! `mlxcel generate -m <org>/<repo>` flow, so [`ModelArgs::validate`] rejects
//! every scalar that could size an allocation, divide, truncate through an
//! `as i32` cast, or violate an undocumented precondition of an MLX C++ entry
//! point, and [`validate_weights`] rejects every tensor whose real shape
//! disagrees with the config, on both axes and on the quantized path too. An MLX
//! C++ exception crossing the cxx bridge is an uncatchable `std::terminate`, not
//! a Rust error, so a check that happens at the first forward pass is not a
//! check.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::models::gpt2::{dim_eq, validate_embedding_table};
use crate::models::switch_layers::{
    SwitchGLU, fused_moe_enabled, group_mask_scores, moe_weighted_sum,
};

// Configuration.

/// Bailing MoE `config.json`.
///
/// Field-for-field upstream's `ModelArgs`, plus the `quantization` block an
/// mlx-community conversion would add. Every routing knob is optional with the
/// upstream default, because `inclusionAI/Ling-lite-1.5` declares none of them:
/// `n_group`, `topk_group`, `score_function`, `routed_scaling_factor`,
/// `moe_router_enable_expert_bias`, `moe_router_enable_routed_scaling`,
/// `moe_shared_expert_intermediate_size`, `moe_router_enable_shared_expert`,
/// `use_qk_norm`, `partial_rotary_factor` and `rotary_dim` are all absent from
/// that file. Declaring any of them required would fail the whole parse.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,

    /// Absent means multi-head attention. Upstream requires the field; the
    /// `Option` only tolerates a config that omits it.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Routed-expert FFN width. Absent falls back to `intermediate_size`, which
    /// is what `Ling-lite-1.5` declares anyway (both are 1408).
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    /// `None` (or zero) means every layer is a plain dense MLP. Upstream gates
    /// its MoE block on `args.num_experts is not None`, so the `Option` is the
    /// faithful shape rather than a convenience.
    #[serde(default)]
    pub num_experts: Option<usize>,

    #[serde(default)]
    pub num_shared_experts: usize,

    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    /// Layers below this index use a dense MLP at `intermediate_size` instead of
    /// the sparse block. Zero on `Ling-lite-1.5`, so all 28 layers are MoE.
    #[serde(default)]
    pub first_k_dense_replace: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Upstream threads this into `initialize_rope`. No published Bailing
    /// checkpoint sets it (`Ling-lite-1.5` writes `null`), and this loader
    /// implements no scaled RoPE variant, so anything but the no-op form is
    /// rejected rather than silently ignored. See
    /// [`ModelArgs::validate_rope_scaling`].
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub use_bias: bool,

    #[serde(default)]
    pub use_qkv_bias: bool,

    /// Live upstream: L2-normalizes `lm_head.weight` along axis 0 at load. See
    /// [`normalize_lm_head_weight`].
    #[serde(default)]
    pub norm_head: bool,

    /// Dead everywhere. Rejected when true; see the module docs.
    #[serde(default)]
    pub norm_softmax: bool,

    #[serde(default)]
    pub use_qk_norm: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,

    /// Explicit rotary width. When absent the width is
    /// `int(head_dim * partial_rotary_factor)`.
    #[serde(default)]
    pub rotary_dim: Option<usize>,

    #[serde(default)]
    pub moe_router_enable_expert_bias: bool,

    /// Parsed for diagnostics only: upstream stores it and never reads it, and
    /// this port mirrors that. See the module docs.
    #[serde(default = "default_true")]
    pub moe_router_enable_routed_scaling: bool,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    /// `"softmax"` (the Bailing default) or `"sigmoid"` (DeepSeek-V3's default).
    /// Taking DeepSeek's gate as-is would change the routing distribution on
    /// every token while producing perfectly finite output.
    #[serde(default = "default_score_function")]
    pub score_function: String,

    #[serde(default = "default_n_group")]
    pub n_group: usize,

    #[serde(default = "default_topk_group")]
    pub topk_group: usize,

    /// Per-shared-expert width. `None` falls back to `moe_intermediate_size`.
    #[serde(default)]
    pub moe_shared_expert_intermediate_size: Option<usize>,

    #[serde(default = "default_true")]
    pub moe_router_enable_shared_expert: bool,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// A `config.json` token-id field, which may be a single int or a list of ints.
///
/// Same shape as the per-family enums in [`crate::models::gpt_neox`] and
/// [`crate::models::helium`]; serde fails the whole config when one field does
/// not match its declared type, so the list form has to be accepted even though
/// `Ling-lite-1.5` writes a single int.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenIdField {
    Single(i32),
    Multiple(Vec<i32>),
}

impl TokenIdField {
    fn ids(&self) -> Vec<i32> {
        match self {
            Self::Single(id) => vec![*id],
            Self::Multiple(ids) => ids.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "bailing_moe".to_string()
}
fn default_num_experts_per_tok() -> usize {
    1
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_max_position_embeddings() -> usize {
    32_768
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    600_000.0
}
fn default_partial_rotary_factor() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_score_function() -> String {
    "softmax".to_string()
}
fn default_n_group() -> usize {
    1
}
fn default_topk_group() -> usize {
    4
}

/// Which non-linearity turns router logits into routing probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFunction {
    /// The Bailing default.
    Softmax,
    /// DeepSeek-V3's default, reachable here only through an explicit config.
    Sigmoid,
}

/// Upper bounds on the architecture scalars a Bailing `config.json` may declare.
///
/// `config.json` is untrusted input: `mlxcel generate -m <org>/<repo>` downloads
/// a third-party HuggingFace repo and loads it in the same command, and the
/// download layer validates repo ids, filenames and transport but never parses
/// the file, so these fields arrive exactly as the checkpoint author wrote them.
///
/// Each ceiling sits orders of magnitude above `Ling-lite-1.5` (28 layers,
/// `hidden_size` 2048, 64 experts, `vocab_size` 126464). They exist so
/// `num_hidden_layers` and `num_experts` cannot size a `Vec::with_capacity` or a
/// per-expert weight probe loop, and so the `as i32` casts these values feed
/// (`head_dim`, the fused QKV width, `moe_intermediate_size`, the shared-expert
/// width) stay inside `i32` instead of truncating to a negative number.
const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_HIDDEN_SIZE: usize = 65_536;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_NUM_ATTENTION_HEADS: usize = 4096;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_MAX_POSITION_EMBEDDINGS: usize = 1 << 22;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_VOCAB_SIZE: usize = 1 << 24;
/// See [`MAX_NUM_HIDDEN_LAYERS`]. Also bounds the per-expert weight-probe loop
/// in [`validate_experts`], which runs once per projection per layer.
const MAX_NUM_EXPERTS: usize = 4096;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_NUM_SHARED_EXPERTS: usize = 1024;

impl ModelArgs {
    /// Head width. Upstream computes `hidden_size // num_attention_heads` and
    /// Bailing configs carry no `head_dim` field at all.
    ///
    /// Only valid after [`ModelArgs::validate`], which rejects
    /// `num_attention_heads == 0`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Output width of the fused `query_key_value` projection:
    /// `(num_attention_heads + 2 * num_key_value_heads) * head_dim`.
    ///
    /// 3072 on `Ling-lite-1.5`, which is `(16 + 2 * 4) * 128`.
    pub fn qkv_out_features(&self) -> usize {
        (self.num_attention_heads + 2 * self.num_kv_heads()) * self.head_dim()
    }

    /// Channel offsets that split the fused projection into Q, K and V:
    /// `(q_size, q_size + kv_size)`.
    ///
    /// Under GQA the K and V blocks are narrower than the Q block, so this is
    /// **not** an even three-way split. Splitting evenly would produce three
    /// tensors of the wrong widths, and MLX's `slice` clamps an out-of-range
    /// stop rather than throwing, so a too-wide read silently drops trailing
    /// channels instead of failing. Expressed as a pure function of the config
    /// so it can be asserted without a checkpoint.
    pub fn qkv_split_offsets(&self) -> (usize, usize) {
        let head_dim = self.head_dim();
        let q_size = self.num_attention_heads * head_dim;
        (q_size, q_size + self.num_kv_heads() * head_dim)
    }

    /// Channels per head that RoPE rotates: `rotary_dim` when the config gives
    /// one, `int(head_dim * partial_rotary_factor)` otherwise.
    ///
    /// The cast saturates rather than wrapping, so a NaN `partial_rotary_factor`
    /// becomes 0 and an infinite one becomes `i32::MAX`;
    /// [`ModelArgs::validate_rope`] rejects both, and everything else that is
    /// not an even value in `2..=head_dim`, before it can reach `fast_rope`.
    pub fn rope_dims(&self) -> i32 {
        match self.rotary_dim {
            Some(dims) => i32::try_from(dims).unwrap_or(i32::MAX),
            None => (self.partial_rotary_factor * self.head_dim() as f32) as i32,
        }
    }

    /// Routed-expert FFN width.
    pub fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// Number of routed experts, or zero when the config declares none.
    pub fn num_experts(&self) -> usize {
        self.num_experts.unwrap_or(0)
    }

    /// Whether the config describes any routed experts at all. Mirrors
    /// upstream's `args.num_experts is not None` guard.
    pub fn has_routed_experts(&self) -> bool {
        self.num_experts() > 0
    }

    /// Width of the single shared MLP: `shared_dim * num_shared_experts`, where
    /// `shared_dim` is `moe_shared_expert_intermediate_size or
    /// moe_intermediate_size`.
    ///
    /// This is **one wide MLP**, not `num_shared_experts` separate experts. On
    /// `Ling-lite-1.5` it is `1408 * 2 = 2816`, which the checkpoint confirms:
    /// `mlp.shared_experts.gate_proj.weight` is `[2816, 2048]`. There is no
    /// per-shared-expert axis anywhere, and it is never packed into the switch
    /// tensors.
    pub fn shared_expert_intermediate_size(&self) -> usize {
        let shared_dim = self
            .moe_shared_expert_intermediate_size
            .unwrap_or_else(|| self.moe_intermediate_size());
        shared_dim.saturating_mul(self.num_shared_experts)
    }

    /// Whether the sparse block builds the shared MLP at all.
    pub fn has_shared_expert(&self) -> bool {
        self.num_shared_experts > 0 && self.moe_router_enable_shared_expert
    }

    /// Whether layer `layer_idx` is sparse. Mirrors upstream's
    /// `args.num_experts is not None and layer_idx >= args.first_k_dense_replace`.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.has_routed_experts() && layer_idx >= self.first_k_dense_replace
    }

    /// Parsed [`ScoreFunction`]. Only valid after [`ModelArgs::validate`], which
    /// rejects any other spelling rather than falling back to a default.
    pub fn score_function(&self) -> ScoreFunction {
        match self.score_function.as_str() {
            "sigmoid" => ScoreFunction::Sigmoid,
            _ => ScoreFunction::Softmax,
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

    /// Stop tokens. `Ling-lite-1.5` declares `eos_token_id` 126081
    /// (`<|endoftext|>`), which is also its `pad_token_id`.
    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdField::ids)
            .unwrap_or_default()
    }

    /// Reject a `config.json` that cannot describe a real Bailing MoE, before
    /// any of its fields sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        // The zero checks come first. `0.is_multiple_of(0)` is true, so a config
        // with `hidden_size == num_attention_heads == 0` would pass the
        // divisibility check below and then divide by zero in `head_dim()`.
        if self.num_attention_heads == 0 || self.num_attention_heads > MAX_NUM_ATTENTION_HEADS {
            return Err(format!(
                "Bailing MoE num_attention_heads ({}) must be between 1 and \
                 {MAX_NUM_ATTENTION_HEADS}",
                self.num_attention_heads
            ));
        }
        if self.hidden_size == 0 || self.hidden_size > MAX_HIDDEN_SIZE {
            return Err(format!(
                "Bailing MoE hidden_size ({}) must be between 1 and {MAX_HIDDEN_SIZE}",
                self.hidden_size
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "Bailing MoE hidden_size ({}) must be divisible by num_attention_heads ({}); \
                 upstream derives head_dim as hidden_size // num_attention_heads",
                self.hidden_size, self.num_attention_heads
            ));
        }

        let num_kv_heads = self.num_kv_heads();
        if num_kv_heads == 0 || num_kv_heads > self.num_attention_heads {
            return Err(format!(
                "Bailing MoE num_key_value_heads ({num_kv_heads}) must be between 1 and \
                 num_attention_heads ({})",
                self.num_attention_heads
            ));
        }
        if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(format!(
                "Bailing MoE num_attention_heads ({}) must be divisible by num_key_value_heads \
                 ({num_kv_heads}) for grouped-query attention",
                self.num_attention_heads
            ));
        }

        if self.num_hidden_layers == 0 || self.num_hidden_layers > MAX_NUM_HIDDEN_LAYERS {
            return Err(format!(
                "Bailing MoE num_hidden_layers ({}) must be between 1 and {MAX_NUM_HIDDEN_LAYERS}",
                self.num_hidden_layers
            ));
        }
        if self.intermediate_size == 0 || self.intermediate_size > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Bailing MoE intermediate_size ({}) must be between 1 and {MAX_INTERMEDIATE_SIZE}",
                self.intermediate_size
            ));
        }
        let moe_intermediate = self.moe_intermediate_size();
        if moe_intermediate == 0 || moe_intermediate > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Bailing MoE moe_intermediate_size ({moe_intermediate}) must be between 1 and \
                 {MAX_INTERMEDIATE_SIZE}"
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(format!(
                "Bailing MoE vocab_size ({}) must be between 1 and {MAX_VOCAB_SIZE}",
                self.vocab_size
            ));
        }
        if self.max_position_embeddings == 0
            || self.max_position_embeddings > MAX_MAX_POSITION_EMBEDDINGS
        {
            return Err(format!(
                "Bailing MoE max_position_embeddings ({}) must be between 1 and \
                 {MAX_MAX_POSITION_EMBEDDINGS}",
                self.max_position_embeddings
            ));
        }
        // A dense prefix longer than the stack means the config and the
        // checkpoint disagree about which layers carry experts at all.
        if self.first_k_dense_replace > self.num_hidden_layers {
            return Err(format!(
                "Bailing MoE first_k_dense_replace ({}) must not exceed num_hidden_layers ({})",
                self.first_k_dense_replace, self.num_hidden_layers
            ));
        }

        self.validate_routing()?;
        self.validate_shared_expert()?;
        self.validate_rope()?;
        self.validate_rope_scaling()?;
        self.validate_norm_eps()?;
        self.validate_vendor_flags()?;
        self.validate_quantization()
    }

    /// Reject routing parameters that would index out of range inside MLX.
    ///
    /// Two of these guard against inheriting an upstream bug rather than against
    /// a hostile config. `group_expert_select` computes `k = n_group -
    /// topk_group` and calls `argpartition(kth = k - 1)`, which is out of range
    /// the moment `topk_group >= n_group`, and it reshapes the score row into
    /// `n_group` groups without checking that `num_experts` divides evenly.
    /// Neither is reachable on `Ling-lite-1.5`, where `n_group` defaults to 1 and
    /// the whole branch is gated on `n_group > 1`, but both are reachable from a
    /// config file. MLX signals an out-of-range `argpartition` by throwing, and
    /// an MLX C++ exception crossing the cxx bridge is an uncatchable
    /// `std::terminate` at the first forward pass rather than a load error.
    ///
    /// The group-scoring step additionally takes the top **two** experts of each
    /// group, so a group of one expert cannot be scored at all.
    fn validate_routing(&self) -> Result<(), String> {
        if !self.has_routed_experts() {
            return Ok(());
        }
        let num_experts = self.num_experts();
        if num_experts > MAX_NUM_EXPERTS {
            return Err(format!(
                "Bailing MoE num_experts ({num_experts}) must be between 1 and {MAX_NUM_EXPERTS}"
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > num_experts {
            return Err(format!(
                "Bailing MoE num_experts_per_tok ({}) must be between 1 and num_experts \
                 ({num_experts}); the router selects that many indices out of a row of \
                 num_experts scores",
                self.num_experts_per_tok
            ));
        }
        if !matches!(self.score_function.as_str(), "softmax" | "sigmoid") {
            return Err(format!(
                "Bailing MoE score_function ({:?}) must be \"softmax\" (the Bailing default) or \
                 \"sigmoid\"; an unrecognized value is rejected rather than silently falling back, \
                 because either fallback changes the routing distribution on every token while \
                 leaving the output finite and plausible",
                self.score_function
            ));
        }
        if self.n_group == 0 {
            return Err(
                "Bailing MoE n_group must be at least 1 (1 disables grouped routing)".to_string(),
            );
        }
        if self.n_group > 1 {
            if !num_experts.is_multiple_of(self.n_group) {
                return Err(format!(
                    "Bailing MoE num_experts ({num_experts}) must be divisible by n_group ({}); \
                     grouped routing reshapes the score row into n_group equal groups",
                    self.n_group
                ));
            }
            let experts_per_group = num_experts / self.n_group;
            if experts_per_group < 2 {
                return Err(format!(
                    "Bailing MoE n_group ({}) leaves {experts_per_group} expert(s) per group; \
                     grouped routing scores each group by the sum of its top two experts, so a \
                     group must hold at least 2",
                    self.n_group
                ));
            }
            if self.topk_group == 0 || self.topk_group >= self.n_group {
                return Err(format!(
                    "Bailing MoE topk_group ({}) must be between 1 and n_group - 1 ({}); the \
                     grouped-routing step zeroes the bottom n_group - topk_group groups via \
                     argpartition(kth = n_group - topk_group - 1), which is out of range as soon \
                     as topk_group reaches n_group, and an MLX C++ exception crossing the cxx \
                     bridge is an uncatchable abort at the first forward pass rather than a load \
                     error",
                    self.topk_group,
                    self.n_group - 1
                ));
            }
        }
        if !self.routed_scaling_factor.is_finite() {
            return Err(format!(
                "Bailing MoE routed_scaling_factor ({}) must be finite; it multiplies every routed \
                 expert weight, so a non-finite value makes every MoE output NaN and that NaN \
                 reaches the logits without anything throwing",
                self.routed_scaling_factor
            ));
        }
        Ok(())
    }

    /// Reject a shared-expert width that cannot be built.
    fn validate_shared_expert(&self) -> Result<(), String> {
        if self.num_shared_experts > MAX_NUM_SHARED_EXPERTS {
            return Err(format!(
                "Bailing MoE num_shared_experts ({}) must not exceed {MAX_NUM_SHARED_EXPERTS}",
                self.num_shared_experts
            ));
        }
        if !self.has_shared_expert() {
            return Ok(());
        }
        let shared_dim = self
            .moe_shared_expert_intermediate_size
            .unwrap_or_else(|| self.moe_intermediate_size());
        let width = shared_dim
            .checked_mul(self.num_shared_experts)
            .ok_or_else(|| {
                format!(
                    "Bailing MoE shared-expert width overflows: \
                     moe_shared_expert_intermediate_size ({shared_dim}) * num_shared_experts ({})",
                    self.num_shared_experts
                )
            })?;
        if width == 0 || width > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Bailing MoE shared-expert width ({width}, from a per-expert width of \
                 {shared_dim} times num_shared_experts {}) must be between 1 and \
                 {MAX_INTERMEDIATE_SIZE}",
                self.num_shared_experts
            ));
        }
        Ok(())
    }

    /// Reject RoPE parameters that MLX would throw on, or that would silently
    /// NaN every rotated channel.
    ///
    /// `mlx::core::fast::rope` requires `dims` to be positive, even, and no
    /// larger than the input's last axis, and enforces that by throwing
    /// `std::invalid_argument`. `fast_rope` crosses the cxx bridge as
    /// `UniquePtr<MlxArray>` rather than a `Result`, so that throw is an
    /// uncatchable `std::terminate` at the **first forward pass**, long after the
    /// checkpoint appeared to load cleanly.
    ///
    /// `partial_rotary_factor` is a float, which widens what a config can
    /// express beyond the integer fields: the `as i32` cast in
    /// [`ModelArgs::rope_dims`] saturates, so NaN becomes 0 and an infinity
    /// becomes `i32::MAX`. Both are caught by the range check below, but the
    /// non-finite case is rejected explicitly so the message names the field.
    fn validate_rope(&self) -> Result<(), String> {
        if self.rotary_dim.is_none() && !self.partial_rotary_factor.is_finite() {
            return Err(format!(
                "Bailing MoE partial_rotary_factor ({}) must be a finite number",
                self.partial_rotary_factor
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!(
                "Bailing MoE rope_theta ({}) must be a finite positive number; RoPE exponentiates \
                 it per channel, so a zero, negative or non-finite base makes every rotated \
                 channel NaN and that NaN reaches the logits without anything throwing",
                self.rope_theta
            ));
        }
        let head_dim = self.head_dim();
        let rope_dims = self.rope_dims();
        // A negative `rope_dims` fails the conversion; folding it to zero puts it
        // in the same arm as a zero one, which MLX refuses for the same reason.
        let dims = usize::try_from(rope_dims).unwrap_or(0);
        if dims == 0 || dims > head_dim {
            return Err(format!(
                "Bailing MoE rotary width resolves to {rope_dims} for a head of width {head_dim}; \
                 it must be an even number between 2 and {head_dim}. MLX throws on a rope `dims` \
                 outside that range, and an MLX C++ exception crossing the cxx bridge is an \
                 uncatchable abort at the first forward pass rather than a load error. Check \
                 rotary_dim ({:?}) and partial_rotary_factor ({}).",
                self.rotary_dim, self.partial_rotary_factor
            ));
        }
        if !dims.is_multiple_of(2) {
            return Err(format!(
                "Bailing MoE rotary width resolves to an odd {rope_dims} for a head of width \
                 {head_dim}; RoPE rotates channel pairs, so the rope `dims` must be even, and MLX \
                 throws on an odd one. Check rotary_dim ({:?}) and partial_rotary_factor ({}).",
                self.rotary_dim, self.partial_rotary_factor
            ));
        }
        Ok(())
    }

    /// Reject a `rope_scaling` block this loader does not implement.
    ///
    /// Upstream passes the block to `initialize_rope`, which builds a scaled
    /// RoPE variant from it. No published Bailing checkpoint sets one
    /// (`Ling-lite-1.5` writes `null`), and this loader always builds the plain
    /// rotation, so accepting a non-trivial block would silently place every
    /// token at the wrong position. Only the no-op spelling is accepted.
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
            "Bailing MoE rope_scaling ({scaling:?}) is not implemented for this family; upstream \
             threads it into initialize_rope, but this loader always builds the plain rotation, so \
             accepting a scaled block would place every token at the wrong position while the \
             model still loaded and still generated fluent text. Only an absent, empty or \
             \"default\" block is accepted."
        ))
    }

    /// Reject an `rms_norm_eps` that would NaN every hidden state.
    ///
    /// `fast::rms_norm` computes `x * weight * rsqrt(mean(x^2) + eps)` and never
    /// inspects `eps`, so a non-finite, negative or zero value produces NaN
    /// hidden states with no error at all: the checkpoint loads, generation runs,
    /// and the output is uniform garbage.
    fn validate_norm_eps(&self) -> Result<(), String> {
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "Bailing MoE rms_norm_eps ({}) must be a finite positive number; it is added to \
                 the mean square under an rsqrt, so a non-finite, negative or zero value makes \
                 every normalized hidden state NaN and that NaN reaches the logits without \
                 anything throwing",
                self.rms_norm_eps
            ));
        }
        Ok(())
    }

    /// Reject `norm_softmax: true`.
    ///
    /// mlx-lm declares the field in `ModelArgs` and never reads it, it does not
    /// appear anywhere in the vendored `modeling_bailing_moe.py`, and
    /// `configuration_bailing_moe.py` does not even name it as a parameter, so it
    /// survives only through `**kwargs`. A checkpoint that sets it true is asking
    /// for behavior no released implementation defines, and silently ignoring it
    /// would produce output that is wrong in a way the checkpoint author asked us
    /// not to produce.
    fn validate_vendor_flags(&self) -> Result<(), String> {
        if self.norm_softmax {
            return Err(
                "Bailing MoE config sets norm_softmax: true, which no released implementation \
                 defines. mlx-lm declares the field and never reads it, the vendored \
                 modeling_bailing_moe.py does not mention it, and configuration_bailing_moe.py \
                 does not name it as a parameter. Loading anyway would silently ignore a flag the \
                 checkpoint author set on purpose."
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Reject a `quantization` block that would abort an MLX kernel.
    ///
    /// Kept as a family-level early diagnostic even though the shared loaders now
    /// enforce the same bound (issue #929): failing here names Bailing MoE and
    /// fires during `ModelArgs::validate`, before any tensor is touched, rather
    /// than at the first quantized projection the loader happens to reach.
    /// [`mlxcel_core::layers::validate_quantization_params`] carries the rationale
    /// for the bound being a range rather than an allowlist.
    fn validate_quantization(&self) -> Result<(), String> {
        let Some(quantization) = self.quantization.as_ref() else {
            return Ok(());
        };
        mlxcel_core::layers::validate_quantization_params(
            quantization.group_size,
            quantization.bits,
        )
        .map_err(|e| format!("Bailing MoE config.json: {e}"))
    }

    /// Whether this config is one where mirroring upstream's unconditional
    /// routed scaling is observably different from honoring the flag.
    ///
    /// Upstream stores `moe_router_enable_routed_scaling` and never reads it, so
    /// the multiply is unconditional. That is what this port does, so a Bailing
    /// checkpoint decodes here exactly as it decodes under the reference. The two
    /// readings coincide whenever the factor is 1.0, which is its default and the
    /// only value any published checkpoint uses; they diverge only for a non-unit
    /// factor with the flag off, which this predicate identifies so the loader can
    /// say so out loud instead of leaving the choice invisible.
    pub fn routed_scaling_flag_is_ignored_observably(&self) -> bool {
        self.has_routed_experts()
            && !self.moe_router_enable_routed_scaling
            && self.routed_scaling_factor != 1.0
    }

    /// Print the diagnostic for [`ModelArgs::routed_scaling_flag_is_ignored_observably`].
    ///
    /// `eprintln!` rather than `tracing::warn!`: only `mlxcel-server` installs a
    /// tracing subscriber, so a `tracing` event is a no-op on the CLI path this
    /// is reachable from.
    fn warn_on_ignored_routed_scaling_flag(&self) {
        if self.routed_scaling_flag_is_ignored_observably() {
            eprintln!(
                "Bailing MoE config sets moe_router_enable_routed_scaling: false with \
                 routed_scaling_factor {}. Upstream mlx-lm stores that flag and never reads it, so \
                 the scaling multiply is unconditional; this loader mirrors upstream and applies \
                 the factor anyway. Routed expert weights will be scaled by {}.",
                self.routed_scaling_factor, self.routed_scaling_factor
            );
        }
    }
}

// Weight-shape validation.

/// Reconstruct the quantization mode `UnifiedLinear::from_weights` would pick.
///
/// Affine stores zero-point `biases`, so their absence means a block-float
/// scheme distinguished by bits and group size. Mirrored here rather than
/// assumed, because the reconciliation below behaves differently per mode.
fn quant_mode(weights: &WeightMap, prefix: &str, group_size: i32, bits: i32) -> &'static str {
    let has_biases = weights.contains_key(&format!("{prefix}.biases"));
    mlxcel_core::layers::infer_quantization_mode(has_biases, group_size, bits)
}

/// Check a quantized tensor's packing against the input width `config.json`
/// claims, and its `biases` against its `scales`.
///
/// Thin `WeightMap` adapter over
/// [`mlxcel_core::layers::validate_quantized_packing`], which carries the
/// reasoning and the rejection messages. This wrapper resolves the three
/// tensors by key, skips silently when the tensor is not quantized, and picks
/// the mode the loader will pick.
///
/// The `mode` fed into the shared reconciliation is `quant_mode`'s guess from
/// `.biases` presence, and that guess matches the loader exactly for every
/// plain projection here (`UnifiedLinear::from_weights` picks its mode through
/// the same `infer_quantization_mode`). It does **not** match for the routed
/// experts: `SwitchLinear::from_weights`
/// (https://github.com/lablup/mlxcel/blob/main/src/models/switch_layers.rs) pins
/// `mode` to `"affine"` unconditionally, regardless of whether `.biases` is
/// present, so a stacked or per-expert tensor quantized in a block-float scheme
/// without zero-point `biases` would be reconciled here under the wrong mode.
/// `validate_experts` inherits that gap.
fn validate_quantized_packing(
    weights: &WeightMap,
    prefix: &str,
    in_features: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let Some(scales) = weights.get(&format!("{prefix}.scales")) else {
        return Ok(());
    };
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight (but {prefix}.scales exists)"))?;
    let w_shape = mlxcel_core::array_shape(weight);
    let s_shape = mlxcel_core::array_shape(scales);
    let bias_shape = weights
        .get(&format!("{prefix}.biases"))
        .map(|biases| mlxcel_core::array_shape(biases));

    mlxcel_core::layers::validate_quantized_packing(
        prefix,
        &mlxcel_core::layers::QuantizedTensorShapes {
            weight: &w_shape,
            scales: &s_shape,
            biases: bias_shape.as_deref(),
        },
        in_features,
        group_size,
        bits,
        quant_mode(weights, prefix, group_size, bits),
    )
}

/// Check one `[out_features, in_features]` projection against the config.
///
/// A quantized weight is packed along the input axis only, so its row count is
/// still `out_features` and is checked on the same path as a float weight. The
/// input axis is checked too, through the scales rather than the packed
/// `.weight`, because the packed width alone does not fix a width without a bit
/// count. See [`validate_quantized_packing`].
fn validate_projection(
    weights: &WeightMap,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let shape = mlxcel_core::array_shape(weight);
    if shape.len() != 2 {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected a 2-D [out, in] projection"
        ));
    }
    let quantized = weights.contains_key(&format!("{prefix}.scales"));

    if !dim_eq(shape[0], out_features) {
        let looks_transposed = !quantized && dim_eq(shape[0], in_features);
        let hint = if looks_transposed {
            " That is the [in, out] orientation; Bailing builds every projection with nn.Linear, \
             so a genuine checkpoint is already [out, in] and must not be transposed."
        } else {
            ""
        };
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected {out_features} rows.{hint}"
        ));
    }
    if !quantized && !dim_eq(shape[1], in_features) {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected [{out_features}, {in_features}]"
        ));
    }
    validate_quantized_packing(weights, prefix, in_features, group_size, bits)?;

    if let Some(bias) = weights.get(&format!("{prefix}.bias")) {
        let bias_shape = mlxcel_core::array_shape(bias);
        if bias_shape.len() != 1 || !dim_eq(bias_shape[0], out_features) {
            return Err(format!(
                "unexpected {prefix}.bias shape {bias_shape:?}: expected [{out_features}]"
            ));
        }
    }
    Ok(())
}

/// Check a 1-D RMSNorm weight.
fn validate_norm(weights: &WeightMap, key: &str, dim: usize) -> Result<(), String> {
    let weight = weights
        .get(key)
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    let shape = mlxcel_core::array_shape(weight);
    if shape.len() != 1 || !dim_eq(shape[0], dim) {
        return Err(format!(
            "unexpected {key} shape {shape:?}: expected [{dim}]"
        ));
    }
    Ok(())
}

/// Check a pre-stacked `[num_experts, out_features, in_features]` expert tensor.
///
/// The leading axis is the gather axis of `gather_mm` / `gather_qmm`, and the
/// router can emit any index below `num_experts`. MLX's gather adds the axis size
/// to a negative index but performs no range check on a positive one, so a
/// stacked tensor with fewer planes than the config claims turns an ordinary
/// token into an out-of-bounds read whose result reaches the logits. More planes
/// than claimed is accepted: the router can never reach them.
fn validate_stacked_experts(
    weights: &WeightMap,
    prefix: &str,
    num_experts: usize,
    out_features: usize,
    in_features: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let shape = mlxcel_core::array_shape(weight);
    if shape.len() != 3 {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected a 3-D [num_experts, out, in] \
             stacked expert tensor"
        ));
    }
    let planes = usize::try_from(shape[0]).unwrap_or(0);
    if planes < num_experts {
        return Err(format!(
            "config num_experts ({num_experts}) exceeds the {planes} expert planes present in \
             {weight_name}. The router selects indices below num_experts and the gather behind \
             gather_mm / gather_qmm does not range-check a positive index, so the missing planes \
             would be read out of bounds and the result would reach the logits."
        ));
    }
    if !dim_eq(shape[1], out_features) {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected {out_features} rows per expert"
        ));
    }
    let quantized = weights.contains_key(&format!("{prefix}.scales"));
    if !quantized && !dim_eq(shape[2], in_features) {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected [{planes}, {out_features}, \
             {in_features}]"
        ));
    }
    validate_quantized_packing(weights, prefix, in_features, group_size, bits)
}

/// Check the routed experts of one MoE layer, in whichever of the two layouts
/// the checkpoint uses.
///
/// A raw `inclusionAI` checkpoint stores every expert separately at
/// `mlp.experts.{e}.{gate,up,down}_proj`, which is 64 * 3 tensors per layer and
/// 5376 across `Ling-lite-1.5`. `SwitchLinear` joins those into the stacked
/// layout at load. An mlx-lm conversion has already joined them into
/// `mlp.switch_mlp.{proj}`. Both are checked here, and the per-expert form is
/// checked for **every** index below `num_experts` rather than only index 0,
/// because `stack_individual_experts` gathers contiguously from 0 until the first
/// gap and would otherwise register a short stack that the router can index past.
fn validate_experts(weights: &WeightMap, args: &ModelArgs, mlp_prefix: &str) -> Result<(), String> {
    let num_experts = args.num_experts();
    let hidden = args.hidden_size;
    let moe_intermediate = args.moe_intermediate_size();
    let group_size = args.group_size();
    let bits = args.bits();

    for (leaf, out_features, in_features) in [
        ("gate_proj", moe_intermediate, hidden),
        ("up_proj", moe_intermediate, hidden),
        ("down_proj", hidden, moe_intermediate),
    ] {
        let stacked = format!("{mlp_prefix}.switch_mlp.{leaf}");
        if weights.contains_key(&format!("{stacked}.weight")) {
            validate_stacked_experts(
                weights,
                &stacked,
                num_experts,
                out_features,
                in_features,
                group_size,
                bits,
            )?;
            continue;
        }
        for expert in 0..num_experts {
            validate_projection(
                weights,
                &format!("{mlp_prefix}.experts.{expert}.{leaf}"),
                out_features,
                in_features,
                group_size,
                bits,
            )?;
        }
    }
    Ok(())
}

/// Resolve the router projection prefix for one MoE layer.
///
/// **This is the router-rename collision, handled by construction.** A raw
/// checkpoint stores the router at `mlp.gate.weight`; upstream's `sanitize`
/// renames it to `mlp.gate.gate_proj.weight`, which ends in the same `gate_proj`
/// segment every routed expert's SwiGLU gate uses. Both keys are probed by exact
/// name, so nothing here can match an expert tensor: the expert keys all carry an
/// `.experts.{idx}.` segment that neither candidate contains. The converted
/// spelling is probed first because it is the more specific of the two, and the
/// two can never both be present.
///
/// Returns the prefix without the `.weight` suffix, ready for
/// `UnifiedLinear::from_weights`.
pub fn router_prefix(weights: &WeightMap, mlp_prefix: &str) -> Result<String, String> {
    let converted = format!("{mlp_prefix}.gate.gate_proj");
    if weights.contains_key(&format!("{converted}.weight")) {
        return Ok(converted);
    }
    let raw = format!("{mlp_prefix}.gate");
    if weights.contains_key(&format!("{raw}.weight")) {
        return Ok(raw);
    }
    Err(format!(
        "Bailing MoE router weight not found: expected {mlp_prefix}.gate.weight (raw checkpoint) \
         or {mlp_prefix}.gate.gate_proj.weight (mlx-lm conversion, which renames the router into \
         the gate_proj spelling the routed experts also use)"
    ))
}

/// Reject a checkpoint whose real tensor shapes disagree with `config.json`,
/// before any of them reaches MLX.
///
/// This has to run **before** the model is built, not after. The fused
/// `query_key_value` output is split at config-derived offsets and reshaped with
/// config-derived head counts, and MLX's `slice` clamps an out-of-range stop
/// rather than throwing, so a too-narrow projection silently yields a short V
/// block and the reshape aborts the process instead of returning an error.
pub fn validate_weights(weights: &WeightMap, args: &ModelArgs) -> Result<(), String> {
    let hidden = args.hidden_size;
    let head_dim = args.head_dim();
    let q_size = args.num_attention_heads * head_dim;
    let group_size = args.group_size();
    let bits = args.bits();

    validate_norm(weights, "model.norm.weight", hidden)?;

    // The token table's row count is checked against `vocab_size` by
    // `validate_embedding_table` in `from_weights`, which since issue #929 also
    // reconstructs the dequantized width through the same shared check this line
    // calls. Kept as a redundant early diagnostic: it runs during
    // `validate_weights`, ahead of the loader, and names the table rather than
    // the constructor that reached it. A no-op for an unquantized table.
    validate_quantized_packing(weights, "model.word_embeddings", hidden, group_size, bits)?;

    // The output head, which nothing else checks. The axis that aborts the
    // process is not the row count: rows only bound an argmax over the logits.
    // It is the input width, the inner dimension of the matmul that produces
    // those logits. MLX throws `std::invalid_argument` when it disagrees with
    // the hidden state, and `matmul` and `quantized_matmul` cross the cxx bridge
    // as `UniquePtr<MlxArray>` rather than a `Result`, so that throw is an
    // uncatchable `std::terminate` at the first forward pass rather than a load
    // error. The row count is checked exactly because upstream loads this tensor
    // into an `nn.Linear(hidden_size, vocab_size)` under a strict
    // `load_weights`, so an inexact head is not a checkpoint the reference
    // accepts either.
    if !args.tie_word_embeddings {
        validate_projection(
            weights,
            "lm_head",
            args.vocab_size,
            hidden,
            group_size,
            bits,
        )?;
    }

    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        let attention = format!("{prefix}.attention");

        validate_projection(
            weights,
            &format!("{attention}.query_key_value"),
            args.qkv_out_features(),
            hidden,
            group_size,
            bits,
        )?;
        validate_projection(
            weights,
            &format!("{attention}.dense"),
            hidden,
            q_size,
            group_size,
            bits,
        )?;
        if args.use_qk_norm {
            validate_norm(
                weights,
                &format!("{attention}.query_layernorm.weight"),
                head_dim,
            )?;
            validate_norm(
                weights,
                &format!("{attention}.key_layernorm.weight"),
                head_dim,
            )?;
        }
        validate_norm(weights, &format!("{prefix}.input_layernorm.weight"), hidden)?;
        validate_norm(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
        )?;

        let mlp = format!("{prefix}.mlp");
        if args.is_moe_layer(layer) {
            let router = router_prefix(weights, &mlp)?;
            validate_projection(
                weights,
                &router,
                args.num_experts(),
                hidden,
                group_size,
                bits,
            )?;
            if args.moe_router_enable_expert_bias
                && let Some(bias) = weights.get(&format!("{mlp}.gate.expert_bias"))
            {
                let bias_shape = mlxcel_core::array_shape(bias);
                if bias_shape.len() != 1 || !dim_eq(bias_shape[0], args.num_experts()) {
                    return Err(format!(
                        "unexpected {mlp}.gate.expert_bias shape {bias_shape:?}: expected [{}]; it \
                         is added to a row of that many router scores",
                        args.num_experts()
                    ));
                }
            }
            validate_experts(weights, args, &mlp)?;
            if args.has_shared_expert() {
                validate_mlp(
                    weights,
                    &format!("{mlp}.shared_experts"),
                    hidden,
                    args.shared_expert_intermediate_size(),
                    group_size,
                    bits,
                )?;
            }
        } else {
            validate_mlp(
                weights,
                &mlp,
                hidden,
                args.intermediate_size,
                group_size,
                bits,
            )?;
        }
    }
    Ok(())
}

/// Check the three projections of a dense SwiGLU MLP.
fn validate_mlp(
    weights: &WeightMap,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    validate_projection(
        weights,
        &format!("{prefix}.gate_proj"),
        intermediate,
        hidden,
        group_size,
        bits,
    )?;
    validate_projection(
        weights,
        &format!("{prefix}.up_proj"),
        intermediate,
        hidden,
        group_size,
        bits,
    )?;
    validate_projection(
        weights,
        &format!("{prefix}.down_proj"),
        hidden,
        intermediate,
        group_size,
        bits,
    )
}

// lm_head normalization (`norm_head`).

/// L2-normalize an `lm_head` weight along axis 0, upstream's `norm_head`.
///
/// Upstream's `sanitize` computes
/// `w / (linalg.norm(w.astype(float32), axis=0, keepdims=True) + 1e-7)` and casts
/// the result back to the weight's own dtype; the vendored
/// `modeling_bailing_moe.py` does the same once at inference and then clears the
/// flag. Axis 0 of an `[vocab_size, hidden_size]` head is the vocabulary axis, so
/// this scales each hidden channel by the reciprocal of its norm across the whole
/// vocabulary. That is unusual, and it is exactly what both reference
/// implementations do.
///
/// The epsilon is added to the norm, not to the squared sum, so it cannot be
/// folded into the `sqrt`.
pub fn normalize_lm_head_weight(weight: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(weight);
    let w32 = mlxcel_core::astype(weight, mlxcel_core::dtype::FLOAT32);
    let squared = mlxcel_core::square(&w32);
    let summed = mlxcel_core::sum_axis(&squared, 0, true);
    let norm = mlxcel_core::sqrt(&summed);
    let eps = mlxcel_core::full_f32(&[1], 1e-7, mlxcel_core::dtype::FLOAT32);
    let norm = mlxcel_core::add(&norm, &eps);
    let normalized = mlxcel_core::divide(&w32, &norm);
    mlxcel_core::astype(&normalized, dtype)
}

/// Build the output head, applying `norm_head` when the config asks for it.
///
/// A quantized head is refused under `norm_head`: the stored `.weight` is a
/// packed `uint32` bit field, so dividing it by a column norm is not the
/// normalization upstream performs, and doing it anyway would corrupt every
/// logit while leaving the checkpoint apparently loadable. No published Bailing
/// checkpoint combines the two (`norm_head` is false everywhere), so refusing
/// costs nothing a real checkpoint needs.
fn load_lm_head(weights: &WeightMap, args: &ModelArgs) -> Result<UnifiedLinear, String> {
    let group_size = args.group_size();
    let bits = args.bits();
    if !args.norm_head {
        return UnifiedLinear::from_weights(weights, "lm_head", group_size, bits);
    }
    if weights.contains_key("lm_head.scales") {
        return Err(
            "Bailing MoE config sets norm_head: true, but lm_head is quantized. norm_head \
             L2-normalizes the raw lm_head weight along axis 0, and the stored tensor is a packed \
             bit field rather than the weight itself, so applying the normalization to it would \
             corrupt every logit. Dequantize the head or clear norm_head."
                .to_string(),
        );
    }
    let weight = weights
        .get("lm_head.weight")
        .ok_or_else(|| "Weight not found: lm_head.weight".to_string())?;
    let mut normalized = WeightMap::new();
    normalized.insert(
        "lm_head.weight".to_string(),
        normalize_lm_head_weight(weight),
    );
    if let Some(bias) = weights.get("lm_head.bias") {
        normalized.insert("lm_head.bias".to_string(), mlxcel_core::copy(bias));
    }
    UnifiedLinear::from_weights(&normalized, "lm_head", group_size, bits)
}

// Router.

/// The Bailing router, upstream's `BailingMoeGate` plus `group_expert_select`.
pub struct BailingMoeGate {
    pub gate_proj: UnifiedLinear,
    /// Router correction bias, present only when
    /// `moe_router_enable_expert_bias` is set. Applied to the **selection**
    /// scores only.
    pub expert_bias: Option<UniquePtr<MlxArray>>,
    pub top_k: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub score_function: ScoreFunction,
}

impl BailingMoeGate {
    /// Returns `(expert_indices, expert_weights)` for a `[n_tokens, hidden]`
    /// input.
    ///
    /// The ordering below is upstream's, and three steps of it are load-bearing
    /// in ways nothing downstream can detect:
    ///
    /// 1. **The bias is applied to the selection copy only.** `orig_scores` is
    ///    captured before the add, the top-k indices come from the biased copy,
    ///    and the returned weights are gathered from `orig_scores`. Gathering
    ///    from the biased scores instead leaves the output finite and plausible
    ///    while misweighting every routed contribution.
    /// 2. **`norm_topk_prob` is conditional and has an epsilon.** Upstream
    ///    normalizes only when `top_k > 1`, and divides by
    ///    `scores.sum(-1) + 1e-20` rather than a bare sum.
    /// 3. **The whole computation runs in float32** and is cast back to the
    ///    router logits' own dtype only at the very end, after normalization and
    ///    scaling.
    ///
    /// The routed-scaling multiply is unconditional, mirroring upstream. See the
    /// module docs for why the config flag does not gate it.
    pub fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let gates = self.gate_proj.forward(x);
        let in_type = mlxcel_core::array_dtype(&gates);
        let gates = mlxcel_core::astype(&gates, mlxcel_core::dtype::FLOAT32);

        let orig_scores = match self.score_function {
            ScoreFunction::Sigmoid => mlxcel_core::sigmoid(&gates),
            ScoreFunction::Softmax => mlxcel_core::softmax(&gates, -1),
        };

        // Selection scores: the correction bias and the grouped-routing mask
        // both act HERE ONLY. `orig_scores` is untouched from this point on.
        let selection = match &self.expert_bias {
            Some(bias) => mlxcel_core::add(&orig_scores, bias),
            None => mlxcel_core::copy(&orig_scores),
        };
        let selection = if self.n_group > 1 {
            group_mask_scores(&selection, self.n_group, self.topk_group)
        } else {
            selection
        };

        // Top-k over the selection scores, mirroring upstream's
        // `argpartition(scores, kth=-k)[..., -k:]`. The expert count comes from
        // the real score row rather than from the config, so the partition pivot
        // is in range even if the two ever disagree.
        let selection_shape = mlxcel_core::array_shape(&selection);
        let n_experts = *selection_shape.last().unwrap_or(&0);
        let kth = (n_experts - self.top_k).max(0);
        let order = mlxcel_core::argpartition(&selection, kth, -1);
        let indices = slice_axis(&order, -1, kth, -1);

        // Weights from the UNBIASED, unmasked scores.
        let scores = mlxcel_core::take_along_axis(&orig_scores, &indices, -1);

        let scores = if self.top_k > 1 && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&scores, -1, true);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::dtype::FLOAT32);
            let denominator = mlxcel_core::add(&sum, &eps);
            mlxcel_core::divide(&scores, &denominator)
        } else {
            scores
        };

        let scale = mlxcel_core::full_f32(
            &[1],
            self.routed_scaling_factor,
            mlxcel_core::dtype::FLOAT32,
        );
        let scores = mlxcel_core::multiply(&scores, &scale);

        (indices, mlxcel_core::astype(&scores, in_type))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        mlp_prefix: &str,
    ) -> Result<Self, String> {
        let prefix = router_prefix(weights, mlp_prefix)?;
        let gate_proj =
            UnifiedLinear::from_weights(weights, &prefix, args.group_size(), args.bits())?;

        // Upstream initializes `expert_bias` to zeros when the flag is set, so a
        // checkpoint that carries no tensor still routes with a zero bias rather
        // than failing to load. The key is unaffected by `sanitize`'s router
        // rename, which touches only `.weight` and `.bias`.
        let expert_bias = if args.moe_router_enable_expert_bias {
            let key = format!("{mlp_prefix}.gate.expert_bias");
            Some(match weights.get(&key) {
                Some(bias) => mlxcel_core::astype(bias, mlxcel_core::dtype::FLOAT32),
                None => mlxcel_core::full_f32(
                    &[args.num_experts() as i32],
                    0.0,
                    mlxcel_core::dtype::FLOAT32,
                ),
            })
        } else {
            None
        };

        Ok(Self {
            gate_proj,
            expert_bias,
            top_k: args.num_experts_per_tok as i32,
            n_group: args.n_group as i32,
            // `validate_routing` bounds `topk_group` to `1..=n_group - 1` only
            // on the grouped branch. With `n_group == 1`, the upstream default
            // and the value `Ling-lite-1.5` inherits, the field is never read
            // and never bounded, so a plain `as i32` would truncate an absurd
            // config value to a negative number and store it. Saturate instead,
            // the same way `ModelArgs::rope_dims` handles an absurd
            // `rotary_dim`: an unreachable field must still not hold a value
            // that would be out of range if a later change did reach it.
            topk_group: i32::try_from(args.topk_group).unwrap_or(i32::MAX),
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
            score_function: args.score_function(),
        })
    }
}

// MLP and MoE block.

/// Upstream's `BailingMoeMLP`: a SwiGLU feed-forward block.
///
/// Used both for the dense prefix layers (at `intermediate_size`) and for the
/// single wide shared expert (at `shared_dim * num_shared_experts`).
pub struct BailingMoeMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl BailingMoeMLP {
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
}

/// Upstream's `BailingMoeSparseMoeBlock`.
pub struct BailingMoeSparseBlock {
    pub gate: BailingMoeGate,
    pub switch_mlp: SwitchGLU,
    /// One wide MLP, not `num_shared_experts` experts, and never packed into the
    /// switch tensors. Its output is added at a fixed weight of 1.0.
    pub shared_experts: Option<BailingMoeMLP>,
}

impl BailingMoeSparseBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = *orig_shape.last().unwrap_or(&0);
        let x_flat = if orig_shape.len() > 2 {
            let tokens: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[tokens, hidden])
        } else {
            mlxcel_core::copy(x)
        };

        let (indices, scores) = self.gate.forward(&x_flat);

        // Fused single-token decode kernel (#268), on by default across every
        // SwitchGLU family; it declines any config it does not support and the
        // caller falls back to gather_qmm + moe_weighted_sum. A raw bf16 Bailing
        // checkpoint has no quantized expert parts, so it always declines there.
        let routed = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1 && fused_moe_enabled() {
                self.switch_mlp
                    .forward_fused_kernel(&x_flat, &indices, &scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.switch_mlp.forward(&x_flat, &indices);
                    moe_weighted_sum(&expert_out, &scores, mlxcel_core::array_dtype(&x_flat))
                }
            }
        };

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
        let gate = BailingMoeGate::from_weights(weights, args, prefix)?;
        let switch_mlp = SwitchGLU::from_weights(
            weights,
            &format!("{prefix}.switch_mlp"),
            args.group_size(),
            args.bits(),
        )?;
        let shared_experts = if args.has_shared_expert() {
            Some(BailingMoeMLP::from_weights(
                weights,
                args,
                &format!("{prefix}.shared_experts"),
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
}

/// Either the dense prefix MLP or the sparse block, per layer.
pub enum FeedForward {
    Dense(BailingMoeMLP),
    Sparse(Box<BailingMoeSparseBlock>),
}

impl FeedForward {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(block) => block.forward(x),
        }
    }
}

// Attention.

/// Bailing attention: fused GQA `query_key_value`, optional per-head QK norm,
/// RoPE, and a `dense` output projection.
pub struct Attention {
    pub query_key_value: UnifiedLinear,
    /// Output projection. Bailing names it `dense`, not `o_proj`.
    pub dense: UnifiedLinear,
    /// Optional per-head RMSNorm on the queries, applied after the head reshape
    /// and before RoPE.
    pub query_layernorm: Option<RMSNorm>,
    /// Optional per-head RMSNorm on the keys.
    pub key_layernorm: Option<RMSNorm>,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;

        // Fused QKV. The split is [q_size, q_size + kv_size], not an even
        // three-way split: under GQA the K and V blocks are `num_kv_heads` heads
        // wide while the Q block is `num_heads` heads wide.
        let qkv = self.query_key_value.forward(x);
        let q = mlxcel_core::slice_last_dim(&qkv, 0, q_size);
        let k = mlxcel_core::slice_last_dim(&qkv, q_size, q_size + kv_size);
        let v = mlxcel_core::slice_last_dim(&qkv, q_size + kv_size, q_size + 2 * kv_size);

        // [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim].
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // QK norm runs after the head reshape and before RoPE, normalizing each
        // head's channels, exactly as upstream orders it.
        let q = match &self.query_layernorm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.key_layernorm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

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
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, q_size]);
        self.dense.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.head_dim() as i32;

        let query_key_value = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.query_key_value"),
            group_size,
            bits,
        )?;
        let dense =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.dense"), group_size, bits)?;

        let (query_layernorm, key_layernorm) = if args.use_qk_norm {
            (
                Some(rms_norm_from_weights(
                    weights,
                    &format!("{prefix}.query_layernorm.weight"),
                    args.rms_norm_eps,
                )?),
                Some(rms_norm_from_weights(
                    weights,
                    &format!("{prefix}.key_layernorm.weight"),
                    args.rms_norm_eps,
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            query_key_value,
            dense,
            query_layernorm,
            key_layernorm,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: args.rope_dims(),
            rope_base: args.rope_theta,
        })
    }
}

fn rms_norm_from_weights(weights: &WeightMap, key: &str, eps: f32) -> Result<RMSNorm, String> {
    let weight = weights
        .get(key)
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    Ok(RMSNorm::new(mlxcel_core::copy(weight), eps))
}

// Decoder layer and model.

pub struct DecoderLayer {
    pub attention: Attention,
    pub mlp: FeedForward,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.attention.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let attention = Attention::from_weights(weights, args, &format!("{prefix}.attention"))?;

        let mlp = if args.is_moe_layer(layer_idx) {
            FeedForward::Sparse(Box::new(BailingMoeSparseBlock::from_weights(
                weights,
                args,
                &format!("{prefix}.mlp"),
            )?))
        } else {
            FeedForward::Dense(BailingMoeMLP::from_weights(
                weights,
                args,
                &format!("{prefix}.mlp"),
            )?)
        };

        let input_layernorm = rms_norm_from_weights(
            weights,
            &format!("{prefix}.input_layernorm.weight"),
            args.rms_norm_eps,
        )?;
        let post_attention_layernorm = rms_norm_from_weights(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            args.rms_norm_eps,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

/// Ant Group Ling / Bailing MoE.
pub struct BailingMoeModel {
    /// Token table. Bailing names it `word_embeddings`, not `embed_tokens`.
    pub word_embeddings: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    /// Separate output head. `None` only when `tie_word_embeddings` is true;
    /// every published Bailing checkpoint ships it.
    pub lm_head: Option<UnifiedLinear>,
    eos_token_ids: Vec<i32>,
}

impl BailingMoeModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.word_embeddings.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }
        let h = self.norm.forward(&h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.word_embeddings.as_linear(&h),
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        // Reject an impossible config before reading 33 GB of weights, not after.
        // `from_weights` validates again for the owned-weights route.
        args.validate()?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        // `config.json` is untrusted, and `from_weights` is also reachable
        // directly from the registration table, so revalidate here rather than
        // trusting that `load` ran first.
        args.validate()?;
        args.warn_on_ignored_routed_scaling_flag();
        validate_weights(weights, args)?;

        let group_size = args.group_size();
        let bits = args.bits();

        let word_embeddings =
            UnifiedEmbedding::from_weights(weights, "model.word_embeddings", group_size, bits)?;
        // Token ids are bounded by `vocab_size`, a config field, and an embedding
        // gather wraps a negative index but does not range-check a positive one,
        // so a config that overstates the table turns an ordinary prompt into an
        // out-of-bounds read whose result reaches the logits.
        validate_embedding_table(
            &word_embeddings,
            "model.word_embeddings",
            args.vocab_size,
            "vocab_size",
            args.hidden_size,
            "hidden_size",
        )?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(weights, args, i)?);
        }

        let norm = rms_norm_from_weights(weights, "model.norm.weight", args.rms_norm_eps)?;

        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(load_lm_head(weights, args)?)
        };

        Ok(Self {
            word_embeddings,
            layers,
            norm,
            lm_head,
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

// LanguageModel trait implementation.

impl LanguageModel for BailingMoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        BailingMoeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        BailingMoeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "bailing_moe_tests.rs"]
mod tests;
