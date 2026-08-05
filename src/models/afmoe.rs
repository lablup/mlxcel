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

//! Arcee AFMoE (`afmoe`), the architecture behind the Trinity family.
//!
//! Ported from mlx-lm's
//! [`afmoe.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/afmoe.py).
//!
//! The original issue described this as "structurally a qwen3_moe / deepseek MoE
//! derivative with QK-norm", which is true of the expert machinery and wrong
//! about the model class. Five features change how every layer is built, and
//! four of them are invisible to a shape check.
//!
//! # It is a hybrid sliding / full attention model, not a plain MoE decoder
//!
//! `layer_types` interleaves `sliding_attention` and `full_attention`. On
//! `Trinity-Nano-Preview` that is three sliding layers then one global, 56
//! times over. Sliding layers need a `RotatingKVCache` and global layers a
//! `KVCache`, which does not fit the trait's homogeneous `&mut [KVCache]`, so
//! this model owns its heterogeneous per-sequence state the way
//! [`crate::models::gemma3`] does and reuses that module's `Cache` enum rather
//! than restating it.
//!
//! The schedule comes from the **list**, not from a modulus.
//! `global_attn_every_n_layers` is declared alongside it and is not what
//! upstream reads; a config whose list disagrees with its modulus is honoured
//! as the list says. See [`ModelArgs::is_sliding_layer`].
//!
//! # Full-attention layers are NoPE
//!
//! `Attention.__init__` builds `self.rope` **only** when `is_local_attention`,
//! so global layers apply no positional encoding at all. Applying RoPE
//! everywhere is the obvious mistake, costs nothing at load, and is invisible on
//! a short prompt: the rotation at position 0 is the identity, and the first few
//! positions differ too little to change an argmax. See [`Attention::forward`].
//!
//! # Attention carries a sigmoid output gate
//!
//! `gate = mx.sigmoid(self.gate_proj(x)); output = output * gate` runs **before**
//! `o_proj`, so every attention block has an extra `self_attn.gate_proj` weight
//! that qwen3_moe does not. Note the name collision: this `gate_proj` is an
//! attention gate, unrelated to the `gate_proj` of a SwiGLU MLP and unrelated to
//! the MoE router at `mlp.router.gate`.
//!
//! # Four norms per layer, in a sandwich
//!
//! `input_layernorm`, `post_attention_layernorm`, `pre_mlp_layernorm`, and
//! `post_mlp_layernorm`, applied as
//!
//! ```text
//! h   = x + post_attention_layernorm(attn(input_layernorm(x)))
//! out = h + post_mlp_layernorm(mlp(pre_mlp_layernorm(h)))
//! ```
//!
//! so the two "post" norms normalize each branch's **output** before it joins
//! the residual, rather than normalizing the residual on the way in. A two-norm
//! pre-norm block loads every tensor this checkpoint ships except two per layer
//! and still generates.
//!
//! # muP embedding scale
//!
//! When `mup_enabled` (true on Trinity) the token embeddings are multiplied by
//! `sqrt(hidden_size)` before the stack, the same trick Gemma uses. Omitting it
//! shrinks every hidden state by 32x on Trinity-Nano and still produces finite
//! logits.
//!
//! # Dense prefix
//!
//! The first `num_dense_layers` layers (2 on Trinity) use a plain SwiGLU MLP;
//! MoE starts at layer 2. This is the one feature the original issue's
//! qwen3_moe framing would have got right by accident, since that family has the
//! same knob under a different name.
//!
//! # Routing details, for the record
//!
//! `score_func: sigmoid`; the `expert_bias` is added to a copy of the scores used
//! for **selection only** while the returned weights are gathered from the
//! unbiased scores (the same contract [`crate::models::bailing_moe`] documents,
//! and the same silent misweighting if it is got wrong); `route_norm`
//! renormalizes the selected weights; and `route_scale` (2.826 on Trinity)
//! multiplies them. Group routing exists but `n_group` is 1 on Trinity, so it
//! never fires there and is unit-tested instead.
//!
//! # Untrusted config
//!
//! Same contract as the other ports in this tree: [`ModelArgs::validate`] rejects
//! every scalar that could size an allocation, divide, or violate an
//! undocumented MLX C++ precondition, and [`validate_weights`] rejects every
//! tensor whose real shape disagrees with the config. An MLX C++ exception
//! crossing the cxx bridge is an uncatchable `std::terminate` at the first
//! forward pass, not a Rust error.

use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::models::gemma3::{Cache, CacheInterface};
use crate::models::gpt2::{dim_eq, validate_embedding_table};
use crate::models::model_owned::ModelOwnedSequenceState;
use crate::models::switch_layers::{SwitchGLU, fused_moe_enabled, group_mask_scores};

// Configuration.

/// AFMoE `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    /// Per-layer attention kind, `"sliding_attention"` or `"full_attention"`.
    ///
    /// This is the schedule upstream reads. `global_attn_every_n_layers` is
    /// declared beside it and never consulted, so a config whose list disagrees
    /// with its modulus is honoured as the list says.
    #[serde(default)]
    pub layer_types: Vec<String>,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    /// Explicit head width. Unlike most families here, AFMoE does **not** derive
    /// it from `hidden_size / num_attention_heads`: Trinity-Nano declares 128
    /// with `hidden_size` 1024 and 8 heads, so the derived value would be 128 by
    /// coincidence, while Trinity-Mini's geometry does not coincide.
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Upstream threads this into `initialize_rope`. Trinity writes `null`, and
    /// this loader always builds the plain rotation, so anything but the no-op
    /// form is rejected rather than silently ignored.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_num_shared_experts")]
    pub num_shared_experts: usize,
    /// Layers below this index use a plain SwiGLU MLP. 2 on Trinity.
    #[serde(default = "default_num_dense_layers")]
    pub num_dense_layers: usize,

    #[serde(default = "default_true")]
    pub route_norm: bool,
    #[serde(default = "default_route_scale")]
    pub route_scale: f32,
    #[serde(default = "default_score_func")]
    pub score_func: String,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,

    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,

    /// When true the embedding output is multiplied by `sqrt(hidden_size)`.
    #[serde(default = "default_true")]
    pub mup_enabled: bool,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// A `config.json` token-id field, which may be a single int or a list of ints.
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

/// The top-level `quantization` block.
///
/// The per-tensor override entries Trinity also writes into this object (each
/// `model.layers.{i}.mlp.router.gate` at 8 bits while the rest is 4-bit) are
/// deliberately not parsed: `UnifiedLinear`, `SwitchLinear` and
/// `UnifiedEmbedding` each reconcile bits and group size from the tensor shapes
/// they load, so the override is honoured per tensor without this loader having
/// to model the key space.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "afmoe".to_string()
}
fn default_vocab_size() -> usize {
    200_192
}
fn default_hidden_size() -> usize {
    2048
}
fn default_intermediate_size() -> usize {
    6144
}
fn default_moe_intermediate_size() -> usize {
    1024
}
fn default_num_hidden_layers() -> usize {
    32
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_num_key_value_heads() -> usize {
    4
}
fn default_head_dim() -> usize {
    64
}
fn default_max_position_embeddings() -> usize {
    131_072
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_num_experts() -> usize {
    128
}
fn default_num_experts_per_tok() -> usize {
    8
}
fn default_num_shared_experts() -> usize {
    1
}
fn default_num_dense_layers() -> usize {
    2
}
fn default_true() -> bool {
    true
}
fn default_route_scale() -> f32 {
    2.826
}
fn default_score_func() -> String {
    "sigmoid".to_string()
}
fn default_n_group() -> usize {
    1
}
fn default_topk_group() -> usize {
    1
}
fn default_sliding_window() -> usize {
    2048
}

/// Which non-linearity turns router logits into routing probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFunc {
    /// Trinity's default.
    Sigmoid,
    Softmax,
}

/// Upper bounds on the architecture scalars an AFMoE `config.json` may declare.
/// Same rationale as the other ports: `config.json` is untrusted input on the
/// `mlxcel generate -m <org>/<repo>` path. Each sits orders of magnitude above
/// `Trinity-Nano-Preview` (56 layers, hidden 1024, 128 experts, vocab 200192).
const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_HIDDEN_SIZE: usize = 65_536;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_NUM_ATTENTION_HEADS: usize = 4096;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_HEAD_DIM: usize = 8192;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_MAX_POSITION_EMBEDDINGS: usize = 1 << 22;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_VOCAB_SIZE: usize = 1 << 24;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_NUM_EXPERTS: usize = 4096;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_NUM_SHARED_EXPERTS: usize = 1024;

impl ModelArgs {
    /// Whether layer `layer_idx` uses sliding-window attention.
    ///
    /// Read from `layer_types`, which is what upstream indexes. A layer past the
    /// end of the list, or a config with no list at all, falls back to full
    /// attention: that is the conservative direction, since a global layer
    /// attends to everything a sliding one would and more, so a short list
    /// degrades into a slower model rather than a wrong one.
    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "sliding_attention")
            .unwrap_or(false)
    }

    /// Index of the first full-attention layer, whose cache sizes the global
    /// prefill mask. `None` when every layer slides.
    pub fn full_attention_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| !self.is_sliding_layer(i))
    }

    /// Index of the first sliding layer, whose cache sizes the windowed prefill
    /// mask. `None` when no layer slides.
    pub fn sliding_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| self.is_sliding_layer(i))
    }

    /// Whether layer `layer_idx` is sparse.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.num_dense_layers
    }

    /// Width of the always-on shared MLP:
    /// `moe_intermediate_size * num_shared_experts`.
    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
            .saturating_mul(self.num_shared_experts)
    }

    pub fn has_shared_expert(&self) -> bool {
        self.num_shared_experts > 0
    }

    /// muP scale applied to the embeddings, or 1.0 when disabled.
    pub fn embedding_scale(&self) -> f32 {
        if self.mup_enabled {
            (self.hidden_size as f32).sqrt()
        } else {
            1.0
        }
    }

    pub fn score_func(&self) -> ScoreFunc {
        match self.score_func.as_str() {
            "softmax" => ScoreFunc::Softmax,
            _ => ScoreFunc::Sigmoid,
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

    /// Reject a `config.json` that cannot describe a real AFMoE, before any of
    /// its fields sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_attention_heads == 0 || self.num_attention_heads > MAX_NUM_ATTENTION_HEADS {
            return Err(format!(
                "AFMoE num_attention_heads ({}) must be between 1 and {MAX_NUM_ATTENTION_HEADS}",
                self.num_attention_heads
            ));
        }
        if self.hidden_size == 0 || self.hidden_size > MAX_HIDDEN_SIZE {
            return Err(format!(
                "AFMoE hidden_size ({}) must be between 1 and {MAX_HIDDEN_SIZE}",
                self.hidden_size
            ));
        }
        if self.head_dim == 0 || self.head_dim > MAX_HEAD_DIM {
            return Err(format!(
                "AFMoE head_dim ({}) must be between 1 and {MAX_HEAD_DIM}",
                self.head_dim
            ));
        }
        if self.num_key_value_heads == 0 || self.num_key_value_heads > self.num_attention_heads {
            return Err(format!(
                "AFMoE num_key_value_heads ({}) must be between 1 and num_attention_heads ({})",
                self.num_key_value_heads, self.num_attention_heads
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "AFMoE num_attention_heads ({}) must be divisible by num_key_value_heads ({}) for \
                 grouped-query attention",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.num_hidden_layers == 0 || self.num_hidden_layers > MAX_NUM_HIDDEN_LAYERS {
            return Err(format!(
                "AFMoE num_hidden_layers ({}) must be between 1 and {MAX_NUM_HIDDEN_LAYERS}",
                self.num_hidden_layers
            ));
        }
        // `layer_types` is the schedule, so a list that does not cover the stack
        // would silently make the uncovered tail global. Upstream indexes the
        // list directly and would raise; this says which layer is missing.
        if !self.layer_types.is_empty() && self.layer_types.len() < self.num_hidden_layers {
            return Err(format!(
                "AFMoE layer_types has {} entries for {} layers; it is the per-layer \
                 sliding/full attention schedule, and a short list would silently make every \
                 uncovered layer global",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        for (idx, kind) in self.layer_types.iter().enumerate() {
            if !matches!(kind.as_str(), "sliding_attention" | "full_attention") {
                return Err(format!(
                    "AFMoE layer_types[{idx}] is {kind:?}; only \"sliding_attention\" and \
                     \"full_attention\" are defined, and an unrecognized value would fall back to \
                     full attention while the checkpoint expects a window"
                ));
            }
        }
        if self.sliding_index().is_some() && self.sliding_window == 0 {
            return Err(
                "AFMoE sliding_window must be at least 1 when layer_types declares a \
                 sliding_attention layer; a zero window would keep no keys at all"
                    .to_string(),
            );
        }
        if self.intermediate_size == 0 || self.intermediate_size > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "AFMoE intermediate_size ({}) must be between 1 and {MAX_INTERMEDIATE_SIZE}",
                self.intermediate_size
            ));
        }
        if self.moe_intermediate_size == 0 || self.moe_intermediate_size > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "AFMoE moe_intermediate_size ({}) must be between 1 and {MAX_INTERMEDIATE_SIZE}",
                self.moe_intermediate_size
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(format!(
                "AFMoE vocab_size ({}) must be between 1 and {MAX_VOCAB_SIZE}",
                self.vocab_size
            ));
        }
        if self.max_position_embeddings == 0
            || self.max_position_embeddings > MAX_MAX_POSITION_EMBEDDINGS
        {
            return Err(format!(
                "AFMoE max_position_embeddings ({}) must be between 1 and \
                 {MAX_MAX_POSITION_EMBEDDINGS}",
                self.max_position_embeddings
            ));
        }
        if self.num_dense_layers > self.num_hidden_layers {
            return Err(format!(
                "AFMoE num_dense_layers ({}) must not exceed num_hidden_layers ({})",
                self.num_dense_layers, self.num_hidden_layers
            ));
        }

        self.validate_routing()?;
        self.validate_shared_expert()?;
        self.validate_rope()?;
        self.validate_rope_scaling()?;
        self.validate_norm_eps()?;
        self.validate_quantization()
    }

    /// Reject routing parameters that would index out of range inside MLX.
    fn validate_routing(&self) -> Result<(), String> {
        if self.num_dense_layers >= self.num_hidden_layers {
            // Every layer is dense; the router is never built.
            return Ok(());
        }
        if self.num_experts == 0 || self.num_experts > MAX_NUM_EXPERTS {
            return Err(format!(
                "AFMoE num_experts ({}) must be between 1 and {MAX_NUM_EXPERTS}",
                self.num_experts
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(format!(
                "AFMoE num_experts_per_tok ({}) must be between 1 and num_experts ({}); the \
                 router selects that many indices out of a row of num_experts scores",
                self.num_experts_per_tok, self.num_experts
            ));
        }
        if !matches!(self.score_func.as_str(), "sigmoid" | "softmax") {
            return Err(format!(
                "AFMoE score_func ({:?}) must be \"sigmoid\" (the AFMoE default) or \"softmax\"; \
                 an unrecognized value is rejected rather than silently falling back, because \
                 either fallback changes the routing distribution on every token while leaving \
                 the output finite and plausible",
                self.score_func
            ));
        }
        if self.n_group == 0 {
            return Err(
                "AFMoE n_group must be at least 1 (1 disables grouped routing)".to_string(),
            );
        }
        if self.n_group > 1 {
            if !self.num_experts.is_multiple_of(self.n_group) {
                return Err(format!(
                    "AFMoE num_experts ({}) must be divisible by n_group ({}); grouped routing \
                     reshapes the score row into n_group equal groups",
                    self.num_experts, self.n_group
                ));
            }
            let experts_per_group = self.num_experts / self.n_group;
            if experts_per_group < 2 {
                return Err(format!(
                    "AFMoE n_group ({}) leaves {experts_per_group} expert(s) per group; grouped \
                     routing scores each group by the sum of its top two experts, so a group must \
                     hold at least 2",
                    self.n_group
                ));
            }
            if self.topk_group == 0 || self.topk_group >= self.n_group {
                return Err(format!(
                    "AFMoE topk_group ({}) must be between 1 and n_group - 1 ({}); the \
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
        if !self.route_scale.is_finite() {
            return Err(format!(
                "AFMoE route_scale ({}) must be finite; it multiplies every routed expert weight, \
                 so a non-finite value makes every MoE output NaN and that NaN reaches the logits \
                 without anything throwing",
                self.route_scale
            ));
        }
        Ok(())
    }

    fn validate_shared_expert(&self) -> Result<(), String> {
        if self.num_shared_experts > MAX_NUM_SHARED_EXPERTS {
            return Err(format!(
                "AFMoE num_shared_experts ({}) must not exceed {MAX_NUM_SHARED_EXPERTS}",
                self.num_shared_experts
            ));
        }
        if !self.has_shared_expert() {
            return Ok(());
        }
        let width = self
            .moe_intermediate_size
            .checked_mul(self.num_shared_experts)
            .ok_or_else(|| {
                format!(
                    "AFMoE shared-expert width overflows: moe_intermediate_size ({}) * \
                     num_shared_experts ({})",
                    self.moe_intermediate_size, self.num_shared_experts
                )
            })?;
        if width == 0 || width > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "AFMoE shared-expert width ({width}) must be between 1 and {MAX_INTERMEDIATE_SIZE}"
            ));
        }
        Ok(())
    }

    /// Reject RoPE parameters MLX would throw on.
    ///
    /// AFMoE rotates the full head width on its sliding layers and does not
    /// rotate at all on its global ones, so only the sliding width is checked.
    fn validate_rope(&self) -> Result<(), String> {
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!(
                "AFMoE rope_theta ({}) must be a finite positive number; RoPE exponentiates it \
                 per channel, so a zero, negative or non-finite base makes every rotated channel \
                 NaN and that NaN reaches the logits without anything throwing",
                self.rope_theta
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(format!(
                "AFMoE head_dim ({}) must be even; the sliding layers rotate the full head width \
                 and RoPE rotates channel pairs, so MLX throws on an odd `dims`",
                self.head_dim
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
            "AFMoE rope_scaling ({scaling:?}) is not implemented for this family; upstream threads \
             it into initialize_rope, but this loader always builds the plain rotation, so \
             accepting a scaled block would place every token at the wrong position while the \
             model still loaded and still generated fluent text. Only an absent, empty or \
             \"default\" block is accepted."
        ))
    }

    fn validate_norm_eps(&self) -> Result<(), String> {
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "AFMoE rms_norm_eps ({}) must be a finite positive number; it is added to the mean \
                 square under an rsqrt, so a non-finite, negative or zero value makes every \
                 normalized hidden state NaN and that NaN reaches the logits without anything \
                 throwing",
                self.rms_norm_eps
            ));
        }
        Ok(())
    }

    fn validate_quantization(&self) -> Result<(), String> {
        let Some(quantization) = self.quantization.as_ref() else {
            return Ok(());
        };
        mlxcel_core::layers::validate_quantization_params(
            quantization.group_size,
            quantization.bits,
        )
        .map_err(|e| format!("AFMoE config.json: {e}"))
    }
}

fn rms_norm_from_weights(weights: &WeightMap, key: &str, eps: f32) -> Result<RMSNorm, String> {
    let weight = weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    Ok(RMSNorm::new(weight, eps))
}

// Attention.

/// AFMoE attention: GQA with per-head QK-norm, a sigmoid output gate, and RoPE
/// **only** on the sliding layers.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    /// The attention output gate. Unrelated to a SwiGLU `gate_proj` and to the
    /// MoE router, despite sharing the name.
    pub gate_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    /// `false` on a full-attention layer, where upstream builds no rope at all.
    pub uses_rope: bool,
    pub rope_base: f32,
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

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // QK-norm runs after the head reshape and before RoPE.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // **Only the sliding layers rotate.** Upstream builds `self.rope` inside
        // `if is_local_attention`, so a global layer applies no positional
        // encoding at all. Rotating here anyway is invisible on a short prompt
        // and wrong on every long one.
        let offset = cache.offset();
        let (q, k) = if self.uses_rope {
            (
                mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset),
                mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset),
            )
        } else {
            (q, k)
        };

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

        // The sigmoid gate multiplies the attention output BEFORE `o_proj`, and
        // is computed from the block input rather than from the attention
        // result.
        let gate = mlxcel_core::sigmoid(&self.gate_proj.forward(x));
        let gated = mlxcel_core::multiply(&attn_out, &gate);
        self.o_proj.forward(&gated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        uses_rope: bool,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.head_dim as i32;
        let load = |leaf: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{leaf}"), group_size, bits)
        };
        Ok(Self {
            q_proj: load("q_proj")?,
            k_proj: load("k_proj")?,
            v_proj: load("v_proj")?,
            o_proj: load("o_proj")?,
            gate_proj: load("gate_proj")?,
            q_norm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.q_norm.weight"),
                args.rms_norm_eps,
            )?,
            k_norm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.k_norm.weight"),
                args.rms_norm_eps,
            )?,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            uses_rope,
            rope_base: args.rope_theta,
        })
    }
}

// Feed-forward.

/// A SwiGLU MLP, used for the dense prefix layers and for the shared expert.
pub struct Mlp {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl Mlp {
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

/// Upstream's `AfmoeMoE`.
pub struct AfmoeMoe {
    /// The router. Upstream wraps it in a `MoERouter` module purely so the
    /// checkpoint key reads `mlp.router.gate`.
    pub router: UnifiedLinear,
    /// Selection-only correction bias, `[num_experts]`.
    pub expert_bias: UniquePtr<MlxArray>,
    pub experts: SwitchGLU,
    pub shared_experts: Option<Mlp>,
    pub top_k: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub route_norm: bool,
    pub route_scale: f32,
    pub score_func: ScoreFunc,
}

impl AfmoeMoe {
    /// Returns `(expert_indices, expert_weights)` for a `[n_tokens, hidden]`
    /// input.
    ///
    /// The expert bias is added to a **copy** used for selection only; the
    /// returned weights are gathered from the unbiased scores. Gathering from
    /// the biased copy leaves the output finite and plausible while misweighting
    /// every routed contribution.
    fn route(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let gates = self.router.forward(x);
        let in_type = mlxcel_core::array_dtype(&gates);
        let gates = mlxcel_core::astype(&gates, mlxcel_core::dtype::FLOAT32);

        let scores = match self.score_func {
            ScoreFunc::Sigmoid => mlxcel_core::sigmoid(&gates),
            ScoreFunc::Softmax => mlxcel_core::softmax(&gates, -1),
        };

        let selection = mlxcel_core::add(&scores, &self.expert_bias);
        let selection = if self.n_group > 1 {
            group_mask_scores(&selection, self.n_group, self.topk_group)
        } else {
            selection
        };

        // `argpartition(-selection, kth = k - 1)` then the first k, which is
        // upstream's orientation. It matters whenever the selection scores tie:
        // the mirrored form picks a different set, not a different order.
        let selection_shape = mlxcel_core::array_shape(&selection);
        let n_experts = *selection_shape.last().unwrap_or(&0);
        let kth = (self.top_k - 1).clamp(0, (n_experts - 1).max(0));
        let order = mlxcel_core::argpartition(&mlxcel_core::negative(&selection), kth, -1);
        let indices = slice_axis(&order, -1, 0, self.top_k);

        // Weights from the UNBIASED, unmasked scores.
        let selected = mlxcel_core::take_along_axis(&scores, &indices, -1);
        let selected = if self.route_norm && self.top_k > 1 {
            let sum = mlxcel_core::sum_axis(&selected, -1, true);
            mlxcel_core::divide(&selected, &sum)
        } else {
            selected
        };
        let scale = mlxcel_core::full_f32(&[1], self.route_scale, mlxcel_core::dtype::FLOAT32);

        // **The weights stay float32.** `AfmoeMoE.__call__` never casts them
        // back to the activation dtype, unlike the Bailing gate this otherwise
        // resembles, so upstream's `y * selected_scores[..., None]` promotes the
        // combine and sums in f32 before a single cast at the end.
        //
        // Measured honestly: on `Trinity-Nano-Preview-4bit` this changes the
        // first MoE layer's output by less than the print precision of a
        // six-figure mean, so it buys no measurable accuracy there. It is kept
        // because it is what upstream computes, and because this stack amplifies
        // a perturbation roughly twofold per layer over 56 layers, which makes
        // every avoidable rounding worth avoiding. See `AfmoeMoe::forward`.
        let _ = in_type;
        (indices, mlxcel_core::multiply(&selected, &scale))
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = *orig_shape.last().unwrap_or(&0);
        let x_flat = if orig_shape.len() > 2 {
            let tokens: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[tokens, hidden])
        } else {
            mlxcel_core::copy(x)
        };

        let (indices, scores) = self.route(&x_flat);
        let routed = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1 && fused_moe_enabled() {
                // The fused kernel consumes the activation dtype, so it
                // takes a cast copy; it is an opt-in decode fast path
                // (`MLXCEL_FUSED_MOE`) rather than the default.
                let fused_scores = mlxcel_core::astype(&scores, mlxcel_core::array_dtype(&x_flat));
                self.experts
                    .forward_fused_kernel(&x_flat, &indices, &fused_scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.experts.forward(&x_flat, &indices);
                    // Deliberately not `moe_weighted_sum`: that helper casts the
                    // weights down to the expert dtype first, which is right for
                    // the families whose gate already returns them in that dtype
                    // and wrong here, where upstream keeps them in f32 through
                    // the sum. See `AfmoeMoe::route`.
                    let out32 = mlxcel_core::astype(&expert_out, mlxcel_core::dtype::FLOAT32);
                    let weighted =
                        mlxcel_core::multiply(&out32, &mlxcel_core::expand_dims(&scores, -1));
                    let summed = mlxcel_core::sum_axis(&weighted, -2, false);
                    mlxcel_core::astype(&summed, mlxcel_core::array_dtype(&expert_out))
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
        let group_size = args.group_size();
        let bits = args.bits();

        // Upstream initializes `expert_bias` to zeros, so a checkpoint that
        // ships no tensor still routes with a zero bias rather than failing.
        let bias_key = format!("{prefix}.expert_bias");
        let expert_bias = match weights.get(&bias_key) {
            Some(bias) => mlxcel_core::astype(bias, mlxcel_core::dtype::FLOAT32),
            None => {
                mlxcel_core::full_f32(&[args.num_experts as i32], 0.0, mlxcel_core::dtype::FLOAT32)
            }
        };

        Ok(Self {
            router: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.router.gate"),
                group_size,
                bits,
            )?,
            expert_bias,
            experts: SwitchGLU::from_weights(
                weights,
                &format!("{prefix}.experts"),
                group_size,
                bits,
            )?,
            shared_experts: if args.has_shared_expert() {
                Some(Mlp::from_weights(
                    weights,
                    args,
                    &format!("{prefix}.shared_experts"),
                )?)
            } else {
                None
            },
            top_k: args.num_experts_per_tok as i32,
            n_group: args.n_group as i32,
            topk_group: i32::try_from(args.topk_group).unwrap_or(i32::MAX),
            route_norm: args.route_norm,
            route_scale: args.route_scale,
            score_func: args.score_func(),
        })
    }
}

/// Either the dense prefix MLP or the sparse block, per layer.
pub enum FeedForward {
    Dense(Mlp),
    Sparse(Box<AfmoeMoe>),
}

impl FeedForward {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(moe) => moe.forward(x),
        }
    }
}

// Decoder layer and model.

/// One AFMoE block: a sandwich-normed attention branch and FFN branch.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: FeedForward,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub pre_mlp_layernorm: RMSNorm,
    pub post_mlp_layernorm: RMSNorm,
    pub uses_sliding: bool,
}

impl DecoderLayer {
    /// **The two "post" norms normalize each branch's output**, not the
    /// residual. Moving either one to the residual path is a pre-norm block that
    /// loads every tensor this checkpoint ships and still generates.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let attn = self
            .self_attn
            .forward(&self.input_layernorm.forward(x), cache, mask);
        let h = mlxcel_core::add(x, &self.post_attention_layernorm.forward(&attn));

        let ffn = self.mlp.forward(&self.pre_mlp_layernorm.forward(&h));
        mlxcel_core::add(&h, &self.post_mlp_layernorm.forward(&ffn))
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
                args.rms_norm_eps,
            )
        };
        Ok(Self {
            self_attn: Attention::from_weights(
                weights,
                args,
                &format!("{prefix}.self_attn"),
                // Only sliding layers rotate.
                uses_sliding,
            )?,
            mlp: if args.is_moe_layer(layer_idx) {
                FeedForward::Sparse(Box::new(AfmoeMoe::from_weights(
                    weights,
                    args,
                    &format!("{prefix}.mlp"),
                )?))
            } else {
                FeedForward::Dense(Mlp::from_weights(weights, args, &format!("{prefix}.mlp"))?)
            },
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_mlp_layernorm: norm("pre_mlp_layernorm")?,
            post_mlp_layernorm: norm("post_mlp_layernorm")?,
            uses_sliding,
        })
    }
}

/// Arcee AFMoE.
pub struct AfmoeModel {
    pub embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    /// `sqrt(hidden_size)` when `mup_enabled`, 1.0 otherwise.
    embedding_scale: f32,
    sliding_window: i32,
    full_attention_index: Option<usize>,
    sliding_index: Option<usize>,
    eos_token_ids: Vec<i32>,
    /// Sliding layers hold a `RotatingKVCache` and global layers a `KVCache`,
    /// which does not fit the trait's homogeneous `&mut [KVCache]`, so the model
    /// owns its heterogeneous cache here and persists it across forward calls.
    sequence_state: ModelOwnedSequenceState<Cache>,
}

impl AfmoeModel {
    pub(crate) fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(input_ids)[1];

        let mut h = self.embed_tokens.forward(input_ids);
        if self.embedding_scale != 1.0 {
            h = mlxcel_core::multiply_scalar(&h, self.embedding_scale);
        }

        // Both prefill masks are sized from the cache's live window rather than
        // the monotonic offset, so a trimmed cache does not produce a mask wider
        // than the K/V it returns. Decode needs no mask: a RotatingKVCache
        // already returns only its window.
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
                Some(create_causal_mask(seq_len, full_live)),
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
            .map(|layer| {
                if layer.uses_sliding {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window))
                } else {
                    Cache::Standard(KVCache::new())
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

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        // Reject an impossible config before reading the weights, not after.
        args.validate()?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;
        validate_weights(weights, args)?;

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

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(weights, args, i)?);
        }

        let norm = rms_norm_from_weights(weights, "model.norm.weight", args.rms_norm_eps)?;
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
                    Cache::Rotating(RotatingKVCache::new(args.sliding_window as i32))
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
            embedding_scale: args.embedding_scale(),
            sliding_window: args.sliding_window as i32,
            full_attention_index: args.full_attention_index(),
            sliding_index: args.sliding_index(),
            eos_token_ids: args.eos_token_ids(),
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
        })
    }
}

// Weight-shape validation.

fn quant_mode(weights: &WeightMap, prefix: &str, group_size: i32, bits: i32) -> &'static str {
    let has_biases = weights.contains_key(&format!("{prefix}.biases"));
    mlxcel_core::layers::infer_quantization_mode(has_biases, group_size, bits)
}

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
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected {out_features} rows"
        ));
    }
    if !quantized && !dim_eq(shape[1], in_features) {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected [{out_features}, {in_features}]"
        ));
    }
    validate_quantized_packing(weights, prefix, in_features, group_size, bits)
}

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
/// router can emit any index below `num_experts`. MLX's gather adds the axis
/// size to a negative index but performs no range check on a positive one, so a
/// stacked tensor with fewer planes than the config claims turns an ordinary
/// token into an out-of-bounds read whose result reaches the logits.
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

/// Reject a checkpoint whose real tensor shapes disagree with `config.json`,
/// before any of them reaches MLX.
pub fn validate_weights(weights: &WeightMap, args: &ModelArgs) -> Result<(), String> {
    let hidden = args.hidden_size;
    let head_dim = args.head_dim;
    let q_size = args.num_attention_heads * head_dim;
    let kv_size = args.num_key_value_heads * head_dim;
    let group_size = args.group_size();
    let bits = args.bits();

    validate_norm(weights, "model.norm.weight", hidden)?;
    validate_quantized_packing(weights, "model.embed_tokens", hidden, group_size, bits)?;
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
        let attn = format!("{prefix}.self_attn");

        validate_projection(
            weights,
            &format!("{attn}.q_proj"),
            q_size,
            hidden,
            group_size,
            bits,
        )?;
        validate_projection(
            weights,
            &format!("{attn}.k_proj"),
            kv_size,
            hidden,
            group_size,
            bits,
        )?;
        validate_projection(
            weights,
            &format!("{attn}.v_proj"),
            kv_size,
            hidden,
            group_size,
            bits,
        )?;
        validate_projection(
            weights,
            &format!("{attn}.o_proj"),
            hidden,
            q_size,
            group_size,
            bits,
        )?;
        // The attention output gate, which qwen3_moe has no counterpart for.
        validate_projection(
            weights,
            &format!("{attn}.gate_proj"),
            q_size,
            hidden,
            group_size,
            bits,
        )?;
        validate_norm(weights, &format!("{attn}.q_norm.weight"), head_dim)?;
        validate_norm(weights, &format!("{attn}.k_norm.weight"), head_dim)?;

        // All four norms, not two.
        for leaf in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_mlp_layernorm",
            "post_mlp_layernorm",
        ] {
            validate_norm(weights, &format!("{prefix}.{leaf}.weight"), hidden)?;
        }

        let mlp = format!("{prefix}.mlp");
        if args.is_moe_layer(layer) {
            validate_projection(
                weights,
                &format!("{mlp}.router.gate"),
                args.num_experts,
                hidden,
                group_size,
                bits,
            )?;
            if let Some(bias) = weights.get(&format!("{mlp}.expert_bias")) {
                let bias_shape = mlxcel_core::array_shape(bias);
                if bias_shape.len() != 1 || !dim_eq(bias_shape[0], args.num_experts) {
                    return Err(format!(
                        "unexpected {mlp}.expert_bias shape {bias_shape:?}: expected [{}]; it is \
                         added to a row of that many router scores",
                        args.num_experts
                    ));
                }
            }
            for (leaf, out_features, in_features) in [
                ("gate_proj", args.moe_intermediate_size, hidden),
                ("up_proj", args.moe_intermediate_size, hidden),
                ("down_proj", hidden, args.moe_intermediate_size),
            ] {
                validate_stacked_experts(
                    weights,
                    &format!("{mlp}.experts.{leaf}"),
                    args.num_experts,
                    out_features,
                    in_features,
                    group_size,
                    bits,
                )?;
            }
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

// LanguageModel trait implementation.

impl LanguageModel for AfmoeModel {
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
        // No batched (multi-sequence-per-forward) decode: the rotating cache has
        // no batched path here. Concurrent server requests are still isolated
        // per sequence through the model-owned `sequence_state`.
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "afmoe_tests.rs"]
mod tests;
