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

//! Ant Group Ling / Ring linear-attention MoE (`bailing_moe_linear`).
//!
//! Ported from mlx-lm's
//! [`bailing_moe_linear.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/bailing_moe_linear.py).
//!
//! The FFN half is [`crate::models::bailing_moe`] unchanged: the same
//! `BailingMoeGate`, the same `SwitchGLU` routed experts, the same single wide
//! shared MLP, reached through the same `mlp.gate.gate_proj` / `mlp.switch_mlp`
//! key space. This module reuses those types directly rather than restating
//! them. What is new is the attention stack.
//!
//! # The linear attention is GLA, not gated-delta
//!
//! `Ring-mini-linear-2.0` looks like a Qwen3-Next-shaped hybrid, and it is not.
//! Upstream's `recurrent_gla` is
//!
//! ```text
//! h_t = h_{t-1} * exp(g) + k_t^T v_t
//! y_t = (q_t h_t) * scale
//! ```
//!
//! There is no delta term. [`crate::models::gated_delta`] computes
//! `delta = (v - k h_{t-1}) * beta` and adds `k^T delta`, a different update
//! rule that produces a model which loads, runs, and emits fluent text from the
//! wrong recurrence. The two share no code here on purpose.
//!
//! # The decay is a fixed ALiBi schedule, not a projection
//!
//! `g` is not read from the checkpoint at all. It is the ALiBi slope schedule
//! for `num_attention_heads`, negated and scaled by the layer's position in the
//! stack:
//!
//! ```text
//! layer_factor = 1 - max(0, layer_idx - 1) / max(1, num_hidden_layers - 1) + 1e-5
//! g            = -slopes * layer_factor
//! ```
//!
//! so every linear layer decays at its own constant per-head rate, computed
//! once at construction. See [`alibi_slopes`] and [`layer_decay`]. Because `g`
//! is constant in `t`, the recurrence has a closed form and does not need a
//! per-token loop; see [`gla_chunked`].
//!
//! # Two prefill paths, and why the slower one is the default
//!
//! [`gla_sequential`] is upstream's `for t in range(L)` loop transcribed step
//! for step, and it is what runs by default. Because `exp(g)` does not depend on
//! `t`, the recurrence also has a closed form,
//!
//! ```text
//! y_i = scale * [ (q_i h_0) e^{g(i+1)} + sum_{j<=i} e^{g(i-j)} (q_i . k_j) v_j ]
//! ```
//!
//! a decayed, un-softmaxed attention over a chunk plus one carried state term,
//! which [`gla_chunked`] evaluates in `O(T/C)` sequential steps at `C = 64`.
//! **That is the default as of #1040.** It measures 2.6x to 4x faster on
//! prefill and also lower perplexity at every window length measured, by 1.5%
//! at 128 tokens rising to 13.75% at 512, because the intra-chunk sum lands in
//! a matmul accumulator instead of a bf16 running state that compounds its
//! error over the sequence.
//!
//! The two paths do decode different continuations from one checkpoint, since
//! reassociating the sum moves layer 0's activations by 0.04%, layer 3
//! amplifies its input ~23x, and the 256-expert top-8 router turns a sub-ulp
//! score difference into a different expert. The measurement is what settles
//! which of the two differing answers to ship. Set
//! `MLXCEL_BAILING_LINEAR_CHUNKED_PREFILL=0` for the reference's own
//! arithmetic, which is what to use when diffing against mlx-lm.
//! `chunked_gla_matches_the_sequential_recurrence` pins the two against a naive
//! host-side reference.
//!
//! # `GroupRMSNorm` is not RMSNorm
//!
//! The linear path's output normalizer splits the `num_heads * head_dim` axis
//! into `group_norm_size` groups (4 on Ring-mini), RMS-normalizes each group
//! **without** a weight, flattens, and only then multiplies by the full-width
//! weight. Normalizing across the whole axis instead changes every linear
//! layer's output while leaving it finite. See [`GroupRMSNorm`].
//!
//! # Linear layers are MHA, full-attention layers are GQA
//!
//! Upstream's `LinearAttention` sets `num_key_value_heads = num_attention_heads`
//! and asserts the group count is 1, so its fused `query_key_value` is
//! `(H + 2H) * head_dim` wide while a global layer's is `(H + 2H_kv) * head_dim`.
//! On Ring-mini that is `[6144, 2048]` against `[3072, 2048]` in the same
//! checkpoint, and the head width differs too: a linear layer always derives
//! `hidden_size / num_attention_heads` while a global layer honours an explicit
//! `head_dim`. [`ModelArgs::linear_head_dim`] and [`ModelArgs::head_dim`] are
//! separate for that reason.
//!
//! # Untrusted config
//!
//! Same contract as [`crate::models::bailing_moe`]: `config.json` arrives from a
//! third-party HuggingFace repo in the ordinary
//! `mlxcel generate -m <org>/<repo>` flow, so [`ModelArgs::validate`] rejects
//! every scalar that could size an allocation, divide, truncate through an
//! `as i32` cast, or violate an undocumented MLX C++ precondition, and
//! [`validate_weights`] rejects every tensor whose real shape disagrees with the
//! config. An MLX C++ exception crossing the cxx bridge is an uncatchable
//! `std::terminate` at the first forward pass, not a Rust error, so a check that
//! happens at the first forward pass is not a check.

use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::models::bailing_moe::{
    BailingMoeGate, BailingMoeMLP, BailingMoeSparseBlock, ScoreFunction, router_prefix,
};
use crate::models::gpt2::{dim_eq, validate_embedding_table};
use crate::models::model_owned::ModelOwnedSequenceState;
use crate::models::switch_layers::SwitchGLU;

/// Chunk length for [`gla_chunked`].
///
/// Matches [`crate::models::gated_delta`]'s delta-rule chunk length. It trades
/// the `C x C` intra-chunk score matrix against the number of sequential
/// inter-chunk steps; 64 keeps the former small enough that the `[B, H, C, C]`
/// intermediate stays well inside a decode-sized allocation.
const GLA_CHUNK: i32 = 64;

// Configuration.

/// `bailing_moe_linear` `config.json`.
///
/// Field-for-field upstream's `ModelArgs`. It overlaps
/// [`crate::models::bailing_moe::ModelArgs`] almost entirely, and is a separate
/// type rather than a shared one because the three fields that differ
/// (`layer_group_size`, `group_norm_size`, `head_dim`) change how every layer is
/// built. Widening the dense family's config to carry them would make a
/// `bailing_moe` checkpoint parse fields no `bailing_moe` layer reads.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Explicit head width for the **full-attention** layers. 128 on
    /// `Ring-mini-linear-2.0`, where it happens to equal
    /// `hidden_size / num_attention_heads`; the linear layers always derive
    /// theirs and never read this. See [`ModelArgs::linear_head_dim`].
    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    #[serde(default)]
    pub num_experts: Option<usize>,

    #[serde(default)]
    pub num_shared_experts: usize,

    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    /// Layers below this index use a dense MLP. 1 on `Ring-mini-linear-2.0`, so
    /// layer 0 is dense and MoE starts at layer 1.
    #[serde(default)]
    pub first_k_dense_replace: usize,

    /// Every `layer_group_size`-th layer is full attention; the rest are linear.
    /// 5 on `Ring-mini-linear-2.0`, so layers 4, 9, 14 and 19 are global and the
    /// other 16 are linear. See [`ModelArgs::is_global_layer`].
    #[serde(default = "default_layer_group_size")]
    pub layer_group_size: usize,

    /// Number of groups [`GroupRMSNorm`] splits the linear path's output into.
    /// 4 on `Ring-mini-linear-2.0`.
    #[serde(default = "default_group_norm_size")]
    pub group_norm_size: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Upstream threads this into `initialize_rope`. `Ring-mini-linear-2.0`
    /// writes `null`, and this loader always builds the plain rotation, so
    /// anything but the no-op form is rejected rather than silently ignored.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub rope_traditional: bool,

    #[serde(default)]
    pub use_bias: bool,

    #[serde(default)]
    pub use_qkv_bias: bool,

    #[serde(default)]
    pub norm_head: bool,

    /// Declared by upstream's `ModelArgs` and never read, exactly as in
    /// `bailing_moe`. Rejected when true.
    #[serde(default)]
    pub norm_softmax: bool,

    #[serde(default)]
    pub use_qk_norm: bool,

    /// Declared by upstream's `ModelArgs` and never read: every normalizer in
    /// the file is an `nn.RMSNorm` or a [`GroupRMSNorm`] regardless. Rejected
    /// when false, since a checkpoint clearing it is asking for LayerNorm
    /// behaviour no released implementation provides.
    #[serde(default = "default_true")]
    pub use_rmsnorm: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,

    #[serde(default)]
    pub moe_router_enable_expert_bias: bool,

    /// Parsed for diagnostics only: upstream stores it and never reads it, and
    /// this port mirrors that, exactly as `bailing_moe` does.
    #[serde(default = "default_true")]
    pub moe_router_enable_routed_scaling: bool,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    #[serde(default = "default_score_function")]
    pub score_function: String,

    #[serde(default = "default_n_group")]
    pub n_group: usize,

    #[serde(default = "default_topk_group")]
    pub topk_group: usize,

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
/// The per-tensor override entries `Ring-mini-linear-2.0` also writes into this
/// object (`model.layers.1.mlp.gate.gate_proj` at 8 bits while the rest is
/// 4-bit) are deliberately **not** parsed. `UnifiedLinear`, `SwitchLinear` and
/// `UnifiedEmbedding` all reconcile bits and group size from the tensor shapes
/// they actually load, so the override is honoured per tensor without this
/// loader having to model the key space; `serde` ignores the extra keys.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "bailing_moe_linear".to_string()
}
fn default_num_experts_per_tok() -> usize {
    1
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_layer_group_size() -> usize {
    1
}
fn default_group_norm_size() -> usize {
    1
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

/// Upper bounds on the architecture scalars a `bailing_moe_linear`
/// `config.json` may declare. Same rationale and same magnitudes as
/// [`crate::models::bailing_moe`]'s: `config.json` is untrusted input, and each
/// ceiling sits orders of magnitude above `Ring-mini-linear-2.0` (20 layers,
/// `hidden_size` 2048, 256 experts, `vocab_size` 157184).
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
    /// Head width of the **full-attention** layers: explicit `head_dim` when the
    /// config declares one, `hidden_size / num_attention_heads` otherwise.
    ///
    /// Only valid after [`ModelArgs::validate`], which rejects
    /// `num_attention_heads == 0`.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Head width of the **linear** layers.
    ///
    /// Upstream's `LinearAttention.__init__` writes
    /// `self.head_dim = args.hidden_size // args.num_attention_heads` and never
    /// consults `args.head_dim`, so a checkpoint that declares a `head_dim`
    /// different from that ratio has two head widths in one stack. Deriving both
    /// from the same field would size the fused `query_key_value` split wrong on
    /// the linear layers only, and MLX's `slice` clamps rather than throws.
    pub fn linear_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Output width of a full-attention layer's fused `query_key_value`:
    /// `(num_attention_heads + 2 * num_key_value_heads) * head_dim`.
    ///
    /// 3072 on `Ring-mini-linear-2.0`, which is `(16 + 2 * 4) * 128`.
    pub fn attention_qkv_out_features(&self) -> usize {
        (self.num_attention_heads + 2 * self.num_kv_heads()) * self.head_dim()
    }

    /// Output width of a linear layer's fused `query_key_value`:
    /// `3 * num_attention_heads * linear_head_dim`, because the linear path is
    /// multi-head rather than grouped-query.
    ///
    /// 6144 on `Ring-mini-linear-2.0`, twice the full-attention layers' 3072 in
    /// the same checkpoint.
    pub fn linear_qkv_out_features(&self) -> usize {
        3 * self.num_attention_heads * self.linear_head_dim()
    }

    /// Channels per head that RoPE rotates on a full-attention layer:
    /// `int(head_dim * partial_rotary_factor)`.
    ///
    /// The cast saturates rather than wrapping, so a NaN `partial_rotary_factor`
    /// becomes 0 and an infinite one becomes `i32::MAX`;
    /// [`ModelArgs::validate_rope`] rejects both.
    pub fn rope_dims(&self) -> i32 {
        (self.partial_rotary_factor * self.head_dim() as f32) as i32
    }

    /// Channels per head that RoPE rotates on a linear layer. Derived from
    /// [`ModelArgs::linear_head_dim`], which can differ from
    /// [`ModelArgs::head_dim`].
    pub fn linear_rope_dims(&self) -> i32 {
        (self.partial_rotary_factor * self.linear_head_dim() as f32) as i32
    }

    pub fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts.unwrap_or(0)
    }

    pub fn has_routed_experts(&self) -> bool {
        self.num_experts() > 0
    }

    /// Width of the single wide shared MLP:
    /// `(moe_shared_expert_intermediate_size or moe_intermediate_size) *
    /// num_shared_experts`.
    ///
    /// 512 on `Ring-mini-linear-2.0` (`512 * 1`), which the checkpoint confirms:
    /// `mlp.shared_experts.gate_proj` dequantizes to `[512, 2048]`.
    pub fn shared_expert_intermediate_size(&self) -> usize {
        let shared_dim = self
            .moe_shared_expert_intermediate_size
            .unwrap_or_else(|| self.moe_intermediate_size());
        shared_dim.saturating_mul(self.num_shared_experts)
    }

    pub fn has_shared_expert(&self) -> bool {
        self.num_shared_experts > 0 && self.moe_router_enable_shared_expert
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.has_routed_experts() && layer_idx >= self.first_k_dense_replace
    }

    /// Whether layer `layer_idx` is a full-attention layer.
    ///
    /// Upstream's `DecoderLayer.__init__`:
    ///
    /// ```text
    /// (layer_idx + 1) % layer_group_size == 0
    ///   or layer_idx >= num_hidden_layers // layer_group_size * layer_group_size
    /// ```
    ///
    /// The second clause is the tail: when the stack does not divide evenly into
    /// groups, the leftover layers at the end are all global. It never fires on
    /// `Ring-mini-linear-2.0` (20 layers, group 5), where only the modulus
    /// selects layers 4, 9, 14 and 19.
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        if self.layer_group_size == 0 {
            return true;
        }
        (layer_idx + 1).is_multiple_of(self.layer_group_size)
            || layer_idx >= self.num_hidden_layers / self.layer_group_size * self.layer_group_size
    }

    /// Index of the first full-attention layer, whose KV cache carries the
    /// sequence offset every linear layer's RoPE reads.
    ///
    /// Upstream hardcodes `self.attn_idx = args.layer_group_size - 1` and indexes
    /// the cache list with it, which is out of range for a stack shorter than one
    /// group. Resolved by scan here so a short stack still finds its global layer
    /// (the tail clause guarantees one exists whenever there is any layer at all).
    pub fn global_layer_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| self.is_global_layer(i))
    }

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

    /// Stop tokens. `Ring-mini-linear-2.0` declares `eos_token_id` 156892.
    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdField::ids)
            .unwrap_or_default()
    }

    /// Reject a `config.json` that cannot describe a real Ling linear MoE,
    /// before any of its fields sizes an allocation, divides, or reaches an MLX
    /// kernel.
    pub fn validate(&self) -> Result<(), String> {
        // The zero checks come first: `0.is_multiple_of(0)` is true in Rust, so
        // a config with `hidden_size == num_attention_heads == 0` would pass the
        // divisibility check below and then divide by zero.
        if self.num_attention_heads == 0 || self.num_attention_heads > MAX_NUM_ATTENTION_HEADS {
            return Err(format!(
                "Ling linear MoE num_attention_heads ({}) must be between 1 and \
                 {MAX_NUM_ATTENTION_HEADS}",
                self.num_attention_heads
            ));
        }
        if self.hidden_size == 0 || self.hidden_size > MAX_HIDDEN_SIZE {
            return Err(format!(
                "Ling linear MoE hidden_size ({}) must be between 1 and {MAX_HIDDEN_SIZE}",
                self.hidden_size
            ));
        }
        // The linear layers derive their head width from this ratio and never
        // read `head_dim`, so an indivisible pair truncates their split offsets
        // even when an explicit `head_dim` is present.
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "Ling linear MoE hidden_size ({}) must be divisible by num_attention_heads ({}); \
                 the linear-attention layers derive their head width as \
                 hidden_size // num_attention_heads regardless of any explicit head_dim",
                self.hidden_size, self.num_attention_heads
            ));
        }
        let head_dim = self.head_dim();
        if head_dim == 0 || head_dim > MAX_HEAD_DIM {
            return Err(format!(
                "Ling linear MoE head_dim ({head_dim}) must be between 1 and {MAX_HEAD_DIM}"
            ));
        }

        let num_kv_heads = self.num_kv_heads();
        if num_kv_heads == 0 || num_kv_heads > self.num_attention_heads {
            return Err(format!(
                "Ling linear MoE num_key_value_heads ({num_kv_heads}) must be between 1 and \
                 num_attention_heads ({})",
                self.num_attention_heads
            ));
        }
        if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(format!(
                "Ling linear MoE num_attention_heads ({}) must be divisible by \
                 num_key_value_heads ({num_kv_heads}) for grouped-query attention",
                self.num_attention_heads
            ));
        }

        if self.num_hidden_layers == 0 || self.num_hidden_layers > MAX_NUM_HIDDEN_LAYERS {
            return Err(format!(
                "Ling linear MoE num_hidden_layers ({}) must be between 1 and \
                 {MAX_NUM_HIDDEN_LAYERS}",
                self.num_hidden_layers
            ));
        }
        if self.layer_group_size == 0 {
            return Err(
                "Ling linear MoE layer_group_size must be at least 1; it is the modulus that \
                 selects the full-attention layers, and 0 would divide by zero"
                    .to_string(),
            );
        }
        if self.group_norm_size == 0 {
            return Err(
                "Ling linear MoE group_norm_size must be at least 1; the linear path's output \
                 normalizer reshapes its last axis into that many groups"
                    .to_string(),
            );
        }
        let linear_width = self.num_attention_heads * self.linear_head_dim();
        if !linear_width.is_multiple_of(self.group_norm_size) {
            return Err(format!(
                "Ling linear MoE group_norm_size ({}) must divide the linear-attention output \
                 width ({linear_width} = num_attention_heads {} * head width {}); GroupRMSNorm \
                 reshapes that axis into group_norm_size equal groups",
                self.group_norm_size,
                self.num_attention_heads,
                self.linear_head_dim()
            ));
        }

        if self.intermediate_size == 0 || self.intermediate_size > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Ling linear MoE intermediate_size ({}) must be between 1 and \
                 {MAX_INTERMEDIATE_SIZE}",
                self.intermediate_size
            ));
        }
        let moe_intermediate = self.moe_intermediate_size();
        if moe_intermediate == 0 || moe_intermediate > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Ling linear MoE moe_intermediate_size ({moe_intermediate}) must be between 1 and \
                 {MAX_INTERMEDIATE_SIZE}"
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(format!(
                "Ling linear MoE vocab_size ({}) must be between 1 and {MAX_VOCAB_SIZE}",
                self.vocab_size
            ));
        }
        if self.max_position_embeddings == 0
            || self.max_position_embeddings > MAX_MAX_POSITION_EMBEDDINGS
        {
            return Err(format!(
                "Ling linear MoE max_position_embeddings ({}) must be between 1 and \
                 {MAX_MAX_POSITION_EMBEDDINGS}",
                self.max_position_embeddings
            ));
        }
        if self.first_k_dense_replace > self.num_hidden_layers {
            return Err(format!(
                "Ling linear MoE first_k_dense_replace ({}) must not exceed num_hidden_layers ({})",
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
    /// Same contract as [`crate::models::bailing_moe`]'s: `group_expert_select`
    /// computes `k = n_group - topk_group` and calls
    /// `argpartition(kth = k - 1)`, out of range as soon as
    /// `topk_group >= n_group`, and it reshapes the score row into `n_group`
    /// groups without checking that `num_experts` divides evenly. Unlike the
    /// dense family, `Ring-mini-linear-2.0` **does** reach the grouped branch:
    /// it declares `n_group: 8` with `topk_group: 4`.
    fn validate_routing(&self) -> Result<(), String> {
        if !self.has_routed_experts() {
            return Ok(());
        }
        let num_experts = self.num_experts();
        if num_experts > MAX_NUM_EXPERTS {
            return Err(format!(
                "Ling linear MoE num_experts ({num_experts}) must be between 1 and \
                 {MAX_NUM_EXPERTS}"
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > num_experts {
            return Err(format!(
                "Ling linear MoE num_experts_per_tok ({}) must be between 1 and num_experts \
                 ({num_experts}); the router selects that many indices out of a row of \
                 num_experts scores",
                self.num_experts_per_tok
            ));
        }
        if !matches!(self.score_function.as_str(), "softmax" | "sigmoid") {
            return Err(format!(
                "Ling linear MoE score_function ({:?}) must be \"softmax\" or \"sigmoid\"; an \
                 unrecognized value is rejected rather than silently falling back, because either \
                 fallback changes the routing distribution on every token while leaving the \
                 output finite and plausible",
                self.score_function
            ));
        }
        if self.n_group == 0 {
            return Err(
                "Ling linear MoE n_group must be at least 1 (1 disables grouped routing)"
                    .to_string(),
            );
        }
        if self.n_group > 1 {
            if !num_experts.is_multiple_of(self.n_group) {
                return Err(format!(
                    "Ling linear MoE num_experts ({num_experts}) must be divisible by n_group \
                     ({}); grouped routing reshapes the score row into n_group equal groups",
                    self.n_group
                ));
            }
            let experts_per_group = num_experts / self.n_group;
            if experts_per_group < 2 {
                return Err(format!(
                    "Ling linear MoE n_group ({}) leaves {experts_per_group} expert(s) per group; \
                     grouped routing scores each group by the sum of its top two experts, so a \
                     group must hold at least 2",
                    self.n_group
                ));
            }
            if self.topk_group == 0 || self.topk_group >= self.n_group {
                return Err(format!(
                    "Ling linear MoE topk_group ({}) must be between 1 and n_group - 1 ({}); the \
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
                "Ling linear MoE routed_scaling_factor ({}) must be finite; it multiplies every \
                 routed expert weight, so a non-finite value makes every MoE output NaN and that \
                 NaN reaches the logits without anything throwing",
                self.routed_scaling_factor
            ));
        }
        Ok(())
    }

    fn validate_shared_expert(&self) -> Result<(), String> {
        if self.num_shared_experts > MAX_NUM_SHARED_EXPERTS {
            return Err(format!(
                "Ling linear MoE num_shared_experts ({}) must not exceed {MAX_NUM_SHARED_EXPERTS}",
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
                    "Ling linear MoE shared-expert width overflows: \
                     moe_shared_expert_intermediate_size ({shared_dim}) * num_shared_experts ({})",
                    self.num_shared_experts
                )
            })?;
        if width == 0 || width > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Ling linear MoE shared-expert width ({width}, from a per-expert width of \
                 {shared_dim} times num_shared_experts {}) must be between 1 and \
                 {MAX_INTERMEDIATE_SIZE}",
                self.num_shared_experts
            ));
        }
        Ok(())
    }

    /// Reject RoPE parameters that MLX would throw on, on **either** head width.
    ///
    /// `mlx::core::fast::rope` requires `dims` positive, even, and no larger than
    /// the input's last axis. `fast_rope` crosses the cxx bridge as
    /// `UniquePtr<MlxArray>` rather than a `Result`, so a violation is an
    /// uncatchable `std::terminate` at the first forward pass. The linear and
    /// global layers can have different head widths, so both are checked.
    fn validate_rope(&self) -> Result<(), String> {
        if !self.partial_rotary_factor.is_finite() {
            return Err(format!(
                "Ling linear MoE partial_rotary_factor ({}) must be a finite number",
                self.partial_rotary_factor
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!(
                "Ling linear MoE rope_theta ({}) must be a finite positive number; RoPE \
                 exponentiates it per channel, so a zero, negative or non-finite base makes every \
                 rotated channel NaN and that NaN reaches the logits without anything throwing",
                self.rope_theta
            ));
        }
        for (label, head_dim, rope_dims) in [
            ("full-attention", self.head_dim(), self.rope_dims()),
            (
                "linear-attention",
                self.linear_head_dim(),
                self.linear_rope_dims(),
            ),
        ] {
            let dims = usize::try_from(rope_dims).unwrap_or(0);
            if dims == 0 || dims > head_dim {
                return Err(format!(
                    "Ling linear MoE {label} rotary width resolves to {rope_dims} for a head of \
                     width {head_dim}; it must be an even number between 2 and {head_dim}. MLX \
                     throws on a rope `dims` outside that range, and an MLX C++ exception \
                     crossing the cxx bridge is an uncatchable abort at the first forward pass \
                     rather than a load error. Check partial_rotary_factor ({}).",
                    self.partial_rotary_factor
                ));
            }
            if !dims.is_multiple_of(2) {
                return Err(format!(
                    "Ling linear MoE {label} rotary width resolves to an odd {rope_dims} for a \
                     head of width {head_dim}; RoPE rotates channel pairs, so the rope `dims` \
                     must be even, and MLX throws on an odd one. Check partial_rotary_factor ({}).",
                    self.partial_rotary_factor
                ));
            }
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
            "Ling linear MoE rope_scaling ({scaling:?}) is not implemented for this family; \
             upstream threads it into initialize_rope, but this loader always builds the plain \
             rotation, so accepting a scaled block would place every token at the wrong position \
             while the model still loaded and still generated fluent text. Only an absent, empty \
             or \"default\" block is accepted."
        ))
    }

    fn validate_norm_eps(&self) -> Result<(), String> {
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "Ling linear MoE rms_norm_eps ({}) must be a finite positive number; it is added \
                 to the mean square under an rsqrt, so a non-finite, negative or zero value makes \
                 every normalized hidden state NaN and that NaN reaches the logits without \
                 anything throwing",
                self.rms_norm_eps
            ));
        }
        Ok(())
    }

    /// Reject the two flags upstream declares and never reads.
    ///
    /// `norm_softmax` is inherited from `bailing_moe`, where it is equally dead;
    /// `use_rmsnorm` is new here and equally dead. Both are rejected rather than
    /// parsed and ignored, because a checkpoint that sets them is asking for
    /// behaviour no released implementation defines.
    fn validate_vendor_flags(&self) -> Result<(), String> {
        if self.norm_softmax {
            return Err(
                "Ling linear MoE config sets norm_softmax: true, which no released \
                 implementation defines. mlx-lm declares the field and never reads it. Loading \
                 anyway would silently ignore a flag the checkpoint author set on purpose."
                    .to_string(),
            );
        }
        if !self.use_rmsnorm {
            return Err(
                "Ling linear MoE config sets use_rmsnorm: false, which no released \
                 implementation defines. mlx-lm declares the field and never reads it: every \
                 normalizer in bailing_moe_linear.py is an nn.RMSNorm or a GroupRMSNorm \
                 unconditionally. Loading anyway would silently ignore a flag the checkpoint \
                 author set on purpose."
                    .to_string(),
            );
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
        .map_err(|e| format!("Ling linear MoE config.json: {e}"))
    }

    /// Whether mirroring upstream's unconditional routed scaling is observably
    /// different from honoring the flag. See
    /// [`crate::models::bailing_moe`]'s module docs.
    pub fn routed_scaling_flag_is_ignored_observably(&self) -> bool {
        self.has_routed_experts()
            && !self.moe_router_enable_routed_scaling
            && self.routed_scaling_factor != 1.0
    }

    fn warn_on_ignored_routed_scaling_flag(&self) {
        if self.routed_scaling_flag_is_ignored_observably() {
            eprintln!(
                "Ling linear MoE config sets moe_router_enable_routed_scaling: false with \
                 routed_scaling_factor {}. Upstream mlx-lm stores that flag and never reads it, \
                 so the scaling multiply is unconditional; this loader mirrors upstream and \
                 applies the factor anyway.",
                self.routed_scaling_factor
            );
        }
    }
}

// Decay schedule.

/// The ALiBi slope schedule for `n` heads, upstream's `_get_slopes`.
///
/// For a power-of-two head count this is
/// `2^(-(2^-(log2(n) - 3)) * (i + 1))` for `i` in `0..n`. For any other count
/// upstream falls back to the next power of two below `n`, then takes every
/// other slope of the *doubled* schedule to fill the remainder. Both branches
/// are reproduced exactly; the fallback is unreachable on
/// `Ring-mini-linear-2.0` (16 heads) and is the only part of this function a
/// real checkpoint does not exercise.
pub fn alibi_slopes(n: usize) -> Vec<f32> {
    fn power_of_2_slopes(n: usize) -> Vec<f32> {
        let ratio = 2f64.powf(-(2f64.powf(-((n as f64).log2() - 3.0))));
        (0..n).map(|i| ratio.powi(i as i32 + 1) as f32).collect()
    }

    if n == 0 {
        return Vec::new();
    }
    let log2n = (n as f64).log2();
    if log2n.fract() == 0.0 {
        return power_of_2_slopes(n);
    }
    let p = 1usize << (log2n.floor() as u32);
    let mut slopes = power_of_2_slopes(p);
    slopes.extend(power_of_2_slopes(2 * p).into_iter().step_by(2).take(n - p));
    slopes
}

/// The per-head decay exponent `g` for one linear layer, upstream's
/// `LinearAttention._get_slopes` tail.
///
/// ```text
/// layer_factor = 1 - max(0, layer_idx - 1) / max(1, num_hidden_layers - 1) + 1e-5
/// g            = -slopes * layer_factor
/// ```
///
/// Every entry is negative, so `exp(g)` lies in `(0, 1)` and the recurrence
/// decays. Layer 0 and layer 1 share the same factor (the `max(0, ...)` clamps
/// both to 0), and the last layer's factor is `1e-5`, a decay of essentially
/// zero retention.
pub fn layer_decay(num_heads: usize, layer_idx: usize, num_hidden_layers: usize) -> Vec<f32> {
    let denom = num_hidden_layers.max(2) - 1;
    let layer_pos = layer_idx.saturating_sub(1);
    let layer_factor = 1.0 - (layer_pos as f64 / denom as f64) + 1e-5;
    alibi_slopes(num_heads)
        .into_iter()
        .map(|s| (-(s as f64) * layer_factor) as f32)
        .collect()
}

// Gated linear attention.

/// One recurrence step, upstream's `recurrent_gla` body for `L == 1`.
///
/// `q`, `k` are `[B, H, 1, D]`, `v` is `[B, H, 1, Dv]`, `exp_g` is `[H, 1, 1]`,
/// and `state` is `[B, H, D, Dv]` or absent on the first token. Returns the
/// output `[B, H, 1, Dv]` and the new state.
///
/// This is the decode path. It is separate from [`gla_chunked`] only because a
/// chunk of length one costs about twice the ops for the same arithmetic, and
/// decode runs it once per layer per token.
pub fn gla_step(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    exp_g: &MlxArray,
    scale: f32,
    state: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let k_t = mlxcel_core::transpose_axes(k, &[0, 1, 3, 2]);
    let update = mlxcel_core::matmul(&k_t, v);
    let new_state = match state {
        Some(prev) => {
            let decayed = mlxcel_core::multiply(prev, exp_g);
            mlxcel_core::add(&decayed, &update)
        }
        None => update,
    };
    let q_scaled = mlxcel_core::multiply_scalar(q, scale);
    let out = mlxcel_core::matmul(&q_scaled, &new_state);
    (out, new_state)
}

/// Upstream's `recurrent_gla`, transcribed step for step.
///
/// This is the default path, for prefill as well as decode. It is
/// **numerically** upstream's, not merely mathematically: the state is carried
/// in the activation dtype and updated once per token, in the same order, so a
/// layer built from a real checkpoint reproduces the reference's own hidden
/// states rather than a reassociated approximation of them. On
/// `Ring-mini-linear-2.0` that is worth insisting on, because the stack
/// amplifies any perturbation hard: layer 3 multiplies its input magnitude by
/// ~23x, and the 256-expert top-8 router turns a sub-ulp difference into a
/// different expert. See [`gla_chunked`] for the faster reassociation and why it
/// is opt-in.
///
/// The per-step outputs are joined with one `stack` rather than a chain of
/// `concatenate` calls, which would copy the whole prefix `L` times.
pub fn gla_sequential(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    exp_g: &MlxArray,
    scale: f32,
    state: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let len = mlxcel_core::array_shape(q)[2];
    if len == 1 {
        return gla_step(q, k, v, exp_g, scale, state);
    }

    let mut carried: Option<UniquePtr<MlxArray>> = state.map(mlxcel_core::copy);
    let mut steps: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(len as usize);
    for t in 0..len {
        let (out, new_state) = gla_step(
            &slice_axis(q, 2, t, t + 1),
            &slice_axis(k, 2, t, t + 1),
            &slice_axis(v, 2, t, t + 1),
            exp_g,
            scale,
            carried.as_deref(),
        );
        carried = Some(new_state);
        // `[B, H, 1, Dv]` -> `[B, H, Dv]`, so the stack below restores the time
        // axis in one op instead of L pairwise concatenations.
        steps.push(mlxcel_core::squeeze_axis(&out, 2));
    }

    (
        mlxcel_core::stack_owned(&steps, 2),
        carried.expect("a non-empty sequence produces a state"),
    )
}

/// The closed form of `recurrent_gla` over a whole prefill, evaluated in chunks.
///
/// Because `g` does not depend on `t`, unrolling the recurrence gives
///
/// ```text
/// h_i = h_0 e^{g(i+1)} + sum_{j<=i} e^{g(i-j)} k_j^T v_j
/// y_i = scale * q_i h_i
///     = scale * [ (q_i h_0) e^{g(i+1)} + sum_{j<=i} e^{g(i-j)} (q_i . k_j) v_j ]
/// ```
///
/// The second term is an un-softmaxed causal attention whose scores carry a
/// multiplicative decay `e^{g(i-j)}`, which is one `[C, C]` matmul per chunk
/// instead of `C` sequential steps. The carried state advances once per chunk:
///
/// ```text
/// h_C = h_0 e^{gC} + sum_{j<C} e^{g(C-1-j)} k_j^T v_j
/// ```
///
/// Chunks are walked in a Rust loop rather than batched behind a pad, because
/// the decay applied to the carried state is a function of the chunk's real
/// length: padding the time axis would over-decay the returned state by exactly
/// the pad width, which is invisible until the next decode step reads it.
///
/// The decay factors are built in float32 and cast to the activation dtype at
/// the end, mirroring upstream's `mx.exp(g).astype(q.dtype)`.
///
/// # Why this is the default (#1040)
///
/// Measured against [`gla_sequential`] on `Ring-mini-linear-2.0-4bit`, this is
/// 2.6x to 4x faster on prefill (0.07s vs 0.29s at 101 tokens, 1.07s vs 2.81s at
/// 2048), because the intra-chunk sum lands in a matmul accumulator instead of a
/// bf16 running state.
///
/// It shipped opt-in first, on the argument that reassociating the sum is a
/// sub-ulp change per layer that this stack nonetheless amplifies (layer 3
/// multiplies its input magnitude by ~23x, and the 256-expert top-8 router turns
/// a sub-ulp score difference into a different expert), so the two paths decode
/// different continuations from one checkpoint. That much is true. What the
/// argument assumed, and #1040 measured, is that "different" meant "no better".
/// It does not: the accumulator is not just faster but measurably more accurate,
/// and the gap grows with how long the sequential recurrence runs in bf16.
///
/// Teacher-forced perplexity on a WikiText-2 excerpt, both paths, same
/// checkpoint (`examples/perplexity`, deterministic, so a repeat of a
/// configuration reproduces its number exactly):
///
/// | window x windows | tokens | sequential | chunked | delta |
/// |---|---|---|---|---|
/// | 128 x 1 | 128 | 115.83 | 114.09 | -1.50% |
/// | 1024 x 1 | 1024 | 39.39 | 36.39 | **-7.62%** |
/// | 256 x 32 | 8192 | 155.47 | 151.42 | -2.60% |
/// | 512 x 32 | 16384 | 85.96 | 74.14 | **-13.75%** |
///
/// Chunked is better in every configuration, and within a fixed scoring shape
/// the advantage grows with the window (128 to 1024 at one window, 256 to 512
/// across 32), which is what a compounding bf16 accumulation error predicts.
/// Absolute perplexity is not comparable across the rows because each scores a
/// different corpus slice; only the within-row comparison is.
///
/// So the accuracy objection that kept this opt-in is answered in the opposite
/// direction from the one it feared, and the faster path is also the better one.
/// The cost is that a checkpoint decoded here no longer matches the same
/// checkpoint decoded under mlx-lm token for token. Set
/// `MLXCEL_BAILING_LINEAR_CHUNKED_PREFILL=0` (also `false`/`off`/`no`,
/// case-insensitive) to restore upstream's sequential recurrence when that
/// comparability is what you need.
pub fn chunked_prefill_enabled() -> bool {
    chunked_prefill_enabled_from(
        std::env::var("MLXCEL_BAILING_LINEAR_CHUNKED_PREFILL")
            .ok()
            .as_deref(),
    )
}

/// The env-var reading of [`chunked_prefill_enabled`], split out so the
/// default and the off-spellings are testable without mutating the process
/// environment (which would need `test_support::env_lock`).
///
/// Mirrors `switch_layers::fused_moe_enabled_from`: on unless explicitly
/// switched off, so a pre-#1040 `=1` still selects chunked rather than
/// becoming a surprise opt-out.
pub(crate) fn chunked_prefill_enabled_from(value: Option<&str>) -> bool {
    match value {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => true,
    }
}

/// See [`chunked_prefill_enabled`] for when this runs and why it is the
/// default.
pub fn gla_chunked(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    scale: f32,
    state: Option<&MlxArray>,
    chunk: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let dtype = mlxcel_core::array_dtype(q);
    let f32_dtype = mlxcel_core::dtype::FLOAT32;
    let shape = mlxcel_core::array_shape(q);
    let total = shape[2];
    let num_heads = shape[1];

    // `[H] -> [1, H, 1, 1]`, so every broadcast below aligns on the head axis.
    let g4 = mlxcel_core::reshape(g, &[1, num_heads, 1, 1]);

    let q_scaled = mlxcel_core::multiply_scalar(q, scale);
    let mut carried: Option<UniquePtr<MlxArray>> = state.map(mlxcel_core::copy);
    let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::new();

    let mut start = 0;
    while start < total {
        let len = chunk.min(total - start);
        let q_c = slice_axis(&q_scaled, 2, start, start + len);
        let k_c = slice_axis(k, 2, start, start + len);
        let v_c = slice_axis(v, 2, start, start + len);

        // decay[h, r, j] = exp(g_h * (r - j)) for j <= r, 0 otherwise. Built as
        // exp of a non-positive product so it cannot overflow: `g` is negative
        // and the clamped delta is non-negative.
        let pos = mlxcel_core::astype(&mlxcel_core::arange_i32(0, len, 1), f32_dtype);
        let rows = mlxcel_core::reshape(&pos, &[1, 1, len, 1]);
        let cols = mlxcel_core::reshape(&pos, &[1, 1, 1, len]);
        let delta = mlxcel_core::subtract(&rows, &cols);
        let zero = mlxcel_core::full_f32(&[1], 0.0, f32_dtype);
        let delta = mlxcel_core::maximum(&delta, &zero);
        let decay = mlxcel_core::exp(&mlxcel_core::multiply(&g4, &delta));
        let causal = mlxcel_core::tril(&mlxcel_core::ones(&[len, len], f32_dtype), 0);
        let decay = mlxcel_core::astype(&mlxcel_core::multiply(&decay, &causal), dtype);

        // Intra-chunk: (Q K^T . decay) V.
        let k_t = mlxcel_core::transpose_axes(&k_c, &[0, 1, 3, 2]);
        let scores = mlxcel_core::multiply(&mlxcel_core::matmul(&q_c, &k_t), &decay);
        let mut out = mlxcel_core::matmul(&scores, &v_c);

        // Inter-chunk: the state carried in decays by exp(g * (r + 1)).
        if let Some(prev) = carried.as_ref() {
            let steps = mlxcel_core::astype(&mlxcel_core::arange_i32(1, len + 1, 1), f32_dtype);
            let steps = mlxcel_core::reshape(&steps, &[1, 1, len, 1]);
            let carry_decay = mlxcel_core::astype(
                &mlxcel_core::exp(&mlxcel_core::multiply(&g4, &steps)),
                dtype,
            );
            let carried_out = mlxcel_core::matmul(&q_c, prev);
            out = mlxcel_core::add(&out, &mlxcel_core::multiply(&carried_out, &carry_decay));
        }
        outputs.push(out);

        // State after this chunk: h * exp(g * len) + sum_j exp(g * (len-1-j)) k_j^T v_j.
        let tail = mlxcel_core::astype(&mlxcel_core::arange_i32(len - 1, -1, -1), f32_dtype);
        let tail = mlxcel_core::reshape(&tail, &[1, 1, len, 1]);
        let tail_decay =
            mlxcel_core::astype(&mlxcel_core::exp(&mlxcel_core::multiply(&g4, &tail)), dtype);
        let k_weighted = mlxcel_core::multiply(&k_c, &tail_decay);
        let update = mlxcel_core::matmul(
            &mlxcel_core::transpose_axes(&k_weighted, &[0, 1, 3, 2]),
            &v_c,
        );
        carried = Some(match carried {
            Some(prev) => {
                let span = mlxcel_core::full_f32(&[1], len as f32, f32_dtype);
                let chunk_decay = mlxcel_core::astype(
                    &mlxcel_core::exp(&mlxcel_core::multiply(&g4, &span)),
                    dtype,
                );
                mlxcel_core::add(&mlxcel_core::multiply(&prev, &chunk_decay), &update)
            }
            None => update,
        });

        start += len;
    }

    let out = if outputs.len() == 1 {
        outputs.pop().expect("one chunk")
    } else {
        let mut joined = outputs.remove(0);
        for next in outputs.iter() {
            joined = mlxcel_core::concatenate(&joined, next, 2);
        }
        joined
    };
    (out, carried.expect("at least one chunk produces a state"))
}

// Normalization.

/// Upstream's `GroupRMSNorm`.
///
/// Splits the last axis into `groups` equal parts, RMS-normalizes each part
/// **without** a weight, flattens back, and only then multiplies by the
/// full-width weight. A plain RMSNorm over the whole axis is a different
/// function whenever `groups > 1`, and produces finite output either way.
pub struct GroupRMSNorm {
    pub weight: UniquePtr<MlxArray>,
    /// A ones vector of the per-group width. `fast_rms_norm` takes a weight
    /// rather than an `Option`, and upstream normalizes weightlessly here, so
    /// the identity weight is materialized once at construction.
    identity: UniquePtr<MlxArray>,
    pub groups: i32,
    pub eps: f32,
}

impl GroupRMSNorm {
    pub fn new(weight: UniquePtr<MlxArray>, groups: i32, eps: f32) -> Result<Self, String> {
        let shape = mlxcel_core::array_shape(&weight);
        if shape.len() != 1 {
            return Err(format!(
                "GroupRMSNorm weight shape {shape:?}: expected a 1-D vector"
            ));
        }
        if groups <= 0 || shape[0] % groups != 0 {
            return Err(format!(
                "GroupRMSNorm width {} is not divisible by groups {groups}",
                shape[0]
            ));
        }
        let per_group = shape[0] / groups;
        let dtype = mlxcel_core::array_dtype(&weight);
        Ok(Self {
            weight,
            identity: mlxcel_core::ones(&[per_group], dtype),
            groups,
            eps,
        })
    }

    /// `x` is `[..., width]`; the result has the same shape.
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let width = *shape.last().expect("GroupRMSNorm input is not a scalar");
        let per_group = width / self.groups;

        let mut grouped_shape = shape.clone();
        grouped_shape.pop();
        grouped_shape.push(self.groups);
        grouped_shape.push(per_group);

        let grouped = mlxcel_core::reshape(x, &grouped_shape);
        let normed = mlxcel_core::fast_rms_norm(&grouped, &self.identity, self.eps);
        let flat = mlxcel_core::reshape(&normed, &shape);
        mlxcel_core::multiply(&self.weight, &flat)
    }
}

fn rms_norm_from_weights(weights: &WeightMap, key: &str, eps: f32) -> Result<RMSNorm, String> {
    let weight = weights
        .get(key)
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    Ok(RMSNorm::new(mlxcel_core::copy(weight), eps))
}

// Attention.

/// Full-attention layer: fused GQA `query_key_value`, optional per-head QK norm,
/// RoPE, `dense` output projection.
///
/// Structurally the same block as [`crate::models::bailing_moe::Attention`]. It
/// is restated rather than reused because that type's constructor is keyed to
/// the dense family's `ModelArgs`, and because the head width here comes from an
/// explicit `head_dim` the dense family does not have.
pub struct Attention {
    pub query_key_value: UnifiedLinear,
    pub dense: UnifiedLinear,
    pub query_layernorm: Option<RMSNorm>,
    pub key_layernorm: Option<RMSNorm>,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_traditional: bool,
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

        let qkv = self.query_key_value.forward(x);
        let q = mlxcel_core::slice_last_dim(&qkv, 0, q_size);
        let k = mlxcel_core::slice_last_dim(&qkv, q_size, q_size + kv_size);
        let v = mlxcel_core::slice_last_dim(&qkv, q_size + kv_size, q_size + 2 * kv_size);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let q = match &self.query_layernorm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.key_layernorm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(
            &q,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );

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
            rope_traditional: args.rope_traditional,
            rope_base: args.rope_theta,
        })
    }
}

/// Recurrent state of one linear-attention layer.
///
/// `state` is `[B, H, head_dim, head_dim]`, absent until the first token. The
/// offset is tracked so a restored snapshot and the model's own bookkeeping
/// agree; RoPE on this path reads the *global* layers' offset, not this one.
pub struct LinearAttentionCache {
    pub state: Option<UniquePtr<MlxArray>>,
    pub offset: i32,
}

impl LinearAttentionCache {
    pub fn new() -> Self {
        Self {
            state: None,
            offset: 0,
        }
    }
}

impl Default for LinearAttentionCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-layer cache: a KV cache on the global layers, a recurrent state on the
/// linear ones.
pub enum BailingLinearCache {
    Attention(KVCache),
    Linear(LinearAttentionCache),
}

impl BailingLinearCache {
    pub fn offset(&self) -> i32 {
        match self {
            Self::Attention(kv) => kv.offset,
            Self::Linear(linear) => linear.offset,
        }
    }

    fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        match self {
            Self::Attention(kv) => {
                super::recurrent_snapshot::push_optional(
                    snapshot,
                    format!("{prefix}.attention.keys"),
                    &kv.keys,
                );
                super::recurrent_snapshot::push_optional(
                    snapshot,
                    format!("{prefix}.attention.values"),
                    &kv.values,
                );
            }
            Self::Linear(linear) => super::recurrent_snapshot::push_optional(
                snapshot,
                format!("{prefix}.linear.state"),
                &linear.state,
            ),
        }
    }

    fn restore_from(&mut self, snapshot: &mlxcel_core::generate::ModelStateSnapshot, prefix: &str) {
        match self {
            Self::Attention(kv) => {
                kv.keys = super::recurrent_snapshot::restore_optional(
                    snapshot,
                    format!("{prefix}.attention.keys"),
                );
                kv.values = super::recurrent_snapshot::restore_optional(
                    snapshot,
                    format!("{prefix}.attention.values"),
                );
                kv.offset = snapshot.token_len() as i32;
            }
            Self::Linear(linear) => {
                linear.state = super::recurrent_snapshot::restore_optional(
                    snapshot,
                    format!("{prefix}.linear.state"),
                );
                linear.offset = snapshot.token_len() as i32;
            }
        }
    }
}

/// Linear-attention layer: fused MHA `query_key_value`, optional per-head QK
/// norm, RoPE, the GLA recurrence, [`GroupRMSNorm`], a sigmoid output gate, and
/// the `dense` output projection.
pub struct LinearAttention {
    pub query_key_value: UnifiedLinear,
    pub dense: UnifiedLinear,
    pub g_proj: UnifiedLinear,
    pub g_norm: GroupRMSNorm,
    pub query_layernorm: Option<RMSNorm>,
    pub key_layernorm: Option<RMSNorm>,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_traditional: bool,
    pub rope_base: f32,
    /// `g`, the constant per-head decay exponent, `[num_heads]` in float32.
    decay: UniquePtr<MlxArray>,
    /// `exp(g)` reshaped to `[1, H, 1, 1]` for the decode step, cached so the
    /// hot path does not rebuild it per token.
    exp_decay: UniquePtr<MlxArray>,
    /// Whether prefill takes [`gla_chunked`]. Resolved once at construction:
    /// reading the environment inside `forward` would put a syscall in the
    /// decode path. See [`chunked_prefill_enabled`].
    chunked_prefill: bool,
}

impl LinearAttention {
    /// `offset` is the sequence position of the first token in `x`, read from a
    /// global layer's KV cache exactly as upstream does. The recurrence has no
    /// position of its own, so it cannot supply one.
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut LinearAttentionCache,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let width = self.num_heads * self.head_dim;

        let qkv = self.query_key_value.forward(x);
        let q = mlxcel_core::slice_last_dim(&qkv, 0, width);
        let k = mlxcel_core::slice_last_dim(&qkv, width, 2 * width);
        let v = mlxcel_core::slice_last_dim(&qkv, 2 * width, 3 * width);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let q = match &self.query_layernorm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.key_layernorm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

        let q = mlxcel_core::fast_rope(
            &q,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.rope_dims,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );

        let (out, new_state) = if l > 1 && self.chunked_prefill {
            gla_chunked(
                &q,
                &k,
                &v,
                &self.decay,
                self.scale,
                cache.state.as_deref(),
                GLA_CHUNK,
            )
        } else {
            let exp_g = mlxcel_core::astype(&self.exp_decay, mlxcel_core::array_dtype(&q));
            gla_sequential(&q, &k, &v, &exp_g, self.scale, cache.state.as_deref())
        };
        cache.state = Some(new_state);
        cache.offset += l;

        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, width]);
        let gate = mlxcel_core::sigmoid(&self.g_proj.forward(x));
        let gated = mlxcel_core::multiply(&self.g_norm.forward(&out), &gate);
        self.dense.forward(&gated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.linear_head_dim() as i32;

        let query_key_value = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.query_key_value"),
            group_size,
            bits,
        )?;
        let dense =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.dense"), group_size, bits)?;
        let g_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.g_proj"), group_size, bits)?;

        let g_norm_key = format!("{prefix}.g_norm.weight");
        let g_norm_weight = weights
            .get(&g_norm_key)
            .ok_or_else(|| format!("Weight not found: {g_norm_key}"))?;
        let g_norm = GroupRMSNorm::new(
            mlxcel_core::copy(g_norm_weight),
            args.group_norm_size as i32,
            args.rms_norm_eps,
        )?;

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

        let decay_host = layer_decay(args.num_attention_heads, layer_idx, args.num_hidden_layers);
        let decay = mlxcel_core::from_slice_f32(&decay_host, &[args.num_attention_heads as i32]);
        let exp_decay = mlxcel_core::exp(&mlxcel_core::reshape(
            &decay,
            &[1, args.num_attention_heads as i32, 1, 1],
        ));

        Ok(Self {
            query_key_value,
            dense,
            g_proj,
            g_norm,
            query_layernorm,
            key_layernorm,
            num_heads: args.num_attention_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: args.linear_rope_dims(),
            rope_traditional: args.rope_traditional,
            rope_base: args.rope_theta,
            decay,
            exp_decay,
            chunked_prefill: chunked_prefill_enabled(),
        })
    }
}

/// Either attention flavour, per layer.
pub enum AttentionKind {
    Full(Box<Attention>),
    Linear(Box<LinearAttention>),
}

// Feed-forward.

/// Either the dense prefix MLP or the sparse block, per layer.
///
/// Both variants are [`crate::models::bailing_moe`] types: the FFN half of this
/// family is byte-identical to the dense one, down to the router rename and the
/// selection-only expert bias.
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

/// Build a `bailing_moe` SwiGLU MLP at an explicit prefix.
///
/// [`BailingMoeMLP::from_weights`] is keyed to the dense family's `ModelArgs`,
/// so the three projections are loaded here and the struct assembled directly.
fn load_mlp(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<BailingMoeMLP, String> {
    let group_size = args.group_size();
    let bits = args.bits();
    Ok(BailingMoeMLP {
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

/// Build the `bailing_moe` router for one MoE layer.
///
/// Mirrors [`BailingMoeGate::from_weights`] against this family's `ModelArgs`,
/// including the zero-initialized `expert_bias` upstream falls back to when the
/// flag is set but the checkpoint ships no tensor, and the saturating
/// `topk_group` conversion that keeps an unreachable field from holding an
/// out-of-range value.
fn load_gate(
    weights: &WeightMap,
    args: &ModelArgs,
    mlp_prefix: &str,
) -> Result<BailingMoeGate, String> {
    let prefix = router_prefix(weights, mlp_prefix)?;
    let gate_proj = UnifiedLinear::from_weights(weights, &prefix, args.group_size(), args.bits())?;

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

    Ok(BailingMoeGate {
        gate_proj,
        expert_bias,
        top_k: args.num_experts_per_tok as i32,
        n_group: args.n_group as i32,
        topk_group: i32::try_from(args.topk_group).unwrap_or(i32::MAX),
        routed_scaling_factor: args.routed_scaling_factor,
        norm_topk_prob: args.norm_topk_prob,
        score_function: args.score_function(),
    })
}

fn load_sparse_block(
    weights: &WeightMap,
    args: &ModelArgs,
    prefix: &str,
) -> Result<BailingMoeSparseBlock, String> {
    Ok(BailingMoeSparseBlock {
        gate: load_gate(weights, args, prefix)?,
        switch_mlp: SwitchGLU::from_weights(
            weights,
            &format!("{prefix}.switch_mlp"),
            args.group_size(),
            args.bits(),
        )?,
        shared_experts: if args.has_shared_expert() {
            Some(load_mlp(
                weights,
                args,
                &format!("{prefix}.shared_experts"),
            )?)
        } else {
            None
        },
    })
}

// Decoder layer and model.

pub struct DecoderLayer {
    pub attention: AttentionKind,
    pub mlp: FeedForward,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub fn is_global(&self) -> bool {
        matches!(self.attention, AttentionKind::Full(_))
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut BailingLinearCache,
        mask: Option<&MlxArray>,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = match (&self.attention, cache) {
            (AttentionKind::Full(attn), BailingLinearCache::Attention(kv)) => {
                attn.forward(&normed, kv, mask)
            }
            (AttentionKind::Linear(attn), BailingLinearCache::Linear(state)) => {
                attn.forward(&normed, state, offset)
            }
            // Unreachable: every cache vector is built by
            // `make_internal_caches` from the same `is_global` predicate that
            // chose the attention, and `restore_sequence_state` rebuilds it the
            // same way before restoring into it.
            //
            // The fallback is a *silent* wrong answer: a zero attention output
            // turns the layer into `h = x`, which stays finite and decodes
            // fluent text. The `debug_assert` makes a future regression loud
            // under a dev-profile `cargo test`, but note it does NOT fire under
            // the project's own gate: `[profile.test-fast]` inherits from
            // release, where `debug-assertions` defaults off. The invariant is
            // held by construction at both producers rather than by this check.
            (attention, cache) => {
                debug_assert!(
                    false,
                    "cache flavour does not match the layer: attention is {}, cache is {}",
                    if matches!(attention, AttentionKind::Full(_)) {
                        "full"
                    } else {
                        "linear"
                    },
                    if matches!(cache, BailingLinearCache::Attention(_)) {
                        "attention"
                    } else {
                        "linear"
                    },
                );
                mlxcel_core::zeros_like(&normed)
            }
        };
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
        let attention_prefix = format!("{prefix}.attention");
        let attention = if args.is_global_layer(layer_idx) {
            AttentionKind::Full(Box::new(Attention::from_weights(
                weights,
                args,
                &attention_prefix,
            )?))
        } else {
            AttentionKind::Linear(Box::new(LinearAttention::from_weights(
                weights,
                args,
                &attention_prefix,
                layer_idx,
            )?))
        };

        let mlp = if args.is_moe_layer(layer_idx) {
            FeedForward::Sparse(Box::new(load_sparse_block(
                weights,
                args,
                &format!("{prefix}.mlp"),
            )?))
        } else {
            FeedForward::Dense(load_mlp(weights, args, &format!("{prefix}.mlp"))?)
        };

        Ok(Self {
            attention,
            mlp,
            input_layernorm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.input_layernorm.weight"),
                args.rms_norm_eps,
            )?,
            post_attention_layernorm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
                args.rms_norm_eps,
            )?,
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

/// Check one `[out_features, in_features]` projection against the config.
///
/// The input axis is checked through the scales rather than the packed
/// `.weight`, because a packed width does not fix a real width without a bit
/// count, and `Ring-mini-linear-2.0` quantizes different tensors at different
/// bit counts in the same checkpoint.
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
            " That is the [in, out] orientation; every projection here is an nn.Linear, so a \
             genuine checkpoint is already [out, in] and must not be transposed."
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

/// Check the routed experts of one MoE layer, in whichever of the two layouts
/// the checkpoint uses.
///
/// The per-expert form is checked for **every** index below `num_experts`
/// rather than only index 0, because `stack_individual_experts` gathers
/// contiguously from 0 until the first gap and would otherwise register a short
/// stack the router can index past. `Ring-mini-linear-2.0` ships the stacked
/// form.
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
///
/// The linear and global layers are checked against **different** fused-QKV
/// widths, which is the check a shared `bailing_moe` validator could not make:
/// on `Ring-mini-linear-2.0` a linear layer's `query_key_value` is `[6144, 2048]`
/// and a global layer's is `[3072, 2048]`, and MLX's `slice` clamps an
/// out-of-range stop rather than throwing, so validating both against one width
/// would let a mislabeled layer silently read the wrong channels.
pub fn validate_weights(weights: &WeightMap, args: &ModelArgs) -> Result<(), String> {
    let hidden = args.hidden_size;
    let group_size = args.group_size();
    let bits = args.bits();

    validate_norm(weights, "model.norm.weight", hidden)?;
    validate_quantized_packing(weights, "model.word_embeddings", hidden, group_size, bits)?;

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
        let is_global = args.is_global_layer(layer);

        let (qkv_out, head_dim, out_width) = if is_global {
            (
                args.attention_qkv_out_features(),
                args.head_dim(),
                args.num_attention_heads * args.head_dim(),
            )
        } else {
            (
                args.linear_qkv_out_features(),
                args.linear_head_dim(),
                args.num_attention_heads * args.linear_head_dim(),
            )
        };

        validate_projection(
            weights,
            &format!("{attention}.query_key_value"),
            qkv_out,
            hidden,
            group_size,
            bits,
        )?;
        validate_projection(
            weights,
            &format!("{attention}.dense"),
            hidden,
            out_width,
            group_size,
            bits,
        )?;
        if !is_global {
            validate_projection(
                weights,
                &format!("{attention}.g_proj"),
                out_width,
                hidden,
                group_size,
                bits,
            )?;
            validate_norm(weights, &format!("{attention}.g_norm.weight"), out_width)?;
        }
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
                        "unexpected {mlp}.gate.expert_bias shape {bias_shape:?}: expected [{}]; \
                         it is added to a row of that many router scores",
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

/// Build the output head, applying `norm_head` when the config asks for it.
///
/// Same contract as [`crate::models::bailing_moe`]'s: a quantized head under
/// `norm_head` is refused, because the stored `.weight` is a packed `uint32`
/// bit field and dividing it by a column norm would corrupt every logit while
/// leaving the checkpoint apparently loadable.
fn load_lm_head(weights: &WeightMap, args: &ModelArgs) -> Result<UnifiedLinear, String> {
    let group_size = args.group_size();
    let bits = args.bits();
    if !args.norm_head {
        return UnifiedLinear::from_weights(weights, "lm_head", group_size, bits);
    }
    if weights.contains_key("lm_head.scales") {
        return Err(
            "Ling linear MoE config sets norm_head: true, but lm_head is quantized. norm_head \
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
        crate::models::bailing_moe::normalize_lm_head_weight(weight),
    );
    if let Some(bias) = weights.get("lm_head.bias") {
        normalized.insert("lm_head.bias".to_string(), mlxcel_core::copy(bias));
    }
    UnifiedLinear::from_weights(&normalized, "lm_head", group_size, bits)
}

/// Ant Group Ling / Ring linear-attention MoE.
pub struct BailingMoeLinearModel {
    /// Token table. Bailing names it `word_embeddings`, not `embed_tokens`.
    pub word_embeddings: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    /// Index of the layer whose KV cache carries the sequence offset. `None`
    /// only for a stack with no global layer, which
    /// [`ModelArgs::is_global_layer`]'s tail clause makes impossible for a
    /// non-empty stack.
    global_layer_index: Option<usize>,
    eos_token_ids: Vec<i32>,
    /// The linear layers hold a recurrent state and the global ones a KV cache,
    /// which does not fit the trait's homogeneous `&mut [KVCache]`. The model
    /// owns the heterogeneous cache here and persists it across forward calls;
    /// recreating it per call would make decode stateless.
    sequence_state: ModelOwnedSequenceState<BailingLinearCache>,
}

impl BailingMoeLinearModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [BailingLinearCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Upstream reads the offset from a global layer's KV cache once, before
        // any layer runs, and passes the same value to every linear layer. The
        // linear recurrence has no position of its own, so this is the only
        // source of one.
        let offset = self
            .global_layer_index
            .and_then(|idx| caches.get(idx))
            .map(BailingLinearCache::offset)
            .or_else(|| caches.first().map(BailingLinearCache::offset))
            .unwrap_or(0);

        let mut h = self.word_embeddings.forward(input_ids);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, cache, mask, offset);
        }
        let h = self.norm.forward(&h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.word_embeddings.as_linear(&h),
        }
    }

    fn make_internal_caches(&self) -> Vec<BailingLinearCache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.is_global() {
                    BailingLinearCache::Attention(KVCache::new())
                } else {
                    BailingLinearCache::Linear(LinearAttentionCache::new())
                }
            })
            .collect()
    }

    /// Shared forward that routes through the model-owned heterogeneous cache.
    ///
    /// `seq_id = None` uses the fallback cache (offline CLI / benchmark), reset
    /// on prefill so a new prompt does not inherit stale state. `Some(id)` uses
    /// the scheduler's per-sequence state so concurrent server requests stay
    /// isolated.
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
            |caches| {
                let mask = if seq_len > 1 {
                    let offset = self
                        .global_layer_index
                        .and_then(|idx| caches.get(idx))
                        .map(BailingLinearCache::offset)
                        .unwrap_or(0);
                    Some(create_causal_mask(seq_len, offset))
                } else {
                    None
                };
                self.forward(input, caches, mask.as_deref())
            },
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

        let internal_caches: Vec<BailingLinearCache> = layers
            .iter()
            .map(|layer| {
                if layer.is_global() {
                    BailingLinearCache::Attention(KVCache::new())
                } else {
                    BailingLinearCache::Linear(LinearAttentionCache::new())
                }
            })
            .collect();

        Ok(Self {
            word_embeddings,
            layers,
            norm,
            lm_head,
            global_layer_index: args.global_layer_index(),
            eos_token_ids: args.eos_token_ids(),
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
        })
    }
}

// LanguageModel trait implementation.

impl LanguageModel for BailingMoeLinearModel {
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

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<mlxcel_core::generate::ModelStateSnapshot> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot =
                    mlxcel_core::generate::ModelStateSnapshot::new("bailing_moe_linear", token_len);
                for (idx, cache) in state.iter().enumerate() {
                    cache.snapshot_into(&mut snapshot, &format!("layer{idx}"));
                }
                snapshot
            })
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != "bailing_moe_linear" {
            return Err(format!(
                "cannot restore {} snapshot into bailing_moe_linear",
                snapshot.family()
            ));
        }
        let mut state = self.make_internal_caches();
        for (idx, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{idx}"));
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Compatibility only: the real state lives in `sequence_state`.
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_batching(&self) -> bool {
        // No batched (multi-sequence-per-forward) decode: the GLA recurrence has
        // no batched path. Concurrent server requests are still isolated per
        // sequence through the model-owned `sequence_state`.
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "bailing_moe_linear_tests.rs"]
mod tests;
