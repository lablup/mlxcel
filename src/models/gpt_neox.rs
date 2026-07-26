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

//! GPT-NeoX (`gpt_neox`) text model implementation using mlxcel-core.
//!
//! Reference: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gpt_neox.py>
//!
//! GPT-NeoX is the EleutherAI decoder behind the Pythia suite and the
//! NeoX-derived checkpoints (Dolly, RedPajama-INCITE, StableLM Alpha). It sits
//! next to the GPT-2 lineage in this tree because it shares `LayerNorm` with
//! bias and a fused QKV projection, but the three things that define it are its
//! own:
//!
//! - **Interleaved per-head QKV.** `query_key_value.weight` is
//!   `[3 * hidden_size, hidden_size]` in plain `nn.Linear` orientation, but its
//!   output is *not* three contiguous `hidden_size` blocks. Upstream reshapes
//!   the projection output to `(..., num_heads, 3 * head_dim)` and splits *that*
//!   last axis, so the layout is head-major: `[q_0 | k_0 | v_0 | q_1 | k_1 |
//!   v_1 | ...]`, each head contributing `3 * head_dim` contiguous channels. On
//!   `EleutherAI/pythia-1b` that is 8 heads of width 256, so 768 channels per
//!   head. See [`split_interleaved_qkv`], which is the only place this matters
//!   and the one place a mistake would not surface: a flat three-way split of
//!   the last axis (the GPT-2 pattern) produces Q, K and V of exactly the right
//!   shape from the wrong channels, so the model loads, decodes and emits
//!   fluent-looking English. [`ModelArgs::interleaved_qkv_channel_offsets`]
//!   states the same layout as a pure function of the config.
//! - **Partial RoPE.** Only `int(head_dim * rotary_pct)` channels of each head
//!   are rotated; the rest pass through untouched. Pythia uses `rotary_pct`
//!   0.25, so 64 of 256 channels rotate and 192 do not. mlxcel expresses this
//!   with the `dims` argument of `fast_rope`, the same way
//!   [`crate::models::phi`] derives `rope_dims` from `partial_rotary_factor`.
//!   `traditional` is false.
//! - **Optional parallel residual.** With `use_parallel_residual: true` (every
//!   Pythia checkpoint) the attention and MLP sub-layers both read the *same*
//!   pre-norm input and their outputs are summed into the residual:
//!   `x + attn(input_layernorm(x)) + mlp(post_attention_layernorm(x))`. With
//!   `false` the block is the ordinary chained form, where the MLP norm reads
//!   the post-attention residual. Both are implemented and unit-tested; see
//!   [`TransformerBlock::forward`].
//!
//! The remaining differences from the GPT-2 block: there are no learned
//! absolute position embeddings (partial RoPE carries position instead), the
//! block has two norms plus a final `final_layer_norm`, the MLP is
//! `dense_4h_to_h(gelu(dense_h_to_4h(x)))` with no gate/up pattern, and the
//! output head `embed_out` is untied and lives at the top level of the
//! checkpoint.
//!
//! Three registered PyTorch buffers reach the checkpoint and must never be
//! loaded; see [`strip_registered_buffers`].

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use crate::models::gpt2::{dim_eq, layer_norm_from_weights, validate_embedding_table};

// Configuration.

/// GPT-NeoX `config.json`.
///
/// Field names follow the HuggingFace `GPTNeoXConfig`, which is the modern
/// `hidden_size` / `num_attention_heads` naming rather than the GPT-2 one.
///
/// Two fields upstream's `ModelArgs` carries are deliberately absent.
/// `num_key_value_heads` is vestigial: upstream's `__post_init__` defaults it to
/// `num_attention_heads` but its `Attention` never reads it, and HuggingFace
/// `GPTNeoXAttention` has no grouped-query concept at all. The fused
/// `query_key_value` projection is `3 * hidden_size` wide and is reshaped to
/// `(num_heads, 3 * head_dim)`, so Q, K and V always carry the same head count.
/// `attention_bias` / `mlp_bias` style switches do not exist in this family
/// either: every projection ships a bias, and this loader takes the bias from
/// the checkpoint when the tensor is present, which is the weight-presence
/// convention used elsewhere in this tree.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    /// MLP hidden width. `null` or absent means `4 * hidden_size`, which is what
    /// upstream hardcodes and what every published checkpoint declares anyway
    /// (Pythia 1B: 8192 for `hidden_size` 2048).
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Not an index bound anywhere in this model: GPT-NeoX has no learned
    /// position table, so nothing gathers with a position id. It is reported as
    /// the context window by `context_window_from_config` and is validated for
    /// magnitude only.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_rotary_emb_base")]
    pub rotary_emb_base: f32,

    /// Fraction of each head's channels that RoPE rotates. See
    /// [`ModelArgs::rope_dims`].
    #[serde(default = "default_rotary_pct")]
    pub rotary_pct: f32,

    #[serde(default = "default_use_parallel_residual")]
    pub use_parallel_residual: bool,

    /// GPT-NeoX ships a separate `embed_out`; upstream builds it
    /// unconditionally and every published checkpoint declares `false` here.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Only used for a diagnostic. See [`ModelArgs::activation_is_gelu`].
    #[serde(default)]
    pub hidden_act: Option<String>,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub bos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// A `config.json` token-id field, which may be a single int or a list of ints.
///
/// Both `eos_token_id` and `bos_token_id` use this shape. Accepting the list
/// form for `bos_token_id` too matters because serde fails the whole config
/// when any one field does not match its declared type, so a checkpoint with a
/// list-valued `bos_token_id` would otherwise fail to load over a field this
/// model barely uses.
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
    "gpt_neox".to_string()
}
fn default_hidden_size() -> usize {
    2048
}
fn default_num_attention_heads() -> usize {
    8
}
fn default_num_hidden_layers() -> usize {
    16
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_vocab_size() -> usize {
    50304
}
fn default_rotary_emb_base() -> f32 {
    10000.0
}
fn default_rotary_pct() -> f32 {
    0.25
}
fn default_use_parallel_residual() -> bool {
    true
}

/// Upper bounds on the architecture scalars a GPT-NeoX `config.json` may
/// declare.
///
/// `config.json` is untrusted input: the common `mlxcel generate -m <org>/<repo>`
/// flow downloads a third-party HuggingFace repo and loads it in the same
/// command, and the download layer validates repo ids, filenames and transport
/// but never parses the file, so these fields arrive exactly as the checkpoint
/// author wrote them.
///
/// Each ceiling sits orders of magnitude above the largest real GPT-NeoX
/// (NeoX-20B: 44 layers, `hidden_size` 6144, 2048 positions). They exist so
/// `num_hidden_layers` cannot size the `Vec::with_capacity` in
/// [`GptNeoxModel::from_weights`], and so the `as i32` casts these values feed
/// (`hidden_size`, `3 * hidden_size`, `intermediate_size`, `head_dim`) stay
/// inside `i32` instead of truncating to a negative number.
const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_HIDDEN_SIZE: usize = 65_536;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_MAX_POSITION_EMBEDDINGS: usize = 1 << 22;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_VOCAB_SIZE: usize = 1 << 24;

impl ModelArgs {
    /// Reject a `config.json` that cannot describe a real GPT-NeoX, before any
    /// of its fields sizes an allocation, divides, or reaches an MLX kernel.
    ///
    /// Rejecting once at load rather than guarding ad hoc downstream follows the
    /// `ModelArgs::validate` precedent in [`crate::models::gpt_bigcode`]. See
    /// [`MAX_NUM_HIDDEN_LAYERS`] for why the magnitude ceilings exist at all.
    pub fn validate(&self) -> Result<(), String> {
        // The zero check has to come first: `0.is_multiple_of(0)` is true, so a
        // `hidden_size == num_attention_heads == 0` config would otherwise pass
        // the divisibility check below and reach `head_dim()`, which divides by
        // `num_attention_heads`.
        if self.num_attention_heads == 0 {
            return Err(
                "GPT-NeoX num_attention_heads must be non-zero (zero divides by zero in head_dim)"
                    .to_string(),
            );
        }
        if self.hidden_size == 0 || self.hidden_size > MAX_HIDDEN_SIZE {
            return Err(format!(
                "GPT-NeoX hidden_size ({}) must be between 1 and {MAX_HIDDEN_SIZE}",
                self.hidden_size
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "GPT-NeoX hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.num_hidden_layers == 0 || self.num_hidden_layers > MAX_NUM_HIDDEN_LAYERS {
            return Err(format!(
                "GPT-NeoX num_hidden_layers ({}) must be between 1 and {MAX_NUM_HIDDEN_LAYERS}",
                self.num_hidden_layers
            ));
        }
        if self.max_position_embeddings == 0
            || self.max_position_embeddings > MAX_MAX_POSITION_EMBEDDINGS
        {
            return Err(format!(
                "GPT-NeoX max_position_embeddings ({}) must be between 1 and \
                 {MAX_MAX_POSITION_EMBEDDINGS}",
                self.max_position_embeddings
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(format!(
                "GPT-NeoX vocab_size ({}) must be between 1 and {MAX_VOCAB_SIZE}",
                self.vocab_size
            ));
        }
        let intermediate = self.intermediate_size();
        if intermediate == 0 || intermediate > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "GPT-NeoX intermediate_size ({intermediate}) must be between 1 and \
                 {MAX_INTERMEDIATE_SIZE}"
            ));
        }
        self.validate_norm_eps()?;
        self.validate_quantization()?;
        self.validate_rope()
    }

    /// Reject a `layer_norm_eps` that would turn every hidden state into NaN.
    ///
    /// `layer_norm_eps` is the second float a `config.json` author controls, and
    /// unlike [`ModelArgs::validate_rope`] the failure mode here is silence
    /// rather than a crash. MLX's `fast::layer_norm` validates the shapes of its
    /// weight and bias but does not look at `eps` at all: it computes
    /// `x * rsqrt(mean(x^2) + eps)`, so a NaN `eps` makes every element of every
    /// hidden state NaN, and a negative one does the same as soon as
    /// `mean(x^2) + eps` goes below zero. Zero is unsafe for its own reason: an
    /// all-zero row (an unused or zero-initialized embedding row is enough)
    /// gives `rsqrt(0)`, and `0 * inf` is NaN again.
    ///
    /// Nothing throws on any of those, so the NaN propagates through every
    /// remaining layer into the logits and the sampler draws from a NaN
    /// distribution. The result is a checkpoint that loads cleanly and then
    /// generates uniformly garbage output, which is harder to diagnose than the
    /// load error rejecting it here produces. No published GPT-NeoX declares
    /// anything but a small positive value (Pythia: 1e-5).
    fn validate_norm_eps(&self) -> Result<(), String> {
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            return Err(format!(
                "GPT-NeoX layer_norm_eps ({}) must be a finite positive number; it is added to the \
                 variance under an rsqrt, so a non-finite, negative or zero value makes every \
                 normalized hidden state NaN and that NaN reaches the logits without anything \
                 throwing",
                self.layer_norm_eps
            ));
        }
        Ok(())
    }

    /// Reject a `quantization` block that would abort the process inside an MLX
    /// quantized kernel.
    ///
    /// `group_size` and `bits` are read straight out of `config.json` and are
    /// threaded through [`load_linear`] and `UnifiedEmbedding::from_weights`
    /// into MLX's `quantized_matmul` and `dequantize`. They are not reconciled
    /// away first: `mlxcel_core::layers::reconcile_quantization_layout`
    /// deliberately returns the declared pair unchanged when either is
    /// non-positive (it treats that as "insufficient shape info, trust the
    /// caller"), so a hostile value reaches the kernel exactly as written.
    ///
    /// MLX then computes `w.shape(-1) * 32 / bits` in `validate_quantized_input`.
    /// At `bits == 0` that is a division by zero, and at `bits > 32` it is zero,
    /// which cannot match the scales, so both end in a `std::invalid_argument`.
    /// `quantized_matmul` crosses the cxx bridge as `UniquePtr<MlxArray>` rather
    /// than a `Result`, so that throw is an uncatchable `std::terminate` at the
    /// first forward pass rather than a load error, which is the same shape of
    /// failure [`ModelArgs::validate_rope`] exists to prevent for `rope`.
    ///
    /// The bound is a range rather than an allowlist of the widths MLX actually
    /// supports on purpose. mlxcel tolerates a declared bit width that disagrees
    /// with the stored tensors and re-derives the effective one from the shapes,
    /// so an allowlist would reject mixed-precision exports that load correctly
    /// today. Only the values that cannot describe any packing at all are
    /// refused.
    fn validate_quantization(&self) -> Result<(), String> {
        let Some(quantization) = self.quantization.as_ref() else {
            return Ok(());
        };
        if quantization.bits < 1 || quantization.bits > 32 {
            return Err(format!(
                "GPT-NeoX quantization.bits ({}) must be between 1 and 32; MLX derives the \
                 unpacked width as `packed_in * 32 / bits`, which divides by zero at 0 and \
                 collapses to zero above 32, and the resulting MLX C++ exception crossing the cxx \
                 bridge is an uncatchable `std::terminate` at the first forward pass rather than a \
                 load error",
                quantization.bits
            ));
        }
        if quantization.group_size < 1 {
            return Err(format!(
                "GPT-NeoX quantization.group_size ({}) must be positive; it is multiplied by the \
                 scales width to check the packing, and a non-positive value can match no real \
                 tensor, so MLX throws and that throw is an uncatchable `std::terminate` rather \
                 than a load error",
                quantization.group_size
            ));
        }
        Ok(())
    }

    /// Reject RoPE parameters that would abort the process at the first forward
    /// pass, or that would poison every rotated channel with NaN.
    ///
    /// `mlx::core::fast::rope` does validate its `dims` argument: it throws when
    /// `dims` is not positive, when `dims` is odd, and when `dims` exceeds the
    /// size of the input's last axis. None of that is usable as a Rust-side
    /// error. The throw is a C++ `std::invalid_argument`, and `fast_rope` is
    /// declared across the cxx bridge as returning `UniquePtr<MlxArray>` rather
    /// than a `Result`, so it becomes an uncatchable `std::terminate` (SIGABRT).
    /// It also fires at the first forward pass rather than at load, so an
    /// unchecked config loads cleanly and then takes the whole process down on
    /// the first request. Every value MLX would throw on is therefore rejected
    /// here instead, at load, by a message that names `rotary_pct`.
    ///
    /// Evenness is part of that contract and is not a formality: RoPE rotates
    /// channel *pairs*, so an odd `dims` has no meaning. HuggingFace
    /// `GPTNeoXRotaryEmbedding` splits at `dims // 2` and silently drops the odd
    /// channel; MLX refuses outright. No published checkpoint produces one,
    /// since every `head_dim` and `rotary_pct` in this family multiplies out
    /// even, so rejecting costs nothing a real checkpoint needs.
    ///
    /// `rotary_pct` is a float, which widens what a hostile config can express
    /// beyond the integer fields. The `as i32` cast in [`ModelArgs::rope_dims`]
    /// is saturating in Rust, so NaN becomes 0 and an infinity becomes
    /// `i32::MAX`; both are caught by the range check, but the non-finite case
    /// is rejected explicitly so the message names the actual problem.
    fn validate_rope(&self) -> Result<(), String> {
        if !self.rotary_pct.is_finite() {
            return Err(format!(
                "GPT-NeoX rotary_pct ({}) must be a finite number",
                self.rotary_pct
            ));
        }
        if !self.rotary_emb_base.is_finite() || self.rotary_emb_base <= 0.0 {
            return Err(format!(
                "GPT-NeoX rotary_emb_base ({}) must be a finite positive number; RoPE takes its \
                 logarithm, so zero or a negative base makes every rotated channel NaN",
                self.rotary_emb_base
            ));
        }
        let head_dim = self.head_dim();
        let rope_dims = self.rope_dims();
        // A negative `rope_dims` fails the conversion; folding it to zero puts
        // it in the same arm as a zero one, which MLX refuses for the same
        // reason.
        let dims = usize::try_from(rope_dims).unwrap_or(0);
        if dims == 0 || dims > head_dim {
            return Err(format!(
                "GPT-NeoX rotary_pct ({}) gives {rope_dims} rotary dimensions for a head of width \
                 {head_dim}; it must be an even number between 2 and {head_dim}. MLX throws on a \
                 rope `dims` outside that range, and an MLX C++ exception crossing the cxx bridge \
                 is an uncatchable `std::terminate` at the first forward pass rather than a load \
                 error.",
                self.rotary_pct
            ));
        }
        if !dims.is_multiple_of(2) {
            return Err(format!(
                "GPT-NeoX rotary_pct ({}) gives an odd rotary dimension count ({rope_dims}) for a \
                 head of width {head_dim}; RoPE rotates channel pairs, so the rope `dims` must be \
                 even. MLX throws on an odd `dims`, and an MLX C++ exception crossing the cxx \
                 bridge is an uncatchable `std::terminate` at the first forward pass rather than a \
                 load error.",
                self.rotary_pct
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of channels per head that RoPE rotates: `int(head_dim *
    /// rotary_pct)`, truncated exactly as upstream's `int()` truncates.
    ///
    /// Pythia 1B: `int(256 * 0.25)` = 64, leaving 192 channels per head
    /// unrotated. The value is passed straight to `fast_rope`'s `dims`
    /// argument, which rotates the leading `dims` channels of the last axis and
    /// copies the remainder through.
    ///
    /// The cast saturates rather than wrapping, and
    /// [`ModelArgs::validate_rope`] rejects everything that is not an even value
    /// in `2..=head_dim` before it can reach a kernel.
    pub fn rope_dims(&self) -> i32 {
        (self.rotary_pct * self.head_dim() as f32) as i32
    }

    /// Byte-free description of the interleaved QKV layout: the channel offsets
    /// of Q, K and V *within one head's* `3 * head_dim` block.
    ///
    /// This is the whole trap of the family reduced to a pure function so it can
    /// be asserted without a checkpoint. The projection output is head-major,
    /// `[q_i | k_i | v_i]` per head, so after reshaping to
    /// `(..., num_heads, 3 * head_dim)` the three tensors are slices of the last
    /// axis at `[0, head_dim)`, `[head_dim, 2 * head_dim)` and
    /// `[2 * head_dim, 3 * head_dim)`.
    ///
    /// Contrast the flat GPT-2 split, which would take `[0, hidden_size)` and so
    /// on out of the *unreshaped* projection: for Pythia 1B that puts the Q
    /// channels of heads 0 through 2 (plus their K and V channels) into the
    /// tensor called Q. Same shapes, different numbers, fluent output.
    pub fn interleaved_qkv_channel_offsets(&self) -> (usize, usize, usize) {
        let head_dim = self.head_dim();
        (head_dim, 2 * head_dim, 3 * head_dim)
    }

    /// MLP hidden width: `intermediate_size` when the config gives one,
    /// `4 * hidden_size` otherwise.
    ///
    /// Upstream hardcodes `4 * hidden_size` and ignores the config field;
    /// HuggingFace `GPTNeoXMLP` reads the field. They agree on every published
    /// checkpoint (Pythia 1B declares 8192 for `hidden_size` 2048), so reading
    /// the field is the strictly more faithful of the two and a value that
    /// disagrees with the checkpoint is rejected by [`load_linear`] rather than
    /// reaching a matmul.
    pub fn intermediate_size(&self) -> usize {
        match self.intermediate_size {
            Some(n) if n > 0 => n,
            _ => 4usize.saturating_mul(self.hidden_size),
        }
    }

    /// Whether `hidden_act` names a GELU variant.
    ///
    /// Every published GPT-NeoX declares `gelu` or `gelu_fast`, and this loader
    /// applies GELU unconditionally (see [`MLP::forward`]). A checkpoint asking
    /// for something else would still load and still generate, just from the
    /// wrong non-linearity, so the mismatch is reported rather than silently
    /// accepted. It is not a hard rejection: the activation choice cannot
    /// corrupt memory, and refusing an otherwise loadable checkpoint over it
    /// would be worse than saying so.
    pub fn activation_is_gelu(&self) -> bool {
        match &self.hidden_act {
            Some(act) => act.to_ascii_lowercase().contains("gelu"),
            None => true,
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

    /// Stop tokens declared by the config.
    ///
    /// Pythia and the other NeoX tokenizers put `<|endoftext|>` at id 0 and
    /// declare it as both `eos_token_id` and `bos_token_id`, but the family has
    /// no id that is safe to assume for a checkpoint that declares neither
    /// (NeoX-derived instruction models add their own stop tokens). An empty
    /// result is preferred over guessing an id that could truncate every
    /// generation at an ordinary token.
    pub fn eos_token_ids(&self) -> Vec<i32> {
        [self.eos_token_id.as_ref(), self.bos_token_id.as_ref()]
            .into_iter()
            .flatten()
            .map(TokenIdField::ids)
            .find(|ids| !ids.is_empty())
            .unwrap_or_default()
    }
}

// Checkpoint layout.

/// Key layout of the GPT-NeoX checkpoint being loaded.
///
/// Two shapes reach this loader, and unlike the GPT-2 case they differ only in
/// naming, never in weight orientation. HuggingFace `GPTNeoXForCausalLM` nests
/// the decoder under a `gpt_neox` submodule and keeps the output head
/// `embed_out` at the top level. Upstream mlx-lm ships a `sanitize` that
/// prefixes every key with `model.` and then rewrites `.gpt_neox.layers.` to
/// `.h.` and `.gpt_neox.` to `.`, so an MLX conversion produced from that module
/// tree carries the second naming instead.
///
/// The candidate list is fixed at two entries and is probed by the embedding
/// key alone; there is no shape-derived branch, because both layouts store every
/// projection in the same `nn.Linear` `[out, in]` orientation and there is
/// nothing to transpose in this family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptNeoxLayout {
    /// Prefix in front of `embed_in` and `final_layer_norm`.
    pub prefix: &'static str,
    /// Path segment holding the decoder blocks: `layers` in a raw HuggingFace
    /// export, `h` after upstream's `sanitize`.
    pub layers_key: &'static str,
    /// Key of the untied output head, without the `.weight` suffix.
    pub embed_out: &'static str,
}

/// The layouts a GPT-NeoX checkpoint can use, raw HuggingFace first.
///
/// Only the first entry is checkpoint-validated (`EleutherAI/pythia-1b`); the
/// second is derived directly from upstream's `sanitize`, which is the only
/// thing that can produce an MLX conversion of this family.
const GPT_NEOX_LAYOUTS: [GptNeoxLayout; 2] = [
    GptNeoxLayout {
        prefix: "gpt_neox.",
        layers_key: "layers",
        embed_out: "embed_out",
    },
    GptNeoxLayout {
        prefix: "model.",
        layers_key: "h",
        embed_out: "model.embed_out",
    },
];

impl GptNeoxLayout {
    /// Detect which of [`GPT_NEOX_LAYOUTS`] the checkpoint uses.
    pub fn detect(weights: &WeightMap) -> Result<Self, String> {
        GPT_NEOX_LAYOUTS
            .iter()
            .find(|layout| weights.contains_key(&format!("{}embed_in.weight", layout.prefix)))
            .cloned()
            .ok_or_else(|| {
                "GPT-NeoX token embedding not found: expected gpt_neox.embed_in.weight (raw \
                 HuggingFace export) or model.embed_in.weight (MLX conversion)"
                    .to_string()
            })
    }

    /// Key prefix of decoder block `layer_idx`, without a trailing dot.
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        format!("{}{}.{}", self.prefix, self.layers_key, layer_idx)
    }

    fn key(&self, suffix: &str) -> String {
        format!("{}{}", self.prefix, suffix)
    }
}

/// Suffixes of the PyTorch buffers `GPTNeoXAttention` registers, which reach the
/// checkpoint but are not weights.
///
/// On `EleutherAI/pythia-1b`, per layer:
///
/// ```text
/// gpt_neox.layers.N.attention.bias                  U8  [1, 1, 2048, 2048]
/// gpt_neox.layers.N.attention.masked_bias           F16 []
/// gpt_neox.layers.N.attention.rotary_emb.inv_freq   F16 [32]
/// ```
///
/// `.attention.bias` is a lower-triangular causal mask, not a projection bias,
/// exactly as GPT-2's `h.N.attn.bias` is; `masked_bias` is the scalar fill value
/// that goes with it; `inv_freq` is a precomputed RoPE frequency table that
/// `fast_rope` recomputes from `rotary_emb_base` anyway. The real projection
/// biases in this family are `.attention.query_key_value.bias` and
/// `.attention.dense.bias`, neither of which ends in any of these suffixes, so
/// none of them is ever matched.
const REGISTERED_BUFFER_SUFFIXES: [&str; 3] = [
    ".attention.bias",
    ".attention.masked_bias",
    ".attention.rotary_emb.inv_freq",
];

/// Remove the per-layer registered buffers listed in
/// [`REGISTERED_BUFFER_SUFFIXES`], returning how many were removed.
///
/// The model graph never asks for these keys, but dropping them before
/// construction releases the causal-mask buffer, which for Pythia's 2048-token
/// context is 4 MB per layer (64 MB over 16 layers), instead of holding it for
/// the lifetime of the weight map.
///
/// This is one pass over the weight map rather than a probe of
/// `<prefix><layers>.<i>.attention.bias` for every `i` in
/// `0..num_hidden_layers`. `num_hidden_layers` is attacker-controlled and is not
/// checked against anything before this runs on the `load()` path, and the
/// probing form has no early exit, so a config declaring it in the billions
/// would spin here for hours before the first weight lookup could reject it.
/// The same replacement was made in [`crate::models::gpt2`] for the same reason.
pub fn strip_registered_buffers(weights: &mut WeightMap) -> usize {
    let before = weights.len();
    weights.retain(|key, _| {
        !REGISTERED_BUFFER_SUFFIXES
            .iter()
            .any(|suffix| key.ends_with(suffix))
    });
    before - weights.len()
}

/// Load an `nn.Linear` projection, rejecting anything that is not `[out, in]`.
///
/// Every shape a GPT-NeoX projection can legally have is known from the config,
/// so check it here rather than letting a mismatch reach `matmul` or `reshape`:
/// an MLX C++ exception crossing the cxx bridge is an uncatchable
/// `std::terminate`, so the process would die with no diagnostic instead of
/// returning a load error naming the tensor.
///
/// Quantization packs the *input* axis only, so a quantized weight is still
/// `[out_features, packed_in]`. The packed input width matches no float layout
/// and is left to `UnifiedLinear` to reconcile, but the row count is untouched
/// by packing, so the output width is checked on the quantized path too. That
/// check is not cosmetic for this family: [`Attention::forward`] reshapes the
/// fused `query_key_value` output to `[batch, seq, num_heads, 3 * head_dim]`
/// using widths derived from `config.json`, so a packed projection whose real
/// output width disagrees makes the reshape throw, which is the uncatchable
/// `std::terminate` again. The same carve-out had to be closed in
/// [`crate::models::gpt_bigcode`] after review.
pub fn load_linear(
    weights: &WeightMap,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    group_size: i32,
    bits: i32,
) -> Result<UnifiedLinear, String> {
    let quantized = weights.contains_key(&format!("{prefix}.scales"));

    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let shape = mlxcel_core::array_shape(weight);

    let is_expected = shape.len() == 2
        && dim_eq(shape[0], out_features)
        && (quantized || dim_eq(shape[1], in_features));
    if !is_expected {
        let looks_transposed = !quantized
            && shape.len() == 2
            && dim_eq(shape[0], in_features)
            && dim_eq(shape[1], out_features);
        let hint = if looks_transposed {
            " That is the [in, out] orientation. HuggingFace GPTNeoX builds its projections with \
             `nn.Linear`, so a genuine checkpoint is already [out, in] and must not be transposed."
        } else {
            ""
        };
        let expected_in = if quantized {
            "<packed in>".to_string()
        } else {
            in_features.to_string()
        };
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected [{out_features}, \
             {expected_in}].{hint}"
        ));
    }

    if let Some(bias) = weights.get(&format!("{prefix}.bias")) {
        let bias_shape = mlxcel_core::array_shape(bias);
        if bias_shape.len() != 1 || !dim_eq(bias_shape[0], out_features) {
            return Err(format!(
                "unexpected {prefix}.bias shape {bias_shape:?}: expected [{out_features}]"
            ));
        }
    }

    UnifiedLinear::from_weights(weights, prefix, group_size, bits)
}

// Attention (interleaved per-head fused QKV, partial RoPE).

/// Split the fused `query_key_value` output into Q, K and V, each
/// `[batch, num_heads, seq_len, head_dim]`.
///
/// **This is the trap of the family, and the one mistake here that no shape
/// check can catch.** The projection is `3 * hidden_size` wide, but its channels
/// are grouped by head rather than by tensor: head `i` owns the contiguous run
/// `[q_i | k_i | v_i]`, so Q, K and V are interleaved at a stride of
/// `3 * head_dim`. On `EleutherAI/pythia-1b` that is 8 heads of 768 channels
/// each, Q at `[0, 256)` within the head, K at `[256, 512)`, V at `[512, 768)`.
///
/// Reshaping to `[batch, seq, num_heads, 3 * head_dim]` first turns that stride
/// into an axis, after which the three tensors are ordinary slices of the last
/// axis. That is exactly upstream's
///
/// ```python
/// qkv.reshape(*qkv.shape[:-1], num_heads, 3 * head_dim).split(3, -1)
/// ```
///
/// and it is *not* the flat `split(qkv, 3, axis=-1)` that
/// [`crate::models::gpt2`] and [`crate::models::gpt_bigcode`] use on their own
/// fused projections. The flat split produces Q, K and V of byte-for-byte
/// identical shape assembled from the wrong channels, so nothing throws and
/// nothing degenerates visibly: the model loads, decodes, and emits fluent
/// English out of a scrambled attention. It is factored out here so the layout
/// can be pinned by value in a unit test rather than only observed through a
/// whole forward pass.
pub fn split_interleaved_qkv(
    qkv: &MlxArray,
    batch: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let grouped = mlxcel_core::reshape(qkv, &[batch, seq_len, num_heads, 3 * head_dim]);
    let q = mlxcel_core::slice_last_dim(&grouped, 0, head_dim);
    let k = mlxcel_core::slice_last_dim(&grouped, head_dim, 2 * head_dim);
    let v = mlxcel_core::slice_last_dim(&grouped, 2 * head_dim, 3 * head_dim);

    // [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
    (
        mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]),
        mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]),
        mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]),
    )
}

pub struct Attention {
    pub query_key_value: UnifiedLinear,
    /// Output projection. GPT-NeoX names it `dense`, not `o_proj`.
    pub dense: UnifiedLinear,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    /// Channels per head that RoPE rotates. An even value in `2..=head_dim`;
    /// see [`ModelArgs::rope_dims`].
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
        let head_dim = self.head_dim;

        let qkv = self.query_key_value.forward(x);
        let (q, k, v) = split_interleaved_qkv(&qkv, b, l, self.num_heads, head_dim);

        // Partial RoPE: `rope_dims` of the `head_dim` channels rotate, the rest
        // are copied through unchanged. `traditional` is false, matching
        // upstream's `nn.RoPE(dims=int(head_dim * rotary_pct),
        // traditional=False, base=rotary_emb_base)`.
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
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * head_dim]);

        self.dense.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let hidden_size = args.hidden_size;
        let query_key_value = load_linear(
            weights,
            &format!("{prefix}.query_key_value"),
            hidden_size,
            3 * hidden_size,
            group_size,
            bits,
        )?;
        let dense = load_linear(
            weights,
            &format!("{prefix}.dense"),
            hidden_size,
            hidden_size,
            group_size,
            bits,
        )?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            query_key_value,
            dense,
            num_heads: args.num_attention_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: args.rope_dims(),
            rope_base: args.rotary_emb_base,
        })
    }
}

// MLP (GELU, no gate/up pattern).

pub struct MLP {
    pub dense_h_to_4h: UnifiedLinear,
    pub dense_4h_to_h: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.dense_h_to_4h.forward(x);
        // Which GELU, and why.
        //
        // Three candidates are in play. HuggingFace `GPTNeoXMLP` resolves
        // `hidden_act`, and Pythia declares `"gelu"`, which is the exact
        // erf-based GELU. Upstream mlx-lm instead applies `nn.gelu_approx`, the
        // tanh approximation, and notes that it corresponds to transformers'
        // `FastGELUActivation`; MLX documents that approximation as within
        // 5e-4 of the exact form over [-6, 6].
        //
        // `mlxcel_core::utils::gelu_approx` is the *erf-based exact* GELU in
        // this tree despite its name (see its own doc comment and the C++
        // implementation in `mlx_cxx_bridge.cpp`, where the tanh form was
        // deliberately replaced because `power(x, 3)` produced NaN for negative
        // bf16 inputs). Using it here therefore matches HuggingFace exactly,
        // which is the checkpoint's own declared activation, and differs from
        // the mlx-lm reference by under 5e-4 per element. That is the same
        // choice `src/models/gpt2.rs` and `src/models/gpt_bigcode.rs` make, and
        // both are token-exact against their mlx-lm references.
        let h = mlxcel_core::utils::gelu_approx(&h);
        self.dense_4h_to_h.forward(&h)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let hidden_size = args.hidden_size;
        let intermediate = args.intermediate_size();
        let dense_h_to_4h = load_linear(
            weights,
            &format!("{prefix}.dense_h_to_4h"),
            hidden_size,
            intermediate,
            group_size,
            bits,
        )?;
        let dense_4h_to_h = load_linear(
            weights,
            &format!("{prefix}.dense_4h_to_h"),
            intermediate,
            hidden_size,
            group_size,
            bits,
        )?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }
}

// Transformer block (parallel or sequential residual).

pub struct TransformerBlock {
    pub attention: Attention,
    pub mlp: MLP,
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
    /// `true` on every Pythia checkpoint. See [`TransformerBlock::forward`].
    pub use_parallel_residual: bool,
}

impl TransformerBlock {
    /// Parallel residual (`use_parallel_residual: true`):
    ///
    /// ```text
    /// out = x + attention(input_layernorm(x)) + mlp(post_attention_layernorm(x))
    /// ```
    ///
    /// Both sub-layers read the same pre-norm input `x`, so the MLP does not
    /// see the attention output at all. This is what makes the two branches
    /// independent enough to overlap, and it is the layout every Pythia
    /// checkpoint was trained with.
    ///
    /// Sequential residual (`use_parallel_residual: false`) is the ordinary
    /// chained form, where `post_attention_layernorm` reads the post-attention
    /// residual rather than `x`:
    ///
    /// ```text
    /// h   = x + attention(input_layernorm(x))
    /// out = h + mlp(post_attention_layernorm(h))
    /// ```
    ///
    /// Both are real checkpoint configurations, so both are implemented and
    /// unit-tested. Running the wrong one produces a numerically different but
    /// perfectly well-shaped result.
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if self.use_parallel_residual {
            let attn_out = self
                .attention
                .forward(&self.input_layernorm.forward(x), cache, mask);
            let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(x));
            let h = mlxcel_core::add(x, &attn_out);
            mlxcel_core::add(&h, &mlp_out)
        } else {
            let attn_out = self
                .attention
                .forward(&self.input_layernorm.forward(x), cache, mask);
            let h = mlxcel_core::add(x, &attn_out);
            let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
            mlxcel_core::add(&h, &mlp_out)
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layout: &GptNeoxLayout,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = layout.layer_prefix(layer_idx);

        let attention = Attention::from_weights(weights, args, &format!("{prefix}.attention"))?;
        let mlp = MLP::from_weights(weights, args, &format!("{prefix}.mlp"))?;

        let dim = args.hidden_size;
        let eps = args.layer_norm_eps;
        let input_layernorm =
            layer_norm_from_weights(weights, &format!("{prefix}.input_layernorm"), dim, eps)?;
        let post_attention_layernorm = layer_norm_from_weights(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            dim,
            eps,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            use_parallel_residual: args.use_parallel_residual,
        })
    }
}

// GPT-NeoX model.

pub struct GptNeoxModel {
    /// Token embedding, and the output head too when the config ties them.
    pub embed_in: UnifiedEmbedding,
    pub h: Vec<TransformerBlock>,
    pub final_layer_norm: LayerNorm,
    /// Separate output head. Present on every published checkpoint; `None` only
    /// when `tie_word_embeddings` is true.
    pub embed_out: Option<UnifiedLinear>,
    eos_token_ids: Vec<i32>,
}

impl GptNeoxModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // No learned position table: position enters the graph only through the
        // partial RoPE offset inside each attention block.
        let mut h = self.embed_in.forward(input_ids);

        for (i, layer) in self.h.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.final_layer_norm.forward(&h);

        match &self.embed_out {
            Some(head) => head.forward(&h),
            None => self.embed_in.as_linear(&h),
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.h.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        // Reject an impossible config before reading the checkpoint, not after.
        // `from_weights` validates again for the owned-weights route.
        args.validate()?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        strip_registered_buffers(&mut weights);

        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        // `config.json` is untrusted: reject impossible scalars before any of
        // them sizes an allocation, divides, or reaches an MLX kernel.
        args.validate()?;

        if !args.activation_is_gelu() {
            // `eprintln!` rather than `tracing::warn!`: only `mlxcel-server`
            // installs a tracing subscriber, so a `tracing` event is a no-op on
            // the CLI path this is reachable from.
            eprintln!(
                "GPT-NeoX config declares hidden_act {:?}, but this loader always applies GELU. \
                 Generated text will differ from the reference implementation.",
                args.hidden_act.as_deref().unwrap_or("")
            );
        }

        let layout = GptNeoxLayout::detect(weights)?;
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_in_key = layout.key("embed_in");
        let embed_in = UnifiedEmbedding::from_weights(weights, &embed_in_key, group_size, bits)?;
        // Token ids are bounded by `vocab_size`, a config field, and an
        // embedding gather wraps a negative index but does not range-check a
        // positive one, so a config that overstates the table turns an ordinary
        // prompt into an out-of-bounds read whose result reaches the logits.
        validate_embedding_table(
            &embed_in,
            &embed_in_key,
            args.vocab_size,
            "vocab_size",
            args.hidden_size,
            "hidden_size",
        )?;

        let mut h = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            h.push(TransformerBlock::from_weights(weights, args, &layout, i)?);
        }

        let final_layer_norm = layer_norm_from_weights(
            weights,
            &layout.key("final_layer_norm"),
            args.hidden_size,
            args.layer_norm_eps,
        )?;

        // Upstream builds `embed_out` unconditionally and every published
        // checkpoint ships it, which is why `tie_word_embeddings` defaults to
        // false. An untied config must actually carry the tensor; silently
        // falling back to the tied path would produce logits from the wrong
        // matrix.
        let embed_out = if args.tie_word_embeddings {
            None
        } else {
            Some(load_linear(
                weights,
                layout.embed_out,
                args.hidden_size,
                args.vocab_size,
                group_size,
                bits,
            )?)
        };

        Ok(Self {
            embed_in,
            h,
            final_layer_norm,
            embed_out,
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

// LanguageModel trait implementation.

impl LanguageModel for GptNeoxModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        GptNeoxModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        GptNeoxModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.h.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "gpt_neox_tests.rs"]
mod tests;
