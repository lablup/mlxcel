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

//! GPT-BigCode (`gpt_bigcode`) text model implementation using mlxcel-core.
//!
//! Reference: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gpt_bigcode.py>
//!
//! GPT-BigCode is the architecture behind the BigCode StarCoder and SantaCoder
//! code models. It is a GPT-2 derivative, so the block shape is shared with
//! [`crate::models::gpt2`]: learned absolute position embeddings (`wpe`) added
//! to the token embeddings at the input boundary with no RoPE anywhere,
//! `LayerNorm` with bias rather than RMSNorm, one fused `c_attn` projection,
//! and a tied output head. The position helpers, the load-time embedding-table
//! validation and the biased `LayerNorm` loader are reused from that module
//! rather than duplicated.
//!
//! Four deltas versus GPT-2:
//!
//! - **Multi-Query Attention**, the defining feature. With `multi_query: true`
//!   exactly one KV head is shared by every query head, so `c_attn` produces
//!   `dim + 2 * kv_dim` features rather than `3 * dim`, and the split offsets
//!   are `[dim, dim + kv_dim]`, which is *not* an even three-way split. On
//!   `bigcode/gpt_bigcode-santacoder` that is 2048 + 2 * 128 = 2304 for 16
//!   query heads of width 128 sharing a single KV head. This maps onto the
//!   existing GQA attention path with `n_kv_heads = 1`; MLX broadcasts the
//!   single KV head across the query heads exactly as it does for the GQA
//!   families, so no new attention code is involved. See
//!   [`ModelArgs::qkv_split_offsets`].
//! - **No `Conv1D` layout.** HuggingFace `GPTBigCode` builds its projections
//!   with `nn.Linear`, not the `Conv1D` that GPT-2 uses, so `c_attn`, `c_proj`
//!   and `c_fc` are already stored `[out, in]`, which is the layout
//!   `mlxcel_core::layers::Linear` wants. Upstream `gpt_bigcode.py`
//!   correspondingly has no `sanitize` method at all. Verified against the
//!   checkpoint: `transformer.h.0.attn.c_attn.weight` is `[2304, 2048]`
//!   (`[out, in]`), where the GPT-2 equivalent is `[768, 2304]` (`[in, out]`).
//!   Transposing here would silently corrupt every projection, so
//!   [`load_linear`] rejects a transposed weight by name rather than accepting
//!   it.
//! - **MLP width from `n_inner`**, not a hardcoded `4 * n_embd`. `n_inner` may
//!   be absent or `null`, in which case `4 * n_embd` is the fallback.
//! - **No causal-mask buffer.** HuggingFace registers GPT-BigCode's causal mask
//!   with `persistent=False`, so unlike GPT-2 no `h.N.attn.bias` tensor ever
//!   reaches the checkpoint and there is nothing to strip at load. The
//!   santacoder export carries 292 tensors, none of them matching that shape.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use crate::models::gpt2::{
    dim_eq, exceeds_position_table, layer_norm_from_weights, position_ids, validate_embedding_table,
};

// Configuration.

/// GPT-BigCode `config.json`.
///
/// Like GPT-2, GPT-BigCode predates the `hidden_size` / `num_attention_heads`
/// naming that the rest of the tree uses, so the field names are the original
/// OpenAI ones.
///
/// Two upstream `ModelArgs` fields are deliberately absent here.
/// `num_key_value_heads` is vestigial: upstream's `__post_init__` derives it
/// from `multi_query`, but its `Attention` then recomputes `1 if multi_query
/// else n_head` and never reads the field, and HuggingFace `GPTBigCode` has no
/// such concept at all. `attention_bias` / `mlp_bias` both default to `true`
/// and gate whether a projection is built with a bias; this loader instead
/// takes the bias from the checkpoint when the tensor is present, which is the
/// weight-presence convention used elsewhere in this tree and gives the same
/// result for every real export.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_n_embd")]
    pub n_embd: usize,

    #[serde(default = "default_n_head")]
    pub n_head: usize,

    #[serde(default = "default_n_layer")]
    pub n_layer: usize,

    /// MLP hidden width. `null` or absent means `4 * n_embd`.
    #[serde(default)]
    pub n_inner: Option<usize>,

    /// Number of rows in the learned position-embedding table `wpe`.
    #[serde(default = "default_n_positions")]
    pub n_positions: usize,

    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// `true` (the default, and every published checkpoint) means one KV head
    /// shared by all `n_head` query heads. `false` degenerates to plain
    /// multi-head attention.
    #[serde(default = "default_multi_query")]
    pub multi_query: bool,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

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
    "gpt_bigcode".to_string()
}
fn default_n_embd() -> usize {
    2048
}
fn default_n_head() -> usize {
    16
}
fn default_n_layer() -> usize {
    24
}
fn default_n_positions() -> usize {
    2048
}
fn default_layer_norm_epsilon() -> f32 {
    1e-5
}
fn default_vocab_size() -> usize {
    49280
}
fn default_multi_query() -> bool {
    true
}
fn default_tie_word_embeddings() -> bool {
    true
}

/// Upper bounds on the architecture scalars a GPT-BigCode `config.json` may
/// declare.
///
/// `config.json` is untrusted input: the common `mlxcel generate -m <org>/<repo>`
/// flow downloads a third-party HuggingFace repo and loads it in the same
/// command, and the download layer validates repo ids, filenames and transport
/// but never parses the file, so these fields arrive exactly as the checkpoint
/// author wrote them.
///
/// Each ceiling sits orders of magnitude above the largest real GPT-BigCode
/// (StarCoder: 40 layers, `n_embd` 6144, 8192 positions). They exist so
/// `n_layer` cannot size the `Vec::with_capacity` in
/// [`GptBigCodeModel::from_weights`], and so the `as i32` casts these values
/// feed (`n_embd`, `n_embd + 2 * kv_dim`, `n_inner`, `n_positions - 1`) stay
/// inside `i32` instead of truncating to a negative number.
const MAX_N_LAYER: usize = 1024;
/// See [`MAX_N_LAYER`].
const MAX_N_EMBD: usize = 65_536;
/// See [`MAX_N_LAYER`].
const MAX_N_INNER: usize = 1 << 22;
/// See [`MAX_N_LAYER`].
const MAX_N_POSITIONS: usize = 1 << 22;
/// See [`MAX_N_LAYER`].
const MAX_VOCAB_SIZE: usize = 1 << 24;

impl ModelArgs {
    /// Reject a `config.json` that cannot describe a real GPT-BigCode, before
    /// any of its fields sizes an allocation, indexes a table, or divides.
    ///
    /// Rejecting once at load rather than guarding ad hoc downstream follows the
    /// `ModelArgs::validate` precedent in [`crate::models::gpt2`]. See
    /// [`MAX_N_LAYER`] for why the magnitude ceilings exist at all.
    pub fn validate(&self) -> Result<(), String> {
        // The zero check has to come first: `0.is_multiple_of(0)` is true, so a
        // `n_embd == n_head == 0` config would otherwise pass the divisibility
        // check below and reach `head_dim()`, which divides by `n_head`.
        if self.n_head == 0 {
            return Err(
                "GPT-BigCode n_head must be non-zero (zero divides by zero in head_dim)"
                    .to_string(),
            );
        }
        if self.n_embd == 0 || self.n_embd > MAX_N_EMBD {
            return Err(format!(
                "GPT-BigCode n_embd ({}) must be between 1 and {MAX_N_EMBD}",
                self.n_embd
            ));
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err(format!(
                "GPT-BigCode n_embd ({}) must be divisible by n_head ({})",
                self.n_embd, self.n_head
            ));
        }
        if self.n_layer == 0 || self.n_layer > MAX_N_LAYER {
            return Err(format!(
                "GPT-BigCode n_layer ({}) must be between 1 and {MAX_N_LAYER}",
                self.n_layer
            ));
        }
        if self.n_positions == 0 || self.n_positions > MAX_N_POSITIONS {
            return Err(format!(
                "GPT-BigCode n_positions ({}) must be between 1 and {MAX_N_POSITIONS}",
                self.n_positions
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(format!(
                "GPT-BigCode vocab_size ({}) must be between 1 and {MAX_VOCAB_SIZE}",
                self.vocab_size
            ));
        }
        let intermediate = self.intermediate_size();
        if intermediate == 0 || intermediate > MAX_N_INNER {
            return Err(format!(
                "GPT-BigCode n_inner ({intermediate}) must be between 1 and {MAX_N_INNER}"
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// One KV head under Multi-Query Attention, `n_head` otherwise.
    ///
    /// This mirrors upstream's `Attention`, which recomputes the value from
    /// `multi_query` rather than reading the vestigial `num_key_value_heads`
    /// dataclass field.
    pub fn n_kv_heads(&self) -> usize {
        if self.multi_query { 1 } else { self.n_head }
    }

    /// Combined width of one K (or one V) block in the fused `c_attn` output.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads() * self.head_dim()
    }

    /// Output features of the fused `c_attn` projection: `dim + 2 * kv_dim`.
    ///
    /// Under MQA this is *not* `3 * n_embd`. Santacoder: 2048 + 2 * 128 = 2304.
    pub fn c_attn_out_features(&self) -> usize {
        self.n_embd + 2 * self.kv_dim()
    }

    /// Split points of the fused `c_attn` output on the last axis.
    ///
    /// `(dim, dim + kv_dim)`, so Q is `[0, dim)`, K is `[dim, dim + kv_dim)`
    /// and V is `[dim + kv_dim, dim + 2 * kv_dim)`. Under MQA the three blocks
    /// have different widths, so splitting the projection into three equal
    /// parts (the GPT-2 pattern) silently mixes query channels into K and V.
    pub fn qkv_split_offsets(&self) -> (usize, usize) {
        let dim = self.n_embd;
        (dim, dim + self.kv_dim())
    }

    /// MLP hidden width: `n_inner` when the config gives one, `4 * n_embd`
    /// otherwise. A `null` `n_inner` deserializes to `None` and takes the
    /// fallback, which is what HuggingFace does with the same field.
    pub fn intermediate_size(&self) -> usize {
        match self.n_inner {
            Some(n) if n > 0 => n,
            _ => 4usize.saturating_mul(self.n_embd),
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
    /// GPT-BigCode checkpoints do not share one `<|endoftext|>` id the way
    /// GPT-2 does (santacoder uses 49152, StarCoder uses 0), so there is no
    /// safe family-wide constant to fall back on. `bos_token_id` is the second
    /// choice because these configs set it to the same `<|endoftext|>` token,
    /// and an empty result (no model-declared stop token) is preferred over
    /// guessing an id that could truncate every generation at an ordinary
    /// token. Every published checkpoint declares `eos_token_id`.
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

/// The only key prefix a GPT-BigCode checkpoint uses.
///
/// HuggingFace `GPTBigCodeForCausalLM` nests the whole decoder under a
/// `transformer` submodule, and upstream mlx-lm's `Model` names the same
/// submodule `transformer` and ships no `sanitize`, so a conversion has nothing
/// that could rename it. Verified on `bigcode/gpt_bigcode-santacoder`: all 292
/// tensors are `transformer.`-prefixed. GPT-2's multi-prefix probe exists
/// because its raw and converted exports genuinely disagree; there is no such
/// split here, and accepting a prefix no real checkpoint uses would only widen
/// what the loader has to treat as valid.
pub const GPT_BIGCODE_PREFIX: &str = "transformer.";

/// Confirm the checkpoint uses the expected `transformer.` key layout.
pub fn detect_prefix(weights: &WeightMap) -> Result<&'static str, String> {
    let probe = format!("{GPT_BIGCODE_PREFIX}wte.weight");
    if weights.contains_key(&probe) {
        Ok(GPT_BIGCODE_PREFIX)
    } else {
        Err(format!(
            "GPT-BigCode token embedding not found: expected {probe}. Every GPT-BigCode \
             checkpoint nests the decoder under a `transformer` submodule."
        ))
    }
}

/// Load an `nn.Linear` projection, rejecting anything that is not `[out, in]`.
///
/// Every shape a GPT-BigCode projection can legally have is known from the
/// config, so check it here rather than letting a mismatch reach `matmul`: an
/// MLX C++ exception crossing the cxx bridge is an uncatchable
/// `std::terminate`, so the process would die with no diagnostic instead of
/// returning a load error naming the tensor.
///
/// A weight that arrives in the transposed `[in, out]` orientation is called
/// out by name. That is the GPT-2 `Conv1D` layout, and GPT-BigCode does not use
/// it (see the module docs); transposing such a weight into the graph would
/// produce a model that loads and generates fluent-looking output from
/// corrupted projections.
///
/// Quantization packs the *input* axis only, so a quantized weight is still
/// `[out_features, packed_in]`. The packed input width matches no float layout
/// and is left to `UnifiedLinear` to reconcile, but the row count is untouched
/// by packing, so the output width is still checked, and for this family that
/// check is not cosmetic. [`Attention::forward`] slices the fused `c_attn`
/// output at offsets derived from the config, so a packed `c_attn` *wider* than
/// the config claims leaves all three slices in bounds and silently hands K and
/// V the wrong channels: the model loads and decodes fluent-looking output from
/// a projection nothing validated. A *narrower* one makes MLX's `slice` clamp
/// and the following `reshape` throw, which is the uncatchable
/// `std::terminate` again.
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
            " That is the GPT-2 `Conv1D` [in, out] orientation. GPT-BigCode builds its \
             projections with `nn.Linear`, so a genuine checkpoint is already [out, in] and \
             must not be transposed."
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

// Attention (fused c_attn QKV with Multi-Query Attention, no RoPE).

pub struct Attention {
    pub c_attn: UnifiedLinear,
    pub c_proj: UnifiedLinear,
    pub num_heads: i32,
    /// `1` under Multi-Query Attention, `num_heads` otherwise.
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
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
        let dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // Fused QKV projection. The split is [dim, dim + kv_dim], not an even
        // three-way split: under MQA the K and V blocks are one head wide while
        // the Q block is `num_heads` heads wide.
        let qkv = self.c_attn.forward(x);
        let q = mlxcel_core::slice_last_dim(&qkv, 0, dim);
        let k = mlxcel_core::slice_last_dim(&qkv, dim, dim + kv_dim);
        let v = mlxcel_core::slice_last_dim(&qkv, dim + kv_dim, dim + 2 * kv_dim);

        // [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim].
        // K and V carry `num_kv_heads` heads, which MLX broadcasts across the
        // query heads inside the attention call, the same way it does for the
        // GQA families.
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

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
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, dim]);

        self.c_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let c_attn = load_linear(
            weights,
            &format!("{prefix}.c_attn"),
            args.n_embd,
            args.c_attn_out_features(),
            group_size,
            bits,
        )?;
        let c_proj = load_linear(
            weights,
            &format!("{prefix}.c_proj"),
            args.n_embd,
            args.n_embd,
            group_size,
            bits,
        )?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            c_attn,
            c_proj,
            num_heads: args.n_head as i32,
            num_kv_heads: args.n_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }
}

// MLP (GELU, no gate/up pattern, width from n_inner).

pub struct MLP {
    pub c_fc: UnifiedLinear,
    pub c_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.c_fc.forward(x);
        // Upstream applies `nn.gelu`, the erf-based exact GELU, which is what
        // `mlxcel_core::utils::gelu_approx` evaluates in this tree (see its own
        // doc comment). The checkpoint config names `gelu_pytorch_tanh`, whose
        // tanh approximation differs from the erf form by under 1e-3; matching
        // upstream keeps this port comparable against the mlx-lm reference,
        // which is the same choice `src/models/gpt2.rs` makes.
        let h = mlxcel_core::utils::gelu_approx(&h);
        self.c_proj.forward(&h)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let n_embd = args.n_embd;
        let intermediate = args.intermediate_size();
        let c_fc = load_linear(
            weights,
            &format!("{prefix}.c_fc"),
            n_embd,
            intermediate,
            group_size,
            bits,
        )?;
        let c_proj = load_linear(
            weights,
            &format!("{prefix}.c_proj"),
            intermediate,
            n_embd,
            group_size,
            bits,
        )?;

        Ok(Self { c_fc, c_proj })
    }
}

// Transformer block (pre-norm, sequential attention then MLP).

pub struct TransformerBlock {
    pub attn: Attention,
    pub mlp: MLP,
    pub ln_1: LayerNorm,
    pub ln_2: LayerNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.ln_1.forward(x);
        let attn_out = self.attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.ln_2.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("{prefix}h.{layer_idx}");

        let attn = Attention::from_weights(weights, args, &format!("{prefix}.attn"))?;
        let mlp = MLP::from_weights(weights, args, &format!("{prefix}.mlp"))?;
        let eps = args.layer_norm_epsilon;
        let ln_1 = layer_norm_from_weights(weights, &format!("{prefix}.ln_1"), args.n_embd, eps)?;
        let ln_2 = layer_norm_from_weights(weights, &format!("{prefix}.ln_2"), args.n_embd, eps)?;

        Ok(Self {
            attn,
            mlp,
            ln_1,
            ln_2,
        })
    }
}

// GPT-BigCode model.

pub struct GptBigCodeModel {
    /// Token embedding, and the output head too when the config ties them.
    pub wte: UnifiedEmbedding,
    /// Learned absolute position embedding.
    pub wpe: UnifiedEmbedding,
    pub h: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    /// Separate output head, present only when `tie_word_embeddings` is false.
    pub lm_head: Option<UnifiedLinear>,
    /// Clamp bound for the learned position lookup.
    ///
    /// Private, and never larger than the number of rows actually present in
    /// `wpe.weight` (enforced by `validate_embedding_table` in
    /// [`GptBigCodeModel::from_weights`], the only constructor). That invariant
    /// is the whole bounds check on the lookup: MLX's gather wraps a negative
    /// index but does not range-check a positive one, so a value larger than
    /// the table reads past the end of the buffer instead of faulting.
    n_positions: usize,
    eos_token_ids: Vec<i32>,
}

impl GptBigCodeModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let input_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = *input_shape.last().unwrap_or(&0);

        let mut h = self.wte.forward(input_ids);

        // Learned absolute positions. Read the offset before any layer runs:
        // `update_and_fetch` advances `caches[0].offset` during the loop below.
        let offset = caches.first().map(|c| c.offset).unwrap_or(0);
        let positions = position_ids(offset, seq_len, self.n_positions);
        if exceeds_position_table(offset, seq_len, self.n_positions) {
            // Once per process: the condition holds for every remaining step of
            // an over-long generation, and it is a prompt/budget problem, not a
            // per-step event worth repeating.
            static CLAMP_WARNED: std::sync::Once = std::sync::Once::new();
            CLAMP_WARNED.call_once(|| {
                // `eprintln!` rather than `tracing::warn!`: only `mlxcel-server`
                // installs a tracing subscriber, and the overrun this reports is
                // reachable from the CLI, where a `tracing` event is a no-op.
                eprintln!(
                    "GPT-BigCode reached position {} but the learned wpe table has only {} \
                     rows; every token from position {} on is embedded at the same row and \
                     output quality degrades. Keep prompt plus generated tokens within the \
                     model's {}-token context.",
                    offset.saturating_add(seq_len.max(1) - 1),
                    self.n_positions,
                    self.n_positions.saturating_sub(1),
                    self.n_positions
                );
            });
        }
        let position_index = mlxcel_core::from_slice_i32(&positions, &[positions.len() as i32]);
        let position_embeds = self.wpe.forward(&position_index);
        h = mlxcel_core::add(&h, &position_embeds);

        for (i, layer) in self.h.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.ln_f.forward(&h);

        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.wte.as_linear(&h),
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

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        // `config.json` is untrusted: reject impossible scalars before any of
        // them sizes an allocation, indexes a table, or divides.
        args.validate()?;

        let prefix = detect_prefix(weights)?;
        let group_size = args.group_size();
        let bits = args.bits();

        let wte_key = format!("{prefix}wte");
        let wte = UnifiedEmbedding::from_weights(weights, &wte_key, group_size, bits)?;
        validate_embedding_table(&wte, &wte_key, args.vocab_size, "vocab_size", args.n_embd)?;

        let wpe_key = format!("{prefix}wpe");
        let wpe = UnifiedEmbedding::from_weights(weights, &wpe_key, group_size, bits)?;
        validate_embedding_table(&wpe, &wpe_key, args.n_positions, "n_positions", args.n_embd)?;

        let mut h = Vec::with_capacity(args.n_layer);
        for i in 0..args.n_layer {
            h.push(TransformerBlock::from_weights(weights, args, prefix, i)?);
        }

        let ln_f = layer_norm_from_weights(
            weights,
            &format!("{prefix}ln_f"),
            args.n_embd,
            args.layer_norm_epsilon,
        )?;

        // Every published GPT-BigCode checkpoint ties the head to `wte` and
        // ships no `lm_head` tensor, which is why `tie_word_embeddings`
        // defaults to true. An untied config must actually carry the tensor;
        // silently falling back to the tied path would produce a model whose
        // logits come from the wrong matrix.
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(load_linear(
                weights,
                "lm_head",
                args.n_embd,
                args.vocab_size,
                group_size,
                bits,
            )?)
        };

        Ok(Self {
            wte,
            wpe,
            h,
            ln_f,
            lm_head,
            n_positions: args.n_positions,
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

// LanguageModel trait implementation.

impl LanguageModel for GptBigCodeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        GptBigCodeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        GptBigCodeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.h.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "gpt_bigcode_tests.rs"]
mod tests;
