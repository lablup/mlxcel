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

//! Kyutai Helium (`helium`).
//!
//! Ported from mlx-lm's
//! [`helium.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/helium.py).
//!
//! Helium is a dense Llama-shaped decoder: RMSNorm before attention and before
//! the MLP, grouped-query attention, a SwiGLU MLP over `gate_proj` / `up_proj` /
//! `down_proj`, no QK-norm, no MoE, no sliding window, and Llama's own weight
//! key names. The decoder block, attention and MLP therefore come from
//! [`crate::models::llama3`] unchanged rather than being copied.
//!
//! # The one architectural difference
//!
//! Upstream builds `nn.RoPE(head_dim, traditional=True, base=rope_theta)`.
//! Every other Llama-family model in this tree rotates split-half pairs
//! `(i, i + dims/2)`; Helium rotates interleaved pairs `(2i, 2i+1)`. The two
//! produce identically shaped tensors from identical weights, so running the
//! wrong one is a silent quality regression that no shape assertion can catch.
//!
//! [`ModelArgs::to_llama3_args`] therefore sets
//! [`crate::models::llama3::ModelArgs::rope_traditional`], which
//! [`crate::models::llama3::Attention::forward`] honors on the graph path and
//! which additionally disables the two fused RoPE fast paths, because their C++
//! launchers hardcode `traditional = false` and take no flag. See that method's
//! doc comment for the full routing argument.
//!
//! # Untrusted config
//!
//! `config.json` arrives from a third-party HuggingFace repo in the common
//! `mlxcel generate -m <org>/<repo>` flow, so [`ModelArgs::validate`] rejects
//! every scalar that could size an allocation, divide, truncate through an
//! `as i32` cast, or violate an undocumented precondition of an MLX C++ entry
//! point, and [`validate_weights`] rejects every tensor whose real shape
//! disagrees with the config, on both axes and on both the float and the
//! quantized path, before either can reach a kernel. An MLX C++ exception
//! crossing the cxx bridge is an uncatchable `std::terminate`, not a Rust error,
//! so a check that happens at the first forward pass is not a check.

use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use super::llama3::Llama3Model;

// Configuration.

/// Helium `config.json`.
///
/// Field-for-field the upstream `ModelArgs`, plus `eos_token_id` (Helium ships
/// no chat template and no tokenizer-level EOS, so `config.json` is the only
/// source of the stop token) and the standard `quantization` block that an
/// mlx-community conversion adds.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,

    /// Absent means multi-head attention. Upstream asserts this is set and then
    /// uses it directly; the `Option` here only tolerates a config that omits
    /// it, which resolves to `num_attention_heads`.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Present in every published Helium `config.json`, but **not** what
    /// upstream's attention uses. See [`ModelArgs::head_dim`].
    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    pub vocab_size: usize,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// A `config.json` token-id field, which may be a single int or a list of ints.
///
/// Same shape as the per-family enums in [`crate::models::gpt2`] and
/// [`crate::models::gpt_neox`]; serde fails the whole config when one field
/// does not match its declared type, so the list form has to be accepted even
/// though every published Helium checkpoint writes a single int.
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
    "helium".to_string()
}
fn default_rms_norm_eps() -> f32 {
    1e-8
}
fn default_rope_theta() -> f32 {
    100_000.0
}
fn default_max_position_embeddings() -> usize {
    4096
}

/// Upper bounds on the architecture scalars a Helium `config.json` may declare.
///
/// `config.json` is untrusted input: `mlxcel generate -m <org>/<repo>` downloads
/// a third-party HuggingFace repo and loads it in the same command, and the
/// download layer validates repo ids, filenames and transport but never parses
/// the file, so these fields arrive exactly as the checkpoint author wrote them.
///
/// Each ceiling sits orders of magnitude above Helium 1 (24 layers,
/// `hidden_size` 2560, `intermediate_size` 7040, `vocab_size` 48000). They exist
/// so `num_hidden_layers` cannot size the `Vec::with_capacity` in
/// [`Llama3Model::from_weights`], and so the `as i32` casts these values feed
/// (`head_dim`, `num_heads * head_dim`, `num_kv_heads * head_dim`) stay inside
/// `i32` instead of truncating to a negative number.
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
    /// Head width, and the number of channels RoPE rotates.
    ///
    /// Upstream `HeliumAttention` computes `args.hidden_size // n_heads` and
    /// never reads the `head_dim` field, even though the dataclass requires it,
    /// so `hidden_size / num_attention_heads` is authoritative here too. On
    /// `kyutai/helium-1-preview-2b` the two agree (2560 / 20 = 128 = the
    /// declared `head_dim`); [`ModelArgs::validate`] rejects a config where they
    /// disagree rather than silently preferring one, because a checkpoint whose
    /// projections were built for the other value would produce a reshape
    /// mismatch inside MLX at the first forward pass.
    ///
    /// Only valid after [`ModelArgs::validate`], which rejects
    /// `num_attention_heads == 0`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
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

    /// Stop tokens.
    ///
    /// Helium's `tokenizer_config.json` declares no `eos_token` and no chat
    /// template, so `config.json`'s `eos_token_id` (2, `</s>`) is the only
    /// source. Delegating to [`Llama3Model`] instead would return Llama 3's
    /// hardcoded `[128001, 128009]`, which are outside Helium's 48000-entry
    /// vocabulary and would therefore never match, so generation would only ever
    /// stop at the token limit.
    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdField::ids)
            .unwrap_or_default()
    }

    /// Reject a `config.json` that cannot describe a real Helium, before any of
    /// its fields sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        // The zero checks come first. `0.is_multiple_of(0)` is true, so a config
        // with `hidden_size == num_attention_heads == 0` would pass the
        // divisibility check below and then divide by zero in `head_dim()`.
        if self.num_attention_heads == 0 {
            return Err("Helium num_attention_heads must be at least 1".to_string());
        }
        if self.hidden_size == 0 || self.hidden_size > MAX_HIDDEN_SIZE {
            return Err(format!(
                "Helium hidden_size ({}) must be between 1 and {MAX_HIDDEN_SIZE}",
                self.hidden_size
            ));
        }
        if self.num_hidden_layers == 0 || self.num_hidden_layers > MAX_NUM_HIDDEN_LAYERS {
            return Err(format!(
                "Helium num_hidden_layers ({}) must be between 1 and {MAX_NUM_HIDDEN_LAYERS}",
                self.num_hidden_layers
            ));
        }
        if self.intermediate_size == 0 || self.intermediate_size > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Helium intermediate_size ({}) must be between 1 and {MAX_INTERMEDIATE_SIZE}",
                self.intermediate_size
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(format!(
                "Helium vocab_size ({}) must be between 1 and {MAX_VOCAB_SIZE}",
                self.vocab_size
            ));
        }
        if self.max_position_embeddings == 0
            || self.max_position_embeddings > MAX_MAX_POSITION_EMBEDDINGS
        {
            return Err(format!(
                "Helium max_position_embeddings ({}) must be between 1 and \
                 {MAX_MAX_POSITION_EMBEDDINGS}",
                self.max_position_embeddings
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "Helium hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            ));
        }

        let num_kv_heads = self.num_kv_heads();
        if num_kv_heads == 0 {
            return Err("Helium num_key_value_heads must be at least 1".to_string());
        }
        if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(format!(
                "Helium num_attention_heads ({}) must be divisible by num_key_value_heads \
                 ({num_kv_heads}) for grouped-query attention",
                self.num_attention_heads
            ));
        }

        if let Some(declared) = self.head_dim {
            let derived = self.head_dim();
            if declared != derived {
                return Err(format!(
                    "Helium config declares head_dim {declared} but hidden_size ({}) / \
                     num_attention_heads ({}) is {derived}. Upstream Helium builds its \
                     projections from hidden_size // num_attention_heads and never reads the \
                     head_dim field, so the two must agree; a checkpoint built for either value \
                     would mis-shape the other's reshape.",
                    self.hidden_size, self.num_attention_heads
                ));
            }
        }

        self.validate_rope()?;
        self.validate_norm_eps()?;
        self.validate_quantization()?;
        Ok(())
    }

    /// Reject RoPE parameters that MLX would throw on, or that would silently
    /// NaN every rotated channel.
    ///
    /// `mlx::core::fast::rope` requires `dims` to be positive, even, and no
    /// larger than the input's last axis. Helium rotates the full head, so
    /// `dims == head_dim` and the third condition holds by construction, but the
    /// first two are config-controlled through `hidden_size` and
    /// `num_attention_heads`. MLX enforces them by throwing
    /// `std::invalid_argument`, and `fast_rope` crosses the cxx bridge as
    /// `UniquePtr<MlxArray>` rather than a `Result`, so that throw is an
    /// uncatchable `std::terminate` at the **first forward pass**, long after
    /// the checkpoint appeared to load cleanly. Both are therefore rejected
    /// here, at load.
    ///
    /// `rope_theta` is the base of the frequency exponentiation. A zero,
    /// negative or non-finite base makes every rotated channel NaN without
    /// anything throwing at all, which is harder to diagnose than a crash.
    fn validate_rope(&self) -> Result<(), String> {
        let head_dim = self.head_dim();
        if head_dim < 2 || !head_dim.is_multiple_of(2) {
            return Err(format!(
                "Helium head_dim ({head_dim}, from hidden_size {} / num_attention_heads {}) must \
                 be even and at least 2. MLX rotates channel pairs and rejects an odd or \
                 non-positive rope dimension by throwing, which crosses the cxx bridge as an \
                 uncatchable abort at the first forward pass rather than a load error.",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!(
                "Helium rope_theta ({}) must be a finite positive number; RoPE exponentiates it \
                 per channel, so a zero, negative or non-finite base makes every rotated channel \
                 NaN and that NaN reaches the logits without anything throwing",
                self.rope_theta
            ));
        }
        Ok(())
    }

    /// Reject an `rms_norm_eps` that would NaN every hidden state.
    ///
    /// `fast::rms_norm` computes `x * weight * rsqrt(mean(x^2) + eps)` and never
    /// inspects `eps`, so a non-finite, negative or zero value produces NaN
    /// hidden states with no error at all: the checkpoint loads, generation
    /// runs, and the output is uniform garbage.
    ///
    /// The bound is a range, not an allowlist. Helium 1's `rms_norm_eps` is
    /// `1e-08`, which is unusually small for this family (Llama uses `1e-05`),
    /// and must stay accepted.
    fn validate_norm_eps(&self) -> Result<(), String> {
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "Helium rms_norm_eps ({}) must be a finite positive number; it is added to the \
                 mean square under an rsqrt, so a non-finite, negative or zero value makes every \
                 normalized hidden state NaN and that NaN reaches the logits without anything \
                 throwing",
                self.rms_norm_eps
            ));
        }
        Ok(())
    }

    /// Reject a `quantization` block that would abort an MLX kernel.
    ///
    /// Kept as a family-level early diagnostic even though the shared loaders now
    /// enforce the same bound (issue #929): failing here names Helium and fires
    /// during config validation, before any tensor is touched, rather than at the
    /// first quantized projection the loader happens to reach.
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
        .map_err(|e| format!("Helium config.json: {e}"))
    }

    /// Build the `llama3::ModelArgs` that drives the shared dense decoder.
    ///
    /// The struct literal is deliberate: every field of the shared config is
    /// named here, so a future field added to [`crate::models::llama3::ModelArgs`]
    /// fails this file to compile rather than silently defaulting for Helium.
    ///
    /// `rope_traditional` is the whole reason this conversion exists.
    pub fn to_llama3_args(&self) -> super::llama3::ModelArgs {
        super::llama3::ModelArgs {
            model_type: self.model_type.clone(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            // Pass the derived width explicitly rather than the declared field:
            // `validate` has already rejected a config where they disagree, and
            // this keeps `hidden_size / num_attention_heads` the single source.
            head_dim: Some(self.head_dim()),
            num_key_value_heads: Some(self.num_kv_heads()),
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            rope_theta: self.rope_theta,
            // Helium has no RoPE scaling of any kind upstream.
            rope_scaling: None,
            quantization: self
                .quantization
                .as_ref()
                .map(|q| super::llama3::Quantization {
                    group_size: q.group_size,
                    bits: q.bits,
                }),
            tie_word_embeddings: self.tie_word_embeddings,
            rope_traditional: true,
        }
    }
}

// Weight-shape validation.

/// Whether an MLX dimension equals an expected extent.
///
/// `array_shape` returns `i32`; a negative or oversized value can never match a
/// real extent, so the conversion failing is itself a mismatch. Mirrors the same
/// check in the GPT-2 lineage loaders.
fn dim_eq(dim: i32, expected: usize) -> bool {
    usize::try_from(dim).is_ok_and(|d| d == expected)
}

/// What [`validate_projection`] observed about one linear projection, so callers
/// can cross-check projections that get concatenated together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProjectionShape {
    quantized: bool,
    /// Second axis of `.weight`. Equals `in_features` on the float path and the
    /// packed width on the quantized path.
    cols: i32,
    /// Second axis of `.scales`, or `None` on the float path.
    scale_cols: Option<i32>,
    has_quant_biases: bool,
    /// Whether a dense `.bias` is present, cross-checked across `q_proj` /
    /// `k_proj` / `v_proj` for the same reason `has_quant_biases` is.
    has_dense_bias: bool,
}

/// Check a quantized projection's `.scales` (and `.biases`) against the input
/// width `config.json` claims.
///
/// A quantized `[out_features, in_features]` matrix packs along the input axis
/// only, so the row check in [`validate_projection`] still applies but says
/// nothing about the input width. MLX reconstructs that width as
/// `scales.shape(-1) * group_size` and `extract_quantized_matmul_dims` throws
/// `std::invalid_argument` when it disagrees with the activation's last axis;
/// `validate_quantized_input` throws again when `.biases` and `.scales` differ
/// in shape. `quantized_matmul` crosses the cxx bridge as
/// `UniquePtr<MlxArray>` rather than a `Result`, so either throw is an
/// uncatchable `std::terminate` at the **first forward pass**, long after the
/// checkpoint appeared to load. The q/k/v cross-check in [`validate_weights`]
/// cannot substitute: it only proves the three projections agree with each
/// other, not that any of them agrees with `hidden_size`.
///
/// The declared `group_size` is the right one to check against, not a
/// shape-derived one. The affine loader trusts the declared group size and
/// re-derives `bits` from the shapes, and
/// `FusedQKVLinear::from_weights_separate` does so unconditionally, so the
/// declared value is what reaches MLX. That makes this stricter than the loader
/// for one layout the loader would repair (a declared `group_size` of 16 at
/// 4 bits, the NVFP4-fallback repack, where `reconcile_quantization_layout`
/// re-derives the group size instead); no Helium checkpoint is packed that way,
/// and rejecting it names the mismatch instead of aborting on it.
///
/// Returns the scales column count, or `None` when the projection is not
/// quantized.
fn validate_quantized_scales(
    weights: &WeightMap,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    group_size: i32,
) -> Result<Option<i32>, String> {
    let Some(scales) = weights.get(&format!("{prefix}.scales")) else {
        return Ok(None);
    };
    let scale_shape = mlxcel_core::array_shape(scales);
    if scale_shape.len() != 2 || !dim_eq(scale_shape[0], out_features) {
        return Err(format!(
            "unexpected {prefix}.scales shape {scale_shape:?}: expected {out_features} rows to \
             match {prefix}.weight"
        ));
    }

    let groups = usize::try_from(scale_shape[1]).unwrap_or(0);
    let group_size = usize::try_from(group_size).unwrap_or(0);
    let described = groups.checked_mul(group_size);
    if described != Some(in_features) {
        return Err(format!(
            "unexpected {prefix}.scales shape {scale_shape:?}: {groups} quantization groups at \
             group_size {group_size} describe an input width of {}, but the config says \
             {in_features}. MLX reconstructs a quantized matrix's input width as \
             scales.shape(-1) * group_size and throws when it disagrees with the activation, and \
             that throw crosses the cxx bridge as an uncatchable abort at the first forward pass \
             rather than a load error. Packing compresses the input axis only, so the row count \
             is still correct and cannot catch this.",
            described
                .map(|d| d.to_string())
                .unwrap_or_else(|| "an overflowing number of".to_string())
        ));
    }

    if let Some(biases) = weights.get(&format!("{prefix}.biases")) {
        let bias_shape = mlxcel_core::array_shape(biases);
        if bias_shape != scale_shape {
            return Err(format!(
                "unexpected {prefix}.biases shape {bias_shape:?}: the affine zero points must \
                 have the same shape as {prefix}.scales ({scale_shape:?}). MLX rejects a mismatch \
                 by throwing, which crosses the cxx bridge as an uncatchable abort at the first \
                 forward pass."
            ));
        }
    }

    Ok(Some(scale_shape[1]))
}

/// Check one `[out_features, in_features]` projection against the config.
///
/// A quantized weight is packed along the input axis only, so its row count is
/// still `out_features` and is checked on the same path as a float weight;
/// skipping the row check on the quantized path is exactly the carve-out that
/// let K and V be sliced from arbitrary interior channels in an earlier port in
/// this chain. The input axis is checked too, through the scales rather than
/// through the packed `.weight`, because the packed width alone does not fix a
/// width without a bit count: see [`validate_quantized_scales`]. Only the
/// declared-versus-derived bit width itself is left to `UnifiedLinear`, which
/// reconciles it from these two shapes and returns a load error, not an abort,
/// when they cannot agree.
fn validate_projection(
    weights: &WeightMap,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    group_size: i32,
) -> Result<ProjectionShape, String> {
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
            " That is the [in, out] orientation; Llama-shaped checkpoints store [out, in] and \
             must not be transposed."
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

    let scale_cols =
        validate_quantized_scales(weights, prefix, out_features, in_features, group_size)?;

    let dense_bias = weights.get(&format!("{prefix}.bias"));
    if let Some(bias) = dense_bias {
        let bias_shape = mlxcel_core::array_shape(bias);
        if bias_shape.len() != 1 || !dim_eq(bias_shape[0], out_features) {
            return Err(format!(
                "unexpected {prefix}.bias shape {bias_shape:?}: expected [{out_features}]"
            ));
        }
    }

    Ok(ProjectionShape {
        quantized,
        cols: shape[1],
        scale_cols,
        has_quant_biases: weights.contains_key(&format!("{prefix}.biases")),
        has_dense_bias: dense_bias.is_some(),
    })
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

/// Check a 2-D embedding-shaped table and return its row count.
///
/// The row count is the only place a config that overstates the vocabulary can
/// be caught: MLX's gather adds the axis size to a negative index but performs
/// no range check on a positive one, so an id past the last row reads whatever
/// follows the table in the buffer rather than faulting, and the result reaches
/// the logits. A table with *more* rows than the config claims is accepted,
/// which is how a vocabulary-padded head is stored.
fn validate_table(
    weights: &WeightMap,
    key: &str,
    claimed_rows: usize,
    expected_cols: usize,
    group_size: i32,
) -> Result<(), String> {
    let weight_name = format!("{key}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let shape = mlxcel_core::array_shape(weight);
    if shape.len() != 2 {
        return Err(format!(
            "unexpected {weight_name} shape {shape:?}: expected a 2-D [rows, hidden_size] table"
        ));
    }
    let rows = usize::try_from(shape[0]).unwrap_or(0);
    if rows < claimed_rows {
        return Err(format!(
            "config vocab_size ({claimed_rows}) exceeds the {rows} rows present in {weight_name}. \
             Lookups into this table are bounded by vocab_size, so a config that overstates it \
             indexes past the end of the table, and the gather behind an embedding lookup does \
             not range-check a positive index."
        ));
    }
    if weights.contains_key(&format!("{key}.scales")) {
        // Quantized: the row count above is still the vocabulary bound, but the
        // model width now lives in the scales, so it is checked there. Leaving
        // it unchecked lets a table packed for a different width reach either
        // `quantized_matmul` (an output head) or the first RMSNorm (an embedding
        // table), both of which throw inside MLX and abort the process.
        validate_quantized_scales(weights, key, rows, expected_cols, group_size)?;
    } else if !dim_eq(shape[1], expected_cols) {
        return Err(format!(
            "{weight_name} is {shape:?} but hidden_size is {expected_cols}; an embedding table \
             must be the model width"
        ));
    }
    Ok(())
}

/// Reject a checkpoint whose real tensor shapes disagree with `config.json`,
/// before any of them reaches MLX.
///
/// This has to run **before** [`Llama3Model::from_weights`], not after.
/// `FusedQKVLinear::from_weights_separate` concatenates `q_proj`, `k_proj` and
/// `v_proj` (and their `scales` and `biases`) along axis 0 with no shape check
/// of its own, and `Attention::forward` then reshapes the result using
/// config-derived head counts. Both a mismatched concatenate and a mismatched
/// reshape throw inside MLX, and an MLX C++ exception crossing the cxx bridge is
/// an uncatchable `std::terminate` rather than a load error.
pub fn validate_weights(weights: &WeightMap, args: &ModelArgs) -> Result<(), String> {
    let hidden = args.hidden_size;
    let head_dim = args.head_dim();
    let q_out = args.num_attention_heads * head_dim;
    let kv_out = args.num_kv_heads() * head_dim;
    let group_size = args.group_size();

    validate_table(
        weights,
        "model.embed_tokens",
        args.vocab_size,
        hidden,
        group_size,
    )?;
    validate_norm(weights, "model.norm.weight", hidden)?;
    if !args.tie_word_embeddings {
        validate_table(weights, "lm_head", args.vocab_size, hidden, group_size)?;
    }

    for layer in 0..args.num_hidden_layers {
        let attn = format!("model.layers.{layer}.self_attn");
        let q = validate_projection(
            weights,
            &format!("{attn}.q_proj"),
            q_out,
            hidden,
            group_size,
        )?;
        let k = validate_projection(
            weights,
            &format!("{attn}.k_proj"),
            kv_out,
            hidden,
            group_size,
        )?;
        let v = validate_projection(
            weights,
            &format!("{attn}.v_proj"),
            kv_out,
            hidden,
            group_size,
        )?;
        validate_projection(
            weights,
            &format!("{attn}.o_proj"),
            hidden,
            q_out,
            group_size,
        )?;

        // The fused QKV loader decides "is this quantized?" from `q_proj.scales`
        // alone and then concatenates all three, and it keeps the affine
        // `biases` only when all three carry one. A checkpoint where q, k and v
        // disagree therefore either aborts inside `concatenate` or, worse for
        // the `biases` case, silently dequantizes K and V without their zero
        // points. Neither is recoverable downstream, so both are rejected here.
        if q.quantized != k.quantized || q.quantized != v.quantized {
            return Err(format!(
                "{attn}: q_proj, k_proj and v_proj must all be quantized or all be float; they \
                 are concatenated into one fused projection and the loader decides which path to \
                 take from q_proj alone"
            ));
        }
        if q.cols != k.cols
            || q.cols != v.cols
            || q.scale_cols != k.scale_cols
            || q.scale_cols != v.scale_cols
        {
            return Err(format!(
                "{attn}: q_proj, k_proj and v_proj must share an input width (got {}, {}, {}); \
                 they are concatenated along the output axis",
                q.cols, k.cols, v.cols
            ));
        }
        if q.has_quant_biases != k.has_quant_biases || q.has_quant_biases != v.has_quant_biases {
            return Err(format!(
                "{attn}: q_proj, k_proj and v_proj must either all carry quantization biases or \
                 none of them; the fused loader drops the whole set when one is missing, which \
                 dequantizes the others without their zero points"
            ));
        }
        // The dense `.bias` set has the same all-or-nothing rule as the affine
        // `.biases` set above, and the same silent failure: the fused loader
        // concatenates q/k/v biases only when all three are present and drops
        // the whole set otherwise, so a checkpoint carrying a bias on some of
        // them loads without complaint and runs those projections unbiased.
        if q.has_dense_bias != k.has_dense_bias || q.has_dense_bias != v.has_dense_bias {
            return Err(format!(
                "{attn}: q_proj, k_proj and v_proj must either all carry a bias or none of them; \
                 the fused loader concatenates the three biases into one and silently drops the \
                 whole set when any is missing, which runs the projections that had one without \
                 it"
            ));
        }

        let mlp = format!("model.layers.{layer}.mlp");
        validate_projection(
            weights,
            &format!("{mlp}.gate_proj"),
            args.intermediate_size,
            hidden,
            group_size,
        )?;
        validate_projection(
            weights,
            &format!("{mlp}.up_proj"),
            args.intermediate_size,
            hidden,
            group_size,
        )?;
        validate_projection(
            weights,
            &format!("{mlp}.down_proj"),
            hidden,
            args.intermediate_size,
            group_size,
        )?;

        validate_norm(
            weights,
            &format!("model.layers.{layer}.input_layernorm.weight"),
            hidden,
        )?;
        validate_norm(
            weights,
            &format!("model.layers.{layer}.post_attention_layernorm.weight"),
            hidden,
        )?;
    }

    Ok(())
}

// Model.

/// Kyutai Helium.
///
/// A [`Llama3Model`] built with traditional RoPE, plus Helium's own stop tokens.
/// Everything else, including batched decode, paged decode and the prompt cache,
/// is the shared dense path unchanged.
pub struct HeliumModel {
    inner: Llama3Model,
    eos_token_ids: Vec<i32>,
}

impl HeliumModel {
    /// Load from a checkpoint directory.
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
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
        validate_weights(weights, args)?;

        let inner = Llama3Model::from_weights(weights, &args.to_llama3_args())?;

        Ok(Self {
            inner,
            eos_token_ids: args.eos_token_ids(),
        })
    }

    pub fn inner(&self) -> &Llama3Model {
        &self.inner
    }
}

// LanguageModel trait implementation.

impl LanguageModel for HeliumModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Llama3Model::forward(&self.inner, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.inner
            .forward_with_embeddings_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.inner.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Llama3Model::make_caches(&self.inner)
    }

    fn num_layers(&self) -> usize {
        self.inner.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.inner
            .forward_batched_impl(input_ids, batch_caches, mask, None)
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.inner
            .forward_batched_impl(input_ids, batch_caches, mask, context)
    }

    fn supports_batched_prefill(&self) -> bool {
        true
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        true
    }

    fn supports_paged_decode_backend(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "helium_tests.rs"]
mod tests;
