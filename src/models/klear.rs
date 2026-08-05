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

//! Kuaishou Klear (`Klear`), a Qwen3-shaped sparse MoE decoder.
//!
//! Ported from mlx-lm's
//! [`Klear.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/Klear.py).
//!
//! The backbone is unremarkable and correctly described by the issue: GQA with
//! per-head QK-RMSNorm, RMSNorm layer norms, RoPE, optional `attention_bias`,
//! routed top-k experts over the shared `switch_layers` path plus a shared
//! expert. Three things are not, and each is silent when got wrong.
//!
//! # The `model_type` is capital-K `"Klear"`
//!
//! `Kwai-Klear/Klear-46B-A2.5B-Instruct` declares `"model_type": "Klear"`, not
//! `"klear"`. mlx-lm has to ship `Klear.py` **and** a byte-identical `klear.py`
//! to cover both spellings, because it matches the config value as written. This
//! tree needs only the lowercase arm, because `get_model_type` lowercases
//! `model_type` before matching; without that normalization the arm would miss
//! every published checkpoint. `the_capitalized_model_type_reaches_the_detection_arm`
//! drives the real entry point rather than a helper, so the normalization is
//! part of what is asserted.
//!
//! # The shared expert is blended, not added
//!
//! Every other shared-expert family in this tree adds the shared MLP's output to
//! the routed mixture at a fixed weight of 1. Klear does not. It learns a
//! per-token 2-way softmax and mixes the two branches:
//!
//! ```python
//! coef = mx.softmax(self.coefficient(x), axis=-1, precise=True)
//! y = y_experts * coef[..., :1] + shared * coef[..., 1:]
//! ```
//!
//! so there is an extra `mlp.coefficient` weight (and bias) per MoE layer, and a
//! plain add misweights every token's output while leaving it finite and the
//! text fluent. See [`KlearSparseMoeBlock::forward`].
//!
//! # Routing is sigmoid, and the bias is selection-only
//!
//! `routing_weights = mx.sigmoid(self.gate(x).astype(mx.float32))`, not softmax.
//! `expert_bias` is added to a copy used for **selection only**, and the returned
//! scores are gathered from the unbiased weights; gathering from the biased copy
//! is the classic silent misweighting. `norm_topk_prob` then renormalizes.
//!
//! # `routed_scaling_factor` is in the config and in no implementation
//!
//! `Klear-46B-A2.5B-Instruct` declares `routed_scaling_factor: 2.5`, which looks
//! like the DeepSeek-style knob the neighbouring families use. Upstream's
//! `ModelArgs` does not declare the field at all, so it is dropped on parse and
//! `KlearSparseMoeBlock` never scales anything. This port mirrors upstream
//! rather than guessing, and says so out loud at load; see
//! [`ModelArgs::warn_on_unused_routed_scaling`]. Applying it would multiply
//! every routed contribution by 2.5 against the reference.
//!
//! # Untrusted config
//!
//! Same contract as the other ports in this tree: [`ModelArgs::validate`] rejects
//! every scalar that could size an allocation, divide, or violate an
//! undocumented MLX C++ precondition, and [`validate_weights`] rejects every
//! tensor whose real shape disagrees with the config. An MLX C++ exception
//! crossing the cxx bridge is an uncatchable `std::terminate` at the first
//! forward pass, not a Rust error.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::models::gpt2::{dim_eq, validate_embedding_table};
use crate::models::switch_layers::{SwitchGLU, fused_moe_enabled, moe_weighted_sum};

// Configuration.

/// Klear `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default)]
    pub attention_bias: bool,

    /// Layers forced to a plain dense MLP regardless of `decoder_sparse_step`.
    /// Empty on `Klear-46B-A2.5B-Instruct`.
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,

    #[serde(default)]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    /// A layer is sparse when `(layer_idx + 1) % decoder_sparse_step == 0`. 1 on
    /// the published checkpoint, so every layer is sparse.
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,

    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Upstream's `ModelArgs` does not declare `rope_scaling`, so a scaled block
    /// would be silently dropped there. It is parsed here so a non-trivial one
    /// can be rejected rather than ignored; the published checkpoint writes
    /// `null`.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    /// Declared by the published checkpoint and read by no implementation. See
    /// the module docs and [`ModelArgs::warn_on_unused_routed_scaling`].
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// A `config.json` token-id field, which may be a single int or a list of ints.
///
/// `Klear-46B-A2.5B-Instruct` writes the list form, `[151645, 151643]`.
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
/// The per-tensor override entries the checkpoint also writes here (each
/// `model.layers.{i}.mlp.gate` at 8 bits while the rest is 4-bit) are
/// deliberately not parsed: the shared loaders reconcile bits and group size
/// from the tensor shapes they load.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "Klear".to_string()
}
fn default_num_experts_per_tok() -> usize {
    1
}
fn default_decoder_sparse_step() -> usize {
    1
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    500_000.0
}
fn default_max_position_embeddings() -> usize {
    65_536
}

/// Upper bounds on the architecture scalars a Klear `config.json` may declare.
/// Same rationale as the other ports: `config.json` is untrusted input on the
/// `mlxcel generate -m <org>/<repo>` path. Each sits orders of magnitude above
/// `Klear-46B-A2.5B-Instruct` (32 layers, hidden 2048, 256 experts, vocab
/// 151936).
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
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_NUM_EXPERTS: usize = 4096;
/// See [`MAX_NUM_HIDDEN_LAYERS`].
const MAX_NUM_SHARED_EXPERTS: usize = 1024;

impl ModelArgs {
    /// Head width. Upstream computes `hidden_size // num_attention_heads`; Klear
    /// configs carry no `head_dim` field.
    ///
    /// Only valid after [`ModelArgs::validate`], which rejects
    /// `num_attention_heads == 0`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Width of the shared MLP: `moe_intermediate_size * n_shared_experts`.
    ///
    /// This is one wide MLP, not `n_shared_experts` separate experts. It is
    /// `896 * 1` on the published checkpoint, which the checkpoint confirms:
    /// `mlp.shared_experts.gate_proj` dequantizes to `[896, 2048]`.
    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
            .saturating_mul(self.n_shared_experts)
    }

    pub fn has_shared_expert(&self) -> bool {
        self.n_shared_experts > 0
    }

    /// Whether layer `layer_idx` is sparse.
    ///
    /// Mirrors upstream's `layer_idx not in mlp_only_layers and num_experts > 0
    /// and (layer_idx + 1) % decoder_sparse_step == 0`.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if self.mlp_only_layers.contains(&layer_idx) || self.num_experts == 0 {
            return false;
        }
        if self.decoder_sparse_step == 0 {
            return false;
        }
        (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
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

    /// Stop tokens. `Klear-46B-A2.5B-Instruct` declares `[151645, 151643]`.
    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdField::ids)
            .unwrap_or_default()
    }

    /// Whether the config declares a routed scaling factor that no
    /// implementation applies. See the module docs.
    pub fn declares_unused_routed_scaling(&self) -> bool {
        self.routed_scaling_factor
            .is_some_and(|factor| factor != 1.0)
    }

    /// Print the diagnostic for [`ModelArgs::declares_unused_routed_scaling`].
    ///
    /// `eprintln!` rather than `tracing::warn!`: only `mlxcel-server` installs a
    /// tracing subscriber, so a `tracing` event is a no-op on the CLI path this
    /// is reachable from.
    fn warn_on_unused_routed_scaling(&self) {
        if let Some(factor) = self.routed_scaling_factor.filter(|f| *f != 1.0) {
            eprintln!(
                "Klear config declares routed_scaling_factor {factor}, which no released \
                 implementation reads: upstream mlx-lm's ModelArgs does not name the field, so it \
                 is dropped on parse and KlearSparseMoeBlock scales nothing. This loader mirrors \
                 upstream and does NOT apply it, so a checkpoint decoded here matches the \
                 reference. Applying it would multiply every routed expert contribution by \
                 {factor}."
            );
        }
    }

    /// Reject a `config.json` that cannot describe a real Klear, before any of
    /// its fields sizes an allocation, divides, or reaches an MLX kernel.
    pub fn validate(&self) -> Result<(), String> {
        // The zero checks precede the divisibility check: `0.is_multiple_of(0)`
        // is true in Rust, so a config with `hidden_size == num_attention_heads
        // == 0` would pass it and then divide by zero in `head_dim()`.
        if self.num_attention_heads == 0 || self.num_attention_heads > MAX_NUM_ATTENTION_HEADS {
            return Err(format!(
                "Klear num_attention_heads ({}) must be between 1 and {MAX_NUM_ATTENTION_HEADS}",
                self.num_attention_heads
            ));
        }
        if self.hidden_size == 0 || self.hidden_size > MAX_HIDDEN_SIZE {
            return Err(format!(
                "Klear hidden_size ({}) must be between 1 and {MAX_HIDDEN_SIZE}",
                self.hidden_size
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "Klear hidden_size ({}) must be divisible by num_attention_heads ({}); upstream \
                 derives head_dim as hidden_size // num_attention_heads",
                self.hidden_size, self.num_attention_heads
            ));
        }

        let num_kv_heads = self.num_kv_heads();
        if num_kv_heads == 0 || num_kv_heads > self.num_attention_heads {
            return Err(format!(
                "Klear num_key_value_heads ({num_kv_heads}) must be between 1 and \
                 num_attention_heads ({})",
                self.num_attention_heads
            ));
        }
        if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(format!(
                "Klear num_attention_heads ({}) must be divisible by num_key_value_heads \
                 ({num_kv_heads}) for grouped-query attention",
                self.num_attention_heads
            ));
        }

        if self.num_hidden_layers == 0 || self.num_hidden_layers > MAX_NUM_HIDDEN_LAYERS {
            return Err(format!(
                "Klear num_hidden_layers ({}) must be between 1 and {MAX_NUM_HIDDEN_LAYERS}",
                self.num_hidden_layers
            ));
        }
        if self.intermediate_size == 0 || self.intermediate_size > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Klear intermediate_size ({}) must be between 1 and {MAX_INTERMEDIATE_SIZE}",
                self.intermediate_size
            ));
        }
        if self.vocab_size == 0 || self.vocab_size > MAX_VOCAB_SIZE {
            return Err(format!(
                "Klear vocab_size ({}) must be between 1 and {MAX_VOCAB_SIZE}",
                self.vocab_size
            ));
        }
        if self.max_position_embeddings == 0
            || self.max_position_embeddings > MAX_MAX_POSITION_EMBEDDINGS
        {
            return Err(format!(
                "Klear max_position_embeddings ({}) must be between 1 and \
                 {MAX_MAX_POSITION_EMBEDDINGS}",
                self.max_position_embeddings
            ));
        }
        // Upstream computes `(layer_idx + 1) % decoder_sparse_step`, which
        // divides by zero when the step is zero.
        if self.decoder_sparse_step == 0 && self.num_experts > 0 {
            return Err(
                "Klear decoder_sparse_step must be at least 1; upstream computes \
                 (layer_idx + 1) % decoder_sparse_step to decide which layers are sparse, and 0 \
                 would divide by zero"
                    .to_string(),
            );
        }
        for &idx in &self.mlp_only_layers {
            if idx >= self.num_hidden_layers {
                return Err(format!(
                    "Klear mlp_only_layers names layer {idx}, which is past the end of a \
                     {}-layer stack",
                    self.num_hidden_layers
                ));
            }
        }

        self.validate_routing()?;
        self.validate_shared_expert()?;
        self.validate_rope()?;
        self.validate_rope_scaling()?;
        self.validate_norm_eps()?;
        self.validate_quantization()
    }

    /// Reject routing parameters that would index out of range inside MLX.
    ///
    /// Upstream calls `argpartition(-biased_weights, kth = k - 1, axis=-1)`,
    /// which MLX refuses when `k - 1` falls outside the expert row. MLX signals
    /// that by throwing, and an MLX C++ exception crossing the cxx bridge is an
    /// uncatchable `std::terminate` at the first forward pass rather than a load
    /// error.
    fn validate_routing(&self) -> Result<(), String> {
        if self.num_experts == 0 {
            return Ok(());
        }
        if self.num_experts > MAX_NUM_EXPERTS {
            return Err(format!(
                "Klear num_experts ({}) must be between 1 and {MAX_NUM_EXPERTS}",
                self.num_experts
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(format!(
                "Klear num_experts_per_tok ({}) must be between 1 and num_experts ({}); the \
                 router selects that many indices out of a row of num_experts scores",
                self.num_experts_per_tok, self.num_experts
            ));
        }
        if self.moe_intermediate_size == 0 || self.moe_intermediate_size > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Klear moe_intermediate_size ({}) must be between 1 and {MAX_INTERMEDIATE_SIZE} \
                 when the config declares routed experts",
                self.moe_intermediate_size
            ));
        }
        Ok(())
    }

    fn validate_shared_expert(&self) -> Result<(), String> {
        if self.n_shared_experts > MAX_NUM_SHARED_EXPERTS {
            return Err(format!(
                "Klear n_shared_experts ({}) must not exceed {MAX_NUM_SHARED_EXPERTS}",
                self.n_shared_experts
            ));
        }
        if !self.has_shared_expert() {
            return Ok(());
        }
        let width = self
            .moe_intermediate_size
            .checked_mul(self.n_shared_experts)
            .ok_or_else(|| {
                format!(
                    "Klear shared-expert width overflows: moe_intermediate_size ({}) * \
                     n_shared_experts ({})",
                    self.moe_intermediate_size, self.n_shared_experts
                )
            })?;
        if width == 0 || width > MAX_INTERMEDIATE_SIZE {
            return Err(format!(
                "Klear shared-expert width ({width}) must be between 1 and {MAX_INTERMEDIATE_SIZE}"
            ));
        }
        Ok(())
    }

    /// Reject RoPE parameters MLX would throw on.
    ///
    /// Klear rotates the full head width (`nn.RoPE(self.head_dim, ...)`), so the
    /// only failure mode left is an odd width or a degenerate base. `fast_rope`
    /// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than a `Result`, so
    /// a violation aborts the process at the first forward pass.
    fn validate_rope(&self) -> Result<(), String> {
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(format!(
                "Klear rope_theta ({}) must be a finite positive number; RoPE exponentiates it per \
                 channel, so a zero, negative or non-finite base makes every rotated channel NaN \
                 and that NaN reaches the logits without anything throwing",
                self.rope_theta
            ));
        }
        let head_dim = self.head_dim();
        if !head_dim.is_multiple_of(2) {
            return Err(format!(
                "Klear head width resolves to an odd {head_dim} (hidden_size {} / \
                 num_attention_heads {}); Klear rotates the full head width and RoPE rotates \
                 channel pairs, so MLX throws on an odd `dims`",
                self.hidden_size, self.num_attention_heads
            ));
        }
        Ok(())
    }

    /// Reject a `rope_scaling` block this loader does not implement.
    ///
    /// Upstream's `ModelArgs` does not declare the field, so a scaled block is
    /// dropped there without comment. Rejecting is the safer reading: silently
    /// ignoring one would place every token at the wrong position while the
    /// model loaded and generated fluent text.
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
            "Klear rope_scaling ({scaling:?}) is not implemented for this family. Upstream's \
             ModelArgs does not declare the field at all, so it silently drops the block; this \
             loader rejects it rather than placing every token at the wrong position while still \
             generating fluent text. Only an absent, empty or \"default\" block is accepted."
        ))
    }

    fn validate_norm_eps(&self) -> Result<(), String> {
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "Klear rms_norm_eps ({}) must be a finite positive number; it is added to the mean \
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
        .map_err(|e| format!("Klear config.json: {e}"))
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

/// Klear attention: GQA with per-head QK-RMSNorm, RoPE over the full head width,
/// and optional bias on every projection.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
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

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Upstream normalizes before the transpose, on `[B, L, H, D]`. RMSNorm
        // acts on the last axis either way, so the order does not matter, but
        // the reshape must come first so the norm sees one head at a time.
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
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
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.head_dim() as i32;
        let load = |leaf: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{leaf}"), group_size, bits)
        };
        Ok(Self {
            q_proj: load("q_proj")?,
            k_proj: load("k_proj")?,
            v_proj: load("v_proj")?,
            o_proj: load("o_proj")?,
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
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base: args.rope_theta,
        })
    }
}

// Feed-forward.

/// Upstream's `KlearMLP`: a SwiGLU feed-forward block.
///
/// Used both for the dense layers (at `intermediate_size`) and for the shared
/// expert (at `moe_intermediate_size * n_shared_experts`).
pub struct KlearMlp {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl KlearMlp {
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

/// Upstream's `KlearSparseMoeBlock`.
pub struct KlearSparseMoeBlock {
    pub gate: UnifiedLinear,
    pub experts: SwitchGLU,
    pub shared_experts: KlearMlp,
    /// The 2-way blend head. `[2, hidden_size]` plus a `[2]` bias, since
    /// upstream's `nn.Linear(hidden_size, 2)` defaults to `bias=True`.
    pub coefficient: UnifiedLinear,
    /// Selection-only correction bias, `[num_experts]`.
    pub expert_bias: UniquePtr<MlxArray>,
    pub top_k: i32,
    pub norm_topk_prob: bool,
}

impl KlearSparseMoeBlock {
    /// **The shared expert is blended, not added.**
    ///
    /// Every other shared-expert family in this tree adds the shared MLP's
    /// output to the routed mixture at a fixed weight of 1. Klear learns a
    /// per-token 2-way softmax over `coefficient(x)` and mixes:
    /// `y = y_experts * coef[..0] + shared * coef[..1]`. A plain add misweights
    /// every token while leaving the output finite and the text fluent, which is
    /// why the blend has its own unit test rather than only a shape check.
    ///
    /// Routing is sigmoid rather than softmax, and the `expert_bias` is added to
    /// a copy used for **selection only** while the returned scores are gathered
    /// from the unbiased weights.
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = *orig_shape.last().unwrap_or(&0);
        let x_flat = if orig_shape.len() > 2 {
            let tokens: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[tokens, hidden])
        } else {
            mlxcel_core::copy(x)
        };
        let in_type = mlxcel_core::array_dtype(&x_flat);

        // Sigmoid, not softmax, and computed in float32.
        let gates = self.gate.forward(&x_flat);
        let routing_weights =
            mlxcel_core::sigmoid(&mlxcel_core::astype(&gates, mlxcel_core::dtype::FLOAT32));

        // Selection uses the biased copy; the returned scores come from the
        // unbiased weights.
        let biased = mlxcel_core::add(&routing_weights, &self.expert_bias);

        // `argpartition(-biased, kth = k - 1)` then the FIRST k, upstream's
        // orientation. It picks the same set as the mirrored form whenever the
        // scores are distinct and a different set when they tie.
        let biased_shape = mlxcel_core::array_shape(&biased);
        let n_experts = *biased_shape.last().unwrap_or(&0);
        let kth = (self.top_k - 1).clamp(0, (n_experts - 1).max(0));
        let order = mlxcel_core::argpartition(&mlxcel_core::negative(&biased), kth, -1);
        let indices = slice_axis(&order, -1, 0, self.top_k);

        let scores = mlxcel_core::take_along_axis(&routing_weights, &indices, -1);
        let scores = if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&scores, -1, true);
            mlxcel_core::divide(&scores, &sum)
        } else {
            scores
        };
        // Upstream casts back to the activation dtype here, before the combine.
        let scores = mlxcel_core::astype(&scores, in_type);

        let routed = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1 && fused_moe_enabled() {
                self.experts
                    .forward_fused_kernel(&x_flat, &indices, &scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.experts.forward(&x_flat, &indices);
                    moe_weighted_sum(&expert_out, &scores, in_type)
                }
            }
        };

        // The blend. `coefficient` is a 2-wide projection softmaxed in float32,
        // giving a per-token weight for the routed branch and one for the shared
        // branch.
        let coef = mlxcel_core::softmax(
            &mlxcel_core::astype(
                &self.coefficient.forward(&x_flat),
                mlxcel_core::dtype::FLOAT32,
            ),
            -1,
        );
        let coef = mlxcel_core::astype(&coef, in_type);
        let routed_coef = slice_axis(&coef, -1, 0, 1);
        let shared_coef = slice_axis(&coef, -1, 1, 2);

        let shared = self.shared_experts.forward(&x_flat);
        let blended = mlxcel_core::add(
            &mlxcel_core::multiply(&routed, &routed_coef),
            &mlxcel_core::multiply(&shared, &shared_coef),
        );

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&blended, &orig_shape)
        } else {
            blended
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
            gate: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate"),
                group_size,
                bits,
            )?,
            experts: SwitchGLU::from_weights(
                weights,
                &format!("{prefix}.experts"),
                group_size,
                bits,
            )?,
            shared_experts: KlearMlp::from_weights(
                weights,
                args,
                &format!("{prefix}.shared_experts"),
            )?,
            coefficient: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.coefficient"),
                group_size,
                bits,
            )?,
            expert_bias,
            top_k: args.num_experts_per_tok as i32,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

/// Either the dense MLP or the sparse block, per layer.
pub enum FeedForward {
    Dense(KlearMlp),
    Sparse(Box<KlearSparseMoeBlock>),
}

impl FeedForward {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(block) => block.forward(x),
        }
    }
}

// Decoder layer and model.

pub struct DecoderLayer {
    pub self_attn: Attention,
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
        Ok(Self {
            self_attn: Attention::from_weights(weights, args, &format!("{prefix}.self_attn"))?,
            mlp: if args.is_moe_layer(layer_idx) {
                FeedForward::Sparse(Box::new(KlearSparseMoeBlock::from_weights(
                    weights,
                    args,
                    &format!("{prefix}.mlp"),
                )?))
            } else {
                FeedForward::Dense(KlearMlp::from_weights(
                    weights,
                    args,
                    &format!("{prefix}.mlp"),
                )?)
            },
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

/// Kuaishou Klear.
pub struct KlearModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    /// Upstream always builds a separate `lm_head`; `None` only when a config
    /// declares tied embeddings, which no published Klear does.
    pub lm_head: Option<UnifiedLinear>,
    eos_token_ids: Vec<i32>,
}

impl KlearModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }
        let h = self.norm.forward(&h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
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
        // Reject an impossible config before reading 24 GB of weights, not after.
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
        args.warn_on_unused_routed_scaling();
        validate_weights(weights, args)?;

        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;
        // Token ids are bounded by `vocab_size`, a config field, and an
        // embedding gather wraps a negative index but does not range-check a
        // positive one, so a config that overstates the table turns an ordinary
        // prompt into an out-of-bounds read whose result reaches the logits.
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

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            eos_token_ids: args.eos_token_ids(),
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
    let head_dim = args.head_dim();
    let q_size = args.num_attention_heads * head_dim;
    let kv_size = args.num_kv_heads() * head_dim;
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
        validate_norm(weights, &format!("{attn}.q_norm.weight"), head_dim)?;
        validate_norm(weights, &format!("{attn}.k_norm.weight"), head_dim)?;
        validate_norm(weights, &format!("{prefix}.input_layernorm.weight"), hidden)?;
        validate_norm(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
        )?;

        let mlp = format!("{prefix}.mlp");
        if args.is_moe_layer(layer) {
            validate_projection(
                weights,
                &format!("{mlp}.gate"),
                args.num_experts,
                hidden,
                group_size,
                bits,
            )?;
            // The blend head, which no other shared-expert family in this tree
            // has. Missing it is not a checkpoint this loader can reproduce: the
            // fallback would be a plain add, which misweights every token.
            validate_projection(
                weights,
                &format!("{mlp}.coefficient"),
                2,
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
            validate_mlp(
                weights,
                &format!("{mlp}.shared_experts"),
                hidden,
                args.shared_expert_intermediate_size(),
                group_size,
                bits,
            )?;
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

impl LanguageModel for KlearModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        KlearModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        KlearModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "klear_tests.rs"]
mod tests;
