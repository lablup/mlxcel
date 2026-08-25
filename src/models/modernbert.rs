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

//! ModernBERT (`model_type: modernbert`) encoder, served through
//! `/v1/embeddings` and `mlxcel embed`.
//!
//! ModernBERT is an 8192-context bidirectional encoder with pre-norm blocks,
//! RoPE instead of an absolute position table, a fused `Wqkv` projection, a
//! GeGLU MLP over a fused `Wi`, and an attention pattern that alternates two
//! local (sliding-window) layers with one global layer. The two attention
//! kinds differ in more than their mask: a local layer also rotates with
//! `local_rope_theta` (10000) while a global layer rotates with
//! `global_rope_theta` (160000), so getting the parity wrong is a silent
//! quality regression that no shape assertion catches.
//!
//! # Layout notes that are easy to get wrong
//!
//! - **Layer 0 has no `attn_norm`.** Upstream makes it `nn.Identity()`, so the
//!   checkpoint simply ships no `layers.0.attn_norm.weight`. Every other layer
//!   must have one; a missing one there is a real error, not an identity.
//! - **`Wqkv` is `[3 * hidden, hidden]` in Q, K, V order**, each block holding
//!   all heads. Upstream reshapes to `[B, L, 3, heads, head_dim]`, which is
//!   exactly `q = out[..., 0..D]`, `k = out[..., D..2D]`, `v = out[..., 2D..3D]`
//!   followed by a per-block head split.
//! - **`Wi` is `[2 * intermediate, hidden]` in (input, gate) order**: upstream
//!   is `input, gate = Wi(x).chunk(2, -1)` then `Wo(act(input) * gate)`. The
//!   activation lands on the *first* half.
//! - Both fused projections load through [`UnifiedLinear`] and are split after
//!   the matmul, never by slicing the weight, because a quantized export packs
//!   each as one tensor.
//! - `hidden_activation` is `gelu`, meaning the exact erf form
//!   ([`mlxcel_core::gelu`]), not the tanh approximation.
//!
//! # Untrusted config
//!
//! `config.json` arrives from a third-party HuggingFace repo, so
//! [`ModernBertArgs::validate`] rejects every scalar that could size an
//! allocation, divide, or truncate through an `as i32` cast before it reaches
//! an MLX entry point. An MLX C++ exception crossing the cxx bridge is an
//! uncatchable `std::terminate`, not a Rust error.

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{
    create_bidirectional_padding_mask, create_bidirectional_window_mask, slice_axis,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use serde_json::Value;

use crate::embeddings::loader::QuantizationParams;

use super::gpt2::{layer_norm_from_weights, validate_embedding_table};

/// `architectures[0]` of a ModernBERT reranker / classifier export.
pub const MODERNBERT_SEQUENCE_CLASSIFICATION_ARCH: &str = "ModernBertForSequenceClassification";

/// Largest `num_hidden_layers` accepted from an untrusted config.
const MAX_HIDDEN_LAYERS: usize = 512;

fn default_max_position_embeddings() -> usize {
    8192
}

fn default_global_rope_theta() -> f32 {
    160_000.0
}

fn default_global_attn_every_n_layers() -> usize {
    3
}

fn default_local_attention() -> usize {
    128
}

fn default_gelu() -> String {
    "gelu".to_string()
}

fn default_classifier_pooling() -> String {
    "cls".to_string()
}

/// ModernBERT `config.json`.
///
/// `norm_eps` and `layer_norm_eps` are two separate optional fields rather than
/// one field with `#[serde(alias)]`: published checkpoints (both
/// `nomic-ai/modernbert-embed-base` and `Alibaba-NLP/gte-reranker-modernbert-base`)
/// carry **both** keys, and serde's `alias` reports a duplicate-field error when
/// the input supplies the field under two of its names. [`Self::norm_eps`]
/// resolves the pair.
#[derive(Debug, Clone, Deserialize)]
pub struct ModernBertArgs {
    #[serde(default)]
    pub architectures: Vec<String>,

    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Modern spelling of the LayerNorm epsilon.
    #[serde(default)]
    pub norm_eps: Option<f32>,
    /// Legacy spelling; both are present in published checkpoints.
    #[serde(default)]
    pub layer_norm_eps: Option<f32>,
    /// Every LayerNorm in the stack is bias-free in published checkpoints.
    #[serde(default)]
    pub norm_bias: bool,

    #[serde(default = "default_global_rope_theta")]
    pub global_rope_theta: f32,
    /// `null` means "use `global_rope_theta` on local layers too".
    #[serde(default)]
    pub local_rope_theta: Option<f32>,

    /// Layer `i` is global when `i % global_attn_every_n_layers == 0`.
    #[serde(default = "default_global_attn_every_n_layers")]
    pub global_attn_every_n_layers: usize,
    /// Total sliding-window width; each side sees `local_attention / 2`.
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,

    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default = "default_gelu")]
    pub hidden_activation: String,

    #[serde(default)]
    pub pad_token_id: Option<i64>,

    /// `cls` or `mean`; consulted only by the sequence-classification head.
    #[serde(default = "default_classifier_pooling")]
    pub classifier_pooling: String,
    #[serde(default = "default_gelu")]
    pub classifier_activation: String,
    #[serde(default)]
    pub classifier_bias: bool,
    #[serde(default)]
    pub num_labels: Option<usize>,
    #[serde(default)]
    pub id2label: Option<serde_json::Map<String, Value>>,
}

impl ModernBertArgs {
    /// Parse and validate a sanitized `config.json` value.
    pub fn from_config(config: &Value) -> Result<Self, String> {
        let args: Self = serde_json::from_value(config.clone())
            .map_err(|e| format!("failed to parse ModernBERT config.json: {e}"))?;
        args.validate()?;
        Ok(args)
    }

    /// Effective LayerNorm epsilon: `norm_eps`, then `layer_norm_eps`, then the
    /// HuggingFace default.
    pub fn norm_eps(&self) -> f32 {
        self.norm_eps.or(self.layer_norm_eps).unwrap_or(1e-5)
    }

    /// `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Layer `i` uses the sliding window instead of full bidirectional
    /// attention. With `global_attn_every_n_layers = 3`, layers 0, 3, 6, ...
    /// are global and the rest are local.
    pub fn is_local_layer(&self, layer: usize) -> bool {
        !layer.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// RoPE base for layer `i`: `local_rope_theta` on a local layer when the
    /// config sets it, `global_rope_theta` otherwise.
    pub fn rope_base(&self, layer: usize) -> f32 {
        if self.is_local_layer(layer) {
            self.local_rope_theta.unwrap_or(self.global_rope_theta)
        } else {
            self.global_rope_theta
        }
    }

    /// `window` argument for [`create_bidirectional_window_mask`], which blocks
    /// at `|q - k| >= window`. ModernBERT attends `|q - k| <= local_attention / 2`,
    /// so the window is one wider than the per-side reach.
    pub fn local_window(&self) -> i32 {
        (self.local_attention / 2 + 1) as i32
    }

    /// Head width of the classification head: `num_labels`, else the size of
    /// `id2label`, else one.
    pub fn num_labels(&self) -> usize {
        self.num_labels
            .or_else(|| self.id2label.as_ref().map(serde_json::Map::len))
            .filter(|&n| n > 0)
            .unwrap_or(1)
    }

    /// `true` when `architectures[0]` marks a sequence-classification export.
    pub fn is_sequence_classifier(&self) -> bool {
        self.architectures
            .first()
            .is_some_and(|arch| arch == MODERNBERT_SEQUENCE_CLASSIFICATION_ARCH)
    }

    /// Reject a config that would divide by zero, size an unbounded
    /// allocation, or overflow an `as i32` cast inside MLX.
    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: usize| -> Result<(), String> {
            if value == 0 || i32::try_from(value).is_err() {
                return Err(format!(
                    "ModernBERT config {name} must be in 1..={}, got {value}",
                    i32::MAX
                ));
            }
            Ok(())
        };
        positive("vocab_size", self.vocab_size)?;
        positive("hidden_size", self.hidden_size)?;
        positive("num_attention_heads", self.num_attention_heads)?;
        positive("intermediate_size", self.intermediate_size)?;
        positive("num_hidden_layers", self.num_hidden_layers)?;
        positive(
            "global_attn_every_n_layers",
            self.global_attn_every_n_layers,
        )?;
        positive("max_position_embeddings", self.max_position_embeddings)?;

        if self.num_hidden_layers > MAX_HIDDEN_LAYERS {
            return Err(format!(
                "ModernBERT config num_hidden_layers {} exceeds the {MAX_HIDDEN_LAYERS} layer cap",
                self.num_hidden_layers
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "ModernBERT config hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            ));
        }
        // `3 * hidden_size` and `2 * intermediate_size` are the fused
        // projection widths that reach `reshape` as i32.
        if i32::try_from(self.hidden_size.saturating_mul(3)).is_err()
            || i32::try_from(self.intermediate_size.saturating_mul(2)).is_err()
        {
            return Err("ModernBERT config fused projection widths overflow i32".to_string());
        }
        if self.local_attention < 2 || i32::try_from(self.local_attention).is_err() {
            return Err(format!(
                "ModernBERT config local_attention must be >= 2, got {}",
                self.local_attention
            ));
        }
        for (name, theta) in [
            ("global_rope_theta", Some(self.global_rope_theta)),
            ("local_rope_theta", self.local_rope_theta),
        ] {
            if let Some(theta) = theta
                && (!theta.is_finite() || theta <= 0.0)
            {
                return Err(format!(
                    "ModernBERT config {name} must be finite and positive, got {theta}"
                ));
            }
        }
        let eps = self.norm_eps();
        if !eps.is_finite() || eps <= 0.0 {
            return Err(format!(
                "ModernBERT config norm_eps must be finite and positive, got {eps}"
            ));
        }
        if self.hidden_activation != "gelu" {
            return Err(format!(
                "ModernBERT hidden_activation `{}` is not supported; only `gelu` is",
                self.hidden_activation
            ));
        }
        if self.classifier_activation != "gelu" {
            return Err(format!(
                "ModernBERT classifier_activation `{}` is not supported; only `gelu` is",
                self.classifier_activation
            ));
        }
        if !matches!(self.classifier_pooling.as_str(), "cls" | "mean") {
            return Err(format!(
                "ModernBERT classifier_pooling `{}` is not supported; expected `cls` or `mean`",
                self.classifier_pooling
            ));
        }
        Ok(())
    }
}

/// Strip the checkpoint's optional `model.` root and drop the tensors an
/// encoder never runs.
///
/// `ModernBertModel` exports the backbone unprefixed; `ModernBertForMaskedLM`
/// and `ModernBertForSequenceClassification` nest it under `model.` and add
/// their own head at the top level. `decoder.*` (the MLM output projection) and
/// `pooler.*` are always dropped; `head.*` and `classifier.*` survive only for
/// the classification head (`keep_head`), so an MLM checkpoint loads as a plain
/// embedder with its head discarded.
pub fn sanitize_modernbert_weights(weights: WeightMap, keep_head: bool) -> WeightMap {
    let mut out = WeightMap::with_capacity(weights.len());
    for (key, value) in weights {
        let key = key
            .strip_prefix("model.")
            .unwrap_or(key.as_str())
            .to_string();
        if key.starts_with("decoder.") || key.starts_with("pooler.") {
            continue;
        }
        if !keep_head && (key.starts_with("head.") || key.starts_with("classifier.")) {
            continue;
        }
        out.insert(key, value);
    }
    out
}

/// Per-forward shape constants shared by every block.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    hidden_size: i32,
    num_heads: i32,
    head_dim: i32,
    intermediate_size: i32,
    scale: f32,
}

/// Apply the GeGLU half-split of a fused `Wi` output.
///
/// `y` is `[.., 2 * intermediate]`; the first half is the activated input and
/// the second half is the multiplicative gate, matching upstream's
/// `input, gate = Wi(x).chunk(2, dim=-1)`.
pub(crate) fn geglu(y: &MlxArray, intermediate: i32) -> UniquePtr<MlxArray> {
    let input = slice_axis(y, -1, 0, intermediate);
    let gate = slice_axis(y, -1, intermediate, 2 * intermediate);
    mlxcel_core::multiply(&mlxcel_core::gelu(&input), &gate)
}

/// One pre-norm ModernBERT block.
struct ModernBertLayer {
    /// `None` for layer 0, where upstream uses `nn.Identity()`.
    attn_norm: Option<LayerNorm>,
    wqkv: UnifiedLinear,
    wo: UnifiedLinear,
    mlp_norm: LayerNorm,
    wi: UnifiedLinear,
    mlp_wo: UnifiedLinear,
    /// Uses the sliding-window mask and `local_rope_theta`.
    local: bool,
    rope_base: f32,
}

impl ModernBertLayer {
    fn from_weights(
        weights: &WeightMap,
        layer: usize,
        args: &ModernBertArgs,
        quant: Option<QuantizationParams>,
    ) -> Result<Self, String> {
        let prefix = format!("layers.{layer}");
        let eps = args.norm_eps();
        let (group_size, bits) = quant
            .map(|q| (q.group_size, q.bits))
            .unwrap_or((DEFAULT_GROUP_SIZE, DEFAULT_BITS));
        let linear = |name: &str| -> Result<UnifiedLinear, String> {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };

        // Layer 0 legitimately ships no attn_norm; any other layer missing one
        // is a truncated or mismatched checkpoint, not an identity.
        let attn_norm_prefix = format!("{prefix}.attn_norm");
        let attn_norm = if weights.contains_key(&format!("{attn_norm_prefix}.weight")) {
            Some(layer_norm_from_weights(
                weights,
                &attn_norm_prefix,
                args.hidden_size,
                eps,
            )?)
        } else if layer == 0 {
            None
        } else {
            return Err(format!(
                "Weight not found: {attn_norm_prefix}.weight (only layer 0 may omit attn_norm)"
            ));
        };

        Ok(Self {
            attn_norm,
            wqkv: linear("attn.Wqkv")?,
            wo: linear("attn.Wo")?,
            mlp_norm: layer_norm_from_weights(
                weights,
                &format!("{prefix}.mlp_norm"),
                args.hidden_size,
                eps,
            )?,
            wi: linear("mlp.Wi")?,
            mlp_wo: linear("mlp.Wo")?,
            local: args.is_local_layer(layer),
            rope_base: args.rope_base(layer),
        })
    }

    /// Split one `[B, L, 3D]` fused projection block into `[B, heads, L, head_dim]`.
    fn head_block(
        qkv: &MlxArray,
        index: i32,
        b: i32,
        l: i32,
        geo: &Geometry,
    ) -> UniquePtr<MlxArray> {
        let block = slice_axis(
            qkv,
            -1,
            index * geo.hidden_size,
            (index + 1) * geo.hidden_size,
        );
        let block = mlxcel_core::reshape(&block, &[b, l, geo.num_heads, geo.head_dim]);
        mlxcel_core::transpose_axes(&block, &[0, 2, 1, 3])
    }

    fn attention(&self, x: &MlxArray, mask: &MlxArray, geo: &Geometry) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let qkv = self.wqkv.forward(x);

        let q = Self::head_block(&qkv, 0, b, l, geo);
        let k = Self::head_block(&qkv, 1, b, l, geo);
        let v = Self::head_block(&qkv, 2, b, l, geo);
        // Full-head, non-traditional (split-half) RoPE over absolute positions
        // 0..L; right padding keeps every real token at its own index.
        let q = mlxcel_core::fast_rope(&q, geo.head_dim, false, self.rope_base, 1.0, 0);
        let k = mlxcel_core::fast_rope(&k, geo.head_dim, false, self.rope_base, 1.0, 0);

        let out = mlxcel_core::layers::attention(&q, &k, &v, geo.scale, Some(mask), 0.0, 0);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, geo.hidden_size]);
        self.wo.forward(&out)
    }

    fn forward(&self, h: &MlxArray, mask: &MlxArray, geo: &Geometry) -> UniquePtr<MlxArray> {
        // Layer 0's `attn_norm` is the identity, so the pre-norm is skipped
        // rather than materialized as a copy.
        let normed = self.attn_norm.as_ref().map(|norm| norm.forward(h));
        let normed: &MlxArray = normed.as_deref().unwrap_or(h);
        let h = mlxcel_core::add(h, &self.attention(normed, mask, geo));

        let projected = self.wi.forward(&self.mlp_norm.forward(&h));
        let mlp = self
            .mlp_wo
            .forward(&geglu(&projected, geo.intermediate_size));
        mlxcel_core::add(&h, &mlp)
    }
}

/// Affine-quantization defaults used when `config.json` carries no
/// `quantization` block; ignored on a dense checkpoint, where
/// [`UnifiedLinear::from_weights`] takes the regular path.
pub(crate) const DEFAULT_GROUP_SIZE: i32 = 64;
pub(crate) const DEFAULT_BITS: i32 = 4;

/// The ModernBERT encoder stack: embeddings, `num_hidden_layers` blocks, and
/// the final LayerNorm.
pub struct ModernBertEncoder {
    tok_embeddings: UnifiedEmbedding,
    embed_norm: LayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    geometry: Geometry,
    local_window: i32,
    hidden_size: usize,
    vocab_size: usize,
}

impl ModernBertEncoder {
    /// Build the stack from an already-sanitized weight map.
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModernBertArgs,
        quant: Option<QuantizationParams>,
    ) -> Result<Self, String> {
        let eps = args.norm_eps();
        let (group_size, bits) = quant
            .map(|q| (q.group_size, q.bits))
            .unwrap_or((DEFAULT_GROUP_SIZE, DEFAULT_BITS));

        let tok_embeddings =
            UnifiedEmbedding::from_weights(weights, "embeddings.tok_embeddings", group_size, bits)?;
        validate_embedding_table(
            &tok_embeddings,
            "embeddings.tok_embeddings",
            args.vocab_size,
            "vocab_size",
            args.hidden_size,
            "hidden_size",
        )?;

        let layers = (0..args.num_hidden_layers)
            .map(|layer| ModernBertLayer::from_weights(weights, layer, args, quant))
            .collect::<Result<Vec<_>, _>>()?;

        let head_dim = args.head_dim();
        Ok(Self {
            tok_embeddings,
            embed_norm: layer_norm_from_weights(weights, "embeddings.norm", args.hidden_size, eps)?,
            layers,
            final_norm: layer_norm_from_weights(weights, "final_norm", args.hidden_size, eps)?,
            geometry: Geometry {
                hidden_size: args.hidden_size as i32,
                num_heads: args.num_attention_heads as i32,
                head_dim: head_dim as i32,
                intermediate_size: args.intermediate_size as i32,
                scale: (head_dim as f32).powf(-0.5),
            },
            local_window: args.local_window(),
            hidden_size: args.hidden_size,
            vocab_size: args.vocab_size,
        })
    }

    /// Width `D` of one hidden state.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Vocabulary size the token ids are bounded by.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Run the stack over one right-padded micro-batch.
    ///
    /// `input_ids` and `attention_mask` are both `[B, L]` int32; the result is
    /// the `[B, L, D]` post-`final_norm` hidden state.
    pub fn encode(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let mask_shape = mlxcel_core::array_shape(attention_mask);
        if ids_shape.len() != 2 || ids_shape != mask_shape {
            return Err(format!(
                "ModernBERT expects [B, L] input_ids and a matching attention_mask, got \
                 {ids_shape:?} and {mask_shape:?}"
            ));
        }

        let mut hidden = self
            .embed_norm
            .forward(&self.tok_embeddings.forward(input_ids));
        // Both masks are additive f32 `0 / -inf` and broadcast against
        // `[B, heads, L, L]` scores: the global one is `[B, 1, 1, L]`, the
        // sliding one `[B, 1, L, L]`.
        let global_mask = create_bidirectional_padding_mask(attention_mask);
        let sliding_mask = create_bidirectional_window_mask(attention_mask, self.local_window);
        for layer in &self.layers {
            let mask = if layer.local {
                &sliding_mask
            } else {
                &global_mask
            };
            hidden = layer.forward(&hidden, mask, &self.geometry);
        }
        Ok(self.final_norm.forward(&hidden))
    }

    /// The sliding-window bound handed to [`create_bidirectional_window_mask`].
    pub fn local_window_bound(&self) -> i32 {
        self.local_window
    }
}
