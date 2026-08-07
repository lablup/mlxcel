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

//! Florence-2 checkpoint plumbing: whole-model config parsing and the
//! weight-key normalization that has to happen before either half of the
//! model is built.
//!
//! Kept apart from `model.rs` because these two run once at load time and
//! have nothing to do with the forward path.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py

use anyhow::{Result, anyhow};
use serde_json::Value;

use mlxcel_core::weights::WeightMap;

use crate::vision::encoders::florence2_davit;

use super::Florence2VisionConfig;

/// Weight-key prefix of the language model inside a Florence-2 checkpoint.
pub(crate) const FLORENCE2_TEXT_PREFIX: &str = "language_model.";

/// Affine quantization parameters of an `mlx-community` Florence-2 export,
/// read from the top-level `quantization` object of `config.json`.
///
/// Every published quantized conversion (`-3bit`, `-4bit`, `-6bit`, `-8bit`,
/// in both `base-ft` and `large-ft`) ships `{"group_size": 64, "bits": N}`,
/// so in practice only `bits` varies. The group size is still read rather
/// than assumed: [`mlxcel_core::layers::UnifiedLinear`] and
/// [`mlxcel_core::layers::UnifiedEmbedding`] treat the declared group size as
/// trusted and derive the bit width back from the packed shapes, so handing
/// them a wrong group size would dequantize the whole model on the wrong
/// stride and produce plausible-looking garbage rather than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Florence2Quantization {
    pub group_size: i32,
    pub bits: i32,
}

/// Upper bound accepted for `quantization.group_size`. MLX's own affine
/// kernels take 32, 64, or 128; the bound is looser than that so a future
/// group size does not need a code change, while still keeping an absurd
/// value out of the packing reconciliation.
const MAX_QUANT_GROUP_SIZE: i32 = 4096;

impl Default for Florence2Quantization {
    fn default() -> Self {
        Self::DENSE
    }
}

impl Florence2Quantization {
    /// What a checkpoint with no `quantization` object gets. Both fields are
    /// inert in that case: the unified layers fall back to the dense
    /// `Linear` / `Embedding` path for every prefix that has no `.scales`
    /// sibling, and a dense checkpoint has none. The values are MLX's own
    /// `nn.quantize` defaults so a hand-built config that omits the block on
    /// a genuinely quantized export still lands on the common packing.
    pub const DENSE: Self = Self {
        group_size: 64,
        bits: 4,
    };

    /// Read the checkpoint's `quantization` object. A checkpoint without one
    /// is dense and gets [`Self::DENSE`].
    ///
    /// The four locations searched are the four
    /// [`Self::config_is_quantized`] accepts, and they have to stay the same
    /// four. That predicate decides whether the bf16 to f16 conversion is
    /// skipped; this one decides the stride everything is dequantized on. A
    /// checkpoint nesting the block somewhere only the predicate looked would
    /// keep its bf16 scales, correctly, and then dequantize the whole model at
    /// the dense fallback of group 64 / 4 bits, which is the mis-stride the
    /// struct comment warns about. Every published Florence-2 conversion puts
    /// the block at the top level, so this is a guard against divergence
    /// rather than a path any of them take.
    ///
    /// Both fields end up as a divisor and a shift width inside the packing
    /// reconciliation, so they are range-checked here where the offending
    /// `config.json` field can be named.
    ///
    /// A field that is absent falls back to the [`Self::DENSE`] value, but a
    /// field that is *present* and unreadable is an error rather than the same
    /// fallback. Silently substituting 64 for an unparseable `group_size` is
    /// the one failure mode this type cannot detect later: the unified layers
    /// trust the declared group size, so a wrong one dequantizes the whole
    /// model on the wrong stride and produces plausible-looking garbage
    /// instead of an error.
    pub fn from_model_config(config: &Value) -> Result<Self> {
        let text_config = config.get("text_config");
        let Some(quant) = ["quantization", "quantization_config"]
            .into_iter()
            .flat_map(|key| [config.get(key), text_config.and_then(|t| t.get(key))])
            .flatten()
            .find(|value| value.is_object())
        else {
            return Ok(Self::DENSE);
        };
        let group_size = quant_field(quant, "group_size")?.unwrap_or(Self::DENSE.group_size);
        let bits = quant_field(quant, "bits")?.unwrap_or(Self::DENSE.bits);
        if !(1..=MAX_QUANT_GROUP_SIZE).contains(&group_size) {
            return Err(anyhow!(
                "Florence-2 quantization.group_size {group_size} outside 1..={MAX_QUANT_GROUP_SIZE}"
            ));
        }
        // Same range `mlxcel_core::layers::validate_quantization_params`
        // accepts, and permissive for the same reason: the unified layers
        // re-derive an effective bit width from the packed shapes, so an
        // allowlist of the four widths mlx-community publishes today would
        // reject a legitimate future export. The bound only keeps a value that
        // can match no real tensor out of the reconciler.
        if !(1..=32).contains(&bits) {
            return Err(anyhow!(
                "Florence-2 quantization.bits {bits} outside 1..=32"
            ));
        }
        Ok(Self { group_size, bits })
    }

    /// True when `config.json` declares quantization metadata at all.
    ///
    /// Deliberately not `parsed != DENSE`: a genuinely 4-bit, group-64 export
    /// declares exactly the values [`Self::DENSE`] carries, so comparing the
    /// parsed parameters cannot separate it from a dense checkpoint. Only the
    /// presence of the block can, and that is what gates the bf16 to f16
    /// conversion in every `load` on this family.
    pub fn config_is_quantized(config: &Value) -> bool {
        crate::models::sanitize::config_has_quantization_metadata(config)
    }
}

/// Read one integer field of the `quantization` object, separating "absent"
/// from "present but unreadable".
///
/// `Ok(None)` means the key is not there and the caller's default applies.
/// Everything else that is not a JSON integer fitting in `i32` is an error
/// naming the field, which is the difference from the permissive
/// [`config_i32`] used for the shape fields: those are all range-checked
/// against the loaded tensors afterwards, whereas a quantization parameter is
/// taken on trust by [`mlxcel_core::layers::UnifiedLinear`] and
/// [`mlxcel_core::layers::UnifiedEmbedding`] and has no later contradiction to
/// trip over. The `i32` conversion is checked for the same reason: the `as
/// i32` cast [`config_i32`] performs truncates an out-of-range JSON integer
/// into a plausible small number rather than rejecting it.
fn quant_field(quant: &Value, key: &str) -> Result<Option<i32>> {
    let Some(value) = quant.get(key) else {
        return Ok(None);
    };
    let parsed = value.as_i64().and_then(|v| i32::try_from(v).ok());
    parsed.map(Some).ok_or_else(|| {
        anyhow!("Florence-2 quantization.{key} must be an integer that fits in i32, got {value}")
    })
}

/// Tensors the Florence-2 forward path consumes densely, by weight key.
///
/// `image_projection` is a bare right-hand matmul operand and
/// `visual_temporal_embed.pos_idx_to_embed` is a precomputed sinusoidal
/// buffer; neither is an `nn.Linear` or an `nn.Embedding`, so upstream's
/// `nn.quantize` walk leaves both dense and every published conversion does.
/// A checkpoint that packed one anyway would hand a `uint32` tensor to
/// `matmul` / `slice`, and MLX's eager throw cannot cross the cxx bridge, so
/// it has to be refused here rather than surfacing as an error later.
///
/// Only the tensors matched by an exact key or a key prefix live in this list.
/// The convolutions and the normalization weights are matched by shape of key
/// instead, in [`reject_unsupported_quantized_tensors`], because they recur
/// per block and per layer rather than at a fixed path. The normalization
/// weights belong to the same category for the same reason: every LayerNorm
/// here is read as a raw `{prefix}.weight` / `{prefix}.bias` pair and handed
/// to `fast::layer_norm`, which takes a float weight and throws on a packed
/// one.
const FLORENCE2_DENSE_ONLY_TENSORS: &[&str] = &["image_projection", "visual_temporal_embed"];

/// Refuse a quantized checkpoint that packs a tensor this implementation can
/// only consume dense.
///
/// This is the narrowed form of the blanket refusal that #854 installed. The
/// projections and embedding tables now load quantized, so the only remaining
/// gap is the handful of raw parameters, LayerNorms, and convolutions listed
/// above and detected below, all of which upstream leaves dense.
pub(crate) fn reject_unsupported_quantized_tensors(weights: &WeightMap) -> Result<(), String> {
    for key in weights.keys() {
        let Some(stem) = key.strip_suffix(".scales") else {
            continue;
        };
        let unsupported = FLORENCE2_DENSE_ONLY_TENSORS
            .iter()
            .any(|dense| stem == *dense || stem.starts_with(&format!("{dense}.")))
            // The conv stack (`convs.*.proj`, the depthwise `*.dw` inside each
            // block) reaches `mlxcel_core::conv2d`, which has no quantized
            // form here. `nn.quantize` skips `nn.Conv2d`, so this is a guard
            // against a non-standard export rather than a known one.
            || (stem.contains("convs") && stem.ends_with("proj"))
            || stem.ends_with(".dw")
            // Every LayerNorm on this family is loaded as a raw weight/bias
            // pair by `layers::layer_norm` or
            // `florence2_davit_blocks::layer_norm_from_weights` and handed to
            // `mlxcel_core::layers::LayerNorm`, i.e. to `fast::layer_norm`, so
            // a packed one presents a `uint32` weight to that kernel and
            // aborts the same way the conv arm would. `nn.quantize` skips
            // `nn.LayerNorm`, so this guards a non-standard export rather than
            // a known one.
            //
            // Matching on the last dot-separated segment containing `norm`
            // covers every normalization prefix in the family:
            // `layernorm_embedding` on both text stacks,
            // `self_attn_layer_norm` / `encoder_attn_layer_norm` /
            // `final_layer_norm` on every text block, `image_proj_norm` on the
            // fusion stage, and the DaViT `convs.N.norm` /
            // `window_attn.norm` / `channel_attn.norm` / `ffn.norm`. It cannot
            // false-positive on a real conversion: every tensor `nn.quantize`
            // packs here ends in one of `q_proj`, `k_proj`, `v_proj`,
            // `out_proj`, `fc1`, `fc2`, `qkv`, `proj`, `shared`, `lm_head`,
            // `embed_positions`, `row_embeddings`, or `column_embeddings`,
            // none of which contains `norm`.
            || stem
                .rsplit('.')
                .next()
                .is_some_and(|segment| segment.contains("norm"));
        if unsupported {
            return Err(format!(
                "Florence-2 quantized checkpoint packs {stem}, which this implementation consumes as a dense tensor. Projections, embedding tables, and the LM head load quantized; this one does not. Use a bf16 or f16 export, for example mlx-community/Florence-2-base-ft-bf16."
            ));
        }
    }
    Ok(())
}

/// Both halves of a Florence-2 `config.json` plus the fusion-stage fields.
#[derive(Debug, Clone)]
pub struct Florence2Config {
    pub text: Florence2TextConfig,
    pub vision: Florence2VisionConfig,
    /// Placeholder id the processor may emit for the image. Florence-2 does
    /// not actually splice image features into placeholder slots, so any
    /// occurrence is dropped before the prompt is embedded. Upstream defaults
    /// this to the text vocabulary size (51289), one past the last real id.
    pub image_token_id: i32,
}

impl Florence2Config {
    /// Parse a full Florence-2 `config.json` (`model_type: florence2`).
    pub fn from_model_config(config: &Value) -> Result<Self> {
        let text = Florence2TextConfig::from_model_config(config)?;
        let vision = Florence2VisionConfig::from_model_config(config)?;
        let image_token_id = config
            .get("image_token_id")
            .or_else(|| config.get("image_token_index"))
            .and_then(Value::as_i64)
            .map(|v| v as i32)
            .unwrap_or(text.vocab_size);
        Ok(Self {
            text,
            vision,
            image_token_id,
        })
    }
}

/// Normalize a full Florence-2 checkpoint's weight keys.
///
/// Three passes, in order:
///
/// 1. The DaViT conv channels-last remap and `position_ids` drop from
///    [`florence2_davit::sanitize`]. Both of its key guards are specific
///    enough (`convs`/`proj.weight`, `blocks`/`dw.weight`) that running them
///    over the whole checkpoint leaves `language_model.*` untouched, and both
///    are idempotent, so this is a no-op on the `mlx-community` exports that
///    already ship channels-last conv weights.
/// 2. Drop `final_logits_bias`. HuggingFace BART carries it as a registered
///    buffer of zeros; it is not a parameter and there is no LM-head bias in
///    this implementation.
/// 3. BART shared-embedding naming. The encoder and decoder token tables are
///    tied to `model.shared` and exports vary in which of the three they
///    materialize, so fill `model.shared.weight` from an `embed_tokens` copy
///    when only the latter is present.
pub fn sanitize(weights: WeightMap) -> WeightMap {
    let weights = florence2_davit::sanitize(weights);

    let mut out = WeightMap::with_capacity(weights.len());
    for (key, value) in weights {
        if key.contains("final_logits_bias") {
            continue;
        }
        out.insert(key, value);
    }

    // A quantized export carries three planes per table, so the fill has to
    // copy `.scales` and `.biases` alongside `.weight`; leaving them behind
    // would present `model.shared` as a dense table whose `.weight` is packed
    // `uint32`, which reaches MLX and aborts instead of erroring.
    let shared = format!("{FLORENCE2_TEXT_PREFIX}model.shared.weight");
    if !out.contains_key(&shared) {
        let source_prefix = [
            format!("{FLORENCE2_TEXT_PREFIX}model.encoder.embed_tokens"),
            format!("{FLORENCE2_TEXT_PREFIX}model.decoder.embed_tokens"),
        ]
        .into_iter()
        .find(|prefix| out.contains_key(&format!("{prefix}.weight")));
        if let Some(source_prefix) = source_prefix {
            let target_prefix = format!("{FLORENCE2_TEXT_PREFIX}model.shared");
            for plane in ["weight", "scales", "biases"] {
                if let Some(tensor) = out.get(&format!("{source_prefix}.{plane}")) {
                    let copied = mlxcel_core::copy(tensor);
                    out.insert(format!("{target_prefix}.{plane}"), copied);
                }
            }
        }
    }
    out
}

/// Upper bound accepted for `encoder_layers` / `decoder_layers`. Real
/// Florence-2 exports ship 6 (base) or 12 (large). The cap exists because
/// [`super::encoder::Florence2Encoder::from_weights`] and
/// [`super::decoder::Florence2Decoder::from_weights`] size a `Vec` from this
/// field before they look up a single weight, so a negative value out of a
/// hostile `config.json` becomes `usize::MAX` and aborts with a capacity
/// overflow.
const MAX_LAYERS: i32 = 256;

/// Upper bound accepted for `max_position_embeddings`. Florence-2 ships 1024.
///
/// The field bounds two separate allocations: the position-table slice the
/// encoder and decoder take (validated against the loaded table at load time)
/// and the O(n^2) host buffer [`super::layers::additive_causal_mask`] fills
/// for a multi-token decoder call. An unbounded value out of a hostile
/// `config.json` is an out-of-memory abort rather than an error return.
const MAX_POSITION_EMBEDDINGS: i32 = 65_536;

/// BART encoder-decoder shape parameters, parsed from the `text_config`
/// object of a Florence-2 `config.json` (`model_type: florence2_language`).
#[derive(Debug, Clone)]
pub struct Florence2TextConfig {
    pub d_model: i32,
    pub encoder_layers: i32,
    pub decoder_layers: i32,
    pub encoder_attention_heads: i32,
    pub decoder_attention_heads: i32,
    pub encoder_ffn_dim: i32,
    pub decoder_ffn_dim: i32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub scale_embedding: bool,
    pub pad_token_id: i32,
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    pub decoder_start_token_id: i32,
    /// Packing of the projections, the shared token table, the position
    /// tables, and the LM head. [`Florence2Quantization::DENSE`] for a bf16 or
    /// f16 export.
    pub quantization: Florence2Quantization,
}

fn config_i32(config: &Value, key: &str) -> Option<i32> {
    config.get(key).and_then(Value::as_i64).map(|v| v as i32)
}

impl Florence2TextConfig {
    /// Parse from the `text_config` sub-object itself.
    pub fn from_text_config(config: &Value) -> Result<Self> {
        let require = |key: &str| -> Result<i32> {
            config_i32(config, key)
                .ok_or_else(|| anyhow!("Florence-2 text_config missing field: {key}"))
        };
        let d_model = require("d_model")?;
        let parsed = Self {
            d_model,
            encoder_layers: require("encoder_layers")?,
            decoder_layers: require("decoder_layers")?,
            encoder_attention_heads: require("encoder_attention_heads")?,
            decoder_attention_heads: require("decoder_attention_heads")?,
            encoder_ffn_dim: config_i32(config, "encoder_ffn_dim").unwrap_or(4 * d_model),
            decoder_ffn_dim: config_i32(config, "decoder_ffn_dim").unwrap_or(4 * d_model),
            vocab_size: require("vocab_size")?,
            max_position_embeddings: config_i32(config, "max_position_embeddings").unwrap_or(1024),
            scale_embedding: config
                .get("scale_embedding")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            pad_token_id: config_i32(config, "pad_token_id").unwrap_or(1),
            bos_token_id: config_i32(config, "bos_token_id").unwrap_or(0),
            eos_token_id: config_i32(config, "eos_token_id").unwrap_or(2),
            decoder_start_token_id: config_i32(config, "decoder_start_token_id").unwrap_or(2),
            // A bare `text_config` sub-object carries no `quantization` block,
            // so this resolves to `DENSE` on the usual path and
            // `from_model_config` overwrites it from the top level. Reading it
            // here as well means a caller that hands the *whole* document to
            // `from_text_config` still gets the right packing.
            quantization: Florence2Quantization::from_model_config(config)?,
        };
        // Every field below becomes a shape, an index, a divisor, or a `Vec`
        // capacity somewhere downstream. MLX throws on an out-of-range shape
        // and the throw crosses an FFI boundary that cannot carry it, so a bad
        // value aborts the process rather than surfacing as an error; these
        // checks are what keep an untrusted `config.json` from getting that
        // far. They also have to run *before* the divisibility check below,
        // which would itself divide by zero on `encoder_attention_heads: 0`.
        if parsed.d_model < 1 {
            return Err(anyhow!(
                "Florence-2 text_config d_model must be positive, got {}",
                parsed.d_model
            ));
        }
        if parsed.encoder_attention_heads < 1 || parsed.decoder_attention_heads < 1 {
            return Err(anyhow!(
                "Florence-2 text_config attention heads must be positive, got {} enc / {} dec",
                parsed.encoder_attention_heads,
                parsed.decoder_attention_heads
            ));
        }
        if !(1..=MAX_LAYERS).contains(&parsed.encoder_layers)
            || !(1..=MAX_LAYERS).contains(&parsed.decoder_layers)
        {
            return Err(anyhow!(
                "Florence-2 text_config layer counts {} enc / {} dec outside 1..={MAX_LAYERS}",
                parsed.encoder_layers,
                parsed.decoder_layers
            ));
        }
        if parsed.encoder_ffn_dim < 1 || parsed.decoder_ffn_dim < 1 {
            return Err(anyhow!(
                "Florence-2 text_config ffn dims must be positive, got {} enc / {} dec",
                parsed.encoder_ffn_dim,
                parsed.decoder_ffn_dim
            ));
        }
        if parsed.vocab_size < 1 {
            return Err(anyhow!(
                "Florence-2 text_config vocab_size must be positive, got {}",
                parsed.vocab_size
            ));
        }
        if !(1..=MAX_POSITION_EMBEDDINGS).contains(&parsed.max_position_embeddings) {
            return Err(anyhow!(
                "Florence-2 text_config max_position_embeddings {} outside 1..={MAX_POSITION_EMBEDDINGS}",
                parsed.max_position_embeddings
            ));
        }
        // Both of these reach the shared embedding gather as literal token ids
        // (`generate_greedy` seeds the decoder with `decoder_start_token_id`,
        // `shift_tokens_right` substitutes `pad_token_id`), so an
        // out-of-vocabulary or negative value is an out-of-range gather.
        // `bos_token_id` and `eos_token_id` are deliberately left unchecked:
        // they are only ever compared, never used as an index, and some
        // exports legitimately carry sentinel values there.
        if !(0..parsed.vocab_size).contains(&parsed.pad_token_id) {
            return Err(anyhow!(
                "Florence-2 text_config pad_token_id {} outside 0..vocab_size {}",
                parsed.pad_token_id,
                parsed.vocab_size
            ));
        }
        if !(0..parsed.vocab_size).contains(&parsed.decoder_start_token_id) {
            return Err(anyhow!(
                "Florence-2 text_config decoder_start_token_id {} outside 0..vocab_size {}",
                parsed.decoder_start_token_id,
                parsed.vocab_size
            ));
        }
        if parsed.d_model % parsed.encoder_attention_heads != 0
            || parsed.d_model % parsed.decoder_attention_heads != 0
        {
            return Err(anyhow!(
                "Florence-2 d_model {} not divisible by attention heads ({} enc / {} dec)",
                parsed.d_model,
                parsed.encoder_attention_heads,
                parsed.decoder_attention_heads
            ));
        }
        Ok(parsed)
    }

    /// Parse from a full Florence-2 `config.json` (`model_type: florence2`),
    /// descending into its `text_config` sub-object. A bare text config (no
    /// `text_config` key) is also accepted so the sub-object can be passed
    /// directly.
    pub fn from_model_config(config: &Value) -> Result<Self> {
        let mut parsed = match config.get("text_config") {
            Some(text) => Self::from_text_config(text)?,
            None => Self::from_text_config(config)?,
        };
        // `quantization` sits at the top level of a Florence-2 `config.json`,
        // not inside `text_config`, so it has to be read from the document we
        // were handed rather than from the sub-object above. When a bare text
        // config was passed directly this re-reads the same object and is a
        // no-op.
        parsed.quantization = Florence2Quantization::from_model_config(config)?;
        Ok(parsed)
    }
}

#[cfg(test)]
#[path = "florence2_quantized_tests.rs"]
mod florence2_quantized_tests;
