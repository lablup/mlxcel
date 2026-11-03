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

//! Compressed-tensors `float-quantized` FP8 checkpoints: detection and
//! load-time requantization to MLX's native `mxfp8`.
//!
//! `llm-compressor`'s `FP8-dynamic` releases (`RedHatAI/Qwen3.5-4B-FP8-dynamic`
//! among them) declare `quant_method: "compressed-tensors"` with
//! `format: "float-quantized"`. Every converted `Linear` is stored as raw E4M3
//! bytes in `<name>.weight` plus a `<name>.weight_scale` that carries either
//! one multiplier per output channel (`strategy: "channel"`) or one per
//! `block_structure: [rows, cols]` tile. The true weight is
//! `e4m3_decode(byte) * weight_scale[channel or tile]`.
//!
//! MLX has no float8 dtype, so the bytes reach this loader as `uint8` and the
//! scale is never applied; a decision to leave the checkpoint alone means the
//! model runs on the raw codes, off by its own per-channel factor. The
//! projections listed in `quantization_config.ignore` (the vision tower, every
//! `linear_attn` projection, `lm_head`, `^mtp.*`) are bf16 in the shards and
//! have no scale at all.
//!
//! [`requantize_compressed_tensors_fp8_weights`] reconstructs each pair on
//! device and requantizes it to `mxfp8`, the same target the vendor block path
//! uses ([`crate::models::fp8_block`]), so the model constructor sees one
//! native layout. Dequantizing to bf16 (what mlx-vlm does) was rejected: it
//! doubles weight memory against the checkpoint, and `mxfp8` is what the
//! vendor path already ships with a measured error bound. Conversion is one
//! tensor at a time with an `eval` per tensor, so peak memory stays near the
//! checkpoint's own footprint.
//!
//! This module deliberately does not touch `sanitize.rs`'s ModelOpt route,
//! which keys on `.weight_scale` **plus** `.weight_scale_2`. A compressed-
//! tensors FP8 tensor has no `.weight_scale_2`, and a ModelOpt NVFP4 triplet
//! has one, so the two routes are disjoint by construction.
//!
//! Used by: [`crate::models::fp8_checkpoint::fp8_checkpoint_quantization`],
//! which the Qwen3.5 text, Qwen3.5 VLM and special loaders call.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::{Map, Value, json};

use super::fp8_block::{
    MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE, decode_fp8_bytes_to_f32, e4m3_lookup_table,
    reconstruct_block_fp8_tensor,
};

/// Suffix identifying a compressed-tensors per-channel or per-block scale.
///
/// ModelOpt NVFP4 tensors also carry `.weight_scale`, but always beside a
/// `.weight_scale_2`; [`has_compressed_tensors_fp8_weights`] and the
/// per-tensor routing both require that key to be absent.
pub const CT_FP8_SCALE_SUFFIX: &str = ".weight_scale";

/// Suffix of the dynamic activation scale compressed-tensors may store next to
/// a `float-quantized` weight.
///
/// The activation scale applies to the *input* of a dynamic-scheme projection;
/// this loader quantizes only the weight and runs the layer in the checkpoint's
/// activation dtype, so the sidecar is removed rather than applied.
pub const CT_FP8_INPUT_SCALE_SUFFIX: &str = ".input_scale";

/// The scale layout a compressed-tensors `float-quantized` weight uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8CtScaleLayout {
    /// One multiplier per output channel (`strategy: "channel"`), stored as
    /// `[out]` or `[out, 1]`.
    Channel,
    /// One multiplier per `rows x cols` tile, stored as
    /// `[ceil(out / rows), ceil(in / cols)]`.
    Block { rows: usize, cols: usize },
}

/// A detected compressed-tensors `float-quantized` FP8 layout and the MLX
/// quantization it maps onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8CtQuantization {
    /// Where each `weight_scale` multiplier applies.
    pub layout: Fp8CtScaleLayout,
    /// Group size of the MLX quantization the weights are converted to.
    pub group_size: i32,
    /// Bit width of the MLX quantization the weights are converted to.
    pub bits: i32,
    /// MLX quantization mode name.
    pub mode: &'static str,
}

impl Fp8CtQuantization {
    /// The `quantization` block to merge into a model config so the model
    /// constructor builds quantized layers for the converted tensors.
    #[must_use]
    pub fn effective_config(&self) -> Value {
        json!({
            "group_size": self.group_size,
            "bits": self.bits,
            "mode": self.mode,
        })
    }
}

/// Read the `quantization_config` block and report whether this checkpoint is
/// a compressed-tensors `float-quantized` FP8 release this loader can
/// reconstruct.
///
/// Accepts the block at the top level or inside `text_config`, since the VLM
/// wrappers place it in either. Returns:
///
/// * `Ok(None)` when the checkpoint is not a compressed-tensors
///   `float-quantized` export at all. A `mixed-precision` export keeps the real
///   formats in per-group `config_groups` and is out of scope, as is a packed
///   (NVFP4/INT4) export, which the ModelOpt route in `sanitize.rs` owns.
/// * `Ok(Some(..))` for a weights group with `type: "float"`, `num_bits: 8`,
///   static weights and either `strategy: "channel"` or a `block_structure`.
/// * `Err` naming the field for a float-quantized declaration whose layout is
///   one this loader cannot pair a scale with (`strategy: "tensor"`,
///   `"group"`, another bit width, dynamic weights). Failing the load is the
///   point: applying a channel scale to a tensor scale, or the reverse,
///   produces fluent but wrong output rather than an error.
pub fn compressed_tensors_fp8_quantization(
    full_config: &Value,
) -> Result<Option<Fp8CtQuantization>, String> {
    let Some(quant_config) = full_config.get("quantization_config").or_else(|| {
        full_config
            .get("text_config")
            .and_then(|text| text.get("quantization_config"))
    }) else {
        return Ok(None);
    };

    let Some(quant_config) = quant_config.as_object() else {
        return Ok(None);
    };

    let method = quant_config
        .get("quant_method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !method.eq_ignore_ascii_case("compressed-tensors") {
        return Ok(None);
    }

    // The top-level `format` names the export as a whole. Anything but
    // `float-quantized` belongs to another route, and a `mixed-precision`
    // export puts the real formats in its groups.
    match quant_config.get("format").and_then(Value::as_str) {
        None => {
            // No top-level format: the per-group formats decide, and the
            // groups are checked below.
        }
        Some(format) if format.eq_ignore_ascii_case("float-quantized") => {}
        Some(_) => return Ok(None),
    }

    let groups = quant_config
        .get("config_groups")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            "quantization_config declares quant_method \"compressed-tensors\" but carries no \
             config_groups object, so the weight layout of its FP8 tensors cannot be determined"
                .to_string()
        })?;

    let top_level_format = quant_config.get("format").and_then(Value::as_str).is_some();
    let mut layouts = Vec::new();
    for (group_name, group) in groups {
        let Some(weights) = group.get("weights").filter(|value| !value.is_null()) else {
            continue;
        };
        let Some(weights) = weights.as_object() else {
            return Err(format!(
                "quantization_config.config_groups.{group_name}.weights must be an object, got \
                 {weights}"
            ));
        };
        if !top_level_format {
            let group_format = group.get("format").and_then(Value::as_str);
            if !group_format.is_some_and(|f| f.eq_ignore_ascii_case("float-quantized")) {
                // Without a top-level format this is a per-group export
                // (mixed precision); the FP8 group alone is not the layout of
                // the whole checkpoint, so another route owns it.
                return Ok(None);
            }
        }
        layouts.push(ct_weights_layout(group_name, weights)?);
    }

    let Some(first) = layouts.first().copied() else {
        return Err(
            "quantization_config declares quant_method \"compressed-tensors\" but no \
             config_groups entry carries a weights layout, so the scale pairing of its FP8 \
             tensors cannot be determined"
                .to_string(),
        );
    };
    if let Some(other) = layouts.iter().find(|layout| **layout != first) {
        return Err(format!(
            "quantization_config.config_groups disagrees on the FP8 weight layout ({first:?} and \
             {other:?}): mlxcel applies one scale pairing per checkpoint, and a per-group mix of \
             channel and block scales would mis-scale every tensor from the other group"
        ));
    }

    Ok(Some(Fp8CtQuantization {
        layout: first,
        group_size: MXFP8_GROUP_SIZE,
        bits: MXFP8_BITS,
        mode: MXFP8_MODE,
    }))
}

/// Validate one `config_groups.*.weights` block and return its scale layout.
fn ct_weights_layout(
    group_name: &str,
    weights: &Map<String, Value>,
) -> Result<Fp8CtScaleLayout, String> {
    let field =
        |name: &str| format!("quantization_config.config_groups.{group_name}.weights.{name}");

    match weights.get("type").and_then(Value::as_str) {
        Some(kind) if kind.eq_ignore_ascii_case("float") => {}
        other => {
            return Err(format!(
                "{} is {other:?}: mlxcel reconstructs only float-quantized E4M3 weights, and \
                 decoding another element type as E4M3 produces a wrong exponent for every \
                 weight without failing",
                field("type")
            ));
        }
    }

    match weights.get("num_bits").and_then(Value::as_u64) {
        Some(8) => {}
        other => {
            return Err(format!(
                "{} is {other:?}: mlxcel decodes only 8-bit E4M3 float weights, because the \
                 byte-to-value table is what pairs each stored code with its value",
                field("num_bits")
            ));
        }
    }

    if weights
        .get("dynamic")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(format!(
            "{} is true: a dynamically quantized weight computes its scale at run time, so the \
             checkpoint carries none to reconstruct against",
            field("dynamic")
        ));
    }

    if let Some(block) = weights
        .get("block_structure")
        .filter(|value| !value.is_null())
    {
        let dims = block.as_array().ok_or_else(|| {
            format!(
                "{} must be an array of two positive integers, got {block}",
                field("block_structure")
            )
        })?;
        let extents = dims
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        format!(
                            "{} entry {dim} is not a positive integer",
                            field("block_structure")
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let [rows, cols] = extents.as_slice() else {
            return Err(format!(
                "{} is {block}, but a block scale pairs with a two-dimensional [rows, cols] tile",
                field("block_structure")
            ));
        };
        if *rows == 0 || *cols == 0 {
            return Err(format!(
                "{} is {block}, but a zero extent has no block to pair a scale with",
                field("block_structure")
            ));
        }
        return Ok(Fp8CtScaleLayout::Block {
            rows: *rows,
            cols: *cols,
        });
    }

    let strategy = weights
        .get("strategy")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if strategy.eq_ignore_ascii_case("channel") {
        return Ok(Fp8CtScaleLayout::Channel);
    }
    Err(format!(
        "{} is {strategy:?}: mlxcel reconstructs only per-output-channel (\"channel\") and \
         block_structure float-quantized FP8 weights. A tensor or group scale has a different \
         sidecar shape, and applying the channel pairing to it mis-scales every tensor without \
         failing",
        field("strategy")
    ))
}

/// Whether `weights` carries at least one compressed-tensors FP8 scale.
///
/// A ModelOpt NVFP4 triplet also stores `.weight_scale`, so a key only counts
/// when it has no `.weight_scale_2` beside it.
#[must_use]
pub fn has_compressed_tensors_fp8_weights(weights: &WeightMap) -> bool {
    weights
        .keys()
        .any(|key| key.ends_with(CT_FP8_SCALE_SUFFIX) && !weights.contains_key(&format!("{key}_2")))
}

/// Reconstruct one per-channel-scaled FP8 tensor into a plain f32 array.
///
/// `raw` holds the E4M3 bytes as `uint8`, `scale` the per-output-channel
/// multipliers. The result is `[rows, cols]` f32, matching the original weight
/// shape. Exposed so tests can compare this route against an independently
/// expressed reference.
pub fn reconstruct_channel_fp8_tensor(
    name: &str,
    raw: &MlxArray,
    scale: &MlxArray,
    lut: &MlxArray,
) -> Result<UniquePtr<MlxArray>, String> {
    let (decoded, rows, _cols) = decode_fp8_bytes_to_f32(name, raw, lut)?;

    let scale = mlxcel_core::astype(scale, dtype::FLOAT32);
    let scale = match mlxcel_core::array_shape(&scale).as_slice() {
        [channels] if *channels == rows => mlxcel_core::reshape(&scale, &[rows, 1]),
        // llm-compressor writes the channel scale as a column vector.
        [channels, 1] if *channels == rows => scale,
        other => {
            return Err(format!(
                "compressed-tensors channel scale for {name} has shape {other:?}, expected \
                 [{rows}] or [{rows}, 1]: the scale carries one multiplier per output channel, \
                 and pairing it with a different axis mis-scales every row without failing"
            ));
        }
    };

    Ok(mlxcel_core::multiply(&decoded, &scale))
}

/// Reconstruct every compressed-tensors FP8 tensor in `weights` and replace it
/// with an MLX-native `mxfp8` pair.
///
/// A tensor converts only when `<prefix>.weight` is `uint8`, `<prefix>.weight_scale`
/// exists and `<prefix>.weight_scale_2` does not, which is exactly the layout
/// the config's `float-quantized` group describes. Everything in `ignore` (the
/// vision tower, every `linear_attn` projection, `lm_head`, the MTP tensors)
/// arrives as bf16 with no scale and is left untouched; `UnifiedLinear` then
/// picks the quantized or the dense path per tensor from the presence of
/// `.scales`.
///
/// Conversion is one tensor at a time and each quantized pair is evaluated
/// before the next tensor starts, so the f32 reconstruction of the largest
/// projection is the only transient and is released before the next one is
/// built.
pub fn requantize_compressed_tensors_fp8_weights(
    mut weights: WeightMap,
    layout: Fp8CtScaleLayout,
) -> Result<WeightMap, String> {
    let mut scale_keys: Vec<String> = weights
        .keys()
        .filter(|key| key.ends_with(CT_FP8_SCALE_SUFFIX))
        // ModelOpt NVFP4 stores `.weight_scale` beside `.weight_scale_2`; that
        // pair belongs to `sanitize.rs`'s repack route.
        .filter(|key| !weights.contains_key(&format!("{key}_2")))
        .cloned()
        .collect();
    if scale_keys.is_empty() {
        return Ok(weights);
    }
    // Deterministic order so a failure reports the same tensor on every run.
    scale_keys.sort();

    let started = std::time::Instant::now();
    let total = scale_keys.len();
    tracing::info!(
        tensors = total,
        ?layout,
        "reconstructing compressed-tensors FP8 weights and requantizing to mxfp8"
    );

    let lut = e4m3_lookup_table();
    let mut converted = 0usize;

    for scale_key in &scale_keys {
        let prefix = scale_key
            .strip_suffix(CT_FP8_SCALE_SUFFIX)
            .ok_or_else(|| format!("Internal error: {scale_key} lost its FP8 scale suffix"))?
            .to_string();
        let weight_key = format!("{prefix}.weight");

        // Route per tensor rather than by name: the config's `ignore` list is
        // expressed in HF module paths, and a tensor is only a compressed-
        // tensors FP8 weight if it really is an E4M3 byte matrix.
        let is_e4m3_bytes = weights
            .get(&weight_key)
            .is_some_and(|weight| mlxcel_core::array_dtype(weight) == dtype::UINT8);
        if !is_e4m3_bytes {
            continue;
        }

        let scale = weights.remove(scale_key).ok_or_else(|| {
            format!("Internal error: compressed-tensors scale {scale_key} disappeared")
        })?;
        let raw = weights.remove(&weight_key).ok_or_else(|| {
            format!(
                "compressed-tensors scale {scale_key} has no matching {weight_key} in the \
                 checkpoint; the scale cannot be applied to anything"
            )
        })?;

        let restored = match layout {
            Fp8CtScaleLayout::Channel => {
                reconstruct_channel_fp8_tensor(&weight_key, &raw, &scale, &lut)?
            }
            Fp8CtScaleLayout::Block { rows, cols } => {
                reconstruct_block_fp8_tensor(&weight_key, &raw, &scale, rows, cols, &lut)?
            }
        };
        // Drop the E4M3 bytes and the scale before quantizing so the peak
        // holds the reconstruction and the packed result, not all four.
        drop(raw);
        drop(scale);

        let quantized = mlxcel_core::quantize_weights_with_mode(
            &restored,
            MXFP8_GROUP_SIZE,
            MXFP8_BITS,
            MXFP8_MODE,
        );
        if mlxcel_core::quantized_weights_has_biases(&quantized) {
            return Err(format!(
                "MLX returned affine biases for a {MXFP8_MODE} quantization of {weight_key}; \
                 the block-float path must not carry a zero-point plane"
            ));
        }
        let packed = mlxcel_core::quantized_weights_w(&quantized);
        let scales = mlxcel_core::quantized_weights_scales(&quantized);

        // Force realization here rather than at first forward: this is what
        // keeps the reconstruction transient and the peak bounded.
        mlxcel_core::eval(&packed);
        mlxcel_core::eval(&scales);

        weights.insert(weight_key, packed);
        weights.insert(format!("{prefix}.scales"), scales);
        // The activation scale applies to the layer input under a dynamic
        // scheme; the forward pass keeps the activation dtype and never reads
        // it, so leaving it would surface as an unexpected key.
        weights.remove(&format!("{prefix}{CT_FP8_INPUT_SCALE_SUFFIX}"));
        converted += 1;
    }

    tracing::info!(
        tensors = total,
        converted,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "compressed-tensors FP8 requantization complete"
    );

    Ok(weights)
}

#[cfg(test)]
#[path = "fp8_ct_tests.rs"]
mod fp8_ct_tests;
