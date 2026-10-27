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

//! Component weight loading for Z-Image.
//!
//! A port of `weights.py` and `sanitize_vae_weights` in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/z_image>.
//! Raw Diffusers trees load directly: the transformer keys are remapped by
//! [`crate::imagegen::dit_blocks::sanitize_transformer_weights`] and the VAE
//! conv layout is decided by [`crate::imagegen::Flux2Vae`]. An mlx-vlm
//! converted tree (`model.safetensors.index.json` metadata
//! `mlx_vlm_format: "z_image"`) loads too; its quantized transformer and VAE
//! layers are dequantized at load with the mode, bits and group size from
//! that metadata. The text encoder's quantization is read from its own
//! `config.json` by the shared Qwen3 conditioner.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use serde_json::{Map, Value};

/// Marker an mlx-vlm conversion writes into each component's index metadata.
pub const MLX_VLM_FORMAT: &str = "z_image";

/// Quantization recorded in a converted component's index metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantSpec {
    pub mode: String,
    pub bits: i32,
    pub group_size: i32,
}

/// `metadata` of `<dir>/model.safetensors.index.json`, or empty.
pub fn index_metadata(dir: &Path) -> Result<Map<String, Value>> {
    let path = dir.join("model.safetensors.index.json");
    if !path.exists() {
        return Ok(Map::new());
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let index: Value =
        serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))?;
    Ok(index
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default())
}

/// Whether `metadata` marks an mlx-vlm converted Z-Image component.
#[must_use]
pub fn is_mlx_vlm_format(metadata: &Map<String, Value>) -> bool {
    metadata.get("mlx_vlm_format").and_then(Value::as_str) == Some(MLX_VLM_FORMAT)
}

/// A metadata value written as a string (`"4"`) or a number.
fn meta_int(metadata: &Map<String, Value>, key: &str) -> i32 {
    match metadata.get(key) {
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0),
        Some(v) => v.as_i64().and_then(|n| i32::try_from(n).ok()).unwrap_or(0),
        None => 0,
    }
}

/// Validate the quantization metadata the way upstream's `_apply_weights` does:
/// `affine` at any bits and group size, `mxfp4` (4, 32), `mxfp8` (8, 32) and
/// `nvfp4` (4, 16); anything else is an error.
pub fn quant_spec(metadata: &Map<String, Value>) -> Result<QuantSpec> {
    let mode = metadata
        .get("quantization_mode")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let bits = meta_int(metadata, "quantization_level");
    let group_size = meta_int(metadata, "quantization_group_size");
    let expected = match mode.as_str() {
        "affine" => None,
        "mxfp4" => Some((4, 32)),
        "mxfp8" => Some((8, 32)),
        "nvfp4" => Some((4, 16)),
        _ => {
            bail!(
                "Unsupported Z-Image quantization metadata: mode={mode:?}, bits={bits}, group_size={group_size}"
            )
        }
    };
    if expected.is_some_and(|e| e != (bits, group_size)) || bits < 1 || group_size < 1 {
        bail!(
            "Unsupported Z-Image quantization metadata: mode={mode:?}, bits={bits}, group_size={group_size}"
        );
    }
    Ok(QuantSpec {
        mode,
        bits,
        group_size,
    })
}

/// Replace every `<p>.weight` / `<p>.scales` / `<p>.biases` triple by a dense
/// `<p>.weight`. A map without `.scales` is returned unchanged.
pub fn dequantize_weights(
    mut weights: WeightMap,
    metadata: &Map<String, Value>,
) -> Result<WeightMap> {
    let prefixes: Vec<String> = weights
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(str::to_string))
        .collect();
    if prefixes.is_empty() {
        return Ok(weights);
    }
    let spec = quant_spec(metadata)?;
    for prefix in prefixes {
        let weight_key = format!("{prefix}.weight");
        let scales = weights
            .remove(&format!("{prefix}.scales"))
            .context("scales disappeared")?;
        let biases = weights.remove(&format!("{prefix}.biases"));
        let packed = weights
            .remove(&weight_key)
            .with_context(|| format!("Quantized weights have no matching module: {prefix}"))?;
        let bias_ptr = biases
            .as_ref()
            .map_or(std::ptr::null(), |b| &**b as *const mlxcel_core::MlxArray);
        // SAFETY: `bias_ptr` is null or borrowed from `biases`, alive for the call.
        let dense = unsafe {
            mlxcel_core::dequantize(
                &packed,
                &scales,
                bias_ptr,
                spec.group_size,
                spec.bits,
                &spec.mode,
            )
        };
        weights.insert(weight_key, dense);
    }
    Ok(weights)
}

/// All safetensors of a component directory plus its index metadata.
pub fn load_component(dir: &Path) -> Result<(WeightMap, Map<String, Value>)> {
    let weights = mlxcel_core::weights::load_weights_from_dir(dir)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load Z-Image weights from {}", dir.display()))?;
    Ok((weights, index_metadata(dir)?))
}

/// Storage dtype of the transformer and VAE weights.
///
/// Upstream keeps these weights bf16 and feeds them f32 activations (f32 noise
/// and timestep), so MLX promotes every matmul and conv to f32. The CUDA build
/// of mlxcel's MLX promotes bf16 with f32 to bf16 instead (its patched
/// `dtype.cpp` promotion table), which would run the whole latent stream in
/// bf16 (measured: 2.1e-2 relative RMS on one Turbo transformer forward,
/// 1.1e-1 on the final latents). f16 promotes with f32 to f32 on every
/// backend, so the arithmetic matches upstream at the memory of bf16.
///
/// The bf16 to f16 cast is not exact in general: bf16 reaches about 3.4e38
/// where f16 stops at 65504, and bf16 values below f16's normal range
/// (6.1e-5) lose low mantissa bits or flush to zero. On Z-Image-Turbo and
/// Z-Image the largest magnitude is 47.75 and the worst per-tensor round-trip
/// error is 2.0e-7 relative RMS (about two thirds of the tensors change in
/// some subnormal element). The loader measures every tensor
/// ([`f16_round_trip_error`]) and keeps any tensor above
/// [`F16_MAX_REL_RMS`] in f32 instead, which is exact and promotes the same
/// way, so another checkpoint cannot be silently corrupted by the cast.
pub const WEIGHT_DTYPE: i32 = mlxcel_core::dtype::FLOAT16;

/// Largest relative RMS error an f16 copy of a weight may introduce; a tensor
/// above it (or one the cast overflows) is stored as f32. 50x the worst
/// tensor of the published checkpoints.
pub const F16_MAX_REL_RMS: f32 = 1e-5;

/// `||f32(f16(x)) - x|| / ||x||` for a floating tensor `x`: 0 when the cast is
/// exact, infinite when it overflows f16, NaN when `x` holds a non-finite
/// value.
#[must_use]
pub fn f16_round_trip_error(x: &mlxcel_core::MlxArray) -> f32 {
    use mlxcel_core::dtype::{FLOAT16, FLOAT32};
    let x32 = mlxcel_core::astype(x, FLOAT32);
    let back = mlxcel_core::astype(&mlxcel_core::astype(x, FLOAT16), FLOAT32);
    let diff = mlxcel_core::subtract(&back, &x32);
    let num = mlxcel_core::item_f32(&mlxcel_core::sum_all(&mlxcel_core::square(&diff)));
    let den = mlxcel_core::item_f32(&mlxcel_core::sum_all(&mlxcel_core::square(&x32)));
    if den > 0.0 {
        (num / den).sqrt()
    } else if num == 0.0 {
        0.0
    } else {
        f32::INFINITY
    }
}

/// Whether `x` may be stored as f16 ([`F16_MAX_REL_RMS`]).
#[must_use]
pub fn fits_f16(x: &mlxcel_core::MlxArray) -> bool {
    // Written so a NaN error (non-finite source) is rejected too.
    f16_round_trip_error(x) <= F16_MAX_REL_RMS
}

/// Cast every bf16 or f32 tensor of `weights` to [`WEIGHT_DTYPE`], except the
/// keys `keep` selects, which stay in the checkpoint dtype. A tensor that does
/// not [`fits_f16`] is stored as f32 instead; its key is returned so the
/// caller can report it.
#[must_use]
pub fn cast_floats(weights: &mut WeightMap, keep: impl Fn(&str) -> bool) -> Vec<String> {
    let mut widened = Vec::new();
    for (key, value) in weights.iter_mut() {
        if keep(key) {
            continue;
        }
        let dt = mlxcel_core::array_dtype(value);
        if !matches!(
            dt,
            mlxcel_core::dtype::BFLOAT16 | mlxcel_core::dtype::FLOAT32
        ) {
            continue;
        }
        let target = if fits_f16(value) {
            WEIGHT_DTYPE
        } else {
            widened.push(key.clone());
            mlxcel_core::dtype::FLOAT32
        };
        if target != dt {
            *value = mlxcel_core::astype(value, target);
            // Materialize now so the bf16 source is released at load, not
            // held alongside the copy until the first forward.
            mlxcel_core::eval(value);
        }
    }
    widened.sort();
    widened
}

/// The VAE's compute dtype: [`WEIGHT_DTYPE`] when every floating tensor of
/// `weights` [`fits_f16`], else f32 (the VAE casts all its weights to one
/// dtype). Returns the offending keys alongside.
#[must_use]
pub fn vae_weight_dtype(weights: &WeightMap) -> (i32, Vec<String>) {
    let mut offending: Vec<String> = weights
        .iter()
        .filter(|(_, v)| {
            matches!(
                mlxcel_core::array_dtype(v),
                mlxcel_core::dtype::BFLOAT16
                    | mlxcel_core::dtype::FLOAT32
                    | mlxcel_core::dtype::FLOAT16
            ) && !fits_f16(v)
        })
        .map(|(k, _)| k.clone())
        .collect();
    offending.sort();
    let dt = if offending.is_empty() {
        WEIGHT_DTYPE
    } else {
        mlxcel_core::dtype::FLOAT32
    };
    (dt, offending)
}

/// Transformer weights that read the half-precision caption stream before it
/// first meets f32: the caption embedder (RMSNorm and Linear) and the
/// pre-attention path of the first context-refiner block. Upstream computes these in the checkpoint's
/// bf16 (bf16 activations times bf16 weights), so they keep that dtype here;
/// the f32 weights would compute them in f32 instead and miss upstream's bf16
/// rounding, which the joint layers amplify about 25x.
#[must_use]
pub fn reads_caption_stream(key: &str) -> bool {
    const BLOCK0: [&str; 6] = [
        "context_refiner.0.attention_norm1.weight",
        "context_refiner.0.attention.to_q.weight",
        "context_refiner.0.attention.to_k.weight",
        "context_refiner.0.attention.to_v.weight",
        "context_refiner.0.attention.norm_q.weight",
        "context_refiner.0.attention.norm_k.weight",
    ];
    key.starts_with("cap_embedder.") || BLOCK0.contains(&key)
}

/// Upstream's `sanitize_vae_weights` key renames (mflux-style module paths to
/// the Diffusers names [`crate::imagegen::Flux2Vae`] reads). The conv layout
/// is decided by the VAE loader from tensor shapes, so a converted
/// (`mlx_vlm_format`) directory, already MLX layout, is left untouched.
#[must_use]
pub fn sanitize_vae_weights(weights: WeightMap) -> WeightMap {
    const RENAMES: [(&str, &str); 6] = [
        ("encoder.conv_in.conv2d.", "encoder.conv_in."),
        ("encoder.conv_out.conv2d.", "encoder.conv_out."),
        ("encoder.conv_norm_out.norm.", "encoder.conv_norm_out."),
        ("decoder.conv_in.conv.", "decoder.conv_in."),
        ("decoder.conv_out.conv.", "decoder.conv_out."),
        ("decoder.conv_norm_out.norm.", "decoder.conv_norm_out."),
    ];
    weights
        .into_iter()
        .map(|(key, value)| {
            let key = RENAMES
                .iter()
                .fold(key, |k, (from, to)| k.replace(from, to));
            (key, value)
        })
        .collect()
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
