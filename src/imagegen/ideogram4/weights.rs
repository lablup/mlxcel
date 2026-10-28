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

//! Ideogram 4 component loading and per-row FP8 dequantization.
//!
//! A port of
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ideogram4/weights.py>.
//! The public checkpoint stores every linear weight as FP8 E4M3 bytes with
//! one f32 `weight_scale` per output row. [`dequantize_fp8_rowwise`] decodes
//! them to bf16 exactly as upstream's `dequantize_fp8_weight_only` does, and
//! every other floating tensor is cast to bf16. The model then runs in bf16,
//! upstream's precision (`PRECISION = mx.bfloat16`); the f16 load-time
//! conversion is not applied to these components, see
//! `docs/image-generation.md`, "Ideogram 4".
//!
//! This per-row scheme is distinct from the 128x128 block scales of
//! `src/models/fp8_block.rs` (`weight_scale_inv`).

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::Value;

use super::config::Ideogram4TransformerConfig;
use super::transformer::Ideogram4Transformer;
use crate::imagegen::Qwen3TextEncoder;
use crate::imagegen::vae::{Flux2Vae, Flux2VaeConfig};
use crate::models::f8_e4m3_to_f32;

/// Storage and compute dtype of every Ideogram 4 transformer and the text
/// encoder.
pub const PRECISION: i32 = dtype::BFLOAT16;

/// Suffix of a per-row FP8 scale sidecar (`<name>.weight_scale`).
pub const ROW_SCALE_SUFFIX: &str = ".weight_scale";

fn is_floating(x: &MlxArray) -> bool {
    matches!(
        mlxcel_core::array_dtype(x),
        dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16
    )
}

/// The 256 E4M3 values as bf16 (every E4M3 value is exact in bf16).
fn e4m3_lut() -> UniquePtr<MlxArray> {
    let table: Vec<f32> = (0..=u8::MAX).map(f8_e4m3_to_f32).collect();
    mlxcel_core::astype(&mlxcel_core::from_slice_f32(&table, &[256]), PRECISION)
}

/// Decode one FP8 weight: `from_fp8(bytes, bf16) * scale.astype(bf16)[:, None]`.
fn dequantize_one(
    key: &str,
    raw: &MlxArray,
    scale: &MlxArray,
    lut: &MlxArray,
) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(raw);
    let scale_shape = mlxcel_core::array_shape(scale);
    if shape.is_empty() || scale_shape != [shape[0]] {
        bail!(
            "per-row FP8 scale for {key} has shape {scale_shape:?}, expected [{}] for a {shape:?} weight",
            shape.first().copied().unwrap_or(0)
        );
    }
    if mlxcel_core::array_dtype(raw) != dtype::UINT8 {
        bail!(
            "FP8 weight {key} is MLX dtype {}, expected the raw E4M3 bytes (uint8)",
            mlxcel_core::array_dtype(raw)
        );
    }
    let decoded = mlxcel_core::take(lut, &mlxcel_core::astype(raw, dtype::UINT32), 0);
    let mut scale_dims = vec![1; shape.len()];
    scale_dims[0] = shape[0];
    let scale = mlxcel_core::reshape(&mlxcel_core::astype(scale, PRECISION), &scale_dims);
    Ok(mlxcel_core::multiply(&decoded, &scale))
}

/// Upstream `dequantize_fp8_weight_only` (to bf16).
///
/// A `<name>.weight` with a `<name>.weight_scale` is decoded from E4M3 and
/// multiplied by the scale broadcast over its output rows; the scale must be
/// `[out]`, else the error names the tensor. Every `weight_scale` is dropped
/// (paired or not), other floating tensors are cast to bf16, and the rest
/// (integer tensors) pass through. Each converted tensor is evaluated as it
/// is produced so one component's FP8 bytes and bf16 copy are never both
/// fully resident.
pub fn dequantize_fp8_rowwise(weights: WeightMap) -> Result<WeightMap> {
    let (scales, rest): (WeightMap, WeightMap) = weights
        .into_iter()
        .partition(|(k, _)| k.ends_with(ROW_SCALE_SUFFIX));
    let lut = e4m3_lut();
    let mut out = WeightMap::with_capacity(rest.len());
    for (key, value) in rest {
        let converted = match scales.get(&format!("{key}_scale")) {
            Some(scale) if key.ends_with(".weight") => dequantize_one(&key, &value, scale, &lut)?,
            _ if is_floating(&value) => mlxcel_core::astype(&value, PRECISION),
            _ => {
                out.insert(key, value);
                continue;
            }
        };
        drop(value);
        mlxcel_core::try_eval(&converted)
            .with_context(|| format!("cannot dequantize Ideogram 4 weight {key}"))?;
        out.insert(key, converted);
    }
    Ok(out)
}

/// The classifier-free-guidance branch, which takes no text.
const UNCONDITIONAL_SUBFOLDER: &str = "unconditional_transformer";

/// Whether a raw transformer key is read: the LLM projection (`llm_cond_*`)
/// is dropped when the branch takes no text.
fn keeps_transformer_key(needs_llm: bool, key: &str) -> bool {
    needs_llm || !key.starts_with("llm_cond_")
}

/// Load and dequantize `<root>/<subfolder>/` into an [`Ideogram4Transformer`].
pub fn load_transformer(root: &Path, subfolder: &str) -> Result<Ideogram4Transformer> {
    let dir = root.join(subfolder);
    let config = Ideogram4TransformerConfig::from_dir(&dir)?;
    // The unconditional transformer never reads text: skip its LLM projection
    // (about 0.5 GB) so it is neither read nor dequantized.
    let needs_llm = subfolder != UNCONDITIONAL_SUBFOLDER;
    let raw = mlxcel_core::weights::load_weights_from_dir_filtered(&dir, |k| {
        keeps_transformer_key(needs_llm, k)
    })
    .map_err(anyhow::Error::msg)
    .with_context(|| format!("cannot load Ideogram 4 weights from {}", dir.display()))?;
    let weights = dequantize_fp8_rowwise(raw)?;
    let transformer = Ideogram4Transformer::from_weights(config, &weights)
        .with_context(|| format!("cannot build the Ideogram 4 {subfolder}"))?;
    if needs_llm && !transformer.has_llm_conditioning() {
        bail!("the Ideogram 4 {subfolder} has no llm_cond_norm/llm_cond_proj weights");
    }
    Ok(transformer)
}

/// Whether a raw `text_encoder/` key belongs to the language model upstream
/// keeps (`language_model.` stripped, then `embed_tokens.`, `layers.` or
/// `norm.`). The vision tower is never read.
fn is_text_key(key: &str) -> bool {
    let rest = key.strip_prefix("language_model.").unwrap_or(key);
    ["embed_tokens.", "layers.", "norm."]
        .iter()
        .any(|p| rest.starts_with(p))
}

/// Upstream `load_text_encoder`: the Qwen3-VL text tower of `text_encoder/`
/// (nested `text_config`), FP8 dequantized to bf16 before it reaches the
/// shared conditioner.
pub fn load_text_encoder(root: &Path) -> Result<Qwen3TextEncoder> {
    let dir = root.join("text_encoder");
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let config: Value = serde_json::from_str(&crate::models::sanitize_config_json(&text))
        .with_context(|| format!("cannot parse {}", path.display()))?;
    if !config.get("text_config").is_some_and(Value::is_object) {
        bail!("{} has no text_config", path.display());
    }
    let raw = mlxcel_core::weights::load_weights_from_dir_filtered(&dir, is_text_key)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load text encoder weights from {}", dir.display()))?;
    let weights = dequantize_fp8_rowwise(raw)?;
    Qwen3TextEncoder::from_weights(&config, weights)
}

/// Upstream `load_vae` (FLUX.2's): decoder only, in [`Flux2Vae::default_dtype`].
pub fn load_vae(root: &Path) -> Result<Flux2Vae> {
    let dir = root.join("vae");
    let config = Flux2VaeConfig::from_dir(&dir)?;
    let raw = mlxcel_core::weights::load_weights_from_dir(&dir)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load Ideogram 4 VAE weights from {}", dir.display()))?;
    Flux2Vae::from_raw_weights(config, raw, false, Flux2Vae::default_dtype())
        .context("cannot build the Ideogram 4 VAE")
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
