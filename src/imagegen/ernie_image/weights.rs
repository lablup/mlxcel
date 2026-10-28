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

//! ERNIE-Image transformer and VAE loading.
//!
//! A port of `load_transformer`, `sanitize_transformer_weights`,
//! `match_conv_layout`, `_tensor_layout` and `load_vae` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie_image/weights.py>.
//! The raw Diffusers tree loads directly. Quantized conversions (upstream
//! `convert.py`) are out of scope and rejected by name.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::{Map, Value};

use super::config::ErnieTransformerConfig;
use super::transformer::ErnieTransformer;
use crate::imagegen::vae::{Flux2Vae, Flux2VaeConfig};
use crate::imagegen::z_image::weights::index_metadata;

/// Storage and compute dtype of the transformer.
///
/// Upstream casts every floating transformer weight to bf16 and draws bf16
/// noise, so the whole latent stream is bf16 apart from its explicit f32
/// islands (modulation, gates, the scheduler step). The port keeps that on
/// every backend; see `docs/image-generation.md`, "ERNIE-Image precision".
pub const TRANSFORMER_DTYPE: i32 = dtype::BFLOAT16;

/// Batch-norm epsilon upstream's `load_vae` sets on the FLUX.2 VAE
/// (`vae.bn.eps = 1e-5`), overriding the checkpoint's `batch_norm_eps: 1e-4`.
pub const VAE_BN_EPS: f32 = 1e-5;

/// Key of the patch-embed conv kernel after sanitizing.
const PATCH_EMBED_KEY: &str = "x_embedder.proj.weight";

/// How the conv kernels of a component are laid out on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvLayout {
    /// `[O, C, kh, kw]` (Diffusers / PyTorch).
    PytorchNchw,
    /// `[O, kh, kw, C]` (MLX / mflux conversions).
    MlxNhwc,
}

/// Upstream `_tensor_layout`: an explicit `tensor_layout`, else MLX when the
/// metadata names mflux or the MLX format, else PyTorch. Decided once per
/// component from metadata, so a kernel is never transposed twice.
#[must_use]
pub fn tensor_layout(metadata: &Map<String, Value>) -> ConvLayout {
    match metadata.get("tensor_layout").and_then(Value::as_str) {
        Some("mlx_nhwc") => return ConvLayout::MlxNhwc,
        Some("pytorch_nchw") => return ConvLayout::PytorchNchw,
        _ => {}
    }
    if metadata.contains_key("mflux_version")
        || metadata.get("format").and_then(Value::as_str) == Some("mlx")
    {
        ConvLayout::MlxNhwc
    } else {
        ConvLayout::PytorchNchw
    }
}

/// Upstream `match_conv_layout` for a known layout: transpose a PyTorch kernel
/// to `[O, kh, kw, C]`, and require the result (or an MLX kernel as stored) to
/// have `target` shape. The shape check is the double-permute guard: an MLX
/// kernel mislabelled as PyTorch, or one already transposed, fails by name
/// instead of being permuted again.
pub fn match_conv_layout(
    value: &MlxArray,
    target: &[i32],
    key: &str,
    layout: ConvLayout,
) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(value);
    let out = match layout {
        ConvLayout::PytorchNchw => {
            if shape.len() != 4 {
                bail!("convolution weight {key} must be 4-D, got {shape:?}");
            }
            mlxcel_core::transpose_axes(value, &[0, 2, 3, 1])
        }
        ConvLayout::MlxNhwc => mlxcel_core::copy(value),
    };
    let got = mlxcel_core::array_shape(&out);
    if got != target {
        bail!(
            "{} convolution weight shape mismatch for {key}: checkpoint={shape:?}, expected={target:?}",
            match layout {
                ConvLayout::PytorchNchw => "PyTorch",
                ConvLayout::MlxNhwc => "MLX",
            }
        );
    }
    Ok(out)
}

/// Upstream `sanitize_transformer_weights`: Diffusers key renames, the
/// patch-embed kernel in MLX layout, every floating tensor cast to
/// [`TRANSFORMER_DTYPE`].
pub fn sanitize_transformer_weights(
    raw: WeightMap,
    config: &ErnieTransformerConfig,
    layout: ConvLayout,
) -> Result<WeightMap> {
    if let Some(key) = raw.keys().find(|k| k.ends_with(".scales")) {
        bail!(
            "quantized ERNIE-Image transformers are not supported (found {key}); \
             load the bf16 Diffusers checkpoint"
        );
    }
    let p = config.patch_size;
    let target = [config.hidden_size, p, p, config.in_channels];
    let mut weights = WeightMap::new();
    for (raw_key, value) in raw {
        let key = raw_key
            .replace("adaLN_modulation.1.", "adaln_modulation.")
            .replace(".to_out.0.", ".to_out.");
        let mut tensor = if is_floating(&value) {
            mlxcel_core::astype(&value, TRANSFORMER_DTYPE)
        } else {
            value
        };
        if key == PATCH_EMBED_KEY {
            tensor = match_conv_layout(&tensor, &target, &key, layout)?;
        }
        weights.insert(key, tensor);
    }
    Ok(weights)
}

fn is_floating(x: &MlxArray) -> bool {
    matches!(
        mlxcel_core::array_dtype(x),
        dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16
    )
}

/// Load `<root>/transformer/`.
pub fn load_transformer(root: &Path) -> Result<ErnieTransformer> {
    let dir = root.join("transformer");
    let config = ErnieTransformerConfig::from_dir(&dir)?;
    let raw = mlxcel_core::weights::load_weights_from_dir(&dir)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load ERNIE-Image weights from {}", dir.display()))?;
    let layout = tensor_layout(&index_metadata(&dir)?);
    let weights = sanitize_transformer_weights(raw, &config, layout)?;
    ErnieTransformer::from_weights(config, &weights)
        .context("cannot build the ERNIE-Image transformer")
}

/// Load `<root>/vae/` (with the encoder when `include_encoder`) in
/// [`Flux2Vae::default_dtype`], with upstream's batch-norm epsilon.
pub fn load_vae(root: &Path, include_encoder: bool) -> Result<Flux2Vae> {
    let dir = root.join("vae");
    let mut config = Flux2VaeConfig::from_dir(&dir)?;
    config.batch_norm_eps = VAE_BN_EPS;
    let raw = mlxcel_core::weights::load_weights_from_dir(&dir)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load ERNIE-Image VAE weights from {}", dir.display()))?;
    Flux2Vae::from_raw_weights(config, raw, include_encoder, Flux2Vae::default_dtype())
        .context("cannot build the ERNIE-Image VAE")
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
