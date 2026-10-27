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

//! FLUX.2 component loaders.
//!
//! Counterparts of `load_text_encoder` and `load_vae` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/weights.py>.
//! MLX-quantized exports (`mlx-community/flux2-klein-*-4bit`) ship no component
//! `config.json` and quantize the VAE's mid-block attention projections; the
//! text-encoder config is synthesized from the variant and those few VAE
//! tensors are dequantized here, so the shared VAE stays dense.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::{Flux2Variant, read_quant_params, text_encoder_config};
use crate::imagegen::{Flux2Vae, Flux2VaeConfig, Qwen3TextEncoder};

/// Load `text_encoder/`: through [`Qwen3TextEncoder::load`] when it ships a
/// `config.json`, else from a config synthesized for `variant`.
pub fn load_text_encoder(dir: &Path, variant: Flux2Variant) -> Result<Qwen3TextEncoder> {
    if dir.join("config.json").exists() {
        return Qwen3TextEncoder::load(dir);
    }
    let quant = read_quant_params(dir)?;
    let config = text_encoder_config(dir, variant, quant)?;
    let weights = mlxcel_core::weights::load_weights_from_dir(dir)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load text encoder weights from {}", dir.display()))?;
    Qwen3TextEncoder::from_weights(&config, weights)
}

/// `bits` and `group_size` of one affine-quantized linear, inferred from its
/// shapes as upstream `_quantization_config` does: `packed_in * 32 / in` and
/// `in / groups`, with `in` the dense input width `out_dim`.
fn infer_layout(packed: &MlxArray, scales: &MlxArray, in_dim: i32) -> Result<(i32, i32)> {
    let p = mlxcel_core::array_shape(packed);
    let s = mlxcel_core::array_shape(scales);
    let (packed_in, groups) = (p[p.len() - 1], s[s.len() - 1]);
    if in_dim <= 0 || groups <= 0 || (packed_in * 32) % in_dim != 0 || in_dim % groups != 0 {
        bail!("cannot infer quantization of a {p:?} weight with {s:?} scales");
    }
    Ok(((packed_in * 32) / in_dim, in_dim / groups))
}

/// Replace every `<prefix>.weight` that has `.scales`/`.biases` with its dense
/// dequantization. The VAE's quantized layers are square attention
/// projections (`in == out`), which fixes their dense input width.
pub(super) fn dequantize_square_linears(weights: &mut WeightMap) -> Result<usize> {
    let prefixes: Vec<String> = weights
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(str::to_string))
        .collect();
    for prefix in &prefixes {
        let take = |w: &mut WeightMap, suffix: &str| {
            w.remove(&format!("{prefix}.{suffix}"))
                .with_context(|| format!("Incomplete quantized weights for module: {prefix}"))
        };
        let packed = take(weights, "weight")?;
        let scales = take(weights, "scales")?;
        let biases = take(weights, "biases")?;
        if mlxcel_core::array_dtype(&packed) != dtype::UINT32 {
            bail!("Expected packed uint32 weights for {prefix}");
        }
        let out_dim = mlxcel_core::array_shape(&packed)[0];
        let (bits, group_size) = infer_layout(&packed, &scales, out_dim)?;
        // SAFETY: `biases` is a live array for the duration of the call.
        let dense: UniquePtr<MlxArray> = unsafe {
            mlxcel_core::dequantize(&packed, &scales, &*biases, group_size, bits, "affine")
        };
        weights.insert(format!("{prefix}.weight"), dense);
    }
    Ok(prefixes.len())
}

/// Load `vae/` with its encoder, casting floats to `compute_dtype`.
pub fn load_vae(dir: &Path, compute_dtype: i32) -> Result<Flux2Vae> {
    let config = Flux2VaeConfig::from_dir(dir)?;
    let mut raw = mlxcel_core::weights::load_weights_from_dir(dir)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load VAE weights from {}", dir.display()))?;
    dequantize_square_linears(&mut raw)?;
    Flux2Vae::from_raw_weights(config, raw, true, compute_dtype)
}
