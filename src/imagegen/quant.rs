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

//! Quantization checks shared by the image families that load mlx-vlm
//! converted (quantized) checkpoints.
//!
//! Each family reads the mode, bits and group size from its own metadata
//! (Z-Image: the component index metadata; Mage-Flow: `transformer/config.json`)
//! and validates them with [`QuantSpec::validated`]. Every packed tensor is
//! then checked by [`validate_quantized_triple`] before MLX sees it, so a
//! malformed checkpoint is an error naming the tensor instead of a C++
//! exception thrown across the FFI boundary.
//!
//! Used by: ZImage (dequantize at load), MageFlow (quantized matmul).

use anyhow::{Context, Result, bail};
use mlxcel_core::MlxArray;
use mlxcel_core::weights::WeightMap;

/// A validated quantization mode, bit width and group size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantSpec {
    pub mode: String,
    pub bits: i32,
    pub group_size: i32,
}

impl QuantSpec {
    /// Accept what MLX's `quantize` / `dequantize` accept: `affine` at bits 2,
    /// 3, 4, 5, 6 or 8 and group size 32, 64 or 128, `mxfp4` (4, 32), `mxfp8`
    /// (8, 32) and `nvfp4` (4, 16). `family` names the model in the error.
    pub fn validated(family: &str, mode: &str, bits: i32, group_size: i32) -> Result<Self> {
        let unsupported = || {
            anyhow::anyhow!(
                "Unsupported {family} quantization metadata: mode={mode:?}, bits={bits}, group_size={group_size}"
            )
        };
        let expected = match mode {
            "affine" => {
                if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) || !matches!(group_size, 32 | 64 | 128) {
                    return Err(unsupported());
                }
                None
            }
            "mxfp4" => Some((4, 32)),
            "mxfp8" => Some((8, 32)),
            "nvfp4" => Some((4, 16)),
            _ => return Err(unsupported()),
        };
        if expected.is_some_and(|e| e != (bits, group_size)) || bits < 1 || group_size < 1 {
            return Err(unsupported());
        }
        Ok(Self {
            mode: mode.to_string(),
            bits,
            group_size,
        })
    }
}

/// Check one packed `<p>.weight` / `<p>.scales` / optional `<p>.biases` triple
/// against `spec`: uint32 packed weights whose unpacked width is whole and a
/// multiple of the group size, scales shaped `[..., in / group_size]`, a
/// floating scale dtype with matching biases for `affine`, uint8 scales and no
/// biases for the block-float modes.
pub fn validate_quantized_triple(
    key: &str,
    spec: &QuantSpec,
    packed: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
) -> Result<()> {
    use mlxcel_core::dtype::{BFLOAT16, FLOAT16, FLOAT32, UINT8, UINT32};
    let affine = spec.mode == "affine";
    let packed_shape = mlxcel_core::array_shape(packed);
    let scales_shape = mlxcel_core::array_shape(scales);
    if mlxcel_core::array_dtype(packed) != UINT32 {
        bail!("Quantized tensor {key} must be uint32 packed weights");
    }
    if packed_shape.len() < 2 || packed_shape.iter().any(|&d| d < 1) {
        bail!("Quantized tensor {key} has an invalid packed shape {packed_shape:?}");
    }
    // Each uint32 holds 32 / bits values, so the unpacked input width is
    // `last * 32 / bits`; it must be whole and a multiple of the group size.
    let last = i64::from(*packed_shape.last().expect("checked non-empty"));
    let unpacked = last * 32;
    if unpacked % i64::from(spec.bits) != 0 {
        bail!(
            "Quantized tensor {key} packed width {last} is not whole at {} bits",
            spec.bits
        );
    }
    let in_dim = unpacked / i64::from(spec.bits);
    if in_dim % i64::from(spec.group_size) != 0 {
        bail!(
            "Quantized tensor {key} input width {in_dim} is not a multiple of group size {}",
            spec.group_size
        );
    }
    let groups = i32::try_from(in_dim / i64::from(spec.group_size))
        .with_context(|| format!("Quantized tensor {key} is too wide"))?;
    let mut expected = packed_shape.clone();
    *expected.last_mut().expect("checked non-empty") = groups;
    if scales_shape != expected {
        bail!("Quantized tensor {key} scales shape {scales_shape:?} does not match {expected:?}");
    }
    let scales_dtype = mlxcel_core::array_dtype(scales);
    if affine {
        if !matches!(scales_dtype, FLOAT16 | BFLOAT16 | FLOAT32) {
            bail!("Quantized tensor {key} affine scales must be a floating dtype");
        }
        let Some(biases) = biases else {
            bail!("Quantized tensor {key} is affine but has no biases");
        };
        if mlxcel_core::array_shape(biases) != expected
            || mlxcel_core::array_dtype(biases) != scales_dtype
        {
            bail!("Quantized tensor {key} biases do not match its scales in shape or dtype");
        }
    } else {
        if scales_dtype != UINT8 {
            bail!("Quantized tensor {key} {} scales must be uint8", spec.mode);
        }
        if biases.is_some() {
            bail!("Quantized tensor {key} is {} but has biases", spec.mode);
        }
    }
    Ok(())
}

/// Module prefixes `<p>` of every `<p>.scales` in `weights`, sorted.
#[must_use]
pub fn quantized_prefixes(weights: &WeightMap) -> Vec<String> {
    let mut prefixes: Vec<String> = weights
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(str::to_string))
        .collect();
    prefixes.sort();
    prefixes
}

/// [`validate_quantized_triple`] for every quantized module of `weights`.
pub fn validate_quantized_weights(weights: &WeightMap, spec: &QuantSpec) -> Result<()> {
    for prefix in quantized_prefixes(weights) {
        let weight_key = format!("{prefix}.weight");
        let packed = weights
            .get(&weight_key)
            .with_context(|| format!("Quantized weights have no matching module: {prefix}"))?;
        let scales = weights
            .get(&format!("{prefix}.scales"))
            .context("scales disappeared")?;
        let biases = weights.get(&format!("{prefix}.biases"));
        validate_quantized_triple(&weight_key, spec, packed, scales, biases.map(|b| &**b))?;
    }
    Ok(())
}
