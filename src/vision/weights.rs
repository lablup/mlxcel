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

//! Quantization-aware loading for learned vision parameters that are consumed
//! as whole dense arrays rather than through a linear or embedding layer.

use mlxcel_core::dtype::{BFLOAT16, FLOAT16, FLOAT32, UINT8, UINT32};
use mlxcel_core::layers::{
    QuantizedTensorShapes, infer_quantization_mode, reconcile_quantization_layout,
    validate_affine_quantization_bits, validate_affine_quantization_group_size,
    validate_one_bit_layout, validate_quantization_biases,
    validate_quantization_mode_for_running_backend, validate_quantized_packing,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn sidecar_key(key: &str, suffix: &str) -> String {
    key.strip_suffix(".weight").map_or_else(
        || format!("{key}.{suffix}"),
        |prefix| format!("{prefix}.{suffix}"),
    )
}

fn checked_rows(shape: &[i32], key: &str) -> Result<i32, String> {
    if shape.is_empty() || shape.iter().any(|&dim| dim <= 0) {
        return Err(format!(
            "Learned parameter {key} has invalid expected shape {shape:?}"
        ));
    }
    shape[..shape.len() - 1]
        .iter()
        .try_fold(1_i32, |rows, &dim| rows.checked_mul(dim))
        .ok_or_else(|| format!("Learned parameter {key} expected shape {shape:?} is too large"))
}

fn validate_quantized_plane_shape(
    key: &str,
    plane: &str,
    actual: &[i32],
    expected: &[i32],
) -> Result<(), String> {
    if actual.len() != expected.len() || actual.is_empty() || actual.iter().any(|&dim| dim <= 0) {
        return Err(format!(
            "Quantized learned parameter {key} {plane} has invalid shape {actual:?}; expected rank {} with positive dimensions",
            expected.len()
        ));
    }
    if actual[..actual.len() - 1] != expected[..expected.len() - 1] {
        return Err(format!(
            "Quantized learned parameter {key} {plane} leading dimensions {:?} do not match expected {:?}",
            &actual[..actual.len() - 1],
            &expected[..expected.len() - 1]
        ));
    }
    Ok(())
}

struct ValidatedQuantizedParam {
    rows: i32,
    packed_width: i32,
    groups: i32,
    group_size: i32,
    bits: i32,
    mode: &'static str,
}

fn validate_param(
    weights: &WeightMap,
    key: &str,
    expected_shape: &[i32],
    quant: Option<(i32, i32)>,
) -> Result<Option<ValidatedQuantizedParam>, String> {
    let rows = checked_rows(expected_shape, key)?;
    let expected_width = expected_shape[expected_shape.len() - 1];
    let weight = weights
        .get(key)
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    let scales_key = sidecar_key(key, "scales");
    let biases_key = sidecar_key(key, "biases");
    let Some(scales) = weights.get(&scales_key) else {
        let actual = mlxcel_core::array_shape(weight);
        let dtype = mlxcel_core::array_dtype(weight);
        if dtype == UINT32 {
            return Err(format!(
                "Quantized learned parameter {key} is missing its scales sidecar {scales_key}"
            ));
        }
        if !matches!(dtype, FLOAT16 | BFLOAT16 | FLOAT32) {
            return Err(format!(
                "Learned parameter {key} dense weights must have a floating dtype"
            ));
        }
        if weights.contains_key(&biases_key) {
            return Err(format!(
                "Learned parameter {key} has biases sidecar {biases_key} without scales sidecar {scales_key}"
            ));
        }
        if actual != expected_shape {
            return Err(format!(
                "Learned parameter {key} shape mismatch: expected {expected_shape:?}, got {actual:?}"
            ));
        }
        return Ok(None);
    };

    let biases = weights.get(&biases_key);
    let weight_shape = mlxcel_core::array_shape(weight);
    let scales_shape = mlxcel_core::array_shape(scales);
    validate_quantized_plane_shape(key, "packed weight", &weight_shape, expected_shape)?;
    validate_quantized_plane_shape(key, "scales", &scales_shape, expected_shape)?;
    if mlxcel_core::array_dtype(weight) != UINT32 {
        return Err(format!(
            "Quantized learned parameter {key} packed weight must have uint32 dtype"
        ));
    }

    let scales_dtype = mlxcel_core::array_dtype(scales);
    if let Some(biases) = biases {
        let biases_shape = mlxcel_core::array_shape(biases);
        if biases_shape != scales_shape || mlxcel_core::array_dtype(biases) != scales_dtype {
            return Err(format!(
                "Quantized learned parameter {key} biases shape and dtype must match its scales"
            ));
        }
        if !matches!(scales_dtype, FLOAT16 | BFLOAT16 | FLOAT32) {
            return Err(format!(
                "Quantized learned parameter {key} affine scales must have a floating dtype"
            ));
        }
    } else if scales_dtype != UINT8 {
        return Err(format!(
            "Quantized learned parameter {key} block-float scales must have uint8 dtype"
        ));
    }

    let packed_width = weight_shape[weight_shape.len() - 1];
    let groups = scales_shape[scales_shape.len() - 1];
    let packed_bits = packed_width
        .checked_mul(32)
        .ok_or_else(|| format!("Quantized learned parameter {key} packed width overflows i32"))?;
    if packed_bits % expected_width != 0 || expected_width % groups != 0 {
        return Err(format!(
            "Quantized learned parameter {key} shape mismatch: expected {expected_shape:?}, packed weight {weight_shape:?} and scales {scales_shape:?} do not describe an integral layout"
        ));
    }
    let effective_bits = packed_bits / expected_width;
    let effective_group_size = expected_width / groups;
    let mode = infer_quantization_mode(biases.is_some(), effective_group_size, effective_bits);
    if mode == "affine" {
        if effective_bits != 1 {
            validate_affine_quantization_bits(effective_bits)
                .map_err(|error| format!("{error} (parameter: {key})"))?;
        }
        validate_affine_quantization_group_size(effective_group_size)
            .map_err(|error| format!("{error} (parameter: {key})"))?;
    } else {
        let supported = matches!(
            (mode, effective_group_size, effective_bits),
            ("mxfp4", 32, 4) | ("mxfp8", 32, 8) | ("nvfp4", 16, 4)
        );
        if !supported {
            return Err(format!(
                "Quantized learned parameter {key} has unsupported block-float layout mode={mode}, group_size={effective_group_size}, bits={effective_bits}"
            ));
        }
    }
    let flat_weight_shape = [rows, packed_width];
    let flat_scales_shape = [rows, groups];
    let layout = reconcile_quantization_layout(
        &flat_weight_shape,
        &flat_scales_shape,
        effective_group_size,
        effective_bits,
        mode,
    )
    .map_err(|error| format!("{error} (parameter: {key})"))?;
    let (declared_group_size, declared_bits) = quant.unwrap_or((64, 4));
    reconcile_quantization_layout(&[], &[], declared_group_size, declared_bits, mode)
        .map_err(|error| format!("{error} (parameter: {key})"))?;
    validate_quantization_mode_for_running_backend(mode)
        .map_err(|error| format!("{error} (parameter: {key})"))?;
    validate_quantization_biases(mode, biases.is_some())
        .map_err(|error| format!("{error} (parameter: {key})"))?;
    validate_one_bit_layout(layout.bits, layout.group_size, mode, biases.is_some())
        .map_err(|error| format!("{error} (parameter: {key})"))?;
    validate_quantized_packing(
        key,
        &QuantizedTensorShapes {
            weight: &flat_weight_shape,
            scales: &flat_scales_shape,
            biases: biases.map(|_| flat_scales_shape.as_slice()),
        },
        usize::try_from(expected_width)
            .map_err(|_| format!("Learned parameter {key} has invalid width {expected_width}"))?,
        layout.group_size,
        layout.bits,
        mode,
    )?;
    if (declared_group_size, declared_bits) != (layout.group_size, layout.bits) {
        tracing::warn!(
            target: "mlxcel::quant",
            parameter = key,
            mode,
            declared_bits,
            declared_group_size,
            effective_bits = layout.bits,
            effective_group_size = layout.group_size,
            "quantized learned-parameter layout does not match declared metadata; using shape-derived params"
        );
    }
    Ok(Some(ValidatedQuantizedParam {
        rows,
        packed_width,
        groups,
        group_size: layout.group_size,
        bits: layout.bits,
        mode,
    }))
}

/// Return shape-derived effective `(group_size, bits)` for a packed parameter
/// after fully validating its geometry and sidecar dtypes before any embedding
/// constructor or MLX FFI operation observes it. Dense parameters return
/// `None` after their shape is validated.
///
/// Used by: GLM-4V, Hunyuan-VL, Phi4-SigLIP, DeepSeek-OCR CLIP,
/// Learnable2DInterpPosEmb (LFM2-VL/Kimi-VL), and SAM3 prompt embeddings.
pub(crate) fn resolve_param_quantization(
    weights: &WeightMap,
    key: &str,
    expected_shape: &[i32],
    quant: Option<(i32, i32)>,
) -> Result<Option<(i32, i32)>, String> {
    validate_param(weights, key, expected_shape, quant)
        .map(|layout| layout.map(|layout| (layout.group_size, layout.bits)))
}

/// Load a learned vision parameter as a dense array, dequantizing a packed
/// `<key>` / sibling `.scales` / optional `.biases` triple when present.
///
/// Used by: Gemma4 and Gemma4-Unified; Step3p7; Llama4; Molmo, Molmo2 and
/// MolmoPoint; Jina-VLM; InternVL; DeepSeek-VL2 and DeepSeek-OCR; Moondream;
/// MoonViT3D; Nemotron-H Nano Omni; Idefics2; Hunyuan-VL framing; Granite and
/// Granite4 Vision; DINOv2; RF-DETR; SAM3, SAM3.1 and SAM3D Body; and the
/// shared Molmo/Jina text embedding loader.
pub(crate) fn load_dense_param(
    weights: &WeightMap,
    key: &str,
    expected_shape: &[i32],
    quant: Option<(i32, i32)>,
) -> Result<UniquePtr<MlxArray>, String> {
    let validated = validate_param(weights, key, expected_shape, quant)?;
    let weight = weights
        .get(key)
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    let Some(validated) = validated else {
        return Ok(mlxcel_core::copy(weight));
    };
    let scales_key = sidecar_key(key, "scales");
    let biases_key = sidecar_key(key, "biases");
    let scales = weights
        .get(&scales_key)
        .ok_or_else(|| format!("Weight not found: {scales_key}"))?;
    let biases = weights.get(&biases_key);
    let flat_weight_shape = [validated.rows, validated.packed_width];
    let flat_scales_shape = [validated.rows, validated.groups];
    let flat_weight = mlxcel_core::reshape(weight, &flat_weight_shape);
    let flat_scales = mlxcel_core::reshape(scales, &flat_scales_shape);
    let flat_biases = biases.map(|value| mlxcel_core::reshape(value, &flat_scales_shape));
    let bias_ptr = flat_biases.as_ref().map_or(std::ptr::null(), |value| {
        value.as_ref().expect("reshape returned a non-null array") as *const MlxArray
    });
    // SAFETY: the packed/scales/biases dtypes and shapes, mode, group size and
    // bit width are validated above; `bias_ptr` is null or remains alive for
    // the duration of this call.
    let dense = unsafe {
        mlxcel_core::dequantize(
            &flat_weight,
            &flat_scales,
            bias_ptr,
            validated.group_size,
            validated.bits,
            validated.mode,
        )
    };
    let dense_shape = mlxcel_core::array_shape(&dense);
    let expected_flat = [validated.rows, expected_shape[expected_shape.len() - 1]];
    if dense_shape != expected_flat {
        return Err(format!(
            "Learned parameter {key} dequantized to shape {dense_shape:?}; expected {expected_shape:?}"
        ));
    }
    Ok(mlxcel_core::reshape(&dense, expected_shape))
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
