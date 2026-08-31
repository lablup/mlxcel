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

use mlxcel_core::weights::WeightMap;

pub(super) fn exact(weights: &WeightMap, name: &str, expected: &[i32]) -> Result<(), String> {
    let shape = weights
        .get(name)
        .map(|value| mlxcel_core::array_shape(value))
        .ok_or_else(|| format!("Weight not found: {name}"))?;
    if shape != expected {
        return Err(format!("{name}: expected {expected:?}, got {shape:?}"));
    }
    Ok(())
}

pub(super) fn vector(weights: &WeightMap, name: &str, length: usize) -> Result<(), String> {
    exact(weights, name, &[length as i32])
}

pub(super) fn matrix(
    weights: &WeightMap,
    prefix: &str,
    rows: usize,
    columns: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let name = format!("{prefix}.weight");
    let shape = weights
        .get(&name)
        .map(|value| mlxcel_core::array_shape(value))
        .ok_or_else(|| format!("Weight not found: {name}"))?;
    let quantized = weights.contains_key(&format!("{prefix}.scales"));
    if shape.len() != 2 || shape[0] != rows as i32 || (!quantized && shape[1] != columns as i32) {
        return Err(format!(
            "{name}: expected [{rows}, {columns}]{}; got {shape:?}",
            if quantized {
                " (packed columns allowed)"
            } else {
                ""
            }
        ));
    }
    validate_quantized_packing(weights, prefix, columns, group_size, bits, false)
}

pub(super) fn expert(
    weights: &WeightMap,
    prefix: &str,
    experts: usize,
    rows: usize,
    columns: usize,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let name = format!("{prefix}.weight");
    let shape = weights
        .get(&name)
        .map(|value| mlxcel_core::array_shape(value))
        .ok_or_else(|| format!("Weight not found: {name}"))?;
    let quantized = weights.contains_key(&format!("{prefix}.scales"));
    if shape.len() != 3
        || shape[0] != experts as i32
        || shape[1] != rows as i32
        || (!quantized && shape[2] != columns as i32)
    {
        return Err(format!(
            "{name}: expected [{experts}, {rows}, {columns}]{}; got {shape:?}",
            if quantized {
                " (packed columns allowed)"
            } else {
                ""
            }
        ));
    }
    validate_quantized_packing(weights, prefix, columns, group_size, bits, true)
}

fn validate_quantized_packing(
    weights: &WeightMap,
    prefix: &str,
    in_features: usize,
    group_size: i32,
    bits: i32,
    native_expert_allowed: bool,
) -> Result<(), String> {
    let Some(scales) = weights.get(&format!("{prefix}.scales")) else {
        if weights.contains_key(&format!("{prefix}.biases")) {
            return Err(format!(
                "{prefix}.biases exists without the required {prefix}.scales"
            ));
        }
        return Ok(());
    };
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let weight_shape = mlxcel_core::array_shape(weight);
    let scales_shape = mlxcel_core::array_shape(scales);
    let biases_shape = weights
        .get(&format!("{prefix}.biases"))
        .map(|value| mlxcel_core::array_shape(value));
    let has_biases = biases_shape.is_some();
    let (group_size, bits, mode) = if native_expert_allowed && !has_biases {
        (16, 4, "nvfp4")
    } else {
        (
            group_size,
            bits,
            mlxcel_core::layers::infer_quantization_mode(has_biases, group_size, bits),
        )
    };
    mlxcel_core::layers::validate_quantized_packing(
        prefix,
        &mlxcel_core::layers::QuantizedTensorShapes {
            weight: &weight_shape,
            scales: &scales_shape,
            biases: biases_shape.as_deref(),
        },
        in_features,
        group_size,
        bits,
        mode,
    )
}

pub(super) fn conv(
    weights: &WeightMap,
    name: &str,
    channels: usize,
    kernel: usize,
) -> Result<(), String> {
    exact(weights, name, &[channels as i32, kernel as i32, 1])
}
