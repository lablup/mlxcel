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

use safetensors::{Dtype, tensor::TensorView};

pub(super) fn tensor_i64_as_f32(tensor: TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::I64);
    tensor
        .data()
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()) as f32)
        .collect()
}

pub(super) fn tensor_i32(tensor: TensorView<'_>) -> Vec<[usize; 2]> {
    assert_eq!(tensor.dtype(), Dtype::I32);
    let values: Vec<usize> = tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()) as usize)
        .collect();
    values
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect()
}

pub(super) fn tensor_bf16_as_f32(tensor: TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::BF16);
    tensor
        .data()
        .chunks_exact(2)
        .map(|bytes| {
            let bits = u16::from_le_bytes(bytes.try_into().unwrap());
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect()
}

pub(super) fn assert_bf16_stage(
    name: &str,
    actual: &mlxcel_core::MlxArray,
    expected: TensorView<'_>,
    max_tolerance: f32,
    mean_tolerance: f32,
) {
    let actual = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        actual,
        mlxcel_core::dtype::FLOAT32,
    ));
    let expected = tensor_bf16_as_f32(expected);
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    let mut total = 0.0_f32;
    for (actual, expected) in actual.iter().zip(&expected) {
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        total += error;
    }
    let mean = total / actual.len() as f32;
    assert!(
        maximum <= max_tolerance && mean <= mean_tolerance,
        "{name} error: max {maximum}, mean {mean}"
    );
}

pub(super) fn check_bf16_values_exact(
    name: &str,
    actual: &mlxcel_core::MlxArray,
    expected: TensorView<'_>,
) -> Result<(), String> {
    if expected.dtype() != Dtype::BF16 {
        return Err(format!("{name} reference is not BF16"));
    }
    let actual = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        actual,
        mlxcel_core::dtype::FLOAT32,
    ));
    let expected = tensor_bf16_as_f32(expected);
    if actual.len() != expected.len() {
        return Err(format!(
            "{name} element count differs: actual={}, expected={}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        if actual != expected {
            return Err(format!(
                "{name} value {index} differs: actual={actual}, expected={expected}"
            ));
        }
    }
    Ok(())
}

pub(super) fn report_bf16_rows(
    name: &str,
    actual: &mlxcel_core::MlxArray,
    expected: TensorView<'_>,
    active: &[f32],
) {
    let shape = expected.shape();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], active.len());
    let width = shape[2];
    let actual = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        actual,
        mlxcel_core::dtype::FLOAT32,
    ));
    let expected = tensor_bf16_as_f32(expected);
    for (row, (actual, expected)) in actual
        .chunks_exact(width)
        .zip(expected.chunks_exact(width))
        .enumerate()
    {
        let mut maximum = 0.0_f32;
        let mut total = 0.0_f32;
        for (&actual, &expected) in actual.iter().zip(expected) {
            let error = (actual - expected).abs();
            maximum = maximum.max(error);
            total += error;
        }
        eprintln!(
            "{name} token {row} active={} max_error={maximum} mean_error={}",
            active[row] != 0.0,
            total / width as f32
        );
    }
}

pub(super) fn report_bf16_stage(
    name: &str,
    actual: &mlxcel_core::MlxArray,
    expected: TensorView<'_>,
) {
    let actual = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        actual,
        mlxcel_core::dtype::FLOAT32,
    ));
    let expected = tensor_bf16_as_f32(expected);
    assert_eq!(actual.len(), expected.len(), "{name} element count");
    let mut maximum = 0.0_f32;
    let mut total = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(&expected) {
        let error = if actual == expected {
            0.0
        } else {
            (actual - expected).abs()
        };
        maximum = maximum.max(error);
        total += error;
    }
    eprintln!(
        "text stage {name}: max_error={maximum} mean_error={}",
        total / actual.len() as f32
    );
}

pub(super) fn tensor_f32(tensor: TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::F32);
    tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}
