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

//! Load-time checks of Qwen4-Exp tensor shapes against the config.
//!
//! MLX validates operand shapes when an op is built, and the cxx bridge turns
//! that throw into `std::terminate`, so a tensor whose stored shape disagrees
//! with the config aborts the process at the first forward pass instead of
//! failing the load. MLX's gathers do not check their indices at all: a router
//! wider than the stacked experts makes `gather_qmm` read past the expert
//! planes.

use mlxcel_core::layers::{infer_quantization_mode, reconcile_quantization_layout};
use mlxcel_core::weights::WeightMap;

/// The logical shape of the matrix (or stack of matrices) stored under
/// `module`: a quantized plane reports its unpacked column count, derived the
/// way `UnifiedLinear` reconciles it.
fn logical_shape(
    weights: &WeightMap,
    module: &str,
    (group_size, bits): (i32, i32),
) -> Result<Vec<i32>, String> {
    let key = format!("{module}.weight");
    let weight = weights
        .get(&key)
        .ok_or_else(|| format!("Qwen4-Exp checkpoint is missing {key}"))?;
    let mut shape = mlxcel_core::array_shape(weight);
    if shape.len() < 2 {
        return Err(format!("{key} has shape {shape:?}, expected a matrix"));
    }
    let Some(scales) = weights.get(&format!("{module}.scales")) else {
        return Ok(shape);
    };
    let scales_shape = mlxcel_core::array_shape(scales);
    let rows = &shape[..shape.len() - 1];
    if scales_shape.len() != shape.len() || scales_shape[..rows.len()] != rows[..] {
        return Err(format!(
            "{module}.scales has shape {scales_shape:?}, which does not match {key} {shape:?}"
        ));
    }
    let biases_key = format!("{module}.biases");
    let biases = weights.get(&biases_key);
    if let Some(biases) = biases {
        let biases_shape = mlxcel_core::array_shape(biases);
        if biases_shape != scales_shape {
            return Err(format!(
                "{biases_key} has shape {biases_shape:?}, but {module}.scales has {scales_shape:?}"
            ));
        }
    }
    let mode = infer_quantization_mode(biases.is_some(), group_size, bits);
    let layout = reconcile_quantization_layout(&shape, &scales_shape, group_size, bits, mode)
        .map_err(|error| format!("{module}: {error}"))?;
    let groups = scales_shape[scales_shape.len() - 1];
    *shape.last_mut().expect("rank >= 2") = groups.saturating_mul(layout.group_size);
    Ok(shape)
}

/// `module` stores `expected` (`[rows, columns]`, or `[experts, rows,
/// columns]` for stacked experts) in logical terms, and a 2-D module's bias,
/// when present, has `rows` entries.
pub(crate) fn expect_matrix(
    weights: &WeightMap,
    module: &str,
    quantization: (i32, i32),
    expected: &[usize],
) -> Result<(), String> {
    let shape = logical_shape(weights, module, quantization)?;
    if !shape
        .iter()
        .map(|&dim| dim as usize)
        .eq(expected.iter().copied())
    {
        return Err(format!(
            "Qwen4-Exp {module} holds a {shape:?} matrix, but the config implies {expected:?}"
        ));
    }
    if let [rows, _] = expected {
        expect_vector_if_present(weights, &format!("{module}.bias"), *rows)?;
    }
    Ok(())
}

/// The 1-D tensor under `key` exists and holds `len` entries.
pub(crate) fn expect_vector(weights: &WeightMap, key: &str, len: usize) -> Result<(), String> {
    if !weights.contains_key(key) {
        return Err(format!("Qwen4-Exp checkpoint is missing {key}"));
    }
    expect_vector_if_present(weights, key, len)
}

fn expect_vector_if_present(weights: &WeightMap, key: &str, len: usize) -> Result<(), String> {
    let Some(tensor) = weights.get(key) else {
        return Ok(());
    };
    let shape = mlxcel_core::array_shape(tensor);
    if shape.len() != 1 || shape[0] as usize != len {
        return Err(format!(
            "Qwen4-Exp {key} has shape {shape:?}, but the config implies [{len}]"
        ));
    }
    Ok(())
}
