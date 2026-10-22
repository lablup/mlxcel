// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact kernel-4 transposed convolution used by the upstream Sapiens2 heads.

use mlxcel_core::{MlxArray, UniquePtr};

const PHASE_TAPS: [[i32; 2]; 2] = [[3, 1], [2, 0]];

/// Convert `[out, 4, 4, in]` weights to the upstream one-GEMM layout.
pub(super) fn gemm_weight(weight: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(weight);
    let output = shape[0];
    let input = shape[3];
    let taps = PHASE_TAPS.map(|indices| mlxcel_core::from_slice_i32(&indices, &[2]));
    let mut phases = Vec::with_capacity(4);
    for rows in &taps {
        for columns in &taps {
            let rows = mlxcel_core::take(weight, rows, 1);
            phases.push(mlxcel_core::take(&rows, columns, 2));
        }
    }
    let phases = mlxcel_core::stack_owned(&phases, 0);
    let phases = mlxcel_core::transpose_axes(&phases, &[2, 3, 4, 0, 1]);
    mlxcel_core::reshape(&phases, &[4 * input, 4 * output])
}

/// Run kernel-4, stride-2, padding-1 transposed convolution as upstream does.
pub(super) fn forward(
    x: &MlxArray,
    weight: &MlxArray,
    output_channels: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let [batch, height, width, input] = [shape[0], shape[1], shape[2], shape[3]];
    let padded = mlxcel_core::pad(x, &[0, 0, 1, 1, 1, 1, 0, 0], 0.0);
    let mut patches = Vec::with_capacity(4);
    for row in 0..2 {
        for column in 0..2 {
            patches.push(mlxcel_core::slice(
                &padded,
                &[0, row, column, 0],
                &[batch, row + height + 1, column + width + 1, input],
            ));
        }
    }
    let patch_refs: Vec<&MlxArray> = patches
        .iter()
        .map(|patch| patch.as_ref().expect("MLX patch allocation"))
        .collect();
    let columns = mlxcel_core::concatenate_many(&patch_refs, -1);
    let columns = mlxcel_core::reshape(&columns, &[-1, 4 * input]);
    let output = mlxcel_core::matmul(&columns, weight);
    let output = mlxcel_core::reshape(
        &output,
        &[batch, height + 1, width + 1, 2, 2, output_channels],
    );

    let mut phases = Vec::with_capacity(4);
    for row in 0..2 {
        for column in 0..2 {
            let phase = mlxcel_core::slice(
                &output,
                &[0, row, column, row, column, 0],
                &[
                    batch,
                    row + height,
                    column + width,
                    row + 1,
                    column + 1,
                    output_channels,
                ],
            );
            phases.push(mlxcel_core::reshape(
                &phase,
                &[batch, height, width, output_channels],
            ));
        }
    }
    let output = mlxcel_core::stack_owned(&phases, 3);
    let output = mlxcel_core::reshape(&output, &[batch, height, width, 2, 2, output_channels]);
    let output = mlxcel_core::transpose_axes(&output, &[0, 1, 3, 2, 4, 5]);
    mlxcel_core::reshape(&output, &[batch, 2 * height, 2 * width, output_channels])
}
