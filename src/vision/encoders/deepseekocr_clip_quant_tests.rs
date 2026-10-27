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

use super::*;

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

fn sidecar_key(key: &str, suffix: &str) -> String {
    key.strip_suffix(".weight").map_or_else(
        || format!("{key}.{suffix}"),
        |prefix| format!("{prefix}.{suffix}"),
    )
}

fn quantize_table(weights: &mut WeightMap, key: &str, dense: &MlxArray, logical_shape: &[i32]) {
    let width = logical_shape[logical_shape.len() - 1];
    let rows: i32 = logical_shape[..logical_shape.len() - 1].iter().product();
    let flat = mlxcel_core::reshape(dense, &[rows, width]);
    let quantized = mlxcel_core::quantize_weights(&flat, GROUP_SIZE, BITS);
    let packed = mlxcel_core::quantized_weights_w(&quantized);
    let scales = mlxcel_core::quantized_weights_scales(&quantized);
    let biases = mlxcel_core::quantized_weights_biases(&quantized);
    let mut packed_shape = logical_shape.to_vec();
    *packed_shape.last_mut().expect("non-empty table shape") = mlxcel_core::array_shape(&packed)[1];
    let mut sidecar_shape = logical_shape.to_vec();
    *sidecar_shape.last_mut().expect("non-empty table shape") =
        mlxcel_core::array_shape(&scales)[1];
    weights.insert(
        key.to_string(),
        mlxcel_core::reshape(&packed, &packed_shape),
    );
    weights.insert(
        sidecar_key(key, "scales"),
        mlxcel_core::reshape(&scales, &sidecar_shape),
    );
    weights.insert(
        sidecar_key(key, "biases"),
        mlxcel_core::reshape(&biases, &sidecar_shape),
    );
}

fn explicitly_dequantize(
    weights: &WeightMap,
    key: &str,
    logical_shape: &[i32],
) -> UniquePtr<MlxArray> {
    let packed = &weights[key];
    let scales = &weights[&sidecar_key(key, "scales")];
    let biases = &weights[&sidecar_key(key, "biases")];
    let rows: i32 = logical_shape[..logical_shape.len() - 1].iter().product();
    let packed_width = mlxcel_core::array_shape(packed)[logical_shape.len() - 1];
    let groups = mlxcel_core::array_shape(scales)[logical_shape.len() - 1];
    let packed = mlxcel_core::reshape(packed, &[rows, packed_width]);
    let scales = mlxcel_core::reshape(scales, &[rows, groups]);
    let biases = mlxcel_core::reshape(biases, &[rows, groups]);
    // SAFETY: this affine triple is produced by `quantize_weights` above;
    // all planes stay alive for the duration of the call.
    let dense = unsafe {
        mlxcel_core::dequantize(
            &packed,
            &scales,
            biases.as_ref().expect("non-null test biases") as *const MlxArray,
            GROUP_SIZE,
            BITS,
            "affine",
        )
    };
    let dense = mlxcel_core::reshape(&dense, logical_shape);
    assert_eq!(mlxcel_core::array_shape(&dense), logical_shape);
    dense
}

fn assert_close(actual: &MlxArray, expected: &MlxArray) {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected)
    );
    let actual = mlxcel_core::utils::array_to_vec_f32(actual);
    let expected = mlxcel_core::utils::array_to_vec_f32(expected);
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "entry {index}: expected {expected}, got {actual}"
        );
    }
}

fn dense_values(shape: &[i32], phase: f32) -> UniquePtr<MlxArray> {
    let len = shape.iter().map(|&dim| dim as usize).product();
    let values = (0..len)
        .map(|index| ((index as f32 * 0.071) + phase).sin())
        .collect::<Vec<_>>();
    mlxcel_core::from_slice_f32(&values, shape)
}

#[test]
fn quantized_class_and_position_tables_match_explicit_dequantized_clip() {
    let prefix = "vision_model";
    let class_key = format!("{prefix}.embeddings.class_embedding");
    let position_key = format!("{prefix}.embeddings.position_embedding.weight");
    let class_dense = dense_values(&[64], 0.2);
    let position_dense = dense_values(&[5, 64], 0.7);
    let mut quantized_weights = WeightMap::new();
    quantize_table(&mut quantized_weights, &class_key, &class_dense, &[64]);
    quantize_table(
        &mut quantized_weights,
        &position_key,
        &position_dense,
        &[5, 64],
    );
    quantized_weights.insert(
        format!("{prefix}.pre_layrnorm.weight"),
        mlxcel_core::ones(&[64], mlxcel_core::dtype::FLOAT32),
    );
    quantized_weights.insert(
        format!("{prefix}.pre_layrnorm.bias"),
        mlxcel_core::zeros(&[64], mlxcel_core::dtype::FLOAT32),
    );

    let mut dense_weights = WeightMap::new();
    dense_weights.insert(
        class_key.clone(),
        explicitly_dequantize(&quantized_weights, &class_key, &[64]),
    );
    dense_weights.insert(
        position_key.clone(),
        explicitly_dequantize(&quantized_weights, &position_key, &[5, 64]),
    );
    dense_weights.insert(
        format!("{prefix}.pre_layrnorm.weight"),
        mlxcel_core::ones(&[64], mlxcel_core::dtype::FLOAT32),
    );
    dense_weights.insert(
        format!("{prefix}.pre_layrnorm.bias"),
        mlxcel_core::zeros(&[64], mlxcel_core::dtype::FLOAT32),
    );

    let config = ClipConfig {
        hidden_size: 64,
        num_heads: 1,
        num_layers: 0,
        layer_norm_eps: 1e-6,
        pos_grid: 2,
    };
    let quantized =
        ClipEncoder::from_weights(&quantized_weights, prefix, config.clone(), GROUP_SIZE, BITS)
            .expect("quantized CLIP tables");
    assert_eq!(mlxcel_core::array_shape(&quantized.class_embedding), [64]);
    assert_eq!(
        mlxcel_core::array_shape(&quantized.position_embedding),
        [5, 64]
    );
    let dense = ClipEncoder::from_weights(&dense_weights, prefix, config, GROUP_SIZE, BITS)
        .expect("explicitly dequantized CLIP tables");

    assert_close(&quantized.abs_pos(10), &dense.abs_pos(10));
    let patches = dense_values(&[2, 3, 3, 64], 1.1);
    assert_close(&quantized.forward(&patches), &dense.forward(&patches));
}
