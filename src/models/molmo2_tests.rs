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

use super::Molmo2Embedding;
use mlxcel_core::weights::WeightMap;

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
const HIDDEN: i32 = 128;

fn dense(rows: i32, offset: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let values: Vec<f32> = (0..rows * HIDDEN)
        .map(|index| offset + (index as f32 * 0.037).sin())
        .collect();
    mlxcel_core::from_slice_f32(&values, &[rows, HIDDEN])
}

fn insert_quantized(
    weights: &mut WeightMap,
    key: &str,
    source: &mlxcel_core::MlxArray,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let quantized = mlxcel_core::quantize_weights(source, GROUP_SIZE, BITS);
    let packed = mlxcel_core::quantized_weights_w(&quantized);
    let scales = mlxcel_core::quantized_weights_scales(&quantized);
    let biases = mlxcel_core::quantized_weights_biases(&quantized);
    let reference = unsafe {
        mlxcel_core::dequantize(
            &packed,
            &scales,
            biases.as_ref().expect("affine biases") as *const _,
            GROUP_SIZE,
            BITS,
            "affine",
        )
    };
    weights.insert(key.to_string(), packed);
    weights.insert(format!("{key}.scales"), scales);
    weights.insert(format!("{key}.biases"), biases);
    reference
}

#[test]
fn quantized_base_and_extension_are_dequantized_before_concat_and_lookup() {
    let base = dense(3, -0.25);
    let extension = dense(2, 0.5);
    let mut weights = WeightMap::new();
    let base_reference = insert_quantized(&mut weights, "wte.embedding", &base);
    let extension_reference = insert_quantized(&mut weights, "wte.new_embedding", &extension);
    let reference = mlxcel_core::concatenate(&base_reference, &extension_reference, 0);

    let embedding =
        Molmo2Embedding::from_weights(&weights, "wte", 3, 2, HIDDEN as usize, GROUP_SIZE, BITS)
            .expect("quantized dual embedding");
    assert_eq!(mlxcel_core::array_shape(&embedding.table), vec![5, HIDDEN]);

    let ids = mlxcel_core::from_slice_i32(&[0, 2, 3, 4], &[4]);
    let actual = embedding.forward(&ids);
    let expected = mlxcel_core::embedding(&reference, &ids);
    let difference = mlxcel_core::abs(&mlxcel_core::subtract(&actual, &expected));
    let maximum = mlxcel_core::max_axis(&difference, 0, false);
    let maximum = mlxcel_core::max_axis(&maximum, 0, false);
    mlxcel_core::eval(&maximum);
    assert!(
        mlxcel_core::item_f32(&maximum) <= 1e-3,
        "quantized dual-table lookup must match explicit dequantize"
    );
}
