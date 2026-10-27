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

#[test]
fn quantized_hunyuan_cls_grid_table_matches_explicit_dequantize() {
    let values: Vec<f32> = (0..5 * 64)
        .map(|index| (index as f32 * 0.023).sin())
        .collect();
    let dense = mlxcel_core::from_slice_f32(&values, &[5, 64]);
    let quantized = mlxcel_core::quantize_weights(&dense, 64, 4);
    let packed = mlxcel_core::quantized_weights_w(&quantized);
    let scales = mlxcel_core::quantized_weights_scales(&quantized);
    let biases = mlxcel_core::quantized_weights_biases(&quantized);
    let dequantized = unsafe {
        mlxcel_core::dequantize(
            &packed,
            &scales,
            biases.as_ref().expect("affine biases") as *const _,
            64,
            4,
            "affine",
        )
    };
    let mut packed_map = WeightMap::new();
    packed_map.insert("tower.embeddings.position_embedding.weight".into(), packed);
    packed_map.insert("tower.embeddings.position_embedding.scales".into(), scales);
    packed_map.insert("tower.embeddings.position_embedding.biases".into(), biases);
    let mut dense_map = WeightMap::new();
    dense_map.insert(
        "tower.embeddings.position_embedding.weight".into(),
        dequantized,
    );

    let (actual, actual_edge) =
        load_position_table(&packed_map, "tower", 64, 64, 4).expect("quantized table");
    assert_eq!(mlxcel_core::array_shape(&actual), [2, 2, 64]);
    let (expected, expected_edge) =
        load_position_table(&dense_map, "tower", 64, 64, 4).expect("dense table");
    assert_eq!((actual_edge, expected_edge), (2, 2));
    let diff = mlxcel_core::reshape(&mlxcel_core::subtract(&actual, &expected), &[-1]);
    let max = mlxcel_core::max_axis(&mlxcel_core::abs(&diff), 0, false);
    mlxcel_core::eval(&max);
    assert!(mlxcel_core::item_f32(&max) <= 1e-6);
}
