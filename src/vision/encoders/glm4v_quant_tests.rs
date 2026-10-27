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

fn maps() -> (WeightMap, WeightMap) {
    let values: Vec<f32> = (0..4 * 64)
        .map(|index| (index as f32 * 0.031).sin())
        .collect();
    let dense = mlxcel_core::from_slice_f32(&values, &[4, 64]);
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
    packed_map.insert("vision.position_embedding.weight".into(), packed);
    packed_map.insert("vision.position_embedding.scales".into(), scales);
    packed_map.insert("vision.position_embedding.biases".into(), biases);
    let mut dense_map = WeightMap::new();
    dense_map.insert("vision.position_embedding.weight".into(), dequantized);
    (packed_map, dense_map)
}

#[test]
fn quantized_glm_position_lookup_and_interpolation_matches_explicit_dequantize() {
    let (packed, dense) = maps();
    let config = Glm4vVisionConfig {
        depth: 0,
        hidden_size: 64,
        intermediate_size: 64,
        num_heads: 1,
        patch_size: 1,
        out_hidden_size: 64,
        spatial_merge_size: 1,
        temporal_patch_size: 1,
        image_size: 2,
        in_channels: 3,
        rms_norm_eps: 1e-5,
        quant_group_size: 64,
        quant_bits: 4,
    };
    let quantized = VisionEmbeddings::from_weights(&packed, &config, "vision")
        .expect("quantized GLM position table");
    assert_eq!(
        mlxcel_core::array_shape(&quantized.position_embedding.dequantized_weight()),
        [4, 64]
    );
    let reference = VisionEmbeddings::from_weights(&dense, &config, "vision")
        .expect("explicitly dequantized GLM position table");
    let input = mlxcel_core::zeros(&[4, 64], mlxcel_core::dtype::FLOAT32);
    let h = [0, 1, 0, 1];
    let w = [0, 0, 1, 1];
    let shape = [2, 2, 2, 2];
    let actual = quantized.forward(&input, &h, &w, &shape, &shape);
    let expected = reference.forward(&input, &h, &w, &shape, &shape);
    let diff = mlxcel_core::reshape(&mlxcel_core::subtract(&actual, &expected), &[-1]);
    let max = mlxcel_core::max_axis(&mlxcel_core::abs(&diff), 0, false);
    mlxcel_core::eval(&max);
    assert!(mlxcel_core::item_f32(&max) <= 1e-6);
}
