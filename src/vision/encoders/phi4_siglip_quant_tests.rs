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
        .map(|index| (index as f32 * 0.029).cos())
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
    packed_map.insert(
        "emb.patch_embedding.weight".into(),
        mlxcel_core::zeros(&[64, 64], mlxcel_core::dtype::FLOAT32),
    );
    packed_map.insert("emb.position_embedding.weight".into(), packed);
    packed_map.insert("emb.position_embedding.scales".into(), scales);
    packed_map.insert("emb.position_embedding.biases".into(), biases);
    let mut dense_map = WeightMap::new();
    dense_map.insert(
        "emb.patch_embedding.weight".into(),
        mlxcel_core::zeros(&[64, 64], mlxcel_core::dtype::FLOAT32),
    );
    dense_map.insert("emb.position_embedding.weight".into(), dequantized);
    (packed_map, dense_map)
}

#[test]
fn quantized_phi4_position_resize_matches_explicit_dequantize() {
    let (packed, dense) = maps();
    let config = Phi4SigLipVisionConfig {
        hidden_size: 64,
        intermediate_size: 64,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_channels: 3,
        image_size: 2,
        patch_size: 1,
        num_patches: 4,
        layer_norm_eps: 1e-6,
    };
    let quantized = Phi4SigLipVisionEmbeddings::from_weights(&packed, "emb", &config, 64, 4)
        .expect("quantized Phi4 position table");
    assert_eq!(
        mlxcel_core::array_shape(&quantized.position_embedding.dequantized_weight()),
        [4, 64]
    );
    let reference = Phi4SigLipVisionEmbeddings::from_weights(&dense, "emb", &config, 64, 4)
        .expect("explicitly dequantized Phi4 position table");
    let actual = quantized.resize_positional_embeddings((3, 2), 6);
    let expected = reference.resize_positional_embeddings((3, 2), 6);
    let diff = mlxcel_core::reshape(&mlxcel_core::subtract(&actual, &expected), &[-1]);
    let max = mlxcel_core::max_axis(&mlxcel_core::abs(&diff), 0, false);
    mlxcel_core::eval(&max);
    assert!(mlxcel_core::item_f32(&max) <= 1e-6);
}

#[test]
fn out_of_range_phi4_position_config_is_rejected_before_grid_arithmetic() {
    let (weights, _) = maps();
    let config = Phi4SigLipVisionConfig {
        hidden_size: 64,
        intermediate_size: 64,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_channels: 3,
        image_size: 2,
        patch_size: 1,
        num_patches: usize::MAX,
        layer_norm_eps: 1e-6,
    };
    let error = Phi4SigLipVisionEmbeddings::from_weights(&weights, "emb", &config, 64, 4)
        .err()
        .expect("out-of-range position count must fail");
    assert!(error.contains("exceeds the supported i32 range"), "{error}");
}
