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
    let values: Vec<f32> = (0..2 * 2 * 64)
        .map(|index| (index as f32 * 0.019).sin())
        .collect();
    let dense = mlxcel_core::from_slice_f32(&values, &[2, 2, 64]);
    let flat = mlxcel_core::reshape(&dense, &[4, 64]);
    let quantized = mlxcel_core::quantize_weights(&flat, 64, 4);
    let packed = mlxcel_core::reshape(&mlxcel_core::quantized_weights_w(&quantized), &[2, 2, 8]);
    let scales = mlxcel_core::reshape(
        &mlxcel_core::quantized_weights_scales(&quantized),
        &[2, 2, 1],
    );
    let biases = mlxcel_core::reshape(
        &mlxcel_core::quantized_weights_biases(&quantized),
        &[2, 2, 1],
    );
    let packed_flat = mlxcel_core::reshape(&packed, &[4, 8]);
    let scales_flat = mlxcel_core::reshape(&scales, &[4, 1]);
    let biases_flat = mlxcel_core::reshape(&biases, &[4, 1]);
    let flat_reference = unsafe {
        mlxcel_core::dequantize(
            &packed_flat,
            &scales_flat,
            biases_flat.as_ref().expect("affine biases") as *const _,
            64,
            4,
            "affine",
        )
    };
    let dequantized = mlxcel_core::reshape(&flat_reference, &[2, 2, 64]);
    let base = || {
        let mut weights = WeightMap::new();
        weights.insert(
            "vision.patch_ln1.weight".into(),
            mlxcel_core::ones(&[3], mlxcel_core::dtype::FLOAT32),
        );
        weights.insert(
            "vision.patch_dense.weight".into(),
            mlxcel_core::zeros(&[64, 3], mlxcel_core::dtype::FLOAT32),
        );
        weights.insert(
            "vision.patch_ln2.weight".into(),
            mlxcel_core::ones(&[64], mlxcel_core::dtype::FLOAT32),
        );
        weights.insert(
            "vision.pos_norm.weight".into(),
            mlxcel_core::ones(&[64], mlxcel_core::dtype::FLOAT32),
        );
        weights
    };
    let mut packed_map = base();
    packed_map.insert("vision.pos_embedding".into(), packed);
    packed_map.insert("vision.pos_embedding.scales".into(), scales);
    packed_map.insert("vision.pos_embedding.biases".into(), biases);
    let mut dense_map = base();
    dense_map.insert("vision.pos_embedding".into(), dequantized);
    (packed_map, dense_map)
}

#[test]
fn quantized_whole_array_position_add_matches_explicit_dequantize() {
    let (packed, dense) = maps();
    let config = Gemma4UnifiedVisionConfig {
        patch_size: 1,
        pooling_kernel_size: 1,
        model_patch_size: 1,
        mm_embed_dim: 64,
        mm_posemb_size: 2,
        num_soft_tokens: 4,
        output_proj_dims: 64,
        rms_norm_eps: 1e-6,
    };
    let quantized = Gemma4UnifiedVisionEmbedder::from_weights(&packed, "vision", &config, 64, 4)
        .expect("quantized position table");
    assert_eq!(
        mlxcel_core::array_shape(&quantized.pos_embedding),
        [2, 2, 64]
    );
    let reference = Gemma4UnifiedVisionEmbedder::from_weights(&dense, "vision", &config, 64, 4)
        .expect("dequantized position table");
    let patches = mlxcel_core::zeros(&[4, 3], mlxcel_core::dtype::FLOAT32);
    let positions = mlxcel_core::from_slice_i32(&[0, 0, 1, 0, 0, 1, -1, -1], &[4, 2]);
    let actual = quantized.forward(&patches, &positions);
    let expected = reference.forward(&patches, &positions);
    let diff = mlxcel_core::reshape(&mlxcel_core::subtract(&actual, &expected), &[-1]);
    let max = mlxcel_core::max_axis(&mlxcel_core::abs(&diff), 0, false);
    mlxcel_core::eval(&max);
    assert!(mlxcel_core::item_f32(&max) <= 1e-5);
}
