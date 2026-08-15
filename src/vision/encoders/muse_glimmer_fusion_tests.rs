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

use super::*;
use crate::models::muse_glimmer::{
    MuseGlimmerConfig, MuseGlimmerTextConfig, MuseGlimmerVisionConfig,
};
use mlxcel_core::MlxArray;
use mlxcel_core::weights::WeightMap;

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-4,
        "actual={actual}, expected={expected}"
    );
}

fn put(weights: &mut WeightMap, key: &str, data: Vec<f32>, shape: &[i32]) {
    weights.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

fn put_linear(weights: &mut WeightMap, prefix: &str, out: usize, input: usize, seed: f32) {
    let data = (0..out * input)
        .map(|idx| seed + (idx % 7) as f32 * 0.01)
        .collect::<Vec<_>>();
    put(
        weights,
        &format!("{prefix}.weight"),
        data,
        &[out as i32, input as i32],
    );
}

#[allow(clippy::field_reassign_with_default)]
fn tiny_config() -> MuseGlimmerConfig {
    let mut vision_config = MuseGlimmerVisionConfig::default();
    vision_config.hidden_size = 2;
    vision_config.intermediate_size = 8;
    vision_config.num_attention_heads = 1;
    vision_config.patch_size = 1;
    vision_config.patch_temporal = 2;
    vision_config.merge_size = 2;
    vision_config.pos_emb_height = 2;
    vision_config.pos_emb_width = 2;
    vision_config.num_hidden_layers = 4;
    vision_config.layer_types = vec![
        "window_attention".to_string(),
        "window_attention".to_string(),
        "window_attention".to_string(),
        "full_attention".to_string(),
    ];

    MuseGlimmerConfig {
        text_config: MuseGlimmerTextConfig {
            model_type: "muse_glimmer_text".to_string(),
            hidden_size: 6,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            post_norm_eps: 1e-8,
            vocab_size: 32,
            tie_word_embeddings: false,
            layer_types: vec!["full_attention".to_string()],
            sliding_window: 8,
            qk_scale_factor: 1.0,
            output_multiplier: 1.0,
            final_logit_softcapping: None,
            layer_rope_theta: vec![None],
            rope_parameters: None,
            quantization: None,
        },
        vision_config,
        image_token_id: None,
        video_token_id: None,
        out_hidden_size: 8,
        projector_hidden_size: 4,
        projector_hidden_act: "gelu".to_string(),
    }
}

fn tiny_fusion_weights(config: &MuseGlimmerConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    put_linear(
        &mut weights,
        &format!("{MUSE_GLIMMER_VISION_ADAPTER_ROOT}.fc1"),
        config.projector_hidden_size,
        config.out_hidden_size,
        0.03,
    );
    put_linear(
        &mut weights,
        &format!("{MUSE_GLIMMER_VISION_ADAPTER_ROOT}.fc2"),
        config.projector_hidden_size,
        config.projector_hidden_size,
        0.02,
    );
    put_linear(
        &mut weights,
        MUSE_GLIMMER_VISION_PROJECTION_ROOT,
        config.text_config.hidden_size,
        config.projector_hidden_size,
        0.04,
    );
    weights
}

#[test]
fn pixel_shuffle_uses_reference_channel_major_order() {
    let input =
        mlxcel_core::from_slice_f32(&[0.0, 100.0, 1.0, 101.0, 2.0, 102.0, 3.0, 103.0], &[4, 2]);
    let out = pixel_shuffle_2x2(&input, &[(1, 2, 2)], 2, 2).unwrap();

    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 8]);
    assert_eq!(
        to_vec_f32(&out),
        vec![0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0]
    );
}

#[test]
fn pixel_shuffle_preserves_ordered_multi_image_boundaries() {
    let input =
        mlxcel_core::from_slice_f32(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[12, 1]);
    let out = pixel_shuffle_2x2(&input, &[(1, 2, 2), (1, 2, 4)], 2, 1).unwrap();

    assert_eq!(mlxcel_core::array_shape(&out), vec![3, 4]);
    assert_eq!(
        to_vec_f32(&out),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 6.0, 7.0, 10.0, 11.0]
    );
}

#[test]
fn pixel_shuffle_rejects_bad_grid_and_feature_rows() {
    let input = mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[4, 1]);
    let err = match pixel_shuffle_2x2(&input, &[(1, 3, 2)], 2, 1) {
        Ok(_) => panic!("non-divisible Muse Glimmer grid must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("divisible by 2"));

    let err = match pixel_shuffle_2x2(&input, &[(1, 2, 4)], 2, 1) {
        Ok(_) => panic!("Muse Glimmer feature-row mismatch must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("do not match image_grid_thw"));
}

#[test]
fn fusion_adapter_projection_output_shape_is_finite() {
    let config = tiny_config();
    let weights = tiny_fusion_weights(&config);
    let fusion = MuseGlimmerVisionFusion::from_weights(&weights, &config).unwrap();
    let tower_features =
        mlxcel_core::from_slice_f32(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[4, 2]);
    let out = fusion.forward(&tower_features, &[(1, 2, 2)]).unwrap();
    let values = to_vec_f32(&out);

    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 6]);
    assert!(values.iter().all(|v| v.is_finite()));
}

#[test]
fn weightless_perception_norm_matches_rms_without_scale() {
    let input = mlxcel_core::from_slice_f32(&[3.0, 4.0], &[1, 2]);
    let out = weightless_perception_norm(&input, 1e-6);
    let values = to_vec_f32(&out);
    let denom = (12.5f32 + 1e-6).sqrt();

    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 2]);
    assert_close(values[0], 3.0 / denom);
    assert_close(values[1], 4.0 / denom);
}
