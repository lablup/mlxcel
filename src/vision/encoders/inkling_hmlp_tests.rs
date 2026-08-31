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

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;

use super::*;

fn reference_config() -> InklingVisionConfig {
    InklingVisionConfig {
        model_type: "inkling_vision".into(),
        patch_size: 40,
        temporal_patch_size: 2,
        num_channels: 3,
        n_layers: 4,
        text_hidden_size: 6144,
        rms_norm_eps: 1e-6,
    }
}

#[test]
fn reference_scale_and_layer_plans_match_upstream() {
    let config = reference_config();
    assert_eq!(
        plan_out_scales(&config).unwrap(),
        vec![
            [1, 1, 1, 3],
            [1, 5, 5, 128],
            [1, 10, 10, 320],
            [1, 40, 40, 4800],
            [2, 40, 40, 9600],
        ]
    );
    assert_eq!(
        layer_plan(&config).unwrap(),
        vec![
            InklingHmlpLayerPlan {
                input_dim: 75,
                output_dim: 128,
                t_fold: 1,
                hw_fold: 5,
                add_norm: true,
            },
            InklingHmlpLayerPlan {
                input_dim: 512,
                output_dim: 320,
                t_fold: 1,
                hw_fold: 2,
                add_norm: true,
            },
            InklingHmlpLayerPlan {
                input_dim: 5120,
                output_dim: 4800,
                t_fold: 1,
                hw_fold: 4,
                add_norm: true,
            },
            InklingHmlpLayerPlan {
                input_dim: 9600,
                output_dim: 6144,
                t_fold: 2,
                hw_fold: 1,
                add_norm: false,
            },
        ]
    );
}

#[test]
fn fold_preserves_reference_timespace_channel_order() {
    let input = mlxcel_core::from_slice_f32(
        &(0..32).map(|value| value as f32).collect::<Vec<_>>(),
        &[1, 2, 4, 4, 1],
    );
    let folded = fold_timespace_to_depth(&input, 1, 2).unwrap();
    mlxcel_core::eval(&folded);
    assert_eq!(mlxcel_core::array_shape(&folded), vec![1, 2, 2, 2, 4]);
    assert_eq!(
        array_to_vec_f32(&folded),
        vec![
            0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0, 15.0,
            16.0, 17.0, 20.0, 21.0, 18.0, 19.0, 22.0, 23.0, 24.0, 25.0, 28.0, 29.0, 26.0, 27.0,
            30.0, 31.0,
        ]
    );
}

#[test]
fn one_layer_tower_accepts_reference_tiles_and_emits_text_width() {
    let mut config = reference_config();
    config.n_layers = 1;
    config.text_hidden_size = 8;
    let mut weights = WeightMap::new();
    weights.insert(
        "vision_tower.encoder_layers.0.projection.weight".into(),
        mlxcel_core::zeros(&[8, 9600], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        "vision_tower.final_norm.weight".into(),
        mlxcel_core::ones(&[8], mlxcel_core::dtype::FLOAT32),
    );
    let tower = InklingHmlpEncoder::from_weights(&weights, &config, 64, 4).unwrap();
    let tiles = mlxcel_core::zeros(&[3, 2, 40, 40, 3], mlxcel_core::dtype::FLOAT32);
    let output = tower.forward(&tiles).unwrap();
    mlxcel_core::eval(&output);
    assert_eq!(mlxcel_core::array_shape(&output), vec![3, 8]);
    assert_eq!(array_to_vec_f32(&output), vec![0.0; 24]);
}

#[test]
fn tower_rejects_non_reference_tile_geometry() {
    let mut config = reference_config();
    config.n_layers = 1;
    config.text_hidden_size = 8;
    let mut weights = WeightMap::new();
    weights.insert(
        "vision_tower.encoder_layers.0.projection.weight".into(),
        mlxcel_core::zeros(&[8, 9600], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        "vision_tower.final_norm.weight".into(),
        mlxcel_core::ones(&[8], mlxcel_core::dtype::FLOAT32),
    );
    let tower = InklingHmlpEncoder::from_weights(&weights, &config, 64, 4).unwrap();
    let invalid = mlxcel_core::zeros(&[1, 1, 40, 40, 3], mlxcel_core::dtype::FLOAT32);
    assert!(tower.forward(&invalid).is_err());
}

#[test]
fn tower_rejects_projection_shape_before_first_forward() {
    let mut config = reference_config();
    config.n_layers = 1;
    config.text_hidden_size = 8;
    let mut weights = WeightMap::new();
    weights.insert(
        "vision_tower.encoder_layers.0.projection.weight".into(),
        mlxcel_core::zeros(&[7, 9600], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        "vision_tower.final_norm.weight".into(),
        mlxcel_core::ones(&[8], mlxcel_core::dtype::FLOAT32),
    );
    let error = InklingHmlpEncoder::from_weights(&weights, &config, 64, 4)
        .err()
        .expect("malformed projection must be rejected");
    assert!(error.contains("expected [8, 9600]"), "{error}");
}
