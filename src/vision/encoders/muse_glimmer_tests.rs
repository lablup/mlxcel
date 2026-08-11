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
use crate::models::muse_glimmer::MuseGlimmerVisionConfig;
use crate::vision::encoders::muse_glimmer_layout::window_index_plan;
use mlxcel_core::MlxArray;
use mlxcel_core::weights::WeightMap;

fn put(weights: &mut WeightMap, key: &str, data: Vec<f32>, shape: &[i32]) {
    weights.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

fn put_fill(weights: &mut WeightMap, key: &str, shape: &[i32], value: f32) {
    let len = shape.iter().product::<i32>() as usize;
    put(weights, key, vec![value; len], shape);
}

fn put_linear(weights: &mut WeightMap, prefix: &str, out: usize, input: usize, bias: bool) {
    put_fill(
        weights,
        &format!("{prefix}.weight"),
        &[out as i32, input as i32],
        0.0,
    );
    if bias {
        put_fill(weights, &format!("{prefix}.bias"), &[out as i32], 0.0);
    }
}

fn put_identity_patch(weights: &mut WeightMap, prefix: &str, hidden: usize, input: usize) {
    let mut data = vec![0.0; hidden * input];
    for idx in 0..hidden.min(input) {
        data[idx * input + idx] = 1.0;
    }
    put(
        weights,
        &format!("{prefix}.weight"),
        data,
        &[hidden as i32, input as i32],
    );
}

#[allow(clippy::field_reassign_with_default)]
fn tiny_config() -> MuseGlimmerVisionConfig {
    let mut cfg = MuseGlimmerVisionConfig::default();
    cfg.hidden_size = 8;
    cfg.intermediate_size = 16;
    cfg.num_hidden_layers = 4;
    cfg.num_attention_heads = 1;
    cfg.patch_size = 1;
    cfg.patch_temporal = 2;
    cfg.merge_size = 2;
    cfg.pos_emb_height = 2;
    cfg.pos_emb_width = 2;
    cfg.max_position_embeddings = 4;
    cfg.layer_types = vec![
        "window_attention".to_string(),
        "window_attention".to_string(),
        "window_attention".to_string(),
        "full_attention".to_string(),
    ];
    cfg
}

fn tiny_tower_weights(config: &MuseGlimmerVisionConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    let root = MUSE_GLIMMER_VISION_TOWER_ROOT;
    let hidden = config.hidden_size;
    let input = patch_input_size(config);
    put_identity_patch(
        &mut weights,
        &format!("{root}.patch_embedder.patch_embedding"),
        hidden,
        input,
    );
    put_fill(
        &mut weights,
        &format!("{root}.patch_embedder.position_embedding_table.weight"),
        &[
            (config.pos_emb_height * config.pos_emb_width) as i32,
            hidden as i32,
        ],
        0.0,
    );
    for norm in ["ln_pre", "ln_post"] {
        put_fill(
            &mut weights,
            &format!("{root}.{norm}.weight"),
            &[hidden as i32],
            1.0,
        );
        put_fill(
            &mut weights,
            &format!("{root}.{norm}.bias"),
            &[hidden as i32],
            0.0,
        );
    }
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("{root}.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            put_fill(
                &mut weights,
                &format!("{prefix}.{norm}.weight"),
                &[hidden as i32],
                1.0,
            );
            put_fill(
                &mut weights,
                &format!("{prefix}.{norm}.bias"),
                &[hidden as i32],
                0.0,
            );
        }
        for proj in ["q_proj", "k_proj", "v_proj", "proj"] {
            put_linear(
                &mut weights,
                &format!("{prefix}.attn.{proj}"),
                hidden,
                hidden,
                true,
            );
        }
        put_linear(
            &mut weights,
            &format!("{prefix}.mlp.fc1"),
            config.intermediate_size,
            hidden,
            true,
        );
        put_linear(
            &mut weights,
            &format!("{prefix}.mlp.fc2"),
            hidden,
            config.intermediate_size,
            true,
        );
    }
    weights
}

fn patterned_pixels(tokens: usize, input: usize) -> Vec<f32> {
    (0..tokens * input)
        .map(|idx| ((idx * 13 % 23) as f32 - 11.0) / 7.0)
        .collect()
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn assert_vec_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-4,
            "idx={idx}, actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn patch_embedder_loads_exact_default_projection_shape_and_position_add() {
    let config = MuseGlimmerVisionConfig::default();
    let root = MUSE_GLIMMER_VISION_TOWER_ROOT;
    let mut weights = WeightMap::new();
    put_fill(
        &mut weights,
        &format!("{root}.patch_embedder.patch_embedding.weight"),
        &[1536, 2 * 3 * 14 * 14],
        0.0,
    );
    put_fill(
        &mut weights,
        &format!("{root}.patch_embedder.position_embedding_table.weight"),
        &[32 * 32, 1536],
        1.0,
    );

    let embedder = MuseGlimmerPatchEmbedder::from_weights(&weights, &config, root).unwrap();
    assert_eq!(embedder.patch_input_size(), 2 * 3 * 14 * 14);
    let pixels = mlxcel_core::from_slice_f32(
        &vec![0.0; embedder.patch_input_size()],
        &[1, 2 * 3 * 14 * 14],
    );
    let out = embedder.forward(&pixels, &[(1, 1, 1)]).unwrap();
    let values = to_vec_f32(&out);

    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 1536]);
    assert!(values.iter().take(16).all(|v| (*v - 1.0).abs() < 1e-5));
}

#[test]
fn tower_records_three_window_one_full_schedule_and_rejects_mutation() {
    let config = tiny_config();
    let weights = tiny_tower_weights(&config);
    let tower = MuseGlimmerVisionTower::from_weights(&weights, &config).unwrap();
    assert_eq!(
        tower.layer_kinds(),
        vec![
            "window_attention",
            "window_attention",
            "window_attention",
            "full_attention"
        ]
    );

    let mut mutated = config.clone();
    mutated.layer_types[3] = "window_attention".to_string();
    let err = match MuseGlimmerVisionTower::from_weights(&weights, &mutated) {
        Ok(_) => panic!("mutated Muse Glimmer vision schedule must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("must be \"full_attention\""));
}

#[test]
fn tower_restores_window_order_after_layer_boundary() {
    let config = tiny_config();
    let weights = tiny_tower_weights(&config);
    let tower = MuseGlimmerVisionTower::from_weights(&weights, &config).unwrap();
    let grid = [(1, 2, 4)];
    let plan = window_index_plan(&grid, config.pos_emb_height as i32).unwrap();
    assert_ne!(plan.indices, (0..8).collect::<Vec<_>>());

    let pixels =
        mlxcel_core::from_slice_f32(&patterned_pixels(8, patch_input_size(&config)), &[8, 6]);
    let ordered = tower
        .forward_window_ordered_for_tests(&pixels, &grid)
        .unwrap();
    let restored = tower.forward(&pixels, &grid).unwrap();
    let inverse =
        mlxcel_core::from_slice_i32(&plan.inverse_indices, &[plan.inverse_indices.len() as i32]);
    let expected = mlxcel_core::take(&ordered, &inverse, 0);

    let ordered_values = to_vec_f32(&ordered);
    let restored_values = to_vec_f32(&restored);
    assert_ne!(ordered_values, restored_values);
    assert_vec_close(&restored_values, &to_vec_f32(&expected));
}

#[test]
fn reduced_tower_forward_is_finite() {
    let config = tiny_config();
    let weights = tiny_tower_weights(&config);
    let tower = MuseGlimmerVisionTower::from_weights(&weights, &config).unwrap();
    let pixels =
        mlxcel_core::from_slice_f32(&patterned_pixels(8, patch_input_size(&config)), &[8, 6]);
    let out = tower.forward(&pixels, &[(1, 2, 4)]).unwrap();
    let values = to_vec_f32(&out);

    assert_eq!(mlxcel_core::array_shape(&out), vec![8, 8]);
    assert!(values.iter().all(|v| v.is_finite()));
}
