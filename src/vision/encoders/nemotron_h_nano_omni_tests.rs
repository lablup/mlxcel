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

//! Tests for the Nemotron H Nano Omni vision tower.
//!
//! These tests build the encoder from synthetic, deterministic weights
//! and confirm:
//! - the patch generator output shape matches the upstream contract
//!   (channels-first input → `[B, num_skip + num_patches, embed_dim]`),
//! - the `RadioOutput { summary, features }` split matches the
//!   `num_cls_tokens` / `num_skip` derived from the config,
//! - smaller-than-stored grids slice the position-embed table without
//!   panicking,
//! - the loader rejects checkpoints that are missing required weights.

use super::*;
use mlxcel_core::weights::WeightMap;

const PATCH_SIZE: usize = 16;
const EMBED_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const INTERMEDIATE_SIZE: usize = 64;
const MAX_RES: usize = 64;
const IMAGE_SIZE: usize = 32;

fn small_config() -> NemotronHNanoOmniVisionConfig {
    NemotronHNanoOmniVisionConfig {
        args: None,
        hidden_size: EMBED_DIM,
        num_hidden_layers: 2,
        num_attention_heads: NUM_HEADS,
        intermediate_size: INTERMEDIATE_SIZE,
        image_size: IMAGE_SIZE,
        patch_size: PATCH_SIZE,
        max_resolution: MAX_RES,
        video_temporal_patch_size: 2,
    }
}

fn linear_weight(out_dim: usize, in_dim: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let total = out_dim * in_dim;
    let data: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 1e-3).collect();
    mlxcel_core::from_slice_f32(&data, &[out_dim as i32, in_dim as i32])
}

fn linear_bias(dim: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 1e-4).collect();
    mlxcel_core::from_slice_f32(&data, &[dim as i32])
}

fn weight_3d(d0: usize, d1: usize, d2: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let total = d0 * d1 * d2;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 1e-4).collect();
    mlxcel_core::from_slice_f32(&data, &[d0 as i32, d1 as i32, d2 as i32])
}

fn weight_2d(d0: usize, d1: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let total = d0 * d1;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 1e-4).collect();
    mlxcel_core::from_slice_f32(&data, &[d0 as i32, d1 as i32])
}

fn ones_1d(n: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::ones(&[n as i32], mlxcel_core::dtype::FLOAT32)
}

fn zeros_1d(n: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::zeros(&[n as i32], mlxcel_core::dtype::FLOAT32)
}

fn build_synthetic_weights(prefix: &str, config: &NemotronHNanoOmniVisionConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    let num_rows = config.cpe_max_size() / config.patch_size;
    let num_cols = num_rows;
    let num_patches = num_rows * num_cols;

    // Input conditioner — `[3, 1, 1]` shapes mirror upstream
    // `mx.zeros((3, 1, 1))` / `mx.ones((3, 1, 1))` so the broadcast
    // against a `[B, 3, H, W]` pixel tensor works without surprises.
    weights.insert(
        format!("{prefix}.input_conditioner.norm_mean"),
        mlxcel_core::zeros(&[3, 1, 1], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        format!("{prefix}.input_conditioner.norm_std"),
        mlxcel_core::ones(&[3, 1, 1], mlxcel_core::dtype::FLOAT32),
    );

    // Patch generator.
    let pg_prefix = format!("{prefix}.model.patch_generator");
    weights.insert(
        format!("{pg_prefix}.cls_token.token"),
        weight_2d(1, EMBED_DIM),
    );
    weights.insert(
        format!("{pg_prefix}.embedder.weight"),
        linear_weight(EMBED_DIM, 3 * PATCH_SIZE * PATCH_SIZE),
    );
    // Provide the optional video_embedder weight so video probing in
    // the loader picks it up.
    weights.insert(
        format!("{pg_prefix}.video_embedder.weight"),
        linear_weight(EMBED_DIM, 2 * 3 * PATCH_SIZE * PATCH_SIZE),
    );
    weights.insert(
        format!("{pg_prefix}.pos_embed"),
        weight_3d(1, num_patches, EMBED_DIM),
    );

    // Two transformer blocks.
    for layer_idx in 0..config.num_hidden_layers {
        let bp = format!("{prefix}.model.blocks.{layer_idx}");
        weights.insert(format!("{bp}.norm1.weight"), ones_1d(EMBED_DIM));
        weights.insert(format!("{bp}.norm1.bias"), zeros_1d(EMBED_DIM));
        weights.insert(format!("{bp}.norm2.weight"), ones_1d(EMBED_DIM));
        weights.insert(format!("{bp}.norm2.bias"), zeros_1d(EMBED_DIM));
        weights.insert(
            format!("{bp}.attn.qkv.weight"),
            linear_weight(EMBED_DIM * 3, EMBED_DIM),
        );
        weights.insert(format!("{bp}.attn.qkv.bias"), linear_bias(EMBED_DIM * 3));
        weights.insert(
            format!("{bp}.attn.proj.weight"),
            linear_weight(EMBED_DIM, EMBED_DIM),
        );
        weights.insert(format!("{bp}.attn.proj.bias"), linear_bias(EMBED_DIM));
        weights.insert(
            format!("{bp}.mlp.fc1.weight"),
            linear_weight(INTERMEDIATE_SIZE, EMBED_DIM),
        );
        weights.insert(format!("{bp}.mlp.fc1.bias"), linear_bias(INTERMEDIATE_SIZE));
        weights.insert(
            format!("{bp}.mlp.fc2.weight"),
            linear_weight(EMBED_DIM, INTERMEDIATE_SIZE),
        );
        weights.insert(format!("{bp}.mlp.fc2.bias"), linear_bias(EMBED_DIM));
    }
    weights
}

#[test]
fn config_derives_skip_and_cls_counts() {
    let mut config = small_config();
    let derived_cls = config.num_cls_tokens();
    let derived_skip = config.num_skip();
    // Default: 1 cls token, 0 register tokens.
    assert_eq!(derived_cls, 1);
    assert_eq!(derived_skip, 1);

    // Three teacher names with `register_multiple = 8` should pad up to 8.
    // For (n_tokens=3, multiple=8) the upstream `register_multiple - (n_tokens % multiple)` = 8 - 3 = 5.
    config.args = Some(NemotronHNanoOmniVisionArgs {
        cpe_max_size: None,
        cls_token_per_teacher: true,
        register_multiple: Some(8),
        num_distinct_teachers: Some(3),
    });
    assert_eq!(config.num_cls_tokens(), 3);
    assert_eq!(config.num_registers(), 5);
    assert_eq!(config.num_skip(), 3 + 5);
}

#[test]
fn vision_tower_loads_synthetic_weights_and_produces_radio_output() {
    let config = small_config();
    let weights = build_synthetic_weights("vision_model.radio_model", &config);
    let model = NemotronHNanoOmniVisionModel::from_weights(
        &weights,
        "vision_model.radio_model",
        &config,
        64,
        4,
    )
    .expect("build vision tower");

    // Synthetic input: batch=1, channels=3, image_size square.
    let image_size = config.image_size as i32;
    let pixel_values =
        mlxcel_core::ones(&[1, 3, image_size, image_size], mlxcel_core::dtype::FLOAT32);

    let output = model.forward(&pixel_values, false);
    let summary_shape = mlxcel_core::array_shape(&output.summary);
    let features_shape = mlxcel_core::array_shape(&output.features);

    let num_cls = config.num_cls_tokens() as i32;
    let patch_h = image_size / config.patch_size as i32;
    let total_patches = patch_h * patch_h;

    assert_eq!(summary_shape, vec![1, num_cls * EMBED_DIM as i32]);
    assert_eq!(features_shape, vec![1, total_patches, EMBED_DIM as i32]);
}

#[test]
fn vision_tower_features_count_matches_patch_grid() {
    let config = small_config();
    let weights = build_synthetic_weights("vision_model.radio_model", &config);
    let model = NemotronHNanoOmniVisionModel::from_weights(
        &weights,
        "vision_model.radio_model",
        &config,
        64,
        4,
    )
    .unwrap();

    let pixel_values = mlxcel_core::ones(
        &[1, 3, config.image_size as i32, config.image_size as i32],
        mlxcel_core::dtype::FLOAT32,
    );
    let output = model.forward(&pixel_values, false);
    let features_shape = mlxcel_core::array_shape(&output.features);
    let patch_h = (config.image_size / config.patch_size) as i32;
    let total_patches = patch_h * patch_h;
    assert_eq!(features_shape[1], total_patches);
    assert_eq!(features_shape[2], EMBED_DIM as i32);
}

#[test]
fn missing_weight_is_a_loader_error_not_a_panic() {
    let config = small_config();
    let mut weights = build_synthetic_weights("vision_model.radio_model", &config);
    weights.remove("vision_model.radio_model.model.patch_generator.embedder.weight");
    let result = NemotronHNanoOmniVisionModel::from_weights(
        &weights,
        "vision_model.radio_model",
        &config,
        64,
        4,
    );
    assert!(result.is_err());
}
