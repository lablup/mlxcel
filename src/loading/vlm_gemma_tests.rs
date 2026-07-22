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

use super::{
    gemma3n_language_model_prefix, gemma3n_metadata, gemma3n_needs_conv_transpose,
    sanitize_gemma3n_weights, sanitize_gemma4_audio_weights,
};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

#[test]
fn gemma3n_metadata_applies_defaults_and_overrides() {
    let defaults = gemma3n_metadata(&json!({}));
    assert_eq!(defaults.vision_hidden_size, 2048);
    assert_eq!(defaults.image_size, 256);
    assert_eq!(defaults.image_token_id, 262_145);
    assert_eq!(defaults.boi_token_id, 255_999);
    assert_eq!(defaults.eoi_token_id, 262_144);
    assert!((defaults.vision_rms_eps - 1e-6).abs() < f32::EPSILON);
    assert_eq!(defaults.audio_token_id, 262_273);
    assert_eq!(defaults.boa_token_id, 256_000);
    assert_eq!(defaults.eoa_token_id, 262_272);
    assert_eq!(defaults.audio_soft_tokens_per_clip, 188);

    let overrides = gemma3n_metadata(&json!({
        "vision_config": {
            "hidden_size": 3072,
            "image_size": 384,
            "rms_norm_eps": 1e-5
        },
        "image_token_index": 9,
        "boi_token_id": 10,
        "eoi_token_id": 11
        ,"audio_token_id": 12,
        "boa_token_id": 13,
        "eoa_token_id": 14,
        "audio_soft_tokens_per_image": 15
    }));
    assert_eq!(overrides.vision_hidden_size, 3072);
    assert_eq!(overrides.image_size, 384);
    assert_eq!(overrides.image_token_id, 9);
    assert_eq!(overrides.boi_token_id, 10);
    assert_eq!(overrides.eoi_token_id, 11);
    assert_eq!(overrides.audio_token_id, 12);
    assert_eq!(overrides.boa_token_id, 13);
    assert_eq!(overrides.eoa_token_id, 14);
    assert_eq!(overrides.audio_soft_tokens_per_clip, 15);
    assert!((overrides.vision_rms_eps - 1e-5).abs() < f32::EPSILON);
}

#[test]
fn gemma3n_language_model_prefix_prefers_quantized_prefix_when_present() {
    let mut quantized = WeightMap::new();
    quantized.insert(
        "language_model.model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    assert_eq!(
        gemma3n_language_model_prefix(&quantized),
        "language_model.model"
    );

    let mut dense = WeightMap::new();
    dense.insert(
        "language_model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    assert_eq!(gemma3n_language_model_prefix(&dense), "language_model");
}

#[test]
fn gemma3n_needs_conv_transpose_checks_reference_conv_shape() {
    let mut raw_weights = WeightMap::new();
    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 16, 3, 3], dtype::FLOAT32),
    );
    assert!(gemma3n_needs_conv_transpose(&raw_weights));

    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 3, 3, 16], dtype::FLOAT32),
    );
    assert!(!gemma3n_needs_conv_transpose(&raw_weights));
}

#[test]
fn sanitize_gemma3n_weights_strips_model_prefix_and_transposes_conv_weights() {
    let mut raw_weights = WeightMap::new();
    raw_weights.insert(
        "model.vision_tower.timm_model.blocks.0.0.conv_exp.weight".to_string(),
        mlxcel_core::ones(&[8, 16, 3, 3], dtype::FLOAT32),
    );
    raw_weights.insert(
        "model.language_model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    raw_weights.insert(
        "model.audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight".to_string(),
        mlxcel_core::ones(&[1536, 1, 5], dtype::FLOAT32),
    );

    let sanitized = sanitize_gemma3n_weights(raw_weights);

    assert!(sanitized.contains_key("vision_tower.timm_model.blocks.0.0.conv_exp.weight"));
    assert!(sanitized.contains_key("language_model.embed_tokens.weight"));
    assert_eq!(
        mlxcel_core::array_shape(
            sanitized
                .get("vision_tower.timm_model.blocks.0.0.conv_exp.weight")
                .unwrap()
        ),
        vec![8, 3, 3, 16]
    );
    assert_eq!(
        mlxcel_core::array_shape(
            sanitized
                .get("audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight")
                .unwrap()
        ),
        vec![1536, 5, 1]
    );
}

fn audio_conv_shape(weights: &WeightMap, key: &str) -> Vec<i32> {
    mlxcel_core::array_shape(weights.get(key).unwrap())
}

const LAYER0_CONV: &str = "audio_tower.subsample_conv_projection.layer0.conv.weight";
const LAYER1_CONV: &str = "audio_tower.subsample_conv_projection.layer1.conv.weight";
const LCONV1D: &str = "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight";

#[test]
fn sanitize_gemma4_audio_weights_preserves_channel_last_checkpoint() {
    // The mlx-community/gemma-4-e4b-it-qat-4bit checkpoint stores these conv
    // weights already in MLX channel-last layout. The sanitizer must leave them
    // untouched so the audio subsample conv receives a C_in=1 weight matching
    // the C_in=1 input (regression for issue #428).
    let mut weights = WeightMap::new();
    weights.insert(
        LAYER0_CONV.to_string(),
        mlxcel_core::ones(&[128, 3, 3, 1], dtype::FLOAT32),
    );
    weights.insert(
        LAYER1_CONV.to_string(),
        mlxcel_core::ones(&[32, 3, 3, 128], dtype::FLOAT32),
    );
    weights.insert(
        LCONV1D.to_string(),
        mlxcel_core::ones(&[1024, 5, 1], dtype::FLOAT32),
    );

    sanitize_gemma4_audio_weights(&mut weights);

    assert_eq!(audio_conv_shape(&weights, LAYER0_CONV), vec![128, 3, 3, 1]);
    assert_eq!(audio_conv_shape(&weights, LAYER1_CONV), vec![32, 3, 3, 128]);
    assert_eq!(audio_conv_shape(&weights, LCONV1D), vec![1024, 5, 1]);
}

#[test]
fn sanitize_gemma4_audio_weights_transposes_pytorch_layout() {
    // Synthetic PyTorch-layout weights: conv2d [out, in, kH, kW] and depthwise
    // conv1d [out, in=1, kW]. Both must be transposed to channel-last.
    let mut weights = WeightMap::new();
    weights.insert(
        LAYER0_CONV.to_string(),
        mlxcel_core::ones(&[128, 1, 3, 3], dtype::FLOAT32),
    );
    weights.insert(
        LCONV1D.to_string(),
        mlxcel_core::ones(&[1024, 1, 5], dtype::FLOAT32),
    );

    sanitize_gemma4_audio_weights(&mut weights);

    // conv2d [128, 1, 3, 3] --transpose[0,2,3,1]--> [128, 3, 3, 1].
    assert_eq!(audio_conv_shape(&weights, LAYER0_CONV), vec![128, 3, 3, 1]);
    // depthwise conv1d [1024, 1, 5] --transpose[0,2,1]--> [1024, 5, 1].
    assert_eq!(audio_conv_shape(&weights, LCONV1D), vec![1024, 5, 1]);
}

#[test]
fn sanitize_gemma4_audio_weights_is_idempotent() {
    // Running the sanitizer twice must equal running it once: the first pass
    // converts the PyTorch-layout weights to channel-last, and the second pass
    // must detect that layout and leave them unchanged.
    let mut once = WeightMap::new();
    once.insert(
        LAYER0_CONV.to_string(),
        mlxcel_core::ones(&[128, 1, 3, 3], dtype::FLOAT32),
    );
    once.insert(
        LCONV1D.to_string(),
        mlxcel_core::ones(&[1024, 1, 5], dtype::FLOAT32),
    );
    sanitize_gemma4_audio_weights(&mut once);

    let mut twice = WeightMap::new();
    twice.insert(
        LAYER0_CONV.to_string(),
        mlxcel_core::ones(&[128, 1, 3, 3], dtype::FLOAT32),
    );
    twice.insert(
        LCONV1D.to_string(),
        mlxcel_core::ones(&[1024, 1, 5], dtype::FLOAT32),
    );
    sanitize_gemma4_audio_weights(&mut twice);
    sanitize_gemma4_audio_weights(&mut twice);

    assert_eq!(
        audio_conv_shape(&once, LAYER0_CONV),
        audio_conv_shape(&twice, LAYER0_CONV)
    );
    assert_eq!(
        audio_conv_shape(&once, LCONV1D),
        audio_conv_shape(&twice, LCONV1D)
    );
    // And the idempotent result is the channel-last layout.
    assert_eq!(audio_conv_shape(&twice, LAYER0_CONV), vec![128, 3, 3, 1]);
    assert_eq!(audio_conv_shape(&twice, LCONV1D), vec![1024, 5, 1]);
}
