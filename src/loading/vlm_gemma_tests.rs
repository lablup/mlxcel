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
    normalize_gemma4_unified_key, sanitize_gemma3n_weights, sanitize_gemma4_unified_weights,
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

    let overrides = gemma3n_metadata(&json!({
        "vision_config": {
            "hidden_size": 3072,
            "image_size": 384,
            "rms_norm_eps": 1e-5
        },
        "image_token_index": 9,
        "boi_token_id": 10,
        "eoi_token_id": 11
    }));
    assert_eq!(overrides.vision_hidden_size, 3072);
    assert_eq!(overrides.image_size, 384);
    assert_eq!(overrides.image_token_id, 9);
    assert_eq!(overrides.boi_token_id, 10);
    assert_eq!(overrides.eoi_token_id, 11);
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
}

// ---- Gemma 4 Unified sanitize remaps -------------------------------------

#[test]
fn unified_key_normalization_prefixes() {
    // Bare language_model.X -> language_model.model.X.
    assert_eq!(
        normalize_gemma4_unified_key("language_model.layers.0.input_layernorm.weight"),
        "language_model.model.layers.0.input_layernorm.weight"
    );
    // Already-normalized key is left untouched.
    assert_eq!(
        normalize_gemma4_unified_key("language_model.model.norm.weight"),
        "language_model.model.norm.weight"
    );
    // model.language_model.X -> language_model.model.X.
    assert_eq!(
        normalize_gemma4_unified_key("model.language_model.norm.weight"),
        "language_model.model.norm.weight"
    );
    // Leading model. is stripped from the multimodal prefixes.
    assert_eq!(
        normalize_gemma4_unified_key("model.vision_embedder.patch_dense.weight"),
        "vision_embedder.patch_dense.weight"
    );
    assert_eq!(
        normalize_gemma4_unified_key("model.embed_vision.embedding_projection.weight"),
        "embed_vision.embedding_projection.weight"
    );
    // Already-clean multimodal keys are untouched.
    assert_eq!(
        normalize_gemma4_unified_key("vision_embedder.pos_embedding"),
        "vision_embedder.pos_embedding"
    );
}

#[test]
fn unified_sanitize_drops_rotary_and_lm_head() {
    let mut raw = WeightMap::new();
    raw.insert(
        "language_model.model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
        mlxcel_core::ones(&[8], dtype::FLOAT32),
    );
    raw.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    raw.insert(
        "vision_embedder.pos_embedding".to_string(),
        mlxcel_core::ones(&[1120, 2, 8], dtype::FLOAT32),
    );

    let out = sanitize_gemma4_unified_weights(raw, true);
    assert!(!out.contains_key("language_model.model.layers.0.self_attn.rotary_emb.inv_freq"));
    assert!(!out.contains_key("lm_head.weight"));
    assert!(out.contains_key("vision_embedder.pos_embedding"));
}

#[test]
fn unified_sanitize_drops_audio_when_absent() {
    let mut raw = WeightMap::new();
    raw.insert(
        "embed_audio.embedding_projection.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );
    raw.insert(
        "embed_vision.embedding_projection.weight".to_string(),
        mlxcel_core::ones(&[4, 4], dtype::FLOAT32),
    );

    // has_audio = false drops embed_audio.*, keeps embed_vision.*.
    let out = sanitize_gemma4_unified_weights(raw, false);
    assert!(!out.contains_key("embed_audio.embedding_projection.weight"));
    assert!(out.contains_key("embed_vision.embedding_projection.weight"));
}

#[test]
fn unified_sanitize_splits_moe_switch_glu() {
    // Fused experts.gate_up_proj [num_experts=2, in=3, 2*ffn=8] splits into
    // gate/up [2, 4, 3] (axes swapped, doubled dim halved); down_proj renamed.
    let mut raw = WeightMap::new();
    raw.insert(
        "language_model.model.layers.0.mlp.experts.gate_up_proj".to_string(),
        mlxcel_core::ones(&[2, 3, 8], dtype::FLOAT32),
    );
    raw.insert(
        "language_model.model.layers.0.mlp.experts.down_proj".to_string(),
        mlxcel_core::ones(&[2, 4, 3], dtype::FLOAT32),
    );

    let out = sanitize_gemma4_unified_weights(raw, true);

    assert!(!out.contains_key("language_model.model.layers.0.mlp.experts.gate_up_proj"));
    let gate = out
        .get("language_model.model.layers.0.mlp.experts.switch_glu.gate_proj.weight")
        .expect("gate_proj split present");
    let up = out
        .get("language_model.model.layers.0.mlp.experts.switch_glu.up_proj.weight")
        .expect("up_proj split present");
    assert_eq!(mlxcel_core::array_shape(gate), vec![2, 4, 3]);
    assert_eq!(mlxcel_core::array_shape(up), vec![2, 4, 3]);
    assert!(
        out.contains_key("language_model.model.layers.0.mlp.experts.switch_glu.down_proj.weight"),
        "down_proj renamed under switch_glu"
    );
}
