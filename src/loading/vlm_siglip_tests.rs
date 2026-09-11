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

use super::{
    build_paligemma_text_model, inject_aya_text_defaults, inject_paligemma_text_defaults,
    quantization_params, resolve_image_token_id, rewrite_aya_weight_key,
    rewrite_paligemma_weight_key,
};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

#[test]
fn rewrite_aya_weight_key_maps_known_prefixes() {
    assert_eq!(
        rewrite_aya_weight_key("model.vision_tower.encoder.layers.0"),
        Some("vision_tower.encoder.layers.0".to_string())
    );
    assert_eq!(
        rewrite_aya_weight_key("model.multi_modal_projector.linear.weight"),
        Some("multi_modal_projector.linear.weight".to_string())
    );
    assert_eq!(
        rewrite_aya_weight_key("model.language_model.layers.0.self_attn.q_proj.weight"),
        Some("model.layers.0.self_attn.q_proj.weight".to_string())
    );
    assert_eq!(rewrite_aya_weight_key("lm_head.weight"), None);
}

#[test]
fn inject_aya_text_defaults_sets_missing_values() {
    let mut text_config = json!({
        "hidden_size": 4096,
        "num_attention_heads": 32
    });
    let weights = WeightMap::new();

    inject_aya_text_defaults(&mut text_config, &weights).unwrap();

    assert_eq!(text_config["vocab_size"], 256000);
    assert_eq!(text_config["layer_norm_eps"], 1e-5);
    assert_eq!(text_config["head_dim"], 128);
    assert_eq!(text_config["sliding_window"], 4096);
    assert_eq!(text_config["tie_word_embeddings"], true);
}

#[test]
fn inject_aya_text_defaults_preserves_explicit_values() {
    let mut text_config = json!({
        "vocab_size": 123,
        "layer_norm_eps": 1e-6,
        "head_dim": 64,
        "sliding_window": 1024,
        "tie_word_embeddings": false
    });
    let mut weights = WeightMap::new();
    weights.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
    );

    inject_aya_text_defaults(&mut text_config, &weights).unwrap();

    assert_eq!(text_config["vocab_size"], 123);
    assert_eq!(text_config["layer_norm_eps"], 1e-6);
    assert_eq!(text_config["head_dim"], 64);
    assert_eq!(text_config["sliding_window"], 1024);
    assert_eq!(text_config["tie_word_embeddings"], false);
}

#[test]
fn rewrite_paligemma_weight_key_rewrites_projector_prefix() {
    assert_eq!(
        rewrite_paligemma_weight_key("multi_modal_projector.linear.weight"),
        Some("multi_modal_projector.linear_1.weight".to_string())
    );
    assert_eq!(rewrite_paligemma_weight_key("model.layers.0"), None);
}

#[test]
fn inject_paligemma_text_defaults_uses_query_scalar_when_present() {
    let mut text_config = json!({
        "query_pre_attn_scalar": 192
    });

    inject_paligemma_text_defaults(&mut text_config).unwrap();

    assert_eq!(text_config["rms_norm_eps"], 1e-6);
    assert_eq!(text_config["head_dim"], 192);
}

#[test]
fn inject_aya_text_defaults_rejects_non_object_configs() {
    let mut text_config = json!(null);
    let err = inject_aya_text_defaults(&mut text_config, &WeightMap::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("Aya Vision text_config"));
}

#[test]
fn inject_paligemma_text_defaults_rejects_non_object_configs() {
    let mut text_config = json!([]);
    let err = inject_paligemma_text_defaults(&mut text_config)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PaliGemma text_config"));
}

#[test]
fn quantization_and_token_helpers_apply_defaults() {
    assert_eq!(quantization_params(&json!({})), (64, 4));
    assert_eq!(resolve_image_token_id(&json!({}), 77), 77);
    assert_eq!(
        resolve_image_token_id(&json!({"image_token_index": 99}), 77),
        99
    );
}

#[test]
fn build_paligemma_text_model_rejects_unknown_backend() {
    let err = match build_paligemma_text_model(&WeightMap::new(), &json!({"model_type": "bad"})) {
        Ok(_) => panic!("expected unsupported backend to fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("Unsupported PaliGemma text backend")
    );
}
