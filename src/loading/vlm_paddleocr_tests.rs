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

use super::{remap_key, remap_paddleocr_weights};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use std::collections::BTreeSet;

// Layer counts from the published `paddleocr-vl-bfloat16` config.json:
// text `num_hidden_layers` and `vision_config.num_hidden_layers`.
const TEXT_LAYERS: usize = 18;
const VISION_LAYERS: usize = 27;

/// The real published checkpoint's weight-key set, as read (read-only) from
/// `model.safetensors.index.json` in `paddleocr-vl-bfloat16`: the ERNIE-4.5
/// text backbone wrapped under `language_model.*`, plus an already-sanitized,
/// already-qkv-fused `visual.*` vision tower with a nested `visual.projector`.
fn real_checkpoint_keys() -> Vec<String> {
    let mut keys = Vec::new();

    // Text backbone, wrapped under `language_model`.
    keys.push("language_model.model.embed_tokens.weight".to_string());
    keys.push("language_model.lm_head.weight".to_string());
    keys.push("language_model.model.norm.weight".to_string());
    for i in 0..TEXT_LAYERS {
        let p = format!("language_model.model.layers.{i}");
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            keys.push(format!("{p}.self_attn.{proj}.weight"));
        }
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            keys.push(format!("{p}.mlp.{proj}.weight"));
        }
        keys.push(format!("{p}.input_layernorm.weight"));
        keys.push(format!("{p}.post_attention_layernorm.weight"));
    }

    // Vision tower: embeddings + transformer stack + post-layernorm.
    keys.push("visual.embeddings.patch_embedding.weight".to_string());
    keys.push("visual.embeddings.patch_embedding.bias".to_string());
    keys.push("visual.embeddings.position_embedding.weight".to_string());
    for i in 0..VISION_LAYERS {
        let p = format!("visual.layers.{i}");
        for wb in ["weight", "bias"] {
            keys.push(format!("{p}.layer_norm1.{wb}"));
            keys.push(format!("{p}.layer_norm2.{wb}"));
            keys.push(format!("{p}.self_attn.qkv.{wb}"));
            keys.push(format!("{p}.self_attn.out_proj.{wb}"));
            keys.push(format!("{p}.mlp.fc1.{wb}"));
            keys.push(format!("{p}.mlp.fc2.{wb}"));
        }
    }
    keys.push("visual.post_layernorm.weight".to_string());
    keys.push("visual.post_layernorm.bias".to_string());

    // Spatial-merge connector, nested under `visual`.
    for wb in ["weight", "bias"] {
        keys.push(format!("visual.projector.pre_norm.{wb}"));
        keys.push(format!("visual.projector.linear_1.{wb}"));
        keys.push(format!("visual.projector.linear_2.{wb}"));
    }

    keys
}

/// Weight keys the ERNIE-4.5 backbone requests in
/// `PaddleOcrTextModel::from_weights` (tie_word_embeddings = false).
fn required_text_keys() -> Vec<String> {
    let mut keys = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
        "lm_head.weight".to_string(),
    ];
    for i in 0..TEXT_LAYERS {
        let p = format!("model.layers.{i}");
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            keys.push(format!("{p}.self_attn.{proj}.weight"));
        }
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            keys.push(format!("{p}.mlp.{proj}.weight"));
        }
        keys.push(format!("{p}.input_layernorm.weight"));
        keys.push(format!("{p}.post_attention_layernorm.weight"));
    }
    keys
}

/// Weight keys the NaViT vision encoder requests in
/// `PaddleOcrVisionEncoder::from_weights(prefix = "visual")` plus the connector
/// keys from `PaddleOcrProjector::from_weights(prefix = "visual.projector")`.
fn required_vision_keys() -> Vec<String> {
    let mut keys = vec![
        "visual.embeddings.patch_embedding.weight".to_string(),
        "visual.embeddings.position_embedding.weight".to_string(),
        "visual.post_layernorm.weight".to_string(),
        "visual.projector.pre_norm.weight".to_string(),
        "visual.projector.linear_1.weight".to_string(),
        "visual.projector.linear_2.weight".to_string(),
    ];
    for i in 0..VISION_LAYERS {
        let p = format!("visual.layers.{i}");
        keys.push(format!("{p}.layer_norm1.weight"));
        keys.push(format!("{p}.layer_norm2.weight"));
        keys.push(format!("{p}.self_attn.qkv.weight"));
        keys.push(format!("{p}.self_attn.out_proj.weight"));
        keys.push(format!("{p}.mlp.fc1.weight"));
        keys.push(format!("{p}.mlp.fc2.weight"));
    }
    keys
}

fn weight_map_from_keys(keys: &[String]) -> WeightMap {
    let mut wm = WeightMap::new();
    for k in keys {
        wm.insert(k.clone(), mlxcel_core::ones(&[1, 1], dtype::FLOAT32));
    }
    wm
}

#[test]
fn remap_produces_text_and_vision_keys_for_real_checkpoint() {
    let raw = weight_map_from_keys(&real_checkpoint_keys());
    let out = remap_paddleocr_weights(raw).expect("remap of real checkpoint keys should succeed");
    let produced: BTreeSet<String> = out.keys().cloned().collect();

    // Every key the text backbone requests must be present. The reported
    // failure was `Weight not found: model.embed_tokens.weight`.
    for key in required_text_keys() {
        assert!(
            produced.contains(&key),
            "text weight missing after remap: {key}"
        );
    }
    // Every key the vision tower + connector request must be present.
    for key in required_vision_keys() {
        assert!(
            produced.contains(&key),
            "vision weight missing after remap: {key}"
        );
    }

    // No raw `language_model.` wrapper (or legacy vision aliases) may survive.
    for key in &produced {
        assert!(
            !key.starts_with("language_model."),
            "unstripped language_model wrapper survived: {key}"
        );
        assert!(
            !key.contains("visual.vision_model"),
            "unmapped vision_model alias survived: {key}"
        );
        assert!(
            !key.starts_with("mlp_AR"),
            "unmapped mlp_AR projector alias survived: {key}"
        );
    }
}

#[test]
fn remap_is_lossless_for_real_checkpoint() {
    // The published checkpoint is already sanitized + fused, so the remap is a
    // pure `language_model.`-strip on the text side and an identity map on the
    // vision side: nothing may be dropped, duplicated, or collapsed.
    let raw_keys = real_checkpoint_keys();
    let expected: BTreeSet<String> = raw_keys.iter().map(|k| remap_key(k)).collect();
    let out = remap_paddleocr_weights(weight_map_from_keys(&raw_keys)).expect("remap ok");
    let produced: BTreeSet<String> = out.keys().cloned().collect();

    assert_eq!(produced, expected);
    assert_eq!(
        produced.len(),
        raw_keys.len(),
        "remap must be a bijection for the published checkpoint"
    );
}

#[test]
fn remap_key_strips_language_model_wrapper() {
    assert_eq!(
        remap_key("language_model.model.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(remap_key("language_model.lm_head.weight"), "lm_head.weight");
    assert_eq!(
        remap_key("language_model.model.layers.5.self_attn.q_proj.weight"),
        "model.layers.5.self_attn.q_proj.weight"
    );
    assert_eq!(
        remap_key("language_model.model.norm.weight"),
        "model.norm.weight"
    );
}

#[test]
fn remap_key_passes_through_sanitized_vision_keys() {
    for key in [
        "visual.embeddings.patch_embedding.weight",
        "visual.embeddings.position_embedding.weight",
        "visual.layers.3.self_attn.qkv.weight",
        "visual.layers.3.self_attn.out_proj.bias",
        "visual.post_layernorm.weight",
        "visual.projector.linear_1.weight",
    ] {
        assert_eq!(remap_key(key), key, "vision key should pass through: {key}");
    }
}

#[test]
fn remap_key_still_handles_reference_sanitize_layout() {
    // The older reference `Model.sanitize` layout must keep working.
    assert_eq!(
        remap_key("visual.vision_model.encoder.layers.0.self_attn.q_proj.weight"),
        "visual.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        remap_key("visual.vision_model.embeddings.patch_embedding.weight"),
        "visual.embeddings.patch_embedding.weight"
    );
    assert_eq!(
        remap_key("visual.vision_model.post_layernorm.weight"),
        "visual.post_layernorm.weight"
    );
    assert_eq!(
        remap_key("mlp_AR.linear_1.weight"),
        "visual.projector.linear_1.weight"
    );
}
