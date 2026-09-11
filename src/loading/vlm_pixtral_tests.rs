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
    apply_mistral_attention_head_override, build_mistral_text_config,
    inherit_quantization_if_missing,
};
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

fn mistral_weight_map() -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::ones(&[5120], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        mlxcel_core::ones(&[14336, 5120], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.31.self_attn.q_proj.weight".to_string(),
        mlxcel_core::ones(&[4096, 5120], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        mlxcel_core::ones(&[4096, 5120], dtype::FLOAT32),
    );
    weights
}

#[test]
fn apply_mistral_attention_head_override_uses_q_proj_shape() {
    let mut text_config = json!({
        "hidden_size": 5120,
        "head_dim": 128,
        "num_attention_heads": 40
    });

    apply_mistral_attention_head_override(&mut text_config, &mistral_weight_map());

    assert_eq!(text_config["num_attention_heads"], 32);
}

#[test]
fn build_mistral_text_config_infers_and_inherits_quantization() {
    let text_config = build_mistral_text_config(
        &json!({
            "text_config": {
                "head_dim": 128
            },
            "quantization": {
                "group_size": 128,
                "bits": 8
            }
        }),
        &mistral_weight_map(),
    )
    .unwrap();

    assert_eq!(text_config["hidden_size"], 5120);
    assert_eq!(text_config["num_hidden_layers"], 32);
    assert_eq!(text_config["intermediate_size"], 14336);
    assert_eq!(text_config["num_attention_heads"], 32);
    assert_eq!(text_config["quantization"]["group_size"], 128);
    assert_eq!(text_config["quantization"]["bits"], 8);
}

#[test]
fn build_mistral_text_config_preserves_existing_quantization() {
    let text_config = build_mistral_text_config(
        &json!({
            "text_config": {
                "head_dim": 128,
                "quantization": {
                    "group_size": 64,
                    "bits": 4
                }
            },
            "quantization": {
                "group_size": 128,
                "bits": 8
            }
        }),
        &mistral_weight_map(),
    )
    .unwrap();

    assert_eq!(text_config["quantization"]["group_size"], 64);
    assert_eq!(text_config["quantization"]["bits"], 4);
}

#[test]
fn inherit_quantization_if_missing_rejects_non_object_configs() {
    let mut text_config = json!(true);
    let full_config = json!({
        "quantization": {"group_size": 128, "bits": 8}
    });

    let err = inherit_quantization_if_missing(&mut text_config, &full_config)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Pixtral/Mistral3 text_config"));
}
