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

fn pinned_config() -> String {
    serde_json::json!({
        "model_type": "molmo2",
        "image_patch_id": 151938,
        "vit_config": {
            "hidden_size": 8, "intermediate_size": 16,
            "num_attention_heads": 2, "head_dim": 4,
            "num_hidden_layers": 27, "image_default_input_size": [28, 28],
            "image_patch_size": 14, "image_num_pos": 4, "layer_norm_eps": 1e-6
        },
        "adapter_config": {
            "hidden_size": 8, "intermediate_size": 12, "text_hidden_size": 10,
            "num_attention_heads": 2, "head_dim": 4, "vit_layers": [-3, -9],
            "pooling_attention_mask": true
        }
    })
    .to_string()
}

fn pinned_processor() -> String {
    serde_json::json!({
        "patch_size": 14, "max_crops": 8, "overlap_margins": [0, 0],
        "pooling_size": [2, 2], "size": {"height": 28, "width": 28}
    })
    .to_string()
}

#[test]
fn resolves_pinned_layers_and_static_bucket_identity() {
    let config = Molmo2VisionConfig::from_json_strs(&pinned_config(), &pinned_processor()).unwrap();
    assert_eq!(config.layers, 25);
    assert_eq!(config.selected_layers, vec![22, 16]);
    assert_eq!(config.emitted_layers, 23);
    assert_eq!(config.static_crops, 9);
    assert_eq!(config.static_pool_groups, 9);
    assert!(config.fingerprint().contains("position=exact-default"));
    assert!(config.fingerprint().contains("selected=[22, 16]"));
    assert!(config.fingerprint().contains("pool-mask=true"));
}

#[test]
fn validates_runtime_crop_grid_relationship() {
    let config = Molmo2VisionConfig::from_json_strs(&pinned_config(), &pinned_processor()).unwrap();
    assert!(config.valid_runtime_geometry(3, [1, 1, 1, 2]));
    assert!(config.valid_runtime_geometry(3, [1, 1, 2, 1]));
    assert!(!config.valid_runtime_geometry(4, [1, 1, 1, 2]));
}

#[test]
fn rejects_position_grid_and_selected_layer_drift() {
    let mut config: Value = serde_json::from_str(&pinned_config()).unwrap();
    config["vit_config"]["image_num_pos"] = Value::from(5);
    assert!(
        Molmo2VisionConfig::from_json_strs(&config.to_string(), &pinned_processor())
            .unwrap_err()
            .contains("exact position")
    );
    config["vit_config"]["image_num_pos"] = Value::from(4);
    config["adapter_config"]["vit_layers"] = serde_json::json!([-26]);
    assert!(
        Molmo2VisionConfig::from_json_strs(&config.to_string(), &pinned_processor())
            .unwrap_err()
            .contains("outside")
    );
}
