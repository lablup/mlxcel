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

//! Unit tests for Jina VLM config normalization and routing.

use super::{build_processor, parse_vision_config, read_quantization, resolve_eos_token_ids};
use crate::models::ModelType;
use serde_json::{Value, json};
use std::path::Path;

/// The `vision_config` block as `jinaai/jina-vlm-mlx` ships it, trimmed to the
/// keys the parser reads.
fn released_vision_config() -> Value {
    json!({
        "block_config": {
            "attn_config": {
                "head_dim": 72,
                "k_bias": true,
                "lnorm_config": { "bias": true, "eps": 1e-06, "type": "default", "with_affine": true },
                "n_heads": 16,
                "n_kv_heads": null,
                "o_bias": true,
                "q_bias": true,
                "qkv_lnorm_on_heads": false,
                "v_bias": true
            },
            "ffn_config": {
                "activation_type": "gelu_pytorch_tanh",
                "bias": true,
                "gated_activation": false,
                "ratio": 4,
                "size": 4304
            },
            "lnorm_config": { "bias": true, "eps": 1e-06, "type": "default", "with_affine": true }
        },
        "hidden_size": 1152,
        "input_size": [378, 378],
        "linear_patch_embedding": true,
        "model_type": "jvlm",
        "n_channels": 3,
        "n_layers": 27,
        "output_size": 2048,
        "patch_embedding_bias": true,
        "patch_size": 14,
        "positional_interpolation": "bicubic",
        "post_lnorm": true,
        "pre_lnorm": false,
        "use_absolute_positional_embeddings": true,
        "use_cls_token": false,
        "vit_layers": [-4, -10],
        "vl_connector_config": {
            "attn_pooling_config": {
                "head_dim": 72,
                "n_heads": 16,
                "n_kv_heads": null,
                "o_bias": true,
                "q_bias": true
            },
            "mlp_projector_config": {
                "activation_type": "silu",
                "gated_activation": true,
                "size": 6144
            },
            "padding_embed_type": "pad_and_partial_pad",
            "pooling_h": 2,
            "pooling_type": "attention_meanq",
            "pooling_w": 2,
            "projector_type": "mlp",
            "spatial_merge_size": 2
        }
    })
}

#[test]
fn the_nested_vision_config_flattens_to_the_released_shapes() {
    let config = parse_vision_config(&released_vision_config(), 64, 4);

    assert_eq!(config.hidden_size, 1152);
    assert_eq!(config.num_hidden_layers, 27);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.head_dim, 72);
    assert_eq!(config.patch_size, 14);
    assert_eq!(config.image_size, 378);
    assert_eq!(config.num_channels, 3);
    assert_eq!(config.intermediate_size, 4304);
    assert_eq!(config.output_size, 2048);
    assert_eq!(config.vit_layers, vec![-4, -10]);
    assert!(config.post_layer_norm);
    assert!(!config.use_cls_token);
    assert_eq!(config.layer_norm_eps, 1e-6);
    assert_eq!((config.group_size, config.bits), (64, 4));
}

#[test]
fn the_connector_reads_its_own_pooling_and_projector_blocks() {
    let config = parse_vision_config(&released_vision_config(), 64, 4);
    // Not the tower's `n_heads`/`head_dim` by accident: they happen to match
    // here, but the connector block is where they must be read from.
    assert_eq!(config.pooling_num_heads, 16);
    assert_eq!(config.pooling_head_dim, 72);
    assert_eq!(config.connector_hidden_size, 6144);
    assert_eq!((config.pooling_h, config.pooling_w), (2, 2));
}

#[test]
fn a_diverging_pooling_head_shape_is_not_taken_from_the_tower() {
    let mut value = released_vision_config();
    value["vl_connector_config"]["attn_pooling_config"]["n_heads"] = json!(8);
    value["vl_connector_config"]["attn_pooling_config"]["head_dim"] = json!(144);
    let config = parse_vision_config(&value, 64, 4);
    assert_eq!(config.pooling_num_heads, 8);
    assert_eq!(config.pooling_head_dim, 144);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.head_dim, 72);
}

#[test]
fn quantization_is_inherited_from_the_top_level_block() {
    let config = json!({ "quantization": { "bits": 4, "group_size": 64 } });
    assert_eq!(read_quantization(&config), (64, 4));

    // An unquantized conversion still has to load, at the affine defaults.
    assert_eq!(read_quantization(&json!({})), (64, 4));
}

#[test]
fn eos_ids_fall_back_to_the_config_when_generation_config_is_absent() {
    let missing = Path::new("/nonexistent/jina-vlm");
    assert_eq!(
        resolve_eos_token_ids(missing, &json!({ "eos_token_id": 151643 })),
        vec![151643]
    );
    assert_eq!(
        resolve_eos_token_ids(missing, &json!({ "eos_token_id": [151643, 151645] })),
        vec![151643, 151645]
    );
    assert_eq!(
        resolve_eos_token_ids(missing, &json!({})),
        crate::models::jina_vlm::JINA_VLM_DEFAULT_EOS_IDS.to_vec()
    );
}

#[test]
fn a_zero_geometry_divisor_in_the_preprocessor_config_does_not_panic_the_load() {
    // `patch_size` / `pooling_h` / `pooling_w` are divisors; a malformed config
    // must fail into the defaults rather than take the process down.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        serde_json::to_string(&json!({
            "patch_size": 0,
            "pooling_h": 0,
            "pooling_w": 0
        }))
        .unwrap(),
    )
    .unwrap();

    let processor = build_processor(dir.path());
    assert_eq!(processor.patch_size, 1);
    assert_eq!(processor.pooling_h, 1);
    assert_eq!(processor.pooling_w, 1);
}

#[test]
fn jvlm_is_the_routed_model_type_with_jina_vlm_as_an_alias() {
    // The released checkpoints declare `jvlm`; routing on `jina_vlm` alone would
    // mean the model never loads.
    for model_type in ["jvlm", "jina_vlm"] {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&json!({
                "architectures": ["JinaVLMForConditionalGeneration"],
                "model_type": model_type,
                "text_config": { "model_type": model_type },
                "vision_config": { "model_type": model_type }
            }))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            crate::loading::get_model_type(dir.path()).unwrap(),
            ModelType::JinaVLM,
            "{model_type} did not route to the Jina VLM loader"
        );
    }
}
