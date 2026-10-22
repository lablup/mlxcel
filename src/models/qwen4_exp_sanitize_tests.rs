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

use serde_json::json;

use super::*;
use crate::models::qwen4_exp::config::Qwen4ExpConfig;

fn config(patch: serde_json::Value) -> ResolvedQwen4ExpConfig {
    let mut raw = json!({
        "model_type": "qwen4_exp_text",
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "num_experts": 3,
        "num_experts_per_tok": 1,
        "shared_expert_intermediate_size": 4,
        "moe_intermediate_size": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 16,
        "max_position_embeddings": 64,
        "ple_embed_dim": 16,
        "heads_per_ngram": 1,
        "eos_token_id": 1,
        "rope_parameters": {
            "type": "default",
            "mrope_section": [1, 1, 0],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
        }
    });
    for (key, value) in patch.as_object().expect("patch is an object") {
        raw[key] = value.clone();
    }
    let config: Qwen4ExpConfig = serde_json::from_value(raw).unwrap();
    config.resolve().unwrap()
}

fn tensor(shape: &[i32]) -> UniquePtr<MlxArray> {
    let count: i32 = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| i as f32).collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn floats(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct KeyCase {
    checkpoint: &'static str,
    canonical: &'static str,
}

#[test]
fn checkpoint_layouts_map_onto_canonical_keys() {
    for case in [
        KeyCase {
            checkpoint: "model.language_model.layers.0.mlp.gate.weight",
            canonical: "model.layers.0.mlp.gate.weight",
        },
        KeyCase {
            checkpoint: "language_model.model.layers.0.mlp.gate.weight",
            canonical: "model.layers.0.mlp.gate.weight",
        },
        KeyCase {
            checkpoint: "model.layers.0.mlp.gate.weight",
            canonical: "model.layers.0.mlp.gate.weight",
        },
        KeyCase {
            checkpoint: "language_model.lm_head.weight",
            canonical: "lm_head.weight",
        },
        KeyCase {
            checkpoint: "model.visual.blocks.0.attn.qkv.weight",
            canonical: "vision_tower.blocks.0.attn.qkv.weight",
        },
        KeyCase {
            checkpoint: "model.language_model.visual.merger.norm.weight",
            canonical: "vision_tower.merger.norm.weight",
        },
        KeyCase {
            checkpoint: "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_12.scales",
            canonical: "model.layers.1.ple.ple_embedding.ngram_embedding.shards.12.scales",
        },
        KeyCase {
            checkpoint: "model.layers.1.ple.ple_embedding.ngram_embedding.shards.12.weight",
            canonical: "model.layers.1.ple.ple_embedding.ngram_embedding.shards.12.weight",
        },
    ] {
        assert_eq!(canonical_key(case.checkpoint), case.canonical, "{case:?}");
    }
}

#[test]
fn sanitize_splits_vision_drops_mtp_and_moves_conv_axes() {
    let mut weights = WeightMap::new();
    weights.insert("mtp.layers.0.mlp.gate.weight".into(), tensor(&[2, 2]));
    weights.insert("model.visual.patch_embed.proj.bias".into(), tensor(&[2]));
    weights.insert(
        "model.language_model.layers.0.linear_attn.conv1d.weight".into(),
        tensor(&[6, 1, 4]),
    );
    weights.insert(
        "model.language_model.layers.1.ple.conv1d.weight".into(),
        tensor(&[6, 4, 1]),
    );
    weights.insert("lm_head.weight".into(), tensor(&[16, 8]));

    let sanitized = sanitize_weights(weights, &config(json!({}))).unwrap();
    assert!(sanitized.text.keys().all(|key| !key.contains("mtp")));
    assert!(
        sanitized
            .vision
            .contains_key("vision_tower.patch_embed.proj.bias")
    );
    assert!(sanitized.text.contains_key("lm_head.weight"));
    let moved = &sanitized.text["model.layers.0.linear_attn.conv1d.weight"];
    assert_eq!(mlxcel_core::array_shape(moved), vec![6, 4, 1]);
    assert_eq!(&floats(moved)[..4], &[0.0, 1.0, 2.0, 3.0]);
    let kept = &sanitized.text["model.layers.1.ple.conv1d.weight"];
    assert_eq!(mlxcel_core::array_shape(kept), vec![6, 4, 1]);
}

#[test]
fn tied_embeddings_drop_the_lm_head() {
    let mut weights = WeightMap::new();
    weights.insert("lm_head.weight".into(), tensor(&[16, 8]));
    let sanitized =
        sanitize_weights(weights, &config(json!({ "tie_word_embeddings": true }))).unwrap();
    assert!(!sanitized.text.contains_key("lm_head.weight"));
}

#[test]
fn per_expert_projections_are_stacked_in_expert_order() {
    let mut weights = WeightMap::new();
    for expert in 0..3 {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let values = vec![expert as f32; 8];
            weights.insert(
                format!("model.layers.0.mlp.experts.{expert}.{projection}.weight"),
                mlxcel_core::from_slice_f32(&values, &[2, 4]),
            );
        }
    }
    let sanitized = sanitize_weights(weights, &config(json!({}))).unwrap();
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        let stacked =
            &sanitized.text[&format!("model.layers.0.mlp.switch_mlp.{projection}.weight")];
        assert_eq!(mlxcel_core::array_shape(stacked), vec![3, 2, 4]);
        let values = floats(stacked);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[8], 1.0);
        assert_eq!(values[16], 2.0);
    }
    assert!(sanitized.text.keys().all(|key| !key.contains(".experts.")));
}

#[test]
fn fused_gate_up_experts_are_split_into_gate_and_up() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.experts.gate_up_proj".into(),
        tensor(&[3, 4, 2]),
    );
    weights.insert(
        "model.layers.0.mlp.experts.down_proj".into(),
        tensor(&[3, 2, 2]),
    );
    let sanitized = sanitize_weights(weights, &config(json!({}))).unwrap();
    let gate = &sanitized.text["model.layers.0.mlp.switch_mlp.gate_proj.weight"];
    let up = &sanitized.text["model.layers.0.mlp.switch_mlp.up_proj.weight"];
    assert_eq!(mlxcel_core::array_shape(gate), vec![3, 2, 2]);
    assert_eq!(&floats(gate)[..4], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&floats(up)[..4], &[4.0, 5.0, 6.0, 7.0]);
    assert!(
        sanitized
            .text
            .contains_key("model.layers.0.mlp.switch_mlp.down_proj.weight")
    );
}

#[test]
fn fp8_checkpoints_are_refused_by_name() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv".into(),
        tensor(&[1, 1]),
    );
    let error = sanitize_weights(weights, &config(json!({})))
        .err()
        .expect("FP8 is refused");
    assert!(error.contains("weight_scale_inv"), "{error}");
}

/// Sanitize walks the expert blocks the checkpoint stores, so its work is
/// bounded by the tensors and not by the declared layer count.
#[test]
fn sanitize_visits_the_expert_blocks_the_checkpoint_stores() {
    let per_expert = || {
        let mut weights = WeightMap::new();
        for layer in [0, 5] {
            for expert in 0..3 {
                weights.insert(
                    format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
                    tensor(&[2, 4]),
                );
            }
        }
        weights
    };
    for layers in [4u64, 1 << 40] {
        let sanitized = sanitize_weights(
            per_expert(),
            &config(json!({ "num_hidden_layers": layers })),
        )
        .unwrap();
        for layer in [0, 5] {
            assert!(
                sanitized.text.contains_key(&format!(
                    "model.layers.{layer}.mlp.switch_mlp.gate_proj.weight"
                )),
                "{layers} declared layers: layer {layer} is not stacked"
            );
        }
        assert!(sanitized.text.keys().all(|key| !key.contains(".experts.")));
    }
}

/// `stack` throws inside MLX on parts of different shapes, which aborts the
/// process; the loader names the odd expert instead.
#[test]
fn per_expert_tensors_of_different_shapes_are_refused() {
    let mut weights = WeightMap::new();
    for expert in 0..3 {
        let shape: &[i32] = if expert == 2 { &[2, 5] } else { &[2, 4] };
        weights.insert(
            format!("model.layers.0.mlp.experts.{expert}.gate_proj.weight"),
            tensor(shape),
        );
    }
    let error = sanitize_weights(weights, &config(json!({})))
        .err()
        .expect("mismatched experts must be refused");
    assert!(
        error.contains("model.layers.0.mlp.experts.2.gate_proj.weight has shape [2, 5]"),
        "{error}"
    );
}

#[derive(Debug, Clone, Copy)]
struct FusedRankCase {
    suffix: &'static str,
    shape: &'static [i32],
}

/// Every fused plane is split on its row axis, so a plane of another rank is
/// refused by name before that axis is indexed.
#[test]
fn fused_expert_planes_of_the_wrong_rank_are_refused() {
    for case in [
        FusedRankCase {
            suffix: "scales",
            shape: &[4],
        },
        FusedRankCase {
            suffix: "biases",
            shape: &[3, 4],
        },
        FusedRankCase {
            suffix: "bias",
            shape: &[],
        },
    ] {
        let mut weights = WeightMap::new();
        weights.insert(
            "model.layers.0.mlp.experts.gate_up_proj.weight".into(),
            tensor(&[3, 4, 2]),
        );
        weights.insert(
            format!("model.layers.0.mlp.experts.gate_up_proj.{}", case.suffix),
            tensor(case.shape),
        );
        let error = sanitize_weights(weights, &config(json!({})))
            .err()
            .unwrap_or_else(|| panic!("{case:?} must be refused"));
        assert!(
            error.contains(&format!(
                "gate_up_proj.{} must be [num_experts",
                case.suffix
            )),
            "{case:?}: {error}"
        );
    }
}
