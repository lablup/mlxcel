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

//! Tiny end-to-end MiMo-V2 forward: image features must reach the decoder's
//! merged embeddings, and a placeholder/feature count mismatch must fail.

use crate::models::mimo_v2_flash::ModelArgs;
use crate::vision::encoders::mimovl::{MimovlVisionConfig, MimovlVisionEncoder};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

// ---------------------------------------------------------------------------
// Tiny end-to-end forward with image placeholders
// ---------------------------------------------------------------------------

fn read_all(array: &MlxArray) -> Vec<f32> {
    let flat = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::eval(&flat);
    let bytes = mlxcel_core::array_to_raw_bytes(&flat);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn random(shape: &[i32], seed: u32, scale: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let values: Vec<f32> = noise(n as usize, seed)
        .into_iter()
        .map(|value| value * scale)
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn tiny_text_args() -> ModelArgs {
    ModelArgs {
        model_type: "mimo_v2".to_string(),
        hidden_size: 32,
        num_hidden_layers: 4,
        intermediate_size: 64,
        moe_intermediate_size: 16,
        vocab_size: 48,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        v_head_dim: 4,
        swa_num_attention_heads: 4,
        swa_num_key_value_heads: 2,
        swa_head_dim: 8,
        swa_v_head_dim: 4,
        hybrid_layer_pattern: vec![0, 1, 1, 0],
        moe_layer_freq: vec![0, 1, 1, 1],
        sliding_window_size: 2,
        max_position_embeddings: 64,
        layernorm_epsilon: 1e-5,
        rope_theta: 10_000.0,
        swa_rope_theta: 100.0,
        partial_rotary_factor: 0.5,
        add_swa_attention_sink_bias: true,
        add_full_attention_sink_bias: false,
        attention_value_scale: Some(0.707),
        n_routed_experts: Some(4),
        num_experts_per_tok: 2,
        n_shared_experts: None,
        topk_method: "noaux_tc".to_string(),
        scoring_func: "sigmoid".to_string(),
        norm_topk_prob: true,
        n_group: 1,
        topk_group: 1,
        tie_word_embeddings: false,
        ..ModelArgs::default()
    }
}

fn tiny_text_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let mut weights = WeightMap::new();
    let mut seed = 0x1234_5678u32;
    let mut next = |shape: &[i32]| {
        seed = seed.wrapping_add(17);
        random(shape, seed, 0.2)
    };
    weights.insert(
        "model.embed_tokens.weight".into(),
        next(&[args.vocab_size as i32, hidden]),
    );
    weights.insert(
        "model.norm.weight".into(),
        mlxcel_core::ones(&[hidden], dtype::FLOAT32),
    );
    weights.insert(
        "lm_head.weight".into(),
        next(&[args.vocab_size as i32, hidden]),
    );
    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        let attn = format!("{prefix}.self_attn");
        let sliding = args.is_sliding_layer(layer);
        let (heads, kv_heads, head_dim, v_head_dim) = if sliding {
            (
                args.swa_num_attention_heads,
                args.swa_num_key_value_heads,
                args.swa_head_dim,
                args.swa_v_head_dim,
            )
        } else {
            (
                args.num_attention_heads,
                args.num_key_value_heads,
                args.head_dim,
                args.v_head_dim,
            )
        };
        weights.insert(
            format!("{attn}.q_proj.weight"),
            next(&[(heads * head_dim) as i32, hidden]),
        );
        weights.insert(
            format!("{attn}.k_proj.weight"),
            next(&[(kv_heads * head_dim) as i32, hidden]),
        );
        weights.insert(
            format!("{attn}.v_proj.weight"),
            next(&[(kv_heads * v_head_dim) as i32, hidden]),
        );
        weights.insert(
            format!("{attn}.o_proj.weight"),
            next(&[hidden, (heads * v_head_dim) as i32]),
        );
        if sliding && args.add_swa_attention_sink_bias {
            weights.insert(format!("{attn}.attention_sink_bias"), next(&[heads as i32]));
        }
        weights.insert(
            format!("{prefix}.input_layernorm.weight"),
            mlxcel_core::ones(&[hidden], dtype::FLOAT32),
        );
        weights.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            mlxcel_core::ones(&[hidden], dtype::FLOAT32),
        );
        let mlp = format!("{prefix}.mlp");
        if args.is_moe_layer(layer) {
            let experts = args.num_routed_experts() as i32;
            let moe = args.moe_intermediate_size as i32;
            weights.insert(format!("{mlp}.gate.weight"), next(&[experts, hidden]));
            weights.insert(
                format!("{mlp}.gate.e_score_correction_bias"),
                mlxcel_core::zeros(&[experts], dtype::FLOAT32),
            );
            weights.insert(
                format!("{mlp}.switch_mlp.gate_proj.weight"),
                next(&[experts, moe, hidden]),
            );
            weights.insert(
                format!("{mlp}.switch_mlp.up_proj.weight"),
                next(&[experts, moe, hidden]),
            );
            weights.insert(
                format!("{mlp}.switch_mlp.down_proj.weight"),
                next(&[experts, hidden, moe]),
            );
        } else {
            let intermediate = args.intermediate_size as i32;
            weights.insert(
                format!("{mlp}.gate_proj.weight"),
                next(&[intermediate, hidden]),
            );
            weights.insert(
                format!("{mlp}.up_proj.weight"),
                next(&[intermediate, hidden]),
            );
            weights.insert(
                format!("{mlp}.down_proj.weight"),
                next(&[hidden, intermediate]),
            );
        }
    }
    weights
}

fn tiny_vision_config() -> MimovlVisionConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "mimovl",
        "depth": 3,
        "hidden_size": 8,
        "intermediate_size": 16,
        "out_hidden_size": 32,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "in_chans": 3,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "qk_channels": 8,
        "use_sink": true,
        "visual_token_window_size": 3,
        "fullatt_block_indexes": [0],
        "vit_window_attn_types": [-1, 1, 0]
    }))
    .expect("tiny vision config")
}

fn tiny_vision_weights(config: &MimovlVisionConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    let in_features =
        config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
    weights.insert(
        "vision_tower.patch_embed.proj.weight".into(),
        random(&[config.hidden_size as i32, in_features as i32], 11, 0.2),
    );
    for index in 0..config.depth {
        let prefix = format!("vision_tower.blocks.{index}");
        let qkv = (config.num_heads + 2 * config.num_key_value_heads) * config.qk_channels;
        weights.insert(
            format!("{prefix}.norm1.weight"),
            mlxcel_core::ones(&[config.hidden_size as i32], dtype::FLOAT32),
        );
        weights.insert(
            format!("{prefix}.norm2.weight"),
            mlxcel_core::ones(&[config.hidden_size as i32], dtype::FLOAT32),
        );
        weights.insert(
            format!("{prefix}.attn.qkv.weight"),
            random(
                &[qkv as i32, config.hidden_size as i32],
                20 + index as u32,
                0.2,
            ),
        );
        weights.insert(
            format!("{prefix}.attn.qkv.bias"),
            random(&[qkv as i32], 40 + index as u32, 0.1),
        );
        weights.insert(
            format!("{prefix}.attn.proj.weight"),
            random(
                &[
                    config.hidden_size as i32,
                    (config.num_heads * config.qk_channels) as i32,
                ],
                60 + index as u32,
                0.2,
            ),
        );
        weights.insert(
            format!("{prefix}.attn.proj.bias"),
            random(&[config.hidden_size as i32], 80 + index as u32, 0.1),
        );
        if config.use_sink && !config.fullatt_block_indexes.contains(&index) {
            weights.insert(
                format!("{prefix}.attn.sinks"),
                random(&[config.num_heads as i32], 100 + index as u32, 0.3),
            );
        }
        for (leaf, out, input) in [
            ("gate_proj", config.intermediate_size, config.hidden_size),
            ("up_proj", config.intermediate_size, config.hidden_size),
            ("down_proj", config.hidden_size, config.intermediate_size),
        ] {
            weights.insert(
                format!("{prefix}.mlp.{leaf}.weight"),
                random(&[out as i32, input as i32], 120 + index as u32, 0.2),
            );
            weights.insert(
                format!("{prefix}.mlp.{leaf}.bias"),
                random(&[out as i32], 140 + index as u32, 0.1),
            );
        }
    }
    let merged = config.hidden_size * config.spatial_merge_size * config.spatial_merge_size;
    weights.insert(
        "vision_tower.merger.ln_q.weight".into(),
        mlxcel_core::ones(&[config.hidden_size as i32], dtype::FLOAT32),
    );
    weights.insert(
        "vision_tower.merger.mlp.0.weight".into(),
        random(&[merged as i32, merged as i32], 200, 0.2),
    );
    weights.insert(
        "vision_tower.merger.mlp.2.weight".into(),
        random(&[config.out_hidden_size as i32, merged as i32], 201, 0.2),
    );
    weights
}

fn tiny_vlm() -> crate::vision::MimoV2VlModel {
    let text_args = tiny_text_args();
    let text_model =
        crate::models::MiMoV2FlashModel::from_weights(&tiny_text_weights(&text_args), &text_args)
            .expect("tiny text model");
    let vision_config = tiny_vision_config();
    let encoder = MimovlVisionEncoder::from_weights(
        &tiny_vision_weights(&vision_config),
        &vision_config,
        "vision_tower",
    )
    .expect("tiny vision tower");
    crate::vision::MimoV2VlModel {
        text_model,
        vision_encoder: Some(encoder),
        text_only_path: None,
        processor: crate::vision::processors::qwen2_vl::Qwen2VLProcessor::new(2, 2, 2),
        image_token_id: 7,
        video_token_id: 8,
        vision_start_token_id: 5,
        vision_end_token_id: 6,
        audio_token_id: 9,
        spatial_merge_size: 2,
    }
}

#[test]
fn tiny_forward_scatters_image_features_into_the_decoder() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let vlm = tiny_vlm();
    // 4 image placeholders for a 4x4 grid at merge size 2.
    let input_ids = mlxcel_core::from_slice_i32(&[1, 7, 7, 7, 7, 2], &[1, 6]);
    let pixel_values = mlxcel_core::from_slice_f32(&noise(32 * 12, 0xBEEF), &[32, 12]);
    let prepared = vlm.get_input_embeddings(&input_ids, &pixel_values, &[(1, 4, 4)]);
    assert_eq!(
        mlxcel_core::array_shape(&prepared.inputs_embeds),
        [1, 6, 32]
    );

    // The four image rows are the vision features, not token embeddings.
    let features = vlm
        .vision_encoder
        .as_ref()
        .expect("vision tower")
        .forward_with_grid(&pixel_values, &[(1, 4, 4)])
        .expect("features")
        .hidden_states;
    let expected = read_all(&features);
    let merged = read_all(&prepared.inputs_embeds);
    for row in 0..4 {
        for column in 0..32 {
            let actual = merged[(row + 1) * 32 + column];
            let expected = expected[row * 32 + column];
            assert!(
                (actual - expected).abs() < 1e-5,
                "image row {row} column {column}: merged {actual} != feature {expected}"
            );
        }
    }

    let mut wrapper_caches = Vec::new();
    let wrapper_logits = LanguageModel::forward_last_logits_with_embeddings_and_sequence_id(
        &vlm,
        &input_ids,
        Some(&prepared.inputs_embeds),
        Some(mlxcel_core::cache::SequenceId::from_raw(701)),
        &mut wrapper_caches,
        None,
        5,
    );
    let direct = vlm.text_model.forward_prepared_embeddings_with_sequence_id(
        &prepared.inputs_embeds,
        Some(mlxcel_core::cache::SequenceId::from_raw(702)),
    );
    let direct_last = mlxcel_core::slice(&direct, &[0, 5, 0], &[1, 6, 48]);
    let wrapper_values = read_all(&wrapper_logits);
    let direct_values = read_all(&direct_last);
    assert_eq!(wrapper_values.len(), direct_values.len());
    for (left, right) in wrapper_values.iter().zip(direct_values.iter()) {
        assert!(
            (left - right).abs() < 1e-5,
            "wrapper logits must come from the merged embeddings: {left} != {right}"
        );
    }
}

#[test]
#[should_panic(expected = "placeholders")]
fn tiny_forward_rejects_a_placeholder_count_mismatch() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let vlm = tiny_vlm();
    // Three placeholders for four merged features.
    let input_ids = mlxcel_core::from_slice_i32(&[1, 7, 7, 7, 2], &[1, 5]);
    let pixel_values = mlxcel_core::from_slice_f32(&noise(32 * 12, 0xBEEF), &[32, 12]);
    let _ = vlm.get_input_embeddings(&input_ids, &pixel_values, &[(1, 4, 4)]);
}
