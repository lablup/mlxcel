// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Runtime native-batch coverage for the MRoPE VLM wrappers in #2563.
//!
//! Decode does not read the vision-only fields, but every fixture still owns a
//! valid zero-depth vision tower (and connector where the family has one). The
//! shared contract drives the concrete wrapper through
//! `forward_batched_with_ids` with three unequal cache histories and distinct
//! per-sequence MRoPE deltas, then covers the exact B=1 and empty B=0 routes.

use super::{
    CohereCompassModel, Ernie45MoeVlModel, Glm4vModel, Glm4vMoeModel, GlmOcrModel, HunyuanVlModel,
    PaddleOcrVlModel,
};
use crate::LanguageModel;
use crate::models::embedding_test_support::mlx_test_guard;
use crate::models::vlm_family_batched_decode_test_support::{
    tiny_cohere_compass_model, tiny_ernie45_moe_vl_model, tiny_glm4v_model, tiny_glm4v_moe_model,
    tiny_hunyuan_vl_model, tiny_paddleocr_model,
};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::BatchedDecodeKind;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn filled(shape: &[i32], phase: usize) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|index| ((index as usize + phase) as f32).sin() * 0.2)
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn insert_linear(weights: &mut WeightMap, prefix: &str, output: i32, input: i32, phase: usize) {
    weights.insert(format!("{prefix}.weight"), filled(&[output, input], phase));
}

fn glm_vision_config() -> crate::vision::encoders::glm4v::Glm4vVisionConfig {
    serde_json::from_value(serde_json::json!({
        "depth": 0,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_heads": 1,
        "patch_size": 1,
        "out_hidden_size": 16,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "image_size": 1,
        "in_channels": 3
    }))
    .expect("tiny GLM vision config parses")
}

fn glm_vision_weights(include_glm4v_only_weights: bool) -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert("vision.patch_embed.proj.weight".into(), filled(&[4, 3], 1));
    if include_glm4v_only_weights {
        weights.insert("vision.post_conv_layernorm.weight".into(), ones(&[4]));
        weights.insert(
            "vision.embeddings.position_embedding.weight".into(),
            filled(&[1, 4], 3),
        );
    }
    weights.insert("vision.post_layernorm.weight".into(), ones(&[4]));
    weights.insert("vision.downsample.weight".into(), filled(&[16, 4], 5));
    insert_linear(&mut weights, "vision.merger.proj", 4, 16, 7);
    weights.insert(
        "vision.merger.post_projection_norm.weight".into(),
        ones(&[4]),
    );
    insert_linear(&mut weights, "vision.merger.gate_proj", 8, 4, 11);
    insert_linear(&mut weights, "vision.merger.up_proj", 8, 4, 13);
    insert_linear(&mut weights, "vision.merger.down_proj", 16, 8, 17);
    weights
}

fn unused_glm4v_vision() -> crate::vision::encoders::glm4v::Glm4vVisionEncoder {
    crate::vision::encoders::glm4v::Glm4vVisionEncoder::from_weights(
        &glm_vision_weights(true),
        &glm_vision_config(),
        "vision",
    )
    .expect("tiny zero-depth GLM-4V tower builds")
}

fn unused_glm_ocr_vision() -> crate::vision::encoders::glm_ocr::GlmOcrVisionEncoder {
    crate::vision::encoders::glm_ocr::GlmOcrVisionEncoder::from_weights(
        &glm_vision_weights(false),
        &glm_vision_config(),
        "vision",
    )
    .expect("tiny zero-depth GLM-OCR tower builds")
}

fn unused_hunyuan_vision() -> crate::vision::encoders::hunyuan_vl::HunyuanVlVisionEncoder {
    let config: crate::vision::encoders::hunyuan_vl::HunyuanVlVisionConfig =
        serde_json::from_value(serde_json::json!({
            "hidden_size": 4,
            "out_hidden_size": 16,
            "num_hidden_layers": 0,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "patch_size": 1,
            "spatial_merge_size": 1,
            "max_image_size": 1
        }))
        .expect("tiny Hunyuan vision config parses");
    let mut weights = WeightMap::new();
    weights.insert(
        "vision.embeddings.patch_embedding.weight".into(),
        filled(&[4, 3], 19),
    );
    weights.insert(
        "vision.embeddings.patch_embedding.bias".into(),
        filled(&[4], 23),
    );
    weights.insert(
        "vision.embeddings.position_embedding.weight".into(),
        filled(&[2, 4], 29),
    );
    weights.insert("vision.perceive.before_rms.weight".into(), ones(&[4]));
    weights.insert("vision.perceive.after_rms.weight".into(), ones(&[16]));
    weights.insert(
        "vision.perceive.proj.0.weight".into(),
        filled(&[16, 1, 1, 4], 31),
    );
    weights.insert("vision.perceive.proj.0.bias".into(), filled(&[16], 37));
    weights.insert(
        "vision.perceive.proj.2.weight".into(),
        filled(&[16, 1, 1, 16], 41),
    );
    weights.insert("vision.perceive.proj.2.bias".into(), filled(&[16], 43));
    insert_linear(&mut weights, "vision.perceive.mlp", 16, 16, 47);
    for (name, phase) in [
        ("image_newline", 53),
        ("image_begin", 59),
        ("image_end", 61),
    ] {
        weights.insert(format!("vision.perceive.{name}"), filled(&[16], phase));
    }
    crate::vision::encoders::hunyuan_vl::HunyuanVlVisionEncoder::from_weights(
        &weights, &config, "vision",
    )
    .expect("tiny zero-depth Hunyuan-VL tower builds")
}

fn unused_ernie_vision() -> crate::vision::encoders::ernie4_5_vl::Ernie45VlVisionEncoder {
    let config: crate::vision::encoders::ernie4_5_vl::Ernie45VlVisionConfig =
        serde_json::from_value(serde_json::json!({
            "depth": 0,
            "embed_dim": 4,
            "num_heads": 1,
            "mlp_ratio": 2.0,
            "patch_size": 1,
            "spatial_merge_size": 1
        }))
        .expect("tiny ERNIE vision config parses");
    let mut weights = WeightMap::new();
    insert_linear(&mut weights, "vision.patch_embed.proj", 4, 3, 67);
    weights.insert("vision.ln.weight".into(), ones(&[4]));
    crate::vision::encoders::ernie4_5_vl::Ernie45VlVisionEncoder::from_weights(
        &weights, &config, "vision",
    )
    .expect("tiny zero-depth ERNIE-VL tower builds")
}

fn unused_ernie_resampler() -> crate::vision::connectors::ernie4_5_vl::Ernie45VlResampler {
    let mut weights = WeightMap::new();
    insert_linear(&mut weights, "resampler.spatial_linear.layers.0", 4, 4, 71);
    insert_linear(&mut weights, "resampler.spatial_linear.layers.2", 4, 4, 73);
    weights.insert(
        "resampler.spatial_linear.layers.3.weight".into(),
        ones(&[4]),
    );
    insert_linear(&mut weights, "resampler.mlp", 16, 4, 79);
    weights.insert("resampler.after_norm.weight".into(), ones(&[16]));
    crate::vision::connectors::ernie4_5_vl::Ernie45VlResampler::from_weights(
        &weights,
        "resampler",
        1,
        false,
        0,
        0,
    )
    .expect("tiny ERNIE-VL resampler builds")
}

fn unused_paddle_vision() -> crate::vision::encoders::paddleocr_vl::PaddleOcrVisionEncoder {
    let config: crate::vision::encoders::paddleocr_vl::PaddleOcrVisionConfig =
        serde_json::from_value(serde_json::json!({
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 0,
            "num_attention_heads": 1,
            "num_channels": 3,
            "image_size": 1,
            "patch_size": 1,
            "spatial_merge_size": 1
        }))
        .expect("tiny PaddleOCR vision config parses");
    let mut weights = WeightMap::new();
    weights.insert(
        "vision.embeddings.patch_embedding.weight".into(),
        filled(&[4, 3], 83),
    );
    weights.insert(
        "vision.embeddings.position_embedding.weight".into(),
        filled(&[1, 4], 89),
    );
    weights.insert("vision.post_layernorm.weight".into(), ones(&[4]));
    crate::vision::encoders::paddleocr_vl::PaddleOcrVisionEncoder::from_weights(
        &weights, &config, "vision",
    )
    .expect("tiny zero-depth PaddleOCR-VL tower builds")
}

fn unused_paddle_connector() -> crate::vision::connectors::paddleocr_vl::PaddleOcrProjector {
    let mut weights = WeightMap::new();
    weights.insert("connector.pre_norm.weight".into(), ones(&[4]));
    insert_linear(&mut weights, "connector.linear_1", 4, 4, 97);
    insert_linear(&mut weights, "connector.linear_2", 16, 4, 101);
    crate::vision::connectors::paddleocr_vl::PaddleOcrProjector::from_weights(
        &weights,
        "connector",
        1,
        0,
        0,
    )
    .expect("tiny PaddleOCR-VL connector builds")
}

fn unused_qwen3_vision() -> crate::vision::encoders::qwen3_vl::Qwen3VLVisionEncoder {
    let config: crate::vision::encoders::qwen3_vl::Qwen3VLVisionConfig =
        serde_json::from_value(serde_json::json!({
            "depth": 0,
            "hidden_size": 4,
            "intermediate_size": 4,
            "out_hidden_size": 16,
            "num_heads": 1,
            "patch_size": 1,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "in_channels": 3,
            "num_position_embeddings": 1,
            "deepstack_visual_indexes": []
        }))
        .expect("tiny Qwen3 vision config parses");
    let mut weights = WeightMap::new();
    weights.insert(
        "vision.patch_embed.proj.weight".into(),
        filled(&[4, 3], 103),
    );
    weights.insert("vision.pos_embed.weight".into(), filled(&[1, 4], 107));
    weights.insert("vision.merger.norm.weight".into(), ones(&[4]));
    insert_linear(&mut weights, "vision.merger.linear_fc1", 4, 4, 109);
    insert_linear(&mut weights, "vision.merger.linear_fc2", 16, 4, 113);
    crate::vision::encoders::qwen3_vl::Qwen3VLVisionEncoder::from_weights(
        &weights, &config, "vision",
    )
    .expect("tiny zero-depth Qwen3-VL tower builds")
}

fn qwen_processor() -> crate::vision::processors::qwen2_vl::Qwen2VLProcessor {
    crate::vision::processors::qwen2_vl::Qwen2VLProcessor::new(1, 1, 1)
}

fn qwen3_processor() -> crate::vision::processors::qwen2_vl::Qwen2VLProcessor {
    crate::vision::processors::qwen2_vl::Qwen2VLProcessor::new_with_norm(
        1, 1, 1, [0.5; 3], [0.5; 3],
    )
}

fn hunyuan_processor() -> crate::vision::processors::hunyuan_vl::HunyuanVlProcessor {
    crate::vision::processors::hunyuan_vl::HunyuanVlProcessor {
        patch_size: 1,
        spatial_merge_size: 1,
        min_pixels: 1,
        max_pixels: 16,
        mean: [0.5; 3],
        std: [0.5; 3],
    }
}

fn ernie_processor() -> crate::vision::processors::ernie4_5_vl::Ernie45VlProcessor {
    crate::vision::processors::ernie4_5_vl::Ernie45VlProcessor {
        patch_size: 1,
        spatial_merge_size: 1,
        min_pixels: 1,
        max_pixels: 16,
        mean: [0.5; 3],
        std: [0.5; 3],
    }
}

fn empty_mrope_positions(axes: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(&[], &[axes, 1, 0])
}

fn assert_runtime_delegation<M, S>(model: &M, family: &str, axes: i32, set_state: S)
where
    M: LanguageModel,
    S: Fn(SequenceId, UniquePtr<MlxArray>, i32),
{
    assert_eq!(
        model.batched_decode_kind(),
        BatchedDecodeKind::Native,
        "{family} must advertise native batched decode"
    );

    for base in [10_000_u64, 20_000] {
        for (row, delta) in [2, 7, 11].into_iter().enumerate() {
            set_state(
                SequenceId::from_raw(base + row as u64),
                empty_mrope_positions(axes),
                delta,
            );
        }
    }
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential_with_ids(
        model,
        &[&[1, 2], &[3, 4, 5, 6], &[7, 8, 9, 10, 11, 12]],
        &[13, 14, 15],
        1e-4,
    );

    for id in [30_000_u64, 40_000] {
        set_state(SequenceId::from_raw(id), empty_mrope_positions(axes), 5);
    }
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward_with_ids(
        model,
        &[1, 3, 5],
        7,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(model);
}

#[test]
fn glm4v_wrapper_executes_native_batched_decode_with_per_row_mrope() {
    let _guard = mlx_test_guard();
    let model = Glm4vModel {
        text_model: tiny_glm4v_model(),
        vision_encoder: unused_glm4v_vision(),
        processor: qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    };
    assert_runtime_delegation(&model, "GLM-4V", 3, |id, positions, delta| {
        model.text_model.set_mrope_state(positions, delta);
        model.text_model.bind_mrope_state_to_sequence(id);
    });
}

#[test]
fn glm_ocr_wrapper_executes_native_batched_decode_with_per_row_mrope() {
    let _guard = mlx_test_guard();
    let model = GlmOcrModel {
        text_model: tiny_glm4v_model(),
        vision_encoder: unused_glm_ocr_vision(),
        processor: qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    };
    assert_runtime_delegation(&model, "GLM-OCR", 3, |id, positions, delta| {
        model.text_model.set_mrope_state(positions, delta);
        model.text_model.bind_mrope_state_to_sequence(id);
    });
}

#[test]
fn glm4v_moe_wrapper_executes_native_batched_decode_with_per_row_mrope() {
    let _guard = mlx_test_guard();
    let model = Glm4vMoeModel {
        text_model: tiny_glm4v_moe_model(),
        vision_encoder: unused_glm4v_vision(),
        processor: qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    };
    assert_runtime_delegation(&model, "GLM-4V MoE", 3, |id, positions, delta| {
        model.text_model.set_mrope_state(positions, delta);
        model.text_model.bind_mrope_state_to_sequence(id);
    });
}

#[test]
fn hunyuan_vl_wrapper_executes_native_batched_decode_with_per_row_mrope() {
    let _guard = mlx_test_guard();
    let model = tiny_hunyuan_vl_wrapper();
    assert_runtime_delegation(&model, "Hunyuan-VL", 4, |id, positions, delta| {
        model.text_model.set_mrope_state(positions, delta);
        model.text_model.bind_mrope_state_to_sequence(id);
    });
}

#[test]
fn ernie45_moe_vl_wrapper_executes_native_batched_decode_with_per_row_mrope() {
    let _guard = mlx_test_guard();
    let model = tiny_ernie45_moe_vl_wrapper();
    assert_runtime_delegation(&model, "ERNIE-4.5-MoE-VL", 3, |id, positions, delta| {
        model.text_model.set_mrope_state(positions, delta);
        model.text_model.bind_mrope_state_to_sequence(id);
    });
}

#[test]
fn paddleocr_vl_wrapper_executes_native_batched_decode_with_per_row_mrope() {
    let _guard = mlx_test_guard();
    let model = tiny_paddleocr_vl_wrapper();
    assert_runtime_delegation(&model, "PaddleOCR-VL", 3, |id, positions, delta| {
        model.text_model.set_mrope_state(positions, delta);
        model.text_model.bind_mrope_state_to_sequence(id);
    });
}

#[test]
fn cohere_compass_wrapper_executes_native_batched_decode_with_per_row_mrope() {
    let _guard = mlx_test_guard();
    let model = CohereCompassModel {
        text_model: tiny_cohere_compass_model(),
        vision_encoder: unused_qwen3_vision(),
        processor: qwen3_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    };
    assert_runtime_delegation(&model, "Cohere Compass", 3, |id, positions, delta| {
        model.text_model.set_mrope_state(positions, delta);
        model.text_model.bind_mrope_state_to_sequence(id);
    });
}

/// Tiny Hunyuan-VL wrapper over a zero-depth vision tower (text paths only).
pub(crate) fn tiny_hunyuan_vl_wrapper() -> HunyuanVlModel {
    HunyuanVlModel {
        text_model: tiny_hunyuan_vl_model(),
        vision_encoder: unused_hunyuan_vision(),
        processor: hunyuan_processor(),
        image_token_id: 29,
        image_start_token_id: 28,
        image_end_token_id: 27,
        image_newline_token_id: 26,
        spatial_merge_size: 1,
    }
}

/// Tiny ERNIE-4.5-MoE-VL wrapper over a zero-depth vision tower (text paths
/// only).
pub(crate) fn tiny_ernie45_moe_vl_wrapper() -> Ernie45MoeVlModel {
    Ernie45MoeVlModel {
        text_model: tiny_ernie45_moe_vl_model(),
        vision_encoder: unused_ernie_vision(),
        resampler: unused_ernie_resampler(),
        processor: ernie_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        vision_end_token_id: 27,
        video_start_token_id: 26,
        video_end_token_id: 25,
        spatial_merge_size: 1,
    }
}

/// Tiny PaddleOCR-VL wrapper over a zero-depth vision tower (text paths only).
pub(crate) fn tiny_paddleocr_vl_wrapper() -> PaddleOcrVlModel {
    PaddleOcrVlModel {
        text_model: tiny_paddleocr_model(),
        vision_encoder: unused_paddle_vision(),
        connector: unused_paddle_connector(),
        processor: crate::vision::processors::paddleocr_vl::PaddleOcrVlProcessor::new(1, 1),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}
