// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Runtime delegation coverage for dense vision-language wrappers (#2556).
//!
//! Decode never reads the vision-only fields, but these fixtures still build
//! valid zero-layer towers and projectors so the tests exercise the concrete
//! wrapper implementations rather than source-text assertions.

use super::{GraniteVisionVLModel, Phi3VLModel, Phi4SigLipVLModel};
use crate::LanguageModel;
use crate::models::embedding_test_support::mlx_test_guard;
use crate::vision::config::VisionConfig;
use crate::vision::connectors::mlp::MLPProjector;
use crate::vision::encoders::phi4_siglip::{Phi4SigLipVisionConfig, Phi4SigLipVisionEncoder};
use crate::vision::encoders::siglip::SigLipVisionModel;
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

fn tiny_siglip_vision(prefix: &str) -> SigLipVisionModel {
    let config: VisionConfig = serde_json::from_value(serde_json::json!({
        "model_type": "siglip_vision_model",
        "num_hidden_layers": 0,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_attention_heads": 1,
        "patch_size": 1,
        "image_size": 2,
        "num_channels": 3,
        "layer_norm_eps": 1e-6
    }))
    .expect("tiny SigLIP config parses");
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.embeddings.patch_embedding.weight"),
        filled(&[4, 1, 1, 3], 1),
    );
    weights.insert(
        format!("{prefix}.embeddings.position_embedding.weight"),
        filled(&[4, 4], 3),
    );
    weights.insert(format!("{prefix}.post_layernorm.weight"), ones(&[4]));
    SigLipVisionModel::from_weights(&weights, &config, prefix)
        .expect("tiny zero-layer SigLIP tower builds")
}

fn tiny_phi4_siglip_vision(prefix: &str) -> Phi4SigLipVisionEncoder {
    let config: Phi4SigLipVisionConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "num_channels": 3,
        "image_size": 2,
        "patch_size": 1,
        "num_patches": 4
    }))
    .expect("tiny Phi4-SigLIP config parses");
    let mut weights = WeightMap::new();
    insert_linear(
        &mut weights,
        &format!("{prefix}.embeddings.patch_embedding"),
        4,
        3,
        5,
    );
    weights.insert(
        format!("{prefix}.embeddings.position_embedding.weight"),
        filled(&[4, 4], 7),
    );
    Phi4SigLipVisionEncoder::from_weights(&weights, &config, prefix, 64, 4)
        .expect("tiny zero-layer Phi4-SigLIP tower builds")
}

fn tiny_projector(prefix: &str, input: i32, hidden: i32) -> MLPProjector {
    let mut weights = WeightMap::new();
    insert_linear(
        &mut weights,
        &format!("{prefix}.linear_1"),
        hidden,
        input,
        11,
    );
    insert_linear(
        &mut weights,
        &format!("{prefix}.linear_2"),
        hidden,
        hidden,
        13,
    );
    MLPProjector::from_weights(&weights, prefix, 64, 4).expect("tiny projector builds")
}

/// Callers hold `mlx_test_guard`: the sequential and batched arms both run on
/// the process-wide default device, and the single-row check compares them
/// bit for bit, so a concurrent `DefaultDeviceGuard::cpu()` must not move the
/// device between them.
fn assert_runtime_delegation<M: LanguageModel>(model: &M) {
    assert_eq!(model.batched_decode_kind(), BatchedDecodeKind::Native);
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        model,
        &[&[1, 2], &[3, 4, 5, 6], &[7, 8, 9, 10, 11, 12]],
        &[13, 14, 15],
        1e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        model,
        &[1, 3, 5],
        7,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(model);
}

#[test]
fn phi3_vl_delegates_native_batched_decode_to_text_tower() {
    let _guard = mlx_test_guard();
    let mut projector_weights = WeightMap::new();
    insert_linear(&mut projector_weights, "img_projection.0", 32, 16, 17);
    insert_linear(&mut projector_weights, "img_projection.2", 32, 32, 19);
    let model = Phi3VLModel {
        text_model: crate::models::phi3::tests::tiny_longrope_model(),
        vision_encoder: tiny_siglip_vision("vision_tower.vision_model"),
        glb_gn: filled(&[1, 1, 16], 23),
        sub_gn: filled(&[1, 1, 1, 16], 29),
        img_proj_linear1: mlxcel_core::layers::UnifiedLinear::from_weights(
            &projector_weights,
            "img_projection.0",
            64,
            4,
        )
        .expect("first Phi3-V projector builds"),
        img_proj_linear2: mlxcel_core::layers::UnifiedLinear::from_weights(
            &projector_weights,
            "img_projection.2",
            64,
            4,
        )
        .expect("second Phi3-V projector builds"),
        processor: crate::vision::processors::phi3_v::Phi3VProcessor::new(1),
        image_dim_out: 4,
    };
    assert_runtime_delegation(&model);
}

#[test]
fn phi4_siglip_vl_delegates_native_batched_decode_to_text_tower() {
    let _guard = mlx_test_guard();
    let mut projector_weights = WeightMap::new();
    insert_linear(&mut projector_weights, "mm_projector_linear1", 32, 4, 31);
    insert_linear(&mut projector_weights, "mm_projector_linear2", 32, 32, 37);
    let model = Phi4SigLipVLModel {
        text_model: crate::models::phi3::tests::tiny_longrope_model(),
        vision_tower: tiny_phi4_siglip_vision("vision_tower"),
        mm_projector_linear1: mlxcel_core::layers::UnifiedLinear::from_weights(
            &projector_weights,
            "mm_projector_linear1",
            64,
            4,
        )
        .expect("first Phi4-SigLIP projector builds"),
        mm_projector_linear2: mlxcel_core::layers::UnifiedLinear::from_weights(
            &projector_weights,
            "mm_projector_linear2",
            64,
            4,
        )
        .expect("second Phi4-SigLIP projector builds"),
        processor: crate::vision::processors::phi4_siglip::Phi4SigLipProcessor::new(1, 1, 4),
        select_layer: -1,
        eos_token_ids: vec![2],
    };
    assert_runtime_delegation(&model);
}

#[test]
fn granite_vision_delegates_native_batched_decode_to_text_tower() {
    let _guard = mlx_test_guard();
    let model = GraniteVisionVLModel {
        text_model: crate::models::popular_dense_batched_decode_tests::tiny_granite_model(),
        vision_tower: tiny_siglip_vision("vision_tower.vision_model"),
        projector: tiny_projector("multi_modal_projector", 4, 8),
        image_newline: filled(&[8], 41),
        processor: crate::vision::processors::anyres::AnyResProcessor::new(vec![(2, 2)], 2),
        image_token_index: 11,
        vision_feature_layers: vec![-1],
        feature_side: 2,
        base_tokens: 4,
        eos_token_id: 2,
    };
    assert_runtime_delegation(&model);
}
