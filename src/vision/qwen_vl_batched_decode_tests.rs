// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

use mlxcel_core::weights::WeightMap;

use crate::audio::qwen3_omni_moe::{Qwen3OmniAudioConfig, Qwen3OmniAudioEncoder};
use crate::models::embedding_test_support::mlx_test_guard;
use crate::models::qwen_vl_batched_decode_test_support::assert_native_qwen_vl_contract;
use crate::models::qwen2_vl::batched_tests::{set_qwen2_state, tiny_qwen2_vl_model};
use crate::models::qwen3_vl::qwen3_vl_tests::{set_qwen3_state, tiny_qwen3_vl_model};
use crate::models::qwen3_vl_moe::tests::{set_qwen3_moe_state, tiny_qwen3_vl_moe_model};
use crate::vision::encoders::qwen3_vl::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};
use crate::vision::processors::qwen2_vl::Qwen2VLProcessor;

fn qwen_processor() -> Qwen2VLProcessor {
    Qwen2VLProcessor::new(1, 1, 1)
}

fn qwen3_processor() -> Qwen2VLProcessor {
    Qwen2VLProcessor::new_with_norm(1, 1, 1, [0.5; 3], [0.5; 3])
}

fn unused_qwen3_vision_encoder() -> Qwen3VLVisionEncoder {
    let config: Qwen3VLVisionConfig = serde_json::from_value(serde_json::json!({
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
    .expect("tiny unused Qwen3 vision config parses");
    let mut weights = WeightMap::new();
    weights.insert(
        "visual.patch_embed.proj.weight".into(),
        mlxcel_core::from_slice_f32(&[0.02; 12], &[4, 3]),
    );
    weights.insert(
        "visual.pos_embed.weight".into(),
        mlxcel_core::from_slice_f32(&[0.01; 4], &[1, 4]),
    );
    weights.insert(
        "visual.merger.norm.weight".into(),
        mlxcel_core::from_slice_f32(&[1.0; 4], &[4]),
    );
    weights.insert(
        "visual.merger.linear_fc1.weight".into(),
        mlxcel_core::from_slice_f32(&[0.03; 16], &[4, 4]),
    );
    weights.insert(
        "visual.merger.linear_fc2.weight".into(),
        mlxcel_core::from_slice_f32(&[0.04; 64], &[16, 4]),
    );
    Qwen3VLVisionEncoder::from_weights(&weights, &config, "visual")
        .expect("unused Qwen3 vision tower is still a valid object")
}

fn unused_qwen3_audio_encoder() -> Qwen3OmniAudioEncoder {
    let config: Qwen3OmniAudioConfig = serde_json::from_value(serde_json::json!({
        "d_model": 4,
        "encoder_layers": 0,
        "encoder_attention_heads": 1,
        "encoder_ffn_dim": 8,
        "num_mel_bins": 4,
        "output_dim": 16,
        "downsample_hidden_size": 4,
        "conv_chunksize": 8,
        "n_window": 2,
        "n_window_infer": 4,
        "max_source_positions": 4
    }))
    .expect("tiny unused Qwen3 audio config parses");
    let mut weights = WeightMap::new();
    for name in ["conv2d1", "conv2d2", "conv2d3"] {
        weights.insert(
            format!("audio.{name}.weight"),
            mlxcel_core::from_slice_f32(&[0.01; 36], &[4, 3, 3, 1]),
        );
    }
    for (name, rows, cols) in [("conv_out", 4, 4), ("proj1", 4, 4), ("proj2", 16, 4)] {
        weights.insert(
            format!("audio.{name}.weight"),
            mlxcel_core::from_slice_f32(&vec![0.02; rows * cols], &[rows as i32, cols as i32]),
        );
    }
    weights.insert(
        "audio.ln_post.weight".into(),
        mlxcel_core::from_slice_f32(&[1.0; 4], &[4]),
    );
    Qwen3OmniAudioEncoder::from_weights(&weights, &config, "audio")
        .expect("unused Qwen3 audio tower is still a valid object")
}

#[test]
fn all_five_qwen_vl_wrappers_execute_mixed_native_batched_decode() {
    let _guard = mlx_test_guard();

    let qwen2 = super::qwen2_vl::Qwen2VLModel {
        text_model: tiny_qwen2_vl_model(),
        vision_encoder: None,
        text_only_path: Some("synthetic".into()),
        processor: qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    };
    assert_native_qwen_vl_contract(&qwen2, |id, state, len| {
        set_qwen2_state(&qwen2.text_model, id, state, len)
    });

    let qwen25 = super::qwen2_5_vl::Qwen25VLModel {
        text_model: tiny_qwen2_vl_model(),
        vision_encoder: None,
        text_only_path: Some("synthetic".into()),
        processor: qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    };
    assert_native_qwen_vl_contract(&qwen25, |id, state, len| {
        set_qwen2_state(&qwen25.text_model, id, state, len)
    });

    let qwen3 = super::qwen3_vl::Qwen3VLModel {
        text_model: tiny_qwen3_vl_model(),
        vision_encoder: None,
        text_only_path: Some("synthetic".into()),
        processor: qwen3_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    };
    assert_native_qwen_vl_contract(&qwen3, |id, state, len| {
        set_qwen3_state(&qwen3.text_model, id, state, len)
    });

    let qwen3_moe = super::qwen3_vl_moe::Qwen3VLMoeModel {
        text_model: tiny_qwen3_vl_moe_model(),
        vision_encoder: None,
        text_only_path: Some("synthetic".into()),
        processor: qwen3_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    };
    assert_native_qwen_vl_contract(&qwen3_moe, |id, state, len| {
        set_qwen3_moe_state(&qwen3_moe.text_model, id, state, len)
    });

    let omni = super::qwen3_omni_moe::Qwen3OmniMoeModel {
        text_model: tiny_qwen3_vl_moe_model(),
        vision_encoder: unused_qwen3_vision_encoder(),
        audio_encoder: unused_qwen3_audio_encoder(),
        processor: qwen3_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        audio_token_id: 27,
        audio_start_token_id: 26,
        audio_end_token_id: 25,
        im_end_token_id: 24,
        spatial_merge_size: 1,
    };
    assert_native_qwen_vl_contract(&omni, |id, state, len| {
        set_qwen3_moe_state(&omni.text_model, id, state, len)
    });
}
