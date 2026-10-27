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

//! Runtime native-batch coverage for the remaining vision-language wrappers.
//!
//! Decode does not touch the vision-only fields, but each fixture still owns a
//! valid, minimal tower/connector rather than relying on a mock wrapper.

use super::dots_ocr::DotsOcrVlModel;
use super::jina_vlm::JinaVlmModel;
use super::minicpmo_vl::MiniCPMOVLModel;
use super::molmo_point_vl::{MolmoPointConfig, MolmoPointVLModel};
use super::molmo_vl::MolmoVLModel;
use super::molmo2_vl::Molmo2VLModel;
use super::moondream2_vl::Moondream2VLModel;
use super::moondream3_vl::Moondream3VLModel;
use crate::LanguageModel;
use crate::models::wrapper_batched_decode_test_support::{
    tiny_jina_vlm_text_model, tiny_molmo_model, tiny_molmo_point_language_model, tiny_molmo2_model,
    tiny_moondream2_model, tiny_moondream3_model, tiny_qwen2_model, tiny_qwen3_model,
};
use crate::vision::encoders::dots_ocr::{DotsVisionConfig, DotsVisionEncoder};
use crate::vision::encoders::minicpmo::{
    MiniCPMOResampler, MiniCPMOVisionConfig, MiniCPMOVisionModel,
};
use crate::vision::encoders::molmo::{MolmoVisionConfig, MolmoVisionModel};
use crate::vision::encoders::molmo_point::{MolmoPointConnector, PointPredictor};
use crate::vision::encoders::molmo2::{Molmo2VisionModel, Molmo2VisionTransformer};
use crate::vision::encoders::moondream3::{Moondream3VisionConfig, Moondream3VisionModel};
use crate::vision::processors::dots_ocr::DotsOcrProcessor;
use crate::vision::processors::jina_vlm::JinaVlmProcessor;
use crate::vision::processors::minicpmo::MiniCPMOProcessor;
use crate::vision::processors::molmo::{MolmoImageTokens, MolmoProcessor};
use crate::vision::processors::molmo2::Molmo2Processor;
use crate::vision::processors::moondream3::Moondream3Processor;
use mlxcel_core::generate::BatchedDecodeKind;
use mlxcel_core::layers::Linear;
use mlxcel_core::utils::GeluActivation;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const HIDDEN: i32 = 4;

fn filled(shape: &[i32], phase: usize) -> UniquePtr<MlxArray> {
    let len: i32 = shape.iter().product();
    let values = (0..len)
        .map(|index| ((index as usize + phase) as f32).sin() * 0.2)
        .collect::<Vec<_>>();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let len: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; len as usize], shape)
}

fn insert_linear(weights: &mut WeightMap, prefix: &str, output: i32, input: i32, phase: usize) {
    weights.insert(format!("{prefix}.weight"), filled(&[output, input], phase));
}

fn insert_layer_norm(weights: &mut WeightMap, prefix: &str, width: i32) {
    weights.insert(format!("{prefix}.weight"), ones(&[width]));
    weights.insert(
        format!("{prefix}.bias"),
        mlxcel_core::zeros(&[width], mlxcel_core::dtype::FLOAT32),
    );
}

fn insert_molmo2_vit(weights: &mut WeightMap, prefix: &str, layers: usize) {
    insert_linear(
        weights,
        &format!("{prefix}.patch_embedding"),
        HIDDEN,
        HIDDEN,
        1,
    );
    weights.insert(
        format!("{prefix}.positional_embedding"),
        filled(&[1, HIDDEN], 2),
    );
    for layer in 0..layers {
        let block = format!("{prefix}.transformer.{layer}");
        for projection in ["wq", "wk", "wv", "wo"] {
            insert_linear(
                weights,
                &format!("{block}.attention.{projection}"),
                HIDDEN,
                HIDDEN,
                3 + layer,
            );
        }
        insert_linear(
            weights,
            &format!("{block}.feed_forward.w1"),
            6,
            HIDDEN,
            7 + layer,
        );
        insert_linear(
            weights,
            &format!("{block}.feed_forward.w2"),
            HIDDEN,
            6,
            11 + layer,
        );
        insert_layer_norm(weights, &format!("{block}.attention_norm"), HIDDEN);
        insert_layer_norm(weights, &format!("{block}.ffn_norm"), HIDDEN);
    }
}

fn insert_projector(weights: &mut WeightMap, prefix: &str, input: i32, output: i32) {
    insert_linear(weights, &format!("{prefix}.w1"), 6, input, 13);
    insert_linear(weights, &format!("{prefix}.w3"), 6, input, 17);
    insert_linear(weights, &format!("{prefix}.w2"), output, 6, 19);
}

fn tiny_molmo_vision() -> MolmoVisionModel {
    let config = MolmoVisionConfig {
        image_num_layers: 0,
        image_emb_dim: HIDDEN,
        image_num_heads: 1,
        image_num_kv_heads: 1,
        image_head_dim: HIDDEN,
        image_num_pos: 2,
        image_norm_eps: 1e-5,
        image_num_patch: (1, 1),
        image_pooling_h: 1,
        image_pooling_w: 1,
        vit_layers: Vec::new(),
        group_size: 64,
        bits: 4,
    };
    let mut weights = WeightMap::new();
    weights.insert(
        "vision.image_vit.class_embedding".into(),
        filled(&[HIDDEN], 1),
    );
    weights.insert(
        "vision.image_vit.positional_embedding".into(),
        filled(&[2, HIDDEN], 2),
    );
    insert_linear(
        &mut weights,
        "vision.image_vit.patch_embedding",
        HIDDEN,
        14 * 14 * 3,
        3,
    );
    insert_layer_norm(&mut weights, "vision.image_vit.pre_ln", HIDDEN);
    for projection in ["wq", "wk", "wv", "wo"] {
        insert_linear(
            &mut weights,
            &format!("vision.image_pooling_2d.{projection}"),
            HIDDEN,
            HIDDEN,
            5,
        );
    }
    insert_projector(&mut weights, "vision.image_projector", HIDDEN, 8);
    MolmoVisionModel::from_weights(&weights, "vision", config)
        .expect("tiny zero-layer Molmo vision tower builds")
}

fn tiny_molmo2_vision() -> Molmo2VisionModel {
    let mut weights = WeightMap::new();
    insert_molmo2_vit(&mut weights, "vision.image_vit", 1);
    for projection in ["wq", "wk", "wv", "wo"] {
        insert_linear(
            &mut weights,
            &format!("vision.image_pooling_2d.{projection}"),
            HIDDEN,
            HIDDEN,
            23,
        );
    }
    insert_projector(&mut weights, "vision.image_projector", HIDDEN, 8);
    Molmo2VisionModel::from_weights(
        &weights,
        "vision",
        1,
        HIDDEN,
        6,
        1,
        1,
        HIDDEN,
        1,
        1e-5,
        false,
        HIDDEN,
        6,
        8,
        1,
        1,
        HIDDEN,
        false,
        &[0],
        false,
        64,
        4,
    )
    .expect("tiny Molmo2 vision tower builds")
}

fn tiny_molmo_point_parts() -> (
    Molmo2VisionTransformer,
    MolmoPointConnector,
    PointPredictor,
    Linear,
) {
    let mut weights = WeightMap::new();
    insert_molmo2_vit(&mut weights, "vision", 0);
    insert_projector(&mut weights, "connector.image_projector", HIDDEN, 8);
    for projection in ["wq", "wk", "wv"] {
        insert_linear(
            &mut weights,
            &format!("connector.image_pooling_2d.{projection}"),
            HIDDEN,
            HIDDEN,
            29,
        );
    }
    for projection in ["patch_q", "patch_k", "subpatch_q", "subpatch_k"] {
        insert_linear(
            &mut weights,
            &format!("point_predictor.{projection}"),
            HIDDEN,
            HIDDEN,
            31,
        );
    }
    weights.insert(
        "point_predictor.add_no_point_class_embed.vector".into(),
        filled(&[HIDDEN], 37),
    );
    insert_linear(&mut weights, "build_vit_embedding", 8, HIDDEN, 41);

    let vision = Molmo2VisionTransformer::from_weights(
        &weights, "vision", 0, HIDDEN, 6, 1, 1, HIDDEN, 1, 1e-5, false, 64, 4,
    )
    .expect("tiny zero-layer Molmo-Point vision tower builds");
    let connector =
        MolmoPointConnector::from_weights(&weights, "connector", 1, 1, HIDDEN, None, false, 64, 4)
            .expect("tiny Molmo-Point connector builds");
    let predictor = PointPredictor::from_weights(
        &weights,
        "point_predictor",
        1e-5,
        false,
        "none",
        10_000.0,
        HIDDEN,
        false,
    )
    .expect("tiny Molmo-Point predictor builds");
    let embedding = Linear::from_weights(&weights, "build_vit_embedding")
        .expect("tiny Molmo-Point ViT embedding builds");
    (vision, connector, predictor, embedding)
}

fn tiny_moondream_vision() -> Moondream3VisionModel {
    let config: Moondream3VisionConfig = serde_json::from_value(serde_json::json!({
        "enc_dim": 4,
        "enc_patch_size": 1,
        "enc_n_layers": 0,
        "enc_ff_dim": 6,
        "enc_n_heads": 1,
        "proj_out_dim": 8,
        "crop_size": 2,
        "in_channels": 3,
        "max_crops": 1,
        "overlap_margin": 0,
        "proj_inner_dim": 6
    }))
    .expect("tiny Moondream vision config parses");
    let mut weights = WeightMap::new();
    insert_linear(&mut weights, "vision.patch_emb", HIDDEN, 3, 43);
    weights.insert("vision.pos_emb".into(), filled(&[4, HIDDEN], 47));
    insert_layer_norm(&mut weights, "vision.post_ln", HIDDEN);
    insert_linear(&mut weights, "vision.proj_mlp.fc1", 6, HIDDEN, 53);
    insert_linear(&mut weights, "vision.proj_mlp.fc2", 8, 6, 59);
    Moondream3VisionModel::from_weights(&weights, &config)
        .expect("tiny zero-layer Moondream vision tower builds")
}

fn tiny_dots_vision() -> DotsVisionEncoder {
    let config: DotsVisionConfig = serde_json::from_value(serde_json::json!({
        "embed_dim": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "patch_size": 1,
        "temporal_patch_size": 1,
        "spatial_merge_size": 1,
        "rms_norm_eps": 1e-5,
        "post_norm": false
    }))
    .expect("tiny dots.ocr vision config parses");
    let mut weights = WeightMap::new();
    insert_linear(
        &mut weights,
        "vision.patch_embed.patchifier.proj",
        HIDDEN,
        3,
        61,
    );
    weights.insert(
        "vision.patch_embed.patchifier.norm.weight".into(),
        ones(&[HIDDEN]),
    );
    insert_layer_norm(&mut weights, "vision.merger.ln_q", HIDDEN);
    insert_linear(&mut weights, "vision.merger.mlp.0", 6, HIDDEN, 67);
    insert_linear(&mut weights, "vision.merger.mlp.2", 8, 6, 71);
    DotsVisionEncoder::from_weights(&weights, &config, "vision")
        .expect("tiny zero-layer dots.ocr vision tower builds")
}

fn tiny_minicpmo_parts() -> (MiniCPMOVisionModel, MiniCPMOResampler) {
    let config = MiniCPMOVisionConfig {
        hidden_size: HIDDEN as usize,
        intermediate_size: 6,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_channels: 3,
        image_size: 1,
        patch_size: 1,
        layer_norm_eps: 1e-6,
        mlp_activation: GeluActivation::default(),
    };
    let mut weights = WeightMap::new();
    weights.insert(
        "vision.embeddings.patch_embedding.weight".into(),
        filled(&[HIDDEN, 1, 1, 3], 73),
    );
    weights.insert(
        "vision.embeddings.position_embedding.weight".into(),
        filled(&[1, HIDDEN], 79),
    );
    insert_layer_norm(&mut weights, "vision.post_layernorm", HIDDEN);

    weights.insert("resampler.query".into(), filled(&[1, HIDDEN], 83));
    insert_linear(&mut weights, "resampler.kv_proj", HIDDEN, HIDDEN, 89);
    for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        insert_linear(
            &mut weights,
            &format!("resampler.attn.{projection}"),
            HIDDEN,
            HIDDEN,
            97,
        );
    }
    for norm in ["ln_q", "ln_kv", "ln_post"] {
        insert_layer_norm(&mut weights, &format!("resampler.{norm}"), HIDDEN);
    }
    weights.insert("resampler.proj".into(), filled(&[HIDDEN, 8], 101));

    let vision = MiniCPMOVisionModel::from_weights(&weights, &config, "vision", 64, 4)
        .expect("tiny zero-layer MiniCPM-o vision tower builds");
    let resampler =
        MiniCPMOResampler::from_weights(&weights, "resampler", HIDDEN as usize, 1, 64, 4)
            .expect("tiny MiniCPM-o resampler builds");
    (vision, resampler)
}

fn assert_native_wrapper<M: LanguageModel>(model: &M) {
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
fn molmo_wrapper_delegates_native_batched_decode_to_text_tower() {
    let model = MolmoVLModel {
        text_model: tiny_molmo_model(),
        vision_tower: tiny_molmo_vision(),
        processor: MolmoProcessor::new(
            1,
            None,
            None,
            None,
            None,
            None,
            None,
            MolmoImageTokens::default(),
        ),
    };
    assert_native_wrapper(&model);
}

#[test]
fn molmo2_wrapper_delegates_native_batched_decode_to_text_tower() {
    let model = Molmo2VLModel {
        text_model: tiny_molmo2_model(),
        vision_tower: tiny_molmo2_vision(),
        processor: Molmo2Processor::new(1, None, None, None, None),
        image_patch_id: 20,
        image_end_token_id: 21,
    };
    assert_native_wrapper(&model);
}

#[test]
fn molmo_point_wrapper_delegates_native_batched_decode_to_text_tower() {
    let (vision_model, connector, point_predictor, build_vit_embedding) = tiny_molmo_point_parts();
    let model = MolmoPointVLModel {
        language_model: tiny_molmo_point_language_model(),
        vision_model,
        connector,
        point_predictor,
        build_vit_embedding,
        processor: Molmo2Processor::new(1, None, None, None, None),
        config: MolmoPointConfig {
            image_patch_id: 20,
            image_end_token_id: 21,
            image_non_indexable_patch_id: 22,
            patch_token_id: 23,
            subpatch_token_id: 24,
            location_token_id: 25,
            no_more_points_class: true,
            norm_logits: false,
            patch_location: None,
            vit_layers: Vec::new(),
            hidden_size: 8,
        },
        vit_layers: Vec::new(),
    };
    assert_native_wrapper(&model);
}

#[test]
fn moondream2_wrapper_delegates_native_batched_decode_to_text_tower() {
    let model = Moondream2VLModel {
        text_model: tiny_moondream2_model(),
        vision_tower: tiny_moondream_vision(),
        processor: Moondream3Processor::new(2, 1, 1, 0),
        eos_token_ids: vec![2],
        prompt_style: crate::moondream2_prompt::Moondream2PromptStyle::LegacyQuestionAnswer,
    };
    assert_native_wrapper(&model);
}

#[test]
fn moondream3_wrapper_delegates_native_batched_decode_to_text_tower() {
    let model = Moondream3VLModel {
        text_model: tiny_moondream3_model(),
        vision_tower: tiny_moondream_vision(),
        processor: Moondream3Processor::new(2, 1, 1, 0),
        eos_token_ids: vec![2],
    };
    assert_native_wrapper(&model);
}

#[test]
fn jina_vlm_wrapper_delegates_native_batched_decode_to_text_tower() {
    let (_, vision_model) = crate::vision::encoders::jina_vlm::tests::build_tiny_vision_model();
    let model = JinaVlmModel {
        text_model: tiny_jina_vlm_text_model(),
        vision_model,
        processor: JinaVlmProcessor::default(),
        image_prompt_token_id: 20,
        always_start_with_space: true,
    };
    assert_native_wrapper(&model);
}

#[test]
fn dots_ocr_wrapper_delegates_native_batched_decode_to_text_tower() {
    let model = DotsOcrVlModel {
        text_model: tiny_qwen2_model(),
        vision_encoder: tiny_dots_vision(),
        processor: DotsOcrProcessor::default(),
        image_token_id: 20,
        video_token_id: 21,
        vision_start_token_id: 22,
        spatial_merge_size: 1,
        eos_token_ids: vec![2],
    };
    assert_native_wrapper(&model);
}

#[test]
fn minicpmo_wrapper_delegates_native_batched_decode_to_text_tower() {
    let (vision_tower, resampler) = tiny_minicpmo_parts();
    let model = MiniCPMOVLModel {
        text_model: tiny_qwen3_model(),
        vision_tower,
        resampler,
        processor: MiniCPMOProcessor::new(1, 2, 1),
        eos_token_ids: vec![2],
    };
    assert_native_wrapper(&model);
}
