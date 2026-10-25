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

use crate::models::deepseek::{DeepSeekModel, ModelArgs};
use crate::vision::deepseekocr::DeepSeekOcrVlModel;
use crate::vision::deepseekocr_2::DeepSeekOcr2VlModel;
use crate::vision::encoders::deepseekocr_clip::{ClipConfig, ClipEncoder};
use crate::vision::encoders::deepseekocr_qwen2::{Qwen2Resampler, Qwen2ResamplerConfig};
use crate::vision::encoders::deepseekocr_sam::{SamConfig, SamEncoder};
use crate::vision::processors::deepseekocr::DeepSeekOcrProcessor;
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

fn array(shape: &[i32], value: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![value; n as usize], shape)
}

fn minimal_sam(final_out_chans: i32) -> SamEncoder {
    let mut weights = WeightMap::new();
    let prefix = "sam_model";
    weights.insert(
        format!("{prefix}.patch_embed.proj.weight"),
        array(&[1, 1, 1, 1], 0.0),
    );
    weights.insert(format!("{prefix}.patch_embed.proj.bias"), array(&[1], 0.0));
    weights.insert(format!("{prefix}.pos_embed"), array(&[1, 1, 1, 1], 0.0));
    weights.insert(format!("{prefix}.neck.0.weight"), array(&[1, 1, 1, 1], 0.0));
    weights.insert(format!("{prefix}.neck.1.weight"), array(&[1], 1.0));
    weights.insert(format!("{prefix}.neck.2.weight"), array(&[1, 1, 1, 1], 0.0));
    weights.insert(format!("{prefix}.neck.3.weight"), array(&[1], 1.0));
    weights.insert(format!("{prefix}.net_2.weight"), array(&[1, 1, 1, 1], 0.0));
    weights.insert(
        format!("{prefix}.net_3.weight"),
        array(&[final_out_chans, 1, 1, 1], 0.0),
    );
    SamEncoder::from_weights(
        &weights,
        prefix,
        SamConfig {
            embed_dim: 1,
            num_heads: 1,
            depth: 0,
            window_size: 0,
            global_attn_indexes: vec![],
            out_chans: 1,
            final_out_chans,
            grid: 1,
        },
    )
    .unwrap()
}

fn minimal_clip() -> ClipEncoder {
    let mut weights = WeightMap::new();
    let prefix = "vision_model";
    weights.insert(
        format!("{prefix}.embeddings.class_embedding"),
        array(&[1], 0.0),
    );
    weights.insert(
        format!("{prefix}.embeddings.position_embedding.weight"),
        array(&[1, 1], 0.0),
    );
    weights.insert(format!("{prefix}.pre_layrnorm.weight"), array(&[1], 1.0));
    weights.insert(format!("{prefix}.pre_layrnorm.bias"), array(&[1], 0.0));
    ClipEncoder::from_weights(
        &weights,
        prefix,
        ClipConfig {
            hidden_size: 1,
            num_heads: 1,
            num_layers: 0,
            layer_norm_eps: 1e-6,
            pos_grid: 1,
        },
    )
    .unwrap()
}

fn minimal_resampler() -> Qwen2Resampler {
    let mut weights = WeightMap::new();
    let prefix = "vision_model.qwen2_encoder";
    weights.insert(format!("{prefix}.query_1024"), array(&[1, 1], 0.0));
    weights.insert(format!("{prefix}.query_768"), array(&[1, 1], 0.0));
    weights.insert(format!("{prefix}.norm.weight"), array(&[1], 1.0));
    Qwen2Resampler::from_weights(
        &weights,
        prefix,
        Qwen2ResamplerConfig {
            dim: 1,
            num_layers: 0,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            intermediate: 1,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            query_len_global: 1,
            query_len_tile: 1,
        },
    )
    .unwrap()
}

fn minimal_projector() -> UnifiedLinear {
    let mut weights = WeightMap::new();
    weights.insert("projector.weight".into(), array(&[1, 1], 1.0));
    UnifiedLinear::from_weights(&weights, "projector", 64, 4).unwrap()
}

fn minimal_text_model() -> DeepSeekModel {
    let args = ModelArgs {
        model_type: "deepseek".to_string(),
        vocab_size: 16,
        hidden_size: 2,
        intermediate_size: 2,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        max_position_embeddings: 16,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        moe_intermediate_size: None,
        n_shared_experts: None,
        n_routed_experts: None,
        num_experts_per_tok: None,
        moe_layer_freq: 1,
        first_k_dense_replace: 0,
        routed_scaling_factor: 1.0,
        attention_bias: false,
        quantization: None,
        group_size: None,
        bits: None,
    };
    let mut weights = WeightMap::new();
    let embedding: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.01).collect();
    weights.insert(
        "model.embed_tokens.weight".into(),
        mlxcel_core::from_slice_f32(&embedding, &[16, 2]),
    );
    weights.insert("model.norm.weight".into(), array(&[2], 1.0));
    weights.insert("lm_head.weight".into(), array(&[16, 2], 0.05));
    DeepSeekModel::from_weights(&weights, &args).unwrap()
}

fn assert_delegating_contract<M: LanguageModel>(model: &M) {
    assert_eq!(model.batched_decode_kind(), BatchedDecodeKind::Native);
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        model,
        &[&[1, 2, 3], &[4, 5], &[6]],
        &[7, 8, 9],
        1e-5,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        model,
        &[1, 3, 5],
        7,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(model);
}

#[test]
fn deepseek_ocr_delegates_actual_batched_decode_to_text_model() {
    let model = DeepSeekOcrVlModel {
        text_model: minimal_text_model(),
        sam: minimal_sam(1),
        clip: minimal_clip(),
        projector: minimal_projector(),
        image_newline: array(&[1], 0.0),
        view_separator: array(&[1], 0.0),
        processor: DeepSeekOcrProcessor::default(),
        image_token_id: 15,
        eos_token_id: 1,
        n_embed: 1,
    };
    assert_delegating_contract(&model);
}

#[test]
fn deepseek_ocr2_delegates_actual_batched_decode_to_text_model() {
    let model = DeepSeekOcr2VlModel {
        text_model: minimal_text_model(),
        sam: minimal_sam(1),
        resampler: minimal_resampler(),
        projector: minimal_projector(),
        view_separator: array(&[1], 0.0),
        processor: DeepSeekOcrProcessor::v2(),
        image_token_id: 15,
        eos_token_id: 1,
        n_embed: 1,
    };
    assert_delegating_contract(&model);
}
