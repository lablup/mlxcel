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

//! Coverage of the `last_row_logits` hook (#3070) for the VLM-internal text
//! models outside the Qwen and GLM families, and for their wrappers. Each
//! case builds a tiny random-weight model and runs the shared harness
//! (`last_row_logits_harness`), which asserts the hook is taken and returns
//! `[1, 1, vocab]` logits equal to the full forward's row, on the text path
//! and on the embeddings path.
//!
//! The Youtu-VL text model runs the same harness from `youtu_vl_lm_tests`,
//! where its fixture lives.

use serde_json::json;

use super::molmo_point_vl::MolmoPointConfig;
use super::vlm_family_batched_decode_tests::{
    qwen3_processor, tiny_ernie45_moe_vl_wrapper, tiny_hunyuan_vl_wrapper,
    tiny_paddleocr_vl_wrapper, unused_qwen3_vision,
};
use super::wrapper_native_batched_decode_tests::{
    tiny_molmo_point_parts, tiny_molmo_vision, tiny_molmo2_vision, tiny_moondream_vision,
};
use super::{
    CohereCompassModel, FalconOcrVlModel, JinaVlmModel, Molmo2VLModel, MolmoPointVLModel,
    MolmoVLModel, Moondream2VLModel, Moondream3VLModel,
};
use crate::models::last_row_logits_harness::last_row_case;
use crate::models::vlm_family_batched_decode_test_support::{
    tiny_cohere_compass_full_mrope_model, tiny_cohere_compass_model,
    tiny_cohere_compass_sliding_nope_model, tiny_ernie45_moe_vl_model, tiny_hunyuan_vl_model,
    tiny_paddleocr_model,
};
use crate::models::wrapper_batched_decode_test_support::{
    tiny_jina_vlm_text_model, tiny_molmo_model, tiny_molmo_point_language_model, tiny_molmo2_model,
    tiny_moondream2_model, tiny_moondream3_model,
};
use crate::models::{FalconOcrConfig, FalconOcrTextModel};
use crate::vision::processors::falcon_ocr::FalconOcrProcessor;
use crate::vision::processors::jina_vlm::JinaVlmProcessor;
use crate::vision::processors::molmo::{MolmoImageTokens, MolmoProcessor};
use crate::vision::processors::molmo2::Molmo2Processor;
use crate::vision::processors::moondream3::Moondream3Processor;
use mlxcel_core::weights::WeightMap;

// Text models.

last_row_case!(
    hunyuan_vl_text_last_row_logits,
    tiny_hunyuan_vl_model,
    embeddings
);
last_row_case!(
    paddleocr_vl_text_last_row_logits,
    tiny_paddleocr_model,
    embeddings
);
last_row_case!(
    ernie45_moe_vl_text_last_row_logits,
    tiny_ernie45_moe_vl_model,
    embeddings
);
last_row_case!(
    cohere_compass_text_last_row_logits,
    tiny_cohere_compass_model,
    embeddings
);
last_row_case!(
    cohere_compass_sliding_nope_text_last_row_logits,
    tiny_cohere_compass_sliding_nope_model,
    embeddings
);
last_row_case!(
    cohere_compass_full_mrope_text_last_row_logits,
    tiny_cohere_compass_full_mrope_model,
    embeddings
);
last_row_case!(
    jina_vlm_text_last_row_logits,
    tiny_jina_vlm_text_model,
    embeddings
);
last_row_case!(molmo_text_last_row_logits, tiny_molmo_model, embeddings);
last_row_case!(molmo2_text_last_row_logits, tiny_molmo2_model, embeddings);
last_row_case!(
    moondream2_text_last_row_logits,
    tiny_moondream2_model,
    embeddings
);
last_row_case!(
    moondream3_text_last_row_logits,
    tiny_moondream3_model,
    embeddings
);

/// One-layer Falcon-OCR text model: fused QKV with attention sinks, a
/// weightless pre-norm stack and the final weighted norm before `output`.
fn tiny_falcon_ocr_text_model() -> FalconOcrTextModel {
    const DIM: i32 = 8;
    const VOCAB: i32 = 32;
    const FFN: i32 = 16;
    let cfg: FalconOcrConfig = serde_json::from_value(json!({
        "dim": DIM, "n_layers": 1, "n_heads": 2, "head_dim": 4, "n_kv_heads": 1,
        "vocab_size": VOCAB, "ffn_dim": FFN, "spatial_patch_size": 1
    }))
    .expect("tiny Falcon-OCR config parses");
    let mut weights = WeightMap::new();
    let mut put = |name: &str, shape: &[i32]| {
        let len: i32 = shape.iter().product();
        let values: Vec<f32> = (0..len)
            .map(|i| ((i * 37 + name.len() as i32 * 11) % 23) as f32 * 0.02 - 0.2)
            .collect();
        weights.insert(
            name.to_string(),
            mlxcel_core::from_slice_f32(&values, shape),
        );
    };
    put("tok_embeddings.weight", &[VOCAB, DIM]);
    put("img_projector.weight", &[DIM, 3]);
    // Two query heads and one key/value head, each four wide.
    put("layers.0.attention.wqkv.weight", &[16, DIM]);
    put("layers.0.attention.wo.weight", &[DIM, DIM]);
    put("layers.0.attention.sinks", &[2]);
    put("layers.0.feed_forward.w1.weight", &[FFN, DIM]);
    put("layers.0.feed_forward.w3.weight", &[FFN, DIM]);
    put("layers.0.feed_forward.w2.weight", &[DIM, FFN]);
    put("norm.weight", &[DIM]);
    put("output.weight", &[VOCAB, DIM]);
    put("freqs_cis_golden", &[2, 1, 2]);
    FalconOcrTextModel::from_weights(&weights, &cfg).expect("tiny Falcon-OCR builds")
}

last_row_case!(
    falcon_ocr_text_last_row_logits,
    tiny_falcon_ocr_text_model,
    embeddings
);

// Wrappers. Prefill never touches the vision-only fields, so each fixture
// carries the minimal tower its struct needs.

last_row_case!(
    hunyuan_vl_wrapper_last_row_logits,
    tiny_hunyuan_vl_wrapper,
    embeddings
);
last_row_case!(
    paddleocr_vl_wrapper_last_row_logits,
    tiny_paddleocr_vl_wrapper,
    embeddings
);
last_row_case!(
    ernie45_moe_vl_wrapper_last_row_logits,
    tiny_ernie45_moe_vl_wrapper,
    embeddings
);

fn tiny_cohere_compass_wrapper() -> CohereCompassModel {
    CohereCompassModel {
        text_model: tiny_cohere_compass_model(),
        vision_encoder: unused_qwen3_vision(),
        processor: qwen3_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}

last_row_case!(
    cohere_compass_wrapper_last_row_logits,
    tiny_cohere_compass_wrapper,
    embeddings
);

fn tiny_falcon_ocr_wrapper() -> FalconOcrVlModel {
    FalconOcrVlModel {
        text_model: tiny_falcon_ocr_text_model(),
        processor: FalconOcrProcessor::new(1, 3),
        ocr_task_token_id: None,
        eos_token_ids: vec![11],
    }
}

last_row_case!(
    falcon_ocr_wrapper_last_row_logits,
    tiny_falcon_ocr_wrapper,
    embeddings
);

fn tiny_jina_vlm_wrapper() -> JinaVlmModel {
    let (_, vision_model) = crate::vision::encoders::jina_vlm::tests::build_tiny_vision_model();
    JinaVlmModel {
        text_model: tiny_jina_vlm_text_model(),
        vision_model,
        processor: JinaVlmProcessor::default(),
        image_prompt_token_id: 20,
        always_start_with_space: true,
    }
}

last_row_case!(
    jina_vlm_wrapper_last_row_logits,
    tiny_jina_vlm_wrapper,
    embeddings
);

fn tiny_molmo_wrapper() -> MolmoVLModel {
    MolmoVLModel {
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
    }
}

last_row_case!(
    molmo_wrapper_last_row_logits,
    tiny_molmo_wrapper,
    embeddings
);

fn tiny_molmo2_wrapper() -> Molmo2VLModel {
    Molmo2VLModel {
        text_model: tiny_molmo2_model(),
        vision_tower: tiny_molmo2_vision(),
        processor: Molmo2Processor::new(1, None, None, None, None),
        image_patch_id: 20,
        image_end_token_id: 21,
    }
}

last_row_case!(
    molmo2_wrapper_last_row_logits,
    tiny_molmo2_wrapper,
    embeddings
);

/// Molmo-Point's text model is not a `LanguageModel` on its own, so this
/// wrapper case is the coverage of `MolmoPointLanguageModel::last_row_logits`
/// (pre-norm row, final norm, extended head).
fn tiny_molmo_point_wrapper() -> MolmoPointVLModel {
    let (vision_model, connector, point_predictor, build_vit_embedding) = tiny_molmo_point_parts();
    MolmoPointVLModel {
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
    }
}

last_row_case!(
    molmo_point_wrapper_last_row_logits,
    tiny_molmo_point_wrapper,
    embeddings
);

fn tiny_moondream2_wrapper() -> Moondream2VLModel {
    Moondream2VLModel {
        text_model: tiny_moondream2_model(),
        vision_tower: tiny_moondream_vision(),
        processor: Moondream3Processor::new(2, 1, 1, 0),
        eos_token_ids: vec![2],
        prompt_style: crate::moondream2_prompt::Moondream2PromptStyle::LegacyQuestionAnswer,
    }
}

last_row_case!(
    moondream2_wrapper_last_row_logits,
    tiny_moondream2_wrapper,
    embeddings
);

fn tiny_moondream3_wrapper() -> Moondream3VLModel {
    Moondream3VLModel {
        text_model: tiny_moondream3_model(),
        vision_tower: tiny_moondream_vision(),
        processor: Moondream3Processor::new(2, 1, 1, 0),
        eos_token_ids: vec![2],
    }
}

last_row_case!(
    moondream3_wrapper_last_row_logits,
    tiny_moondream3_wrapper,
    embeddings
);
