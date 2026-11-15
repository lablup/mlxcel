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

//! VLM text-model and wrapper coverage of the `last_row_logits` hook (#3070): each case builds
//! a tiny random-weight model and runs the shared harness
//! (`last_row_logits_harness`), which asserts the hook is taken and returns
//! `[1, 1, vocab]` logits equal to the full forward's row.

use crate::LanguageModel;
use crate::models::embedding_test_support::mlx_test_guard;
use crate::models::last_row_logits_harness::TOKENS;
use crate::models::last_row_logits_harness::last_row_case;
use crate::models::qwen_vl_batched_decode_test_support::RowPositionState;
use crate::models::qwen2_vl::batched_tests::tiny_qwen2_vl_model;
use crate::models::qwen3_vl::qwen3_vl_tests::{set_qwen3_state, tiny_qwen3_vl_model};
use crate::models::qwen3_vl_moe::tests::{set_qwen3_moe_state, tiny_qwen3_vl_moe_model};
use crate::models::vlm_family_batched_decode_test_support::{
    tiny_glm4v_model, tiny_glm4v_moe_model,
};
use crate::vision::qwen_vl_batched_decode_tests as qwen_fixtures;
use crate::vision::vlm_family_batched_decode_tests as family_fixtures;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::logits_at_position;
use mlxcel_core::{MlxArray, UniquePtr};

fn tiny_qwen2_vl_wrapper() -> super::qwen2_vl::Qwen2VLModel {
    super::qwen2_vl::Qwen2VLModel {
        text_model: tiny_qwen2_vl_model(),
        vision_encoder: None,
        text_only_path: Some("synthetic".into()),
        processor: qwen_fixtures::qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}

fn tiny_qwen2_5_vl_wrapper() -> super::qwen2_5_vl::Qwen25VLModel {
    super::qwen2_5_vl::Qwen25VLModel {
        text_model: tiny_qwen2_vl_model(),
        vision_encoder: None,
        text_only_path: Some("synthetic".into()),
        processor: qwen_fixtures::qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}

fn tiny_qwen3_vl_wrapper() -> super::qwen3_vl::Qwen3VLModel {
    super::qwen3_vl::Qwen3VLModel {
        text_model: tiny_qwen3_vl_model(),
        vision_encoder: None,
        text_only_path: Some("synthetic".into()),
        processor: qwen_fixtures::qwen3_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}

fn tiny_qwen3_vl_moe_wrapper() -> super::qwen3_vl_moe::Qwen3VLMoeModel {
    super::qwen3_vl_moe::Qwen3VLMoeModel {
        text_model: tiny_qwen3_vl_moe_model(),
        vision_encoder: None,
        text_only_path: Some("synthetic".into()),
        processor: qwen_fixtures::qwen3_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}

fn tiny_qwen3_omni_wrapper() -> super::qwen3_omni_moe::Qwen3OmniMoeModel {
    super::qwen3_omni_moe::Qwen3OmniMoeModel {
        text_model: tiny_qwen3_vl_moe_model(),
        vision_encoder: qwen_fixtures::unused_qwen3_vision_encoder(),
        audio_encoder: qwen_fixtures::unused_qwen3_audio_encoder(),
        processor: qwen_fixtures::qwen3_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        audio_token_id: 27,
        audio_start_token_id: 26,
        audio_end_token_id: 25,
        im_end_token_id: 24,
        spatial_merge_size: 1,
    }
}

fn tiny_glm4v_wrapper() -> super::glm4v::Glm4vModel {
    super::glm4v::Glm4vModel {
        text_model: tiny_glm4v_model(),
        vision_encoder: family_fixtures::unused_glm4v_vision(),
        processor: family_fixtures::qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}

fn tiny_glm_ocr_wrapper() -> super::glm_ocr::GlmOcrModel {
    super::glm_ocr::GlmOcrModel {
        text_model: tiny_glm4v_model(),
        vision_encoder: family_fixtures::unused_glm_ocr_vision(),
        processor: family_fixtures::qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}

fn tiny_glm4v_moe_wrapper() -> super::glm4v_moe::Glm4vMoeModel {
    super::glm4v_moe::Glm4vMoeModel {
        text_model: tiny_glm4v_moe_model(),
        vision_encoder: family_fixtures::unused_glm4v_vision(),
        processor: family_fixtures::qwen_processor(),
        image_token_id: 29,
        video_token_id: 30,
        vision_start_token_id: 28,
        spatial_merge_size: 1,
    }
}

// Text models: the text-only fast path and, through the embeddings case, the
// MRoPE path with sequential positions.
last_row_case!(
    qwen2_vl_text_last_row_logits,
    tiny_qwen2_vl_model,
    embeddings
);
last_row_case!(
    qwen3_vl_text_last_row_logits,
    tiny_qwen3_vl_model,
    embeddings
);
last_row_case!(
    qwen3_vl_moe_text_last_row_logits,
    tiny_qwen3_vl_moe_model,
    embeddings
);
last_row_case!(glm4v_text_last_row_logits, tiny_glm4v_model, embeddings);
last_row_case!(
    glm4v_moe_text_last_row_logits,
    tiny_glm4v_moe_model,
    embeddings
);

// Wrappers: the engine reaches the same last-row forward through each one.
last_row_case!(
    qwen2_vl_wrapper_last_row_logits,
    tiny_qwen2_vl_wrapper,
    embeddings
);
last_row_case!(
    qwen2_5_vl_wrapper_last_row_logits,
    tiny_qwen2_5_vl_wrapper,
    embeddings
);
last_row_case!(
    qwen3_vl_wrapper_last_row_logits,
    tiny_qwen3_vl_wrapper,
    embeddings
);
last_row_case!(
    qwen3_vl_moe_wrapper_last_row_logits,
    tiny_qwen3_vl_moe_wrapper,
    embeddings
);
last_row_case!(
    qwen3_omni_wrapper_last_row_logits,
    tiny_qwen3_omni_wrapper,
    embeddings
);
last_row_case!(
    glm4v_wrapper_last_row_logits,
    tiny_glm4v_wrapper,
    embeddings
);
last_row_case!(
    glm_ocr_wrapper_last_row_logits,
    tiny_glm_ocr_wrapper,
    embeddings
);
last_row_case!(
    glm4v_moe_wrapper_last_row_logits,
    tiny_glm4v_moe_wrapper,
    embeddings
);

fn image_seq() -> SequenceId {
    SequenceId::from_raw(307_100)
}

fn ids() -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(TOKENS, &[1, TOKENS.len() as i32])
}

fn assert_close(actual: &MlxArray, expected: &MlxArray, what: &str) {
    mlxcel_core::eval(actual);
    mlxcel_core::eval(expected);
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected),
        "{what}: shape"
    );
    let close = mlxcel_core::allclose(actual, expected, 1e-4, 1e-4);
    mlxcel_core::eval(&close);
    assert!(mlxcel_core::item_bool(&close), "{what}: logits differ");
}

/// An image prefill: embeddings forwarded under a sequence id whose image
/// state `prepare` installs (stored MRoPE positions, DeepStack features).
/// The last-row forward must resolve that state exactly as the full forward
/// does, and `prepare` must actually change the logits, so the comparison is
/// not the text-only one again.
fn assert_image_prefill_rows<M: LanguageModel>(
    make: impl Fn() -> M,
    prepare: impl Fn(&M, SequenceId, &MlxArray),
) {
    let input = ids();
    let last = TOKENS.len() - 1;
    let embeds = make()
        .embed_tokens(&input)
        .expect("Qwen-VL models expose embed_tokens");
    mlxcel_core::eval(&embeds);
    let full_of = |with_state: bool| {
        let model = make();
        if with_state {
            prepare(&model, image_seq(), &embeds);
        }
        let mut caches = model.make_caches();
        let full = model.forward_with_embeddings_and_sequence_id(
            &input,
            Some(&embeds),
            Some(image_seq()),
            &mut caches,
            None,
        );
        mlxcel_core::eval(&full);
        full
    };
    let full = full_of(true);
    let plain = full_of(false);
    let same = mlxcel_core::allclose(&full, &plain, 1e-4, 1e-4);
    mlxcel_core::eval(&same);
    assert!(
        !mlxcel_core::item_bool(&same),
        "the image state must change the logits for this case to mean anything"
    );

    let model = make();
    prepare(&model, image_seq(), &embeds);
    let mut caches = model.make_caches();
    let row = model
        .last_row_logits(
            &input,
            Some(&embeds),
            Some(image_seq()),
            &mut caches,
            None,
            last,
        )
        .expect("the image prefill must take the last-row forward");
    assert_close(&row, &logits_at_position(&full, last), "last_row_logits");

    let model = make();
    prepare(&model, image_seq(), &embeds);
    let mut caches = model.make_caches();
    let row = model.forward_last_logits_with_embeddings_and_sequence_id(
        &input,
        Some(&embeds),
        Some(image_seq()),
        &mut caches,
        None,
        2,
    );
    assert_close(
        &row,
        &logits_at_position(&full, 2),
        "engine embeddings entry",
    );
}

const STORED: RowPositionState = RowPositionState::Stored { delta: 3 };

/// DeepStack state for an image run at positions 1..=3, one injected layer.
fn deepstack_inputs(embeds: &MlxArray) -> (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) {
    let hidden = mlxcel_core::array_shape(embeds)[2];
    let mask: Vec<f32> = (0..TOKENS.len())
        .map(|i| if (1..=3).contains(&i) { 1.0 } else { 0.0 })
        .collect();
    let mask = mlxcel_core::from_slice_f32(&mask, &[1, TOKENS.len() as i32]);
    let mask = mlxcel_core::astype(&mask, mlxcel_core::dtype::BOOL);
    let values: Vec<f32> = (0..3 * hidden)
        .map(|i| 0.5 + (i % 7) as f32 * 0.25)
        .collect();
    let visual = mlxcel_core::from_slice_f32(&values, &[3, hidden]);
    (mask, vec![visual])
}

/// Stored `[3, 1, len]` MRoPE positions of an image grid: the three axes
/// advance differently, so they are not a uniform shift that RoPE's relative
/// encoding would cancel.
fn image_positions() -> UniquePtr<MlxArray> {
    let len = TOKENS.len() as i32 + 1;
    let mut values = Vec::with_capacity(3 * len as usize);
    values.extend((0..len).map(|position| position * 2));
    values.extend((0..len).map(|position| position / 3));
    values.extend((0..len).map(|position| (position % 3) * 2));
    mlxcel_core::from_slice_i32(&values, &[3, 1, len])
}

#[test]
fn qwen2_vl_image_prefill_last_row_uses_stored_mrope_positions() {
    let _guard = mlx_test_guard();
    assert_image_prefill_rows(tiny_qwen2_vl_wrapper, |model, seq, _| {
        model
            .text_model
            .set_mrope_state_for_sequence(seq, image_positions(), 3);
    });
}

#[test]
fn qwen3_vl_image_prefill_last_row_uses_mrope_and_deepstack_state() {
    let _guard = mlx_test_guard();
    assert_image_prefill_rows(tiny_qwen3_vl_wrapper, |model, seq, embeds| {
        set_qwen3_state(&model.text_model, seq, STORED, TOKENS.len());
        let (mask, visual) = deepstack_inputs(embeds);
        model.text_model.set_deepstack_state(mask, visual);
    });
}

#[test]
fn qwen3_vl_moe_image_prefill_last_row_uses_mrope_and_deepstack_state() {
    let _guard = mlx_test_guard();
    assert_image_prefill_rows(tiny_qwen3_vl_moe_wrapper, |model, seq, embeds| {
        set_qwen3_moe_state(&model.text_model, seq, STORED, TOKENS.len());
        let (mask, visual) = deepstack_inputs(embeds);
        model.text_model.set_deepstack_state(mask, visual);
    });
}

#[test]
fn glm4v_image_prefill_last_row_uses_stored_mrope_positions() {
    let _guard = mlx_test_guard();
    assert_image_prefill_rows(tiny_glm4v_wrapper, |model, seq, _| {
        model.text_model.set_mrope_state(image_positions(), 3);
        model.text_model.bind_mrope_state_to_sequence(seq);
    });
}
