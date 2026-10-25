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

//! Runtime parity for the issue #2564 VLM-owned text towers.

use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel};
use mlxcel_core::layers::KVCache;

use super::wrapper_batched_decode_test_support::{
    tiny_jina_vlm_text_model, tiny_molmo_model, tiny_molmo_point_language_model, tiny_molmo2_model,
    tiny_moondream2_model, tiny_moondream3_model, tiny_moondream3_model_with_router,
};

const PROMPTS: &[&[i32]] = &[&[1, 2], &[3, 4, 5, 6], &[7, 8, 9, 10, 11, 12]];
const NEXT: &[i32] = &[13, 14, 15];

fn assert_tower<M: LanguageModel>(model: &M) {
    assert_eq!(model.batched_decode_kind(), BatchedDecodeKind::Native);
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        model, PROMPTS, NEXT, 1e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        model,
        &[1, 3, 5],
        7,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(model);
}

#[test]
fn molmo_tower_runs_native_batched_decode() {
    assert_tower(&tiny_molmo_model());
}

#[test]
fn molmo2_tower_runs_native_batched_decode_with_per_head_qk_norm() {
    assert_tower(&tiny_molmo2_model());
}

#[test]
fn moondream2_tower_runs_native_batched_decode_with_partial_rope() {
    assert_tower(&tiny_moondream2_model());
}

#[test]
fn moondream3_tower_runs_native_batched_decode_when_rows_share_an_expert() {
    let model = tiny_moondream3_model();
    let tokens = mlxcel_core::from_slice_i32(NEXT, &[NEXT.len() as i32, 1]);
    let selected = experts_as_i32(&model.first_layer_selected_experts(&tokens));
    assert!(selected.windows(2).all(|pair| pair[0] == pair[1]));
    assert_tower(&model);
}

#[test]
fn moondream3_tower_runs_native_batched_decode_when_rows_split_across_experts() {
    let model = tiny_moondream3_model_with_router(true);
    let tokens = mlxcel_core::from_slice_i32(NEXT, &[NEXT.len() as i32, 1]);
    let selected = experts_as_i32(&model.first_layer_selected_experts(&tokens));
    assert!(
        selected.windows(2).any(|pair| pair[0] != pair[1]),
        "engineered rows must route to disjoint experts: {selected:?}"
    );
    assert_tower(&model);
}

fn experts_as_i32(experts: &mlxcel_core::MlxArray) -> Vec<i32> {
    mlxcel_core::eval(experts);
    mlxcel_core::array_to_raw_bytes(experts)
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four-byte expert index")))
        .collect()
}

#[test]
fn jina_vlm_tower_runs_native_batched_decode_with_per_head_qk_norm() {
    assert_tower(&tiny_jina_vlm_text_model());
}

fn point_prefill(
    model: &crate::models::molmo_point::MolmoPointLanguageModel,
    prompt: &[i32],
) -> Vec<KVCache> {
    let mut caches = model.make_caches();
    let input = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = model.forward(&input, &mut caches, None);
    mlxcel_core::eval(&logits);
    caches
}

#[test]
fn molmo_point_batched_transformer_preserves_pre_ln_and_extended_head_semantics() {
    let model = tiny_molmo_point_language_model();

    let expected: Vec<_> = PROMPTS
        .iter()
        .zip(NEXT)
        .map(|(prompt, &token)| {
            let mut caches = point_prefill(&model, prompt);
            let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            let (post, pre) =
                model
                    .model
                    .forward_with_pre_ln(Some(&input), None, &mut caches, None);
            mlxcel_core::eval(&post);
            mlxcel_core::eval(&pre);
            (post, pre)
        })
        .collect();

    let mut caches: Vec<_> = PROMPTS
        .iter()
        .map(|prompt| point_prefill(&model, prompt))
        .collect();
    let input = mlxcel_core::from_slice_i32(NEXT, &[NEXT.len() as i32, 1]);
    let (post, pre) = {
        let mut rows: Vec<&mut [KVCache]> = caches.iter_mut().map(Vec::as_mut_slice).collect();
        model
            .model
            .forward_batched_with_pre_ln(&input, &mut rows, None)
    };
    let logits = model.lm_head.forward(&post);
    mlxcel_core::eval(&pre);
    mlxcel_core::eval(&logits);

    let shape = mlxcel_core::array_shape(&logits);
    for (row, (expected_post, expected_pre)) in expected.iter().enumerate() {
        let actual_post = mlxcel_core::slice(&post, &[row as i32, 0, 0], &[row as i32 + 1, 1, 8]);
        let actual_pre = mlxcel_core::slice(&pre, &[row as i32, 0, 0], &[row as i32 + 1, 1, 8]);
        assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
            &actual_post,
            expected_post,
            1e-4,
            1e-4,
        )));
        assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
            &actual_pre,
            expected_pre,
            1e-4,
            1e-4,
        )));
        let expected_logits = model.lm_head.forward(expected_post);
        let actual_logits =
            mlxcel_core::slice(&logits, &[row as i32, 0, 0], &[row as i32 + 1, 1, shape[2]]);
        assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
            &actual_logits,
            &expected_logits,
            1e-4,
            1e-4,
        )));
    }
}
