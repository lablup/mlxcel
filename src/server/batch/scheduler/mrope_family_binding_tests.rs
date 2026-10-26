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

//! Issue #2634: PaddleOCR-VL, ERNIE-4.5-MoE-VL and Hunyuan-VL must own their
//! MRoPE state per sequence in the server, like the Qwen VL family.
//!
//! Their vision wrappers write the request's MRoPE positions and rope delta
//! into the text model's fallback slot during media preparation. Unless
//! admission binds that slot to the request's sequence id, every in-flight
//! request decodes from whichever request was prepared last, and a text-only
//! request inherits the previous image request's positions.
//!
//! The tests seed the fallback slot directly, standing in for the media
//! preparation, and drive the `LoadedModel` routing used by admission
//! (bind), preemption (take and install) and admission aborts (clear), plus
//! the real `enqueue_request` flow. One test per family pins the CLI
//! contract: an unbound sequence id still reads the fallback.

use super::vlm_fallback_abort_tests::{
    enqueue, exhaust_pool, expect_error, queued_seq_id, scheduler,
};
use super::*;
use crate::models::qwen_mrope_state::MRopeEntry;
use crate::vision::vlm_family_batched_decode_tests::{
    tiny_ernie45_moe_vl_wrapper, tiny_hunyuan_vl_wrapper, tiny_paddleocr_vl_wrapper,
};

/// Prefill length of the seeded image request.
const PREFILL_LEN: i32 = 4;

/// Distinct `[axes, 1, PREFILL_LEN]` position ids, tagged by `tag` so two
/// requests' tensors can be told apart.
fn positions(axes: i32, tag: i32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let values: Vec<i32> = (0..axes * PREFILL_LEN)
        .map(|index| index + tag * 100)
        .collect();
    mlxcel_core::from_slice_i32(&values, &[axes, 1, PREFILL_LEN])
}

/// Read back the first position id of a stored entry (its tag marker).
fn first_position(entry: &MRopeEntry) -> Option<i32> {
    let stored = entry.position_ids.as_ref()?;
    let first = mlxcel_core::slice(stored, &[0, 0, 0], &[1, 1, 1]);
    mlxcel_core::eval(&first);
    Some(mlxcel_core::item_i32(&first))
}

fn assert_image_entry(entry: Option<MRopeEntry>, axes: i32, tag: i32, delta: i32, what: &str) {
    let entry = entry.unwrap_or_else(|| panic!("{what}: no per-sequence MRoPE entry was bound"));
    assert_eq!(entry.rope_deltas, Some(delta), "{what}: wrong rope delta");
    let stored = entry
        .position_ids
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: the entry lost its position ids"));
    assert_eq!(
        mlxcel_core::array_shape(stored),
        vec![axes, 1, PREFILL_LEN],
        "{what}: wrong position-id shape"
    );
    assert_eq!(
        first_position(&entry),
        Some(tag * 100),
        "{what}: holds another request's position ids"
    );
}

fn assert_text_only_entry(entry: Option<MRopeEntry>, what: &str) {
    let entry = entry.unwrap_or_else(|| panic!("{what}: no per-sequence MRoPE entry was bound"));
    assert_eq!(
        entry.rope_deltas, None,
        "{what}: inherited an image request's rope delta"
    );
    assert!(
        entry.position_ids.is_none(),
        "{what}: inherited an image request's position ids"
    );
}

/// Decode one token after a text prefill and return its logits.
fn decode_after_prefill<M: crate::LanguageModel>(
    model: &M,
    caches: &mut [mlxcel_core::layers::KVCache],
    seq_id: Option<SequenceId>,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let token = mlxcel_core::from_slice_i32(&[9], &[1, 1]);
    let logits = model.forward_with_sequence_id(&token, seq_id, caches, None);
    mlxcel_core::eval(&logits);
    logits
}

fn prefilled_caches<M: crate::LanguageModel>(model: &M) -> Vec<mlxcel_core::layers::KVCache> {
    let mut caches = model.make_caches();
    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let logits = model.forward_with_sequence_id(&prompt, None, &mut caches, None);
    mlxcel_core::eval(&logits);
    caches
}

fn max_abs_diff(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) -> f32 {
    let diff = mlxcel_core::abs(&mlxcel_core::subtract(a, b));
    let max = mlxcel_core::max_all(&diff);
    mlxcel_core::eval(&max);
    mlxcel_core::item_f32(&max)
}

macro_rules! mrope_family_binding_tests {
    (
        $family:ident,
        variant: $variant:ident,
        wrapper: $wrapper:ident,
        text_model: $text_ty:ty,
        axes: $axes:expr $(,)?
    ) => {
        mod $family {
            use super::*;

            const AXES: i32 = $axes;

            fn text(model: &LoadedModel) -> &$text_ty {
                let LoadedModel::$variant(wrapper) = model else {
                    panic!("the test must hold the {} model", stringify!($variant));
                };
                &wrapper.text_model
            }

            /// Stand-in for an image request's media preparation.
            fn seed(model: &LoadedModel, tag: i32, delta: i32) {
                text(model).set_mrope_state(positions(AXES, tag), delta);
            }

            /// Two image requests and a text-only request admitted in one
            /// drain tick: each keeps its own positions and delta, and the
            /// text-only request inherits neither.
            #[test]
            fn bind_keeps_interleaved_sequences_apart() {
                let _guard = crate::models::embedding_test_support::mlx_test_guard();
                let model = LoadedModel::$variant($wrapper());
                let (a, b, t) = (
                    SequenceId::from_raw(26_340),
                    SequenceId::from_raw(26_341),
                    SequenceId::from_raw(26_342),
                );

                seed(&model, 1, 5);
                model.bind_qwen_vl_mrope_state_to_sequence(a);
                seed(&model, 2, 9);
                model.bind_qwen_vl_mrope_state_to_sequence(b);
                model.bind_qwen_vl_mrope_state_to_sequence(t);

                assert_image_entry(text(&model).take_mrope_entry(a), AXES, 1, 5, "request A");
                assert_image_entry(text(&model).take_mrope_entry(b), AXES, 2, 9, "request B");
                assert_text_only_entry(text(&model).take_mrope_entry(t), "text-only request");
            }

            /// Preemption takes the entry out under the evicted id and
            /// reinstalls it under the freshly allocated one.
            #[test]
            fn preemption_carries_entry_to_new_sequence_id() {
                let _guard = crate::models::embedding_test_support::mlx_test_guard();
                let model = LoadedModel::$variant($wrapper());
                let (old, new) = (SequenceId::from_raw(26_343), SequenceId::from_raw(26_344));

                seed(&model, 3, 7);
                model.bind_qwen_vl_mrope_state_to_sequence(old);
                let snapshot = model.take_qwen_vl_mrope_entry(old);
                assert!(
                    !snapshot.is_empty(),
                    "preemption must carry the evicted sequence's MRoPE entry"
                );
                assert!(text(&model).take_mrope_entry(old).is_none());
                model.install_qwen_vl_mrope_entry(new, snapshot);

                assert_image_entry(
                    text(&model).take_mrope_entry(new),
                    AXES,
                    3,
                    7,
                    "reinstalled entry",
                );
            }

            /// An admission abort drops the prepared fallback, so the next
            /// bound request does not adopt it.
            #[test]
            fn clear_vlm_fallback_state_drops_prepared_slot() {
                let _guard = crate::models::embedding_test_support::mlx_test_guard();
                let model = LoadedModel::$variant($wrapper());
                let next = SequenceId::from_raw(26_345);

                seed(&model, 4, 11);
                model.clear_vlm_fallback_state();
                text(&model).bind_mrope_state_to_sequence(next);

                assert_text_only_entry(text(&model).take_mrope_entry(next), "next request");
            }

            /// CLI contract: a forward with no id or an unbound id keeps
            /// reading the fallback slot.
            #[test]
            fn unbound_sequence_id_reads_fallback() {
                let _guard = crate::models::embedding_test_support::mlx_test_guard();
                let model = $wrapper();
                let mut no_id = prefilled_caches(&model);
                let mut unbound = prefilled_caches(&model);
                let mut cleared = prefilled_caches(&model);

                model
                    .text_model
                    .set_mrope_state(mlxcel_core::from_slice_i32(&[], &[AXES, 1, 0]), 37);
                let expected = decode_after_prefill(&model, &mut no_id, None);
                let actual =
                    decode_after_prefill(&model, &mut unbound, Some(SequenceId::from_raw(26_346)));
                model.text_model.clear_mrope_state();
                let text_only = decode_after_prefill(&model, &mut cleared, None);

                assert!(
                    max_abs_diff(&expected, &actual) < 1e-5,
                    "an unbound id must resolve through the fallback slot"
                );
                assert!(
                    max_abs_diff(&expected, &text_only) > 1e-4,
                    "the fallback delta must change decode logits for this test to discriminate"
                );
            }

            /// Real admission flow: an image request's prepared state binds
            /// to its own sequence, and the text-only request admitted next
            /// gets an empty entry.
            #[test]
            fn admission_binds_prepared_state_to_the_admitted_sequence() {
                let _guard = crate::models::embedding_test_support::mlx_test_guard();
                let mut scheduler = scheduler(loaded(), false);

                seed(scheduler.engine.model(), 5, 13);
                let _image = enqueue(&mut scheduler, vec![]);
                let image_seq = queued_seq_id(&scheduler);
                scheduler.prefill_queue.dequeue();
                let _text = enqueue(&mut scheduler, vec![]);
                let text_seq = queued_seq_id(&scheduler);

                let model = scheduler.engine.model();
                assert_image_entry(
                    text(model).take_mrope_entry(image_seq),
                    AXES,
                    5,
                    13,
                    "image request",
                );
                assert_text_only_entry(text(model).take_mrope_entry(text_seq), "text-only request");
            }

            /// Releasing a finished image request through the scheduler's
            /// release path drops its per-sequence entry and leaves the
            /// other in-flight request's entry in place.
            #[test]
            fn release_sequence_caches_drops_the_bound_entry() {
                let _guard = crate::models::embedding_test_support::mlx_test_guard();
                let mut scheduler = scheduler(loaded(), false);

                seed(scheduler.engine.model(), 7, 19);
                let _image = enqueue(&mut scheduler, vec![]);
                let image_seq = queued_seq_id(&scheduler);
                scheduler.prefill_queue.dequeue();
                let _text = enqueue(&mut scheduler, vec![]);
                let text_seq = queued_seq_id(&scheduler);

                scheduler.release_sequence_caches(image_seq);

                let model = scheduler.engine.model();
                assert!(
                    text(model).take_mrope_entry(image_seq).is_none(),
                    "releasing the sequence must drop its MRoPE entry"
                );
                assert_text_only_entry(text(model).take_mrope_entry(text_seq), "text-only request");
            }

            /// `Server busy` after media preparation: the next text-only
            /// request must not bind the aborted request's MRoPE state.
            #[test]
            fn allocation_failure_after_media_preparation_clears_fallback() {
                let _guard = crate::models::embedding_test_support::mlx_test_guard();
                let mut scheduler = scheduler(loaded(), true);
                let mut held = exhaust_pool(&mut scheduler);

                seed(scheduler.engine.model(), 6, 17);
                let busy = enqueue(&mut scheduler, vec![]);
                let message = expect_error(&busy);
                assert!(message.starts_with("Server busy"), "got: {message}");

                scheduler.release_sequence_caches(held.pop().unwrap());
                let _next = enqueue(&mut scheduler, vec![]);
                let seq_id = queued_seq_id(&scheduler);
                assert_text_only_entry(
                    text(scheduler.engine.model()).take_mrope_entry(seq_id),
                    "text-only request",
                );
            }

            fn loaded() -> LoadedModel {
                LoadedModel::$variant($wrapper())
            }
        }
    };
}

mrope_family_binding_tests!(
    paddleocr_vl,
    variant: PaddleOcrVL,
    wrapper: tiny_paddleocr_vl_wrapper,
    text_model: crate::models::PaddleOcrTextModel,
    axes: 3,
);
mrope_family_binding_tests!(
    ernie45_moe_vl,
    variant: Ernie45MoeVLM,
    wrapper: tiny_ernie45_moe_vl_wrapper,
    text_model: crate::models::ernie4_5_moe_vl::Ernie45MoeVlTextModel,
    axes: 3,
);
mrope_family_binding_tests!(
    hunyuan_vl,
    variant: HunyuanVLM,
    wrapper: tiny_hunyuan_vl_wrapper,
    text_model: crate::models::hunyuan_vl::HunyuanVlTextModel,
    axes: 4,
);
