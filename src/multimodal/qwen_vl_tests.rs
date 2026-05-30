// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

use super::{
    InsertedQwenVlmTokens, forward_batched_with_seq_ids_dispatch, insert_qwen_vl_image_tokens,
};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};
use std::cell::RefCell;

#[test]
fn insert_qwen_vl_image_tokens_inserts_blocks_after_bos() {
    let mut prompt_tokens = vec![1, 42, 43];
    let stats = insert_qwen_vl_image_tokens(&mut prompt_tokens, &[(1, 4, 4)], 2, 100, 103);

    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 1,
            total_image_tokens: 4,
        })
    );
    assert_eq!(prompt_tokens, vec![1, 100, 103, 103, 103, 103, 101, 42, 43]);
}

// A prompt that already carries one `<|image_pad|>` per image
// (the canonical framing emitted by the model chat template, e.g. qwen2_vl on
// the CLI image path) must have that single placeholder EXPANDED to the grid
// count, not skipped. Previously this returned `None`, leaving one placeholder
// to face N vision features — a count mismatch that produced zero tokens.
#[test]
fn insert_qwen_vl_image_tokens_expands_single_placeholder_per_image() {
    // [<|im_start|>, <|vision_start|>(100), <|image_pad|>(103), <|vision_end|>(101), 42, 43]
    let mut prompt_tokens = vec![1, 100, 103, 101, 42, 43];
    let stats = insert_qwen_vl_image_tokens(&mut prompt_tokens, &[(1, 4, 4)], 2, 100, 103);

    // (1*4/2*4/2) = 4 image tokens; the single placeholder expands in place,
    // keeping the template's vision_start/vision_end framing.
    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 1,
            total_image_tokens: 4,
        })
    );
    assert_eq!(prompt_tokens, vec![1, 100, 103, 103, 103, 103, 101, 42, 43]);
}

// Two images, one placeholder each, with differing grids → expand each in
// place to its own count.
#[test]
fn insert_qwen_vl_image_tokens_expands_one_placeholder_per_image_multi() {
    let mut prompt_tokens = vec![1, 203, 7, 203, 8];
    let stats =
        insert_qwen_vl_image_tokens(&mut prompt_tokens, &[(1, 4, 4), (2, 2, 2)], 2, 200, 203);

    // image 0: 1*2*2 = 4; image 1: 2*1*1 = 2; total 6.
    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 2,
            total_image_tokens: 6,
        })
    );
    assert_eq!(prompt_tokens, vec![1, 203, 203, 203, 203, 7, 203, 203, 8]);
}

// A placeholder count that does not equal the image count (e.g. the prompt was
// already expanded to N>images placeholders) must be left untouched so we never
// double-expand.
#[test]
fn insert_qwen_vl_image_tokens_noop_when_already_expanded() {
    // 4 placeholders for a single image that expects 4 → already expanded.
    let mut prompt_tokens = vec![1, 100, 103, 103, 103, 103, 101, 42];
    let original = prompt_tokens.clone();

    let stats = insert_qwen_vl_image_tokens(&mut prompt_tokens, &[(1, 4, 4)], 2, 100, 103);

    assert_eq!(stats, None);
    assert_eq!(prompt_tokens, original);
}

#[test]
fn insert_qwen_vl_image_tokens_supports_multiple_images() {
    let mut prompt_tokens = vec![1, 7];
    let stats =
        insert_qwen_vl_image_tokens(&mut prompt_tokens, &[(1, 4, 4), (2, 2, 2)], 2, 200, 203);

    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 2,
            total_image_tokens: 6,
        })
    );
    assert_eq!(
        prompt_tokens,
        vec![1, 200, 203, 203, 203, 203, 201, 200, 203, 203, 201, 7]
    );
}

// -- dispatch helper integration tests ----------------------
//
// Drives `forward_batched_with_seq_ids_dispatch` against a tiny stub
// `LanguageModel` so we can assert that each row of a batched call
// resolves its own `seq_id`. This is the wrapper-level coverage gap the
// pr-reviewer flagged: without an override on each `vision::Qwen*VLModel`
// the trait default would discard `seq_ids` before they reach the text
// model, and a mixed VL+text batch would silently use text-only positions
// for the VL row.

/// Stub `LanguageModel` that records every (seq_id, row_value) pair it
/// observes via `forward_with_sequence_id`. Forward returns a 1-element
/// logits tensor whose value is `row_value as f32`, so callers can also
/// assert that the row ordering survived the per-row dispatch.
struct StubTextModel {
    /// Append-only log of `(seq_id_or_neg1, row_value)` calls.
    calls: RefCell<Vec<(i64, i32)>>,
}

impl StubTextModel {
    fn new() -> Self {
        Self {
            calls: RefCell::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<(i64, i32)> {
        self.calls.borrow().clone()
    }
}

impl LanguageModel for StubTextModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // No seq id — record the row value with sentinel -1.
        let row_val = read_first_i32(input_ids);
        self.calls.borrow_mut().push((-1, row_val));
        // Return shape [1, 1, 1] so concatenate along axis 0 yields
        // [B, 1, 1] which the helper can stack.
        let payload = row_val as f32;
        mlxcel_core::from_slice_f32(&[payload], &[1, 1, 1])
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let row_val = read_first_i32(input_ids);
        let id = seq_id.map(|s| s.as_u64() as i64).unwrap_or(-1);
        self.calls.borrow_mut().push((id, row_val));
        let payload = row_val as f32;
        mlxcel_core::from_slice_f32(&[payload], &[1, 1, 1])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
}

fn read_first_i32(arr: &MlxArray) -> i32 {
    mlxcel_core::eval(arr);
    let head = mlxcel_core::slice(arr, &[0, 0], &[1, 1]);
    mlxcel_core::eval(&head);
    mlxcel_core::item_i32(&head)
}

/// Mixed VL+text batch: row 0 is bound to seq id 5 (VL row), row 1 is
/// bound to seq id 9 (text row). The helper must call
/// `forward_with_sequence_id` once per row with the correct id so that
/// each row's per-sequence MRoPE state resolves independently — the
/// exact regression the wrapper-level overrides on
/// `forward_batched_with_context_and_ids` were added to fix.
#[test]
#[ignore = "requires serial MLX execution"]
fn forward_batched_with_seq_ids_dispatch_routes_each_row_to_its_seq_id() {
    let model = StubTextModel::new();

    // input_ids is [B, T] = [2, 1] with row 0 = 100, row 1 = 200 so the
    // stub can identify which row it received.
    let input_ids = mlxcel_core::from_slice_i32(&[100, 200], &[2, 1]);

    // Two empty per-row caches.
    let mut row0_caches: Vec<KVCache> = Vec::new();
    let mut row1_caches: Vec<KVCache> = Vec::new();
    let mut batch_caches: Vec<&mut [KVCache]> =
        vec![row0_caches.as_mut_slice(), row1_caches.as_mut_slice()];

    let seq_ids = [SequenceId::from_raw(5), SequenceId::from_raw(9)];

    let logits = forward_batched_with_seq_ids_dispatch(
        &model,
        &input_ids,
        Some(&seq_ids),
        batch_caches.as_mut_slice(),
        None,
        None,
    );
    mlxcel_core::eval(&logits);

    // Two calls — one per row — each carrying the right (id, row_val) pair.
    let calls = model.calls();
    assert_eq!(
        calls.len(),
        2,
        "must dispatch once per row, got {:?}",
        calls
    );
    assert_eq!(
        calls[0],
        (5, 100),
        "row 0 must be forwarded with seq id 5 and row value 100"
    );
    assert_eq!(
        calls[1],
        (9, 200),
        "row 1 must be forwarded with seq id 9 and row value 200"
    );

    // Sanity: stacked logits preserved per-row identity.
    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(shape, vec![2, 1, 1]);
}

/// Single-row fast path: the helper should still route through
/// `forward_with_sequence_id` so the per-sequence MRoPE entry for that
/// row resolves. We assert the call log shows exactly one entry under the
/// correct id.
#[test]
#[ignore = "requires serial MLX execution"]
fn forward_batched_with_seq_ids_dispatch_single_row_uses_forward_with_sequence_id() {
    let model = StubTextModel::new();
    let input_ids = mlxcel_core::from_slice_i32(&[42], &[1, 1]);
    let mut row_caches: Vec<KVCache> = Vec::new();
    let mut batch_caches: Vec<&mut [KVCache]> = vec![row_caches.as_mut_slice()];
    let seq_ids = [SequenceId::from_raw(11)];

    let logits = forward_batched_with_seq_ids_dispatch(
        &model,
        &input_ids,
        Some(&seq_ids),
        batch_caches.as_mut_slice(),
        None,
        None,
    );
    mlxcel_core::eval(&logits);

    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0], (11, 42));
}

/// When `seq_ids` is `None` the helper must NOT bypass the model's
/// batched fast path — falling through to `forward_batched_with_context`
/// preserves CLI/single-process callers that have never used per-seq
/// dispatch (they still resolve via the legacy fallback slot).
#[test]
#[ignore = "requires serial MLX execution"]
fn forward_batched_with_seq_ids_dispatch_no_seq_ids_falls_through_to_batched() {
    let model = StubTextModel::new();
    let input_ids = mlxcel_core::from_slice_i32(&[7, 8], &[2, 1]);
    let mut row0_caches: Vec<KVCache> = Vec::new();
    let mut row1_caches: Vec<KVCache> = Vec::new();
    let mut batch_caches: Vec<&mut [KVCache]> =
        vec![row0_caches.as_mut_slice(), row1_caches.as_mut_slice()];

    let logits = forward_batched_with_seq_ids_dispatch(
        &model,
        &input_ids,
        None,
        batch_caches.as_mut_slice(),
        None,
        None,
    );
    mlxcel_core::eval(&logits);

    // Without seq_ids the trait default `forward_batched_with_context`
    // routes through `forward_batched`, which loops calling `forward()`
    // — so each row's call is logged under the sentinel id `-1`.
    let calls = model.calls();
    assert_eq!(
        calls,
        vec![(-1, 7), (-1, 8)],
        "no seq_ids → forward (without id) per row; got {:?}",
        calls
    );
}
