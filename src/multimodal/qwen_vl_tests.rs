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

use super::{
    InsertedQwenVlmTokens, QwenVisualGrid, QwenVisualKind, forward_batched_with_seq_ids_dispatch,
    insert_qwen_vl_image_tokens, insert_qwen_vl_media_tokens,
    insert_qwen_vl_media_tokens_timestamped, qwen_video_group_timestamps,
    qwen_video_timestamp_text, qwen_vl_mrope_positions_from_tokens,
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
            video_blocks: 0,
            total_image_tokens: 4,
            total_video_tokens: 0,
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
            video_blocks: 0,
            total_image_tokens: 4,
            total_video_tokens: 0,
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
            video_blocks: 0,
            total_image_tokens: 6,
            total_video_tokens: 0,
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
            video_blocks: 0,
            total_image_tokens: 6,
            total_video_tokens: 0,
        })
    );
    assert_eq!(
        prompt_tokens,
        vec![1, 200, 203, 203, 203, 203, 201, 200, 203, 203, 201, 7]
    );
}

#[test]
fn insert_qwen_vl_media_tokens_expands_video_placeholder() {
    let mut prompt_tokens = vec![1, 100, 104, 101, 42];
    let stats = insert_qwen_vl_media_tokens(
        &mut prompt_tokens,
        &[QwenVisualGrid {
            kind: QwenVisualKind::Video,
            grid_thw: (2, 4, 4),
        }],
        2,
        100,
        103,
        104,
    )
    .unwrap();

    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 0,
            video_blocks: 1,
            total_image_tokens: 0,
            total_video_tokens: 8,
        })
    );
    assert_eq!(
        prompt_tokens,
        vec![1, 100, 104, 104, 104, 104, 104, 104, 104, 104, 101, 42]
    );
}

#[test]
fn insert_qwen_vl_media_tokens_preserves_mixed_prompt_order() {
    let mut prompt_tokens = vec![1, 100, 104, 101, 9, 100, 103, 101, 42];
    let stats = insert_qwen_vl_media_tokens(
        &mut prompt_tokens,
        &[
            QwenVisualGrid {
                kind: QwenVisualKind::Video,
                grid_thw: (2, 4, 4),
            },
            QwenVisualGrid {
                kind: QwenVisualKind::Image,
                grid_thw: (1, 4, 4),
            },
        ],
        2,
        100,
        103,
        104,
    )
    .unwrap();

    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 1,
            video_blocks: 1,
            total_image_tokens: 4,
            total_video_tokens: 8,
        })
    );
    assert_eq!(
        prompt_tokens,
        vec![
            1, 100, 104, 104, 104, 104, 104, 104, 104, 104, 101, 9, 100, 103, 103, 103, 103, 101,
            42,
        ]
    );
}

#[test]
fn insert_qwen_vl_media_tokens_rejects_wrong_expanded_cardinality() {
    let mut prompt_tokens = vec![1, 100, 104, 104, 104, 101, 42];
    let error = insert_qwen_vl_media_tokens(
        &mut prompt_tokens,
        &[QwenVisualGrid {
            kind: QwenVisualKind::Video,
            grid_thw: (2, 4, 4),
        }],
        2,
        100,
        103,
        104,
    )
    .unwrap_err();

    assert!(
        error.to_string().contains("cardinality mismatch"),
        "{error}"
    );
}

#[test]
fn insert_qwen_vl_media_tokens_rejects_adjacent_cross_kind_runs() {
    let mut prompt_tokens = vec![
        1, 103, 103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 104, 42,
    ];
    let error = insert_qwen_vl_media_tokens(
        &mut prompt_tokens,
        &[
            QwenVisualGrid {
                kind: QwenVisualKind::Image,
                grid_thw: (1, 4, 4),
            },
            QwenVisualGrid {
                kind: QwenVisualKind::Video,
                grid_thw: (2, 4, 4),
            },
        ],
        2,
        100,
        103,
        104,
    )
    .unwrap_err();

    assert!(
        error.to_string().contains("adjacent"),
        "error should reject MRoPE-ambiguous adjacent runs, got: {error}"
    );
}

#[test]
fn qwen_vl_mrope_positions_keep_video_temporal_axis() {
    let tokens = vec![1, 100, 104, 104, 104, 104, 104, 104, 104, 104, 101, 42];
    let positions =
        qwen_vl_mrope_positions_from_tokens(&tokens, &[(2, 4, 4)], 2, 103, 104).unwrap();

    assert_eq!(
        &positions[0][2..10],
        &[2, 2, 2, 2, 3, 3, 3, 3],
        "video temporal positions must advance across grid_t"
    );
    assert_eq!(
        &positions[1][2..10],
        &[2, 2, 3, 3, 2, 2, 3, 3],
        "height positions must use h / spatial_merge_size"
    );
    assert_eq!(
        &positions[2][2..10],
        &[2, 3, 2, 3, 2, 3, 2, 3],
        "width positions must use w / spatial_merge_size"
    );
    assert_eq!(positions[0].len(), tokens.len());
}

// -- per-group video timestamps (#2402) ---------------------

/// Two-group `(2, 4, 4)` video with spatial merge 2: each group is
/// `(4 / 2) * (4 / 2) = 4` video tokens.
fn two_group_video() -> [QwenVisualGrid; 1] {
    [QwenVisualGrid {
        kind: QwenVisualKind::Video,
        grid_thw: (2, 4, 4),
    }]
}

/// Timestamp text tokens: group 0 is one token, group 1 is two.
fn two_group_prefixes() -> Vec<Vec<Vec<i32>>> {
    vec![vec![vec![50], vec![51, 52]]]
}

/// The prompt a two-group video expands to, with ids start=100, end=101,
/// video=104, trailing text 42.
fn two_group_expanded() -> Vec<i32> {
    vec![
        1, 50, 100, 104, 104, 104, 104, 101, 51, 52, 100, 104, 104, 104, 104, 101, 42,
    ]
}

#[test]
fn timestamped_media_tokens_render_one_framed_run_per_group() {
    let mut prompt_tokens = vec![1, 100, 104, 101, 42];
    let stats = insert_qwen_vl_media_tokens_timestamped(
        &mut prompt_tokens,
        &two_group_video(),
        2,
        100,
        103,
        104,
        &two_group_prefixes(),
    )
    .unwrap();

    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 0,
            video_blocks: 1,
            total_image_tokens: 0,
            total_video_tokens: 8,
        }),
        "timestamp text is not a visual token and must not change the feature count"
    );
    assert_eq!(
        prompt_tokens,
        two_group_expanded(),
        "the template's own framing pair is folded into the per-group framing"
    );
}

#[test]
fn timestamped_media_tokens_insert_after_bos_without_placeholders() {
    let mut prompt_tokens = vec![1, 42];
    insert_qwen_vl_media_tokens_timestamped(
        &mut prompt_tokens,
        &two_group_video(),
        2,
        100,
        103,
        104,
        &two_group_prefixes(),
    )
    .unwrap();

    assert_eq!(prompt_tokens, two_group_expanded());
}

#[test]
fn timestamped_media_tokens_keep_images_flat_and_order_mixed_media() {
    // image (2x2 merged -> 4 tokens at grid (1, 4, 4) / merge 2), then video.
    let media = [
        QwenVisualGrid {
            kind: QwenVisualKind::Image,
            grid_thw: (1, 4, 4),
        },
        QwenVisualGrid {
            kind: QwenVisualKind::Video,
            grid_thw: (2, 4, 4),
        },
    ];
    let mut prompt_tokens = vec![1, 100, 103, 101, 9, 100, 104, 101, 42];
    insert_qwen_vl_media_tokens_timestamped(
        &mut prompt_tokens,
        &media,
        2,
        100,
        103,
        104,
        &two_group_prefixes(),
    )
    .unwrap();

    assert_eq!(
        prompt_tokens,
        vec![
            1, 100, 103, 103, 103, 103, 101, 9, 50, 100, 104, 104, 104, 104, 101, 51, 52, 100, 104,
            104, 104, 104, 101, 42,
        ]
    );
}

#[test]
fn timestamped_media_tokens_accept_already_expanded_prompts() {
    // Per-group layout is left untouched.
    let mut grouped = two_group_expanded();
    insert_qwen_vl_media_tokens_timestamped(
        &mut grouped,
        &two_group_video(),
        2,
        100,
        103,
        104,
        &two_group_prefixes(),
    )
    .unwrap();
    assert_eq!(grouped, two_group_expanded());

    // A flat run holding every group's features is also accepted.
    let mut flat = vec![1, 100, 104, 104, 104, 104, 104, 104, 104, 104, 101, 42];
    let flat_before = flat.clone();
    insert_qwen_vl_media_tokens_timestamped(
        &mut flat,
        &two_group_video(),
        2,
        100,
        103,
        104,
        &two_group_prefixes(),
    )
    .unwrap();
    assert_eq!(flat, flat_before);
}

#[test]
fn timestamped_media_tokens_reject_malformed_group_layouts() {
    // Second group is truncated to 3 tokens.
    let mut truncated = two_group_expanded();
    truncated.remove(12);
    let error = insert_qwen_vl_media_tokens_timestamped(
        &mut truncated,
        &two_group_video(),
        2,
        100,
        103,
        104,
        &two_group_prefixes(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("cardinality mismatch"),
        "truncated group must be rejected, got: {error}"
    );

    // One timestamp for a two-group video.
    let mut prompt_tokens = vec![1, 100, 104, 101, 42];
    let error = insert_qwen_vl_media_tokens_timestamped(
        &mut prompt_tokens,
        &two_group_video(),
        2,
        100,
        103,
        104,
        &[vec![vec![50]]],
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("temporal group(s)"),
        "timestamp/group count mismatch must be rejected, got: {error}"
    );
    assert_eq!(prompt_tokens, vec![1, 100, 104, 101, 42]);
}

#[test]
fn qwen_vl_mrope_positions_restart_temporal_axis_per_timestamped_group() {
    let tokens = two_group_expanded();
    let positions =
        qwen_vl_mrope_positions_from_tokens(&tokens, &[(2, 4, 4)], 2, 103, 104).unwrap();

    // Each group is positioned as its own `(1, 4, 4)` grid; the timestamp
    // text and framing tokens between groups advance as plain text.
    assert_eq!(
        positions[0],
        vec![0, 1, 2, 3, 3, 3, 3, 5, 6, 7, 8, 9, 9, 9, 9, 11, 12],
        "temporal axis is constant within a group and jumps past the timestamp text"
    );
    assert_eq!(
        positions[1],
        vec![0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 12]
    );
    assert_eq!(
        positions[2],
        vec![0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12]
    );
    assert_eq!(positions[0].len(), tokens.len());
    assert_eq!(positions[1].len(), tokens.len());
    assert_eq!(positions[2].len(), tokens.len());
}

#[test]
fn qwen_vl_mrope_positions_reject_timestamped_groups_missing_a_run() {
    // Only one of the two groups is present.
    let tokens = vec![1, 50, 100, 104, 104, 104, 104, 101, 42];
    let error = qwen_vl_mrope_positions_from_tokens(&tokens, &[(2, 4, 4)], 2, 103, 104)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("temporal group") || error.contains("expected"),
        "a missing group run must not silently pass, got: {error}"
    );
}

#[test]
fn video_group_timestamps_use_mean_of_first_and_last_frame_time() {
    // 4 frames at 30 fps, temporal patch 2 -> groups (0, 15) and (30, 45).
    let seconds = qwen_video_group_timestamps(&[0, 15, 30, 45], 30.0, 2, 2).unwrap();
    assert_eq!(seconds, vec![0.25, 1.25]);

    // An odd frame count pads by repeating the last frame: (0, 10), (20, 20).
    let seconds = qwen_video_group_timestamps(&[0, 10, 20], 10.0, 2, 2).unwrap();
    assert_eq!(seconds, vec![0.5, 2.0]);

    // Resampled clips keep real time: indices come from the sampler, not an
    // assumed 2 fps.
    let seconds = qwen_video_group_timestamps(&[0, 60, 120, 180], 60.0, 2, 2).unwrap();
    assert_eq!(seconds, vec![0.5, 2.5]);
}

#[test]
fn video_group_timestamps_reject_inconsistent_inputs() {
    assert!(qwen_video_group_timestamps(&[0, 1, 2, 3], 30.0, 2, 3).is_err());
    assert!(qwen_video_group_timestamps(&[0, 1], 0.0, 2, 1).is_err());
    assert!(qwen_video_group_timestamps(&[0, 1], f64::NAN, 2, 1).is_err());
    assert!(qwen_video_group_timestamps(&[0, 1], f64::INFINITY, 2, 1).is_err());
    assert!(qwen_video_group_timestamps(&[], 30.0, 2, 1).is_err());
    assert!(qwen_video_group_timestamps(&[0, 1], 30.0, 0, 1).is_err());
}

#[test]
fn video_timestamp_text_has_one_decimal() {
    assert_eq!(qwen_video_timestamp_text(0.5), "<0.5 seconds>");
    assert_eq!(qwen_video_timestamp_text(12.0), "<12.0 seconds>");
    assert_eq!(qwen_video_timestamp_text(2.04), "<2.0 seconds>");
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
/// `forward_batched_with_ids` were added to fix.
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
    );
    mlxcel_core::eval(&logits);

    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0], (11, 42));
}

/// When `seq_ids` is `None` the helper must NOT bypass the model's
/// batched fast path — falling through to `forward_batched`
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
    );
    mlxcel_core::eval(&logits);

    // Without seq_ids the trait default `forward_batched`
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
