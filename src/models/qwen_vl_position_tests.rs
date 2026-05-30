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

//! Regression tests for chunked-prefill MRoPE position-IDs shape check.
//!
//! These tests verify:
//! - the sufficiency check (`shape[-1] >= cache_offset + seq_len`) correctly
//!   replaces the old strict-equality guard, allowing cached position_ids to be reused
//!   across chunked-prefill passes without panicking on shape mismatch.
//! - the batch-size check (`shape[1] == batch_size`) is validated alongside
//!   the seq-length check, so sequential requests with different batch_sizes do not reuse
//!   stale position IDs and crash on broadcast_shapes.

/// Helper: build a synthetic [3, batch, total_len] position_ids tensor holding values 0..total_len.
fn make_position_ids_with_batch(
    batch: i32,
    total_len: i32,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let pos = mlxcel_core::arange_i32(0, total_len, 1);
    let pos = mlxcel_core::reshape(&pos, &[1, total_len]);
    let pos = mlxcel_core::broadcast_to(&pos, &[1, batch, total_len]);
    mlxcel_core::broadcast_to(&pos, &[3, batch, total_len])
}

/// Helper: build a synthetic [3, 1, total_len] position_ids tensor holding values 0..total_len.
fn make_position_ids(total_len: i32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    make_position_ids_with_batch(1, total_len)
}

/// Emulates the shape-check and slice logic from all Qwen VL text-model
/// `forward_with_mrope_state` / `forward_impl` functions after and.
///
/// Returns `Some(sliced_ids)` when the cached tensor is sufficient (matching both batch
/// dimension and seq-length range), `None` otherwise.
fn try_reuse_position_ids(
    stored_pos: &mlxcel_core::MlxArray,
    batch: i32,
    cache_offset: i32,
    seq_len: i32,
) -> Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> {
    let pos_shape = mlxcel_core::array_shape(stored_pos);
    // validate batch dimension (pos_shape[1]) in addition to seq-length
    // sufficiency (pos_shape[2] >= cache_offset + seq_len), matching upstream Python:
    //   self._position_ids.shape[1] == batch_size
    //   and self._position_ids.shape[-1] >= cache_offset + seq_length
    if pos_shape.len() == 3 && pos_shape[1] == batch && pos_shape[2] >= cache_offset + seq_len {
        Some(mlxcel_core::slice(
            stored_pos,
            &[0, 0, cache_offset],
            &[pos_shape[0], pos_shape[1], cache_offset + seq_len],
        ))
    } else {
        None
    }
}

/// Verify that the sufficiency check accepts a cache when shape[-1] equals exactly
/// cache_offset + seq_len (boundary case).
#[test]
#[ignore = "requires serial MLX execution"]
fn sufficiency_check_accepts_exact_boundary() {
    // total_len = 20, chunk [0, 20) → cache_offset=0, seq_len=20, batch=1
    let stored = make_position_ids(20);
    let result = try_reuse_position_ids(&stored, 1, 0, 20);
    assert!(
        result.is_some(),
        "Should reuse when shape[-1] == cache_offset + seq_len (exact boundary)"
    );

    // Verify shape of sliced result is [3, 1, 20]
    let sliced = result.unwrap();
    mlxcel_core::eval(&sliced);
    let shape = mlxcel_core::array_shape(&sliced);
    assert_eq!(
        shape,
        vec![3, 1, 20],
        "Sliced ids should have shape [3, 1, 20]"
    );
}

/// Verify that the sufficiency check accepts cached ids during chunked prefill
/// when cache_offset > 0 (previously this would have failed the old == 0 guard).
#[test]
#[ignore = "requires serial MLX execution"]
fn sufficiency_check_accepts_chunked_prefill_with_nonzero_cache_offset() {
    // Simulate: full prefill position_ids = [3, 1, 20], sent in two 10-token chunks.
    // Second chunk: cache_offset=10, seq_len=10, batch=1
    let stored = make_position_ids(20);
    let result = try_reuse_position_ids(&stored, 1, 10, 10);
    assert!(
        result.is_some(),
        "Should reuse when shape[-1]=20 >= cache_offset(10) + seq_len(10)=20"
    );

    let sliced = result.unwrap();
    mlxcel_core::eval(&sliced);

    // Shape should be [3, 1, 10]
    let shape = mlxcel_core::array_shape(&sliced);
    assert_eq!(
        shape,
        vec![3, 1, 10],
        "Sliced ids should have shape [3, 1, 10]"
    );
}

/// Verify that slice values are correct: the chunk [cache_offset, cache_offset+seq_len)
/// should contain positions {cache_offset, cache_offset+1, ..., cache_offset+seq_len-1}.
#[test]
#[ignore = "requires serial MLX execution"]
fn chunked_prefill_slice_contains_correct_position_values() {
    // total_len = 20; first chunk cache_offset=0, seq_len=8; second cache_offset=8, seq_len=8
    let stored = make_position_ids(20);

    for (cache_offset, seq_len) in [(0i32, 8i32), (8, 8), (16, 4)] {
        let result = try_reuse_position_ids(&stored, 1, cache_offset, seq_len);
        assert!(
            result.is_some(),
            "Should reuse chunk cache_offset={cache_offset} seq_len={seq_len}"
        );

        let sliced = result.unwrap();
        mlxcel_core::eval(&sliced);

        // Check the first element of each chunk equals cache_offset
        let first = mlxcel_core::slice(&sliced, &[0, 0, 0], &[1, 1, 1]);
        mlxcel_core::eval(&first);
        let first_val = mlxcel_core::item_i32(&first);
        assert_eq!(
            first_val, cache_offset,
            "First position in chunk should be {cache_offset}, got {first_val}"
        );

        // Check the last element equals cache_offset + seq_len - 1
        let last_idx = seq_len - 1;
        let last = mlxcel_core::slice(&sliced, &[0, 0, last_idx], &[1, 1, last_idx + 1]);
        mlxcel_core::eval(&last);
        let last_val = mlxcel_core::item_i32(&last);
        assert_eq!(
            last_val,
            cache_offset + seq_len - 1,
            "Last position in chunk should be {}, got {last_val}",
            cache_offset + seq_len - 1
        );
    }
}

/// Verify that the sufficiency check REJECTS stale cached ids when they no longer
/// cover the needed window (decode steps far into generation).
#[test]
#[ignore = "requires serial MLX execution"]
fn sufficiency_check_rejects_when_cached_ids_exhausted() {
    // Stored covers [0, 20). During decode at cache_offset=50, seq_len=1: not sufficient.
    let stored = make_position_ids(20);
    let result = try_reuse_position_ids(&stored, 1, 50, 1);
    assert!(
        result.is_none(),
        "Should NOT reuse when shape[-1]=20 < cache_offset(50) + seq_len(1)=51"
    );
}

/// Verify the old strict-equality guard (cache_offset == 0) was the bug:
/// a chunk with cache_offset=10, seq_len=10 previously fell through to delta-based
/// computation even though the cached ids had enough range.
///
/// This test documents the regression: with the old guard the second chunk would
/// have silently been recomputed via rope_deltas, potentially producing wrong
/// positions for video tokens.
#[test]
#[ignore = "requires serial MLX execution"]
fn old_strict_equality_guard_would_reject_second_chunk() {
    let stored = make_position_ids(20);
    let cache_offset = 10i32;
    let seq_len = 10i32;

    // Old (broken) condition: `cache_offset == 0`
    let old_guard_passes = cache_offset == 0;
    assert!(
        !old_guard_passes,
        "Old guard correctly does not pass for cache_offset={cache_offset}"
    );

    // New (fixed) condition: `pos_shape[2] >= cache_offset + seq_len`
    let pos_shape = mlxcel_core::array_shape(&stored);
    let new_guard_passes =
        pos_shape.len() == 3 && pos_shape[1] == 1 && pos_shape[2] >= cache_offset + seq_len;
    assert!(
        new_guard_passes,
        "New sufficiency guard must accept cache_offset={cache_offset} seq_len={seq_len} shape[-1]={}",
        pos_shape[2]
    );
}

/// Verify that a cached [3, 1, 8] position_ids tensor is REJECTED when the next request
/// has batch_size=2 (regression guard).
///
/// This covers the upstream mlx-vlm PR #1040 fix: sequential requests with different
/// batch_sizes must not reuse stale position IDs, which would produce wrong RoPE
/// cos/sin tensors and crash on broadcast_shapes.
#[test]
#[ignore = "requires serial MLX execution"]
fn batch_size_mismatch_rejects_cached_position_ids() {
    // Request 1: batch_size=1, prompt_len=8 → caches [3, 1, 8] position_ids.
    let stored = make_position_ids_with_batch(1, 8);

    // Request 2: batch_size=2, seq_len=4, cache_offset=0 → must NOT reuse the [3, 1, 8] cache.
    let result = try_reuse_position_ids(&stored, 2, 0, 4);
    assert!(
        result.is_none(),
        "Should NOT reuse cached [3, 1, 8] position_ids for batch_size=2 request (shape[1]=1 != 2)"
    );
}

/// Verify that the batch-size check does NOT break the common single-batch path:
/// a [3, 1, 8] cache is still accepted for a subsequent single-batch request that
/// fits within the seq-length range.
#[test]
#[ignore = "requires serial MLX execution"]
fn same_batch_size_still_accepted_with_sufficient_seq_len() {
    // Request 1: batch_size=1, prompt_len=8 → caches [3, 1, 8] position_ids.
    let stored = make_position_ids_with_batch(1, 8);

    // Request 2: batch_size=1, seq_len=4, cache_offset=0 → must reuse (seq sufficient).
    let result = try_reuse_position_ids(&stored, 1, 0, 4);
    assert!(
        result.is_some(),
        "Should reuse [3, 1, 8] for batch_size=1 seq_len=4 (shape[1]=1 matches, shape[2]=8 >= 4)"
    );
    let sliced = result.unwrap();
    mlxcel_core::eval(&sliced);
    let shape = mlxcel_core::array_shape(&sliced);
    assert_eq!(
        shape,
        vec![3, 1, 4],
        "Sliced ids should have shape [3, 1, 4]"
    );
}

/// Verify that a [3, 2, 8] batch-2 cache is correctly accepted for a batch-2 request.
#[test]
#[ignore = "requires serial MLX execution"]
fn batch_two_cache_accepted_for_matching_batch_two_request() {
    // Prefill with batch_size=2, prompt_len=8 → caches [3, 2, 8] position_ids.
    let stored = make_position_ids_with_batch(2, 8);

    // Next request: same batch_size=2, seq_len=1 (decode step), cache_offset=8.
    // shape[-1]=8 < cache_offset(8)+seq_len(1)=9 → not sufficient, must recompute.
    let result_decode = try_reuse_position_ids(&stored, 2, 8, 1);
    assert!(
        result_decode.is_none(),
        "Should NOT reuse [3, 2, 8] when shape[-1]=8 < cache_offset(8)+seq_len(1)=9"
    );

    // Decode step within the prefill window (cache_offset=4, seq_len=4 → exactly covers).
    let result_within = try_reuse_position_ids(&stored, 2, 4, 4);
    assert!(
        result_within.is_some(),
        "Should reuse [3, 2, 8] for batch_size=2 cache_offset=4 seq_len=4 (8 >= 4+4)"
    );
    let sliced = result_within.unwrap();
    mlxcel_core::eval(&sliced);
    let shape = mlxcel_core::array_shape(&sliced);
    assert_eq!(
        shape,
        vec![3, 2, 4],
        "Sliced ids should have shape [3, 2, 4]"
    );
}
