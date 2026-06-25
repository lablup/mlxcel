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

//! Unit tests for the physical main-K/V pool storage added to
//! `PagedBlockPool` in #118 (Phase 1 of the unified paged KV cache, #116).
//!
//! Organised as:
//! 1. write_block + gather_visible byte-identity vs a dense contiguous buffer
//!    (the acceptance criterion).
//! 2. Fragmented / out-of-order physical rows gather in block-table order.
//! 3. Partial final block gathers exactly `visible_len` tokens (no padding).
//! 4. `logical_start > 0` (post-trim) gathers the correct visible window.
//! 5. INT8 main K/V round-trips byte-identically (dtype preserved, no astype).
//! 6. FP16 main K/V round-trips byte-identically (dtype preserved, no astype).
//! 7. `release_block` to refcount 0 frees and recycles a row with fresh data.
//! 8. `pool_tensor_bytes` reflects allocated pool tensors.
//! 9. Turbo4 sidecars coexist with main-K/V rows.
//!
//! Plus the bulk prefill writer added in #120 (Phase 3):
//!
//! - `write_prefill` cold round-trip is byte-identical to a dense prefill.
//! - `write_prefill` of a block-aligned prompt round-trips.
//! - `write_prefill` copy-on-write forks a shared partial tail block so two
//!   sequences' suffixes never corrupt each other or the shared prefix.

use super::KVCacheMode;
use super::paged::{PagedBlockId, PagedBlockPool, PagedKvLayout, PagedSequenceState};

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const H: i32 = 2; // n_kv_heads
const D: i32 = 3; // head_dim

fn fp16_pool(block_size: usize, num_layers: usize) -> PagedBlockPool {
    // bytes_per_block is only a scheduling budget; geometry is inferred from
    // the first written block, so any positive multiple of block_size works.
    let layout = PagedKvLayout::uniform(
        num_layers,
        block_size,
        block_size * H as usize * D as usize * 2,
    )
    .unwrap();
    PagedBlockPool::new(layout)
}

/// Deterministic `[1, H, n_slots, D]` FP32 block. Element value encodes
/// `(base, head, slot, dim)` so any misplacement is caught exactly. FP32 round
/// trips bit-exactly through MLX, so byte comparison is meaningful.
fn make_block(base: f32, n_slots: i32) -> UniquePtr<MlxArray> {
    let mut values = Vec::with_capacity((H * n_slots * D) as usize);
    for head in 0..H {
        for slot in 0..n_slots {
            for dim in 0..D {
                values.push(base + head as f32 * 1000.0 + slot as f32 * 10.0 + dim as f32 * 0.1);
            }
        }
    }
    ffi::from_slice_f32(&values, &[1, H, n_slots, D])
}

/// Same logical block as [`make_block`], emitted as the bare layout-A slab
/// `[n_slots, H, D]` to exercise the 3D write convention.
fn make_block_3d(base: f32, n_slots: i32) -> UniquePtr<MlxArray> {
    // Build [1, H, n_slots, D] then transpose to [n_slots, H, D].
    let four_d = make_block(base, n_slots);
    let t = ffi::transpose_axes(&four_d, &[0, 2, 1, 3]); // [1, n_slots, H, D]
    ffi::reshape(&t, &[n_slots, H, D])
}

/// Deterministic `[1, H, n_slots, D]` INT8 block.
fn make_int8_block(base: i8, n_slots: i32) -> UniquePtr<MlxArray> {
    let mut bytes = Vec::with_capacity((H * n_slots * D) as usize);
    for head in 0..H {
        for slot in 0..n_slots {
            for dim in 0..D {
                let v = base
                    .wrapping_add((head as i8).wrapping_mul(7))
                    .wrapping_add((slot as i8).wrapping_mul(3))
                    .wrapping_add(dim as i8);
                bytes.push(v as u8);
            }
        }
    }
    ffi::from_bytes(&bytes, &[1, H, n_slots, D], dtype::INT8)
}

/// Flatten any tensor to a row-major Vec<f32> (after an FP32 cast) so contents
/// can be compared independent of stride. Mirrors `detach_tests::flatten_fp32`.
fn flatten_fp32(arr: &MlxArray) -> Vec<f32> {
    let a = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&a);
    let bytes = ffi::array_to_raw_bytes(&a);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Raw little-endian bytes of an evaluated tensor (no dtype cast). Used for the
/// INT8 byte-identity check.
fn raw_bytes(arr: &MlxArray) -> Vec<u8> {
    ffi::eval(arr);
    ffi::array_to_raw_bytes(arr)
}

/// Build the dense contiguous reference `[1, H, T, D]` by writing each
/// `[1, H, n_slots, D]` block at its logical token offset, then return the
/// visible slice `[0, 0, 0, visible_len]` as `[1, H, visible_len, D]`.
fn dense_reference(
    blocks: &[(UniquePtr<MlxArray>, usize)], // (block, slot_start within the dense buffer)
    total_tokens: i32,
    visible_start: i32,
    visible_len: i32,
) -> UniquePtr<MlxArray> {
    let mut dense = ffi::zeros(&[1, H, total_tokens, D], dtype::FLOAT32);
    for (block, offset) in blocks {
        let shape = ffi::array_shape(block);
        let n_slots = shape[2];
        let starts = [0, 0, *offset as i32, 0];
        let stops = [1, H, *offset as i32 + n_slots, D];
        dense = ffi::slice_update(&dense, block, &starts, &stops);
    }
    ffi::slice(
        &dense,
        &[0, 0, visible_start, 0],
        &[1, H, visible_start + visible_len, D],
    )
}

// ---------------------------------------------------------------------------
// 1. Acceptance: contiguous write -> gather is byte-identical to dense
// ---------------------------------------------------------------------------

#[test]
fn contiguous_gather_is_byte_identical_to_dense() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    // 3 full blocks => 12 tokens.
    let n_blocks = 3i32;
    let total = n_blocks * block_size as i32;
    pool.append_tokens(&mut state, 0, total as usize).unwrap();
    let block_ids = state.layer(0).unwrap().block_ids.clone();
    assert_eq!(block_ids.len(), n_blocks as usize);

    // Write each block with distinct data; collect the same data for the
    // dense reference.
    let mut dense_blocks: Vec<(UniquePtr<MlxArray>, usize)> = Vec::new();
    for (b, &block_id) in block_ids.iter().enumerate() {
        let k = make_block(100.0 + b as f32, block_size as i32);
        let v = make_block(500.0 + b as f32, block_size as i32);
        pool.write_block(block_id, 0, 0, &k, &v).unwrap();
        dense_blocks.push((k, b * block_size));
        // V reference handled separately below by re-deriving the same base.
        let _ = v;
    }

    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    // Shape is [1, H, visible_len, D].
    assert_eq!(ffi::array_shape(&gk), vec![1, H, total, D]);

    let dense_k = dense_reference(&dense_blocks, total, 0, total);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));

    // Rebuild dense V reference from the same value bases.
    let mut dense_v_blocks: Vec<(UniquePtr<MlxArray>, usize)> = Vec::new();
    for b in 0..n_blocks as usize {
        dense_v_blocks.push((
            make_block(500.0 + b as f32, block_size as i32),
            b * block_size,
        ));
    }
    let dense_v = dense_reference(&dense_v_blocks, total, 0, total);
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

// ---------------------------------------------------------------------------
// 2. Fragmented / out-of-order physical rows gather in block-table order
// ---------------------------------------------------------------------------

#[test]
fn fragmented_rows_gather_in_block_table_order() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);

    // Allocate three sequences' worth of blocks so rows interleave, then build
    // ONE sequence whose block table references rows out of physical order.
    // We acquire blocks by appending to three throwaway states, write into
    // them in a scrambled order, and then assemble a sequence state by hand.
    let mut s = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut s, 0, 3 * block_size).unwrap();
    let ids = s.layer(0).unwrap().block_ids.clone(); // [b0, b1, b2] -> rows 0,1,2

    // Write the THIRD block first, then first, then second, so the row
    // assignment order (0,1,2 by acquisition) is decoupled from write order.
    // Row assignment happens on first write, so writing in (b2, b0, b1) order
    // assigns rows 0->b2, 1->b0, 2->b1.
    pool.write_block(
        ids[2],
        0,
        0,
        &make_block(20.0, block_size as i32),
        &make_block(70.0, block_size as i32),
    )
    .unwrap();
    pool.write_block(
        ids[0],
        0,
        0,
        &make_block(10.0, block_size as i32),
        &make_block(60.0, block_size as i32),
    )
    .unwrap();
    pool.write_block(
        ids[1],
        0,
        0,
        &make_block(30.0, block_size as i32),
        &make_block(80.0, block_size as i32),
    )
    .unwrap();

    // The sequence's block table is in logical order [b0, b1, b2], whose
    // physical rows are [1, 2, 0] — genuinely scattered.
    let (gk, _gv) = pool
        .gather_visible(&s, 0)
        .unwrap()
        .expect("gather must return data");

    // Dense reference in block-table order: b0 (base 10), b1 (30), b2 (20).
    let total = 3 * block_size as i32;
    let dense_blocks = vec![
        (make_block(10.0, block_size as i32), 0),
        (make_block(30.0, block_size as i32), block_size),
        (make_block(20.0, block_size as i32), 2 * block_size),
    ];
    let dense_k = dense_reference(&dense_blocks, total, 0, total);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
}

// ---------------------------------------------------------------------------
// 3. Partial final block gathers exactly visible_len (no padding leakage)
// ---------------------------------------------------------------------------

#[test]
fn partial_final_block_gathers_exactly_visible_len() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    // 6 tokens => 2 blocks, last block half-full (slots 0,1 valid; 2,3 pad).
    pool.append_tokens(&mut state, 0, 6).unwrap();
    let ids = state.layer(0).unwrap().block_ids.clone();
    assert_eq!(ids.len(), 2);

    // Block 0 full (4 slots), block 1 has 2 slots written.
    pool.write_block(ids[0], 0, 0, &make_block(100.0, 4), &make_block(500.0, 4))
        .unwrap();
    pool.write_block(ids[1], 0, 0, &make_block(200.0, 2), &make_block(600.0, 2))
        .unwrap();

    let (gk, _gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    // Exactly 6 visible tokens, NOT 8.
    assert_eq!(ffi::array_shape(&gk), vec![1, H, 6, D]);

    // Dense reference: 6-wide buffer (no padding slots exist at all).
    let dense_blocks = vec![(make_block(100.0, 4), 0), (make_block(200.0, 2), 4)];
    let dense_k = dense_reference(&dense_blocks, 6, 0, 6);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
}

// ---------------------------------------------------------------------------
// 4. logical_start > 0 (post-trim) gathers the correct window
// ---------------------------------------------------------------------------

#[test]
fn logical_start_offset_gathers_correct_window() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    // 12 tokens, 3 full blocks, all written.
    pool.append_tokens(&mut state, 0, 12).unwrap();
    let ids = state.layer(0).unwrap().block_ids.clone();
    for (b, &id) in ids.iter().enumerate() {
        pool.write_block(
            id,
            0,
            0,
            &make_block(100.0 + b as f32, 4),
            &make_block(500.0 + b as f32, 4),
        )
        .unwrap();
    }

    // Simulate a sliding-window trim of the first 5 tokens: bump
    // logical_start. (This is the post-trim state the decode path would see;
    // gather must return tokens [5, 12) = 7 tokens.)
    state.layer_mut(0).unwrap().logical_start = 5;
    assert_eq!(state.layer(0).unwrap().visible_len(), 7);

    let (gk, _gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    assert_eq!(ffi::array_shape(&gk), vec![1, H, 7, D]);

    // Dense reference: full 12-wide buffer, sliced to the visible window [5,12).
    let dense_blocks = vec![
        (make_block(100.0, 4), 0),
        (make_block(101.0, 4), 4),
        (make_block(102.0, 4), 8),
    ];
    let dense_k = dense_reference(&dense_blocks, 12, 5, 7);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
}

// ---------------------------------------------------------------------------
// 5. INT8 main K/V round-trips byte-identically (dtype preserved)
// ---------------------------------------------------------------------------

#[test]
fn int8_main_kv_round_trips_byte_identically() {
    let block_size = 4usize;
    let layout =
        PagedKvLayout::new_with_mode(block_size, vec![512], KVCacheMode::Int8, Vec::new()).unwrap();
    let mut pool = PagedBlockPool::new(layout);
    let mut state = PagedSequenceState::new(pool.layout());

    pool.append_tokens(&mut state, 0, 8).unwrap(); // 2 blocks
    let ids = state.layer(0).unwrap().block_ids.clone();
    pool.write_block(
        ids[0],
        0,
        0,
        &make_int8_block(1, 4),
        &make_int8_block(50, 4),
    )
    .unwrap();
    pool.write_block(
        ids[1],
        0,
        0,
        &make_int8_block(100, 4),
        &make_int8_block(-40, 4),
    )
    .unwrap();

    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    // dtype must remain INT8 (no astype on the K/V path).
    assert_eq!(ffi::array_dtype(&gk), dtype::INT8);
    assert_eq!(ffi::array_dtype(&gv), dtype::INT8);
    assert_eq!(ffi::array_shape(&gk), vec![1, H, 8, D]);

    // Byte-identity vs a dense INT8 buffer.
    let mut dense_k = ffi::zeros(&[1, H, 8, D], dtype::INT8);
    dense_k = ffi::slice_update(
        &dense_k,
        &make_int8_block(1, 4),
        &[0, 0, 0, 0],
        &[1, H, 4, D],
    );
    dense_k = ffi::slice_update(
        &dense_k,
        &make_int8_block(100, 4),
        &[0, 0, 4, 0],
        &[1, H, 8, D],
    );
    assert_eq!(raw_bytes(&gk), raw_bytes(&dense_k));

    let mut dense_v = ffi::zeros(&[1, H, 8, D], dtype::INT8);
    dense_v = ffi::slice_update(
        &dense_v,
        &make_int8_block(50, 4),
        &[0, 0, 0, 0],
        &[1, H, 4, D],
    );
    dense_v = ffi::slice_update(
        &dense_v,
        &make_int8_block(-40, 4),
        &[0, 0, 4, 0],
        &[1, H, 8, D],
    );
    assert_eq!(raw_bytes(&gv), raw_bytes(&dense_v));
}

// ---------------------------------------------------------------------------
// 5b. FP16 main K/V round-trips byte-identically (dtype preserved, no astype)
// ---------------------------------------------------------------------------

#[test]
fn fp16_main_kv_round_trips_byte_identically() {
    // Build FP16 blocks by casting FP32 data through astype.  FP16 is exact
    // through pure data-movement ops (take/reshape/slice/transpose), so a
    // raw-byte comparison is meaningful and proves no silent dtype coercion.
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    pool.append_tokens(&mut state, 0, 8).unwrap(); // 2 blocks
    let ids = state.layer(0).unwrap().block_ids.clone();

    // Create FP16 blocks by casting known FP32 data.
    let k0 = ffi::astype(&make_block(1.0, 4), dtype::FLOAT16);
    let v0 = ffi::astype(&make_block(50.0, 4), dtype::FLOAT16);
    let k1 = ffi::astype(&make_block(100.0, 4), dtype::FLOAT16);
    let v1 = ffi::astype(&make_block(150.0, 4), dtype::FLOAT16);

    pool.write_block(ids[0], 0, 0, &k0, &v0).unwrap();
    pool.write_block(ids[1], 0, 0, &k1, &v1).unwrap();

    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");

    // dtype must remain FLOAT16 — no astype coercion on the K/V path.
    assert_eq!(ffi::array_dtype(&gk), dtype::FLOAT16);
    assert_eq!(ffi::array_dtype(&gv), dtype::FLOAT16);
    assert_eq!(ffi::array_shape(&gk), vec![1, H, 8, D]);

    // Byte-identity vs a dense FP16 reference built the same way.
    let mut dense_k = ffi::zeros(&[1, H, 8, D], dtype::FLOAT16);
    dense_k = ffi::slice_update(
        &dense_k,
        &ffi::astype(&make_block(1.0, 4), dtype::FLOAT16),
        &[0, 0, 0, 0],
        &[1, H, 4, D],
    );
    dense_k = ffi::slice_update(
        &dense_k,
        &ffi::astype(&make_block(100.0, 4), dtype::FLOAT16),
        &[0, 0, 4, 0],
        &[1, H, 8, D],
    );
    assert_eq!(raw_bytes(&gk), raw_bytes(&dense_k));

    let mut dense_v = ffi::zeros(&[1, H, 8, D], dtype::FLOAT16);
    dense_v = ffi::slice_update(
        &dense_v,
        &ffi::astype(&make_block(50.0, 4), dtype::FLOAT16),
        &[0, 0, 0, 0],
        &[1, H, 4, D],
    );
    dense_v = ffi::slice_update(
        &dense_v,
        &ffi::astype(&make_block(150.0, 4), dtype::FLOAT16),
        &[0, 0, 4, 0],
        &[1, H, 8, D],
    );
    assert_eq!(raw_bytes(&gv), raw_bytes(&dense_v));
}

// ---------------------------------------------------------------------------
// 6. release_block to refcount 0 frees + recycles a row with fresh data
// ---------------------------------------------------------------------------

#[test]
fn released_row_is_recycled_and_serves_fresh_data() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    // One block, written, then trimmed away (refcount -> 0, row freed).
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let first_id = state.layer(0).unwrap().block_ids[0];
    pool.write_block(first_id, 0, 0, &make_block(999.0, 4), &make_block(999.0, 4))
        .unwrap();
    let bytes_after_first = pool.pool_tensor_bytes();
    assert!(bytes_after_first > 0);

    pool.trim_tokens(&mut state, 0, 4).unwrap();
    assert_eq!(pool.refcount(first_id), 0);
    assert_eq!(state.layer(0).unwrap().block_ids.len(), 0);

    // Re-acquire (recycles the freed block id AND its row) and write NEW data.
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let second_id = state.layer(0).unwrap().block_ids[0];
    assert_eq!(second_id, first_id, "block id should be recycled");
    pool.write_block(second_id, 0, 0, &make_block(11.0, 4), &make_block(22.0, 4))
        .unwrap();

    // Pool tensor must not have grown (the row was reused, not appended).
    assert_eq!(pool.pool_tensor_bytes(), bytes_after_first);

    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    // Must serve the NEW data, with no stale bytes from the 999.0 write.
    let dense_k = dense_reference(&[(make_block(11.0, 4), 0)], 4, 0, 4);
    let dense_v = dense_reference(&[(make_block(22.0, 4), 0)], 4, 0, 4);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

// ---------------------------------------------------------------------------
// 7. pool_tensor_bytes reflects allocated pool tensors
// ---------------------------------------------------------------------------

#[test]
fn pool_tensor_bytes_tracks_allocation() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 2);
    let mut state = PagedSequenceState::new(pool.layout());

    // No writes yet: lazily-allocated pools stay None => zero bytes.
    assert_eq!(pool.pool_tensor_bytes(), 0);

    pool.append_tokens(&mut state, 0, 4).unwrap();
    let id0 = state.layer(0).unwrap().block_ids[0];
    pool.write_block(id0, 0, 0, &make_block(1.0, 4), &make_block(2.0, 4))
        .unwrap();
    // Layer 0 K and V pool tensors allocated at the grow chunk: 2 tensors *
    // (32 blocks * 4 slots * H * D * 4 bytes).
    let expected_layer0 = 2 * (32 * block_size as i32 * H * D * 4) as usize;
    assert_eq!(pool.pool_tensor_bytes(), expected_layer0);

    // Writing to layer 1 allocates an independent pair of pool tensors.
    pool.append_tokens(&mut state, 1, 4).unwrap();
    let id1 = state.layer(1).unwrap().block_ids[0];
    pool.write_block(id1, 1, 0, &make_block(3.0, 4), &make_block(4.0, 4))
        .unwrap();
    assert_eq!(pool.pool_tensor_bytes(), 2 * expected_layer0);
}

// ---------------------------------------------------------------------------
// 8. Turbo4 sidecars coexist with main-K/V rows
// ---------------------------------------------------------------------------

#[test]
fn turbo4_sidecars_coexist_with_main_kv_rows() {
    // Turbo4 pool: install a sidecar AND write the main K/V row for the same
    // block; both must be independently retrievable, and release frees both.
    let layout = PagedKvLayout::uniform_with_mode(1, 4, 128, KVCacheMode::Turbo4Asym, 16).unwrap();
    let mut pool = PagedBlockPool::new(layout);
    let mut state = PagedSequenceState::new(pool.layout());

    pool.append_tokens(&mut state, 0, 4).unwrap();
    let id = state.layer(0).unwrap().block_ids[0];

    // Main K/V row.
    pool.write_block(id, 0, 0, &make_block(7.0, 4), &make_block(8.0, 4))
        .unwrap();
    // Turbo4 per-page sidecar on the same block.
    pool.install_v_packed(
        id,
        ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]),
    )
    .unwrap();

    assert!(pool.v_packed_for(id).is_some());
    assert!(pool.pool_tensor_bytes() > 0);
    assert!(pool.turbo_sidecar_bytes() > 0);

    // Gather still returns the main K/V independent of the sidecar.
    let (gk, _gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    let dense_k = dense_reference(&[(make_block(7.0, 4), 0)], 4, 0, 4);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));

    // Releasing the block to refcount 0 frees BOTH the sidecar and the row.
    pool.release_sequence(&mut state).unwrap();
    assert_eq!(pool.refcount(id), 0);
    assert!(pool.v_packed_for(id).is_none());
    assert_eq!(pool.turbo_sidecar_bytes(), 0);

    // The freed row is reusable: a fresh block reuses it and the pool tensor
    // does not grow.
    let bytes_before_reuse = pool.pool_tensor_bytes();
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let id2 = state.layer(0).unwrap().block_ids[0];
    pool.write_block(id2, 0, 0, &make_block(9.0, 4), &make_block(10.0, 4))
        .unwrap();
    assert_eq!(pool.pool_tensor_bytes(), bytes_before_reuse);
}

// ---------------------------------------------------------------------------
// 9. Write convention: the bare 3D [n_slots, H, D] slab is accepted too
// ---------------------------------------------------------------------------

#[test]
fn write_accepts_bare_3d_block_layout() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let id = state.layer(0).unwrap().block_ids[0];

    // Write via the 3D convention; gather and compare to the 4D-built dense
    // reference using the same value base.
    pool.write_block(id, 0, 0, &make_block_3d(42.0, 4), &make_block_3d(84.0, 4))
        .unwrap();
    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    let dense_k = dense_reference(&[(make_block(42.0, 4), 0)], 4, 0, 4);
    let dense_v = dense_reference(&[(make_block(84.0, 4), 0)], 4, 0, 4);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

// ---------------------------------------------------------------------------
// 10. Validation: shape/dtype mismatch and empty-window behaviour
// ---------------------------------------------------------------------------

#[test]
fn write_rejects_geometry_mismatch_after_first_write() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 8).unwrap();
    let ids = state.layer(0).unwrap().block_ids.clone();

    pool.write_block(ids[0], 0, 0, &make_block(1.0, 4), &make_block(2.0, 4))
        .unwrap();

    // A second write with a different head_dim must be rejected.
    let bad = ffi::from_slice_f32(&vec![0.0; (H * 4 * (D + 1)) as usize], &[1, H, 4, D + 1]);
    let err = pool.write_block(ids[1], 0, 0, &bad, &bad).unwrap_err();
    assert!(err.contains("head_dim") || err.contains("expects"), "{err}");
}

#[test]
fn gather_returns_none_for_empty_layer() {
    let block_size = 4usize;
    let pool = fp16_pool(block_size, 1);
    let state = PagedSequenceState::new(pool.layout());
    // No tokens appended, no writes -> no visible window, no pool storage.
    assert!(pool.gather_visible(&state, 0).unwrap().is_none());
}

#[test]
fn write_rejects_unknown_block_and_oob_slot() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let bogus = PagedBlockId::from_raw(123_456);
    let k = make_block(1.0, 4);
    let err = pool.write_block(bogus, 0, 0, &k, &k).unwrap_err();
    assert!(err.contains("unknown block"), "{err}");

    // Known block, but slot range overruns block_size.
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let id = state.layer(0).unwrap().block_ids[0];
    let two = make_block(1.0, 2);
    let err = pool.write_block(id, 0, 3, &two, &two).unwrap_err();
    assert!(err.contains("out of bounds"), "{err}");
}

// ---------------------------------------------------------------------------
// 11. write_prefill (#120): cold bulk write -> gather is byte-identical to dense
// ---------------------------------------------------------------------------

#[test]
fn write_prefill_cold_round_trip_is_byte_identical_to_dense() {
    // T = 10 with block_size 4 => 2 full blocks + a half-full third block, so
    // the bulk write must chunk across a partial last block.
    let block_size = 4usize;
    let total = 10i32;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    // A single [1, H, T, D] prefill whose per-token values are distinct so any
    // mis-slotting is caught exactly (make_block encodes slot index in value).
    let k_prefill = make_block(100.0, total);
    let v_prefill = make_block(500.0, total);

    pool.write_prefill(&mut state, 0, &k_prefill, &v_prefill)
        .unwrap();

    // The bulk write must have allocated ceil(10/4) = 3 blocks and advanced len.
    assert_eq!(state.layer(0).unwrap().block_ids.len(), 3);
    assert_eq!(state.layer(0).unwrap().len, total as usize);
    assert_eq!(state.layer(0).unwrap().visible_len(), total as usize);

    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    assert_eq!(ffi::array_shape(&gk), vec![1, H, total, D]);

    // Dense reference: the same [1, H, T, D] buffer written at offset 0.
    let dense_k = dense_reference(&[(make_block(100.0, total), 0)], total, 0, total);
    let dense_v = dense_reference(&[(make_block(500.0, total), 0)], total, 0, total);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

#[test]
fn write_prefill_block_aligned_prompt_round_trips() {
    // Block-aligned T (== 2 * block_size) takes the no-COW fresh-block path.
    let block_size = 4usize;
    let total = 8i32;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    let k_prefill = make_block(11.0, total);
    let v_prefill = make_block(77.0, total);
    pool.write_prefill(&mut state, 0, &k_prefill, &v_prefill)
        .unwrap();
    assert_eq!(state.layer(0).unwrap().block_ids.len(), 2);

    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    let dense_k = dense_reference(&[(make_block(11.0, total), 0)], total, 0, total);
    let dense_v = dense_reference(&[(make_block(77.0, total), 0)], total, 0, total);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

// ---------------------------------------------------------------------------
// 12. write_prefill copy-on-write: a shared partial tail block is forked so two
//     sequences' suffixes never corrupt each other or the shared prefix.
//
// This is the PagedBlockPool-level COW proof. Two sequence states share the
// same block table (the second adopts the first's block_ids with a refcount
// bump via retain_block, exactly as CachePool::adopt_paged does), where the
// last shared block is PARTIALLY filled (the prefix ends mid-block). Each
// sequence then write_prefills a DIFFERENT suffix; the partial tail block is
// refcount==2 at suffix time, so write_prefill must copy-on-write it for the
// writer rather than mutating the block the sibling still references.
// ---------------------------------------------------------------------------

#[test]
fn write_prefill_cow_forks_shared_partial_tail_block() {
    let block_size = 4usize;
    let prefix_len = 6i32; // 2 blocks; second block half-full (slots 0,1).
    let mut pool = fp16_pool(block_size, 1);

    // --- Build the shared prefix on sequence A. ---
    let mut state_a = PagedSequenceState::new(pool.layout());
    let prefix_k = make_block(1000.0, prefix_len);
    let prefix_v = make_block(5000.0, prefix_len);
    pool.write_prefill(&mut state_a, 0, &prefix_k, &prefix_v)
        .unwrap();
    let prefix_blocks = state_a.layer(0).unwrap().block_ids.clone();
    assert_eq!(prefix_blocks.len(), 2);
    let tail_block = prefix_blocks[1];
    assert_eq!(pool.refcount(tail_block), 1);

    // --- Sequence B adopts the SAME block table, pinning every prefix block
    //     (this is what CachePool::adopt_paged does for a shared prefix). ---
    let mut state_b = PagedSequenceState::new(pool.layout());
    {
        let layer_b = state_b.layer_mut(0).unwrap();
        layer_b.block_ids = prefix_blocks.clone();
        layer_b.len = prefix_len as usize;
        layer_b.logical_start = 0;
    }
    for &id in &prefix_blocks {
        pool.retain_block(id).unwrap();
    }
    // The partial tail block is now shared by both sequences.
    assert_eq!(pool.refcount(tail_block), 2);

    // --- Each sequence writes a DIFFERENT 5-token suffix (positions [6, 11)).
    //     The first suffix block is the shared partial tail (slots 2,3), which
    //     must be copy-on-written for each writer. ---
    let suffix_a_k = make_block(2000.0, 5);
    let suffix_a_v = make_block(6000.0, 5);
    pool.write_prefill(&mut state_a, 0, &suffix_a_k, &suffix_a_v)
        .unwrap();

    let suffix_b_k = make_block(3000.0, 5);
    let suffix_b_v = make_block(7000.0, 5);
    pool.write_prefill(&mut state_b, 0, &suffix_b_k, &suffix_b_v)
        .unwrap();

    // --- COW accounting. A wrote first: at that moment the tail was shared
    //     (refcount 2), so A copy-on-wrote it to a fresh block and released its
    //     reference to the original (2 -> 1). B wrote second: by then B was the
    //     SOLE owner of the original tail (refcount 1), so the in-place write is
    //     safe and no fork is needed — B keeps the original block. This is
    //     exactly the refcount-driven COW contract: copy only while shared. ---
    let a_tail = state_a.layer(0).unwrap().block_ids[1];
    let b_tail = state_b.layer(0).unwrap().block_ids[1];
    assert_ne!(
        a_tail, tail_block,
        "A wrote while the tail was shared, so it must have forked a fresh copy"
    );
    assert_eq!(
        b_tail, tail_block,
        "B wrote while it was the sole owner of the tail, so it keeps the block in place"
    );
    assert_ne!(a_tail, b_tail, "A and B must hold independent tail blocks");
    assert_eq!(
        pool.refcount(tail_block),
        1,
        "the original tail is now solely owned by B"
    );
    assert_eq!(
        pool.refcount(a_tail),
        1,
        "A's forked tail copy is solely owned by A"
    );
    // The first shared block (fully inside the prefix) is never written, so it
    // is still shared by both sequences.
    assert_eq!(pool.refcount(prefix_blocks[0]), 2);

    // --- Correctness: each sequence gathers its OWN shared-prefix + own suffix,
    //     proving the two suffixes did not corrupt each other or the prefix. ---
    // Sequence A: prefix tokens [0,6) (base 1000) then suffix tokens [6,11)
    // (base 2000, i.e. dense-token index 6 carries suffix slot 0).
    let total = 11i32;
    let (gk_a, gv_a) = pool.gather_visible(&state_a, 0).unwrap().expect("gather A");
    let dense_a_k = dense_reference(
        &[
            (make_block(1000.0, prefix_len), 0),
            (make_block(2000.0, 5), prefix_len as usize),
        ],
        total,
        0,
        total,
    );
    let dense_a_v = dense_reference(
        &[
            (make_block(5000.0, prefix_len), 0),
            (make_block(6000.0, 5), prefix_len as usize),
        ],
        total,
        0,
        total,
    );
    assert_eq!(flatten_fp32(&gk_a), flatten_fp32(&dense_a_k));
    assert_eq!(flatten_fp32(&gv_a), flatten_fp32(&dense_a_v));

    // Sequence B: same prefix, DIFFERENT suffix (base 3000 / 7000).
    let (gk_b, gv_b) = pool.gather_visible(&state_b, 0).unwrap().expect("gather B");
    let dense_b_k = dense_reference(
        &[
            (make_block(1000.0, prefix_len), 0),
            (make_block(3000.0, 5), prefix_len as usize),
        ],
        total,
        0,
        total,
    );
    let dense_b_v = dense_reference(
        &[
            (make_block(5000.0, prefix_len), 0),
            (make_block(7000.0, 5), prefix_len as usize),
        ],
        total,
        0,
        total,
    );
    assert_eq!(flatten_fp32(&gk_b), flatten_fp32(&dense_b_k));
    assert_eq!(flatten_fp32(&gv_b), flatten_fp32(&dense_b_v));
}

// ---------------------------------------------------------------------------
// 13. read_block_contents + acquire_and_write_block (#125): a multi-block
//     prefill round-trips byte-identically through a SECOND (decode-node) pool.
//
// Models the cross-node block-content transfer: the origin pool's blocks are
// read out slab-by-slab (`read_block_contents`), then re-materialized on a
// fresh pool (`acquire_and_write_block`) with a remapped block table. The
// gathered visible window must be byte-identical, the fresh ids must be
// independent of (here, disjoint from) the origin ids, and block accounting
// (per-block refcount + live count) must match the origin.
// ---------------------------------------------------------------------------

#[test]
fn extract_then_restore_with_contents_round_trips() {
    let block_size = 4usize;
    let total = 10i32; // 2 full blocks + a half-full third (partial-tail padding).
    let mut origin = fp16_pool(block_size, 1);
    let mut origin_state = PagedSequenceState::new(origin.layout());

    let k_prefill = make_block(100.0, total);
    let v_prefill = make_block(500.0, total);
    origin
        .write_prefill(&mut origin_state, 0, &k_prefill, &v_prefill)
        .unwrap();

    let origin_blocks = origin_state.layer(0).unwrap().block_ids.clone();
    assert_eq!(origin_blocks.len(), 3);
    let origin_live = origin.live_block_count();
    assert_eq!(origin_live, 3);
    let origin_max_id = origin_blocks
        .iter()
        .map(|b| b.as_u64())
        .max()
        .expect("non-empty block table");

    // Baseline visible window from the origin pool.
    let (origin_gk, origin_gv) = origin
        .gather_visible(&origin_state, 0)
        .unwrap()
        .expect("origin gather must return data");

    // "Transfer": read every block's full [block_size, H, D] slab out of origin.
    let transferred: Vec<(u64, UniquePtr<MlxArray>, UniquePtr<MlxArray>)> = origin_blocks
        .iter()
        .map(|&block_id| {
            let (k, v) = origin.read_block_contents(block_id, 0).unwrap();
            (block_id.as_u64(), k, v)
        })
        .collect();

    // Decode node: a fresh pool whose id counter is pre-advanced past the
    // origin's max (a real decode pool is rarely pristine), so the restored
    // blocks get provably disjoint fresh ids.
    let mut decode = fp16_pool(block_size, 1);
    let mut resident = PagedSequenceState::new(decode.layout());
    decode
        .append_tokens(&mut resident, 0, block_size * origin_blocks.len())
        .unwrap();
    let resident_live = decode.live_block_count();
    assert_eq!(resident_live, origin_blocks.len());

    // Acquire + write each transferred block, remapping origin id -> fresh id.
    let mut id_map: HashMap<u64, PagedBlockId> = HashMap::new();
    for (origin_id, k, v) in &transferred {
        let fresh = decode.acquire_and_write_block(0, k, v).unwrap();
        id_map.insert(*origin_id, fresh);
    }

    // Rebuild the block table over the fresh ids (same len / logical_start).
    let mut decode_state = PagedSequenceState::new(decode.layout());
    {
        let layer = decode_state.layer_mut(0).unwrap();
        layer.block_ids = origin_blocks
            .iter()
            .map(|origin_id| id_map[&origin_id.as_u64()])
            .collect();
        layer.len = origin_state.layer(0).unwrap().len;
        layer.logical_start = origin_state.layer(0).unwrap().logical_start;
    }

    // Fresh ids are independent of the origin ids (disjoint here)...
    for fresh in id_map.values() {
        assert!(
            fresh.as_u64() > origin_max_id,
            "decode pool must allocate fresh ids past the origin's (got {}, origin max {origin_max_id})",
            fresh.as_u64()
        );
    }

    // ...but the gathered content is byte-identical to the origin window.
    let (decode_gk, decode_gv) = decode
        .gather_visible(&decode_state, 0)
        .unwrap()
        .expect("decode gather must return data");
    assert_eq!(flatten_fp32(&origin_gk), flatten_fp32(&decode_gk));
    assert_eq!(flatten_fp32(&origin_gv), flatten_fp32(&decode_gv));

    // Block accounting matches the origin: every restored block is solely owned
    // (refcount 1) and the decode pool's live count is the resident blocks plus
    // exactly the origin's live count (no double-allocation, no leak).
    for fresh in id_map.values() {
        assert_eq!(decode.refcount(*fresh), 1);
    }
    assert_eq!(decode.live_block_count(), resident_live + origin_live);
}

/// The distributed handoff (#125) ships pool blocks through `array_to_raw_bytes`
/// -> wire -> `from_bytes`. That byte round-trip must preserve 16-bit float
/// content EXACTLY. A `from_bytes` bug used to read fp16/bf16 bytes as per-byte
/// `uint8`->float casts (reading half the bytes, one value per byte), silently
/// corrupting every transferred block; the single-layer FP32 round-trip above
/// could not catch it because FP32 has an explicit `from_bytes` case. This
/// exercises the exact serde path in both 16-bit dtypes across two physical
/// blocks (one partial tail), comparing the gathered window byte-for-byte.
fn assert_byte_roundtrip_preserves_content(cast_dtype: i32) {
    let block_size = 4usize;
    let total = 10i32; // 2 full blocks + a half-full third (partial tail).
    let mut origin = fp16_pool(block_size, 1);
    let mut origin_state = PagedSequenceState::new(origin.layout());

    // Cast the deterministic content to the real pool-backed 16-bit dtype.
    let k_prefill = ffi::astype(&make_block(100.0, total), cast_dtype);
    let v_prefill = ffi::astype(&make_block(500.0, total), cast_dtype);
    origin
        .write_prefill(&mut origin_state, 0, &k_prefill, &v_prefill)
        .unwrap();

    let origin_blocks = origin_state.layer(0).unwrap().block_ids.clone();
    let (origin_gk, origin_gv) = origin
        .gather_visible(&origin_state, 0)
        .unwrap()
        .expect("origin gather");

    // Transfer each block through the raw byte wire (read -> bytes -> from_bytes).
    let transferred: Vec<(u64, UniquePtr<MlxArray>, UniquePtr<MlxArray>)> = origin_blocks
        .iter()
        .map(|&block_id| {
            let (k, v) = origin.read_block_contents(block_id, 0).unwrap();
            let kk = ffi::from_bytes(
                &ffi::array_to_raw_bytes(&k),
                &ffi::array_shape(&k),
                ffi::array_dtype(&k),
            );
            let vv = ffi::from_bytes(
                &ffi::array_to_raw_bytes(&v),
                &ffi::array_shape(&v),
                ffi::array_dtype(&v),
            );
            (block_id.as_u64(), kk, vv)
        })
        .collect();

    let mut decode = fp16_pool(block_size, 1);
    let mut decode_state = PagedSequenceState::new(decode.layout());
    let mut id_map: HashMap<u64, PagedBlockId> = HashMap::new();
    for (origin_id, k, v) in &transferred {
        id_map.insert(*origin_id, decode.acquire_and_write_block(0, k, v).unwrap());
    }
    {
        let layer = decode_state.layer_mut(0).unwrap();
        layer.block_ids = origin_blocks.iter().map(|o| id_map[&o.as_u64()]).collect();
        layer.len = origin_state.layer(0).unwrap().len;
        layer.logical_start = origin_state.layer(0).unwrap().logical_start;
    }
    let (decode_gk, decode_gv) = decode
        .gather_visible(&decode_state, 0)
        .unwrap()
        .expect("decode gather");

    assert_eq!(
        flatten_fp32(&origin_gk),
        flatten_fp32(&decode_gk),
        "K content corrupted by the byte round-trip (dtype {cast_dtype})"
    );
    assert_eq!(
        flatten_fp32(&origin_gv),
        flatten_fp32(&decode_gv),
        "V content corrupted by the byte round-trip (dtype {cast_dtype})"
    );
}

/// `acquire_and_write_block` mints a fresh block before the fallible write. If
/// the write is rejected (a malformed or oversized transferred slab on the #125
/// restore path) the just-minted block must be released, not leaked. After a
/// rejected write the pool's live block count must be unchanged.
#[test]
fn acquire_and_write_block_releases_on_write_failure() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);

    // Capture this layer's geometry (n_kv_heads = H, head_dim = D) with one
    // good block, so a later mismatched slab is rejected by write_block.
    let _good = pool
        .acquire_and_write_block(
            0,
            &make_block_3d(0.0, block_size as i32),
            &make_block_3d(500.0, block_size as i32),
        )
        .unwrap();
    assert_eq!(pool.live_block_count(), 1);

    // A slab whose n_kv_heads disagrees with the captured meta (H + 1 vs H) is
    // rejected by write_block; the block acquired for it must be released.
    let bad = ffi::from_slice_f32(
        &vec![0.0f32; (block_size as i32 * (H + 1) * D) as usize],
        &[block_size as i32, H + 1, D],
    );
    assert!(pool.acquire_and_write_block(0, &bad, &bad).is_err());
    assert_eq!(
        pool.live_block_count(),
        1,
        "a rejected write must not leak the freshly acquired block"
    );
}

#[test]
fn paged_block_byte_roundtrip_preserves_fp16() {
    assert_byte_roundtrip_preserves_content(dtype::FLOAT16);
}

#[test]
fn paged_block_byte_roundtrip_preserves_bf16() {
    assert_byte_roundtrip_preserves_content(dtype::BFLOAT16);
}

// ---------------------------------------------------------------------------
// Issue #196: absolute block indexing under logical_start > 0
// ---------------------------------------------------------------------------

/// `write_prefill` after a sliding-window advance (`logical_start > 0`) must
/// land the suffix on the correct ABSOLUTE blocks and round-trip through
/// `gather_visible`.
#[test]
fn write_prefill_round_trips_with_logical_start() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    // Prefill 12 tokens cold (3 blocks), then slide the window forward by 5.
    let k0 = make_block(100.0, 12);
    let v0 = make_block(500.0, 12);
    pool.write_prefill(&mut state, 0, &k0, &v0).unwrap();
    state.layer_mut(0).unwrap().logical_start = 5;

    // Write a 6-token suffix: absolute positions [12, 18) spanning blocks 3-4.
    let k1 = make_block(112.0, 6);
    let v1 = make_block(512.0, 6);
    pool.write_prefill(&mut state, 0, &k1, &v1).unwrap();

    let layer = state.layer(0).unwrap();
    assert_eq!(layer.len, 18);
    assert_eq!(layer.visible_len(), 13);
    // Absolute sizing: ceil(18 / 4) = 5 blocks (visible-based sizing would
    // have allocated only ceil(13 / 4) = 4 and written past the table).
    assert_eq!(layer.block_ids.len(), 5);

    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    assert_eq!(ffi::array_shape(&gk), vec![1, H, 13, D]);

    // Dense reference: 18-token buffer (two writes), visible window [5, 18).
    let dense_blocks = vec![(make_block(100.0, 12), 0), (make_block(112.0, 6), 12)];
    let dense_k = dense_reference(&dense_blocks, 18, 5, 13);
    let dense_blocks_v = vec![(make_block(500.0, 12), 0), (make_block(512.0, 6), 12)];
    let dense_v = dense_reference(&dense_blocks_v, 18, 5, 13);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

/// A back-trim with `logical_start` past a block boundary must NOT release
/// tail blocks that still hold visible tokens (the old visible-length sizing
/// released them, and `gather_visible` then failed).
#[test]
fn trim_preserves_visible_window_with_logical_start_past_block_boundary() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    let k0 = make_block(100.0, 12);
    let v0 = make_block(500.0, 12);
    pool.write_prefill(&mut state, 0, &k0, &v0).unwrap();
    // logical_start (5) is past the first block boundary (4).
    state.layer_mut(0).unwrap().logical_start = 5;

    // Back-trim 2 tokens: len 12 -> 10, visible window [5, 10).
    let trimmed = pool.trim_tokens(&mut state, 0, 2).unwrap();
    assert_eq!(trimmed, 2);
    let layer = state.layer(0).unwrap();
    assert_eq!(layer.len, 10);
    assert_eq!(layer.visible_len(), 5);
    // Absolute sizing keeps ceil(10 / 4) = 3 blocks (visible-based sizing
    // would have popped block 2, which holds positions 8 and 9).
    assert_eq!(layer.block_ids.len(), 3);

    let (gk, _gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    assert_eq!(ffi::array_shape(&gk), vec![1, H, 5, D]);
    let dense_k = dense_reference(&[(make_block(100.0, 12), 0)], 12, 5, 5);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
}

/// A back-trim that consumes the whole visible window empties the layer:
/// origin reset, all blocks released, nothing left to gather.
#[test]
fn trim_to_logical_start_empties_the_layer() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    let k0 = make_block(100.0, 12);
    let v0 = make_block(500.0, 12);
    pool.write_prefill(&mut state, 0, &k0, &v0).unwrap();
    state.layer_mut(0).unwrap().logical_start = 5;

    // Trim the entire visible window (7 tokens).
    let trimmed = pool.trim_tokens(&mut state, 0, 7).unwrap();
    assert_eq!(trimmed, 7);
    let layer = state.layer(0).unwrap();
    assert_eq!(layer.len, 0);
    assert_eq!(layer.logical_start, 0);
    assert_eq!(layer.block_ids.len(), 0);
    assert!(pool.gather_visible(&state, 0).unwrap().is_none());
}

// ---------------------------------------------------------------------------
// 12. write_prefill presize (#224): one allocation for a multi-chunk span
// ---------------------------------------------------------------------------

#[test]
fn write_prefill_presizes_pool_in_one_step_for_multi_chunk_span() {
    // 100 blocks of 4 slots = 400 tokens, spanning four 32-row slabs.
    // presize_for_span must create the layer pool in one step at
    // ceil(100/32)*32 = 128 rows (four slabs) instead of growing through
    // intermediate capacities.
    let block_size = 4usize;
    let total = 400i32;
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    let k_prefill = make_block(100.0, total);
    let v_prefill = make_block(500.0, total);
    pool.write_prefill(&mut state, 0, &k_prefill, &v_prefill)
        .unwrap();

    assert_eq!(state.layer(0).unwrap().block_ids.len(), 100);
    assert_eq!(state.layer(0).unwrap().len, total as usize);

    // Capacity must be the single presized target: 128 rows for K and V, in
    // FP32 (make_block writes f32), 4 bytes per element.
    let expected_capacity_rows = 128usize;
    let expected_bytes = 2 * expected_capacity_rows * block_size * (H as usize) * (D as usize) * 4;
    assert_eq!(pool.pool_tensor_bytes(), expected_bytes);

    // The regression #224 fixed: presize allocates the pool at the final
    // size directly, so NO growth episode happened. The old incremental path
    // converged to the same 128 rows via three growth steps, so this
    // assertion (not the final capacity above) is what pins presize.
    assert_eq!(pool.pool_grow_events(), 0);

    // Round-trip stays byte-identical to the dense reference.
    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    let dense_k = dense_reference(&[(make_block(100.0, total), 0)], total, 0, total);
    let dense_v = dense_reference(&[(make_block(500.0, total), 0)], total, 0, total);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

#[test]
fn write_prefill_after_presize_grows_incrementally_and_round_trips() {
    // First prefill presizes to 32 rows (8 blocks used). A second prefill
    // pushing past the presized capacity must take the growth path
    // (ensure_layer_capacity appending a slab) and stay byte-identical.
    let block_size = 4usize;
    let first = 32i32; // 8 blocks
    let second = 160i32; // 40 more blocks -> 48 total, beyond the 32-row chunk
    let mut pool = fp16_pool(block_size, 1);
    let mut state = PagedSequenceState::new(pool.layout());

    let k1 = make_block(100.0, first);
    let v1 = make_block(500.0, first);
    pool.write_prefill(&mut state, 0, &k1, &v1).unwrap();
    let bytes_after_first = pool.pool_tensor_bytes();

    let k2 = make_block(200.0, second);
    let v2 = make_block(600.0, second);
    pool.write_prefill(&mut state, 0, &k2, &v2).unwrap();
    assert!(pool.pool_tensor_bytes() > bytes_after_first);
    assert_eq!(state.layer(0).unwrap().len, (first + second) as usize);

    // The second span needed 40 more rows past the presized 32: exactly one
    // growth episode (one ensure_layer_capacity call appending slabs).
    assert_eq!(pool.pool_grow_events(), 1);

    let total = first + second;
    let (gk, gv) = pool
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    let dense_k = dense_reference(
        &[
            (make_block(100.0, first), 0),
            (make_block(200.0, second), first as usize),
        ],
        total,
        0,
        total,
    );
    let dense_v = dense_reference(
        &[
            (make_block(500.0, first), 0),
            (make_block(600.0, second), first as usize),
        ],
        total,
        0,
        total,
    );
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

// ---------------------------------------------------------------------------
// 13. Chunked slabs (#235): growth appends slabs without copying, and gather
//     stays byte-identical when a block table crosses slabs non-monotonically.
// ---------------------------------------------------------------------------

#[test]
fn slab_growth_appends_without_copy_and_fragmented_gather_round_trips() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size, 1);

    // A: 40 blocks (160 tokens) -> rows 0..39, spanning two 32-row slabs.
    let mut state_a = PagedSequenceState::new(pool.layout());
    let a_tokens = 160i32;
    pool.write_prefill(
        &mut state_a,
        0,
        &make_block(100.0, a_tokens),
        &make_block(500.0, a_tokens),
    )
    .unwrap();
    let bytes_two_slabs = pool.pool_tensor_bytes();
    assert_eq!(
        pool.pool_grow_events(),
        0,
        "the presized first prefill creates slabs; creation is not growth"
    );

    // B: 26 more blocks push past the presized 64-row capacity (40 + 26 = 66)
    // and need a third slab. Exactly one growth episode and exactly one
    // slab's worth of new bytes per side (no copy, no ladder).
    let mut state_b = PagedSequenceState::new(pool.layout());
    let b_tokens = 104i32;
    pool.write_prefill(
        &mut state_b,
        0,
        &make_block(200.0, b_tokens),
        &make_block(600.0, b_tokens),
    )
    .unwrap();
    let slab_bytes_per_side = 32 * block_size * (H as usize) * (D as usize) * 4;
    assert_eq!(pool.pool_grow_events(), 1);
    assert_eq!(
        pool.pool_tensor_bytes(),
        bytes_two_slabs + 2 * slab_bytes_per_side,
        "growth must append exactly one slab per side"
    );

    // D: 6 more blocks (rows 66..71) so two release batches can interleave.
    let mut state_d = PagedSequenceState::new(pool.layout());
    pool.write_prefill(
        &mut state_d,
        0,
        &make_block(400.0, 24),
        &make_block(800.0, 24),
    )
    .unwrap();

    // Release A then D: the free list ends with D's rows, so C's LIFO reuse
    // starts in the THIRD slab (rows 66..71) and then drops back to row 0,
    // crossing slab boundaries non-monotonically. The gather must split the
    // row list into per-slab runs and still round-trip byte-identically.
    pool.release_sequence(&mut state_a).unwrap();
    pool.release_sequence(&mut state_d).unwrap();
    let mut state_c = PagedSequenceState::new(pool.layout());
    let c_tokens = 184i32;
    pool.write_prefill(
        &mut state_c,
        0,
        &make_block(300.0, c_tokens),
        &make_block(700.0, c_tokens),
    )
    .unwrap();
    {
        let layer = state_c.layer(0).unwrap();
        let rows: Vec<usize> = layer
            .block_ids
            .iter()
            .map(|id| pool.debug_row_of(*id, 0).unwrap())
            .collect();
        assert!(
            rows.windows(2).any(|w| w[0] > w[1]),
            "C must reuse freed rows non-monotonically to exercise run splitting (rows: {rows:?})"
        );
        let crosses = rows.windows(2).any(|w| w[0] / 32 != w[1] / 32);
        assert!(
            crosses,
            "C's rows must cross a slab boundary (rows: {rows:?})"
        );
    }

    let (gk, gv) = pool
        .gather_visible(&state_c, 0)
        .unwrap()
        .expect("gather must return data");
    let dense_k = dense_reference(&[(make_block(300.0, c_tokens), 0)], c_tokens, 0, c_tokens);
    let dense_v = dense_reference(&[(make_block(700.0, c_tokens), 0)], c_tokens, 0, c_tokens);
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));

    // B (rows spanning the second and third slabs) still round-trips too.
    let (gk_b, gv_b) = pool
        .gather_visible(&state_b, 0)
        .unwrap()
        .expect("gather must return data");
    let dense_kb = dense_reference(&[(make_block(200.0, b_tokens), 0)], b_tokens, 0, b_tokens);
    let dense_vb = dense_reference(&[(make_block(600.0, b_tokens), 0)], b_tokens, 0, b_tokens);
    assert_eq!(flatten_fp32(&gk_b), flatten_fp32(&dense_kb));
    assert_eq!(flatten_fp32(&gv_b), flatten_fp32(&dense_vb));
}
