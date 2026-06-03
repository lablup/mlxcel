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

use super::paged::{PagedBlockId, PagedBlockPool, PagedKvLayout, PagedSequenceState};
use super::KVCacheMode;

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;

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
