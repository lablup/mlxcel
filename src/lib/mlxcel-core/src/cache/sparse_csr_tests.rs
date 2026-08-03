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

//! Structural tests for the sparse page table (issue #904).
//!
//! These are host-only: no MLX kernel runs here. What they pin is that the
//! selected set survives the trip through the CSR encoding unchanged, which is
//! the property the GPU test in `paged_v2/sparse_tests.rs` then checks the
//! kernel actually honours.

use super::*;

fn layout(batch: i32, heads: i32, capacity: i32) -> ContiguousCacheLayout {
    ContiguousCacheLayout {
        batch,
        buffer_heads: heads,
        capacity,
    }
}

#[test]
fn the_pool_row_of_a_token_is_its_offset_in_the_flat_allocation() {
    let l = layout(2, 3, 16);
    // [B, H, Cap, D] row-major: (b * H + h) * Cap + t.
    assert_eq!(l.row(0, 0, 0), 0);
    assert_eq!(l.row(0, 0, 5), 5);
    assert_eq!(l.row(0, 1, 0), 16);
    assert_eq!(l.row(0, 2, 15), 47);
    assert_eq!(l.row(1, 0, 0), 48);
    assert_eq!(l.row(1, 2, 15), 95);
    assert_eq!(l.pool_rows(), 96);
}

#[test]
fn the_layout_is_read_off_the_allocation_shape_not_the_live_window() {
    // The live window is a prefix of a step-padded allocation; the row stride
    // is the reserved capacity. Reading `live_len` here would alias heads.
    let l = ContiguousCacheLayout::from_shape(&[1, 8, 512, 128]).unwrap();
    assert_eq!(l.capacity, 512);
    assert_eq!(l.row(0, 1, 0), 512);
    assert!(ContiguousCacheLayout::from_shape(&[1, 8, 512]).is_err());
    assert!(ContiguousCacheLayout::from_shape(&[1, 8, 0, 128]).is_err());
}

#[test]
fn equal_shaped_k_and_v_allocations_share_a_row_mapping_at_any_batch() {
    let k = layout(4, 8, 256);
    let v = layout(4, 8, 256);
    shared_row_mapping(&k, &v, 8).unwrap();
}

#[test]
fn a_wider_k_allocation_shares_a_mapping_only_at_batch_one() {
    // MiniMax-M3 rides its index key on the K head axis, so K has one head
    // more than V. At batch 1 both bases collapse to h * Cap; beyond that the
    // sequence stride diverges and the selection is inexpressible.
    shared_row_mapping(&layout(1, 9, 256), &layout(1, 8, 256), 8).unwrap();
    let err = shared_row_mapping(&layout(2, 9, 256), &layout(2, 8, 256), 8).unwrap_err();
    assert!(err.contains("only share a row mapping at batch 1"), "{err}");
}

#[test]
fn a_capacity_disagreement_is_declined() {
    let err = shared_row_mapping(&layout(1, 8, 256), &layout(1, 8, 512), 8).unwrap_err();
    assert!(err.contains("capacity"), "{err}");
}

#[test]
fn attention_heads_may_not_exceed_the_allocation() {
    let err = shared_row_mapping(&layout(1, 8, 256), &layout(1, 8, 256), 9).unwrap_err();
    assert!(err.contains("exceed the allocation"), "{err}");
}

#[test]
fn the_csr_structure_is_derived_without_touching_the_rows() {
    let sel = SparseSelection::from_host(3, 4, (0..12).collect()).unwrap();
    let s = sel.structure();
    assert_eq!(s.page_size, TOKEN_EXACT_PAGE_SIZE);
    assert_eq!(s.indptr, vec![0, 4, 8, 12]);
    assert_eq!(s.last_page_len, vec![1, 1, 1]);
    assert_eq!(s.first_page_offset, vec![0, 0, 0]);
    assert_eq!(s.seq_lens, vec![4, 4, 4]);
    assert_eq!(s.page_counts(), vec![4, 4, 4]);
    assert_eq!(s.total_selected(), 12);
}

#[test]
fn the_csr_view_of_a_selection_passes_the_paged_invariants() {
    // The geometry identity `(pages - 1) * page_size + lpl - fpo == seq_len`
    // is what stops the kernel reading out of bounds, so a sparse view has to
    // satisfy the same check a dense one does.
    let sel = SparseSelection::from_host(2, 5, vec![9, 3, 7, 1, 0, 40, 33, 37, 31, 30]).unwrap();
    let view = sel.to_csr_view().unwrap();
    view.validate().unwrap();
    assert_eq!(view.page_size, 1);
    assert_eq!(view.batch(), 2);
    assert_eq!(view.total_pages(), 10);
    assert_eq!(view.page_counts(), vec![5, 5]);
    assert_eq!(view.max_seq_len(), 5);
    assert!(view.any_visible());
}

#[test]
fn an_unordered_selection_is_legal_because_page_lists_are_arbitrary() {
    // `argpartition` returns the selected positions in no particular order and
    // the kernel never assumes one; softmax is order-independent. Pinning this
    // stops a future "sort the indices" from being added as if it were needed.
    let sel = SparseSelection::from_host(1, 4, vec![31, 2, 17, 0]).unwrap();
    sel.to_csr_view().unwrap().validate().unwrap();
}

#[test]
fn a_selection_round_trips_through_the_encoding_unchanged() {
    let l = layout(2, 2, 32);
    let positions = vec![
        vec![vec![0, 5, 9], vec![1, 2, 3]],
        vec![vec![7, 8, 9], vec![0, 4, 8]],
    ];
    let sel = selection_from_positions(&l, 2, 10, &positions).unwrap();
    assert_eq!(sel.requests, 4);
    assert_eq!(sel.per_request, 3);

    // Decoding a row back to (b, h, t) must return the position that produced it.
    let rows = sel.materialize();
    assert_eq!(rows.len(), 4);
    for (b, heads) in positions.iter().enumerate() {
        for (h, want) in heads.iter().enumerate() {
            let r = b * 2 + h;
            let got: Vec<i32> = rows[r]
                .iter()
                .map(|&row| row - l.base(b as i32, h as i32))
                .collect();
            assert_eq!(&got, want, "request {r} (sequence {b}, head {h})");
        }
    }
}

#[test]
fn a_ragged_selection_is_rejected_rather_than_silently_padded() {
    let l = layout(1, 2, 32);
    let positions = vec![vec![vec![0, 1, 2], vec![0, 1]]];
    let err = selection_from_positions(&l, 2, 10, &positions).unwrap_err();
    assert!(err.contains("uniform"), "{err}");
}

#[test]
fn a_position_past_the_live_window_is_rejected() {
    // Requests are adjacent in the pool view, so an over-long position reads
    // the next head's tokens instead of failing. It has to be caught here.
    let l = layout(1, 2, 32);
    let positions = vec![vec![vec![0, 10], vec![0, 1]]];
    let err = selection_from_positions(&l, 2, 10, &positions).unwrap_err();
    assert!(err.contains("outside [0, 10)"), "{err}");
}

#[test]
fn block_expansion_names_exactly_the_selected_blocks_plus_the_tail() {
    // 1 sequence, 2 heads, capacity 64, live window 50, blocks of 16.
    // Whole blocks are 0..3 (48 tokens); the tail is [48, 50).
    let l = layout(1, 2, 64);
    let blocks = crate::ffi::from_slice_i32(&[2, 0], &[1, 2]);
    let sel = selection_from_blocks(&l, 2, 16, &blocks, 48, 50).unwrap();
    assert_eq!(sel.requests, 2);
    assert_eq!(sel.per_request, 2 * 16 + 2);

    let mut want: Vec<i32> = (32..48).collect(); // block 2
    want.extend(0..16); // block 0
    want.extend([48, 49]); // the partial final block

    for (h, row) in sel.materialize().iter().enumerate() {
        let decoded: Vec<i32> = row.iter().map(|&r| r - l.base(0, h as i32)).collect();
        assert_eq!(decoded, want, "head {h}");
    }
}

#[test]
fn block_expansion_gives_every_sequence_its_own_row_space() {
    let l = layout(2, 2, 32);
    // Sequence 0 picks block 1, sequence 1 picks block 0.
    let blocks = crate::ffi::from_slice_i32(&[1, 0], &[2, 1]);
    let sel = selection_from_blocks(&l, 2, 8, &blocks, 16, 20).unwrap();
    assert_eq!(sel.per_request, 8 + 4);
    let rows = sel.materialize();
    for b in 0..2usize {
        for h in 0..2usize {
            let r = b * 2 + h;
            let decoded: Vec<i32> = rows[r]
                .iter()
                .map(|&x| x - l.base(b as i32, h as i32))
                .collect();
            let block = if b == 0 { 1 } else { 0 };
            let mut want: Vec<i32> = (block * 8..block * 8 + 8).collect();
            want.extend(16..20);
            assert_eq!(decoded, want, "sequence {b} head {h}");
        }
    }
}

#[test]
fn block_expansion_rejects_a_tail_outside_the_live_window() {
    let l = layout(1, 1, 64);
    let blocks = crate::ffi::from_slice_i32(&[0], &[1, 1]);
    assert!(selection_from_blocks(&l, 1, 16, &blocks, 50, 50).is_err());
    assert!(selection_from_blocks(&l, 1, 16, &blocks, 48, 65).is_err());
    assert!(selection_from_blocks(&l, 1, 0, &blocks, 48, 50).is_err());
    assert!(selection_from_blocks(&l, 2, 16, &blocks, 48, 50).is_err());
}

#[test]
fn an_empty_or_mis_sized_selection_is_rejected() {
    assert!(SparseSelection::from_host(0, 4, vec![]).is_err());
    assert!(SparseSelection::from_host(2, 0, vec![]).is_err());
    assert!(SparseSelection::from_host(2, 3, vec![0, 1, 2]).is_err());
    assert!(SparseSelection::from_host(1, 2, vec![0, -1]).is_err());
}
