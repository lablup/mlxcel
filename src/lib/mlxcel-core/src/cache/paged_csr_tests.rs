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

//! Unit tests for the CSR page-table view (issue #898).
//!
//! Two layers of coverage:
//!
//! 1. The pure builder against synthetic [`PagedLayerState`]s, where the block
//!    tables, the shared rows, and the trimmed windows can be constructed
//!    exactly. This is where the page-geometry identity is pinned.
//! 2. The pool method [`PagedBlockPool::paged_csr_view`] against a real pool,
//!    so the row resolution and the refcounted-block path are exercised
//!    end to end.

use super::*;
use crate::cache::paged::{PagedBlockPool, PagedKvLayout, PagedSequenceState};

/// Synthetic layer state: block ids `ids`, written length `len`, visible window
/// starting at `logical_start`.
fn layer(ids: &[u64], len: usize, logical_start: usize) -> PagedLayerState {
    PagedLayerState {
        block_ids: ids.iter().copied().map(PagedBlockId::from_raw).collect(),
        len,
        logical_start,
    }
}

/// Identity row mapping: block id `n` lives at pool row `n`.
fn identity_rows(id: PagedBlockId) -> Option<usize> {
    Some(id.as_u64() as usize)
}

fn build(page_size: usize, layers: &[&PagedLayerState]) -> PagedCsrView {
    build_paged_csr_view(page_size, layers, identity_rows).expect("view builds")
}

// ---------------------------------------------------------------------------
// Pure builder
// ---------------------------------------------------------------------------

#[test]
fn full_pages_produce_a_dense_csr() {
    // 3 full pages of 4 tokens, nothing trimmed.
    let a = layer(&[7, 8, 9], 12, 0);
    let view = build(4, &[&a]);

    assert_eq!(view.indices, vec![7, 8, 9]);
    assert_eq!(view.indptr, vec![0, 3]);
    assert_eq!(view.first_page_offset, vec![0]);
    assert_eq!(view.last_page_len, vec![4]);
    assert_eq!(view.seq_lens, vec![12]);
    assert_eq!(view.rope_offsets, vec![12]);
    assert_eq!(view.page_counts(), vec![3]);
    assert_eq!(view.max_seq_len(), 12);
    assert!(view.any_visible());
}

#[test]
fn partial_last_page_reports_its_valid_entries() {
    // 9 tokens over 4-token pages: 2 full pages plus 1 valid entry.
    let a = layer(&[0, 1, 2], 9, 0);
    let view = build(4, &[&a]);

    assert_eq!(view.indptr, vec![0, 3]);
    assert_eq!(view.last_page_len, vec![1]);
    assert_eq!(view.seq_lens, vec![9]);
}

#[test]
fn a_full_final_page_reports_page_size_not_zero() {
    // The `(len - 1) % page_size + 1` form exists so an exactly-full final page
    // reports `page_size`; a plain `len % page_size` would report 0 and make
    // the kernel skip the whole page.
    for page_size in [16usize, 32, 64] {
        let a = layer(&[0, 1], 2 * page_size, 0);
        let view = build(page_size, &[&a]);
        assert_eq!(view.last_page_len, vec![page_size as i32]);
        assert_eq!(view.seq_lens, vec![2 * page_size as i32]);
    }
}

#[test]
fn a_trimmed_window_drops_retired_pages_and_offsets_into_the_first() {
    // Sliding window trimmed the first 6 of 12 tokens (page size 4): pages 0 is
    // fully retired, page 1 starts at entry 2.
    let a = layer(&[10, 11, 12], 12, 6);
    let view = build(4, &[&a]);

    assert_eq!(view.indices, vec![11, 12], "retired page 0 is not emitted");
    assert_eq!(view.first_page_offset, vec![2]);
    assert_eq!(view.last_page_len, vec![4]);
    assert_eq!(view.seq_lens, vec![6]);
    // The absolute position of the next token is unaffected by the trim.
    assert_eq!(view.rope_offsets, vec![12]);
}

#[test]
fn geometry_identity_holds_across_a_randomized_sweep() {
    // (pages - 1) * page_size + last_page_len - first_page_offset == seq_len is
    // the identity the kernel's token->page arithmetic inverts, so sweep it.
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..500 {
        let page_size = [16usize, 32, 64][(next() % 3) as usize];
        let len = 1 + (next() % 400) as usize;
        let logical_start = (next() % len as u64) as usize;
        let n_blocks = len.div_ceil(page_size);
        let ids: Vec<u64> = (0..n_blocks as u64).collect();
        let l = layer(&ids, len, logical_start);
        let view = build(page_size, &[&l]);
        // `validate` already asserts the identity; re-derive it here so the
        // test fails on the identity itself rather than on a helper.
        let pages = view.pages_for(0) as i64;
        let expected = (pages - 1) * page_size as i64 + i64::from(view.last_page_len[0])
            - i64::from(view.first_page_offset[0]);
        assert_eq!(expected, i64::from(view.seq_lens[0]));
        assert_eq!(view.seq_lens[0] as usize, len - logical_start);
    }
}

#[test]
fn shared_blocks_appear_in_every_owner() {
    // Two requests share a two-page prefix (blocks 3 and 4) and diverge after.
    let a = layer(&[3, 4, 5], 12, 0);
    let b = layer(&[3, 4, 6], 10, 0);
    let view = build(4, &[&a, &b]);

    assert_eq!(view.indices, vec![3, 4, 5, 3, 4, 6]);
    assert_eq!(view.indptr, vec![0, 3, 6]);
    assert_eq!(view.seq_lens, vec![12, 10]);
    assert_eq!(view.last_page_len, vec![4, 2]);
    assert_eq!(view.page_counts(), vec![3, 3]);
}

#[test]
fn an_empty_request_contributes_no_pages() {
    let a = layer(&[1, 2], 8, 0);
    let empty = layer(&[], 0, 0);
    let c = layer(&[3], 3, 0);
    let view = build(4, &[&a, &empty, &c]);

    assert_eq!(view.indptr, vec![0, 2, 2, 3]);
    assert_eq!(view.seq_lens, vec![8, 0, 3]);
    assert_eq!(view.last_page_len, vec![4, 0, 3]);
    assert_eq!(view.first_page_offset, vec![0, 0, 0]);
    assert_eq!(view.pages_for(1), 0);
    assert!(view.any_visible());
}

#[test]
fn a_fully_trimmed_request_is_empty_but_keeps_its_rope_offset() {
    // logical_start == len: every token retired. The request contributes no
    // pages, but its next token still lands at absolute position `len`.
    let a = layer(&[0, 1], 8, 8);
    let view = build(4, &[&a]);

    assert!(view.indices.is_empty());
    assert_eq!(view.seq_lens, vec![0]);
    assert_eq!(view.rope_offsets, vec![8]);
    assert!(!view.any_visible());
}

#[test]
fn all_empty_batch_reports_no_visible_tokens() {
    let a = layer(&[], 0, 0);
    let b = layer(&[], 0, 0);
    let view = build(32, &[&a, &b]);
    assert!(!view.any_visible());
    assert_eq!(view.total_pages(), 0);
    assert_eq!(view.max_seq_len(), 0);
}

#[test]
fn len_outrunning_the_block_table_is_rejected() {
    // Two 4-token pages cannot hold 12 tokens.
    let a = layer(&[0, 1], 12, 0);
    let err = build_paged_csr_view(4, &[&a], identity_rows).unwrap_err();
    assert!(err.contains("exceeds"), "unexpected error: {err}");
}

#[test]
fn an_unwritten_block_is_rejected_rather_than_read() {
    let a = layer(&[0, 1], 8, 0);
    let err = build_paged_csr_view(4, &[&a], |id| if id.as_u64() == 1 { None } else { Some(0) })
        .unwrap_err();
    assert!(err.contains("no pool row"), "unexpected error: {err}");
}

#[test]
fn zero_page_size_is_rejected() {
    let a = layer(&[0], 1, 0);
    let err = build_paged_csr_view(0, &[&a], identity_rows).unwrap_err();
    assert!(err.contains("page_size"), "unexpected error: {err}");
}

#[test]
fn validate_rejects_a_hand_corrupted_view() {
    let a = layer(&[0, 1], 8, 0);
    let mut view = build(4, &[&a]);
    view.seq_lens[0] = 7; // contradicts the page geometry
    let err = view.validate().unwrap_err();
    assert!(err.contains("contradicts"), "unexpected error: {err}");

    let mut view = build(4, &[&a]);
    view.indptr.pop();
    assert!(view.validate().is_err());

    let mut view = build(4, &[&a]);
    view.last_page_len[0] = 0;
    assert!(view.validate().is_err());
}

// ---------------------------------------------------------------------------
// Through a real pool
// ---------------------------------------------------------------------------

const H: i32 = 2;
const D: i32 = 3;

fn fp16_pool(block_size: usize) -> PagedBlockPool {
    let layout =
        PagedKvLayout::uniform(1, block_size, block_size * H as usize * D as usize * 2).unwrap();
    PagedBlockPool::new(layout)
}

fn write_all_blocks(pool: &mut PagedBlockPool, state: &PagedSequenceState, block_size: usize) {
    let ids = state.layer(0).unwrap().block_ids.clone();
    for id in ids {
        let k = crate::ffi::zeros(&[1, H, block_size as i32, D], crate::dtype::FLOAT16);
        let v = crate::ffi::zeros(&[1, H, block_size as i32, D], crate::dtype::FLOAT16);
        pool.write_block(id, 0, 0, &k, &v).unwrap();
    }
}

#[test]
fn pool_view_resolves_physical_rows_in_block_table_order() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size);
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 10).unwrap();
    write_all_blocks(&mut pool, &state, block_size);

    let view = pool.paged_csr_view(&[&state], 0).unwrap();
    assert_eq!(view.page_size, block_size as i32);
    assert_eq!(view.indptr, vec![0, 3]);
    assert_eq!(view.seq_lens, vec![10]);
    assert_eq!(view.last_page_len, vec![2]);
    assert_eq!(view.first_page_offset, vec![0]);
    // Rows are assigned in first-write order, which is block-table order here.
    assert_eq!(view.indices, vec![0, 1, 2]);
}

#[test]
fn pool_view_shares_rows_between_prefix_sharing_sequences() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size);

    let mut a = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut a, 0, 8).unwrap();
    write_all_blocks(&mut pool, &a, block_size);

    // Fork: b adopts a's two blocks (refcount bump) and appends its own.
    let mut b = PagedSequenceState::new(pool.layout());
    let shared = a.layer(0).unwrap().block_ids.clone();
    for id in &shared {
        pool.retain_block(*id).unwrap();
    }
    {
        let layer_b = b.layer_mut(0).unwrap();
        layer_b.block_ids = shared.clone();
        layer_b.len = 8;
    }
    pool.append_tokens(&mut b, 0, 2).unwrap();
    write_all_blocks(&mut pool, &b, block_size);

    let view = pool.paged_csr_view(&[&a, &b], 0).unwrap();
    assert_eq!(view.indptr, vec![0, 2, 5]);
    // Both requests resolve the shared prefix to the same physical rows.
    assert_eq!(view.indices[0..2], view.indices[2..4]);
    assert_eq!(view.seq_lens, vec![8, 10]);
    assert_eq!(view.last_page_len, vec![4, 2]);
    for id in &shared {
        assert_eq!(pool.refcount(*id), 2);
    }
}

#[test]
fn pool_view_after_a_trim_drops_retired_pages() {
    let block_size = 4usize;
    let mut pool = fp16_pool(block_size);
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 12).unwrap();
    write_all_blocks(&mut pool, &state, block_size);

    // Retire the first 6 tokens from the front.
    state.layer_mut(0).unwrap().logical_start = 6;

    let view = pool.paged_csr_view(&[&state], 0).unwrap();
    assert_eq!(view.pages_for(0), 2);
    assert_eq!(view.first_page_offset, vec![2]);
    assert_eq!(view.seq_lens, vec![6]);
    assert_eq!(view.rope_offsets, vec![12]);
}

#[test]
fn pool_view_rejects_an_out_of_range_layer() {
    let mut pool = fp16_pool(4);
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let err = pool.paged_csr_view(&[&state], 7).unwrap_err();
    assert!(err.contains("out of range"), "unexpected error: {err}");
}
