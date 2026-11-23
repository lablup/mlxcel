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

//! Block accounting of window-bounded sliding layers (issue #3256). These
//! drive the block table through `append_tokens`, the settle and the trim
//! without pool tensors, so they pin the bookkeeping alone.

use super::super::{PagedBlockPool, PagedKvLayout, PagedLayerState, PagedSequenceState};
use super::REWIND_RESERVE_BLOCKS;

const BLOCK: usize = 16;
const WINDOW: usize = 32;

fn pool() -> PagedBlockPool {
    PagedBlockPool::new(PagedKvLayout::uniform(1, BLOCK, BLOCK).unwrap())
}

fn sliding_state() -> PagedSequenceState {
    PagedSequenceState {
        block_size: BLOCK,
        layers: vec![PagedLayerState::sliding(WINDOW)],
    }
}

/// One append the way `write_prefill` performs it: settle, then grow.
fn append(pool: &mut PagedBlockPool, state: &mut PagedSequenceState, n: usize) {
    pool.settle_window_before_append(&mut state.layers[0])
        .unwrap();
    pool.append_tokens(state, 0, n).unwrap();
}

#[test]
fn decode_keeps_the_live_table_within_the_window_bound() {
    let mut pool = pool();
    let mut state = sliding_state();
    append(&mut pool, &mut state, 100);
    let bound = WINDOW.div_ceil(BLOCK) + 1;
    for _ in 0..200 {
        append(&mut pool, &mut state, 1);
        let layer = &state.layers[0];
        assert_eq!(layer.visible_len(), WINDOW, "a decode step sees the window");
        assert!(
            layer.block_ids.len() <= bound,
            "{} live blocks exceed {bound}",
            layer.block_ids.len()
        );
        assert!(layer.retired.len() <= REWIND_RESERVE_BLOCKS);
        assert_eq!(layer.end_block(), layer.len.div_ceil(BLOCK));
    }
    // Only the live table and the reserve hold blocks.
    let held = state.layers[0].block_ids.len() + state.layers[0].retired.len();
    assert_eq!(pool.live_block_count(), held);
}

#[test]
fn prefill_chunk_sees_window_minus_one_earlier_keys() {
    let mut pool = pool();
    let mut state = sliding_state();
    append(&mut pool, &mut state, 70);
    append(&mut pool, &mut state, 20);
    let layer = &state.layers[0];
    assert_eq!(layer.logical_start, 70 - (WINDOW - 1));
    assert_eq!(layer.visible_len(), WINDOW - 1 + 20);
}

#[test]
fn retention_keeps_every_retired_block_pinned() {
    let mut pool = pool();
    pool.set_retain_retired(true);
    let mut state = sliding_state();
    for _ in 0..10 {
        append(&mut pool, &mut state, 16);
    }
    append(&mut pool, &mut state, 1);
    let layer = &state.layers[0];
    let abs: Vec<usize> = layer.retired.iter().map(|(abs, _)| *abs).collect();
    assert_eq!(abs, (0..layer.block_base).collect::<Vec<_>>());
    for &(_, id) in &layer.retired {
        assert_eq!(pool.refcount(id), 1);
    }
    pool.release_sequence(&mut state).unwrap();
    assert_eq!(pool.live_block_count(), 0);
}

#[test]
fn decode_rewind_brings_the_head_back_from_the_reserve() {
    let mut pool = pool();
    let mut state = sliding_state();
    append(&mut pool, &mut state, 63);
    // Step to 64 and 65; the settle before the second step retires block 1.
    append(&mut pool, &mut state, 1);
    append(&mut pool, &mut state, 1);
    assert_eq!(state.layers[0].block_base, 2);
    // Rewind both steps: the next step at length 63 needs position 32.
    assert_eq!(pool.trim_tokens(&mut state, 0, 2).unwrap(), 2);
    let layer = &state.layers[0];
    assert_eq!(layer.len, 63);
    assert!(layer.logical_start <= 63 - (WINDOW - 1));
    assert!(layer.block_base * BLOCK <= layer.logical_start);
    append(&mut pool, &mut state, 1);
    assert_eq!(state.layers[0].visible_len(), WINDOW);
}

#[test]
fn rewind_past_the_reserve_is_an_error() {
    let mut pool = pool();
    let mut state = sliding_state();
    append(&mut pool, &mut state, 160);
    append(&mut pool, &mut state, 1);
    assert!(pool.trim_tokens(&mut state, 0, 100).is_err());
}

#[test]
fn pad_trim_after_a_prefill_needs_no_retired_block() {
    let mut pool = pool();
    let mut state = sliding_state();
    append(&mut pool, &mut state, 50);
    append(&mut pool, &mut state, 64);
    let before = state.layers[0].clone();
    assert_eq!(pool.trim_tokens(&mut state, 0, 14).unwrap(), 14);
    let layer = &state.layers[0];
    assert_eq!(layer.len, 100);
    assert_eq!(layer.block_base, before.block_base);
    assert_eq!(layer.logical_start, before.logical_start);
}
