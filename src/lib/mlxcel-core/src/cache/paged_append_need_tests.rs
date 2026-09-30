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

//! `PagedBlockPool::blocks_to_append` (issue #1982): the scheduler reserves
//! decode budget from this figure, so it must match what `append_tokens` and
//! the copy-on-write in `write_prefill` actually acquire.

use super::paged::{PagedBlockPool, PagedKvLayout, PagedSequenceState};

fn layout() -> PagedKvLayout {
    // 2 layers, block_size 4.
    PagedKvLayout::uniform(2, 4, 128).unwrap()
}

fn state_with_len(
    pool: &mut PagedBlockPool,
    layout: &PagedKvLayout,
    len: usize,
) -> PagedSequenceState {
    let mut state = PagedSequenceState::new(layout);
    for layer in 0..layout.num_layers {
        pool.append_tokens(&mut state, layer, len).unwrap();
    }
    state
}

#[test]
fn mid_block_append_needs_no_block() {
    let layout = layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let state = state_with_len(&mut pool, &layout, 5);
    assert_eq!(pool.blocks_to_append(&state, 1), 0);
    assert_eq!(pool.blocks_to_append(&state, 3), 0);
    assert_eq!(pool.blocks_to_append(&state, 0), 0);
}

#[test]
fn boundary_append_needs_one_block_per_layer() {
    let layout = layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = state_with_len(&mut pool, &layout, 4);
    assert_eq!(pool.blocks_to_append(&state, 1), 2);
    // Spanning two fresh blocks per layer.
    assert_eq!(pool.blocks_to_append(&state, 5), 4);

    // The estimate is exactly what append_tokens acquires.
    let live_before = pool.live_block_count();
    for layer in 0..layout.num_layers {
        pool.append_tokens(&mut state, layer, 1).unwrap();
    }
    assert_eq!(pool.live_block_count() - live_before, 2);
}

#[test]
fn shared_partial_tail_counts_the_copy_on_write_fork() {
    let layout = layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let state = state_with_len(&mut pool, &layout, 6);
    assert_eq!(pool.blocks_to_append(&state, 1), 0);

    // Another owner (a parked prompt-cache prefix) pins the partial tail on
    // layer 0: the next write there forks it first.
    let tail = state.layers[0].block_ids[1];
    pool.retain_block(tail).unwrap();
    assert_eq!(pool.blocks_to_append(&state, 1), 1);

    // A block-aligned length writes into a fresh block, so a shared full block
    // behind it costs nothing extra.
    let aligned = state_with_len(&mut pool, &layout, 8);
    pool.retain_block(aligned.layers[0].block_ids[1]).unwrap();
    assert_eq!(pool.blocks_to_append(&aligned, 1), 2);
}
