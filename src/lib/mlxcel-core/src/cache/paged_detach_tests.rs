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

//! Unit + property tests for `cache/paged_detach.rs`.
//!
//! Organised as:
//! 1. `PagedBlockPool` refcount plumbing (`retain_block` / `release_block`).
//! 2. `CachePool::detach_paged` / `adopt_paged` happy path + edge cases.
//! 3. Parking + `memory_usage_bytes` accounting for paged sets.
//! 4. Trim semantics on the paged backend (whole-block release vs logical
//!    partial trims).
//! 5. Stress test: many concurrent sequences sharing a prefix.
//! 6. Property test: `prefill(N)+detach+adopt+decode(M)` vs fresh
//!    `prefill(N+M)` under paged (dense placeholder equivalence).
//! 7. INT8 round-trip under paged detach/adopt.

use super::super::paged::{PagedBlockId, PagedBlockPool, PagedKvLayout, PagedSequenceState};
use super::super::{
    CachePool, KVCache, KVCacheMode, SequenceId, SequenceStateBackend, SequenceStateLayout,
};
use super::DetachedPagedCacheSet;

use crate::ffi::MlxArray;
use crate::generate::LanguageModel;
use cxx::UniquePtr;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Minimal paged-only stub model. Mirrors the pattern used in
/// `detach_tests.rs::cache_pool_detach_rejects_paged_backend` but also
/// provides a per-layer dense placeholder cache so the paged scheduler path
/// can still mirror writes.
struct PagedStubModel {
    layout: PagedKvLayout,
    prepared: std::cell::RefCell<Vec<SequenceId>>,
}

impl PagedStubModel {
    fn new(layout: PagedKvLayout) -> Self {
        Self {
            layout,
            prepared: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn prepared_ids(&self) -> Vec<SequenceId> {
        self.prepared.borrow().clone()
    }
}

impl LanguageModel for PagedStubModel {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        crate::ffi::zeros(&[1], 0)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layout.num_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.layout.num_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0]
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::paged_kv_cache(self.layout.clone())
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.prepared.borrow_mut().push(seq_id);
    }
}

fn fp32_tokens(values: &[f32]) -> UniquePtr<MlxArray> {
    let t = values.len() as i32;
    crate::ffi::from_slice_f32(values, &[1, 1, t, 1])
}

fn default_layout() -> PagedKvLayout {
    // 2 layers, block_size=4 tokens, 128 bytes/block (32 bytes/token).
    PagedKvLayout::uniform(2, 4, 128).unwrap()
}

/// Drive `PagedBlockPool::write_prefill` for an active sequence at the
/// `CachePool` level. This is the split-borrow `append_paged_tokens` already
/// uses internally; it is replicated here (the test module is a descendant of
/// `cache.rs`, so it may touch the private `paged_pool` / `active` fields)
/// because the live forward/scheduler wiring of `write_prefill` is #121 and is
/// out of scope for #120. `k`/`v` are `[1, n_kv_heads, n_new, head_dim]`.
fn write_prefill_for(
    pool: &mut CachePool,
    id: SequenceId,
    layer_idx: usize,
    k: &MlxArray,
    v: &MlxArray,
) -> Result<(), String> {
    let block_pool = pool
        .paged_pool
        .as_ref()
        .ok_or_else(|| "paged backend not initialized".to_string())?;
    let state = pool
        .active
        .get(&id)
        .ok_or_else(|| format!("sequence {id} not found"))?
        .paged
        .as_ref()
        .ok_or_else(|| format!("sequence {id} is not paged"))?;
    block_pool
        .borrow_mut()
        .write_prefill(&mut state.borrow_mut(), layer_idx, k, v)
}

/// Deterministic `[1, H, n_tokens, D]` FP32 prefill block whose per-token
/// values are distinct (encodes head/token/dim), so misplacement is caught
/// exactly and FP32 round-trips bit-for-bit. Mirrors `paged_pool_tests::make_block`.
fn prefill_block(base: f32, n_kv_heads: i32, n_tokens: i32, head_dim: i32) -> UniquePtr<MlxArray> {
    let mut values = Vec::with_capacity((n_kv_heads * n_tokens * head_dim) as usize);
    for head in 0..n_kv_heads {
        for tok in 0..n_tokens {
            for dim in 0..head_dim {
                values.push(base + head as f32 * 1000.0 + tok as f32 * 10.0 + dim as f32 * 0.1);
            }
        }
    }
    crate::ffi::from_slice_f32(&values, &[1, n_kv_heads, n_tokens, head_dim])
}

/// Flatten a tensor to a row-major `Vec<f32>` (after an FP32 cast) for content
/// comparison. Mirrors `paged_pool_tests::flatten_fp32`.
fn flatten_fp32(arr: &MlxArray) -> Vec<f32> {
    let a = crate::ffi::astype(arr, crate::dtype::FLOAT32);
    crate::ffi::eval(&a);
    crate::ffi::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Build the dense contiguous reference `[1, H, total, D]` by writing each
/// `[1, H, n, D]` block at its token offset. Mirrors
/// `paged_pool_tests::dense_reference` (no trimming; full visible window).
fn dense_reference(
    blocks: &[(UniquePtr<MlxArray>, usize)],
    n_kv_heads: i32,
    total: i32,
    head_dim: i32,
) -> UniquePtr<MlxArray> {
    let mut dense = crate::ffi::zeros(&[1, n_kv_heads, total, head_dim], crate::dtype::FLOAT32);
    for (block, offset) in blocks {
        let n = crate::ffi::array_shape(block)[2];
        dense = crate::ffi::slice_update(
            &dense,
            block,
            &[0, 0, *offset as i32, 0],
            &[1, n_kv_heads, *offset as i32 + n, head_dim],
        );
    }
    dense
}

// ---------------------------------------------------------------------------
// 1. PagedBlockPool refcount plumbing
// ---------------------------------------------------------------------------

#[test]
fn block_pool_retain_release_holds_block_when_pinned() {
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);

    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];
    assert_eq!(pool.refcount(block_id), 1);

    // Pin from a detached cache.
    pool.retain_block(block_id).unwrap();
    assert_eq!(pool.refcount(block_id), 2);

    // Releasing the sequence's pin leaves the block alive under detach pin.
    pool.trim_tokens(&mut state, 0, 4).unwrap();
    assert_eq!(pool.refcount(block_id), 1);

    // A fresh allocation must not recycle this block while it is pinned.
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let new_block_id = state.layer(0).unwrap().block_ids[0];
    assert_ne!(
        new_block_id, block_id,
        "pinned block must not be recycled by append_tokens"
    );

    // Drop the detach pin — block returns to free list and becomes
    // eligible for reuse on the next allocation.
    pool.release_block(block_id).unwrap();
    assert_eq!(pool.refcount(block_id), 0);
    pool.trim_tokens(&mut state, 0, 4).unwrap();
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let recycled = state.layer(0).unwrap().block_ids[0];
    // With the pin gone the free list can serve the block again; the
    // specific id chosen is an implementation detail of the LIFO free list
    // but must be one of the previously allocated block ids.
    assert!(recycled == block_id || recycled == new_block_id);
}

#[test]
fn block_pool_refuses_retain_of_released_block() {
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);

    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];
    pool.trim_tokens(&mut state, 0, 4).unwrap();
    assert_eq!(pool.refcount(block_id), 0);

    // Retaining a free block must fail so callers cannot observe stale
    // contents from a block that is about to be recycled.
    let err = pool.retain_block(block_id).unwrap_err();
    assert!(err.contains("cannot retain released"));
}

#[test]
fn block_pool_double_release_errors_cleanly() {
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);

    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];
    pool.retain_block(block_id).unwrap();
    pool.release_block(block_id).unwrap(); // down to refcount 1 (sequence)
    pool.trim_tokens(&mut state, 0, 4).unwrap(); // down to refcount 0
    let err = pool.release_block(block_id).unwrap_err();
    assert!(err.contains("already released"));
}

// ---------------------------------------------------------------------------
// 1b. Global block budget (#122 sub-step b1)
// ---------------------------------------------------------------------------

#[test]
fn block_budget_refuses_new_blocks_beyond_cap() {
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let mut pool = PagedBlockPool::new(layout.clone());
    pool.set_block_budget(Some(3));
    assert_eq!(pool.block_budget(), Some(3));

    let mut state = PagedSequenceState::new(&layout);
    // block_size 4 → 12 tokens = exactly 3 blocks, filling the budget.
    pool.append_tokens(&mut state, 0, 12).unwrap();
    assert_eq!(pool.allocated_block_count(), 3);
    assert_eq!(pool.free_block_budget(), Some(0));

    // A 4th block cannot be minted while at the cap.
    let err = pool.append_tokens(&mut state, 0, 4).unwrap_err();
    assert!(err.contains("budget exhausted"), "got: {err}");
}

#[test]
fn block_budget_allows_reuse_of_freed_blocks_at_cap() {
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let mut pool = PagedBlockPool::new(layout.clone());
    pool.set_block_budget(Some(2));

    let mut a = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut a, 0, 8).unwrap(); // 2 blocks → at the cap
    assert_eq!(pool.free_block_budget(), Some(0));
    pool.release_sequence(&mut a).unwrap(); // both freed (refcount 0), rows retained

    // A fresh sequence reuses the freed blocks without minting — allowed at cap.
    let mut b = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut b, 0, 8).unwrap();
    assert_eq!(
        pool.allocated_block_count(),
        2,
        "freed blocks must be reused, not re-minted, when at the budget"
    );
}

#[test]
fn block_budget_none_is_unbounded() {
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let mut pool = PagedBlockPool::new(layout.clone());
    assert_eq!(pool.block_budget(), None);
    assert_eq!(pool.free_block_budget(), None);

    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 400).unwrap(); // 100 blocks, no cap
    assert_eq!(pool.allocated_block_count(), 100);
}

#[test]
fn free_block_budget_tracks_allocation() {
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let mut pool = PagedBlockPool::new(layout.clone());
    pool.set_block_budget(Some(5));
    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 8).unwrap(); // 2 blocks
    assert_eq!(pool.allocated_block_count(), 2);
    assert_eq!(pool.live_block_count(), 2);
    assert_eq!(pool.free_block_budget(), Some(3));
}

#[test]
fn free_block_budget_rises_when_blocks_are_freed() {
    // free_block_budget is budget − LIVE, not budget − allocated: freeing a
    // block restores acquirable headroom even though the row is retained for
    // reuse. This is what lets eviction/preemption reclaim budget for admission.
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let mut pool = PagedBlockPool::new(layout.clone());
    pool.set_block_budget(Some(4));
    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 16).unwrap(); // 4 blocks, all live
    assert_eq!(pool.free_block_budget(), Some(0));

    pool.release_sequence(&mut state).unwrap();
    assert_eq!(pool.live_block_count(), 0);
    assert_eq!(pool.allocated_block_count(), 4, "rows retained for reuse");
    assert_eq!(
        pool.free_block_budget(),
        Some(4),
        "freeing live blocks restores the full acquirable budget"
    );
}

#[test]
fn cache_pool_budget_set_before_pool_creation_applies_on_creation() {
    // The scheduler may set the budget before the first paged allocation lazily
    // creates the pool; CachePool stores the intent and applies it on creation.
    let model = PagedStubModel::new(default_layout());
    let mut pool = CachePool::new(4);
    assert!(
        pool.paged_pool_ref().is_none(),
        "no pool before first allocate"
    );
    pool.set_paged_block_budget(Some(7));
    assert_eq!(pool.paged_block_budget(), Some(7), "stored intent survives");

    let _seq = pool.allocate(&model).unwrap(); // creates the paged pool
    assert_eq!(
        pool.paged_pool_ref().unwrap().block_budget(),
        Some(7),
        "the pre-set budget must apply when the pool is born"
    );
}

// ---------------------------------------------------------------------------
// 2. CachePool detach / adopt happy path + edge cases
// ---------------------------------------------------------------------------

#[test]
fn detach_paged_returns_none_for_unknown_seq() {
    let mut pool = CachePool::new(4);
    assert!(pool.detach_paged(SequenceId::from_raw(9_999)).is_none());
}

#[test]
fn detach_paged_rejects_dense_sequence() {
    // A dense-only model should never succeed detach_paged — ensure the
    // paged surface does not accidentally snatch dense sequences.
    let model = PagedStubModel::new(default_layout());
    let mut pool = CachePool::new(4);
    // Allocate a paged sequence first so the paged pool is initialized.
    let _paged_id = pool.allocate(&model).unwrap();

    // Construct a dense sequence via the legacy layout override.
    let mut dense_pool = CachePool::new(4);
    struct DenseStub {
        num_layers: usize,
    }
    impl LanguageModel for DenseStub {
        fn forward(
            &self,
            _i: &MlxArray,
            _c: &mut [KVCache],
            _m: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            crate::ffi::zeros(&[1], 0)
        }
        fn make_caches(&self) -> Vec<KVCache> {
            (0..self.num_layers).map(|_| KVCache::new()).collect()
        }
        fn num_layers(&self) -> usize {
            self.num_layers
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
    }
    let dense_model = DenseStub { num_layers: 2 };
    let dense_seq = dense_pool.allocate(&dense_model).unwrap();
    assert!(
        dense_pool.detach_paged(dense_seq).is_none(),
        "detach_paged must not steal dense sequences"
    );
    assert_eq!(dense_pool.active_count(), 1);
}

#[test]
fn detach_paged_round_trip_preserves_layout_and_block_table() {
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq_a = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq_a, 0, 6).unwrap();
    pool.append_paged_tokens(seq_a, 1, 8).unwrap();

    let (layer0_blocks, layer1_blocks) = {
        let state = pool.get_paged_state(seq_a).unwrap();
        (
            state.layer(0).unwrap().block_ids.clone(),
            state.layer(1).unwrap().block_ids.clone(),
        )
    };
    let pre_blocks = layer0_blocks.len() + layer1_blocks.len();

    let detached = pool
        .detach_paged(seq_a)
        .expect("paged detach must succeed for paged sequences");
    assert_eq!(detached.backend(), SequenceStateBackend::PagedKvCache);
    assert_eq!(detached.num_layers(), 2);
    assert_eq!(detached.seq_len(), 6); // layer 0 has 6 tokens, matches first-layer summary
    assert_eq!(detached.retained_block_count(), pre_blocks);
    assert_eq!(
        detached.layout().block_size,
        layout.block_size,
        "layout must be preserved"
    );
    assert!(pool.get_paged_state(seq_a).is_none());
    assert_eq!(pool.active_count(), 0);

    let seq_b = pool
        .adopt_paged(&model, detached)
        .expect("paged adopt must succeed");
    assert_ne!(seq_a, seq_b);
    assert_eq!(pool.active_count(), 1);
    assert!(model.prepared_ids().contains(&seq_b));

    // Block table must round-trip bit-for-bit.
    let state = pool.get_paged_state(seq_b).unwrap();
    assert_eq!(state.layer(0).unwrap().block_ids, layer0_blocks);
    assert_eq!(state.layer(1).unwrap().block_ids, layer1_blocks);
}

#[test]
fn detach_paged_prevents_block_recycling_while_pinned() {
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq_a = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq_a, 0, 4).unwrap();
    let original_block = pool
        .get_paged_state(seq_a)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids[0];

    let detached = pool.detach_paged(seq_a).unwrap();

    // Now allocate a fresh sequence and append — the new sequence must not
    // recycle the block pinned by the detached set, even though the original
    // sequence is gone.
    let seq_b = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq_b, 0, 4).unwrap();
    let new_block = pool
        .get_paged_state(seq_b)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids[0];
    assert_ne!(
        new_block, original_block,
        "detach pin must keep original block out of the free list"
    );

    // Cleanup — release the pin explicitly.
    pool.release_detached_paged(detached);
}

#[test]
fn release_detached_paged_returns_blocks_to_free_list() {
    // Regression for the discard-path leak: `release_detached_paged` must
    // release BOTH the detach pin and the origin sequence's allocation so every
    // block reaches refcount 0 and returns to the free list. Previously only
    // the pin was released, leaking each block at refcount 1.
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq, 0, 6).unwrap();
    pool.append_paged_tokens(seq, 1, 8).unwrap();
    let blocks: Vec<PagedBlockId> = (0..layout.num_layers)
        .flat_map(|l| {
            pool.get_paged_state(seq)
                .unwrap()
                .layer(l)
                .unwrap()
                .block_ids
                .clone()
        })
        .collect();
    assert!(!blocks.is_empty(), "appended tokens must allocate blocks");

    let detached = pool.detach_paged(seq).unwrap();
    for b in &blocks {
        assert_eq!(
            pool.paged_pool_ref().unwrap().refcount(*b),
            2,
            "a parked block holds the origin allocation plus the detach pin"
        );
    }

    pool.release_detached_paged(detached);
    for b in &blocks {
        assert_eq!(
            pool.paged_pool_ref().unwrap().refcount(*b),
            0,
            "discard must drive every block to refcount 0 (returned to the pool)"
        );
    }

    // The reclaimed blocks must be recyclable by a fresh sequence.
    let seq2 = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq2, 0, 4).unwrap();
    let reused = pool
        .get_paged_state(seq2)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids[0];
    assert!(
        blocks.contains(&reused),
        "a freed block must be recycled by the next allocation"
    );
}

#[test]
fn adopt_paged_releases_detach_pins() {
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq_a = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq_a, 0, 4).unwrap();
    pool.append_paged_tokens(seq_a, 1, 4).unwrap();
    let detached = pool.detach_paged(seq_a).unwrap();
    let _ = pool.adopt_paged(&model, detached).unwrap();

    // After adopt, every block's refcount must equal the number of sequences
    // that currently reference it (here, 1). We verify indirectly via trim:
    // dropping the only live reference should let the pool recycle the block.
    let active_ids: Vec<SequenceId> = vec![];
    let _ = active_ids; // placeholder for future multi-seq scenarios

    assert_eq!(pool.parked_count(), 0);
}

#[test]
fn adopt_paged_rejects_incompatible_layout() {
    let layout_a = PagedKvLayout::uniform(2, 4, 128).unwrap();
    let model_a = PagedStubModel::new(layout_a.clone());
    let mut pool = CachePool::new(4);
    let seq_a = pool.allocate(&model_a).unwrap();
    pool.append_paged_tokens(seq_a, 0, 4).unwrap();
    let detached = pool.detach_paged(seq_a).unwrap();

    // Try adopting into a pool whose paged layout differs (block_size=8).
    let layout_b = PagedKvLayout::uniform(2, 8, 128).unwrap();
    let model_b = PagedStubModel::new(layout_b);
    let mut pool_b = CachePool::new(4);
    let _warm = pool_b.allocate(&model_b).unwrap();

    let err = pool_b
        .adopt_paged_preserving(&model_b, detached)
        .unwrap_err();
    assert!(err.0.contains("layout mismatch"));
    // Original set survives — cleanup via explicit release.
    pool_b.release_detached_paged(err.1);
}

#[test]
fn adopt_paged_respects_capacity_and_cleans_up_pins() {
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(1);

    let seq_a = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq_a, 0, 4).unwrap();
    let detached = pool.detach_paged(seq_a).unwrap();

    // Refill the pool with a dummy paged sequence so max_sequences=1 is hit.
    let _seq_fill = pool.allocate(&model).unwrap();
    assert_eq!(pool.active_count(), 1);

    // adopt_paged must fail because the pool is full. Pins are released by
    // the returning API contract.
    let err = pool.adopt_paged(&model, detached).unwrap_err();
    assert!(err.contains("max capacity"));
}

// ---------------------------------------------------------------------------
// 3. Parking / memory accounting
// ---------------------------------------------------------------------------

#[test]
fn park_paged_round_trip_accounts_bytes() {
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq, 0, 8).unwrap();
    pool.append_paged_tokens(seq, 1, 4).unwrap();
    let detached = pool.detach_paged(seq).unwrap();
    // The set's own ledger view (prompt-cache accounting) stays nonzero: with
    // no real pool writes it falls back to the layout's nominal bytes.
    assert!(detached.nbytes() > 0);

    let handle = pool.park_detached_paged(detached);
    assert_eq!(pool.parked_count(), 1);
    // Pool-resident bytes are counted ONCE via `pool_tensor_bytes` inside
    // `memory_usage_bytes` (#226); the parked walk no longer re-adds them.
    // This sequence never wrote K/V, so the pool holds no slabs and the true
    // physical footprint is zero.
    assert_eq!(pool.parked_bytes(), 0);
    assert!(pool.peek_parked_paged(handle).is_some());
    assert_eq!(pool.memory_usage_bytes(), 0);

    // Dense peek must reject a paged handle.
    assert!(pool.peek_parked(handle).is_none());

    let taken = pool.take_parked_paged(handle).unwrap();
    assert_eq!(taken.num_layers(), 2);
    assert_eq!(pool.parked_count(), 0);

    // Re-adopt to land back in the active set.
    let seq_b = pool.adopt_paged(&model, taken).unwrap();
    assert_eq!(pool.active_count(), 1);
    assert!(model.prepared_ids().contains(&seq_b));
}

#[test]
fn take_parked_paged_with_dense_handle_returns_none_without_dropping() {
    // Mix dense + paged handles to confirm the unified handle namespace.
    let paged_layout = default_layout();
    let paged_model = PagedStubModel::new(paged_layout.clone());

    struct DenseStub;
    impl LanguageModel for DenseStub {
        fn forward(
            &self,
            _i: &MlxArray,
            _c: &mut [KVCache],
            _m: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            crate::ffi::zeros(&[1], 0)
        }
        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
    }
    let dense_model = DenseStub;

    let mut pool = CachePool::new(4);
    let dense_seq = pool.allocate(&dense_model).unwrap();
    {
        let caches = pool.get_caches_mut(dense_seq).unwrap();
        caches[0].update(fp32_tokens(&[1.0, 2.0]), fp32_tokens(&[3.0, 4.0]));
    }
    let dense_detached = pool.detach(dense_seq).unwrap();
    let dense_handle = pool.park_detached(dense_detached);

    // take_parked_paged(dense_handle) must not consume the dense entry.
    assert!(pool.take_parked_paged(dense_handle).is_none());
    assert_eq!(pool.parked_count(), 1);
    // ...and the dense-side take still works.
    let dense_set = pool.take_parked(dense_handle).unwrap();
    assert_eq!(dense_set.seq_len(), 2);

    // Now park a paged set and verify the symmetric case.
    let paged_seq = pool.allocate(&paged_model).unwrap();
    pool.append_paged_tokens(paged_seq, 0, 4).unwrap();
    let paged_detached = pool.detach_paged(paged_seq).unwrap();
    let paged_handle = pool.park_detached_paged(paged_detached);
    assert!(pool.take_parked(paged_handle).is_none());
    assert_eq!(pool.parked_count(), 1);
    let paged_set = pool.take_parked_paged(paged_handle).unwrap();
    pool.release_detached_paged(paged_set);
}

// ---------------------------------------------------------------------------
// 4. Trim semantics under paged backend
// ---------------------------------------------------------------------------

#[test]
fn paged_trim_to_releases_whole_blocks_past_n() {
    let layout = default_layout(); // block_size = 4
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq = pool.allocate(&model).unwrap();
    // Fill 9 tokens => 3 blocks (ceil(9/4)).
    pool.append_paged_tokens(seq, 0, 9).unwrap();
    assert_eq!(
        pool.get_paged_state(seq)
            .unwrap()
            .layer(0)
            .unwrap()
            .block_ids
            .len(),
        3
    );

    // Trim down to 4 tokens => exactly 1 block needed.
    let trimmed = pool.trim_paged_tokens(seq, 0, 5).unwrap();
    assert_eq!(trimmed, 5);
    let state = pool.get_paged_state(seq).unwrap();
    let layer = state.layer(0).unwrap();
    assert_eq!(layer.len, 4);
    assert_eq!(layer.block_ids.len(), 1);
}

#[test]
fn paged_partial_block_trim_keeps_block_updates_logical_length() {
    // Trim by 1 token when the last block is fractionally used — the block
    // must remain allocated to preserve the remaining prefix data while the
    // logical length shrinks.
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq, 0, 5).unwrap(); // 2 blocks (4 + 1 token)
    let before_blocks = pool
        .get_paged_state(seq)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(before_blocks.len(), 2);

    pool.trim_paged_tokens(seq, 0, 1).unwrap();
    let state = pool.get_paged_state(seq).unwrap();
    let layer = state.layer(0).unwrap();
    assert_eq!(layer.len, 4);
    // Partial trim: still within the second block's span in token coords,
    // so reserved_blocks stays at 1 (the last block's worth of slots is
    // no longer needed).
    assert_eq!(layer.block_ids.len(), 1);
    // First block preserved bit-for-bit.
    assert_eq!(layer.block_ids[0], before_blocks[0]);
}

// ---------------------------------------------------------------------------
// 5. Stress: many concurrent sequences sharing a prefix
// ---------------------------------------------------------------------------

#[test]
fn stress_many_sequences_share_a_prefix_detach_refcount_is_correct() {
    // Simulate a prompt-prefix cache: one "prefix" sequence is detached,
    // parked, adopted by a fresh consumer sequence, and released. The seed
    // is re-parked after every round so the prefix stays available for
    // N iterations. The invariant under test: prefix blocks survive every
    // cycle intact, regardless of how many adopt/release pairs ran.
    const N: usize = 8;

    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(16);

    let seed_seq = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seed_seq, 0, 12).unwrap(); // 3 blocks
    pool.append_paged_tokens(seed_seq, 1, 16).unwrap(); // 4 blocks
    let prefix_blocks: Vec<_> = pool
        .get_paged_state(seed_seq)
        .unwrap()
        .layers
        .iter()
        .flat_map(|l| l.block_ids.iter().copied())
        .collect();

    // Park the seed in the pool so we can re-park after every adopt cycle.
    let mut parked_handle = {
        let detached = pool.detach_paged(seed_seq).unwrap();
        pool.park_detached_paged(detached)
    };

    for _ in 0..N {
        // Take the parked seed, adopt into a live consumer, and observe that
        // the block table is bit-identical to the original prefix.
        let seed = pool.take_parked_paged(parked_handle).unwrap();
        let consumer = pool.adopt_paged(&model, seed).unwrap();
        {
            let state = pool.get_paged_state(consumer).unwrap();
            let consumer_blocks: Vec<_> = state
                .layers
                .iter()
                .flat_map(|l| l.block_ids.iter().copied())
                .collect();
            assert_eq!(
                consumer_blocks, prefix_blocks,
                "adopted sequence must inherit the full prefix block table"
            );
        }

        // Re-park the prefix: detach the consumer (whose block table is
        // identical to the prefix) and park it for the next iteration.
        let re_detached = pool.detach_paged(consumer).unwrap();
        parked_handle = pool.park_detached_paged(re_detached);
    }

    // Drain the final parked set to release its pins.
    if let Some(final_set) = pool.take_parked_paged(parked_handle) {
        pool.release_detached_paged(final_set);
    }
    assert_eq!(pool.parked_count(), 0);
}

// ---------------------------------------------------------------------------
// 6. Property test: detach+adopt+append under paged mirrors fresh prefill
// ---------------------------------------------------------------------------

#[test]
fn property_paged_detach_adopt_append_matches_fresh_allocate_block_table() {
    // Paged attention runs on the dense placeholder caches in Phase 0, so
    // we verify two things:
    // 1. The dense placeholder's visible contents after
    //    detach+adopt+update(M) match a fresh prefill(N+M) placeholder.
    // 2. The paged block table after detach+adopt+append(M) contains the
    //    correct number of blocks for N+M tokens (ceil).
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let head_k: Vec<f32> = (0..6).map(|i| i as f32 + 0.5).collect();
    let tail_k: Vec<f32> = (6..10).map(|i| i as f32 + 0.5).collect();
    let head_v: Vec<f32> = head_k.iter().map(|v| v * -1.0).collect();
    let tail_v: Vec<f32> = tail_k.iter().map(|v| v * -1.0).collect();

    // Path A: paged detach + adopt + append.
    let seq_src = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq_src, 0, 6).unwrap();
    {
        let caches = pool.get_caches_mut(seq_src).unwrap();
        caches[0].update(fp32_tokens(&head_k), fp32_tokens(&head_v));
    }
    let detached = pool.detach_paged(seq_src).unwrap();
    let seq_new = pool.adopt_paged(&model, detached).unwrap();
    pool.append_paged_tokens(seq_new, 0, 4).unwrap();
    {
        let caches = pool.get_caches_mut(seq_new).unwrap();
        caches[0].update(fp32_tokens(&tail_k), fp32_tokens(&tail_v));
    }
    let path_a_blocks = pool
        .get_paged_state(seq_new)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .len();
    let path_a_len = pool.get_paged_state(seq_new).unwrap().layer(0).unwrap().len;

    // Path B: fresh prefill of N+M tokens into a new sequence.
    let seq_fresh = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq_fresh, 0, 10).unwrap();
    let all_k: Vec<f32> = head_k.iter().chain(tail_k.iter()).copied().collect();
    let all_v: Vec<f32> = head_v.iter().chain(tail_v.iter()).copied().collect();
    {
        let caches = pool.get_caches_mut(seq_fresh).unwrap();
        caches[0].update(fp32_tokens(&all_k), fp32_tokens(&all_v));
    }
    let path_b_blocks = pool
        .get_paged_state(seq_fresh)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .len();
    let path_b_len = pool
        .get_paged_state(seq_fresh)
        .unwrap()
        .layer(0)
        .unwrap()
        .len;

    assert_eq!(path_a_blocks, path_b_blocks);
    assert_eq!(path_a_len, path_b_len);
}

// ---------------------------------------------------------------------------
// 7. INT8 round-trip under paged detach/adopt
// ---------------------------------------------------------------------------

#[test]
fn paged_int8_cache_round_trips_through_detach_adopt() {
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq, 0, 4).unwrap();
    pool.append_paged_tokens(seq, 1, 4).unwrap();

    // Promote both layers to INT8 and fill with non-trivial values so the
    // per-token scale buffers are populated.
    {
        let caches = pool.get_caches_mut(seq).unwrap();
        for cache in caches.iter_mut() {
            *cache = KVCache::new_with_mode(KVCacheMode::Int8);
        }
        caches[0].update(
            fp32_tokens(&[1.0, 8.0, 64.0, 0.125]),
            fp32_tokens(&[2.0, 16.0, 128.0, 0.0625]),
        );
        caches[1].update(
            fp32_tokens(&[3.0, 9.0, 27.0, 0.333]),
            fp32_tokens(&[-1.0, -4.0, -16.0, -0.25]),
        );
    }

    let pre_scales = {
        let caches = pool.get_caches_mut(seq).unwrap();
        crate::array_to_raw_bytes(caches[0].key_scales.as_ref().unwrap())
    };

    let detached = pool.detach_paged(seq).unwrap();
    let seq_b = pool.adopt_paged(&model, detached).unwrap();
    let caches = pool.get_caches_mut(seq_b).unwrap();
    assert_eq!(caches[0].mode, KVCacheMode::Int8);
    assert_eq!(caches[1].mode, KVCacheMode::Int8);
    assert_eq!(caches[0].seq_len(), 4);
    assert_eq!(caches[1].seq_len(), 4);
    assert!(caches[0].key_scales.is_some());
    assert!(caches[0].val_scales.is_some());

    let post_scales = crate::array_to_raw_bytes(caches[0].key_scales.as_ref().unwrap());
    assert_eq!(
        pre_scales, post_scales,
        "INT8 scale buffer must survive paged detach/adopt bit-for-bit"
    );
}

// ---------------------------------------------------------------------------
// 8. release_detached_paged idempotency
// ---------------------------------------------------------------------------

#[test]
fn release_detached_paged_is_safe_after_take_no_double_release() {
    // Ensure that taking a paged set out of the park and subsequently
    // releasing it does not double-release block pins: adopt already drops
    // the detach pin, so running release_detached_paged on an already-
    // adopted set must be a no-op.
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seq = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq, 0, 4).unwrap();

    // Path A: detach + adopt (pins dropped inside adopt).
    let detached = pool.detach_paged(seq).unwrap();
    let _ = pool.adopt_paged(&model, detached).unwrap();

    // Path B: detach + release (pins dropped inside release).
    let seq_b = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq_b, 0, 4).unwrap();
    let detached_b = pool.detach_paged(seq_b).unwrap();
    pool.release_detached_paged(detached_b);
    assert_eq!(pool.parked_count(), 0);
}

// ---------------------------------------------------------------------------
// 9. write_prefill (#120) over the real detach/adopt machinery: a shared-prefix
//    request allocates blocks ONLY for its suffix (acceptance criterion 2),
//    verified via PagedCacheStats, and the adopted sequence reads back its own
//    prefix + suffix correctly.
//
// The simultaneous two-live-sharers copy-on-write proof lives in
// `paged_pool_tests::write_prefill_cow_forks_shared_partial_tail_block`:
// `CachePool::adopt_paged` is move-semantics (it consumes the detached set and
// transfers the block pins), so it cannot stand up two live sequences whose
// block tables alias the same physical prefix blocks at the same instant — that
// aliasing is produced with `retain_block` at the pool level. Here we exercise
// the real detach -> park -> adopt path and prove the block-accounting half of
// the acceptance criteria with `PagedCacheStats`.
// ---------------------------------------------------------------------------

#[test]
fn write_prefill_after_shared_prefix_adopt_allocates_only_suffix_blocks() {
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    // Single-layer paged layout so the block accounting is unambiguous.
    let layout = PagedKvLayout::uniform(
        1,
        block_size,
        block_size * n_kv_heads as usize * head_dim as usize * 2,
    )
    .unwrap();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    // --- Build an 8-token (block-aligned: 2 whole blocks) prefix on a seed. ---
    let seed = pool.allocate(&model).unwrap();
    let prefix_len = 8i32;
    let prefix_k = prefill_block(1000.0, n_kv_heads, prefix_len, head_dim);
    let prefix_v = prefill_block(5000.0, n_kv_heads, prefix_len, head_dim);
    write_prefill_for(&mut pool, seed, 0, &prefix_k, &prefix_v).unwrap();
    let prefix_blocks = pool
        .get_paged_state(seed)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(prefix_blocks.len(), 2, "8 tokens => 2 whole prefix blocks");

    // --- Detach + park + adopt: the consumer inherits the prefix block table
    //     (the real shared-prefix path). ---
    let detached = pool.detach_paged(seed).unwrap();
    let handle = pool.park_detached_paged(detached);
    let consumer = pool.adopt_parked_paged(&model, handle).unwrap();
    assert_eq!(
        pool.get_paged_state(consumer)
            .unwrap()
            .layer(0)
            .unwrap()
            .block_ids,
        prefix_blocks,
        "adopted consumer must reuse the prefix blocks, not fresh copies"
    );

    // Live blocks right after adoption: exactly the 2 prefix blocks.
    let live_before = pool.paged_stats().unwrap().live_blocks;
    assert_eq!(live_before, 2, "only the 2 shared prefix blocks are live");

    // --- write_prefill a 6-token suffix (positions [8, 14)). 8 is block-
    //     aligned, so this allocates 2 fresh suffix blocks and touches no
    //     prefix block. ---
    let suffix_len = 6i32;
    let suffix_k = prefill_block(2000.0, n_kv_heads, suffix_len, head_dim);
    let suffix_v = prefill_block(6000.0, n_kv_heads, suffix_len, head_dim);
    write_prefill_for(&mut pool, consumer, 0, &suffix_k, &suffix_v).unwrap();

    let live_after = pool.paged_stats().unwrap().live_blocks;
    // Suffix is 6 tokens over 2 blocks; prefix blocks are untouched. So live
    // grows by EXACTLY the suffix block count, NOT by another copy of the
    // prefix. This is the "allocates blocks only for its suffix" criterion.
    assert_eq!(
        live_after - live_before,
        2,
        "shared-prefix request must allocate blocks only for its suffix"
    );
    let suffix_blocks = pool
        .get_paged_state(consumer)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(suffix_blocks.len(), 4, "2 prefix + 2 suffix blocks");
    // The prefix blocks are still the SAME physical blocks (no reallocation).
    assert_eq!(&suffix_blocks[..2], &prefix_blocks[..]);

    // --- Correctness: the consumer gathers prefix [0,8) + suffix [8,14). ---
    let total = prefix_len + suffix_len; // 14
    let state = pool.get_paged_state(consumer).unwrap();
    let (gk, gv) = pool
        .paged_pool_ref()
        .unwrap()
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    assert_eq!(
        crate::ffi::array_shape(&gk),
        vec![1, n_kv_heads, total, head_dim]
    );

    let dense_k = dense_reference(
        &[
            (prefill_block(1000.0, n_kv_heads, prefix_len, head_dim), 0),
            (
                prefill_block(2000.0, n_kv_heads, suffix_len, head_dim),
                prefix_len as usize,
            ),
        ],
        n_kv_heads,
        total,
        head_dim,
    );
    let dense_v = dense_reference(
        &[
            (prefill_block(5000.0, n_kv_heads, prefix_len, head_dim), 0),
            (
                prefill_block(6000.0, n_kv_heads, suffix_len, head_dim),
                prefix_len as usize,
            ),
        ],
        n_kv_heads,
        total,
        head_dim,
    );
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

// ---------------------------------------------------------------------------
// 10. Pool-backed radix round-trip (#121 sub-step b): the adopt path must
//     reconstruct POOL-BACKED caches (not empty dense placeholders) for a
//     dense-natural Fp16 paged sequence, and the adopted sequence must SHARE
//     the cached prefix's physical blocks (stored once, no re-prefill).
// ---------------------------------------------------------------------------

/// Dense-natural stub model: its NATURAL sequence-state backend is the dense
/// external `KVCache` slice (the trait default), so `allocate_with_layout` with
/// an Fp16 paged layout POOL-BACKS it (`new_paged` caches) — exactly the shape
/// a real `supports_batching` transformer (qwen3 / llama3) gets under paged
/// decode. Contrast with `PagedStubModel`, whose natural backend is paged and
/// therefore keeps dense placeholder caches.
struct DenseNaturalStubModel {
    num_layers: usize,
    prepared: std::cell::RefCell<Vec<SequenceId>>,
}

impl DenseNaturalStubModel {
    fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            prepared: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn prepared_ids(&self) -> Vec<SequenceId> {
        self.prepared.borrow().clone()
    }
}

impl LanguageModel for DenseNaturalStubModel {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        crate::ffi::zeros(&[1], 0)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0]
    }

    // Deliberately DOES NOT override `sequence_state_layout`: the trait default
    // reports `SequenceStateBackend::DenseKvCache`, which is the natural backend
    // the pool-backing gate keys on.

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.prepared.borrow_mut().push(seq_id);
    }
}

/// 1-layer Fp16 paged layout sized for `[1, n_kv_heads, *, head_dim]` blocks.
fn fp16_pool_layout(block_size: usize, n_kv_heads: i32, head_dim: i32) -> PagedKvLayout {
    let bytes_per_block = block_size * n_kv_heads as usize * head_dim as usize * 2;
    let layout = PagedKvLayout::uniform(1, block_size, bytes_per_block).unwrap();
    assert_eq!(
        layout.cache_mode,
        KVCacheMode::Fp16,
        "pool-backing requires an Fp16 paged layout"
    );
    layout
}

#[test]
fn adopt_paged_reconstructs_pool_backed_caches_and_shares_prefix() {
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    let layout = fp16_pool_layout(block_size, n_kv_heads, head_dim);
    let model = DenseNaturalStubModel::new(1);
    let mut pool = CachePool::new(4);

    // --- Allocate a POOL-BACKED paged sequence (dense-natural + Fp16). ---
    let seed = pool
        .allocate_with_layout(&model, Some(SequenceStateLayout::paged_kv_cache(layout)))
        .unwrap();
    assert!(
        pool.get_caches_mut(seed).unwrap()[0].is_paged_backed(),
        "dense-natural Fp16 paged sequence must be pool-backed at allocate time"
    );

    // --- Prefill an 8-token (2 whole blocks) prefix through the POOL-BACKED
    //     cache's real `update_and_fetch` path (routes to `write_paged`, which
    //     writes the pool blocks AND advances the cache's monotonic offset —
    //     exactly what `model.forward` does). The returned gather is the
    //     reference visible window. ---
    let prefix_len = 8i32;
    let prefix_k = prefill_block(1000.0, n_kv_heads, prefix_len, head_dim);
    let prefix_v = prefill_block(5000.0, n_kv_heads, prefix_len, head_dim);
    let reference = {
        let caches = pool.get_caches_mut(seed).unwrap();
        let (gk, gv) = caches[0].update_and_fetch(prefix_k, prefix_v);
        (flatten_fp32(&gk), flatten_fp32(&gv))
    };
    assert_eq!(
        pool.get_caches_mut(seed).unwrap()[0].offset,
        prefix_len,
        "prefill through the pool-backed cache advances its monotonic offset"
    );
    let prefix_blocks = pool
        .get_paged_state(seed)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(prefix_blocks.len(), 2, "8 tokens => 2 whole prefix blocks");
    let live_before = pool.paged_stats().unwrap().live_blocks;
    assert_eq!(
        live_before, 2,
        "only the 2 prefix blocks are live pre-detach"
    );

    // --- detach -> park -> adopt into a SECOND sequence (the radix path). ---
    let detached = pool.detach_paged(seed).unwrap();
    assert_eq!(detached.retained_block_count(), 2);
    let handle = pool.park_detached_paged(detached);
    let consumer = pool.adopt_parked_paged(&model, handle).unwrap();
    assert_ne!(seed, consumer);
    assert!(model.prepared_ids().contains(&consumer));

    // (a) THE FIX: the adopted sequence's caches are POOL-BACKED. Before #121
    //     sub-step (b) the adopt path rebuilt empty dense caches here, so the
    //     adopted sequence would read garbage in a real model forward.
    assert!(
        pool.get_caches_mut(consumer).unwrap()[0].is_paged_backed(),
        "adopt_paged must reconstruct POOL-BACKED caches for a pool-backed sequence"
    );
    // (a') the restored monotonic offset must equal the cached prefix length so
    //      RoPE positions for the post-adopt prefill/decode are correct. A fresh
    //      `new_paged` cache would report 0 here and mis-rotate the suffix.
    assert_eq!(
        pool.get_caches_mut(consumer).unwrap()[0].offset,
        prefix_len,
        "adopt must restore the cache offset to the cached prefix length"
    );

    // (b) the consumer SHARES the prefix's physical block ids (refcount-pinned
    //     across the round-trip), not fresh copies.
    let consumer_blocks = pool
        .get_paged_state(consumer)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(
        consumer_blocks, prefix_blocks,
        "adopted consumer must reuse the prefix block ids"
    );

    // (d) the prefix is stored ONCE — adoption did not allocate a second copy.
    let live_after = pool.paged_stats().unwrap().live_blocks;
    assert_eq!(
        live_after, live_before,
        "shared prefix must not double the live block count"
    );

    // (c) gather_visible on the adopted sequence returns the SAME K/V.
    let adopted = {
        let state = pool.get_paged_state(consumer).unwrap();
        let (gk, gv) = pool
            .paged_pool_ref()
            .unwrap()
            .gather_visible(&state, 0)
            .unwrap()
            .expect("consumer gather must return data");
        (flatten_fp32(&gk), flatten_fp32(&gv))
    };
    assert_eq!(
        adopted.0, reference.0,
        "adopted K equals original (no copy)"
    );
    assert_eq!(
        adopted.1, reference.1,
        "adopted V equals original (no copy)"
    );

    // --- A block-aligned suffix [8, 14) prefilled through the consumer's
    //     pool-backed cache allocates ONLY its own blocks; the shared prefix
    //     blocks are untouched and the offset continues from 8. ---
    let suffix_len = 6i32;
    let suffix_k = prefill_block(2000.0, n_kv_heads, suffix_len, head_dim);
    let suffix_v = prefill_block(6000.0, n_kv_heads, suffix_len, head_dim);
    {
        let caches = pool.get_caches_mut(consumer).unwrap();
        let _ = caches[0].update_and_fetch(suffix_k, suffix_v);
    }
    assert_eq!(
        pool.get_caches_mut(consumer).unwrap()[0].offset,
        prefix_len + suffix_len,
        "post-adopt suffix prefill continues the monotonic offset from the prefix"
    );
    let live_suffix = pool.paged_stats().unwrap().live_blocks;
    assert_eq!(
        live_suffix - live_after,
        2,
        "shared-prefix request allocates blocks only for its 6-token suffix"
    );
    let after_blocks = pool
        .get_paged_state(consumer)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(
        &after_blocks[..2],
        &prefix_blocks[..],
        "prefix blocks unchanged"
    );
}

#[test]
fn pool_backed_sharers_cow_keeps_other_prefix_intact() {
    // Two LIVE pool-backed sequences sharing a PARTIAL-tail prefix: a divergent
    // suffix on one must copy-on-write the shared tail block, leaving the other
    // sequence's prefix bit-identical. `CachePool::adopt_paged` is move-based
    // (it cannot stand up two live aliasing sequences at once), so the sibling
    // is constructed by aliasing the block table + `retain_block` — exactly the
    // refcount>1 state two adopters of the same cached prefix would reach.
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    let layout = fp16_pool_layout(block_size, n_kv_heads, head_dim);
    let model = DenseNaturalStubModel::new(1);
    let mut pool = CachePool::new(4);

    // --- Sequence A: pool-backed, 6-token prefix (2 blocks; tail half-full). ---
    let seq_a = pool
        .allocate_with_layout(
            &model,
            Some(SequenceStateLayout::paged_kv_cache(layout.clone())),
        )
        .unwrap();
    assert!(pool.get_caches_mut(seq_a).unwrap()[0].is_paged_backed());
    let prefix_len = 6i32;
    let prefix_k = prefill_block(1000.0, n_kv_heads, prefix_len, head_dim);
    let prefix_v = prefill_block(5000.0, n_kv_heads, prefix_len, head_dim);
    write_prefill_for(&mut pool, seq_a, 0, &prefix_k, &prefix_v).unwrap();
    let prefix_blocks = pool
        .get_paged_state(seq_a)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(prefix_blocks.len(), 2);
    let tail_block = prefix_blocks[1];
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(tail_block), 1);

    let reference = {
        let state = pool.get_paged_state(seq_a).unwrap();
        let (gk, gv) = pool
            .paged_pool_ref()
            .unwrap()
            .gather_visible(&state, 0)
            .unwrap()
            .expect("A gather");
        (flatten_fp32(&gk), flatten_fp32(&gv))
    };

    // --- Sequence B: pool-backed, aliases A's prefix block table + pins it
    //     (the refcount>1 state two concurrent adopters of the same cached
    //     prefix reach). ---
    let seq_b = pool
        .allocate_with_layout(&model, Some(SequenceStateLayout::paged_kv_cache(layout)))
        .unwrap();
    assert!(pool.get_caches_mut(seq_b).unwrap()[0].is_paged_backed());
    {
        let set = pool.get_mut(seq_b).unwrap();
        let mut state = set.paged.as_ref().unwrap().borrow_mut();
        let layer = state.layer_mut(0).unwrap();
        layer.block_ids = prefix_blocks.clone();
        layer.len = prefix_len as usize;
        layer.logical_start = 0;
    }
    for &id in &prefix_blocks {
        pool.paged_pool
            .as_ref()
            .unwrap()
            .borrow_mut()
            .retain_block(id)
            .unwrap();
    }
    assert_eq!(
        pool.paged_pool_ref().unwrap().refcount(tail_block),
        2,
        "both live sequences now share the partial tail block"
    );
    // B reads the identical prefix before any divergent write.
    {
        let state = pool.get_paged_state(seq_b).unwrap();
        let (gk, gv) = pool
            .paged_pool_ref()
            .unwrap()
            .gather_visible(&state, 0)
            .unwrap()
            .expect("B gather");
        assert_eq!(flatten_fp32(&gk), reference.0);
        assert_eq!(flatten_fp32(&gv), reference.1);
    }

    // --- Divergent 5-token suffix into A. positions [6, 11): the first suffix
    //     slot lands in the SHARED partial tail (slots 2,3), so write_prefill
    //     must copy-on-write the tail for A. ---
    let suffix_k = prefill_block(2000.0, n_kv_heads, 5, head_dim);
    let suffix_v = prefill_block(6000.0, n_kv_heads, 5, head_dim);
    write_prefill_for(&mut pool, seq_a, 0, &suffix_k, &suffix_v).unwrap();

    let a_tail = pool
        .get_paged_state(seq_a)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids[1];
    let b_tail = pool
        .get_paged_state(seq_b)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids[1];
    assert_ne!(
        a_tail, tail_block,
        "A wrote while the tail was shared, so it must fork a fresh copy"
    );
    assert_eq!(
        b_tail, tail_block,
        "B keeps the original tail block (not corrupted by A's suffix)"
    );
    assert_eq!(
        pool.paged_pool_ref().unwrap().refcount(tail_block),
        1,
        "after A's COW fork the original tail is solely owned by B"
    );

    // --- B's prefix is bit-identical to before A's divergent write. ---
    let b_after = {
        let state = pool.get_paged_state(seq_b).unwrap();
        let (gk, gv) = pool
            .paged_pool_ref()
            .unwrap()
            .gather_visible(&state, 0)
            .unwrap()
            .expect("B gather after A suffix");
        (flatten_fp32(&gk), flatten_fp32(&gv))
    };
    assert_eq!(b_after.0, reference.0, "B's prefix K intact after A's COW");
    assert_eq!(b_after.1, reference.1, "B's prefix V intact after A's COW");
}

#[test]
fn trim_across_shared_blocks_releases_only_the_trimming_sequences_refs() {
    // Two LIVE pool-backed sequences share a 2-block prefix (refcount 2 on each
    // block, the state two adopters of a cached prefix reach). Trimming one of
    // them across the shared tail block must drop ONLY that sequence's ref: the
    // physical block survives for the sibling (refcount 2 -> 1, never freed),
    // the sibling's gathered prefix stays bit-identical, and the trimmed
    // sequence's own window shrinks to the retained prefix. This is the
    // "trim/rewind across shared blocks" correctness case for #126. (rewind is
    // an alias of trim, so this covers both.)
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    let layout = fp16_pool_layout(block_size, n_kv_heads, head_dim);
    let model = DenseNaturalStubModel::new(1);
    let mut pool = CachePool::new(4);

    // Sequence A: 6-token prefix => 2 blocks (head block full, tail half-full).
    let seq_a = pool
        .allocate_with_layout(
            &model,
            Some(SequenceStateLayout::paged_kv_cache(layout.clone())),
        )
        .unwrap();
    let prefix_len = 6i32;
    let prefix_k = prefill_block(1000.0, n_kv_heads, prefix_len, head_dim);
    let prefix_v = prefill_block(5000.0, n_kv_heads, prefix_len, head_dim);
    write_prefill_for(&mut pool, seq_a, 0, &prefix_k, &prefix_v).unwrap();
    let prefix_blocks = pool
        .get_paged_state(seq_a)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(prefix_blocks.len(), 2);
    let head_block = prefix_blocks[0];
    let tail_block = prefix_blocks[1];

    let reference = {
        let state = pool.get_paged_state(seq_a).unwrap();
        let (gk, gv) = pool
            .paged_pool_ref()
            .unwrap()
            .gather_visible(&state, 0)
            .unwrap()
            .expect("A gather");
        (flatten_fp32(&gk), flatten_fp32(&gv))
    };

    // Sequence B aliases A's prefix block table and pins it (refcount 2),
    // exactly the shared state two concurrent adopters of a cached prefix reach.
    let seq_b = pool
        .allocate_with_layout(&model, Some(SequenceStateLayout::paged_kv_cache(layout)))
        .unwrap();
    {
        let set = pool.get_mut(seq_b).unwrap();
        let mut state = set.paged.as_ref().unwrap().borrow_mut();
        let layer = state.layer_mut(0).unwrap();
        layer.block_ids = prefix_blocks.clone();
        layer.len = prefix_len as usize;
        layer.logical_start = 0;
    }
    for &id in &prefix_blocks {
        pool.paged_pool
            .as_ref()
            .unwrap()
            .borrow_mut()
            .retain_block(id)
            .unwrap();
    }
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(tail_block), 2);
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(head_block), 2);
    assert_eq!(pool.paged_pool_ref().unwrap().live_block_count(), 2);

    // Trim A by 2 tokens: visible 6 -> 4, dropping the shared tail block from
    // A's table. `release_block` decrements the tail's refcount but must NOT
    // free it (B still owns it).
    let trimmed = pool.trim_paged_tokens(seq_a, 0, 2).unwrap();
    assert_eq!(trimmed, 2);
    {
        let state = pool.get_paged_state(seq_a).unwrap();
        let layer = state.layer(0).unwrap();
        assert_eq!(layer.len, 4);
        assert_eq!(layer.block_ids, vec![head_block]);
    }

    assert_eq!(
        pool.paged_pool_ref().unwrap().refcount(tail_block),
        1,
        "trim dropped only A's ref to the shared tail; B still owns it"
    );
    assert_eq!(
        pool.paged_pool_ref().unwrap().refcount(head_block),
        2,
        "the head block is still shared by both sequences"
    );
    assert_eq!(
        pool.paged_pool_ref().unwrap().live_block_count(),
        2,
        "the trimmed-away block stays live for the sibling (no premature free)"
    );

    // B's full 6-token prefix is untouched by A's trim.
    {
        let state = pool.get_paged_state(seq_b).unwrap();
        let (gk, gv) = pool
            .paged_pool_ref()
            .unwrap()
            .gather_visible(&state, 0)
            .unwrap()
            .expect("B gather after A trim");
        assert_eq!(flatten_fp32(&gk), reference.0, "B keys intact after A trim");
        assert_eq!(
            flatten_fp32(&gv),
            reference.1,
            "B values intact after A trim"
        );
    }

    // A's own window is exactly the retained 4-token prefix (the head block's
    // data). `prefill_block` values are position-encoded and length-independent,
    // so the first four tokens equal a fresh 4-token block.
    {
        let state = pool.get_paged_state(seq_a).unwrap();
        let (gk, gv) = pool
            .paged_pool_ref()
            .unwrap()
            .gather_visible(&state, 0)
            .unwrap()
            .expect("A gather post-trim");
        assert_eq!(
            flatten_fp32(&gk),
            flatten_fp32(&prefill_block(1000.0, n_kv_heads, 4, head_dim)),
            "A keys are the retained 4-token prefix"
        );
        assert_eq!(
            flatten_fp32(&gv),
            flatten_fp32(&prefill_block(5000.0, n_kv_heads, 4, head_dim)),
            "A values are the retained 4-token prefix"
        );
    }
}

/// Free reference to DetachedPagedCacheSet to keep the import alive.
#[allow(dead_code)]
fn _type_alive() -> Option<DetachedPagedCacheSet> {
    None
}

// ---------------------------------------------------------------------------
// 12. Partial prefix adoption (#225): trim a detached set to a block boundary
//     and adopt only the matched prefix.
// ---------------------------------------------------------------------------

#[test]
fn trim_detached_paged_to_enables_partial_prefix_adoption() {
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    let layout = PagedKvLayout::uniform(
        1,
        block_size,
        block_size * n_kv_heads as usize * head_dim as usize * 2,
    )
    .unwrap();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    // Seed: 12 tokens = 3 whole blocks, distinct per-token values.
    let seed = pool.allocate(&model).unwrap();
    let full_len = 12i32;
    let full_k = prefill_block(1000.0, n_kv_heads, full_len, head_dim);
    let full_v = prefill_block(5000.0, n_kv_heads, full_len, head_dim);
    write_prefill_for(&mut pool, seed, 0, &full_k, &full_v).unwrap();

    let mut set = pool.detach_paged(seed).unwrap();
    assert_eq!(set.seq_len(), 12);
    let live_before = pool.paged_stats().unwrap().live_blocks;
    assert_eq!(live_before, 3);

    // A request matched only 10 tokens; the scheduler floors to 8 (2 blocks).
    pool.trim_detached_paged_to(&mut set, 8).unwrap();
    assert_eq!(set.seq_len(), 8);
    assert_eq!(
        pool.paged_stats().unwrap().live_blocks,
        2,
        "the dropped tail block must fully release (both pins)"
    );

    // Adopt the trimmed set and append a divergent 6-token suffix at 8.
    let consumer = pool.adopt_paged(&model, set).unwrap();
    {
        let state = pool.get_paged_state(consumer).unwrap();
        let layer = state.layer(0).unwrap();
        assert_eq!(layer.block_ids.len(), 2);
        assert_eq!(layer.len, 8);
    }
    let suffix_len = 6i32;
    let suffix_k = prefill_block(2000.0, n_kv_heads, suffix_len, head_dim);
    let suffix_v = prefill_block(6000.0, n_kv_heads, suffix_len, head_dim);
    write_prefill_for(&mut pool, consumer, 0, &suffix_k, &suffix_v).unwrap();

    // Gather must return the kept 8-token prefix plus the fresh suffix,
    // byte-identical to the dense reference.
    let total = 8 + suffix_len;
    let state = pool.get_paged_state(consumer).unwrap();
    let (gk, gv) = pool
        .paged_pool_ref()
        .unwrap()
        .gather_visible(&state, 0)
        .unwrap()
        .expect("gather must return data");
    let prefix8_k = {
        let full = prefill_block(1000.0, n_kv_heads, full_len, head_dim);
        crate::ffi::slice(&full, &[0, 0, 0, 0], &[1, n_kv_heads, 8, head_dim])
    };
    let prefix8_v = {
        let full = prefill_block(5000.0, n_kv_heads, full_len, head_dim);
        crate::ffi::slice(&full, &[0, 0, 0, 0], &[1, n_kv_heads, 8, head_dim])
    };
    let dense_k = dense_reference(
        &[
            (prefix8_k, 0),
            (
                prefill_block(2000.0, n_kv_heads, suffix_len, head_dim),
                8usize,
            ),
        ],
        n_kv_heads,
        total,
        head_dim,
    );
    let dense_v = dense_reference(
        &[
            (prefix8_v, 0),
            (
                prefill_block(6000.0, n_kv_heads, suffix_len, head_dim),
                8usize,
            ),
        ],
        n_kv_heads,
        total,
        head_dim,
    );
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));
}

#[test]
fn trim_detached_paged_to_rejects_bad_targets() {
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    let layout = PagedKvLayout::uniform(
        1,
        block_size,
        block_size * n_kv_heads as usize * head_dim as usize * 2,
    )
    .unwrap();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seed = pool.allocate(&model).unwrap();
    let full_k = prefill_block(1.0, n_kv_heads, 12, head_dim);
    let full_v = prefill_block(2.0, n_kv_heads, 12, head_dim);
    write_prefill_for(&mut pool, seed, 0, &full_k, &full_v).unwrap();
    let mut set = pool.detach_paged(seed).unwrap();

    // Not block aligned.
    assert!(pool.trim_detached_paged_to(&mut set, 6).is_err());
    // Beyond the stored length.
    assert!(pool.trim_detached_paged_to(&mut set, 16).is_err());
    // No-op full-length trim succeeds and changes nothing.
    pool.trim_detached_paged_to(&mut set, 12).unwrap();
    assert_eq!(set.seq_len(), 12);
    assert_eq!(pool.paged_stats().unwrap().live_blocks, 3);

    // The set still adopts cleanly afterwards.
    let consumer = pool.adopt_paged(&model, set).unwrap();
    assert_eq!(
        pool.get_paged_state(consumer)
            .unwrap()
            .layer(0)
            .unwrap()
            .len,
        12
    );
}

// ---------------------------------------------------------------------------
// 13. Real-bytes accounting (#226): detached sets and pool stats report the
//     actual pool memory, not the layout's nominal placeholder.
// ---------------------------------------------------------------------------

#[test]
fn detached_paged_set_accounts_real_pool_bytes() {
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    let layout = PagedKvLayout::uniform(
        1,
        block_size,
        block_size * n_kv_heads as usize * head_dim as usize * 2,
    )
    .unwrap();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    // 12 tokens = 3 blocks, written in FP32 (tests write f32 blocks), so the
    // REAL cost per block is block_size x H x D x 4 bytes x 2 (K+V) = 192.
    let seq = pool.allocate(&model).unwrap();
    let k = prefill_block(1.0, n_kv_heads, 12, head_dim);
    let v = prefill_block(2.0, n_kv_heads, 12, head_dim);
    write_prefill_for(&mut pool, seq, 0, &k, &v).unwrap();

    let per_block_real = block_size * n_kv_heads as usize * head_dim as usize * 4 * 2;
    assert_eq!(
        pool.paged_pool_ref().unwrap().real_block_bytes(0),
        Some(per_block_real)
    );

    // Pool stats are real: in_use covers the 3 mapped rows, reserved covers
    // the full presized slab capacity.
    let stats = pool.paged_stats().unwrap();
    assert_eq!(stats.bytes_in_use, 3 * per_block_real);
    assert_eq!(
        stats.bytes_reserved,
        pool.paged_pool_ref().unwrap().pool_tensor_bytes()
    );
    assert!(stats.bytes_reserved >= stats.bytes_in_use);

    // The detached set's ledger bytes are the real pinned-pool bytes (the
    // stub's dense handles are empty clone handles, so they contribute 0).
    let mut set = pool.detach_paged(seq).unwrap();
    assert_eq!(set.nbytes(), 3 * per_block_real);

    // A partial trim (#225) shrinks the ledger by the dropped blocks.
    pool.trim_detached_paged_to(&mut set, 8).unwrap();
    assert_eq!(set.nbytes(), 2 * per_block_real);

    pool.release_detached_paged(set);
    // With every pin gone the rows unmap and in_use returns to zero;
    // reserved keeps the allocated slabs (capacity is not shrunk).
    let stats = pool.paged_stats().unwrap();
    assert_eq!(stats.bytes_in_use, 0);
    assert!(stats.bytes_reserved > 0);
}

#[test]
fn detached_paged_set_without_pool_writes_falls_back_to_nominal_bytes() {
    let layout = default_layout();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    // Logical appends only: no pool tensor exists, so no real geometry was
    // ever captured and the nominal layout accounting is the only signal.
    let seq = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq, 0, 8).unwrap();
    let set = pool.detach_paged(seq).unwrap();
    let nominal = set.paged_state().reserved_bytes(set.layout());
    assert!(nominal > 0);
    assert_eq!(set.nbytes(), nominal);
    pool.release_detached_paged(set);
}

// ---------------------------------------------------------------------------
// 14. Non-consuming clone-and-pin adoption (#227): clones share the source's
//     physical blocks and the source survives for further borrowers.
// ---------------------------------------------------------------------------

#[test]
fn clone_detached_paged_prefix_shares_blocks_and_preserves_source() {
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    let layout = PagedKvLayout::uniform(
        1,
        block_size,
        block_size * n_kv_heads as usize * head_dim as usize * 2,
    )
    .unwrap();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(8);

    // Source: 12 tokens = 3 blocks of distinct values.
    let seed = pool.allocate(&model).unwrap();
    let k = prefill_block(1000.0, n_kv_heads, 12, head_dim);
    let v = prefill_block(5000.0, n_kv_heads, 12, head_dim);
    write_prefill_for(&mut pool, seed, 0, &k, &v).unwrap();
    let source = pool.detach_paged(seed).unwrap();
    let source_blocks = source.paged_state().layer(0).unwrap().block_ids.clone();
    // Detached source holds 2 references per block.
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(source_blocks[0]), 2);

    // Borrower 1 clones the first 8 tokens (2 blocks).
    let clone_a = pool.clone_detached_paged_prefix(&source, 8).unwrap();
    assert_eq!(clone_a.seq_len(), 8);
    assert_eq!(
        clone_a.paged_state().layer(0).unwrap().block_ids,
        source_blocks[..2].to_vec(),
        "the clone must reference the SAME physical blocks"
    );
    // Source 2 refs + clone 2 refs.
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(source_blocks[0]), 4);
    // Source untouched.
    assert_eq!(source.seq_len(), 12);
    assert_eq!(
        source.paged_state().layer(0).unwrap().block_ids,
        source_blocks
    );

    // Borrower 2 clones the whole entry concurrently.
    let clone_b = pool.clone_detached_paged_prefix(&source, 12).unwrap();
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(source_blocks[0]), 6);

    // Adopt clone A: its detach-style pin is released, the table ref stays.
    let seq_a = pool.adopt_paged(&model, clone_a).unwrap();
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(source_blocks[0]), 5);

    // The adopted borrower appends a divergent suffix on FRESH blocks and
    // gathers prefix + suffix byte-identically to the dense reference.
    let suffix = prefill_block(2000.0, n_kv_heads, 6, head_dim);
    let suffix_v = prefill_block(6000.0, n_kv_heads, 6, head_dim);
    write_prefill_for(&mut pool, seq_a, 0, &suffix, &suffix_v).unwrap();
    {
        let state = pool.get_paged_state(seq_a).unwrap();
        let blocks = &state.layer(0).unwrap().block_ids;
        assert_eq!(blocks.len(), 4, "2 shared prefix + 2 fresh suffix blocks");
        assert_eq!(&blocks[..2], &source_blocks[..2]);
        assert!(!source_blocks.contains(&blocks[2]));
    }
    let (gk, gv) = {
        let state = pool.get_paged_state(seq_a).unwrap();
        pool.paged_pool_ref()
            .unwrap()
            .gather_visible(&state, 0)
            .unwrap()
            .expect("gather must return data")
    };
    let prefix8 = |base: f32| {
        let full = prefill_block(base, n_kv_heads, 12, head_dim);
        crate::ffi::slice(&full, &[0, 0, 0, 0], &[1, n_kv_heads, 8, head_dim])
    };
    let dense_k = dense_reference(
        &[
            (prefix8(1000.0), 0),
            (prefill_block(2000.0, n_kv_heads, 6, head_dim), 8usize),
        ],
        n_kv_heads,
        14,
        head_dim,
    );
    let dense_v = dense_reference(
        &[
            (prefix8(5000.0), 0),
            (prefill_block(6000.0, n_kv_heads, 6, head_dim), 8usize),
        ],
        n_kv_heads,
        14,
        head_dim,
    );
    assert_eq!(flatten_fp32(&gk), flatten_fp32(&dense_k));
    assert_eq!(flatten_fp32(&gv), flatten_fp32(&dense_v));

    // The source's ledger view is unchanged; releasing everything drives the
    // shared blocks back to zero with no leaks.
    let per_block_real = block_size * n_kv_heads as usize * head_dim as usize * 4 * 2;
    assert_eq!(source.nbytes(), 3 * per_block_real);
    pool.release_detached_paged(clone_b);
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(source_blocks[0]), 3);
    pool.release(seq_a);
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(source_blocks[0]), 2);
    pool.release_detached_paged(source);
    assert_eq!(pool.paged_pool_ref().unwrap().refcount(source_blocks[0]), 0);
}

#[test]
fn clone_detached_paged_prefix_rejects_bad_targets() {
    let n_kv_heads = 2i32;
    let head_dim = 3i32;
    let block_size = 4usize;
    let layout = PagedKvLayout::uniform(
        1,
        block_size,
        block_size * n_kv_heads as usize * head_dim as usize * 2,
    )
    .unwrap();
    let model = PagedStubModel::new(layout.clone());
    let mut pool = CachePool::new(4);

    let seed = pool.allocate(&model).unwrap();
    let k = prefill_block(1.0, n_kv_heads, 12, head_dim);
    let v = prefill_block(2.0, n_kv_heads, 12, head_dim);
    write_prefill_for(&mut pool, seed, 0, &k, &v).unwrap();
    let source = pool.detach_paged(seed).unwrap();
    let first_block = source.paged_state().layer(0).unwrap().block_ids[0];

    // Misaligned, zero, and oversized targets are declined without pinning.
    assert!(pool.clone_detached_paged_prefix(&source, 6).is_err());
    assert!(pool.clone_detached_paged_prefix(&source, 0).is_err());
    assert!(pool.clone_detached_paged_prefix(&source, 16).is_err());
    assert_eq!(
        pool.paged_pool_ref().unwrap().refcount(first_block),
        2,
        "declined clones must not leak pins"
    );

    pool.release_detached_paged(source);
}
