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

//! Decode-time paged block reclaim (issue #1982).
//!
//! `BatchScheduler` needs a loaded model, so these cases drive the shared
//! reclaim loop and the decode reservation through [`PagedBlockReclaimer`]
//! over a real `CachePool` (with a block budget) and a real
//! `PromptCacheStore` holding paged prefixes. The harness mirrors the
//! scheduler's trait impl: eviction is `evict_one_lru` + releasing the queued
//! pins, preemption releases a running sequence's blocks.

use std::sync::Arc;
use std::time::Duration;

use mlxcel_core::cache::{CachePool, KVCache, PagedKvLayout, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{
    PagedBlockReclaimer, RequestPriority, reclaim_blocks, reserve_decode_blocks, take_paged_room,
};
use crate::server::prompt_cache::{
    CacheEntry, DetachedKvSet, PromptCacheConfig, PromptCacheStore,
    key::{MultimodalDigest, PromptCacheKey},
};

const LAYERS: usize = 2;
const BLOCK: usize = 4;

struct PagedStub {
    layout: PagedKvLayout,
}

impl LanguageModel for PagedStub {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::zeros(&[1], 0)
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
}

/// A scheduler stand-in: pool + prompt-cache store + the active decode batch.
struct Harness {
    model: PagedStub,
    pool: CachePool,
    store: Arc<PromptCacheStore>,
    active: Vec<SequenceId>,
    preempted: Vec<SequenceId>,
    /// Blocks set aside for a parked chunked prefill, like
    /// `BatchScheduler::chunked_prefill_reserved_blocks`.
    reserved: usize,
}

impl Harness {
    fn new(budget: usize) -> Self {
        let mut pool = CachePool::new(16);
        pool.set_paged_block_budget(Some(budget));
        Self {
            model: PagedStub {
                layout: PagedKvLayout::uniform(LAYERS, BLOCK, 128).unwrap(),
            },
            pool,
            store: Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
                true,
                1 << 30,
                32,
                Duration::from_secs(3600),
                1,
            ))),
            active: Vec::new(),
            preempted: Vec::new(),
            reserved: 0,
        }
    }

    /// Start a decoding sequence holding `tokens` KV positions on every layer.
    fn decoding(&mut self, tokens: usize) -> SequenceId {
        let id = self.pool.allocate(&self.model).unwrap();
        self.append_all_layers(id, tokens).unwrap();
        self.active.push(id);
        id
    }

    /// Park a `tokens`-long paged prefix in the prompt cache under `salt`.
    fn cache_prefix(&mut self, tokens: usize, salt: i32) {
        let id = self.pool.allocate(&self.model).unwrap();
        self.append_all_layers(id, tokens).unwrap();
        let set = self.pool.detach_paged(id).expect("detach_paged");
        let key_tokens: Vec<i32> = (0..16).map(|t| salt * 1000 + t).collect();
        let key = PromptCacheKey::new_full(
            "m",
            None,
            "tpl",
            Some("s"),
            MultimodalDigest::empty(),
            &key_tokens,
        );
        self.store
            .insert(
                &key,
                CacheEntry::new(key_tokens.clone(), DetachedKvSet::Paged(set)),
            )
            .expect("insert paged prefix");
    }

    fn append_all_layers(&mut self, id: SequenceId, tokens: usize) -> Result<(), String> {
        for layer in 0..LAYERS {
            self.pool.append_paged_tokens(id, layer, tokens)?;
        }
        Ok(())
    }

    fn release_store(&mut self) {
        self.store.clear();
        for paged in self.store.drain_pending_paged_releases() {
            self.pool.release_detached_paged(paged);
        }
    }
}

impl PagedBlockReclaimer for Harness {
    fn free_blocks(&self) -> Option<usize> {
        self.pool
            .free_paged_block_budget()
            .map(|free| free.saturating_sub(self.reserved))
    }

    fn evict_cold_prefix(&mut self) -> bool {
        if self.store.evict_one_lru() == 0 {
            return false;
        }
        for paged in self.store.drain_pending_paged_releases() {
            self.pool.release_detached_paged(paged);
        }
        true
    }

    fn preempt_one(&mut self, below: Option<RequestPriority>) -> bool {
        // Every harness row is Normal priority, so a limit at or below Normal
        // leaves no eligible victim.
        if below.is_some_and(|limit| limit <= RequestPriority::Normal) {
            return false;
        }
        // Newest (least-progressed) row first, the order
        // `select_block_reclaim_victim_from` uses for equal priorities.
        let Some(victim) = self.active.pop() else {
            return false;
        };
        self.pool.release(victim);
        self.preempted.push(victim);
        true
    }

    fn active_len(&self) -> usize {
        self.active.len()
    }

    fn blocks_to_append(&self, id: SequenceId, tokens: usize) -> usize {
        self.pool.paged_blocks_to_append(id, tokens)
    }

    fn is_active(&self, id: SequenceId) -> bool {
        self.active.contains(&id)
    }

    fn reserved_blocks(&self) -> usize {
        self.reserved
    }
}

/// Budget 8 blocks: a decoding row at a block boundary (1 block x 2 layers)
/// plus two cached prefixes (1 and 2 blocks per layer) fill it exactly.
fn full_pool_with_cached_prefixes() -> (Harness, SequenceId) {
    let mut h = Harness::new(8);
    let row = h.decoding(BLOCK);
    h.cache_prefix(BLOCK, 1); // oldest: 2 blocks
    h.cache_prefix(2 * BLOCK, 2); // newer: 4 blocks
    assert_eq!(h.free_blocks(), Some(0), "cached prefixes fill the budget");
    assert_eq!(h.store.len(), 2);
    (h, row)
}

#[test]
fn decode_past_block_boundary_without_reclaim_exhausts_the_budget() {
    // The pre-#1982 decode path: nothing reclaims, the append fails (inside a
    // real forward this is the `write_paged` panic).
    let (mut h, row) = full_pool_with_cached_prefixes();
    let err = h.append_all_layers(row, 1).unwrap_err();
    assert!(err.contains("block budget exhausted"), "{err}");
    h.release_store();
}

#[test]
fn decode_reclaim_evicts_lru_prefix_so_the_row_crosses_the_boundary() {
    let (mut h, row) = full_pool_with_cached_prefixes();

    let reservation = reserve_decode_blocks(&mut h, &[row], 1);
    assert_eq!(reservation.run, vec![row], "the row decodes");
    assert!(reservation.shed.is_empty());
    let outcome = reservation.reclaim.expect("the pool had no room");
    assert!(outcome.fits);
    assert_eq!(outcome.evicted, 1, "only the LRU prefix is evicted");
    assert_eq!(outcome.preempted, 0);
    assert_eq!(h.store.len(), 1, "the newer prefix stays cached");

    h.append_all_layers(row, 1)
        .expect("the reserved blocks let the row cross the boundary");
    h.release_store();
}

#[test]
fn decode_with_free_budget_evicts_and_preempts_nothing() {
    let mut h = Harness::new(32);
    let row = h.decoding(BLOCK);
    let other = h.decoding(BLOCK + 1);
    h.cache_prefix(BLOCK, 1);
    let free_before = h.free_blocks();

    let reservation = reserve_decode_blocks(&mut h, &[row, other], 1);
    assert_eq!(reservation.run, vec![row, other]);
    assert!(reservation.shed.is_empty());
    assert_eq!(reservation.reclaim, None, "no reclaim pass runs with room");
    assert_eq!(h.store.len(), 1, "cached prefix untouched");
    assert!(h.preempted.is_empty());
    assert_eq!(h.free_blocks(), free_before);
    h.release_store();
}

#[test]
fn decode_without_evictable_prefixes_falls_back_to_preemption() {
    // Budget 4: two rows at a block boundary hold 2 blocks each, and the store
    // is empty, so only preemption can make room.
    let mut h = Harness::new(4);
    let first = h.decoding(BLOCK);
    let second = h.decoding(BLOCK);
    assert_eq!(h.free_blocks(), Some(0));

    let reservation = reserve_decode_blocks(&mut h, &[first, second], 1);
    assert_eq!(h.preempted, vec![second], "one running row is preempted");
    assert_eq!(
        reservation.run,
        vec![first],
        "the preempted row leaves the tick"
    );
    assert!(
        reservation.shed.is_empty(),
        "the survivor fits after preemption"
    );
    let outcome = reservation.reclaim.expect("reclaim ran");
    assert_eq!((outcome.evicted, outcome.preempted), (0, 1));

    h.append_all_layers(first, 1)
        .expect("preemption freed the blocks the survivor needs");
}

#[test]
fn decode_that_cannot_reclaim_sheds_the_row_instead_of_failing_the_forward() {
    // One row owns the whole 2-block budget and nothing is cached. It is not
    // preempted (the floor keeps the last row: re-prefilling it would only
    // return here), so it is shed and the tick never runs its forward.
    let mut h = Harness::new(2);
    let row = h.decoding(BLOCK);

    let reservation = reserve_decode_blocks(&mut h, &[row], 1);
    assert!(
        h.preempted.is_empty(),
        "the last row is never self-preempted"
    );
    assert!(reservation.run.is_empty());
    assert_eq!(reservation.shed, vec![row]);
    assert_eq!(reservation.reclaim.map(|o| o.fits), Some(false));
}

#[test]
fn shedding_keeps_rows_that_need_no_new_block() {
    // Budget 4, nothing cached, and no preemption victim (e.g. every row
    // carries a structured-output constraint). Only the row that needs a
    // block is shed; the mid-block row still decodes.
    let mut h = Harness::new(4);
    let boundary = h.decoding(BLOCK);
    let mid_block = h.decoding(BLOCK - 1);
    assert_eq!(h.free_blocks(), Some(0));

    let reservation = reserve_decode_blocks(&mut NoPreempt(&mut h), &[boundary, mid_block], 1);
    assert_eq!(reservation.run, vec![mid_block]);
    assert_eq!(reservation.shed, vec![boundary]);
    h.append_all_layers(mid_block, 1)
        .expect("a mid-block write needs no new block");
}

#[test]
fn admission_floor_zero_may_preempt_every_lower_priority_row() {
    // Prefill admission's candidate is not in the batch, so a higher-priority
    // request's reclaim may preempt down to an empty batch.
    let mut h = Harness::new(4);
    h.decoding(BLOCK);
    h.decoding(BLOCK);
    let outcome = reclaim_blocks(&mut h, 4, 0, Some(RequestPriority::High));
    assert!(outcome.fits);
    assert_eq!(outcome.preempted, 2);
    assert!(h.active.is_empty());
}

/// Wraps a harness whose preemption policy finds no victim (e.g. every row
/// carries a structured-output constraint).
struct NoPreempt<'a>(&'a mut Harness);

impl PagedBlockReclaimer for NoPreempt<'_> {
    fn free_blocks(&self) -> Option<usize> {
        self.0.free_blocks()
    }
    fn evict_cold_prefix(&mut self) -> bool {
        self.0.evict_cold_prefix()
    }
    fn preempt_one(&mut self, _below: Option<RequestPriority>) -> bool {
        false
    }
    fn active_len(&self) -> usize {
        self.0.active_len()
    }
    fn blocks_to_append(&self, id: SequenceId, tokens: usize) -> usize {
        self.0.blocks_to_append(id, tokens)
    }
    fn is_active(&self, id: SequenceId) -> bool {
        self.0.is_active(id)
    }
    fn reserved_blocks(&self) -> usize {
        self.0.reserved_blocks()
    }
}

#[test]
fn batched_prefill_window_stops_at_the_first_row_over_the_paged_budget() {
    // Rows needing 40 + 50 fit a 100-block room; the next 20-block row does not
    // fit the remaining 10 and stays queued.
    let mut room = Some(100);
    assert!(take_paged_room(&mut room, 40));
    assert!(take_paged_room(&mut room, 50));
    assert!(!take_paged_room(&mut room, 20));
    assert_eq!(room, Some(10), "a refused row charges nothing");
    // No budget configured: every row is admitted.
    let mut unbounded = None;
    assert!(take_paged_room(&mut unbounded, usize::MAX));
}

#[test]
fn decode_does_not_take_blocks_a_parked_chunked_prefill_still_needs() {
    // Budget 8: the row (2 blocks) and a cached prefix (2 blocks) leave 4
    // free, all of which a parked chunked prefill still has to write. The
    // row's boundary crossing must reclaim the prefix instead of consuming
    // them, or the next prefill chunk would hit an exhausted pool.
    let mut h = Harness::new(8);
    let row = h.decoding(BLOCK);
    h.cache_prefix(BLOCK, 1);
    assert_eq!(h.pool.free_paged_block_budget(), Some(4));
    h.reserved = 4;

    let reservation = reserve_decode_blocks(&mut h, &[row], 1);
    assert_eq!(reservation.run, vec![row]);
    assert_eq!(reservation.reclaim.map(|o| o.evicted), Some(1));
    assert_eq!(h.store.len(), 0);
    assert_eq!(h.free_blocks(), Some(2), "the chunk's 4 blocks stay free");
}

#[test]
fn admission_never_preempts_rows_of_equal_priority() {
    // Two equal-priority requests that cannot both fit would otherwise preempt
    // each other after every prefill; the queued one waits instead.
    let mut h = Harness::new(4);
    h.decoding(BLOCK);
    h.decoding(BLOCK);
    let outcome = reclaim_blocks(&mut h, 4, 0, Some(RequestPriority::Normal));
    assert!(!outcome.fits);
    assert_eq!(outcome.preempted, 0);
    assert_eq!(h.active.len(), 2);
}

#[test]
fn a_row_short_only_by_the_chunked_prefill_reservation_waits_instead_of_failing() {
    // Budget 4: the last running row holds 2 blocks, the other 2 are set aside
    // for a parked chunked prefill, and nothing is cached. The row cannot be
    // preempted (floor) and cannot take the reserved blocks, but the prefill
    // becomes preemptible once it joins the batch, so the row is deferred for
    // this tick rather than failed.
    let mut h = Harness::new(4);
    let row = h.decoding(BLOCK);
    h.reserved = 2;

    let reservation = reserve_decode_blocks(&mut h, &[row], 1);
    assert!(reservation.run.is_empty());
    assert!(reservation.shed.is_empty());
    assert_eq!(reservation.deferred, vec![row]);
}
