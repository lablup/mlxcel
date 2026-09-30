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

//! Paged KV block reclaim under a block budget (#122 b2, #1982).
//!
//! Prefill admission and the decode tick both need pool blocks that a
//! configured `--kv-cache-budget` may not have free. They share one reclaim
//! loop: evict cold prompt-cache prefixes (LRU) first, then preempt running
//! sequences. The loop and the decode-side reservation are written against the
//! small [`PagedBlockReclaimer`] trait so they can be exercised over a real
//! `CachePool` and `PromptCacheStore` without a loaded model.
//!
//! Decode needs this because a pool-backed paged write that cannot acquire a
//! block panics inside the model forward (`write_paged` expects the write to
//! succeed), so every block a tick will mint has to be reserved before the
//! forward runs.

use super::*;

/// The scheduler operations the block reclaim loop drives.
pub(super) trait PagedBlockReclaimer {
    /// Blocks still acquirable before the budget is hit, `None` when unbounded.
    fn free_blocks(&self) -> Option<usize>;
    /// Evict the least recently used prompt-cache entry and return its pinned
    /// blocks to the pool. `false` when there was nothing to evict.
    fn evict_cold_prefix(&mut self) -> bool;
    /// Preempt one running sequence (it re-prefills on resume), restricted to
    /// rows of priority strictly below `below` when set. `false` when no
    /// eligible victim exists.
    fn preempt_one(&mut self, below: Option<RequestPriority>) -> bool;
    /// Number of sequences in the active decode batch.
    fn active_len(&self) -> usize;
    /// Blocks appending `tokens` positions to every layer of `id` would mint.
    fn blocks_to_append(&self, id: SequenceId, tokens: usize) -> usize;
    /// Whether `id` is still in the active batch (a preemption removes it).
    fn is_active(&self, id: SequenceId) -> bool;
    /// Blocks [`Self::free_blocks`] sets aside for a parked chunked prefill.
    /// That prefill joins the batch (and becomes preemptible) once its last
    /// chunk runs, so a row short only by these blocks waits a tick instead of
    /// being shed.
    fn reserved_blocks(&self) -> usize;
}

/// What one reclaim pass did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ReclaimOutcome {
    /// `need` blocks are acquirable now.
    pub fits: bool,
    /// Prompt-cache entries evicted.
    pub evicted: usize,
    /// Running sequences preempted.
    pub preempted: usize,
}

/// Reclaim pool blocks until at least `need` are acquirable, or no further
/// reclamation is possible. Evicts cold prompt-cache prefixes first, then
/// preempts running sequences while more than `preempt_floor` remain active.
/// The scheduler picks preemption victims with
/// [`select_block_reclaim_victim_from`] (least progress first), so the row
/// closest to finishing is never the one discarded.
///
/// Prefill admission passes a floor of 0 (the admitted sequence is not in the
/// batch yet) and its own priority as `preempt_below`, so it only displaces
/// strictly lower-priority rows. Decode passes a floor of 1 and no priority
/// limit: the rows asking for blocks are themselves in the batch, and
/// preempting the last one to make room for itself would only re-prefill it
/// back to the same point.
pub(super) fn reclaim_blocks<R: PagedBlockReclaimer + ?Sized>(
    r: &mut R,
    need: usize,
    preempt_floor: usize,
    preempt_below: Option<RequestPriority>,
) -> ReclaimOutcome {
    let room = |r: &R| r.free_blocks().is_none_or(|free| free >= need);
    let mut outcome = ReclaimOutcome::default();
    // 1. Evict cold cross-request prefixes; releasing their pins frees blocks.
    while !room(r) && r.evict_cold_prefix() {
        outcome.evicted += 1;
    }
    // 2. Preempt running sequences (drop their KV; they re-prefill on resume).
    while !room(r) && r.active_len() > preempt_floor && r.preempt_one(preempt_below) {
        outcome.preempted += 1;
    }
    outcome.fits = room(r);
    outcome
}

/// Blocks a decode tick appending `tokens_per_row` positions to each of `ids`
/// will mint.
pub(super) fn decode_block_need<R: PagedBlockReclaimer + ?Sized>(
    r: &R,
    ids: &[SequenceId],
    tokens_per_row: usize,
) -> usize {
    ids.iter()
        .map(|&id| r.blocks_to_append(id, tokens_per_row))
        .sum()
}

/// Outcome of reserving block budget for one decode tick.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct DecodeReservation {
    /// Rows that decode this tick, in the caller's order.
    pub run: Vec<SequenceId>,
    /// Rows that could not get their blocks even after reclaim; the caller
    /// finishes them with an error instead of running the forward.
    pub shed: Vec<SequenceId>,
    /// Rows that skip this tick: the pool could hold their blocks, but a
    /// parked chunked prefill has them set aside. They decode again once that
    /// prefill joins the batch and reclaim can preempt it.
    pub deferred: Vec<SequenceId>,
    /// The reclaim pass, when the pool did not already have room.
    pub reclaim: Option<ReclaimOutcome>,
}

/// Reserve the blocks a decode tick over `ids` will mint (#1982).
///
/// With room, returns `ids` unchanged and touches nothing. Otherwise runs the
/// shared [`reclaim_blocks`] loop with a preemption floor of 1, drops rows a
/// preemption removed, and, if the budget still cannot cover the remaining
/// rows, removes rows that need a new block (latest first) until it can. A
/// removed row is deferred to a later tick when the blocks it lacks are only
/// the ones set aside for a parked chunked prefill, and shed otherwise. Rows
/// that need no block always run.
pub(super) fn reserve_decode_blocks<R: PagedBlockReclaimer + ?Sized>(
    r: &mut R,
    ids: &[SequenceId],
    tokens_per_row: usize,
) -> DecodeReservation {
    let need = decode_block_need(r, ids, tokens_per_row);
    if r.free_blocks().is_none_or(|free| free >= need) {
        return DecodeReservation {
            run: ids.to_vec(),
            ..DecodeReservation::default()
        };
    }
    let outcome = reclaim_blocks(r, need, 1, None);
    let mut run: Vec<SequenceId> = ids.iter().copied().filter(|&id| r.is_active(id)).collect();
    let mut shed = Vec::new();
    let mut deferred = Vec::new();
    loop {
        let free = r.free_blocks().unwrap_or(usize::MAX);
        let need = decode_block_need(r, &run, tokens_per_row);
        if need <= free {
            break;
        }
        let can_wait = need <= free.saturating_add(r.reserved_blocks());
        match run
            .iter()
            .rposition(|&id| r.blocks_to_append(id, tokens_per_row) > 0)
        {
            Some(pos) if can_wait => deferred.push(run.remove(pos)),
            Some(pos) => shed.push(run.remove(pos)),
            None => break,
        }
    }
    DecodeReservation {
        run,
        shed,
        deferred,
        reclaim: Some(outcome),
    }
}

/// Charge `need` blocks against a batched-prefill window's remaining paged
/// budget. `room == None` means no budget is configured and always admits.
/// Returns `false`, leaving `room` untouched, when the row does not fit.
pub(super) fn take_paged_room(room: &mut Option<usize>, need: usize) -> bool {
    match room {
        None => true,
        Some(free) if need <= *free => {
            *free -= need;
            true
        }
        Some(_) => false,
    }
}

impl PagedBlockReclaimer for BatchScheduler {
    fn free_blocks(&self) -> Option<usize> {
        self.available_paged_blocks()
    }

    fn evict_cold_prefix(&mut self) -> bool {
        let Some(store) = self.prompt_cache.clone() else {
            return false;
        };
        if store.evict_one_lru() == 0 {
            return false;
        }
        self.drain_store_paged_releases();
        true
    }

    fn preempt_one(&mut self, below: Option<RequestPriority>) -> bool {
        match select_block_reclaim_victim_from(self.active_batch.iter_sequences(), below) {
            Some(victim_id) => self.preempt_sequence(victim_id),
            None => false,
        }
    }

    fn active_len(&self) -> usize {
        self.active_batch.len()
    }

    fn blocks_to_append(&self, id: SequenceId, tokens: usize) -> usize {
        self.cache_pool.paged_blocks_to_append(id, tokens)
    }

    fn is_active(&self, id: SequenceId) -> bool {
        self.active_batch.get(id).is_some()
    }

    fn reserved_blocks(&self) -> usize {
        self.chunked_prefill_reserved_blocks()
    }
}

impl BatchScheduler {
    /// Blocks the parked chunked prefill still has to write. Its admission
    /// checked the whole prompt against the budget, but the blocks are only
    /// acquired chunk by chunk, so decode growth and later admissions between
    /// chunks must leave them free or the next chunk hits an exhausted pool.
    pub(super) fn chunked_prefill_reserved_blocks(&self) -> usize {
        self.chunked_prefill_seq.as_ref().map_or(0, |seq| {
            let remaining = seq.prompt_tokens.len().saturating_sub(seq.prefill_offset);
            self.cache_pool
                .paged_blocks_to_append(seq.seq_id, remaining)
        })
    }

    /// Paged blocks acquirable without reclaim once the parked chunked
    /// prefill's remaining blocks are set aside, or `None` when no budget is
    /// configured or the pool does not exist yet. Every block-budget decision
    /// (prefill admission, decode reservation, lookahead prime) reads this
    /// rather than the raw pool figure (#1982).
    pub(super) fn available_paged_blocks(&self) -> Option<usize> {
        self.cache_pool
            .free_paged_block_budget()
            .map(|free| free.saturating_sub(self.chunked_prefill_reserved_blocks()))
    }

    /// Reclaim paged pool blocks until at least `need` are acquirable, or no
    /// further reclamation is possible, for admitting a prefill of
    /// `priority`. Only strictly lower-priority rows may be preempted; an
    /// equal-priority request waits for running rows to finish instead. See
    /// [`reclaim_blocks`]. Returns whether `need` blocks are now acquirable.
    pub(super) fn reclaim_paged_blocks(&mut self, need: usize, priority: RequestPriority) -> bool {
        reclaim_blocks(self, need, 0, Some(priority)).fits
    }

    /// Reserve pool blocks for the decode tick over `seq_ids` (#1982). Returns
    /// `None` when every row may run, or `Some(rows)` with the rows that may.
    ///
    /// A no-op returning `None` when no budget is configured or the pool
    /// already has room. Under pressure it tears down any prebuilt lookahead
    /// (preemption changes batch membership, the #632 invalidation rule), runs
    /// the shared reclaim loop, and finishes any row that still cannot get a
    /// block with an error, so the forward never hits an exhausted pool.
    pub(super) fn reserve_decode_step_blocks(
        &mut self,
        seq_ids: &[SequenceId],
    ) -> Option<Vec<SequenceId>> {
        self.cache_pool.paged_block_budget()?;
        // A tick writes one position per row. The lookahead prime that may
        // follow a synchronous step writes one more; it is gated separately by
        // `prime_fits_block_budget`, so it never needs a reservation here.
        let tokens_per_row = 1;
        let need = decode_block_need(self, seq_ids, tokens_per_row);
        let free_before = self.available_paged_blocks();
        if free_before.is_none_or(|free| free >= need) {
            return None;
        }
        self.discard_lookahead();
        let reservation = reserve_decode_blocks(self, seq_ids, tokens_per_row);
        if let Some(outcome) = reservation.reclaim {
            tracing::info!(
                need,
                free_before = free_before.unwrap_or_default(),
                free_after = self.available_paged_blocks().unwrap_or_default(),
                evicted_prefixes = outcome.evicted,
                preempted = outcome.preempted,
                shed = reservation.shed.len(),
                deferred = reservation.deferred.len(),
                "Decode reclaimed paged KV blocks"
            );
        }
        for &id in &reservation.shed {
            let total = self.cache_pool.paged_block_budget().unwrap_or_default();
            Self::abort_sequence_with_error(
                self.active_batch.get_mut(id),
                "KV cache budget exhausted",
                &format!("no free block in the {total}-block KV cache budget to continue decoding"),
            );
        }
        Some(reservation.run)
    }

    /// Whether the pool can take the one speculative position a lookahead
    /// prime over `seq_ids` appends. A prime that would need a block the
    /// budget cannot give is skipped (the next tick runs synchronously and
    /// reserves through [`Self::reserve_decode_step_blocks`]), rather than
    /// reclaiming for speculative work.
    pub(super) fn prime_fits_block_budget(&self, seq_ids: &[SequenceId]) -> bool {
        self.available_paged_blocks()
            .is_none_or(|free| free >= decode_block_need(self, seq_ids, 1))
    }
}

#[cfg(test)]
#[path = "block_reclaim_tests.rs"]
mod block_reclaim_tests;
