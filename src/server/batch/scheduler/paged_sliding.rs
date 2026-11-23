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

//! Prompt-cache plumbing for paged sliding-window sequences (issue #3256).
//!
//! A family that answers `supports_paged_sliding_state()` keeps every layer
//! of an FP16 sequence in the paged pool, sliding layers as window-bounded
//! block tables. Such a sequence donates and adopts through the paged KV
//! path, not through snapshots. Blocks that leave a sliding window stay
//! pinned on the sequence (its retired list) while the prompt cache is on,
//! so a later request that diverges anywhere can still adopt the largest
//! block boundary whose window is intact.
//!
//! The retired blocks count against the prompt-cache byte budget in two
//! steps. A live sequence keeps at most a per-layer cap of them, derived
//! from the store's capacity and the server's parallelism, evicting
//! unanchored blocks oldest-first; the donated entry carries them in its
//! byte size, so the store's LRU then evicts whole entries.

use super::*;
use mlxcel_core::cache::{LayerState, LayerStateKind};

impl BatchScheduler {
    /// Whether the loaded model stores FP16 sliding layers as paged blocks.
    pub(super) fn paged_sliding_family(&self) -> bool {
        self.engine.model().supports_paged_sliding_state()
    }

    /// Whether `seq_id` was allocated with pool-backed layer state.
    pub(super) fn sequence_is_paged_sliding(&self, seq_id: SequenceId) -> bool {
        self.paged_sliding_family()
            && self
                .engine
                .pool()
                .layer_state(seq_id)
                .is_some_and(|layers| layers.iter().any(LayerState::is_pool_backed))
    }

    /// Turn retired-block retention on (with its per-layer cap) while the
    /// prompt cache is enabled, off otherwise. Runs after every allocation,
    /// so the cap follows the pool's real block size once a layer is written.
    pub(super) fn configure_sliding_retention(&mut self) {
        if !self.paged_sliding_family() {
            return;
        }
        let retain = self.prompt_cache_active() && !self.request_memory_policy_enabled();
        let cap = if retain {
            self.sliding_retired_cap_blocks()
        } else {
            0
        };
        self.engine
            .pool_mut()
            .configure_sliding_retention(retain, cap);
    }

    /// Per-layer retired-block cap: the store's byte capacity shared by the
    /// pool's sequences, divided by the bytes one retired block costs across
    /// every sliding layer. Unbounded until the pool's rows exist.
    fn sliding_retired_cap_blocks(&self) -> usize {
        let Some(store) = self.prompt_cache.as_ref() else {
            return 0;
        };
        let Some(kinds) = self.engine.model().layer_state_kinds() else {
            return 0;
        };
        let per_block: usize = match self.engine.pool().paged_pool_ref() {
            Some(pool) => kinds
                .iter()
                .enumerate()
                .filter(|(_, kind)| matches!(kind, LayerStateKind::Sliding { .. }))
                .map(|(layer_idx, _)| pool.real_block_bytes(layer_idx).unwrap_or(0))
                .sum(),
            None => 0,
        };
        if per_block == 0 {
            return usize::MAX;
        }
        let sequences = self.engine.pool().max_sequences().max(1);
        (store.capacity_bytes() / sequences / per_block)
            .max(mlxcel_core::cache::REWIND_RESERVE_BLOCKS)
    }

    /// Record the history boundary and system-prompt root of `seq_id`'s
    /// request as anchors on its sliding layers, so retirement and donation
    /// keep the blocks those reuse points need.
    pub(super) fn install_sliding_anchors(&mut self, seq_id: SequenceId) {
        if !self.sequence_is_paged_sliding(seq_id) {
            return;
        }
        let Some(ctx) = self.prompt_cache_seq_ctx.get(&seq_id) else {
            return;
        };
        let anchors: Vec<usize> = [
            ctx.history_prefix_tokens.as_ref().map(Vec::len),
            ctx.root_prefix_tokens.as_ref().map(Vec::len),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !anchors.is_empty() {
            self.engine.pool_mut().set_sliding_anchors(seq_id, &anchors);
        }
    }

    /// Publish the retired sliding footprint of live sequences and stored
    /// entries to `/v1/cache/stats`.
    pub(super) fn publish_sliding_retired_gauges(&self) {
        if !self.paged_sliding_family() {
            return;
        }
        let pool = self.engine.pool();
        let (mut blocks, mut bytes) = pool.sliding_retired_footprint();
        if let Some(store) = self.prompt_cache.as_ref() {
            (blocks, bytes) = store.fold_paged_sets((blocks, bytes), |(b, n), set| {
                let (bb, nn) = pool.detached_retired_footprint(set);
                (b + bb, n + nn)
            });
        }
        self.batch_observability
            .update_sliding_retired_gauges(blocks as u64, bytes as u64);
    }
}

impl BatchScheduler {
    /// Trim a donated set's unanchored retired blocks to the live cap, so the
    /// entry carries its anchors' windows and the newest history only.
    pub(super) fn prune_donated_retired(
        &mut self,
        set: &mut mlxcel_core::cache::DetachedPagedCacheSet,
    ) {
        let cap = self.sliding_retired_cap_blocks();
        if let Err(err) = self.engine.pool_mut().prune_detached_retired(set, cap) {
            tracing::warn!("prompt-cache donate: retired sliding prune failed: {err}");
        }
    }
}
