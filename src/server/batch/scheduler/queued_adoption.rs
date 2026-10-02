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

//! Giving back a queued request's adopted prompt-cache prefix (issue #2088).
//!
//! A request adopts a cached prefix when it is enqueued, which pins the
//! prefix's paged blocks for as long as it waits for admission. Under a tight
//! `--kv-cache-budget` (and more often with an admission watermark, which
//! lengthens the wait) those pins can be the only reclaimable blocks left
//! when a running row needs one. The block reclaim loop drops them before it
//! preempts running work.

use super::*;

impl BatchScheduler {
    /// Drop the cached prefix the lowest-priority, newest queued text request
    /// adopted at enqueue (issue #2088): release its cache slot, allocate a
    /// fresh one, and reset its offsets so it prefills cold when admitted, the
    /// way [`Self::preempt_sequence`] resets a running row. Its prompt-cache
    /// context moves to the new id, so it still donates its prefix back when
    /// it finishes. Multimodal requests are skipped: an adopted VLM request
    /// dropped its full-prompt embeddings and could not re-prefill cold.
    /// Returns whether an adoption was dropped.
    pub(super) fn drop_queued_adoption(&mut self) -> bool {
        let Some(old_id) = self.prefill_queue.find_lowest_first(|seq| {
            seq.already_cached_tokens > 0
                && !seq.is_vlm_request()
                && seq.audio.is_empty()
                && self
                    .cache_pool
                    .paged_blocks_to_reach(seq.seq_id, 0)
                    .is_some()
        }) else {
            return false;
        };
        self.release_sequence_caches(old_id);
        let new_id = match self.allocate_sequence_state() {
            Ok(id) => id,
            Err(err) => {
                self.prompt_cache_seq_ctx.remove(&old_id);
                if let Some(seq) = self.prefill_queue.remove(old_id) {
                    let _ = seq.response_tx.send(GenerateEvent::Error(format!(
                        "Server busy: re-allocation after dropping the cached prefix failed: {err}"
                    )));
                }
                return true;
            }
        };
        if let Some(ctx) = self.prompt_cache_seq_ctx.remove(&old_id) {
            self.prompt_cache_seq_ctx.insert(new_id, ctx);
        }
        if let Some(seq) = self.prefill_queue.get_mut(old_id) {
            tracing::info!(
                old = %old_id,
                new = %new_id,
                cached_tokens = seq.already_cached_tokens,
                "Dropped a queued request's adopted prefix to free paged KV blocks"
            );
            seq.seq_id = new_id;
            seq.prefill_offset = 0;
            seq.prefill_start_offset = 0;
            seq.already_cached_tokens = 0;
        }
        true
    }
}
