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

//! Paged KV admission watermark (issue #2088).
//!
//! Under a tight `--kv-cache-budget`, admitting a prefill into exactly the
//! free blocks leaves the running rows no room to grow, so the next decode
//! ticks preempt them and they re-prefill. The watermark keeps a fraction of
//! the budget free at admission while a decode batch is live, trading a later
//! admission for fewer preemptions.

use super::block_reclaim::{PagedBlockReclaimer, reclaim_blocks};
use super::*;

/// Blocks the admission watermark keeps free: `fraction` of the `total`
/// budget, rounded up, but only while a decode batch is live. With no
/// running row there is no decode growth to protect, and holding headroom
/// back would wedge a request that fits the budget but not the budget minus
/// the watermark.
pub(super) fn admission_watermark_blocks(total: usize, fraction: f64, decode_live: bool) -> usize {
    if !decode_live || fraction <= 0.0 {
        return 0;
    }
    // `fraction` is clamped to `0.0..=MAX_KV_ADMISSION_WATERMARK`, so the
    // product fits `total`.
    ((total as f64) * fraction).ceil() as usize
}

/// Whether a prefill needing `need` blocks may be admitted now, keeping
/// `watermark` blocks free for decode growth (issue #2088). Admits when the
/// free blocks minus the watermark cover the request; otherwise reclaims
/// toward `need + watermark` (cold prefixes first, then rows of strictly
/// lower priority than `priority`) and admits only if that much is free.
/// `false` means the caller defers the request until running rows finish.
pub(super) fn admit_with_watermark<R: PagedBlockReclaimer + ?Sized>(
    r: &mut R,
    need: usize,
    watermark: usize,
    priority: RequestPriority,
) -> bool {
    let target = need.saturating_add(watermark);
    if r.free_blocks().is_none_or(|free| free >= target) {
        return true;
    }
    reclaim_blocks(r, target, 0, Some(priority), false).fits
}

impl BatchScheduler {
    /// Blocks the admission watermark currently keeps free (issue #2088); 0
    /// without a budget or while no row is decoding.
    pub(super) fn paged_admission_headroom(&self) -> usize {
        self.cache_pool.paged_block_budget().map_or(0, |total| {
            admission_watermark_blocks(
                total,
                self.paged_admission_watermark,
                !self.active_batch.is_empty(),
            )
        })
    }

    /// Admit a prefill needing `need` blocks at `priority` under the
    /// admission watermark, reclaiming if needed. Only strictly lower-priority
    /// rows may be preempted; an equal-priority request waits for running rows
    /// to finish instead. See [`admit_with_watermark`].
    pub(super) fn reclaim_paged_blocks(&mut self, need: usize, priority: RequestPriority) -> bool {
        let watermark = self.paged_admission_headroom();
        admit_with_watermark(self, need, watermark, priority)
    }
}
