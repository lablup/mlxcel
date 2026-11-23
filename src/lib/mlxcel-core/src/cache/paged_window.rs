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

//! Window-bounded sliding layers in the paged pool (issue #3256).
//!
//! A sliding layer with window `w` attends, for a query at position `p`, the
//! keys `(p - w, p]`. Before an append at length `len`, the pool therefore
//! moves `logical_start` up to `len - (w - 1)`: a decode step then sees `w`
//! keys, and a prefill chunk of `L` rows sees the `w - 1` earlier keys plus
//! its own, with the caller's band mask (or the windowed causal kernel)
//! hiding the rest. Settling before the append, not after it, keeps a pad
//! trim exact: the pad positions trail the chunk, so dropping them never
//! needs a key the settle already dropped.
//!
//! Blocks wholly before `logical_start` leave the live table: `block_base`
//! advances and each one moves to `retired` with its pin held. With
//! retention on (the prompt cache is enabled) they stay there for donation.
//! With it off, all but the newest [`REWIND_RESERVE_BLOCKS`] go back to the
//! pool at once; the reserve covers the decode lookahead rewind, which can
//! move the window start back by one position.

use super::{PagedBlockPool, PagedLayerState};

/// Retired blocks a sliding layer keeps pinned when retention is off, so a
/// decode lookahead rewind (at most
/// [`crate::cache::DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS`] positions, which
/// moves the window start back by at most one) can bring its head back.
pub const REWIND_RESERVE_BLOCKS: usize = 1;

impl PagedBlockPool {
    /// Keep blocks that leave a sliding window pinned on their sequence for
    /// prompt-cache donation (`true`), or release them beyond the rewind
    /// reserve (`false`, the default).
    pub fn set_retain_retired(&mut self, retain: bool) {
        self.retain_retired = retain;
    }

    /// Cap the retired blocks one sliding layer keeps while retention is on
    /// (the prompt-cache byte budget, converted to blocks by the caller).
    pub fn set_retired_cap_blocks(&mut self, cap: usize) {
        self.retired_cap_blocks = cap;
    }

    /// Retired sliding blocks `state` pins and their real bytes (issue #3256).
    /// Bytes are `0` for a layer whose pool rows are not allocated yet.
    #[must_use]
    pub fn retired_footprint(&self, state: &super::PagedSequenceState) -> (usize, usize) {
        let mut blocks = 0;
        let mut bytes = 0;
        for (layer_idx, layer) in state.layers.iter().enumerate() {
            blocks += layer.retired.len();
            bytes += layer.retired.len() * self.real_block_bytes(layer_idx).unwrap_or(0);
        }
        (blocks, bytes)
    }

    /// Whether retired sliding blocks stay pinned on their sequence.
    #[must_use]
    pub fn retains_retired(&self) -> bool {
        self.retain_retired
    }

    /// Move `logical_start` of a sliding layer to `len - (window - 1)` and
    /// retire the head blocks wholly before it. A no-op for a full layer.
    pub(super) fn settle_window_before_append(
        &mut self,
        layer: &mut PagedLayerState,
    ) -> Result<(), String> {
        let Some(window) = layer.window else {
            return Ok(());
        };
        let start = layer.len.saturating_sub(window.saturating_sub(1));
        if start > layer.logical_start {
            layer.logical_start = start;
        }
        self.retire_head_blocks(layer)
    }

    /// Move every live block wholly before `logical_start` to `retired`, then
    /// release retired blocks beyond the reserve when retention is off.
    fn retire_head_blocks(&mut self, layer: &mut PagedLayerState) -> Result<(), String> {
        let block_size = self.layout.block_size.max(1);
        let first_visible = layer.logical_start / block_size;
        let retire = first_visible
            .saturating_sub(layer.block_base)
            .min(layer.block_ids.len());
        if retire == 0 {
            return Ok(());
        }
        for (i, block_id) in layer.block_ids.drain(..retire).enumerate() {
            layer.retired.push((layer.block_base + i, block_id));
        }
        layer.block_base += retire;
        if !self.retain_retired {
            let excess = layer.retired.len().saturating_sub(REWIND_RESERVE_BLOCKS);
            for (_, block_id) in layer.retired.drain(..excess).collect::<Vec<_>>() {
                self.release_block(block_id)?;
            }
        } else {
            let cap = self.retired_cap_blocks.max(REWIND_RESERVE_BLOCKS);
            while layer.retired.len() > cap {
                let victim = evict_retired_index(layer, block_size);
                let (_, block_id) = layer.retired.remove(victim);
                self.release_block(block_id)?;
            }
        }
        self.bump_block_epoch();
        Ok(())
    }

    /// Release `layer`'s retired blocks that no anchor needs, oldest first,
    /// until at most `keep_unanchored` of them remain. Returns how many were
    /// released. The donation path calls this so an entry carries its
    /// anchors and the newest history, not every block the window passed.
    pub fn prune_retired(
        &mut self,
        layer: &mut PagedLayerState,
        keep_unanchored: usize,
    ) -> Result<usize, String> {
        let block_size = self.layout.block_size.max(1);
        let unanchored: Vec<usize> = (0..layer.retired.len())
            .filter(|&i| !anchored(layer, layer.retired[i].0, block_size))
            .collect();
        let drop = unanchored.len().saturating_sub(keep_unanchored);
        let mut released = 0;
        for &i in unanchored[..drop].iter().rev() {
            let (_, block_id) = layer.retired.remove(i);
            self.release_block(block_id)?;
            released += 1;
        }
        if released > 0 {
            self.bump_block_epoch();
        }
        Ok(released)
    }

    /// Trim `token_count` positions off a sliding layer's tail and bring back
    /// the retired head blocks the shorter sequence's window needs.
    ///
    /// # Errors
    ///
    /// A block the window needs was already released (retention off and the
    /// rewind went past the reserve, or an evicted retired block).
    pub(super) fn trim_window_tokens(
        &mut self,
        layer: &mut PagedLayerState,
        token_count: usize,
    ) -> Result<usize, String> {
        let window = layer.window.unwrap_or(usize::MAX);
        let trimmed = token_count.min(layer.len);
        if trimmed == 0 {
            return Ok(0);
        }
        let new_len = layer.len - trimmed;
        if new_len == 0 {
            while let Some(block_id) = layer.block_ids.pop() {
                self.release_block(block_id)?;
            }
            for (_, block_id) in std::mem::take(&mut layer.retired) {
                self.release_block(block_id)?;
            }
            layer.len = 0;
            layer.logical_start = 0;
            layer.block_base = 0;
            self.bump_block_epoch();
            return Ok(trimmed);
        }

        let block_size = self.layout.block_size.max(1);
        let start = new_len.saturating_sub(window.saturating_sub(1));
        let need_base = layer.logical_start.min(start) / block_size;
        while layer.block_base > need_base {
            match layer.retired.last().copied() {
                Some((abs, block_id)) if abs + 1 == layer.block_base => {
                    layer.retired.pop();
                    layer.block_ids.insert(0, block_id);
                    layer.block_base = abs;
                }
                _ => {
                    return Err(format!(
                        "PagedBlockPool: sliding rewind to length {new_len} needs block {} which \
                         is no longer held",
                        layer.block_base - 1
                    ));
                }
            }
        }
        layer.logical_start = layer.logical_start.min(start);
        layer.len = new_len;
        let required_blocks = new_len.div_ceil(block_size);
        while layer.end_block() > required_blocks {
            if let Some(block_id) = layer.block_ids.pop() {
                self.release_block(block_id)?;
            } else {
                break;
            }
        }
        self.bump_block_epoch();
        Ok(trimmed)
    }

    /// Drop every retired block of `layer` (a sequence that will not donate,
    /// or whose retained blocks are reclaimed). Returns how many were freed.
    pub fn release_retired(&mut self, layer: &mut PagedLayerState) -> Result<usize, String> {
        let n = layer.retired.len();
        for (_, block_id) in std::mem::take(&mut layer.retired) {
            self.release_block(block_id)?;
        }
        if n > 0 {
            self.bump_block_epoch();
        }
        Ok(n)
    }
}

/// Whether absolute block `abs` lies in the window an anchor of `layer` needs,
/// `[anchor - window, anchor)`.
fn anchored(layer: &PagedLayerState, abs: usize, block_size: usize) -> bool {
    let window = layer.window.unwrap_or(0);
    layer.anchors.iter().any(|&anchor| {
        let first = anchor.saturating_sub(window) / block_size;
        let end = anchor.div_ceil(block_size);
        (first..end).contains(&abs)
    })
}

/// The retired entry to evict next: the oldest unanchored block, or the
/// oldest block when every one is anchored. Never the newest
/// [`REWIND_RESERVE_BLOCKS`], which a decode rewind may need.
fn evict_retired_index(layer: &PagedLayerState, block_size: usize) -> usize {
    let protected = layer.retired.len().saturating_sub(REWIND_RESERVE_BLOCKS);
    (0..protected)
        .find(|&i| !anchored(layer, layer.retired[i].0, block_size))
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "paged_window_tests.rs"]
mod tests;
