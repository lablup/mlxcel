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

//! [`LagunaDFlashContextCache`]: the drafter's per-layer context K/V window.
//!
//! The Laguna drafter attends from the proposal block to the target's
//! captured context through sliding-window attention (window `W`, 512 on
//! the published checkpoints). Only the context K/V is cached; the proposal
//! K/V is concatenated per forward and never stored, exactly as the Qwen 3.5
//! DFlash port does with its dense `KVCache`.
//!
//! The cache keeps the newest `W - 1` context positions in temporal order.
//! That is the whole set any proposal query can still reach: the first block
//! position `p` attends `[p - W + 1, p]`, which covers the `W - 1` context
//! positions before it; later block positions see strictly fewer context
//! entries, which the attention mask (not this cache) enforces. Keeping a
//! temporal buffer instead of a ring makes the per-query window a plain
//! `[block, prior + block]` additive mask, and `offset` stays the absolute
//! position of the next context row so RoPE is applied at the target's own
//! positions.

use crate::ffi::{self, MlxArray};
use crate::ops::concatenate;
use cxx::UniquePtr;

/// Temporal context K/V buffer capped at `window - 1` positions.
pub struct LagunaDFlashContextCache {
    keys: Option<UniquePtr<MlxArray>>,
    values: Option<UniquePtr<MlxArray>>,
    /// Sliding window `W`; the buffer keeps at most `W - 1` positions.
    window: i32,
    /// Absolute position of the next context row.
    offset: i32,
}

impl LagunaDFlashContextCache {
    /// Empty cache for a layer with sliding window `window` (`>= 2`).
    pub fn new(window: i32) -> Self {
        debug_assert!(window >= 2, "sliding window must be at least 2");
        Self {
            keys: None,
            values: None,
            window,
            offset: 0,
        }
    }

    /// Sliding window this cache serves.
    pub fn window(&self) -> i32 {
        self.window
    }

    /// Largest number of context positions the cache retains.
    pub fn capacity(&self) -> i32 {
        self.window - 1
    }

    /// Absolute position of the next context row (the position the first
    /// proposal token of the next block occupies after the update).
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Number of context positions currently held.
    pub fn len(&self) -> i32 {
        self.keys
            .as_ref()
            .map(|k| ffi::array_shape(k)[2])
            .unwrap_or(0)
    }

    /// Whether no context position is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Skip `n` context positions that were dropped before projection (they
    /// are older than any position the next block can attend). Advances the
    /// absolute offset without touching the buffer.
    pub fn advance(&mut self, n: i32) {
        if n > 0 {
            self.offset += n;
        }
    }

    /// Append `new_keys` / `new_values` (`[B, H, T, D]`, already RoPE'd at
    /// positions `offset..offset + T`) and return the retained context
    /// window in temporal order: the newest `window - 1` positions,
    /// including the rows just appended.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let incoming = ffi::array_shape(&new_keys)[2];
        let (keys, values) = match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => (
                concatenate(&k, &new_keys, 2),
                concatenate(&v, &new_values, 2),
            ),
            _ => (new_keys, new_values),
        };
        let total = ffi::array_shape(&keys)[2];
        let keep = self.capacity();
        let (keys, values) = if total > keep {
            let k_shape = ffi::array_shape(&keys);
            let v_shape = ffi::array_shape(&values);
            let start = total - keep;
            (
                ffi::slice(
                    &keys,
                    &[0, 0, start, 0],
                    &[k_shape[0], k_shape[1], total, k_shape[3]],
                ),
                ffi::slice(
                    &values,
                    &[0, 0, start, 0],
                    &[v_shape[0], v_shape[1], total, v_shape[3]],
                ),
            )
        } else {
            (keys, values)
        };
        self.offset += incoming;
        self.keys = Some(ffi::contiguous(&keys, false));
        self.values = Some(ffi::contiguous(&values, false));
        (ffi::copy(&keys), ffi::copy(&values))
    }
}
