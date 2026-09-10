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

//! Per-layer context cache of the Muse Glimmer assistant drafter (issue
//! #1343).
//!
//! The drafter's attention reads its keys under a bidirectional sliding mask
//! built from ABSOLUTE positions, `|query_pos - key_pos| <= sliding_window`,
//! so the cache has to hand back its rows in temporal order together with
//! the absolute position of the first row. A plain [`crate::layers::RotatingKVCache`]
//! cannot: once its ring has wrapped, a single-row append returns the buffer
//! in physical slot order, and the drafter appends one row after every
//! zero-accept round. This cache is the temporal window that the mask needs:
//! append-only, never trimmed, at most `window` rows kept, and `offset`
//! counting every row the layer has ever seen (skipped prompt rows
//! included).

use crate::ffi::{self, MlxArray};
use crate::ops::concatenate;
use cxx::UniquePtr;

/// Temporal K/V window of one drafter layer.
pub struct MuseAssistantContextCache {
    keys: Option<UniquePtr<MlxArray>>,
    values: Option<UniquePtr<MlxArray>>,
    window: i32,
    /// Absolute position the next appended row takes; the row count the
    /// layer has consumed so far, skipped prompt rows included.
    pub offset: i32,
}

impl MuseAssistantContextCache {
    pub fn new(window: i32) -> Self {
        Self {
            keys: None,
            values: None,
            window,
            offset: 0,
        }
    }

    /// The sliding window this cache keeps, in rows.
    pub fn window(&self) -> i32 {
        self.window
    }

    /// Rows currently held, `<= window`.
    pub fn len(&self) -> i32 {
        self.keys
            .as_ref()
            .map(|k| ffi::array_shape(k)[2])
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Absolute position of the first held row.
    pub fn key_start(&self) -> i32 {
        self.offset - self.len()
    }

    /// Account for `n` rows that were dropped before they reached the cache
    /// (a prompt longer than the window keeps only its last `window` rows).
    pub fn skip(&mut self, n: i32) {
        self.offset += n.max(0);
    }

    /// Append `[B, H, n, D]` context rows and return the rows now held, in
    /// temporal order: the last `min(len + n, window)` positions.
    pub fn append(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let incoming = ffi::array_shape(&new_keys)[2];
        let (all_keys, all_values) = match (self.keys.as_ref(), self.values.as_ref()) {
            (Some(keys), Some(values)) => (
                concatenate(keys, &new_keys, 2),
                concatenate(values, &new_values, 2),
            ),
            _ => (new_keys, new_values),
        };
        let total = ffi::array_shape(&all_keys)[2];
        let (keys, values) = if total > self.window {
            let k_shape = ffi::array_shape(&all_keys);
            let v_shape = ffi::array_shape(&all_values);
            let start = total - self.window;
            (
                ffi::contiguous(
                    &ffi::slice(
                        &all_keys,
                        &[0, 0, start, 0],
                        &[k_shape[0], k_shape[1], total, k_shape[3]],
                    ),
                    false,
                ),
                ffi::contiguous(
                    &ffi::slice(
                        &all_values,
                        &[0, 0, start, 0],
                        &[v_shape[0], v_shape[1], total, v_shape[3]],
                    ),
                    false,
                ),
            )
        } else {
            (all_keys, all_values)
        };
        self.offset += incoming;
        let out = (ffi::copy(&keys), ffi::copy(&values));
        self.keys = Some(keys);
        self.values = Some(values);
        out
    }

    /// The rows currently held, in temporal order, without appending.
    pub fn fetch(&self) -> Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        match (self.keys.as_ref(), self.values.as_ref()) {
            (Some(keys), Some(values)) => Some((ffi::copy(keys), ffi::copy(values))),
            _ => None,
        }
    }

    /// Drop every row and rewind `offset` to zero.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
    }
}
