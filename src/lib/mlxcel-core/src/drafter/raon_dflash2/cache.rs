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

//! [`ContextWindow`]: one drafter layer's context K/V.
//!
//! Each layer keeps the K/V of the newest `window` context positions in
//! temporal order (`[1, kv_heads, len, head_dim]`), RoPE'd at their absolute
//! target positions. A proposal block of `n` rows attends the newest
//! `window - n` of them plus itself, so keys never exceed `window`, as the
//! vendor `DFlash2Draft._attn` does. The block's own K/V is never stored.

use crate::ffi::{self, MlxArray};
use crate::ops::concatenate;
use cxx::UniquePtr;

/// Temporal context K/V buffer of one layer, capped at `window` positions.
pub struct ContextWindow {
    keys: Option<UniquePtr<MlxArray>>,
    values: Option<UniquePtr<MlxArray>>,
    window: i32,
}

impl ContextWindow {
    pub fn new(window: i32) -> Self {
        Self {
            keys: None,
            values: None,
            window,
        }
    }

    /// Positions held.
    pub fn len(&self) -> i32 {
        self.keys.as_ref().map_or(0, |k| ffi::array_shape(k)[2])
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every held position (the context restarted past the window).
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
    }

    /// Append `[1, H, T, D]` keys and values and keep the newest `window`.
    pub fn append(&mut self, keys: UniquePtr<MlxArray>, values: UniquePtr<MlxArray>) {
        let (keys, values) = match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => (concatenate(&k, &keys, 2), concatenate(&v, &values, 2)),
            _ => (keys, values),
        };
        let total = ffi::array_shape(&keys)[2];
        let (keys, values) = if total > self.window {
            (
                tail(&keys, total - self.window),
                tail(&values, total - self.window),
            )
        } else {
            (keys, values)
        };
        self.keys = Some(ffi::contiguous(&keys, false));
        self.values = Some(ffi::contiguous(&values, false));
    }

    /// The newest `window - block` positions (keys, values), or `None` when
    /// the window is empty.
    pub fn recent(&self, block: i32) -> Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let keys = self.keys.as_ref()?;
        let values = self.values.as_ref()?;
        let len = ffi::array_shape(keys)[2];
        let keep = (self.window - block).max(0);
        let skip = (len - keep).max(0);
        Some((tail(keys, skip), tail(values, skip)))
    }
}

/// `x[:, :, skip:]` of a `[B, H, S, D]` tensor.
fn tail(x: &MlxArray, skip: i32) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(x);
    ffi::slice(x, &[0, 0, skip, 0], &[s[0], s[1], s[2], s[3]])
}
