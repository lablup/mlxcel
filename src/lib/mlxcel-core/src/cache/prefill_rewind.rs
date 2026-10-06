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

//! Rewinding the pad tail of a tile-aligned padded prefill out of a
//! [`RotatingKVCache`] (issue #1755).
//!
//! [`RotatingKVCache::trim`] mirrors upstream mlx-lm and only rewinds
//! `offset`/`idx`, leaving the physical buffer at its previous length. That is
//! what speculative decode wants, but it is not enough after a padded prefill
//! whose chunk outgrew the window: the next single-token update pre-trims the
//! buffer by its PHYSICAL length (keeping the last `max_size` slots), so the
//! pad keys would survive into the decode window. The rewind here also cuts the
//! physical buffer back to `idx`, which leaves exactly the state an unpadded
//! prefill of the same tokens produces.

use super::{KVCacheMode, RotatingKVCache};
use crate::ffi;

impl RotatingKVCache {
    /// Drop the last `n` positions written by a multi-token (prefill) append,
    /// leaving the cache as if those positions had never been appended.
    ///
    /// Only valid while the buffer is chronological, which is what a
    /// multi-token append always leaves behind: no speculative buffering, and a
    /// physical length equal to the write cursor. A ring that a single-token
    /// decode has wrapped is refused, as is Turbo4Asym storage (its V side is
    /// packed into sidecar buffers this slice does not cover). Every refusal is
    /// an `Err` and leaves the cache untouched.
    ///
    /// Used by: model-owned families' sequence-aware pad trim
    /// (`LanguageModel::trim_sequence_state`), Gemma 3.
    pub fn rewind_padded_prefill(&mut self, n: i32) -> Result<(), String> {
        if n < 0 {
            return Err(format!("rotating prefill rewind: negative count {n}"));
        }
        if n == 0 {
            return Ok(());
        }
        if self.mode == KVCacheMode::Turbo4Asym {
            return Err("rotating prefill rewind: Turbo4Asym storage is not supported".into());
        }
        if self.buffer_size != 0 {
            return Err("rotating prefill rewind: speculative buffering is active".into());
        }
        let (Some(keys), Some(values)) = (self.keys.as_ref(), self.values.as_ref()) else {
            return Err("rotating prefill rewind: the cache holds no keys/values".into());
        };
        let k_shape = ffi::array_shape(keys);
        let v_shape = ffi::array_shape(values);
        let physical = k_shape[2];
        if physical != self.idx || v_shape[2] != physical {
            return Err(format!(
                "rotating prefill rewind: buffer is not chronological \
                 (physical {physical}, idx {}, offset {})",
                self.idx, self.offset
            ));
        }
        if n > self.idx || n > self.offset {
            return Err(format!(
                "rotating prefill rewind: cannot drop {n} of {} written positions",
                self.idx
            ));
        }

        let keep = self.idx - n;
        let k = ffi::slice(
            keys,
            &[0, 0, 0, 0],
            &[k_shape[0], k_shape[1], keep, k_shape[3]],
        );
        let v = ffi::slice(
            values,
            &[0, 0, 0, 0],
            &[v_shape[0], v_shape[1], keep, v_shape[3]],
        );
        let trimmed = self.trim(n);
        debug_assert_eq!(trimmed, n, "chronological trim drops exactly n");
        debug_assert_eq!(self.idx, keep);
        self.keys = Some(ffi::contiguous(&k, false));
        self.values = Some(ffi::contiguous(&v, false));
        // The buffers were replaced, so no earlier write owns them any more.
        self.inplace_marker = None;
        Ok(())
    }
}
