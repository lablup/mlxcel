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

//! Keeping only the window on a [`RotatingKVCache`]'s first multi-token fill
//! (issue #3062).
//!
//! A first fill used to store every prompt position, and only the first
//! single-token update cut the buffer back to `max_size`. In a single-pass
//! prefill each sliding layer therefore kept its whole prompt alive until
//! decode began: 8.75 GiB over Gemma 4 12B's 40 sliding layers at 28.7K
//! tokens, against 320 MiB for the windows.
//!
//! A first fill longer than [`Self::first_fill_keep`]'s bound now stores the
//! newest `max_size + FIRST_FILL_REWIND_SLACK` positions as a fresh buffer,
//! scheduled with `async_eval` so the full-length keys are released once the
//! layer's own attention has consumed them. The caller still receives every
//! key for that attention. The stored buffer has the shape a chunked prefill
//! continuation already leaves (chronological, `idx` equal to its length,
//! `offset` past it), so every reader that handles a chunked prefill handles
//! it unchanged:
//!
//! - The next single-token update keeps the newest `max_size` slots before
//!   `idx`, the same positions as before.
//! - The next multi-token append keeps the newest `max_size - 1`, which the
//!   stored tail covers.
//! - A rewind of up to `FIRST_FILL_REWIND_SLACK` positions ([`Self::trim`] or
//!   [`Self::rewind_padded_prefill`]) still leaves a full window behind it.
//!   That bound covers a tile-aligned prefill plan's pad tail, the only rewind
//!   that follows a first fill directly. A rewind past the slack would need
//!   positions the cache no longer holds, so a caller that can issue one (the
//!   server's ragged batched prefill, whose shorter rows rewind the cohort's
//!   whole length difference) wraps the forward in a [`FullFirstFillScope`].
//! - A speculative rollback after a verify block rewinds that block, which a
//!   later append stored in full, and buffered speculative mode
//!   (`buffer_size > 0`) never reaches `update_concat`.
//!
//! Every storage mode is covered: FP16 and the modes that fall back to it, and
//! Turbo4Asym, whose V sidecars are per-token and slice like the keys.

use std::cell::Cell;
use std::marker::PhantomData;

use super::RotatingKVCache;
use crate::ffi::MlxArray;
use cxx::UniquePtr;

/// Positions past `max_size` that a trimmed first fill keeps for a rewind.
///
/// One Neural Accelerator tile: a tile-aligned prefill pads its last piece by
/// at most `NA_TILE_SIZE - 1` positions and rewinds them after the forward.
pub const FIRST_FILL_REWIND_SLACK: i32 = crate::utils::NA_TILE_SIZE as i32;

thread_local! {
    static FULL_FIRST_FILL_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// While alive, rotating caches updated on this thread store their whole first
/// fill, as they did before issue #3062.
///
/// For a forward whose rows will be rewound by more than
/// [`FIRST_FILL_REWIND_SLACK`] positions afterwards. Scopes nest; the cache
/// trims again once the outermost one drops. It is tied to the thread that
/// created it because the cache updates run on the forward's thread.
///
/// Used by: the server's ragged batched prefill cohort.
#[must_use = "the scope retains full first fills only while it is alive"]
pub struct FullFirstFillScope {
    _not_send: PhantomData<*const ()>,
}

impl FullFirstFillScope {
    /// Retain full first fills on this thread until the scope drops.
    pub fn enter() -> Self {
        FULL_FIRST_FILL_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Self {
            _not_send: PhantomData,
        }
    }

    /// Whether a scope is alive on this thread.
    pub fn is_active() -> bool {
        FULL_FIRST_FILL_DEPTH.with(|depth| depth.get() > 0)
    }
}

impl Drop for FullFirstFillScope {
    fn drop(&mut self) {
        FULL_FIRST_FILL_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

impl RotatingKVCache {
    /// How many positions a first fill of `new_seq_len` stores, or `None` to
    /// store all of them: a fill no longer than `max_size +
    /// FIRST_FILL_REWIND_SLACK`, or one made inside a [`FullFirstFillScope`].
    ///
    /// Used by: [`Self::update_concat`], [`Self::update_turbo4_concat`].
    pub(super) fn first_fill_keep(&self, new_seq_len: i32) -> Option<i32> {
        if self.max_size <= 0 || FullFirstFillScope::is_active() {
            return None;
        }
        let keep = self.max_size.saturating_add(FIRST_FILL_REWIND_SLACK);
        (new_seq_len > keep).then_some(keep)
    }

    /// The newest `keep` slots of a first-fill array along the time axis, as
    /// a fresh buffer that does not hold the full array alive.
    ///
    /// Used by: [`Self::update_concat`], [`Self::update_turbo4_concat`].
    pub(super) fn first_fill_tail(arr: &MlxArray, keep: i32) -> UniquePtr<MlxArray> {
        crate::utils::materialized_axis_tail(arr, 2, keep)
    }
}
