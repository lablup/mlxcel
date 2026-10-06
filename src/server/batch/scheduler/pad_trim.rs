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

//! Tile-alignment decision and the post-pad trim shared by every prefill site
//! (issue #1755).
//!
//! A padded prefill writes `excess` pad positions after the real tokens. For a
//! dense or paged sequence they sit in the `CachePool` entry's `KVCache`s; for
//! a family whose own layout is model-owned they sit in the model's
//! per-sequence state, which only [`LanguageModel::trim_sequence_state`] can
//! reach. [`trim_padded_prefill`] covers both, so no site can trim one and
//! forget the other.

use mlxcel_core::cache::{SequenceId, SequenceStateBackend};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;

#[cfg(test)]
thread_local! {
    static ALIGNMENT_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Whether prefill pads to the 32-token Neural Accelerator tile.
///
/// Follows [`mlxcel_core::generate::prefill_tile_alignment_enabled`]: M5+ with
/// a Neural Accelerator, `MLXCEL_NO_PADDED_PREFILL` to disable,
/// `MLXCEL_FORCE_PADDED_PREFILL` to force on any hardware. Tests pin it per
/// thread through [`set_alignment_override_for_test`].
#[inline]
pub(super) fn should_align_prefill() -> bool {
    #[cfg(test)]
    if let Some(forced) = ALIGNMENT_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
    mlxcel_core::generate::prefill_tile_alignment_enabled()
}

/// Pin [`should_align_prefill`] for the current thread (`None` restores the
/// hardware/env decision). Thread-local so concurrent tests cannot see it.
#[cfg(test)]
pub(crate) fn set_alignment_override_for_test(value: Option<bool>) {
    ALIGNMENT_OVERRIDE.with(|cell| cell.set(value));
}

/// Drop the `excess` pad positions a padded prefill pass wrote for `seq_id`.
///
/// `caches` is the sequence's `CachePool` entry. It is trimmed for every
/// layout, and is empty (or shadow paged accounting) for a model-owned family,
/// whose own state is rewound through the model hook. The model's natural
/// layout decides, not the allocated backend: a model-owned family allocated on
/// the paged backend for block accounting still keeps its K/V to itself
/// (#1346).
///
/// `Err` means the sequence's state may disagree with its token count; the
/// caller must abort the request.
pub(super) fn trim_padded_prefill<M: LanguageModel + ?Sized>(
    model: &M,
    seq_id: SequenceId,
    caches: &mut [KVCache],
    excess: i32,
) -> Result<(), String> {
    if excess <= 0 {
        return Ok(());
    }
    for cache in caches.iter_mut() {
        cache.trim(excess);
    }
    if model.sequence_state_layout().backend == SequenceStateBackend::ModelOwned {
        model.trim_sequence_state(seq_id, excess)?;
    }
    Ok(())
}
