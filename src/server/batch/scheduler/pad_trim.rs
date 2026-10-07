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

//! Tile-alignment decision shared by every prefill site (issue #1755).
//!
//! The post-pad trim that pairs with it lives in the engine
//! ([`mlxcel_core::engine::Engine::trim_padding`], run inside
//! [`mlxcel_core::engine::Engine::prefill`] for a planned piece), so no site
//! can trim the pool caches and forget a model-owned family's own state.

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
