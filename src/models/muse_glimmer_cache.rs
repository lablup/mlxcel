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

use crate::models::kv_snapshot::{self, KvSnapshotNames};
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{KVCache, RotatingKVCache};
use mlxcel_core::{MlxArray, UniquePtr};

/// Tensor-name and error-message vocabulary Muse Glimmer hands to the shared
/// serializers in [`crate::models::kv_snapshot`]. The `full` / `sliding`
/// segments are what Muse Glimmer has always written, so a snapshot taken by
/// an earlier build still restores.
const MUSE_KV_SNAPSHOT_NAMES: KvSnapshotNames =
    KvSnapshotNames::new("Muse Glimmer", "full", "sliding");

pub enum MuseCache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl MuseCache {
    pub(crate) fn offset(&self) -> i32 {
        match self {
            Self::Standard(cache) => cache.offset,
            Self::Rotating(cache) => cache.offset,
        }
    }

    #[cfg(test)]
    pub(crate) fn live_len(&self) -> i32 {
        match self {
            Self::Standard(cache) => cache.live_len(),
            Self::Rotating(cache) => cache.offset.min(cache.snapshot_state().max_size),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_sliding(&self) -> bool {
        matches!(self, Self::Rotating(_))
    }

    pub(crate) fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self {
            Self::Standard(cache) => cache.update_and_fetch(k, v),
            Self::Rotating(cache) => cache.update_and_fetch(k, v),
        }
    }

    pub(crate) fn snapshot_into(
        &self,
        snapshot: &mut ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        match self {
            Self::Standard(cache) => {
                kv_snapshot::snapshot_standard(cache, snapshot, prefix, MUSE_KV_SNAPSHOT_NAMES)
            }
            Self::Rotating(cache) => {
                kv_snapshot::snapshot_rotating(cache, snapshot, prefix, MUSE_KV_SNAPSHOT_NAMES)
            }
        }
    }

    pub(crate) fn restore_from(
        &mut self,
        snapshot: &ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        match self {
            Self::Standard(cache) => {
                kv_snapshot::restore_standard(cache, snapshot, prefix, MUSE_KV_SNAPSHOT_NAMES)
            }
            Self::Rotating(cache) => {
                kv_snapshot::restore_rotating(cache, snapshot, prefix, MUSE_KV_SNAPSHOT_NAMES)
            }
        }
    }
}
