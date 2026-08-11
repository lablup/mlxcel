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

use crate::models::recurrent_snapshot::{push_i32, push_optional, restore_i32, restore_optional};
use mlxcel_core::cache::{KVCacheMode, RotatingKVCacheSnapshotState};
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{KVCache, RotatingKVCache};
use mlxcel_core::{MlxArray, UniquePtr};

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
            Self::Standard(cache) => snapshot_standard(cache, snapshot, prefix),
            Self::Rotating(cache) => snapshot_rotating(cache, snapshot, prefix),
        }
    }

    pub(crate) fn restore_from(
        &mut self,
        snapshot: &ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        match self {
            Self::Standard(cache) => restore_standard(cache, snapshot, prefix),
            Self::Rotating(cache) => restore_rotating(cache, snapshot, prefix),
        }
    }
}

fn snapshot_standard(
    cache: &KVCache,
    snapshot: &mut ModelStateSnapshot,
    prefix: &str,
) -> Result<(), String> {
    if cache.keys.is_none() && cache.values.is_none() {
        return Ok(());
    }
    if cache.keys.is_some() != cache.values.is_some() {
        return Err(format!(
            "Muse Glimmer snapshot {prefix}: full cache has only one of keys/values"
        ));
    }
    if cache.mode != KVCacheMode::Fp16 {
        return Err(format!(
            "Muse Glimmer snapshot {prefix}: full cache mode {:?} is not supported by model-state snapshots",
            cache.mode
        ));
    }
    push_optional(snapshot, format!("{prefix}.full.keys"), &cache.keys);
    push_optional(snapshot, format!("{prefix}.full.values"), &cache.values);
    push_i32(snapshot, format!("{prefix}.full.offset"), cache.offset);
    push_i32(snapshot, format!("{prefix}.full.mode"), 0);
    Ok(())
}

fn snapshot_rotating(
    cache: &RotatingKVCache,
    snapshot: &mut ModelStateSnapshot,
    prefix: &str,
) -> Result<(), String> {
    if cache.keys.is_none() && cache.values.is_none() {
        return Ok(());
    }
    if cache.keys.is_some() != cache.values.is_some() {
        return Err(format!(
            "Muse Glimmer snapshot {prefix}: sliding cache has only one of keys/values"
        ));
    }
    let state = cache.snapshot_state();
    if state.mode != KVCacheMode::Fp16 {
        return Err(format!(
            "Muse Glimmer snapshot {prefix}: sliding cache mode {:?} is not supported by model-state snapshots",
            state.mode
        ));
    }
    push_optional(snapshot, format!("{prefix}.sliding.keys"), &cache.keys);
    push_optional(snapshot, format!("{prefix}.sliding.values"), &cache.values);
    push_i32(
        snapshot,
        format!("{prefix}.sliding.max_size"),
        state.max_size,
    );
    push_i32(
        snapshot,
        format!("{prefix}.sliding.buffer_size"),
        state.buffer_size,
    );
    push_i32(snapshot, format!("{prefix}.sliding.offset"), state.offset);
    push_i32(
        snapshot,
        format!("{prefix}.sliding.start_position"),
        state.start_position,
    );
    push_i32(snapshot, format!("{prefix}.sliding.idx"), state.idx);
    push_i32(snapshot, format!("{prefix}.sliding.step"), state.step);
    push_i32(snapshot, format!("{prefix}.sliding.mode"), 0);
    Ok(())
}

fn restore_standard(
    cache: &mut KVCache,
    snapshot: &ModelStateSnapshot,
    prefix: &str,
) -> Result<(), String> {
    let keys = restore_optional(snapshot, format!("{prefix}.full.keys"));
    let values = restore_optional(snapshot, format!("{prefix}.full.values"));
    if keys.is_none() && values.is_none() {
        return Ok(());
    }
    if keys.is_some() != values.is_some() {
        return Err(format!(
            "Muse Glimmer restore {prefix}: full snapshot has only one of keys/values"
        ));
    }
    validate_fp16_mode(snapshot, format!("{prefix}.full.mode"))?;
    cache.keys = keys;
    cache.values = values;
    cache.offset = restore_i32(snapshot, format!("{prefix}.full.offset"))
        .unwrap_or(snapshot.token_len() as i32);
    cache.mode = KVCacheMode::Fp16;
    Ok(())
}

fn restore_rotating(
    cache: &mut RotatingKVCache,
    snapshot: &ModelStateSnapshot,
    prefix: &str,
) -> Result<(), String> {
    let keys = restore_optional(snapshot, format!("{prefix}.sliding.keys"));
    let values = restore_optional(snapshot, format!("{prefix}.sliding.values"));
    if keys.is_none() && values.is_none() {
        return Ok(());
    }
    if keys.is_some() != values.is_some() {
        return Err(format!(
            "Muse Glimmer restore {prefix}: sliding snapshot has only one of keys/values"
        ));
    }
    validate_fp16_mode(snapshot, format!("{prefix}.sliding.mode"))?;
    let current = cache.snapshot_state();
    let state = RotatingKVCacheSnapshotState {
        max_size: restore_i32(snapshot, format!("{prefix}.sliding.max_size"))
            .unwrap_or(current.max_size),
        buffer_size: restore_i32(snapshot, format!("{prefix}.sliding.buffer_size")).unwrap_or(0),
        offset: restore_i32(snapshot, format!("{prefix}.sliding.offset"))
            .unwrap_or(snapshot.token_len() as i32),
        start_position: restore_i32(snapshot, format!("{prefix}.sliding.start_position"))
            .unwrap_or(0),
        idx: restore_i32(snapshot, format!("{prefix}.sliding.idx"))
            .unwrap_or(snapshot.token_len() as i32),
        step: restore_i32(snapshot, format!("{prefix}.sliding.step")).unwrap_or(current.step),
        mode: KVCacheMode::Fp16,
        turbo_seed: current.turbo_seed,
    };
    cache.restore_fp16_snapshot_state(state, keys, values)
}

fn validate_fp16_mode(snapshot: &ModelStateSnapshot, name: String) -> Result<(), String> {
    match restore_i32(snapshot, &name).unwrap_or(0) {
        0 => Ok(()),
        value => Err(format!(
            "Muse Glimmer restore {name}: snapshot cache mode {value} is not supported"
        )),
    }
}
