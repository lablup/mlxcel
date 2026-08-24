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

//! Small helpers for exact-prefix recurrent-state snapshots.
//!
//! The snapshot container itself lives in `mlxcel-core` so it can be exposed
//! through the `LanguageModel` trait. These helpers keep the model files from
//! repeating the same optional-tensor copy/restore boilerplate.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) fn push_i32(snapshot: &mut ModelStateSnapshot, name: impl Into<String>, value: i32) {
    let scalar = mlxcel_core::from_slice_i32(&[value], &[1]);
    snapshot.push_tensor(
        name,
        scalar.as_ref().expect("from_slice_i32 returns an array"),
    );
}

pub(crate) fn push_optional(
    snapshot: &mut ModelStateSnapshot,
    name: impl Into<String>,
    array: &Option<UniquePtr<MlxArray>>,
) {
    if let Some(array) = array.as_ref().and_then(|a| a.as_ref()) {
        snapshot.push_tensor(name, array);
    }
}

pub(crate) fn restore_optional(
    snapshot: &ModelStateSnapshot,
    name: impl AsRef<str>,
) -> Option<UniquePtr<MlxArray>> {
    snapshot.tensor(name.as_ref()).map(mlxcel_core::copy)
}

pub(crate) fn restore_i32(snapshot: &ModelStateSnapshot, name: impl AsRef<str>) -> Option<i32> {
    snapshot.tensor(name.as_ref()).map(mlxcel_core::item_i32)
}

pub(crate) fn push_kv_cache_mode(
    snapshot: &mut ModelStateSnapshot,
    name: impl Into<String>,
    mode: KVCacheMode,
) {
    push_i32(snapshot, name, kv_cache_mode_to_i32(mode));
}

pub(crate) fn restore_kv_cache_mode(
    snapshot: &ModelStateSnapshot,
    name: impl AsRef<str>,
) -> Result<Option<KVCacheMode>, String> {
    let Some(tag) = restore_i32(snapshot, name.as_ref()) else {
        return Ok(None);
    };
    kv_cache_mode_from_i32(tag).map(Some)
}

fn kv_cache_mode_to_i32(mode: KVCacheMode) -> i32 {
    match mode {
        KVCacheMode::Fp16 => 0,
        KVCacheMode::Int8 => 1,
        KVCacheMode::Turbo4Asym => 2,
        KVCacheMode::Turbo3Asym => 3,
        KVCacheMode::Turbo4 => 4,
        KVCacheMode::Turbo4Delegated => 5,
    }
}

fn kv_cache_mode_from_i32(tag: i32) -> Result<KVCacheMode, String> {
    match tag {
        0 => Ok(KVCacheMode::Fp16),
        1 => Ok(KVCacheMode::Int8),
        2 => Ok(KVCacheMode::Turbo4Asym),
        3 => Ok(KVCacheMode::Turbo3Asym),
        4 => Ok(KVCacheMode::Turbo4),
        5 => Ok(KVCacheMode::Turbo4Delegated),
        other => Err(format!("unknown serialized KV cache mode tag {other}")),
    }
}
