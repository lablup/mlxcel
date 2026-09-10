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

//! Shared exact-prefix snapshot serialization for the three ordinary
//! attention cache types.
//!
//! A model-owned family opts into prompt-cache reuse by implementing
//! `LanguageModel::supports_snapshot_reuse()` plus the four state hooks around
//! it. Each family used to hand-write the per-cache-type tensor and scalar
//! layout; this module owns that layout once, so a new family only has to map
//! its cache enum onto the functions here.
//!
//! Three cache types are covered, and they differ in what a truncating restore
//! is allowed to do:
//!
//! * [`KVCache`] keeps every token at its own slot, so a truncation is always
//!   sound.
//! * [`RotatingKVCache`] keeps only a window, and once the ring has wrapped
//!   the logical token `t` no longer sits at slot `t`. Truncation is allowed
//!   only while [`RotatingKVCacheSnapshotState::can_truncate_to`] holds.
//! * [`ChunkedKVCache`] drops tokens off the front once the visible window
//!   exceeds `chunk_size`. Truncation is allowed only while the front is
//!   untrimmed (`start_position == 0`), because a cold prefill of the shorter
//!   prompt would hold the window `[target_len - chunk_size, target_len)`,
//!   which is wider than the `[start_position, target_len)` a trimmed buffer
//!   can offer.
//!
//! Only [`KVCacheMode::Fp16`] is snapshot-capable. The quantized modes carry
//! sidecar buffers the snapshot container does not model, so both snapshot and
//! restore refuse them and the caller falls back to a cold prefill rather than
//! restoring a mismatched layout.

use crate::models::recurrent_snapshot::{
    kv_cache_mode_from_i32, kv_cache_mode_to_i32, push_i32, push_optional, restore_i32,
    restore_optional,
};
use mlxcel_core::cache::{KVCacheMode, RotatingKVCacheSnapshotState};
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{ChunkedKVCache, KVCache, RotatingKVCache};

/// Tensor-name and error-message vocabulary for one family.
///
/// The tensor names are part of a family's on-the-wire snapshot contract, and
/// two families that shipped before this module existed spell the same two
/// cache types differently: Gemma 4 writes `standard` / `rotating` and Muse
/// Glimmer writes `full` / `sliding`. Parameterizing the names keeps both
/// byte-identical to what they stored before, so their existing snapshot tests
/// hold without edits.
///
/// `chunked` is not parameterized: Llama 4 is its only holder and it has no
/// pre-existing spelling to preserve.
///
/// Used by: Gemma 3, AFMoE, Gemma 4, Llama 4, Muse Glimmer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvSnapshotNames {
    /// Human-readable family label used in error messages, for example
    /// `"Gemma 4"`.
    pub family: &'static str,
    /// Tensor-name segment and error-message word for [`KVCache`] state.
    pub standard: &'static str,
    /// Tensor-name segment and error-message word for [`RotatingKVCache`]
    /// state.
    pub rotating: &'static str,
}

impl KvSnapshotNames {
    pub(crate) const fn new(
        family: &'static str,
        standard: &'static str,
        rotating: &'static str,
    ) -> Self {
        Self {
            family,
            standard,
            rotating,
        }
    }
}

/// Tensor-name segment and error-message word for [`ChunkedKVCache`] state.
const CHUNKED: &str = "chunked";

// ---------------------------------------------------------------------------
// KVCache (full attention)
// ---------------------------------------------------------------------------

/// Copy a full-attention cache into `snapshot` under `prefix`.
///
/// Stores `{prefix}.{standard}.keys` and `.values` as the physical buffers are
/// held (`[B, H_kv, T_alloc, D]`, where `T_alloc >= offset` because the buffer
/// grows in `step`-sized blocks), plus the `.offset` and `.mode` scalars.
///
/// An empty cache stores nothing and is not an error: KV-shared layers never
/// populate their own cache, and `restore_standard` leaves such a layer fresh.
///
/// Used by: Gemma 3, AFMoE, Gemma 4, Llama 4, Muse Glimmer.
pub(crate) fn snapshot_standard(
    cache: &KVCache,
    snapshot: &mut ModelStateSnapshot,
    prefix: &str,
    names: KvSnapshotNames,
) -> Result<(), String> {
    let (family, kind) = (names.family, names.standard);
    if cache.keys.is_none() && cache.values.is_none() {
        return Ok(());
    }
    if cache.keys.is_some() != cache.values.is_some() {
        return Err(format!(
            "{family} snapshot {prefix}: {kind} cache has only one of keys/values"
        ));
    }
    if cache.mode != KVCacheMode::Fp16 {
        return Err(format!(
            "{family} snapshot {prefix}: {kind} cache mode {:?} is not supported by model-state snapshots",
            cache.mode
        ));
    }
    push_optional(snapshot, format!("{prefix}.{kind}.keys"), &cache.keys);
    push_optional(snapshot, format!("{prefix}.{kind}.values"), &cache.values);
    push_i32(snapshot, format!("{prefix}.{kind}.offset"), cache.offset);
    push_i32(
        snapshot,
        format!("{prefix}.{kind}.mode"),
        kv_cache_mode_to_i32(cache.mode),
    );
    Ok(())
}

/// Restore a full-attention cache from `snapshot` under `prefix`.
///
/// Assigns `keys`, `values` and `offset`. Both the stored mode and the live
/// cache's configured mode must be [`KVCacheMode::Fp16`]; a mismatch is an
/// error rather than a silent reinterpretation of a quantized buffer.
///
/// Used by: Gemma 3, AFMoE, Gemma 4, Llama 4, Muse Glimmer.
pub(crate) fn restore_standard(
    cache: &mut KVCache,
    snapshot: &ModelStateSnapshot,
    prefix: &str,
    names: KvSnapshotNames,
) -> Result<(), String> {
    let (family, kind) = (names.family, names.standard);
    let keys = restore_optional(snapshot, format!("{prefix}.{kind}.keys"));
    let values = restore_optional(snapshot, format!("{prefix}.{kind}.values"));
    if keys.is_none() && values.is_none() {
        return Ok(());
    }
    if keys.is_some() != values.is_some() {
        return Err(format!(
            "{family} restore {prefix}: {kind} snapshot has only one of keys/values"
        ));
    }
    let mode = snapshot_mode(snapshot, &format!("{prefix}.{kind}.mode"))?;
    if mode != KVCacheMode::Fp16 {
        return Err(format!(
            "{family} restore {prefix}: {kind} snapshot mode {mode:?} is not supported"
        ));
    }
    if cache.mode != mode {
        return Err(format!(
            "{family} restore {prefix}: {kind} snapshot mode {mode:?} does not match configured cache mode {:?}",
            cache.mode
        ));
    }
    cache.keys = keys;
    cache.values = values;
    cache.offset = restore_i32(snapshot, format!("{prefix}.{kind}.offset"))
        .unwrap_or(snapshot.token_len() as i32);
    Ok(())
}

/// Whether the full-attention state stored under `prefix` can be restored
/// covering only its first `target_len` tokens.
///
/// Every token sits at its own slot, so the only constraints are the mode and
/// `target_len <= offset`. A prefix with no stored tensors is vacuously
/// truncatable: the layer captured nothing, `restore_standard` leaves a fresh
/// cache, and there is no tail to drop.
///
/// Used by: Gemma 3, AFMoE, Gemma 4, Llama 4.
pub(crate) fn standard_truncatable_to(
    snapshot: &ModelStateSnapshot,
    prefix: &str,
    target_len: i32,
    names: KvSnapshotNames,
) -> bool {
    let kind = names.standard;
    if target_len < 0 {
        return false;
    }
    if snapshot.tensor(&format!("{prefix}.{kind}.keys")).is_none() {
        return true;
    }
    let Ok(mode) = snapshot_mode(snapshot, &format!("{prefix}.{kind}.mode")) else {
        return false;
    };
    if mode != KVCacheMode::Fp16 {
        return false;
    }
    let Some(offset) = restore_i32(snapshot, format!("{prefix}.{kind}.offset")) else {
        return false;
    };
    target_len <= offset
}

/// Drop everything past `target_len` tokens from a full-attention cache.
///
/// Rewinds through [`KVCache::trim`], which is a logical rewind: the physical
/// buffer keeps its capacity and the next update overwrites the rewound tail
/// in place.
///
/// Used by: Gemma 3, AFMoE, Gemma 4, Llama 4.
pub(crate) fn truncate_standard(
    cache: &mut KVCache,
    target_len: i32,
    names: KvSnapshotNames,
) -> Result<(), String> {
    if cache.keys.is_none() && cache.values.is_none() {
        return Ok(());
    }
    let offset = cache.offset;
    check_truncate_target(target_len, offset, names)?;
    let drop = offset - target_len;
    if drop == 0 {
        return Ok(());
    }
    check_trimmed(cache.trim(drop), drop, names)
}

// ---------------------------------------------------------------------------
// RotatingKVCache (sliding window)
// ---------------------------------------------------------------------------

/// Copy a sliding-window cache into `snapshot` under `prefix`.
///
/// Stores `{prefix}.{rotating}.keys` and `.values` as the physical ring buffer
/// is held, plus every scalar
/// [`RotatingKVCache::restore_fp16_snapshot_state`] needs to reproduce the ring
/// geometry: `.max_size`, `.buffer_size`, `.offset`, `.start_position`, `.idx`,
/// `.step`, `.mode` and `.turbo_seed`.
///
/// Used by: Gemma 3, AFMoE, Gemma 4, Muse Glimmer.
pub(crate) fn snapshot_rotating(
    cache: &RotatingKVCache,
    snapshot: &mut ModelStateSnapshot,
    prefix: &str,
    names: KvSnapshotNames,
) -> Result<(), String> {
    let (family, kind) = (names.family, names.rotating);
    if cache.keys.is_none() && cache.values.is_none() {
        return Ok(());
    }
    if cache.keys.is_some() != cache.values.is_some() {
        return Err(format!(
            "{family} snapshot {prefix}: {kind} cache has only one of keys/values"
        ));
    }
    let state = cache.snapshot_state();
    if state.mode != KVCacheMode::Fp16 {
        return Err(format!(
            "{family} snapshot {prefix}: {kind} cache mode {:?} is not supported by model-state snapshots",
            state.mode
        ));
    }
    push_optional(snapshot, format!("{prefix}.{kind}.keys"), &cache.keys);
    push_optional(snapshot, format!("{prefix}.{kind}.values"), &cache.values);
    push_i32(
        snapshot,
        format!("{prefix}.{kind}.max_size"),
        state.max_size,
    );
    push_i32(
        snapshot,
        format!("{prefix}.{kind}.buffer_size"),
        state.buffer_size,
    );
    push_i32(snapshot, format!("{prefix}.{kind}.offset"), state.offset);
    push_i32(
        snapshot,
        format!("{prefix}.{kind}.start_position"),
        state.start_position,
    );
    push_i32(snapshot, format!("{prefix}.{kind}.idx"), state.idx);
    push_i32(snapshot, format!("{prefix}.{kind}.step"), state.step);
    push_i32(
        snapshot,
        format!("{prefix}.{kind}.mode"),
        kv_cache_mode_to_i32(state.mode),
    );
    push_i32(
        snapshot,
        format!("{prefix}.{kind}.turbo_seed"),
        state.turbo_seed as i32,
    );
    Ok(())
}

/// Restore a sliding-window cache from `snapshot` under `prefix`.
///
/// Every scalar falls back to the live cache's own value when absent, so a
/// snapshot written by an earlier build that stored fewer fields still
/// restores. The assignment itself goes through
/// [`RotatingKVCache::restore_fp16_snapshot_state`], which validates the ring
/// geometry against the copied buffers.
///
/// Used by: Gemma 3, AFMoE, Gemma 4, Muse Glimmer.
pub(crate) fn restore_rotating(
    cache: &mut RotatingKVCache,
    snapshot: &ModelStateSnapshot,
    prefix: &str,
    names: KvSnapshotNames,
) -> Result<(), String> {
    let (family, kind) = (names.family, names.rotating);
    let keys = restore_optional(snapshot, format!("{prefix}.{kind}.keys"));
    let values = restore_optional(snapshot, format!("{prefix}.{kind}.values"));
    if keys.is_none() && values.is_none() {
        return Ok(());
    }
    if keys.is_some() != values.is_some() {
        return Err(format!(
            "{family} restore {prefix}: {kind} snapshot has only one of keys/values"
        ));
    }
    let current = cache.snapshot_state();
    let mode = snapshot_mode(snapshot, &format!("{prefix}.{kind}.mode"))?;
    if mode != KVCacheMode::Fp16 {
        return Err(format!(
            "{family} restore {prefix}: {kind} snapshot mode {mode:?} is not supported"
        ));
    }
    if current.mode != mode {
        return Err(format!(
            "{family} restore {prefix}: {kind} snapshot mode {mode:?} does not match configured cache mode {:?}",
            current.mode
        ));
    }
    let state = RotatingKVCacheSnapshotState {
        max_size: restore_i32(snapshot, format!("{prefix}.{kind}.max_size"))
            .unwrap_or(current.max_size),
        buffer_size: restore_i32(snapshot, format!("{prefix}.{kind}.buffer_size")).unwrap_or(0),
        offset: restore_i32(snapshot, format!("{prefix}.{kind}.offset"))
            .unwrap_or(snapshot.token_len() as i32),
        start_position: restore_i32(snapshot, format!("{prefix}.{kind}.start_position"))
            .unwrap_or(0),
        idx: restore_i32(snapshot, format!("{prefix}.{kind}.idx"))
            .unwrap_or(snapshot.token_len() as i32),
        step: restore_i32(snapshot, format!("{prefix}.{kind}.step")).unwrap_or(current.step),
        mode,
        turbo_seed: restore_i32(snapshot, format!("{prefix}.{kind}.turbo_seed"))
            .map(|seed| seed as u32)
            .unwrap_or(current.turbo_seed),
    };
    cache.restore_fp16_snapshot_state(state, keys, values)
}

/// Whether the sliding-window state stored under `prefix` can be restored
/// covering only its first `target_len` tokens.
///
/// Defers to [`RotatingKVCacheSnapshotState::can_truncate_to`], which is the
/// load-bearing check: it holds only while the ring has not wrapped, because a
/// wrapped ring no longer keeps logical token `t` at slot `t`.
///
/// Used by: Gemma 3, AFMoE, Gemma 4.
pub(crate) fn rotating_truncatable_to(
    snapshot: &ModelStateSnapshot,
    prefix: &str,
    target_len: i32,
    names: KvSnapshotNames,
) -> bool {
    let kind = names.rotating;
    if target_len < 0 {
        return false;
    }
    if snapshot.tensor(&format!("{prefix}.{kind}.keys")).is_none() {
        return true;
    }
    let Ok(mode) = snapshot_mode(snapshot, &format!("{prefix}.{kind}.mode")) else {
        return false;
    };
    let Some(max_size) = restore_i32(snapshot, format!("{prefix}.{kind}.max_size")) else {
        return false;
    };
    let Some(offset) = restore_i32(snapshot, format!("{prefix}.{kind}.offset")) else {
        return false;
    };
    let Some(idx) = restore_i32(snapshot, format!("{prefix}.{kind}.idx")) else {
        return false;
    };
    let state = RotatingKVCacheSnapshotState {
        max_size,
        buffer_size: restore_i32(snapshot, format!("{prefix}.{kind}.buffer_size")).unwrap_or(0),
        offset,
        start_position: restore_i32(snapshot, format!("{prefix}.{kind}.start_position"))
            .unwrap_or(0),
        idx,
        step: 256,
        mode,
        turbo_seed: 0,
    };
    state.can_truncate_to(target_len)
}

/// Drop everything past `target_len` tokens from a sliding-window cache.
///
/// Refuses a wrapped ring outright: [`RotatingKVCache::trim`] only rewinds
/// `offset` and `idx`, which is sound while slot and logical position still
/// agree and silently wrong once they do not.
///
/// Used by: Gemma 3, AFMoE, Gemma 4.
pub(crate) fn truncate_rotating(
    cache: &mut RotatingKVCache,
    target_len: i32,
    names: KvSnapshotNames,
) -> Result<(), String> {
    if cache.keys.is_none() && cache.values.is_none() {
        return Ok(());
    }
    let (family, kind) = (names.family, names.rotating);
    let offset = cache.offset;
    check_truncate_target(target_len, offset, names)?;
    if !cache.is_trimmable() {
        return Err(format!(
            "{family} truncate: {kind} cache has wrapped and cannot be trimmed"
        ));
    }
    let drop = offset - target_len;
    if drop == 0 {
        return Ok(());
    }
    check_trimmed(cache.trim(drop), drop, names)
}

// ---------------------------------------------------------------------------
// ChunkedKVCache (Llama 4 iGQA)
// ---------------------------------------------------------------------------

/// Copy a chunked cache into `snapshot` under `prefix`.
///
/// Stores `{prefix}.chunked.keys` and `.values` as the physical buffer is held
/// (`[B, H_kv, T_alloc, D]`, where `T_alloc >= offset - start_position`), plus
/// the `.chunk_size`, `.offset` and `.start_position` scalars. There is no mode
/// scalar: [`ChunkedKVCache`] has no quantized variant.
///
/// The private `step` field is not stored. It only controls how the buffer
/// grows from here and stays at its constructor default on the restored cache.
///
/// Used by: Llama 4.
pub(crate) fn snapshot_chunked(
    cache: &ChunkedKVCache,
    snapshot: &mut ModelStateSnapshot,
    prefix: &str,
    names: KvSnapshotNames,
) -> Result<(), String> {
    let family = names.family;
    if cache.keys.is_none() && cache.values.is_none() {
        return Ok(());
    }
    if cache.keys.is_some() != cache.values.is_some() {
        return Err(format!(
            "{family} snapshot {prefix}: {CHUNKED} cache has only one of keys/values"
        ));
    }
    push_optional(snapshot, format!("{prefix}.{CHUNKED}.keys"), &cache.keys);
    push_optional(
        snapshot,
        format!("{prefix}.{CHUNKED}.values"),
        &cache.values,
    );
    push_i32(
        snapshot,
        format!("{prefix}.{CHUNKED}.chunk_size"),
        cache.chunk_size,
    );
    push_i32(snapshot, format!("{prefix}.{CHUNKED}.offset"), cache.offset);
    push_i32(
        snapshot,
        format!("{prefix}.{CHUNKED}.start_position"),
        cache.start_position,
    );
    Ok(())
}

/// Restore a chunked cache from `snapshot` under `prefix`.
///
/// A stored `chunk_size` that disagrees with the live cache means the snapshot
/// came from a different model configuration, so the restore is refused rather
/// than installed against a window of the wrong width.
///
/// Used by: Llama 4.
pub(crate) fn restore_chunked(
    cache: &mut ChunkedKVCache,
    snapshot: &ModelStateSnapshot,
    prefix: &str,
    names: KvSnapshotNames,
) -> Result<(), String> {
    let family = names.family;
    let keys = restore_optional(snapshot, format!("{prefix}.{CHUNKED}.keys"));
    let values = restore_optional(snapshot, format!("{prefix}.{CHUNKED}.values"));
    if keys.is_none() && values.is_none() {
        return Ok(());
    }
    if keys.is_some() != values.is_some() {
        return Err(format!(
            "{family} restore {prefix}: {CHUNKED} snapshot has only one of keys/values"
        ));
    }
    let chunk_size =
        restore_i32(snapshot, format!("{prefix}.{CHUNKED}.chunk_size")).unwrap_or(cache.chunk_size);
    if chunk_size != cache.chunk_size {
        return Err(format!(
            "{family} restore {prefix}: {CHUNKED} snapshot chunk size {chunk_size} does not match configured chunk size {}",
            cache.chunk_size
        ));
    }
    cache.keys = keys;
    cache.values = values;
    cache.offset = restore_i32(snapshot, format!("{prefix}.{CHUNKED}.offset"))
        .unwrap_or(snapshot.token_len() as i32);
    cache.start_position =
        restore_i32(snapshot, format!("{prefix}.{CHUNKED}.start_position")).unwrap_or(0);
    Ok(())
}

/// Whether the chunked state stored under `prefix` can be restored covering
/// only its first `target_len` tokens.
///
/// Allowed only while the front is untrimmed. Once `start_position > 0` the
/// buffer holds `[start_position, offset)`, and a cold prefill of `target_len`
/// tokens would instead hold `[target_len - chunk_size, target_len)`. Whenever
/// `start_position > target_len - chunk_size` the truncated restore attends
/// over strictly fewer tokens than the cold run, so it is not exact.
///
/// Used by: Llama 4.
pub(crate) fn chunked_truncatable_to(
    snapshot: &ModelStateSnapshot,
    prefix: &str,
    target_len: i32,
) -> bool {
    if target_len < 0 {
        return false;
    }
    if snapshot
        .tensor(&format!("{prefix}.{CHUNKED}.keys"))
        .is_none()
    {
        return true;
    }
    let Some(start_position) = restore_i32(snapshot, format!("{prefix}.{CHUNKED}.start_position"))
    else {
        return false;
    };
    if start_position != 0 {
        return false;
    }
    let Some(offset) = restore_i32(snapshot, format!("{prefix}.{CHUNKED}.offset")) else {
        return false;
    };
    target_len <= offset
}

/// Drop everything past `target_len` tokens from a chunked cache.
///
/// Unlike the other two types this is a physical trim: [`ChunkedKVCache`] has
/// no `trim`, and its `update_and_fetch` reads the buffer length back through
/// `get_buffer_size`, so a merely logical rewind would leave the abandoned tail
/// visible to the next growth decision.
///
/// Used by: Llama 4.
pub(crate) fn truncate_chunked(
    cache: &mut ChunkedKVCache,
    target_len: i32,
    names: KvSnapshotNames,
) -> Result<(), String> {
    if cache.keys.is_none() && cache.values.is_none() {
        return Ok(());
    }
    let family = names.family;
    let offset = cache.offset;
    check_truncate_target(target_len, offset, names)?;
    if cache.start_position != 0 {
        return Err(format!(
            "{family} truncate: {CHUNKED} cache has trimmed its front and cannot be truncated"
        ));
    }
    if target_len == offset {
        return Ok(());
    }
    cache.keys = Some(slice_leading_tokens(
        cache.keys.as_ref(),
        target_len,
        family,
        "keys",
    )?);
    cache.values = Some(slice_leading_tokens(
        cache.values.as_ref(),
        target_len,
        family,
        "values",
    )?);
    cache.offset = target_len;
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared internals
// ---------------------------------------------------------------------------

/// Read a stored `KVCacheMode` tag, defaulting to `Fp16` when the scalar is
/// absent (snapshots written before the tag existed were always FP16).
fn snapshot_mode(snapshot: &ModelStateSnapshot, name: &str) -> Result<KVCacheMode, String> {
    match restore_i32(snapshot, name) {
        Some(tag) => kv_cache_mode_from_i32(tag),
        None => Ok(KVCacheMode::Fp16),
    }
}

fn check_truncate_target(
    target_len: i32,
    offset: i32,
    names: KvSnapshotNames,
) -> Result<(), String> {
    if target_len > offset {
        return Err(format!(
            "{} truncate: target {target_len} exceeds cached offset {offset}",
            names.family
        ));
    }
    Ok(())
}

fn check_trimmed(trimmed: i32, drop: i32, names: KvSnapshotNames) -> Result<(), String> {
    if trimmed != drop {
        return Err(format!(
            "{} truncate: trimmed {trimmed} of {drop} requested tokens",
            names.family
        ));
    }
    Ok(())
}

/// Slice the leading `target_len` positions off the sequence axis of a cache
/// buffer.
fn slice_leading_tokens(
    array: Option<&mlxcel_core::UniquePtr<mlxcel_core::MlxArray>>,
    target_len: i32,
    family: &str,
    field: &str,
) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>, String> {
    let array = array.and_then(|a| a.as_ref()).ok_or_else(|| {
        format!("{family} truncate: {CHUNKED} cache is missing its {field} buffer")
    })?;
    let shape = mlxcel_core::array_shape(array);
    if shape.len() != 4 {
        return Err(format!(
            "{family} truncate: expected rank-4 {CHUNKED} {field}, got shape {shape:?}"
        ));
    }
    if target_len > shape[2] {
        return Err(format!(
            "{family} truncate: target {target_len} exceeds {CHUNKED} {field} length {}",
            shape[2]
        ));
    }
    Ok(mlxcel_core::slice(
        array,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], target_len, shape[3]],
    ))
}

#[cfg(test)]
#[path = "kv_snapshot_tests.rs"]
mod tests;
