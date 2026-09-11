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

//! Round-trip and truncation-rule tests for the shared KV snapshot
//! serializers (issue #1335).
//!
//! Every assertion is made against what the cache hands back to attention on
//! the next `update_and_fetch`, not only against the scalars, because the
//! failure these tests exist to catch is a restore that reproduces the
//! bookkeeping while pointing at the wrong slots.

use super::*;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{ChunkedKVCache, KVCache, RotatingKVCache};
use mlxcel_core::{MlxArray, UniquePtr};

const H: i32 = 2;
const D: i32 = 4;

const TEST_NAMES: KvSnapshotNames = KvSnapshotNames::new("Test", "standard", "rotating");
/// The Muse Glimmer vocabulary, used here only to pin that the names really do
/// select the stored tensor names.
const ALT_NAMES: KvSnapshotNames = KvSnapshotNames::new("Muse Glimmer", "full", "sliding");

/// Build `n` tokens of K or V starting at absolute position `start`.
///
/// The value at `[0, h, t, d]` depends on the absolute position `start + t`
/// and never on how the tokens were chunked, so the first four tokens of a
/// six-token push are bitwise equal to a four-token push. A naive
/// `base + linear_index` fill does not have that property, because the head
/// stride changes with `n`, and every cold-versus-restored comparison below
/// would then be comparing different data.
fn make_kv(start: i32, n: i32, base: f32) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity((H * n * D) as usize);
    for h in 0..H {
        for t in 0..n {
            for d in 0..D {
                data.push(base + (h as f32) * 1000.0 + ((start + t) as f32) * 10.0 + (d as f32));
            }
        }
    }
    mlxcel_core::from_slice_f32(&data, &[1, H, n, D])
}

fn keys_at(start: i32, n: i32) -> UniquePtr<MlxArray> {
    make_kv(start, n, 0.0)
}

fn values_at(start: i32, n: i32) -> UniquePtr<MlxArray> {
    make_kv(start, n, 100_000.0)
}

fn assert_arrays_equal(actual: &MlxArray, expected: &MlxArray, what: &str) {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected),
        "{what}: shape mismatch"
    );
    let equal = mlxcel_core::array_equal(actual, expected, false);
    assert!(mlxcel_core::item_bool(&equal), "{what}: values differ");
}

fn fill_standard(cache: &mut KVCache, n: i32) {
    let _ = cache.update_and_fetch(keys_at(0, n), values_at(0, n));
}

fn fill_rotating_one_at_a_time(cache: &mut RotatingKVCache, n: i32) {
    for t in 0..n {
        let _ = cache.update_and_fetch(keys_at(t, 1), values_at(t, 1));
    }
}

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

#[test]
fn standard_round_trip_preserves_keys_values_offset() {
    let mut source = KVCache::new();
    fill_standard(&mut source, 37);
    assert_eq!(source.offset, 37);

    let mut snapshot = ModelStateSnapshot::new("test", 37);
    snapshot_standard(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = KVCache::new();
    restore_standard(&mut restored, &snapshot, "layer0", TEST_NAMES).expect("restore");
    assert_eq!(restored.offset, 37, "restored offset");

    let (source_k, source_v) = source.update_and_fetch(keys_at(37, 1), values_at(37, 1));
    let (restored_k, restored_v) = restored.update_and_fetch(keys_at(37, 1), values_at(37, 1));
    assert_eq!(
        mlxcel_core::array_shape(&restored_k)[2],
        38,
        "restored cache must return all 38 tokens"
    );
    assert_arrays_equal(&restored_k, &source_k, "keys after restore");
    assert_arrays_equal(&restored_v, &source_v, "values after restore");
}

#[test]
fn standard_truncated_restore_matches_a_cold_fill_of_the_same_prefix() {
    let mut source = KVCache::new();
    fill_standard(&mut source, 12);
    let mut snapshot = ModelStateSnapshot::new("test", 12);
    snapshot_standard(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");
    assert!(standard_truncatable_to(&snapshot, "layer0", 5, TEST_NAMES));

    let mut restored = KVCache::new();
    restore_standard(&mut restored, &snapshot, "layer0", TEST_NAMES).expect("restore");
    truncate_standard(&mut restored, 5, TEST_NAMES).expect("truncate");
    assert_eq!(restored.offset, 5);

    let mut cold = KVCache::new();
    fill_standard(&mut cold, 5);

    let (cold_k, cold_v) = cold.update_and_fetch(keys_at(5, 1), values_at(5, 1));
    let (warm_k, warm_v) = restored.update_and_fetch(keys_at(5, 1), values_at(5, 1));
    assert_arrays_equal(&warm_k, &cold_k, "keys after truncated restore");
    assert_arrays_equal(&warm_v, &cold_v, "values after truncated restore");
}

#[test]
fn standard_truncate_rejects_a_target_past_the_cached_offset() {
    let mut cache = KVCache::new();
    fill_standard(&mut cache, 6);
    let err = truncate_standard(&mut cache, 9, TEST_NAMES).expect_err("target past offset");
    assert!(
        err.contains("exceeds cached offset"),
        "unexpected error: {err}"
    );
}

#[test]
fn an_empty_cache_snapshots_nothing_and_truncates_to_a_no_op() {
    let source = KVCache::new();
    let mut snapshot = ModelStateSnapshot::new("test", 0);
    snapshot_standard(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");
    assert!(snapshot.is_empty(), "an empty cache must store nothing");
    // A KV-shared layer stores nothing and must not block a truncating restore.
    assert!(standard_truncatable_to(&snapshot, "layer0", 4, TEST_NAMES));
    assert!(rotating_truncatable_to(&snapshot, "layer0", 4, TEST_NAMES));
    assert!(chunked_truncatable_to(&snapshot, "layer0", 4));

    let mut restored = KVCache::new();
    restore_standard(&mut restored, &snapshot, "layer0", TEST_NAMES).expect("restore");
    assert_eq!(restored.offset, 0);
    truncate_standard(&mut restored, 4, TEST_NAMES).expect("empty truncate is a no-op");
}

// ---------------------------------------------------------------------------
// RotatingKVCache
// ---------------------------------------------------------------------------

#[test]
fn rotating_round_trip_preserves_ring_state_after_wrap() {
    let mut source = RotatingKVCache::new(16);
    fill_rotating_one_at_a_time(&mut source, 40);
    let state = source.snapshot_state();
    assert_eq!(state.offset, 40);
    assert!(
        !source.is_trimmable(),
        "40 tokens through a 16-token window must have wrapped"
    );

    let mut snapshot = ModelStateSnapshot::new("test", 40);
    snapshot_rotating(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = RotatingKVCache::new(16);
    restore_rotating(&mut restored, &snapshot, "layer0", TEST_NAMES).expect("restore");
    let restored_state = restored.snapshot_state();
    assert_eq!(restored_state.offset, state.offset, "offset");
    assert_eq!(restored_state.idx, state.idx, "ring write index");
    assert_eq!(restored_state.max_size, state.max_size, "max_size");

    let (source_k, source_v) = source.update_and_fetch(keys_at(40, 1), values_at(40, 1));
    let (restored_k, restored_v) = restored.update_and_fetch(keys_at(40, 1), values_at(40, 1));
    assert_eq!(
        mlxcel_core::array_shape(&restored_k)[2],
        16,
        "a wrapped ring returns exactly one window"
    );
    assert_arrays_equal(&restored_k, &source_k, "keys after wrapped restore");
    assert_arrays_equal(&restored_v, &source_v, "values after wrapped restore");
}

#[test]
fn rotating_round_trip_survives_an_over_window_prefill() {
    // A prompt longer than the window in one push is the shape a first turn
    // actually has, and `update_concat` answers it by storing all 20 tokens
    // and pinning `idx` to that length rather than to `max_size`. The state is
    // temporary (the next single-token step re-slices from the back) but a
    // snapshot taken in between has to survive it: this is the state that made
    // `restore_fp16_snapshot_state` refuse the whole donation (#1335).
    let mut source = RotatingKVCache::new(8);
    let _ = source.update_and_fetch(keys_at(0, 20), values_at(0, 20));
    let state = source.snapshot_state();
    assert_eq!(state.offset, 20);
    assert_eq!(
        state.idx, 20,
        "an over-window prefill pins idx to the stored length, not to max_size"
    );

    let mut snapshot = ModelStateSnapshot::new("test", 20);
    snapshot_rotating(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = RotatingKVCache::new(8);
    restore_rotating(&mut restored, &snapshot, "layer0", TEST_NAMES)
        .expect("an over-window prefill must round-trip");
    let restored_state = restored.snapshot_state();
    assert_eq!(restored_state.idx, state.idx, "ring write index");
    assert_eq!(restored_state.offset, state.offset, "offset");

    assert!(
        !rotating_truncatable_to(&snapshot, "layer0", 12, TEST_NAMES),
        "an over-window buffer is not truncatable even though idx == offset"
    );

    let (source_k, source_v) = source.update_and_fetch(keys_at(20, 1), values_at(20, 1));
    let (restored_k, restored_v) = restored.update_and_fetch(keys_at(20, 1), values_at(20, 1));
    assert_eq!(
        mlxcel_core::array_shape(&restored_k)[2],
        8,
        "the step after an over-window prefill re-slices back to one window"
    );
    assert_arrays_equal(&restored_k, &source_k, "keys after over-window restore");
    assert_arrays_equal(&restored_v, &source_v, "values after over-window restore");
}

#[test]
fn rotating_truncatable_only_while_unwrapped() {
    let mut cache = RotatingKVCache::new(16);
    fill_rotating_one_at_a_time(&mut cache, 10);
    let mut unwrapped = ModelStateSnapshot::new("test", 10);
    snapshot_rotating(&cache, &mut unwrapped, "layer0", TEST_NAMES).expect("snapshot");
    assert!(
        rotating_truncatable_to(&unwrapped, "layer0", 7, TEST_NAMES),
        "10 of 16 tokens is still linear, so slot 7 holds logical token 7"
    );

    for t in 10..40 {
        let _ = cache.update_and_fetch(keys_at(t, 1), values_at(t, 1));
    }
    let mut wrapped = ModelStateSnapshot::new("test", 40);
    snapshot_rotating(&cache, &mut wrapped, "layer0", TEST_NAMES).expect("snapshot");
    assert!(
        !rotating_truncatable_to(&wrapped, "layer0", 7, TEST_NAMES),
        "a wrapped ring no longer keeps logical token t at slot t"
    );
}

#[test]
fn rotating_truncate_refuses_a_wrapped_ring() {
    let mut cache = RotatingKVCache::new(16);
    fill_rotating_one_at_a_time(&mut cache, 40);
    let err = truncate_rotating(&mut cache, 7, TEST_NAMES).expect_err("wrapped ring");
    assert!(
        err.contains("has wrapped and cannot be trimmed"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// ChunkedKVCache
// ---------------------------------------------------------------------------

#[test]
fn chunked_round_trip_after_front_trim() {
    let mut source = ChunkedKVCache::new(8);
    let _ = source.update_and_fetch(keys_at(0, 20), values_at(0, 20));
    source.maybe_trim_front();
    assert_eq!(source.start_position, 12, "front trim start_position");
    assert_eq!(source.offset, 20, "front trim keeps the monotonic offset");

    let mut snapshot = ModelStateSnapshot::new("test", 20);
    snapshot_chunked(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = ChunkedKVCache::new(8);
    restore_chunked(&mut restored, &snapshot, "layer0", TEST_NAMES).expect("restore");
    assert_eq!(restored.start_position, 12);
    assert_eq!(restored.offset, 20);

    let source_window = source.keys.as_ref().and_then(|k| k.as_ref()).expect("keys");
    let restored_window = restored
        .keys
        .as_ref()
        .and_then(|k| k.as_ref())
        .expect("restored keys");
    assert_eq!(
        mlxcel_core::array_shape(restored_window)[2],
        8,
        "the visible window is offset - start_position"
    );
    assert_arrays_equal(restored_window, source_window, "restored visible window");

    let (source_k, source_v) = source.update_and_fetch(keys_at(20, 1), values_at(20, 1));
    let (restored_k, restored_v) = restored.update_and_fetch(keys_at(20, 1), values_at(20, 1));
    assert_arrays_equal(&restored_k, &source_k, "keys after chunked restore");
    assert_arrays_equal(&restored_v, &source_v, "values after chunked restore");
}

#[test]
fn chunked_truncatable_only_before_front_trim() {
    let mut untrimmed = ChunkedKVCache::new(8);
    let _ = untrimmed.update_and_fetch(keys_at(0, 6), values_at(0, 6));
    untrimmed.maybe_trim_front();
    assert_eq!(untrimmed.start_position, 0, "6 of 8 tokens does not trim");
    let mut before = ModelStateSnapshot::new("test", 6);
    snapshot_chunked(&untrimmed, &mut before, "layer0", TEST_NAMES).expect("snapshot");
    assert!(chunked_truncatable_to(&before, "layer0", 4));

    let mut trimmed = ChunkedKVCache::new(8);
    let _ = trimmed.update_and_fetch(keys_at(0, 20), values_at(0, 20));
    trimmed.maybe_trim_front();
    let mut after = ModelStateSnapshot::new("test", 20);
    snapshot_chunked(&trimmed, &mut after, "layer0", TEST_NAMES).expect("snapshot");
    assert!(
        !chunked_truncatable_to(&after, "layer0", 4),
        "a front-trimmed window is narrower than a cold prefill of the same prefix"
    );
}

#[test]
fn chunked_truncated_restore_matches_a_cold_fill_of_the_same_prefix() {
    let mut source = ChunkedKVCache::new(8);
    let _ = source.update_and_fetch(keys_at(0, 6), values_at(0, 6));
    let mut snapshot = ModelStateSnapshot::new("test", 6);
    snapshot_chunked(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = ChunkedKVCache::new(8);
    restore_chunked(&mut restored, &snapshot, "layer0", TEST_NAMES).expect("restore");
    truncate_chunked(&mut restored, 4, TEST_NAMES).expect("truncate");
    assert_eq!(restored.offset, 4);
    assert_eq!(restored.start_position, 0);

    let mut cold = ChunkedKVCache::new(8);
    let _ = cold.update_and_fetch(keys_at(0, 4), values_at(0, 4));

    let (cold_k, cold_v) = cold.update_and_fetch(keys_at(4, 1), values_at(4, 1));
    let (warm_k, warm_v) = restored.update_and_fetch(keys_at(4, 1), values_at(4, 1));
    assert_arrays_equal(&warm_k, &cold_k, "keys after truncated chunked restore");
    assert_arrays_equal(&warm_v, &cold_v, "values after truncated chunked restore");
}

#[test]
fn chunked_truncate_refuses_a_front_trimmed_cache() {
    let mut cache = ChunkedKVCache::new(8);
    let _ = cache.update_and_fetch(keys_at(0, 20), values_at(0, 20));
    cache.maybe_trim_front();
    let err = truncate_chunked(&mut cache, 4, TEST_NAMES).expect_err("front-trimmed cache");
    assert!(err.contains("trimmed its front"), "unexpected error: {err}");
}

#[test]
fn chunked_restore_rejects_a_chunk_size_mismatch() {
    let mut source = ChunkedKVCache::new(8);
    let _ = source.update_and_fetch(keys_at(0, 6), values_at(0, 6));
    let mut snapshot = ModelStateSnapshot::new("test", 6);
    snapshot_chunked(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut other = ChunkedKVCache::new(16);
    let err = restore_chunked(&mut other, &snapshot, "layer0", TEST_NAMES)
        .expect_err("chunk size mismatch");
    assert!(
        err.contains("does not match configured chunk size"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Mode gate and family naming
// ---------------------------------------------------------------------------

#[test]
fn non_fp16_mode_is_rejected() {
    // Populate the buffers directly rather than through `update_and_fetch`:
    // this test is about the mode gate, not about the Int8 quantizer.
    let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
    int8.keys = Some(keys_at(0, 4));
    int8.values = Some(values_at(0, 4));
    int8.offset = 4;

    let mut snapshot = ModelStateSnapshot::new("test", 4);
    let err = snapshot_standard(&int8, &mut snapshot, "layer0", TEST_NAMES)
        .expect_err("Int8 cache must not be snapshotted");
    assert!(
        err.contains("is not supported by model-state snapshots"),
        "unexpected error: {err}"
    );
    assert!(snapshot.is_empty(), "a refused snapshot stores nothing");

    let mut fp16 = KVCache::new();
    fill_standard(&mut fp16, 4);
    let mut good = ModelStateSnapshot::new("test", 4);
    snapshot_standard(&fp16, &mut good, "layer0", TEST_NAMES).expect("fp16 snapshot");

    let mut target = KVCache::new_with_mode(KVCacheMode::Int8);
    let err = restore_standard(&mut target, &good, "layer0", TEST_NAMES)
        .expect_err("an Fp16 snapshot must not land in an Int8 cache");
    assert!(
        err.contains("does not match configured cache mode"),
        "unexpected error: {err}"
    );
    assert!(
        target.keys.is_none(),
        "a refused restore must leave the cache untouched"
    );
}

#[test]
fn family_names_select_the_stored_tensor_names() {
    let mut cache = KVCache::new();
    fill_standard(&mut cache, 4);
    let mut snapshot = ModelStateSnapshot::new("test", 4);
    snapshot_standard(&cache, &mut snapshot, "layer0", ALT_NAMES).expect("snapshot");
    assert!(
        snapshot.tensor("layer0.full.keys").is_some(),
        "Muse Glimmer spells the full-attention segment `full`"
    );
    assert!(
        snapshot.tensor("layer0.standard.keys").is_none(),
        "the Gemma 4 spelling must not appear under Muse Glimmer names"
    );

    let mut rotating = RotatingKVCache::new(16);
    fill_rotating_one_at_a_time(&mut rotating, 4);
    let mut rotating_snapshot = ModelStateSnapshot::new("test", 4);
    snapshot_rotating(&rotating, &mut rotating_snapshot, "layer0", ALT_NAMES).expect("snapshot");
    assert!(rotating_snapshot.tensor("layer0.sliding.keys").is_some());
    assert!(
        rotating_snapshot
            .tensor("layer0.sliding.turbo_seed")
            .is_some(),
        "the shared serializer stores turbo_seed for every family"
    );
    assert!(rotating_snapshot.tensor("layer0.rotating.keys").is_none());
}

// ---------------------------------------------------------------------------
// Restore-time geometry validation
// ---------------------------------------------------------------------------
//
// A snapshot only ever comes from this process's own live caches today, so
// these shapes are unreachable through the server. They are still refused
// rather than installed: `restore_*` is the boundary at which a stored state
// becomes a live GPU cache, and a state whose declared length runs past its own
// buffer makes the next append slice past the end of the sequence axis, which
// surfaces as an MLX throw at the FFI boundary rather than as a declined
// restore.

#[test]
fn standard_restore_refuses_an_offset_past_its_buffer() {
    let mut source = KVCache::new();
    source.keys = Some(keys_at(0, 4));
    source.values = Some(values_at(0, 4));
    source.offset = 9;

    let mut snapshot = ModelStateSnapshot::new("test", 9);
    snapshot_standard(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = KVCache::new();
    let err = restore_standard(&mut restored, &snapshot, "layer0", TEST_NAMES)
        .expect_err("offset past the stored buffer");
    assert!(
        err.contains("declares 9 tokens but its buffer holds 4"),
        "unexpected error: {err}"
    );
    assert!(
        restored.keys.is_none() && restored.offset == 0,
        "a refused restore must leave the cache untouched"
    );
}

#[test]
fn standard_restore_refuses_keys_and_values_of_different_lengths() {
    let mut source = KVCache::new();
    source.keys = Some(keys_at(0, 4));
    source.values = Some(values_at(0, 6));
    source.offset = 4;

    let mut snapshot = ModelStateSnapshot::new("test", 4);
    snapshot_standard(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = KVCache::new();
    let err = restore_standard(&mut restored, &snapshot, "layer0", TEST_NAMES)
        .expect_err("keys and values of different lengths");
    assert!(
        err.contains("disagree on batch, heads or length"),
        "unexpected error: {err}"
    );
}

#[test]
fn chunked_restore_refuses_a_window_past_its_buffer() {
    let mut source = ChunkedKVCache::new(8);
    source.keys = Some(keys_at(0, 4));
    source.values = Some(values_at(0, 4));
    source.offset = 20;
    source.start_position = 0;

    let mut snapshot = ModelStateSnapshot::new("test", 20);
    snapshot_chunked(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = ChunkedKVCache::new(8);
    let err = restore_chunked(&mut restored, &snapshot, "layer0", TEST_NAMES)
        .expect_err("visible window past the stored buffer");
    assert!(
        err.contains("declares 20 tokens but its buffer holds 4"),
        "unexpected error: {err}"
    );
    assert!(
        restored.keys.is_none() && restored.offset == 0 && restored.start_position == 0,
        "a refused restore must leave the cache untouched"
    );
}

#[test]
fn chunked_restore_refuses_an_inverted_window() {
    let mut source = ChunkedKVCache::new(8);
    source.keys = Some(keys_at(0, 4));
    source.values = Some(values_at(0, 4));
    source.offset = 2;
    source.start_position = 5;

    let mut snapshot = ModelStateSnapshot::new("test", 5);
    snapshot_chunked(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = ChunkedKVCache::new(8);
    let err = restore_chunked(&mut restored, &snapshot, "layer0", TEST_NAMES)
        .expect_err("start_position past offset");
    assert!(
        err.contains("is not a valid range"),
        "unexpected error: {err}"
    );
}

#[test]
fn rotating_restore_refuses_a_window_mismatch() {
    let mut source = RotatingKVCache::new(16);
    fill_rotating_one_at_a_time(&mut source, 10);
    let mut snapshot = ModelStateSnapshot::new("test", 10);
    snapshot_rotating(&source, &mut snapshot, "layer0", TEST_NAMES).expect("snapshot");

    let mut restored = RotatingKVCache::new(8);
    let err = restore_rotating(&mut restored, &snapshot, "layer0", TEST_NAMES)
        .expect_err("window mismatch");
    assert!(
        err.contains("does not match configured window"),
        "unexpected error: {err}"
    );
    assert!(
        restored.keys.is_none(),
        "a refused restore must leave the cache untouched"
    );
}

#[test]
fn truncate_refuses_a_negative_target() {
    let mut standard = KVCache::new();
    fill_standard(&mut standard, 6);
    let err = truncate_standard(&mut standard, -1, TEST_NAMES).expect_err("negative target");
    assert!(err.contains("is negative"), "unexpected error: {err}");

    let mut chunked = ChunkedKVCache::new(8);
    let _ = chunked.update_and_fetch(keys_at(0, 6), values_at(0, 6));
    let err = truncate_chunked(&mut chunked, -1, TEST_NAMES).expect_err("negative target");
    assert!(err.contains("is negative"), "unexpected error: {err}");
    assert_eq!(chunked.offset, 6, "a refused truncate must not move offset");

    let mut rotating = RotatingKVCache::new(16);
    fill_rotating_one_at_a_time(&mut rotating, 6);
    let err = truncate_rotating(&mut rotating, -1, TEST_NAMES).expect_err("negative target");
    assert!(err.contains("is negative"), "unexpected error: {err}");
}

#[test]
fn chunked_truncate_does_not_shorten_keys_when_the_value_slice_fails() {
    // `truncate_chunked` builds both slices before installing either, so a
    // failure slicing the second buffer must not leave the cache holding a
    // shortened key buffer beside its original, full-length value buffer
    // (983b8bae). Giving keys and values different physical lengths up front
    // is the only way to make the second slice fail on its own.
    let mut cache = ChunkedKVCache::new(8);
    cache.keys = Some(keys_at(0, 8));
    cache.values = Some(values_at(0, 4));
    cache.offset = 8;

    let err = truncate_chunked(&mut cache, 6, TEST_NAMES).expect_err("short values buffer");
    assert!(
        err.contains("exceeds") && err.contains("length"),
        "unexpected error: {err}"
    );
    let keys = cache
        .keys
        .as_ref()
        .and_then(|k| k.as_ref())
        .expect("keys must still be present");
    assert_eq!(
        mlxcel_core::array_shape(keys)[2],
        8,
        "a refused truncate must not shorten the key buffer either"
    );
    assert_eq!(cache.offset, 8, "a refused truncate must not move offset");
}

#[test]
fn error_messages_carry_the_family_label() {
    let mut cache = ChunkedKVCache::new(8);
    let _ = cache.update_and_fetch(keys_at(0, 4), values_at(0, 4));
    let err = truncate_chunked(&mut cache, 9, ALT_NAMES).expect_err("target past offset");
    assert!(
        err.starts_with("Muse Glimmer truncate:"),
        "unexpected error: {err}"
    );
}
