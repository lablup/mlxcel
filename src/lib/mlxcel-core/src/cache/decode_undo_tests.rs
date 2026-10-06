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

//! Unit tests for [`RotatingKVCache::rewind_decode_writes`] (issue #2159).
//!
//! A rewind of `n` decode writes must leave keys, values, `offset`, and `idx`
//! exactly as they were before those writes, including after the ring wrapped
//! and the writes overwrote valid slots. Every refusal must leave the cache
//! untouched.

use super::*;
use crate::utils::array_to_vec_f32;

const HEADS: i32 = 2;
const DIM: i32 = 2;

/// Append `count` positions (one multi-token append when `count > 1`) whose
/// K/V values encode their absolute position plus `tag`, distinct per head
/// and lane so a misplaced row cannot compare equal.
fn append(cache: &mut RotatingKVCache, count: i32, tag: f32) {
    let mut keys = Vec::new();
    for h in 0..HEADS {
        for i in 0..count {
            for d in 0..DIM {
                keys.push((cache.offset + i) as f32 + tag + 0.25 * h as f32 + 0.125 * d as f32);
            }
        }
    }
    let values: Vec<f32> = keys.iter().map(|k| -k).collect();
    let _ = cache.update_and_fetch(
        ffi::from_slice_f32(&keys, &[1, HEADS, count, DIM]),
        ffi::from_slice_f32(&values, &[1, HEADS, count, DIM]),
    );
}

/// One single-token decode write tagged as speculative, made inside the
/// lookahead append scope as the scheduler's prime forward makes it.
fn speculative_write(cache: &mut RotatingKVCache) {
    let _scope = DecodeLookaheadAppendScope::enter();
    append(cache, 1, 1000.0);
}

fn state(cache: &RotatingKVCache) -> (i32, i32, Vec<i32>, Vec<f32>, Vec<f32>) {
    let keys = cache.keys.as_ref().expect("keys");
    (
        cache.offset,
        cache.idx,
        ffi::array_shape(keys),
        array_to_vec_f32(keys),
        array_to_vec_f32(cache.values.as_ref().expect("values")),
    )
}

/// A depth-2 logging cache of window `window` holding `prompt` prefill
/// positions followed by `decoded` single-token writes.
fn cache_after(window: i32, prompt: i32, decoded: i32) -> RotatingKVCache {
    let mut cache = RotatingKVCache::new(window);
    cache.set_decode_undo_depth(DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS);
    append(&mut cache, prompt, 0.0);
    for _ in 0..decoded {
        append(&mut cache, 1, 0.0);
    }
    cache
}

/// A second cache sharing `cache`'s buffers and cursors (no undo log).
fn fork(cache: &RotatingKVCache) -> RotatingKVCache {
    let mut copy = RotatingKVCache::new(cache.max_size);
    copy.keys = Some(ffi::array_handle_clone(cache.keys.as_ref().unwrap()));
    copy.values = Some(ffi::array_handle_clone(cache.values.as_ref().unwrap()));
    copy.offset = cache.offset;
    copy.idx = cache.idx;
    copy
}

fn assert_rewind_restores(mut cache: RotatingKVCache, writes: i32, label: &str) {
    let before = state(&cache);
    for _ in 0..writes {
        speculative_write(&mut cache);
    }
    cache
        .rewind_decode_writes(writes)
        .unwrap_or_else(|err| panic!("{label}: {err}"));
    assert_eq!(state(&cache), before, "{label}: rewound state");

    // The rewound cache keeps decoding exactly like one that never saw the
    // speculative writes.
    let mut reference = fork(&cache);
    for _ in 0..3 {
        append(&mut cache, 1, 0.0);
        append(&mut reference, 1, 0.0);
    }
    assert_eq!(state(&cache), state(&reference), "{label}: decode after");
}

#[test]
fn rewinds_one_and_two_writes_after_the_ring_wrapped() {
    // Window 4: a 3-token prompt plus 6 decode steps has wrapped the ring.
    assert_rewind_restores(cache_after(4, 3, 6), 1, "wrapped, one write");
    assert_rewind_restores(cache_after(4, 3, 6), 2, "wrapped, two writes");
    // The second write rolls the cursor over the end of the buffer.
    assert_rewind_restores(cache_after(4, 3, 4), 2, "wrapped, cursor rollover");
}

#[test]
fn rewinds_after_an_over_window_prefill() {
    assert_rewind_restores(cache_after(4, 9, 1), 1, "over-window, one write");
    assert_rewind_restores(cache_after(4, 9, 1), 2, "over-window, two writes");
}

#[test]
fn rewind_of_the_reslicing_write_decodes_like_the_prefill_state() {
    // The first decode write after a prefill longer than the window re-slices
    // the buffer to the window before overwriting slot 0. The rewind restores
    // the re-sliced layout (cursor at the window end), which every later
    // write treats exactly like the over-window buffer.
    let mut cache = cache_after(4, 9, 0);
    let mut reference = fork(&cache);
    speculative_write(&mut cache);
    speculative_write(&mut cache);
    cache.rewind_decode_writes(2).expect("rewind");
    assert_eq!(cache.offset, 9);
    for step in 0..5 {
        append(&mut cache, 1, 0.0);
        append(&mut reference, 1, 0.0);
        assert_eq!(state(&cache), state(&reference), "decode step {step}");
    }
}

#[test]
fn rewind_spanning_the_warmup_boundary_is_exact() {
    // One warmup write onto a never-valid slot, then one overwrite.
    assert_rewind_restores(cache_after(4, 2, 1), 2, "warmup + overwrite");
    // Both writes still inside the warmup.
    assert_rewind_restores(cache_after(8, 2, 1), 2, "warmup only");

    // The warmup write is older than the log (enabled late): the cursors
    // alone undo it, the logged overwrite restores its row.
    let mut cache = RotatingKVCache::new(4);
    append(&mut cache, 2, 0.0);
    append(&mut cache, 1, 0.0); // grows the buffer to the window
    let before = state(&cache);
    speculative_write(&mut cache); // warmup write, not logged
    cache.set_decode_undo_depth(1);
    speculative_write(&mut cache); // overwrite of position 0's slot, logged
    cache
        .rewind_decode_writes(2)
        .expect("an unlogged warmup write is undone by its cursors");
    assert_eq!(state(&cache), before, "late-enabled log");
}

#[test]
fn plain_trim_after_wrap_leaves_the_speculative_row_behind() {
    // The defect the log exists for: trim rewinds the cursors only.
    let mut cache = cache_after(4, 3, 6);
    let before = state(&cache);
    speculative_write(&mut cache);
    speculative_write(&mut cache);
    cache.trim(2);
    assert_ne!(
        state(&cache),
        before,
        "trim cannot restore overwritten rows"
    );
    assert_eq!(cache.decode_undo_len(), 0, "trim clears the log");
}

#[test]
fn refusals_leave_the_cache_untouched() {
    let refuse = |cache: &mut RotatingKVCache, n: i32, label: &str| {
        let before = (state(cache), cache.decode_undo_len());
        assert!(cache.rewind_decode_writes(n).is_err(), "{label}: refused");
        assert_eq!(
            (state(cache), cache.decode_undo_len()),
            before,
            "{label}: untouched"
        );
    };

    // `n` beyond the logged depth for writes past the warmup.
    let mut cache = cache_after(4, 3, 6);
    speculative_write(&mut cache);
    speculative_write(&mut cache);
    speculative_write(&mut cache);
    refuse(&mut cache, 3, "beyond depth after wrap");

    // `n` beyond the offset.
    let mut cache = cache_after(8, 2, 1);
    refuse(&mut cache, 4, "more than the offset");
    refuse(&mut cache, -1, "negative");
    assert!(cache.rewind_decode_writes(0).is_ok(), "zero is a no-op");

    // Log disabled.
    let mut cache = RotatingKVCache::new(4);
    append(&mut cache, 6, 0.0);
    speculative_write(&mut cache);
    refuse(&mut cache, 1, "log disabled");

    // Speculative buffering keeps its own rollback slack.
    let mut cache = cache_after(4, 3, 6);
    cache
        .enable_speculative_buffer(2)
        .expect("fp16 cache buffers");
    speculative_write(&mut cache);
    let before = (
        cache.offset,
        cache.idx,
        array_to_vec_f32(cache.keys.as_ref().unwrap()),
    );
    assert!(cache.rewind_decode_writes(1).is_err(), "buffered: refused");
    assert_eq!(
        (
            cache.offset,
            cache.idx,
            array_to_vec_f32(cache.keys.as_ref().unwrap())
        ),
        before,
        "buffered: untouched"
    );

    // Non-FP16 storage (INT8 rotating falls back to FP16 buffers, but its
    // mode carries sidecars the log does not cover).
    let mut cache = RotatingKVCache::new_with_mode(4, KVCacheMode::Int8);
    cache.set_decode_undo_depth(2);
    append(&mut cache, 6, 0.0);
    speculative_write(&mut cache);
    refuse(&mut cache, 1, "int8");
}

#[test]
fn multi_token_appends_clear_the_log() {
    let mut cache = cache_after(4, 3, 6);
    speculative_write(&mut cache);
    assert_eq!(cache.decode_undo_len(), 2, "the log holds its depth");
    append(&mut cache, 3, 0.0);
    assert_eq!(cache.decode_undo_len(), 0);
    assert_eq!(cache.decode_undo_depth(), 2, "the depth survives a clear");
    // With the log cleared, a rewind into the prefill is refused.
    assert!(cache.rewind_decode_writes(1).is_err());
}

#[test]
fn flushed_rows_still_restore_exactly() {
    let mut cache = cache_after(4, 3, 6);
    let before = state(&cache);
    speculative_write(&mut cache);
    flush_decode_undo_rows([&mut cache]);
    speculative_write(&mut cache);
    flush_decode_undo_rows([&mut cache]);
    // Flushing twice schedules nothing new.
    flush_decode_undo_rows([&mut cache]);
    cache.rewind_decode_writes(2).expect("rewind");
    assert_eq!(state(&cache), before);
}

#[test]
fn overwrites_outside_the_lookahead_scope_copy_nothing_and_are_not_rewindable() {
    // A synchronous step (no scope) after the wrap logs its cursors only.
    let mut cache = cache_after(4, 3, 6);
    append(&mut cache, 1, 0.0);
    let mut pending = Vec::new();
    cache.take_pending_decode_undo_rows(&mut pending);
    assert!(pending.is_empty(), "no row copy outside the scope");
    let before = (state(&cache), cache.decode_undo_len());
    assert!(cache.rewind_decode_writes(1).is_err());
    assert_eq!(
        (state(&cache), cache.decode_undo_len()),
        before,
        "untouched"
    );

    // The scope records rows, and ends with its guard.
    {
        let _scope = DecodeLookaheadAppendScope::enter();
        append(&mut cache, 1, 1000.0);
    }
    cache.take_pending_decode_undo_rows(&mut pending);
    assert_eq!(pending.len(), 2, "one K and one V row copy");
    append(&mut cache, 1, 0.0);
    cache.take_pending_decode_undo_rows(&mut pending);
    assert_eq!(pending.len(), 2, "the guard ended the scope");
}
