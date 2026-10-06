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

//! `RotatingKVCache::rewind_padded_prefill` (issue #1755): a padded append
//! followed by the rewind must leave the cache byte-identical to an unpadded
//! append of the real tokens, including after the next decode step.

use super::*;
use crate::utils::array_to_vec_f32;

/// Append `count` positions whose K/V values are their absolute positions, or
/// `PAD` for the trailing `pad` of them.
fn append(cache: &mut RotatingKVCache, count: i32, pad: i32) {
    let values: Vec<f32> = (0..count)
        .map(|i| {
            if i >= count - pad {
                -1.0
            } else {
                (cache.offset + i) as f32
            }
        })
        .collect();
    let _ = cache.update_and_fetch(
        ffi::from_slice_f32(&values, &[1, 1, count, 1]),
        ffi::from_slice_f32(&values, &[1, 1, count, 1]),
    );
}

fn state(cache: &RotatingKVCache) -> (i32, i32, Vec<f32>, Vec<f32>) {
    (
        cache.offset,
        cache.idx,
        array_to_vec_f32(cache.keys.as_ref().unwrap()),
        array_to_vec_f32(cache.values.as_ref().unwrap()),
    )
}

/// Real-token chunks, each followed by the pad width the last chunk carries.
fn assert_rewind_matches_unpadded(window: i32, prior: &[i32], real: i32, pad: i32) {
    let mut padded = RotatingKVCache::new(window);
    let mut plain = RotatingKVCache::new(window);
    for &chunk in prior {
        append(&mut padded, chunk, 0);
        append(&mut plain, chunk, 0);
    }
    append(&mut padded, real + pad, pad);
    padded
        .rewind_padded_prefill(pad)
        .expect("chronological rewind");
    append(&mut plain, real, 0);
    assert_eq!(
        state(&padded),
        state(&plain),
        "window {window}, prior {prior:?}, real {real}, pad {pad}: after the rewind"
    );

    // The next decode step pre-trims by the physical length; the pad must not
    // be what survives that.
    for _ in 0..3 {
        append(&mut padded, 1, 0);
        append(&mut plain, 1, 0);
    }
    assert_eq!(
        state(&padded),
        state(&plain),
        "window {window}, prior {prior:?}, real {real}, pad {pad}: after decode"
    );
}

#[test]
fn rewind_within_the_window_matches_unpadded_prefill() {
    assert_rewind_matches_unpadded(32, &[], 13, 19);
}

#[test]
fn rewind_of_an_over_window_padded_prefill_matches_unpadded_prefill() {
    // Padded length 32 > window 8, real length 13 > window 8: plain `trim`
    // would leave the pad tail in the physical buffer the decode pre-trim keeps.
    assert_rewind_matches_unpadded(8, &[], 13, 19);
    // Real length inside the window, padded length beyond it.
    assert_rewind_matches_unpadded(8, &[], 5, 27);
}

#[test]
fn rewind_of_a_padded_continuation_chunk_matches_unpadded_prefill() {
    assert_rewind_matches_unpadded(8, &[10], 3, 29);
    assert_rewind_matches_unpadded(8, &[3, 4], 6, 26);
}

#[test]
fn rewind_refuses_a_wrapped_ring() {
    let mut cache = RotatingKVCache::new(4);
    append(&mut cache, 6, 0);
    append(&mut cache, 1, 0); // single-token decode wraps the ring
    let before = state(&cache);
    assert!(cache.rewind_padded_prefill(1).is_err());
    assert_eq!(
        state(&cache),
        before,
        "a refused rewind leaves the cache untouched"
    );
}

#[test]
fn rewind_refuses_more_than_was_written_and_turbo4asym() {
    let mut cache = RotatingKVCache::new(8);
    append(&mut cache, 4, 0);
    assert!(cache.rewind_padded_prefill(5).is_err());
    assert!(cache.rewind_padded_prefill(-1).is_err());
    assert!(cache.rewind_padded_prefill(0).is_ok());
    assert_eq!(cache.offset, 4);

    let mut turbo = RotatingKVCache::new_with_mode(32, KVCacheMode::Turbo4Asym);
    assert!(turbo.rewind_padded_prefill(1).is_err());
}
