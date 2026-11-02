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

//! `KVCache::trim`, the rollback primitive behind `Engine::unwind_appends` and
//! the server's speculative rewinds (issue #1209).

use super::*;
use crate::dtype;

#[test]
fn test_kv_cache_trim_basic() {
    let mut cache = KVCache::new();

    // Add some data: [batch=1, heads=2, seq_len=5, head_dim=4]
    let keys = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
    let values = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
    cache.update(keys, values);
    assert_eq!(cache.offset, 5);
    let cap_before = ffi::array_shape(cache.keys.as_ref().unwrap())[2];

    // Trim 2. A dense trim is a logical rewind (issue #1209): only the
    // offset moves, and the buffer keeps its pre-allocated capacity so
    // the next append does not regrow.
    let trimmed = cache.trim(2);
    assert_eq!(trimmed, 2);
    assert_eq!(cache.offset, 3);
    assert_eq!(cache.seq_len(), 3);
    let k_shape = ffi::array_shape(cache.keys.as_ref().unwrap());
    assert_eq!(k_shape, vec![1, 2, cap_before, 4]);
    let v_shape = ffi::array_shape(cache.values.as_ref().unwrap());
    assert_eq!(v_shape, vec![1, 2, cap_before, 4]);

    // The visible window is what the offset says, not the buffer length.
    let (k_win, v_win) = cache.update_and_fetch(
        ffi::ones(&[1, 2, 1, 4], dtype::FLOAT32),
        ffi::ones(&[1, 2, 1, 4], dtype::FLOAT32),
    );
    assert_eq!(ffi::array_shape(&k_win), vec![1, 2, 4, 4]);
    assert_eq!(ffi::array_shape(&v_win), vec![1, 2, 4, 4]);
}

/// The rollback contract behind the logical trim (issue #1209): a
/// trimmed-then-overwritten cache must fetch a window byte-identical to a
/// cache that never saw the rejected block, in Fp16 and in Int8 (whose
/// scale sidecars must rewind in lockstep).
#[test]
fn kv_cache_trim_then_append_matches_untrimmed_reference() {
    let fill = |v: f32, n: i32| {
        let data: Vec<f32> = (0..(2 * n * 4)).map(|i| v + i as f32 * 0.125).collect();
        ffi::from_slice_f32(&data, &[1, 2, n, 4])
    };
    let bytes = |arr: &crate::ffi::MlxArray| {
        ffi::eval(arr);
        ffi::array_to_raw_bytes(arr)
    };
    for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
        let mut cache = KVCache::new_with_mode(mode);
        let mut reference = KVCache::new_with_mode(mode);
        let _ = cache.update_and_fetch(fill(1.0, 5), fill(100.0, 5));
        let _ = reference.update_and_fetch(fill(1.0, 5), fill(100.0, 5));

        // A rejected speculative block reaches only the trimmed cache.
        let _ = cache.update_and_fetch(fill(-7.0, 3), fill(-70.0, 3));
        assert_eq!(cache.trim(3), 3);

        let (k, v) = cache.update_and_fetch(fill(2.0, 2), fill(200.0, 2));
        let (k_ref, v_ref) = reference.update_and_fetch(fill(2.0, 2), fill(200.0, 2));
        assert_eq!(ffi::array_shape(&k), ffi::array_shape(&k_ref));
        assert_eq!(bytes(&k), bytes(&k_ref), "{mode:?} keys diverge after trim");
        assert_eq!(
            bytes(&v),
            bytes(&v_ref),
            "{mode:?} values diverge after trim"
        );
    }
}

#[test]
fn test_kv_cache_trim_all() {
    let mut cache = KVCache::new();
    let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
    let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
    cache.update(keys, values);

    // Trim all
    let trimmed = cache.trim(3);
    assert_eq!(trimmed, 3);
    assert_eq!(cache.offset, 0);
    assert!(cache.keys.is_none());
    assert!(cache.values.is_none());
}

#[test]
fn test_kv_cache_trim_zero() {
    let mut cache = KVCache::new();
    let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
    let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
    cache.update(keys, values);

    let trimmed = cache.trim(0);
    assert_eq!(trimmed, 0);
    assert_eq!(cache.offset, 3);
}

#[test]
fn test_kv_cache_trim_more_than_available() {
    let mut cache = KVCache::new();
    let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
    let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
    cache.update(keys, values);

    // Trim more than available - should trim only what's available
    let trimmed = cache.trim(10);
    assert_eq!(trimmed, 3);
    assert_eq!(cache.offset, 0);
    assert!(cache.keys.is_none());
}
