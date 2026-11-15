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

//! `KVCache::reserve_rows` (issue #3084): an eager growth must leave the cache
//! exactly as the lazy growth inside `update_fp16` would have.

use super::*;
use crate::dtype;

/// `[1, 2, n, d]` rows with distinct values, so a misplaced row shows.
fn rows(base: f32, n: i32, d: i32) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..(2 * n * d)).map(|i| base + i as f32 * 0.25).collect();
    ffi::from_slice_f32(&data, &[1, 2, n, d])
}

fn bytes(arr: &MlxArray) -> Vec<u8> {
    ffi::eval(arr);
    ffi::array_to_raw_bytes(arr)
}

fn buffer_len(cache: &KVCache) -> i32 {
    ffi::array_shape(cache.keys.as_ref().unwrap())[2]
}

/// Appends `pieces` multi-token blocks (K width 6, V width 4, as MLA's K and V
/// widths differ) through two eager layers and one lazy reference, reserving
/// only on the eager pair, and checks layer 0's fetched windows and whole
/// buffers match the reference after every piece.
fn assert_eager_matches_lazy(pieces: &[i32], trim_after_first: i32) {
    // Two layers, so a reservation's flush runs while another layer's grown
    // copy is still queued.
    let mut eager = [KVCache::new(), KVCache::new()];
    let mut lazy = KVCache::new();
    let mut grew_any = false;
    for (i, &n) in pieces.iter().enumerate() {
        let cap_before = eager[0].keys.as_ref().map(|_| buffer_len(&eager[0]));
        let grown = eager
            .iter_mut()
            .map(|c| usize::from(c.reserve_rows(n)))
            .sum::<usize>();
        let cap_after = eager[0].keys.as_ref().map(|_| buffer_len(&eager[0]));
        // Both layers hold the same row count, so they grow together, and a
        // reservation that grew must leave the append nothing to grow.
        assert!(grown == 0 || grown == 2, "piece {i}: {grown} caches grew");
        assert_eq!(grown == 2, cap_before != cap_after, "piece {i}");
        grew_any |= grown == 2;

        let _ = eager[1].update_and_fetch(rows(i as f32, n, 6), rows(i as f32, n, 4));
        let k_new = rows(i as f32 * 10.0, n, 6);
        let v_new = rows(-(i as f32) * 10.0, n, 4);
        let (k, v) = eager[0].update_and_fetch(k_new, v_new);
        let (k_ref, v_ref) =
            lazy.update_and_fetch(rows(i as f32 * 10.0, n, 6), rows(-(i as f32) * 10.0, n, 4));
        assert_eq!(ffi::array_shape(&k), ffi::array_shape(&k_ref), "piece {i}");
        assert_eq!(bytes(&k), bytes(&k_ref), "piece {i} keys");
        assert_eq!(bytes(&v), bytes(&v_ref), "piece {i} values");
        assert_eq!(
            buffer_len(&eager[0]),
            buffer_len(&lazy),
            "piece {i} capacity"
        );
        assert_eq!(eager[0].offset, lazy.offset);
        if i == 0 && trim_after_first > 0 {
            for cache in eager.iter_mut().chain(std::iter::once(&mut lazy)) {
                assert_eq!(cache.trim(trim_after_first), trim_after_first);
            }
        }
    }
    assert!(grew_any, "the piece plan must exercise at least one growth");
}

#[test]
fn eager_growth_matches_lazy_growth_across_chunked_pieces() {
    // The first piece allocates two 256-row steps; the second overflows them,
    // and every later growth starts from a non-step-aligned buffer.
    assert_eager_matches_lazy(&[300, 300, 300, 7], 0);
}

#[test]
fn eager_growth_drops_dead_slots_after_a_trim() {
    // A trim keeps the old capacity; the growth must not carry the trimmed
    // rows into the new buffer.
    assert_eager_matches_lazy(&[400, 300, 600], 50);
}

#[test]
fn reserve_declines_when_there_is_nothing_to_release() {
    let one = |c: &mut KVCache, n: i32| usize::from(c.reserve_rows(n));
    let mut cache = KVCache::new();
    // First fill: no buffer yet.
    assert_eq!(one(&mut cache, 8), 0);
    let _ = cache.update_and_fetch(rows(0.0, 8, 6), rows(0.0, 8, 4));
    let cap = buffer_len(&cache);
    // Fits in the current buffer.
    assert_eq!(one(&mut cache, 4), 0);
    // A decode row never reserves, even past the buffer edge.
    assert_eq!(one(&mut cache, 1), 0);
    assert_eq!(buffer_len(&cache), cap);
    // A quantized layout keeps its own growth path.
    let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
    let _ = int8.update_and_fetch(
        ffi::ones(&[1, 2, 8, 4], dtype::FLOAT32),
        ffi::ones(&[1, 2, 8, 4], dtype::FLOAT32),
    );
    assert_eq!(one(&mut int8, 600), 0);
}
