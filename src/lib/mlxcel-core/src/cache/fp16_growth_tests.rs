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

//! The eager, releasing FP16 growth inside `update_fp16` (issues #3084,
//! #3092): it must fire only for a multi-token growth of an existing buffer,
//! and leave the cache holding exactly the rows and capacity the lazy growth
//! leaves. The memory effect is measured in `tests/kv_growth_release.rs`.

use std::cell::RefCell;

use super::*;
use crate::cache::{PagedBlockPool, PagedKvLayout, PagedSequenceState};
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

/// The capacity the lazy growth rule leaves after appending `n` rows to a
/// buffer of `cap` slots whose live prefix is `live` rows.
fn lazy_capacity(cache: &KVCache, cap: Option<i32>, live: i32, n: i32) -> i32 {
    match cap {
        Some(cap) if live + n <= cap => cap,
        _ => {
            let grown = ((n + cache.step - 1) / cache.step) * cache.step;
            if cap.is_some() { live + grown } else { grown }
        }
    }
}

/// Appends `pieces` blocks (K width 6, V width 4, as MLA's K and V widths
/// differ) to two layers, the way a chunked prefill does, and checks after
/// every piece that layer 0's fetched window holds exactly the rows appended
/// so far, its capacity is the lazy rule's, and both layers released exactly
/// when a multi-token piece outgrew an existing buffer.
fn assert_growth_matches_lazy_rule(pieces: &[i32], trim_after_first: i32) {
    // Two layers, so one layer's flush runs while the other's grown copy is
    // still queued.
    let mut layers = [KVCache::new(), KVCache::new()];
    let mut want_k: Option<UniquePtr<MlxArray>> = None;
    let mut want_v: Option<UniquePtr<MlxArray>> = None;
    let mut released_any = false;
    for (i, &n) in pieces.iter().enumerate() {
        let cap = layers[0].keys.as_ref().map(|_| buffer_len(&layers[0]));
        let live = layers[0].buffer_idx();
        let should_release = n > 1 && cap.is_some_and(|c| live + n > c);
        let want_cap = lazy_capacity(&layers[0], cap, live, n);
        assert_eq!(
            layers[0].fp16_growth_releases(n) && layers[0].fp16_append_needs_growth(n),
            should_release,
            "piece {i}"
        );
        released_any |= should_release;

        let _ = layers[1].update_and_fetch(rows(i as f32, n, 6), rows(i as f32, n, 4));
        let k_new = rows(i as f32 * 10.0, n, 6);
        let v_new = rows(-(i as f32) * 10.0, n, 4);
        want_k = Some(match want_k.take() {
            Some(w) => concatenate(&w, &k_new, 2),
            None => ffi::contiguous(&k_new, false),
        });
        want_v = Some(match want_v.take() {
            Some(w) => concatenate(&w, &v_new, 2),
            None => ffi::contiguous(&v_new, false),
        });
        let (k, v) = layers[0].update_and_fetch(k_new, v_new);
        let (wk, wv) = (want_k.as_ref().unwrap(), want_v.as_ref().unwrap());
        assert_eq!(ffi::array_shape(&k), ffi::array_shape(wk), "piece {i}");
        assert_eq!(bytes(&k), bytes(wk), "piece {i} keys");
        assert_eq!(bytes(&v), bytes(wv), "piece {i} values");
        assert_eq!(buffer_len(&layers[0]), want_cap, "piece {i} capacity");
        assert_eq!(buffer_len(&layers[1]), want_cap, "piece {i} layer 1");
        if i == 0 && trim_after_first > 0 {
            for cache in layers.iter_mut() {
                assert_eq!(cache.trim(trim_after_first), trim_after_first);
            }
            let keep = ffi::array_shape(wk)[2] - trim_after_first;
            want_k = Some(ffi::slice(wk, &[0, 0, 0, 0], &[1, 2, keep, 6]));
            want_v = Some(ffi::slice(wv, &[0, 0, 0, 0], &[1, 2, keep, 4]));
        }
    }
    assert!(
        released_any,
        "the piece plan must exercise a releasing growth"
    );
}

#[test]
fn releasing_growth_matches_lazy_growth_across_chunked_pieces() {
    // The first piece allocates two 256-row steps; the second overflows them,
    // and every later growth starts from a non-step-aligned buffer.
    assert_growth_matches_lazy_rule(&[300, 300, 300, 7], 0);
}

#[test]
fn releasing_growth_drops_dead_slots_after_a_trim() {
    // A trim keeps the old capacity; the growth must not carry the trimmed
    // rows into the new buffer.
    assert_growth_matches_lazy_rule(&[400, 300, 600], 50);
}

#[test]
fn growth_stays_lazy_when_there_is_nothing_to_release() {
    let mut cache = KVCache::new();
    // First fill: no buffer yet.
    assert!(!cache.fp16_growth_releases(8));
    let _ = cache.update_and_fetch(rows(0.0, 8, 6), rows(0.0, 8, 4));
    let cap = buffer_len(&cache);
    // An append that fits allocates nothing.
    assert!(!cache.fp16_append_needs_growth(4));
    // A decode row never releases, even at the buffer edge.
    assert!(!cache.fp16_growth_releases(1));
    let edge = cap - cache.buffer_idx();
    let _ = cache.update_and_fetch(rows(1.0, edge, 6), rows(1.0, edge, 4));
    assert!(cache.fp16_append_needs_growth(1));
    let (k, _) = cache.update_and_fetch(rows(2.0, 1, 6), rows(2.0, 1, 4));
    assert_eq!(ffi::array_shape(&k)[2], cap + 1);
    assert_eq!(buffer_len(&cache), cap + cache.step);
}

#[test]
fn decode_and_first_fill_growths_report_no_release() {
    let mut cache = KVCache::new();
    let (k, v) = (rows(0.0, 8, 6), rows(0.0, 8, 4));
    assert!(!cache.grow_fp16_buffers(&k, &v), "first fill");
    let (k1, v1) = (rows(0.0, 1, 6), rows(0.0, 1, 4));
    assert!(!cache.grow_fp16_buffers(&k1, &v1), "decode row");
    assert!(cache.grow_fp16_buffers(&k, &v), "multi-token growth");
}

#[test]
fn quantized_and_pooled_layouts_skip_the_fp16_growth() {
    // INT8 keeps its own int8 buffers across multi-token growths; the FP16
    // growth would have rebuilt them in the incoming f32.
    let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
    for _ in 0..3 {
        let _ = int8.update_and_fetch(
            ffi::ones(&[1, 2, 300, 4], dtype::FLOAT32),
            ffi::ones(&[1, 2, 300, 4], dtype::FLOAT32),
        );
    }
    assert_eq!(int8.offset, 900);
    assert_eq!(ffi::array_dtype(int8.keys.as_ref().unwrap()), dtype::INT8);

    // A pool-backed cache writes into the pool and never allocates the
    // dense buffers the FP16 growth would grow.
    const PAGE: usize = 32;
    let layout = PagedKvLayout::uniform(1, PAGE, PAGE * 2 * 4 * 2).expect("valid layout");
    let mut pool = PagedBlockPool::new(layout.clone());
    pool.set_slab_blocks(64).expect("fresh pool has no storage");
    let state = Rc::new(RefCell::new(PagedSequenceState::new(&layout)));
    let mut paged = KVCache::new_paged(Rc::new(RefCell::new(pool)), state, 0);
    for _ in 0..3 {
        let k = ffi::astype(&ffi::ones(&[1, 2, 300, 4], dtype::FLOAT32), dtype::FLOAT16);
        let v = ffi::astype(&ffi::ones(&[1, 2, 300, 4], dtype::FLOAT32), dtype::FLOAT16);
        let _ = paged.update_and_fetch(k, v);
    }
    assert!(paged.keys.is_none() && paged.values.is_none());
}
