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

//! A first fill longer than the window keeps only the window plus the rewind
//! slack (issue #3062), and every operation after it returns exactly what the
//! full-length first fill returned.
//!
//! Each test drives two caches through the same operations: `trimmed` the
//! default way, and `full` with every update made inside a
//! [`FullFirstFillScope`], which reproduces the pre-#3062 layout. Returned
//! keys and values are compared exactly.

use super::first_fill::{FIRST_FILL_REWIND_SLACK, FullFirstFillScope};
use super::*;
use crate::dtype;

const HEAD_DIM: i32 = 64;

fn tensor(tokens: i32, seed: u32) -> UniquePtr<MlxArray> {
    let total = (2 * tokens * HEAD_DIM) as usize;
    let mut state = seed.max(1);
    let data: Vec<f32> = (0..total)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 1) as f32 / (i32::MAX as f32)
        })
        .collect();
    ffi::astype(
        &ffi::from_slice_f32(&data, &[1, 2, tokens, HEAD_DIM]),
        dtype::FLOAT16,
    )
}

fn to_f32(arr: &MlxArray) -> Vec<f32> {
    let a = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&a);
    ffi::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn stored_len(cache: &RotatingKVCache) -> i32 {
    ffi::array_shape(cache.keys.as_ref().expect("cache holds keys"))[2]
}

struct Twin {
    trimmed: RotatingKVCache,
    full: RotatingKVCache,
    seed: u32,
}

impl Twin {
    fn new(max_size: i32, mode: KVCacheMode) -> Self {
        Self {
            trimmed: RotatingKVCache::new_with_mode(max_size, mode),
            full: RotatingKVCache::new_with_mode(max_size, mode),
            seed: 1,
        }
    }

    /// Append `tokens` to both caches and assert they return the same K/V.
    fn append(&mut self, tokens: i32) {
        self.seed += 2;
        let (kt, vt) = self
            .trimmed
            .update_and_fetch(tensor(tokens, self.seed), tensor(tokens, self.seed + 1));
        let (kf, vf) = {
            let _scope = FullFirstFillScope::enter();
            self.full
                .update_and_fetch(tensor(tokens, self.seed), tensor(tokens, self.seed + 1))
        };
        let at = format!("append of {tokens} at offset {}", self.full.offset);
        assert_eq!(ffi::array_shape(&kt), ffi::array_shape(&kf), "{at}: keys");
        assert_eq!(to_f32(&kt), to_f32(&kf), "{at}: keys");
        assert_eq!(to_f32(&vt), to_f32(&vf), "{at}: values");
        assert_eq!(self.trimmed.offset, self.full.offset, "{at}: offset");
        assert_eq!(
            self.trimmed.logical_start(),
            self.full.logical_start(),
            "{at}: logical start"
        );
    }

    fn decode(&mut self, steps: i32) {
        for _ in 0..steps {
            self.append(1);
        }
    }

    fn trim(&mut self, n: i32) {
        assert_eq!(self.trimmed.trim(n), self.full.trim(n));
    }
}

#[test]
fn an_over_window_first_fill_stores_the_window_and_slack() {
    let window = 16;
    let keep = window + FIRST_FILL_REWIND_SLACK;
    let mut cache = RotatingKVCache::new(window);
    let (k, v) = cache.update_and_fetch(tensor(200, 3), tensor(200, 4));
    assert_eq!(ffi::array_shape(&k)[2], 200, "attention sees every key");
    assert_eq!(ffi::array_shape(&v)[2], 200, "attention sees every value");
    assert_eq!(stored_len(&cache), keep);
    assert_eq!(cache.idx, keep);
    assert_eq!(cache.offset, 200);
    assert_eq!(cache.visible_len(), keep);
    assert_eq!(cache.logical_start(), 0);
    let tail = ffi::slice(&k, &[0, 0, 200 - keep, 0], &[1, 2, 200, HEAD_DIM]);
    assert_eq!(to_f32(cache.keys.as_ref().unwrap()), to_f32(&tail));
    assert!(
        !cache.is_trimmable(),
        "an over-window fill was never trimmable"
    );
}

#[test]
fn fills_within_the_slack_and_scoped_fills_store_everything() {
    let window = 16;
    let keep = window + FIRST_FILL_REWIND_SLACK;
    let mut short = RotatingKVCache::new(window);
    let _ = short.update_and_fetch(tensor(keep, 3), tensor(keep, 4));
    assert_eq!(stored_len(&short), keep);
    assert_eq!(short.idx, keep);

    let mut scoped = RotatingKVCache::new(window);
    {
        let _outer = FullFirstFillScope::enter();
        {
            let _inner = FullFirstFillScope::enter();
        }
        assert!(FullFirstFillScope::is_active(), "scopes nest");
        let _ = scoped.update_and_fetch(tensor(200, 3), tensor(200, 4));
    }
    assert!(!FullFirstFillScope::is_active());
    assert_eq!(stored_len(&scoped), 200);
}

#[test]
fn decode_after_a_trimmed_first_fill_matches_the_full_fill() {
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
    ] {
        let mut twin = Twin::new(32, mode);
        twin.append(300);
        // Past one full wrap of the ring.
        twin.decode(70);
    }
}

#[test]
fn a_continuation_after_a_trimmed_first_fill_matches_the_full_fill() {
    for mode in [KVCacheMode::Fp16, KVCacheMode::Turbo4Asym] {
        let mut twin = Twin::new(32, mode);
        twin.append(300);
        twin.append(50);
        twin.append(7);
        twin.decode(40);
    }
}

#[test]
fn a_trim_within_the_slack_matches_the_full_fill() {
    for n in [1, 5, FIRST_FILL_REWIND_SLACK] {
        for mode in [KVCacheMode::Fp16, KVCacheMode::Turbo4Asym] {
            // Trim, then decode.
            let mut twin = Twin::new(32, mode);
            twin.append(300);
            twin.trim(n);
            twin.decode(40);

            // Trim, then a multi-token append (a re-prefill or verify block).
            let mut twin = Twin::new(32, mode);
            twin.append(300);
            twin.trim(n);
            twin.append(6);
            twin.decode(5);
        }
    }
}

#[test]
fn a_padded_rewind_within_the_slack_matches_an_unpadded_fill() {
    let window = 32;
    let real = 300;
    for pad in [1, 17, FIRST_FILL_REWIND_SLACK - 1, FIRST_FILL_REWIND_SLACK] {
        let mut padded = RotatingKVCache::new(window);
        let mut plain = RotatingKVCache::new(window);
        let prompt = tensor(real + pad, 9);
        let real_rows = ffi::slice(&prompt, &[0, 0, 0, 0], &[1, 2, real, HEAD_DIM]);
        let _ = padded.update_and_fetch(ffi::copy(&prompt), ffi::copy(&prompt));
        let _ = plain.update_and_fetch(ffi::copy(&real_rows), ffi::copy(&real_rows));
        padded
            .rewind_padded_prefill(pad)
            .expect("a rewind within the slack is supported");
        assert_eq!(padded.offset, plain.offset);
        for step in 0..40u32 {
            let (kp, vp) = padded.update_and_fetch(tensor(1, 100 + step), tensor(1, 200 + step));
            let (kq, vq) = plain.update_and_fetch(tensor(1, 100 + step), tensor(1, 200 + step));
            assert_eq!(
                to_f32(&kp),
                to_f32(&kq),
                "pad {pad}, decode step {step}: keys"
            );
            assert_eq!(
                to_f32(&vp),
                to_f32(&vq),
                "pad {pad}, decode step {step}: values"
            );
        }
    }
}

#[test]
fn a_padded_rewind_past_the_slack_is_refused_and_supported_in_scope() {
    let window = 32;
    let pad = FIRST_FILL_REWIND_SLACK + 1;
    let mut cache = RotatingKVCache::new(window);
    let _ = cache.update_and_fetch(tensor(300, 3), tensor(300, 4));
    let before = (cache.offset, cache.idx, stored_len(&cache));
    let err = cache
        .rewind_padded_prefill(pad)
        .expect_err("the positions the rewind would expose were not stored");
    assert!(err.contains("FullFirstFillScope"), "{err}");
    assert_eq!(
        (cache.offset, cache.idx, stored_len(&cache)),
        before,
        "a refused rewind leaves the cache untouched"
    );

    let mut scoped = RotatingKVCache::new(window);
    {
        let _scope = FullFirstFillScope::enter();
        let _ = scoped.update_and_fetch(tensor(300, 3), tensor(300, 4));
    }
    scoped
        .rewind_padded_prefill(pad)
        .expect("a scoped first fill keeps every position");
    assert_eq!(scoped.offset, 300 - pad);
}

#[test]
fn speculative_buffering_after_a_trimmed_first_fill_matches_the_full_fill() {
    let mut twin = Twin::new(32, KVCacheMode::Fp16);
    twin.append(300);
    twin.trimmed
        .enable_speculative_buffer(4)
        .expect("FP16 buffering");
    twin.full
        .enable_speculative_buffer(4)
        .expect("FP16 buffering");
    assert_eq!(
        twin.trimmed.speculative_ring_cursor(),
        twin.full.speculative_ring_cursor()
    );
    assert_eq!(
        twin.trimmed.snapshot_state(),
        twin.full.snapshot_state(),
        "buffering re-lays both caches out as the same window"
    );
    // Verify block, rollback, decode: the MTP round.
    twin.append(4);
    twin.trim(3);
    twin.decode(3);
    twin.append(4);
    twin.trim(1);
    twin.decode(2);
}

#[test]
fn a_snapshot_of_a_trimmed_first_fill_restores_and_decodes() {
    let window = 32;
    let mut cache = RotatingKVCache::new(window);
    let _ = cache.update_and_fetch(tensor(300, 3), tensor(300, 4));
    let state = cache.snapshot_state();
    assert!(!state.can_truncate_to(299), "as before, no partial restore");
    let mut restored = RotatingKVCache::new(window);
    restored
        .restore_snapshot_state(
            state,
            RotatingKVCacheSnapshotBuffers::fp16(
                Some(ffi::copy(cache.keys.as_ref().unwrap())),
                Some(ffi::copy(cache.values.as_ref().unwrap())),
            ),
        )
        .expect("idx equals the stored length");
    for step in 0..10u32 {
        let (ka, _) = cache.update_and_fetch(tensor(1, 50 + step), tensor(1, 60 + step));
        let (kb, _) = restored.update_and_fetch(tensor(1, 50 + step), tensor(1, 60 + step));
        assert_eq!(to_f32(&ka), to_f32(&kb), "decode step {step}");
    }
}
