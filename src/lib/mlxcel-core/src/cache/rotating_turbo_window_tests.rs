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

//! A Turbo4Asym `RotatingKVCache` must return the same keys as an FP16 one
//! (issue #1793).
//!
//! Keys are stored as FP16 in both modes, so for FP16 inputs the returned key
//! tensors must be identical, not merely close, after every operation: the
//! window rule (which prior slots a multi-token append keeps, in which order)
//! is a property of the ring, not of how V is stored. The returned V is
//! compared with a reconstruction tolerance instead, which still catches a
//! sidecar sliced differently from the keys (that pairs a key with another
//! token's value). The turbo multi-token append used to concatenate the whole
//! physical buffer, so it returned decode-growth zero slots, rolled-back draft
//! keys and pre-wrap physical order, and clamped an over-window result to
//! `max_size` against the #678 rule.

use super::*;
use crate::dtype;
use crate::ffi;

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

/// The same ring in both storage modes, driven by identical operations.
struct Pair {
    fp16: RotatingKVCache,
    turbo: RotatingKVCache,
    seed: u32,
}

impl Pair {
    fn new(max_size: i32) -> Self {
        Self {
            fp16: RotatingKVCache::new_with_mode(max_size, KVCacheMode::Fp16),
            turbo: RotatingKVCache::new_with_mode(max_size, KVCacheMode::Turbo4Asym),
            seed: 1,
        }
    }

    /// Append `tokens` to both caches, assert that they return the same keys
    /// and matching values, and return the key axis length.
    fn append(&mut self, tokens: i32) -> i32 {
        self.seed += 2;
        let (k16, v16) = self
            .fp16
            .update_and_fetch(tensor(tokens, self.seed), tensor(tokens, self.seed + 1));
        let (kt, vt) = self
            .turbo
            .update_and_fetch(tensor(tokens, self.seed), tensor(tokens, self.seed + 1));
        let returned = ffi::array_shape(&k16)[2];
        assert_eq!(
            ffi::array_shape(&kt)[2],
            returned,
            "turbo returned {} keys where FP16 returned {returned} (append of {tokens}, \
             offset {})",
            ffi::array_shape(&kt)[2],
            self.fp16.offset
        );
        assert_eq!(
            to_f32(&kt),
            to_f32(&k16),
            "turbo keys differ from FP16 keys after an append of {tokens}"
        );
        assert_eq!(ffi::array_shape(&vt), ffi::array_shape(&v16));
        let (v16, vt) = (to_f32(&v16), to_f32(&vt));
        let num: f32 = v16.iter().zip(&vt).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = v16.iter().map(|a| a * a).sum();
        let rel = (num / den.max(1e-12)).sqrt();
        assert!(
            rel < 0.15,
            "turbo V is not the FP16 V reconstructed (relative L2 {rel:.4}); a sidecar was \
             sliced differently from the keys"
        );
        assert_eq!(self.turbo.offset, self.fp16.offset);
        assert_eq!(self.turbo.visible_len(), self.fp16.visible_len());
        returned
    }

    fn trim(&mut self, n: i32) {
        assert_eq!(self.fp16.trim(n), self.turbo.trim(n));
    }
}

/// Decode grows the turbo buffer to the window ahead of `offset`; the next
/// append must take only the `visible_len()` prior keys, not the zero slots.
#[test]
fn turbo_append_after_decode_growth_takes_visible_prior_keys() {
    let mut pair = Pair::new(64);
    pair.append(6);
    pair.append(1);
    assert!(
        ffi::array_shape(pair.turbo.keys.as_ref().unwrap())[2] > pair.turbo.offset,
        "fixture must reach physical > offset"
    );
    let live = pair.turbo.visible_len();
    assert_eq!(pair.append(3), live + 3);
    pair.append(1);
}

/// A speculative rollback rewinds `offset`/`idx` without moving data; the
/// next verify append must not return the rolled-back draft keys.
#[test]
fn turbo_append_after_trim_rollback_drops_rolled_back_keys() {
    let mut pair = Pair::new(64);
    pair.append(6);
    pair.append(1);
    pair.append(4);
    pair.trim(3);
    let live = pair.turbo.visible_len();
    assert_eq!(pair.append(2), live + 2);
    // A rollback of single-token decode writes, then a block again.
    pair.append(1);
    pair.append(1);
    pair.trim(2);
    let live = pair.turbo.visible_len();
    assert_eq!(pair.append(3), live + 3);
    pair.append(1);
}

/// After the ring wraps, the prior keys must come back in chronological
/// order, newest `max_size - 1` of them.
#[test]
fn turbo_append_after_wrap_is_chronological() {
    let mut pair = Pair::new(32);
    pair.append(4);
    for _ in 0..40 {
        pair.append(1);
    }
    assert!(pair.turbo.offset > 32, "fixture must wrap the ring");
    assert_eq!(pair.append(3), 31 + 3);
    pair.append(1);
    pair.append(5);
}

/// An over-window append keeps `max_size - 1` prior keys plus every new one
/// (#678); the following decode step sees the newest `max_size`.
#[test]
fn turbo_over_window_append_returns_every_new_key() {
    let mut pair = Pair::new(32);
    pair.append(40);
    pair.append(1);
    pair.append(10);
    assert_eq!(pair.append(40), 31 + 40);
    assert_eq!(pair.append(1), 32);
    pair.append(2);
}

/// A first install longer than the window stores every token; the next
/// decode step must still see exactly the newest `max_size` keys.
#[test]
fn turbo_decode_after_over_window_install_sees_the_window() {
    let mut pair = Pair::new(32);
    pair.append(40);
    assert_eq!(pair.append(1), 32);
    for _ in 0..40 {
        assert_eq!(pair.append(1), 32);
    }
    pair.append(4);
}
