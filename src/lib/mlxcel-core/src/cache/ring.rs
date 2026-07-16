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

//! Ring-sliding KV cache for the Unlimited-OCR decoder.
//!
//! Semantics mirror the reference `SlidingWindowLlamaAttention` ring buffer in
//! baidu's `modeling_deepseekv2.py`:
//!
//! - **Prefill** (`q_len > 1`): the full prompt + image KV is appended and
//!   retained *permanently*. The physical length after prefill is recorded as
//!   `prefill_len` on the first decode step.
//! - **Warmup decode**: while the physical length is below `prefill_len +
//!   window`, decode tokens are appended (the buffer grows). When it reaches
//!   `prefill_len + window`, `ring_pos` initialises to 0.
//! - **Steady-state decode**: each new token's K/V *overwrites* physical slot
//!   `prefill_len + ring_pos`, then `ring_pos = (ring_pos + 1) % window`. The
//!   physical length stays `prefill_len + window` forever.
//!
//! RoPE positions are absolute and keep increasing past the window: the caller
//! reads [`RingSlidingKVCache::offset`] *before* the update and applies RoPE to
//! Q and the fresh K at that absolute position, so every retained key carries
//! its write-time rotation. Because RoPE bakes the absolute position into each
//! vector and softmax over the key axis is order-invariant, the circular
//! physical ordering of the ring slots does not affect attention output.
//!
//! The mask policy comes from [`RingSlidingKVCache::prefill_causal`]: causal
//! during prefill (`q_len > 1`), and `None` during steady single-token decode
//! (every retained entry is attendable).

use crate::concatenate;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;

/// Ring-sliding KV cache holding the full prefill KV plus a bounded circular
/// window of the most recent decode tokens. See the module docs for the exact
/// prefill / warmup / steady-state contract.
///
/// One instance is created per transformer layer and stored model-owned (the
/// Unlimited-OCR decoder keeps its ring caches inside the model rather than in
/// the external `KVCache` slice, so the standard `KVCache`-based prompt-cache
/// and distributed-handoff paths are declined for this family).
pub struct RingSlidingKVCache {
    /// Physical K buffer `[B, H, phys, D]`, or `None` before the first write.
    keys: Option<UniquePtr<MlxArray>>,
    /// Physical V buffer `[B, H, phys, D]`, or `None` before the first write.
    values: Option<UniquePtr<MlxArray>>,
    /// Absolute write position, monotonically increasing. Used as the RoPE
    /// position for new Q/K tokens; keeps growing even after the physical
    /// length is bounded by the ring window.
    offset: i32,
    /// Sliding window size `W` (number of decode slots that rotate).
    window: i32,
    /// Physical length recorded at the prefill -> decode boundary, or `-1`
    /// before any decode step. Everything below `prefill_len` is retained
    /// permanently; the ring rotates within `[prefill_len, prefill_len + W)`.
    prefill_len: i32,
    /// Circular write cursor within the ring, or `-1` until the warmup fills
    /// the window.
    ring_pos: i32,
}

impl RingSlidingKVCache {
    /// Create an empty ring cache with sliding window `window` (clamped to
    /// `>= 1`).
    #[must_use]
    pub fn new(window: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            window: window.max(1),
            prefill_len: -1,
            ring_pos: -1,
        }
    }

    /// Absolute RoPE position for the next token(s). Read this *before*
    /// [`Self::update_and_fetch`] and apply RoPE at this offset.
    #[must_use]
    #[inline]
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Sliding window size `W`.
    #[must_use]
    #[inline]
    pub fn window(&self) -> i32 {
        self.window
    }

    /// Recorded prefill length once decode has begun, else `None`.
    #[must_use]
    #[inline]
    pub fn prefill_len(&self) -> Option<i32> {
        (self.prefill_len >= 0).then_some(self.prefill_len)
    }

    /// Whether the attention step for `q_len` new tokens must apply the causal
    /// prefill mask. `true` for prefill (`q_len > 1`); `false` for steady
    /// single-token decode, where every retained entry is attendable and no
    /// mask is needed.
    #[must_use]
    #[inline]
    pub fn prefill_causal(&self, q_len: i32) -> bool {
        q_len > 1
    }

    /// Current physical length of the K/V buffers (`0` when empty).
    #[must_use]
    #[inline]
    pub fn physical_len(&self) -> i32 {
        match &self.keys {
            Some(k) => ffi::array_shape(k)[2],
            None => 0,
        }
    }

    /// Clear all state for a fresh sequence.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.prefill_len = -1;
        self.ring_pos = -1;
    }

    /// Append `new_keys` / `new_values` (`[B, H, L, D]`) at the end of the
    /// physical buffer, growing it by `L`.
    fn append(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        self.keys = Some(match self.keys.take() {
            Some(k) => concatenate(&k, &new_keys, 2),
            None => new_keys,
        });
        self.values = Some(match self.values.take() {
            Some(v) => concatenate(&v, &new_values, 2),
            None => new_values,
        });
    }

    /// Overwrite a single physical slot with one token's K/V.
    fn overwrite_slot(&mut self, slot: i32, k_tok: &MlxArray, v_tok: &MlxArray) {
        let k = self.keys.take().expect("ring keys present in steady state");
        let v = self
            .values
            .take()
            .expect("ring values present in steady state");
        let ks = ffi::array_shape(&k);
        let vs = ffi::array_shape(&v);
        self.keys = Some(ffi::slice_update(
            &k,
            k_tok,
            &[0, 0, slot, 0],
            &[ks[0], ks[1], slot + 1, ks[3]],
        ));
        self.values = Some(ffi::slice_update(
            &v,
            v_tok,
            &[0, 0, slot, 0],
            &[vs[0], vs[1], slot + 1, vs[3]],
        ));
    }

    /// Slice a single token `[B, H, 1, D]` out of an `[B, H, L, D]` tensor.
    fn token_slice(src: &MlxArray, t: i32) -> UniquePtr<MlxArray> {
        let s = ffi::array_shape(src);
        ffi::slice(src, &[0, 0, t, 0], &[s[0], s[1], t + 1, s[3]])
    }

    /// Return the full physical K/V window for attention (a lazy view over the
    /// whole buffer).
    fn full_view(&self) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let k = self.keys.as_ref().expect("ring keys present after update");
        let v = self
            .values
            .as_ref()
            .expect("ring values present after update");
        let ks = ffi::array_shape(k);
        let vs = ffi::array_shape(v);
        (
            ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], ks[2], ks[3]]),
            ffi::slice(v, &[0, 0, 0, 0], &[vs[0], vs[1], vs[2], vs[3]]),
        )
    }

    /// Write `new_keys` / `new_values` (`[B, H, L, D]`, already RoPE-rotated at
    /// [`Self::offset`]) into the cache following the prefill / warmup /
    /// steady-state ring contract, then return the full physical K/V window for
    /// attention.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let l = ffi::array_shape(&new_keys)[2];

        // Prefill: no decode step has been seen yet.
        if self.prefill_len < 0 {
            if l > 1 {
                // Prefill chunk: append and keep growing (still prefill).
                self.append(new_keys, new_values);
                self.offset += l;
                return self.full_view();
            }
            // First decode token: freeze the prefill boundary at the current
            // physical length before writing this token.
            self.prefill_len = self.physical_len();
        }

        let bound = self.prefill_len + self.window;
        if self.physical_len() < bound {
            // Warmup: append decode tokens until the ring window is full.
            self.append(new_keys, new_values);
            self.offset += l;
            if self.physical_len() >= bound {
                self.ring_pos = 0;
            }
            return self.full_view();
        }

        // Steady state: overwrite ring slots in place, absolute offset keeps
        // advancing so subsequent RoPE positions stay correct.
        if self.ring_pos < 0 {
            self.ring_pos = 0;
        }
        for t in 0..l {
            let slot = self.prefill_len + self.ring_pos;
            let k_tok = Self::token_slice(&new_keys, t);
            let v_tok = Self::token_slice(&new_values, t);
            self.overwrite_slot(slot, &k_tok, &v_tok);
            self.ring_pos = (self.ring_pos + 1) % self.window;
        }
        self.offset += l;
        self.full_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a device array back to a flat `Vec<f32>`.
    fn to_f32(arr: &MlxArray) -> Vec<f32> {
        ffi::eval(arr);
        ffi::array_to_raw_bytes(arr)
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Build a `[1, 1, n, 1]` token chunk for positions `range`. Keys carry the
    /// position, values carry `position * 10` so K/V alignment is checkable.
    fn tokens(range: std::ops::Range<i32>) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let k: Vec<f32> = range.clone().map(|p| p as f32).collect();
        let v: Vec<f32> = range.clone().map(|p| (p * 10) as f32).collect();
        let n = k.len() as i32;
        (
            ffi::from_slice_f32(&k, &[1, 1, n, 1]),
            ffi::from_slice_f32(&v, &[1, 1, n, 1]),
        )
    }

    /// A plain "keep everything" reference cache: append forever, never evict.
    #[derive(Default)]
    struct FullCache {
        keys: Option<UniquePtr<MlxArray>>,
        values: Option<UniquePtr<MlxArray>>,
    }

    impl FullCache {
        fn push(&mut self, k: UniquePtr<MlxArray>, v: UniquePtr<MlxArray>) {
            match (&self.keys, &self.values) {
                (Some(pk), Some(pv)) => {
                    self.keys = Some(concatenate(pk, &k, 2));
                    self.values = Some(concatenate(pv, &v, 2));
                }
                _ => {
                    self.keys = Some(k);
                    self.values = Some(v);
                }
            }
        }
    }

    #[test]
    fn ring_reports_absolute_offset_and_prefill_boundary() {
        let mut cache = RingSlidingKVCache::new(4);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.prefill_len(), None);

        let (k, v) = tokens(0..6);
        cache.update_and_fetch(k, v); // prefill of 6 tokens
        assert_eq!(cache.offset(), 6);
        assert_eq!(
            cache.prefill_len(),
            None,
            "prefill_len frozen only at first decode"
        );

        // Ten decode steps; offset must stay absolute (never bounded).
        for p in 6..16 {
            let (k, v) = tokens(p..p + 1);
            cache.update_and_fetch(k, v);
        }
        assert_eq!(cache.offset(), 16);
        assert_eq!(cache.prefill_len(), Some(6));
        // Physical length is bounded: prefill (6) + window (4).
        assert_eq!(cache.physical_len(), 10);
    }

    #[test]
    fn ring_matches_full_cache_until_eviction_begins() {
        let window = 4;
        let prefill = 5;
        let mut ring = RingSlidingKVCache::new(window);
        let mut full = FullCache::default();

        let (rk, rv) = tokens(0..prefill);
        let (fk, fv) = tokens(0..prefill);
        ring.update_and_fetch(rk, rv);
        full.push(fk, fv);

        // Warmup decode: for the first `window` decode steps the ring buffer is
        // byte-identical to the keep-everything reference, so attention over
        // either is identical.
        for step in 0..window {
            let pos = prefill + step;
            let (rk, rv) = tokens(pos..pos + 1);
            let (fk, fv) = tokens(pos..pos + 1);
            let (ring_k, ring_v) = ring.update_and_fetch(rk, rv);
            full.push(fk, fv);
            let full_k = to_f32(full.keys.as_ref().unwrap());
            let full_v = to_f32(full.values.as_ref().unwrap());
            assert_eq!(
                to_f32(&ring_k),
                full_k,
                "keys diverged during warmup at decode step {step}"
            );
            assert_eq!(
                to_f32(&ring_v),
                full_v,
                "values diverged during warmup at decode step {step}"
            );
        }
    }

    #[test]
    fn ring_retains_prefill_and_last_window_after_wraparound() {
        let window = 8;
        let prefill = 6;
        let mut ring = RingSlidingKVCache::new(window);

        let (k, v) = tokens(0..prefill);
        ring.update_and_fetch(k, v);

        // Decode well past eviction: 3x the window length of steady-state steps
        // plus the warmup window.
        let decode_steps = window * 3 + window;
        let mut last_out: Option<(Vec<f32>, Vec<f32>)> = None;
        for step in 0..decode_steps {
            let pos = prefill + step;
            let (k, v) = tokens(pos..pos + 1);
            let (ok, ov) = ring.update_and_fetch(k, v);
            last_out = Some((to_f32(&ok), to_f32(&ov)));
        }

        let (k, v) = last_out.expect("decode produced output");
        // Physical length stays bounded.
        assert_eq!(k.len() as i32, prefill + window);
        // All entries finite (no corruption over long generation).
        assert!(k.iter().all(|x| x.is_finite()), "keys not finite: {k:?}");
        assert!(v.iter().all(|x| x.is_finite()), "values not finite: {v:?}");
        // K/V pairs stay aligned (value == key * 10) across the whole buffer.
        for (kk, vv) in k.iter().zip(&v) {
            assert_eq!(*vv, kk * 10.0, "K/V misaligned: k={k:?} v={v:?}");
        }

        // Prefill region (permanent) holds positions 0..prefill in order.
        let prefill_region: Vec<i32> = k[..prefill as usize].iter().map(|x| *x as i32).collect();
        assert_eq!(prefill_region, (0..prefill).collect::<Vec<_>>());

        // Ring region holds exactly the last `window` decode positions (in some
        // circular order): decode covered positions prefill..prefill+decode_steps.
        let last_pos = prefill + decode_steps; // exclusive upper bound
        let mut ring_region: Vec<i32> = k[prefill as usize..].iter().map(|x| *x as i32).collect();
        ring_region.sort_unstable();
        let expected: Vec<i32> = (last_pos - window..last_pos).collect();
        assert_eq!(
            ring_region, expected,
            "ring region must hold the last {window} decode tokens"
        );
    }

    #[test]
    fn ring_reset_clears_state() {
        let mut cache = RingSlidingKVCache::new(4);
        let (k, v) = tokens(0..3);
        cache.update_and_fetch(k, v);
        let (k, v) = tokens(3..4);
        cache.update_and_fetch(k, v);
        assert!(cache.physical_len() > 0);

        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.physical_len(), 0);
        assert_eq!(cache.prefill_len(), None);

        // Reusable after reset.
        let (k, v) = tokens(0..4);
        let (ok, _ov) = cache.update_and_fetch(k, v);
        assert_eq!(to_f32(&ok), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(cache.offset(), 4);
    }
}
