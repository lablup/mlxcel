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

//! Tests for the cache-owned attention entry (issue #2171).
//!
//! The dense half pins [`KVCache::attend`] and [`attend_batched`] against the
//! `update_and_fetch` plus SDPA pair the models used to spell out, op for op,
//! so the outputs are bit-identical. The paged half drives a real
//! [`PagedBlockPool`] through [`KVCache::new_paged`] and checks that an
//! unmasked single-token step is served by the pooled entry (the production
//! counters move) while a masked one is not, and that both agree with the
//! gather-then-SDPA baseline computed from the same post-write pool state.

use std::cell::RefCell;
use std::rc::Rc;

use super::*;
use crate::cache::{
    KVCacheMode, PagedBlockPool, PagedKvLayout, PagedSequenceState, paged_batch_decode_stats,
};
use crate::dtype;

const PAGE: usize = 32;
const HEADS: i32 = 4;
const KV_HEADS: i32 = 2;
const DIM: i32 = 64;
const SCALE: f32 = 0.125;

/// xorshift64* in [-1, 1), deterministic so a failure reproduces exactly.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn random_array(rng: &mut Rng, shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|d| *d as usize).product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let f32_arr = ffi::from_slice_f32(&data, shape);
    ffi::astype(&f32_arr, dtype::FLOAT16)
}

fn to_vec_f32(arr: &MlxArray) -> Vec<f32> {
    let as_f32 = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&as_f32);
    ffi::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

/// Bit patterns of `arr` widened to f32, for comparisons where NaN must count.
///
/// `f32` equality treats every NaN as unequal to itself, so two outputs that
/// hold NaN in the same slot fail `assert_eq!` on their values even when they
/// are the same bits. Comparing `to_bits` makes "identical" mean identical.
fn to_bits_f32(arr: &MlxArray) -> Vec<u32> {
    to_vec_f32(arr).into_iter().map(f32::to_bits).collect()
}

fn relative_rms(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "outputs must have the same length");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    ((num / den.max(1e-12)) as f32).sqrt()
}

/// `(q, k, v)` for one step of `len` tokens over `batch` rows.
fn step(
    rng: &mut Rng,
    batch: i32,
    len: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    (
        random_array(rng, &[batch, HEADS, len, DIM]),
        random_array(rng, &[batch, KV_HEADS, len, DIM]),
        random_array(rng, &[batch, KV_HEADS, len, DIM]),
    )
}

/// The pair every model used to spell out: dense update, live window, SDPA.
fn reference(
    cache: &mut KVCache,
    q: &MlxArray,
    k: UniquePtr<MlxArray>,
    v: UniquePtr<MlxArray>,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let (ck, cv) = cache.update_and_fetch(k, v);
    let q_len = ffi::array_shape(q)[2];
    if q_len > 1 && mask.is_none() {
        crate::causal_attention(q, &ck, &cv, SCALE, 0.0, 0)
    } else {
        crate::layers::attention(q, &ck, &cv, SCALE, mask, 0.0, 0)
    }
}

fn fresh_pool(
    num_layers: usize,
) -> (
    Rc<RefCell<PagedBlockPool>>,
    Vec<Rc<RefCell<PagedSequenceState>>>,
) {
    let bytes_per_block = PAGE * KV_HEADS as usize * DIM as usize * 2;
    let layout = PagedKvLayout::uniform(num_layers, PAGE, bytes_per_block).expect("valid layout");
    let mut pool = PagedBlockPool::new(layout.clone());
    pool.set_slab_blocks(512)
        .expect("fresh pool has no storage");
    let states = (0..4)
        .map(|_| Rc::new(RefCell::new(PagedSequenceState::new(&layout))))
        .collect();
    (Rc::new(RefCell::new(pool)), states)
}

#[test]
fn dense_prefill_then_decode_matches_update_and_fetch_plus_sdpa() {
    let mut rng = Rng::new(0x2171);
    let mut got = KVCache::new();
    let mut want = KVCache::new();

    // Causal prefill of 8 tokens, no explicit mask.
    let (q, k, v) = step(&mut rng, 1, 8);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    let a = got.attend(&q, k, v, SCALE, None);
    let b = reference(&mut want, &q, k2, v2, None);
    assert_eq!(to_vec_f32(&a), to_vec_f32(&b), "prefill output differs");
    assert_eq!(got.offset, want.offset);
    assert_eq!(got.offset, 8);

    // Three decode steps.
    for _ in 0..3 {
        let (q, k, v) = step(&mut rng, 1, 1);
        let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
        let a = got.attend(&q, k, v, SCALE, None);
        let b = reference(&mut want, &q, k2, v2, None);
        assert_eq!(to_vec_f32(&a), to_vec_f32(&b), "decode output differs");
        assert_eq!(got.offset, want.offset);
    }
    assert_eq!(got.offset, 11);
}

#[test]
fn dense_masked_step_matches_masked_sdpa() {
    let mut rng = Rng::new(0x2172);
    let mut got = KVCache::new();
    let mut want = KVCache::new();
    let (q, k, v) = step(&mut rng, 1, 4);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    got.attend(&q, k, v, SCALE, None);
    reference(&mut want, &q, k2, v2, None);

    // A two-token verify step with an explicit causal mask over 6 keys.
    let (q, k, v) = step(&mut rng, 1, 2);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    let mask = crate::utils::create_causal_mask(2, 4);
    let a = got.attend(&q, k, v, SCALE, Some(&mask));
    let b = reference(&mut want, &q, k2, v2, Some(&mask));
    assert_eq!(to_vec_f32(&a), to_vec_f32(&b), "masked output differs");
    assert_eq!(got.offset, 6);
}

#[test]
fn batched_dense_matches_per_row_attend() {
    let mut rng = Rng::new(0x2173);
    let mut c0 = KVCache::new();
    let mut c1 = KVCache::new();
    let mut r0 = KVCache::new();
    let mut r1 = KVCache::new();
    for (c, r) in [(&mut c0, &mut r0), (&mut c1, &mut r1)] {
        let (q, k, v) = step(&mut rng, 1, 5);
        let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
        c.attend(&q, k, v, SCALE, None);
        reference(r, &q, k2, v2, None);
    }

    let (q, k, v) = step(&mut rng, 2, 1);
    let mut rows = Vec::new();
    for (b, r) in [&mut r0, &mut r1].into_iter().enumerate() {
        let q_b = slice_row(&q, b);
        let k_b = slice_row(&k, b);
        let v_b = slice_row(&v, b);
        rows.push(reference(r, &q_b, k_b, v_b, None));
    }
    let want = crate::concatenate(&rows[0], &rows[1], 0);

    let mut caches: Vec<&mut KVCache> = vec![&mut c0, &mut c1];
    let got = attend_batched(&q, &k, &v, &mut caches, SCALE, None);
    assert_eq!(ffi::array_shape(&got), vec![2, HEADS, 1, DIM]);
    assert_eq!(to_vec_f32(&got), to_vec_f32(&want));
    assert_eq!(c0.offset, 6);
    assert_eq!(c1.offset, 6);
}

#[test]
fn batched_masked_rows_get_their_own_mask_row() {
    let mut rng = Rng::new(0x2174);
    let mut c0 = KVCache::new();
    let mut c1 = KVCache::new();
    let mut r0 = KVCache::new();
    let mut r1 = KVCache::new();

    // Padded batched prefill: 3 tokens, mask `[B, 3, 3]`, row 1 masks its
    // first key so the two rows must not share a mask.
    //
    // Query 0 of row 1 is left with no unmasked key (key 0 by the row mask,
    // keys 1 and 2 causally). That is deliberate: it is the row whose output
    // depends most on which mask row it got. What a fully masked query yields
    // is backend-defined. Metal's vector SDPA skips the masked keys and writes
    // 0; on ROCm an array-masked SDPA at this head dim has no fused kernel and
    // MLX's graph fallback softmaxes an all -inf row (-1e9 rounds to -inf in
    // f16) into NaN. The batched and per-row paths run the same SDPA either
    // way, so the comparison below is bitwise: the same NaN in the same slot is
    // a match, a NaN on one side only is not.
    let (q, k, v) = step(&mut rng, 2, 3);
    let mut mask_data = vec![0.0f32; 2 * 3 * 3];
    for t in 0..3 {
        for s in 0..3 {
            if s > t {
                mask_data[t * 3 + s] = -1e9;
                mask_data[9 + t * 3 + s] = -1e9;
            }
        }
    }
    for t in 0..3 {
        mask_data[9 + t * 3] = -1e9;
    }
    let mask = ffi::astype(&ffi::from_slice_f32(&mask_data, &[2, 3, 3]), dtype::FLOAT16);

    let mut rows = Vec::new();
    for (b, r) in [&mut r0, &mut r1].into_iter().enumerate() {
        let q_b = slice_row(&q, b);
        let k_b = slice_row(&k, b);
        let v_b = slice_row(&v, b);
        let m_b = ffi::squeeze_axis(
            &ffi::slice(&mask, &[b as i32, 0, 0], &[b as i32 + 1, 3, i32::MAX]),
            0,
        );
        rows.push(reference(r, &q_b, k_b, v_b, Some(&m_b)));
    }
    let want = crate::concatenate(&rows[0], &rows[1], 0);

    let mut caches: Vec<&mut KVCache> = vec![&mut c0, &mut c1];
    let got = attend_batched(&q, &k, &v, &mut caches, SCALE, Some(&mask));
    assert_eq!(to_bits_f32(&got), to_bits_f32(&want));
}

#[test]
fn paged_cache_single_token_goes_through_the_pooled_entry() {
    let (pool, states) = fresh_pool(1);
    let mut rng = Rng::new(0x2175);
    let mut paged = KVCache::new_paged(pool.clone(), states[0].clone(), 0);
    let mut baseline = KVCache::new_paged(pool, states[1].clone(), 0);

    // Prefill both sequences with the same 40 tokens (crosses a page).
    let (q, k, v) = step(&mut rng, 1, 40);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    let before = paged_batch_decode_stats();
    let a = paged.attend(&q, k, v, SCALE, None);
    let b = reference(&mut baseline, &q, k2, v2, None);
    let after = paged_batch_decode_stats();
    assert_eq!(to_vec_f32(&a), to_vec_f32(&b), "paged prefill differs");
    assert_eq!(
        after.v2_launches + after.gather_fallbacks,
        before.v2_launches + before.gather_fallbacks,
        "a multi-token step must not take the pooled single-token entry"
    );
    assert_eq!(paged.offset, 40);

    // Unmasked decode: the pooled entry serves it (one launch or one gather
    // fallback, either way a served step) and agrees with the gather path.
    let (q, k, v) = step(&mut rng, 1, 1);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    let before = paged_batch_decode_stats();
    let a = paged.attend(&q, k, v, SCALE, None);
    let after = paged_batch_decode_stats();
    assert_eq!(
        after.v2_launches + after.gather_fallbacks,
        before.v2_launches + before.gather_fallbacks + 1,
        "an unmasked single-token step on a pool-backed cache is served by the pooled entry"
    );
    let b = reference(&mut baseline, &q, k2, v2, None);
    let rms = relative_rms(&to_vec_f32(&a), &to_vec_f32(&b));
    assert!(
        rms < 2e-2,
        "paged decode diverged from the gather baseline: rms {rms}"
    );
    assert_eq!(paged.offset, 41);
    assert_eq!(baseline.offset, 41);

    // Masked single token (a verify step): never the pooled kernel, and the
    // pool still receives the row through the dense route's intercept.
    let (q, k, v) = step(&mut rng, 1, 1);
    let (k2, v2) = (ffi::copy(&k), ffi::copy(&v));
    let mask = crate::utils::create_causal_mask(1, 41);
    let before = paged_batch_decode_stats();
    let a = paged.attend(&q, k, v, SCALE, Some(&mask));
    let after = paged_batch_decode_stats();
    assert_eq!(
        after.v2_launches + after.gather_fallbacks,
        before.v2_launches + before.gather_fallbacks,
        "a masked step must not take the pooled single-token entry"
    );
    let b = reference(&mut baseline, &q, k2, v2, Some(&mask));
    assert_eq!(to_vec_f32(&a), to_vec_f32(&b), "masked paged step differs");
    assert_eq!(paged.offset, 42);
}

#[test]
fn batched_paged_decode_is_one_pooled_launch() {
    let (pool, states) = fresh_pool(1);
    let mut rng = Rng::new(0x2176);
    let mut c0 = KVCache::new_paged(pool.clone(), states[0].clone(), 0);
    let mut c1 = KVCache::new_paged(pool, states[1].clone(), 0);
    for c in [&mut c0, &mut c1] {
        let (q, k, v) = step(&mut rng, 1, 12);
        c.attend(&q, k, v, SCALE, None);
    }
    let (q, k, v) = step(&mut rng, 2, 1);
    let before = paged_batch_decode_stats();
    let mut caches: Vec<&mut KVCache> = vec![&mut c0, &mut c1];
    let out = attend_batched(&q, &k, &v, &mut caches, SCALE, None);
    let after = paged_batch_decode_stats();
    assert_eq!(ffi::array_shape(&out), vec![2, HEADS, 1, DIM]);
    assert_eq!(
        after.v2_launches + after.gather_fallbacks,
        before.v2_launches + before.gather_fallbacks + 1,
        "the batch is served by exactly one pooled call"
    );
    assert_eq!(c0.offset, 13);
    assert_eq!(c1.offset, 13);
}

/// Batched decode over Turbo4Asym caches takes the same dequant-first route a
/// lone sequence takes (#2171): each row of [`attend_batched`] is bit-identical
/// to that row's single-sequence dequant-first call, and stays within float
/// noise of the full-dequant `update_and_fetch` plus SDPA pair the batched
/// loop ran before. Pins the behavior change ADR 0008 records, so a future
/// batched Turbo route cannot silently diverge from the single-sequence one.
#[test]
fn batched_turbo4_asym_rows_match_single_sequence_dequant_first() {
    let mut rng = Rng::new(0x2174);
    let mode = KVCacheMode::Turbo4Asym;
    let mut c0 = KVCache::new_with_mode(mode);
    let mut c1 = KVCache::new_with_mode(mode);
    let mut s0 = KVCache::new_with_mode(mode);
    let mut s1 = KVCache::new_with_mode(mode);
    let mut f0 = KVCache::new_with_mode(mode);
    let mut f1 = KVCache::new_with_mode(mode);
    for (c, s, f) in [(&mut c0, &mut s0, &mut f0), (&mut c1, &mut s1, &mut f1)] {
        let (q, k, v) = step(&mut rng, 1, 6);
        c.attend(&q, ffi::copy(&k), ffi::copy(&v), SCALE, None);
        s.attend(&q, ffi::copy(&k), ffi::copy(&v), SCALE, None);
        reference(f, &q, k, v, None);
    }

    let (q, k, v) = step(&mut rng, 2, 1);
    let mut single_rows = Vec::new();
    let mut full_rows = Vec::new();
    for (b, (s, f)) in [(&mut s0, &mut f0), (&mut s1, &mut f1)]
        .into_iter()
        .enumerate()
    {
        let q_b = slice_row(&q, b);
        single_rows.push(s.update_and_turbo4_asym_dequant_sdpa_attention(
            &q_b,
            slice_row(&k, b),
            slice_row(&v, b),
            SCALE,
            None,
        ));
        full_rows.push(reference(f, &q_b, slice_row(&k, b), slice_row(&v, b), None));
    }
    let single = crate::concatenate(&single_rows[0], &single_rows[1], 0);
    let full = crate::concatenate(&full_rows[0], &full_rows[1], 0);

    let mut caches: Vec<&mut KVCache> = vec![&mut c0, &mut c1];
    let got = attend_batched(&q, &k, &v, &mut caches, SCALE, None);
    let got = to_vec_f32(&got);
    assert_eq!(
        got,
        to_vec_f32(&single),
        "batched row left the dequant-first route"
    );
    let drift = relative_rms(&got, &to_vec_f32(&full));
    assert!(
        drift < 1e-2,
        "dequant-first drifted {drift} from the full-dequant reference"
    );
    assert_eq!(c0.offset, 7);
    assert_eq!(c1.offset, 7);
}

#[test]
#[should_panic(expected = "empty batch")]
fn batched_attend_rejects_an_empty_batch() {
    let mut rng = Rng::new(0x2177);
    let (q, k, v) = step(&mut rng, 1, 1);
    let mut caches: Vec<&mut KVCache> = Vec::new();
    attend_batched(&q, &k, &v, &mut caches, SCALE, None);
}

/// A one-row batch is what the lookahead prime and `forward_batched` hand over
/// when no `DecodeBatchContext` exists. Dense storage matches the per-row
/// `attend` bit for bit.
#[test]
fn batched_dense_one_row_matches_attend() {
    let mut rng = Rng::new(0x2178);
    let mut batched = KVCache::new();
    let mut single = KVCache::new();
    let (q, k, v) = step(&mut rng, 1, 5);
    batched.attend(&q, ffi::copy(&k), ffi::copy(&v), SCALE, None);
    single.attend(&q, k, v, SCALE, None);

    let (q, k, v) = step(&mut rng, 1, 1);
    let want = single.attend(&q, ffi::copy(&k), ffi::copy(&v), SCALE, None);
    let mut caches: Vec<&mut KVCache> = vec![&mut batched];
    let got = attend_batched(&q, &k, &v, &mut caches, SCALE, None);
    assert_eq!(ffi::array_shape(&got), vec![1, HEADS, 1, DIM]);
    assert_eq!(to_vec_f32(&got), to_vec_f32(&want));
    assert_eq!(batched.offset, 6);
}

/// The pooled launch keys on `caches[0]`, so a one-row pool-backed batch
/// reaches it at a single-token step with no scheduler context.
#[test]
fn batched_paged_one_row_is_one_pooled_launch() {
    let (pool, states) = fresh_pool(1);
    let mut rng = Rng::new(0x2179);
    let mut paged = KVCache::new_paged(pool, states[0].clone(), 0);
    let (q, k, v) = step(&mut rng, 1, 12);
    paged.attend(&q, k, v, SCALE, None);

    let (q, k, v) = step(&mut rng, 1, 1);
    let before = paged_batch_decode_stats();
    let mut caches: Vec<&mut KVCache> = vec![&mut paged];
    let out = attend_batched(&q, &k, &v, &mut caches, SCALE, None);
    let after = paged_batch_decode_stats();
    assert_eq!(ffi::array_shape(&out), vec![1, HEADS, 1, DIM]);
    assert_eq!(
        after.v2_launches + after.gather_fallbacks,
        before.v2_launches + before.gather_fallbacks + 1,
        "a one-row pool-backed batch is served by the pooled entry"
    );
    assert_eq!(paged.offset, 13);
}

/// A batch that mixes a pool-backed row with a dense row cannot share one
/// launch. The pooled entry declines before it writes to the pool, and every
/// row then runs `attend` on its own storage, in either row order.
#[test]
fn batched_mixed_storage_runs_each_row_on_its_own_route() {
    for paged_first in [true, false] {
        let (pool, states) = fresh_pool(1);
        let mut rng = Rng::new(0x217a);
        let mut paged = KVCache::new_paged(pool.clone(), states[0].clone(), 0);
        let mut paged_ref = KVCache::new_paged(pool, states[1].clone(), 0);
        let mut dense = KVCache::new();
        let mut dense_ref = KVCache::new();

        let (q, k, v) = step(&mut rng, 1, 12);
        paged.attend(&q, ffi::copy(&k), ffi::copy(&v), SCALE, None);
        paged_ref.attend(&q, ffi::copy(&k), ffi::copy(&v), SCALE, None);
        let (q, k, v) = step(&mut rng, 1, 12);
        dense.attend(&q, ffi::copy(&k), ffi::copy(&v), SCALE, None);
        dense_ref.attend(&q, k, v, SCALE, None);

        let (q, k, v) = step(&mut rng, 2, 1);
        let (paged_row, dense_row) = if paged_first { (0, 1) } else { (1, 0) };
        let want_paged = paged_ref.attend(
            &slice_row(&q, paged_row),
            slice_row(&k, paged_row),
            slice_row(&v, paged_row),
            SCALE,
            None,
        );
        let want_dense = dense_ref.attend(
            &slice_row(&q, dense_row),
            slice_row(&k, dense_row),
            slice_row(&v, dense_row),
            SCALE,
            None,
        );

        let before = paged_batch_decode_stats();
        let mut caches: Vec<&mut KVCache> = if paged_first {
            vec![&mut paged, &mut dense]
        } else {
            vec![&mut dense, &mut paged]
        };
        let got = attend_batched(&q, &k, &v, &mut caches, SCALE, None);
        let after = paged_batch_decode_stats();
        assert_eq!(ffi::array_shape(&got), vec![2, HEADS, 1, DIM]);
        assert_eq!(
            after.v2_launches + after.gather_fallbacks,
            before.v2_launches + before.gather_fallbacks + 1,
            "only the pool-backed row reaches the pooled entry (paged_first={paged_first})"
        );

        let got_paged = to_vec_f32(&slice_row(&got, paged_row));
        let got_dense = to_vec_f32(&slice_row(&got, dense_row));
        assert_eq!(
            got_dense,
            to_vec_f32(&want_dense),
            "dense row differs (paged_first={paged_first})"
        );
        assert_eq!(
            got_paged,
            to_vec_f32(&want_paged),
            "pool-backed row differs (paged_first={paged_first})"
        );
        assert_eq!(paged.offset, 13);
        assert_eq!(dense.offset, 13);
    }
}
