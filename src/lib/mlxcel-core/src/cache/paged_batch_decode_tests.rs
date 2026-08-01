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

//! Tests for the production whole-batch paged decode (issue #899).
//!
//! Two halves. The decline checks are pure and run anywhere: they pin the
//! contract that the function touches nothing before it has decided it can
//! serve the batch. The parity checks drive a real [`PagedBlockPool`] through
//! real [`KVCache::new_paged`] caches, run the batched entry point, and compare
//! against the gather-then-SDPA path computed from the *same* post-write pool
//! state, which is the exact baseline #899 has to preserve.

use std::cell::RefCell;

use super::*;
use crate::cache::{PagedBlockPool, PagedKvLayout, PagedSequenceState};
use crate::dtype;

const PAGE: usize = 32;

// ── decline contract ────────────────────────────────────────────────────────

#[test]
fn a_dense_batch_is_declined() {
    let mut a = KVCache::new();
    let mut b = KVCache::new();
    let caches: Vec<&mut KVCache> = vec![&mut a, &mut b];
    assert!(batch_is_servable(&caches, 0.0).is_err());
}

#[test]
fn an_empty_batch_is_declined() {
    let caches: Vec<&mut KVCache> = Vec::new();
    assert!(batch_is_servable(&caches, 0.0).is_err());
}

#[test]
fn a_softcap_batch_is_declined() {
    let (pool, states) = fresh_pool(2, 2, 64);
    let mut a = KVCache::new_paged(pool.clone(), states[0].clone(), 0);
    let mut b = KVCache::new_paged(pool, states[1].clone(), 0);
    let caches: Vec<&mut KVCache> = vec![&mut a, &mut b];
    assert!(batch_is_servable(&caches, 0.0).is_ok());
    assert!(batch_is_servable(&caches, 30.0).is_err());
}

#[test]
fn a_mixed_dense_and_paged_batch_is_declined() {
    let (pool, states) = fresh_pool(1, 2, 64);
    let mut paged = KVCache::new_paged(pool, states[0].clone(), 0);
    let mut dense = KVCache::new();
    let caches: Vec<&mut KVCache> = vec![&mut paged, &mut dense];
    assert!(batch_is_servable(&caches, 0.0).is_err());
}

#[test]
fn caches_from_two_pools_are_declined() {
    let (pool_a, states_a) = fresh_pool(1, 2, 64);
    let (pool_b, states_b) = fresh_pool(1, 2, 64);
    let mut a = KVCache::new_paged(pool_a, states_a[0].clone(), 0);
    let mut b = KVCache::new_paged(pool_b, states_b[0].clone(), 0);
    let caches: Vec<&mut KVCache> = vec![&mut a, &mut b];
    assert!(batch_is_servable(&caches, 0.0).is_err());
}

#[test]
fn caches_from_two_layers_are_declined() {
    let (pool, states) = fresh_pool(2, 2, 64);
    let mut a = KVCache::new_paged(pool.clone(), states[0].clone(), 0);
    let mut b = KVCache::new_paged(pool, states[1].clone(), 1);
    let caches: Vec<&mut KVCache> = vec![&mut a, &mut b];
    assert!(batch_is_servable(&caches, 0.0).is_err());
}

#[test]
fn shape_checks_require_a_single_query_token() {
    assert!(is_single_token_batch(&[4, 8, 1, 64], 4));
    assert!(!is_single_token_batch(&[4, 8, 2, 64], 4));
    assert!(!is_single_token_batch(&[2, 8, 1, 64], 4));
    assert!(!is_single_token_batch(&[8, 1, 64], 4));
}

#[test]
fn a_multi_token_step_declines_without_touching_the_pool() {
    // Batched prefill and speculative / MTP verify both arrive with more than
    // one query token; the decline must happen before any pool write.
    let (pool, states) = fresh_pool(1, 2, 64);
    let mut a = KVCache::new_paged(pool.clone(), states[0].clone(), 0);
    let mut b = KVCache::new_paged(pool.clone(), states[1].clone(), 0);
    let mut caches: Vec<&mut KVCache> = vec![&mut a, &mut b];

    let q = ffi::zeros(&[2, 8, 3, 64], dtype::FLOAT16);
    let k = ffi::zeros(&[2, 2, 3, 64], dtype::FLOAT16);
    let v = ffi::zeros(&[2, 2, 3, 64], dtype::FLOAT16);
    assert!(paged_batch_decode_attention(&q, &k, &v, &mut caches, 0.125, 0.0).is_none());

    assert_eq!(states[0].borrow().layer(0).unwrap().len, 0);
    assert_eq!(states[1].borrow().layer(0).unwrap().len, 0);
    assert_eq!(pool.borrow().allocated_block_count(), 0);
}

// ── parity against the gather path ──────────────────────────────────────────

/// A pool sized for the fused path plus one `PagedSequenceState` per sequence.
///
/// `set_slab_blocks` is what makes the fused kernels reachable at all: they
/// read one contiguous pool buffer per side, so every row the batch touches has
/// to live in the first slab.
fn fresh_pool(
    num_layers: usize,
    kv_heads: usize,
    head_dim: usize,
) -> (
    Rc<RefCell<PagedBlockPool>>,
    Vec<Rc<RefCell<PagedSequenceState>>>,
) {
    let bytes_per_block = PAGE * kv_heads * head_dim * 2;
    let layout = PagedKvLayout::uniform(num_layers, PAGE, bytes_per_block).expect("valid layout");
    let mut pool = PagedBlockPool::new(layout.clone());
    pool.set_slab_blocks(512)
        .expect("fresh pool has no storage");
    let states = (0..4)
        .map(|_| Rc::new(RefCell::new(PagedSequenceState::new(&layout))))
        .collect();
    (Rc::new(RefCell::new(pool)), states)
}

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

/// Relative RMS between two flattened outputs.
fn relative_rms(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "outputs must have the same length");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += f64::from(x - y) * f64::from(x - y);
        den += f64::from(*y) * f64::from(*y);
    }
    if den == 0.0 {
        return num.sqrt() as f32;
    }
    (num / den).sqrt() as f32
}

/// Drive one whole-batch decode step and return `(batched output, gather
/// reference computed from the same post-write pool state, cache stats)`.
fn run_step(
    prompt_lens: &[usize],
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, crate::paged_v2::PlanCacheStats) {
    let batch = prompt_lens.len();
    let (pool, states) = fresh_pool(1, kv_heads as usize, head_dim as usize);
    let mut rng = Rng::new(seed);
    let mut caches: Vec<KVCache> = (0..batch)
        .map(|b| KVCache::new_paged(pool.clone(), states[b].clone(), 0))
        .collect();

    // Prefill each sequence into the pool.
    for (b, &len) in prompt_lens.iter().enumerate() {
        let k = random_array(&mut rng, &[1, kv_heads, len as i32, head_dim]);
        let v = random_array(&mut rng, &[1, kv_heads, len as i32, head_dim]);
        caches[b].update(k, v);
    }

    let q = random_array(&mut rng, &[batch as i32, q_heads, 1, head_dim]);
    let k_step = random_array(&mut rng, &[batch as i32, kv_heads, 1, head_dim]);
    let v_step = random_array(&mut rng, &[batch as i32, kv_heads, 1, head_dim]);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let out = {
        let mut refs: Vec<&mut KVCache> = caches.iter_mut().collect();
        paged_batch_decode_attention(&q, &k_step, &v_step, &mut refs, scale, 0.0)
            .expect("the batch is servable")
    };

    // Reference: the pre-#899 path, run against the pool state the batched
    // entry point just left behind, so the two see identical K/V.
    let reference = {
        let pool_ref = pool.borrow();
        let borrowed: Vec<std::cell::Ref<'_, PagedSequenceState>> =
            states.iter().take(batch).map(|s| s.borrow()).collect();
        let state_refs: Vec<&PagedSequenceState> = borrowed.iter().map(|s| &**s).collect();
        gather_fallback(&q, &pool_ref, &state_refs, 0, scale)
    };

    let stats = pool.borrow().decode_plan_cache_stats();
    (to_vec_f32(&out), to_vec_f32(&reference), stats)
}

#[test]
fn batched_decode_matches_the_gather_path_above_the_floor() {
    // 4 x 1024 = 4096 visible tokens, exactly the dispatch floor, so v2 runs.
    let (out, reference, stats) = run_step(&[1024, 1024, 1024, 1024], 8, 2, 64, 0xA11CE);
    assert!(
        stats.view_rebuilds >= 1,
        "the CSR view should have been built"
    );
    assert!(stats.plan_rebuilds >= 1, "the plan should have been built");
    let rms = relative_rms(&out, &reference);
    assert!(rms < 5e-3, "relative RMS {rms} exceeds 5e-3");
}

#[test]
fn batched_decode_matches_the_gather_path_with_ragged_lengths() {
    // Mixed lengths, none a multiple of the page size, still above the floor.
    let (out, reference, _) = run_step(&[1500, 900, 1201, 777], 8, 2, 64, 0xBEEF);
    let rms = relative_rms(&out, &reference);
    assert!(rms < 5e-3, "relative RMS {rms} exceeds 5e-3");
}

#[test]
fn batched_decode_matches_the_gather_path_at_gqa_one() {
    let (out, reference, _) = run_step(&[2048, 2048], 4, 4, 64, 0xC0FFEE);
    let rms = relative_rms(&out, &reference);
    assert!(rms < 5e-3, "relative RMS {rms} exceeds 5e-3");
}

#[test]
fn below_the_floor_the_batch_still_answers_correctly() {
    // 2 x 256 = 512 visible tokens against a 2 x 512 = 1024 batched floor, so
    // the entry point takes its own gather fallback. It must still produce the
    // reference answer exactly, not merely within tolerance, and must still
    // have written the step's K/V once.
    let (out, reference, stats) = run_step(&[256, 256], 8, 2, 64, 0xD00D);
    assert_eq!(
        stats.plan_rebuilds, 0,
        "a below-floor launch must not build a plan"
    );
    let rms = relative_rms(&out, &reference);
    assert!(
        rms < 1e-6,
        "the gather fallback must reproduce the reference"
    );
}

#[test]
fn a_single_sequence_above_its_floor_takes_the_fused_path() {
    // The single-sequence decode path (`decode_single_step` -> the model's
    // one-sequence `forward`) reaches this entry point with a batch of one. Two
    // of the five scenarios in the issue's benchmark matrix are that shape, so
    // a batch-1 launch above the single-request floor has to fuse.
    let (out, reference, stats) = run_step(&[4096], 8, 2, 64, 0x1EAF);
    assert!(
        stats.plan_rebuilds >= 1,
        "a 4096-token single-sequence launch should have built a plan, got {stats:?}"
    );
    let rms = relative_rms(&out, &reference);
    assert!(rms < 5e-3, "relative RMS {rms} exceeds 5e-3");
}

#[test]
fn a_short_single_sequence_stays_on_gather() {
    // The one measured loss in #898 is batch 1 at 1024 tokens, so a lone
    // request below 4096 must not fuse even though a batched launch of the same
    // per-request size would.
    let (_, _, stats) = run_step(&[1024], 8, 2, 64, 0x1EAF);
    assert_eq!(
        stats.plan_rebuilds, 0,
        "a 1024-token single-sequence launch must stay on gather, got {stats:?}"
    );
}

#[test]
fn a_batched_launch_just_under_a_nominal_1k_prompt_fuses() {
    // The exact shape the production benchmark delivered for its nominal 1K
    // scenario: 4 requests of 956 tokens. The first floor formulation summed to
    // 3824 and declined it; the two-regime floor requires 4 x 512 = 2048.
    let (out, reference, stats) = run_step(&[956, 956, 956, 956], 8, 2, 64, 0x956);
    assert!(
        stats.plan_rebuilds >= 1,
        "4 x 956 tokens should fuse, got {stats:?}"
    );
    let rms = relative_rms(&out, &reference);
    assert!(rms < 5e-3, "relative RMS {rms} exceeds 5e-3");
}

#[test]
fn the_step_is_written_exactly_once() {
    let (pool, states) = fresh_pool(1, 2, 64);
    let mut rng = Rng::new(0xF00D);
    let mut caches: Vec<KVCache> = (0..2)
        .map(|b| KVCache::new_paged(pool.clone(), states[b].clone(), 0))
        .collect();
    for cache in caches.iter_mut() {
        let k = random_array(&mut rng, &[1, 2, 100, 64]);
        let v = random_array(&mut rng, &[1, 2, 100, 64]);
        cache.update(k, v);
    }
    assert_eq!(states[0].borrow().layer(0).unwrap().len, 100);

    let q = random_array(&mut rng, &[2, 8, 1, 64]);
    let k = random_array(&mut rng, &[2, 2, 1, 64]);
    let v = random_array(&mut rng, &[2, 2, 1, 64]);
    {
        let mut refs: Vec<&mut KVCache> = caches.iter_mut().collect();
        paged_batch_decode_attention(&q, &k, &v, &mut refs, 0.125, 0.0).expect("servable");
    }
    for state in states.iter().take(2) {
        assert_eq!(state.borrow().layer(0).unwrap().len, 101);
    }
    // `offset` drives RoPE for the next step and must advance in lockstep.
    assert_eq!(caches[0].offset, 101);
    assert_eq!(caches[1].offset, 101);
}

#[test]
fn a_steady_step_reuses_the_page_table() {
    let (pool, states) = fresh_pool(1, 2, 64);
    let mut rng = Rng::new(0x5EED);
    let mut caches: Vec<KVCache> = (0..4)
        .map(|b| KVCache::new_paged(pool.clone(), states[b].clone(), 0))
        .collect();
    for cache in caches.iter_mut() {
        // 1024 = exactly 32 pages, so the next 31 decode tokens all land in a
        // fresh page 32 opened by the first of them.
        let k = random_array(&mut rng, &[1, 2, 1024, 64]);
        let v = random_array(&mut rng, &[1, 2, 1024, 64]);
        cache.update(k, v);
    }

    let scale = 0.125;
    let mut rebuilds = Vec::new();
    for _ in 0..4 {
        let q = random_array(&mut rng, &[4, 8, 1, 64]);
        let k = random_array(&mut rng, &[4, 2, 1, 64]);
        let v = random_array(&mut rng, &[4, 2, 1, 64]);
        let mut refs: Vec<&mut KVCache> = caches.iter_mut().collect();
        paged_batch_decode_attention(&q, &k, &v, &mut refs, scale, 0.0).expect("servable");
        drop(refs);
        rebuilds.push(pool.borrow().decode_plan_cache_stats().view_rebuilds);
    }

    // Step 1 opens page 32 for every request (epoch bump, rebuild). Steps 2, 3
    // and 4 stay inside it, so at most one further rebuild is allowed for the
    // trailing epoch bump the first step left behind.
    assert!(
        rebuilds[3] - rebuilds[0] <= 1,
        "steady steps should reuse the page table, rebuild counts were {rebuilds:?}"
    );
    let stats = pool.borrow().decode_plan_cache_stats();
    assert!(
        stats.view_reuses >= 2,
        "expected page-table reuse, got {stats:?}"
    );
}

#[test]
fn a_multi_slab_layer_falls_back_to_gather() {
    // The pool default (32 rows) cannot hold 4 x 1024 tokens in one slab, so
    // the fused path declines and the entry point answers from gather.
    let bytes_per_block = PAGE * 2 * 64 * 2;
    let layout = PagedKvLayout::uniform(1, PAGE, bytes_per_block).expect("valid layout");
    let pool = Rc::new(RefCell::new(PagedBlockPool::new(layout.clone())));
    let states: Vec<Rc<RefCell<PagedSequenceState>>> = (0..4)
        .map(|_| Rc::new(RefCell::new(PagedSequenceState::new(&layout))))
        .collect();
    let mut rng = Rng::new(0x51AB);
    let mut caches: Vec<KVCache> = (0..4)
        .map(|b| KVCache::new_paged(pool.clone(), states[b].clone(), 0))
        .collect();
    for cache in caches.iter_mut() {
        let k = random_array(&mut rng, &[1, 2, 1024, 64]);
        let v = random_array(&mut rng, &[1, 2, 1024, 64]);
        cache.update(k, v);
    }
    assert!(
        pool.borrow().slab_count(0) > 1,
        "the default slab size should not hold this batch"
    );

    let q = random_array(&mut rng, &[4, 8, 1, 64]);
    let k = random_array(&mut rng, &[4, 2, 1, 64]);
    let v = random_array(&mut rng, &[4, 2, 1, 64]);
    let scale = 0.125;
    let out = {
        let mut refs: Vec<&mut KVCache> = caches.iter_mut().collect();
        paged_batch_decode_attention(&q, &k, &v, &mut refs, scale, 0.0).expect("servable")
    };
    let reference = {
        let pool_ref = pool.borrow();
        let borrowed: Vec<std::cell::Ref<'_, PagedSequenceState>> =
            states.iter().map(|s| s.borrow()).collect();
        let state_refs: Vec<&PagedSequenceState> = borrowed.iter().map(|s| &**s).collect();
        gather_fallback(&q, &pool_ref, &state_refs, 0, scale)
    };
    let rms = relative_rms(&to_vec_f32(&out), &to_vec_f32(&reference));
    assert!(
        rms < 1e-6,
        "the gather fallback must reproduce the reference"
    );
    // Nothing was cached, because the fused path was never reached.
    assert_eq!(pool.borrow().decode_plan_cache_stats().plan_rebuilds, 0);
}

// ── outcome reporting ───────────────────────────────────────────────────────

/// Drive one launch through the pool directly and return only its outcome.
fn outcome_for(prompt_len: usize, slabbed: bool) -> PagedDecodeOutcome {
    let (pool, states) = if slabbed {
        fresh_pool(1, 2, 64)
    } else {
        let bytes_per_block = PAGE * 2 * 64 * 2;
        let layout = PagedKvLayout::uniform(1, PAGE, bytes_per_block).expect("valid layout");
        let pool = Rc::new(RefCell::new(PagedBlockPool::new(layout.clone())));
        let states = (0..4)
            .map(|_| Rc::new(RefCell::new(PagedSequenceState::new(&layout))))
            .collect();
        (pool, states)
    };
    let mut rng = Rng::new(0xD1A6);
    let mut cache = KVCache::new_paged(pool.clone(), states[0].clone(), 0);
    let k = random_array(&mut rng, &[1, 2, prompt_len as i32, 64]);
    let v = random_array(&mut rng, &[1, 2, prompt_len as i32, 64]);
    cache.update(k, v);

    let q = random_array(&mut rng, &[1, 8, 1, 64]);
    let state_ref = states[0].borrow();
    let refs: Vec<&PagedSequenceState> = vec![&state_ref];
    let mut pool_mut = pool.borrow_mut();
    pool_mut
        .paged_decode_batched(&q, &refs, 0, 0.125)
        .expect("no hard error")
        .1
}

#[test]
fn a_multi_slab_layer_reports_the_knob_that_would_fix_it() {
    // The decline that silently disabled the whole fused path in the first #899
    // production benchmark. It has to name the slab counts and the knob, not
    // just return `None`.
    let outcome = outcome_for(2048, false);
    match &outcome {
        PagedDecodeOutcome::MultiSlab {
            k_slabs,
            slab_blocks,
            ..
        } => {
            assert!(*k_slabs > 1, "expected several slabs, got {k_slabs}");
            assert_eq!(*slab_blocks, crate::cache::POOL_SLAB_BLOCKS);
        }
        other => panic!("expected a multi-slab decline, got {other:?}"),
    }
    assert!(!outcome.is_fused());
    assert!(outcome.describe().contains("MLXCEL_PAGED_SLAB_BLOCKS"));
}

#[test]
fn a_below_floor_launch_reports_both_numbers() {
    match outcome_for(64, true) {
        PagedDecodeOutcome::BelowFloor {
            batch,
            visible_tokens,
            floor,
        } => {
            assert_eq!(batch, 1);
            assert_eq!(visible_tokens, 64);
            assert_eq!(floor, crate::paged_v2::MIN_SINGLE_REQUEST_KV_TOKENS);
        }
        other => panic!("expected a below-floor decline, got {other:?}"),
    }
}

#[test]
fn a_fused_launch_reports_its_shape() {
    match outcome_for(8192, true) {
        PagedDecodeOutcome::Fused {
            batch,
            visible_tokens,
            chunks,
            ..
        } => {
            assert_eq!(batch, 1);
            assert_eq!(visible_tokens, 8192);
            assert!(chunks >= 1);
        }
        other => panic!("expected a fused launch, got {other:?}"),
    }
}

#[test]
fn an_unservable_batch_declines_without_touching_the_pool() {
    reset_reported();
    let (pool, states) = fresh_pool(1, 2, 64);
    let mut a = KVCache::new_paged(pool.clone(), states[0].clone(), 0);
    let mut dense = KVCache::new();
    let mut caches: Vec<&mut KVCache> = vec![&mut a, &mut dense];
    let q = ffi::zeros(&[2, 8, 1, 64], dtype::FLOAT16);
    let k = ffi::zeros(&[2, 2, 1, 64], dtype::FLOAT16);
    let v = ffi::zeros(&[2, 2, 1, 64], dtype::FLOAT16);
    assert!(paged_batch_decode_attention(&q, &k, &v, &mut caches, 0.125, 0.0).is_none());
    assert_eq!(states[0].borrow().layer(0).unwrap().len, 0);
    assert_eq!(pool.borrow().allocated_block_count(), 0);
}
