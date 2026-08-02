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

//! The production paged decode entry point really dispatches cascade (#903).
//!
//! This is a **separate integration binary on purpose.** `cascade_enabled()`
//! and the two threshold reads are `OnceLock`s over environment variables, so a
//! unit test inside `mlxcel-core` cannot flip them: whichever test ran first in
//! the shared process would pin the value for every other. An integration test
//! is its own process, so setting the variables before the first read is
//! deterministic here and nowhere else.
//!
//! It exists because of the most expensive lesson of epic #909. Issue #899
//! shipped a fused decode path that silently never activated, and its
//! before/after benchmark compared the fallback against itself and produced a
//! clean-looking null. A cascade implementation that is correct in isolation but
//! is never reached by `PagedBlockPool::paged_decode_batched` would reproduce
//! that failure exactly, and no unit test in `paged_v2` would notice.

use mlxcel_core::cache::{PagedBlockId, PagedBlockPool, PagedKvLayout, PagedSequenceState};
use mlxcel_core::paged_v2::PagedDecodeOutcome;
use mlxcel_core::{MlxArray, UniquePtr};

const PAGE: usize = 32;
const HQ: i32 = 8;
const HKV: i32 = 2;
const DIM: i32 = 32;
/// 32 pages, comfortably over both the cascade span floor and the #899
/// per-request dispatch floor of 512 visible tokens.
const SHARED_PAGES: usize = 32;
const TAIL: usize = 64;
const BATCH: usize = 4;

/// xorshift64* in [-1, 1).
struct Rng(u64);

impl Rng {
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
                (unit * 2.0 - 1.0) * 0.5
            })
            .collect()
    }
}

/// A batch of `BATCH` sequences behind one genuinely refcounted shared prefix.
fn shared_prefix_batch() -> (PagedBlockPool, Vec<PagedSequenceState>, UniquePtr<MlxArray>) {
    let hkv = HKV as usize;
    let dim = DIM as usize;
    let layout = PagedKvLayout::uniform(1, PAGE, PAGE * hkv * dim * 2).unwrap();
    let mut pool = PagedBlockPool::new(layout);
    let total_len = SHARED_PAGES * PAGE + TAIL;
    // The fused kernels read one contiguous pool buffer per side, so the whole
    // batch has to fit the first slab.
    pool.set_slab_blocks((BATCH * total_len.div_ceil(PAGE) + 8).next_power_of_two())
        .unwrap();

    let mut rng = Rng(0x903_5EED);
    let mut states: Vec<PagedSequenceState> = Vec::with_capacity(BATCH);
    let mut shared_blocks: Vec<PagedBlockId> = Vec::new();
    for b in 0..BATCH {
        let mut state = PagedSequenceState::new(pool.layout());
        let from_page = if b == 0 {
            pool.append_tokens(&mut state, 0, total_len).unwrap();
            shared_blocks = state.layer(0).unwrap().block_ids[..SHARED_PAGES].to_vec();
            0
        } else {
            for id in &shared_blocks {
                pool.retain_block(*id).unwrap();
            }
            {
                let layer = state.layer_mut(0).unwrap();
                layer.block_ids = shared_blocks.clone();
                layer.len = SHARED_PAGES * PAGE;
            }
            pool.append_tokens(&mut state, 0, TAIL).unwrap();
            SHARED_PAGES
        };
        let block_ids = state.layer(0).unwrap().block_ids.clone();
        for block_id in block_ids.iter().skip(from_page) {
            let shape = [1, HKV, PAGE as i32, DIM];
            let k = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&rng.vec(hkv * PAGE * dim), &shape),
                mlxcel_core::dtype::FLOAT16,
            );
            let v = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&rng.vec(hkv * PAGE * dim), &shape),
                mlxcel_core::dtype::FLOAT16,
            );
            pool.write_block(*block_id, 0, 0, &k, &v).unwrap();
        }
        states.push(state);
    }
    for id in &shared_blocks {
        assert_eq!(
            pool.refcount(*id),
            BATCH as u32,
            "the fixture's prefix is not actually shared"
        );
    }

    let q = mlxcel_core::from_slice_f32(
        &rng.vec(BATCH * HQ as usize * dim),
        &[BATCH as i32, HQ, 1, DIM],
    );
    (pool, states, q)
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn the_production_entry_point_dispatches_cascade_and_agrees_with_the_flat_launch() {
    if !mlxcel_core::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    // SAFETY: single-threaded, before any other thread exists in this test
    // binary, and before anything has read the gate. Every `OnceLock` behind
    // these variables latches on first read, and the first read in this process
    // happens inside the `paged_decode_batched` call below.
    unsafe {
        std::env::set_var("MLXCEL_CASCADE_ATTENTION", "1");
        std::env::set_var("MLXCEL_CASCADE_MIN_SHARED_PAGES", "8");
        std::env::set_var("MLXCEL_CASCADE_MIN_MEMBERS", "2");
    }

    let (mut pool, states, q) = shared_prefix_batch();
    let refs: Vec<&PagedSequenceState> = states.iter().collect();
    let scale = 1.0 / (DIM as f32).sqrt();

    // The flat baseline comes from the #898 library entry, which has no cascade
    // branch at all, so enabling the feature cannot contaminate it.
    let view = pool.paged_csr_view(&refs, 0).unwrap();
    let (pool_k, pool_v) = pool.single_slab_tensors(0).expect("single-slab pool");
    let flat = mlxcel_core::paged_v2::run_decode_v2(&q, pool_k, pool_v, &view, scale)
        .unwrap()
        .expect("v2 serves this shape");
    let flat = to_vec_f32(&flat);

    let (cascade, cascade_outcome) = pool.paged_decode_batched(&q, &refs, 0, scale).unwrap();
    let cascade = to_vec_f32(&cascade.expect("v2 serves this shape"));

    match &cascade_outcome {
        PagedDecodeOutcome::FusedCascade {
            batch,
            members,
            shared_pages,
            shared_tokens,
            ..
        } => {
            assert_eq!(*batch, BATCH);
            assert_eq!(*members, BATCH, "every sequence shares the prefix");
            assert_eq!(*shared_pages, SHARED_PAGES);
            assert_eq!(*shared_tokens, SHARED_PAGES * PAGE);
        }
        other => panic!("the production entry point did not dispatch cascade: {other:?}"),
    }
    assert!(cascade_outcome.is_cascade() && cascade_outcome.is_fused());

    // The counters a server reads must agree with the outcome, or the /metrics
    // view of "is cascade running" is decorative.
    let stats = mlxcel_core::cache::paged_batch_decode_stats();
    assert_eq!(stats.cascade_failures, 0, "a cascade launch fell back");

    let scale_ref = flat
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    let err = flat
        .iter()
        .zip(&cascade)
        .fold(0.0f32, |acc, (f, c)| acc.max((f - c).abs() / scale_ref));
    assert!(
        err < 5e-3,
        "cascade and flat disagree by {err} on the production entry point"
    );
}
