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

//! GPU correctness tests for the cascade decode launch (issue #903).
//!
//! The batch here holds *genuinely* shared pool blocks: sequence 0 writes the
//! shared prompt, the others adopt its block ids through `retain_block` and
//! append their own tokens, which is the same fork
//! `CachePool::clone_detached_paged_prefix` performs for an APC hit. So the
//! sharing the detector reads off the page table is the sharing the pool really
//! has, not a hand-built page table that happens to repeat a row.
//!
//! Two of these tests are negative controls, and they exist because both
//! failures are silent:
//!
//! * **Head-stacking order.** Level 0 folds the member axis into the query-head
//!   axis, and the kernel maps head `h` to KV head `h / NRep`. Stack
//!   member-major instead of KV-head-major and every query head reads the wrong
//!   KV head: same shapes, successful launch, wrong answer.
//! * **LSE units.** Issue #898's merge kernel takes log2 LSE. A natural-log LSE
//!   still merges and returns a plausible weighted average, which is what
//!   `mla::split_kv_tests::merge_rejects_natural_log_lse_units` pins for the MLA
//!   caller. The same control is restated here on cascade-shaped partials.

use super::*;
use crate::cache::{PagedBlockId, PagedBlockPool, PagedKvLayout, PagedSequenceState};
use crate::dtype;
use crate::paged_v2::cascade::{CascadeGroup, build_cascade_plan, detect_shared_prefix};

/// xorshift64* in [-1, 1). Deterministic so a failure reproduces exactly.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
        unit * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// A decode batch whose first `shared_pages` blocks are one physical prefix
/// shared by `members` sequences, plus optional non-sharing sequences.
struct CascadeBatch {
    pool: PagedBlockPool,
    states: Vec<PagedSequenceState>,
    /// Per request, `[token][kv head][dim]` flattened as `((t * hkv) + h) * d + i`.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// `[B][Hq * D]` decode query.
    q: Vec<Vec<f32>>,
    lens: Vec<usize>,
    hq: i32,
    hkv: i32,
    dim: i32,
    page_size: usize,
}

impl CascadeBatch {
    /// `member_tails[m]` is the private token count of member `m`; `loners` are
    /// whole-sequence lengths for requests that share nothing.
    #[allow(clippy::too_many_arguments)]
    fn new(
        page_size: usize,
        hq: i32,
        hkv: i32,
        dim: i32,
        shared_pages: usize,
        member_tails: &[usize],
        loners: &[usize],
        pool_dtype: i32,
        seed: u64,
    ) -> Self {
        let layout =
            PagedKvLayout::uniform(1, page_size, page_size * hkv as usize * dim as usize * 2)
                .unwrap();
        let mut pool = PagedBlockPool::new(layout);
        let mut rng = Rng::new(seed);
        let per_token = hkv as usize * dim as usize;
        let shared_len = shared_pages * page_size;
        let shared_k = rng.vec(shared_len * per_token);
        let shared_v = rng.vec(shared_len * per_token);

        let mut states: Vec<PagedSequenceState> = Vec::new();
        let mut k_all: Vec<Vec<f32>> = Vec::new();
        let mut v_all: Vec<Vec<f32>> = Vec::new();
        let mut shared_blocks: Vec<PagedBlockId> = Vec::new();

        for (m, &tail) in member_tails.iter().enumerate() {
            let mut state = PagedSequenceState::new(pool.layout());
            let mut k_seq = shared_k.clone();
            let mut v_seq = shared_v.clone();
            k_seq.extend(rng.vec(tail * per_token));
            v_seq.extend(rng.vec(tail * per_token));
            let len = shared_len + tail;

            let first_private_page = if m == 0 {
                // The anchor writes the shared prefix itself.
                pool.append_tokens(&mut state, 0, len).unwrap();
                shared_blocks = state.layer(0).unwrap().block_ids[..shared_pages].to_vec();
                0
            } else {
                // Every other member adopts the anchor's blocks, exactly as an
                // APC prefix clone does, then appends its own tokens.
                for id in &shared_blocks {
                    pool.retain_block(*id).unwrap();
                }
                {
                    let layer = state.layer_mut(0).unwrap();
                    layer.block_ids = shared_blocks.clone();
                    layer.len = shared_len;
                }
                pool.append_tokens(&mut state, 0, tail).unwrap();
                shared_pages
            };
            write_pages(
                &mut pool,
                &state,
                page_size,
                hkv,
                dim,
                &k_seq,
                &v_seq,
                len,
                first_private_page,
                pool_dtype,
            );
            states.push(state);
            k_all.push(k_seq);
            v_all.push(v_seq);
        }

        for &len in loners {
            let mut state = PagedSequenceState::new(pool.layout());
            pool.append_tokens(&mut state, 0, len).unwrap();
            let k_seq = rng.vec(len * per_token);
            let v_seq = rng.vec(len * per_token);
            write_pages(
                &mut pool, &state, page_size, hkv, dim, &k_seq, &v_seq, len, 0, pool_dtype,
            );
            states.push(state);
            k_all.push(k_seq);
            v_all.push(v_seq);
        }

        let lens: Vec<usize> = member_tails
            .iter()
            .map(|t| shared_len + t)
            .chain(loners.iter().copied())
            .collect();
        let q = (0..lens.len())
            .map(|_| rng.vec(hq as usize * dim as usize))
            .collect();

        Self {
            pool,
            states,
            k: k_all,
            v: v_all,
            q,
            lens,
            hq,
            hkv,
            dim,
            page_size,
        }
    }

    fn state_refs(&self) -> Vec<&PagedSequenceState> {
        self.states.iter().collect()
    }

    fn q_array(&self) -> UniquePtr<MlxArray> {
        let flat: Vec<f32> = self.q.iter().flat_map(|s| s.iter().copied()).collect();
        ffi::from_slice_f32(&flat, &[self.q.len() as i32, self.hq, 1, self.dim])
    }

    fn scale(&self) -> f32 {
        1.0 / (self.dim as f32).sqrt()
    }

    fn geometry(&self) -> PagedDecodeGeometry {
        PagedDecodeGeometry {
            q_heads: self.hq,
            kv_heads: self.hkv,
            head_dim: self.dim,
            page_size: self.page_size as i32,
        }
    }

    fn view(&self) -> crate::cache::paged_csr::PagedCsrView {
        self.pool.paged_csr_view(&self.state_refs(), 0).unwrap()
    }

    fn pools(&self) -> (&MlxArray, &MlxArray) {
        self.pool.single_slab_tensors(0).expect("single-slab pool")
    }

    /// Host attention over the whole written range, in f64.
    fn reference(&self) -> Vec<f32> {
        let hq = self.hq as usize;
        let hkv = self.hkv as usize;
        let dim = self.dim as usize;
        let n_rep = hq / hkv;
        let scale = f64::from(self.scale());
        let mut out = vec![0.0f32; self.lens.len() * hq * dim];
        for (r, &len) in self.lens.iter().enumerate() {
            for h in 0..hq {
                let kv_head = h / n_rep;
                let scores: Vec<f64> = (0..len)
                    .map(|t| {
                        let mut dot = 0.0f64;
                        for i in 0..dim {
                            dot += f64::from(self.q[r][h * dim + i])
                                * f64::from(self.k[r][(t * hkv + kv_head) * dim + i]);
                        }
                        dot * scale
                    })
                    .collect();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0.0f64;
                let mut acc = vec![0.0f64; dim];
                for (t, score) in scores.iter().enumerate() {
                    let p = (score - m).exp();
                    denom += p;
                    for (i, slot) in acc.iter_mut().enumerate() {
                        *slot += p * f64::from(self.v[r][(t * hkv + kv_head) * dim + i]);
                    }
                }
                let base = (r * hq + h) * dim;
                for (i, value) in acc.iter().enumerate() {
                    out[base + i] = (value / denom) as f32;
                }
            }
        }
        out
    }
}

/// Write pages `[from_page, end)` of `state` from the host token arrays.
#[allow(clippy::too_many_arguments)]
fn write_pages(
    pool: &mut PagedBlockPool,
    state: &PagedSequenceState,
    page_size: usize,
    hkv: i32,
    dim: i32,
    k_seq: &[f32],
    v_seq: &[f32],
    len: usize,
    from_page: usize,
    pool_dtype: i32,
) {
    let block_ids = state.layer(0).unwrap().block_ids.clone();
    for (p, block_id) in block_ids.iter().enumerate().skip(from_page) {
        let t0 = p * page_size;
        if t0 >= len {
            break;
        }
        let n_slots = (len - t0).min(page_size);
        let mut kb = vec![0.0f32; hkv as usize * n_slots * dim as usize];
        let mut vb = vec![0.0f32; hkv as usize * n_slots * dim as usize];
        for h in 0..hkv as usize {
            for s in 0..n_slots {
                for i in 0..dim as usize {
                    let src = ((t0 + s) * hkv as usize + h) * dim as usize + i;
                    let dst = (h * n_slots + s) * dim as usize + i;
                    kb[dst] = k_seq[src];
                    vb[dst] = v_seq[src];
                }
            }
        }
        let shape = [1, hkv, n_slots as i32, dim];
        let k_arr = ffi::astype(&ffi::from_slice_f32(&kb, &shape), pool_dtype);
        let v_arr = ffi::astype(&ffi::from_slice_f32(&vb, &shape), pool_dtype);
        pool.write_block(*block_id, 0, 0, &k_arr, &v_arr).unwrap();
    }
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = ffi::astype(a, dtype::FLOAT32);
    ffi::eval(&f);
    ffi::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn max_rel_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "output length mismatch");
    let scale = want
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |acc, (g, w)| acc.max((g - w).abs() / scale))
}

/// The flat v2 launch over the whole batch, the thing cascade must reproduce.
fn run_flat(batch: &CascadeBatch) -> Vec<f32> {
    let q = batch.q_array();
    let view = batch.view();
    let (pool_k, pool_v) = batch.pools();
    let out = crate::paged_v2::run_decode_v2(&q, pool_k, pool_v, &view, batch.scale())
        .unwrap()
        .expect("v2 serves this shape");
    to_vec_f32(&out)
}

fn plan_for(batch: &CascadeBatch, min_shared: usize) -> CascadePlan {
    let view = batch.view();
    let group = detect_shared_prefix(&view, min_shared, 2).expect("the batch shares a prefix");
    build_cascade_plan(&view, group).expect("plan builds")
}

fn run_cascade(batch: &CascadeBatch, plan: &CascadePlan) -> Vec<f32> {
    let q = batch.q_array();
    let (pool_k, pool_v) = batch.pools();
    let (out, stats) = run_cascade_decode(
        &q,
        pool_k,
        pool_v,
        plan,
        batch.geometry(),
        batch.scale(),
        128,
    )
    .expect("cascade launch");
    // The stacked level-0 launch is what makes this a cascade rather than two
    // flat launches; assert the fold really happened.
    assert_eq!(
        stats.prefix_q_heads,
        batch.hq * plan.members() as i32,
        "level 0 did not stack the member queries onto the head axis"
    );
    assert!(stats.prefix_chunks >= 1 && stats.suffix_chunks >= 1);
    to_vec_f32(&out)
}

/// `run_cascade_decode` reimplemented with two deliberate defects available, so
/// the negative controls exercise the same launches the production path does.
///
/// `kv_head_major` false stacks the member queries member-major (the wrong
/// order); `lse_scale` other than 1.0 rescales the log2 LSE before the merge
/// (`LN_2` turns it into natural log, the unit mistake #898's contract warns
/// about).
fn run_cascade_variant(
    batch: &CascadeBatch,
    plan: &CascadePlan,
    kv_head_major: bool,
    lse_scale: f32,
) -> Vec<f32> {
    let geometry = batch.geometry();
    let hq = geometry.q_heads;
    let dim = geometry.head_dim;
    let hkv = geometry.kv_heads;
    let n_rep = geometry.n_rep();
    let members = plan.members() as i32;
    let b = plan.batch() as i32;
    let (pool_k, pool_v) = batch.pools();
    let q = batch.q_array();

    let member_index = ffi::from_slice_i32(&plan.member_rows, &[members]);
    let q_members = ffi::take(&q, &member_index, 0);
    let q_prefix = if kv_head_major {
        let split = ffi::reshape(&q_members, &[members, hkv, n_rep, dim]);
        let major = ffi::contiguous(&ffi::transpose_axes(&split, &[1, 0, 2, 3]), false);
        ffi::reshape(&major, &[1, hkv * members * n_rep, 1, dim])
    } else {
        ffi::reshape(&q_members, &[1, members * hq, 1, dim])
    };

    let geometry0 = prefix_geometry(geometry, plan.members()).unwrap();
    let prefix_plan =
        PagedDecodePlan::heuristic(geometry0, &plan.prefix_view.page_counts(), 128);
    let ctx0 = V2Context::build(
        &q_prefix,
        pool_k,
        pool_v,
        &plan.prefix_view,
        geometry0,
        batch.scale(),
    )
    .unwrap();
    let (v0, lse0) = ctx0.launch_with_lse(&prefix_plan).unwrap();
    let (v0, lse0) = if kv_head_major {
        let split = ffi::reshape(&v0, &[hkv, members, n_rep, dim]);
        let major = ffi::contiguous(&ffi::transpose_axes(&split, &[1, 0, 2, 3]), false);
        let split_lse = ffi::reshape(&lse0, &[hkv, members, n_rep]);
        let major_lse = ffi::contiguous(&ffi::transpose_axes(&split_lse, &[1, 0, 2]), false);
        (
            ffi::reshape(&major, &[members, hq, dim]),
            ffi::reshape(&major_lse, &[members, hq]),
        )
    } else {
        (
            ffi::reshape(&v0, &[members, hq, dim]),
            ffi::reshape(&lse0, &[members, hq]),
        )
    };

    let suffix_plan =
        PagedDecodePlan::heuristic(geometry, &plan.suffix_view.page_counts(), 128);
    let ctx1 = V2Context::build(
        &q,
        pool_k,
        pool_v,
        &plan.suffix_view,
        geometry,
        batch.scale(),
    )
    .unwrap();
    let (v1, lse1) = ctx1.launch_with_lse(&suffix_plan).unwrap();

    let order = ffi::from_slice_i32(&plan.merge_order, &[plan.merge_order.len() as i32]);
    let v_cat = crate::concatenate(&v1, &v0, 0);
    let lse_cat = crate::concatenate(&lse1, &lse0, 0);
    let lse_cat = if (lse_scale - 1.0).abs() < f32::EPSILON {
        lse_cat
    } else {
        ffi::multiply(
            &lse_cat,
            &ffi::full_f32(&[1], lse_scale, dtype::FLOAT32),
        )
    };
    let v_in = ffi::take(&v_cat, &order, 0);
    let lse_in = ffi::take(&lse_cat, &order, 0);
    let o_indptr = ffi::from_slice_i32(&plan.o_indptr, &[b + 1]);
    let mut merged_v = UniquePtr::null();
    let mut merged_lse = UniquePtr::null();
    ffi::paged_attention_merge_states(&v_in, &lse_in, &o_indptr, &mut merged_v, &mut merged_lse);
    to_vec_f32(&merged_v)
}

#[test]
fn cascade_matches_the_flat_launch_on_an_exact_f32_pool() {
    // 8 shared pages of 8 tokens each, three members with different tails.
    let batch = CascadeBatch::new(8, 8, 2, 16, 8, &[5, 11, 3], &[], dtype::FLOAT32, 0x5EED);
    let plan = plan_for(&batch, 4);
    assert_eq!(plan.group.shared_pages, 8);
    assert_eq!(plan.group.members, vec![0, 1, 2]);
    assert!(plan.members_are_whole_batch());

    let want = batch.reference();
    let flat = run_flat(&batch);
    let cascade = run_cascade(&batch, &plan);
    assert!(
        max_rel_error(&flat, &want) < 2e-5,
        "the flat baseline itself drifted"
    );
    let err = max_rel_error(&cascade, &want);
    assert!(err < 2e-5, "cascade deviates from the host reference by {err}");
    let err = max_rel_error(&cascade, &flat);
    assert!(err < 2e-5, "cascade deviates from the flat launch by {err}");
}

#[test]
fn cascade_matches_the_flat_launch_with_an_f16_pool() {
    let batch = CascadeBatch::new(16, 8, 4, 32, 6, &[9, 20, 1, 33], &[], dtype::FLOAT16, 0xBEEF);
    let plan = plan_for(&batch, 4);
    let flat = run_flat(&batch);
    let cascade = run_cascade(&batch, &plan);
    // f16 KV: the two paths sum the same products in a different order, so the
    // bound is the f16 tolerance rather than the f32 one.
    let err = max_rel_error(&cascade, &flat);
    assert!(err < 5e-3, "cascade deviates from the flat launch by {err}");
}

#[test]
fn a_non_sharing_request_rides_along_unchanged() {
    // Two members plus two loners: the loners take a merge group of one, which
    // the merge kernel resolves to the identity.
    let batch = CascadeBatch::new(
        8,
        8,
        2,
        16,
        6,
        &[7, 2],
        &[13, 40],
        dtype::FLOAT32,
        0x1234_5678,
    );
    let plan = plan_for(&batch, 4);
    assert_eq!(plan.group.members, vec![0, 1]);
    assert!(!plan.members_are_whole_batch());
    assert_eq!(plan.batch(), 4);

    let want = batch.reference();
    let cascade = run_cascade(&batch, &plan);
    let err = max_rel_error(&cascade, &want);
    assert!(err < 2e-5, "mixed cascade deviates by {err}");
}

#[test]
fn member_major_head_stacking_reads_the_wrong_kv_head() {
    // Negative control for the level-0 permutation. Needs GQA (n_rep > 1) and
    // more than one member, because with n_rep == 1 the two orders coincide.
    let batch = CascadeBatch::new(8, 4, 2, 16, 6, &[5, 9], &[], dtype::FLOAT32, 0xC0FFEE);
    let plan = plan_for(&batch, 4);
    assert!(batch.geometry().n_rep() > 1 && plan.members() > 1);

    let want = batch.reference();
    let right = run_cascade_variant(&batch, &plan, true, 1.0);
    let wrong = run_cascade_variant(&batch, &plan, false, 1.0);
    assert!(
        max_rel_error(&right, &want) < 2e-5,
        "the KV-head-major stacking must reproduce the reference"
    );
    assert!(
        max_rel_error(&wrong, &want) > 1e-2,
        "member-major stacking produced the right answer, so this control proves nothing"
    );
}

#[test]
fn merge_rejects_natural_log_lse_units_on_the_cascade_path() {
    // The same clause `mla::split_kv_tests::merge_rejects_natural_log_lse_units`
    // pins for the MLA caller, restated on cascade partials: issue #898's merge
    // kernel takes log2 LSE, and a natural-log LSE merges without complaint
    // into a plausible wrong weighted average.
    let batch = CascadeBatch::new(8, 8, 2, 16, 6, &[5, 9, 21], &[], dtype::FLOAT32, 0xFEED);
    let plan = plan_for(&batch, 4);

    let want = batch.reference();
    let log2 = run_cascade_variant(&batch, &plan, true, 1.0);
    let natural = run_cascade_variant(&batch, &plan, true, std::f32::consts::LN_2);
    assert!(
        max_rel_error(&log2, &want) < 2e-5,
        "the log2 merge must reproduce the reference"
    );
    assert!(
        max_rel_error(&natural, &want) > 1e-3,
        "a natural-log LSE produced the right answer, so this control proves nothing"
    );
}

#[test]
fn the_shared_blocks_really_are_shared_in_the_pool() {
    // The premise of the whole feature: detection reads equal physical rows off
    // the page table, and equal rows mean one refcounted block, not two copies.
    let batch = CascadeBatch::new(8, 8, 2, 16, 5, &[4, 6, 8], &[], dtype::FLOAT32, 0x900D);
    let shared = batch.states[0].layer(0).unwrap().block_ids[..5].to_vec();
    for id in &shared {
        assert_eq!(batch.pool.refcount(*id), 3, "block {id} is not shared by 3");
    }
    let view = batch.view();
    assert_eq!(view.indices[0..5], view.indices[view.indptr[1] as usize..][..5]);

    let group = detect_shared_prefix(&view, 4, 2).expect("group");
    assert_eq!(
        group,
        CascadeGroup {
            shared_pages: 5,
            members: vec![0, 1, 2],
        }
    );
}
