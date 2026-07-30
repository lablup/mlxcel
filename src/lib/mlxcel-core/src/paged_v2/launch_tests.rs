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

//! GPU correctness tests for the paged decode v2 launch (issue #898).
//!
//! Deliberately small so they run inside the normal unit-test budget; the wide
//! matrix (head dims, GQA ratios, block sizes, contexts to 32K, batches) lives
//! in `examples/paged_decode_v2_correctness.rs`, which is opt-in.
//!
//! The reference here is a host-side attention computed in f64 from the same
//! values that were written into the pool, not another GPU path. That makes a
//! failure attributable: a mismatch is the kernel, not a disagreement between
//! two kernels that could both be wrong. The pools are f32 so no quantization
//! sits between the reference and the kernel, and one test additionally checks
//! the realistic f16-pool case against the gather-then-SDPA reference.

use super::*;
use crate::cache::{PagedBlockPool, PagedKvLayout, PagedSequenceState};
use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::paged_v2::plan::PagedDecodePlan;
use cxx::UniquePtr;

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
}

/// One synthetic decode batch: the pool, the sequence states, and the host copy
/// of every value written, so a reference can be computed without reading back
/// from the GPU.
struct Batch {
    pool: PagedBlockPool,
    states: Vec<PagedSequenceState>,
    /// Per request, `[token][head][dim]` flattened as `((t * hkv) + h) * d + i`.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// `[B][Hq * D]` decode query.
    q: Vec<Vec<f32>>,
    lens: Vec<usize>,
    starts: Vec<usize>,
    hq: i32,
    hkv: i32,
    dim: i32,
    page_size: usize,
}

impl Batch {
    /// Build a batch with the given per-request written lengths and visible
    /// window starts. `dtype` is the pool dtype (f32 for exact tests, f16 for
    /// the realistic one).
    #[allow(clippy::too_many_arguments)]
    fn new(
        page_size: usize,
        hq: i32,
        hkv: i32,
        dim: i32,
        lens: &[usize],
        starts: &[usize],
        pool_dtype: i32,
        seed: u64,
    ) -> Self {
        let layout =
            PagedKvLayout::uniform(1, page_size, page_size * hkv as usize * dim as usize * 2)
                .unwrap();
        let mut pool = PagedBlockPool::new(layout);
        let mut rng = Rng::new(seed);
        let mut states = Vec::new();
        let mut k_all = Vec::new();
        let mut v_all = Vec::new();
        let mut q_all = Vec::new();

        for (r, &len) in lens.iter().enumerate() {
            let mut state = PagedSequenceState::new(pool.layout());
            if len > 0 {
                pool.append_tokens(&mut state, 0, len).unwrap();
            }
            let mut k_seq = vec![0.0f32; len * hkv as usize * dim as usize];
            let mut v_seq = vec![0.0f32; len * hkv as usize * dim as usize];
            for value in k_seq.iter_mut() {
                *value = rng.next_f32();
            }
            for value in v_seq.iter_mut() {
                *value = rng.next_f32();
            }

            // Write page by page in the `[1, Hkv, n_slots, D]` layout
            // `write_block` accepts.
            let block_ids = state.layer(0).unwrap().block_ids.clone();
            for (p, block_id) in block_ids.iter().enumerate() {
                let t0 = p * page_size;
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
                let k_arr = cast(&ffi::from_slice_f32(&kb, &shape), pool_dtype);
                let v_arr = cast(&ffi::from_slice_f32(&vb, &shape), pool_dtype);
                pool.write_block(*block_id, 0, 0, &k_arr, &v_arr).unwrap();
            }

            state.layer_mut(0).unwrap().logical_start = starts[r];
            states.push(state);
            k_all.push(k_seq);
            v_all.push(v_seq);

            let mut q_seq = vec![0.0f32; hq as usize * dim as usize];
            for value in q_seq.iter_mut() {
                *value = rng.next_f32();
            }
            q_all.push(q_seq);
        }

        Self {
            pool,
            states,
            k: k_all,
            v: v_all,
            q: q_all,
            lens: lens.to_vec(),
            starts: starts.to_vec(),
            hq,
            hkv,
            dim,
            page_size,
        }
    }

    fn state_refs(&self) -> Vec<&PagedSequenceState> {
        self.states.iter().collect()
    }

    fn q_array(&self, dtype_id: i32) -> UniquePtr<MlxArray> {
        let flat: Vec<f32> = self.q.iter().flat_map(|s| s.iter().copied()).collect();
        let arr = ffi::from_slice_f32(&flat, &[self.q.len() as i32, self.hq, 1, self.dim]);
        cast(&arr, dtype_id)
    }

    fn scale(&self) -> f32 {
        1.0 / (self.dim as f32).sqrt()
    }

    /// Host attention over the visible window, in f64.
    fn reference(&self) -> Vec<f32> {
        let hq = self.hq as usize;
        let hkv = self.hkv as usize;
        let dim = self.dim as usize;
        let n_rep = hq / hkv;
        let scale = f64::from(self.scale());
        let mut out = vec![0.0f32; self.lens.len() * hq * dim];
        for (r, &len) in self.lens.iter().enumerate() {
            let start = self.starts[r];
            for h in 0..hq {
                let kv_head = h / n_rep;
                let mut scores: Vec<f64> = Vec::with_capacity(len - start);
                for t in start..len {
                    let mut dot = 0.0f64;
                    for i in 0..dim {
                        let q = f64::from(self.q[r][h * dim + i]);
                        let k = f64::from(self.k[r][(t * hkv + kv_head) * dim + i]);
                        dot += q * k;
                    }
                    scores.push(dot * scale);
                }
                let out_base = (r * hq + h) * dim;
                if scores.is_empty() {
                    continue; // an empty window contributes zeros
                }
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0.0f64;
                let mut acc = vec![0.0f64; dim];
                for (idx, t) in (start..len).enumerate() {
                    let p = (scores[idx] - m).exp();
                    denom += p;
                    for (i, slot) in acc.iter_mut().enumerate() {
                        *slot += p * f64::from(self.v[r][(t * hkv + kv_head) * dim + i]);
                    }
                }
                for (i, value) in acc.iter().enumerate() {
                    out[out_base + i] = (value / denom) as f32;
                }
            }
        }
        out
    }

    /// Geometry the plan is built against.
    fn geometry(&self) -> PagedDecodeGeometry {
        PagedDecodeGeometry {
            q_heads: self.hq,
            kv_heads: self.hkv,
            head_dim: self.dim,
            page_size: self.page_size as i32,
        }
    }
}

fn cast(a: &MlxArray, dtype_id: i32) -> UniquePtr<MlxArray> {
    if dtype_id == dtype::FLOAT32 {
        ffi::astype(a, dtype::FLOAT32)
    } else {
        ffi::astype(a, dtype_id)
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

/// Max absolute deviation relative to the reference's own scale, which is the
/// tolerance form the issue asks for (a plain absolute bound would be
/// meaningless for an output whose magnitude depends on the value distribution).
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

/// Run v2 with an explicit chunk size, bypassing the heuristic so a test can
/// pin the single-chunk and the many-chunk paths independently.
fn run_with_chunk(batch: &Batch, pages_per_chunk: i32) -> Vec<f32> {
    let q = batch.q_array(dtype::FLOAT32);
    let view = batch.pool.paged_csr_view(&batch.state_refs(), 0).unwrap();
    let (pool_k, pool_v) = batch.pool.single_slab_tensors(0).expect("single-slab pool");
    let ctx = V2Context::build(&q, pool_k, pool_v, &view, batch.geometry(), batch.scale()).unwrap();
    let plan = PagedDecodePlan::with_chunk_size(
        batch.geometry(),
        &view.page_counts(),
        pages_per_chunk,
        128,
        crate::autotune::Source::Default,
    );
    plan.validate().unwrap();
    let out = ctx.launch(&plan).unwrap();
    to_vec_f32(&out)
}

// ---------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------

#[test]
fn single_chunk_matches_a_host_reference() {
    let batch = Batch::new(16, 4, 2, 64, &[40, 33], &[0, 0], dtype::FLOAT32, 0x51);
    let got = run_with_chunk(&batch, 64); // one chunk per request, no merge
    let err = max_rel_error(&got, &batch.reference());
    assert!(err < 2e-3, "single-chunk relative error {err}");
}

#[test]
fn many_chunks_merge_to_the_same_answer() {
    let batch = Batch::new(16, 4, 2, 64, &[40, 33], &[0, 0], dtype::FLOAT32, 0x51);
    // One page per chunk: 3 chunks for request 0, 3 for request 1, so the merge
    // kernel is exercised with a variable-length grouping.
    let merged = run_with_chunk(&batch, 1);
    let err = max_rel_error(&merged, &batch.reference());
    assert!(err < 2e-3, "merged relative error {err}");

    // And the two paths agree with each other, which is the property #899's
    // dispatch will rely on when it switches chunk sizes at runtime.
    let single = run_with_chunk(&batch, 64);
    let cross = max_rel_error(&merged, &single);
    assert!(cross < 2e-3, "single-chunk vs merged deviation {cross}");
}

#[test]
fn chunk_size_does_not_change_the_answer() {
    let batch = Batch::new(
        32,
        8,
        2,
        64,
        &[200, 97, 128],
        &[0, 0, 0],
        dtype::FLOAT32,
        0xbeef,
    );
    let want = batch.reference();
    for ppc in [1, 2, 3, 4, 7, 16, 64] {
        let got = run_with_chunk(&batch, ppc);
        let err = max_rel_error(&got, &want);
        assert!(err < 2e-3, "pages_per_chunk {ppc} relative error {err}");
    }
}

#[test]
fn gqa_groups_read_the_right_kv_head() {
    // n_rep = 8: the CTA's q-head group split is exercised, and a mis-mapped
    // KV head would show up as a large error rather than a subtle one.
    let batch = Batch::new(16, 8, 1, 64, &[70], &[0], dtype::FLOAT32, 0x1234);
    for ppc in [1, 5, 32] {
        let err = max_rel_error(&run_with_chunk(&batch, ppc), &batch.reference());
        assert!(err < 2e-3, "gqa pages_per_chunk {ppc} relative error {err}");
    }
}

#[test]
fn a_trimmed_window_attends_only_to_visible_tokens() {
    // logical_start lands mid-page, so first_page_offset is non-zero and the
    // retired pages are absent from the CSR view entirely.
    let batch = Batch::new(16, 4, 2, 64, &[100, 64], &[37, 16], dtype::FLOAT32, 0x77);
    for ppc in [1, 2, 8] {
        let err = max_rel_error(&run_with_chunk(&batch, ppc), &batch.reference());
        assert!(
            err < 2e-3,
            "trimmed pages_per_chunk {ppc} relative error {err}"
        );
    }
}

#[test]
fn an_empty_request_yields_zeros_without_poisoning_its_neighbours() {
    // Request 1 is fully trimmed: its chunk produces an all-empty partial
    // (lse = -inf), which must merge to zeros rather than NaN.
    let batch = Batch::new(
        16,
        4,
        2,
        64,
        &[48, 32, 40],
        &[0, 32, 5],
        dtype::FLOAT32,
        0x99,
    );
    let want = batch.reference();
    for ppc in [1, 4] {
        let got = run_with_chunk(&batch, ppc);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "output has non-finite values"
        );
        let hq_d = (batch.hq * batch.dim) as usize;
        assert!(
            got[hq_d..2 * hq_d].iter().all(|&v| v == 0.0),
            "the empty request should read back as zeros"
        );
        let err = max_rel_error(&got, &want);
        assert!(
            err < 2e-3,
            "empty-request pages_per_chunk {ppc} relative error {err}"
        );
    }
}

#[test]
fn f16_pools_match_the_gather_reference() {
    // The realistic dtype: f16 pool, f16 query, compared against the
    // gather-then-SDPA path the tree already trusts (ADR 0001 strategy A).
    let batch = Batch::new(32, 8, 2, 128, &[300, 129], &[0, 0], dtype::FLOAT16, 0x2026);
    let q = batch.q_array(dtype::FLOAT16);
    let want = to_vec_f32(
        &crate::layers::paged_decode_attention_pooled_fallback(
            &q,
            &batch.pool,
            &batch.state_refs(),
            0,
            batch.scale(),
        )
        .unwrap(),
    );
    let got = to_vec_f32(
        &batch
            .pool
            .paged_decode_fused_v2(&q, &batch.state_refs(), 0, batch.scale())
            .unwrap()
            .expect("v2 serves this shape"),
    );
    let err = max_rel_error(&got, &want);
    assert!(
        err <= 2e-2,
        "f16 relative error {err} vs the gather reference"
    );
}

#[test]
fn the_entry_point_declines_a_batch_with_no_visible_tokens() {
    let batch = Batch::new(16, 4, 2, 64, &[32], &[32], dtype::FLOAT32, 0x5);
    let q = batch.q_array(dtype::FLOAT32);
    let out = batch
        .pool
        .paged_decode_fused_v2(&q, &batch.state_refs(), 0, batch.scale())
        .unwrap();
    assert!(out.is_none(), "an all-empty batch must decline, not launch");
}

// ---------------------------------------------------------------------------
// The merge kernel on its own (the piece issue #903 reuses unchanged)
// ---------------------------------------------------------------------------

#[test]
fn merge_kernel_matches_the_closed_form() {
    // Three output rows over 6 partials with a ragged grouping, including an
    // empty partial (lse = -inf) and a row with a single partial.
    let heads = 3usize;
    let dim = 8usize;
    let indptr = vec![0i32, 3, 4, 6];
    let n = 6usize;
    let mut rng = Rng::new(0xfeed);
    let v: Vec<f32> = (0..n * heads * dim).map(|_| rng.next_f32()).collect();
    let mut lse: Vec<f32> = (0..n * heads).map(|_| rng.next_f32() * 4.0).collect();
    lse[heads] = f32::NEG_INFINITY; // an empty chunk inside row 0

    let v_arr = ffi::from_slice_f32(&v, &[n as i32, heads as i32, dim as i32]);
    let lse_arr = ffi::from_slice_f32(&lse, &[n as i32, heads as i32]);
    let indptr_arr = ffi::from_slice_i32(&indptr, &[indptr.len() as i32]);
    let mut out_v = UniquePtr::null();
    let mut out_lse = UniquePtr::null();
    ffi::paged_attention_merge_states(&v_arr, &lse_arr, &indptr_arr, &mut out_v, &mut out_lse);
    let got_v = to_vec_f32(&out_v);
    let got_lse = to_vec_f32(&out_lse);

    for o in 0..indptr.len() - 1 {
        let begin = indptr[o] as usize;
        let end = indptr[o + 1] as usize;
        for h in 0..heads {
            let finite: Vec<usize> = (begin..end)
                .filter(|&i| lse[i * heads + h].is_finite())
                .collect();
            let m = finite
                .iter()
                .map(|&i| f64::from(lse[i * heads + h]))
                .fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for &i in &finite {
                denom += (f64::from(lse[i * heads + h]) - m).exp2();
            }
            for d in 0..dim {
                let mut acc = 0.0f64;
                for &i in &finite {
                    let w = (f64::from(lse[i * heads + h]) - m).exp2();
                    acc += w * f64::from(v[(i * heads + h) * dim + d]);
                }
                let want = if denom > 0.0 { acc / denom } else { 0.0 };
                let got = f64::from(got_v[(o * heads + h) * dim + d]);
                assert!(
                    (got - want).abs() < 1e-5,
                    "row {o} head {h} dim {d}: got {got} want {want}"
                );
            }
            let want_lse = if denom > 0.0 {
                m + denom.log2()
            } else {
                f64::NEG_INFINITY
            };
            let got = f64::from(got_lse[o * heads + h]);
            assert!(
                (got - want_lse).abs() < 1e-4,
                "row {o} head {h} lse: got {got} want {want_lse}"
            );
        }
    }
}

#[test]
fn merge_kernel_returns_zeros_for_an_all_empty_row() {
    let heads = 2usize;
    let dim = 4usize;
    let indptr = vec![0i32, 2];
    let v = vec![1.0f32; 2 * heads * dim];
    let lse = vec![f32::NEG_INFINITY; 2 * heads];
    let v_arr = ffi::from_slice_f32(&v, &[2, heads as i32, dim as i32]);
    let lse_arr = ffi::from_slice_f32(&lse, &[2, heads as i32]);
    let indptr_arr = ffi::from_slice_i32(&indptr, &[2]);
    let mut out_v = UniquePtr::null();
    let mut out_lse = UniquePtr::null();
    ffi::paged_attention_merge_states(&v_arr, &lse_arr, &indptr_arr, &mut out_v, &mut out_lse);
    assert!(to_vec_f32(&out_v).iter().all(|&x| x == 0.0));
    assert!(
        to_vec_f32(&out_lse)
            .iter()
            .all(|x| x.is_infinite() && *x < 0.0)
    );
}

#[test]
fn merge_is_associative_across_regroupings() {
    // The property issue #903 depends on: merging (a, b) then c equals merging
    // a then (b, c), so a cascade can decompose the same partials differently.
    let heads = 2usize;
    let dim = 8usize;
    let n = 4usize;
    let mut rng = Rng::new(0xabcd);
    let v: Vec<f32> = (0..n * heads * dim).map(|_| rng.next_f32()).collect();
    let lse: Vec<f32> = (0..n * heads).map(|_| rng.next_f32() * 3.0).collect();
    let v_arr = ffi::from_slice_f32(&v, &[n as i32, heads as i32, dim as i32]);
    let lse_arr = ffi::from_slice_f32(&lse, &[n as i32, heads as i32]);

    let merge = |ptr: &[i32], v_in: &MlxArray, lse_in: &MlxArray| {
        let arr = ffi::from_slice_i32(ptr, &[ptr.len() as i32]);
        let mut ov = UniquePtr::null();
        let mut ol = UniquePtr::null();
        ffi::paged_attention_merge_states(v_in, lse_in, &arr, &mut ov, &mut ol);
        (ov, ol)
    };

    // One shot over all four.
    let (all_v, _) = merge(&[0, 4], &v_arr, &lse_arr);
    // Two-stage: (0,1) and (2,3), then merge the two results.
    let (stage_v, stage_lse) = merge(&[0, 2, 4], &v_arr, &lse_arr);
    let (two_v, _) = merge(&[0, 2], &stage_v, &stage_lse);

    let a = to_vec_f32(&all_v);
    let b = to_vec_f32(&two_v);
    let err = max_rel_error(&b, &a);
    assert!(err < 1e-5, "regrouped merge deviates by {err}");
}
