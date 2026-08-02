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

//! GPU correctness tests for fused sparse decode (issue #904).
//!
//! ## What is being proved, and why it is not the output tensor alone
//!
//! A sparse kernel that reads the wrong rows still produces a plausible output:
//! same shape, same magnitude, a convex combination of *some* value vectors. So
//! the tests here check the selection twice over, from both ends.
//!
//! - [`the_fused_output_matches_a_dense_reference_restricted_to_the_selection`]
//!   compares against a host reference that attends to exactly the selected
//!   positions and nothing else. That reference is, by construction, the dense
//!   softmax with a `-inf` mask on every unselected column, which is the
//!   implementation this path replaces.
//! - [`the_fused_output_differs_from_full_dense_attention`] requires the answer
//!   to be *wrong* as a dense answer. Without it, a kernel that quietly ignored
//!   the page list and swept the whole context could still pass a
//!   "close to reference" test whenever the selection happened to be broad.
//! - [`changing_one_selected_position_changes_the_output`] pins the sensitivity
//!   the other direction: a single different row must move the result.
//! - [`the_selected_rows_decode_back_to_the_intended_positions`] checks the
//!   encoding itself, so a failure attributes to the page table rather than to
//!   the kernel.

use super::*;
use crate::cache::sparse_csr::{ContiguousCacheLayout, selection_from_positions};
use crate::dtype;
use crate::ffi;

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

/// Shape of the synthetic decode step.
///
/// `k_buffer_heads` may exceed `kv_heads`: MiniMax-M3 rides its index key on
/// the K head axis, so the K allocation is one head wider than the V one, and
/// the sparse path has to address both from one row list.
struct Case {
    batch: i32,
    kv_heads: i32,
    n_rep: i32,
    head_dim: i32,
    capacity: i32,
    live_len: i32,
    k_buffer_heads: i32,
    selected_per_request: usize,
}

impl Case {
    /// The default shape: one sequence, four KV heads, two query heads each.
    ///
    /// `4 requests x 512 selected` is 2048 selected rows, which is exactly the
    /// shipped batched floor (`4 * MIN_BATCHED_KV_TOKENS_PER_REQUEST`). Sized
    /// deliberately so the test exercises the dispatched path under the real
    /// policy instead of an environment override, since the floor is cached in
    /// a process-wide `OnceLock` that a test cannot safely move.
    fn default_case() -> Self {
        Self {
            batch: 1,
            kv_heads: 4,
            n_rep: 2,
            head_dim: 64,
            capacity: 1024,
            live_len: 1024,
            k_buffer_heads: 4,
            selected_per_request: 512,
        }
    }

    fn q_heads(&self) -> i32 {
        self.kv_heads * self.n_rep
    }

    fn scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }

    fn k_layout(&self) -> ContiguousCacheLayout {
        ContiguousCacheLayout {
            batch: self.batch,
            buffer_heads: self.k_buffer_heads,
            capacity: self.capacity,
        }
    }

    fn v_layout(&self) -> ContiguousCacheLayout {
        ContiguousCacheLayout {
            batch: self.batch,
            buffer_heads: self.kv_heads,
            capacity: self.capacity,
        }
    }
}

/// The host-side values behind one synthetic step, so a reference can be
/// computed without reading anything back from the GPU.
struct Fixture {
    case: Case,
    /// `[B * H_k * Cap * D]` K allocation, flat.
    k: Vec<f32>,
    /// `[B * H_v * Cap * D]` V allocation, flat.
    v: Vec<f32>,
    /// `[B * Hq * D]` decode query, flat.
    q: Vec<f32>,
    /// `[b][h]` selected live-window positions.
    positions: Vec<Vec<Vec<i32>>>,
}

impl Fixture {
    fn new(case: Case, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let k_len = (case.batch * case.k_buffer_heads * case.capacity * case.head_dim) as usize;
        let v_len = (case.batch * case.kv_heads * case.capacity * case.head_dim) as usize;
        let q_len = (case.batch * case.q_heads() * case.head_dim) as usize;
        let k: Vec<f32> = (0..k_len).map(|_| rng.next_f32()).collect();
        let v: Vec<f32> = (0..v_len).map(|_| rng.next_f32()).collect();
        let q: Vec<f32> = (0..q_len).map(|_| rng.next_f32()).collect();

        // A scattered, unordered, per-head-distinct selection. Distinct per
        // head matters: a bug that reuses one head's row list for every head
        // would otherwise be invisible.
        let stride = (case.live_len as usize / case.selected_per_request).max(1);
        let positions = (0..case.batch)
            .map(|b| {
                (0..case.kv_heads)
                    .map(|h| {
                        let offset = ((b * case.kv_heads + h) as usize * 7) % stride;
                        let mut list: Vec<i32> = (0..case.selected_per_request)
                            .map(|i| ((i * stride + offset) % case.live_len as usize) as i32)
                            .collect();
                        // Reverse odd heads so the row list is not monotone.
                        if h % 2 == 1 {
                            list.reverse();
                        }
                        list
                    })
                    .collect()
            })
            .collect();

        Self {
            case,
            k,
            v,
            q,
            positions,
        }
    }

    fn k_alloc(&self, dtype_id: i32) -> UniquePtr<MlxArray> {
        let shape = [
            self.case.batch,
            self.case.k_buffer_heads,
            self.case.capacity,
            self.case.head_dim,
        ];
        cast(&ffi::from_slice_f32(&self.k, &shape), dtype_id)
    }

    fn v_alloc(&self, dtype_id: i32) -> UniquePtr<MlxArray> {
        let shape = [
            self.case.batch,
            self.case.kv_heads,
            self.case.capacity,
            self.case.head_dim,
        ];
        cast(&ffi::from_slice_f32(&self.v, &shape), dtype_id)
    }

    fn q_array(&self) -> UniquePtr<MlxArray> {
        ffi::from_slice_f32(
            &self.q,
            &[self.case.batch, self.case.q_heads(), 1, self.case.head_dim],
        )
    }

    fn selection(&self) -> SparseSelection {
        selection_from_positions(
            &self.case.k_layout(),
            self.case.kv_heads,
            self.case.live_len,
            &self.positions,
        )
        .expect("selection")
    }

    /// Host attention over an explicit position list, in f64.
    ///
    /// This *is* the dense reference with a `-inf` mask on every unselected
    /// column: masking a column to `-inf` contributes `exp(-inf) = 0` to both
    /// the softmax denominator and the value sum, which is the same number as
    /// omitting it from the sum. Computing it as a sum over the kept set rather
    /// than a masked sweep keeps the reference independent of the code under
    /// test.
    fn reference(&self, positions: &[Vec<Vec<i32>>]) -> Vec<f32> {
        let c = &self.case;
        let (hq, d) = (c.q_heads() as usize, c.head_dim as usize);
        let mut out = vec![0.0f32; (c.batch as usize) * hq * d];
        for b in 0..c.batch as usize {
            for h in 0..c.kv_heads as usize {
                let kb = c.k_layout().base(b as i32, h as i32) as usize;
                let vb = c.v_layout().base(b as i32, h as i32) as usize;
                for g in 0..c.n_rep as usize {
                    let hq_idx = h * c.n_rep as usize + g;
                    let q0 = (b * hq + hq_idx) * d;
                    let mut scores: Vec<f64> = Vec::with_capacity(positions[b][h].len());
                    for &t in &positions[b][h] {
                        let k0 = (kb + t as usize) * d;
                        let mut dot = 0.0f64;
                        for i in 0..d {
                            dot += f64::from(self.q[q0 + i]) * f64::from(self.k[k0 + i]);
                        }
                        scores.push(dot * f64::from(c.scale()));
                    }
                    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                    let denom: f64 = exps.iter().sum();
                    let o0 = (b * hq + hq_idx) * d;
                    for (p, &t) in positions[b][h].iter().enumerate() {
                        let v0 = (vb + t as usize) * d;
                        let w = exps[p] / denom;
                        for i in 0..d {
                            out[o0 + i] += (w * f64::from(self.v[v0 + i])) as f32;
                        }
                    }
                }
            }
        }
        out
    }

    /// Every live position, for the "this must not be dense" check.
    fn all_positions(&self) -> Vec<Vec<Vec<i32>>> {
        (0..self.case.batch)
            .map(|_| {
                (0..self.case.kv_heads)
                    .map(|_| (0..self.case.live_len).collect())
                    .collect()
            })
            .collect()
    }

    fn inputs<'a>(&self, q: &'a MlxArray, k: &'a MlxArray, v: &'a MlxArray) -> SparseDecodeInputs<'a> {
        SparseDecodeInputs {
            q,
            k_alloc: k,
            v_alloc: v,
            kv_heads: self.case.kv_heads,
            scale: self.case.scale(),
        }
    }
}

fn cast(a: &MlxArray, dtype_id: i32) -> UniquePtr<MlxArray> {
    ffi::astype(a, dtype_id)
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = ffi::astype(a, dtype::FLOAT32);
    ffi::eval(&f);
    ffi::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// Max absolute deviation relative to the reference's own scale.
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

/// Run the fused path and assert it dispatched, returning the output.
fn run_fused(fx: &Fixture, dtype_id: i32) -> (Vec<f32>, SparseDecodeOutcome) {
    let q = fx.q_array();
    let k = fx.k_alloc(dtype_id);
    let v = fx.v_alloc(dtype_id);
    let inputs = fx.inputs(&q, &k, &v);
    let (out, outcome) = run_sparse_decode(&inputs, &fx.selection());
    let out = out.unwrap_or_else(|| panic!("sparse decode declined: {}", outcome.describe()));
    (to_vec_f32(&out), outcome)
}

// ---------------------------------------------------------------------------
// Correctness of the selection, from both ends
// ---------------------------------------------------------------------------

#[test]
fn the_selected_rows_decode_back_to_the_intended_positions() {
    let fx = Fixture::new(Case::default_case(), 0x5eed_0904);
    let sel = fx.selection();
    let rows = sel.materialize();
    assert_eq!(rows.len(), (fx.case.batch * fx.case.kv_heads) as usize);
    for b in 0..fx.case.batch as usize {
        for h in 0..fx.case.kv_heads as usize {
            let r = b * fx.case.kv_heads as usize + h;
            let base = fx.case.k_layout().base(b as i32, h as i32);
            let decoded: Vec<i32> = rows[r].iter().map(|&row| row - base).collect();
            assert_eq!(
                decoded, fx.positions[b][h],
                "request {r} (sequence {b}, head {h}) does not decode to its selection"
            );
        }
    }
}

#[test]
fn the_fused_output_matches_a_dense_reference_restricted_to_the_selection() {
    let fx = Fixture::new(Case::default_case(), 0x5eed_0001);
    let (got, outcome) = run_fused(&fx, dtype::FLOAT32);
    assert!(outcome.is_fused(), "{}", outcome.describe());
    let want = fx.reference(&fx.positions);
    let err = max_rel_error(&got, &want);
    assert!(err < 2e-3, "max relative error {err} against the masked-dense reference");
}

#[test]
fn the_fused_output_differs_from_full_dense_attention() {
    // The load-bearing negative control. A kernel that ignored the page list
    // and swept the whole live window would still pass a "close to reference"
    // test whenever the selection is broad, so the sparse answer is required to
    // be a *wrong* dense answer.
    let fx = Fixture::new(Case::default_case(), 0x5eed_0002);
    let (got, _) = run_fused(&fx, dtype::FLOAT32);
    let dense = fx.reference(&fx.all_positions());
    let err = max_rel_error(&got, &dense);
    assert!(
        err > 1e-2,
        "sparse output is within {err} of full dense attention; the page list may be ignored"
    );
}

#[test]
fn changing_one_selected_position_changes_the_output() {
    let fx = Fixture::new(Case::default_case(), 0x5eed_0003);
    let (base, _) = run_fused(&fx, dtype::FLOAT32);

    let mut moved = fx.positions.clone();
    // Swap one row of one head for a position that head did not select.
    let taken: std::collections::HashSet<i32> = moved[0][0].iter().copied().collect();
    let replacement = (0..fx.case.live_len)
        .find(|t| !taken.contains(t))
        .expect("an unselected position exists");
    moved[0][0][0] = replacement;

    let q = fx.q_array();
    let k = fx.k_alloc(dtype::FLOAT32);
    let v = fx.v_alloc(dtype::FLOAT32);
    let sel = selection_from_positions(
        &fx.case.k_layout(),
        fx.case.kv_heads,
        fx.case.live_len,
        &moved,
    )
    .unwrap();
    let (out, outcome) = run_sparse_decode(&fx.inputs(&q, &k, &v), &sel);
    let out = to_vec_f32(&out.unwrap_or_else(|| panic!("{}", outcome.describe())));

    let d = fx.case.head_dim as usize;
    let changed = base[..d]
        .iter()
        .zip(&out[..d])
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        changed,
        "replacing a selected row left head 0 unchanged; the row list is not reaching the kernel"
    );
    // And the moved selection must still be right, not merely different.
    let want = fx.reference(&moved);
    assert!(max_rel_error(&out, &want) < 2e-3);
}

#[test]
fn an_f16_cache_matches_the_reference_within_its_own_precision() {
    // The realistic dtype: the production allocations are f16, so the pool the
    // kernel reads is f16 and the tolerance is set by the storage, not the
    // kernel.
    let fx = Fixture::new(Case::default_case(), 0x5eed_0004);
    let (got, outcome) = run_fused(&fx, dtype::FLOAT16);
    assert!(outcome.is_fused(), "{}", outcome.describe());
    let want = fx.reference(&fx.positions);
    let err = max_rel_error(&got, &want);
    assert!(err < 3e-2, "max relative error {err} on an f16 cache");
}

#[test]
fn a_k_allocation_with_a_side_head_is_addressed_by_the_same_row_list() {
    // MiniMax-M3's shape: the index key rides at head `kv_heads` of the K
    // allocation, so K is one head wider than V. At batch 1 the row bases
    // coincide and the extra head sits past every row the selection names.
    let mut case = Case::default_case();
    case.k_buffer_heads = case.kv_heads + 1;
    let fx = Fixture::new(case, 0x5eed_0005);
    let (got, outcome) = run_fused(&fx, dtype::FLOAT32);
    assert!(outcome.is_fused(), "{}", outcome.describe());
    let want = fx.reference(&fx.positions);
    assert!(max_rel_error(&got, &want) < 2e-3);
}

#[test]
fn the_plan_splits_and_merges_without_changing_the_answer() {
    // A wider selection forces the plan past one chunk per request, so the
    // merge pass runs. The answer must not move.
    let mut case = Case::default_case();
    case.selected_per_request = 1024;
    let fx = Fixture::new(case, 0x5eed_0006);
    let (got, outcome) = run_fused(&fx, dtype::FLOAT32);
    assert!(outcome.is_fused(), "{}", outcome.describe());
    let want = fx.reference(&fx.positions);
    assert!(max_rel_error(&got, &want) < 2e-3);
}

// ---------------------------------------------------------------------------
// Declines
// ---------------------------------------------------------------------------

#[test]
fn a_small_selection_is_declined_by_the_dispatch_floor() {
    let mut case = Case::default_case();
    case.selected_per_request = 32;
    let fx = Fixture::new(case, 0x5eed_0007);
    let q = fx.q_array();
    let k = fx.k_alloc(dtype::FLOAT32);
    let v = fx.v_alloc(dtype::FLOAT32);
    let (out, outcome) = run_sparse_decode(&fx.inputs(&q, &k, &v), &fx.selection());
    assert!(out.is_none());
    match outcome {
        SparseDecodeOutcome::BelowFloor {
            requests,
            selected,
            floor,
        } => {
            assert_eq!(requests, 4);
            assert_eq!(selected, 128);
            assert!(floor > selected, "floor {floor} must exceed {selected}");
        }
        other => panic!("expected BelowFloor, got {other:?}"),
    }
}

#[test]
fn a_batched_side_head_layout_is_declined_rather_than_mis_addressed() {
    // Beyond batch 1 the K and V sequence strides diverge, so one row list
    // cannot address both. Declining is the only correct answer; reading the
    // K row out of the V allocation would silently attend to another
    // sequence's values.
    let mut case = Case::default_case();
    case.batch = 2;
    case.k_buffer_heads = case.kv_heads + 1;
    case.selected_per_request = 512;
    let fx = Fixture::new(case, 0x5eed_0008);
    let q = fx.q_array();
    let k = fx.k_alloc(dtype::FLOAT32);
    let v = fx.v_alloc(dtype::FLOAT32);
    let (out, outcome) = run_sparse_decode(&fx.inputs(&q, &k, &v), &fx.selection());
    assert!(out.is_none());
    assert!(
        matches!(outcome, SparseDecodeOutcome::UnservableLayout(_)),
        "{outcome:?}"
    );
}

#[test]
fn a_selection_whose_request_count_disagrees_with_the_launch_is_rejected() {
    let fx = Fixture::new(Case::default_case(), 0x5eed_0009);
    let q = fx.q_array();
    let k = fx.k_alloc(dtype::FLOAT32);
    let v = fx.v_alloc(dtype::FLOAT32);
    let bogus = SparseSelection::from_host(3, 512, vec![0; 3 * 512]).unwrap();
    let (out, outcome) = run_sparse_decode(&fx.inputs(&q, &k, &v), &bogus);
    assert!(out.is_none());
    assert!(
        matches!(outcome, SparseDecodeOutcome::SelectionRejected(_)),
        "{outcome:?}"
    );
}

#[test]
fn every_outcome_kind_has_a_distinct_index_and_a_message() {
    let all = [
        SparseDecodeOutcome::Fused {
            requests: 4,
            selected_per_request: 512,
            chunks: 8,
            merged: true,
        },
        SparseDecodeOutcome::KillSwitch,
        SparseDecodeOutcome::BelowFloor {
            requests: 4,
            selected: 128,
            floor: 2048,
        },
        SparseDecodeOutcome::UnservableGeometry("x".to_string()),
        SparseDecodeOutcome::UnservableLayout("x".to_string()),
        SparseDecodeOutcome::SelectionRejected("x".to_string()),
        SparseDecodeOutcome::PlanRejected("x".to_string()),
        SparseDecodeOutcome::NotServable("x"),
    ];
    assert_eq!(all.len(), SPARSE_DECODE_OUTCOME_KINDS);
    let mut seen = [false; SPARSE_DECODE_OUTCOME_KINDS];
    for outcome in &all {
        let i = outcome.kind_index();
        assert!(i < SPARSE_DECODE_OUTCOME_KINDS, "{outcome:?} index {i}");
        assert!(!seen[i], "duplicate index for {outcome:?}");
        seen[i] = true;
        assert!(!outcome.describe().is_empty());
    }
    assert!(all[0].is_fused());
    assert!(!all[1].is_fused());
}

#[test]
fn the_floor_message_carries_both_numbers() {
    let text = SparseDecodeOutcome::BelowFloor {
        requests: 4,
        selected: 128,
        floor: 2048,
    }
    .describe();
    assert!(text.contains("128"), "{text}");
    assert!(text.contains("2048"), "{text}");
}

#[test]
fn the_kill_switch_names_the_variable_that_set_it() {
    let text = SparseDecodeOutcome::KillSwitch.describe();
    assert!(text.contains(SPARSE_PAGED_ENV), "{text}");
}

#[test]
fn reporting_a_kind_twice_emits_once_and_counts_twice() {
    reset_sparse_reported();
    let before = sparse_decode_stats();
    let outcome = SparseDecodeOutcome::NotServable("test");
    report_sparse_outcome_once(&outcome);
    report_sparse_outcome_once(&outcome);
    let after = sparse_decode_stats();
    assert_eq!(after.fallbacks - before.fallbacks, 2);
    reset_sparse_reported();
}
