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

//! Run step for two-level cascade decode (issue #903).
//!
//! Three launches and no new kernel:
//!
//! 1. **Level 0**, the shared span, as a *one-request* v2 launch whose query is
//!    every member's query stacked onto the head axis.
//! 2. **Level 1**, the whole batch, as an ordinary v2 launch over the page
//!    ranges the members do not share.
//! 3. **The merge**, issue #898's `paged_attention_merge_states`, over a
//!    grouping that pairs each member's two states and leaves everyone else's
//!    single state alone.
//!
//! ## Why level 0 is one request with many heads
//!
//! This is the whole trick, and it is what makes the shared span read once
//! instead of once per member.
//!
//! The v2 partial kernel gives one threadgroup all query heads of one KV head
//! for one `(request, page tile)` pair, and that threadgroup loads each K and V
//! element exactly once and reuses it across every query head it owns
//! (`paged_attention_v2.cpp`, the `for (uint g = 0; g < QHeads; g++)` loop over
//! a single `k_reg` / `v_reg` load). Handing the same page list to `M` separate
//! requests would therefore read the span `M` times. Handing it to one request
//! with `M` times as many query heads reads it once.
//!
//! The kernel maps query head `h` to KV head `h / NRep`, so the stacking order
//! is not free: it has to be **KV-head major**. With `G = Hq / Hkv`, member `m`
//! and group slot `g`, the level-0 head index is
//!
//! ```text
//!   h0 = kv_head * (M * G) + m * G + g
//! ```
//!
//! which gives `h0 / NRep0 = h0 / (M * G) = kv_head` exactly. That is one
//! `[M, Hkv, G, D] -> [Hkv, M, G, D]` transpose on the way in and its inverse
//! on the way out. Order the stack member-major instead and every query head
//! silently reads the wrong KV head: the launch succeeds, the shapes agree, and
//! the answer is wrong. `cascade_launch_tests` pins the correct order against a
//! flat reference.
//!
//! The consequence for the plan is that level 0's `NRep` is `M * G` rather than
//! `G`, so `PagedDecodeGeometry::check` has to pass at the larger value and the
//! kernel JIT-specializes once per distinct `M`. Both are handled by building a
//! separate [`PagedDecodeGeometry`] for level 0 and checking it before launch.
//!
//! ## The merge kernel is #898's, unchanged
//!
//! Same contract [`crate::mla::split_kv`] documents and consumes: `v_in`
//! `[N, H, D]` already normalized per partial, `lse_in` `[N, H]` **in log2
//! units**, `o_indptr` `[M + 1]` grouping contiguous rows. Both levels here are
//! v2 launches, so their LSE is already the `m + log2(l)` the partial kernel
//! emits and no unit conversion happens anywhere on this path. That is the
//! silent-failure clause: a natural-log LSE still merges and returns a
//! plausible weighted average, so `cascade_launch_tests` carries the same
//! negative control `mla::split_kv_tests::merge_rejects_natural_log_lse_units`
//! does, restated for cascade partials.
//!
//! Nothing in `paged_attention_v2.cpp`, `paged_attention_v2_merge.cpp` or the
//! FFI signature was edited for this issue.

use cxx::UniquePtr;

use crate::ffi::{self, MlxArray};
use crate::paged_v2::cascade::CascadePlan;
use crate::paged_v2::launch::V2Context;
use crate::paged_v2::plan::{PagedDecodeGeometry, PagedDecodePlan};

/// What a cascade launch actually did, for the caller's outcome report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CascadeLaunchStats {
    /// Chunks the level-0 (shared span) plan emitted.
    pub prefix_chunks: usize,
    /// Chunks the level-1 (per-request suffix) plan emitted.
    pub suffix_chunks: usize,
    /// Query heads of the stacked level-0 launch, `Hkv * M * G`.
    pub prefix_q_heads: i32,
}

/// Level-0 geometry: the flat geometry with the member count folded into the
/// query-head axis.
///
/// Returns `Err` when the fold overflows i32, which a batch large enough to
/// reach cannot be scheduled anyway.
pub fn prefix_geometry(
    geometry: PagedDecodeGeometry,
    members: usize,
) -> Result<PagedDecodeGeometry, String> {
    let members = i64::try_from(members).unwrap_or(i64::MAX);
    let stacked = i64::from(geometry.q_heads) * members;
    let q_heads = i32::try_from(stacked)
        .map_err(|_| format!("cascade: {stacked} stacked query heads overflows i32"))?;
    Ok(PagedDecodeGeometry {
        q_heads,
        ..geometry
    })
}

/// Stack `[M, Hq, 1, D]` member queries into the KV-head-major
/// `[1, Hkv * M * G, 1, D]` level-0 query.
fn stack_member_queries(
    q_members: &MlxArray,
    members: i32,
    geometry: PagedDecodeGeometry,
) -> UniquePtr<MlxArray> {
    let hkv = geometry.kv_heads;
    let n_rep = geometry.n_rep();
    let dim = geometry.head_dim;
    let split = ffi::reshape(q_members, &[members, hkv, n_rep, dim]);
    let kv_major = ffi::contiguous(&ffi::transpose_axes(&split, &[1, 0, 2, 3]), false);
    ffi::reshape(&kv_major, &[1, hkv * members * n_rep, 1, dim])
}

/// Undo [`stack_member_queries`] on a `[1, Hkv * M * G, trailing...]` result,
/// producing `[M, Hq, trailing...]`.
fn unstack_member_rows(
    stacked: &MlxArray,
    members: i32,
    geometry: PagedDecodeGeometry,
    trailing: Option<i32>,
) -> UniquePtr<MlxArray> {
    let hkv = geometry.kv_heads;
    let n_rep = geometry.n_rep();
    let hq = geometry.q_heads;
    match trailing {
        Some(dim) => {
            let split = ffi::reshape(stacked, &[hkv, members, n_rep, dim]);
            let request_major = ffi::contiguous(&ffi::transpose_axes(&split, &[1, 0, 2, 3]), false);
            ffi::reshape(&request_major, &[members, hq, dim])
        }
        None => {
            let split = ffi::reshape(stacked, &[hkv, members, n_rep]);
            let request_major = ffi::contiguous(&ffi::transpose_axes(&split, &[1, 0, 2]), false);
            ffi::reshape(&request_major, &[members, hq])
        }
    }
}

/// Run one two-level cascade decode step.
///
/// `q` is `[B, Hq, 1, D]` f32, the pools are the layer's single-slab tensors,
/// and the result is `[B, Hq, 1, D]` f32 in request order, numerically the same
/// attention the flat v2 launch computes (the two levels partition the same key
/// range, and the merge of disjoint softmax states is exact up to f32
/// rounding).
///
/// Errors rather than declining: the caller has already decided the shape is
/// servable through [`crate::paged_v2::cascade::build_cascade_plan`], so a
/// failure here is a bug and must not be papered over with a silent fallback
/// that would make a benchmark compare the flat path against itself. The one
/// production caller answers an error by falling back *and saying so*.
pub fn run_cascade_decode(
    q: &MlxArray,
    k_pool: &MlxArray,
    v_pool: &MlxArray,
    plan: &CascadePlan,
    geometry: PagedDecodeGeometry,
    scale: f32,
    target_ctas: usize,
) -> Result<(UniquePtr<MlxArray>, CascadeLaunchStats), String> {
    plan.validate()?;
    geometry.check()?;
    let batch = i32::try_from(plan.batch())
        .map_err(|_| format!("cascade: batch {} overflows i32", plan.batch()))?;
    let members = i32::try_from(plan.members())
        .map_err(|_| format!("cascade: member count {} overflows i32", plan.members()))?;
    let hq = geometry.q_heads;
    let dim = geometry.head_dim;

    // -- Level 0: the shared span, one request, member queries stacked. --
    let member_index = ffi::from_slice_i32(&plan.member_rows, &[members]);
    let q_members = if plan.members_are_whole_batch() {
        // The gather would be the identity; skip it rather than pay for a copy
        // in the case this feature exists for (every client behind one prompt).
        ffi::contiguous(q, false)
    } else {
        ffi::take(q, &member_index, 0)
    };
    let q_prefix = stack_member_queries(&q_members, members, geometry);

    let geometry0 = prefix_geometry(geometry, plan.members())?;
    geometry0
        .check()
        .map_err(|e| format!("cascade: stacked level-0 geometry is unservable ({e})"))?;
    let prefix_pages = plan.prefix_view.page_counts();
    let prefix_plan = PagedDecodePlan::heuristic(geometry0, &prefix_pages, target_ctas);
    let prefix_ctx = V2Context::build(
        &q_prefix,
        k_pool,
        v_pool,
        &plan.prefix_view,
        geometry0,
        scale,
    )?;
    let (prefix_v, prefix_lse) = prefix_ctx.launch_with_lse(&prefix_plan)?;
    let prefix_v = unstack_member_rows(&prefix_v, members, geometry, Some(dim));
    let prefix_lse = unstack_member_rows(&prefix_lse, members, geometry, None);

    // -- Level 1: the whole batch over the unshared ranges. --
    let suffix_pages = plan.suffix_view.page_counts();
    let suffix_plan = PagedDecodePlan::heuristic(geometry, &suffix_pages, target_ctas);
    let suffix_ctx = V2Context::build(q, k_pool, v_pool, &plan.suffix_view, geometry, scale)?;
    let (suffix_v, suffix_lse) = suffix_ctx.launch_with_lse(&suffix_plan)?;

    // -- Merge: #898's kernel over `concat(level 1, level 0)` reordered so each
    // request's partials are contiguous. --
    let order = ffi::from_slice_i32(&plan.merge_order, &[plan.merge_order.len() as i32]);
    let v_cat = crate::concatenate(&suffix_v, &prefix_v, 0);
    let lse_cat = crate::concatenate(&suffix_lse, &prefix_lse, 0);
    let v_in = ffi::take(&v_cat, &order, 0);
    let lse_in = ffi::take(&lse_cat, &order, 0);
    let o_indptr = ffi::from_slice_i32(&plan.o_indptr, &[batch + 1]);

    let mut merged_v = UniquePtr::null();
    let mut merged_lse = UniquePtr::null();
    ffi::paged_attention_merge_states(&v_in, &lse_in, &o_indptr, &mut merged_v, &mut merged_lse);

    let stats = CascadeLaunchStats {
        prefix_chunks: prefix_plan.num_chunks,
        suffix_chunks: suffix_plan.num_chunks,
        prefix_q_heads: geometry0.q_heads,
    };
    Ok((ffi::reshape(&merged_v, &[batch, hq, 1, dim]), stats))
}

#[cfg(test)]
#[path = "cascade_launch_tests.rs"]
mod cascade_launch_tests;
