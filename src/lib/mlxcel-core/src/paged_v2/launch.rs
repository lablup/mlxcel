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

//! Run step for paged decode v2 (issue #898).
//!
//! Turns a [`PagedCsrView`] plus a [`PagedDecodePlan`] into MLX arrays and the
//! one or two kernel launches they imply:
//!
//! 1. The partial kernel always runs, producing `(partial_v, lse)` over the
//!    plan's chunks.
//! 2. The merge kernel runs only when some request has more than one chunk.
//!    When every request fits in a single chunk the partial output already *is*
//!    the answer (the plan emits chunks in request order, so chunk `r` is
//!    request `r`), and it is reshaped rather than merged. That is the issue's
//!    "write O directly" case, decided on the host so no output element can be
//!    left unwritten by a kernel that skipped it.
//!
//! The workspace is the partial kernel's output arrays. An MLX custom kernel
//! cannot write through a caller-owned buffer (outputs are values in the graph),
//! so "caller-provided workspace" becomes "MLX-allocated outputs whose exact
//! size the plan states up front" via [`PagedDecodePlan::workspace_bytes`].

use cxx::UniquePtr;

use crate::autotune::Source;
use crate::cache::paged_csr::PagedCsrView;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::paged_v2::plan::{PagedDecodeGeometry, PagedDecodePlan, device_target_ctas};

/// Everything a v2 launch needs except the plan.
///
/// Holds the CSR arrays because they are plan-independent: the autotuner
/// profiles several plans against one context, and rebuilding the page table
/// per candidate would measure array construction instead of the kernel.
pub struct V2Context<'a> {
    /// `[B, Hq, 1, D]` f32 decode query.
    pub q: &'a MlxArray,
    /// `[num_blocks, page_size, Hkv, D]` f16 pool tensors.
    pub k_pool: &'a MlxArray,
    pub v_pool: &'a MlxArray,
    pub indices: UniquePtr<MlxArray>,
    pub indptr: UniquePtr<MlxArray>,
    pub last_page_len: UniquePtr<MlxArray>,
    pub first_page_offset: UniquePtr<MlxArray>,
    pub scale: f32,
    pub geometry: PagedDecodeGeometry,
}

impl<'a> V2Context<'a> {
    /// Materialize the CSR view as MLX arrays.
    pub fn build(
        q: &'a MlxArray,
        k_pool: &'a MlxArray,
        v_pool: &'a MlxArray,
        view: &PagedCsrView,
        geometry: PagedDecodeGeometry,
        scale: f32,
    ) -> Result<Self, String> {
        view.validate()?;
        let batch = view.batch() as i32;
        Ok(Self {
            q,
            k_pool,
            v_pool,
            indices: ffi::from_slice_i32(&view.indices, &[view.indices.len() as i32]),
            indptr: ffi::from_slice_i32(&view.indptr, &[batch + 1]),
            last_page_len: ffi::from_slice_i32(&view.last_page_len, &[batch]),
            first_page_offset: ffi::from_slice_i32(&view.first_page_offset, &[batch]),
            scale,
            geometry,
        })
    }

    /// Launch the plan and return `[B, Hq, 1, D]` f32.
    pub fn launch(&self, plan: &PagedDecodePlan) -> Result<UniquePtr<MlxArray>, String> {
        plan.validate()?;
        let n = i32::try_from(plan.num_chunks)
            .map_err(|_| format!("paged decode v2: {} chunks overflows i32", plan.num_chunks))?;
        let request_indices = ffi::from_slice_i32(&plan.request_indices, &[n]);
        let kv_tile_indices = ffi::from_slice_i32(&plan.kv_tile_indices, &[n]);
        let params = ffi::from_slice_i32(&[plan.pages_per_chunk], &[1]);

        let mut partial_v = UniquePtr::null();
        let mut lse = UniquePtr::null();
        ffi::paged_attention_decode_v2_partial(
            self.q,
            self.k_pool,
            self.v_pool,
            &self.indices,
            &self.indptr,
            &self.last_page_len,
            &self.first_page_offset,
            &request_indices,
            &kv_tile_indices,
            &params,
            self.scale,
            &mut partial_v,
            &mut lse,
        );

        let batch = i32::try_from(plan.batch)
            .map_err(|_| format!("paged decode v2: batch {} overflows i32", plan.batch))?;
        let hq = self.geometry.q_heads;
        let dim = self.geometry.head_dim;

        if !plan.needs_merge {
            // One chunk per request, emitted in request order: the partial
            // output is [B, Hq, D] already normalized.
            return Ok(ffi::reshape(&partial_v, &[batch, hq, 1, dim]));
        }

        let o_indptr = ffi::from_slice_i32(&plan.o_indptr, &[batch + 1]);
        let mut merged_v = UniquePtr::null();
        let mut merged_lse = UniquePtr::null();
        ffi::paged_attention_merge_states(
            &partial_v,
            &lse,
            &o_indptr,
            &mut merged_v,
            &mut merged_lse,
        );
        Ok(ffi::reshape(&merged_v, &[batch, hq, 1, dim]))
    }
}

/// Geometry implied by a query and a pool tensor.
///
/// `q` is `[B, Hq, 1, D]` and the pool is `[num_blocks, page_size, Hkv, D]`,
/// the layout-A shapes v1 already uses.
pub fn geometry_from_shapes(
    q: &MlxArray,
    k_pool: &MlxArray,
) -> Result<(usize, PagedDecodeGeometry), String> {
    let q_shape = ffi::array_shape(q);
    let pool_shape = ffi::array_shape(k_pool);
    if q_shape.len() != 4 || pool_shape.len() != 4 {
        return Err(format!(
            "paged decode v2: expected 4-D q and pool, got {q_shape:?} and {pool_shape:?}"
        ));
    }
    let batch = q_shape[0].max(0) as usize;
    let geometry = PagedDecodeGeometry {
        q_heads: q_shape[1],
        kv_heads: pool_shape[2],
        head_dim: q_shape[3],
        page_size: pool_shape[1],
    };
    if geometry.head_dim != pool_shape[3] {
        return Err(format!(
            "paged decode v2: q head_dim {} disagrees with pool head_dim {}",
            geometry.head_dim, pool_shape[3]
        ));
    }
    Ok((batch, geometry))
}

/// Full v2 decode over one layer's pool.
///
/// Returns `Ok(None)` when v2 declines the shape (a geometry it cannot serve,
/// or a batch with no visible tokens at all), which the caller answers by
/// falling back exactly as it would if the fused kernel were unavailable.
/// `Ok(Some(out))` is `[B, Hq, 1, D]` f32.
pub fn run_decode_v2(
    q: &MlxArray,
    k_pool: &MlxArray,
    v_pool: &MlxArray,
    view: &PagedCsrView,
    scale: f32,
) -> Result<Option<UniquePtr<MlxArray>>, String> {
    if !view.any_visible() {
        return Ok(None);
    }
    let (batch, geometry) = geometry_from_shapes(q, k_pool)?;
    if batch != view.batch() {
        return Err(format!(
            "paged decode v2: q batch {batch} disagrees with the {} requests in the page view",
            view.batch()
        ));
    }
    if geometry.page_size != view.page_size {
        return Err(format!(
            "paged decode v2: pool page_size {} disagrees with the view's {}",
            geometry.page_size, view.page_size
        ));
    }
    if let Err(reason) = geometry.check() {
        tracing::debug!("paged decode v2 declines this shape: {reason}");
        return Ok(None);
    }

    let ctx = V2Context::build(q, k_pool, v_pool, view, geometry, scale)?;
    let page_counts = view.page_counts();
    let plan = resolve_plan(&ctx, &page_counts);
    if let Err(reason) = plan.validate() {
        tracing::debug!("paged decode v2 declines this plan: {reason}");
        return Ok(None);
    }
    ctx.launch(&plan).map(Some)
}

/// Pick the plan for this launch: the autotuned chunk size when one is
/// available, the binary-search heuristic otherwise.
///
/// With `MLXCEL_AUTOTUNE` unset (the default) this is one env read behind a
/// `OnceLock` plus the heuristic, with no cache read, no lock, and no
/// filesystem access.
pub fn resolve_plan(ctx: &V2Context<'_>, page_counts: &[usize]) -> PagedDecodePlan {
    let target = device_target_ctas();
    let heuristic = PagedDecodePlan::heuristic(ctx.geometry, page_counts, target);
    let (chunk, source) = crate::autotune::ops::paged_decode_v2_chunk::resolve_pages_per_chunk(
        ctx,
        page_counts,
        target,
        heuristic.pages_per_chunk,
    );
    if chunk == heuristic.pages_per_chunk && source == Source::Default {
        return heuristic;
    }
    PagedDecodePlan::with_chunk_size(ctx.geometry, page_counts, chunk, target, source)
}

#[cfg(test)]
#[path = "launch_tests.rs"]
mod launch_tests;
