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

//! Stage 2: split-KV absorbed MLA decode over the latent cache (issue #907).
//!
//! ## Why split at all
//!
//! Absorbed decode is one query row per head against `S` latent rows. However
//! that work is expressed, the parallelism available without a split is bounded
//! by `batch * num_heads`, and adding context adds none: it makes each unit of
//! work longer. Small batch times long context is exactly the shape that leaves
//! the machine idle, which is the same observation issue #898 made about paged
//! decode. The fix is the same: cut the KV range into chunks that are reduced
//! independently and combine the partial softmax states afterwards.
//!
//! ## The merge kernel is #898's, unchanged
//!
//! [`crate::ffi::paged_attention_merge_states`] takes
//!
//! ```text
//!   v_in     [N, H, D] f32   partials, each already divided by its own denominator
//!   lse_in   [N, H]    f32   matching log-sum-exp, in log2 units
//!   o_indptr [M + 1]   i32   output row o merges partial rows [o_indptr[o], o_indptr[o+1])
//! ```
//!
//! and knows nothing about pages, block tables, or GQA. Its contract note calls
//! that out explicitly, and `paged_v2::launch_tests::merge_is_associative_across_regroupings`
//! pins the property this path depends on: the result does not depend on how the
//! partials were grouped, only on which partials a row covers.
//!
//! MLA fits the contract without changing it. `H` is `num_heads` (all of which
//! share the single latent KV head), `D` is `kv_lora_rank`, `M` is the batch,
//! and `N` is `batch * chunks`. Nothing in this module edits the merge kernel,
//! the FFI signature, or the C++ launcher. Two contract details are load-bearing
//! and are honoured here rather than worked around: the LSE must be in **log2**
//! units (this module multiplies a natural-log `logsumexp` by `log2(e)`, see
//! [`LOG2_E`]), and a partial with `-inf` LSE contributes nothing (which is what
//! makes a trailing short chunk safe).
//!
//! ## What produces the partials
//!
//! The partial producer here is composed from MLX ops, one chunk at a time. It
//! is correct, it exercises the full split-and-merge decomposition, and it is
//! the reference a fused partial kernel is validated against. It is **not** a
//! speed win on its own: `C` small matmuls do the same arithmetic as one large
//! one with `C` times the launch overhead. Ship the fused partial kernel before
//! quoting a Stage 2 throughput number.

use cxx::UniquePtr;

use crate::ffi::{self, MlxArray};
use crate::mla::absorb::MlaAbsorbedProjections;
use crate::mla::decode::{absorb_queries, unabsorb_output};
use crate::mla::stats::{self, MlaDecodePath};

/// `log2(e)`. Converts a natural-log `logsumexp` into the merge kernel's log2
/// units, which is the single most likely thing to get silently wrong when
/// reusing #898's kernel from a new caller: a natural-log LSE still merges, it
/// just weights the partials by `e^(lse)` raised to `log2(e)` and produces a
/// plausible-looking wrong answer.
pub const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// The smallest chunk worth cutting. Below this the per-chunk fixed cost
/// dominates the reduction it performs.
pub const MIN_CHUNK_LEN: i32 = 128;

/// How the latent range is cut for one decode step.
///
/// Uniform: a dense (non-paged) KV cache is rectangular, so every request in
/// the batch has the same live length and therefore the same chunk count. That
/// is what lets [`Self::o_indptr`] be an arithmetic sequence instead of a
/// per-request scan. A ragged batch would need per-request chunk counts, which
/// the merge kernel already supports through `o_indptr`; only this plan would
/// change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaSplitPlan {
    /// Live latent rows per request.
    pub kv_len: i32,
    /// Rows per chunk. The last chunk of each request may be shorter.
    pub chunk_len: i32,
    /// Chunks per request, `ceil(kv_len / chunk_len)`.
    pub num_chunks: usize,
    /// Requests in the batch.
    pub batch: usize,
}

impl MlaSplitPlan {
    /// Cut `kv_len` so the batch produces at least `target_ctas` independent
    /// units, without going below [`MIN_CHUNK_LEN`].
    ///
    /// `batch * num_heads` units already exist before any split, so the split
    /// only needs to make up the shortfall; a batch that already saturates the
    /// device gets one chunk and the merge is skipped entirely.
    #[must_use]
    pub fn heuristic(batch: usize, num_heads: usize, kv_len: i32, target_ctas: usize) -> Self {
        let batch = batch.max(1);
        let kv_len = kv_len.max(0);
        let existing = batch.saturating_mul(num_heads.max(1));
        let wanted_chunks = target_ctas.div_ceil(existing.max(1)).max(1);
        let chunk_len = if kv_len <= 0 {
            MIN_CHUNK_LEN
        } else {
            let by_target = (kv_len as usize).div_ceil(wanted_chunks) as i32;
            by_target.max(MIN_CHUNK_LEN)
        };
        let num_chunks = if kv_len <= 0 {
            0
        } else {
            (kv_len as usize).div_ceil(chunk_len as usize)
        };
        Self {
            kv_len,
            chunk_len,
            num_chunks,
            batch,
        }
    }

    /// A plan with an explicit chunk length, for tests and for an autotuner.
    #[must_use]
    pub fn with_chunk_len(batch: usize, kv_len: i32, chunk_len: i32) -> Self {
        let batch = batch.max(1);
        let kv_len = kv_len.max(0);
        let chunk_len = chunk_len.max(1);
        Self {
            kv_len,
            chunk_len,
            num_chunks: if kv_len <= 0 {
                0
            } else {
                (kv_len as usize).div_ceil(chunk_len as usize)
            },
            batch,
        }
    }

    /// Whether the merge launch is needed at all.
    ///
    /// One chunk per request means the partial already is the answer, the same
    /// "write O directly" case `paged_v2::launch` decides on the host.
    #[must_use]
    pub const fn needs_merge(&self) -> bool {
        self.num_chunks > 1
    }

    /// Total partial rows, `batch * num_chunks`.
    #[must_use]
    pub const fn num_partials(&self) -> usize {
        self.batch * self.num_chunks
    }

    /// The merge kernel's `o_indptr`: `[0, C, 2C, ..., B*C]`.
    ///
    /// Request-major, so partial rows for one request are contiguous, which is
    /// what the grouping means.
    #[must_use]
    pub fn o_indptr(&self) -> Vec<i32> {
        (0..=self.batch)
            .map(|b| (b * self.num_chunks) as i32)
            .collect()
    }

    /// Half-open row range of chunk `c`, clamped to `kv_len`.
    #[must_use]
    pub fn chunk_range(&self, c: usize) -> (i32, i32) {
        let start = (c as i32).saturating_mul(self.chunk_len).min(self.kv_len);
        let stop = (start.saturating_add(self.chunk_len)).min(self.kv_len);
        (start, stop)
    }

    /// Reject a plan the split path cannot serve.
    pub fn validate(&self) -> Result<(), String> {
        if self.batch == 0 {
            return Err("mla split: batch must be non-zero".to_string());
        }
        if self.kv_len <= 0 || self.num_chunks == 0 {
            return Err("mla split: no visible latent rows to split".to_string());
        }
        if self.chunk_len <= 0 {
            return Err(format!(
                "mla split: chunk_len {} must be positive",
                self.chunk_len
            ));
        }
        let covered = (self.num_chunks as i64) * (self.chunk_len as i64);
        if covered < self.kv_len as i64 {
            return Err(format!(
                "mla split: {} chunks of {} cover {covered} rows, short of kv_len {}",
                self.num_chunks, self.chunk_len, self.kv_len
            ));
        }
        Ok(())
    }
}

/// Absorbed single-token decode, split across latent chunks and merged with
/// issue #898's `paged_attention_merge_states`.
///
/// * `q_nope` `[B, H, 1, qk_nope_head_dim]`, `q_pe` `[B, H, 1, qk_rope_head_dim]`
/// * `ckv` `[B, 1, S, kv_lora_rank]`, `kpe` `[B, 1, S, qk_rope_head_dim]`
///
/// Returns `[B, H, 1, v_head_dim]`. Decode only: a multi-token step needs a
/// causal mask per chunk, which the partial format has no place for.
pub fn absorbed_decode_split_kv(
    q_nope: &MlxArray,
    q_pe: &MlxArray,
    ckv: &MlxArray,
    kpe: &MlxArray,
    proj: &MlaAbsorbedProjections,
    scale: f32,
    plan: &MlaSplitPlan,
) -> Result<UniquePtr<MlxArray>, String> {
    plan.validate()?;
    let q_shape = ffi::array_shape(q_nope);
    if q_shape.len() != 4 || q_shape[2] != 1 {
        return Err(format!(
            "mla split: expected a single-token query [B, H, 1, D], got {q_shape:?}"
        ));
    }
    let batch = q_shape[0];
    let heads = q_shape[1];
    if batch as usize != plan.batch {
        return Err(format!(
            "mla split: query batch {batch} disagrees with the plan's {}",
            plan.batch
        ));
    }
    let rank = proj.geometry().kv_lora_rank as i32;

    stats::record(MlaDecodePath::AbsorbedSplitKv);

    let q_absorbed = absorb_queries(q_nope, proj);
    let scale_scalar = ffi::full_f32(&[1], scale, ffi::array_dtype(&q_absorbed));

    let mut partial_v = Vec::with_capacity(plan.num_chunks);
    let mut partial_lse = Vec::with_capacity(plan.num_chunks);
    let ckv_shape = ffi::array_shape(ckv);
    let kpe_shape = ffi::array_shape(kpe);

    for c in 0..plan.num_chunks {
        let (start, stop) = plan.chunk_range(c);
        let ckv_c = ffi::slice(
            ckv,
            &[0, 0, start, 0],
            &[ckv_shape[0], ckv_shape[1], stop, ckv_shape[3]],
        );
        let kpe_c = ffi::slice(
            kpe,
            &[0, 0, start, 0],
            &[kpe_shape[0], kpe_shape[1], stop, kpe_shape[3]],
        );

        // scores = scale * (q_absorbed . ckv + q_pe . kpe), [B, H, 1, chunk]
        let latent = ffi::matmul(&q_absorbed, &ffi::transpose_axes(&ckv_c, &[0, 1, 3, 2]));
        let rope = ffi::matmul(q_pe, &ffi::transpose_axes(&kpe_c, &[0, 1, 3, 2]));
        let scores = ffi::multiply(&ffi::add(&latent, &rope), &scale_scalar);
        let scores = ffi::astype(&scores, crate::dtype::FLOAT32);

        // The merge kernel wants each partial already divided by its own
        // denominator, and the matching LSE in log2 units.
        let probs = ffi::softmax_precise(&scores, -1);
        let lse_ln = ffi::logsumexp_axis(&scores, -1, false);
        let lse_log2 = ffi::multiply(&lse_ln, &ffi::full_f32(&[1], LOG2_E, crate::dtype::FLOAT32));

        let probs_k = ffi::astype(&probs, ffi::array_dtype(&ckv_c));
        let v_c = ffi::matmul(&probs_k, &ckv_c);
        let v_c = ffi::astype(&v_c, crate::dtype::FLOAT32);

        // [B, H, 1, D] -> [B, H, D] and [B, H, 1] -> [B, H]: the merge kernel's
        // partial layout has no query axis.
        partial_v.push(ffi::reshape(&v_c, &[batch, heads, rank]));
        partial_lse.push(ffi::reshape(&lse_log2, &[batch, heads]));
    }

    let merged = if plan.needs_merge() {
        // Stack to [C, B, H, D], move the request axis outermost, then flatten
        // to [B*C, H, D] so each request's chunks are contiguous, which is what
        // `o_indptr` groups over.
        let v_stacked = crate::ops::stack_owned(&partial_v, 0);
        let v_in = ffi::reshape(
            &ffi::contiguous(&ffi::transpose_axes(&v_stacked, &[1, 0, 2, 3]), false),
            &[batch * plan.num_chunks as i32, heads, rank],
        );
        let lse_stacked = crate::ops::stack_owned(&partial_lse, 0);
        let lse_in = ffi::reshape(
            &ffi::contiguous(&ffi::transpose_axes(&lse_stacked, &[1, 0, 2]), false),
            &[batch * plan.num_chunks as i32, heads],
        );
        let o_indptr_host = plan.o_indptr();
        let o_indptr = ffi::from_slice_i32(&o_indptr_host, &[o_indptr_host.len() as i32]);

        let mut v_out = UniquePtr::null();
        let mut lse_out = UniquePtr::null();
        ffi::paged_attention_merge_states(&v_in, &lse_in, &o_indptr, &mut v_out, &mut lse_out);
        v_out
    } else {
        // One chunk per request: the partial is the answer, no merge launch.
        partial_v.pop().ok_or_else(|| {
            "mla split: validate() accepted a plan that produced no partials".to_string()
        })?
    };

    let o_latent = ffi::reshape(&merged, &[batch, heads, 1, rank]);
    let o_latent = ffi::astype(&o_latent, ffi::array_dtype(ckv));
    Ok(unabsorb_output(&o_latent, proj))
}

#[cfg(test)]
#[path = "split_kv_tests.rs"]
mod split_kv_tests;
