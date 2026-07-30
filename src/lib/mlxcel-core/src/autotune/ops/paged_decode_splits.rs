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

//! Autotuned `NumSplits` for the v1 paged-attention decode kernel (issue #906).
//!
//! ## Why this knob
//!
//! `src/lib/mlx-cpp/turbo/paged_attention.cpp` launches the fused decode kernel
//! over `(32, NumSplits, B*Hq)`: 32 lanes partition the head dimension and
//! `NumSplits` SIMD groups sweep strided token stripes, then SIMD group 0
//! merges the partials with a flash rescale. Before #906 `NumSplits` came from
//! a single shape-blind expression: the largest value the `tg_acc[NumSplits *
//! Dim]` threadgroup-memory budget and the 1024-thread cap allow.
//!
//! That ceiling is a *feasibility* bound. It says nothing about which value is
//! fastest, and the two costs it ignores pull in opposite directions:
//!
//! - Wide splits shorten the per-group token sweep (good at long context).
//! - Wide splits also cost a wider threadgroup-memory reduction and more
//!   occupancy pressure per `(batch, head)` slot, and at short context most
//!   groups get zero or one token to sweep and contribute nothing but a
//!   reduction slot.
//!
//! So the optimum is a function of visible context length and of how many
//! `(batch, head)` slots are competing for the GPU, which is exactly a
//! shape-bucketed decision.
//!
//! ## Tactic space
//!
//! `NumSplits` is a template argument, so every candidate is a distinct JIT
//! specialization and therefore a genuine tactic rather than a runtime branch.
//! The candidate set is the powers of two up to the budget ceiling, plus the
//! ceiling itself when it is not a power of two (head dims like 256 give a
//! ceiling of 28). Including the ceiling matters: it is the pre-#906 default,
//! and the profiler only switches away from the default when the win clears
//! the noise margin, which requires the default to be measured.
//!
//! ## Cost when the autotuner is off
//!
//! [`resolve_num_splits`] returns `0` (meaning "use the ceiling") without
//! touching the cache, the memo lock, or the filesystem when `MLXCEL_AUTOTUNE`
//! is unset and `MLXCEL_PAGED_DECODE_SPLITS` is not set. The C++ launcher then
//! takes exactly the pre-#906 path.

use std::sync::OnceLock;

use crate::MlxArray;
use crate::autotune::bucket::{ShapeBucket, powers_of_two_up_to, round_up_pow2};
use crate::autotune::tactic::{Tactic, TunableOp, TuneError};
use crate::autotune::{Source, mode};
use crate::ffi;

/// Logical op name; part of the cache key.
pub const OP_NAME: &str = "paged_attention_decode_num_splits";

/// Explicit operator override. Wins over any tuned or cached value, matching
/// the `MLXCEL_QMM_TILE_*` / `MLXCEL_QMV_MULTIROW` escape-hatch convention.
pub const SPLITS_ENV: &str = "MLXCEL_PAGED_DECODE_SPLITS";

/// Tactic parameter index for the split count.
const PARAM_NUM_SPLITS: usize = 0;

/// The explicit `MLXCEL_PAGED_DECODE_SPLITS` value, read once.
///
/// A non-positive or unparseable value is ignored (with a warning) rather than
/// clamped, so a typo degrades to the default instead of silently pinning an
/// arbitrary split count. The C++ launcher clamps whatever survives into
/// `[1, cap]`, so an over-large value is safe.
fn env_num_splits() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var(SPLITS_ENV).ok()?;
        match raw.trim().parse::<i32>() {
            Ok(v) if v >= 1 => Some(v),
            _ => {
                tracing::warn!(
                    "{SPLITS_ENV}={raw:?} is not a positive integer; ignoring it and using the default split count"
                );
                None
            }
        }
    })
}

/// Feasible split ceiling for a head dimension, from the C++ launcher.
#[must_use]
pub fn num_splits_cap(dim: i32) -> i32 {
    ffi::paged_attention_num_splits_cap(dim)
}

/// Candidate split counts for a ceiling: powers of two up to it, plus the
/// ceiling itself when it is not already a power of two.
#[must_use]
pub fn split_candidates(cap: i32) -> Vec<i64> {
    if cap < 1 {
        return vec![1];
    }
    let cap = cap as u32;
    let mut out: Vec<i64> = powers_of_two_up_to(cap)
        .into_iter()
        .map(i64::from)
        .collect();
    if out.last().copied() != Some(i64::from(cap)) {
        out.push(i64::from(cap));
    }
    out
}

/// Launch shape of one fused paged-decode call, used to build the cache key.
///
/// `batch` and `context` are bucketed (they drift continuously during
/// serving); `q_heads`, `kv_heads`, and `head_dim` are kept exact because they
/// are model constants and each one changes the kernel specialization or the
/// feasible ceiling, so merging them across buckets would mix incomparable
/// regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeShape {
    pub batch: usize,
    pub q_heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    /// Largest visible context length across the sequences in this launch.
    pub context: usize,
}

impl DecodeShape {
    #[must_use]
    pub fn bucket(&self) -> ShapeBucket {
        ShapeBucket::from_exact(&[
            round_up_pow2(self.batch),
            self.q_heads.max(0) as u32,
            self.kv_heads.max(0) as u32,
            self.head_dim.max(0) as u32,
            round_up_pow2(self.context),
        ])
    }
}

/// [`TunableOp`] over the real decode call.
///
/// Profiling re-runs the *actual* launch with the actual pool and block-table
/// arrays rather than a synthetic stand-in, so the measurement includes the
/// real scatter pattern. That is what makes lazy first-use tuning meaningful:
/// the cost is a handful of extra decode-step launches, paid once per bucket.
pub struct PagedDecodeSplitsOp<'a> {
    q: &'a MlxArray,
    k_pool: &'a MlxArray,
    v_pool: &'a MlxArray,
    rows: &'a MlxArray,
    row_offsets: &'a MlxArray,
    logical_starts: &'a MlxArray,
    visible_lens: &'a MlxArray,
    scale: f32,
    shape: DecodeShape,
    cap: i32,
}

impl<'a> PagedDecodeSplitsOp<'a> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        q: &'a MlxArray,
        k_pool: &'a MlxArray,
        v_pool: &'a MlxArray,
        rows: &'a MlxArray,
        row_offsets: &'a MlxArray,
        logical_starts: &'a MlxArray,
        visible_lens: &'a MlxArray,
        scale: f32,
        shape: DecodeShape,
    ) -> Self {
        let cap = num_splits_cap(shape.head_dim);
        Self {
            q,
            k_pool,
            v_pool,
            rows,
            row_offsets,
            logical_starts,
            visible_lens,
            scale,
            shape,
            cap,
        }
    }
}

impl TunableOp for PagedDecodeSplitsOp<'_> {
    fn op_name(&self) -> &str {
        OP_NAME
    }

    fn runner_id(&self) -> String {
        // The launcher picks the Metal or CUDA JIT body from the live backend,
        // and the two are different kernels; keep their tactics apart.
        if crate::metal_is_available() {
            "metal".to_string()
        } else {
            "cuda".to_string()
        }
    }

    fn dtype_tag(&self) -> String {
        // The kernel always accumulates and emits f32 regardless of the pool
        // dtype (the caller casts around it), so the tactic is dtype-invariant
        // and the tag is fixed rather than absent.
        "f32".to_string()
    }

    fn bucket(&self) -> ShapeBucket {
        self.shape.bucket()
    }

    fn candidates(&self, _bucket: &ShapeBucket) -> Vec<Tactic> {
        split_candidates(self.cap)
            .into_iter()
            .map(|v| Tactic::scalar("num_splits", v))
            .collect()
    }

    fn default_tactic(&self, _bucket: &ShapeBucket) -> Tactic {
        Tactic::scalar("num_splits", i64::from(self.cap))
    }

    fn env_override(&self) -> Option<Tactic> {
        env_num_splits().map(|v| Tactic::scalar("num_splits", i64::from(v)))
    }

    fn run(&self, tactic: &Tactic) -> Result<(), TuneError> {
        let splits = tactic
            .param(PARAM_NUM_SPLITS)
            .ok_or_else(|| TuneError::infeasible(tactic, "tactic carries no split count"))?;
        let splits = i32::try_from(splits)
            .map_err(|_| TuneError::infeasible(tactic, "split count out of range"))?;
        if splits < 1 || splits > self.cap {
            return Err(TuneError::infeasible(
                tactic,
                format!("split count outside [1, {}]", self.cap),
            ));
        }
        let out = ffi::paged_attention_decode(
            self.q,
            self.k_pool,
            self.v_pool,
            self.rows,
            self.row_offsets,
            self.logical_starts,
            self.visible_lens,
            self.scale,
            splits,
        );
        crate::eval(&out);
        Ok(())
    }
}

/// Resolve the `NumSplits` to launch this decode call with.
///
/// Returns the value to pass as `num_splits_override`: `0` means "let the C++
/// launcher use its budget ceiling", which is the pre-#906 behavior.
///
/// Fast path: with `MLXCEL_AUTOTUNE` unset and no explicit
/// `MLXCEL_PAGED_DECODE_SPLITS`, this is one `OnceLock` read and one atomic
/// mode read, then `0`. No cache read, no lock, no filesystem access.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn resolve_num_splits(
    q: &MlxArray,
    k_pool: &MlxArray,
    v_pool: &MlxArray,
    rows: &MlxArray,
    row_offsets: &MlxArray,
    logical_starts: &MlxArray,
    visible_lens: &MlxArray,
    scale: f32,
    shape: DecodeShape,
) -> i32 {
    if let Some(v) = env_num_splits() {
        return v;
    }
    if !mode().reads_cache() {
        return 0;
    }
    let op = PagedDecodeSplitsOp::new(
        q,
        k_pool,
        v_pool,
        rows,
        row_offsets,
        logical_starts,
        visible_lens,
        scale,
        shape,
    );
    let resolution = crate::autotune::resolve(&op);
    if resolution.source.is_default() {
        // Preserve the exact pre-#906 launch: `0` re-derives the ceiling in C++
        // rather than round-tripping it through the tactic.
        return 0;
    }
    match resolution.param(PARAM_NUM_SPLITS) {
        Some(v) if v >= 1 => i32::try_from(v).unwrap_or(0),
        _ => 0,
    }
}

/// Whether a resolution came from a tuned or cached tactic (for logs/tests).
#[must_use]
pub fn is_tuned(source: Source) -> bool {
    matches!(source, Source::Cache | Source::Tuned)
}
