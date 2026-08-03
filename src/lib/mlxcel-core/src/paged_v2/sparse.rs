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

//! Fused sparse decode over a contiguous KV cache (issue #904).
//!
//! Takes a [`SparseSelection`] and runs the #898 v2 decode kernels over exactly
//! the selected rows. No kernel change: the selection becomes a `page_size = 1`
//! page table and the cache allocation becomes the pool (see
//! [`crate::cache::sparse_csr`] for the addressing argument).
//!
//! ## What this replaces
//!
//! The gather path materializes the selected KV into a fresh contiguous buffer
//! and then runs dense SDPA over it: `S * D` elements are read scattered,
//! written once, and read twice more (K and V). The mask path is worse still,
//! computing every dropped position before adding `-inf` to it. Here the
//! scattered read happens inside the attention loop and nothing is written, so
//! the transient allocation the gather needed disappears entirely.
//!
//! ## Where the dispatch floor sits for a sparse launch
//!
//! #899 measured the fused kernel against gather and found exactly one losing
//! shape, batch 1 at 1024 visible tokens, which produced the two-regime floor in
//! [`crate::paged_v2::dispatch`]: 4096 visible tokens for a single request, 512
//! per request when batched. Two things move that floor here, in opposite
//! directions, and both are stated rather than assumed:
//!
//! - **The token count that must clear the floor is the selected count, not the
//!   context.** The launch reads `S` rows per request, so a 32K context with a
//!   4K selection is a 4K-token launch. Applying the floor to `kv_len` would
//!   dispatch launches whose real work is eight times smaller than the number
//!   that justified the dispatch.
//! - **The request count is multiplied by the KV heads.** Folding the head axis
//!   into the request axis turns a single sequence into `Hkv` CSR requests, so a
//!   batch-1 sparse decode with `Hkv > 1` faces the *batched* floor, not the
//!   single-request one. That is not a loophole: the losing measurement was
//!   caused by a plan that degenerates to a couple of pages per chunk and pays a
//!   merge for nothing, and `Hkv` requests of `S` pages each is the same batched
//!   shape that measured 1.41x and 1.47x, not the batch-1 shape that measured
//!   0.91x.
//!
//! The net effect is that a single-sequence sparse decode needs `Hkv * 512`
//! selected rows rather than 4096. For MiniMax-M3's shipped configuration
//! (`Hkv` of 4, a ~2048-row selection) that is cleared with margin; for a small
//! selection it is not, and the launch declines with
//! [`SparseDecodeOutcome::BelowFloor`] carrying both numbers.
//!
//! **The `Hkv == 1` case does not get that relief and is worth stating
//! explicitly**, because it is exactly MLA. An absorbed-MLA decode has one KV
//! head, so head folding yields one CSR request per sequence and a batch-1
//! launch faces the 4096-row single-request floor. DeepSeek-V3.2's default
//! `index_topk` is 2048, so such a launch would be declined here even once the
//! kernel can express MLA's positional term (see `docs/sparse-paged-decode.md`
//! for why it cannot yet). The floor has to be re-derived for `Hkv == 1` sparse
//! launches before MLA can be expected to dispatch at all.
//!
//! **That part of the reasoning is derived, not measured.** #899's table was
//! taken on dense page lists. The floor is the same code path and the same
//! environment overrides, so a re-measurement moves both together.
//!
//! ## Why a token floor alone is not enough, and the sparsity gate
//!
//! A token floor asks "is this launch big enough". It does not ask the question
//! that decides a *sparse* launch, which is "is skipping worth the kernel you
//! have to skip with". `mlx::fast::scaled_dot_product_attention` is a heavily
//! tuned dense kernel; the v2 partial kernel is a scalar per-lane sweep. Reading
//! half the data through a kernel with twice the constant factor is a loss, and
//! `examples/sparse_paged_decode_bench` measures exactly that (MiniMax-M3
//! geometry, one repetition of 40 steps on an idle M-series host, so indicative
//! rather than a recorded result):
//!
//! | context | sparsity | fused vs mask |
//! |---|---|---|
//! | 4096 | 2.0x | **0.67x** |
//! | 8192 | 4.0x | **0.77x** |
//! | 16384 | 8.0x | 1.22x |
//! | 32768 | 16.0x | 1.17x |
//! | 65536 | 32.0x | 2.06x |
//!
//! So the launch must also clear [`MIN_SPARSITY_RATIO`]: the live window has to
//! be at least that many times the selected count. The default sits **on top of
//! the measured win** rather than interpolated into the unmeasured 4x-to-8x
//! band. #899 argued the opposite way for its token floor, and deliberately sat
//! below its weakest measured point, because what lay under that point was a
//! missed opportunity. Here what lies under the point is a *measured
//! regression*, and shipping a regression in a narrow band is worse than
//! declining a modest win in one.
//!
//! Two caveats on that table, both pointing the same way. The harness builds the
//! mask on the host outside the timed region, while the real mask path rebuilds
//! it on device every step through several `O(kv_len)` passes; and the harness
//! hands the sparse arm a prebuilt block list, while the real path expands the
//! selection in one `O(selected)` pass. Both flatter the mask arm, so the real
//! crossover should sit at a lower sparsity than 8x, which makes this default
//! conservative in the direction that cannot regress anything.
//!
//! For MiniMax-M3's shipped configuration (`topk_blocks` 16, `block_size` 128,
//! so a ~2048-row selection) the gate opens at a 16K context, which is exactly
//! where issue #904 requires decode to improve.
//!
//! ## Proving which path ran
//!
//! Every call returns a [`SparseDecodeOutcome`] alongside its result, and
//! [`report_sparse_outcome_once`] announces the first occurrence of each
//! *kind* on stderr as well as through `tracing`. The `mlxcel` CLI installs no
//! tracing subscriber, so a `tracing::info!` on a CLI-only path prints nothing
//! at any `RUST_LOG`; a benchmark that cannot see which kernel it measured is
//! how #899 first compared the fallback against itself. One flag per kind, not
//! one global flag, because a single one-shot reports only the warmup request.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cxx::UniquePtr;

use crate::cache::sparse_csr::{
    ContiguousCacheLayout, SparseCsrStructure, SparseIndices, SparseSelection, shared_row_mapping,
};
use crate::ffi;
use crate::ffi::MlxArray;
use crate::paged_v2::dispatch::{
    PagedV2Dispatch, active_required_visible_tokens, select_paged_v2_dispatch,
};
use crate::paged_v2::launch::V2Context;
use crate::paged_v2::plan::PagedDecodeGeometry;
use crate::paged_v2::resolve_plan;

/// Kill switch. `MLXCEL_SPARSE_PAGED_ATTENTION=0` pins every caller back on its
/// pre-#904 gather or mask path.
pub const SPARSE_PAGED_ENV: &str = "MLXCEL_SPARSE_PAGED_ATTENTION";

/// Opt-in dump of the selected rows. Synchronizes, so it is for debugging a
/// selection only, never for a timed run.
pub const SPARSE_PAGED_DUMP_ENV: &str = "MLXCEL_SPARSE_PAGED_DUMP";

/// Environment override for [`MIN_SPARSITY_RATIO`].
pub const MIN_SPARSITY_ENV: &str = "MLXCEL_SPARSE_PAGED_MIN_SPARSITY";

/// How many times larger the live window must be than the selection.
///
/// See the module docs for the measurements: the fused path loses at 2x and 4x
/// and wins from 8x, so this sits on the measured win rather than inside the
/// unmeasured band below it. `0` disables the gate.
pub const MIN_SPARSITY_RATIO: usize = 8;

/// The active sparsity gate, read once per process.
#[must_use]
pub fn min_sparsity_ratio() -> usize {
    static RATIO: OnceLock<usize> = OnceLock::new();
    *RATIO.get_or_init(|| match std::env::var(MIN_SPARSITY_ENV) {
        Ok(raw) => raw.trim().parse::<usize>().unwrap_or(MIN_SPARSITY_RATIO),
        Err(_) => MIN_SPARSITY_RATIO,
    })
}

/// Whether skipping is worth the kernel it has to be skipped with.
///
/// Pure, so the numbers that decided a dispatch can be logged next to the ratio
/// they were measured against. A non-positive selection never clears the gate;
/// a zero ratio always does.
#[must_use]
pub fn clears_sparsity_gate(live_len: usize, selected_per_request: usize, ratio: usize) -> bool {
    if ratio == 0 {
        return true;
    }
    if selected_per_request == 0 {
        return false;
    }
    // `checked_mul`, not `saturating_mul`: saturating the requirement to
    // `usize::MAX` makes an absurd selection *pass* the gate, because the
    // comparison then reads `usize::MAX >= usize::MAX`. A requirement that
    // overflows `usize` is one no window can meet.
    match selected_per_request.checked_mul(ratio) {
        Some(required) => live_len >= required,
        None => false,
    }
}

/// Whether the fused sparse path is enabled, read once per process.
///
/// Enabled by default. `0`, `false`, `off` and `no` disable it; anything else
/// enables it, so a typo leaves the measured path in place rather than
/// silently reverting to the one being replaced.
#[must_use]
pub fn sparse_paged_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(SPARSE_PAGED_ENV) {
        Ok(raw) => !matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    })
}

/// Whether to dump selections, read once per process.
#[must_use]
pub fn sparse_paged_dump_enabled() -> bool {
    static DUMP: OnceLock<bool> = OnceLock::new();
    *DUMP.get_or_init(|| match std::env::var(SPARSE_PAGED_DUMP_ENV) {
        Ok(raw) => matches!(raw.trim(), "1" | "true" | "on" | "yes"),
        Err(_) => false,
    })
}

/// What one sparse decode call did, and why.
///
/// A value rather than a log line, for the reason [`crate::paged_v2::outcome`]
/// spells out: a sparse path that silently falls back to dense is
/// indistinguishable from "sparsity does not help".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseDecodeOutcome {
    /// The fused kernel ran over the selection.
    Fused {
        /// CSR requests, i.e. sequences times KV heads.
        requests: usize,
        /// Rows each request selected.
        selected_per_request: usize,
        /// Chunks the plan emitted.
        chunks: usize,
        /// Whether the merge pass ran.
        merged: bool,
    },
    /// `MLXCEL_SPARSE_PAGED_ATTENTION` pinned the caller's own path.
    KillSwitch,
    /// The launch selects fewer rows than the dispatch floor.
    BelowFloor {
        requests: usize,
        selected: usize,
        floor: usize,
    },
    /// The selection is too large a fraction of the live window for skipping to
    /// pay for the kernel it has to be skipped with.
    BelowSparsity {
        live_len: usize,
        selected_per_request: usize,
        required_ratio: usize,
    },
    /// The kernel cannot serve this head geometry.
    UnservableGeometry(String),
    /// The K and V allocations do not share a row mapping, or a shape is not
    /// what the pool view needs.
    UnservableLayout(String),
    /// The selection itself is malformed.
    SelectionRejected(String),
    /// The chunk plan failed its structural check.
    PlanRejected(String),
    /// The caller declined before building a selection: a cache mode whose
    /// rows are not raw K/V, a multi-token step, or a shape it does not handle.
    NotServable(&'static str),
}

/// Number of distinct outcome kinds, for the one-shot report table.
pub const SPARSE_DECODE_OUTCOME_KINDS: usize = 9;

impl SparseDecodeOutcome {
    /// Whether the fused sparse kernel actually ran.
    #[must_use]
    pub fn is_fused(&self) -> bool {
        matches!(self, Self::Fused { .. })
    }

    /// Stable index for this kind.
    #[must_use]
    pub fn kind_index(&self) -> usize {
        match self {
            Self::Fused { .. } => 0,
            Self::KillSwitch => 1,
            Self::BelowFloor { .. } => 2,
            Self::UnservableGeometry(_) => 3,
            Self::UnservableLayout(_) => 4,
            Self::SelectionRejected(_) => 5,
            Self::PlanRejected(_) => 6,
            Self::NotServable(_) => 7,
            Self::BelowSparsity { .. } => 8,
        }
    }

    /// One-line summary carrying the numbers that produced the decision.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Fused {
                requests,
                selected_per_request,
                chunks,
                merged,
            } => format!(
                "fused sparse launch ({requests} request(s) x {selected_per_request} selected \
                 rows, {chunks} chunks, merge {})",
                if *merged { "on" } else { "skipped" }
            ),
            Self::KillSwitch => {
                format!("fallback: pinned by {SPARSE_PAGED_ENV}")
            }
            Self::BelowFloor {
                requests,
                selected,
                floor,
            } => format!(
                "fallback: {selected} selected rows across {requests} request(s) is below the \
                 {floor}-row dispatch floor"
            ),
            Self::BelowSparsity {
                live_len,
                selected_per_request,
                required_ratio,
            } => format!(
                "fallback: a {selected_per_request}-row selection out of a {live_len}-token \
                 window is {:.1}x sparsity, below the {required_ratio}x the fused kernel needs \
                 to beat dense SDPA",
                *live_len as f64 / (*selected_per_request).max(1) as f64
            ),
            Self::UnservableGeometry(reason) => {
                format!("fallback: the kernel cannot serve this geometry ({reason})")
            }
            Self::UnservableLayout(reason) => {
                format!("fallback: the cache cannot be addressed as a page pool ({reason})")
            }
            Self::SelectionRejected(reason) => {
                format!("fallback: the selection was rejected ({reason})")
            }
            Self::PlanRejected(reason) => {
                format!("fallback: the chunk plan was rejected ({reason})")
            }
            Self::NotServable(reason) => format!("fallback: not servable ({reason})"),
        }
    }
}

/// Fused-launch and fallback counters, for a harness that wants a total rather
/// than a first-occurrence line.
#[derive(Debug, Default)]
pub struct SparseDecodeCounters {
    pub fused: AtomicU64,
    pub fallbacks: AtomicU64,
}

static COUNTERS: SparseDecodeCounters = SparseDecodeCounters {
    fused: AtomicU64::new(0),
    fallbacks: AtomicU64::new(0),
};

/// Snapshot of the counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseDecodeStats {
    pub fused: u64,
    pub fallbacks: u64,
}

/// Read the sparse decode counters.
#[must_use]
pub fn sparse_decode_stats() -> SparseDecodeStats {
    SparseDecodeStats {
        fused: COUNTERS.fused.load(Ordering::Relaxed),
        fallbacks: COUNTERS.fallbacks.load(Ordering::Relaxed),
    }
}

static REPORTED: [AtomicBool; SPARSE_DECODE_OUTCOME_KINDS] =
    [const { AtomicBool::new(false) }; SPARSE_DECODE_OUTCOME_KINDS];

/// Announce an outcome the first time its kind occurs, and count it.
///
/// Emits on **stderr** as well as through `tracing`. There are at most
/// [`SPARSE_DECODE_OUTCOME_KINDS`] such lines in a process lifetime, so always
/// emitting them costs nothing and makes a benchmark arm self-identifying even
/// under a binary with no tracing subscriber installed. stderr rather than
/// stdout so a piped `mlxcel generate` still produces clean token output.
pub fn report_sparse_outcome_once(outcome: &SparseDecodeOutcome) {
    if outcome.is_fused() {
        COUNTERS.fused.fetch_add(1, Ordering::Relaxed);
    } else {
        COUNTERS.fallbacks.fetch_add(1, Ordering::Relaxed);
    }
    let slot = &REPORTED[outcome.kind_index()];
    if slot
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        let line = outcome.describe();
        tracing::info!("sparse paged decode: {line}");
        eprintln!("sparse paged decode: {line}");
    }
}

/// Forget which outcomes have been announced. Test-only.
#[cfg(test)]
pub(crate) fn reset_sparse_reported() {
    for slot in &REPORTED {
        slot.store(false, Ordering::Relaxed);
    }
}

/// Everything the sparse launch needs to know about the caller's cache.
///
/// The two allocations are the **raw reserved buffers**, not the fetched
/// windows: `update_and_fetch` returns a token-axis slice of a step-padded
/// buffer, and reshaping that slice into a pool would copy the whole cache.
pub struct SparseDecodeInputs<'a> {
    /// `[B, Hq, 1, D]` decode query, any float dtype.
    pub q: &'a MlxArray,
    /// `[B, H_k, Cap, D]` raw K allocation.
    pub k_alloc: &'a MlxArray,
    /// `[B, H_v, Cap, D]` raw V allocation.
    pub v_alloc: &'a MlxArray,
    /// KV heads attention uses. May be fewer than the allocation's heads.
    pub kv_heads: i32,
    /// Live tokens in the window the selection was drawn from. This is the
    /// denominator of the sparsity gate: what matters is not how many rows the
    /// launch reads but how many it *skips*.
    pub live_len: i32,
    /// Softmax scale.
    pub scale: f32,
}

/// Run a fused sparse decode, or say why it did not run.
///
/// On success the output is `[B, Hq, 1, D]` f32, which the caller casts back to
/// its working dtype. `None` means the caller must run its own gather or mask
/// path; the paired outcome says which decline it was.
///
/// The caller is responsible for reporting the outcome (see
/// [`report_sparse_outcome_once`]); returning it rather than logging it here
/// keeps the decision testable.
pub fn run_sparse_decode(
    inputs: &SparseDecodeInputs<'_>,
    selection: &SparseSelection,
) -> (Option<UniquePtr<MlxArray>>, SparseDecodeOutcome) {
    if !sparse_paged_enabled() {
        return (None, SparseDecodeOutcome::KillSwitch);
    }
    match prepare(inputs, selection) {
        Err(outcome) => (None, outcome),
        Ok(prepared) => launch(inputs, selection, &prepared),
    }
}

/// The shape facts a launch needs, all derived before any MLX work happens.
struct Prepared {
    /// CSR requests: `B * kv_heads`.
    requests: i32,
    /// `Hq / kv_heads`.
    n_rep: i32,
    head_dim: i32,
    k_layout: ContiguousCacheLayout,
    v_layout: ContiguousCacheLayout,
    geometry: PagedDecodeGeometry,
    structure: SparseCsrStructure,
    batch: i32,
    q_heads: i32,
}

fn prepare(
    inputs: &SparseDecodeInputs<'_>,
    selection: &SparseSelection,
) -> Result<Prepared, SparseDecodeOutcome> {
    selection
        .validate()
        .map_err(SparseDecodeOutcome::SelectionRejected)?;

    let q_shape = ffi::array_shape(inputs.q);
    if q_shape.len() != 4 || q_shape[2] != 1 {
        return Err(SparseDecodeOutcome::UnservableLayout(format!(
            "expected a [B, Hq, 1, D] decode query, got {q_shape:?}"
        )));
    }
    let (batch, q_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[3]);

    let k_layout = ContiguousCacheLayout::from_shape(&ffi::array_shape(inputs.k_alloc))
        .map_err(SparseDecodeOutcome::UnservableLayout)?;
    let v_layout = ContiguousCacheLayout::from_shape(&ffi::array_shape(inputs.v_alloc))
        .map_err(SparseDecodeOutcome::UnservableLayout)?;
    let k_dim = ffi::array_shape(inputs.k_alloc)[3];
    let v_dim = ffi::array_shape(inputs.v_alloc)[3];
    if k_dim != head_dim || v_dim != head_dim {
        return Err(SparseDecodeOutcome::UnservableLayout(format!(
            "query head_dim {head_dim} disagrees with the cache ({k_dim} K, {v_dim} V)"
        )));
    }
    if k_layout.batch != batch {
        return Err(SparseDecodeOutcome::UnservableLayout(format!(
            "query batch {batch} disagrees with the cache batch {}",
            k_layout.batch
        )));
    }
    shared_row_mapping(&k_layout, &v_layout, inputs.kv_heads)
        .map_err(SparseDecodeOutcome::UnservableLayout)?;

    if inputs.kv_heads <= 0 || q_heads % inputs.kv_heads != 0 {
        return Err(SparseDecodeOutcome::UnservableGeometry(format!(
            "q_heads {q_heads} is not a multiple of kv_heads {}",
            inputs.kv_heads
        )));
    }
    let n_rep = q_heads / inputs.kv_heads;
    let requests = batch.saturating_mul(inputs.kv_heads);
    if selection.requests != requests as usize {
        return Err(SparseDecodeOutcome::SelectionRejected(format!(
            "the selection has {} requests but the launch has {requests} \
             (batch {batch} x kv_heads {})",
            selection.requests, inputs.kv_heads
        )));
    }

    // The head axis is folded into the request axis, so the pool has one KV
    // head and each CTA owns a group of the `n_rep` query heads that share it.
    let geometry = PagedDecodeGeometry {
        q_heads: n_rep,
        kv_heads: 1,
        head_dim,
        page_size: crate::cache::sparse_csr::TOKEN_EXACT_PAGE_SIZE,
    };
    geometry
        .check()
        .map_err(SparseDecodeOutcome::UnservableGeometry)?;

    let structure = selection.structure();
    let selected = structure.total_selected();
    let floor = active_required_visible_tokens(structure.requests());
    if select_paged_v2_dispatch(selected, floor) == PagedV2Dispatch::Gather {
        return Err(SparseDecodeOutcome::BelowFloor {
            requests: structure.requests(),
            selected,
            floor,
        });
    }
    let ratio = min_sparsity_ratio();
    let live_len = inputs.live_len.max(0) as usize;
    if !clears_sparsity_gate(live_len, selection.per_request, ratio) {
        return Err(SparseDecodeOutcome::BelowSparsity {
            live_len,
            selected_per_request: selection.per_request,
            required_ratio: ratio,
        });
    }

    Ok(Prepared {
        requests,
        n_rep,
        head_dim,
        k_layout,
        v_layout,
        geometry,
        structure,
        batch,
        q_heads,
    })
}

fn launch(
    inputs: &SparseDecodeInputs<'_>,
    selection: &SparseSelection,
    p: &Prepared,
) -> (Option<UniquePtr<MlxArray>>, SparseDecodeOutcome) {
    if sparse_paged_dump_enabled() {
        for (r, rows) in selection.materialize().iter().enumerate() {
            eprintln!("sparse paged decode: request {r} selects rows {rows:?}");
        }
    }

    // Pure reshapes of the allocations. `[B, H, Cap, D]` row-major is already
    // `[B * H * Cap, 1, 1, D]`, so no data moves and no copy is made.
    let k_pool = ffi::reshape(inputs.k_alloc, &[p.k_layout.pool_rows(), 1, 1, p.head_dim]);
    let v_pool = ffi::reshape(inputs.v_alloc, &[p.v_layout.pool_rows(), 1, 1, p.head_dim]);
    // `[B, Hq, 1, D]` to `[B * Hkv, n_rep, 1, D]`: MLX GQA numbers query head
    // `i` under KV head `i / n_rep`, so this reshape lands each query head in
    // the request that owns its KV head.
    let q = ffi::astype(inputs.q, crate::dtype::FLOAT32);
    let q = ffi::reshape(&q, &[p.requests, p.n_rep, 1, p.head_dim]);

    let indices = match &selection.indices {
        SparseIndices::Host(v) => ffi::from_slice_i32(v, &[v.len() as i32]),
        SparseIndices::Device(a) => {
            let flat = ffi::reshape(a, &[selection.total() as i32]);
            ffi::astype(&flat, crate::dtype::INT32)
        }
    };

    let ctx = V2Context {
        q: &q,
        k_pool: &k_pool,
        v_pool: &v_pool,
        indices,
        indptr: ffi::from_slice_i32(&p.structure.indptr, &[p.requests + 1]),
        last_page_len: ffi::from_slice_i32(&p.structure.last_page_len, &[p.requests]),
        first_page_offset: ffi::from_slice_i32(&p.structure.first_page_offset, &[p.requests]),
        scale: inputs.scale,
        geometry: p.geometry,
    };

    let page_counts = p.structure.page_counts();
    let plan = resolve_plan(&ctx, &page_counts);
    if let Err(reason) = plan.validate() {
        return (None, SparseDecodeOutcome::PlanRejected(reason));
    }
    let outcome = SparseDecodeOutcome::Fused {
        requests: p.structure.requests(),
        selected_per_request: selection.per_request,
        chunks: plan.num_chunks,
        merged: plan.needs_merge,
    };
    match ctx.launch(&plan) {
        Ok(out) => {
            // `[B * Hkv, n_rep, 1, D]` back to `[B, Hq, 1, D]`, the inverse of
            // the query reshape.
            let out = ffi::reshape(&out, &[p.batch, p.q_heads, 1, p.head_dim]);
            (Some(out), outcome)
        }
        Err(reason) => (None, SparseDecodeOutcome::PlanRejected(reason)),
    }
}

#[cfg(test)]
#[path = "sparse_tests.rs"]
mod sparse_tests;
