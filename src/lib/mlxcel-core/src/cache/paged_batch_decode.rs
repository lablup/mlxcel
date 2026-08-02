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

//! Whole-batch decode over pool-backed KV caches (issue #899).
//!
//! ## What this replaces
//!
//! Before #899, a model's batched decode handed pool-backed caches to the
//! per-sequence loop: for each sequence, `KVCache::update_and_fetch` appended
//! the new token to the pool and called `gather_visible` to materialize a dense
//! `[1, Hkv, visible_len, D]` copy, then the model ran an ordinary fused SDPA
//! against it. That is ADR 0001 strategy A, and the ADR measures its overhead
//! at 2x-3x of SDPA time past 4096 tokens.
//!
//! [`paged_batch_decode_attention`] takes the whole batch instead: it performs
//! the same pool appends, then issues **one** fused v2 launch across every
//! sequence and every KV head, with no gather copy anywhere.
//!
//! ## Contract
//!
//! The function either owns the step or has not touched anything:
//!
//! - `None` means it declined **before writing**, so the caller runs its
//!   unchanged per-sequence loop. Every decline check is evaluated first, for
//!   this reason.
//! - `Some(out)` means the pool appends happened here and `out` is
//!   `[B, Hq, 1, D]` attention output. If v2 was not the right choice for the
//!   shape, or declined it, the gather-then-SDPA fallback ran *inside* this
//!   function; the caller must not run its own loop, which would double-write
//!   the pool.
//!
//! ## Variant coverage
//!
//! | family shape | handling |
//! |---|---|
//! | full attention, f16 pool | v2 above the token floor, gather below |
//! | sliding window (`logical_start > 0`) | v2; the window is expressed as a CSR range, see below |
//! | logit softcap | declined, gather (`softcap != 0.0`) |
//! | explicit attention mask | declined (`mask` is not `None` at the call site) |
//! | speculative / MTP verify | never reached: those steps have `seq_len > 1` |
//! | Int8 / Turbo KV modes | declined: those sequences are never pool-backed |
//!
//! ### Sliding window
//!
//! v2 applies **no windowing mask of its own**. A trimmed window is expressed
//! entirely in the page table: [`build_paged_csr_view`] emits only the pages
//! from `logical_start / page_size` onward and sets `first_page_offset` to
//! `logical_start % page_size`, so the kernel's token `i` of request `r`
//! resolves to absolute position `logical_start + i` and the retired prefix is
//! never addressed. That is the same `[logical_start, len)` window
//! `gather_visible` slices, which is why the two paths agree.
//!
//! Two consequences worth stating explicitly, because both are silent if
//! violated:
//!
//! 1. **The window must be contiguous.** A future attention-sink retention on
//!    the paged path (`[0, keep) ++ [start, len)`, the shape
//!    `KVCache::trim_front_keep_sink` produces on dense caches) is *not*
//!    expressible in one CSR range and must not be routed here. Today it cannot
//!    be: `trim_front_keep_sink` returns `0` for pool-backed caches, so
//!    `logical_start` on a served sequence only ever moves through
//!    `PagedBlockPool::trim_tokens`, which keeps the window contiguous.
//! 2. **RoPE is applied upstream.** The model rotates Q and K from the
//!    scheduler's `BatchedAttentionMetadata::rope_offsets` before this function
//!    is called, so [`PagedCsrView::rope_offsets`] is not consumed here. It
//!    records the same quantity (the absolute next-token position, which is the
//!    written length and not the visible length) and exists for callers that
//!    have to derive it themselves.
//!
//! [`build_paged_csr_view`]: super::paged_csr::build_paged_csr_view
//! [`PagedCsrView::rope_offsets`]: super::paged_csr::PagedCsrView::rope_offsets

use std::cell::Ref;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cxx::UniquePtr;

use super::{KVCache, KVCacheMode, PagedSequenceState};
use crate::ffi;
use crate::ffi::MlxArray;
use crate::paged_v2::{PAGED_DECODE_OUTCOME_KINDS, PagedDecodeOutcome};

/// Counters for what the production paged decode actually did.
///
/// Process-wide and relaxed: they exist so tests can assert that v2 really took
/// the step (rather than silently falling back) and so an operator can see the
/// split without a profiler. They are never read on the hot path.
#[derive(Debug, Default)]
pub struct PagedBatchDecodeCounters {
    /// Layer steps served by a fused v2 launch, cascade included.
    pub v2_launches: AtomicU64,
    /// Layer steps served by the in-function gather-then-SDPA fallback.
    pub gather_fallbacks: AtomicU64,
    /// Calls that declined before writing, leaving the caller's own loop to run.
    pub declines: AtomicU64,
    /// Layer steps served by the two-level cascade decomposition (issue #903).
    /// A subset of [`Self::v2_launches`].
    pub cascade_launches: AtomicU64,
    /// Cumulative KV tokens hoisted into a shared-span launch. Divided by
    /// [`Self::cascade_launches`] this is the mean shared-span length; kept as a
    /// sum so the two numbers stay consistent under concurrent updates.
    pub cascade_shared_tokens: AtomicU64,
    /// Cumulative member count across cascade launches. Divided by
    /// [`Self::cascade_launches`] this is the mean subgroup size.
    pub cascade_member_seqs: AtomicU64,
    /// Cascade launches that failed and fell back to the flat v2 launch.
    /// Non-zero means a planned decomposition is not running; see the
    /// `CascadeFailed` line in the log for the reason.
    pub cascade_failures: AtomicU64,
}

static COUNTERS: PagedBatchDecodeCounters = PagedBatchDecodeCounters {
    v2_launches: AtomicU64::new(0),
    gather_fallbacks: AtomicU64::new(0),
    declines: AtomicU64::new(0),
    cascade_launches: AtomicU64::new(0),
    cascade_shared_tokens: AtomicU64::new(0),
    cascade_member_seqs: AtomicU64::new(0),
    cascade_failures: AtomicU64::new(0),
};

/// One flag per [`PagedDecodeOutcome`] kind, so the *first* launch of each
/// distinct outcome is announced and the rest are silent.
///
/// Per kind rather than one global flag on purpose. A single one-shot reports
/// only whatever happened first, which in a real server is a short warmup
/// request; a later, permanent decline for an entirely different reason then
/// never surfaces. That is exactly how the #899 production benchmark ran a full
/// sweep on the gather path without any line saying so.
static REPORTED: [AtomicBool; PAGED_DECODE_OUTCOME_KINDS] =
    [const { AtomicBool::new(false) }; PAGED_DECODE_OUTCOME_KINDS];

/// Announce an outcome the first time its kind occurs.
///
/// **Info, not debug.** A decode-path diagnostic that only an operator who
/// already suspects a problem can enable is worthless for the case it exists
/// for: knowing which kernel a benchmark actually measured. There are at most
/// [`PAGED_DECODE_OUTCOME_KINDS`] of these lines in a process lifetime, so the
/// cost of always emitting them is bounded and tiny.
fn report_once(outcome: &PagedDecodeOutcome) {
    let slot = &REPORTED[outcome.kind_index()];
    if slot
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        tracing::info!("paged decode v2: {}", outcome.describe());
    }
}

/// Forget which outcomes have been announced. Test-only: the flags are
/// process-wide, and a test that asserts on logging needs a clean slate.
#[cfg(test)]
fn reset_reported() {
    for slot in &REPORTED {
        slot.store(false, Ordering::Relaxed);
    }
}

/// Snapshot of [`COUNTERS`] as plain integers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedBatchDecodeStats {
    pub v2_launches: u64,
    pub gather_fallbacks: u64,
    pub declines: u64,
    /// Cascade launches (issue #903), a subset of `v2_launches`.
    pub cascade_launches: u64,
    /// Cumulative shared-span tokens across cascade launches.
    pub cascade_shared_tokens: u64,
    /// Cumulative member count across cascade launches.
    pub cascade_member_seqs: u64,
    /// Planned cascade launches that failed and fell back to flat v2.
    pub cascade_failures: u64,
}

impl PagedBatchDecodeStats {
    /// Mean shared-span length in KV tokens, or `0.0` before the first cascade
    /// launch.
    #[must_use]
    pub fn mean_shared_tokens(&self) -> f64 {
        if self.cascade_launches == 0 {
            0.0
        } else {
            self.cascade_shared_tokens as f64 / self.cascade_launches as f64
        }
    }

    /// Mean number of sequences sharing the span, or `0.0` before the first
    /// cascade launch.
    #[must_use]
    pub fn mean_cascade_members(&self) -> f64 {
        if self.cascade_launches == 0 {
            0.0
        } else {
            self.cascade_member_seqs as f64 / self.cascade_launches as f64
        }
    }
}

/// Read the production paged decode counters.
#[must_use]
pub fn paged_batch_decode_stats() -> PagedBatchDecodeStats {
    PagedBatchDecodeStats {
        v2_launches: COUNTERS.v2_launches.load(Ordering::Relaxed),
        gather_fallbacks: COUNTERS.gather_fallbacks.load(Ordering::Relaxed),
        declines: COUNTERS.declines.load(Ordering::Relaxed),
        cascade_launches: COUNTERS.cascade_launches.load(Ordering::Relaxed),
        cascade_shared_tokens: COUNTERS.cascade_shared_tokens.load(Ordering::Relaxed),
        cascade_member_seqs: COUNTERS.cascade_member_seqs.load(Ordering::Relaxed),
        cascade_failures: COUNTERS.cascade_failures.load(Ordering::Relaxed),
    }
}

/// Fold a cascade outcome into the process counters.
///
/// Separate from the `v2_launches` bump so the cascade numbers stay a strict
/// subset of it: a step that ran the cascade decomposition is still a fused v2
/// launch, and reporting it as anything else would make the two counters
/// disagree about what the fused path served.
fn record_cascade(outcome: &PagedDecodeOutcome) {
    match outcome {
        PagedDecodeOutcome::FusedCascade {
            members,
            shared_tokens,
            ..
        } => {
            COUNTERS.cascade_launches.fetch_add(1, Ordering::Relaxed);
            COUNTERS
                .cascade_shared_tokens
                .fetch_add(*shared_tokens as u64, Ordering::Relaxed);
            COUNTERS
                .cascade_member_seqs
                .fetch_add(*members as u64, Ordering::Relaxed);
        }
        PagedDecodeOutcome::CascadeFailed(_) => {
            COUNTERS.cascade_failures.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// Whether every cache in the batch can be served by the whole-batch path.
///
/// Pure and MLX-free apart from the shape reads, and evaluated in full before
/// any pool mutation: a `false` here means the caller's per-sequence loop runs
/// with the pool untouched.
fn batch_is_servable(caches: &[&mut KVCache], softcap: f32) -> Result<(), &'static str> {
    if caches.is_empty() {
        return Err("empty batch");
    }
    // A logit soft-cap is applied inside SDPA; the v2 kernel has no soft-cap
    // term and the gather fallback in this function does not thread one either,
    // so the whole batch goes back to the caller (issue #899 scopes soft-cap
    // families to a follow-up).
    if softcap != 0.0 {
        return Err("logit soft-cap requested");
    }
    let Some(first) = caches[0].paged_backing.as_ref() else {
        return Err("caches are not pool-backed");
    };
    for cache in caches {
        let Some(backing) = cache.paged_backing.as_ref() else {
            return Err("caches are not pool-backed");
        };
        if cache.mode != KVCacheMode::Fp16 {
            return Err("cache mode is not Fp16");
        }
        if backing.layer_idx != first.layer_idx {
            return Err("caches name different layers");
        }
        // One launch reads one pool; sequences backed by different pools cannot
        // share a page table.
        if !Rc::ptr_eq(&backing.pool, &first.pool) {
            return Err("caches are backed by different pools");
        }
    }
    Ok(())
}

/// `[B, H, 1, D]` shape check for the decode tensors.
fn is_single_token_batch(shape: &[i32], batch: usize) -> bool {
    shape.len() == 4 && shape[0] as usize == batch && shape[2] == 1
}

/// Whole-batch pooled paged decode: append this step's K/V for every sequence,
/// then run attention once over the batch.
///
/// See the module docs for the decline contract. `q_batched` is `[B, Hq, 1, D]`,
/// `k_batched` / `v_batched` are `[B, Hkv, 1, D]`, and `caches[b]` is sequence
/// `b`'s cache **for one layer** (all entries must name the same `layer_idx`
/// of the same pool).
///
/// Used by: Llama3 / Qwen2 / Qwen2.5 / Helium `forward_split_attention`, Qwen3
/// `forward_split_attention`, and every VLM whose text backbone is one of
/// those.
pub fn paged_batch_decode_attention(
    q_batched: &MlxArray,
    k_batched: &MlxArray,
    v_batched: &MlxArray,
    caches: &mut [&mut KVCache],
    scale: f32,
    softcap: f32,
) -> Option<UniquePtr<MlxArray>> {
    let batch = caches.len();
    let servable = batch_is_servable(caches, softcap).and_then(|()| {
        if !is_single_token_batch(&ffi::array_shape(q_batched), batch)
            || !is_single_token_batch(&ffi::array_shape(k_batched), batch)
            || !is_single_token_batch(&ffi::array_shape(v_batched), batch)
        {
            // Batched prefill and speculative / MTP verify land here: more than
            // one query token per sequence, which the kernel does not serve.
            return Err("not a single-token decode step");
        }
        Ok(())
    });
    if let Err(reason) = servable {
        COUNTERS.declines.fetch_add(1, Ordering::Relaxed);
        report_once(&PagedDecodeOutcome::NotServable(reason));
        return None;
    }

    let backing = caches[0]
        .paged_backing
        .clone()
        .expect("batch_is_servable established a paged backing");
    let layer_idx = backing.layer_idx;

    // Past this point the pool is mutated, so every exit must return `Some`.
    for (b, cache) in caches.iter_mut().enumerate() {
        let k_i = slice_row(k_batched, b);
        let v_i = slice_row(v_batched, b);
        cache.write_paged(&k_i, &v_i);
    }

    let states: Vec<Ref<'_, PagedSequenceState>> = caches
        .iter()
        .map(|cache| {
            cache
                .paged_backing
                .as_ref()
                .expect("checked above")
                .state
                .borrow()
        })
        .collect();
    let state_refs: Vec<&PagedSequenceState> = states.iter().map(|state| &**state).collect();

    let mut pool = backing.pool.borrow_mut();
    match pool.paged_decode_batched(q_batched, &state_refs, layer_idx, scale) {
        Ok((launched, outcome)) => {
            // Which kernel a server actually ran has to be visible without a
            // profiler and without an operator opting into debug logging; a
            // before/after benchmark is uninterpretable otherwise, and the
            // first #899 sweep measured gather against gather because this was
            // not true.
            report_once(&outcome);
            record_cascade(&outcome);
            match launched {
                Some(out) => {
                    COUNTERS.v2_launches.fetch_add(1, Ordering::Relaxed);
                    Some(out)
                }
                None => {
                    COUNTERS.gather_fallbacks.fetch_add(1, Ordering::Relaxed);
                    Some(gather_fallback(
                        q_batched,
                        &pool,
                        &state_refs,
                        layer_idx,
                        scale,
                    ))
                }
            }
        }
        Err(reason) => {
            // A hard error here means the batch's own bookkeeping is
            // inconsistent (a length outrunning its blocks, a block with no
            // pool row). The gather path applies the same guards and would fail
            // the same way, so surface it rather than papering over it: the
            // pre-#899 path panicked on exactly these conditions too.
            panic!("PagedBlockPool::paged_decode_batched failed on layer {layer_idx}: {reason}");
        }
    }
}

/// Per-sequence `gather_visible` + fused SDPA, concatenated on the batch axis.
///
/// Byte-identical to what the caller's per-sequence loop produced before #899:
/// the same gather, the same `attention_from_ptr` call, and a batch-axis concat
/// that commutes with the transpose/reshape the caller applies afterward
/// (neither touches axis 0).
fn gather_fallback(
    q_batched: &MlxArray,
    pool: &super::PagedBlockPool,
    states: &[&PagedSequenceState],
    layer_idx: usize,
    scale: f32,
) -> UniquePtr<MlxArray> {
    let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(states.len());
    for (b, state) in states.iter().enumerate() {
        let q_i = slice_row(q_batched, b);
        let (key_visible, value_visible) = pool
            .gather_visible(state, layer_idx)
            .expect("PagedBlockPool::gather_visible failed for pool-backed cache")
            .expect("gather_visible returned None for a pool-backed cache");
        outputs.push(unsafe {
            crate::layers::attention_from_ptr(
                &q_i,
                &key_visible,
                &value_visible,
                scale,
                std::ptr::null(),
                0.0,
                0,
            )
        });
    }
    let mut outputs = outputs.into_iter();
    let mut result = outputs
        .next()
        .expect("a servable batch has at least one sequence");
    for output in outputs {
        result = crate::concatenate(&result, &output, 0);
    }
    result
}

/// Slice row `b` out of a `[B, H, T, D]` tensor, keeping the batch axis.
fn slice_row(arr: &MlxArray, b: usize) -> UniquePtr<MlxArray> {
    ffi::slice(
        arr,
        &[b as i32, 0, 0, 0],
        &[b as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
    )
}

#[cfg(test)]
#[path = "paged_batch_decode_tests.rs"]
mod paged_batch_decode_tests;
