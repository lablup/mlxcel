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

//! Core batch scheduler with iteration-level scheduling and chunked prefill.
//!
//! [`BatchScheduler`] replaces the sequential `loop { request_rx.recv() }`
//! pattern in the model worker. At each tick it decides whether to:
//!
//! - **Prefill** (or continue a chunked prefill of) a queued request,
//! - **Decode** one token for each active sequence, or
//! - **Idle** (block until the next request arrives).
//!
//! When `prefill_chunk_size > 0`, long prompts are broken into chunks and
//! decode steps are interleaved between chunks so active sequences are not
//! starved during prefill of large prompts.

use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::{
    BatchKvQuantConfig, CachePool, DetachedPagedCacheSet, KVCacheMode, PagedKvLayout, SequenceId,
    SequenceStateBackend, SequenceStateLayout,
};
use mlxcel_core::generate::{
    DecodeBatchContext, DecodeStorageBackend as CoreDecodeStorageBackend, LanguageModel,
};
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use mlxcel_core::hardware;
use mlxcel_core::sampling::{
    FusedSampleParams, TokenBiasMap, batched_fused_sample, compute_logprobs,
    row_supports_fused_batch, sample_token_optimized, sample_token_optimized_with_state,
};
use mlxcel_core::streams::{
    install_thread_local_default_stream, new_thread_local_generation_stream,
};
use mlxcel_core::utils::{align_to_na_tile, create_padded_prefill_mask};
use mlxcel_core::{MlxThreadLocalStream, UniquePtr};

use crate::LoadedModel;
use crate::server::ServerGenerateOptions;
use crate::server::batch::observability::BatchObservability;
use crate::server::config::{
    DecodeStorageBackend, PreemptionPolicy, PromptCacheRequestContext, ReasoningBudgetOverride,
};
use crate::server::model_provider::model_worker::{
    StreamingDecodeState, build_generation_result_with_cache, merge_config_stop_tokens,
    prepare_request_vlm_embeddings,
};
use crate::server::model_provider::{GenerateEvent, ModelRequest};
use crate::server::prompt_cache::key::PromptCacheKey;
use crate::server::prompt_cache::{
    CacheEntry, DetachedKvSet, ModelSnapshotEntry, PromptCacheStore,
};
use crate::server::state::BatchMetrics;
use crate::server::thinking_budget::{
    ThinkingBudget, ThinkingDecision, ThinkingState, ThinkingTokenIds,
};
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::feature_cache::ModelVisionCaches;
use crate::vlm_runtime::prepared_embedding_refs;

use super::active::ActiveBatch;
use super::prefill_cohort::{
    PrefillCohortKind, PrefillRow, batched_window_admits, default_batched_prefill_token_budget,
    plan_prefill_cohorts,
};
use super::queue::PrefillQueue;
use super::sequence::{
    BatchSchedulerAction, FinishReason, RequestPriority, SequenceInfo, SequenceState,
};

/// Returns true when the current hardware is M5+ with Neural Accelerator
/// support and tile-aligned prefill should be applied.
#[inline]
fn should_align_prefill() -> bool {
    let hw = hardware::get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

pub(crate) const DEFAULT_PAGED_BLOCK_SIZE: usize = 32;

/// Environment override for the #715 batched-prefill token budget.
const MAX_BATCH_PREFILL_TOKENS_ENV: &str = "MLXCEL_MAX_BATCH_PREFILL_TOKENS";

/// Resolve the effective batched-prefill padded-token budget (#715).
///
/// Precedence (matching the `--flag` > env convention of the other server
/// knobs): the explicit CLI/config value (`configured`, `Some` only when the
/// flag was passed) wins; otherwise the `MLXCEL_MAX_BATCH_PREFILL_TOKENS` env
/// var if set to a valid integer; otherwise the derived default
/// [`default_batched_prefill_token_budget`]. A resolved value of `0` disables
/// the cap (uncapped escape hatch) at every level.
fn resolve_max_batch_prefill_tokens(
    configured: Option<usize>,
    prefill_chunk_size: usize,
    max_batch_prefill: usize,
) -> usize {
    if let Some(value) = configured {
        return value;
    }
    if let Ok(raw) = std::env::var(MAX_BATCH_PREFILL_TOKENS_ENV)
        && let Ok(parsed) = raw.trim().parse::<usize>()
    {
        return parsed;
    }
    default_batched_prefill_token_budget(prefill_chunk_size, max_batch_prefill)
}

/// Decide whether a request may participate in experimental VLM prompt-prefix
/// cache sharing (#124 step c).
///
/// Sharing is allowed only when all three hold:
/// 1. the operator opted in (`--enable-vlm-prefix-cache`),
/// 2. the request actually carries multimodal content, and
/// 3. it has no video payload (video frame bytes are NOT folded into the
///    request's multimodal digest, so a video prefix could collide with a
///    different video in the same bucket).
///
/// Text-only requests and any request with video always return `false`,
/// preserving the legacy cold-prefill path. Keeping the decision in one pure
/// function lets the safety conditions be pinned by a unit test so a future
/// edit cannot silently enable-by-default or allow video sharing.
#[inline]
fn vlm_prefix_sharing_allowed(enabled: bool, is_multimodal: bool, has_videos: bool) -> bool {
    enabled && is_multimodal && !has_videos
}

fn effective_decode_storage_backend(
    requested: DecodeStorageBackend,
    max_batch_size: usize,
    supports_batching: bool,
    supports_paged_decode_backend: bool,
) -> DecodeStorageBackend {
    let paged_available = max_batch_size > 1 && supports_batching && supports_paged_decode_backend;
    match requested {
        DecodeStorageBackend::Auto | DecodeStorageBackend::Paged if paged_available => {
            DecodeStorageBackend::Paged
        }
        DecodeStorageBackend::Auto | DecodeStorageBackend::Paged => DecodeStorageBackend::Dense,
        DecodeStorageBackend::Dense => DecodeStorageBackend::Dense,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkedPrefillRange {
    start: usize,
    end: usize,
    is_terminal: bool,
}

#[inline]
fn next_chunked_prefill_range(
    prompt_len: usize,
    offset: usize,
    chunk_size: usize,
) -> Option<ChunkedPrefillRange> {
    if chunk_size == 0 || offset >= prompt_len {
        return None;
    }
    let end = offset.saturating_add(chunk_size).min(prompt_len);
    Some(ChunkedPrefillRange {
        start: offset,
        end,
        is_terminal: end >= prompt_len,
    })
}

/// Core batch scheduler that drives the model worker loop.
///
/// Replaces the old sequential `recv()` loop with an iteration-level scheduler
/// that interleaves prefill and decode operations. When `max_batch_size == 1`
/// (the default), behavior is identical to the pre-scheduler worker loop.
///
/// When `prefill_chunk_size > 0`, long prompts are processed in chunks with
/// decode interleaving to prevent latency spikes for active sequences.
pub struct BatchScheduler {
    // -- Pool & scheduling structures --
    cache_pool: CachePool,
    prefill_queue: PrefillQueue,
    active_batch: ActiveBatch,

    // -- Model & tokenizer --
    model: LoadedModel,
    tokenizer: MlxcelTokenizer,

    // -- Generation infrastructure --
    //
    // Thread-local generation stream owned by this scheduler. The TLS
    // handle resolves to a per-thread `MlxStream` on demand, which
    // keeps dispatch and synchronization paired on the same thread even
    // if construction and execution happen on different threads.
    // Currently constructed and run on the same thread; the TLS design
    // leaves room to separate them in the future if needed.
    // See `mlxcel_core::streams` and.
    generation_stream: Option<UniquePtr<MlxThreadLocalStream>>,

    // -- Request channel --
    request_rx: mpsc::Receiver<ModelRequest>,

    // -- Metrics --
    /// Shared metrics updated atomically for HTTP handlers to read.
    batch_metrics: Arc<BatchMetrics>,
    /// Detailed observability counters (prefill, decode, cache).
    batch_observability: Arc<BatchObservability>,

    // -- Configuration --
    config_eos: Vec<i32>,
    /// Number of prompt tokens per prefill chunk. 0 = chunking disabled.
    prefill_chunk_size: usize,
    /// Whether preemptive eviction is enabled.
    enable_preemption: bool,
    /// Policy for selecting the eviction victim.
    preemption_policy: PreemptionPolicy,

    // -- Chunked prefill in-progress state --
    /// Sequence currently undergoing chunked prefill. `None` when no chunked
    /// prefill is in progress.
    chunked_prefill_seq: Option<SequenceInfo>,

    // -- Shutdown flag --
    shutdown_requested: bool,

    // -- Batched prefill config --
    /// Maximum number of pending requests to batch together for prefill.
    /// When `> 1`, the scheduler may collect multiple queued requests and
    /// run a single batched forward pass. Falls back to sequential prefill
    /// when only one request is pending or on any error.
    max_batch_prefill: usize,
    /// #715: padded-token budget bounding a single batched-prefill window.
    ///
    /// The padded batched path pads every row of a cohort to the window's
    /// longest prompt `L` and materializes a stacked `[B, L, L]` FP32 attention
    /// mask, an `O(B*L^2)` transient. This caps `B*L` (`rows * max_len`) so the
    /// mask stays within `budget^2 / 2` elements; rows past the budget spill to
    /// the next tick and prefill via the chunked single-sequence path. `0`
    /// disables the cap (uncapped). Resolved once in [`Self::with_config`] /
    /// [`Self::with_max_batch_prefill_tokens`] from the env override, the CLI
    /// value, or the derived default (`2 * max_batch_prefill * prefill_chunk_size`).
    max_batch_prefill_tokens: usize,
    /// Decode-time sequence-state backend used by this scheduler.
    decode_storage_backend: DecodeStorageBackend,

    /// Server-wide KV cache quantization mode.
    ///
    /// When non-Fp16 the scheduler upgrades each newly-allocated sequence's
    /// per-layer `KVCache` to the configured mode (with Boundary-V policy
    /// applied for Turbo4 modes) immediately after `model.make_caches()`
    /// returns its default Fp16 caches. For paged decode this also picks
    /// the Turbo4-aware [`PagedKvLayout`] constructor so the per-page sidecar
    /// budget is reserved.
    ///
    /// Defaults to [`KVCacheMode::Fp16`] (bit-exact baseline).
    kv_cache_mode: KVCacheMode,

    /// server-wide batch KV quantization configuration.
    ///
    /// When [`BatchKvQuantConfig::is_enabled`] returns `true`, the
    /// scheduler resolves per-layer modes from this config (with the
    /// last layer optionally forced to FP16 per
    /// [`BatchKvQuantConfig::skip_last_layer`]) and uses them in place of
    /// the legacy [`Self::kv_cache_mode`]-driven path. When disabled
    /// (the default), the legacy path is bit-exact preserved.
    batch_kv_quant: BatchKvQuantConfig,

    /// maximum KV cache size for plain (non-sliding) `KVCache`
    /// instances managed by this scheduler's `CachePool`.
    ///
    /// When `Some(N)`, the scheduler enforces a hard cap on the **live KV
    /// window** of each per-layer plain `KVCache`: after every prefill
    /// chunk (full prefill, chunked prefill start, chunked prefill
    /// continuation) and every decode step, [`KVCache::trim_front`] is
    /// invoked on each cache whose `live_len()` exceeds `N`, dropping the
    /// oldest excess tokens. **Crucially, the cache's monotonic `offset`
    /// is never decremented** — `trim_front` advances `live_start` so
    /// RoPE relative positions stay correct (see [`KVCache::trim_front`] for the position invariant).
    ///
    /// Sliding-window models that manage their own internal
    /// `RotatingKVCache` go through a separate model-level code path and
    /// are unaffected. `None` (the default) preserves the legacy unbounded
    /// behaviour. Turbo-quantized caches (`Turbo4Asym` / `Turbo4` /
    /// `Turbo4Delegated` / `Turbo3Asym`) are skipped by
    /// [`KVCache::trim_front`] (which returns `0` for those modes) with a
    /// one-time startup warning logged in [`Self::with_max_kv_size`] —
    /// the warning inspects both `kv_cache_mode` and the per-layer modes
    /// resolved from `batch_kv_quant` so the combination
    /// `--kv-quant-scheme=turboquant --max-kv-size=M` is flagged even
    /// when the legacy `--kv-cache-mode` flag is left at FP16.
    max_kv_size: Option<usize>,

    /// Experimental VLM prompt-prefix cache sharing toggle (#124 step c,
    /// `--enable-vlm-prefix-cache`).
    ///
    /// `false` (the default) keeps every multimodal request on the cold-prefill
    /// path (the pre-#124 behavior). When `true`, a multimodal chat request may
    /// adopt a previously donated KV prefix and donate its own, restricted to a
    /// whole-entry match so the prefilled suffix is the newly-appended text
    /// turn (multi-turn same-image conversations). Text-only and non-VLM
    /// requests are unaffected either way.
    enable_vlm_prefix_cache: bool,

    // -- Vision feature cache --
    /// Per-model vision feature cache bundle. Contains LRU caches for
    /// post-projection image features so multi-turn VLM conversations can
    /// skip the vision tower when the same image is referenced across turns.
    ///
    /// Stored as `Rc<..>` because the scheduler is single-threaded (all MLX
    /// work runs on the worker thread). The cache is cleared automatically
    /// when the scheduler (and thus this loaded model) is dropped.
    vision_caches: Rc<ModelVisionCaches>,

    // -- Axis B / — language-bias token map --
    /// Cached per-scheduler `TokenBiasMap` resolved once from the server-level
    /// `LangBiasConfig` at worker startup.
    ///
    /// **Phase 1 limitation — single policy per batch**: every sequence in
    /// this scheduler's active batch receives the same bias, regardless of
    /// per-request preferences. Per-sequence override via the
    /// `/v1/chat/completions` request body is reserved for a follow-up
    /// issue (B12) tracked outside this Epic. The bias is attached to each
    /// queued sequence's [`SamplingConfig`] at `enqueue_request` time so
    /// per-step sampling (`sample_token_optimized`) observes it with no
    /// additional hot-path overhead beyond the existing
    /// [`mlxcel_core::sampling::apply_token_bias`] fast path.
    ///
    /// Empty map = bit-exact baseline path (no sampling change, no alloc).
    token_bias: TokenBiasMap,

    /// Reserved output-illegal token ids for the loaded model (issue #350).
    ///
    /// Multimodal models (e.g. `gemma4_unified`) carry audio / image / video
    /// span placeholder ids that must never be emitted as text. Resolved once
    /// at construction from [`LanguageModel::output_suppressed_token_ids`] and
    /// force-masked to `-inf` on every sequence's sampling config at
    /// [`Self::enqueue_request`] time, unconditionally (suppression always
    /// wins over any per-request bias). Empty for non-multimodal models, which
    /// keeps the enqueue path zero-cost.
    model_output_suppressed: Vec<i32>,

    // -- — thinking-token budget --
    /// Server-wide default thinking-token budget. `None` means unrestricted.
    /// Per-request `thinking_budget_tokens` overrides this at enqueue time.
    reasoning_budget: Option<ThinkingBudget>,
    /// Cached `<think>` / `</think>` token id pair resolved once from the
    /// tokenizer at worker startup. `None` for non-thinking models; when
    /// `None`, every sequence's [`ThinkingState`] is constructed as disabled
    /// regardless of any budget configuration.
    thinking_token_ids: Option<ThinkingTokenIds>,

    // -- cross-request prompt-prefix KV cache --
    /// Shared store that hands out detached KV caches on prefix match and
    /// absorbs donated caches on sequence finish. `None` when the feature is
    /// disabled at config time so the hot path has zero overhead.
    prompt_cache: Option<Arc<PromptCacheStore>>,

    /// Parallel map indexed by `SequenceId`: remembers the
    /// [`PromptCacheRequestContext`] per in-flight sequence so the donate-back
    /// path on completion can rebuild the cache key without touching the HTTP
    /// request layer again. Dropped automatically when the sequence is
    /// removed from the map on finish / error.
    prompt_cache_seq_ctx: std::collections::HashMap<SequenceId, PromptCacheRequestContext>,

    /// resolved speculative-decoding dispatch shape.
    ///
    /// Defaults to [`crate::server::SpeculativeDispatch::Disabled`]
    /// (constructed by [`Self::with_config`]). When the scheduler is
    /// driven by a worker that wires `--draft-model` /
    /// `--draft-kind` / `--draft-block-size`, the matching kind-specific
    /// variant is attached via
    /// [`Self::with_speculative_dispatch`].
    ///
    /// **Hot-path semantics**: every per-request decode-tick first
    /// inspects `self.speculative_dispatch` via [`Self::should_dispatch_speculative`].
    /// When the dispatch is `Disabled` (the default), the inspect is a
    /// single match-arm short-circuit so the bit-exact classic decode
    /// path stays zero-overhead. When the dispatch is a kind-specific
    /// variant AND the per-request preconditions hold (single active
    /// sequence, target supports the matching trait), the scheduler
    /// delegates to the speculative round-loop driver — see the
    /// per-kind dispatch hooks scheduled to land alongside this field.
    /// Until those hooks land, the field is plumbed end-to-end and the
    /// scheduler logs the auto-detected dispatch once at startup, but
    /// still falls back to classic decode at request time. This is the
    /// minimum-viable architectural foundation; full B=1 / B>1 decode
    /// integration completes in the peer follow-up sketched in the
    /// implementation plan.
    speculative_dispatch: crate::server::SpeculativeDispatch,

    /// lazy-loaded drafter handle for the speculative burst
    /// path. Constructed empty alongside the dispatch; the drafter's
    /// weights are loaded from disk on the **first** speculative
    /// request and cached for subsequent requests on the same worker
    /// (mandate 2). For
    /// [`crate::server::SpeculativeDispatch::Disabled`] this slot stays
    /// empty for the worker's lifetime and the burst path is never
    /// entered.
    speculative_drafter_slot: super::speculative_burst::WorkerDrafterSlot,

    /// Adaptive MTP enable/decline policy (issue #333). `Some` only when the
    /// dispatch is [`crate::server::SpeculativeDispatch::Mtp`] and the adaptive
    /// path is enabled (`MLXCEL_MTP_ADAPTIVE` not set to an off value); built
    /// by [`Self::with_mtp_policy`]. When `None` the B=1 MTP gate falls back to
    /// the static per-hardware default
    /// ([`super::speculative_burst::mtp_b1_burst_enabled`]). The policy profiles
    /// the first few B=1 bursts of the (target, drafter, hardware) pairing and
    /// settles to a data-driven verdict; consulted via
    /// [`super::mtp_policy::MtpPolicy::should_attempt_b1`] and fed via
    /// [`super::mtp_policy::MtpPolicy::record_b1_sample`].
    mtp_policy: Option<super::mtp_policy::MtpPolicy>,

    // -- Disaggregated serving-role handoff --
    /// Cached decode-model paged block geometry for cross-node KV handoff
    /// ([`crate::distributed::disaggregated::handoff_impl`]).
    ///
    /// Lazily probed on the first [`Self::ingest_sequence_handoff`] and reused
    /// for every later restore, so the one-token geometry probe runs at most
    /// once per worker. `None` until the first ingest. Wired to a live caller by
    /// the disaggregated decode serving role (a later step); the hooks below are
    /// the in-crate seam it builds on.
    #[allow(dead_code)]
    paged_handoff_geometry: Option<crate::distributed::kv_cache_serde::ExpectedBlockGeometry>,

    // -- Lookahead async_eval decode pipeline (issue #632) --
    /// Prebuilt next-step tokens for the active decode batch, scheduled with
    /// `async_eval` one tick ahead so the GPU forward overlaps the host-side
    /// commit / stop-check / next-tick bookkeeping. `None` outside a steady
    /// pipelined run (the bit-exact synchronous fallback). See
    /// [`DecodeLookahead`] and [`Self::execute_decode_step`] for the state
    /// machine and every invalidation trigger.
    decode_lookahead: Option<DecodeLookahead>,

    /// `MLXCEL_FORCE_SYNC` kill switch, read once at construction (mirrors the
    /// CLI generation loop's hoisted env probe). When `true` the scheduler
    /// never engages the lookahead pipeline and every decode tick runs
    /// synchronously, so the sync path stays available for A/B equivalence.
    lookahead_force_sync: bool,
}

/// One prebuilt decode step held across ticks by the lookahead pipeline.
///
/// The `tokens` array was sampled from a forward that has already been issued
/// (`async_eval`), and that forward appended exactly one KV position per
/// sequence (the last committed token). `tokens` themselves are NOT yet
/// committed and NOT yet in the KV cache. Tearing the pipeline down
/// (`discard_lookahead`) trims that one speculative KV position per sequence
/// and drops `tokens`, restoring the exact synchronous-decode cache invariant.
struct DecodeLookahead {
    /// Active sequence id set this step was built for, in batch order. The
    /// pipeline is only consumed when the next tick decodes this identical set.
    ids: Vec<SequenceId>,
    /// `[B]` device token-id array (uint32), one per `ids` entry, already
    /// scheduled with `async_eval`.
    tokens: UniquePtr<mlxcel_core::MlxArray>,
}

/// Copy a `[B]` device token-id array to host as `Vec<i32>`. `fused_sample`
/// returns a row-contiguous `uint32` array; the raw bytes are reinterpreted as
/// `i32`, exact for any token id in `0..vocab_size`.
///
/// Uses [`mlxcel_core::array_evaluated_bytes`] (surgical per-array `eval`, no
/// `contiguous()` op) rather than `array_to_raw_bytes`: the steady pipeline has
/// already scheduled the next forward on the same stream before this read, and
/// `array_to_raw_bytes`' `contiguous()` would enqueue a fresh op behind that
/// forward, making the read block on it and collapsing the overlap. This reader
/// waits only on the token array's own completion event.
fn lookahead_tokens_to_host(tokens: &mlxcel_core::MlxArray) -> Vec<i32> {
    let bytes = mlxcel_core::array_evaluated_bytes(tokens);
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Pure decision: may the lookahead pipeline stay engaged for the next tick?
///
/// False when the next tick would change batch membership: a queued request is
/// waiting to be admitted, a chunked prefill is interleaving, or a preemption is
/// pending. Extracted from [`BatchScheduler::lookahead_safe`] so unit tests can
/// exercise every invalidation trigger without constructing a live model.
fn lookahead_pipeline_safe(queue_empty: bool, chunked_in_progress: bool, preempting: bool) -> bool {
    queue_empty && !chunked_in_progress && !preempting
}

/// Pure decision: does committing `next_token` finish this sequence?
///
/// A finishing token (cancellation, an EOS / stop id, or the length limit) forces
/// the steady pipeline to discard its prebuilt step and re-run the tick
/// synchronously, so completion and prompt-cache donation always flow through the
/// untouched synchronous path from a clean cache state. Mirrors the per-row
/// pre-check in [`BatchScheduler::pipelined_steady_decode`].
fn lookahead_token_finishes(
    next_token: i32,
    generated_len: usize,
    max_tokens: usize,
    merged_eos: &[i32],
    cancelled: bool,
) -> bool {
    cancelled || merged_eos.contains(&next_token) || generated_len + 1 >= max_tokens
}

/// Number of speculative KV positions a lookahead teardown must unwind.
///
/// The steady tick issues step n+1's prime forward BEFORE it learns step n's
/// finish outcome, so a teardown after a successful prime (`next_prime_issued`)
/// must unwind two speculative appends (step n plus step n+1). A prime that
/// bailed (its forward returned `None`) or any teardown that runs before a prime
/// (admission, preemption, stale id set, cancellation in `finalize_completed`)
/// unwinds one (step n only). Mirrors the count selection in
/// [`BatchScheduler::pipelined_steady_decode`] and the `1`-position teardowns.
fn lookahead_teardown_positions(next_prime_issued: bool) -> usize {
    if next_prime_issued { 2 } else { 1 }
}

impl BatchScheduler {
    fn release_sequence_caches(&mut self, seq_id: SequenceId) {
        self.model.release_sequence_state_by_id(seq_id);
        if let Some(caches) = self.cache_pool.get_caches_mut(seq_id) {
            self.model.release_sequence_state(caches);
        }
        self.cache_pool.release(seq_id);
    }

    fn begin_prefill(seq: &mut SequenceInfo) -> Result<(), String> {
        seq.state.transition_to(SequenceState::Prefilling)?;
        seq.prefill_start = Some(Instant::now());
        seed_rng_if_needed(&seq.sampling);
        Ok(())
    }

    // ── Disaggregated serving-role handoff hooks ─────────────────────────
    //
    // The in-crate seam that lets a serving-role worker move a finished
    // pool-backed sequence's KV across nodes. The mechanism (serialize /
    // anchored restore / one-token geometry probe) lives in
    // `crate::distributed::disaggregated::handoff_impl` and is exercised
    // byte-for-byte against real models by `tests/paged_handoff_parity.rs`.
    // A live caller (the decode / prefill role serve loop) lands in a later
    // step, so these stay `#[allow(dead_code)]` until then.

    /// Prefill role: serialize sequence `seq_id`'s pool-backed KV into a single
    /// wire frame for handoff to a decode node. `token_history` is the
    /// sequence's prompt token ids (so the decode node continues with the same
    /// context).
    #[allow(dead_code)]
    pub(crate) fn extract_sequence_handoff(
        &self,
        seq_id: SequenceId,
        token_history: Vec<i32>,
        generated_tokens: Vec<i32>,
    ) -> anyhow::Result<Vec<u8>> {
        crate::distributed::disaggregated::handoff_impl::extract_sequence_handoff(
            &self.cache_pool,
            seq_id,
            None,
            token_history,
            generated_tokens,
        )
    }

    /// Decode role: reconstruct a handed-off sequence from `bytes` onto a fresh
    /// pool-backed slot, anchored to this worker model's real block geometry,
    /// and return the new local sequence id. The geometry probe runs once on
    /// the first call and is cached in `paged_handoff_geometry`.
    /// Return this worker model's paged block geometry for handoff restores,
    /// probing it once on the first call and caching it in
    /// `paged_handoff_geometry`. `ExpectedBlockGeometry` is `Copy`, so the
    /// cached value is returned by value and leaves no borrow on `self`.
    fn ensure_handoff_geometry(
        &mut self,
    ) -> anyhow::Result<crate::distributed::kv_cache_serde::ExpectedBlockGeometry> {
        if let Some(geometry) = self.paged_handoff_geometry {
            return Ok(geometry);
        }
        let probed = crate::distributed::disaggregated::handoff_impl::probe_block_geometry(
            &self.model,
            DEFAULT_PAGED_BLOCK_SIZE,
        )?;
        self.paged_handoff_geometry = Some(probed);
        Ok(probed)
    }

    #[allow(dead_code)]
    pub(crate) fn ingest_sequence_handoff(&mut self, bytes: &[u8]) -> anyhow::Result<SequenceId> {
        let geometry = self.ensure_handoff_geometry()?;
        crate::distributed::disaggregated::handoff_impl::ingest_sequence_handoff(
            &mut self.cache_pool,
            &self.model,
            bytes,
            &crate::distributed::kv_cache_serde::CacheIngestLimits::default(),
            &geometry,
            DEFAULT_PAGED_BLOCK_SIZE,
        )
    }

    /// Whether this node's model can participate in the disaggregated pool-block
    /// KV handoff (#125). The handoff extracts pool-backed Fp16 KV, which only the
    /// dense external KV-cache families (natural backend `DenseKvCache`: qwen3 /
    /// llama3) produce. Model-owned paged families (gemma3 / gemma4 / llama4 /
    /// qwen3_5 / qwen3_next, natural backend `ModelOwned`) keep their KV
    /// model-internal and are routed through the paged backend for shadow
    /// accounting only, so there is nothing pool-paged to extract: a handoff
    /// attempt reads unwritten pool tensors and used to crash the prefill serving
    /// loop (#708). The whole node serves one model, so this is a node-level fact
    /// the serving-role loop checks once and applies to every request.
    pub(crate) fn handoff_supported(&self) -> bool {
        self.model.sequence_state_layout().backend == SequenceStateBackend::DenseKvCache
    }

    /// Prefill role (#126 B2b): run a full prefill for `seq`, then extract its
    /// pool-backed KV as a handoff frame for a decode node and release the local
    /// caches.
    ///
    /// Returns `Ok(None)` when the request completed during prefill (an immediate
    /// EOS at the first token), in which case [`finish_prefill`] already finalized
    /// and released it and there is nothing to hand off.
    ///
    /// This is the "reuse then extract" factoring (option C): it drives the
    /// standard [`Self::execute_full_prefill`] path (or, for prompts longer than
    /// `--prefill-chunk-size`, the standard chunked-prefill path driven to
    /// completion, issue #197) verbatim, so first-token sampling, structured
    /// output, thinking budget, and logprobs are byte-for-byte identical to a
    /// single-node prefill, then lifts the finished sequence back out of the
    /// active batch before any local decode step runs. The hot [`Self::run`] loop
    /// is never touched. Speculative burst is bypassed (it would complete the
    /// request locally, defeating the handoff).
    ///
    /// [`finish_prefill`]: Self::finish_prefill
    #[allow(dead_code)]
    pub(crate) fn prefill_request_for_handoff(
        &mut self,
        mut seq: SequenceInfo,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        // The serving-role intake is strictly sequential, so a chunked prefill
        // left in progress would be a wiring bug; refuse rather than clobber it.
        if self.chunked_prefill_seq.is_some() {
            anyhow::bail!("prefill-role handoff: another chunked prefill is already in progress");
        }
        let seq_id = seq.seq_id;
        // The decode node restores the prompt context from `token_history`; the
        // first sampled token rides in `generated_tokens` so it seeds decode.
        let prompt_tokens = seq.prompt_tokens.clone();
        if let Err(err) = Self::begin_prefill(&mut seq) {
            self.abort_sequence(seq, &err);
            anyhow::bail!("prefill-role handoff: begin_prefill failed: {err}");
        }
        // Reuse the standard prefill machinery: it samples the first token and
        // transitions the sequence into the active batch (no speculative burst on
        // the handoff path). Long prompts take the same chunked path the run()
        // loop uses, driven to completion here so the full prompt's KV is in the
        // pool before extraction (issue #197); the final chunk samples the first
        // token via finish_prefill exactly like a single-node chunked prefill.
        // The handoff path is text-only, so the VLM-embeddings full-prefill
        // exemption in execute_prefill's dispatch cannot apply.
        if self.prefill_chunk_size > 0 && prompt_tokens.len() > self.prefill_chunk_size {
            self.start_chunked_prefill(seq);
            while self.chunked_prefill_seq.is_some() {
                self.continue_chunked_prefill();
            }
        } else {
            self.execute_full_prefill(seq);
        }
        // Lift the just-prefilled sequence back out before any local decode runs.
        // If it finished at prefill (immediate EOS) it is already finalized and
        // released, so there is nothing to hand off.
        let Some(seq) = self.active_batch.remove(seq_id) else {
            return Ok(None);
        };
        let generated_tokens = seq.generated_tokens.clone();
        let bytes = self.extract_sequence_handoff(seq_id, prompt_tokens, generated_tokens);
        // Release the local caches and per-sequence tracking on BOTH outcomes:
        // on success the KV now belongs to the decode node (no donate-back, the
        // prefix left this node); on an extract error the sequence was already
        // lifted out of the active batch, so skipping the release would leak
        // its pool slot.
        self.prompt_cache_seq_ctx.remove(&seq_id);
        self.release_sequence_caches(seq_id);
        Ok(Some(bytes?))
    }

    /// Prefill role (#126 B3a): build a queued text sequence from the raw request
    /// parts the serving-role prefill loop carries, run a full prefill, and
    /// extract the pool-backed KV as a handoff frame for a decode node.
    ///
    /// The disaggregated path is text-only over the pool-backed Fp16 families
    /// (qwen3 / llama3), so this builds the minimal text sequence (no VLM
    /// embeddings, prompt-prefix adoption, thinking budget, or structured
    /// output) and reuses [`Self::prefill_request_for_handoff`] for the prefill
    /// and extract. `response_tx` carries the first sampled token's text back to
    /// the caller, which is the prefill node's half of the streamed output (the
    /// decode node emits the continuation, mirroring the router's first-token +
    /// decode-token merge). `cancelled` is the client's cancellation flag.
    /// Returns `Ok(None)` when the request hit EOS at the first token, in which
    /// case there is nothing to hand off.
    ///
    /// The empty-prompt guard runs before a cache slot is allocated, so a
    /// rejected request leaks no pool state. Prompts longer than
    /// `--prefill-chunk-size` are prefilled in chunks by
    /// [`Self::prefill_request_for_handoff`] (issue #197).
    #[allow(dead_code)]
    pub(crate) fn prefill_text_request_for_handoff(
        &mut self,
        prompt_tokens: Vec<i32>,
        sampling: mlxcel_core::generate::SamplingConfig,
        max_tokens: usize,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        // Reject model-owned paged families up front, before any allocation or
        // prefill work: the pool-block handoff only supports pool-backed dense
        // Fp16 families (#125). A model-owned family keeps its KV model-internal,
        // so extraction would read unwritten pool tensors (#708). The serving-role
        // loop turns this error into a per-request failure frame and keeps serving.
        if !self.handoff_supported() {
            anyhow::bail!(
                "prefill-role handoff: the disaggregated handoff does not support this model's \
                 {:?} sequence-state backend; only pool-backed dense Fp16 families (qwen3 / \
                 llama3) can be handed off. Model-owned paged families (gemma3 / gemma4 / llama4 \
                 / qwen3_5 / qwen3_next) keep their KV model-internal (issue #708).",
                self.model.sequence_state_layout().backend
            );
        }
        if prompt_tokens.is_empty() {
            anyhow::bail!("prefill-role handoff: empty prompt has no tokens to prefill");
        }
        // Admission cap for the network-facing intake: with chunked prefill
        // supported (issue #197) the old chunk-size bail no longer bounds the
        // accepted prompt, so an oversized PrefillRequestFrame could drive a
        // multi-minute synchronous chunk loop on the node's only prefill
        // worker. 1M tokens matches the CacheIngestLimits philosophy (far
        // above any realistic context; the pool budget is the real bound).
        const MAX_HANDOFF_PROMPT_TOKENS: usize = 1 << 20;
        if prompt_tokens.len() > MAX_HANDOFF_PROMPT_TOKENS {
            anyhow::bail!(
                "prefill-role handoff: prompt of {} tokens exceeds the admission cap \
                 ({MAX_HANDOFF_PROMPT_TOKENS})",
                prompt_tokens.len()
            );
        }
        // Merge the model's configured stop tokens into the request sampling
        // exactly as the single-node intake (`enqueue_request`) does, so the
        // handoff prefill samples the first token identically.
        let sampling = merge_config_stop_tokens(sampling, &self.config_eos);
        let seq_id = self
            .allocate_sequence_state()
            .map_err(|e| anyhow::anyhow!("prefill-role handoff: allocate sequence: {e}"))?;
        let decode_state = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);
        let seq = SequenceInfo {
            seq_id,
            state: SequenceState::Queued,
            prompt_tokens,
            sampling,
            max_tokens,
            eos_token_ids: self.config_eos.clone(),
            priority: RequestPriority::default(),
            logprobs_config: mlxcel_core::sampling::LogprobsConfig::default(),
            vlm_embeddings: None,
            images: Vec::new(),
            audio: Vec::new(),
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            decode_state,
            prefill_offset: 0,
            prefill_start_offset: 0,
            already_cached_tokens: 0,
            response_tx,
            cancelled,
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
            token_history: Vec::new(),
            sampler_state: None,
            merged_eos: Vec::new(),
            thinking: crate::server::thinking_budget::ThinkingState::disabled(),
            structured: None,
        };
        self.prefill_request_for_handoff(seq)
    }

    /// Decode role (#126 B2b): reconstruct a handed-off sequence onto a fresh pool
    /// slot and register it as a live decode sequence in the active batch, seeded
    /// with the prefill node's generated token(s) so the next decode step feeds
    /// the right token.
    ///
    /// `max_tokens`, `sampling`, and `response_tx` are coordination parameters
    /// supplied by the decode node's request layer (the router / stream bridge in
    /// a real deployment, the test harness in B2c): the KV handoff frame carries
    /// the cache, the prompt token history, and the generated tokens, while the
    /// per-request budget, sampling policy, and output stream stay with the node
    /// that holds the client connection. The deserialization happens once here and
    /// is reused for both the paged restore and the request context.
    ///
    /// Logprobs, thinking-budget, and structured-output continuation across the
    /// handoff are out of scope for this step.
    #[allow(dead_code)]
    pub(crate) fn ingest_handoff_as_active(
        &mut self,
        bytes: &[u8],
        max_tokens: usize,
        sampling: mlxcel_core::generate::SamplingConfig,
        response_tx: mpsc::Sender<GenerateEvent>,
    ) -> anyhow::Result<SequenceId> {
        let geometry = self.ensure_handoff_geometry()?;
        let limits = crate::distributed::kv_cache_serde::CacheIngestLimits::default();
        // Deserialize once: the restore below consumes the KV blocks, and the
        // request context (prompt token history + the prefill node's generated
        // tokens) seeds the live decode sequence.
        let state = crate::distributed::kv_cache_serde::deserialize_cache_state_with_limits(
            bytes, &limits,
        )?;
        let seq_id =
            crate::distributed::disaggregated::handoff_impl::ingest_sequence_handoff_state(
                &mut self.cache_pool,
                &self.model,
                &state,
                &limits,
                &geometry,
                DEFAULT_PAGED_BLOCK_SIZE,
            )?;

        let prompt_tokens = state.token_history.clone();
        let generated_tokens = state.generated_tokens.clone();
        let needs_history = sampling.needs_token_history();
        // Rebuild the penalty history exactly as a single-node run would have it
        // after prefill: prompt prefix (when penalties need it) plus whatever the
        // prefill node already generated.
        let mut token_history = initial_token_history(&prompt_tokens, needs_history);
        if needs_history {
            token_history.extend_from_slice(&generated_tokens);
        }
        let merged_eos = merged_eos_token_ids(self.model.eos_token_ids(), &sampling.stop_token_ids);
        // Seed the incremental detokenizer with everything already produced (the
        // prompt plus the handed-off tokens) so the decode node's text continues
        // from the correct boundary.
        let detok_seed: Vec<i32> = prompt_tokens
            .iter()
            .chain(generated_tokens.iter())
            .copied()
            .collect();
        let decode_state = StreamingDecodeState::new(&self.tokenizer, &detok_seed);
        let prefill_offset = prompt_tokens.len();

        let seq = SequenceInfo {
            seq_id,
            state: SequenceState::Decoding,
            prompt_tokens,
            sampling,
            max_tokens,
            eos_token_ids: self.config_eos.clone(),
            priority: RequestPriority::default(),
            logprobs_config: mlxcel_core::sampling::LogprobsConfig::default(),
            vlm_embeddings: None,
            images: Vec::new(),
            audio: Vec::new(),
            generated_tokens,
            generated_text: String::new(),
            decode_state,
            prefill_offset,
            prefill_start_offset: 0,
            already_cached_tokens: 0,
            response_tx,
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: Some(Instant::now()),
            token_history,
            sampler_state: None,
            merged_eos,
            thinking: crate::server::thinking_budget::ThinkingState::disabled(),
            structured: None,
        };

        if self.active_batch.add(seq).is_err() {
            // No room in the active batch: release the restored KV so the rejected
            // handoff does not leak a sequence or its pool blocks.
            self.release_sequence_caches(seq_id);
            anyhow::bail!("handoff decode admission failed: active batch is full");
        }
        Ok(seq_id)
    }

    /// Decode role (#126 B2b): drive every active sequence to completion, reusing
    /// the same per-tick `execute_decode_step` + `finalize_completed` that the hot
    /// [`Self::run`] loop calls, without touching `run()` itself. Returns once the
    /// active batch has drained (each sequence reached its EOS or token budget).
    #[allow(dead_code)]
    pub(crate) fn decode_handoff_until_idle(&mut self) {
        while self.decode_handoff_step() {}
    }

    /// One decode tick of the handoff drive loop (issue #199): run a single
    /// `execute_decode_step` + `finalize_completed` over the active batch and
    /// report whether any sequence remains. The networked decode role calls
    /// this per tick so it can drain and ship newly produced tokens
    /// incrementally instead of buffering the whole continuation.
    ///
    /// Returns `false` (without stepping) when the active batch is already
    /// empty, so `while decode_handoff_step() {}` is exactly
    /// [`Self::decode_handoff_until_idle`].
    #[allow(dead_code)]
    pub(crate) fn decode_handoff_step(&mut self) -> bool {
        if self.active_batch.is_empty() {
            return false;
        }
        let ids = self.active_batch.sequence_ids();
        self.execute_decode_step(&ids);
        self.finalize_completed();
        !self.active_batch.is_empty()
    }

    /// Create a new batch scheduler, taking ownership of the model and channel.
    ///
    /// Currently unused at the call site (`with_config` is what production
    /// constructs) but retained as a convenience for future tests/benches.
    /// The scheduler API is `pub(crate)` after the refactor so
    /// `dead_code` is silenced explicitly rather than dropped, keeping the
    /// preserved-behavior intent visible.
    #[allow(dead_code)]
    pub(crate) fn new(
        model: LoadedModel,
        tokenizer: MlxcelTokenizer,
        config_eos: Vec<i32>,
        request_rx: mpsc::Receiver<ModelRequest>,
        max_batch_size: usize,
        max_queue_depth: usize,
        batch_metrics: Arc<BatchMetrics>,
    ) -> Self {
        Self::with_config(
            model,
            tokenizer,
            config_eos,
            request_rx,
            max_batch_size,
            max_queue_depth,
            batch_metrics,
            Arc::new(BatchObservability::new()),
            0,
            false,
            PreemptionPolicy::default(),
            1,
            DecodeStorageBackend::Dense,
        )
    }

    /// Create a new batch scheduler with chunked-prefill and preemption config.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_config(
        model: LoadedModel,
        tokenizer: MlxcelTokenizer,
        config_eos: Vec<i32>,
        request_rx: mpsc::Receiver<ModelRequest>,
        max_batch_size: usize,
        max_queue_depth: usize,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
        prefill_chunk_size: usize,
        enable_preemption: bool,
        preemption_policy: PreemptionPolicy,
        max_batch_prefill: usize,
        decode_storage_backend: DecodeStorageBackend,
    ) -> Self {
        let generation_stream = new_thread_local_generation_stream();
        let max_batch_size = max_batch_size.max(1);
        let effective_decode_storage = effective_decode_storage_backend(
            decode_storage_backend,
            max_batch_size,
            model.supports_batching(),
            model.supports_paged_decode_backend(),
        );
        if decode_storage_backend == DecodeStorageBackend::Paged
            && effective_decode_storage != decode_storage_backend
        {
            tracing::info!(
                "Paged decode storage requested but unavailable for this worker; falling back to dense"
            );
            batch_observability.record_decode_storage_fallback();
        }
        // Non-batching models use lightweight placeholder entries in the pool
        // (no real KV caches), so we size the pool to cover both the active
        // batch and the prefill queue so requests can be queued while another
        // sequence is generating.
        let pool_capacity = max_batch_size + max_queue_depth;
        // issue #350: resolve the model's reserved output-illegal placeholder
        // ids once, before `model` is moved into the scheduler. Empty for
        // non-multimodal models (zero cost on the enqueue path).
        let model_output_suppressed = model.output_suppressed_token_ids();
        Self {
            cache_pool: CachePool::new(pool_capacity),
            prefill_queue: PrefillQueue::with_capacity(max_queue_depth),
            active_batch: ActiveBatch::new(max_batch_size),
            model,
            tokenizer,
            generation_stream,
            request_rx,
            batch_metrics,
            batch_observability,
            config_eos,
            prefill_chunk_size,
            enable_preemption,
            preemption_policy,
            chunked_prefill_seq: None,
            shutdown_requested: false,
            max_batch_prefill: max_batch_prefill.max(1),
            // #715: resolve the batched-prefill token budget from the env
            // override or the derived default (`max_batch_prefill *
            // prefill_chunk_size`). `with_max_batch_prefill_tokens` overrides
            // this later with an explicit CLI value when one was passed.
            max_batch_prefill_tokens: resolve_max_batch_prefill_tokens(
                None,
                prefill_chunk_size,
                max_batch_prefill.max(1),
            ),
            decode_storage_backend: effective_decode_storage,
            vision_caches: Rc::new(ModelVisionCaches::new(
                crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE,
            )),
            token_bias: TokenBiasMap::default(),
            model_output_suppressed,
            reasoning_budget: None,
            thinking_token_ids: None,
            prompt_cache: None,
            prompt_cache_seq_ctx: std::collections::HashMap::new(),
            kv_cache_mode: KVCacheMode::Fp16,
            batch_kv_quant: BatchKvQuantConfig::default(),
            max_kv_size: None,
            // multimodal prefix-cache sharing stays off until the operator
            // opts in via `with_vlm_prefix_cache` (#124 step c).
            enable_vlm_prefix_cache: false,
            // dispatch defaults to Disabled so the scheduler's
            // hot path stays bit-exact for the non-speculative case. The
            // worker thread overrides this via `with_speculative_dispatch`
            // when the operator passed `--draft-model`.
            speculative_dispatch: crate::server::SpeculativeDispatch::Disabled,
            // empty slot, populated lazily on the first
            // speculative request when `speculative_dispatch` is
            // kind-specific. `with_speculative_dispatch` rebuilds this
            // slot from the dispatch passed by the worker.
            speculative_drafter_slot: super::speculative_burst::WorkerDrafterSlot::from_dispatch(
                &crate::server::SpeculativeDispatch::Disabled,
            ),
            // No adaptive MTP policy until `with_mtp_policy` builds one for an
            // MTP dispatch. The non-speculative hot path never touches it.
            mtp_policy: None,
            paged_handoff_geometry: None,
            decode_lookahead: None,
            // Honor MLXCEL_FORCE_SYNC=1 as the pipeline kill switch, probed once
            // here (the server worker is long-lived, so a per-tick getenv would
            // be pure overhead).
            lookahead_force_sync: std::env::var("MLXCEL_FORCE_SYNC").is_ok(),
        }
    }

    /// Override the #715 batched-prefill padded-token budget with the explicit
    /// CLI/config value (`--max-batch-prefill-tokens`).
    ///
    /// `configured` is `None` when the flag was not passed (keep the env value
    /// or the derived default already resolved in [`Self::with_config`]),
    /// `Some(0)` for the uncapped escape hatch, or `Some(n)` for an explicit
    /// cap. An explicit value wins over `MLXCEL_MAX_BATCH_PREFILL_TOKENS` (see
    /// [`resolve_max_batch_prefill_tokens`]).
    pub fn with_max_batch_prefill_tokens(mut self, configured: Option<usize>) -> Self {
        self.max_batch_prefill_tokens = resolve_max_batch_prefill_tokens(
            configured,
            self.prefill_chunk_size,
            self.max_batch_prefill,
        );
        self
    }

    /// Returns the resolved batched-prefill padded-token budget (0 = uncapped).
    /// Exposed for tests.
    pub fn max_batch_prefill_tokens(&self) -> usize {
        self.max_batch_prefill_tokens
    }

    /// Attach the server-wide KV cache quantization mode.
    ///
    /// When `mode == KVCacheMode::Fp16` this builder is a no-op — every new
    /// sequence keeps the default Fp16 caches and the paged layout uses
    /// `PagedKvLayout::uniform`. For Turbo4 modes (`Turbo4Asym`, `Turbo4`,
    /// `Turbo4Delegated`) the scheduler additionally picks
    /// [`PagedKvLayout::uniform_with_mode`] so per-page sidecars are
    /// reserved, preserving the cross-tenant isolation contract.
    pub fn with_kv_cache_mode(mut self, mode: KVCacheMode) -> Self {
        self.kv_cache_mode = mode;
        self
    }

    /// Returns the server-wide KV cache quantization mode (for tests).
    pub fn kv_cache_mode(&self) -> KVCacheMode {
        self.kv_cache_mode
    }

    /// Attach the server-wide batch KV quantization configuration
    ///
    /// When `config.is_enabled()` returns `true`, every newly-allocated
    /// sequence's per-layer caches are upgraded from the default Fp16
    /// to the resolved nominal [`KVCacheMode`], with the last layer
    /// optionally forced back to Fp16 per `config.skip_last_layer`. When
    /// `config.is_enabled()` is `false` (the default) this builder is a
    /// no-op and the legacy [`Self::with_kv_cache_mode`] path is used
    /// bit-exactly.
    pub fn with_batch_kv_quant(mut self, config: BatchKvQuantConfig) -> Self {
        self.batch_kv_quant = config;
        self
    }

    /// Returns the server-wide batch KV quantization configuration (for
    /// tests).
    pub fn batch_kv_quant(&self) -> BatchKvQuantConfig {
        self.batch_kv_quant
    }

    /// Set the maximum KV cache size for plain (non-sliding) caches.
    ///
    /// When `max_kv_size.is_some()`, the scheduler advances `live_start` on
    /// every plain `KVCache` after each prefill chunk and each decode step
    /// so the live window stays bounded. `self.offset` stays monotonic so
    /// RoPE relative positions are preserved across the cap — see
    /// [`KVCache::trim_front`] for the position invariant. Sliding-window
    /// model caches (managed by the model itself as internal
    /// `RotatingKVCache` instances, not via this pool's `Vec<KVCache>`)
    /// are unaffected.
    ///
    /// Turbo-quantized caches (`Turbo4Asym` / `Turbo4` / `Turbo4Delegated`
    /// / `Turbo3Asym`) are silently skipped by `KVCache::trim_front` — a
    /// warning is emitted here so the operator knows the combination is
    /// unsupported. ** H3**: the warning now inspects *both* the
    /// legacy `kv_cache_mode` flag *and* the per-layer modes resolved
    /// from `batch_kv_quant` so the combination
    /// `--kv-bits=N --kv-quant-scheme=turboquant --max-kv-size=M` is
    /// flagged even when `kv_cache_mode` is the default `Fp16`.
    ///
    /// Mirrors upstream mlx-lm `BatchGenerator(max_kv_size=N)` (PR #1106).
    pub fn with_max_kv_size(mut self, max_kv_size: Option<usize>) -> Self {
        if max_kv_size.is_some() {
            // Legacy `--kv-cache-mode`-driven path.
            let legacy_is_turbo = Self::is_turbo_mode(self.kv_cache_mode);
            // `--kv-bits` / `--kv-quant-scheme` path. When
            // batch KV quant is enabled, `base_mode()` reports the
            // effective per-layer mode driving the paged-layout
            // selection; we treat any Turbo base mode the same way as
            // the legacy Turbo flags.
            let batched_is_turbo = self.batch_kv_quant.is_enabled()
                && Self::is_turbo_mode(self.batch_kv_quant.base_mode());
            if legacy_is_turbo || batched_is_turbo {
                tracing::warn!(
                    "--max-kv-size is set together with a Turbo KV quantization mode \
                     (legacy_mode={:?}, batch_kv_quant_base_mode={:?}); Turbo-quantized \
                     layers will NOT be capped — the cap only applies to plain Fp16/Int8 \
                     KVCache layers. Consider omitting --max-kv-size or switching to a \
                     non-Turbo KV cache mode.",
                    self.kv_cache_mode,
                    if self.batch_kv_quant.is_enabled() {
                        Some(self.batch_kv_quant.base_mode())
                    } else {
                        None
                    },
                );
            }
            // Int8 KV forces the dense decode backend (only genuine Fp16
            // sequences are pool-backed on the paged path). The dense
            // batched-decode + front-trim path currently produces incorrect
            // output once a prompt exceeds the cap (a pre-existing,
            // mode-independent defect, tracked in issue #718): `--kv-cache-mode fp16
            // --decode-storage-backend dense --max-kv-size N` mis-decodes the
            // same way, while the KV-cache-layer Int8 trim is proven correct
            // by unit tests. Warn loudly so operators do not silently get
            // garbage; the cap is reliable today only on the paged backend
            // (default Fp16).
            let legacy_is_int8 = self.kv_cache_mode == KVCacheMode::Int8;
            let batched_is_int8 = self.batch_kv_quant.is_enabled()
                && self.batch_kv_quant.base_mode() == KVCacheMode::Int8;
            if legacy_is_int8 || batched_is_int8 {
                tracing::warn!(
                    "--max-kv-size is set together with Int8 KV, which runs on the dense \
                     decode backend. The dense batched-decode front-trim currently \
                     mis-decodes prompts longer than the cap (a pre-existing defect that \
                     also affects `--kv-cache-mode fp16 --decode-storage-backend dense`). \
                     Omit --max-kv-size with Int8, or keep prompts within the cap, until \
                     the dense-decode trim path is fixed."
                );
            }
        }
        self.max_kv_size = max_kv_size;
        self
    }

    /// Returns the configured maximum KV cache size (for tests).
    pub fn max_kv_size(&self) -> Option<usize> {
        self.max_kv_size
    }

    /// Enable experimental VLM prompt-prefix cache sharing (#124 step c,
    /// `--enable-vlm-prefix-cache`).
    ///
    /// Default off. When on, multimodal chat requests may adopt and donate KV
    /// prefixes for multi-turn same-image conversations (whole-entry match, so
    /// the prefilled suffix is the newly-appended text turn). Text-only and
    /// non-VLM behavior is unchanged.
    pub fn with_vlm_prefix_cache(mut self, enabled: bool) -> Self {
        self.enable_vlm_prefix_cache = enabled;
        self
    }

    /// Install the paged KV block budget (epic #116 #122 b3).
    ///
    /// `Some(n)` caps the paged pool at `n` blocks — the admission gate in
    /// [`Self::admit_paged_prefill`] then evicts cold prefixes / preempts to
    /// stay within it. `None` (the default) keeps the pool unbounded, the
    /// behaviour-preserving path. The block count is resolved from the
    /// operator's `--kv-cache-budget` directive by
    /// [`crate::memory_estimate::resolve_paged_block_budget`] on the worker
    /// thread (where the model geometry is known). Only meaningful for
    /// pool-backed (Fp16, dense-natural-backend) sequences; inert for
    /// model-owned / quantized families that keep dense caches and never mint
    /// pool blocks.
    pub fn with_paged_block_budget(mut self, budget: Option<usize>) -> Self {
        self.cache_pool.set_paged_block_budget(budget);
        self
    }

    /// Attach the resolved speculative-decoding dispatch.
    ///
    /// Default (constructed by [`Self::with_config`]) is
    /// [`crate::server::SpeculativeDispatch::Disabled`], so callers that
    /// don't pass `--draft-model` keep the bit-exact classic decode path
    /// with zero overhead.
    ///
    /// When `dispatch` is one of [`crate::server::SpeculativeDispatch::Mtp`],
    /// [`crate::server::SpeculativeDispatch::DFlash`], or
    /// [`crate::server::SpeculativeDispatch::Classic`], the scheduler logs
    /// the dispatch at the next decode tick and (in a follow-up issue)
    /// constructs the matching round-loop driver per request when the
    /// per-request preconditions hold.
    ///
    /// **Preconditions for the kind-specific dispatch (Mtp / DFlash)**:
    ///
    /// 1. The active batch has size exactly 1 (continuous batching at
    ///    B>1 is incompatible with the existing self-contained round-loop
    ///    drivers — they own the full round loop, not a single tick — so
    ///    the integration falls back to classic decode at B>1 and logs a
    ///    one-time warning at worker startup; see `model_worker.rs`).
    /// 2. The target wraps a model that implements the matching
    ///    [`mlxcel_core::speculative::mtp::target::MtpTarget`] trait (for
    ///    MTP) or
    ///    [`mlxcel_core::drafter::dflash::SpeculativeTarget`] (for
    ///    DFlash). Today that means:
    ///    - **MTP**: `Gemma4Wrapper` / `Gemma4VLModel` — see the
    ///      `MtpTarget` impls in
    ///      [`crate::models::gemma4_mtp_target`].
    ///    - **DFlash**: `Qwen35Model` / `Qwen35VLModel` — see the
    ///      `SpeculativeTarget` impl in `crate::models::qwen3_5`.
    /// 3. The drafter weights are loadable at the recorded
    ///    `draft_model_path`. Drafter loading itself happens lazily on
    ///    the worker thread the first time the dispatch arm is selected
    ///    (so a never-used drafter never costs anything beyond the
    ///    config-file parse already done at startup).
    pub fn with_speculative_dispatch(
        mut self,
        dispatch: crate::server::SpeculativeDispatch,
    ) -> Self {
        // rebuild the (still-empty) drafter slot to carry
        // the path + kind from the new dispatch. The drafter weights
        // are NOT loaded here — `ensure_loaded` on the first
        // speculative request is what reads from disk.
        self.speculative_drafter_slot =
            super::speculative_burst::WorkerDrafterSlot::from_dispatch(&dispatch);
        self.speculative_dispatch = dispatch;
        self
    }

    /// Attach the adaptive MTP enable/decline policy (issue #333).
    ///
    /// Must be chained **after** [`Self::with_speculative_dispatch`] so the
    /// resolved dispatch (and the drafter checkpoint identity) are in place.
    /// The policy is built only for [`crate::server::SpeculativeDispatch::Mtp`]
    /// and only when the adaptive path is enabled; for any other dispatch, or
    /// when `MLXCEL_MTP_ADAPTIVE` is set to an off value, the field stays
    /// `None` and the B=1 gate keeps the pre-#333 static per-hardware default.
    ///
    /// `target_model_id` is the coarse, non-request-identifying target
    /// identity (the model directory basename) used as one third of the
    /// persisted-hint key; the worker passes the served model's basename.
    /// Building the policy reads the persisted hint from disk once here, at
    /// worker startup, so the per-request gate performs no IO.
    pub fn with_mtp_policy(mut self, target_model_id: Option<String>) -> Self {
        if let crate::server::SpeculativeDispatch::Mtp {
            draft_model_path,
            block_size,
            ..
        } = &self.speculative_dispatch
        {
            let drafter_id = draft_model_path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown-drafter".to_string());
            let target_id = target_model_id.unwrap_or_else(|| "unknown-target".to_string());
            self.mtp_policy = super::mtp_policy::MtpPolicy::initialize(
                target_id,
                drafter_id,
                *block_size,
                self.model.supports_batching(),
            );
        }
        self
    }

    /// Returns the configured speculative-decoding dispatch (for tests
    /// and operator-visible diagnostic endpoints).
    pub fn speculative_dispatch(&self) -> &crate::server::SpeculativeDispatch {
        &self.speculative_dispatch
    }

    /// Whether the scheduler has a kind-specific speculative dispatch
    /// configured (wired this into the actual runtime dispatch).
    ///
    /// Returns `true` when [`Self::speculative_dispatch`] is one of the
    /// kind-specific variants ([`crate::server::SpeculativeDispatch::Mtp`]
    /// or [`crate::server::SpeculativeDispatch::DFlash`]).
    ///
    /// **Semantics**: a `true` return only means a
    /// speculative *path* is configured — the actual per-request
    /// decision happens inside [`Self::execute_prefill`] via
    /// [`super::speculative_burst::should_burst_for_sequence`], which
    /// adds per-sequence preconditions (no multimodal payload / VLM
    /// embeddings, no structured-output constraint, no adopted
    /// prompt-cache prefix).
    /// The active-batch size is NOT consulted by this gate any more:
    /// the burst takes the full request lifecycle (prefill + decode)
    /// in one tick, so it never enters [`Self::active_batch`] and the
    /// B-size of concurrent classic requests is independent of whether
    /// this gate fires for a speculative request.
    ///
    /// Backwards compatibility: earlier callers (the worker-startup log and the integration tests) used this method as a
    /// "would we dispatch?" probe. The semantics remain compatible:
    /// `true` ↔ "speculative is on and a future request could enter
    /// the burst path"; `false` ↔ "every request takes the classic
    /// path." The active-batch-size restriction was a earlier
    /// over-approximation that the burst design removes.
    pub fn should_dispatch_speculative(&self) -> bool {
        self.speculative_dispatch.is_kind_specific()
    }

    /// Replace the default vision feature cache with one sized per the server
    /// configuration.
    ///
    /// `max_size == 0` disables the cache entirely; non-zero values mirror
    /// the `--vision-cache-size` CLI flag. Callers that do not invoke this
    /// method get the default size from
    /// [`crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE`].
    pub fn with_vision_cache_size(mut self, max_size: usize) -> Self {
        self.vision_caches = Rc::new(ModelVisionCaches::new(max_size));
        self
    }

    /// Attach a pre-resolved Axis B `TokenBiasMap` to this scheduler (B8).
    ///
    /// The bias is cached for the scheduler's lifetime and applied to every
    /// queued sequence's [`SamplingConfig`] at enqueue time (see the merge in
    /// [`Self::enqueue_request`]). An empty map is a zero-overhead no-op on
    /// the hot sampling path — [`sample_token_optimized`] still short-circuits
    /// via the existing `config.token_bias.is_empty()` branch.
    ///
    /// **Phase 1 limitation**: one policy per batch (scheduler-wide).
    /// Per-sequence overrides via request-body `lang_bias` are reserved for
    /// the B12 follow-up outside this Epic.
    pub fn with_token_bias(mut self, bias: TokenBiasMap) -> Self {
        self.token_bias = bias;
        self
    }

    /// Returns a reference to the cached token-bias map (for tests).
    pub fn token_bias(&self) -> &TokenBiasMap {
        &self.token_bias
    }

    /// Attach the server-wide thinking-token budget and resolved
    /// `<think>` / `</think>` token ids.
    ///
    /// `token_ids == None` means the model is non-thinking; the budget is
    /// then silently ignored for every sequence. Callers resolve the token
    /// ids once via
    /// [`crate::server::thinking_budget::resolve_thinking_token_ids`] after
    /// the tokenizer is loaded.
    pub fn with_reasoning_budget(
        mut self,
        budget: Option<ThinkingBudget>,
        token_ids: Option<ThinkingTokenIds>,
    ) -> Self {
        self.reasoning_budget = budget;
        self.thinking_token_ids = token_ids;
        self
    }

    /// Attach the shared prompt-prefix KV cache store
    ///
    /// When `Some(..)`, the scheduler:
    /// * Looks up either a longest-prefix KV match or an exact-prefix
    ///   recurrent-state snapshot on each new request, then adopts/restores on
    ///   hit to skip re-prefill of the shared prefix.
    /// * Donates the sequence's full KV cache or model-owned snapshot back to
    ///   the store on a healthy finish (normal stop / length / cancelled
    ///   without error).
    /// * Never donates back on OOM, transition errors, or
    ///   `Finished(FinishReason::Error(..))`.
    ///
    /// When `None` every hot path short-circuits on the `is_some()` check
    /// before any store access so the bit-exact baseline is preserved.
    pub fn with_prompt_cache(mut self, store: Option<Arc<PromptCacheStore>>) -> Self {
        self.prompt_cache = store;
        self
    }

    /// Whether the installed prompt-cache store is currently accepting
    /// lookups and inserts (scheduler-level gate).
    #[inline]
    fn prompt_cache_active(&self) -> bool {
        self.prompt_cache
            .as_ref()
            .map(|s| s.is_enabled())
            .unwrap_or(false)
    }

    /// Build a [`PromptCacheKey`] bound to the per-request metadata the
    /// scheduler captured at enqueue time. Returns `None` when the request
    /// carried no [`PromptCacheRequestContext`] (e.g. non-chat endpoints).
    fn compose_prompt_cache_key<'a>(
        ctx: &'a PromptCacheRequestContext,
        tokens: &'a [i32],
    ) -> PromptCacheKey<'a> {
        // The digest is computed over the request's resolved image/audio bytes
        // in the route layer (#124 step b). For text-only requests it is
        // `MultimodalDigest::empty()`, so the key is byte-identical to the
        // pre-#124 path. Today multimodal requests still bypass adopt/donate at
        // the `is_multimodal` gate, so a non-empty digest only starts mattering
        // once that gate is lifted (#124 step c); folding it in now keeps the
        // bucket safe from a text↔image collision the moment sharing turns on.
        PromptCacheKey::new_full(
            ctx.model_id.as_str(),
            ctx.lora_id.as_deref(),
            ctx.template_sig.as_str(),
            Some(ctx.session_key.as_str()),
            ctx.mm_digest,
            tokens,
        )
    }

    /// Attempt to adopt a cached prefix for a freshly tokenized request,
    /// returning the adopted `SequenceId` together with the matched-prefix
    /// length on success.
    ///
    /// The caller invokes this **before** [`Self::allocate_sequence_state`]
    /// so the adopted id becomes the sequence's canonical id and no
    /// seq_id rebinding dance is required. On any miss path the caller
    /// proceeds with a fresh allocation under a brand-new id.
    ///
    /// Gating (all of these yield `None`, which maps to a cold prefill):
    /// * feature disabled at config time,
    /// * request carried no [`PromptCacheRequestContext`] (non-chat endpoint),
    /// * store miss / match shorter than `min_prefix_tokens`,
    /// * race with another worker that already consumed the entry,
    /// * empty detached set (e.g. stored against an aborted seq),
    /// * backend mismatch (a `Dense` entry under the paged decode backend, or
    ///   a `Paged` entry under the dense backend — the entry's KV shape cannot
    ///   be installed into the active pool),
    /// * `require_whole_entry` is set (multimodal sharing) and the match is
    ///   shorter than the full stored entry (see below),
    /// * [`CachePool::adopt`] / [`CachePool::adopt_paged`] error (capacity,
    ///   layout mismatch, …).
    ///
    /// `require_whole_entry` is set for multimodal requests (#124 step c). It
    /// forces a whole-entry match so the prefilled suffix is guaranteed to be
    /// the newly-appended text turn: every image/audio placeholder token sits
    /// inside the matched prefix (a different media payload lands in a
    /// different digest bucket), so the suffix can safely run through the
    /// token path. Text-only requests pass `false` and keep accepting partial
    /// (APC block-aligned) matches.
    ///
    /// Both dense and paged entries are adopted in-place: dense via
    /// [`CachePool::adopt`], paged via [`CachePool::adopt_paged`] (which shares
    /// the cached prefix's refcounted pool blocks so the prefix is never
    /// re-prefilled — #121 sub-step b).
    fn try_adopt_cached_prefix(
        &mut self,
        ctx: &PromptCacheRequestContext,
        tokens: &[i32],
        require_whole_entry: bool,
    ) -> Option<(SequenceId, usize)> {
        if !self.prompt_cache_active() {
            return None;
        }
        // Return to the pool any paged pins a prior store op queued (an insert
        // eviction, or a previous lookup's TTL / drained-shell sweep). The
        // lookup below may sweep more; those drain on the next store touch.
        self.drain_store_paged_releases();

        let store = self.prompt_cache.as_ref()?.clone();
        let key = Self::compose_prompt_cache_key(ctx, tokens);
        if self.model.supports_snapshot_reuse()
            && let Some((snapshot_entry, matched_len)) = store.lookup_snapshot_prefix(&key, tokens)
        {
            let seq_id = match self.allocate_sequence_state() {
                Ok(id) => id,
                Err(err) => {
                    tracing::warn!("Cache pool allocation failed during snapshot restore: {err}");
                    return None;
                }
            };
            let restore = snapshot_entry
                .with_snapshot(|snapshot| self.model.restore_sequence_state(seq_id, snapshot));
            match restore {
                Ok(()) => {
                    tracing::debug!(
                        seq_id = %seq_id,
                        matched = matched_len,
                        total = tokens.len(),
                        "prompt-cache snapshot hit: restored {matched_len}/{} tokens",
                        tokens.len()
                    );
                    self.batch_observability
                        .record_prompt_cache_hit(matched_len);
                    self.batch_metrics
                        .record_prompt_cache_snapshot_hit(matched_len);
                    if let Some(ref store) = self.prompt_cache {
                        self.batch_metrics
                            .update_prompt_cache_gauges(store.bytes(), store.len());
                    }
                    return Some((seq_id, matched_len));
                }
                Err(err) => {
                    tracing::warn!(
                        seq_id = %seq_id,
                        "prompt-cache snapshot restore failed ({err}); falling back to cold prefill"
                    );
                    self.release_sequence_caches(seq_id);
                    return None;
                }
            }
        }
        let (entry, matched_len) = store.lookup_longest_prefix(&key, tokens)?;
        // #124 step c: multimodal sharing requires the matched prefix to cover
        // the ENTIRE stored entry. A partial (e.g. APC block-clamped) match
        // could leave image/audio placeholder tokens in the suffix, which the
        // token-path suffix prefill would mis-handle. Decline here (falling
        // back to a cold prefill) before consuming anything; the entry stays
        // available for a later exact match.
        if require_whole_entry && matched_len < entry.tokens.len() {
            return None;
        }
        // Length the adopted cache actually covers. The dense path truncates
        // to exactly `matched_len`; the paged paths floor to the pool block
        // boundary (#225), so they report their own value.
        let mut adopted_len = matched_len;

        // #227: pool-backed paged entries adopt by CLONE, leaving the stored
        // entry intact for concurrent same-prefix siblings and deeper future
        // matches. The one-shot take below destroyed the entry on first use;
        // combined with the #225 trim, a short partial match (even the
        // chat-template preamble) could gut a multi-thousand-token entry.
        enum PagedCloneOutcome {
            /// Adoptable clone built; the source entry stays in the store.
            Cloned(Box<DetachedPagedCacheSet>, usize),
            /// Cold prefill, entry untouched (below minimum, pin failure,
            /// or cross-backend).
            Decline(String),
            /// Dense entry, or a clone-ineligible paged shape (dense-compat
            /// handles, Turbo4, sliding-window): the consuming take path
            /// below can still adopt those.
            TakePath,
        }
        let backend_is_paged = matches!(self.decode_storage_backend, DecodeStorageBackend::Paged);
        let min_prefix = store.min_prefix_tokens().max(1);
        let cache_pool = &mut self.cache_pool;
        // `with_detached` itself returns `None` for a drained shell; the take
        // below then also yields `None` (cold prefill).
        let clone_attempt: Option<PagedCloneOutcome> = entry.with_detached(|set| match set {
            DetachedKvSet::Paged(paged) if backend_is_paged && paged.clone_eligible() => {
                let block_size = paged.layout().block_size.max(1);
                // Floor BOTH the partial and the whole-entry match to the
                // pool block boundary: a donated entry's length
                // (prompt + generated tokens) is almost never block-aligned,
                // and the clone shares whole blocks only. The caller
                // re-prefills everything past the adopted length, which
                // re-covers the dropped partial tail.
                let adoptable = (matched_len.min(paged.seq_len()) / block_size) * block_size;
                if adoptable < min_prefix {
                    return PagedCloneOutcome::Decline(format!(
                        "block-floored match {adoptable} below the minimum prefix {min_prefix}"
                    ));
                }
                match cache_pool.clone_detached_paged_prefix(paged, adoptable) {
                    Ok(clone) => PagedCloneOutcome::Cloned(Box::new(clone), adoptable),
                    Err(err) => PagedCloneOutcome::Decline(err),
                }
            }
            // Paged entry under a dense decode backend: cross-backend
            // adoption is invalid; decline without touching the entry
            // (the old path took the set just to release it).
            DetachedKvSet::Paged(_) if !backend_is_paged => {
                PagedCloneOutcome::Decline("paged entry under a dense decode backend".into())
            }
            // Clone-ineligible paged shapes and dense entries.
            DetachedKvSet::Paged(_) | DetachedKvSet::Dense(_) => PagedCloneOutcome::TakePath,
        });
        match clone_attempt {
            Some(PagedCloneOutcome::Cloned(clone, adoptable)) => {
                adopted_len = adoptable;
                let adopt_result = self
                    .cache_pool
                    .adopt_paged(&self.model as &dyn LanguageModel, *clone);
                return self.finish_prompt_cache_adopt(adopt_result, adopted_len, tokens.len());
            }
            Some(PagedCloneOutcome::Decline(reason)) => {
                tracing::debug!(
                    "prompt-cache adopt: paged clone declined ({reason}); falling back to cold prefill (entry preserved)"
                );
                return None;
            }
            Some(PagedCloneOutcome::TakePath) | None => {}
        }

        // Dense entries (and clone-ineligible paged shapes) keep the legacy
        // one-shot consume: their adoption genuinely moves buffers.
        // `take_detached` returns `None` if a racing lookup already consumed
        // this entry; the miss path is safe (fresh prefill).
        let detached = entry.take_detached()?;
        if detached.is_empty() {
            // A paged set drained on this path would leak its block pins via
            // `Drop`; release them explicitly so the pool budget stays honest.
            self.release_unused_detached(detached);
            return None;
        }

        // Reject cross-backend adoption: the active decode backend determines
        // the KV shape the model worker can install. Adopting the wrong variant
        // would corrupt the sequence, so fall through to a cold prefill (and
        // release any paged pins we took).
        let backend_mismatch = matches!(
            (&detached, self.decode_storage_backend),
            (DetachedKvSet::Dense(_), DecodeStorageBackend::Paged)
                | (DetachedKvSet::Paged(_), DecodeStorageBackend::Dense)
        );
        if backend_mismatch {
            self.release_unused_detached(detached);
            return None;
        }

        let adopt_result = match detached {
            DetachedKvSet::Dense(mut dense) => {
                // APC block-level partial adoption. When APC clamps
                // `matched_len` to a block boundary shorter than the cached
                // entry's full token length, the request diverged from the
                // cached prefix at the next block. Truncate the detached KV
                // state to exactly `matched_len` so the adopted cache covers
                // only the consistent prefix; the prefill loop then re-prefills
                // the divergent tail. When `matched_len == seq_len` this branch
                // is skipped, preserving the bit-exact full-prefix path.
                let detached_seq_len = dense.seq_len();
                if matched_len < detached_seq_len as usize {
                    let target = matched_len as i32;
                    if let Err(err) = dense.truncate_to(target) {
                        tracing::warn!(
                            "prompt-cache adopt: APC partial truncate to {target} failed ({err}); falling back to cold prefill"
                        );
                        return None;
                    }
                    tracing::debug!(
                        from = detached_seq_len,
                        to = target,
                        "prompt-cache adopt: APC partial adoption truncated detached cache to block boundary"
                    );
                }
                self.cache_pool
                    .adopt(&self.model as &dyn LanguageModel, dense)
            }
            DetachedKvSet::Paged(mut paged) => {
                // Paged partial prefix adoption (#225). An APC block-clamped
                // lookup (or a request that diverges inside the stored entry)
                // matches only `matched_len` of the set. Floor that to the
                // POOL block boundary: no partially filled tail block survives
                // the trim, so the suffix re-prefill starts on a fresh block
                // and never needs copy-on-write against a shared tail. A
                // whole-entry match skips the trim and stays bit-exact with
                // the pre-#225 path.
                let paged_seq_len = paged.seq_len();
                let block_size = paged.layout().block_size.max(1);
                let adoptable = if matched_len < paged_seq_len {
                    (matched_len / block_size) * block_size
                } else {
                    paged_seq_len
                };
                let min_prefix = store.min_prefix_tokens().max(1);
                if adoptable < min_prefix {
                    tracing::debug!(
                        from = paged_seq_len,
                        to = adoptable,
                        "prompt-cache adopt: block-floored paged match below the minimum prefix; releasing and falling back to cold prefill"
                    );
                    self.cache_pool.release_detached_paged(paged);
                    return None;
                }
                if adoptable < paged_seq_len {
                    if let Err(err) = self
                        .cache_pool
                        .trim_detached_paged_to(&mut paged, adoptable)
                    {
                        tracing::warn!(
                            "prompt-cache adopt: paged partial trim to {adoptable} failed ({err}); falling back to cold prefill"
                        );
                        self.cache_pool.release_detached_paged(paged);
                        return None;
                    }
                    tracing::debug!(
                        from = paged_seq_len,
                        to = adoptable,
                        "prompt-cache adopt: paged partial adoption trimmed detached set to the pool block boundary"
                    );
                    adopted_len = adoptable;
                }
                self.cache_pool
                    .adopt_paged(&self.model as &dyn LanguageModel, paged)
            }
        };

        self.finish_prompt_cache_adopt(adopt_result, adopted_len, tokens.len())
    }

    /// Shared tail of [`Self::try_adopt_cached_prefix`]: record hit metrics
    /// and gauges on success, log and fall back to a cold prefill on failure.
    fn finish_prompt_cache_adopt(
        &mut self,
        adopt_result: Result<SequenceId, String>,
        adopted_len: usize,
        total_tokens: usize,
    ) -> Option<(SequenceId, usize)> {
        match adopt_result {
            Ok(adopted_id) => {
                tracing::debug!(
                    seq_id = %adopted_id,
                    matched = adopted_len,
                    total = total_tokens,
                    "prompt-cache hit: adopted {adopted_len}/{total_tokens} tokens"
                );
                self.batch_observability
                    .record_prompt_cache_hit(adopted_len);
                // also increment BatchMetrics Prometheus counters.
                self.batch_metrics.record_prompt_cache_hit(adopted_len);
                // Update byte/entry gauges so /metrics reflects current state.
                if let Some(ref store) = self.prompt_cache {
                    self.batch_metrics
                        .update_prompt_cache_gauges(store.bytes(), store.len());
                }
                Some((adopted_id, adopted_len))
            }
            Err(err) => {
                // `adopt_paged` already releases paged pins on its error path;
                // `adopt` (dense) simply drops the buffers. Nothing to reclaim.
                tracing::debug!("prompt-cache adopt failed ({err}); falling back to cold prefill");
                None
            }
        }
    }

    /// Release a detached set the adopt path decided not to use.
    ///
    /// A dense set just drops its MLX buffers. A paged set additionally owns
    /// refcount pins on physical pool blocks, which [`Drop`] cannot release on
    /// its own (it has no pool handle) — so route it through
    /// [`CachePool::release_detached_paged`] to return the pins and keep the
    /// block budget accurate.
    fn release_unused_detached(&mut self, detached: DetachedKvSet) {
        match detached {
            DetachedKvSet::Dense(_) => {}
            DetachedKvSet::Paged(paged) => {
                self.cache_pool.release_detached_paged(paged);
            }
        }
    }

    /// Return to the pool any paged block pins the prompt-cache store queued
    /// for release. The store evicts (LRU / TTL) and declines (oversized)
    /// paged entries but cannot return their pool pins — it holds no
    /// `CachePool` handle — so it stashes them. The scheduler owns the pool,
    /// so it drains the queue here and routes each set through
    /// [`CachePool::release_detached_paged`]. Called from the store-touching
    /// paths so pins are reclaimed promptly during serving; a cheap no-op when
    /// the queue is empty (#122 sub-step a).
    fn drain_store_paged_releases(&mut self) {
        let store = match self.prompt_cache.as_ref() {
            Some(s) if s.has_pending_paged_releases() => s.clone(),
            _ => return,
        };
        for paged in store.drain_pending_paged_releases() {
            self.cache_pool.release_detached_paged(paged);
        }
    }

    /// Donate a finished sequence's KV cache back to the store so future
    /// requests sharing a prefix can adopt it.
    ///
    /// The caller must invoke this **before** calling
    /// [`Self::release_sequence_caches`] — once release runs the underlying
    /// tensors are gone. Safe to call unconditionally; all the gating checks
    /// (feature enabled, healthy finish, context present, detachable backend)
    /// live inside this method so the caller can keep its hot-path code
    /// simple.
    ///
    /// Both dense and paged sequences are donated: dense via
    /// [`CachePool::detach`] (→ [`DetachedKvSet::Dense`]) and paged via
    /// [`CachePool::detach_paged`] (→ [`DetachedKvSet::Paged`], which pins the
    /// prefix's physical pool blocks so a later `adopt_paged` can share them).
    /// `ModelOwned` sequences carry no detachable cross-request KV; families
    /// that opt into snapshot reuse donate a copied model-owned snapshot, while
    /// the rest are skipped.
    fn donate_finished_sequence_cache(
        &mut self,
        seq_id: SequenceId,
        prompt_tokens: &[i32],
        generated_tokens: &[i32],
        healthy_finish: bool,
    ) {
        if !healthy_finish {
            return;
        }
        if !self.prompt_cache_active() {
            return;
        }
        // Remove the context regardless of whether the donate-back succeeds
        // so the map doesn't grow unbounded across sequences that never
        // qualified for a donate-back.
        let ctx = match self.prompt_cache_seq_ctx.remove(&seq_id) {
            Some(c) => c,
            None => return,
        };

        // Tokens stored against both KV entries and recurrent snapshots are
        // the full prompt + generated tail, so the next turn can restore the
        // exact previous conversation prefix and prefill only the appended
        // user turn.
        let mut tokens = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
        tokens.extend_from_slice(prompt_tokens);
        tokens.extend_from_slice(generated_tokens);

        // Families with model-owned recurrent or linear-attention state opt
        // into exact-prefix snapshots explicitly. Check this capability before
        // consulting the allocated storage backend: under the paged decode
        // override these families may still carry a shadow `PagedKvCache`
        // placeholder even though the real state lives in
        // `ModelOwnedSequenceState` and cannot be detached as KV blocks.
        if self.model.supports_snapshot_reuse() {
            let store = match self.prompt_cache.as_ref() {
                Some(s) => s.clone(),
                None => return,
            };
            if tokens.len() < store.min_prefix_tokens() {
                return;
            }
            let snapshot = match self.model.snapshot_sequence_state(seq_id, tokens.len()) {
                Some(s) if !s.is_empty() => s,
                Some(_) => {
                    tracing::debug!(
                        seq_id = %seq_id,
                        token_len = tokens.len(),
                        "prompt-cache snapshot donate skipped: captured snapshot was empty"
                    );
                    return;
                }
                None => {
                    tracing::debug!(
                        seq_id = %seq_id,
                        token_len = tokens.len(),
                        "prompt-cache snapshot donate skipped: no model-owned state for sequence"
                    );
                    return;
                }
            };
            let entry = ModelSnapshotEntry::new(tokens, snapshot);
            let key_tokens = entry.tokens.clone();
            let key = Self::compose_prompt_cache_key(&ctx, &key_tokens);
            match store.insert_snapshot(&key, entry) {
                Ok(()) => {
                    tracing::debug!(
                        seq_id = %seq_id,
                        token_len = key_tokens.len(),
                        bytes = store.stats().snapshot_bytes,
                        "prompt-cache snapshot inserted"
                    );
                    self.batch_observability.record_prompt_cache_insert();
                    self.batch_metrics
                        .update_prompt_cache_gauges(store.bytes(), store.len());
                }
                Err(err) => {
                    tracing::debug!("prompt-cache snapshot insert skipped: {err}");
                    self.batch_observability.record_prompt_cache_insert_reject();
                }
            }
            return;
        }

        let backend = self
            .cache_pool
            .get_mut(seq_id)
            .map(|s| s.backend)
            .unwrap_or(SequenceStateBackend::ModelOwned);

        // Other `ModelOwned` families carry no detachable cross-request KV.
        if backend == SequenceStateBackend::ModelOwned {
            return;
        }

        let store = match self.prompt_cache.as_ref() {
            Some(s) => s.clone(),
            None => return,
        };

        // Detach into the backend-appropriate variant.
        let kv_set: DetachedKvSet = match backend {
            SequenceStateBackend::DenseKvCache => match self.cache_pool.detach(seq_id) {
                Some(d) => DetachedKvSet::Dense(d),
                None => return,
            },
            SequenceStateBackend::PagedKvCache => {
                // `detach_paged` pins every physical prefix block, and those
                // pins can only be returned through `release_detached_paged`
                // (the set's `Drop` cannot). If the store would reject the
                // entry for being shorter than `min_prefix_tokens`, screen the
                // length BEFORE detaching so we never take pins we'd have to
                // immediately release. The dense path needs no such screen — a
                // rejected dense entry just drops its buffers.
                if tokens.len() < store.min_prefix_tokens() {
                    return;
                }
                match self.cache_pool.detach_paged(seq_id) {
                    Some(p) => DetachedKvSet::Paged(p),
                    None => return,
                }
            }
            SequenceStateBackend::ModelOwned => return,
        };

        if kv_set.is_empty() {
            // Nothing to cache: aborted before any prefill completed, or the
            // model never populated the KV state. Release any paged pins we
            // took so the pool budget stays honest.
            self.release_unused_detached(kv_set);
            return;
        }

        // The `CacheEntry` takes ownership of `tokens` and the key borrows
        // from the same buffer. Build the entry first, then form the key
        // against `entry.tokens` so both reference the same contiguous
        // allocation without copying the vector.
        let entry = CacheEntry::new(tokens, kv_set);
        let key_tokens = entry.tokens.clone();
        let key = Self::compose_prompt_cache_key(&ctx, &key_tokens);
        match store.insert(&key, entry) {
            Ok(()) => {
                self.batch_observability.record_prompt_cache_insert();
                // refresh byte/entry gauges after a successful insert.
                self.batch_metrics
                    .update_prompt_cache_gauges(store.bytes(), store.len());
            }
            Err(err) => {
                // Oversized / disabled / prefix-too-short — `insert` declines
                // the entry. For dense that frees the buffers; for a paged entry
                // the store stashes its block pins on its pending-release queue
                // (it has no `CachePool` handle), which the
                // `drain_store_paged_releases()` below returns to the pool
                // (#122 sub-step a).
                tracing::debug!(
                    seq_id = %seq_id,
                    "prompt-cache donate-back skipped: {err:?}"
                );
                self.batch_observability.record_prompt_cache_insert_reject();
            }
        }
        // Return to the pool any paged pins this insert's eviction / rejection
        // paths queued: byte/entry-budget `enforce_caps` (LRU), idempotent
        // replacement removal, or an oversized / disabled decline.
        self.drain_store_paged_releases();
    }

    /// Apply thinking-budget enforcement to a freshly sampled
    /// token for a single sequence.
    ///
    /// Returns the final token id to commit to the sequence (either the
    /// sampled value, or the forced `</think>` id when the budget fires).
    /// Caller is responsible for using the returned id for the remainder of
    /// the decode step (EOS check, streaming emission, history update).
    ///
    /// The state advances with the final id so subsequent steps see the
    /// post-close phase.
    ///
    /// # Notes on bypass of sampling knobs
    ///
    /// When the budget fires the forced id bypasses the sampler's logits
    /// pipeline for that step. No retroactive re-penalization happens because
    /// - `token_history` is only appended once per step (caller uses the
    ///   returned id),
    /// - `merged_eos` checks use the returned id,
    /// - the next step samples fresh logits from the underlying model.
    fn apply_thinking_budget(seq_thinking: &mut ThinkingState, sampled: i32) -> i32 {
        if seq_thinking.is_disabled() {
            return sampled;
        }
        let final_id = match seq_thinking.decide_override(sampled) {
            ThinkingDecision::NoOverride => sampled,
            ThinkingDecision::ForceClose(close_id) => close_id,
        };
        seq_thinking.observe(final_id);
        final_id
    }

    /// apply the structured-output mask (if any) to logits before
    /// sampling.
    ///
    /// Returns either the masked logits or `Err(_)` describing why the
    /// matcher refused to advance. The scheduler propagates the error as
    /// `FinishReason::Error(...)` so the SSE stream terminates cleanly
    /// instead of emitting non-conforming output.
    fn apply_structured_mask(
        constraint: &std::sync::Arc<
            std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>,
        >,
        logits: UniquePtr<mlxcel_core::MlxArray>,
        vocab_size_hint: usize,
    ) -> Result<UniquePtr<mlxcel_core::MlxArray>, String> {
        let mut guard = constraint
            .lock()
            .map_err(|e| format!("structured-output constraint poisoned: {e}"))?;
        let masked = crate::server::structured::apply_structured_mask_to_logits(
            &mut guard,
            &logits,
            vocab_size_hint,
        )
        .map_err(|e| e.to_string())?;
        Ok(masked)
    }

    /// advance the matcher state by the just-sampled token.
    ///
    /// Returns `Ok(())` on success, `Err(msg)` when `consume_token` fails or
    /// the matcher is in an error state. The caller transitions the sequence
    /// to `Finished(Error(msg))` on error.
    fn consume_structured_token(
        constraint: &std::sync::Arc<
            std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>,
        >,
        token: i32,
    ) -> Result<(), String> {
        let mut guard = constraint
            .lock()
            .map_err(|e| format!("structured-output constraint poisoned: {e}"))?;
        guard.consume_token(token).map_err(|e| e.to_string())
    }

    /// send a clean SSE error event and transition the sequence
    /// to `Finished(Error(msg))`. Used by the structured-output path to
    /// abort cleanly when the matcher refuses to advance.
    fn abort_sequence_with_error(seq: Option<&mut SequenceInfo>, prefix: &str, msg: &str) {
        if let Some(seq) = seq {
            let _ = seq
                .response_tx
                .send(GenerateEvent::Error(format!("{prefix}: {msg}")));
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Error(
                    msg.to_string(),
                )))
            {
                tracing::error!("State transition error: {err}");
            }
        }
    }

    /// Effective thinking-budget for a single sequence.
    ///
    /// Combines the server default with any per-request override attached to
    /// the request's `ServerGenerateOptions`. Returns a [`ThinkingState`]
    /// ready to be stored on `SequenceInfo`.
    ///
    /// `enter_block_on_start` is passed through to the [`ThinkingState`].
    /// Chat endpoints set `true` (the Qwen3 chat template primes `<think>\n`);
    /// raw text endpoints (`/v1/completions`, `/completion`) set `false` so
    /// the model must emit `<think>` before any in-block counting begins.
    fn build_thinking_state(
        &self,
        override_: ReasoningBudgetOverride,
        enter_block_on_start: bool,
    ) -> ThinkingState {
        // No thinking tokens -> always disabled regardless of config.
        let Some(token_ids) = self.thinking_token_ids else {
            return ThinkingState::disabled();
        };
        let effective = match override_ {
            ReasoningBudgetOverride::InheritServerDefault => self.reasoning_budget,
            ReasoningBudgetOverride::Explicit(v) => v,
        };
        ThinkingState::new(Some(token_ids), effective, enter_block_on_start)
    }

    /// Run the scheduler loop until shutdown or channel close.
    pub fn run(&mut self) {
        install_thread_local_default_stream(self.generation_stream.as_ref());

        loop {
            // 1. Non-blocking drain of all pending requests
            self.drain_incoming_requests();

            if self.shutdown_requested {
                break;
            }

            self.publish_metrics();

            // 2. Decide what to do this tick
            let action = self.decide_action();

            // 3. Execute
            match action {
                BatchSchedulerAction::Prefill(seq_id) => {
                    // Admission / chunked-prefill interleave / preemption all
                    // change batch membership; tear down any prebuilt lookahead
                    // (trimming its speculative KV) before the prefill runs so
                    // the next decode rebuilds against the new batch (#632
                    // invalidation).
                    self.discard_lookahead();
                    // Use batched prefill when max_batch_prefill > 1 and at
                    // least 2 requests are waiting, otherwise take the regular
                    // single-request path so there is zero overhead for the
                    // common case. #715: also require the head-of-queue prompt
                    // to be short enough to join a padded batch within the
                    // token budget; a head too long to batch takes the
                    // chunk-aware single-sequence path (which keeps the
                    // attention mask chunked to `[chunk, L]` instead of the
                    // unchunked `[L, L]` a single-row batched forward would
                    // build).
                    if self.max_batch_prefill > 1
                        && self.prefill_queue.len() >= 2
                        && self.chunked_prefill_seq.is_none()
                        && self.batched_prefill_admits_head()
                    {
                        self.execute_batched_prefill();
                    } else {
                        self.execute_prefill(seq_id);
                    }
                    self.publish_metrics();
                }
                BatchSchedulerAction::Decode(ids) => {
                    self.execute_decode_step(&ids);
                }
                BatchSchedulerAction::Idle => match self.request_rx.recv() {
                    Ok(req) => {
                        if self.handle_incoming(req) {
                            break;
                        }
                        self.publish_metrics();
                    }
                    Err(_) => {
                        tracing::info!("Request channel closed, scheduler exiting");
                        break;
                    }
                },
            }

            // 4. Clean up completed sequences
            self.finalize_completed();
        }
    }

    fn publish_metrics(&self) {
        let active = self.active_batch.len();
        let queued = self.prefill_queue.len();
        let paged_stats = self.cache_pool.paged_stats();
        let paged_block_size = self.cache_pool.paged_block_size().unwrap_or(0);
        self.batch_metrics.set_active_count(active);
        self.batch_metrics.set_queue_depth(queued);
        self.batch_observability.update_gauges(
            active,
            queued,
            self.cache_pool.active_count(),
            self.cache_pool.memory_usage_bytes() as u64,
            paged_block_size,
            paged_stats,
            // #122 c: surface the configured block-budget cap (0 = unbounded)
            // so `/v1/cache/stats` and `/metrics` can report admission headroom.
            self.cache_pool.paged_block_budget().unwrap_or(0) as u64,
        );
    }

    fn allocate_sequence_state(&mut self) -> Result<SequenceId, String> {
        let layout_override = self.sequence_state_layout_override();
        let seq_id = self
            .cache_pool
            .allocate_with_layout(&self.model, layout_override)?;
        // apply the configured KV cache mode (with
        // Boundary-V policy for Turbo4 modes) to the freshly allocated
        // per-layer caches. `model.make_caches()` always returns Fp16
        // caches; this step upgrades them to the requested mode while
        // keeping boundary layers at FP16 quality. No-op when the
        // configured mode is `Fp16`.
        self.apply_kv_cache_mode_to(seq_id);
        self.model.prepare_sequence_state(seq_id);
        Ok(seq_id)
    }

    /// Apply the configured `kv_cache_mode` to every per-layer cache of
    /// `seq_id` (with the Boundary-V upgrade for Turbo4 modes).
    ///
    /// No-op for `KVCacheMode::Fp16` — the cache pool already returns
    /// FP16 caches from `model.make_caches()`. For Turbo4 modes the
    /// per-layer mode is resolved via
    /// [`mlxcel_core::cache::turbo::resolve_layer_modes`] so boundary
    /// layers stay FP16 for quality.
    ///
    /// when [`BatchKvQuantConfig::is_enabled`] returns
    /// `true`, the per-layer mode table is sourced from
    /// [`BatchKvQuantConfig::resolve_layer_modes`] instead — that table
    /// honours the `skip_last_layer` policy, which is distinct from
    /// (and composes with) the existing Boundary-V mechanism.
    ///
    /// Used by: [`Self::allocate_sequence_state`].
    fn apply_kv_cache_mode_to(&mut self, seq_id: SequenceId) {
        let Some(caches) = self.cache_pool.get_caches_mut(seq_id) else {
            return;
        };
        if caches.is_empty() {
            // Model-owned / paged sequences without dense placeholder
            // caches — nothing to upgrade. The model's own decode path
            // is responsible for honoring the configured mode.
            return;
        }
        let n_layers = caches.len();

        // when the batched KV quant config is active, it
        // takes precedence over the legacy `kv_cache_mode` path. The
        // resolved table already encodes the per-layer mode (with the
        // last-layer skip applied), so apply it directly.
        if self.batch_kv_quant.is_enabled() {
            let layer_modes = self.batch_kv_quant.resolve_layer_modes(n_layers);
            for (cache, mode) in caches.iter_mut().zip(layer_modes) {
                cache.mode = mode;
            }
            return;
        }

        // Legacy path: nominal mode + Boundary-V
        // protection only.
        let nominal = self.kv_cache_mode;
        if nominal == KVCacheMode::Fp16 {
            return;
        }
        let requested = mlxcel_core::cache::turbo::boundary_v_layers_from_env();
        let layer_modes =
            mlxcel_core::cache::turbo::resolve_layer_modes(nominal, n_layers, requested);
        for (cache, mode) in caches.iter_mut().zip(layer_modes) {
            cache.mode = mode;
        }
    }

    /// Prepare Turbo4Delegated cache state before a sequence enters decode.
    ///
    /// `finish_prefill` has already emitted the first token; when that token
    /// also satisfies `max_tokens` or EOS, the sequence never decodes and this
    /// helper is intentionally not called.
    fn prepare_turbo4_delegated_for_sequence_decode(&mut self, seq_id: SequenceId) {
        let Some(caches) = self.cache_pool.get_caches_mut(seq_id) else {
            return;
        };
        for cache in caches {
            cache.prepare_turbo4_delegated_for_decode();
        }
    }

    /// Enforce the `--max-kv-size` cap on a sequence's KV caches.
    ///
    /// Trims the oldest `live_len(cache) - max_kv_size` tokens from every
    /// plain `KVCache` layer whose live window exceeds the configured
    /// bound. Turbo-mode caches return `0` from `KVCache::trim_front`
    /// (safe no-op — see [`KVCache::trim_front`] for the per-mode
    /// support matrix). Sliding-window models manage their own internal
    /// `RotatingKVCache` and are never stored in the pool's
    /// `Vec<KVCache>`, so they are unaffected.
    ///
    /// ** H1**: `max_kv_size` has already been validated to fit
    /// in `i32` by [`crate::server::cli_input::resolve_max_kv_size`], so
    /// the `i32::try_from` here is a defensive belt-and-suspenders fallback
    /// — it returns silently rather than panicking, because the validation
    /// at startup ensures we never reach the failure branch in practice.
    /// We compare against `cache.live_len()` (not `cache.offset`) so the
    /// cap is enforced on the **live window** — the monotonic `offset`
    /// keeps growing past the cap by design.
    ///
    /// ** H2**: called from every cache-mutating path:
    /// [`Self::execute_full_prefill`], [`Self::start_chunked_prefill`],
    /// [`Self::continue_chunked_prefill`], [`Self::decode_single_step`],
    /// and [`Self::execute_batched_decode`].
    fn enforce_max_kv_size_for(&mut self, seq_id: SequenceId) {
        let Some(max) = self.max_kv_size else {
            return;
        };
        // Defensive: even though `resolve_max_kv_size` already clamps this
        // to `i32::MAX` at startup, a future caller that bypasses the CLI
        // validation could still construct an out-of-range scheduler. Use
        // `checked` arithmetic so the worst case is a no-op trim rather
        // than a wraparound that corrupts every cache.
        let Ok(max_i32) = i32::try_from(max) else {
            tracing::error!(
                "--max-kv-size value {max} does not fit in i32; skipping trim. \
                 This should have been rejected by ServerStartupInput::into_startup_config — \
                 please file a bug if you see this in production."
            );
            return;
        };
        let Some(caches) = self.cache_pool.get_caches_mut(seq_id) else {
            return;
        };
        for cache in caches {
            // `live_len() = offset - live_start`. We trim against the live
            // window length (what attention sees), not the monotonic
            // `offset` (which keeps growing by design — RoPE invariant in `KVCache::trim_front`).
            let live_len = cache.live_len();
            // `checked_sub` so a future arithmetic regression cannot
            // silently wrap into a negative trim depth that produces a
            // 4-billion-element slice and crashes Metal.
            if let Some(excess) = live_len.checked_sub(max_i32)
                && excess > 0
            {
                cache.trim_front(excess);
            }
        }
    }

    fn sequence_state_layout_override(&self) -> Option<SequenceStateLayout> {
        if self.decode_storage_backend != DecodeStorageBackend::Paged {
            return None;
        }

        let num_layers = self.model.num_layers();
        // prefer the batched KV quant config when active so
        // its `base_mode()` drives paged-layout selection (Turbo-aware
        // when scheme is TurboQuant, otherwise the legacy uniform path).
        let effective_mode = if self.batch_kv_quant.is_enabled() {
            self.batch_kv_quant.base_mode()
        } else {
            self.kv_cache_mode
        };
        // when a Turbo4 cache mode is configured, build a
        // packed-aware paged layout so per-page
        // sidecar accounting and detach/adopt round-trip work correctly.
        // Fp16/Int8 keep the historical `PagedKvLayout::uniform` path —
        // bit-identical to earlier.
        let paged_layout = if Self::is_turbo_mode(effective_mode) {
            // The actual per-token packed sidecar size depends on the
            // model's V head_dim, which is not known to the scheduler
            // at construction time (the dense `KVCache::update_turbo4_*`
            // path lazily allocates the right shape on first write).
            // We charge a per-block budget equal to `DEFAULT_PAGED_BLOCK_SIZE`
            // as a placeholder so the layout passes the
            // `bytes % block_size == 0` validation in
            // [`PagedKvLayout::new_with_mode`]; the runtime
            // `turbo_sidecars.nbytes()` reports the true footprint via
            // `CachePool::memory_usage_bytes`.
            let sidecar_bytes_per_block = DEFAULT_PAGED_BLOCK_SIZE;
            PagedKvLayout::uniform_with_mode(
                num_layers,
                DEFAULT_PAGED_BLOCK_SIZE,
                DEFAULT_PAGED_BLOCK_SIZE,
                effective_mode,
                sidecar_bytes_per_block,
            )
            .expect("valid paged Turbo4 decode layout")
        } else {
            // Carry the actual cache mode so the pool-backing gate in
            // `CachePool::allocate_with_layout` (`paged_layout.cache_mode ==
            // Fp16`) pool-backs ONLY genuine Fp16 sequences. Int8 (and
            // Turbo3Asym) keep their dense KV path — memory saving preserved —
            // until the pool gains native quantized storage; these modes carry
            // no per-page sidecars, so the sidecar budget is 0 (uniform_with_mode
            // treats non-Turbo4 modes as sidecar-free).
            PagedKvLayout::uniform_with_mode(
                num_layers,
                DEFAULT_PAGED_BLOCK_SIZE,
                DEFAULT_PAGED_BLOCK_SIZE,
                effective_mode,
                0,
            )
            .expect("valid paged decode layout")
        };
        Some(SequenceStateLayout::paged_kv_cache(paged_layout))
    }

    /// Whether the supplied KV cache mode requires Turbo*-aware paged
    /// layout (per-page sidecar storage on `PagedBlockPool`) and is
    /// **incompatible** with the `--max-kv-size` cap.
    ///
    /// All Turbo modes carry per-token rotation state in their sidecars
    /// (`turbo_params` / `turbo3_params` / `v_packed` / `v_norms` /
    /// `cold_offset`) that `KVCache::trim_front` cannot safely truncate
    /// from the head. H3: `Turbo3Asym` belongs in this set
    /// the 3-bit V sidecars (`v_packed` with 24-bit groups + `v_norms`)
    /// have the same per-token contract as `Turbo4*`. Omitting it from
    /// this match silently allowed `--max-kv-size` + `fp16+turbo3` to ship
    /// without the operator-facing warning that the cap will not be
    /// honoured on the V side.
    ///
    /// Used by: scheduler dispatch for sequence allocation, the
    /// `--max-kv-size` + Turbo combination warning.
    #[inline]
    fn is_turbo_mode(mode: KVCacheMode) -> bool {
        matches!(
            mode,
            KVCacheMode::Turbo4Asym
                | KVCacheMode::Turbo4
                | KVCacheMode::Turbo4Delegated
                | KVCacheMode::Turbo3Asym
        )
    }

    fn sync_sequence_storage(&mut self, seq_id: SequenceId) {
        if let Err(err) = self
            .model
            .sync_sequence_storage(seq_id, &mut self.cache_pool)
        {
            tracing::warn!("Failed to sync paged state for {seq_id}: {err}");
        }
    }

    // ------------------------------------------------------------------
    // Request ingestion
    // ------------------------------------------------------------------

    fn drain_incoming_requests(&mut self) {
        loop {
            match self.request_rx.try_recv() {
                Ok(req) => {
                    if self.handle_incoming(req) {
                        self.shutdown_requested = true;
                        return;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.shutdown_requested = true;
                    return;
                }
            }
        }
    }

    fn handle_incoming(&mut self, req: ModelRequest) -> bool {
        match req {
            ModelRequest::Generate {
                prompt,
                prompt_token_ids,
                options,
                images,
                audio,
                videos,
                response_tx,
                cancelled,
            } => {
                self.enqueue_request(
                    prompt,
                    prompt_token_ids,
                    options,
                    images,
                    audio,
                    videos,
                    response_tx,
                    cancelled,
                );
                false
            }
            ModelRequest::Shutdown => {
                tracing::info!("BatchScheduler received shutdown signal");
                true
            }
        }
    }

    fn enqueue_request(
        &mut self,
        prompt: String,
        prompt_token_ids: Option<Vec<i32>>,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<crate::server::media::ResolvedVideo>,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
    ) {
        // Prefer the ids tokenized on the request-dispatch thread (issue #633);
        // fall back to scheduler-side tokenization when the dispatcher had no
        // pre-tokenizer. `tokenize_prompt_for_generation` is the shared
        // `add_special` convention so both paths are byte-identical.
        let mut prompt_tokens: Vec<i32> = match prompt_token_ids {
            Some(ids) => ids,
            None => match crate::server::model_provider::tokenize_prompt_for_generation(
                &self.tokenizer,
                &prompt,
            ) {
                Ok(ids) => ids,
                Err(err) => {
                    let _ = response_tx
                        .send(GenerateEvent::Error(format!("Tokenization error: {err}")));
                    return;
                }
            },
        };

        // Empty-prompt guard (null/empty-cache safety):
        //
        // A zero-token prompt cannot be prefilled — the forward pass would
        // run with a `[1, 0]` input and the per-sequence KV cache would
        // remain in the `keys is None, offset == 0` state. Admitting such a
        // request into the batch could later crash the scheduler when the
        // cache is used alongside populated caches in `execute_batched_*`.
        // Mirrors the upstream `mlx-lm` `BatchKVCache.extend` null-guard
        // that refuses to pad/concatenate a cache with no tensors. VLM
        // requests may legitimately start with an empty token list (image
        // tokens are injected later by `prepare_request_vlm_embeddings`),
        // so this guard only applies to pure-text requests without images,
        // audio, or videos.
        if prompt_tokens.is_empty() && images.is_empty() && audio.is_empty() && videos.is_empty() {
            let _ = response_tx.send(GenerateEvent::Error(
                "Empty prompt: request has no input tokens to process".to_string(),
            ));
            return;
        }

        let mut sampling = merge_config_stop_tokens(options.sampling.clone(), &self.config_eos);

        // Axis B (B8): attach the scheduler-wide token bias to each sequence's
        // sampling config when no per-request override is present. Empty
        // cached bias = bit-exact baseline (the `is_empty()` short-circuit in
        // `sample_token_optimized` keeps hot-path cost at zero).
        //
        // Phase 1 limitation: one policy per batch. Per-request overrides
        // via `/v1/chat/completions` request body are deferred to B12.
        if !self.token_bias.is_empty() && sampling.token_bias.is_empty() {
            sampling.token_bias = self.token_bias.clone();
        }

        // issue #350: force-suppress the model's reserved multimodal
        // placeholder tokens (audio / image / video span markers) on every
        // sequence's output logits. Applied after the lang-bias merge and
        // unconditionally, so suppression always wins over any per-request
        // bias and a placeholder id can never become the sampled argmax. A
        // no-op (and zero alloc) for non-multimodal models whose suppressed
        // set is empty.
        if !self.model_output_suppressed.is_empty() {
            sampling
                .token_bias
                .suppress_tokens(&self.model_output_suppressed);
        }

        let is_multimodal = !images.is_empty() || !audio.is_empty() || !videos.is_empty();

        // Experimental VLM prompt-prefix cache sharing (#124 step c). Off by
        // default; the operator opts in with `--enable-vlm-prefix-cache`. Video
        // payloads are excluded because video frame bytes are not folded into
        // the request's multimodal digest yet, so a video prefix could collide
        // with a different one in the same bucket.
        let vlm_sharing_ok = vlm_prefix_sharing_allowed(
            self.enable_vlm_prefix_cache,
            is_multimodal,
            !videos.is_empty(),
        );

        // For VLM sharing the image/audio placeholder tokens must be expanded
        // BEFORE probing the cache: `prepare_request_vlm_embeddings` rewrites
        // `prompt_tokens` into the post-injection stream the KV cache is built
        // over, and both the cache key (via the request's multimodal digest)
        // and the matched-prefix length are computed against that stream. No
        // sequence id exists yet, so a preparation error just aborts the
        // request with nothing to clean up. `Some(_)` marks "prepared early";
        // the inner value is the optional merged embeddings.
        let prepared_early = if vlm_sharing_ok {
            match prepare_request_vlm_embeddings(
                &self.model,
                &self.tokenizer,
                &prompt,
                &mut prompt_tokens,
                &images,
                &audio,
                &videos,
                Some(self.vision_caches.as_ref()),
            ) {
                Ok(emb) => Some(emb),
                Err(err) => {
                    let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
                    return;
                }
            }
        } else {
            None
        };

        // before allocating a fresh KV-cache slot,
        // probe the prompt-prefix cache for a reusable detached set. On a
        // hit, adopt under a brand-new SequenceId and record how many
        // leading tokens the prefill can skip. On a miss (which includes
        // feature-disabled, no ctx, and race paths), fall through to the
        // cold-allocation path below.
        //
        // Text-only requests use the cache whenever the route attached a
        // context. Multimodal requests opt in only under `vlm_sharing_ok`;
        // when they do, the adopt is restricted to a whole-entry match
        // (`require_whole_entry == is_multimodal`) so the prefilled suffix is
        // guaranteed to be the newly-appended text turn: every image/audio
        // token sits inside the matched prefix, and a different media payload
        // lands in a different digest bucket. Multimodal requests with sharing
        // off keep the legacy cold-prefill path (their pre-injection token
        // stream is not self-describing).
        let ctx_ref = if is_multimodal && !vlm_sharing_ok {
            None
        } else {
            options.prompt_cache_ctx.as_ref()
        };
        let (seq_id, prefill_start_offset, already_cached_tokens) = match ctx_ref
            .and_then(|ctx| self.try_adopt_cached_prefix(ctx, &prompt_tokens, is_multimodal))
        {
            Some((adopted_id, adopted_len)) => (adopted_id, adopted_len, adopted_len),
            None => {
                // Miss or feature disabled → regular allocate.
                // count misses only when the cache is actually
                // active (ctx_ref is Some) to avoid inflating the miss
                // counter for multimodal or cache-disabled requests.
                if ctx_ref.is_some() {
                    self.batch_metrics.record_prompt_cache_miss();
                }
                let seq_id = match self.allocate_sequence_state() {
                    Ok(id) => id,
                    Err(err) => {
                        tracing::warn!("Cache pool allocation failed: {err}");
                        let _ =
                            response_tx.send(GenerateEvent::Error(format!("Server busy: {err}")));
                        return;
                    }
                };
                (seq_id, 0, 0)
            }
        };

        // Resolve the per-sequence input embeddings. On the VLM-sharing path
        // the tokens were already expanded above; if a prefix was adopted
        // (`prefill_start_offset > 0`) the remaining suffix is the appended
        // text turn, so the full-prompt embeddings are dropped and the suffix
        // runs through the token path (the adopted KV already holds the image
        // rows, and the MRoPE / per-layer state bound below covers the suffix
        // positions). Otherwise the request prepares embeddings here exactly as
        // before.
        let vlm_embeddings = match prepared_early {
            Some(emb) => {
                if prefill_start_offset > 0 {
                    None
                } else {
                    emb
                }
            }
            None => match prepare_request_vlm_embeddings(
                &self.model,
                &self.tokenizer,
                &prompt,
                &mut prompt_tokens,
                &images,
                &audio,
                &videos,
                Some(self.vision_caches.as_ref()),
            ) {
                Ok(emb) => emb,
                Err(err) => {
                    // Clean up the context map so a donate-back won't fire for
                    // a sequence that never reached a healthy finish.
                    self.prompt_cache_seq_ctx.remove(&seq_id);
                    self.release_sequence_caches(seq_id);
                    let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
                    return;
                }
            },
        };

        // mlx-vlm PR #1095: per-sequence MRoPE alignment.
        //
        // The Qwen VL families compute the MRoPE position-id tensor and
        // `rope_deltas` scalar inside `prepare_request_vlm_embeddings`
        // (it runs the vision encoder and writes the result to the text
        // model's fallback slot). Without binding that state to *this*
        // sequence id, the next text-only request's decode step would
        // pick up the previous VL row's delta and produce wrong
        // attention positions. Bind unconditionally for Qwen VL models;
        // the call is a no-op for everything else.
        self.model.bind_qwen_vl_mrope_state_to_sequence(seq_id);

        // per-sequence `per_layer_inputs` for Gemma 4
        // E2B/E4B. `prepare_request_vlm_embeddings` writes the
        // freshly projected tensor to the VL model's fallback slot;
        // this call drains the slot into a per-`SequenceId` map so a
        // burst of Gemma 4 VLM requests in a single drain tick cannot
        // have one row consume another row's tensor. No-op for
        // everything that is not a Gemma 4 VLM.
        self.model.bind_gemma4_per_layer_inputs_to_sequence(seq_id);

        // Issue #85: same lifecycle invariant for Gemma 3n VLM. The
        // legacy `Gemma3nVLModel.cached_per_layer_inputs` cell was a
        // single fallback slot with no per-sequence binding; under a
        // burst of Gemma 3n VLM requests the next prepare would
        // overwrite the slot before the first prefill consumed it (or
        // panic on `Option::unwrap` when the timing flipped). The
        // call below is a no-op for everything that is not a
        // Gemma 3n VLM.
        self.model.bind_gemma3n_per_layer_inputs_to_sequence(seq_id);

        let decode_state = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);

        // resolve the effective thinking-token budget for this
        // sequence from the per-request override + server default. The route
        // layer supplies `thinking_enter_block_on_start` as `true` when the
        // rendered prompt primes `<think>` (chat endpoints) and `false` for
        // raw text endpoints.
        let thinking = self.build_thinking_state(
            options.reasoning_budget,
            options.thinking_enter_block_on_start,
        );

        // Record the per-request prompt-cache context so the donate-back
        // path can compose the insert key without reaching back into the
        // HTTP layer. Only stored when the feature is active and the
        // request actually carried a context — otherwise the map stays
        // empty and the donate-back short-circuits. Multimodal requests are
        // stored only when VLM sharing is enabled (#124 step c); otherwise
        // they keep opting out of the cache entirely.
        if self.prompt_cache_active()
            && (!is_multimodal || vlm_sharing_ok)
            && let Some(ctx) = options.prompt_cache_ctx.clone()
        {
            self.prompt_cache_seq_ctx.insert(seq_id, ctx);
        }

        // Guard against a degenerate cache hit where the adopted prefix
        // covers the entire tokenized prompt. This can legitimately happen
        // when a client replays an identical prompt. Back off one token so
        // the prefill path still runs and the sampler sees fresh logits.
        let prefill_start_offset =
            if prefill_start_offset >= prompt_tokens.len() && !prompt_tokens.is_empty() {
                tracing::debug!(
                    seq_id = %seq_id,
                    "prompt-cache hit covered the entire prompt; re-running the \
                     last token through prefill to produce a sampling logit"
                );
                prompt_tokens.len() - 1
            } else {
                prefill_start_offset
            };

        let seq = SequenceInfo {
            seq_id,
            state: SequenceState::Queued,
            prompt_tokens,
            sampling,
            max_tokens: options.max_tokens,
            eos_token_ids: self.config_eos.clone(),
            priority: options.priority,
            logprobs_config: options.logprobs,
            vlm_embeddings,
            images,
            audio,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            decode_state,
            prefill_offset: 0,
            prefill_start_offset,
            already_cached_tokens,
            response_tx,
            cancelled,
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
            token_history: Vec::new(),
            sampler_state: None,
            merged_eos: Vec::new(),
            thinking,
            // forward the structured-output constraint built by
            // the route layer so the per-step sampling path can consult it.
            structured: options.structured.clone(),
        };

        if let Err(rejected) = self.prefill_queue.enqueue(seq) {
            self.prompt_cache_seq_ctx.remove(&rejected.seq_id);
            self.release_sequence_caches(rejected.seq_id);
            let _ = rejected.response_tx.send(GenerateEvent::Error(
                "Server busy: prefill queue full".to_string(),
            ));
        }
    }

    // ------------------------------------------------------------------
    // Scheduling decision
    // ------------------------------------------------------------------

    /// Determine the next action. Runs in O(1) time.
    ///
    /// Policy:
    /// 1. If a chunked prefill is in progress and active sequences exist,
    ///    decode first (interleave).
    /// 2. If a chunked prefill is in progress and no active sequences,
    ///    continue the prefill.
    /// 3. If active sequences exist, decode first.
    /// 4. If the batch is not full and the queue has work, prefill.
    /// 5. Otherwise idle.
    fn decide_action(&self) -> BatchSchedulerAction {
        tracing::debug!(
            active = self.active_batch.len(),
            queued = self.prefill_queue.len(),
            chunked_in_progress = self.chunked_prefill_seq.is_some(),
            "scheduler tick"
        );
        // Chunked prefill in progress: interleave decode with prefill
        if self.chunked_prefill_seq.is_some() {
            if !self.active_batch.is_empty() {
                // Interleave: decode active sequences first, then continue
                // prefill on the next tick.
                return BatchSchedulerAction::Decode(self.active_batch.sequence_ids());
            }
            // No active sequences, continue the prefill
            return BatchSchedulerAction::Prefill(SequenceId::from_raw(0));
        }

        if self.active_batch.is_empty() && self.prefill_queue.is_empty() {
            return BatchSchedulerAction::Idle;
        }

        // When active sequences exist:
        // 1. If batch is NOT full and queue has work → admit one new sequence
        //    (this grows the batch to improve decode throughput via batching)
        // 2. If batch is full or queue is empty → decode existing sequences
        // 3. Preemption overrides when enabled and a higher-priority request waits
        if !self.active_batch.is_empty() {
            if self.should_preempt() {
                return BatchSchedulerAction::Prefill(SequenceId::from_raw(0));
            }
            if !self.active_batch.is_full() && !self.prefill_queue.is_empty() {
                // Admit one queued request to grow the batch before decoding.
                // This is critical for throughput: larger batches amortize
                // weight-loading bandwidth across more sequences.
                return BatchSchedulerAction::Prefill(SequenceId::from_raw(0));
            }
            return BatchSchedulerAction::Decode(self.active_batch.sequence_ids());
        }

        // Batch is empty but queue has work
        BatchSchedulerAction::Prefill(SequenceId::from_raw(0))
    }

    /// Check if preemption should occur: batch is full, preemption is
    /// enabled, and a higher-priority request is waiting.
    fn should_preempt(&self) -> bool {
        if !self.enable_preemption || !self.active_batch.is_full() {
            return false;
        }
        // Only preempt if waiting request has higher priority than some
        // active sequence.
        let waiting_priority = match self.prefill_queue.peek_priority() {
            Some(p) => p,
            None => return false,
        };
        // Find the lowest-priority active sequence
        let min_active_priority = self
            .active_batch
            .iter_min_priority()
            .unwrap_or(RequestPriority::High);

        waiting_priority > min_active_priority
    }

    // ------------------------------------------------------------------
    // Paged KV block-budget admission (#122 b2)
    // ------------------------------------------------------------------

    /// Estimate the pool blocks a sequence's prefill will pin: one block per
    /// `block_size` prompt tokens, per layer. Returns 0 when there is no paged
    /// pool (the budget gate is then a no-op for this model).
    fn estimate_prefill_blocks(&self, prompt_len: usize) -> usize {
        match self.cache_pool.paged_block_size() {
            Some(block_size) if block_size > 0 => prompt_len
                .div_ceil(block_size)
                .saturating_mul(self.model.num_layers()),
            _ => 0,
        }
    }

    /// Reclaim paged pool blocks until at least `need` are acquirable, or no
    /// further reclamation is possible. First evicts cold prompt-cache prefixes
    /// (LRU; releasing their pins frees real blocks), then preempts running
    /// sequences (which re-prefill on resume). Returns whether `need` blocks are
    /// now acquirable.
    fn reclaim_paged_blocks(&mut self, need: usize) -> bool {
        let room = |pool: &CachePool| pool.free_paged_block_budget().is_none_or(|f| f >= need);
        // 1. Evict cold cross-request prefixes; releasing their pins frees blocks.
        if let Some(store) = self.prompt_cache.clone() {
            while !room(&self.cache_pool) {
                if store.evict_one_lru() == 0 {
                    break; // nothing left to evict
                }
                self.drain_store_paged_releases();
            }
        }
        if room(&self.cache_pool) {
            return true;
        }
        // 2. Preempt running sequences (drop their KV; they re-prefill on resume).
        while !room(&self.cache_pool) {
            if !self.try_evict_for_preemption() {
                break; // no preemptible victim left
            }
        }
        room(&self.cache_pool)
    }

    /// Paged block-budget admission gate. Returns `Some(seq)` to proceed with
    /// the prefill, or `None` when the sequence was deferred (re-queued for a
    /// later tick once decodes free blocks) or rejected (it cannot fit the whole
    /// budget). A no-op (`Some(seq)`) when no budget is configured.
    fn admit_paged_prefill(&mut self, seq: SequenceInfo) -> Option<SequenceInfo> {
        // Opt-in: no budget configured ⇒ admit (default unbounded behaviour).
        let total = match self.cache_pool.paged_block_budget() {
            Some(t) => t,
            None => return Some(seq),
        };
        let need = self.estimate_prefill_blocks(seq.prompt_tokens.len());
        if need == 0 {
            return Some(seq); // model does not use the paged pool
        }
        // If it cannot fit the entire budget, reject — deferring forever would
        // wedge the queue behind a request that can never run.
        if need > total {
            self.abort_sequence(
                seq,
                &format!(
                    "prompt needs {need} KV blocks, exceeding the {total}-block KV cache budget"
                ),
            );
            return None;
        }
        // Acquirable blocks (budget − live). `None` means the pool is not yet
        // created (nothing allocated ⇒ the whole budget is free).
        let free = self.cache_pool.free_paged_block_budget().unwrap_or(total);
        if need <= free {
            return Some(seq);
        }
        if self.reclaim_paged_blocks(need) {
            return Some(seq);
        }
        // Still no room — defer to a later tick. Decodes in flight will free
        // blocks as their sequences finish; this request retries then.
        if let Err(rejected) = self.prefill_queue.enqueue(seq) {
            self.prompt_cache_seq_ctx.remove(&rejected.seq_id);
            self.release_sequence_caches(rejected.seq_id);
            let _ = rejected.response_tx.send(GenerateEvent::Error(
                "Server busy: prefill queue full".to_string(),
            ));
        }
        None
    }

    // ------------------------------------------------------------------
    // Prefill execution (chunked or full)
    // ------------------------------------------------------------------

    /// Prefill a sequence. If `prefill_chunk_size > 0` and the prompt
    /// exceeds one chunk, the prefill is split across multiple ticks with
    /// decode interleaving.
    fn execute_prefill(&mut self, _action_id: SequenceId) {
        // Resume a chunked prefill already in progress?
        if self.chunked_prefill_seq.is_some() {
            self.continue_chunked_prefill();
            return;
        }

        // Preemption: if batch is full and preemption is enabled, evict
        // a lower-priority sequence to make room.
        if self.active_batch.is_full() && self.enable_preemption && !self.try_evict_for_preemption()
        {
            // Cannot evict -- skip prefill this tick
            return;
        }

        let seq = match self.prefill_queue.dequeue() {
            Some(s) => s,
            None => return,
        };

        // #122 b2: paged KV block-budget admission gate. Opt-in — a no-op
        // unless a budget is configured (`free_paged_block_budget()` is `None`
        // otherwise). When the sequence would not fit, this evicts cold
        // prompt-cache prefixes, then preempts running sequences, and as a last
        // resort re-queues the sequence for a later tick (or rejects it if it
        // can never fit the whole budget). Returning `None` means the sequence
        // was deferred or rejected and this tick is done.
        let seq = match self.admit_paged_prefill(seq) {
            Some(s) => s,
            None => return,
        };

        // speculative-decoding burst path.
        //
        // Why: the existing speculative round loops
        // (`MtpGenerator::generate`, `DFlashGenerator::run`, plus their
        // batched B>1 peers) are self-contained drive loops that own
        // prefill + decode + finish in a single function call. Folding
        // them into the scheduler's tick-based decode_single_step would
        // require refactoring every generator into a per-tick step API —
        // a much larger and riskier change. Instead, we run the entire
        // speculative request lifecycle as one "burst" right at prefill
        // time, bypassing the standard prefill → finish_prefill →
        // active_batch → decode pipeline. The classic non-speculative
        // path is bit-exact preserved (the gate's per-sequence
        // preconditions also include: no multimodal payload / VLM
        // embeddings, no structured output, no adopted prompt-cache
        // prefix — see
        // `speculative_burst::should_burst_for_sequence`).
        //
        // adds the B>1 batched burst: when `max_batch_size >
        // 1` and the dequeued head sequence is speculative-eligible, the
        // scheduler collects an equal-prompt-length window of additional
        // eligible requests and drives them all through the batched
        // round-loop driver in one tick. A window of size 1 falls back
        // to the B=1 burst.
        let seq = match self.try_speculative_burst(seq) {
            // Burst (B=1 or batched) handled the request(s) end-to-end.
            None => return,
            // Burst declined; route the returned head sequence through
            // the classic prefill path. Any sibling rows that were
            // collected into a declined window were re-routed by
            // `try_speculative_burst` already.
            Some(rejected_seq) => rejected_seq,
        };

        let mut seq = seq;
        if let Err(err) = Self::begin_prefill(&mut seq) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }

        let prompt_len = seq.prompt_tokens.len();

        // Decide: chunked vs full prefill.
        //
        // VLM image requests carry pre-merged input embeddings spanning the
        // full (unpadded) prompt length and are consumed whole by
        // `forward_with_embeddings` (which ignores `input_ids` length when
        // embeddings are present). Chunked prefill would (a) feed the entire
        // embedding sequence on chunk 0 while advancing `prefill_offset` by
        // only one chunk — corrupting the cache/offset bookkeeping — and
        // (b) re-introduce the NA-tile padding/embedding shape mismatch that
        // `execute_full_prefill` guards against. So embeddings-bearing
        // sequences always take the full-prefill path, mirroring the batched
        // dispatch which already forces VLM requests to `execute_full_prefill`.
        if self.prefill_chunk_size > 0
            && prompt_len > self.prefill_chunk_size
            && seq.vlm_embeddings.is_none()
        {
            // Start chunked prefill: process first chunk
            self.start_chunked_prefill(seq);
        } else {
            // Full-prompt prefill (original path)
            self.execute_full_prefill(seq);
        }
    }

    /// Attempt to handle the dequeued head sequence through the
    /// speculative-decoding burst path (B=1 / batched B>1).
    ///
    /// Returns:
    /// - `None` — a burst (B=1 or batched) handled the request(s)
    ///   end-to-end. The caller must `return` immediately; the
    ///   sequence(s) are already finalized and their caches released.
    /// - `Some(seq)` — the burst declined (or the head was not
    ///   speculative-eligible). The caller routes `seq` through the
    ///   classic prefill path. If a batched window had been collected
    ///   and then declined, the sibling rows were re-enqueued onto the
    ///   prefill queue here so they retry on the next tick.
    ///
    /// ## Batched-window collection
    ///
    /// When `max_batch_size > 1` and the head is speculative-eligible,
    /// this method drains an *equal-prompt-length* window of additional
    /// eligible requests from the front of the head's priority lane (via
    /// [`super::queue::PrefillQueue::drain_matching_window`]). The
    /// batched MTP target adapter requires a rectangular `[B, L]`
    /// prefill, and equal-length prompts make the batched prefill
    /// byte-identical to B separate B=1 prefills (acceptance item 1). The window also requires matching `max_tokens` and
    /// sampling config so the single per-window sampler / budget the
    /// batched round loop takes is correct for every row. B > 1 also
    /// applies [`super::speculative_burst::can_join_batched_burst_window`]
    /// so requests that need payloads unsupported by the batched round
    /// loops (currently logprobs) stay on the B = 1 burst path. A window
    /// that collapses to size 1 falls back to the B=1 burst.
    /// Whether to run the B=1 MTP burst for the next singleton request.
    ///
    /// Routes through the adaptive policy (issue #333) when one is attached: it
    /// forces MTP on while profiling, then returns its settled verdict. Without
    /// a policy (adaptive disabled) this is the pre-#333 static per-hardware
    /// gate ([`super::speculative_burst::mtp_b1_burst_enabled`]), which reads
    /// `MLXCEL_ENABLE_MTP_B1` and the hardware default. A pure read with no
    /// per-token cost.
    fn mtp_b1_should_run(&self) -> bool {
        match &self.mtp_policy {
            Some(policy) => policy.should_attempt_b1(),
            None => super::speculative_burst::mtp_b1_burst_enabled(self.model.supports_batching()),
        }
    }

    fn try_speculative_burst(&mut self, seq: SequenceInfo) -> Option<SequenceInfo> {
        // Fast path: speculative dispatch off, or the head fails the
        // per-sequence gate (multimodal payload / VLM embeddings /
        // structured output / adopted cache prefix). History-dependent
        // penalties, logprobs, and thinking budgets are supported by
        // the B = 1 burst path.
        // Route straight to classic only for the hard per-request gates.
        if !super::speculative_burst::should_burst_for_sequence(&self.speculative_dispatch, &seq) {
            return Some(seq);
        }

        // Try to assemble a B>1 batched window. The window head is
        // `seq`; siblings must (1) be speculative-eligible, (2) have the
        // same prompt length, (3) have the same `max_tokens`, and (4)
        // share the same sampling config — the batched round loop takes
        // one sampler and one per-row budget for the whole window. B>1
        // also excludes requests whose response payloads need per-row
        // data that batched round loops do not return yet (for example
        // logprobs), leaving them on the B=1 burst arm. The window cap
        // is the configured `max_batch_size`, surfaced via the active
        // batch's capacity (the scheduler constructs
        // `ActiveBatch::new(max_batch_size)`).
        let max_batch_size = self.active_batch.max_size();
        let head_can_join_batched = super::speculative_burst::can_join_batched_burst_window(&seq);
        // Ragged (variable-length-prompt) windows are gated behind a separate
        // opt-in subordinate to `MLXCEL_ENABLE_MTP_BATCH`. When off (the
        // default), the window collector keeps its original equal-prompt-length
        // constraint so the validated same-length batched burst is unchanged.
        // When on, the prompt-length equality is dropped and burst-eligible
        // rows of different lengths join one window; the batched MTP adapter
        // left-pads them to `max_prompt_len` and threads per-row positions /
        // valid lengths so greedy parity is preserved.
        let allow_ragged = super::speculative_burst::mtp_batched_ragged_window_enabled();
        let window: Vec<SequenceInfo> = if max_batch_size > 1 && head_can_join_batched {
            let head_prompt_len = seq.prompt_tokens.len();
            let head_max_tokens = seq.max_tokens;
            let head_sampling = seq.sampling.clone();
            let head_lane = seq.priority;
            // Reserve one slot for the head itself.
            let max_extra = max_batch_size.saturating_sub(1);
            let dispatch = &self.speculative_dispatch;
            let extra =
                self.prefill_queue
                    .drain_matching_window(head_lane, max_extra, |candidate| {
                        (allow_ragged || candidate.prompt_tokens.len() == head_prompt_len)
                            && candidate.max_tokens == head_max_tokens
                            && super::speculative_burst::sampling_config_eq(
                                &candidate.sampling,
                                &head_sampling,
                            )
                            && super::speculative_burst::should_burst_for_sequence(
                                dispatch, candidate,
                            )
                            && super::speculative_burst::can_join_batched_burst_window(candidate)
                    });
            let mut window = Vec::with_capacity(extra.len() + 1);
            window.push(seq);
            window.extend(extra);
            window
        } else {
            vec![seq]
        };

        if window.len() == 1
            && matches!(
                self.speculative_dispatch,
                crate::server::SpeculativeDispatch::Mtp { .. }
            )
            && !self.mtp_b1_should_run()
        {
            // B=1 MTP decision (issue #333, adaptive): when an adaptive policy
            // is attached it profiles the first few B=1 bursts of this
            // (target, drafter, hardware) pairing and settles to a data-driven
            // verdict, overriding the static per-hardware gate where the
            // measured profile is clearly favorable or unfavorable. Without a
            // policy (MLXCEL_MTP_ADAPTIVE off) this falls back to the static
            // per-hardware default (issue #165): non-batchable 12B targets keep
            // B=1 MTP on everywhere; batch-capable 31B targets default it on
            // only on M5+, since pre-M5 GPU cores measured a consistent
            // regression. `MLXCEL_ENABLE_MTP_B1` overrides in both directions;
            // on decline the request falls back to classic decode.
            let seq = window.into_iter().next().expect("singleton window");
            tracing::info!(
                "MTP B=1 speculative burst declined for seq {} (adaptive policy \
                 verdict, per-hardware default, or MLXCEL_ENABLE_MTP_B1=0); \
                 falling back to classic decode",
                seq.seq_id,
            );
            return Some(seq);
        }

        if window.len() >= 2
            && matches!(
                self.speculative_dispatch,
                crate::server::SpeculativeDispatch::Mtp { .. }
            )
            && !super::speculative_burst::mtp_batched_burst_enabled()
        {
            // The B>1 batched MTP burst is off by default because it is not
            // consistently faster than classic batched decode on the 31B
            // (M5 Max: ~1.06x for a same-length window, ~0.78x once prompt
            // lengths differ and requests serialize into head-of-line-blocking
            // B=1 bursts). M5 Max runs observed greedy parity holding at
            // temperature 0. Set MLXCEL_ENABLE_MTP_BATCH=1 to force the path.
            let mut rejected_window = window;
            let head = rejected_window.remove(0);
            tracing::info!(
                "MTP batched speculative burst declined for seq {}: B>1 MTP is not \
                 consistently faster than classic batched decode on the 31B; falling \
                 back to classic decode. Set MLXCEL_ENABLE_MTP_BATCH=1 to force the \
                 experimental batched MTP path",
                head.seq_id,
            );
            for sibling in rejected_window {
                let sibling_id = sibling.seq_id;
                if let Err(boxed) = self.prefill_queue.enqueue(sibling) {
                    tracing::warn!(
                        "MTP batched speculative burst declined and prefill queue \
                         full; aborting sibling seq {sibling_id}"
                    );
                    self.prompt_cache_seq_ctx.remove(&sibling_id);
                    self.abort_sequence(
                        *boxed,
                        "MTP batched speculative burst declined and prefill queue full",
                    );
                }
            }
            return Some(head);
        }

        if window.len() >= 2 {
            // ---- Batched B>1 burst ----
            let ctx = super::speculative_burst::BurstContext {
                model: &self.model,
                tokenizer: &self.tokenizer,
                drafter_slot: &mut self.speculative_drafter_slot,
                dispatch: &self.speculative_dispatch,
            };
            match super::speculative_burst::try_run_burst_batched(ctx, window) {
                Ok(super::speculative_burst::BatchedBurstFinalized { rows }) => {
                    // Every row handled inline. For each row: donate the
                    // finished sequence's KV cache back to the
                    // prompt-cache store, release its cache slot, and
                    // record its per-sequence metric — the batched
                    // analogue of the B=1 cleanup below.
                    //
                    // `BatchedBurstRow` now carries the
                    // per-row prompt/committed token vectors and the
                    // healthy-finish flag, so the batched arm calls
                    // `donate_finished_sequence_cache` per row BEFORE
                    // the `remove`/`release` — symmetric with the B=1
                    // arm and the classic `finalize_completed` path.
                    // The donate helper chooses the model's supported
                    // cross-request reuse path: exact-prefix snapshots for
                    // opt-in model-owned families, otherwise detached KV for
                    // dense/paged families. Wiring it in removes the structural
                    // asymmetry between the two burst arms and future-proofs
                    // the batched path for any reusable family that later
                    // becomes batched-burst-eligible. Error /
                    // transition-failure rows carry an empty/`false`
                    // payload so the donate is a guaranteed no-op on
                    // those tainted-cache rows.
                    for super::speculative_burst::BatchedBurstRow {
                        seq_id,
                        tokens_generated,
                        prompt_tokens,
                        generated_tokens,
                        healthy_finish,
                    } in rows
                    {
                        self.donate_finished_sequence_cache(
                            seq_id,
                            &prompt_tokens,
                            &generated_tokens,
                            healthy_finish,
                        );
                        // Defensive non-donate cleanup: the donate path
                        // above already removed the `prompt_cache_seq_ctx`
                        // entry on the dense-KV path.
                        self.prompt_cache_seq_ctx.remove(&seq_id);
                        self.release_sequence_caches(seq_id);
                        self.batch_metrics
                            .record_sequence_completed(tokens_generated);
                        self.batch_observability.record_sequence_completed();
                    }
                    self.publish_metrics();
                    None
                }
                Err(mut rejected_window) => {
                    // The batched burst declined the whole window (e.g.
                    // unsupported model variant). The head goes back to
                    // the caller for the classic path; the sibling rows
                    // are re-enqueued so they retry next tick (they
                    // re-evaluate as their own potential window heads).
                    let head = rejected_window.remove(0);
                    for sibling in rejected_window {
                        let sibling_id = sibling.seq_id;
                        if let Err(boxed) = self.prefill_queue.enqueue(sibling) {
                            // Queue full: the sibling cannot be retried.
                            // Abort it with a clear error rather than
                            // dropping it silently. This is extremely
                            // unlikely — the sibling was just dequeued
                            // from this same queue.
                            tracing::warn!(
                                "speculative burst window declined and prefill queue \
                                 full; aborting sibling seq {sibling_id}"
                            );
                            self.prompt_cache_seq_ctx.remove(&sibling_id);
                            self.abort_sequence(
                                *boxed,
                                "speculative burst declined and prefill queue full",
                            );
                        }
                    }
                    Some(head)
                }
            }
        } else {
            // ---- B=1 burst (window collapsed to the head only) ----
            let seq = window.into_iter().next().expect("window has the head");
            let ctx = super::speculative_burst::BurstContext {
                model: &self.model,
                tokenizer: &self.tokenizer,
                drafter_slot: &mut self.speculative_drafter_slot,
                dispatch: &self.speculative_dispatch,
            };
            match super::speculative_burst::try_run_burst_b1(ctx, seq) {
                Ok(super::speculative_burst::BurstFinalized {
                    seq_id,
                    tokens_generated,
                    prompt_tokens,
                    generated_tokens,
                    healthy_finish,
                    mtp_profile,
                    burst_wall_ms,
                }) => {
                    // Burst handled the full request lifecycle inline.
                    //
                    // donate the finished sequence's KV
                    // cache back to the prompt-cache store BEFORE the
                    // `remove`/`release` below — `donate_finished_sequence_cache`
                    // both consumes the `prompt_cache_seq_ctx` entry and
                    // needs the cache slot still attached. This mirrors
                    // the classic path's `finalize_completed`, keeping
                    // the burst and classic donate paths symmetric. The
                    // donate helper snapshots opt-in model-owned families and
                    // detaches dense/paged KV for the regular backends. Wiring
                    // it in removes the structural asymmetry between the two
                    // paths and future-proofs the burst for any reusable model
                    // family that later becomes burst-eligible.
                    self.donate_finished_sequence_cache(
                        seq_id,
                        &prompt_tokens,
                        &generated_tokens,
                        healthy_finish,
                    );
                    // Release the pre-allocated cache slot for symmetry
                    // with `finalize_completed`, and mirror the classic
                    // path's per-sequence metric recording so Prometheus
                    // counters cover burst completions too. The `remove`
                    // here is the defensive non-donate cleanup (the
                    // donate path above already removed the
                    // `prompt_cache_seq_ctx` entry on the dense-KV path).
                    self.prompt_cache_seq_ctx.remove(&seq_id);
                    self.release_sequence_caches(seq_id);
                    self.batch_metrics
                        .record_sequence_completed(tokens_generated);
                    self.batch_observability.record_sequence_completed();
                    // Feed the adaptive MTP policy (issue #333) the coarse
                    // profile of this B=1 burst. Only present for MTP runs that
                    // executed a speculative round; a no-op once the policy has
                    // settled, so there is no steady-state per-request cost.
                    if let (Some(policy), Some(profile)) = (self.mtp_policy.as_mut(), mtp_profile) {
                        policy.record_b1_sample(profile);
                    }
                    // Observability (issue #638): the B=1 burst runs the whole
                    // request to completion in this single scheduler tick, so
                    // `burst_wall_ms` is the head-of-line stall it imposed on
                    // every concurrent classic-decode row (none advanced while
                    // the burst held the worker). Surface it with the round /
                    // accepted-token counts so the HOL cost of the
                    // run-to-completion burst is measurable until the
                    // tick-cooperative slice lands. `queued` shows how many
                    // rows waited behind it.
                    let (rounds, accepted) = mtp_profile
                        .map(|p| (p.rounds, p.accepted_draft_tokens))
                        .unwrap_or((0, 0));
                    tracing::info!(
                        seq_id = %seq_id,
                        burst_wall_ms,
                        tokens_generated,
                        rounds,
                        accepted_draft_tokens = accepted,
                        hol_waiters = self.active_batch.len() + self.prefill_queue.len(),
                        "speculative B=1 burst finalized (burst_wall_ms is the HOL \
                         stall on concurrent rows)"
                    );
                    self.publish_metrics();
                    None
                }
                Err(rejected_seq) => Some(rejected_seq),
            }
        }
    }

    /// #715: whether the head-of-queue prompt is short enough to enter the
    /// batched-prefill path under the padded-token budget.
    ///
    /// A batched cohort pads every row to the window's longest prompt `L`, so
    /// the head can only ever join a `>= 2`-row batch when `2 * head_len` stays
    /// within the budget. When it cannot (or the queue is empty), batching
    /// would collapse to a single-row unchunked `[L, L]` forward, so the head
    /// is instead routed to the normal chunk-aware single-sequence path. `0`
    /// (uncapped) always admits.
    fn batched_prefill_admits_head(&self) -> bool {
        let budget = self.max_batch_prefill_tokens;
        if budget == 0 {
            return true;
        }
        match self.prefill_queue.peek_prompt_len() {
            Some(head_len) => head_len.saturating_mul(2) <= budget,
            None => false,
        }
    }

    /// Batched prefill: drain up to `max_batch_prefill` requests from the
    /// prefill queue and process eligible cold text rows in a single forward
    /// pass.
    ///
    /// The drained window can mix requests the padded batched path supports
    /// (cold text: zero KV-history offset, no custom embeddings) with requests
    /// it does not (adopted prompt-cache prefixes, VLM / custom embeddings).
    /// #332: instead of falling the whole window back to sequential prefill
    /// when it contains one incompatible request, the window is split into
    /// cohorts ([`plan_prefill_cohorts`]). Cold cohorts run batched; everything
    /// else takes the offset-aware single-sequence path. Cohorts are dispatched
    /// in window order, and the queue dequeues in priority order, so request
    /// priority / FIFO fairness is preserved across cohort boundaries.
    fn execute_batched_prefill(&mut self) {
        let batch_size = self.max_batch_prefill.min(self.prefill_queue.len());

        // Collect up to `batch_size` requests from the queue. The queue
        // dequeues in priority order (high lane, then normal, then low; FIFO
        // within a lane), so `seqs` is already in priority order.
        //
        // #715: the drain is also bounded by the padded-token budget
        // (`max_batch_prefill_tokens`). Because the padded batched path pads
        // every row to the window's longest prompt `L`, the drained window
        // costs `rows * L` padded tokens and materializes an `O(rows * L^2)`
        // FP32 mask. Draining stops before a row that would push `rows * L`
        // past the budget; the remaining rows stay queued and are prefilled on
        // a later tick (short ones re-batch, long ones take the chunked
        // single-sequence path). The head row is always taken so the drain
        // makes forward progress; the dispatch-time guard
        // ([`Self::batched_prefill_admits_head`]) has already kept a head too
        // long to batch out of this path entirely.
        let budget = self.max_batch_prefill_tokens;
        let mut seqs: Vec<SequenceInfo> = Vec::with_capacity(batch_size);
        let mut window_max_len = 0usize;
        while seqs.len() < batch_size {
            let Some(next_len) = self.prefill_queue.peek_prompt_len() else {
                break;
            };
            if !batched_window_admits(seqs.len(), window_max_len, next_len, budget) {
                break;
            }
            let Some(seq) = self.prefill_queue.dequeue() else {
                break;
            };
            window_max_len = window_max_len.max(seq.prompt_tokens.len());
            seqs.push(seq);
        }

        if seqs.is_empty() {
            return;
        }

        // Classify each row, then plan cohorts. A row is "cold" only when it
        // has no VLM / custom embeddings AND no adopted prompt-cache prefix.
        // That is exactly the precondition the padded batched path assumes (a
        // zero cache offset for every row), so the planner's guarantee that a
        // BatchedCold cohort holds only cold rows is what keeps cache offsets
        // correct: an adopted prefix can never be folded into a batch and have
        // its KV resumed at the wrong position.
        let can_batch = self.model.supports_batched_prefill();
        let can_pad = self.model.supports_padded_prefill();
        let rows: Vec<PrefillRow> = seqs
            .iter()
            .map(|s| PrefillRow {
                is_cold: s.vlm_embeddings.is_none() && s.prefill_start_offset == 0,
                prompt_len: s.prompt_tokens.len(),
            })
            .collect();
        let plan = plan_prefill_cohorts(&rows, can_batch, can_pad);

        // Move sequences into index-addressable slots so each cohort can take
        // ownership of exactly its members. Dispatching cohorts in plan order
        // reproduces window (priority) order across the cohort boundaries.
        let mut slots: Vec<Option<SequenceInfo>> = seqs.into_iter().map(Some).collect();
        for cohort in plan {
            match cohort.kind {
                // Behavior note (#332): a cold row that previously fell back to
                // *sequential* prefill (because the collected window held an
                // incompatible sibling) now runs batched here. A padded batched
                // forward (B > 1) is not bitwise-identical to single-sequence
                // prefill on Metal, so such a row's greedy decode can differ
                // from its old sequential output by an early near-tie token flip
                // (the documented #203 / #325 / #326 jitter class). That is the
                // intended effect of cohort splitting, not a correctness
                // regression: the guarantee is that a cohort-split cold row
                // decodes identically to the same row in an all-cold batched
                // window of the same composition (pinned by
                // scheduler_cohort_parity_tests).
                PrefillCohortKind::BatchedCold => {
                    let group: Vec<SequenceInfo> = cohort
                        .members
                        .iter()
                        .filter_map(|&i| slots[i].take())
                        .collect();
                    self.run_padded_batched_prefill(group);
                }
                PrefillCohortKind::Sequential => {
                    for &i in &cohort.members {
                        let Some(mut seq) = slots[i].take() else {
                            continue;
                        };
                        if let Err(err) = Self::begin_prefill(&mut seq) {
                            tracing::error!("Batched prefill state transition error: {err}");
                            self.abort_sequence(seq, &err);
                            continue;
                        }
                        self.execute_full_prefill(seq);
                    }
                }
            }
        }
    }

    /// Run a single padded batched prefill over a cohort of cold text rows.
    ///
    /// Every sequence in `seqs` must be cold (zero KV-history offset, no custom
    /// embeddings); [`plan_prefill_cohorts`] guarantees this, so the pipeline
    /// below can assume a zero cache offset for every row. Sequences are padded
    /// to the longest prompt in the cohort (aligned to a 32-token tile on M5+),
    /// each with a per-sequence causal + padding mask, and run in one forward
    /// pass. On any error the affected sequences fall back to the
    /// single-sequence prefill path so no request is lost.
    fn run_padded_batched_prefill(&mut self, mut seqs: Vec<SequenceInfo>) {
        // Defensive: the planner only emits BatchedCold cohorts of >= 2 rows,
        // but keep the empty / single cases correct if called directly.
        if seqs.is_empty() {
            return;
        }
        if seqs.len() == 1 {
            let mut seq = seqs.remove(0);
            if let Err(err) = Self::begin_prefill(&mut seq) {
                tracing::error!("Batched prefill state transition error: {err}");
                self.abort_sequence(seq, &err);
                return;
            }
            self.execute_full_prefill(seq);
            return;
        }

        // Transition all sequences to Prefilling up front so every fallback
        // below routes through `execute_full_prefill` from the correct state.
        for seq in &mut seqs {
            if let Err(err) = Self::begin_prefill(seq) {
                tracing::error!("Batched prefill state transition error: {err}");
            }
        }

        let b = seqs.len();
        let max_len = seqs.iter().map(|s| s.prompt_tokens.len()).max().unwrap();
        let can_pad_prefill = self.model.supports_padded_prefill();
        if !can_pad_prefill && seqs.iter().any(|s| s.prompt_tokens.len() != max_len) {
            // Should not happen for a planner-approved cohort (it only batches
            // equal-length rows on equal-length-only models), but stay safe.
            tracing::debug!(
                "batched prefill: cohort fell back to sequential (model requires equal prompt lengths)"
            );
            for seq in seqs {
                self.execute_full_prefill(seq);
            }
            return;
        }

        let padded_len = if can_pad_prefill && should_align_prefill() {
            align_to_na_tile(max_len)
        } else {
            max_len
        };

        tracing::debug!("batched prefill: {} requests, padded to {}", b, padded_len);

        // Build padded input: [B, padded_len]
        let mut flat_tokens: Vec<i32> = Vec::with_capacity(b * padded_len);
        for seq in &seqs {
            let tokens = &seq.prompt_tokens;
            flat_tokens.extend_from_slice(tokens);
            // Pad with 0 to padded_len
            flat_tokens.extend(std::iter::repeat_n(0, padded_len - tokens.len()));
        }
        let input = mlxcel_core::from_slice_i32(&flat_tokens, &[b as i32, padded_len as i32]);

        // Build per-sequence attention masks and collect cache pointers.
        // Each mask has shape [padded_len, padded_len]. Stacking on axis 0
        // produces [B, padded_len, padded_len], which model batched-prefill
        // paths slice per sequence into [padded_len, padded_len].
        let stacked_mask = if seqs.iter().any(|s| s.prompt_tokens.len() != padded_len) {
            let mut batch_masks: Vec<UniquePtr<mlxcel_core::MlxArray>> = Vec::with_capacity(b);
            for seq in &seqs {
                let actual = seq.prompt_tokens.len() as i32;
                let padded = padded_len as i32;
                let mask = create_padded_prefill_mask(actual, padded, 0);
                batch_masks.push(mask);
            }
            Some(mlxcel_core::stack_owned(&batch_masks, 0))
        } else {
            None
        };

        let batch_ids: Vec<SequenceId> = seqs.iter().map(|seq| seq.seq_id).collect();
        let mut batch_caches = match self.cache_pool.get_batch_caches_mut(&batch_ids) {
            Ok(caches) => caches,
            Err(err) => {
                tracing::warn!("batched prefill: {err}, falling back");
                // Re-queue all sequences for sequential processing.
                for seq in seqs {
                    self.execute_full_prefill(seq);
                }
                return;
            }
        };

        if batch_caches.len() != b {
            // Re-queue all sequences for sequential processing.
            for seq in seqs {
                self.execute_full_prefill(seq);
            }
            return;
        }

        // Single batched forward pass: [B, padded_len] → [B, padded_len, vocab]
        let raw_logits = self.model.forward_batched_with_context_and_ids(
            &input,
            Some(&batch_ids),
            &mut batch_caches,
            stacked_mask.as_deref(),
            None,
        );

        mlxcel_core::eval(&raw_logits);
        mlxcel_core::clear_memory_cache();

        // Process per-sequence results.
        for (i, mut seq) in seqs.into_iter().enumerate() {
            let actual_len = seq.prompt_tokens.len();
            let padded = padded_len;

            // Extract logits at the last real token position: index [i, actual_len-1, :]
            let last_pos = actual_len as i32 - 1;
            let vocab = {
                let shape = mlxcel_core::array_shape(&raw_logits);
                shape[2]
            };
            let seq_logits = mlxcel_core::slice(
                &raw_logits,
                &[i as i32, last_pos, 0],
                &[i as i32 + 1, last_pos + 1, vocab],
            );

            // Trim padding positions from this sequence's KV cache so that the
            // decode phase starts with the correct cache offset.
            let excess = (padded - actual_len) as i32;
            if excess > 0
                && let Some(caches) = self.cache_pool.get_caches_mut(seq.seq_id)
            {
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }

            self.sync_sequence_storage(seq.seq_id);

            seq.prefill_offset = actual_len;
            self.batch_observability.record_prefill_start(actual_len);

            let eos_tokens =
                merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
            let needs_history = seq.sampling.needs_token_history();
            let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

            self.finish_prefill(seq, seq_logits, eos_tokens, token_history, needs_history);
        }
    }

    /// Full-prompt prefill: process the entire prompt in one pass.
    ///
    /// when `seq.prefill_start_offset > 0`, a
    /// prompt-cache hit has installed the first `prefill_start_offset` tokens
    /// of KV state on this sequence. Only the suffix tokens are fed to the
    /// model. The VLM-prefix path deliberately opts out of cache adoption at
    /// the enqueue site, so this branch never has to mix the two.
    fn execute_full_prefill(&mut self, mut seq: SequenceInfo) {
        let _span = tracing::info_span!(
            "prefill",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
            cached = seq.prefill_start_offset,
        )
        .entered();
        // Only the suffix enters the prefill counters — the first
        // `prefill_start_offset` tokens were resolved from the adopted
        // detached cache with zero model work.
        let suffix_len = seq.prompt_tokens.len() - seq.prefill_start_offset;
        self.batch_observability.record_prefill_start(suffix_len);

        // Non-batching models use internal RefCell caches that are shared
        // across all sequences.  Reset them now (at prefill time) rather
        // than at enqueue time so that queued requests don't corrupt an
        // in-flight generation.
        if !self.model.supports_batching() {
            let _ = self.model.make_caches();
        }

        let eos_tokens =
            merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
        let needs_history = seq.sampling.needs_token_history();
        let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

        // Feed only the suffix tokens to the model when a cached prefix was
        // adopted. For cold prefills `start == 0` and this is identical to
        // the legacy behavior.
        let suffix_tokens: Vec<i32> = seq.prompt_tokens[seq.prefill_start_offset..].to_vec();

        // Run prefill (with or without VLM embeddings).
        // On M5+ hardware pad the prompt to a 32-token tile boundary for
        // optimal Neural Accelerator throughput.
        let actual_len = suffix_tokens.len();
        // VLM image requests inject pre-merged input embeddings at the real
        // (unpadded) sequence length and run through `forward_with_embeddings`
        // below. NA-tile alignment pads only the token-id vector and builds a
        // matching padded mask — it does NOT pad the injected embeddings. So
        // aligning here would hand the model a padded mask (e.g. 320x320) that
        // cannot broadcast against the unpadded embeddings (e.g. [1,H,293,293]),
        // aborting the process. Skip alignment when embeddings are present; the
        // text backbone then builds a causal mask sized to the embeddings,
        // matching the CLI generate path. Token-id (text-only) prefill — for
        // VLMs and plain text models alike — is unaffected.
        let (effective_tokens, pad_mask_opt) =
            if should_align_prefill() && seq.vlm_embeddings.is_none() {
                let padded_len = align_to_na_tile(actual_len);
                if padded_len > actual_len {
                    let mut padded = suffix_tokens.clone();
                    padded.resize(padded_len, 0);
                    // The padding mask anchors to the adopted cache offset so
                    // the newly-prefilled positions see the correct KV-history
                    // positions on M5+ hardware.
                    let mask = create_padded_prefill_mask(
                        actual_len as i32,
                        padded_len as i32,
                        seq.prefill_start_offset as i32,
                    );
                    (padded, Some(mask))
                } else {
                    (suffix_tokens.clone(), None)
                }
            } else {
                (suffix_tokens.clone(), None)
            };

        let eff_len = effective_tokens.len() as i32;
        let input = mlxcel_core::from_slice_i32(&effective_tokens, &[1, eff_len]);
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found for sequence during prefill");
                    return;
                }
            };

            let raw_logits = if let Some(ref embeddings) = seq.vlm_embeddings {
                // VLM path: apply provided mask or the tile-alignment mask.
                match prepared_embedding_refs(embeddings) {
                    Ok((input_embeds, caller_mask)) => {
                        // Caller-supplied mask takes precedence; tile-alignment mask
                        // is used only when the caller does not provide one.
                        let effective_mask =
                            caller_mask.or(pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()));
                        let logits = self.model.forward_with_embeddings_and_sequence_id(
                            &input,
                            Some(input_embeds),
                            Some(seq.seq_id),
                            caches,
                            effective_mask,
                        );
                        mlxcel_core::eval(&logits);
                        self.model.after_prefill();
                        logits
                    }
                    Err(err) => {
                        self.abort_sequence(seq, &err.to_string());
                        return;
                    }
                }
            } else {
                self.model.forward_with_sequence_id(
                    &input,
                    Some(seq.seq_id),
                    caches,
                    pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
                )
            };

            // Extract logits at the last real token position and trim padding from
            // KV caches so the decode phase begins with the correct cache offset.
            if pad_mask_opt.is_some() && effective_tokens.len() > actual_len {
                let padded_len = effective_tokens.len();
                let shape = mlxcel_core::array_shape(&raw_logits);
                let vocab = shape[2];
                let sliced = mlxcel_core::slice(
                    &raw_logits,
                    &[0, actual_len as i32 - 1, 0],
                    &[shape[0], actual_len as i32, vocab],
                );
                // Trim padding positions from all KV caches.
                let excess = (padded_len - actual_len) as i32;
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
                sliced
            } else {
                raw_logits
            }
        };
        self.sync_sequence_storage(seq.seq_id);

        // H2: enforce the `--max-kv-size` cap at the end of a
        // full prefill before the sequence transitions to decode. A long
        // prompt can overshoot the cap during a single forward pass; without
        // this trim the first decode step would start with a too-wide live
        // window. With no cap configured this is a cheap early-return.
        self.enforce_max_kv_size_for(seq.seq_id);

        mlxcel_core::clear_memory_cache();
        // `prefill_offset` is a cursor into `prompt_tokens`, so it must
        // include the adopted prefix even though those tokens bypassed the
        // forward pass.
        seq.prefill_offset = seq.prefill_start_offset + actual_len;

        self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
    }

    /// Begin a chunked prefill: process the first chunk and store the
    /// sequence for continuation on subsequent ticks.
    ///
    /// `seq.prefill_start_offset` skips over the
    /// leading tokens that the adopted prompt-cache entry already covers,
    /// so the first chunk starts *after* the cached prefix.
    fn start_chunked_prefill(&mut self, mut seq: SequenceInfo) {
        let _span = tracing::info_span!(
            "chunked_prefill_start",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
            chunk_size = self.prefill_chunk_size,
            cached = seq.prefill_start_offset,
        )
        .entered();

        // Reset internal caches for non-batching models (same as execute_full_prefill).
        if !self.model.supports_batching() {
            let _ = self.model.make_caches();
        }

        let chunk_size = self.prefill_chunk_size;
        let chunk_range = match next_chunked_prefill_range(
            seq.prompt_tokens.len(),
            seq.prefill_start_offset,
            chunk_size,
        ) {
            Some(range) => range,
            None => {
                self.abort_sequence(seq, "Chunked prefill start had no suffix tokens to process");
                return;
            }
        };
        // Counter reflects only the work the model actually runs.
        let suffix_len = seq.prompt_tokens.len() - seq.prefill_start_offset;
        self.batch_observability.record_prefill_start(suffix_len);

        let start = chunk_range.start;
        let end = chunk_range.end;
        let chunk = &seq.prompt_tokens[start..end];

        // Align the first chunk to a 32-token tile boundary on M5+ hardware.
        let actual_chunk_len = chunk.len();
        let (eff_chunk, pad_mask_opt) = if should_align_prefill() {
            let padded_len = align_to_na_tile(actual_chunk_len);
            if padded_len > actual_chunk_len {
                let mut padded = chunk.to_vec();
                padded.resize(padded_len, 0);
                // Mask anchored to the KV offset the adopted prefix already
                // installed (starts at zero for cold prefills).
                let mask = create_padded_prefill_mask(
                    actual_chunk_len as i32,
                    padded_len as i32,
                    start as i32,
                );
                (padded, Some(mask))
            } else {
                (chunk.to_vec(), None)
            }
        } else {
            (chunk.to_vec(), None)
        };

        let eff_len = eff_chunk.len() as i32;
        let input = mlxcel_core::from_slice_i32(&eff_chunk, &[1, eff_len]);
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found for sequence during chunked prefill");
                    return;
                }
            };

            // VLM embeddings are applied only on the first chunk.
            let logits = if let Some(ref embeddings) = seq.vlm_embeddings {
                match prepared_embedding_refs(embeddings) {
                    Ok((input_embeds, caller_mask)) => {
                        let effective_mask =
                            caller_mask.or(pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()));
                        let logits = self.model.forward_with_embeddings_and_sequence_id(
                            &input,
                            Some(input_embeds),
                            Some(seq.seq_id),
                            caches,
                            effective_mask,
                        );
                        mlxcel_core::eval(&logits);
                        self.model.after_prefill();
                        logits
                    }
                    Err(err) => {
                        self.abort_sequence(seq, &err.to_string());
                        return;
                    }
                }
            } else {
                let logits = self.model.forward_with_sequence_id(
                    &input,
                    Some(seq.seq_id),
                    caches,
                    pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
                );
                mlxcel_core::eval(&logits);
                logits
            };

            // Trim padding positions from KV caches when the chunk was padded.
            if pad_mask_opt.is_some() && eff_chunk.len() > actual_chunk_len {
                let excess = (eff_chunk.len() - actual_chunk_len) as i32;
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }
            logits
        };
        self.sync_sequence_storage(seq.seq_id);

        // H2: enforce the `--max-kv-size` cap after each
        // prefill chunk so the live window cannot grow unbounded across
        // chunks of a long prompt. A 100k-token prompt with `--max-kv-size
        // 4096` would otherwise see the cap engage only after the entire
        // prefill completes — defeating the memory-bound the operator
        // configured. With no cap configured this is a cheap early-return.
        self.enforce_max_kv_size_for(seq.seq_id);

        mlxcel_core::clear_memory_cache();
        seq.prefill_offset = end;

        tracing::debug!(
            "Chunked prefill: seq {} chunk 0..{end}/{} tokens",
            seq.seq_id,
            seq.prompt_tokens.len()
        );

        // The chunked-vs-full decision in `prefill_sequence` keys off the
        // *full* prompt length, but the work we just ran covers only the
        // suffix `[prefill_start_offset..]`. When a prompt-cache hit adopts a
        // long prefix, that suffix can fit entirely in chunk 0 even though the
        // full prompt cleared the chunking threshold — so this first chunk has
        // already reached the end of the prompt and there is nothing to
        // continue. Finish the prefill now (mirroring the final-chunk handling
        // in `continue_chunked_prefill`). Storing the sequence for
        // continuation instead would feed an empty `[end..end]` chunk on the
        // next tick, producing a zero-length forward whose `[1, 0, vocab]`
        // logits crash in `slice_last_logits` (issue #179).
        if chunk_range.is_terminal {
            let eos_tokens =
                merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
            let needs_history = seq.sampling.needs_token_history();
            let token_history = initial_token_history(&seq.prompt_tokens, needs_history);
            self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
            return;
        }

        // Store the sequence for continuation
        self.chunked_prefill_seq = Some(seq);
    }

    /// Continue a chunked prefill that is already in progress.
    fn continue_chunked_prefill(&mut self) {
        let mut seq = match self.chunked_prefill_seq.take() {
            Some(s) => s,
            None => return,
        };

        let _span = tracing::info_span!(
            "chunked_prefill_continue",
            seq_id = %seq.seq_id,
            offset = seq.prefill_offset,
            total = seq.prompt_tokens.len(),
        )
        .entered();

        let chunk_size = self.prefill_chunk_size;
        let offset = seq.prefill_offset;
        let total = seq.prompt_tokens.len();
        let chunk_range = match next_chunked_prefill_range(total, offset, chunk_size) {
            Some(range) => range,
            None => {
                self.abort_sequence(
                    seq,
                    "Chunked prefill continuation had no remaining tokens to process",
                );
                return;
            }
        };
        self.batch_observability.record_prefill_chunk();

        let end = chunk_range.end;
        let chunk = &seq.prompt_tokens[offset..end];

        // Align each continuation chunk to a 32-token tile boundary on M5+.
        let actual_chunk_len = chunk.len();
        // For non-batching models the scheduler's dummy caches always have
        // offset=0.  Use the prefill_offset (number of tokens already
        // processed) as the KV offset instead, which is accurate regardless
        // of whether the model uses internal or scheduler-managed caches.
        let kv_offset = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found during chunked prefill continuation");
                    return;
                }
            };
            if self.model.supports_batching() {
                caches.first().map_or(0, |c| c.offset)
            } else {
                offset as i32
            }
        };
        let (eff_chunk, pad_mask_opt) = if should_align_prefill() {
            let padded_len = align_to_na_tile(actual_chunk_len);
            if padded_len > actual_chunk_len {
                let mut padded = chunk.to_vec();
                padded.resize(padded_len, 0);
                let mask = create_padded_prefill_mask(
                    actual_chunk_len as i32,
                    padded_len as i32,
                    kv_offset,
                );
                (padded, Some(mask))
            } else {
                (chunk.to_vec(), None)
            }
        } else {
            (chunk.to_vec(), None)
        };

        let eff_len = eff_chunk.len() as i32;
        let input = mlxcel_core::from_slice_i32(&eff_chunk, &[1, eff_len]);
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found during chunked prefill continuation");
                    return;
                }
            };

            let logits = self.model.forward_with_sequence_id(
                &input,
                Some(seq.seq_id),
                caches,
                pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
            );

            // Trim padding positions from KV caches when the chunk was padded.
            if pad_mask_opt.is_some() && eff_chunk.len() > actual_chunk_len {
                let excess = (eff_chunk.len() - actual_chunk_len) as i32;
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }
            logits
        };
        self.sync_sequence_storage(seq.seq_id);

        // H2: enforce the `--max-kv-size` cap after each
        // continuation chunk so a multi-chunk prefill stays bounded across
        // all chunks, not just at the very end. Cheap early-return when no
        // cap is configured.
        self.enforce_max_kv_size_for(seq.seq_id);

        seq.prefill_offset = end;

        tracing::debug!(
            "Chunked prefill: seq {} chunk {offset}..{end}/{total} tokens",
            seq.seq_id,
        );

        if !chunk_range.is_terminal {
            // More chunks remain -- store and yield back to the scheduler
            mlxcel_core::eval(&logits);
            mlxcel_core::clear_memory_cache();
            self.chunked_prefill_seq = Some(seq);
            return;
        }

        // Final chunk -- complete the prefill and sample the first token
        mlxcel_core::clear_memory_cache();

        let eos_tokens =
            merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
        let needs_history = seq.sampling.needs_token_history();
        let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

        self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
    }

    /// Complete a prefill (full or chunked): sample the first token,
    /// handle EOS, and either finish immediately or move to the active
    /// decode batch.
    fn finish_prefill(
        &mut self,
        mut seq: SequenceInfo,
        logits: UniquePtr<mlxcel_core::MlxArray>,
        eos_tokens: Vec<i32>,
        mut token_history: Vec<i32>,
        needs_history: bool,
    ) {
        // apply structured-output mask to the prefill logits
        // before sampling the first token so the very first emitted token
        // already conforms to the schema.
        let logits_for_sampling = if let Some(constraint) = seq.structured.clone() {
            // Read the vocab dimension from the prefill logits so the mask
            // matches the sampler's vocabulary exactly.
            let shape = mlxcel_core::array_shape(&logits);
            let vocab = *shape.last().unwrap_or(&0) as usize;
            match Self::apply_structured_mask(&constraint, mlxcel_core::copy(&logits), vocab) {
                Ok(masked) => masked,
                Err(msg) => {
                    let _ = seq
                        .response_tx
                        .send(GenerateEvent::Error(format!("structured output: {msg}")));
                    if let Err(err) = seq
                        .state
                        .transition_to(SequenceState::Finished(FinishReason::Error(msg)))
                    {
                        tracing::error!("State transition error: {err}");
                    }
                    self.prompt_cache_seq_ctx.remove(&seq.seq_id);
                    self.release_sequence_caches(seq.seq_id);
                    return;
                }
            }
        } else {
            mlxcel_core::copy(&logits)
        };
        // #347: reseed the global MLX RNG to THIS row's own seed at the exact
        // point it samples its first token. `begin_prefill` already seeded once,
        // but a batched cohort runs every row's `begin_prefill` up front before
        // any row reaches `finish_prefill`, so by the time row 0 samples here the
        // global RNG holds the LAST cohort row's seed ("last-seed-wins"). The
        // fused sampler draws from that process-global RNG with no per-call key
        // (`fused_sample` takes only the scalar params), so without this reseed a
        // seeded row's first token would depend on its siblings' seeds. Reseeding
        // here, rather than only in `begin_prefill`, guarantees the seed is live
        // at the exact sample point and makes each row's first token depend only
        // on its own seed. Greedy / `temperature == 0` / `top_k == 1` rows take
        // the argmax path and consume no RNG, so this is a no-op for them. The
        // batched fused DECODE path shares one global-RNG draw across the whole
        // `[B, vocab]` batch and is out of scope here (see issue #347).
        seed_rng_if_needed(&seq.sampling);
        let (first_token_arr, adjusted_logits) =
            sample_token_optimized(&logits_for_sampling, &seq.sampling, &token_history);
        mlxcel_core::eval(&first_token_arr);
        let sampled_first_token = mlxcel_core::item_i32(&first_token_arr);

        // advance the matcher state with the just-sampled token.
        // If consume_token errors, transition the sequence to Finished(Error)
        // and surface a clean SSE error event rather than leaking
        // non-conforming output.
        if let Some(constraint) = seq.structured.clone()
            && let Err(msg) = Self::consume_structured_token(&constraint, sampled_first_token)
        {
            let _ = seq
                .response_tx
                .send(GenerateEvent::Error(format!("structured output: {msg}")));
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Error(msg)))
            {
                tracing::error!("State transition error: {err}");
            }
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            return;
        }

        // thinking-budget override. Qwen3 chat templates prime
        // `<think>\n`, so the first prefill-completion token is already
        // inside the reasoning block when `enter_block_on_start == true`.
        let first_token = Self::apply_thinking_budget(&mut seq.thinking, sampled_first_token);

        seq.first_token_time = Some(Instant::now());

        // if the budget fired and substituted the first token,
        // drop the logprob below (computed against the sampled token) so the
        // streamed metadata stays consistent with the emitted token text.
        let override_fired = first_token != sampled_first_token;

        // Check for immediate EOS
        if eos_tokens.contains(&first_token) {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
            {
                tracing::error!("State transition error: {err}");
            }
            let result = build_generation_result_with_cache(
                String::new(),
                seq.prompt_tokens.len(),
                0,
                seq.created_at.elapsed().as_millis() as u64,
                seq.prefill_start
                    .map(|t| (Instant::now() - t).as_millis() as u64)
                    .unwrap_or(0),
                seq.max_tokens,
                seq.already_cached_tokens,
            );
            tracing::info!(
                prompt_tokens = seq.prompt_tokens.len(),
                cached_tokens = seq.already_cached_tokens,
                saved_ms = 0,
                "prompt-cache: request completed (eos-at-prefill): \
                 cached={}/{} prompt tokens, saved ~0ms",
                seq.already_cached_tokens,
                seq.prompt_tokens.len(),
            );
            let _ = seq.response_tx.send(GenerateEvent::Done(result));
            // Prefill produced a valid KV cache (EOS on turn 1 is a healthy
            // stop). Donate it back so the next turn can reuse the prompt
            // prefix. `generated_tokens` is empty here by construction.
            self.donate_finished_sequence_cache(seq.seq_id, &seq.prompt_tokens, &[], true);
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            return;
        }

        // Optionally compute logprobs for the first token. When the override
        // fired, the sampled token differs from the emitted `first_token`;
        // suppress logprob emission in that case to keep token text and
        // logprob metadata consistent.
        let token_lp = if override_fired {
            None
        } else {
            compute_logprobs(&adjusted_logits, first_token, &seq.logprobs_config)
        };

        seq.generated_tokens.push(first_token);
        if needs_history {
            token_history.push(first_token);
        }

        // Store merged EOS and token history on the sequence so decode_single_step
        // can reuse them without per-step reconstruction.
        seq.merged_eos = eos_tokens;
        seq.token_history = token_history;

        if let Some(new_text) = seq.decode_state.on_token(first_token, &self.tokenizer) {
            let event = match token_lp {
                Some(lp) => GenerateEvent::TokenWithLogprobs(new_text, lp),
                None => GenerateEvent::Token(new_text),
            };
            let _ = seq.response_tx.send(event);
        }

        if seq.generated_tokens.len() >= seq.max_tokens {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }
            // Forward any tail the incremental detokenizer held back (a final
            // token carrying complete text plus a trailing incomplete UTF-8
            // byte) as one last token event before Done, so streaming clients
            // receive it (issue #633).
            if let Some(tail) = seq.decode_state.flush(&self.tokenizer) {
                let _ = seq.response_tx.send(GenerateEvent::Token(tail));
            }
            let cached = seq.already_cached_tokens;
            let result = seq.decode_state.finish_with_cache(
                seq.created_at,
                seq.prompt_tokens.len(),
                seq.max_tokens,
                cached,
            );
            tracing::info!(
                prompt_tokens = seq.prompt_tokens.len(),
                cached_tokens = cached,
                generation_time_ms = result.generation_time_ms,
                "prompt-cache: request completed (max-tokens): \
                 cached={}/{} prompt tokens, total {}ms",
                cached,
                seq.prompt_tokens.len(),
                result.generation_time_ms,
            );
            let _ = seq.response_tx.send(GenerateEvent::Done(result));
            self.donate_finished_sequence_cache(
                seq.seq_id,
                &seq.prompt_tokens,
                &seq.generated_tokens,
                true,
            );
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            return;
        }

        self.prepare_turbo4_delegated_for_sequence_decode(seq.seq_id);

        if let Err(err) = seq.state.transition_to(SequenceState::Decoding) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }

        let prompt_len = seq.prompt_tokens.len() as i32;
        if let Some(cache_set) = self.cache_pool.get_mut(seq.seq_id) {
            cache_set.prompt_len = seq.prompt_tokens.len();
            cache_set.current_offset = prompt_len + 1;
        }

        if let Err(err) = self.active_batch.add(seq) {
            tracing::error!("Failed to add sequence to active batch: {err}");
        }
    }

    // ------------------------------------------------------------------
    // Preemptive eviction
    // ------------------------------------------------------------------

    /// Attempt to evict one sequence from the active batch to make room
    /// for a higher-priority queued request.
    ///
    /// Returns `true` if eviction succeeded (a slot is now free).
    ///
    /// **Streaming caveat:** Tokens already streamed to the client via
    /// `GenerateEvent::Token` are not recalled. When the evicted sequence
    /// is re-prefilled, duplicate tokens may be streamed. This is
    /// acceptable for preemptive scheduling (the client sees a retry)
    /// and is consistent with vLLM's eviction semantics.
    fn try_evict_for_preemption(&mut self) -> bool {
        let victim_id = match self.select_eviction_victim() {
            Some(id) => id,
            None => return false,
        };

        if let Some(mut victim) = self.active_batch.remove(victim_id) {
            tracing::info!(
                "Preempting sequence {} (priority={:?}, {} tokens generated)",
                victim.seq_id,
                victim.priority,
                victim.generated_tokens.len()
            );

            // follow-up: when the victim is a VL request the
            // text model holds a per-sequence MRoPE entry under the old
            // seq id. `release_sequence_caches` below would drop it, but
            // `prepare_request_vlm_embeddings` does NOT re-run on
            // re-prefill so the entry would never be rebuilt under the
            // new id. Take the entry out *before* the release so we can
            // rebind it under the freshly allocated id below.
            //
            // For non-Qwen-VL models / text-only requests this returns
            // an empty snapshot and the rebind is a no-op.
            let mrope_snapshot = self.model.take_qwen_vl_mrope_entry(victim.seq_id);

            // same lifecycle invariant for Gemma 4 E2B/E4B
            // `per_layer_inputs`. The tensor is projected exactly once
            // by `prepare_request_vlm_embeddings` at enqueue time and
            // is consumed at prefill time; preemption-and-reallocate
            // would otherwise drop it and the re-prefill would observe
            // `per_layer_inputs == None` for an E2B/E4B request. Take
            // it out before `release_sequence_caches` drains the map.
            let pli_snapshot = self.model.take_gemma4_per_layer_inputs_entry(victim.seq_id);

            // Issue #85: same for Gemma 3n VLM `per_layer_inputs`.
            // Without this round trip the re-prefill would panic in
            // `Gemma3nVLModel::forward_with_embeddings_and_sequence_id`
            // (per_layer_inputs missing for this sequence).
            let pli3n_snapshot = self
                .model
                .take_gemma3n_per_layer_inputs_entry(victim.seq_id);

            // Release its KV cache
            self.release_sequence_caches(victim.seq_id);

            // Reset the sequence for re-prefill: clear generated tokens,
            // reset decode state, and re-allocate a cache slot.
            //
            // Preemption discards the adopted prefix cache as well — the
            // victim must re-prefill from scratch to stay consistent with
            // the fresh `allocate_sequence_state` that follows.
            victim.generated_tokens.clear();
            victim.generated_text.clear();
            victim.prefill_offset = 0;
            victim.prefill_start_offset = 0;
            victim.already_cached_tokens = 0;
            victim.decode_state = StreamingDecodeState::new(&self.tokenizer, &victim.prompt_tokens);
            victim.token_history.clear();
            victim.merged_eos.clear();

            // Allocate a fresh cache slot
            match self.allocate_sequence_state() {
                Ok(new_id) => {
                    victim.seq_id = new_id;
                    // Re-install the previously-saved MRoPE entry under
                    // the new seq id so re-prefill resolves the same
                    // per-row delta the original prefill computed
                    // (follow-up).
                    self.model
                        .install_qwen_vl_mrope_entry(new_id, mrope_snapshot);
                    // same for Gemma 4 `per_layer_inputs`.
                    // The tensor is reused unchanged across re-prefill
                    // because both depend only on the request's
                    // input_ids (no decode-time updates).
                    self.model
                        .install_gemma4_per_layer_inputs_entry(new_id, pli_snapshot);
                    // Issue #85: same for Gemma 3n `per_layer_inputs`.
                    self.model
                        .install_gemma3n_per_layer_inputs_entry(new_id, pli3n_snapshot);
                    if let Err(err) = victim.state.transition_to(SequenceState::Queued) {
                        tracing::error!("Eviction state transition error: {err}");
                        self.release_sequence_caches(new_id);
                        let _ = victim
                            .response_tx
                            .send(GenerateEvent::Error(format!("Eviction state error: {err}")));
                        return true; // Slot is still freed
                    }
                }
                Err(err) => {
                    tracing::warn!("Re-allocation failed for evicted sequence: {err}");
                    let _ = victim.response_tx.send(GenerateEvent::Error(format!(
                        "Preemption re-queue failed: {err}"
                    )));
                    // The snapshots are dropped here; the request is
                    // about to error out so the entries have no further
                    // consumer.
                    drop(mrope_snapshot);
                    drop(pli_snapshot);
                    return true; // Slot is still freed
                }
            }

            // Re-queue the evicted sequence (it will re-prefill when admitted)
            if let Err(rejected) = self.prefill_queue.enqueue(victim) {
                self.release_sequence_caches(rejected.seq_id);
                let _ = rejected.response_tx.send(GenerateEvent::Error(
                    "Preemption re-queue failed: prefill queue full".to_string(),
                ));
            }

            self.batch_metrics.record_preemption();
            true
        } else {
            false
        }
    }

    /// Select the eviction victim based on the configured policy.
    ///
    /// follow-up: sequences with an attached structured-output
    /// constraint are excluded from the candidate set. Preemption resets
    /// `generated_tokens`, the streaming decoder, and the KV cache, but the
    /// `llguidance` matcher carries grammar progress that cannot be safely
    /// rewound — re-prefill would advance the matcher from a state that
    /// reflects the discarded tokens, producing either an empty mask error
    /// or silent grammar mis-advance. Skipping these sequences trades a
    /// rare scheduling stall for correctness; if no other candidate is
    /// available, `try_evict_for_preemption` falls through to its existing
    /// "no candidate" path and the new request stays queued.
    fn select_eviction_victim(&self) -> Option<SequenceId> {
        match self.preemption_policy {
            PreemptionPolicy::LongestFirst => {
                // Evict the sequence with the most generated tokens
                self.active_batch
                    .iter_sequences()
                    .filter(|seq| seq.structured.is_none())
                    .max_by_key(|seq| seq.generated_tokens.len())
                    .map(|seq| seq.seq_id)
            }
            PreemptionPolicy::LowestPriority => {
                // Evict the lowest-priority sequence; break ties by longest
                self.active_batch
                    .iter_sequences()
                    .filter(|seq| seq.structured.is_none())
                    .min_by(|a, b| {
                        a.priority
                            .cmp(&b.priority)
                            .then_with(|| b.generated_tokens.len().cmp(&a.generated_tokens.len()))
                    })
                    .map(|seq| seq.seq_id)
            }
        }
    }

    // ------------------------------------------------------------------
    // Decode execution (batched when B > 1, sequential fallback otherwise)
    // ------------------------------------------------------------------

    /// Run one decode step for the active sequences.
    fn execute_decode_step(&mut self, seq_ids: &[SequenceId]) {
        // Filter-to-empty guard: a zero-sized decode step is a no-op, not a
        // failure. The observability counter already reflects length 0 for
        // caller-side traceability, so we still record it, then skip the
        // dispatch entirely. This matches the null-guard pattern upstream
        // `mlx-lm` added to `BatchKVCache.filter` when the filtered index
        // list is empty.
        if seq_ids.is_empty() {
            self.batch_observability.record_decode_step(0);
            return;
        }

        let _span = tracing::info_span!("decode_step", batch_size = seq_ids.len(),).entered();
        self.batch_observability.record_decode_step(seq_ids.len());

        // Lookahead async_eval pipeline (issue #632). Eligible batches overlap
        // the next forward with the current tick's host bookkeeping; anything
        // outside the narrow eligibility window (see `lookahead_params`) runs
        // the untouched synchronous path. `run_decode_tick` owns the state
        // machine and always leaves the caches in the synchronous-decode
        // invariant on any teardown, so the fallback is bit-exact.
        self.run_decode_tick(seq_ids);
    }

    /// Raw synchronous decode dispatch for `seq_ids` (no observability
    /// recording; the caller already counted the step). B=1 and non-batching
    /// models take the per-sequence path; larger batches take the batched
    /// forward. This is the exact pre-#632 behavior and the pipeline's
    /// guaranteed fallback.
    fn dispatch_sync_decode(&mut self, seq_ids: &[SequenceId]) {
        if seq_ids.len() <= 1 || !self.model.supports_batching() {
            for &seq_id in seq_ids {
                self.decode_single_step(seq_id);
            }
            return;
        }
        self.execute_batched_decode(seq_ids);
    }

    /// Drive one decode tick through the lookahead pipeline state machine.
    ///
    /// States:
    /// - A prebuilt lookahead for the identical id set and still-safe
    ///   conditions -> steady pipelined commit + re-prime.
    /// - A prebuilt lookahead that is stale (id set changed) or now unsafe ->
    ///   discard (trim + drop), run synchronously, then re-prime if eligible.
    /// - No prebuilt lookahead -> run synchronously, then prime if eligible
    ///   (bootstrap).
    fn run_decode_tick(&mut self, seq_ids: &[SequenceId]) {
        let params = self.lookahead_params(seq_ids);

        match self.decode_lookahead.take() {
            Some(la) if la.ids == seq_ids && params.is_some() && self.lookahead_safe() => {
                self.pipelined_steady_decode(la, seq_ids, &params.unwrap());
            }
            Some(la) => {
                // Stale id set or no longer eligible/safe: no step n+1 prime was
                // issued, so undo the one speculative KV position and fall back
                // to a clean sync step.
                self.apply_lookahead_trim(&la.ids, lookahead_teardown_positions(false));
                drop(la);
                self.dispatch_sync_decode(seq_ids);
                self.maybe_prime_lookahead(seq_ids);
            }
            None => {
                self.dispatch_sync_decode(seq_ids);
                self.maybe_prime_lookahead(seq_ids);
            }
        }
    }

    /// Batched decode-storage context for the active backend. Shared by the
    /// synchronous batched decode and the lookahead prime so both drive the
    /// identical dense / native-paged execution path.
    fn decode_batch_context(&self) -> DecodeBatchContext {
        match self.decode_storage_backend {
            DecodeStorageBackend::Auto | DecodeStorageBackend::Dense => {
                debug_assert_ne!(
                    self.decode_storage_backend,
                    DecodeStorageBackend::Auto,
                    "scheduler should normalize decode storage backend before decode dispatch"
                );
                DecodeBatchContext::dense()
            }
            DecodeStorageBackend::Paged => DecodeBatchContext {
                storage_backend: CoreDecodeStorageBackend::Paged,
                paged_block_size: DEFAULT_PAGED_BLOCK_SIZE as i32,
                use_native_paged_kernel: true,
            },
        }
    }

    /// Whether the active decode batch is eligible for the lookahead pipeline
    /// this tick, returning the shared fused sampling params on success. A
    /// `None` return routes the tick to the synchronous path. The gate is
    /// deliberately narrow: it reuses the batched-fused predicate (which
    /// already rejects penalties/token-history, token bias, structured-output
    /// masks, thinking-budget overrides, and per-token logprobs) and further
    /// requires a trimmable KV tail (dense or pool-backed paged; model-owned
    /// SSM / hybrid / mixed-cache backends stay synchronous), no
    /// `--max-kv-size`, no speculative dispatch, and `MLXCEL_FORCE_SYNC` unset.
    fn lookahead_params(&self, seq_ids: &[SequenceId]) -> Option<FusedSampleParams> {
        if self.lookahead_force_sync {
            return None;
        }
        // Speculative decoding drives its own decode loop.
        if self.should_dispatch_speculative() {
            return None;
        }
        // --max-kv-size trims the live window mid-decode (trim_front); keep
        // those runs synchronous so the speculative +1 accounting stays simple.
        if self.max_kv_size.is_some() {
            return None;
        }
        // The batched fused gate rejects every per-row feature the device
        // feedback path cannot honor; reusing it keeps pipeline sampling
        // bit-identical to the fast path it accelerates.
        let params = self.batched_decode_fused_params(seq_ids)?;
        for &seq_id in seq_ids {
            let seq = self.active_batch.get(seq_id)?;
            // Loop detection needs a post-commit host scan the steady path
            // skips (off by default, so no common-case cost).
            if seq.sampling.loop_detection.is_enabled() {
                return None;
            }
            // Dense and pool-backed paged sequences both have a trimmable KV
            // tail (dense via KVCache::trim, paged via the pool rewind API in
            // apply_lookahead_trim). Model-owned families (SSM / hybrid /
            // mixed-cache) carry no such tail, so they stay synchronous.
            match self.cache_pool.get(seq_id) {
                Some(set)
                    if matches!(
                        set.backend,
                        SequenceStateBackend::DenseKvCache | SequenceStateBackend::PagedKvCache
                    ) => {}
                _ => return None,
            }
        }
        Some(params)
    }

    /// Conditions under which priming the next forward is safe: the next tick
    /// will decode this identical id set. False on a pending admission
    /// (queue non-empty), a chunked-prefill interleave, or a pending
    /// preemption, each of which mutates batch membership next tick.
    fn lookahead_safe(&self) -> bool {
        lookahead_pipeline_safe(
            self.prefill_queue.is_empty(),
            self.chunked_prefill_seq.is_some(),
            self.should_preempt(),
        )
    }

    /// Trim the one speculative KV position the prime forward appended from
    /// each sequence, restoring the synchronous-decode invariant (the last
    /// committed token is not yet in the KV cache). Called on every pipeline
    /// teardown before the synchronous path, a completion, or a prompt-cache
    /// donation runs, so slot reuse and detach always see clean caches.
    ///
    /// Pool-backed paged sequences rewind through the pool block table (one
    /// token per layer, releasing any tail block); dense (and dense-natural
    /// paged mirror) sequences trim the dense KV tail and re-mirror the shorter
    /// length into the paged bookkeeping state.
    ///
    /// `positions` is the number of speculative appends to unwind: `1` for a
    /// teardown before the step-n+1 prime forward has run (admission,
    /// preemption, stale id set, cancellation seen in `finalize_completed`),
    /// `2` for the steady-tick teardown that already issued the step-n+1 prime
    /// (both the step-n and step-n+1 appends).
    fn apply_lookahead_trim(&mut self, ids: &[SequenceId], positions: usize) {
        if positions == 0 {
            return;
        }
        let num_layers = self.model.num_layers();
        for &seq_id in ids {
            let paged_backed = self
                .cache_pool
                .get(seq_id)
                .map(|s| s.caches.iter().any(|c| c.is_paged_backed()))
                .unwrap_or(false);
            if paged_backed {
                for layer in 0..num_layers {
                    // A failed rewind silently leaks the speculative KV
                    // position(s), which would corrupt a later donation of this
                    // sequence's cache; surface it so the leak is diagnosable.
                    if let Err(err) = self
                        .cache_pool
                        .rewind_paged_tokens(seq_id, layer, positions)
                    {
                        tracing::warn!(
                            seq_id = %seq_id,
                            layer,
                            positions,
                            "lookahead teardown: paged rewind failed, speculative KV \
                             position may leak: {err}"
                        );
                    }
                }
            } else if let Some(caches) = self.cache_pool.get_caches_mut(seq_id) {
                let want = positions as i32;
                for (layer, cache) in caches.iter_mut().enumerate() {
                    // KVCache::trim clamps to the live window and returns the
                    // count actually removed; a short trim means a speculative
                    // position was not unwound (e.g. an unexpectedly short cache),
                    // which would desync the KV against generated_tokens.
                    let trimmed = cache.trim(want);
                    if trimmed != want {
                        tracing::warn!(
                            seq_id = %seq_id,
                            layer,
                            requested = want,
                            trimmed,
                            "lookahead teardown: dense trim removed fewer positions \
                             than requested, KV may be out of sync"
                        );
                    }
                }
                // Re-mirror the shorter dense length into any paged bookkeeping
                // (no-op for a pure dense pool).
                self.sync_sequence_storage(seq_id);
            }
        }
    }

    /// Tear down any live lookahead: trim the speculative KV position from each
    /// of its sequences and drop the prebuilt tokens. Idempotent no-op when the
    /// pipeline is idle. Invoked before admission / preemption (`run`) and
    /// before completion / cancellation donation (`finalize_completed`).
    fn discard_lookahead(&mut self) {
        if let Some(la) = self.decode_lookahead.take() {
            // A stored lookahead carries exactly one speculative append per
            // sequence (the prime forward that produced its tokens); no step
            // n+1 prime has been issued on this teardown path.
            //
            // Unlike the steady finishing path, no pre-trim eval is needed
            // here even if that prime is still in flight: KVCache::trim only
            // adjusts a host-tracked offset, and all decode work runs on the
            // single generation stream, so the lazy slice the trim enqueues is
            // dependency-ordered after the append. The finishing path's eval
            // is defensive, not required for safety.
            self.apply_lookahead_trim(&la.ids, lookahead_teardown_positions(false));
        }
    }

    /// Prime the next forward for `seq_ids` after a synchronous step (pipeline
    /// bootstrap). Reads each sequence's last committed token from host state,
    /// builds the `[B, 1]` input, and schedules the forward + fused sample.
    /// No-op unless eligible, safe, and every sequence is still live.
    fn maybe_prime_lookahead(&mut self, seq_ids: &[SequenceId]) {
        let Some(params) = self.lookahead_params(seq_ids) else {
            return;
        };
        if !self.lookahead_safe() {
            return;
        }
        // A sequence that just finished (EOS / length) leaves the batch next
        // tick; do not prime across a membership change.
        let mut last_tokens: Vec<i32> = Vec::with_capacity(seq_ids.len());
        for &seq_id in seq_ids {
            match self.active_batch.get(seq_id) {
                Some(seq) if !seq.state.is_finished() => {
                    last_tokens.push(*seq.generated_tokens.last().unwrap_or(&0));
                }
                _ => return,
            }
        }
        let input = mlxcel_core::from_slice_i32(&last_tokens, &[seq_ids.len() as i32, 1]);
        self.decode_lookahead = self.prime_lookahead_with_input(seq_ids, &input, &params);
    }

    /// Run one forward for `seq_ids` on `input` (`[B, 1]`), fused-sample the
    /// next tokens on-device, schedule them with `async_eval`, and return the
    /// prebuilt step. The forward appends one speculative KV position per
    /// sequence (undone by [`Self::apply_lookahead_trim`]). Returns `None` if a
    /// sequence's caches vanished. The caller decides whether to keep the step
    /// (store it in `decode_lookahead`) or unwind it.
    fn prime_lookahead_with_input(
        &mut self,
        seq_ids: &[SequenceId],
        input: &mlxcel_core::MlxArray,
        params: &FusedSampleParams,
    ) -> Option<DecodeLookahead> {
        let logits = self.lookahead_forward(seq_ids, input)?;
        let last_logits = mlxcel_core::slice_last_logits(&logits);
        let tokens = mlxcel_core::fused_sample(
            &last_logits,
            params.temperature,
            params.top_k,
            params.top_p,
            params.min_p,
        );
        // Schedule the sampled tokens (and thus the whole forward graph) without
        // reading them to host, so the GPU runs ahead while the caller returns
        // to the scheduler loop and reads the PREVIOUS step's tokens.
        mlxcel_core::async_eval(&tokens);
        Some(DecodeLookahead {
            ids: seq_ids.to_vec(),
            tokens,
        })
    }

    /// Forward pass for the lookahead pipeline, mirroring the synchronous decode
    /// forward exactly: the B=1 per-sequence path
    /// ([`Self::decode_single_step`]) or the batched path
    /// ([`Self::execute_batched_decode`]) with the same decode-storage context.
    /// Returns `None` if a sequence's caches vanished (the caller then skips
    /// priming).
    fn lookahead_forward(
        &mut self,
        seq_ids: &[SequenceId],
        input: &mlxcel_core::MlxArray,
    ) -> Option<UniquePtr<mlxcel_core::MlxArray>> {
        let logits = if seq_ids.len() == 1 {
            let seq_id = seq_ids[0];
            let caches = self.cache_pool.get_caches_mut(seq_id)?;
            self.model
                .forward_with_sequence_id(input, Some(seq_id), caches, None)
        } else {
            let decode_context = self.decode_batch_context();
            let mut batch_caches = self.cache_pool.get_batch_caches_mut(seq_ids).ok()?;
            let logits = self.model.forward_batched_with_context_and_ids(
                input,
                Some(seq_ids),
                &mut batch_caches,
                None,
                Some(&decode_context),
            );
            drop(batch_caches);
            logits
        };
        for &seq_id in seq_ids {
            self.sync_sequence_storage(seq_id);
        }
        Some(logits)
    }

    /// Steady pipelined decode, ordered exactly like the CLI generation loop
    /// (`generate.rs`) so the GPU never idles on the host read:
    ///
    /// 1. FIRST build and `async_eval` step n+1 from `la.tokens` fed back
    ///    device-side (no host knowledge needed). The GPU starts the next
    ///    forward immediately.
    /// 2. THEN read step n's tokens to host (this blocks on the PREVIOUS tick's
    ///    prime forward, which by now has finished, while step n+1 runs on the
    ///    GPU) and run the finish pre-check.
    /// 3. If a row finishes (EOS / length / cancel) or the shape is off, unwind
    ///    BOTH speculative appends (step n and the just-issued step n+1) and
    ///    re-run the tick synchronously, so completion / donation flows through
    ///    the untouched sync path from a clean cache state.
    /// 4. Otherwise commit step n and keep the step n+1 prebuilt step.
    fn pipelined_steady_decode(
        &mut self,
        la: DecodeLookahead,
        seq_ids: &[SequenceId],
        params: &FusedSampleParams,
    ) {
        // Step 1: speculatively build + schedule step n+1 FIRST. Feed la.tokens
        // ([B]) back as the next [B, 1] input device-side (reshape + int32 cast
        // to match the synchronous from_slice_i32 dtype), keeping the GPU busy
        // through the host read below. This appends a second speculative KV
        // position per sequence (the overshoot the issue accepts).
        let col = mlxcel_core::reshape_token_for_forward(&la.tokens);
        let next_input = mlxcel_core::astype(&col, mlxcel_core::dtype::INT32);
        let next = self.prime_lookahead_with_input(seq_ids, &next_input, params);

        // Step 2: read step n's tokens to host (the sync point) and finish-check.
        let toks = lookahead_tokens_to_host(&la.tokens);
        let mut finishing = toks.len() != seq_ids.len();
        if !finishing {
            for (i, &seq_id) in seq_ids.iter().enumerate() {
                let Some(seq) = self.active_batch.get(seq_id) else {
                    finishing = true;
                    break;
                };
                if lookahead_token_finishes(
                    toks[i],
                    seq.generated_tokens.len(),
                    seq.max_tokens,
                    &seq.merged_eos,
                    seq.cancelled.load(Ordering::Relaxed),
                ) {
                    finishing = true;
                    break;
                }
            }
        }

        if finishing {
            // Step 3: tear down. Sync the in-flight step n+1 forward so its
            // kernels are not still writing KV when we rewind, then unwind both
            // speculative appends (step n from the previous prime plus step n+1
            // when it was actually issued) back to the synchronous-decode
            // invariant and re-run the tick synchronously.
            if let Some(nla) = &next {
                mlxcel_core::eval(&nla.tokens);
            }
            let positions = lookahead_teardown_positions(next.is_some());
            self.apply_lookahead_trim(&la.ids, positions);
            drop(next);
            drop(la);
            // The sync re-dispatch below re-samples step n's token. fused_sample
            // draws from MLX's global RNG (random::categorical without an
            // explicit key), so at temperature > 0 the re-drawn token can
            // differ from the discarded lookahead sample that triggered this
            // finish pre-check; greedy (temp 0) is unaffected, matching the
            // byte-equivalence gate. Bounded to one token per completing
            // request, and stochastic runs carry no cross-mode determinism
            // guarantee.
            self.dispatch_sync_decode(seq_ids);
            self.maybe_prime_lookahead(seq_ids);
            return;
        }

        // Step 4: every row continues. Commit step n (reusing the batched
        // fast-path bookkeeping; no finish can fire after the pre-check) and
        // keep the already-primed step n+1.
        drop(la);
        self.apply_fused_decode_tokens(seq_ids, &toks);
        self.decode_lookahead = next;
        self.batch_observability.record_lookahead_step();
    }

    /// Batched decode: one forward_batched() call for all active sequences.
    ///
    /// # Null/empty-cache safety
    ///
    /// Early-exits on `seq_ids.is_empty()`. Though the scheduler's current
    /// [`Self::decide_action`] never produces a `Decode(ids)` action with an
    /// empty list (it returns [`BatchSchedulerAction::Idle`] first), this
    /// guard makes the method robust against future policy changes and any
    /// direct caller. Dispatching a zero-batch forward pass would otherwise
    /// materialize an empty `[0, 1]` input tensor and invoke the model
    /// kernel with no work to do, which is both wasteful and potentially
    /// undefined behavior in downstream MLX kernels.
    ///
    /// This mirrors the upstream `mlx-lm` `BatchKVCache.filter` / `extend`
    /// null-guards that prevent cache operations from crashing when all
    /// sequences have been filtered out of the batch.
    fn execute_batched_decode(&mut self, seq_ids: &[SequenceId]) {
        if seq_ids.is_empty() {
            // Filter-to-empty case: nothing to do. Bookkeeping is handled by
            // the caller (`execute_decode_step`) via its own length guard.
            return;
        }

        let b = seq_ids.len();

        // trim per-sequence plain KVCache layers before the batched
        // forward pass so all sequences stay within the --max-kv-size bound.
        // Sliding-window (model-internal RotatingKVCache) and Turbo-quantized
        // caches are unaffected (trim_front returns 0 for Turbo modes).
        for &seq_id in seq_ids {
            self.enforce_max_kv_size_for(seq_id);
        }

        let mut last_tokens: Vec<i32> = Vec::with_capacity(b);

        for &seq_id in seq_ids {
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => {
                    self.execute_decode_step_sequential_remaining(seq_ids, last_tokens.len());
                    return;
                }
            };
            last_tokens.push(*seq.generated_tokens.last().unwrap_or(&0));
        }

        let input = mlxcel_core::from_slice_i32(&last_tokens, &[b as i32, 1]);

        debug_assert!(
            {
                let unique: HashSet<_> = seq_ids.iter().collect();
                unique.len() == seq_ids.len()
            },
            "execute_batched_decode: duplicate SequenceId in seq_ids"
        );

        let decode_context = self.decode_batch_context();
        let mut batch_caches = match self.cache_pool.get_batch_caches_mut(seq_ids) {
            Ok(caches) => caches,
            Err(err) => {
                tracing::error!("{err} during batched decode");
                return;
            }
        };

        let logits = self.model.forward_batched_with_context_and_ids(
            &input,
            Some(seq_ids),
            &mut batch_caches,
            None,
            Some(&decode_context),
        );
        drop(batch_caches);

        for &seq_id in seq_ids {
            self.sync_sequence_storage(seq_id);
        }

        // Fast path: when every active row shares a fused-compatible sampling
        // config and none needs a structured-output mask, a thinking-budget
        // override, or a per-token logprobs payload, sample all B rows in ONE
        // fused `[B, vocab] -> [B]` dispatch + eval instead of B per-row
        // slice/sample/eval/extract round trips. The per-row loop below stays
        // the exact fallback for every other case (structured output,
        // row-specific logprobs, token-bias observability, thinking budgets,
        // mixed sampling configs).
        if let Some(params) = self.batched_decode_fused_params(seq_ids) {
            let tokens = batched_fused_sample(&logits, &params);
            self.apply_fused_decode_tokens(seq_ids, &tokens);
            return;
        }

        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let seq_logits =
                mlxcel_core::slice(&logits, &[i as i32, 0, 0], &[i as i32 + 1, 1, i32::MAX]);

            // when the sequence has a structured-output
            // constraint, apply the schema mask to the per-sequence logits
            // before sampling. Failures here surface as a clean
            // FinishReason::Error rather than silent non-conforming output.
            let constraint_clone = self
                .active_batch
                .get_mut(seq_id)
                .and_then(|s| s.structured.clone());
            let logits_for_sampling = if let Some(constraint) = constraint_clone.as_ref() {
                let shape = mlxcel_core::array_shape(&seq_logits);
                let vocab = *shape.last().unwrap_or(&0) as usize;
                match Self::apply_structured_mask(constraint, mlxcel_core::copy(&seq_logits), vocab)
                {
                    Ok(masked) => masked,
                    Err(msg) => {
                        Self::abort_sequence_with_error(
                            self.active_batch.get_mut(seq_id),
                            "structured output",
                            &msg,
                        );
                        continue;
                    }
                }
            } else {
                mlxcel_core::copy(&seq_logits)
            };

            // Use cached token_history (incrementally maintained) instead of
            // rebuilding per step. Use cached merged_eos computed once at prefill.
            //
            // follow-up: we capture `sampled` separately from
            // `final_id` so the structured-output matcher (below) can be
            // advanced by the *pre-override* token. The matcher's mask
            // describes which token ids are grammatically legal at this
            // step; feeding it the post-override forced `</think>` would
            // hand it a token outside its allowed set and cause a parser
            // error or silent mis-advance.
            let (sampled_token, token_val, token_lp) = {
                let seq = match self.active_batch.get_mut(seq_id) {
                    Some(s) => s,
                    None => continue,
                };
                // Penalty rows use the incremental per-sequence sampler state
                // (lazily created); the no-penalty rows that reach this per-row
                // fallback take the original rebuild-free path unchanged.
                let (token_arr, adjusted_logits) = if seq.sampling.needs_token_history() {
                    sample_token_optimized_with_state(
                        &logits_for_sampling,
                        &seq.sampling,
                        &seq.token_history,
                        &mut seq.sampler_state,
                    )
                } else {
                    sample_token_optimized(&logits_for_sampling, &seq.sampling, &seq.token_history)
                };
                mlxcel_core::eval(&token_arr);
                let sampled = mlxcel_core::item_i32(&token_arr);
                // apply the thinking-budget override first so that
                // when the override fires (sampled != final_id) we can skip
                // the log-softmax work entirely. The logprob metadata would
                // be dropped anyway because the emitted `</think>` differs
                // from the token the logits describe, so computing it first
                // is wasted GPU work on the decode hot path.
                let final_id = Self::apply_thinking_budget(&mut seq.thinking, sampled);
                let lp = if final_id == sampled {
                    compute_logprobs(&adjusted_logits, sampled, &seq.logprobs_config)
                } else {
                    // Override fired; token text and logprob metadata must
                    // stay consistent, so drop the logprob for this step.
                    None
                };
                (sampled, final_id, lp)
            };

            // advance the matcher state with the *pre-override*
            // sampled token (`sampled_token`), not the post-override
            // `token_val`. The matcher derived its mask from the unaltered
            // logits, so feeding it `final_id` after a thinking-budget
            // override would hand it a token outside its allowed set and
            // either cause a parser error or silently mis-advance. Mirrors
            // the pattern in `finish_prefill` which uses
            // `sampled_first_token`.
            //
            // If `consume_token` fails (matcher hit an error state),
            // transition the sequence to `Finished(Error)` and skip
            // emission so non-conforming output never reaches the client.
            if let Some(constraint) = constraint_clone
                && let Err(msg) = Self::consume_structured_token(&constraint, sampled_token)
            {
                Self::abort_sequence_with_error(
                    self.active_batch.get_mut(seq_id),
                    "structured output",
                    &msg,
                );
                continue;
            }

            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => continue,
            };

            if seq.merged_eos.contains(&token_val) {
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Stop))
                {
                    tracing::error!("State transition error: {err}");
                }
                continue;
            }

            seq.generated_tokens.push(token_val);

            // Incrementally update token_history
            if seq.sampling.needs_token_history() {
                seq.token_history.push(token_val);
            }

            if let Some(new_text) = seq.decode_state.on_token(token_val, &self.tokenizer) {
                let event = match token_lp {
                    Some(lp) => GenerateEvent::TokenWithLogprobs(new_text, lp),
                    None => GenerateEvent::Token(new_text),
                };
                let _ = seq.response_tx.send(event);
            }

            if seq.generated_tokens.len() >= seq.max_tokens
                && let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }

            // Loop / repetition guard (issue #432): end early when the raw
            // generated stream collapses into a short repeated pattern. Skip if
            // the length limit already finished this sequence; the detector is
            // a zero-overhead no-op when loop detection is disabled (default).
            if !seq.state.is_finished()
                && mlxcel_core::detect_repetition_loop(
                    &seq.generated_tokens,
                    &seq.sampling.loop_detection,
                )
            {
                match seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
                {
                    Ok(()) => tracing::info!(
                        generated = seq.generated_tokens.len(),
                        "loop detection: ending generation early (repetition loop)"
                    ),
                    Err(err) => tracing::error!("State transition error: {err}"),
                }
            }

            // Periodic cache clearing, backend-aware cadence (#627): disabled by
            // default on CUDA (clear churns the pool and defeats CUDA-graph
            // reuse, mlx#2358), 256 on Metal, MLXCEL_CACHE_CLEAR_INTERVAL overrides.
            if mlxcel_core::memory::should_clear_cache_at(
                seq.generated_tokens.len(),
                mlxcel_core::memory::cache_clear_interval(),
            ) {
                mlxcel_core::clear_memory_cache();
            }

            if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
                cache_set.current_offset += 1;
            }
        }
    }

    /// Decide whether the batched decode fast path applies to `seq_ids`.
    ///
    /// Returns `Some(params)` with the shared scalar sampling parameters when
    /// EVERY active row can be sampled by a single `[B, vocab] -> [B]` fused
    /// dispatch: all rows share the same scalar parameters, none needs a
    /// history-based penalty or token bias, and none needs a structured-output
    /// mask, a thinking-budget override, or a per-token logprobs payload. Any
    /// row that needs per-row treatment returns `None`, which routes the caller
    /// to the unchanged per-row fallback loop.
    ///
    /// The per-row obligations map onto the generic predicate
    /// [`mlxcel_core::sampling::row_supports_fused_batch`] as: structured-output
    /// mask -> `needs_logit_mask` (`seq.structured`); thinking-budget override
    /// -> `needs_token_override` (`seq.thinking`); per-token logprobs ->
    /// `needs_per_token_payload` (`seq.logprobs_config`).
    fn batched_decode_fused_params(&self, seq_ids: &[SequenceId]) -> Option<FusedSampleParams> {
        let mut shared: Option<FusedSampleParams> = None;
        for &seq_id in seq_ids {
            // A row that vanished from the batch forces the per-row fallback,
            // which carries its own missing-sequence guards.
            let seq = self.active_batch.get(seq_id)?;
            if !row_supports_fused_batch(
                &seq.sampling,
                seq.structured.is_some(),
                !seq.thinking.is_disabled(),
                seq.logprobs_config.enabled,
            ) {
                return None;
            }
            let params = FusedSampleParams::from_config(&seq.sampling);
            match shared {
                None => shared = Some(params),
                Some(first) if !first.matches(&params) => return None,
                Some(_) => {}
            }
        }
        shared
    }

    /// Bookkeeping for the batched fused fast path.
    ///
    /// Consumes the `[B]` token ids produced by
    /// [`mlxcel_core::sampling::batched_fused_sample`] and drives each
    /// sequence's EOS check, token history, streaming decode, length limit,
    /// periodic cache clear, and cache-offset advance. This mirrors the tail of
    /// the per-row loop in [`Self::execute_batched_decode`] minus the per-row
    /// sampling, structured-output, thinking-budget, and logprobs work that
    /// [`Self::batched_decode_fused_params`] already excluded. `tokens[i]` is
    /// the id sampled for `seq_ids[i]`.
    fn apply_fused_decode_tokens(&mut self, seq_ids: &[SequenceId], tokens: &[i32]) {
        debug_assert_eq!(
            seq_ids.len(),
            tokens.len(),
            "apply_fused_decode_tokens: token count must match seq_ids"
        );
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let token_val = tokens[i];
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => continue,
            };

            if seq.merged_eos.contains(&token_val) {
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Stop))
                {
                    tracing::error!("State transition error: {err}");
                }
                continue;
            }

            seq.generated_tokens.push(token_val);

            // The gate guarantees no penalty config reaches the fast path, so
            // this is a no-op today; it is kept for exact parity with the
            // per-row loop in case the gate ever admits history-tracking
            // configs.
            if seq.sampling.needs_token_history() {
                seq.token_history.push(token_val);
            }

            if let Some(new_text) = seq.decode_state.on_token(token_val, &self.tokenizer) {
                let _ = seq.response_tx.send(GenerateEvent::Token(new_text));
            }

            if seq.generated_tokens.len() >= seq.max_tokens
                && let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }

            // Loop / repetition guard (issue #432): end early when the raw
            // generated stream collapses into a short repeated pattern. Skip if
            // the length limit already finished this sequence; the detector is
            // a zero-overhead no-op when loop detection is disabled (default).
            if !seq.state.is_finished()
                && mlxcel_core::detect_repetition_loop(
                    &seq.generated_tokens,
                    &seq.sampling.loop_detection,
                )
            {
                match seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
                {
                    Ok(()) => tracing::info!(
                        generated = seq.generated_tokens.len(),
                        "loop detection: ending generation early (repetition loop)"
                    ),
                    Err(err) => tracing::error!("State transition error: {err}"),
                }
            }

            // Periodic cache clearing, backend-aware cadence (#627): disabled by
            // default on CUDA (clear churns the pool and defeats CUDA-graph
            // reuse, mlx#2358), 256 on Metal, MLXCEL_CACHE_CLEAR_INTERVAL overrides.
            if mlxcel_core::memory::should_clear_cache_at(
                seq.generated_tokens.len(),
                mlxcel_core::memory::cache_clear_interval(),
            ) {
                mlxcel_core::clear_memory_cache();
            }

            if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
                cache_set.current_offset += 1;
            }
        }
    }

    fn execute_decode_step_sequential_remaining(
        &mut self,
        seq_ids: &[SequenceId],
        start_from: usize,
    ) {
        for &seq_id in &seq_ids[start_from..] {
            self.decode_single_step(seq_id);
        }
    }

    fn decode_single_step(&mut self, seq_id: SequenceId) {
        let last_token = {
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => return,
            };
            *seq.generated_tokens.last().unwrap_or(&0)
        };

        // trim the oldest tokens from plain KVCache layers so the
        // live window stays within the configured --max-kv-size bound before
        // each decode forward pass. Sliding-window layers are managed by the
        // model and bypass this pool path; Turbo-quantized caches silently skip
        // the trim (KVCache::trim_front returns 0 for Turbo modes).
        self.enforce_max_kv_size_for(seq_id);

        let input = mlxcel_core::from_slice_i32(&[last_token], &[1, 1]);
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq_id) {
                Some(c) => c,
                None => {
                    tracing::error!("Cache not found for {seq_id} during decode");
                    return;
                }
            };
            self.model
                .forward_with_sequence_id(&input, Some(seq_id), caches, None)
        };
        self.sync_sequence_storage(seq_id);

        // apply structured-output mask to per-step logits when
        // the sequence has an attached constraint. Errors abort the
        // sequence cleanly rather than emitting non-conforming output.
        let constraint_clone = self
            .active_batch
            .get_mut(seq_id)
            .and_then(|s| s.structured.clone());
        let logits_for_sampling = if let Some(constraint) = constraint_clone.as_ref() {
            let shape = mlxcel_core::array_shape(&logits);
            let vocab = *shape.last().unwrap_or(&0) as usize;
            match Self::apply_structured_mask(constraint, mlxcel_core::copy(&logits), vocab) {
                Ok(masked) => masked,
                Err(msg) => {
                    Self::abort_sequence_with_error(
                        self.active_batch.get_mut(seq_id),
                        "structured output",
                        &msg,
                    );
                    return;
                }
            }
        } else {
            mlxcel_core::copy(&logits)
        };

        // Use cached token_history from SequenceInfo (incrementally maintained)
        // and cached merged_eos (computed once during prefill) to avoid
        // per-step allocation and reconstruction overhead.
        //
        // follow-up: we capture `sampled` separately from
        // `final_id`. The structured-output matcher (below) must be
        // advanced by the pre-override token because its mask was derived
        // from the unaltered logits; passing the post-override forced
        // `</think>` would feed it a token outside its allowed set.
        let (sampled_token, token_val, token_lp) = {
            let seq = self.active_batch.get_mut(seq_id).unwrap();
            // Penalty sequences use the incremental per-sequence sampler state
            // (lazily created); a no-penalty sequence takes the original
            // rebuild-free path unchanged.
            let (token_arr, adjusted_logits) = if seq.sampling.needs_token_history() {
                sample_token_optimized_with_state(
                    &logits_for_sampling,
                    &seq.sampling,
                    &seq.token_history,
                    &mut seq.sampler_state,
                )
            } else {
                sample_token_optimized(&logits_for_sampling, &seq.sampling, &seq.token_history)
            };
            mlxcel_core::eval(&token_arr);
            let sampled = mlxcel_core::item_i32(&token_arr);
            // apply the thinking-budget override first so that
            // when the override fires the log-softmax work is skipped — the
            // logprob metadata for the sampled token would be dropped anyway
            // (token text and logprob `token_id` must stay consistent), so
            // computing it up-front wastes GPU time on every override step.
            let final_id = Self::apply_thinking_budget(&mut seq.thinking, sampled);
            let lp = if final_id == sampled {
                compute_logprobs(&adjusted_logits, sampled, &seq.logprobs_config)
            } else {
                None
            };
            (sampled, final_id, lp)
        };

        // advance the matcher state with the *pre-override*
        // sampled token. See the parallel comment in
        // `execute_batched_decode` for why this must not be `token_val`.
        if let Some(constraint) = constraint_clone
            && let Err(msg) = Self::consume_structured_token(&constraint, sampled_token)
        {
            Self::abort_sequence_with_error(
                self.active_batch.get_mut(seq_id),
                "structured output",
                &msg,
            );
            return;
        }

        let seq = match self.active_batch.get_mut(seq_id) {
            Some(s) => s,
            None => return,
        };

        if seq.merged_eos.contains(&token_val) {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
            {
                tracing::error!("State transition error: {err}");
            }
            return;
        }

        seq.generated_tokens.push(token_val);

        // Incrementally update token_history instead of rebuilding from scratch
        if seq.sampling.needs_token_history() {
            seq.token_history.push(token_val);
        }

        if let Some(new_text) = seq.decode_state.on_token(token_val, &self.tokenizer) {
            let event = match token_lp {
                Some(lp) => GenerateEvent::TokenWithLogprobs(new_text, lp),
                None => GenerateEvent::Token(new_text),
            };
            let _ = seq.response_tx.send(event);
        }

        if seq.generated_tokens.len() >= seq.max_tokens
            && let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
        {
            tracing::error!("State transition error: {err}");
        }

        // Loop / repetition guard (issue #432): end early when the raw
        // generated stream collapses into a short repeated pattern. Skip if the
        // length limit already finished this sequence; the detector is a
        // zero-overhead no-op when loop detection is disabled (default).
        if !seq.state.is_finished()
            && mlxcel_core::detect_repetition_loop(
                &seq.generated_tokens,
                &seq.sampling.loop_detection,
            )
        {
            match seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
            {
                Ok(()) => tracing::info!(
                    generated = seq.generated_tokens.len(),
                    "loop detection: ending generation early (repetition loop)"
                ),
                Err(err) => tracing::error!("State transition error: {err}"),
            }
        }

        // Periodic cache clearing, backend-aware cadence (#627): disabled by
        // default on CUDA (clear churns the pool and defeats CUDA-graph
        // reuse, mlx#2358), 256 on Metal, MLXCEL_CACHE_CLEAR_INTERVAL overrides.
        if mlxcel_core::memory::should_clear_cache_at(
            seq.generated_tokens.len(),
            mlxcel_core::memory::cache_clear_interval(),
        ) {
            mlxcel_core::clear_memory_cache();
        }

        if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
            cache_set.current_offset += 1;
        }
    }

    // ------------------------------------------------------------------
    // Completion and cleanup
    // ------------------------------------------------------------------

    fn finalize_completed(&mut self) {
        // Any completion or cancellation changes batch membership and may donate
        // a sequence's KV to the prompt cache. Tear down a live lookahead first
        // so its speculative KV position is trimmed off before donation / slot
        // reuse and the surviving sequences rebuild their pipeline next tick
        // (#632 constraints 2, 3, 6). No-op on the steady no-finish path.
        if self.decode_lookahead.is_some()
            && self
                .active_batch
                .iter_sequences()
                .any(|s| s.state.is_finished() || s.cancelled.load(Ordering::Relaxed))
        {
            self.discard_lookahead();
        }

        // First, transition any cancelled sequences to Finished(Cancelled).
        // This must happen before the finished-ID scan so that newly cancelled
        // sequences are collected in the same pass.
        let cancelled_ids: Vec<SequenceId> = self
            .active_batch
            .iter_sequences()
            .filter(|s| !s.state.is_finished() && s.cancelled.load(Ordering::Relaxed))
            .map(|s| s.seq_id)
            .collect();

        for id in &cancelled_ids {
            if let Some(seq) = self.active_batch.get_mut(*id) {
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                {
                    tracing::warn!("Failed to cancel sequence {id}: {err}");
                } else {
                    tracing::info!("Sequence {id} cancelled (client disconnected)");
                }
            }
        }

        // Cancel a chunked-prefill-in-progress sequence if client disconnected.
        if let Some(ref seq) = self.chunked_prefill_seq
            && seq.cancelled.load(Ordering::Relaxed)
        {
            let seq = self.chunked_prefill_seq.take().unwrap();
            tracing::info!(
                "Chunked-prefill sequence {} cancelled (client disconnected)",
                seq.seq_id
            );
            let _ = seq.response_tx.send(GenerateEvent::Error(
                "Request cancelled: client disconnected".to_string(),
            ));
            // Cancellation during prefill means the KV cache is only
            // partially populated; skip donate-back and just release. The
            // context map still needs cleanup so no dangling entries leak.
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            self.batch_observability.record_sequence_completed();
        }

        // Also cancel queued sequences whose client has already disconnected,
        // so they never enter the active batch.
        self.cancel_queued_disconnected();

        // Collect finished IDs by scanning active sequences. Uses iter_sequences()
        // to avoid allocating a full key snapshot when no sequences are finished.
        let finished_ids: Vec<SequenceId> = self
            .active_batch
            .iter_sequences()
            .filter(|s| s.state.is_finished())
            .map(|s| s.seq_id)
            .collect();

        let has_completed = !finished_ids.is_empty();
        for id in finished_ids {
            if let Some(mut seq) = self.active_batch.remove(id) {
                let tokens_generated = seq.generated_tokens.len();

                // Forward the incremental detokenizer's held tail as one final
                // token event before Done, so streaming clients are not missing
                // text the non-streaming result.text still carries (issue #633).
                if let Some(tail) = seq.decode_state.flush(&self.tokenizer) {
                    let _ = seq.response_tx.send(GenerateEvent::Token(tail));
                }
                let cached = seq.already_cached_tokens;
                let result = seq.decode_state.finish_with_cache(
                    seq.created_at,
                    seq.prompt_tokens.len(),
                    seq.max_tokens,
                    cached,
                );
                // Per-request TTFT / decode-rate telemetry (epic #623 #624).
                // Recorded once here, where the finished sequence's timings are
                // available, never on the per-token hot path.
                self.batch_observability.record_request_completion(
                    result.prompt_eval_ms,
                    result.generation_only_ms,
                    result.completion_tokens,
                );
                tracing::info!(
                    prompt_tokens = seq.prompt_tokens.len(),
                    cached_tokens = cached,
                    generation_time_ms = result.generation_time_ms,
                    "prompt-cache: request completed: \
                     cached={}/{} prompt tokens, total {}ms",
                    cached,
                    seq.prompt_tokens.len(),
                    result.generation_time_ms,
                );
                let _ = seq.response_tx.send(GenerateEvent::Done(result));

                // donate the full KV cache back to
                // the prompt-cache store on *healthy* finishes (Stop /
                // Length / Cancelled) so the next turn of the same
                // conversation can adopt it. `Finished(Error)` paths bypass
                // this branch — their cache is assumed tainted.
                let healthy = matches!(
                    seq.state,
                    SequenceState::Finished(
                        FinishReason::Stop
                            | FinishReason::Length
                            | FinishReason::RepetitionLoop
                            | FinishReason::Cancelled,
                    )
                );
                self.donate_finished_sequence_cache(
                    id,
                    &seq.prompt_tokens,
                    &seq.generated_tokens,
                    healthy,
                );
                // `donate_finished_sequence_cache` already removed the
                // context from `prompt_cache_seq_ctx` on donate; drop it
                // defensively on the non-donate paths so the map cannot
                // grow unbounded across long-lived workers.
                self.prompt_cache_seq_ctx.remove(&id);

                self.release_sequence_caches(id);
                self.batch_metrics
                    .record_sequence_completed(tokens_generated);
                self.batch_observability.record_sequence_completed();

                tracing::debug!("Sequence {id} completed ({tokens_generated} tokens)");
            }
        }

        if has_completed {
            self.publish_metrics();
        }
    }

    /// Remove queued sequences whose client has already disconnected.
    ///
    /// This prevents cancelled requests from ever entering the active batch,
    /// freeing the prefill queue slot immediately.
    fn cancel_queued_disconnected(&mut self) {
        let drained: Vec<SequenceInfo> = self.prefill_queue.drain_cancelled();
        for seq in drained {
            tracing::info!(
                "Queued sequence {} cancelled before prefill (client disconnected)",
                seq.seq_id
            );
            let _ = seq.response_tx.send(GenerateEvent::Error(
                "Request cancelled: client disconnected".to_string(),
            ));
            // No prefill ran → no valid cache to donate. Clear the
            // context entry so it cannot linger.
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            self.batch_observability.record_sequence_completed();
        }
    }

    fn abort_sequence(&mut self, seq: SequenceInfo, error: &str) {
        let _ = seq
            .response_tx
            .send(GenerateEvent::Error(error.to_string()));
        // Abort paths produce an error outcome (OOM / transition failure /
        // invalid cache); the KV cache is untrustworthy and must not be
        // donated back. Dropping the context entry prevents a future
        // finalize pass from trying.
        self.prompt_cache_seq_ctx.remove(&seq.seq_id);
        self.release_sequence_caches(seq.seq_id);
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "serving_handoff_parity_tests.rs"]
mod serving_handoff_parity_tests;

#[cfg(test)]
#[path = "scheduler_cohort_parity_tests.rs"]
mod scheduler_cohort_parity_tests;

#[cfg(test)]
#[path = "scheduler_seed_determinism_tests.rs"]
mod scheduler_seed_determinism_tests;
