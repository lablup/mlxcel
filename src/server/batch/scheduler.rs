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
    BatchKvQuantConfig, CachePool, KVCacheMode, PagedKvLayout, SequenceId, SequenceStateBackend,
    SequenceStateLayout,
};
use mlxcel_core::generate::{
    DecodeBatchContext, DecodeStorageBackend as CoreDecodeStorageBackend, LanguageModel,
};
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use mlxcel_core::hardware;
use mlxcel_core::sampling::{TokenBiasMap, compute_logprobs, sample_token_optimized};
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
use crate::server::prompt_cache::key::{MultimodalDigest, PromptCacheKey};
use crate::server::prompt_cache::{CacheEntry, DetachedKvSet, PromptCacheStore};
use crate::server::state::BatchMetrics;
use crate::server::thinking_budget::{
    ThinkingBudget, ThinkingDecision, ThinkingState, ThinkingTokenIds,
};
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::feature_cache::ModelVisionCaches;
use crate::vlm_runtime::prepared_embedding_refs;

use super::active::ActiveBatch;
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

const DEFAULT_PAGED_BLOCK_SIZE: usize = 32;

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
            decode_storage_backend: effective_decode_storage,
            vision_caches: Rc::new(ModelVisionCaches::new(
                crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE,
            )),
            token_bias: TokenBiasMap::default(),
            reasoning_budget: None,
            thinking_token_ids: None,
            prompt_cache: None,
            prompt_cache_seq_ctx: std::collections::HashMap::new(),
            kv_cache_mode: KVCacheMode::Fp16,
            batch_kv_quant: BatchKvQuantConfig::default(),
            max_kv_size: None,
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
        }
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
        }
        self.max_kv_size = max_kv_size;
        self
    }

    /// Returns the configured maximum KV cache size (for tests).
    pub fn max_kv_size(&self) -> Option<usize> {
        self.max_kv_size
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
    /// * Looks up a longest-prefix match on each new request and calls
    ///   [`CachePool::adopt`] on hit to skip re-prefill of the shared prefix.
    /// * Donates the sequence's full cache back to the store on a healthy
    ///   finish (normal stop / length / cancelled without error).
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
        PromptCacheKey::new_full(
            ctx.model_id.as_str(),
            ctx.lora_id.as_deref(),
            ctx.template_sig.as_str(),
            Some(ctx.session_key.as_str()),
            MultimodalDigest::empty(),
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
    /// * [`CachePool::adopt`] / [`CachePool::adopt_paged`] error (capacity,
    ///   layout mismatch, …).
    ///
    /// Both dense and paged entries are adopted in-place: dense via
    /// [`CachePool::adopt`], paged via [`CachePool::adopt_paged`] (which shares
    /// the cached prefix's refcounted pool blocks so the prefix is never
    /// re-prefilled — #121 sub-step b).
    fn try_adopt_cached_prefix(
        &mut self,
        ctx: &PromptCacheRequestContext,
        tokens: &[i32],
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
        let (entry, matched_len) = store.lookup_longest_prefix(&key, tokens)?;
        // `take_detached` is one-shot: it returns `None` if a racing lookup
        // already consumed this entry. The miss path is safe — the current
        // sequence just does a fresh prefill.
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
            DetachedKvSet::Paged(paged) => {
                // APC block-hash partial adoption for paged entries is deferred
                // (a later #121 sub-step): the paged store path matches the full
                // stored prefix only. If a partial match somehow surfaces,
                // decline rather than adopt an over-long prefix.
                let paged_seq_len = paged.seq_len();
                if matched_len < paged_seq_len {
                    tracing::debug!(
                        from = paged_seq_len,
                        to = matched_len,
                        "prompt-cache adopt: paged partial adoption not yet supported; releasing and falling back to cold prefill"
                    );
                    self.cache_pool.release_detached_paged(paged);
                    return None;
                }
                self.cache_pool
                    .adopt_paged(&self.model as &dyn LanguageModel, paged)
            }
        };

        match adopt_result {
            Ok(adopted_id) => {
                tracing::debug!(
                    seq_id = %adopted_id,
                    matched = matched_len,
                    total = tokens.len(),
                    "prompt-cache hit: adopted {matched_len}/{} tokens",
                    tokens.len()
                );
                self.batch_observability
                    .record_prompt_cache_hit(matched_len);
                // also increment BatchMetrics Prometheus counters.
                self.batch_metrics.record_prompt_cache_hit(matched_len);
                // Update byte/entry gauges so /metrics reflects current state.
                if let Some(ref store) = self.prompt_cache {
                    self.batch_metrics
                        .update_prompt_cache_gauges(store.bytes(), store.len());
                }
                Some((adopted_id, matched_len))
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
    /// `ModelOwned` sequences carry no detachable cross-request KV and are
    /// skipped.
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

        let backend = self
            .cache_pool
            .get_mut(seq_id)
            .map(|s| s.backend)
            .unwrap_or(SequenceStateBackend::ModelOwned);
        // `ModelOwned` families (heterogeneous attention+recurrent caches, e.g.
        // Qwen 3.5 / Gemma 4) carry no detachable cross-request KV. Skip before
        // building the token vector so the burst donate stays a cheap no-op.
        if backend == SequenceStateBackend::ModelOwned {
            return;
        }

        // Tokens stored against the entry are the full prompt + generated
        // tail, so the next turn's `prompt + new user turn` can match at
        // least up through the previous assistant reply.
        let mut tokens = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
        tokens.extend_from_slice(prompt_tokens);
        tokens.extend_from_slice(generated_tokens);

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
                    // Use batched prefill when max_batch_prefill > 1 and at
                    // least 2 requests are waiting, otherwise take the regular
                    // single-request path so there is zero overhead for the
                    // common case.
                    if self.max_batch_prefill > 1
                        && self.prefill_queue.len() >= 2
                        && self.chunked_prefill_seq.is_none()
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
                options,
                images,
                audio,
                videos,
                response_tx,
                cancelled,
            } => {
                self.enqueue_request(
                    prompt,
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
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<crate::server::media::ResolvedVideo>,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
    ) {
        let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
        let token_ids = match self.tokenizer.encode(&prompt, add_special) {
            Ok(ids) => ids,
            Err(err) => {
                let _ =
                    response_tx.send(GenerateEvent::Error(format!("Tokenization error: {err}")));
                return;
            }
        };
        let mut prompt_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();

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

        // before allocating a fresh KV-cache slot,
        // probe the prompt-prefix cache for a reusable detached set. On a
        // hit, adopt under a brand-new SequenceId and record how many
        // leading tokens the prefill can skip. On a miss (which includes
        // feature-disabled, no ctx, and race paths), fall through to the
        // cold-allocation path below.
        //
        // VLM / audio / video requests opt out of the cache path entirely:
        // their pre-injection token stream is not self-describing (image
        // and video frame placeholders expand later inside
        // `prepare_request_vlm_embeddings`), so matching against it risks
        // reusing a KV slice built for a different media payload. Support
        // for image-aware cache keys is tracked separately.
        let is_multimodal = !images.is_empty() || !audio.is_empty() || !videos.is_empty();
        let ctx_ref = if is_multimodal {
            None
        } else {
            options.prompt_cache_ctx.as_ref()
        };
        let (seq_id, prefill_start_offset, already_cached_tokens) =
            match ctx_ref.and_then(|ctx| self.try_adopt_cached_prefix(ctx, &prompt_tokens)) {
                Some((adopted_id, matched_len)) => (adopted_id, matched_len, matched_len),
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
                            let _ = response_tx
                                .send(GenerateEvent::Error(format!("Server busy: {err}")));
                            return;
                        }
                    };
                    (seq_id, 0, 0)
                }
            };

        let vlm_embeddings = match prepare_request_vlm_embeddings(
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
        // empty and the donate-back short-circuits. Multimodal requests
        // opt out of the cache altogether (see above).
        if self.prompt_cache_active()
            && !is_multimodal
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
            && !super::speculative_burst::mtp_b1_burst_enabled()
        {
            // B=1 (single-request) MTP runs by default for every MTP target
            // (~1.87x on the 12B Unified pair, ~1.2 to 1.4x on the 31B, both
            // byte-identical on M5 Max). This decline fires only when an operator
            // opts out with `MLXCEL_ENABLE_MTP_B1=0`, e.g. on lower-bandwidth
            // Apple Silicon where the B=1 verify forward may not pay for itself;
            // the request then falls back to classic decode.
            let seq = window.into_iter().next().expect("singleton window");
            tracing::info!(
                "MTP B=1 speculative burst disabled for seq {} via MLXCEL_ENABLE_MTP_B1=0; \
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
                    // The donate helper is hard-gated on a dense
                    // KV-cache backend; both batched-eligible model
                    // families today — Gemma 4 (MTP) and Qwen 3.5
                    // (DFlash) — are `SequenceStateBackend::ModelOwned`
                    // with heterogeneous attention+recurrent caches, so
                    // the donate is a guarded no-op for them, identical
                    // to the B=1 arm's no-op for those same families.
                    // Wiring it in removes the structural asymmetry
                    // between the two burst arms and future-proofs the
                    // batched path for any dense-KV-cache model that
                    // later becomes batched-burst-eligible. Error /
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
                    // donate helper detaches dense and paged backends but
                    // skips `SequenceStateBackend::ModelOwned`; the two
                    // burst-eligible model families today — Qwen 3.5
                    // (DFlash) and Gemma 4 (MTP) — are both `ModelOwned`
                    // with heterogeneous attention+recurrent caches that
                    // the detach handoff cannot represent, so the donate is
                    // a guarded no-op for them — identical to the classic
                    // path's no-op for those same families. Wiring it in
                    // removes the structural asymmetry between the two
                    // paths and future-proofs the burst for any
                    // dense/paged-KV model that later becomes
                    // burst-eligible.
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
                    self.publish_metrics();
                    None
                }
                Err(rejected_seq) => Some(rejected_seq),
            }
        }
    }

    /// Batched prefill: drain up to `max_batch_prefill` requests from the
    /// prefill queue and process them in a single forward pass.
    ///
    /// Sequences are padded to the longest prompt in the batch (aligned to a
    /// 32-token tile on M5+). Each sequence gets a per-sequence causal +
    /// padding attention mask so padding tokens are excluded from attention.
    ///
    /// On any error the method falls back to sequential single-request prefill
    /// for the remaining requests so no requests are lost.
    fn execute_batched_prefill(&mut self) {
        let batch_size = self.max_batch_prefill.min(self.prefill_queue.len());

        // Collect up to `batch_size` requests from the queue.
        let mut seqs: Vec<SequenceInfo> = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match self.prefill_queue.dequeue() {
                Some(s) => seqs.push(s),
                None => break,
            }
        }

        if seqs.is_empty() {
            return;
        }

        // Single-request fast path: fall through to the regular prefill so
        // there is no overhead for constructing a padded batch.
        if seqs.len() == 1 {
            let seq = seqs.remove(0);
            self.execute_full_prefill(seq);
            return;
        }

        // Most models only implement batched decode (`[B, 1]`) and do not
        // support full-sequence prompt prefill via `forward_batched()`.
        // Keep those on the single-sequence prefill path so correctness and
        // the standard NAX-friendly prefill route are preserved.
        if !self.model.supports_batched_prefill() {
            tracing::debug!(
                "batched prefill: falling back to sequential (model lacks full batched prefill)"
            );
            for mut seq in seqs {
                if let Err(err) = Self::begin_prefill(&mut seq) {
                    tracing::error!("Batched prefill state transition error: {err}");
                    self.abort_sequence(seq, &err);
                    continue;
                }
                self.execute_full_prefill(seq);
            }
            return;
        }

        // Any VLM request cannot currently be batched (embeddings are
        // per-sequence and would need separate handling). Fall back for the
        // whole batch when any request carries VLM embeddings.
        if seqs.iter().any(|s| s.vlm_embeddings.is_some()) {
            tracing::debug!("batched prefill: falling back to sequential (VLM request in batch)");
            for mut seq in seqs {
                if let Err(err) = Self::begin_prefill(&mut seq) {
                    tracing::error!("Batched prefill state transition error: {err}");
                    self.abort_sequence(seq, &err);
                    continue;
                }
                self.execute_full_prefill(seq);
            }
            return;
        }

        // any sequence that adopted a cached prefix
        // cannot participate in the padded batched prefill path because the
        // KV-history offsets differ across sequences. Take the single-
        // sequence path for those so their `prefill_start_offset` is
        // honored correctly; batched-prefill continues for the rest of
        // this batch in the normal padded pipeline below.
        if seqs.iter().any(|s| s.prefill_start_offset > 0) {
            tracing::debug!(
                "batched prefill: falling back to sequential (adopted prompt-cache prefix in batch)"
            );
            for mut seq in seqs {
                if let Err(err) = Self::begin_prefill(&mut seq) {
                    tracing::error!("Batched prefill state transition error: {err}");
                    self.abort_sequence(seq, &err);
                    continue;
                }
                self.execute_full_prefill(seq);
            }
            return;
        }

        let b = seqs.len();
        let max_len = seqs.iter().map(|s| s.prompt_tokens.len()).max().unwrap();
        let can_pad_prefill = self.model.supports_padded_prefill();
        if !can_pad_prefill && seqs.iter().any(|s| s.prompt_tokens.len() != max_len) {
            tracing::debug!(
                "batched prefill: falling back to sequential (model requires equal prompt lengths)"
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

        // Transition all sequences to Prefilling.
        for seq in &mut seqs {
            if let Err(err) = Self::begin_prefill(seq) {
                tracing::error!("Batched prefill state transition error: {err}");
            }
        }

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
        // Counter reflects only the work the model actually runs.
        let suffix_len = seq.prompt_tokens.len() - seq.prefill_start_offset;
        self.batch_observability.record_prefill_start(suffix_len);

        // Reset internal caches for non-batching models (same as execute_full_prefill).
        if !self.model.supports_batching() {
            let _ = self.model.make_caches();
        }

        let chunk_size = self.prefill_chunk_size;
        let start = seq.prefill_start_offset;
        let end = (start + chunk_size).min(seq.prompt_tokens.len());
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
        {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found for sequence during chunked prefill");
                    return;
                }
            };

            // VLM embeddings are applied only on the first chunk.
            if let Some(ref embeddings) = seq.vlm_embeddings {
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
            }

            // Trim padding positions from KV caches when the chunk was padded.
            if pad_mask_opt.is_some() && eff_chunk.len() > actual_chunk_len {
                let excess = (eff_chunk.len() - actual_chunk_len) as i32;
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }
        }
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
        self.batch_observability.record_prefill_chunk();

        let chunk_size = self.prefill_chunk_size;
        let offset = seq.prefill_offset;
        let total = seq.prompt_tokens.len();
        let end = (offset + chunk_size).min(total);
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

        if end < total {
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
            seq.decode_state.flush(&self.tokenizer);
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

        if seq_ids.len() <= 1 || !self.model.supports_batching() {
            for &seq_id in seq_ids {
                self.decode_single_step(seq_id);
            }
            return;
        }

        self.execute_batched_decode(seq_ids);
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

        let mut batch_caches = match self.cache_pool.get_batch_caches_mut(seq_ids) {
            Ok(caches) => caches,
            Err(err) => {
                tracing::error!("{err} during batched decode");
                return;
            }
        };

        let decode_context = match self.decode_storage_backend {
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
                let (token_arr, adjusted_logits) =
                    sample_token_optimized(&logits_for_sampling, &seq.sampling, &seq.token_history);
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

            // Periodic cache clearing (matches Python mlx-lm which clears every 256)
            if seq.generated_tokens.len() % 256 == 0 {
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
            let (token_arr, adjusted_logits) =
                sample_token_optimized(&logits_for_sampling, &seq.sampling, &seq.token_history);
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

        // Periodic cache clearing (matches Python mlx-lm which clears every 256)
        if seq.generated_tokens.len() % 256 == 0 {
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

                seq.decode_state.flush(&self.tokenizer);
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
                        FinishReason::Stop | FinishReason::Length | FinishReason::Cancelled,
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
