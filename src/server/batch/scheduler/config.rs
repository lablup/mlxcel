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

use super::*;

impl BatchScheduler {
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
            mixed_step_enabled: mixed_step_enabled(),
            // #1011: resolve the env override / shipped default now;
            // `with_prefill_grant_interval` overrides this later with an
            // explicit CLI value when one was passed.
            prefill_grant_interval: resolve_prefill_grant_interval(None),
            decode_ticks_since_prefill_grant: 0,
            shutdown_requested: false,
            consecutive_decode_eval_failures: 0,
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
            xtc_newline_token_ids: Vec::new(),
            reasoning_budget: None,
            thinking_token_ids: None,
            prompt_cache: None,
            prompt_cache_seq_ctx: std::collections::HashMap::new(),
            prompt_cache_warmups: std::collections::VecDeque::new(),
            kv_cache_mode: KVCacheMode::Fp16,
            batch_kv_quant: BatchKvQuantConfig::default(),
            max_kv_size: None,
            context_retention: ContextRetentionPolicy::default(),
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
            speculative_drafter_slot:
                crate::server::batch::speculative_burst::WorkerDrafterSlot::from_dispatch(
                    &crate::server::SpeculativeDispatch::Disabled,
                ),
            // No adaptive MTP policy until `with_mtp_policy` builds one for an
            // MTP dispatch. The non-speculative hot path never touches it.
            mtp_policy: None,
            // No tick-cooperative slice in flight at startup (issue #734),
            // and nothing waiting for a slot grant (issue #746).
            speculative_slice: None,
            speculative_slice_yielded: false,
            speculative_slice_backlog: std::collections::VecDeque::new(),
            speculative_slice_grant_slices: 0,
            speculative_slice_grant_budget: 0,
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

    /// Override the #1011 prefill fairness interval with the explicit
    /// CLI/config value (`--prefill-grant-interval`).
    ///
    /// `None` keeps whatever [`Self::with_config`] resolved from
    /// `MLXCEL_PREFILL_GRANT_INTERVAL` or the shipped default; `Some(0)`
    /// disables the grant (pre-#1011 arbitration, unbounded parked-prefill
    /// wait); `Some(n)` sets an explicit interval.
    pub fn with_prefill_grant_interval(mut self, configured: Option<usize>) -> Self {
        self.prefill_grant_interval = resolve_prefill_grant_interval(configured);
        self
    }

    /// Returns the resolved prefill fairness interval (0 = grant disabled).
    /// Exposed for tests.
    pub fn prefill_grant_interval(&self) -> u32 {
        self.prefill_grant_interval
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
        self.inject_model_owned_kv_cache_modes();
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
        self.inject_model_owned_kv_cache_modes();
        self.log_effective_kv_cache_application();
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
            // Note: Int8 KV forces the dense decode backend (only genuine Fp16
            // sequences are pool-backed on the paged path). The dense
            // batched-decode front-trim used to mis-decode prompts longer than
            // the cap because it dropped the leading attention-sink tokens
            // (issue #718). `enforce_max_kv_size_for` now pins a small sink
            // prefix via `KVCache::trim_front_keep_sink`, matching mlx-lm
            // `RotatingKVCache(keep=4)`, so `--max-kv-size` decodes correctly on
            // the dense backend (and therefore under Int8); no warning needed.
        }
        self.max_kv_size = max_kv_size;
        self
    }

    /// Returns the configured maximum KV cache size (for tests).
    pub fn max_kv_size(&self) -> Option<usize> {
        self.max_kv_size
    }

    /// Install the context-retention policy (#1472, b10621 `--context-shift`
    /// / `--keep`). Default: shifting disabled, retain 0, which is upstream's
    /// default and makes the KV bound a hard stop.
    pub fn with_context_retention(mut self, policy: ContextRetentionPolicy) -> Self {
        self.context_retention = policy;
        self
    }

    /// The configured context-retention policy (for tests).
    pub fn context_retention(&self) -> ContextRetentionPolicy {
        self.context_retention
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

    /// Install the paged KV slab size in blocks (issue #899).
    ///
    /// `Some(n)` makes each layer's pool storage one contiguous `n`-row slab,
    /// which is the precondition for the fused paged-attention decode kernels:
    /// they read one pool buffer per side, so a layer spread across several
    /// slabs is declined and falls back to gather-then-SDPA. `None` leaves the
    /// pool's own default (32 rows), which is the pre-#899 behaviour.
    ///
    /// Resolved from the operator's `--ctx-size` / `--parallel` and the KV
    /// budget by [`crate::memory_estimate::resolve_paged_slab_blocks`] on the
    /// worker thread. Applied to the pool when it is lazily created; a failure
    /// (a pool that already has storage) is logged and ignored, because an
    /// unsized slab costs performance, not correctness.
    pub fn with_paged_slab_blocks(mut self, slab_blocks: Option<usize>) -> Self {
        if let Err(reason) = self.cache_pool.set_paged_slab_blocks(slab_blocks) {
            tracing::warn!("could not install the paged KV slab size: {reason}");
        }
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
    ///    - **MTP**: `Gemma4Wrapper` / `Gemma4VLModel` (Gemma 4 assistant
    ///      drafter, see the `MtpTarget` impls in
    ///      [`crate::models::gemma4_mtp_target`]) and, since issue #1165,
    ///      `Qwen35Model` / `Qwen35VLModel` (`qwen3_5_mtp` drafter, Metal
    ///      only, see [`crate::models::qwen3_5_mtp_target`]).
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
            crate::server::batch::speculative_burst::WorkerDrafterSlot::from_dispatch(&dispatch);
        self.speculative_dispatch = dispatch;

        // Warm the block-vs-chain exactness probe here rather than letting
        // the first request pay for it. The probe is memoized per (model,
        // block width), so this is the same call the per-request gate
        // makes; running it at worker startup keeps a few hundred
        // milliseconds of GPU work off the request path and puts the
        // verdict in the startup log, where an operator will see a decline
        // before wondering why throughput looks like classic decode.
        if let crate::server::SpeculativeDispatch::Mtp { block_size, .. } =
            &self.speculative_dispatch
        {
            let block_size = *block_size as usize;
            if block_size >= 2 {
                let _ = crate::server::batch::speculative_burst::mtp_capable_target(
                    &self.model,
                    block_size,
                );
            }
        }
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
    ///
    /// Whatever this resolves to, including "no policy at all", is published
    /// into [`super::observability::BatchObservability`] for the
    /// `GET /v1/internal/mtp-policy` endpoint (issue #1257). Publishing here is
    /// what lets the endpoint answer without touching `MtpPolicy`, which the
    /// worker thread owns unsynchronized.
    pub fn with_mtp_policy(mut self, target_model_id: Option<String>) -> Self {
        use crate::server::batch::mtp_policy::{MtpPolicySnapshot, MtpPolicyUnavailableReason};

        // Snapshot to publish when the exactness gate has vetoed the
        // pairing (issue #1298). Checked here, at attach time, because the
        // gate is memoized per block width and cannot change later in the
        // process: publishing the attached policy instead would report
        // `profiling` forever, since the burst the policy waits to sample
        // never dispatches past `mtp_capable_target`.
        let mut exactness_veto: Option<MtpPolicySnapshot> = None;
        let mtp_dispatch = if let crate::server::SpeculativeDispatch::Mtp {
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
            let block = *block_size as usize;
            if block >= 2
                && !crate::server::batch::speculative_burst::mtp_capable_target(&self.model, block)
            {
                exactness_veto = Some(MtpPolicySnapshot::exactness_declined(
                    target_id.clone(),
                    drafter_id.clone(),
                    *block_size,
                    crate::models::speculative_exactness::decline_reason(*block_size),
                ));
            }
            self.mtp_policy = crate::server::batch::mtp_policy::MtpPolicy::initialize(
                target_id,
                drafter_id,
                *block_size,
                self.model.supports_batching(),
            );
            true
        } else {
            false
        };
        // Publish the resolved policy state for the supported read interface
        // (issue #1257). Doing it here, rather than letting the endpoint reach
        // into `self.mtp_policy`, is what keeps `MtpPolicy` single-threaded and
        // lock-free on the decode path.
        // The exactness veto outranks the attached policy's own view: a
        // vetoed pairing's policy is structurally starved and its
        // `profiling` state would be a lie (issue #1298).
        let published = if let Some(veto) = exactness_veto {
            veto
        } else {
            match (&self.mtp_policy, mtp_dispatch) {
                (Some(policy), _) => policy.snapshot(),
                // MTP dispatch with the adaptive path switched off
                // (`MLXCEL_MTP_ADAPTIVE=0`): the pre-#333 static per-hardware gate
                // decides, so report what it decided rather than leaving a
                // consumer to guess whether MTP runs.
                (None, true) => MtpPolicySnapshot::unavailable(
                    MtpPolicyUnavailableReason::AdaptiveDisabled,
                    Some(
                        crate::server::batch::speculative_burst::mtp_b1_burst_enabled(
                            self.model.supports_batching(),
                        ),
                    ),
                ),
                // No MTP dispatch at all: there is no pairing and no burst.
                (None, false) => MtpPolicySnapshot::unavailable(
                    MtpPolicyUnavailableReason::NoMtpDispatch,
                    Some(false),
                ),
            }
        };
        self.batch_observability.set_mtp_policy(published);
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
    /// [`crate::server::batch::speculative_burst::should_burst_for_sequence`], which
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

    /// Attach the tokenizer's newline token id(s), resolved once at worker
    /// startup, for the XTC special-token allowlist.
    ///
    /// Cached for the scheduler's lifetime and combined with each request's
    /// merged end-of-sequence set at enqueue time (see
    /// [`Self::enqueue_request`]). An empty list is a no-op — it simply
    /// contributes nothing to the allowlist.
    pub fn with_xtc_newline_token_ids(mut self, ids: Vec<i32>) -> Self {
        self.xtc_newline_token_ids = ids;
        self
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
}
