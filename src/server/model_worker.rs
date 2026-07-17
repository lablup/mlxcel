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

//! Server-side generation helpers and worker lifecycle for `ModelProvider`.
//!
//! `ModelProvider` owns the public channel API, while this module owns the
//! long-lived worker thread behavior plus the image/VLM preparation helpers.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageError, ImageReader};
use mlxcel_core::generate::LanguageModel;

use crate::LoadedModel;
use crate::SamplingConfig;
// The backend-neutral serve contract (#449 M3 Stage 2c). The MLX scheduler and the
// OpenXLA worker are both driven through `BatchEngine::serve`.
use crate::server::batch::BatchEngine;
use crate::server::batch::BatchObservability;
use crate::server::media::{ImageInputLimits, current_image_input_limits};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::feature_cache::ModelVisionCaches;
use crate::vision::merge::InputEmbeddings;
use crate::vlm_runtime::{
    prepare_and_compute_vlm_embeddings_with_budget, prepare_and_compute_vlm_embeddings_with_cache,
};
use crate::worker_failfast::run_core_thread_or_abort;

use super::{GenerationResult, ModelRequest};

/// Configuration for the scheduler, passed from `ModelProvider` to the
/// worker thread.
pub(crate) struct WorkerSchedulerConfig {
    pub max_batch_size: usize,
    pub max_queue_depth: usize,
    pub prefill_chunk_size: usize,
    pub enable_preemption: bool,
    pub preemption_policy: crate::server::config::PreemptionPolicy,
    /// Maximum number of requests to batch together for prefill (default: 1).
    pub max_batch_prefill: usize,
    /// #715: explicit `--max-batch-prefill-tokens` value bounding the padded
    /// batched-prefill transient. `None` keeps the env override or the derived
    /// default (`max_batch_prefill * prefill_chunk_size`); `Some(0)` disables
    /// the cap (uncapped); `Some(n)` sets an explicit budget.
    pub max_batch_prefill_tokens: Option<usize>,
    /// Decode-time storage backend for server sequence state.
    pub decode_storage_backend: crate::server::DecodeStorageBackend,
    /// Optional pipeline runtime for in-process or remote coordinator execution.
    pub pipeline_parallel_runtime: Option<crate::server::PipelineParallelRuntimeConfig>,
    /// Tensor-parallel runtime configuration.
    pub tensor_parallel: crate::distributed::ShardConfig,
    /// Maximum number of cached post-projection image features per loaded
    /// VLM. `0` disables caching.
    pub vision_cache_size: usize,
    /// Axis B (B8): optional server-wide language-bias config.
    /// Resolved once on the worker thread into a `TokenBiasMap` after the
    /// tokenizer loads, and attached to the batch scheduler for the rest of
    /// the worker's lifetime.
    pub lang_bias_config: Option<mlxcel_core::lang_analyzer::LangBiasConfig>,
    /// server-wide default thinking-token budget.
    pub reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,
    /// cross-request prompt-prefix KV cache store.
    ///
    /// `None` when the feature is disabled by
    /// [`crate::server::prompt_cache::PromptCacheConfig::enabled`]. When
    /// `Some`, the worker thread can publish detached caches and lookup /
    /// adopt them on later requests. The store is thread-safe, so the same
    /// `Arc` is also handed to `AppState` for observation-only use.
    ///
    /// Store handle passed to [`BatchScheduler::with_prompt_cache`] so the
    /// scheduler can adopt detached prefixes on cache hits and donate-back
    /// finished sequences.
    pub prompt_cache: Option<Arc<crate::server::prompt_cache::PromptCacheStore>>,
    /// (B11) / server-wide KV cache quantization mode.
    ///
    /// Defaults to [`mlxcel_core::cache::KVCacheMode::Fp16`] (bit-exact
    /// baseline). When a Turbo4 variant is configured, the scheduler
    /// applies it to each new sequence's per-layer cache and picks the
    /// Turbo4-aware paged layout.
    pub kv_cache_mode: mlxcel_core::cache::KVCacheMode,
    /// continuous-batching KV quantization configuration.
    ///
    /// When enabled (`bits > 0`), the scheduler resolves per-layer
    /// [`mlxcel_core::cache::KVCacheMode`] values from this config (with
    /// the last layer optionally forced to FP16) and overrides the
    /// nominal [`Self::kv_cache_mode`] for each newly-allocated sequence.
    /// Defaults to a disabled config so existing deployments stay
    /// bit-exact.
    pub batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig,
    /// maximum KV cache size for plain (non-sliding) caches.
    ///
    /// When `Some(N)`, the batch scheduler caps each per-sequence plain
    /// `KVCache` to `N` tokens by trimming the oldest entries once
    /// `offset > N`. Sliding-window models keep their model-specific
    /// window and bypass this cap. `None` (the default) preserves the
    /// legacy unbounded behaviour.
    pub max_kv_size: Option<usize>,
    /// paged KV pool block-budget directive (epic #116 #122 b3,
    /// `--kv-cache-budget`).
    ///
    /// `None` (the default) keeps the paged pool unbounded — the
    /// behaviour-preserving path. `Some(Bytes/Auto)` is resolved to a concrete
    /// block count on this worker thread (where `model_path` + the loaded
    /// model's geometry are both available) via
    /// [`crate::memory_estimate::resolve_paged_block_budget`] and installed on
    /// the scheduler's pool through
    /// [`crate::server::batch::BatchScheduler::with_paged_block_budget`].
    pub kv_cache_budget: Option<crate::memory_estimate::PagedBudgetDirective>,
    /// experimental VLM prompt-prefix cache toggle (#124 step c,
    /// `--enable-vlm-prefix-cache`).
    ///
    /// `false` (the default) preserves the legacy cold-prefill behavior for
    /// every multimodal request. When `true`, the scheduler permits VLM
    /// (image/audio) chat requests to adopt and donate KV prefixes for
    /// multi-turn same-image conversations. Text-only and non-VLM requests are
    /// unaffected either way.
    pub enable_vlm_prefix_cache: bool,
    /// disaggregated serving role for paged KV serving (#126 B2), derived from
    /// `--node-role`.
    ///
    /// [`ServingMode::Hybrid`](crate::distributed::disaggregated::ServingMode::Hybrid)
    /// (the default) is the single-node path: the worker runs the standard
    /// scheduler loop, byte-identical to a server with no distributed flags.
    /// `PrefillOnly` / `DecodeOnly` select a disaggregated serving role. The
    /// worker carries the mode so a serving-role coordinator
    /// ([`crate::distributed::disaggregated::ServingCoordinator`]) can drive the
    /// scheduler over the B1 handoff hooks in a later step (B2b); until then
    /// every mode runs the standard loop.
    pub serving_mode: crate::distributed::disaggregated::ServingMode,
    /// disaggregated serving-role network addresses (#126 B3b2a).
    ///
    /// `serving_bind` is this node's own role-transport listener; `decode_peers`
    /// holds the decode node a prefill worker hands KV off to (the first entry).
    /// Empty / `None` on a `Hybrid` node, where the worker runs the standard
    /// scheduler loop. (`--prefill-peers` is consumed by the future dedicated
    /// router via `ServerConfig`, not by the worker, so it is not threaded here.)
    pub decode_peers: Vec<std::net::SocketAddr>,
    pub serving_bind: Option<std::net::SocketAddr>,
    /// resolved speculative-decoding dispatch shape.
    ///
    /// Constructed once at worker construction (or
    /// [`crate::server::SpeculativeDispatch::Disabled`] when the operator
    /// did not pass `--draft-model`). The scheduler logs the summary at
    /// startup and consumes the variant per request to decide whether to
    /// run classic decode (the default and only currently-wired path
    /// inside the scheduler) or one of the kind-specific speculative
    /// round loops (MTP / DFlash) once the round-loop dispatch hook lands
    /// in [`crate::server::batch::BatchScheduler`]. For
    /// [`crate::server::SpeculativeDispatch::Disabled`] the hot path
    /// short-circuits in [`BatchScheduler::decode_single_step`] with no
    /// overhead.
    pub speculative_dispatch: crate::server::SpeculativeDispatch,
    /// serve-level `--max-denoising-steps` override (diffusion models only).
    ///
    /// `None` (the default) keeps the checkpoint's `generation_config` step
    /// cap. Only the DiffusionGemma worker loop reads it; autoregressive
    /// models ignore it.
    pub max_denoising_steps: Option<usize>,
    /// serve-level `--diffusion-sampler` selection (diffusion models only).
    ///
    /// `"entropy-bound"` (default) or `"confidence-threshold"`; parsed once on
    /// the worker thread. Ignored by non-diffusion models.
    pub diffusion_sampler: String,
    /// serve-level `--diffusion-threshold` for the confidence-threshold
    /// sampler (diffusion models only). Ignored by non-diffusion models.
    pub diffusion_threshold: f32,
}

pub(crate) fn spawn_model_worker_with_batch_config(
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    request_rx: mpsc::Receiver<ModelRequest>,
    loaded: Arc<AtomicBool>,
    worker_model_id: String,
    sched_config: WorkerSchedulerConfig,
    batch_metrics: Arc<BatchMetrics>,
    batch_observability: Arc<BatchObservability>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Re-impose fail-fast on this core generation thread under release
        // `panic = "unwind"` (issue #375): an uncaught panic in the load,
        // scheduler, diffusion, or disaggregated-role path aborts the process
        // for a supervised restart instead of silently unwinding and leaving
        // the server unable to generate. The audio worker and pipeline stage
        // boundaries keep their own `catch_unwind` and are not wrapped.
        run_core_thread_or_abort("model-worker", move || {
            tracing::info!("Model worker thread starting, loading model...");

            let load_start = Instant::now();
            // Route model loading through the compute-backend seam (issue
            // #338). Under default features this folds to the MLX backend with
            // no runtime dispatch. The pipeline-parallel branch below is its
            // own distributed loader and does not go through the seam.
            let backend = crate::backend::select_backend();
            let result = if let Some(ref pipeline_runtime) = sched_config.pipeline_parallel_runtime
            {
                match pipeline_runtime {
                crate::server::PipelineParallelRuntimeConfig::InProcess {
                    layers,
                    micro_batch_size,
                } => {
                    crate::distributed::pipeline::PipelineServerModel::load_in_process_with_adapter(
                        &model_path,
                        Some(layers.as_str()),
                        *micro_batch_size,
                        adapter_path.as_deref(),
                    )
                }
                crate::server::PipelineParallelRuntimeConfig::RemoteCoordinator(config) => {
                    crate::distributed::pipeline::PipelineServerModel::load_remote(
                        &model_path,
                        config.clone(),
                    )
                }
            }
            .and_then(|model| {
                let tokenizer = crate::tokenizer::load_tokenizer(&model_path)?;
                Ok((crate::LoadedModel::PipelineLlama(model), tokenizer))
            })
            } else if sched_config.tensor_parallel.tp_size > 1 {
                backend.load_model_with_tensor_parallel(
                    &model_path,
                    adapter_path.as_deref(),
                    &sched_config.tensor_parallel,
                )
            } else if let Some(adapter) = adapter_path {
                tracing::info!("Loading LoRA adapter from {:?}", adapter);
                backend.load_model_with_adapter(&model_path, &adapter)
            } else {
                backend.load_model(&model_path)
            };

            let (model, tokenizer) = match result {
                Ok((model, tokenizer)) => {
                    let load_elapsed = load_start.elapsed();
                    // Issue #55: log MLX-allocator resident memory after a
                    // successful weight load so operators see the actual
                    // working set the model occupies (not just the tensor
                    // sum). Useful for capacity planning and for the future
                    // preflight (#56) which will compare this against
                    // `MLXCEL_MEMORY_LIMIT` to fail fast.
                    let snap = mlxcel_core::memory::snapshot();
                    tracing::info!(
                        worker_model_id = %worker_model_id,
                        load_seconds = load_elapsed.as_secs_f64(),
                        active_bytes = snap.active_bytes,
                        peak_bytes = snap.peak_bytes,
                        cache_bytes = snap.cache_bytes,
                        limit_bytes = snap.limit_bytes,
                        "Model {worker_model_id} loaded in {:.3}s (resident after load: {:.2} GB)",
                        load_elapsed.as_secs_f64(),
                        snap.active_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    );
                    loaded.store(true, Ordering::Release);
                    (model, tokenizer)
                }
                Err(err) => {
                    tracing::error!("Failed to load model: {err}");
                    return;
                }
            };

            let config_eos = crate::read_eos_token_ids(&model_path);
            if !config_eos.is_empty() {
                tracing::info!("EOS tokens from config: {:?}", config_eos);
            }

            // DiffusionGemma (issue #217 phase 3): block-diffusion models are
            // model-owned single-stream generators (`supports_batching() == false`)
            // and cannot join the BatchScheduler. Serve them on a dedicated
            // batch-1 loop off the same request channel and return; all the
            // scheduler-specific setup below is skipped.
            let model = match model {
                LoadedModel::DiffusionGemma(diffusion) => {
                    let sampler = crate::server::diffusion_worker::parse_diffusion_sampler(
                        &sched_config.diffusion_sampler,
                    )
                    .unwrap_or_else(|err| {
                        tracing::warn!("{err}; defaulting to entropy-bound");
                        crate::models::diffusion_gemma::DiffusionSamplerKind::EntropyBound
                    });
                    let defaults = crate::server::diffusion_worker::DiffusionServeDefaults {
                        sampler,
                        confidence_threshold: sched_config.diffusion_threshold,
                        max_denoising_steps: sched_config.max_denoising_steps,
                    };
                    crate::server::diffusion_worker::run_diffusion_worker_loop(
                        &diffusion,
                        &tokenizer,
                        &model_path,
                        request_rx,
                        defaults,
                        &config_eos,
                    );
                    return;
                }
                // LLaDA-2 MoE (issue #546): masked-diffusion block generator,
                // served on the shared single-stream diffusion worker loop. The
                // serve-level `--max-denoising-steps` flag maps to the LLaDA-2
                // per-block step count.
                LoadedModel::Llada2Moe(llada2) => {
                    crate::server::diffusion_worker::run_llada2_worker_loop(
                        &llada2,
                        &tokenizer,
                        request_rx,
                        sched_config.max_denoising_steps,
                        &config_eos,
                    );
                    return;
                }
                other => other,
            };

            // Axis B (B8): resolve the server-wide LangBiasConfig once,
            // after the tokenizer is available. Empty bias set or an HF-less
            // tokenizer yields an empty map — bit-exact baseline preserved.
            let token_bias = resolve_worker_token_bias(
                sched_config.lang_bias_config.as_ref(),
                &tokenizer,
                &model_path,
            );

            // B9 — emit structured debug trace once at generator construction time
            // (after resolve, before the scheduler is started).
            if let (true, Some(cfg)) = (
                !token_bias.is_empty(),
                sched_config.lang_bias_config.as_ref(),
            ) {
                let langs: Vec<&str> = cfg
                    .bias_set
                    .ordered
                    .iter()
                    .map(|(code, _)| code.as_str())
                    .collect();
                let languages_str = langs.join(",");
                let policy_str = if cfg.policy == mlxcel_core::InclusionPolicy::Strict {
                    "strict"
                } else {
                    "conservative"
                };
                // emit byte_fragment_entries only when non-zero so
                // the existing B9 field shape is preserved for Phase 1 configs.
                let byte_fragment_entries = token_bias.byte_fragment_len();
                if byte_fragment_entries > 0 {
                    tracing::debug!(
                        entries = token_bias.len(),
                        byte_fragment_entries,
                        languages = %languages_str,
                        policy = %policy_str,
                        "lang_bias resolved"
                    );
                } else {
                    tracing::debug!(
                        entries = token_bias.len(),
                        languages = %languages_str,
                        policy = %policy_str,
                        "lang_bias resolved"
                    );
                }
            }

            let chunk_info = if sched_config.prefill_chunk_size > 0 {
                format!(", prefill_chunk_size={}", sched_config.prefill_chunk_size)
            } else {
                String::new()
            };
            let batch_prefill_info = if sched_config.max_batch_prefill > 1 {
                format!(", max_batch_prefill={}", sched_config.max_batch_prefill)
            } else {
                String::new()
            };
            let decode_storage_info = match sched_config.decode_storage_backend {
                crate::server::DecodeStorageBackend::Auto => ", decode_storage=auto".to_string(),
                crate::server::DecodeStorageBackend::Dense => String::new(),
                crate::server::DecodeStorageBackend::Paged => ", decode_storage=paged".to_string(),
            };
            let lang_bias_info = if !token_bias.is_empty() {
                format!(", lang_bias_tokens={}", token_bias.len())
            } else {
                String::new()
            };
            // log the resolved speculative dispatch once at
            // startup. This makes the operator-visible "which path is
            // active" explicit in the worker log without forcing the
            // scheduler to log per request.
            let spec_info = if !matches!(
                sched_config.speculative_dispatch,
                crate::server::SpeculativeDispatch::Disabled
            ) {
                format!(", {}", sched_config.speculative_dispatch.summary())
            } else {
                String::new()
            };

            // Serving-throughput default gate (#628): the `--parallel` /
            // `--max-batch-size` default admits a concurrent decode batch, but
            // SSM / hybrid / mixed-cache families keep recurrent or shared
            // internal state that is not compatible with per-sequence batching.
            // Clamp the effective batch to 1 for those families now that the
            // model is loaded, so the shipped default is safe for every
            // architecture ("4 when supports_batching(), 1 otherwise"). Resolved
            // before the scheduler log so the log reports the effective value.
            let effective_max_batch_size = if model.supports_batching() {
                sched_config.max_batch_size
            } else {
                if sched_config.max_batch_size > 1 {
                    tracing::info!(
                        "Model {} does not support batched decode; clamping \
                         max_batch_size {} -> 1 (single-slot sequential serving)",
                        worker_model_id,
                        sched_config.max_batch_size,
                    );
                }
                1
            };

            tracing::info!(
                "Starting BatchScheduler (max_batch_size={}, \
             max_queue_depth={}{chunk_info}{batch_prefill_info}{decode_storage_info}{lang_bias_info}{spec_info})",
                effective_max_batch_size,
                sched_config.max_queue_depth,
            );

            // speculative dispatch is wired end-to-end via
            // the burst path in `BatchScheduler::execute_prefill`. With
            // `max_batch_size > 1` the scheduler assembles an
            // equal-prompt-length window of concurrently-queued speculative
            // requests and drives them through the batched round-loop driver
            // (`MtpBatchedGenerator` / `DFlashBatchedGenerator`) in one tick
            // — true B>1 batched speculative decoding. A
            // speculative request whose prompt length, `max_tokens`, or
            // sampling config does not match the current window head, or
            // that arrives alone, runs on the B=1 arm. For MTP on the
            // Gemma 4 family that arm is tick-cooperative (issue #734):
            // the request is served one speculative round per scheduler
            // tick, alternating with the classic actions, so concurrent
            // classic-decode rows advance between rounds and the
            // head-of-line stall is bounded by about one round (see
            // `server::batch::speculative_slice`;
            // `MLXCEL_MTP_TICK_SLICE=0` restores the legacy
            // run-to-completion burst). The DFlash B=1 arm and every B>1
            // batched arm still run to completion inside one tick and
            // head-of-line-block concurrent rows for their full duration.
            //
            // Variable-length-prompt MTP bursts (different prompt lengths in one
            // B>1 window) are implemented behind the `MLXCEL_ENABLE_MTP_BATCH_RAGGED`
            // opt-in (subordinate to `MLXCEL_ENABLE_MTP_BATCH`): when enabled the
            // MTP adapter left-pads the window to `max_prompt_len` (eligible while
            // `max_prompt_len <= sliding_window`), preserving greedy parity via the
            // left-padding uniform per-row position shift.
            if sched_config.speculative_dispatch.is_kind_specific()
                && sched_config.max_batch_size > 1
            {
                tracing::info!(
                    "Speculative decoding active ({}) with max_batch_size={}: \
                 concurrently-queued speculative requests that share a \
                 prompt length, max_tokens, and sampling config are driven \
                 as a single B>1 batched burst. A speculative \
                 request that does not match the current window head, or \
                 that arrives alone, runs on the B=1 arm; MTP B=1 requests \
                 are served tick-cooperatively (one speculative round per \
                 scheduler tick, so concurrent classic-decode rows advance \
                 between rounds; MLXCEL_MTP_TICK_SLICE=0 restores the \
                 legacy run-to-completion burst), while DFlash B=1 and B>1 \
                 batched bursts run to completion inside one tick. \
                 Variable-length-prompt MTP batched bursts are \
                 available behind MLXCEL_ENABLE_MTP_BATCH_RAGGED=1 (with \
                 MLXCEL_ENABLE_MTP_BATCH=1).",
                    sched_config.speculative_dispatch.summary(),
                    sched_config.max_batch_size,
                );
            }

            // resolve the thinking-token id pair once, after the
            // tokenizer is loaded. For models without `<think>`/`</think>` tokens
            // (non-thinking models) this returns `None` and the scheduler silently
            // ignores any budget parameter (logging once per model load).
            let thinking_ids =
                crate::server::thinking_budget::resolve_thinking_token_ids(&tokenizer);
            if sched_config.reasoning_budget.is_some() && thinking_ids.is_none() {
                tracing::warn!(
                    "--reasoning-budget / thinking_budget_tokens requested but this model's \
                 tokenizer has no <think> / </think> tokens; thinking-budget enforcement \
                 is disabled for this session"
                );
            }

            // resolve the tokenizer's newline token id(s) once, for the
            // XTC (Exclude Top Choices) special-token allowlist. Combined
            // with each request's merged end-of-sequence set at enqueue time
            // (see `BatchScheduler::enqueue_request`).
            let xtc_newline_token_ids = resolve_xtc_newline_token_ids(&tokenizer);

            // #122 b3: resolve the `--kv-cache-budget` directive into a paged
            // block count now (the model's geometry is known) and install it on
            // the scheduler's pool below. A no-op (unbounded) when the flag is
            // unset. Computed before `model` is moved into `with_config`.
            let paged_block_budget = resolve_worker_paged_block_budget(
                &model_path,
                &model,
                effective_max_batch_size,
                sched_config.kv_cache_budget,
            );

            let mut scheduler = super::super::batch::BatchScheduler::with_config(
                model,
                tokenizer,
                config_eos,
                request_rx,
                effective_max_batch_size,
                sched_config.max_queue_depth,
                batch_metrics,
                batch_observability,
                sched_config.prefill_chunk_size,
                sched_config.enable_preemption,
                sched_config.preemption_policy,
                sched_config.max_batch_prefill,
                sched_config.decode_storage_backend,
            )
            .with_vision_cache_size(sched_config.vision_cache_size)
            // cap the batched-prefill transient to --max-batch-prefill-tokens (#715).
            .with_max_batch_prefill_tokens(sched_config.max_batch_prefill_tokens)
            .with_token_bias(token_bias)
            .with_xtc_newline_token_ids(xtc_newline_token_ids)
            .with_reasoning_budget(sched_config.reasoning_budget, thinking_ids)
            .with_prompt_cache(sched_config.prompt_cache)
            .with_kv_cache_mode(sched_config.kv_cache_mode)
            .with_batch_kv_quant(sched_config.batch_kv_quant)
            // cap plain KVCache growth to --max-kv-size when set.
            .with_max_kv_size(sched_config.max_kv_size)
            // install the resolved paged KV block budget (epic #116 #122 b3).
            .with_paged_block_budget(paged_block_budget)
            // experimental VLM prompt-prefix cache sharing (#124 step c).
            .with_vlm_prefix_cache(sched_config.enable_vlm_prefix_cache)
            // attach the resolved speculative dispatch so the
            // scheduler can branch per-request once the round-loop dispatch
            // hook is wired in `decode_single_step`.
            .with_speculative_dispatch(sched_config.speculative_dispatch)
            // attach the adaptive MTP policy (issue #333). Keyed on the served
            // model's directory basename (the coarse, non-request-identifying
            // target identity) plus the drafter basename and hardware class. A
            // no-op for non-MTP dispatch or when MLXCEL_MTP_ADAPTIVE is off.
            .with_mtp_policy(
                model_path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned()),
            );

            // #126 B3b2a: a non-hybrid `--node-role` runs the live disaggregated
            // serving role rather than the standard single-node loop. The role loop
            // binds this node's `--serving-bind` transport and drives prefill (or
            // decode) over the B1 handoff hooks, returning only when the transport
            // closes. A misconfigured role (no `--serving-bind`, or a prefill node
            // without `--decode-peers`) logs an error and falls back to serving
            // locally rather than hanging a half-configured node. `Hybrid` (the
            // default) and `Router` run the standard loop, byte-identical to before
            // (the dedicated router front lands in a later step).
            use crate::distributed::disaggregated::ServingMode;
            match sched_config.serving_mode {
                ServingMode::PrefillOnly | ServingMode::DecodeOnly => {
                    if !run_disaggregated_serving_role(
                        &mut scheduler,
                        sched_config.serving_mode,
                        sched_config.serving_bind,
                        &sched_config.decode_peers,
                    ) {
                        scheduler.serve();
                    }
                }
                ServingMode::Hybrid | ServingMode::Router => scheduler.serve(),
            }
        })
    })
}

/// Drive the live disaggregated serving role for a non-hybrid worker (#126
/// B3b2a).
///
/// Binds this node's `serving_bind` role transport and runs the prefill or
/// decode role loop ([`serve_prefill_role_networked_blocking`] /
/// [`serve_decode_role_networked_blocking`]), which returns when the transport
/// closes. Returns `false` without starting a loop when the role is
/// misconfigured (no `serving_bind`, or a prefill node with no `decode_peers`),
/// so the caller falls back to the standard single-node scheduler loop rather
/// than hanging a half-configured node. A role-loop error is logged before the
/// worker exits.
///
/// [`serve_prefill_role_networked_blocking`]: crate::distributed::disaggregated::coordinator::serve_prefill_role_networked_blocking
/// [`serve_decode_role_networked_blocking`]: crate::distributed::disaggregated::coordinator::serve_decode_role_networked_blocking
fn run_disaggregated_serving_role(
    scheduler: &mut crate::server::batch::BatchScheduler,
    serving_mode: crate::distributed::disaggregated::ServingMode,
    serving_bind: Option<std::net::SocketAddr>,
    decode_peers: &[std::net::SocketAddr],
) -> bool {
    use crate::distributed::disaggregated::ServingMode;
    use crate::distributed::disaggregated::coordinator::{
        serve_decode_role_networked_blocking, serve_prefill_role_networked_blocking,
    };
    use crate::distributed::tcp_transport::TcpTransportConfig;

    let Some(bind_addr) = serving_bind else {
        tracing::error!(
            serving_mode = %serving_mode,
            "Disaggregated serving role requires --serving-bind; \
             falling back to the single-node scheduler loop"
        );
        return false;
    };
    let bind = TcpTransportConfig {
        bind_address: bind_addr.to_string(),
        ..TcpTransportConfig::default()
    };

    let result = match serving_mode {
        ServingMode::PrefillOnly => {
            let Some(decode_peer) = decode_peers.first() else {
                tracing::error!(
                    "--node-role prefill requires --decode-peers (the decode node to \
                     hand KV off to); falling back to the single-node scheduler loop"
                );
                return false;
            };
            tracing::info!(
                bind = %bind_addr, decode_peer = %decode_peer,
                "Starting the disaggregated prefill serving role"
            );
            // Pass the configured decode peers: the first entry is the static
            // handoff fallback used when the router omits `decode_target`. The
            // router-target allowlist is read separately from the dedicated
            // MLXCEL_DECODE_ALLOWLIST env input in the coordinator (issue #389).
            serve_prefill_role_networked_blocking(bind, decode_peers.to_vec(), scheduler, None)
        }
        ServingMode::DecodeOnly => {
            tracing::info!(bind = %bind_addr, "Starting the disaggregated decode serving role");
            serve_decode_role_networked_blocking(bind, scheduler, None)
        }
        // The caller only invokes this for a non-hybrid serving role.
        ServingMode::Hybrid | ServingMode::Router => return false,
    };

    if let Err(e) = result {
        tracing::error!(
            serving_mode = %serving_mode,
            "Disaggregated serving role loop exited with an error: {e:#}"
        );
    }
    true
}

/// Resolve the operator's `--kv-cache-budget` directive into a concrete paged
/// KV block count for this worker's scheduler pool (epic #116 #122 b3).
///
/// Returns `None` (leave the pool unbounded) when the flag is unset, the model
/// geometry is unavailable, or the budget rounds below one block — in the last
/// case a warning is logged rather than installing a zero budget that would
/// reject every request. `batch` is the configured active-sequence count (it
/// scales the activation reserve under [`PagedBudgetDirective::Auto`]).
///
/// [`PagedBudgetDirective::Auto`]: crate::memory_estimate::PagedBudgetDirective::Auto
fn resolve_worker_paged_block_budget(
    model_path: &std::path::Path,
    model: &LoadedModel,
    batch: usize,
    directive: Option<crate::memory_estimate::PagedBudgetDirective>,
) -> Option<usize> {
    let directive = directive?;
    // Explicit opt-out: leave the pool unbounded without the "geometry
    // unavailable" warning path (#628 default budget guard escape hatch).
    if matches!(
        directive,
        crate::memory_estimate::PagedBudgetDirective::Disabled
    ) {
        tracing::info!("--kv-cache-budget disabled; leaving the paged KV pool unbounded");
        return None;
    }
    let num_layers = model.num_layers();
    let block_size = crate::server::batch::scheduler::DEFAULT_PAGED_BLOCK_SIZE;
    // The paged pool stores Fp16; Int8 / Turbo sequences keep dense caches and
    // ignore the budget, so the per-block cost is computed at Fp16.
    let blocks = crate::memory_estimate::resolve_paged_block_budget(
        model_path,
        num_layers,
        block_size,
        batch.max(1) as u64,
        false,
        directive,
    );
    match blocks {
        Some(n) if n > 0 => {
            tracing::info!(
                "Paged KV block budget: {n} blocks ({num_layers} layers, \
                 {block_size}-token blocks)"
            );
            Some(n)
        }
        Some(_) => {
            tracing::warn!(
                "--kv-cache-budget resolves to 0 KV blocks at this configuration \
                 (model too large for a meaningful paged budget at this batch / \
                 available memory); leaving the paged pool unbounded"
            );
            None
        }
        None => {
            tracing::warn!(
                "--kv-cache-budget was set but the model's KV geometry is \
                 unavailable; leaving the paged pool unbounded"
            );
            None
        }
    }
}

/// Resolve the worker-level Axis B `LangBiasConfig` into a concrete
/// `TokenBiasMap` using the loaded tokenizer (B8).
///
/// Returns an empty map (baseline no-op) in the following cases:
/// - No `lang_bias_config` was supplied.
/// - The bias set is empty.
/// - The tokenizer is not HuggingFace-backed (SentencePiece/Tiktoken are not
///   supported by the Phase 1 vocabulary scanner).
/// - `tokenizer.json` cannot be read from disk.
/// - The resolver itself fails (logged; generation continues without bias).
fn resolve_worker_token_bias(
    config: Option<&mlxcel_core::lang_analyzer::LangBiasConfig>,
    tokenizer: &crate::tokenizer::MlxcelTokenizer,
    model_path: &std::path::Path,
) -> mlxcel_core::sampling::TokenBiasMap {
    let Some(cfg) = config else {
        return mlxcel_core::sampling::TokenBiasMap::default();
    };
    if cfg.bias_set.ordered.is_empty() {
        return mlxcel_core::sampling::TokenBiasMap::default();
    }
    let Some(hf) = tokenizer.hf_tokenizer() else {
        tracing::warn!(
            "--lang-bias/LLAMA_ARG_LANG_BIAS requested but this model uses a \
             non-HuggingFace tokenizer (SentencePiece/Tiktoken); language \
             steering is disabled for this session"
        );
        return mlxcel_core::sampling::TokenBiasMap::default();
    };
    let json_path = model_path.join("tokenizer.json");
    let json_bytes = match std::fs::read(&json_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                "failed to read {json_path:?} for lang-bias vocab-hash cache key: \
                 {err}; language steering disabled for this session"
            );
            return mlxcel_core::sampling::TokenBiasMap::default();
        }
    };
    match cfg.resolve_token_bias(hf, &json_bytes) {
        Ok(map) => map,
        Err(err) => {
            tracing::warn!(
                "failed to resolve language bias (vocab scan): {err}; language \
                 steering disabled for this session"
            );
            mlxcel_core::sampling::TokenBiasMap::default()
        }
    }
}

/// Spawn the legacy sequential model worker.
///
/// This worker processes one request at a time using the `BatchScheduler` with
/// `max_batch_size=1` and no chunked prefill, which is functionally equivalent
/// to the pre-scheduler sequential `recv()` loop. It is activated when
/// `--no-batch` is passed on the CLI.
///
/// Choosing this path explicitly guarantees:
/// - No batch scheduling data structures are allocated beyond size-1.
/// - No prefill chunking interleaving occurs.
/// - Log output clearly indicates the sequential execution mode.
///
/// The CLI `generate` command is unaffected and uses `CxxGenerator` directly.
pub(crate) fn spawn_legacy_model_worker(
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    tensor_parallel: crate::distributed::ShardConfig,
    reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,
    request_rx: mpsc::Receiver<ModelRequest>,
    loaded: Arc<AtomicBool>,
    worker_model_id: String,
    batch_metrics: Arc<BatchMetrics>,
    batch_observability: Arc<BatchObservability>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Same fail-fast posture as the batched worker above (issue #375): the
        // legacy sequential generation thread aborts the process on an uncaught
        // panic under release `panic = "unwind"` rather than unwinding away.
        run_core_thread_or_abort("model-worker-legacy", move || {
            tracing::info!(
                "Model worker thread starting (legacy sequential mode, --no-batch), loading model..."
            );

            let load_start = Instant::now();
            // Route model loading through the compute-backend seam (issue #338);
            // folds to the MLX backend under default features.
            let backend = crate::backend::select_backend();
            let result = if tensor_parallel.tp_size > 1 {
                backend.load_model_with_tensor_parallel(
                    &model_path,
                    adapter_path.as_deref(),
                    &tensor_parallel,
                )
            } else if let Some(adapter) = adapter_path {
                tracing::info!("Loading LoRA adapter from {:?}", adapter);
                backend.load_model_with_adapter(&model_path, &adapter)
            } else {
                backend.load_model(&model_path)
            };

            let (model, tokenizer) = match result {
                Ok((model, tokenizer)) => {
                    let load_elapsed = load_start.elapsed();
                    // Issue #55: log MLX-allocator resident memory after a
                    // successful weight load so operators see the actual
                    // working set the model occupies (not just the tensor
                    // sum). Useful for capacity planning and for the future
                    // preflight (#56) which will compare this against
                    // `MLXCEL_MEMORY_LIMIT` to fail fast.
                    let snap = mlxcel_core::memory::snapshot();
                    tracing::info!(
                        worker_model_id = %worker_model_id,
                        load_seconds = load_elapsed.as_secs_f64(),
                        active_bytes = snap.active_bytes,
                        peak_bytes = snap.peak_bytes,
                        cache_bytes = snap.cache_bytes,
                        limit_bytes = snap.limit_bytes,
                        "Model {worker_model_id} loaded in {:.3}s (resident after load: {:.2} GB)",
                        load_elapsed.as_secs_f64(),
                        snap.active_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    );
                    loaded.store(true, Ordering::Release);
                    (model, tokenizer)
                }
                Err(err) => {
                    tracing::error!("Failed to load model: {err}");
                    return;
                }
            };

            let config_eos = crate::read_eos_token_ids(&model_path);
            if !config_eos.is_empty() {
                tracing::info!("EOS tokens from config: {:?}", config_eos);
            }

            // DiffusionGemma (issue #217 phase 3): serve the block-diffusion model
            // on its dedicated batch-1 loop. The legacy worker has no serve-level
            // diffusion flags wired in (like `--vision-cache-size`), so it uses the
            // engine defaults; operators who need to tune the diffusion knobs use
            // the default batched worker.
            let model = match model {
                LoadedModel::DiffusionGemma(diffusion) => {
                    crate::server::diffusion_worker::run_diffusion_worker_loop(
                        &diffusion,
                        &tokenizer,
                        &model_path,
                        request_rx,
                        crate::server::diffusion_worker::DiffusionServeDefaults::default(),
                        &config_eos,
                    );
                    return;
                }
                // LLaDA-2 MoE (issue #546): served on the shared single-stream
                // loop. The legacy worker has no serve-level diffusion flags, so
                // the engine step default applies (steps_override = None).
                LoadedModel::Llada2Moe(llada2) => {
                    crate::server::diffusion_worker::run_llada2_worker_loop(
                        &llada2,
                        &tokenizer,
                        request_rx,
                        None,
                        &config_eos,
                    );
                    return;
                }
                other => other,
            };

            tracing::info!(
                "Starting legacy sequential worker \
             (max_batch_size=1, prefill_chunk_size=disabled)"
            );

            // resolve the thinking-token id pair once, after the
            // tokenizer is loaded. Mirrors the batched-worker path in
            // `spawn_model_worker_with_batch_config`. For models without
            // `<think>`/`</think>` tokens the helper returns `None` and the
            // scheduler silently ignores any budget parameter (after the
            // warn-once log below).
            let thinking_ids =
                crate::server::thinking_budget::resolve_thinking_token_ids(&tokenizer);
            if reasoning_budget.is_some() && thinking_ids.is_none() {
                tracing::warn!(
                    "--reasoning-budget / thinking_budget_tokens requested but this model's \
                 tokenizer has no <think> / </think> tokens; thinking-budget enforcement \
                 is disabled for this session"
                );
            }

            // resolve the tokenizer's newline token id(s) once, mirroring
            // the batched-worker path above, for the XTC special-token
            // allowlist.
            let xtc_newline_token_ids = resolve_xtc_newline_token_ids(&tokenizer);

            // Reuse BatchScheduler with max_batch_size=1 and chunking disabled.
            // Per the scheduler docs, size-1 behavior is identical to the old
            // sequential recv() loop, with no extra overhead.
            //
            // The legacy worker uses the default vision cache size because it
            // currently does not receive the normalized server config; users who
            // need to tune `--vision-cache-size` should use the default batched
            // worker which wires the flag through `WorkerSchedulerConfig`.
            let mut scheduler = super::super::batch::BatchScheduler::with_config(
                model,
                tokenizer,
                config_eos,
                request_rx,
                1,          // max_batch_size = 1 → sequential, no interleaving
                usize::MAX, // max_queue_depth: unbounded (one at a time anyway)
                batch_metrics,
                batch_observability,
                0,     // prefill_chunk_size = 0 → chunking disabled
                false, // enable_preemption = false
                crate::server::config::PreemptionPolicy::default(),
                1, // max_batch_prefill = 1 → sequential prefill
                crate::server::DecodeStorageBackend::Dense,
            )
            .with_xtc_newline_token_ids(xtc_newline_token_ids)
            .with_reasoning_budget(reasoning_budget, thinking_ids);
            scheduler.serve();
        })
    })
}

/// Spawn the OpenXLA / IREE serve worker (issue #449 M3 Stage 2c).
///
/// Parallel to [`spawn_legacy_model_worker`], but for the OpenXLA backend: it
/// builds the `mlxcel-xla` continuous-batching engine and a standalone tokenizer
/// inside the worker thread (so loading does not block the server start, same as
/// the MLX path), marks the model loaded, then drives the
/// [`XlaServeWorker`](crate::server::batch::XlaServeWorker) through the
/// [`BatchEngine`](crate::server::batch::BatchEngine) contract. `b_max` is one of
/// the engine's bundled slot counts; the HAL device is read from
/// `MLXCEL_XLA_DEVICE` (default [`mlxcel_xla::default_device`]: `metal` on Apple
/// Silicon, `local-task` elsewhere), matching the single-sequence path.
#[cfg(feature = "xla-iree")]
pub(crate) fn spawn_xla_model_worker(
    model_path: PathBuf,
    b_max: usize,
    request_rx: mpsc::Receiver<ModelRequest>,
    loaded: Arc<AtomicBool>,
    worker_model_id: String,
    batch_metrics: Arc<BatchMetrics>,
    batch_observability: Arc<BatchObservability>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Same fail-fast posture as the MLX workers (issue #375): abort the
        // process on an uncaught panic rather than unwinding away a serve thread.
        run_core_thread_or_abort("model-worker-xla", move || {
            let device = std::env::var("MLXCEL_XLA_DEVICE")
                .unwrap_or_else(|_| mlxcel_xla::default_device().to_string());
            tracing::info!(
                "Model worker thread starting (OpenXLA continuous batching, B_max={b_max}, \
                 device={device}), loading model..."
            );

            let load_start = Instant::now();
            let tokenizer = match crate::tokenizer::load_tokenizer(&model_path) {
                Ok(t) => t,
                Err(err) => {
                    tracing::error!("Failed to load tokenizer for the OpenXLA backend: {err}");
                    return;
                }
            };
            let engine = match mlxcel_xla::XlaBatchEngine::load(&model_path, b_max, &device) {
                Ok(engine) => engine,
                Err(err) => {
                    tracing::error!("Failed to load the OpenXLA engine: {err}");
                    return;
                }
            };
            let load_elapsed = load_start.elapsed();
            tracing::info!(
                worker_model_id = %worker_model_id,
                load_seconds = load_elapsed.as_secs_f64(),
                "OpenXLA model {worker_model_id} loaded in {:.3}s (B_max={b_max}, device={device})",
                load_elapsed.as_secs_f64(),
            );
            loaded.store(true, Ordering::Release);

            let mut worker = crate::server::batch::XlaServeWorker::new(
                engine,
                tokenizer,
                request_rx,
                batch_metrics,
                batch_observability,
            );
            worker.serve();
        })
    })
}

pub(crate) fn merge_config_stop_tokens(
    mut sampling: SamplingConfig,
    config_eos: &[i32],
) -> SamplingConfig {
    for &id in config_eos {
        if !sampling.stop_token_ids.contains(&id) {
            sampling.stop_token_ids.push(id);
        }
    }
    sampling
}

pub(crate) fn decode_request_images(images: &[Vec<u8>]) -> Result<Vec<DynamicImage>> {
    decode_request_images_with_limits(images, current_image_input_limits())
}

pub(crate) fn decode_request_images_with_limits(
    images: &[Vec<u8>],
    limits: ImageInputLimits,
) -> Result<Vec<DynamicImage>> {
    let mut decoded_images: Vec<DynamicImage> = Vec::with_capacity(images.len());

    for bytes in images {
        match decode_request_image_with_limits(bytes, limits) {
            Ok(image) => decoded_images.push(image),
            Err(err) if is_image_limit_error(&err) => {
                return Err(anyhow!("Image decode rejected by configured limits: {err}"));
            }
            Err(err) => {
                tracing::warn!("Failed to decode image: {}", err);
            }
        }
    }

    if decoded_images.is_empty() {
        Err(anyhow!("Failed to decode any images"))
    } else {
        Ok(decoded_images)
    }
}

fn decode_request_image_with_limits(
    bytes: &[u8],
    limits: ImageInputLimits,
) -> image::ImageResult<DynamicImage> {
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    reader.limits(limits.image_decode_limits());
    reader.decode()
}

fn is_image_limit_error(err: &ImageError) -> bool {
    matches!(err, ImageError::Limits(_))
}

/// `image_soft_tokens` is the request-scoped Gemma 4 image soft-token budget
/// (`None` = use the checkpoint's configured budget). It is already validated
/// against the supported ladder at the request boundary. Every non-Gemma-4
/// family ignores it.
pub(crate) fn prepare_request_vlm_embeddings(
    model: &LoadedModel,
    tokenizer: &MlxcelTokenizer,
    prompt: &str,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
    videos: &[crate::server::media::ResolvedVideo],
    vision_caches: Option<&ModelVisionCaches>,
    image_soft_tokens: Option<usize>,
) -> Result<Option<InputEmbeddings>> {
    let has_media = !images.is_empty() || !audio.is_empty() || !videos.is_empty();

    if !has_media || !model.is_vlm() {
        // Moondream3 needs special prompt formatting even for text-only
        if images.is_empty() && matches!(model, LoadedModel::Moondream3VLM(_)) {
            let prepared = crate::moondream3_prompt::prepare_moondream3_prompt_tokens(
                prompt,
                0,
                |text, add_special| {
                    tokenizer
                        .encode(text, add_special)
                        .unwrap_or_default()
                        .iter()
                        .map(|&t| t as i32)
                        .collect()
                },
            )
            .map_err(|e| anyhow!("{}", e))?;
            *prompt_tokens = prepared.tokens;
        } else if images.is_empty()
            && let LoadedModel::Moondream2VLM(moondream2) = model
        {
            // Moondream2 frames text-only prompts the same way (BOS + question).
            let prepared = crate::moondream2_prompt::prepare_moondream2_prompt_tokens(
                prompt,
                0,
                moondream2.prompt_style,
                |text, add_special| {
                    tokenizer
                        .encode(text, add_special)
                        .unwrap_or_default()
                        .iter()
                        .map(|&t| t as i32)
                        .collect()
                },
            )
            .map_err(|e| anyhow!("{}", e))?;
            *prompt_tokens = prepared.tokens;
        }
        return Ok(None);
    }

    // video inputs route to the Gemma 4 video embedding path,
    // mirroring the CLI dispatch in `commands/generate_vlm.rs::compute_vlm_embeddings`.
    // Combining --video with --audio is rejected upstream and at the route
    // layer; this branch additionally surfaces a clear error if those happen
    // to coexist (defence in depth).
    if !videos.is_empty() {
        if !audio.is_empty() {
            return Err(anyhow!("Combined video and audio inputs are not supported"));
        }
        return prepare_request_video_embeddings(
            model,
            prompt_tokens,
            images,
            videos,
            image_soft_tokens,
        );
    }

    // Audio-only or audio+images for Gemma4 / Gemma4 Unified
    if !audio.is_empty() {
        // The server renders chat messages text-only, so the prompt carries no
        // `<|audio|>` marker (issue #437). Resolve the Gemma `<end_of_turn>`
        // id so the per-family audio expansion can place the audio block inside
        // the last user turn instead of the model turn (which forces an
        // immediate EOS / 0-token output).
        let end_of_turn_token_id = resolve_end_of_turn_token_id(tokenizer);
        if let Some(embeddings) = prepare_gemma4_unified_audio_embeddings(
            model,
            prompt_tokens,
            images,
            audio,
            end_of_turn_token_id,
            image_soft_tokens,
        )? {
            return Ok(Some(embeddings));
        }
        if let Some(embeddings) = prepare_gemma4_audio_embeddings(
            model,
            prompt_tokens,
            images,
            audio,
            end_of_turn_token_id,
            image_soft_tokens,
        )? {
            return Ok(Some(embeddings));
        }
        if let Some(embeddings) = prepare_qwen3_omni_audio_embeddings(
            model,
            prompt_tokens,
            images,
            audio,
            end_of_turn_token_id,
        )? {
            return Ok(Some(embeddings));
        }
        match prepare_nemotron_h_nano_omni_audio_embeddings(
            model,
            prompt_tokens,
            images,
            audio,
            end_of_turn_token_id,
        )? {
            Some(embeddings) => return Ok(Some(embeddings)),
            None => {
                // Model does not support audio (not Gemma4 / Nemotron H Nano
                // Omni, or no audio tower). Log a warning and fall through to
                // image-only or text-only paths.
                tracing::warn!("Audio input provided but model does not support audio; ignoring");
            }
        }
    }

    // Standard image-only path. When a per-model vision cache is available and
    // enabled, the cache-aware variant is used so repeated images across
    // multi-turn conversations skip the vision tower. The hashing identity is
    // derived from the request-supplied image bytes so path-less (inline)
    // payloads still benefit from de-duplication.
    if !images.is_empty() {
        let decoded_images = decode_request_images(images)?;
        let prepared = if let Some(caches) = vision_caches.filter(|c| c.enabled()) {
            // The cached value is the vision tower's output, which for Gemma 4
            // depends on the soft-token budget as well as the image bytes.
            // Fold the budget into the key so the same image requested at two
            // budgets cannot serve one's features for the other.
            let image_cache_keys: Vec<Option<crate::vision::feature_cache::CacheKey>> = images
                .iter()
                .map(|bytes| {
                    Some(crate::vision::feature_cache::CacheKey::from_hash(
                        crate::vision::feature_cache::image_hash_from_bytes_with_soft_tokens(
                            bytes,
                            image_soft_tokens,
                        ),
                    ))
                })
                .collect();
            prepare_and_compute_vlm_embeddings_with_cache(
                model,
                prompt_tokens,
                prompt,
                &decoded_images,
                Some(&image_cache_keys),
                Some(caches),
                image_soft_tokens,
                |text, add_special| {
                    tokenizer
                        .encode(text, add_special)
                        .unwrap_or_default()
                        .iter()
                        .map(|&t| t as i32)
                        .collect()
                },
            )?
        } else {
            prepare_and_compute_vlm_embeddings_with_budget(
                model,
                prompt_tokens,
                prompt,
                &decoded_images,
                image_soft_tokens,
                |text, add_special| {
                    tokenizer
                        .encode(text, add_special)
                        .unwrap_or_default()
                        .iter()
                        .map(|&t| t as i32)
                        .collect()
                },
            )?
        };
        return Ok(prepared.map(|prepared| prepared.embeddings));
    }

    // If we reach here with no images (audio was present but unsupported),
    // apply model-specific text-only formatting if needed.
    if images.is_empty() && matches!(model, LoadedModel::Moondream3VLM(_)) {
        let prepared = crate::moondream3_prompt::prepare_moondream3_prompt_tokens(
            prompt,
            0,
            |text, add_special| {
                tokenizer
                    .encode(text, add_special)
                    .unwrap_or_default()
                    .iter()
                    .map(|&t| t as i32)
                    .collect()
            },
        )
        .map_err(|e| anyhow!("{}", e))?;
        *prompt_tokens = prepared.tokens;
    } else if images.is_empty()
        && let LoadedModel::Moondream2VLM(moondream2) = model
    {
        let prepared = crate::moondream2_prompt::prepare_moondream2_prompt_tokens(
            prompt,
            0,
            moondream2.prompt_style,
            |text, add_special| {
                tokenizer
                    .encode(text, add_special)
                    .unwrap_or_default()
                    .iter()
                    .map(|&t| t as i32)
                    .collect()
            },
        )
        .map_err(|e| anyhow!("{}", e))?;
        *prompt_tokens = prepared.tokens;
    }

    Ok(None)
}

/// Resolve the per-family end-of-turn token id from the tokenizer.
///
/// The server flattens chat messages to text-only, so an `input_audio` request
/// produces a prompt with no audio placeholder marker. Knowing the end-of-turn
/// id lets the per-family server audio expanders
/// ([`crate::vlm_runtime::expand_gemma4_audio_tokens_for_server`] and
/// [`crate::vlm_runtime::expand_nemotron_h_nano_omni_audio_tokens_for_server`])
/// place the audio block inside the last user turn instead of the assistant
/// turn (issue #437). Returns `None` when no known marker is a single token in
/// this tokenizer, in which case the caller keeps the legacy "before the final
/// token" insertion.
fn resolve_end_of_turn_token_id(tokenizer: &MlxcelTokenizer) -> Option<i32> {
    // The end-of-turn marker differs across families: Gemma 2/3 use
    // "<end_of_turn>", Gemma 4 renamed it to "<turn|>" (id 106, with "<|turn>"
    // for start-of-turn), and the Nemotron H Nano Omni ChatML template closes
    // every turn with "<|im_end|>". Try them in order so the audio block lands
    // inside the last user turn on every supported checkpoint; the Gemma
    // markers are tried first so the Gemma audio path keeps resolving to its
    // own marker even on a tokenizer that also defines "<|im_end|>".
    const EOT_CANDIDATES: &[&str] = &["<end_of_turn>", "<turn|>", "<|im_end|>"];
    if let Some(hf) = tokenizer.hf_tokenizer() {
        for candidate in EOT_CANDIDATES {
            if let Some(id) = hf.token_to_id(candidate) {
                return Some(id as i32);
            }
        }
    }
    // SentencePiece / Tiktoken fallback: accept only when the literal marker
    // encodes to exactly one token, so a tokenizer that splits it into pieces
    // does not yield a bogus mid-vocabulary id.
    for candidate in EOT_CANDIDATES {
        if let Ok(ids) = tokenizer.encode(candidate, false)
            && ids.len() == 1
        {
            return Some(ids[0] as i32);
        }
    }
    None
}

/// Resolve the tokenizer's newline token id(s), once per model load.
///
/// XTC (Exclude Top Choices) sampling must never remove a token needed to end
/// a line, so these ids are folded into the per-request special-token
/// allowlist alongside the merged end-of-sequence set (see
/// `BatchScheduler::enqueue_request`). Encoding the bare `"\n"` character
/// (no special tokens) covers both a single-token byte-level newline piece
/// (e.g. the GPT-2/Llama-style `Ċ`) and any tokenizer that splits it into
/// more than one id. An encode failure (not expected for a model's own
/// tokenizer) yields an empty allowlist contribution rather than aborting
/// startup.
fn resolve_xtc_newline_token_ids(tokenizer: &MlxcelTokenizer) -> Vec<i32> {
    tokenizer
        .encode("\n", false)
        .map(|ids| ids.into_iter().map(|id| id as i32).collect())
        .unwrap_or_default()
}

/// Process audio (and optionally images) for Gemma4 VLM models.
///
/// Returns `Ok(None)` if the model is not a Gemma4 VLM with audio support.
fn prepare_gemma4_audio_embeddings(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    audio_data: &[Vec<u8>],
    end_of_turn_token_id: Option<i32>,
    image_soft_tokens: Option<usize>,
) -> Result<Option<InputEmbeddings>> {
    use crate::audio;

    let gemma4_vl = match model {
        LoadedModel::Gemma4VLM(vl) => vl,
        _ => return Ok(None),
    };

    if gemma4_vl.audio_tower.is_none() {
        tracing::warn!("Gemma4 model has no audio encoder; ignoring audio input");
        return Ok(None);
    }

    if audio_data.len() > 1 {
        tracing::warn!(
            "Multiple audio inputs provided ({}); only the first will be processed",
            audio_data.len()
        );
    }

    // Process the first audio input
    let audio_bytes = &audio_data[0];
    let (samples, sample_rate) = audio::load_wav_from_bytes(audio_bytes)
        .map_err(|e| anyhow!("Failed to decode audio: {}", e))?;

    tracing::info!(
        "Audio input: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate as f64
    );

    let num_audio_tokens = audio::compute_audio_num_tokens(samples.len(), sample_rate, 40, 750);

    // Expand audio tokens: BOA + AUDIO*N + EOA, placed inside the last user
    // turn (issue #437).
    crate::vlm_runtime::expand_gemma4_audio_tokens_for_server(
        prompt_tokens,
        gemma4_vl.audio_token_id,
        gemma4_vl.boa_token_id,
        gemma4_vl.eoa_token_id,
        num_audio_tokens,
        end_of_turn_token_id,
    );

    // `AudioFeatureExtractor::extract` assumes a 16 kHz waveform (160-sample
    // hop = 10 ms). `load_wav_from_bytes` returns native-rate samples, so
    // without resampling the Conformer encoder emits the wrong frame count and
    // desyncs from the duration-based placeholder count above, garbling the
    // audio embeddings (issue #436). Resample to 16 kHz before mel extraction;
    // duration (and thus `num_audio_tokens`) is rate-invariant.
    let samples = if sample_rate != 16_000 {
        audio::whisper_mel::resample_to_16k(&samples, sample_rate)
    } else {
        samples
    };

    // Extract mel spectrogram features with the checkpoint's
    // feature-extractor config (reference defaults when absent).
    let extractor = audio::AudioFeatureExtractor::new(gemma4_vl.audio_extractor_config.clone());
    let (features, mask) = extractor.extract(&samples, None);
    let num_frames = mask.len();

    let audio_features = mlxcel_core::from_slice_f32(
        &features,
        &[1, num_frames as i32, extractor.feature_size() as i32],
    );
    let mask_i32: Vec<i32> = mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let audio_mask = mlxcel_core::from_slice_i32(&mask_i32, &[1, num_frames as i32]);
    let audio_mask = mlxcel_core::astype(&audio_mask, mlxcel_core::dtype::BOOL);

    // Process images if present alongside audio. The placeholder expansion is
    // driven by the `num_soft_tokens` this exact preprocess call produced, so
    // the image-token run in the prompt matches the feature rows one-for-one at
    // whatever budget the request asked for.
    let processed_images = if !images.is_empty() {
        let decoded_images = decode_request_images(images)?;
        let processed = gemma4_vl
            .processor
            .preprocess_with_budget(&decoded_images, image_soft_tokens);
        let num_soft_tokens: Vec<usize> = processed.iter().map(|img| img.num_soft_tokens).collect();
        crate::vlm_runtime::expand_gemma4_image_tokens_pub(
            prompt_tokens,
            gemma4_vl.image_token_id,
            gemma4_vl.boi_token_id,
            gemma4_vl.eoi_token_id,
            &num_soft_tokens,
        )?;
        processed
    } else {
        Vec::new()
    };

    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = gemma4_vl
        .get_input_embeddings_with_audio(
            &input_ids_arr,
            &processed_images,
            Some(&audio_features),
            Some(&audio_mask),
        )
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(Some(embeddings))
}

/// Process audio (and optionally images) for Gemma 4 Unified models.
///
/// Encoder-free: the raw waveform is chunked into `audio_samples_per_token`
/// frames (no mel spectrogram, no Conformer) and projected by `embed_audio`.
/// Returns `Ok(None)` when the model is not a Gemma 4 Unified model with audio.
fn prepare_gemma4_unified_audio_embeddings(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    audio_data: &[Vec<u8>],
    end_of_turn_token_id: Option<i32>,
    image_soft_tokens: Option<usize>,
) -> Result<Option<InputEmbeddings>> {
    use crate::audio;

    let unified = match model {
        LoadedModel::Gemma4Unified(m) => m,
        _ => return Ok(None),
    };

    if unified.embed_audio.is_none() {
        tracing::warn!("Gemma4 Unified model has no audio embedder; ignoring audio input");
        return Ok(None);
    }

    if audio_data.len() > 1 {
        tracing::warn!(
            "Multiple audio inputs provided ({}); only the first will be processed",
            audio_data.len()
        );
    }

    let (samples, sample_rate) = audio::load_wav_from_bytes(&audio_data[0])
        .map_err(|e| anyhow!("Failed to decode audio: {}", e))?;
    tracing::info!(
        "Gemma4 Unified audio input: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate.max(1) as f64
    );

    let audio_input = unified.processor.process_audio(&samples);
    let num_audio_tokens = audio_input.num_frames;

    crate::vlm_runtime::expand_gemma4_audio_tokens_for_server(
        prompt_tokens,
        unified.audio_token_id,
        unified.boa_token_id,
        unified.eoa_token_id,
        num_audio_tokens,
        end_of_turn_token_id,
    );

    // Process images alongside audio (encoder-free patch projector).
    let processed_images = if !images.is_empty() {
        let decoded_images = decode_request_images(images)?;
        let processed = unified
            .processor
            .preprocess_with_budget(&decoded_images, image_soft_tokens);
        let num_soft_tokens: Vec<usize> = processed.iter().map(|img| img.num_soft_tokens).collect();
        crate::vlm_runtime::expand_gemma4_image_tokens_pub(
            prompt_tokens,
            unified.image_token_id,
            unified.boi_token_id,
            unified.eoi_token_id,
            &num_soft_tokens,
        )?;
        processed
    } else {
        Vec::new()
    };

    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = unified.get_input_embeddings_with_audio(
        &input_ids_arr,
        &processed_images,
        Some(&audio_input.features),
        Some(&audio_input.mask),
    );

    Ok(Some(embeddings))
}

/// Process audio (and optionally images) for the Nemotron H Nano Omni VLM.
///
/// Mirrors the CLI builder `compute_nemotron_h_nano_omni_audio_embeddings` in
/// `src/commands/generate_vlm.rs`: it runs the Parakeet feature extractor,
/// derives the post-subsampling audio token count, places the sound-context
/// block inside the last user turn, runs the encoder + projector via
/// `extract_audio_features`, and scatters the audio rows through
/// `get_input_embeddings_full`. The encoder forward is the model method, not a
/// duplicate.
///
/// Returns `Ok(None)` when the model is not a Nemotron H Nano Omni VLM, or it
/// was loaded without an audio bundle or `sound_context_token_id`, so the
/// dispatch in [`prepare_request_vlm_embeddings`] falls through to the next
/// audio handler / the "model does not support audio" warning.
/// Resolve `input_audio` (optionally with images) into Qwen3-Omni thinker
/// embeddings. Mirrors the CLI's `compute_qwen3_omni_audio_embeddings` in
/// `src/commands/generate_vlm.rs`.
fn prepare_qwen3_omni_audio_embeddings(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    audio_data: &[Vec<u8>],
    end_of_turn_token_id: Option<i32>,
) -> Result<Option<InputEmbeddings>> {
    use crate::audio;
    use crate::audio::qwen3_omni_moe::audio_out_len;
    use crate::audio::whisper_mel::{log_mel_spectrogram, resample_to_16k};

    let omni_vl = match model {
        LoadedModel::Qwen3OmniMoe(vl) => vl,
        _ => return Ok(None),
    };

    if audio_data.len() > 1 {
        tracing::warn!(
            "Multiple audio inputs provided ({}); only the first will be processed",
            audio_data.len()
        );
    }

    let (samples, sample_rate) = audio::load_wav_from_bytes(&audio_data[0])
        .map_err(|e| anyhow!("Failed to decode audio: {}", e))?;
    tracing::info!(
        "Audio input: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate.max(1) as f64
    );
    let samples = if sample_rate != 16_000 {
        resample_to_16k(&samples, sample_rate)
    } else {
        samples
    };
    let (mel, num_frames) = log_mel_spectrogram(&samples, 128);
    if num_frames == 0 {
        return Err(anyhow!("Audio clip too short to produce mel frames"));
    }
    let num_audio_tokens = audio_out_len(num_frames).max(1);

    crate::vlm_runtime::expand_qwen3_omni_audio_tokens_for_server(
        prompt_tokens,
        omni_vl.audio_token_id,
        omni_vl.audio_start_token_id,
        omni_vl.audio_end_token_id,
        num_audio_tokens,
        end_of_turn_token_id,
    );

    // Optional image branch: same preprocessing + placeholder insertion the
    // image-only runtime path performs.
    let images_data = if !images.is_empty() {
        let decoded_images = decode_request_images(images)?;
        let (pixel_values, grid_thw) = omni_vl.processor.preprocess_with_grid(&decoded_images);
        crate::qwen_vl::insert_qwen3_omni_image_tokens(
            prompt_tokens,
            &grid_thw,
            omni_vl.spatial_merge_size,
            omni_vl.vision_start_token_id,
            omni_vl.image_token_id,
            omni_vl.im_end_token_id,
        );
        Some((pixel_values, grid_thw))
    } else {
        None
    };

    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let audio_features = omni_vl.extract_audio_features(&mel, num_frames);
    let embeddings = omni_vl.input_embeddings_multimodal(
        &input_ids_arr,
        images_data
            .as_ref()
            .map(|(pv, grid)| (pv.as_ref().unwrap() as &mlxcel_core::MlxArray, &grid[..])),
        Some(&audio_features),
    );

    Ok(Some(embeddings))
}

fn prepare_nemotron_h_nano_omni_audio_embeddings(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    audio_data: &[Vec<u8>],
    end_of_turn_token_id: Option<i32>,
) -> Result<Option<InputEmbeddings>> {
    use crate::audio;
    use crate::audio::nemotron_h_nano_omni::NemotronOmniFeatureExtractor;

    let nemotron_vl = match model {
        LoadedModel::NemotronHNanoOmniVLM(vl) => vl,
        _ => return Ok(None),
    };

    let bundle = match nemotron_vl.audio() {
        Some(bundle) => bundle,
        None => {
            tracing::warn!(
                "Nemotron H Nano Omni model was loaded without audio support; ignoring audio input"
            );
            return Ok(None);
        }
    };

    let sound_context_token_id = match nemotron_vl.config.sound_context_token_id {
        Some(id) => id,
        None => {
            tracing::warn!(
                "Nemotron H Nano Omni model has no sound_context_token_id; ignoring audio input"
            );
            return Ok(None);
        }
    };

    if audio_data.len() > 1 {
        tracing::warn!(
            "Multiple audio inputs provided ({}); only the first will be processed",
            audio_data.len()
        );
    }

    // Server passes inline payload bytes; decode the first clip.
    let (samples, sample_rate) = audio::load_wav_from_bytes(&audio_data[0])
        .map_err(|e| anyhow!("Failed to decode audio: {}", e))?;
    tracing::info!(
        "Audio input: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate.max(1) as f64
    );

    // The Parakeet feature extractor is tied to the configured sampling rate
    // (16 kHz for the released checkpoint). The CLI path hard-errors on a rate
    // mismatch; `load_wav_from_bytes` returns native-rate samples, so resample
    // to the expected rate before mel extraction so the encoder frame count
    // matches the duration-derived placeholder count. The only resampler in
    // core targets 16 kHz, so a checkpoint expecting a different rate is
    // rejected with a clear error (preserving the CLI's hard-error contract).
    let expected_rate = bundle.config.sampling_rate;
    let samples = if sample_rate != expected_rate {
        if expected_rate == 16_000 {
            audio::whisper_mel::resample_to_16k(&samples, sample_rate)
        } else {
            return Err(anyhow!(
                "Audio sample rate {} Hz does not match the model's expected {} Hz; \
                 resample the audio before sending it.",
                sample_rate,
                expected_rate
            ));
        }
    } else {
        samples
    };

    // Run the feature extractor and derive the post-subsampling token count,
    // mirroring the CLI builder. Single server clip, so `feature_lengths` has
    // length 1.
    let extractor = NemotronOmniFeatureExtractor::new(&bundle.config);
    let extracted = extractor.extract_batch(&[&samples[..]]);
    let num_frames = extracted.features_shape[1] as usize;
    let total_frames = extracted
        .feature_lengths
        .first()
        .copied()
        .unwrap_or(num_frames as i32) as usize;
    let num_audio_tokens = bundle.config.subsampling_output_length(total_frames).max(1);

    // Place the sound-context block inside the last user turn (issue #437): the
    // server prompt is text-only with no `<so_embedding>` marker, so the block
    // is spliced before the user turn's closing `<|im_end|>`.
    crate::vlm_runtime::expand_nemotron_h_nano_omni_audio_tokens_for_server(
        prompt_tokens,
        sound_context_token_id,
        nemotron_vl.config.sound_start_token_id,
        nemotron_vl.config.sound_end_token_id,
        num_audio_tokens,
        end_of_turn_token_id,
    );

    // Optional image branch (combined image + audio): preprocess and expand
    // image tokens the same way the image-only runtime path does, so the merged
    // stream matches an image-only request.
    let processed_images = if !images.is_empty() {
        let decoded_images = decode_request_images(images)?;
        let processed = nemotron_vl.processor.preprocess_batch(&decoded_images);
        crate::vlm_runtime::expand_nemotron_h_nano_omni_image_tokens_for_server(
            prompt_tokens,
            nemotron_vl.config.img_context_token_id,
            nemotron_vl.config.image_start_token_id,
            nemotron_vl.config.image_end_token_id,
            &processed,
        );
        processed
    } else {
        Vec::new()
    };

    // Build encoder inputs: row-major f32 features, int32 attention mask and
    // per-clip lengths (the encoder broadcasts the mask via `less`).
    let audio_features_in = mlxcel_core::from_slice_f32(
        &extracted.features,
        &[
            extracted.features_shape[0],
            extracted.features_shape[1],
            extracted.features_shape[2],
        ],
    );
    let audio_attention_mask = mlxcel_core::from_slice_i32(
        &extracted.attention_mask,
        &[
            extracted.attention_mask_shape[0],
            extracted.attention_mask_shape[1],
        ],
    );
    let feature_lengths = mlxcel_core::from_slice_i32(
        &extracted.feature_lengths,
        &[extracted.feature_lengths.len() as i32],
    );

    let audio_features = nemotron_vl
        .extract_audio_features(
            &audio_features_in,
            Some(&audio_attention_mask),
            Some(&feature_lengths),
        )
        .map_err(|e| anyhow!("Audio feature extraction failed: {}", e))?;

    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = nemotron_vl.get_input_embeddings_full(
        &input_ids_arr,
        &processed_images,
        Some(&audio_features),
    );

    Ok(Some(embeddings))
}

/// Resolve `videos` into Gemma 4 video embeddings.
///
/// Mirrors the CLI's `compute_gemma4_video_embeddings` in
/// `src/commands/generate_vlm.rs`: probes for ffmpeg, decodes each video into
/// a frame sequence, runs the Gemma 4 video processor, expands `<|video|>`
/// placeholders in the prompt token stream, and dispatches the combined
/// (images + video frames) tensor through the same vision tower path that
/// powers static image inputs.
///
/// Routing-side guarantees (the route layer in `chat.rs` short-circuits
/// non-Gemma 4 video requests with a 400):
/// * The model is a `Gemma4VLM` — non-Gemma 4 models never reach this path.
/// * The video paths have already been canonicalised and validated against
///   `MLXCEL_VIDEO_DIR_ALLOWLIST` by [`crate::server::media::extract_chat_video_paths`].
///
/// Defence-in-depth: this function still rejects non-Gemma 4 models with a
/// clean error so a future caller that bypasses the route guard cannot
/// silently corrupt a non-VLM run.
fn prepare_request_video_embeddings(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    videos: &[crate::server::media::ResolvedVideo],
    image_soft_tokens: Option<usize>,
) -> Result<Option<InputEmbeddings>> {
    use crate::multimodal::video;

    // Encoder-free Gemma 4 Unified routes to its own video path (issue #164):
    // per-frame patches scatter into video_token_id placeholders rather than
    // through the ViT image tower.
    if let LoadedModel::Gemma4Unified(unified) = model {
        return prepare_gemma4_unified_video_embeddings(
            unified,
            prompt_tokens,
            images,
            videos,
            image_soft_tokens,
        );
    }

    // Kimi-VL / Kimi-VL 2.5 (kimi_k25) MoonViT 3D video path (issue #551).
    if let LoadedModel::KimiVL(kimi) = model {
        return prepare_kimi_vl_video_embeddings(kimi, prompt_tokens, images, videos);
    }

    let gemma4_vl = match model {
        LoadedModel::Gemma4VLM(model) => model,
        _ => {
            return Err(anyhow!(
                "video inputs are only supported by Gemma 4 and Kimi-VL VLM models in this build"
            ));
        }
    };

    if !video::ffmpeg_available() {
        return Err(anyhow!(
            "Video input requires `ffmpeg` on PATH. Install ffmpeg (e.g. `brew install ffmpeg` \
             on macOS or `apt install ffmpeg` on Linux) and retry."
        ));
    }

    // Decode each video honoring the per-video FPS override when supplied;
    // otherwise fall back to `multimodal::video::DEFAULT_FPS`.
    //
    // every `ResolvedVideo` carries a [`VideoSource`] handle
    // on Unix this is fd-backed, so the call to `load_video_source` reads
    // from the open file description the resolver already validated. ffmpeg
    // never re-opens the canonical path, so an attacker cannot win the
    // canonicalise → ffmpeg-open swap race even with write access to an
    // allowlist directory.
    let mut decoded_videos: Vec<Vec<image::DynamicImage>> = Vec::with_capacity(videos.len());
    let mut fps_per_video: Vec<f64> = Vec::with_capacity(videos.len());
    for resolved in videos.iter() {
        let fps = resolved.fps.unwrap_or(video::DEFAULT_FPS);
        let frames =
            video::load_video_source(&resolved.source, Some(fps), None).map_err(|err| {
                anyhow!(
                    "Failed to load video {:?}: {}",
                    resolved.source.canonical_path(),
                    err
                )
            })?;
        decoded_videos.push(frames);
        fps_per_video.push(fps);
    }

    let total_frames: usize = decoded_videos.iter().map(Vec::len).sum();
    tracing::info!(
        "video request: decoded {} video(s) ({} total frames after sampling)",
        decoded_videos.len(),
        total_frames
    );

    // Optional companion images (e.g. user passes both image_url and video_url).
    let decoded_images: Vec<image::DynamicImage> = if images.is_empty() {
        Vec::new()
    } else {
        decode_request_images(images)?
    };

    // Companion still images honor the per-request budget; the video frames keep
    // their own (smaller) per-frame budget, which `--image-soft-tokens` and the
    // `image_url` fields deliberately do not touch.
    let processed_images = gemma4_vl
        .processor
        .preprocess_with_budget(&decoded_images, image_soft_tokens);
    let image_soft_token_counts: Vec<usize> = processed_images
        .iter()
        .map(|img| img.num_soft_tokens)
        .collect();

    let processed_videos = gemma4_vl
        .processor
        .process_videos(&decoded_videos, Some(&fps_per_video));

    // Per-video soft-token-per-frame matrix, matching
    // `commands/generate_vlm::compute_gemma4_video_embeddings`.
    let video_frame_tokens: Vec<Vec<usize>> = processed_videos
        .iter()
        .map(|v| vec![v.num_soft_tokens_per_frame; v.num_frames()])
        .collect();

    // The CLI path uses `i32::MIN` as a sentinel that cannot appear in a
    // tokenised prompt so the placeholder-replace branch of
    // `expand_gemma4_video_tokens` is bypassed and the function takes its
    // "splice after BOS" fallback. Server callers behave the same way today
    // because chat templates do not yet emit a real video token id; future
    // template upgrades that introduce one can pass the proper value via
    // `vlm_runtime::expand_gemma4_video_tokens` directly.
    let video_token_sentinel = i32::MIN;

    if decoded_images.is_empty() {
        crate::vlm_runtime::expand_gemma4_video_tokens(
            prompt_tokens,
            video_token_sentinel,
            gemma4_vl.image_token_id,
            gemma4_vl.boi_token_id,
            gemma4_vl.eoi_token_id,
            &video_frame_tokens,
        )?;
    } else {
        crate::vlm_runtime::expand_gemma4_image_tokens_pub(
            prompt_tokens,
            gemma4_vl.image_token_id,
            gemma4_vl.boi_token_id,
            gemma4_vl.eoi_token_id,
            &image_soft_token_counts,
        )?;
        crate::vlm_runtime::expand_gemma4_video_tokens(
            prompt_tokens,
            video_token_sentinel,
            gemma4_vl.image_token_id,
            gemma4_vl.boi_token_id,
            gemma4_vl.eoi_token_id,
            &video_frame_tokens,
        )?;
    }

    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = gemma4_vl.get_input_embeddings_with_videos(
        &input_ids_arr,
        &processed_images,
        &processed_videos,
    );

    Ok(Some(embeddings))
}

/// Resolve `videos` into Kimi-VL / Kimi-VL 2.5 (`kimi_k25`) MoonViT video
/// embeddings (issue #551).
///
/// Mirrors [`prepare_request_video_embeddings`]'s decode/ffmpeg handling: probes
/// for ffmpeg, decodes each video through its fd-backed [`VideoSource`] (no
/// canonicalize -> ffmpeg-open TOCTOU window), decodes any companion images,
/// then routes both through the shared
/// [`crate::vlm_runtime::compute_kimi_vl_media_embeddings`] helper. That helper
/// patchifies per frame, builds the mixed `(t, h, w)` / `(h, w)` media-grid
/// list, expands `<|media_pad|>` placeholders in media order, and runs the
/// MoonViT tower with the temporal position embedding + temporal pooling.
fn prepare_kimi_vl_video_embeddings(
    kimi: &crate::vision::KimiVLModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    videos: &[crate::server::media::ResolvedVideo],
) -> Result<Option<InputEmbeddings>> {
    use crate::multimodal::video;

    if !video::ffmpeg_available() {
        return Err(anyhow!(
            "Video input requires `ffmpeg` on PATH. Install ffmpeg (e.g. `brew install ffmpeg` \
             on macOS or `apt install ffmpeg` on Linux) and retry."
        ));
    }

    // Decode each video honouring the per-video FPS override when supplied;
    // otherwise fall back to `multimodal::video::DEFAULT_FPS` (2.0 fps). The
    // fd-backed source means ffmpeg reads the open file description the resolver
    // already validated.
    let mut decoded_videos: Vec<Vec<image::DynamicImage>> = Vec::with_capacity(videos.len());
    for resolved in videos.iter() {
        let fps = resolved.fps.unwrap_or(video::DEFAULT_FPS);
        let frames =
            video::load_video_source(&resolved.source, Some(fps), None).map_err(|err| {
                anyhow!(
                    "Failed to load video {:?}: {}",
                    resolved.source.canonical_path(),
                    err
                )
            })?;
        decoded_videos.push(frames);
    }

    let total_frames: usize = decoded_videos.iter().map(Vec::len).sum();
    tracing::info!(
        "Kimi-VL video request: decoded {} video(s) ({} total frames after sampling)",
        decoded_videos.len(),
        total_frames
    );

    // Optional companion images (e.g. user passes both image_url and video_url).
    let decoded_images: Vec<image::DynamicImage> = if images.is_empty() {
        Vec::new()
    } else {
        decode_request_images(images)?
    };

    let (embeddings, _stats) = crate::vlm_runtime::compute_kimi_vl_media_embeddings(
        kimi,
        prompt_tokens,
        &decoded_images,
        &decoded_videos,
    )?;

    Ok(Some(embeddings))
}

/// Resolve `videos` into Gemma 4 Unified (encoder-free) video embeddings.
///
/// Mirrors [`prepare_request_video_embeddings`]'s decode/ffmpeg handling and
/// the CLI's `compute_gemma4_unified_video_embeddings`, but routes through the
/// unified model's `video_token_id` scatter: each decoded frame is patchified
/// at the per-frame `vision_soft_tokens_per_video_frame` budget, the prompt is
/// expanded into per-frame `<boi> video_token*N <eoi>` runs, and the per-frame
/// soft tokens scatter into `video_token_id` placeholders (issue #164).
///
/// `videos` carry [`crate::multimodal::video::VideoSource`] handles that are
/// fd-backed on Unix, so `load_video_source` reads from the open file
/// description the resolver already validated (no canonicalize → ffmpeg-open
/// TOCTOU window).
fn prepare_gemma4_unified_video_embeddings(
    unified: &crate::vision::Gemma4UnifiedModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    videos: &[crate::server::media::ResolvedVideo],
    image_soft_tokens: Option<usize>,
) -> Result<Option<InputEmbeddings>> {
    use crate::multimodal::video;

    if !video::ffmpeg_available() {
        return Err(anyhow!(
            "Video input requires `ffmpeg` on PATH. Install ffmpeg (e.g. `brew install ffmpeg` \
             on macOS or `apt install ffmpeg` on Linux) and retry."
        ));
    }

    // Decode each video honoring the per-video FPS override when supplied;
    // otherwise fall back to `multimodal::video::DEFAULT_FPS` (2.0 fps).
    let mut decoded_videos: Vec<Vec<image::DynamicImage>> = Vec::with_capacity(videos.len());
    for resolved in videos.iter() {
        let fps = resolved.fps.unwrap_or(video::DEFAULT_FPS);
        let frames =
            video::load_video_source(&resolved.source, Some(fps), None).map_err(|err| {
                anyhow!(
                    "Failed to load video {:?}: {}",
                    resolved.source.canonical_path(),
                    err
                )
            })?;
        decoded_videos.push(frames);
    }

    let total_decoded_frames: usize = decoded_videos.iter().map(Vec::len).sum();
    tracing::info!(
        "Gemma4 Unified video request: decoded {} video(s) ({} total frames after sampling)",
        decoded_videos.len(),
        total_decoded_frames
    );

    // Optional companion images (e.g. user passes both image_url and video_url).
    let processed_images = if images.is_empty() {
        Vec::new()
    } else {
        let decoded_images = decode_request_images(images)?;
        let processed = unified
            .processor
            .preprocess_with_budget(&decoded_images, image_soft_tokens);
        let num_soft_tokens: Vec<usize> = processed.iter().map(|img| img.num_soft_tokens).collect();
        crate::vlm_runtime::expand_gemma4_image_tokens_pub(
            prompt_tokens,
            unified.image_token_id,
            unified.boi_token_id,
            unified.eoi_token_id,
            &num_soft_tokens,
        )?;
        processed
    };

    // Patchify every frame of every video. Frames stay flat in (video, frame)
    // order so the scatter sees them in the same order as the expanded
    // video_token_id placeholders.
    let mut video_frames: Vec<crate::vision::processors::gemma4_unified::Gemma4UnifiedImageInput> =
        Vec::with_capacity(total_decoded_frames);
    let mut video_frame_tokens: Vec<Vec<usize>> = Vec::with_capacity(decoded_videos.len());
    for frames in &decoded_videos {
        let processed = unified.processor.preprocess_video_frames(frames);
        video_frame_tokens.push(processed.iter().map(|f| f.num_soft_tokens).collect());
        video_frames.extend(processed);
    }

    crate::vlm_runtime::expand_gemma4_unified_video_tokens(
        prompt_tokens,
        unified.video_token_id,
        unified.boi_token_id,
        unified.eoi_token_id,
        &video_frame_tokens,
    )?;

    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings =
        unified.get_input_embeddings_with_video(&input_ids_arr, &processed_images, &video_frames);

    Ok(Some(embeddings))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn build_generation_result(
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    elapsed_ms: u64,
    prompt_eval_ms: u64,
    max_tokens: usize,
) -> GenerationResult {
    build_generation_result_with_cache(
        text,
        prompt_tokens,
        completion_tokens,
        elapsed_ms,
        prompt_eval_ms,
        max_tokens,
        0,
    )
}

/// Build a `GenerationResult` with prompt-prefix cache information.
///
/// `cached_tokens` is the number of leading prompt tokens that were satisfied
/// by the KV prefix cache. Pass `0` for non-cached requests.
pub(crate) fn build_generation_result_with_cache(
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    elapsed_ms: u64,
    prompt_eval_ms: u64,
    max_tokens: usize,
    cached_tokens: usize,
) -> GenerationResult {
    let finish_reason = if completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    };

    GenerationResult {
        text,
        prompt_tokens,
        completion_tokens,
        generation_time_ms: elapsed_ms,
        prompt_eval_ms,
        generation_only_ms: elapsed_ms.saturating_sub(prompt_eval_ms),
        finish_reason: finish_reason.to_string(),
        logprobs: None,
        cached_tokens,
    }
}

/// Maximum number of unemitted trailing tokens the incremental detokenizer will
/// hold while an incomplete UTF-8 sequence is still forming before it force-emits
/// them (rendering any still-incomplete bytes as U+FFFD replacement characters).
///
/// A legitimate incomplete UTF-8 character is at most four bytes, i.e. at most
/// four byte-fallback `<0xXX>` tokens (or a handful of split byte-level BPE
/// pieces), so this bound is only ever reached on degenerate output that emits
/// an unbounded run of invalid bytes. Capping the held window keeps the
/// per-token decode cost O(window) instead of letting it grow to O(sequence
/// length) on such input.
const MAX_HELD_TOKENS: usize = 32;

/// Incremental, byte-fallback-safe detokenizer for streaming generation.
///
/// Emits only the newly-resolved UTF-8 suffix as each token arrives, holding
/// back incomplete multi-byte sequences (byte-fallback `<0xXX>` tokens and split
/// byte-level BPE pieces) until they form valid UTF-8. This is the canonical
/// detokenizer for the server's streaming responses and is also reused by the
/// offline interactive chat REPL (epic #92 / issue #96) so the two surfaces
/// never diverge.
///
/// ## Incremental windowing (issue #633)
///
/// Rather than re-decoding the entire token history on every token (which is
/// O(N) per token and therefore O(N^2) over a full generation), only a bounded
/// suffix *window* of the history is re-decoded. Two decodes are taken per
/// token, both anchored at the same left boundary `prefix_offset`:
///
/// * `prefix_text = decode(all_ids[prefix_offset..read_offset])` — the text
///   already emitted for the current window.
/// * `new_text = decode(all_ids[prefix_offset..])` — that same text plus the
///   newest, not-yet-emitted tokens.
///
/// Because both decodes share the `prefix_offset` left context, the shared
/// prefix renders identically in each and cancels out, so the emitted delta
/// `new_text[prefix_text.len()..]` is byte-identical to slicing a whole-history
/// decode. This mirrors the mlx-lm `NaiveStreamingDetokenizer`
/// (<https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tokenizer_utils.py>)
/// and the llama.cpp / vLLM incremental detokenizers. `read_offset` only ever
/// advances to a position whose decoded text ends on a complete UTF-8 boundary,
/// so `prefix_text` is always a genuine prefix of `new_text` and the byte slice
/// never splits a multi-byte character.
///
/// This assumes the tokenizer's `decode` does not apply context-sensitive
/// `clean_up_tokenization_spaces` rewriting across the window boundary, which
/// holds for every tokenizer in the supported set (Llama, Qwen, Gemma,
/// DeepSeek, Mixtral, ... all ship `clean_up_tokenization_spaces=false`); it is
/// the same assumption the mlx-lm naive detokenizer makes.
pub struct StreamingDecodeState {
    /// Full running token-id history (prompt followed by generated tokens).
    /// Retained for the window slices and for `finish_truncated`, but never
    /// decoded in full on the hot path.
    all_ids: Vec<u32>,
    /// Left anchor of the re-decode window (token index into `all_ids`). See the
    /// struct-level docs for how the shared-prefix cancellation keeps the
    /// emitted delta byte-identical to a whole-history decode.
    prefix_offset: usize,
    /// Boundary between already-emitted tokens and the newest unemitted tokens
    /// (token index into `all_ids`). Always advanced to a UTF-8-complete
    /// position so it never splits a multi-byte character.
    read_offset: usize,
    generated_text: String,
    completion_tokens: usize,
    first_token_time: Option<Instant>,
}

impl StreamingDecodeState {
    pub fn new(_tokenizer: &MlxcelTokenizer, prompt_tokens: &[i32]) -> Self {
        let all_ids: Vec<u32> = prompt_tokens.iter().map(|&x| x as u32).collect();
        let read_offset = all_ids.len();

        Self {
            all_ids,
            // Anchor the window at the very start so the first generated token's
            // delta carries the full prompt as left context. This makes the
            // first emitted byte identical to slicing a whole-history decode at
            // the prompt boundary (the previous implementation's behaviour),
            // including any leading-space handling at the prompt/generation seam.
            prefix_offset: 0,
            read_offset,
            generated_text: String::new(),
            completion_tokens: 0,
            first_token_time: None,
        }
    }

    /// Decode the token-id window `all_ids[start..end]` to text, tolerating
    /// decode errors by yielding an empty string (matching the previous
    /// `unwrap_or_default` behaviour on the full-history decode).
    fn decode_window(&self, tokenizer: &MlxcelTokenizer, start: usize, end: usize) -> String {
        tokenizer
            .decode(&self.all_ids[start..end], false)
            .unwrap_or_default()
    }

    /// The text newly emittable from the current window: the suffix of
    /// `new_text` past the already-emitted `prefix_text`.
    ///
    /// Normally `prefix_text` is a genuine prefix of `new_text` (the window's
    /// `read_offset` only advances to a UTF-8-complete boundary), so this is a
    /// plain suffix. The fallback covers a decoder that retroactively rewrites
    /// already-emitted text: the `tokenizers` `ByteFallback` decoder renders an
    /// entire consecutive byte-token run as U+FFFD once it holds an incomplete
    /// byte, which can change how an earlier byte in the same run rendered. In
    /// that rare case `strip_prefix` fails; decoding only the un-emitted tail
    /// tokens keeps streaming causal and never panics on a non-char-boundary
    /// slice.
    fn window_delta(
        &self,
        tokenizer: &MlxcelTokenizer,
        prefix_text: &str,
        new_text: &str,
    ) -> String {
        match new_text.strip_prefix(prefix_text) {
            Some(rest) => rest.to_string(),
            None => self.decode_window(tokenizer, self.read_offset, self.all_ids.len()),
        }
    }

    pub fn on_token(&mut self, token_id: i32, tokenizer: &MlxcelTokenizer) -> Option<String> {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
        }
        self.completion_tokens += 1;
        self.all_ids.push(token_id as u32);

        // Re-decode only the bounded suffix window (see struct docs). Both
        // decodes share the `prefix_offset` left context so the shared prefix
        // cancels out and the delta is byte-identical to a whole-history decode.
        let prefix_text = self.decode_window(tokenizer, self.prefix_offset, self.read_offset);
        let new_text = self.decode_window(tokenizer, self.prefix_offset, self.all_ids.len());

        // Hold back an incomplete trailing UTF-8 sequence: a byte-fallback or
        // byte-level piece that only decodes to U+FFFD until the completing
        // token arrives. Emitting it now would corrupt the stream because the
        // byte offset shifts when the replacement character resolves into a
        // shorter real character. The window is re-decoded on the next token, so
        // nothing is lost by waiting.
        //
        // `MAX_HELD_TOKENS` force-emits on degenerate output that never
        // completes a sequence, bounding the held window (and thus the per-token
        // decode cost). A legitimate incomplete character never spans that many
        // tokens.
        let force = self.all_ids.len().saturating_sub(self.read_offset) >= MAX_HELD_TOKENS;
        if new_text.len() <= prefix_text.len() || (!force && new_text.ends_with('\u{FFFD}')) {
            return None;
        }

        let delta = self.window_delta(tokenizer, &prefix_text, &new_text);
        self.prefix_offset = self.read_offset;
        self.read_offset = self.all_ids.len();
        if delta.is_empty() {
            return None;
        }
        self.generated_text.push_str(&delta);
        Some(delta)
    }

    /// Flush any remaining windowed text (including an unresolved trailing
    /// incomplete sequence) at the end of generation, returning the flushed
    /// delta so streaming callers can emit it as one final token event.
    ///
    /// This return value is load-bearing: `on_token` holds back the WHOLE window
    /// while its tail is an incomplete UTF-8 sequence (a single token can carry
    /// complete text plus a trailing incomplete byte), so if generation stops on
    /// such a token the complete text is still buffered here. A streaming caller
    /// that ignores the return would append it to `generated_text` (seen by the
    /// non-streaming `result.text`) but never send it to the client. Every
    /// streaming finish site must forward this delta before its `Done`/end event.
    ///
    /// Unlike [`Self::on_token`], the trailing-U+FFFD guard is not applied: any
    /// bytes that never completed a UTF-8 sequence are emitted here as U+FFFD
    /// replacement characters so that the streamed output matches a
    /// whole-history decode. The window prefix ended on a complete boundary, so
    /// `prefix_text` is a genuine prefix of `new_text` and the slice is safe.
    #[must_use = "the flushed tail must be streamed to the client, not dropped"]
    pub fn flush(&mut self, tokenizer: &MlxcelTokenizer) -> Option<String> {
        let prefix_text = self.decode_window(tokenizer, self.prefix_offset, self.read_offset);
        let new_text = self.decode_window(tokenizer, self.prefix_offset, self.all_ids.len());
        let flushed = if new_text.len() > prefix_text.len() {
            let delta = self.window_delta(tokenizer, &prefix_text, &new_text);
            self.generated_text.push_str(&delta);
            (!delta.is_empty()).then_some(delta)
        } else {
            None
        };
        self.prefix_offset = self.all_ids.len();
        self.read_offset = self.all_ids.len();
        flushed
    }

    #[allow(dead_code)]
    pub(crate) fn finish(
        self,
        start: Instant,
        prompt_token_count: usize,
        max_tokens: usize,
    ) -> GenerationResult {
        self.finish_with_cache(start, prompt_token_count, max_tokens, 0)
    }

    /// Like [`finish`] but records how many prompt tokens were served from
    /// the KV prefix cache.
    pub(crate) fn finish_with_cache(
        self,
        start: Instant,
        prompt_token_count: usize,
        max_tokens: usize,
        cached_tokens: usize,
    ) -> GenerationResult {
        let elapsed_ms = start.elapsed().as_millis() as u64;
        let prompt_eval_ms = self
            .first_token_time
            .map(|t| (t - start).as_millis() as u64)
            .unwrap_or(elapsed_ms);

        build_generation_result_with_cache(
            self.generated_text,
            prompt_token_count,
            self.completion_tokens,
            elapsed_ms,
            prompt_eval_ms,
            max_tokens,
            cached_tokens,
        )
    }

    /// Like [`finish`](Self::finish) but truncates the accumulated text to
    /// `keep_bytes` first.
    ///
    /// The OpenXLA serve worker (issue #449 M3 Stage 2d) uses this when a stop
    /// string matches: streaming already stopped at `keep_bytes` (the stop
    /// string and everything after it were withheld), so truncating here keeps
    /// the non-streaming `result.text` consistent with what was streamed.
    /// `keep_bytes` counts whole emitted pieces and so lands on a char boundary;
    /// it is floored to the nearest boundary defensively rather than panicking.
    #[cfg(feature = "xla-iree")]
    pub(crate) fn finish_truncated(
        mut self,
        keep_bytes: usize,
        start: Instant,
        prompt_token_count: usize,
        max_tokens: usize,
    ) -> GenerationResult {
        if keep_bytes < self.generated_text.len() {
            let mut cut = keep_bytes;
            while cut > 0 && !self.generated_text.is_char_boundary(cut) {
                cut -= 1;
            }
            self.generated_text.truncate(cut);
        }
        self.finish_with_cache(start, prompt_token_count, max_tokens, 0)
    }
}

#[cfg(test)]
#[path = "model_worker_tests.rs"]
mod tests;
