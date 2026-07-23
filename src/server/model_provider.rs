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

//! Model provider with dedicated generation thread
//!
//! Since MLX operations are not thread-safe, we run the model on a dedicated
//! thread and communicate via channels.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use mlxcel_core::sampling::TokenLogprobData;

use crate::server::ServerGenerateOptions;
use crate::server::batch::BatchObservability;
use crate::server::media::{MediaRequestMetadata, ResolvedVideo};
use crate::server::state::BatchMetrics;

/// Request to the model thread
pub(crate) enum ModelRequest {
    Generate {
        prompt: String,
        /// Pre-tokenized prompt ids, produced on the request-dispatch thread via
        /// [`tokenize_prompt_for_generation`] (issue #633). When `Some`, the
        /// scheduler uses these directly instead of tokenizing `prompt` on its
        /// own thread; `None` falls back to scheduler-side tokenization. The
        /// `prompt` string is still carried for VLM prompt formatting (e.g.
        /// Moondream3) and diagnostics.
        prompt_token_ids: Option<Vec<i32>>,
        options: ServerGenerateOptions,
        /// Raw image bytes for VLM (empty for text-only)
        images: Vec<Vec<u8>>,
        /// Raw audio bytes for audio-language models (empty for text/vision-only)
        audio: Vec<Vec<u8>>,
        /// Resolved video items with optional per-video FPS overrides
        /// (hardened). Each entry carries a
        /// [`crate::multimodal::video::VideoSource`] handle the worker
        /// passes to [`crate::multimodal::video::load_video_source`]. On
        /// Unix the handle is fd-backed: the resolver opened the file
        /// after canonicalising and matching against
        /// `MLXCEL_VIDEO_DIR_ALLOWLIST`, and ffmpeg consumes that open
        /// file description (via `/dev/fd/N`) rather than re-opening the
        /// path — closing the canonicalise → ffmpeg-open TOCTOU window.
        /// Empty for non-video requests.
        videos: Vec<ResolvedVideo>,
        /// Declared and resolved media counts retained from the HTTP boundary.
        ///
        /// MLX/diffusion workers ignore this metadata and keep their tolerant
        /// resolver behavior. XLA validates it before deciding whether a
        /// request is text-only.
        #[cfg_attr(not(feature = "xla-iree"), allow(dead_code))]
        media: MediaRequestMetadata,
        response_tx: mpsc::Sender<GenerateEvent>,
        /// Cancellation flag set by the SSE sender when the client disconnects.
        /// The `BatchScheduler` polls this to abort orphaned sequences.
        cancelled: Arc<AtomicBool>,
    },
    Shutdown,
}

/// Events from generation
pub enum GenerateEvent {
    Token(String),
    /// Token with associated log probability data (emitted when logprobs are enabled)
    TokenWithLogprobs(String, TokenLogprobData),
    Done(GenerationResult),
    Error(String),
}

/// Result of a generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub generation_time_ms: u64,
    pub prompt_eval_ms: u64,
    pub generation_only_ms: u64,
    pub finish_reason: String,
    /// Per-token log probability data; `None` when logprobs were not requested
    pub logprobs: Option<Vec<TokenLogprobData>>,
    /// Number of prompt tokens that were satisfied by the KV prefix cache.
    ///
    /// Non-zero only when the prompt-prefix cache feature is active and the
    /// scheduler adopted a detached cache for this request. Exposed in the
    /// OpenAI response body as `usage.prompt_tokens_details.cached_tokens`.
    pub cached_tokens: usize,
}

// `pub` (not `pub(crate)`) so the offline interactive chat REPL
// (`mlxcel::commands::chat`, epic #92 / issue #96) can reuse
// [`model_worker::StreamingDecodeState`] for incremental, byte-fallback-safe
// detokenization instead of forking a second detokenizer. The server's
// streaming path is the canonical owner of this logic; the REPL is a second
// consumer of the exact same code.
#[path = "model_worker.rs"]
pub mod model_worker;

/// Thread-safe model provider using channels
pub struct ModelProvider {
    request_tx: mpsc::Sender<ModelRequest>,
    model_id: String,
    created_at: i64,
    loaded: Arc<AtomicBool>,
    batch_metrics: Arc<BatchMetrics>,
    batch_observability: Arc<BatchObservability>,
    /// Shared cross-request prompt-prefix KV cache.
    /// `None` when the feature is disabled by config.
    prompt_cache: Option<Arc<crate::server::prompt_cache::PromptCacheStore>>,
    /// Tokenizer used to encode prompts on the request-dispatch (HTTP-side)
    /// thread before enqueueing, so a long prompt no longer tokenizes on the
    /// scheduler thread and stalls concurrent decode ticks (issue #633). `None`
    /// on paths that do not pre-tokenize (legacy/XLA/tests); the scheduler then
    /// tokenizes as before. Its encoding is byte-identical to the scheduler's
    /// because both go through [`tokenize_prompt_for_generation`].
    prompt_tokenizer: Option<Arc<crate::tokenizer::MlxcelTokenizer>>,
    /// Bounded wait applied during the decode phase of `drain_generation_events*`
    /// to detect a hung model worker.
    ///
    /// Resolved at startup from the `--timeout` CLI flag via
    /// [`validated_decode_hang_timeout`]. Constructors that do not receive a
    /// `ServerConfig` initialise this to [`DECODE_HANG_TIMEOUT`].
    decode_hang_timeout: Duration,
    _worker_handle: thread::JoinHandle<()>,
}

impl ModelProvider {
    /// Create and start a new model provider
    pub fn new(model_path: PathBuf) -> Result<Self> {
        Self::new_with_adapter(model_path, None)
    }

    /// Create and start a new model provider with an optional LoRA adapter.
    ///
    /// Uses default batch settings (max_batch_size=1, max_queue_depth=1024).
    pub fn new_with_adapter(model_path: PathBuf, adapter_path: Option<PathBuf>) -> Result<Self> {
        Self::new_with_batch_config(model_path, adapter_path, 1, 1024)
    }

    /// Create and start a new model provider with batch scheduling config.
    pub fn new_with_batch_config(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
    ) -> Result<Self> {
        let batch_metrics = Arc::new(BatchMetrics::new());
        Self::new_with_metrics(
            model_path,
            adapter_path,
            max_batch_size,
            max_queue_depth,
            batch_metrics,
        )
    }

    /// Create and start a new model provider with full server config.
    ///
    /// When `config.no_batch` is true, the legacy sequential worker is spawned
    /// instead of the batch scheduler, regardless of `max_batch_size`.
    pub fn new_with_server_config(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        config: &crate::server::ServerConfig,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        Self::new_with_server_config_and_prompt_cache(
            model_path,
            adapter_path,
            config,
            None,
            batch_metrics,
            batch_observability,
        )
    }

    /// Same as [`Self::new_with_server_config`] but also wires the
    /// cross-request prompt-prefix KV cache store.
    /// The legacy (`config.no_batch`) worker ignores the store because that
    /// path never calls the batch scheduler that manages the cache.
    pub fn new_with_server_config_and_prompt_cache(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        config: &crate::server::ServerConfig,
        prompt_cache_store: Option<Arc<crate::server::prompt_cache::PromptCacheStore>>,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        // Validate `--timeout` once at construction time and stash it on the
        // provider so the same `Duration` is used by every drain loop. Issue
        // a value of 0 falls back to `DECODE_HANG_TIMEOUT` with a logged
        // warning so an operator typo never silently expires every request.
        let decode_hang_timeout = validated_decode_hang_timeout(config.timeout_seconds);

        // resolve the speculative-decoding dispatch once
        // from the `ServerConfig::{draft_model_path, draft_kind,
        // draft_block_size}` fields. The resolution reads the drafter's
        // `config.json` so it must happen on the main thread before we
        // hand the resolved value to the worker thread. Failures are
        // surfaced as `anyhow::Error` here so the operator gets a clear
        // startup-time error rather than a per-request 5xx.
        let speculative_dispatch = crate::server::SpeculativeDispatch::resolve(config)
            .map_err(|e| anyhow::anyhow!("Speculative decoding dispatch resolution failed: {e}"))?;

        // OpenXLA backend (issue #449 M3 Stage 2c): when `MLXCEL_BACKEND=xla` is
        // selected on an `xla-iree` build, serve through the continuous-batching
        // XLA engine instead of the MLX scheduler. The MLX path below is the
        // default and is unaffected. The XLA engine is greedy and owns its own
        // KV/scheduling, so most scheduler config (`no_batch`, preemption,
        // speculative dispatch, prompt cache) does not apply; `max_batch_size`
        // maps to the engine's bundled slot count.
        #[cfg(feature = "xla-iree")]
        if std::env::var("MLXCEL_BACKEND").ok().as_deref() == Some("xla") {
            if config.no_batch {
                tracing::warn!(
                    "--no-batch is ignored by the OpenXLA backend; it serves through the \
                     continuous-batching engine"
                );
            }
            let b_max = xla_serve_b_max(config.max_batch_size);
            let mut provider =
                Self::new_with_xla_worker(model_path, b_max, batch_metrics, batch_observability)?;
            provider.prompt_cache = prompt_cache_store;
            provider.decode_hang_timeout = decode_hang_timeout;
            return Ok(provider);
        }

        // Pre-tokenizer for the request-dispatch thread (issue #633). Loaded
        // once here (borrowing `model_path` before it is moved into the worker
        // constructor) so `send_generate_request_with_cancellation` can encode a
        // prompt off the scheduler thread. A load failure leaves it `None` and
        // the scheduler tokenizes as before, so this is never fatal.
        let prompt_tokenizer = crate::tokenizer::load_tokenizer(&model_path)
            .ok()
            .map(std::sync::Arc::new);

        if config.no_batch {
            let mut provider = Self::new_with_legacy_worker(
                model_path,
                adapter_path,
                config.tensor_parallel.clone(),
                config.reasoning_budget,
                batch_metrics,
                batch_observability,
            )?;
            // Keep the store visible on the provider even though the
            // legacy path won't exercise it yet — `AppState` should still
            // be able to observe it via the model provider handle.
            provider.prompt_cache = prompt_cache_store;
            provider.prompt_tokenizer = prompt_tokenizer;
            provider.decode_hang_timeout = decode_hang_timeout;
            // log a warning if the operator asked for
            // speculative decoding but selected `--no-batch`. The legacy
            // sequential worker bypasses the BatchScheduler entirely so
            // the dispatch is inactive on this path.
            if !matches!(
                speculative_dispatch,
                crate::server::SpeculativeDispatch::Disabled
            ) {
                tracing::warn!(
                    "Speculative decoding requested ({}) but --no-batch \
                     is enabled; the legacy sequential worker does not \
                     run the speculative dispatch. Drop --no-batch to \
                     enable speculative decoding.",
                    speculative_dispatch.summary(),
                );
            }
            Ok(provider)
        } else {
            let mut provider = Self::new_with_full_config_and_speculative_dispatch(
                model_path,
                adapter_path,
                config.max_batch_size,
                config.max_queue_depth,
                config.prefill_chunk_size,
                config.enable_preemption,
                config.preemption_policy,
                config.max_batch_prefill,
                // forward the --max-batch-prefill-tokens cap to the worker (#715).
                config.max_batch_prefill_tokens,
                config.decode_storage_backend,
                config.pipeline_parallel_runtime.clone(),
                config.vision_cache_size,
                config.lang_bias_config.clone(),
                config.reasoning_budget,
                prompt_cache_store,
                config.kv_cache_mode,
                config.batch_kv_quant,
                // forward the --max-kv-size cap to the scheduler.
                config.max_kv_size,
                // forward the --kv-cache-budget directive to the worker.
                config.kv_cache_budget,
                // experimental VLM prompt-prefix cache toggle (#124 step c).
                config.enable_vlm_prefix_cache,
                // disaggregated serving role from `--node-role` (#126 B2).
                config.serving_mode,
                // disaggregated serving-role network addresses (#126 B3b2a). The
                // worker uses `decode_peers` + `serving_bind`; `--prefill-peers`
                // stays in `ServerConfig` for the future dedicated router.
                config.decode_peers.clone(),
                config.serving_bind,
                speculative_dispatch,
                // serve-level diffusion knobs (#217 phase 3).
                config.max_denoising_steps,
                config.diffusion_sampler.clone(),
                config.diffusion_threshold,
                batch_metrics,
                batch_observability,
            )?;
            provider.prompt_tokenizer = prompt_tokenizer;
            provider.decode_hang_timeout = decode_hang_timeout;
            Ok(provider)
        }
    }

    /// Create and start a new model provider using the legacy sequential worker.
    ///
    /// This is activated by `--no-batch`. The worker uses the `BatchScheduler`
    /// in size-1 mode (no interleaving, no chunked prefill) which is equivalent
    /// to the pre-scheduler sequential request loop.
    pub(crate) fn new_with_legacy_worker(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        tensor_parallel: crate::distributed::ShardConfig,
        reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        let model_id = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let created_at = chrono::Utc::now().timestamp();
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();
        let loaded = Arc::new(AtomicBool::new(false));
        let loaded_clone = loaded.clone();
        let worker_model_id = model_id.clone();
        let metrics_clone = batch_metrics.clone();
        let obs_clone = batch_observability.clone();

        let worker_handle = model_worker::spawn_legacy_model_worker(
            model_path,
            adapter_path,
            tensor_parallel,
            reasoning_budget,
            request_rx,
            loaded_clone,
            worker_model_id,
            metrics_clone,
            obs_clone,
        );

        Ok(Self {
            request_tx,
            model_id,
            created_at,
            loaded,
            batch_metrics,
            batch_observability,
            prompt_cache: None,
            prompt_tokenizer: None,
            decode_hang_timeout: DECODE_HANG_TIMEOUT,
            _worker_handle: worker_handle,
        })
    }

    /// Create and start a model provider backed by the OpenXLA / IREE
    /// continuous-batching engine (issue #449 M3 Stage 2c).
    ///
    /// Spawns [`spawn_xla_model_worker`](model_worker::spawn_xla_model_worker),
    /// which builds the engine + tokenizer on the worker thread and serves through
    /// the [`BatchEngine`](crate::server::batch::BatchEngine) contract. The shared
    /// `batch_metrics` handle is passed to the worker, which populates the active
    /// count, queue depth, and per-sequence completion the `/metrics` endpoint
    /// reports (the `batch_observability` handle is held for the provider API
    /// surface; the XLA path has no prompt cache or preemption to report there).
    #[cfg(feature = "xla-iree")]
    pub(crate) fn new_with_xla_worker(
        model_path: PathBuf,
        b_max: usize,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        let model_id = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let created_at = chrono::Utc::now().timestamp();
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();
        let loaded = Arc::new(AtomicBool::new(false));

        let worker_handle = model_worker::spawn_xla_model_worker(
            model_path,
            b_max,
            request_rx,
            loaded.clone(),
            model_id.clone(),
            batch_metrics.clone(),
            batch_observability.clone(),
        );

        Ok(Self {
            request_tx,
            model_id,
            created_at,
            loaded,
            batch_metrics,
            batch_observability,
            prompt_cache: None,
            prompt_tokenizer: None,
            decode_hang_timeout: DECODE_HANG_TIMEOUT,
            _worker_handle: worker_handle,
        })
    }

    /// Create and start a new model provider with full scheduler config
    /// and shared batch metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_full_config(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
        prefill_chunk_size: usize,
        enable_preemption: bool,
        preemption_policy: crate::server::config::PreemptionPolicy,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        Self::new_with_full_config_and_batch_prefill(
            model_path,
            adapter_path,
            max_batch_size,
            max_queue_depth,
            prefill_chunk_size,
            enable_preemption,
            preemption_policy,
            1,
            crate::server::DecodeStorageBackend::Dense,
            None,
            crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE,
            None,
            None,
            batch_metrics,
            batch_observability,
        )
    }

    /// Create and start a new model provider with full scheduler config,
    /// shared batch metrics, and batched prefill support.
    ///
    /// `vision_cache_size` maps directly to the `--vision-cache-size` CLI
    /// flag. `0` disables per-image vision feature caching entirely.
    ///
    /// `lang_bias_config` is the Axis B / (B8) server-wide
    /// language-bias configuration. Pass `None` for the baseline bit-exact
    /// path (no sampling changes, no tokenizer-vocab scan).
    ///
    /// `reasoning_budget` is the server-wide default
    /// thinking-token budget for Qwen3-family models. Pass `None` for
    /// unrestricted reasoning (bit-exact baseline); per-request
    /// `thinking_budget_tokens` still takes precedence.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_full_config_and_batch_prefill(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
        prefill_chunk_size: usize,
        enable_preemption: bool,
        preemption_policy: crate::server::config::PreemptionPolicy,
        max_batch_prefill: usize,
        decode_storage_backend: crate::server::DecodeStorageBackend,
        pipeline_parallel_runtime: Option<crate::server::PipelineParallelRuntimeConfig>,
        vision_cache_size: usize,
        lang_bias_config: Option<mlxcel_core::lang_analyzer::LangBiasConfig>,
        reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        Self::new_with_full_config_and_prompt_cache(
            model_path,
            adapter_path,
            max_batch_size,
            max_queue_depth,
            prefill_chunk_size,
            enable_preemption,
            preemption_policy,
            max_batch_prefill,
            decode_storage_backend,
            pipeline_parallel_runtime,
            vision_cache_size,
            lang_bias_config,
            reasoning_budget,
            None,
            mlxcel_core::cache::KVCacheMode::Fp16,
            mlxcel_core::cache::BatchKvQuantConfig::default(),
            None,  // max_kv_size: unbounded
            None,  // kv_cache_budget: unbounded
            false, // enable_vlm_prefix_cache: off
            // serving_mode: single-node Hybrid (this wrapper has no --node-role).
            crate::distributed::disaggregated::ServingMode::Hybrid,
            Vec::new(), // decode_peers: none (hybrid)
            None,       // serving_bind: none (hybrid)
            batch_metrics,
            batch_observability,
        )
    }

    /// Full constructor variant that also accepts a shared prompt-prefix
    /// KV cache store.
    ///
    /// Introduced. `prompt_cache_store` is `None` when
    /// [`crate::server::prompt_cache::PromptCacheConfig::enabled`] is
    /// `false`; in that case the feature is a total no-op.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_full_config_and_prompt_cache(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
        prefill_chunk_size: usize,
        enable_preemption: bool,
        preemption_policy: crate::server::config::PreemptionPolicy,
        max_batch_prefill: usize,
        decode_storage_backend: crate::server::DecodeStorageBackend,
        pipeline_parallel_runtime: Option<crate::server::PipelineParallelRuntimeConfig>,
        vision_cache_size: usize,
        lang_bias_config: Option<mlxcel_core::lang_analyzer::LangBiasConfig>,
        reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,
        prompt_cache_store: Option<Arc<crate::server::prompt_cache::PromptCacheStore>>,
        kv_cache_mode: mlxcel_core::cache::KVCacheMode,
        batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig,
        // maximum KV cache size for plain (non-sliding) caches.
        // `None` preserves the legacy unbounded behaviour.
        max_kv_size: Option<usize>,
        // paged KV pool block-budget directive (`--kv-cache-budget`).
        // `None` keeps the pool unbounded.
        kv_cache_budget: Option<crate::memory_estimate::PagedBudgetDirective>,
        enable_vlm_prefix_cache: bool,
        serving_mode: crate::distributed::disaggregated::ServingMode,
        decode_peers: Vec<std::net::SocketAddr>,
        serving_bind: Option<std::net::SocketAddr>,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        // backward-compatible wrapper that defaults the
        // speculative dispatch to `Disabled`. Callers wiring `--draft-model`
        // / `--draft-kind` use the `_with_speculative_dispatch` variant
        // below.
        Self::new_with_full_config_and_speculative_dispatch(
            model_path,
            adapter_path,
            max_batch_size,
            max_queue_depth,
            prefill_chunk_size,
            enable_preemption,
            preemption_policy,
            max_batch_prefill,
            // this wrapper predates --max-batch-prefill-tokens (#715); let the
            // scheduler use the env override or the derived default.
            None,
            decode_storage_backend,
            pipeline_parallel_runtime,
            vision_cache_size,
            lang_bias_config,
            reasoning_budget,
            prompt_cache_store,
            kv_cache_mode,
            batch_kv_quant,
            max_kv_size,
            kv_cache_budget,
            enable_vlm_prefix_cache,
            serving_mode,
            decode_peers,
            serving_bind,
            crate::server::SpeculativeDispatch::Disabled,
            // diffusion knobs default to the engine defaults in this
            // speculative-dispatch-agnostic wrapper.
            None,
            "entropy-bound".to_string(),
            0.9,
            batch_metrics,
            batch_observability,
        )
    }

    /// variant that also accepts the resolved
    /// [`crate::server::SpeculativeDispatch`].
    ///
    /// Use this from `new_with_server_config_and_prompt_cache` so the
    /// `--draft-model` / `--draft-kind` flags actually reach the worker
    /// thread. The default [`crate::server::SpeculativeDispatch::Disabled`]
    /// preserves bit-exact baseline behaviour for the non-speculative
    /// path.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_full_config_and_speculative_dispatch(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
        prefill_chunk_size: usize,
        enable_preemption: bool,
        preemption_policy: crate::server::config::PreemptionPolicy,
        max_batch_prefill: usize,
        max_batch_prefill_tokens: Option<usize>,
        decode_storage_backend: crate::server::DecodeStorageBackend,
        pipeline_parallel_runtime: Option<crate::server::PipelineParallelRuntimeConfig>,
        vision_cache_size: usize,
        lang_bias_config: Option<mlxcel_core::lang_analyzer::LangBiasConfig>,
        reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,
        prompt_cache_store: Option<Arc<crate::server::prompt_cache::PromptCacheStore>>,
        kv_cache_mode: mlxcel_core::cache::KVCacheMode,
        batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig,
        max_kv_size: Option<usize>,
        kv_cache_budget: Option<crate::memory_estimate::PagedBudgetDirective>,
        enable_vlm_prefix_cache: bool,
        serving_mode: crate::distributed::disaggregated::ServingMode,
        decode_peers: Vec<std::net::SocketAddr>,
        serving_bind: Option<std::net::SocketAddr>,
        speculative_dispatch: crate::server::SpeculativeDispatch,
        max_denoising_steps: Option<usize>,
        diffusion_sampler: String,
        diffusion_threshold: f32,
        batch_metrics: Arc<BatchMetrics>,
        batch_observability: Arc<BatchObservability>,
    ) -> Result<Self> {
        let model_id = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let created_at = chrono::Utc::now().timestamp();
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();
        let loaded = Arc::new(AtomicBool::new(false));
        let loaded_clone = loaded.clone();
        let worker_model_id = model_id.clone();
        let metrics_clone = batch_metrics.clone();
        let obs_clone = batch_observability.clone();

        let sched_config = model_worker::WorkerSchedulerConfig {
            max_batch_size,
            max_queue_depth,
            prefill_chunk_size,
            enable_preemption,
            preemption_policy,
            max_batch_prefill: max_batch_prefill.max(1),
            // #715: forward the explicit --max-batch-prefill-tokens value; the
            // scheduler resolves the env override / derived default otherwise.
            max_batch_prefill_tokens,
            decode_storage_backend,
            pipeline_parallel_runtime,
            tensor_parallel: crate::distributed::ShardConfig::default(),
            vision_cache_size,
            lang_bias_config,
            reasoning_budget,
            prompt_cache: prompt_cache_store.clone(),
            kv_cache_mode,
            batch_kv_quant,
            // cap plain KVCache growth when configured.
            max_kv_size,
            // paged KV pool block-budget directive; resolved to a block count
            // on the worker thread once the model geometry is known.
            kv_cache_budget,
            // experimental VLM prompt-prefix cache toggle (#124 step c).
            enable_vlm_prefix_cache,
            // disaggregated serving role from `--node-role` (#126 B2). The
            // worker carries it so the serving-role coordinator can be wired
            // onto the live scheduler later (B2b); `Hybrid` is the unchanged
            // single-node path.
            serving_mode,
            // disaggregated serving-role network addresses (#126 B3b2a): the
            // worker binds `serving_bind` and hands KV off to `decode_peers`
            // when `serving_mode` is non-hybrid.
            decode_peers,
            serving_bind,
            // forward the resolved speculative dispatch.
            speculative_dispatch,
            // serve-level diffusion knobs (#217 phase 3); consumed only by the
            // DiffusionGemma worker loop.
            max_denoising_steps,
            diffusion_sampler,
            diffusion_threshold,
        };

        let worker_handle = model_worker::spawn_model_worker_with_batch_config(
            model_path,
            adapter_path,
            request_rx,
            loaded_clone,
            worker_model_id,
            sched_config,
            metrics_clone,
            obs_clone,
        );

        Ok(Self {
            request_tx,
            model_id,
            created_at,
            loaded,
            batch_metrics,
            batch_observability,
            prompt_cache: prompt_cache_store,
            prompt_tokenizer: None,
            decode_hang_timeout: DECODE_HANG_TIMEOUT,
            _worker_handle: worker_handle,
        })
    }

    /// Create and start a new model provider with shared batch metrics.
    pub fn new_with_metrics(
        model_path: PathBuf,
        adapter_path: Option<PathBuf>,
        max_batch_size: usize,
        max_queue_depth: usize,
        batch_metrics: Arc<BatchMetrics>,
    ) -> Result<Self> {
        let model_id = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let created_at = chrono::Utc::now().timestamp();

        // Create channel for requests
        let (request_tx, request_rx) = mpsc::channel::<ModelRequest>();

        // Shared loaded flag
        let loaded = Arc::new(AtomicBool::new(false));
        let loaded_clone = loaded.clone();

        // Clone model_id for the worker thread
        let worker_model_id = model_id.clone();
        let metrics_clone = batch_metrics.clone();
        let batch_observability = Arc::new(BatchObservability::new());
        let obs_clone = batch_observability.clone();

        let sched_config = model_worker::WorkerSchedulerConfig {
            max_batch_size,
            max_queue_depth,
            prefill_chunk_size: 0,
            enable_preemption: false,
            preemption_policy: crate::server::config::PreemptionPolicy::default(),
            max_batch_prefill: 1,
            // #715: minimal test path never batches prefill; keep the default.
            max_batch_prefill_tokens: None,
            decode_storage_backend: crate::server::DecodeStorageBackend::Dense,
            pipeline_parallel_runtime: None,
            tensor_parallel: crate::distributed::ShardConfig::default(),
            vision_cache_size: crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE,
            lang_bias_config: None,
            reasoning_budget: None,
            prompt_cache: None,
            kv_cache_mode: mlxcel_core::cache::KVCacheMode::Fp16,
            batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig::default(),
            max_kv_size: None,              // unbounded in minimal test path
            kv_cache_budget: None,          // unbounded in minimal test path
            enable_vlm_prefix_cache: false, // off in minimal test path
            // minimal test path is single-node.
            serving_mode: crate::distributed::disaggregated::ServingMode::Hybrid,
            decode_peers: Vec::new(), // single-node minimal test path
            serving_bind: None,       // single-node minimal test path
            // minimal test path has no drafter; the dispatch
            // defaults to `Disabled` which short-circuits the scheduler
            // hot path to the classic decode loop.
            speculative_dispatch: crate::server::SpeculativeDispatch::Disabled,
            // minimal test path uses the engine diffusion defaults.
            max_denoising_steps: None,
            diffusion_sampler: "entropy-bound".to_string(),
            diffusion_threshold: 0.9,
        };

        let worker_handle = model_worker::spawn_model_worker_with_batch_config(
            model_path,
            adapter_path,
            request_rx,
            loaded_clone,
            worker_model_id,
            sched_config,
            metrics_clone,
            obs_clone,
        );

        Ok(Self {
            request_tx,
            model_id,
            created_at,
            loaded,
            batch_metrics,
            batch_observability,
            prompt_cache: None,
            prompt_tokenizer: None,
            decode_hang_timeout: DECODE_HANG_TIMEOUT,
            _worker_handle: worker_handle,
        })
    }

    /// Get a reference to the shared batch metrics.
    pub fn batch_metrics(&self) -> &Arc<BatchMetrics> {
        &self.batch_metrics
    }

    /// Get a reference to the shared batch observability counters.
    pub fn batch_observability(&self) -> &Arc<BatchObservability> {
        &self.batch_observability
    }

    /// Shared cross-request prompt-prefix KV cache store, if configured.
    ///
    /// `None` when the feature is disabled via
    /// [`crate::server::prompt_cache::PromptCacheConfig::enabled`].
    pub fn prompt_cache(&self) -> Option<&Arc<crate::server::prompt_cache::PromptCacheStore>> {
        self.prompt_cache.as_ref()
    }

    /// Get model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> i64 {
        self.created_at
    }

    /// Check if model is loaded and ready for inference
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Acquire)
    }

    /// Generate text and return the full result
    pub fn generate(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
    ) -> Result<GenerationResult> {
        self.generate_with_media(prompt, options, Vec::new(), Vec::new())
    }

    /// Generate text with optional images and return the full result
    pub fn generate_with_images(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
    ) -> Result<GenerationResult> {
        self.generate_with_media(prompt, options, images, Vec::new())
    }

    /// Generate text with optional images and audio, and return the full result
    pub fn generate_with_media(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
    ) -> Result<GenerationResult> {
        self.generate_with_media_and_videos(prompt, options, images, audio, Vec::new())
    }

    /// Generate text with optional images, audio, and videos, returning the
    /// full result.
    ///
    /// Server video routes pass `videos` through here once the path-traversal
    /// guard in [`crate::server::media::extract_chat_video_paths`] has cleared
    /// each entry against `MLXCEL_VIDEO_DIR_ALLOWLIST`. Empty `videos` matches
    /// earlier behavior bit-for-bit.
    ///
    /// Restricted to `pub(crate)` because the input type [`ResolvedVideo`] is
    /// crate-internal — it carries an [`crate::multimodal::video::VideoSource`]
    /// owning an open fd whose lifecycle is managed by the request handler.
    /// Exposing it across the crate boundary would invite leaks.
    pub(crate) fn generate_with_media_and_videos(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
    ) -> Result<GenerationResult> {
        let media = MediaRequestMetadata::from_resolved(images.len(), audio.len(), videos.len());
        self.generate_with_media_and_videos_declared(prompt, options, images, audio, videos, media)
    }

    /// Generation entry used by prepared HTTP requests that retain declared
    /// media cardinality across tolerant resolution.
    pub(crate) fn generate_with_media_and_videos_declared(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
        media: MediaRequestMetadata,
    ) -> Result<GenerationResult> {
        let response_rx = self
            .send_generate_request_with_metadata(prompt, options, images, audio, videos, media)?;
        drain_generation_events(response_rx, self.decode_hang_timeout, |_| {})
    }

    /// Generate text with streaming callback
    pub fn generate_streaming<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String),
    {
        self.generate_streaming_with_images(prompt, options, Vec::new(), callback)
    }

    /// Generate text with optional images and streaming callback
    pub fn generate_streaming_with_images<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String),
    {
        let response_rx =
            self.send_generate_request(prompt, options, images, Vec::new(), Vec::new())?;
        drain_generation_events(response_rx, self.decode_hang_timeout, callback)
    }

    /// Generate text with optional images/audio and a logprobs-aware streaming callback.
    ///
    /// The callback receives the decoded token text plus optional `TokenLogprobData`.
    /// When `options.logprobs.enabled` is false the logprob argument will always
    /// be `None`, so this method is a strict superset of `generate_streaming_with_images`.
    pub fn generate_streaming_with_logprobs<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String, Option<TokenLogprobData>),
    {
        let response_rx = self.send_generate_request(prompt, options, images, audio, Vec::new())?;
        drain_generation_events_with_logprobs(response_rx, self.decode_hang_timeout, callback)
    }

    /// Generate text with streaming callback and cancellation support.
    ///
    /// Like `generate_streaming` but accepts a `cancelled` token that the SSE
    /// sender sets when the client disconnects.
    ///
    /// Used by: chat.rs, completions.rs, native_completion.rs (streaming routes)
    pub fn generate_streaming_cancellable<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        cancelled: Arc<AtomicBool>,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String),
    {
        let response_rx = self.send_generate_request_with_cancellation(
            prompt,
            options,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            cancelled,
        )?;
        drain_generation_events(response_rx, self.decode_hang_timeout, callback)
    }

    /// Generate text with logprobs-aware streaming callback and cancellation
    /// support.
    ///
    /// Like `generate_streaming_with_logprobs` but accepts a `cancelled` token
    /// that the SSE sender sets when the client disconnects.
    ///
    /// Used by: chat.rs, completions.rs (streaming routes)
    pub fn generate_streaming_with_logprobs_cancellable<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        cancelled: Arc<AtomicBool>,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String, Option<TokenLogprobData>),
    {
        self.generate_streaming_with_logprobs_cancellable_videos(
            prompt,
            options,
            images,
            audio,
            Vec::new(),
            cancelled,
            callback,
        )
    }

    /// Like [`Self::generate_streaming_with_logprobs_cancellable`] but also
    /// forwards resolved video paths to the worker. Empty
    /// `videos` is bit-exact with the no-video variant.
    ///
    /// `pub(crate)` for the same reason as
    /// [`Self::generate_with_media_and_videos`] — the [`ResolvedVideo`]
    /// input type is crate-internal.
    pub(crate) fn generate_streaming_with_logprobs_cancellable_videos<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
        cancelled: Arc<AtomicBool>,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String, Option<TokenLogprobData>),
    {
        let media = MediaRequestMetadata::from_resolved(images.len(), audio.len(), videos.len());
        self.generate_streaming_with_logprobs_cancellable_videos_declared(
            prompt, options, images, audio, videos, media, cancelled, callback,
        )
    }

    /// Streaming entry used by prepared HTTP requests that retain declared
    /// media cardinality across tolerant resolution.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_streaming_with_logprobs_cancellable_videos_declared<F>(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
        media: MediaRequestMetadata,
        cancelled: Arc<AtomicBool>,
        callback: F,
    ) -> Result<GenerationResult>
    where
        F: FnMut(String, Option<TokenLogprobData>),
    {
        let response_rx = self.send_generate_request_with_cancellation_and_metadata(
            prompt, options, images, audio, videos, media, cancelled,
        )?;
        drain_generation_events_with_logprobs(response_rx, self.decode_hang_timeout, callback)
    }

    fn send_generate_request(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        let media = MediaRequestMetadata::from_resolved(images.len(), audio.len(), videos.len());
        self.send_generate_request_with_metadata(prompt, options, images, audio, videos, media)
    }

    fn send_generate_request_with_metadata(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
        media: MediaRequestMetadata,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        self.send_generate_request_with_cancellation_and_metadata(
            prompt,
            options,
            images,
            audio,
            videos,
            media,
            Arc::new(AtomicBool::new(false)),
        )
    }

    /// Send a generation request with an explicit cancellation token.
    ///
    /// The cancellation token is an `Arc<AtomicBool>` shared with the SSE
    /// sender. When the client disconnects the token is set to `true`, and the
    /// `BatchScheduler` will abort the corresponding sequence.
    fn send_generate_request_with_cancellation(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
        cancelled: Arc<AtomicBool>,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        let media = MediaRequestMetadata::from_resolved(images.len(), audio.len(), videos.len());
        self.send_generate_request_with_cancellation_and_metadata(
            prompt, options, images, audio, videos, media, cancelled,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn send_generate_request_with_cancellation_and_metadata(
        &self,
        prompt: String,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
        media: MediaRequestMetadata,
        cancelled: Arc<AtomicBool>,
    ) -> Result<mpsc::Receiver<GenerateEvent>> {
        let (response_tx, response_rx) = mpsc::channel();

        // Tokenize on this (request-dispatch / HTTP-side) thread when a
        // pre-tokenizer is available, so a long prompt no longer stalls the
        // scheduler thread's decode loop (issue #633). A tokenization failure
        // falls back to `None` so the scheduler encodes it and surfaces the
        // error through the normal response channel.
        let prompt_token_ids = self.prompt_tokenizer.as_ref().and_then(|tok| {
            tokenize_prompt_for_generation(tok, &prompt)
                .map_err(|err| {
                    tracing::debug!(
                        "HTTP-side prompt tokenization failed ({err}); deferring to scheduler"
                    );
                    err
                })
                .ok()
        });

        self.request_tx
            .send(ModelRequest::Generate {
                prompt,
                prompt_token_ids,
                options,
                images,
                audio,
                videos,
                media,
                response_tx,
                cancelled,
            })
            .map_err(|e| anyhow::anyhow!("Failed to send request: {e}"))?;

        Ok(response_rx)
    }
}

/// Tokenize a rendered prompt into `i32` ids using the same `add_special`
/// convention the scheduler applies (issue #633).
///
/// `add_special` is suppressed when the prompt already begins with a literal BOS
/// marker (`<bos>` / `<s>`), matching `BatchScheduler::enqueue_request` so that
/// pre-tokenizing on the dispatch thread is byte-identical to tokenizing on the
/// scheduler thread. This is the single source of truth for both sites.
pub(crate) fn tokenize_prompt_for_generation(
    tokenizer: &crate::tokenizer::MlxcelTokenizer,
    prompt: &str,
) -> Result<Vec<i32>> {
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    let ids = tokenizer.encode(prompt, add_special)?;
    Ok(ids.iter().map(|&x| x as i32).collect())
}

fn send_shutdown_signal(request_tx: &mpsc::Sender<ModelRequest>) -> bool {
    request_tx.send(ModelRequest::Shutdown).is_ok()
}

/// Default timeout applied after the first generated token has been received
/// to detect a hung model worker.
///
/// Once the prefill is complete and decoding has begun, each subsequent decode
/// step should finish within a bounded wall-clock time. The model worker thread
/// emits events in a tight loop during decode; if no event arrives within this
/// window, the request is considered hung.
///
/// 300 seconds (5 minutes) is deliberately generous to handle large batch
/// sizes, slow hardware, and long decode chains without causing false-positive
/// timeouts during normal operation. Operators can configure this with
/// `--timeout SECONDS`. Setting `--timeout 0` falls back to this 300 s default
/// (with a logged warning at startup) because `0` would otherwise expire every
/// request instantly.
///
/// At runtime the per-provider `Duration` resolved by
/// [`validated_decode_hang_timeout`] is threaded through
/// [`drain_generation_events`] / [`drain_generation_events_with_logprobs`] /
/// [`drain_generation_events_impl`]; this constant is the fallback used by
/// [`validated_decode_hang_timeout`] and the constructors that do not see a
/// `ServerConfig`.
#[doc(hidden)] // pub(crate) for tests
pub(crate) const DECODE_HANG_TIMEOUT: Duration = Duration::from_secs(300);

/// Validate `timeout_seconds` from the server config and convert it to a
/// `Duration`.
///
/// Returns the configured duration on success. Logs a warning and returns the
/// fallback ([`DECODE_HANG_TIMEOUT`]) when the value is `0`, which would cause
/// every request to time out instantly ("invalid timeout config values produce a clean log message").
///
/// Used by: `ModelProvider::new_with_server_config_and_prompt_cache` (the only
/// constructor that receives a `ServerConfig`); other constructors default to
/// [`DECODE_HANG_TIMEOUT`].
pub(crate) fn validated_decode_hang_timeout(timeout_seconds: u64) -> Duration {
    if timeout_seconds == 0 {
        tracing::warn!(
            "server timeout_seconds is 0, which would expire immediately; \
             using built-in fallback of {}s. \
             Set --timeout to a positive value to suppress this warning.",
            DECODE_HANG_TIMEOUT.as_secs()
        );
        return DECODE_HANG_TIMEOUT;
    }
    Duration::from_secs(timeout_seconds)
}

/// Map the server's `--max-batch-size` to one of the OpenXLA engine's bundled
/// slot counts (issue #449 M3 Stage 2c): the largest bundled `B_max` that does
/// not exceed the request, defaulting to the smallest. The engine compiles one
/// ragged graph per slot count, so the server picks from the bundled set rather
/// than any value.
#[cfg(feature = "xla-iree")]
fn xla_serve_b_max(max_batch_size: usize) -> usize {
    if max_batch_size >= 8 { 8 } else { 4 }
}

/// Drain `response_rx`, forwarding decoded tokens to `on_token` and applying
/// the two-phase timeout policy described on
/// [`drain_generation_events_impl`].
///
/// `decode_hang_timeout` is the per-provider Phase-2 bound resolved at startup
/// via [`validated_decode_hang_timeout`]; pass [`DECODE_HANG_TIMEOUT`] when a
/// caller has no configured value.
pub(super) fn drain_generation_events<F>(
    response_rx: mpsc::Receiver<GenerateEvent>,
    decode_hang_timeout: Duration,
    mut on_token: F,
) -> Result<GenerationResult>
where
    F: FnMut(String),
{
    // Accumulate logprobs from TokenWithLogprobs events so the final
    // GenerationResult can carry them even in non-streaming mode.
    let mut accumulated_logprobs: Vec<TokenLogprobData> = Vec::new();

    let mut result =
        drain_generation_events_impl(&response_rx, decode_hang_timeout, |event| match event {
            GenerateEvent::Token(token) => {
                on_token(token);
                Ok(None)
            }
            // Collect logprobs even when the streaming callback ignores them.
            GenerateEvent::TokenWithLogprobs(token, lp) => {
                accumulated_logprobs.push(lp);
                on_token(token);
                Ok(None)
            }
            GenerateEvent::Done(result) => Ok(Some(result)),
            GenerateEvent::Error(err) => Err(anyhow::anyhow!(err)),
        })?;

    if !accumulated_logprobs.is_empty() {
        result.logprobs = Some(accumulated_logprobs);
    }

    Ok(result)
}

/// Like [`drain_generation_events`] but exposes per-token logprob data to the
/// callback. `decode_hang_timeout` follows the same contract.
pub(super) fn drain_generation_events_with_logprobs<F>(
    response_rx: mpsc::Receiver<GenerateEvent>,
    decode_hang_timeout: Duration,
    mut on_token: F,
) -> Result<GenerationResult>
where
    F: FnMut(String, Option<TokenLogprobData>),
{
    drain_generation_events_impl(&response_rx, decode_hang_timeout, |event| match event {
        GenerateEvent::Token(token) => {
            on_token(token, None);
            Ok(None)
        }
        GenerateEvent::TokenWithLogprobs(token, lp) => {
            on_token(token, Some(lp));
            Ok(None)
        }
        GenerateEvent::Done(result) => Ok(Some(result)),
        GenerateEvent::Error(err) => Err(anyhow::anyhow!(err)),
    })
}

/// Core receive loop that distinguishes "prefill still running" from "hung
/// model".
///
/// Two-phase timeout strategy:
///
/// **Phase 1 — prefill window** (`decode_phase_started == false`): block
/// indefinitely (`recv()`). Long prompts (32k+ tokens) may require minutes of
/// prefill computation before the first generated token appears. Any timeout
/// applied here would incorrectly abort valid in-progress requests.
///
/// **Phase 2 — decode window** (`decode_phase_started == true`): apply
/// `decode_hang_timeout`. Once decoding has started, each subsequent decode
/// step is a single forward pass that should complete in seconds even on slow
/// hardware, so a bounded window catches genuine worker deadlocks without
/// false-positives on legitimate long-running decode chains. The provider
/// resolves this duration at startup from the `--timeout SECONDS` CLI flag via
/// [`validated_decode_hang_timeout`]; `--timeout 0` falls back to the
/// [`DECODE_HANG_TIMEOUT`] (300 s) default with a logged warning.
///
/// `handler` maps a `GenerateEvent` to `Ok(Some(result))` (done),
/// `Ok(None)` (continue), or `Err(...)` (fatal).
pub(super) fn drain_generation_events_impl<H>(
    response_rx: &mpsc::Receiver<GenerateEvent>,
    decode_hang_timeout: Duration,
    mut handler: H,
) -> Result<GenerationResult>
where
    H: FnMut(GenerateEvent) -> Result<Option<GenerationResult>>,
{
    // True once any token, logprob-token, or Done event has been seen.
    // `Done` may arrive before any token (e.g. max_tokens=0 guard), so the
    // flag reflects "the decode phase has begun" rather than "a token arrived".
    let mut decode_phase_started = false;

    loop {
        // Phase 1: infinite wait during prefill.
        // Phase 2: bounded wait during decode to detect hangs.
        let event = if decode_phase_started {
            match response_rx.recv_timeout(decode_hang_timeout) {
                Ok(ev) => ev,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    return Err(anyhow::anyhow!(
                        "model worker did not produce a token within {}s after decode started; \
                         possible hang or crash. \
                         Increase --timeout if this model legitimately takes longer.",
                        decode_hang_timeout.as_secs()
                    ));
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(anyhow::anyhow!("Response channel closed"));
                }
            }
        } else {
            // Prefill may take minutes for very large prompts — wait without
            // any timeout so we never spuriously abort a valid request.
            match response_rx.recv() {
                Ok(ev) => ev,
                Err(_) => return Err(anyhow::anyhow!("Response channel closed")),
            }
        };

        // Transition to decode phase on any token or result event.
        match &event {
            GenerateEvent::Token(_)
            | GenerateEvent::TokenWithLogprobs(_, _)
            | GenerateEvent::Done(_) => {
                decode_phase_started = true;
            }
            GenerateEvent::Error(_) => {}
        }

        if let Some(result) = handler(event)? {
            return Ok(result);
        }
    }
}

impl Drop for ModelProvider {
    fn drop(&mut self) {
        let _ = send_shutdown_signal(&self.request_tx);
    }
}

#[cfg(test)]
#[path = "model_provider_tests.rs"]
mod tests;
