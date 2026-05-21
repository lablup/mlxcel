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

use crate::LoadedModel;
use crate::SamplingConfig;
use crate::server::batch::BatchObservability;
use crate::server::media::{ImageInputLimits, current_image_input_limits};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::feature_cache::ModelVisionCaches;
use crate::vision::merge::InputEmbeddings;
use crate::vlm_runtime::{
    prepare_and_compute_vlm_embeddings, prepare_and_compute_vlm_embeddings_with_cache,
};

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
    /// Decode-time storage backend for server sequence state.
    pub decode_storage_backend: crate::server::DecodeStorageBackend,
    /// Optional pipeline runtime for in-process or remote coordinator execution.
    pub pipeline_parallel_runtime: Option<crate::server::PipelineParallelRuntimeConfig>,
    /// Tensor-parallel runtime configuration.
    pub tensor_parallel: crate::distributed::ShardConfig,
    /// Maximum number of cached post-projection image features per loaded
    /// VLM. `0` disables caching.
    pub vision_cache_size: usize,
    /// Axis B Epic #362 (B8): optional server-wide language-bias config.
    /// Resolved once on the worker thread into a `TokenBiasMap` after the
    /// tokenizer loads, and attached to the batch scheduler for the rest of
    /// the worker's lifetime.
    pub lang_bias_config: Option<mlxcel_core::lang_analyzer::LangBiasConfig>,
    /// Issue #409: server-wide default thinking-token budget.
    pub reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,
    /// Epic #416 / issue #419: cross-request prompt-prefix KV cache store.
    ///
    /// `None` when the feature is disabled by
    /// [`crate::server::prompt_cache::PromptCacheConfig::enabled`]. When
    /// `Some`, the worker thread can publish detached caches and lookup /
    /// adopt them on later requests. The store is thread-safe, so the same
    /// `Arc` is also handed to `AppState` for observation-only use.
    ///
    /// Store handle passed to [`BatchScheduler::with_prompt_cache`] so the
    /// scheduler can adopt detached prefixes on cache hits and donate-back
    /// finished sequences (epic #416 / issue #421).
    pub prompt_cache: Option<Arc<crate::server::prompt_cache::PromptCacheStore>>,
    /// Issue #484 (B11) / #508: server-wide KV cache quantization mode.
    ///
    /// Defaults to [`mlxcel_core::cache::KVCacheMode::Fp16`] (bit-exact
    /// baseline). When a Turbo4 variant is configured, the scheduler
    /// applies it to each new sequence's per-layer cache and picks the
    /// Turbo4-aware paged layout (#482).
    pub kv_cache_mode: mlxcel_core::cache::KVCacheMode,
    /// Issue #545: continuous-batching KV quantization configuration.
    ///
    /// When enabled (`bits > 0`), the scheduler resolves per-layer
    /// [`mlxcel_core::cache::KVCacheMode`] values from this config (with
    /// the last layer optionally forced to FP16) and overrides the
    /// nominal [`Self::kv_cache_mode`] for each newly-allocated sequence.
    /// Defaults to a disabled config so existing deployments stay
    /// bit-exact.
    pub batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig,
    /// Issue #603: maximum KV cache size for plain (non-sliding) caches.
    ///
    /// When `Some(N)`, the batch scheduler caps each per-sequence plain
    /// `KVCache` to `N` tokens by trimming the oldest entries once
    /// `offset > N`. Sliding-window models keep their model-specific
    /// window and bypass this cap. `None` (the default) preserves the
    /// legacy unbounded behaviour.
    pub max_kv_size: Option<usize>,
    /// Issue #666: resolved speculative-decoding dispatch shape.
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
        tracing::info!("Model worker thread starting, loading model...");

        let load_start = Instant::now();
        let result = if let Some(ref pipeline_runtime) = sched_config.pipeline_parallel_runtime {
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
            crate::load_model_with_tensor_parallel(
                &model_path,
                adapter_path.as_deref(),
                &sched_config.tensor_parallel,
            )
        } else if let Some(adapter) = adapter_path {
            tracing::info!("Loading LoRA adapter from {:?}", adapter);
            crate::load_model_with_adapter(&model_path, &adapter)
        } else {
            crate::load_model(&model_path)
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

        // Axis B Epic #362 (B8): resolve the server-wide LangBiasConfig once,
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
            // Issue #405 — emit byte_fragment_entries only when non-zero so
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
        // Issue #666: log the resolved speculative dispatch once at
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
        tracing::info!(
            "Starting BatchScheduler (max_batch_size={}, \
             max_queue_depth={}{chunk_info}{batch_prefill_info}{decode_storage_info}{lang_bias_info}{spec_info})",
            sched_config.max_batch_size,
            sched_config.max_queue_depth,
        );

        // Issue #670 / #674: speculative dispatch is wired end-to-end via
        // the burst path in `BatchScheduler::execute_prefill`. With
        // `max_batch_size > 1` the scheduler assembles an
        // equal-prompt-length window of concurrently-queued speculative
        // requests and drives them through the batched round-loop driver
        // (`MtpBatchedGenerator` / `DFlashBatchedGenerator`) in one tick
        // — true B>1 batched speculative decoding (issue #674). A
        // speculative request whose prompt length, `max_tokens`, or
        // sampling config does not match the current window head, or
        // that arrives alone, still runs as a B=1 burst; in that case
        // the burst occupies the worker thread for its full duration and
        // concurrent classic-decode rows head-of-line-block behind it
        // until it completes. The previous PR-#671 wording described the
        // B=1-only behaviour; this reflects the #674 batched path.
        if sched_config.speculative_dispatch.is_kind_specific() && sched_config.max_batch_size > 1 {
            tracing::info!(
                "Speculative decoding active ({}) with max_batch_size={}: \
                 concurrently-queued speculative requests that share a \
                 prompt length, max_tokens, and sampling config are driven \
                 as a single B>1 batched burst (issue #674). A speculative \
                 request that does not match the current window head, or \
                 that arrives alone, runs as a B=1 burst and head-of-line-\
                 blocks concurrent classic-decode rows for its full \
                 duration. Variable-length-prompt batched bursts are a \
                 documented follow-up.",
                sched_config.speculative_dispatch.summary(),
                sched_config.max_batch_size,
            );
        }

        // Issue #409: resolve the thinking-token id pair once, after the
        // tokenizer is loaded. For models without `<think>`/`</think>` tokens
        // (non-thinking models) this returns `None` and the scheduler silently
        // ignores any budget parameter (logging once per model load).
        let thinking_ids = crate::server::thinking_budget::resolve_thinking_token_ids(&tokenizer);
        if sched_config.reasoning_budget.is_some() && thinking_ids.is_none() {
            tracing::warn!(
                "--reasoning-budget / thinking_budget_tokens requested but this model's \
                 tokenizer has no <think> / </think> tokens; thinking-budget enforcement \
                 is disabled for this session"
            );
        }

        let mut scheduler = super::super::batch::BatchScheduler::with_config(
            model,
            tokenizer,
            config_eos,
            request_rx,
            sched_config.max_batch_size,
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
        .with_token_bias(token_bias)
        .with_reasoning_budget(sched_config.reasoning_budget, thinking_ids)
        .with_prompt_cache(sched_config.prompt_cache)
        .with_kv_cache_mode(sched_config.kv_cache_mode)
        .with_batch_kv_quant(sched_config.batch_kv_quant)
        // Issue #603: cap plain KVCache growth to --max-kv-size when set.
        .with_max_kv_size(sched_config.max_kv_size)
        // Issue #666: attach the resolved speculative dispatch so the
        // scheduler can branch per-request once the round-loop dispatch
        // hook is wired in `decode_single_step`.
        .with_speculative_dispatch(sched_config.speculative_dispatch);
        scheduler.run();
    })
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
        tracing::info!(
            "Model worker thread starting (legacy sequential mode, --no-batch), loading model..."
        );

        let load_start = Instant::now();
        let result = if tensor_parallel.tp_size > 1 {
            crate::load_model_with_tensor_parallel(
                &model_path,
                adapter_path.as_deref(),
                &tensor_parallel,
            )
        } else if let Some(adapter) = adapter_path {
            tracing::info!("Loading LoRA adapter from {:?}", adapter);
            crate::load_model_with_adapter(&model_path, &adapter)
        } else {
            crate::load_model(&model_path)
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

        tracing::info!(
            "Starting legacy sequential worker \
             (max_batch_size=1, prefill_chunk_size=disabled)"
        );

        // Issue #409: resolve the thinking-token id pair once, after the
        // tokenizer is loaded. Mirrors the batched-worker path in
        // `spawn_model_worker_with_batch_config`. For models without
        // `<think>`/`</think>` tokens the helper returns `None` and the
        // scheduler silently ignores any budget parameter (after the
        // warn-once log below).
        let thinking_ids = crate::server::thinking_budget::resolve_thinking_token_ids(&tokenizer);
        if reasoning_budget.is_some() && thinking_ids.is_none() {
            tracing::warn!(
                "--reasoning-budget / thinking_budget_tokens requested but this model's \
                 tokenizer has no <think> / </think> tokens; thinking-budget enforcement \
                 is disabled for this session"
            );
        }

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
        .with_reasoning_budget(reasoning_budget, thinking_ids);
        scheduler.run();
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

pub(crate) fn prepare_request_vlm_embeddings(
    model: &LoadedModel,
    tokenizer: &MlxcelTokenizer,
    prompt: &str,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
    videos: &[crate::server::media::ResolvedVideo],
    vision_caches: Option<&ModelVisionCaches>,
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
        }
        return Ok(None);
    }

    // Issue #596: video inputs route to the Gemma 4 video embedding path,
    // mirroring the CLI dispatch in `commands/generate_vlm.rs::compute_vlm_embeddings`.
    // Combining --video with --audio is rejected upstream and at the route
    // layer; this branch additionally surfaces a clear error if those happen
    // to coexist (defence in depth).
    if !videos.is_empty() {
        if !audio.is_empty() {
            return Err(anyhow!("Combined video and audio inputs are not supported"));
        }
        return prepare_request_video_embeddings(model, prompt_tokens, images, videos);
    }

    // Audio-only or audio+images for Gemma4
    if !audio.is_empty() {
        match prepare_gemma4_audio_embeddings(model, prompt_tokens, images, audio)? {
            Some(embeddings) => return Ok(Some(embeddings)),
            None => {
                // Model does not support audio (not Gemma4 or no audio tower).
                // Log a warning and fall through to image-only or text-only paths.
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
            let image_cache_keys: Vec<Option<crate::vision::feature_cache::CacheKey>> = images
                .iter()
                .map(|bytes| {
                    Some(crate::vision::feature_cache::CacheKey::from_hash(
                        crate::vision::feature_cache::image_hash_from_bytes(bytes),
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
            prepare_and_compute_vlm_embeddings(
                model,
                prompt_tokens,
                prompt,
                &decoded_images,
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
    }

    Ok(None)
}

/// Process audio (and optionally images) for Gemma4 VLM models.
///
/// Returns `Ok(None)` if the model is not a Gemma4 VLM with audio support.
fn prepare_gemma4_audio_embeddings(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[Vec<u8>],
    audio_data: &[Vec<u8>],
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

    // Expand audio tokens: BOA + AUDIO*N + EOA
    crate::vlm_runtime::expand_gemma4_audio_tokens_for_server(
        prompt_tokens,
        gemma4_vl.audio_token_id,
        gemma4_vl.boa_token_id,
        gemma4_vl.eoa_token_id,
        num_audio_tokens,
    );

    // Extract mel spectrogram features
    let extractor =
        audio::AudioFeatureExtractor::new(audio::AudioFeatureExtractorConfig::default());
    let (features, mask) = extractor.extract(&samples, None);
    let num_frames = mask.len();

    let audio_features = mlxcel_core::from_slice_f32(
        &features,
        &[1, num_frames as i32, extractor.feature_size() as i32],
    );
    let mask_i32: Vec<i32> = mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let audio_mask = mlxcel_core::from_slice_i32(&mask_i32, &[1, num_frames as i32]);
    let audio_mask = mlxcel_core::astype(&audio_mask, mlxcel_core::dtype::BOOL);

    // Process images if present alongside audio
    let processed_images = if !images.is_empty() {
        let decoded_images = decode_request_images(images)?;
        let processed = gemma4_vl.processor.preprocess(&decoded_images);
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
    let embeddings = gemma4_vl.get_input_embeddings_with_audio(
        &input_ids_arr,
        &processed_images,
        Some(&audio_features),
        Some(&audio_mask),
    );

    Ok(Some(embeddings))
}

/// Resolve `videos` into Gemma 4 video embeddings (issue #596).
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
) -> Result<Option<InputEmbeddings>> {
    use crate::multimodal::video;

    let gemma4_vl = match model {
        LoadedModel::Gemma4VLM(model) => model,
        _ => {
            return Err(anyhow!(
                "video inputs are only supported by Gemma 4 VLM models in this build"
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
    // Issue #601: every `ResolvedVideo` carries a [`VideoSource`] handle —
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

    let processed_images = gemma4_vl.processor.preprocess(&decoded_images);
    let image_soft_tokens: Vec<usize> = processed_images
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
            &image_soft_tokens,
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
/// by the KV prefix cache (issue #423). Pass `0` for non-cached requests.
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

/// Maximum number of bytes accumulated in the byte-fallback buffer before the
/// buffer is force-flushed as replacement characters to avoid holding tokens
/// indefinitely on pathological model outputs.
const BYTE_FALLBACK_BUFFER_MAX: usize = 4;

pub(crate) struct StreamingDecodeState {
    all_ids: Vec<u32>,
    prev_decoded_len: usize,
    generated_text: String,
    completion_tokens: usize,
    first_token_time: Option<Instant>,
    /// Buffer for raw bytes accumulated from consecutive byte-fallback tokens
    /// (`<0xXX>`). Held until enough bytes arrive to form a valid UTF-8
    /// sequence. Cleared on each successful decode or when a non-byte-fallback
    /// token follows.
    byte_fallback_buffer: Vec<u8>,
}

impl StreamingDecodeState {
    pub(crate) fn new(tokenizer: &MlxcelTokenizer, prompt_tokens: &[i32]) -> Self {
        let all_ids: Vec<u32> = prompt_tokens.iter().map(|&x| x as u32).collect();
        let prev_decoded_len = tokenizer.decode(&all_ids, false).unwrap_or_default().len();

        Self {
            all_ids,
            prev_decoded_len,
            generated_text: String::new(),
            completion_tokens: 0,
            first_token_time: None,
            byte_fallback_buffer: Vec::new(),
        }
    }

    pub(crate) fn on_token(
        &mut self,
        token_id: i32,
        tokenizer: &MlxcelTokenizer,
    ) -> Option<String> {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
        }
        self.completion_tokens += 1;
        self.all_ids.push(token_id as u32);

        // --- Byte-fallback buffering (issue #547) ---
        // Tokenizers that use byte-fallback (e.g. Gemma variants with
        // `byte_fallback = true`) map each byte of a multi-byte UTF-8
        // character to an individual token in the form `<0xXX>`. When decoded
        // one token at a time, incomplete byte sequences produce U+FFFD
        // replacement characters in the output.
        //
        // We detect byte-fallback tokens by looking up the raw token piece. If
        // the piece matches `<0xXX>`, we accumulate the raw byte into a
        // dedicated buffer and defer emission until the buffer forms valid
        // UTF-8. Once a non-byte-fallback token arrives, any remaining bytes in
        // the buffer are force-flushed as replacement characters.
        //
        // The buffer is also bounded to BYTE_FALLBACK_BUFFER_MAX bytes to
        // prevent indefinite buffering on pathological inputs; excess bytes are
        // force-flushed as replacement characters.
        if let Some(piece) = tokenizer.token_piece(token_id as u32)
            && let Some(byte_val) = parse_byte_fallback_token(&piece)
        {
            self.byte_fallback_buffer.push(byte_val);

            // Force-flush if buffer exceeds the maximum size (guards
            // against unbounded growth on unusual model outputs).
            if self.byte_fallback_buffer.len() >= BYTE_FALLBACK_BUFFER_MAX {
                return self.flush_byte_fallback_buffer();
            }

            // Try to decode the buffered bytes as UTF-8.
            match std::str::from_utf8(&self.byte_fallback_buffer) {
                Ok(decoded) if !decoded.is_empty() => {
                    let decoded = decoded.to_string();
                    self.byte_fallback_buffer.clear();
                    self.generated_text.push_str(&decoded);
                    // Advance prev_decoded_len by syncing with the full
                    // re-decode so that subsequent tokens start from the
                    // right position.
                    let full_text = tokenizer.decode(&self.all_ids, false).unwrap_or_default();
                    self.prev_decoded_len = safe_emit_boundary(&full_text);
                    return Some(decoded);
                }
                _ => {
                    // Incomplete sequence -- hold in buffer, emit nothing.
                    return None;
                }
            }
        }

        // Non-byte-fallback token: flush any leftover byte-fallback bytes as
        // replacement characters before continuing with the normal decode path.
        if !self.byte_fallback_buffer.is_empty() {
            let flushed = self.flush_byte_fallback_buffer();
            // Re-run the normal path for the current (non-byte-fallback) token.
            // The flush already updated prev_decoded_len, so the subsequent
            // re-decode will pick up where it left off.
            let extra = self.emit_regular_token(tokenizer);
            return match (flushed, extra) {
                (Some(a), Some(b)) => Some(a + &b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
        }

        self.emit_regular_token(tokenizer)
    }

    /// Emit text from the current `all_ids` using the standard re-decode path.
    ///
    /// Advances `prev_decoded_len` to the safe boundary. Returns the newly
    /// emitted text, or `None` if nothing can be emitted yet.
    fn emit_regular_token(&mut self, tokenizer: &MlxcelTokenizer) -> Option<String> {
        let full_text = tokenizer.decode(&self.all_ids, false).unwrap_or_default();

        // Find the safe emit boundary: skip trailing U+FFFD replacement characters.
        // Byte-level BPE tokenizers split multi-byte UTF-8 sequences across tokens.
        // Incomplete byte sequences decode as U+FFFD, but become valid characters
        // once the completing token arrives. Emitting FFFD prematurely corrupts
        // the output because the byte offset shifts when the replacement chars
        // resolve into shorter real characters.
        let safe_len = safe_emit_boundary(&full_text);

        if safe_len <= self.prev_decoded_len {
            return None;
        }

        let new_text = &full_text[self.prev_decoded_len..safe_len];
        if new_text.is_empty() {
            return None;
        }

        self.generated_text.push_str(new_text);
        self.prev_decoded_len = safe_len;
        Some(new_text.to_string())
    }

    /// Flush the byte-fallback buffer to the output.
    ///
    /// If the buffered bytes form valid UTF-8, emit the decoded string.
    /// Otherwise emit one replacement character (U+FFFD) per buffered byte, in
    /// line with the `ByteFallback` decoder's own fallback behaviour. In both
    /// cases the buffer is cleared and `prev_decoded_len` is re-synced.
    fn flush_byte_fallback_buffer(&mut self) -> Option<String> {
        if self.byte_fallback_buffer.is_empty() {
            return None;
        }
        let buf = std::mem::take(&mut self.byte_fallback_buffer);
        let flushed = match std::str::from_utf8(&buf) {
            Ok(s) => s.to_string(),
            Err(_) => "\u{FFFD}".repeat(buf.len()),
        };
        // Advance prev_decoded_len by the exact byte length of what was just
        // emitted. Using safe_emit_boundary on the full re-decode would advance
        // past any trailing text from the next regular token (which has already
        // been pushed into all_ids), causing that text to be silently dropped.
        self.prev_decoded_len += flushed.len();

        if flushed.is_empty() {
            None
        } else {
            self.generated_text.push_str(&flushed);
            Some(flushed)
        }
    }

    /// Flush any remaining buffered text (including unresolved replacement chars)
    /// at the end of generation.
    ///
    /// Also drains the byte-fallback buffer: any accumulated bytes that did not
    /// form a complete UTF-8 sequence are emitted as U+FFFD replacement
    /// characters so that the streamed output matches the non-streaming result.
    pub(crate) fn flush(&mut self, tokenizer: &MlxcelTokenizer) {
        // Drain the byte-fallback buffer first. Any incomplete byte sequences
        // are flushed as replacement characters here; we then skip past the
        // corresponding replacement chars in the full-decode result below so
        // they are not emitted a second time.
        if !self.byte_fallback_buffer.is_empty() {
            self.flush_byte_fallback_buffer();
            // After a byte-fallback flush, prev_decoded_len sits at
            // safe_emit_boundary (i.e. just before trailing U+FFFD chars
            // from the incomplete sequence). Advance it past those trailing
            // replacement chars so the normal emit path below does not
            // re-emit them.
            let full_text = tokenizer.decode(&self.all_ids, false).unwrap_or_default();
            self.prev_decoded_len = full_text.len();
            return;
        }

        let full_text = tokenizer.decode(&self.all_ids, false).unwrap_or_default();
        if full_text.len() > self.prev_decoded_len {
            let remaining = &full_text[self.prev_decoded_len..];
            self.generated_text.push_str(remaining);
            self.prev_decoded_len = full_text.len();
        }
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
    /// the KV prefix cache (epic #416 / issue #423).
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
}

/// Find the byte position after the last non-U+FFFD character.
/// Trailing replacement characters are buffered because they likely come from
/// incomplete multi-byte UTF-8 sequences that will be completed by the next token.
fn safe_emit_boundary(text: &str) -> usize {
    text.char_indices()
        .rev()
        .find(|(_, c)| *c != '\u{FFFD}')
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0)
}

/// Detect a byte-fallback token piece and return the raw byte value.
///
/// Byte-fallback tokens have the form `<0xXX>` where `XX` is a two-digit
/// hex value, e.g. `<0xE2>`, `<0x80>`, `<0x9C>`. These are produced by
/// tokenizers whose model config has `byte_fallback = true` (common in Gemma
/// and some Llama variants built with SentencePiece byte-fallback vocabulary).
///
/// Returns `None` for regular token pieces that do not match this pattern.
///
/// Used by: StreamingDecodeState (model_worker.rs)
fn parse_byte_fallback_token(piece: &str) -> Option<u8> {
    // Must be exactly 6 bytes: '<', '0', 'x', HI, LO, '>'
    // Use byte-level checks so that `from_str_radix` never sees a leading '+'
    // or '-' sign (defense-in-depth: e.g. `<0x+f>` would otherwise parse as
    // byte 0x0F).
    let bytes = piece.as_bytes();
    if bytes.len() == 6
        && &bytes[..3] == b"<0x"
        && bytes[5] == b'>'
        && bytes[3].is_ascii_hexdigit()
        && bytes[4].is_ascii_hexdigit()
    {
        u8::from_str_radix(std::str::from_utf8(&bytes[3..5]).ok()?, 16).ok()
    } else {
        None
    }
}

#[cfg(test)]
#[path = "model_worker_tests.rs"]
mod tests;
