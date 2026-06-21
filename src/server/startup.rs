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

//! Server startup pipeline shared by `mlxcel serve` and `mlxcel-server`.
//!
//! This module keeps process-level side effects such as tracing initialization,
//! chat-template resolution, model warmup, and socket binding out of
//! `server/mod.rs` so the server root can focus on shared types and state.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use tower::Service;

use crate::SamplingConfig;
use crate::distributed::pipeline::{
    ElasticPpConfig, RemoteStageServiceConfig, RemoteStageServiceHandle,
    resolve_in_process_pipeline_num_layers,
};
use crate::distributed::{
    ClusterConfig, ClusterDiscoveryMode, ClusterInitPlan, ClusterInitRequest, NodeRegistry,
    NodeRole, TransportBackend, plan_cluster, resolve_model_shard_plan, shard_config_from_cli,
    validate_supported_runtime, write_plan_toml,
};

use super::batch::BatchObservability;
use super::state::ModelMediaSupport;
use super::{
    AppState, BatchMetrics, ChatTemplateProcessor, ModelProvider, PipelineParallelRuntimeConfig,
    ServerConfig, ServerGenerateOptions, create_app,
};

struct ResolvedDistributedStartup {
    _node_registry: Option<NodeRegistry>,
    pipeline_runtime: Option<PipelineParallelRuntimeConfig>,
    remote_stage_service: Option<RemoteStageServiceConfig>,
}

/// Minimum effective context window accepted for each request slot.
///
/// `llama-server` treats `--ctx-size` as a total context budget shared by
/// parallel slots. Below this floor, a process can start successfully but
/// become unusable for normal chat/completion traffic, so fail early with a
/// clear operator-facing error.
pub const MIN_PARALLEL_CONTEXT_SIZE: usize = 512;

/// Startup configuration for the server (shared between `mlxcel serve` and `mlxcel-server`).
#[derive(Debug)]
pub struct ServerStartupConfig {
    // Model
    pub model_path: PathBuf,
    pub adapter_path: Option<PathBuf>,
    pub model_alias: Option<String>,

    // Network
    pub host: String,
    pub port: u16,

    // Auth
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,

    // Limits
    pub n_parallel: usize,
    pub ctx_size: usize,
    pub n_predict: i32, // -1 = unlimited
    pub timeout: u64,

    // Speculative decoding
    pub draft_model_path: Option<PathBuf>,
    pub draft_max: usize,
    /// raw `--draft-kind` value. `None` means
    /// "auto-detect from the drafter `config.json::model_type`" when
    /// `draft_model_path` is also supplied. Parsing into
    /// [`mlxcel_core::drafter::DrafterKind`] and reconciliation against
    /// the drafter config happens at the dispatch site via
    /// [`mlxcel_core::drafter::resolve_drafter_kind`].
    pub draft_kind: Option<String>,
    /// explicit `--draft-block-size` override. `None` means
    /// "use the per-kind default" — `4` for MTP, `16` for DFlash. See
    /// [`crate::cli::speculative_args::default_block_size_for_kind`].
    pub draft_block_size: Option<u32>,

    // Chat template
    pub chat_template: Option<String>,
    pub chat_template_file: Option<PathBuf>,

    // Endpoint toggles
    pub enable_slots: bool,
    pub enable_props: bool,
    pub enable_metrics: bool,

    // Batch scheduling
    pub max_batch_size: Option<usize>,
    pub max_queue_depth: usize,
    /// Bound on the audio worker command queue (admission control). Forwarded to
    /// [`super::config::ServerConfig::audio_queue_depth`].
    pub audio_queue_depth: usize,
    /// Per-request reply timeout for the audio worker, in seconds. Forwarded to
    /// [`super::config::ServerConfig::audio_request_timeout_secs`].
    pub audio_request_timeout_secs: u64,
    /// Prefill chunk size in tokens (0 = disabled).
    pub prefill_chunk_size: usize,
    /// Set when `--batch-size` and `--prefill-chunk-size` conflict; triggers a startup warning.
    pub batch_size_conflict: bool,
    /// Set when `--ubatch-size` was provided; triggers a startup info notice.
    pub ubatch_size_provided: bool,
    /// Enable preemptive eviction when batch is full.
    pub enable_preemption: bool,
    /// Enable experimental VLM prompt-prefix cache sharing (#124 step c,
    /// `--enable-vlm-prefix-cache`). Default off; forwarded to the scheduler.
    pub enable_vlm_prefix_cache: bool,
    /// Validated CORS allow-list origins (#244). `None` keeps the permissive
    /// default; `Some(non_empty)` restricts cross-origin requests to exactly
    /// these origins. Built from `--allowed-origins` in
    /// [`super::ServerStartupInput::into_startup_config`] and forwarded to
    /// [`super::config::ServerConfig`].
    pub cors_allowed_origins: Option<Vec<axum::http::HeaderValue>>,
    /// Preemption policy string from CLI (parsed into enum at build_server_config).
    pub preemption_policy: String,
    /// Force the legacy sequential worker, bypassing the batch scheduler.
    pub no_batch: bool,
    /// Maximum number of pending requests to batch together for prefill (default: 1).
    pub max_batch_prefill: usize,
    /// Decode-time storage backend requested by the CLI. `None` preserves the
    /// legacy `MLXCEL_SERVER_DECODE_STORAGE` env-var fallback.
    pub decode_storage_backend: Option<crate::server::DecodeStorageBackend>,

    // Warmup
    pub warmup: bool,

    // Default sampling
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,

    // DRY
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: i32, // -1 = use full context
    pub dry_sequence_breakers: Vec<String>,

    // Logging
    pub verbose: bool,
    pub log_disable: bool,
    pub log_file: Option<PathBuf>,

    // Distributed inference
    /// Path to a TOML cluster configuration file.
    pub distributed_config: Option<PathBuf>,
    /// Node role (CLI shorthand, parsed into `NodeRole` at startup).
    pub node_role: Option<String>,
    /// Unique node identifier (CLI shorthand).
    pub node_id: Option<String>,
    /// Static peer addresses (CLI shorthand).
    pub peers: Vec<SocketAddr>,
    /// Prefill-node peers a decode node receives handoffs from (disaggregated
    /// serving, #126).
    pub prefill_peers: Vec<SocketAddr>,
    /// Decode-node peers a prefill node hands off to (disaggregated serving,
    /// #126).
    pub decode_peers: Vec<SocketAddr>,
    /// This node's own serving-role transport bind address (disaggregated
    /// serving, #126). `Some` enables the live prefill/decode role loop on a
    /// non-hybrid node.
    pub serving_bind: Option<SocketAddr>,
    /// Manual pipeline-parallel layer partition spec (e.g. "0-15,16-31").
    /// When `None`, auto-partition mode is used.
    pub pp_layers: Option<String>,
    /// Micro-batch size for in-process pipeline execution.
    pub pp_micro_batch_size: usize,

    // Zero-config multi-machine pipeline bring-up.
    /// Zero-config coordinator intent: pipeline depth for `mlxcel-server --pp-auto N`.
    pub pp_auto: Option<u32>,
    /// Zero-config peer intent: `mlxcel-server --pp-peer` joins a running cluster.
    pub pp_peer: bool,
    /// Cluster discovery mode string (parsed into `ClusterDiscoveryMode` at startup).
    pub cluster_discovery: String,
    /// Optional override for the zero-config cluster name.
    pub cluster_name: Option<String>,
    /// Static seed peers for the zero-config bring-up.
    pub cluster_peers: Vec<SocketAddr>,
    /// Optional UDP port for the discovery beacon.
    pub cluster_discovery_port: Option<u16>,
    /// Optional coordinator control-plane bind address.
    pub cluster_control_addr: Option<SocketAddr>,
    /// Optional output path for the emitted cluster TOML.
    pub cluster_config_out: Option<PathBuf>,
    /// When `true`, plan the cluster and exit before starting workers.
    pub dry_run: bool,

    /// Number of tensor-parallel ranks (1 = disabled).
    pub tp_size: usize,
    /// MoE expert sharding mode string (parsed into `MoeShardMode` at plan generation).
    pub tp_moe_mode: String,
    /// Embedding sharding mode string (parsed into `EmbeddingMode` at plan generation).
    pub tp_embedding_mode: String,
    /// LM head sharding mode string (parsed into `EmbeddingMode` at plan generation).
    pub tp_lm_head_mode: String,

    // Vision feature cache.
    /// Maximum number of cached post-projection image features per loaded model.
    ///
    /// `0` disables the cache. Default matches
    /// [`DEFAULT_VISION_CACHE_SIZE`](crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE).
    pub vision_cache_size: usize,

    /// Maximum encoded image payload bytes accepted per image content block.
    pub max_image_payload_size: usize,
    /// Maximum number of image content blocks accepted in one request.
    pub max_images_per_request: usize,
    /// Maximum decoded image width passed to `image::Limits`.
    pub max_image_width: u32,
    /// Maximum decoded image height passed to `image::Limits`.
    pub max_image_height: u32,
    /// Maximum decoder allocation budget passed to `image::Limits`.
    pub max_image_decode_alloc_bytes: u64,

    // Elastic pipeline-parallel repartitioning.
    /// When `true`, the runtime constructs the elastic repartition coordinator
    /// described in `docs_internal/architecture/elastic-pipeline-repartition-
    /// 20260418.md`. Off by default so existing deployments are unaffected.
    pub enable_elastic_pp: bool,
    /// Drain timeout (seconds). Only consulted when `enable_elastic_pp` is set.
    pub elastic_pp_drain_timeout: u64,
    /// Memory-pressure trigger fraction. Clamped to `(0.0, 1.0]` at consumption
    /// time.
    pub elastic_pp_pressure_fraction: f64,
    /// Cool-down (seconds) between successive memory-pressure triggers on the
    /// same stage.
    pub elastic_pp_cool_down: u64,

    // Observability.
    /// Port operators requested for `/metrics`. Currently informational —
    /// the endpoint is multiplexed onto `port` because the server has a
    /// single HTTP listener.
    pub metrics_port: Option<u16>,
    /// Optional chrome-tracing JSON output path for pipeline scheduler
    /// actions. `Some(path)` constructs a `PpTracer`.
    pub debug_pp_trace: Option<PathBuf>,

    /// Axis B (B8): server-wide language-bias configuration
    /// already resolved from CLI flags (B6) or the `LLAMA_ARG_LANG_BIAS`
    /// env-var path (B7). `None` preserves the bit-exact baseline path.
    pub lang_bias_config: Option<mlxcel_core::lang_analyzer::LangBiasConfig>,

    /// server-wide default for the thinking-token budget.
    ///
    /// Normalized from the raw `i32` on [`super::ServerStartupInput`] via
    /// [`super::thinking_budget::ThinkingBudget::from_raw_i32`]. `None` means
    /// "unrestricted reasoning" (llama.cpp `-1` semantics); per-request body
    /// fields may still impose or lift a cap on a per-request basis. Applies
    /// only to Qwen3-family thinking models — for models that lack
    /// `<think>` / `</think>` token IDs the scheduler resolves the token pair
    /// to `None` and the budget is silently ignored.
    pub reasoning_budget: Option<super::thinking_budget::ThinkingBudget>,

    /// server-wide default chat-template kwargs.
    ///
    /// Parsed from the raw JSON string on [`super::ServerStartupInput`] via
    /// [`super::chat_template_kwargs::ChatTemplateKwargs::from_json_str`].
    /// `None` means no server defaults; per-request kwargs may still apply.
    pub chat_template_kwargs: Option<super::chat_template_kwargs::ChatTemplateKwargs>,

    /// resolved prompt-prefix KV cache policy.
    ///
    /// Built from CLI flags and env vars via
    /// [`super::cli_input::build_prompt_cache_config`] inside
    /// [`super::ServerStartupInput::into_startup_config`].
    /// The default is [`super::prompt_cache::PromptCacheConfig::default`]
    /// (enabled, 2 GiB cap, 1024 entries, 3600 s TTL, 32 token min).
    pub prompt_cache: super::prompt_cache::PromptCacheConfig,

    /// (B11): resolved KV cache mode for per-sequence cache
    /// construction.
    ///
    /// Resolved from `--cache-type-k`/`--cache-type-v` (split flags,
    /// `LLAMA_ARG_CACHE_TYPE_K`/`LLAMA_ARG_CACHE_TYPE_V` env vars) or the
    /// legacy `--kv-cache-mode` shorthand.  Defaults to `KVCacheMode::Fp16`
    /// (bit-exact baseline, no quantization).
    ///
    /// The split flags take precedence over the legacy shorthand. When only
    /// one of K or V is specified, the unspecified side defaults to `fp16`.
    /// Unsupported K/V combinations are rejected at startup.
    pub kv_cache_mode: mlxcel_core::cache::KVCacheMode,

    /// resolved batch KV cache quantization configuration
    /// (uniform `mx.quantize` or TurboQuant variant) for the
    /// continuous-batching path.
    ///
    /// Built from the `--kv-bits`, `--kv-group-size`, `--kv-quant-scheme`,
    /// and `--kv-skip-last-layer` CLI flags. When `bits == 0`
    /// ([`mlxcel_core::cache::BatchKvQuantConfig::is_enabled`] returns
    /// `false`) the batched scheduler keeps the legacy
    /// `kv_cache_mode`-driven path bit-exactly. Otherwise the scheduler
    /// reuses `BatchKvQuantConfig::resolve_layer_modes` so the last layer
    /// stays at FP16 even when the nominal mode is quantized — preserving
    /// quality on deep models such as gemma-4-31b.
    pub batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig,

    /// maximum KV cache size for plain (non-sliding) KVCache
    /// instances. `None` preserves the legacy unbounded behaviour.
    ///
    /// Resolved from `--max-kv-size` / `LLAMA_ARG_MAX_KV_SIZE`. See the
    /// corresponding field on [`crate::server::ServerConfig`] for full
    /// semantics.
    pub max_kv_size: Option<usize>,

    /// Paged KV pool block-budget directive (`--kv-cache-budget`). See the
    /// corresponding field on [`crate::server::ServerConfig`] for full
    /// semantics. `None` (the default) keeps the pool unbounded.
    pub kv_cache_budget: Option<crate::memory_estimate::PagedBudgetDirective>,

    /// maximum number of responses kept in the
    /// [`crate::server::responses_store::ResponsesStore`]. `0` disables
    /// response persistence entirely, in which case `GET /v1/responses/:id`
    /// and `previous_response_id` return 400. Resolved from
    /// `--responses-store-max-entries` / `LLAMA_ARG_RESPONSES_STORE_MAX_ENTRIES`.
    pub responses_store_max_entries: usize,
    /// TTL (seconds) for in-memory response store entries.
    /// `0` disables TTL (entries are evicted only by capacity pressure).
    /// Resolved from `--responses-store-ttl-secs` / `LLAMA_ARG_RESPONSES_STORE_TTL_SECS`.
    pub responses_store_ttl_secs: u64,
    /// capacity cap for the conversation transcript store.
    /// `0` disables the store; requests referencing `conversation` still
    /// succeed but operate as if the transcript is empty.
    pub conversation_store_max_entries: usize,
    /// TTL (seconds) for conversation transcripts. `0`
    /// disables TTL.
    pub conversation_store_ttl_secs: u64,

    /// (A4): resolved path to a YAML weight-load surgery
    /// configuration. `None` keeps the bit-exact baseline load path.
    ///
    /// The path is parsed into a [`mlxcel_surgery::SurgeryPipeline`]
    /// inside [`start_server`] and installed via
    /// [`crate::surgery::set_active_pipeline`] before the model worker
    /// thread is spawned. The string is propagated through the startup
    /// config (rather than constructing the pipeline at
    /// [`super::cli_input::ServerStartupInput::into_startup_config`]
    /// time) so the `serde::Debug`-friendly shape of this struct is
    /// preserved and so tests that drive `start_server` without
    /// passing a real YAML file (the common case in `tests/`) remain
    /// trivial to construct.
    #[cfg(feature = "surgery")]
    pub surgery_config_path: Option<PathBuf>,

    /// `--max-denoising-steps` (issue #217 phase 3). Serve-level diffusion
    /// step-cap override; `None` keeps the checkpoint default.
    pub max_denoising_steps: Option<usize>,
    /// `--diffusion-sampler` (issue #217 phase 3).
    pub diffusion_sampler: String,
    /// `--diffusion-threshold` (issue #217 phase 3).
    pub diffusion_threshold: f32,
}

impl Default for ServerStartupConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            adapter_path: None,
            model_alias: None,
            host: "127.0.0.1".to_string(),
            port: 8080,
            api_key: None,
            api_key_file: None,
            n_parallel: 1,
            ctx_size: 0,
            n_predict: -1,
            timeout: 600,
            draft_model_path: None,
            draft_max: 16,
            // speculative-decoding selector defaults.
            // `draft_kind = None` means "auto-detect when a drafter is
            // supplied, otherwise inert"; `draft_block_size = None`
            // means "fall back to the per-kind default once the kind
            // has been resolved".
            draft_kind: None,
            draft_block_size: None,
            max_batch_size: None,
            max_queue_depth: 32,
            audio_queue_depth: crate::server::config::DEFAULT_AUDIO_QUEUE_DEPTH,
            audio_request_timeout_secs: crate::server::config::DEFAULT_AUDIO_REQUEST_TIMEOUT_SECS,
            prefill_chunk_size: 512,
            batch_size_conflict: false,
            ubatch_size_provided: false,
            enable_preemption: false,
            enable_vlm_prefix_cache: false,
            cors_allowed_origins: None,
            preemption_policy: "longest-first".to_string(),
            no_batch: false,
            max_batch_prefill: 1,
            decode_storage_backend: None,
            chat_template: None,
            chat_template_file: None,
            enable_slots: true,
            enable_props: false,
            enable_metrics: false,
            warmup: true,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.1,
            seed: None,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: -1,
            dry_sequence_breakers: Vec::new(),
            verbose: false,
            log_disable: false,
            log_file: None,
            distributed_config: None,
            node_role: None,
            node_id: None,
            peers: Vec::new(),
            prefill_peers: Vec::new(),
            decode_peers: Vec::new(),
            serving_bind: None,
            pp_layers: None,
            pp_micro_batch_size: 1,
            pp_auto: None,
            pp_peer: false,
            cluster_discovery: "static".to_string(),
            cluster_name: None,
            cluster_peers: Vec::new(),
            cluster_discovery_port: None,
            cluster_control_addr: None,
            cluster_config_out: None,
            dry_run: false,
            tp_size: 1,
            tp_moe_mode: "expert_parallel".to_string(),
            tp_embedding_mode: "replicated".to_string(),
            tp_lm_head_mode: "replicated".to_string(),
            vision_cache_size: crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE,
            max_image_payload_size: crate::server::DEFAULT_MAX_IMAGE_PAYLOAD_SIZE,
            max_images_per_request: crate::server::DEFAULT_MAX_IMAGES_PER_REQUEST,
            max_image_width: crate::server::DEFAULT_MAX_IMAGE_WIDTH,
            max_image_height: crate::server::DEFAULT_MAX_IMAGE_HEIGHT,
            max_image_decode_alloc_bytes: crate::server::DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES,
            enable_elastic_pp: false,
            elastic_pp_drain_timeout: 120,
            elastic_pp_pressure_fraction: 0.92,
            elastic_pp_cool_down: 30,
            metrics_port: None,
            debug_pp_trace: None,
            lang_bias_config: None,
            reasoning_budget: None,
            chat_template_kwargs: None,
            prompt_cache: super::prompt_cache::PromptCacheConfig::default(),
            kv_cache_mode: mlxcel_core::cache::KVCacheMode::Fp16,
            batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig::default(),
            max_kv_size: None,
            kv_cache_budget: None,
            responses_store_max_entries: 1024,
            responses_store_ttl_secs: 3600,
            conversation_store_max_entries: 256,
            conversation_store_ttl_secs: 3600,
            #[cfg(feature = "surgery")]
            surgery_config_path: None,
            max_denoising_steps: None,
            diffusion_sampler: "entropy-bound".to_string(),
            diffusion_threshold: 0.9,
        }
    }
}

/// Return the number of slots that share the total context budget.
///
/// Continuous batching can admit `--max-batch-size` concurrent decode
/// sequences, so an explicit override becomes the sizing divisor. The legacy
/// sequential worker processes one request at a time and therefore keeps the
/// full context budget for that single active slot.
pub fn effective_parallel_context_slots(
    n_parallel: usize,
    max_batch_size: Option<usize>,
    no_batch: bool,
) -> usize {
    if no_batch {
        1
    } else {
        max_batch_size.unwrap_or(n_parallel).max(1)
    }
}

/// Resolve the effective per-slot context window from a total context budget.
pub fn resolve_parallel_context_size(
    ctx_size: usize,
    n_parallel: usize,
    max_batch_size: Option<usize>,
    no_batch: bool,
) -> usize {
    if ctx_size == 0 {
        return 0;
    }

    let slots = effective_parallel_context_slots(n_parallel, max_batch_size, no_batch);
    ctx_size / slots
}

fn resolve_context_kv_cap(
    per_slot_context_size: usize,
    explicit_max_kv_size: Option<usize>,
) -> Option<usize> {
    if per_slot_context_size == 0 {
        return explicit_max_kv_size;
    }

    Some(match explicit_max_kv_size {
        Some(max_kv_size) => max_kv_size.min(per_slot_context_size),
        None => per_slot_context_size,
    })
}

fn validate_parallel_context_startup(startup: &ServerStartupConfig) -> Result<()> {
    if startup.ctx_size == 0 {
        return Ok(());
    }

    let slots = effective_parallel_context_slots(
        startup.n_parallel,
        startup.max_batch_size,
        startup.no_batch,
    );
    let per_slot_context_size = resolve_parallel_context_size(
        startup.ctx_size,
        startup.n_parallel,
        startup.max_batch_size,
        startup.no_batch,
    );

    anyhow::ensure!(
        per_slot_context_size >= MIN_PARALLEL_CONTEXT_SIZE,
        "--ctx-size {} divided across {} active slot(s) gives {} tokens per slot, below the minimum supported per-slot context size of {}; increase --ctx-size, reduce --parallel/--max-batch-size, or use --no-batch for single-slot serving",
        startup.ctx_size,
        slots,
        per_slot_context_size,
        MIN_PARALLEL_CONTEXT_SIZE
    );

    Ok(())
}

/// Resolve the elastic repartition configuration from CLI flags.
///
/// Returns `None` when `--enable-elastic-pp` is not set, which is the
/// default. Callers that construct a
/// [`super::super::distributed::pipeline::RepartitionCoordinator`] should
/// skip construction when this helper returns `None`.
pub(super) fn resolve_elastic_pp_config(startup: &ServerStartupConfig) -> Option<ElasticPpConfig> {
    if !startup.enable_elastic_pp {
        return None;
    }
    let cfg = ElasticPpConfig::enabled()
        .with_drain_timeout(std::time::Duration::from_secs(
            startup.elastic_pp_drain_timeout,
        ))
        .with_cool_down(std::time::Duration::from_secs(startup.elastic_pp_cool_down))
        .with_trigger_memory_fraction(startup.elastic_pp_pressure_fraction);
    Some(cfg)
}

pub(super) fn resolve_default_max_tokens(n_predict: i32) -> usize {
    if n_predict < 0 {
        4096
    } else {
        n_predict as usize
    }
}

pub(super) fn resolve_dry_penalty_last_n(value: i32) -> usize {
    if value < 0 { 0 } else { value as usize }
}

/// Resolve API key from flag or file.
pub(super) fn resolve_api_key(
    api_key: Option<String>,
    api_key_file: Option<&Path>,
) -> Result<Option<String>> {
    if api_key.is_some() {
        return Ok(api_key);
    }
    if let Some(path) = api_key_file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read API key file: {:?}", path))?;
        let key = content.trim().to_string();
        if key.is_empty() {
            anyhow::bail!("API key file {:?} is empty", path);
        }
        return Ok(Some(key));
    }
    Ok(None)
}

/// Walk the directories named in `MLXCEL_VIDEO_DIR_ALLOWLIST` once at
/// startup and emit a `tracing::warn!` for any entry whose group or world
/// write bits are set (hardening / follow-up).
///
/// Reads the env var via [`super::media::video_dir_allowlist_from_env`]
/// and delegates the actual permission check to
/// [`super::media::scan_insecure_allowlist_dirs`]. Both helpers fail closed
/// when the env var is empty/unset, so this runs as a no-op for operators
/// who haven't opted into the feature.
///
/// closed the dominant canonicalise → ffmpeg-open TOCTOU window
/// at the kernel level: every file open now uses `O_NOFOLLOW` (so a symlink
/// swap in the metadata→open gap returns `ELOOP` instead of silently
/// following the link), and subprocesses receive `/dev/fd/N` rather than a
/// path, so they cannot be redirected post-open regardless. Any residual
/// window is limited to the kernel-internal `namei` compare-and-swap race,
/// which is not practical to exploit. The startup warning is preserved as
/// defence-in-depth and operator-policy guidance: a writable upload
/// directory remains a policy red flag (anyone with shell access on the
/// host can drop arbitrary files into the sandbox), and restricting to
/// mode 0750 or stricter is the recommended posture.
///
/// On non-Unix targets this function emits a single warning when
/// `MLXCEL_VIDEO_DIR_ALLOWLIST` is set, because the `O_NOFOLLOW` +
/// fd-passing security layer is unavailable on those platforms. The
/// video-allowlist feature is a Linux/macOS capability; non-Unix operators
/// should leave the env var unset.
///
/// We log instead of refusing to start so that operators can still bring
/// the server up with a loose-mode directory while they fix the
/// permissions; the resolver itself is safe against the static path
/// checks (canonicalise + allowlist prefix + regular-file + extension)
/// and the fd-passing + `O_NOFOLLOW` guarantee.
fn warn_on_insecure_video_allowlist() {
    let allowlist = super::media::video_dir_allowlist_from_env();
    if allowlist.is_empty() {
        return;
    }
    #[cfg(not(unix))]
    {
        tracing::warn!(
            "{} is set but the O_NOFOLLOW + fd-passing security layer \
 is only available on Unix (Linux, macOS). \
             On this platform the video resolver falls back to path-only \
             mode, which retains a residual TOCTOU window. Leave \
             MLXCEL_VIDEO_DIR_ALLOWLIST unset on non-Unix deployments.",
            super::media::VIDEO_DIR_ALLOWLIST_ENV
        );
    }
    #[cfg(unix)]
    {
        let insecure = super::media::scan_insecure_allowlist_dirs(&allowlist);
        for dir in insecure {
            tracing::warn!(
                "Allowlist directory '{}' is world/group-writable. The dominant \
                 TOCTOU race against video resolution is closed at the kernel level \
                 by the O_NOFOLLOW + fd-passing fix, but a writable \
                 upload directory remains a policy red flag. Restrict permissions \
                 to 0750 or stricter.",
                dir.display()
            );
        }
    }
}

/// Inspect the model's `config.json` and decide which media inputs the chat
/// handler should accept.
///
/// Failure modes are intentionally tolerant: if `config.json` is missing or
/// the type cannot be determined, the loaded model would have failed earlier
/// in startup; falling back to "no media support" here just means video
/// requests get a 400, which is the safe default.
fn detect_model_media_support(model_path: &Path) -> ModelMediaSupport {
    use crate::models::ModelType;

    let model_type = match crate::models::get_model_type(model_path) {
        Ok(t) => t,
        Err(err) => {
            tracing::debug!(
                "Could not determine model type from {:?} for media-support detection: {err}; \
                 disabling media support",
                model_path
            );
            return ModelMediaSupport::default();
        }
    };

    // Currently only Gemma 4 VLM consumes `video_url` content blocks. Mirror
    // the dispatch in `commands/generate_vlm::compute_vlm_embeddings` and add
    // new variants here when more video-capable models land.
    let video = matches!(model_type, ModelType::Gemma4VLM);
    if video {
        tracing::info!(
            "model_type={:?}: enabling video_url content block support",
            model_type
        );
    }

    ModelMediaSupport { video }
}

/// Resolve chat template from override string, file, or model's tokenizer metadata.
pub(super) fn resolve_chat_template(
    template_override: Option<&str>,
    template_file: Option<&Path>,
    model_path: &Path,
) -> Result<ChatTemplateProcessor> {
    if let Some(template) = template_override {
        return Ok(ChatTemplateProcessor::with_template(template.to_string()));
    }
    if let Some(path) = template_file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read chat template file: {:?}", path))?;
        return Ok(ChatTemplateProcessor::with_template(content));
    }
    Ok(ChatTemplateProcessor::from_model_path(model_path)?.unwrap_or_default())
}

/// Parse a preemption policy string from CLI into the enum.
///
/// Accepts "longest-first" (default) and "lowest-priority" (case-insensitive).
fn parse_preemption_policy(s: &str) -> crate::server::PreemptionPolicy {
    match s.trim().to_ascii_lowercase().as_str() {
        "lowest-priority" | "lowestpriority" => crate::server::PreemptionPolicy::LowestPriority,
        _ => crate::server::PreemptionPolicy::LongestFirst,
    }
}

fn resolve_decode_storage_backend() -> crate::server::DecodeStorageBackend {
    match std::env::var("MLXCEL_SERVER_DECODE_STORAGE") {
        Ok(raw) => match raw.parse::<crate::server::DecodeStorageBackend>() {
            Ok(backend) => backend,
            Err(err) => {
                tracing::warn!(
                    "{err}; falling back to automatic decode storage selection (set MLXCEL_SERVER_DECODE_STORAGE=auto|dense|paged)"
                );
                crate::server::DecodeStorageBackend::Auto
            }
        },
        Err(_) => crate::server::DecodeStorageBackend::Auto,
    }
}

pub(super) fn build_server_config(
    startup: &ServerStartupConfig,
    api_key: Option<String>,
) -> ServerConfig {
    let tensor_parallel = shard_config_from_cli(
        startup.tp_size,
        &startup.tp_moe_mode,
        &startup.tp_embedding_mode,
        &startup.tp_lm_head_mode,
    )
    .expect("tensor parallel config was already validated during startup");
    let max_batch_size = startup.max_batch_size.unwrap_or(startup.n_parallel).max(1);
    let context_size = resolve_parallel_context_size(
        startup.ctx_size,
        startup.n_parallel,
        startup.max_batch_size,
        startup.no_batch,
    );
    let max_kv_size = resolve_context_kv_cap(context_size, startup.max_kv_size);
    // Derive the disaggregated serving role from `--node-role` (#126 B2). The
    // role string was already validated in `resolve_distributed_startup`, so a
    // parse failure here falls back to the single-node `Hybrid` default rather
    // than erroring a second time. Absent `--node-role` is `Hybrid` (the
    // byte-identical single-node path).
    // Special-case "router" before the NodeRole parse: the router has no
    // NodeRole variant but does have a ServingMode variant.
    let serving_mode = if startup
        .node_role
        .as_deref()
        .map(|r| r.eq_ignore_ascii_case("router"))
        .unwrap_or(false)
    {
        crate::distributed::disaggregated::ServingMode::Router
    } else {
        startup
            .node_role
            .as_deref()
            .and_then(|role| role.parse::<NodeRole>().ok())
            .map(crate::distributed::disaggregated::ServingMode::from_node_role)
            .unwrap_or(crate::distributed::disaggregated::ServingMode::Hybrid)
    };

    ServerConfig {
        api_key,
        timeout_seconds: startup.timeout,
        model_alias: startup.model_alias.clone(),
        context_size,
        n_parallel: startup.n_parallel,
        enable_slots_endpoint: startup.enable_slots,
        enable_props_endpoint: startup.enable_props,
        enable_metrics_endpoint: startup.enable_metrics,
        default_temperature: startup.temperature,
        default_top_p: startup.top_p,
        default_top_k: startup.top_k,
        default_min_p: startup.min_p,
        default_repetition_penalty: startup.repeat_penalty,
        default_repetition_context_size: startup.repeat_last_n,
        default_max_tokens: resolve_default_max_tokens(startup.n_predict),
        default_seed: startup.seed,
        default_frequency_penalty: startup.frequency_penalty,
        default_presence_penalty: startup.presence_penalty,
        default_dry_multiplier: startup.dry_multiplier,
        default_dry_base: startup.dry_base,
        default_dry_allowed_length: startup.dry_allowed_length,
        default_dry_penalty_last_n: resolve_dry_penalty_last_n(startup.dry_penalty_last_n),
        draft_model_path: startup.draft_model_path.clone(),
        num_draft_tokens: startup.draft_max,
        // forward the speculative-decoding selector flags
        // verbatim. Reconciliation against the drafter `config.json`
        // and dispatch into `MtpGenerator` / `DFlashGenerator` / the
        // classic `SpeculativeGenerator` happens later inside the
        // continuous-batching worker, when both the drafter path and
        // the resolved kind are known.
        draft_kind: startup.draft_kind.clone(),
        draft_block_size: startup.draft_block_size,
        max_batch_size,
        max_queue_depth: startup.max_queue_depth,
        audio_queue_depth: startup.audio_queue_depth,
        audio_request_timeout_secs: startup.audio_request_timeout_secs,
        prefill_chunk_size: startup.prefill_chunk_size,
        enable_preemption: startup.enable_preemption,
        preemption_policy: parse_preemption_policy(&startup.preemption_policy),
        no_batch: startup.no_batch,
        max_batch_prefill: startup.max_batch_prefill.max(1),
        decode_storage_backend: startup
            .decode_storage_backend
            .unwrap_or_else(resolve_decode_storage_backend),
        pipeline_parallel_runtime: startup.pp_layers.as_ref().map(|layers| {
            PipelineParallelRuntimeConfig::InProcess {
                layers: layers.clone(),
                micro_batch_size: startup.pp_micro_batch_size.max(1),
            }
        }),
        remote_pipeline_stage: None,
        tensor_parallel,
        vision_cache_size: startup.vision_cache_size,
        lang_bias_config: startup.lang_bias_config.clone(),
        reasoning_budget: startup.reasoning_budget,
        chat_template_kwargs: startup.chat_template_kwargs.clone(),
        // wire the CLI/env-resolved policy through instead of
        // always using the compiled-in default.
        prompt_cache: startup.prompt_cache.clone(),
        // (B11): wire the resolved KV cache mode through so the
        // model worker can apply it when constructing per-sequence generators.
        kv_cache_mode: startup.kv_cache_mode,
        // wire the resolved batch KV quant config through so
        // the continuous-batching scheduler can apply per-layer modes
        // (with the last-layer skip) at sequence allocation time.
        batch_kv_quant: startup.batch_kv_quant,
        // Issue #57: forward the resolved per-slot context cap (optionally
        // tightened by `--max-kv-size`) so the scheduler can apply a head-trim
        // policy to plain `KVCache` instances. `None` means no explicit
        // context or max-KV bound was configured.
        max_kv_size,
        // forward the paged KV block-budget directive verbatim; the worker
        // resolves it to a concrete block count once the model is loaded.
        kv_cache_budget: startup.kv_cache_budget,
        // forward the experimental VLM prefix-cache toggle (#124 step c).
        enable_vlm_prefix_cache: startup.enable_vlm_prefix_cache,
        // forward the validated CORS allow-list (#244); `None` keeps permissive.
        cors_allowed_origins: startup.cors_allowed_origins.clone(),
        // disaggregated serving role derived from `--node-role` (#126 B2).
        serving_mode,
        // disaggregated serving-role network addresses (#126 B3b2a): the
        // worker uses these to bind its role transport and reach its handoff
        // peer when `serving_mode` is non-hybrid.
        prefill_peers: startup.prefill_peers.clone(),
        decode_peers: startup.decode_peers.clone(),
        serving_bind: startup.serving_bind,
        // serve-level diffusion knobs (#217 phase 3); consumed only by the
        // DiffusionGemma worker loop.
        max_denoising_steps: startup.max_denoising_steps,
        diffusion_sampler: startup.diffusion_sampler.clone(),
        diffusion_threshold: startup.diffusion_threshold,
    }
}

fn initialize_server_logging(startup: &ServerStartupConfig) -> Result<()> {
    if startup.log_disable {
        return Ok(());
    }

    let filter = if startup.verbose { "debug" } else { "info" };
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter));

    if let Some(ref log_path) = startup.log_file {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("Failed to open log file: {:?}", log_path))?;
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_writer(file)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }

    Ok(())
}

fn warmup_model(model_provider: &ModelProvider) -> Result<()> {
    model_provider.generate(
        "Hello".to_string(),
        ServerGenerateOptions {
            max_tokens: 1,
            sampling: SamplingConfig::greedy(),
            stop_sequences: None,
            priority: crate::server::batch::RequestPriority::Normal,
            logprobs: Default::default(),
            reasoning_budget: Default::default(),
            // warmup prompt is the raw literal "Hello", not a
            // chat-templated prompt with `<think>\n` priming, so treat the
            // first token as not-yet-in-block.
            thinking_enter_block_on_start: false,
            // Warmup bypasses the prompt cache entirely — a single literal
            // "Hello" is not worth donating back.
            prompt_cache_ctx: None,
            // Warmup never asks for structured output.
            structured: None,
        },
    )?;
    Ok(())
}

#[cfg_attr(not(test), allow(dead_code))]
fn validate_tensor_parallel_startup(startup: &ServerStartupConfig) -> Result<()> {
    resolve_tensor_parallel_runtime_support(startup).map(|_| ())
}

fn resolve_tensor_parallel_runtime_support(
    startup: &ServerStartupConfig,
) -> Result<crate::distributed::TensorParallelRuntimeSupport> {
    let shard_config = shard_config_from_cli(
        startup.tp_size,
        &startup.tp_moe_mode,
        &startup.tp_embedding_mode,
        &startup.tp_lm_head_mode,
    )?;
    let summary = resolve_model_shard_plan(&startup.model_path, shard_config)?;
    if summary.shard_config.tp_size > 1 {
        tracing::info!("Tensor parallel request: {}", summary.summary_line());
    }
    validate_supported_runtime(
        &startup.model_path,
        summary.shard_config.clone(),
        startup.adapter_path.as_deref(),
    )
}

fn validate_pipeline_parallel_startup(startup: &ServerStartupConfig) -> Result<()> {
    anyhow::ensure!(
        startup.pp_micro_batch_size > 0,
        "--pp-micro-batch-size must be greater than 0"
    );
    let Some(pp_layers) = startup.pp_layers.as_deref() else {
        return Ok(());
    };

    anyhow::ensure!(
        !pp_layers.trim().is_empty(),
        "--pp-layers must not be empty when provided"
    );
    // LoRA adapter composition with PP is supported for in-process stages.
    // Single-adapter only; multi-adapter stacking and runtime hot-swap are
    // out of scope for v1.
    anyhow::ensure!(
        startup.draft_model_path.is_none(),
        "Server pipeline parallelism does not support speculative decoding yet"
    );
    anyhow::ensure!(
        startup.tp_size == 1,
        "Server pipeline parallelism does not support tensor parallelism yet"
    );
    anyhow::ensure!(
        !startup.no_batch,
        "Server pipeline parallelism requires the batch scheduler; remove --no-batch"
    );
    // Announce stage-executor family capabilities for operator visibility
    // and to document the exact set a cluster handshake would advertise.
    // Emitting this as a single comma-separated line keeps log parsers happy
    // and makes cross-version mismatches trivially greppable.
    let family_names: Vec<&'static str> = crate::distributed::pipeline::supported_families()
        .iter()
        .map(|f| f.name())
        .collect();
    tracing::info!(
        "Pipeline-parallel stage-executor families advertised: {}",
        family_names.join(",")
    );

    crate::distributed::pipeline::resolve_in_process_pipeline_num_layers(&startup.model_path)
        .map(|_| ())
}

fn log_endpoints(startup: &ServerStartupConfig, addr: &str) {
    tracing::info!("Starting mlxcel server on {}", addr);
    tracing::info!("Endpoints:");
    tracing::info!("  POST /v1/chat/completions  - OpenAI chat completions");
    tracing::info!("  POST /v1/completions       - OpenAI text completions");
    tracing::info!("  GET  /v1/models            - List models");
    tracing::info!("  POST /completion           - llama-server native completion");
    tracing::info!("  POST /tokenize             - Tokenize text");
    tracing::info!("  POST /detokenize           - Detokenize tokens");
    if startup.enable_props {
        tracing::info!("  GET  /props                - Server properties");
    }
    if startup.enable_slots {
        tracing::info!("  GET  /slots                - Slot status");
    }
    tracing::info!("  GET  /health               - Health check");
}

async fn serve_unix_socket(startup: &ServerStartupConfig, app: axum::Router) -> Result<()> {
    let socket_path = Path::new(&startup.host);

    if socket_path.exists() {
        std::fs::remove_file(socket_path)
            .with_context(|| format!("Failed to remove stale socket: {:?}", socket_path))?;
    }
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create socket directory: {:?}", parent))?;
    }

    log_endpoints(startup, &startup.host);
    let listener = tokio::net::UnixListener::bind(socket_path)
        .with_context(|| format!("Failed to bind Unix socket: {:?}", socket_path))?;

    loop {
        let (socket, _addr) = listener.accept().await?;
        let app = app.clone();
        tokio::spawn(async move {
            let socket = hyper_util::rt::TokioIo::new(socket);
            let hyper_service =
                hyper::service::service_fn(move |request| app.clone().call(request));
            if let Err(err) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .serve_connection(socket, hyper_service)
                    .await
            {
                tracing::debug!("Unix socket connection error: {}", err);
            }
        });
    }
}

async fn serve_tcp(startup: &ServerStartupConfig, app: axum::Router) -> Result<()> {
    let addr = format!("{}:{}", startup.host, startup.port);
    log_endpoints(startup, &addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn parse_startup_listen_addr(startup: &ServerStartupConfig) -> Result<SocketAddr> {
    format!("{}:{}", startup.host, startup.port)
        .parse()
        .context("failed to parse local listen address for distributed config")
}

fn resolve_remote_pipeline_topology(
    startup: &ServerStartupConfig,
    cluster_config: &ClusterConfig,
    local_id: &str,
) -> Result<(
    Option<PipelineParallelRuntimeConfig>,
    Option<RemoteStageServiceConfig>,
)> {
    let pipeline_depth = cluster_config.cluster.pipeline_parallel_size;
    if pipeline_depth <= 1 {
        return Ok((None, None));
    }
    anyhow::ensure!(
        startup.pp_layers.is_none(),
        "remote pipeline startup is configured via cluster topology; remove --pp-layers"
    );
    anyhow::ensure!(
        startup.adapter_path.is_none(),
        "remote pipeline startup does not support adapter loading yet"
    );
    anyhow::ensure!(
        startup.draft_model_path.is_none(),
        "remote pipeline startup does not support speculative decoding yet"
    );
    anyhow::ensure!(
        startup.tp_size == 1,
        "remote pipeline startup does not support tensor parallelism yet"
    );
    anyhow::ensure!(
        !startup.no_batch,
        "remote pipeline startup requires the batch scheduler; remove --no-batch"
    );
    anyhow::ensure!(
        startup.node_id.is_some(),
        "remote pipeline startup requires --node-id so the local cluster node can be identified"
    );

    let local_node = cluster_config.find_node(local_id).ok_or_else(|| {
        anyhow::anyhow!("local node '{local_id}' was not found in cluster config")
    })?;
    let pipeline_nodes = cluster_config.pipeline_stage_nodes();
    anyhow::ensure!(
        !pipeline_nodes.is_empty(),
        "cluster config must define pipeline_stage nodes when pipeline_parallel_size > 1"
    );

    if local_node.role == NodeRole::PipelineStage {
        let stage_index = local_node.stage.ok_or_else(|| {
            anyhow::anyhow!(
                "pipeline stage node '{}' is missing required 'stage' index",
                local_node.id
            )
        })?;
        let num_layers = resolve_in_process_pipeline_num_layers(&startup.model_path)?;
        let (assignments, report) =
            crate::distributed::pipeline::resolve_in_process_stage_assignments_for_model(
                &startup.model_path,
                num_layers,
                Some(pipeline_depth as usize),
                None,
            )?;
        crate::distributed::pipeline::log_partition_quality(&report);
        let stage_assignment = assignments
            .into_iter()
            .find(|assignment| assignment.stage_index == stage_index as usize)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "failed to resolve stage assignment for stage {} of {}",
                    stage_index,
                    pipeline_depth
                )
            })?;
        let upstream_peer = stage_index
            .checked_sub(1)
            .and_then(|idx| cluster_config.pipeline_stage_node(idx))
            .map(|node| node.address.to_string());
        let downstream_peer = cluster_config
            .pipeline_stage_node(stage_index + 1)
            .map(|node| node.address.to_string());
        return Ok((
            None,
            Some(RemoteStageServiceConfig {
                model_dir: startup.model_path.clone(),
                bind_address: local_node.address.to_string(),
                transport_backend: cluster_config.cluster.transport_backend,
                stage_assignment,
                num_stages: pipeline_depth,
                upstream_peer,
                downstream_peer,
            }),
        ));
    }

    anyhow::ensure!(
        local_node.address != parse_startup_listen_addr(startup)?,
        "remote pipeline coordinator control address {} conflicts with HTTP listen address {}; assign a distinct cluster node address/port for control traffic",
        local_node.address,
        parse_startup_listen_addr(startup)?
    );
    let stage_peers = pipeline_nodes
        .into_iter()
        .map(|node| node.address.to_string())
        .collect::<Vec<_>>();
    Ok((
        Some(PipelineParallelRuntimeConfig::RemoteCoordinator(
            crate::distributed::pipeline::RemotePipelineRuntimeConfig {
                stage_peers,
                transport_backend: cluster_config.cluster.transport_backend,
                bind_address: local_node.address.to_string(),
                stage_timeout: std::time::Duration::from_secs(30),
            },
        )),
        None,
    ))
}

/// Parse the discovery mode string, falling back to an actionable error so the
/// operator sees what was accepted.
fn parse_discovery_mode(raw: &str) -> Result<ClusterDiscoveryMode> {
    raw.parse::<ClusterDiscoveryMode>()
        .with_context(|| format!("failed to parse --cluster-discovery={raw}"))
}

/// Derive the default output path for the emitted cluster TOML.
fn default_cluster_config_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".mlxcel")
        .join("cluster.toml")
}

/// Build a [`ClusterInitRequest`] from the coordinator-side CLI inputs.
fn build_cluster_init_request(startup: &ServerStartupConfig) -> Result<ClusterInitRequest> {
    let pp_stages = startup.pp_auto.ok_or_else(|| {
        anyhow::anyhow!("internal: build_cluster_init_request called without --pp-auto")
    })?;
    anyhow::ensure!(
        pp_stages >= 2,
        "--pp-auto requires N >= 2; pass N={pp_stages} instead of a single-node run"
    );
    anyhow::ensure!(
        startup.distributed_config.is_none(),
        "--pp-auto and --distributed-config are mutually exclusive; remove one or the other"
    );
    anyhow::ensure!(
        startup.pp_layers.is_none(),
        "--pp-auto replaces --pp-layers; remove --pp-layers when using --pp-auto"
    );
    anyhow::ensure!(
        !startup.pp_peer,
        "--pp-auto (coordinator) and --pp-peer are mutually exclusive"
    );

    let discovery = parse_discovery_mode(&startup.cluster_discovery)?;
    let http_addr = parse_startup_listen_addr(startup)?;
    let control_addr = startup.cluster_control_addr.unwrap_or_else(|| {
        SocketAddr::new(
            http_addr.ip(),
            crate::distributed::DEFAULT_CONTROL_BASE_PORT,
        )
    });
    let discovery_port = startup
        .cluster_discovery_port
        .unwrap_or(crate::distributed::DEFAULT_DISCOVERY_PORT);
    let data_port_base = control_addr.port().saturating_add(1).max(1);
    let cluster_name = startup
        .cluster_name
        .clone()
        .unwrap_or_else(|| "mlxcel-cluster".to_string());

    Ok(ClusterInitRequest {
        pp_stages,
        cluster_name,
        transport_backend: TransportBackend::Tcp,
        discovery,
        discovery_timeout: None,
        discovery_port,
        coordinator_http_addr: http_addr,
        coordinator_control_addr: control_addr,
        static_peers: startup.cluster_peers.clone(),
        data_port_base,
        output_toml_path: startup.cluster_config_out.clone(),
    })
}

/// Run the zero-config bring-up path: resolve peers (static or
/// mDNS broadcast), emit a deterministic cluster TOML, and rewrite the
/// startup config so the downstream distributed resolution path sees a
/// normal `distributed_config` + `node_id` tuple.
///
/// Returns `Some(plan)` when the path was taken so the caller can honour
/// `--dry-run`. Returns `None` for a no-op when neither `--pp-auto` nor
/// `--pp-peer` is set.
async fn run_zero_config_bring_up(
    startup: &mut ServerStartupConfig,
) -> Result<Option<ClusterInitPlan>> {
    if startup.pp_auto.is_some() {
        let request = build_cluster_init_request(startup)?;
        let resolved_peers = crate::distributed::discover_peers(
            request.discovery,
            &request.cluster_name,
            request.coordinator_control_addr.ip(),
            request.discovery_port,
            &request.static_peers,
            request.pp_stages as usize,
            request
                .discovery_timeout
                .unwrap_or(crate::distributed::DEFAULT_DISCOVERY_TIMEOUT),
        )
        .await?;
        let mut planned_request = request;
        planned_request.static_peers = resolved_peers;
        let plan = plan_cluster(&planned_request)?;

        let output_path = startup
            .cluster_config_out
            .clone()
            .unwrap_or_else(default_cluster_config_path);
        write_plan_toml(&plan, &output_path)?;
        tracing::info!(
            "Zero-config pipeline cluster ready: wrote {} ({} stage(s))",
            output_path.display(),
            plan.cluster.cluster.pipeline_parallel_size,
        );
        for line in plan.summary.lines() {
            tracing::info!("cluster> {line}");
        }

        startup.distributed_config = Some(output_path);
        if startup.node_id.is_none() {
            startup.node_id = Some("coordinator".to_string());
        }
        return Ok(Some(plan));
    }

    if startup.pp_peer {
        anyhow::ensure!(
            startup.distributed_config.is_some(),
            "--pp-peer currently requires --distributed-config pointing at the coordinator-emitted cluster TOML. \
             Future work will remove this once the coordinator push-assigns stages (follow-up)."
        );
        anyhow::ensure!(
            startup.node_id.is_some(),
            "--pp-peer requires --node-id so the coordinator can map this host to a pipeline stage"
        );
    }

    Ok(None)
}

/// Resolve distributed cluster configuration and any remote pipeline startup mode.
async fn resolve_distributed_startup(
    startup: &ServerStartupConfig,
) -> Result<ResolvedDistributedStartup> {
    if let Some(ref config_path) = startup.distributed_config {
        let cluster_config = ClusterConfig::from_file(config_path)?;
        let local_id = startup
            .node_id
            .as_deref()
            .or_else(|| cluster_config.nodes.first().map(|n| n.id.as_str()))
            .unwrap_or("node-0");
        let (pipeline_runtime, remote_stage_service) =
            resolve_remote_pipeline_topology(startup, &cluster_config, local_id)?;
        let registry = crate::distributed::initialize_distributed(
            &cluster_config,
            local_id,
            std::time::Duration::from_secs(5),
        )
        .await?;
        return Ok(ResolvedDistributedStartup {
            _node_registry: Some(registry),
            pipeline_runtime,
            remote_stage_service,
        });
    }

    // CLI shorthand remains non-PP-only; remote PP requires an explicit cluster config.
    if let Some(ref role_str) = startup.node_role {
        // The "router" role is not a cluster inference role; skip distributed
        // cluster init and let the router startup path handle it.
        if role_str.eq_ignore_ascii_case("router") {
            return Ok(ResolvedDistributedStartup {
                _node_registry: None,
                pipeline_runtime: None,
                remote_stage_service: None,
            });
        }
        let role: NodeRole = role_str.parse()?;
        let node_id = startup
            .node_id
            .clone()
            .unwrap_or_else(|| "node-0".to_string());
        let listen_addr = parse_startup_listen_addr(startup)?;
        let cluster_config =
            ClusterConfig::from_cli(node_id.clone(), listen_addr, role, startup.peers.clone());
        let registry = crate::distributed::initialize_distributed(
            &cluster_config,
            &node_id,
            std::time::Duration::from_secs(5),
        )
        .await?;
        return Ok(ResolvedDistributedStartup {
            _node_registry: Some(registry),
            pipeline_runtime: None,
            remote_stage_service: None,
        });
    }

    Ok(ResolvedDistributedStartup {
        _node_registry: None,
        pipeline_runtime: None,
        remote_stage_service: None,
    })
}

async fn serve_remote_pipeline_stage(service_config: RemoteStageServiceConfig) -> Result<()> {
    let bind_address = service_config.bind_address.clone();
    let stage_index = service_config.stage_assignment.stage_index;
    let num_stages = service_config.num_stages;
    let upstream = service_config.upstream_peer.clone();
    let downstream = service_config.downstream_peer.clone();
    let handle = RemoteStageServiceHandle::spawn(service_config)?;
    tracing::info!(
        "Starting remote pipeline stage service on {} (stage={}/{}, upstream={:?}, downstream={:?})",
        bind_address,
        stage_index,
        num_stages,
        upstream,
        downstream
    );
    tokio::signal::ctrl_c()
        .await
        .context("failed to wait for shutdown signal")?;
    tracing::info!(
        "Shutting down remote pipeline stage service on {}",
        handle.local_addr()
    );
    handle.shutdown()
}

/// Install the configured `--surgery <FILE>` YAML pipeline into the
/// process-wide active-pipeline slot, returning early with a friendly
/// `anyhow::Error` on malformed input.
///
/// Called once during [`start_server`] before any model worker thread
/// is spawned. When `surgery_config_path` is `None`, this is a no-op
/// and the server runs on the bit-exact baseline load path.
#[cfg(feature = "surgery")]
fn install_surgery_pipeline_for_server(startup: &ServerStartupConfig) -> Result<()> {
    let Some(ref path) = startup.surgery_config_path else {
        return Ok(());
    };
    if !path.exists() {
        anyhow::bail!("--surgery: config file does not exist: {}", path.display());
    }
    let pipeline = crate::surgery::load_pipeline_from_file(path)
        .map_err(|e| anyhow::anyhow!("--surgery: {e}"))?;
    tracing::info!(
        path = %path.display(),
        ops = pipeline.len(),
        "Surgery: installed weight-load pipeline"
    );
    crate::surgery::set_active_pipeline(Some(std::sync::Arc::new(pipeline)));
    Ok(())
}

/// Start the server with the given startup configuration.
///
/// Shared entry point used by both `mlxcel serve` and `mlxcel-server`.
pub async fn start_server(mut startup: ServerStartupConfig) -> Result<()> {
    initialize_server_logging(&startup)?;
    super::media::configure_image_input_limits(super::media::ImageInputLimits {
        max_payload_bytes: startup.max_image_payload_size,
        max_images_per_request: startup.max_images_per_request,
        max_width: startup.max_image_width,
        max_height: startup.max_image_height,
        max_decode_alloc_bytes: startup.max_image_decode_alloc_bytes,
    });

    // Axis A weight-load surgery. Install the
    // pipeline *before* worker startup so the spawned model loader
    // thread observes it through the active-pipeline snapshot. When
    // `--surgery` is absent this is a no-op and the load path stays
    // bit-exact with the earlier baseline.
    #[cfg(feature = "surgery")]
    install_surgery_pipeline_for_server(&startup)?;

    // Zero-config multi-machine pipeline bring-up. Runs before
    // the tensor-parallel / pipeline-parallel validators so the emitted TOML
    // passes through the existing distributed resolution path unchanged.
    let zero_config_plan = run_zero_config_bring_up(&mut startup).await?;
    if startup.dry_run {
        if let Some(plan) = zero_config_plan.as_ref() {
            // Print the topology summary to stdout so CI gates and operators
            // can consume it without scraping logs.
            println!("{}", plan.summary);
            println!(
                "Emitted cluster TOML at: {}",
                startup
                    .distributed_config
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "<not persisted>".to_string())
            );
            return Ok(());
        }
        anyhow::bail!("--dry-run was requested but --pp-auto was not provided; nothing to plan");
    }

    validate_parallel_context_startup(&startup)?;
    validate_pipeline_parallel_startup(&startup)?;
    let tp_support = resolve_tensor_parallel_runtime_support(&startup)?;

    if startup.ubatch_size_provided {
        tracing::info!("--ubatch-size is not applicable on Apple Silicon unified memory; ignored");
    }
    if startup.batch_size_conflict {
        tracing::warn!(
            "--batch-size and --prefill-chunk-size both provided; \
             --prefill-chunk-size takes precedence"
        );
    }

    // Log hardware capabilities (detection is cached; subsequent calls are free).
    {
        let hw = mlxcel_core::hardware::get_hardware();
        tracing::debug!(
            silicon_gen = %hw.silicon_gen,
            gpu_cores = hw.gpu_core_count,
            memory_gb = hw.unified_memory_gb,
            bandwidth_gbps = hw.memory_bandwidth_gbps,
            neural_accelerator = hw.has_neural_accelerator,
            metal_version = hw.metal_version,
            macos_supports_na = hw.macos_supports_na,
            "Hardware capabilities detected"
        );
    }

    let runtime = crate::initialize_runtime();
    if let Some(invalid) = runtime.invalid_device_override.as_deref() {
        tracing::warn!(
            value = invalid,
            "Ignoring invalid MLXCEL_DEVICE override; using gpu"
        );
    }
    tracing::info!("Runtime device: {}", runtime.device);
    if let Some(max_memory) = runtime.wired_limit_bytes {
        tracing::info!(
            "Wired memory limit: {:.1} GB",
            max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else if runtime.device == crate::RuntimeDevice::Gpu {
        let max_memory = mlxcel_core::gpu_max_memory_size();
        tracing::info!(
            "GPU memory: {:.1} GB (no wired limit)",
            max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // -- Distributed mode initialization --
    let distributed = resolve_distributed_startup(&startup).await?;

    let api_key = resolve_api_key(startup.api_key.clone(), startup.api_key_file.as_deref())?;
    let mut config = build_server_config(&startup, api_key);

    if config.pipeline_parallel_runtime.is_some() && distributed.pipeline_runtime.is_some() {
        anyhow::bail!(
            "server startup resolved both in-process and remote pipeline runtimes; remove either --pp-layers or the remote pipeline cluster topology"
        );
    }
    config.pipeline_parallel_runtime = distributed
        .pipeline_runtime
        .or(config.pipeline_parallel_runtime.take());
    config.no_batch |= tp_support.force_no_batch;
    if config.tensor_parallel.tp_size > 1 {
        if config.no_batch {
            tracing::info!(
                "Tensor parallel runtime enabled; using legacy sequential worker for this runtime"
            );
        } else {
            tracing::info!("Tensor parallel runtime enabled; batch scheduler remains active");
        }
    }
    if let Some(ref pipeline_runtime) = config.pipeline_parallel_runtime {
        tracing::info!(
            "Server pipeline runtime enabled ({})",
            pipeline_runtime.describe()
        );
    }
    if let Some(elastic_cfg) = resolve_elastic_pp_config(&startup) {
        if config.pipeline_parallel_runtime.is_none() {
            tracing::warn!(
                "--enable-elastic-pp was set but no pipeline-parallel runtime is active; \
                 elastic repartitioning requires PP (ignore if launching as a peer)"
            );
        } else {
            tracing::info!(
                drain_timeout_s = elastic_cfg.drain_timeout.as_secs(),
                cool_down_s = elastic_cfg.cool_down.as_secs(),
                pressure_fraction = elastic_cfg.trigger_memory_fraction,
                "Elastic pipeline repartitioning enabled (experimental, see \
                 docs_internal/architecture/elastic-pipeline-repartition-20260418.md)"
            );
        }
    }
    if let Some(service_config) = distributed.remote_stage_service {
        return serve_remote_pipeline_stage(service_config).await;
    }
    let mut chat_template = resolve_chat_template(
        startup.chat_template.as_deref(),
        startup.chat_template_file.as_deref(),
        &startup.model_path,
    )?;
    let tokenizer = crate::tokenizer::load_tokenizer(&startup.model_path)?;

    // align the chat-template `enable_thinking` Jinja kwarg
    // default with upstream `TokenizerWrapper.apply_chat_template`'s
    // `enable_thinking=self.has_thinking` behavior. When the underlying
    // tokenizer recognizes a think marker pair (single-token `<think>` /
    // `</think>`, single-token `<longcat_think>` variants, or multi-token
    // `<|channel>thought` / `<channel|>` for Gemma 4 and friends), the
    // server-side default flips to `true` so a request that does not set
    // `chat_template_kwargs.enable_thinking` still sees thinking enabled
    // by default. Per-request kwargs and the existing CLI/env defaults
    // (`--chat-template-kwargs`, `LLAMA_ARG_CHAT_TEMPLATE_KWARGS`)
    // continue to win on conflict via `merge_server_and_request`.
    let thinking_markers = tokenizer.infer_thinking_markers();
    if thinking_markers.has_thinking() {
        tracing::info!(
            think_start = ?thinking_markers.think_start,
            think_end = ?thinking_markers.think_end,
            think_start_tokens_len = thinking_markers
                .think_start_tokens
                .as_ref()
                .map(Vec::len)
                .unwrap_or(0),
            think_end_tokens_len = thinking_markers
                .think_end_tokens
                .as_ref()
                .map(Vec::len)
                .unwrap_or(0),
            "Tokenizer recognizes a think marker pair; defaulting \
             chat_template kwarg `enable_thinking=true` (\
             upstream PR #1114)"
        );
        chat_template.set_default_enable_thinking(true);
    }

    // If the serving role is "router", start the lightweight HTTP router front-end
    // and return without loading model weights.
    if config.serving_mode == crate::distributed::disaggregated::ServingMode::Router {
        let addr = config.serving_bind.ok_or_else(|| {
            anyhow::anyhow!("the router (--node-role router) requires --serving-bind <host:port>")
        })?;
        let transport = std::sync::Arc::new(
            crate::distributed::tcp_transport::TcpTransport::bind(
                crate::distributed::tcp_transport::TcpTransportConfig {
                    bind_address: addr.to_string(),
                    ..Default::default()
                },
            )
            .await?,
        );
        let reply_to = crate::distributed::transport::Transport::local_addr(transport.as_ref())?;
        let config_arc = std::sync::Arc::new(config.clone());
        let chat_template_arc = std::sync::Arc::new(chat_template);
        let tokenizer_arc = std::sync::Arc::new(tokenizer);
        let state = std::sync::Arc::new(crate::server::router_front::RouterState::build(
            config_arc,
            transport,
            reply_to,
            chat_template_arc,
            tokenizer_arc,
        )?);
        crate::server::router_front::spawn_result_demux(state.clone());
        crate::server::router_front::spawn_health_monitor(state.clone());
        let app = crate::server::router_front::create_router_app(state);
        tracing::info!(
            host = %startup.host,
            port = startup.port,
            "Starting disaggregated router front-end"
        );
        return serve_tcp(&startup, app).await;
    }

    // Create shared batch metrics and observability that both ModelProvider
    // and AppState read/write.
    let batch_metrics = Arc::new(BatchMetrics::new());
    let batch_observability = Arc::new(BatchObservability::new());

    // hybrid SSM / linear-attention models cannot use APC because
    // their recurrent state cannot be reconstructed from a token-prefix hash.
    // Detect by reading model_type / architectures from config.json and
    // force-disable APC at runtime (the whole-prefix prompt cache is still
    // safe and stays enabled).
    if config.prompt_cache.apc.enabled
        && let Ok(Some(family)) =
            crate::server::prompt_cache::detect_hybrid_ssm_from_path(&startup.model_path)
    {
        tracing::warn!(
            model_type = %family,
            "Detected hybrid SSM / linear-attention model family ({family}); \
             auto-disabling APC because recurrent state cannot decompose \
             into hashable blocks. Whole-prefix prompt cache is unaffected."
        );
        config.prompt_cache.apc.enabled = false;
    }

    // Cross-request prompt-prefix KV cache store.
    // Gated on the config flag so a disabled policy reserves zero memory.
    // wire BatchMetrics into the store so hits/misses/evictions
    // are counted and exposed via /metrics.
    let prompt_cache_store = if config.prompt_cache.is_enabled() {
        let cache_metrics = Arc::new(crate::server::state::BatchMetricsCacheAdapter::new(
            batch_metrics.clone(),
        ));
        let store = Arc::new(crate::server::prompt_cache::PromptCacheStore::with_metrics(
            config.prompt_cache.clone(),
            cache_metrics,
        ));
        tracing::info!(
            capacity_bytes = config.prompt_cache.capacity_bytes,
            max_entries = config.prompt_cache.max_entries,
            ttl_seconds = config.prompt_cache.ttl.as_secs(),
            snapshot_capacity_bytes = config.prompt_cache.snapshot_capacity_bytes,
            snapshot_max_entries = config.prompt_cache.snapshot_max_entries,
            snapshot_ttl_seconds = config.prompt_cache.snapshot_ttl.as_secs(),
            min_prefix_tokens = config.prompt_cache.min_prefix_tokens,
            apc_enabled = config.prompt_cache.apc.enabled,
            apc_block_size = config.prompt_cache.apc.block_size,
            apc_hash = %config.prompt_cache.apc.hash,
            "Prompt-prefix cache store enabled (+ APC, snapshots)"
        );
        Some(store)
    } else {
        tracing::debug!("Prompt-prefix KV cache store disabled by config");
        None
    };

    // `--timeout` is validated inside `new_with_server_config_and_prompt_cache` and
    // the resolved `Duration` is stashed on `ModelProvider`, where it flows into the drain loops.
    // A zero value triggers a logged warning and falls back to the 300 s default.
    let model_provider = Arc::new(ModelProvider::new_with_server_config_and_prompt_cache(
        startup.model_path.clone(),
        startup.adapter_path.clone(),
        &config,
        prompt_cache_store.clone(),
        batch_metrics.clone(),
        batch_observability.clone(),
    )?);

    if startup.warmup {
        tracing::info!("Warming up model...");
        match warmup_model(model_provider.as_ref()) {
            Ok(()) => tracing::info!("Warmup complete"),
            Err(err) => tracing::warn!("Warmup failed (non-fatal): {}", err),
        }
    }

    // Warn if operator requested a distinct /metrics port — not yet wired.
    if let Some(requested) = startup.metrics_port
        && requested != startup.port
    {
        tracing::warn!(
            "--metrics-port {} requested, but the /metrics endpoint is \
             multiplexed onto the main HTTP port ({}). A separate metrics \
             listener is deferred to a follow-up rollout.",
            requested,
            startup.port
        );
    }

    // Construct the chrome-tracing writer when --debug-pp-trace is set.
    let pp_tracer = startup.debug_pp_trace.as_ref().map(|path| {
        tracing::info!(
            path = %path.display(),
            "Enabling pipeline scheduler chrome-tracing (--debug-pp-trace)"
        );
        Arc::new(crate::distributed::pipeline::PpTracer::new(path.clone()))
    });

    // detect static media-input capabilities once at startup so
    // the chat handler can short-circuit unsupported requests with a 400.
    let media_support = detect_model_media_support(&startup.model_path);

    // hardening: scan the operator-provided
    // `MLXCEL_VIDEO_DIR_ALLOWLIST` directories for world/group-writable
    // entries. The technical TOCTOU race (attacker swaps the file between
    // canonicalize and ffmpeg open) is now closed's
    // fd-passing fix in `media::extract_chat_video_paths_with_allowlist`,
    // but a loose-mode allowlist directory still violates operator-policy
    // hygiene and can re-enable the race if a future ffmpeg version
    // interprets `/dev/fd/N` differently. We keep the warning as
    // defence-in-depth.
    warn_on_insecure_video_allowlist();

    // build the Responses-API stores from the resolved limits.
    // `max_entries = 0` disables the store entirely; otherwise build with
    // the configured TTL (a TTL of 0 means "no TTL" which we map to a
    // very large duration so the sweep is a no-op).
    let responses_store = if startup.responses_store_max_entries == 0 {
        None
    } else {
        let ttl = if startup.responses_store_ttl_secs == 0 {
            std::time::Duration::from_secs(u64::MAX / 2)
        } else {
            std::time::Duration::from_secs(startup.responses_store_ttl_secs)
        };
        Some(Arc::new(super::responses_store::ResponsesStore::new(
            super::responses_store::ResponsesStoreConfig {
                max_entries: startup.responses_store_max_entries,
                ttl,
            },
        )))
    };
    let conversation_store = if startup.conversation_store_max_entries == 0 {
        None
    } else {
        let ttl = if startup.conversation_store_ttl_secs == 0 {
            std::time::Duration::from_secs(u64::MAX / 2)
        } else {
            std::time::Duration::from_secs(startup.conversation_store_ttl_secs)
        };
        Some(Arc::new(super::conversation_store::ConversationStore::new(
            super::conversation_store::ConversationStoreConfig {
                max_entries: startup.conversation_store_max_entries,
                ttl,
            },
        )))
    };

    // Speech-to-text wiring: when the loaded checkpoint is a Whisper-style ASR
    // model, populate the audio slot so `/v1/audio/transcriptions` and
    // `/v1/audio/translations` are served. `WhisperSttProvider::load` hands the
    // checkpoint path to its own dedicated worker thread, which loads the
    // weights and evaluates every transcription on that one stream-initialized
    // thread (MLX work is thread-affine); the load happens off this startup
    // thread. The chat ModelProvider load above is a no-op for this checkpoint
    // (the worker logs and returns), matching the single-model "speech-to-text
    // only" deployment; serving chat and STT simultaneously is out of scope.
    // Audio admission bounds (#373). Read from the in-scope `config` before it is
    // moved into `AppState`: a bounded command queue (queue depth) plus a
    // per-request reply timeout, shared by the STT and TTS workers. A `0` timeout
    // falls back to the default rather than timing out instantly; a `0` queue
    // depth is clamped at the channel boundary inside `AudioWorker::spawn`.
    let audio_queue_depth = config.audio_queue_depth;
    let audio_request_timeout =
        std::time::Duration::from_secs(if config.audio_request_timeout_secs == 0 {
            crate::server::config::DEFAULT_AUDIO_REQUEST_TIMEOUT_SECS
        } else {
            config.audio_request_timeout_secs
        });
    let audio_model: Option<Arc<dyn crate::server::audio_model::AudioModelProvider>> =
        match crate::models::get_model_type(&startup.model_path) {
            Ok(crate::models::ModelType::Whisper) => {
                tracing::info!(
                    "Detected Whisper speech-to-text checkpoint; loading audio model for \
                     /v1/audio/transcriptions and /v1/audio/translations"
                );
                match crate::server::whisper_stt::WhisperSttProvider::load(
                    &startup.model_path,
                    audio_queue_depth,
                    audio_request_timeout,
                ) {
                    Ok(provider) => Some(Arc::new(provider)),
                    Err(err) => {
                        tracing::error!("Failed to load Whisper speech-to-text model: {err}");
                        None
                    }
                }
            }
            Ok(crate::models::ModelType::Kokoro) => {
                tracing::info!(
                    "Detected Kokoro text-to-speech checkpoint; loading audio model for \
                     /v1/audio/speech"
                );
                match crate::server::kokoro_tts::KokoroTtsProvider::load(
                    &startup.model_path,
                    audio_queue_depth,
                    audio_request_timeout,
                ) {
                    Ok(provider) => Some(Arc::new(provider)),
                    Err(err) => {
                        tracing::error!("Failed to load Kokoro text-to-speech model: {err}");
                        None
                    }
                }
            }
            _ => None,
        };

    let state = AppState::with_observability(
        model_provider,
        config,
        chat_template,
        tokenizer,
        startup.model_path.clone(),
        batch_metrics,
        batch_observability,
    )
    .with_media_support(media_support)
    .with_pp_tracer(pp_tracer)
    .with_prompt_cache(prompt_cache_store)
    .with_responses_store(responses_store)
    .with_conversation_store(conversation_store)
    .with_audio_model(audio_model);
    let app = create_app(state);

    if startup.port == 0 {
        serve_unix_socket(&startup, app).await
    } else {
        serve_tcp(&startup, app).await
    }
}

#[cfg(test)]
#[path = "startup_tests.rs"]
mod tests;
