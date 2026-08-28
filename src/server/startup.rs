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
    // llama-server b10621 chat-template / reasoning / parsing settings
    // (issue #1447).
    /// Where a model's thoughts are reported (`--reasoning-format`).
    pub reasoning_format: crate::server::ReasoningFormat,
    /// `--skip-chat-parsing`: every parser off, everything in `content`.
    pub skip_chat_parsing: bool,
    /// `--no-prefill-assistant`: a trailing assistant message is complete.
    pub no_prefill_assistant: bool,
    /// `--reasoning-budget-message`, injected before the end-of-thinking tag
    /// when the budget is exhausted.
    pub reasoning_budget_message: Option<String>,

    // Model
    pub model_path: PathBuf,
    pub adapter_path: Option<PathBuf>,
    /// Served model id: the first `--alias` entry, or `None`.
    pub model_alias: Option<String>,
    /// Every `--alias` entry, primary first (issue #1434). Empty when the flag
    /// was not given.
    pub model_aliases: Vec<String>,

    // Network
    pub host: String,
    pub port: u16,

    // Auth (#1437). b10621 accepts a set of keys: every `--api-key`
    // occurrence contributes a comma-separated list and every
    // `--api-key-file` occurrence contributes one key per line, with the
    // environment bindings appending to the same set rather than replacing it.
    pub api_keys: Vec<String>,
    pub api_key_files: Vec<PathBuf>,

    // Limits
    pub n_parallel: usize,
    pub ctx_size: usize,
    pub n_predict: i32, // -1 = unlimited

    // HTTP transport (#1432). `http_timeout` is b10621's `--timeout`
    // (socket read/write budget); `decode_timeout` is the mlxcel-native
    // decode watchdog that spelling used to carry.
    pub http_timeout: u64,
    pub decode_timeout: u64,
    /// `--api-prefix`, already validated by
    /// [`crate::server::transport::resolve_api_prefix`]. Empty = no prefix.
    pub api_prefix: String,
    /// `--sse-ping-interval`; `None` = pings disabled (`-1` upstream).
    pub sse_ping_interval: Option<std::time::Duration>,
    /// `--threads-http`, already resolved to a concrete worker count by
    /// [`crate::server::transport::resolve_http_threads`].
    pub threads_http: usize,
    /// `--reuse-port`: also set `SO_REUSEPORT` on the listener.
    pub reuse_port: bool,
    /// `--ssl-cert-file` / `--ssl-key-file`, paired and validated by
    /// [`crate::server::tls::resolve_tls_paths`].
    pub tls: Option<crate::server::tls::TlsPaths>,

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

    /// `--spm-infill`: the Suffix/Prefix/Middle ordering for `POST /infill`
    /// (#1442). Forwarded to [`super::config::ServerConfig::spm_infill`].
    pub spm_infill: bool,

    /// `--embd-normalize` (#1452), `None` when unset.
    pub embd_normalize: Option<crate::embeddings::EmbdNormalize>,
    /// `--embeddings` / `--reranking` (#1452).
    pub embedding_serving_mode: crate::server::config::EmbeddingServingMode,
    /// `--pooling` (#1452), `None` when unset or when it named `rank`.
    pub pooling: Option<crate::embeddings::PoolingMode>,

    // Batch scheduling
    pub max_batch_size: Option<usize>,
    pub max_queue_depth: usize,
    /// Bound on the audio worker command queue (admission control). Forwarded to
    /// [`super::config::ServerConfig::audio_queue_depth`].
    pub audio_queue_depth: usize,
    /// Per-request reply timeout for the audio worker, in seconds. Forwarded to
    /// [`super::config::ServerConfig::audio_request_timeout_secs`].
    pub audio_request_timeout_secs: u64,
    /// `--embedding-model`: embedding checkpoint served on `/v1/embeddings`
    /// next to the chat model. Forwarded to
    /// [`super::config::ServerConfig::embedding_model_path`].
    pub embedding_model_path: Option<PathBuf>,
    /// `--embedding-batch-size`. Forwarded to
    /// [`super::config::ServerConfig::embedding_batch_size`].
    pub embedding_batch_size: usize,
    /// `--embedding-max-length`. Forwarded to
    /// [`super::config::ServerConfig::embedding_max_length`].
    pub embedding_max_length: Option<usize>,
    /// `--embedding-queue-depth`. Forwarded to
    /// [`super::config::ServerConfig::embedding_queue_depth`].
    pub embedding_queue_depth: usize,
    /// `--embedding-request-timeout-secs`. Forwarded to
    /// [`super::config::ServerConfig::embedding_request_timeout_secs`].
    pub embedding_request_timeout_secs: u64,
    /// `--reranker-model`: reranker checkpoint served on `/v1/rerank` next to
    /// the chat model. Forwarded to
    /// [`super::config::ServerConfig::reranker_model_path`].
    pub reranker_model_path: Option<PathBuf>,
    /// `--rerank-batch-size`. Forwarded to
    /// [`super::config::ServerConfig::rerank_batch_size`].
    pub rerank_batch_size: usize,
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
    /// Resolved CORS policy (#244 allow-list, b10621 flags in #1432). Built
    /// in [`super::ServerStartupInput::into_startup_config`] and forwarded to
    /// [`super::config::ServerConfig`].
    pub cors_policy: crate::server::CorsPolicy,
    /// #1011: `--prefill-grant-interval`. `None` keeps the env override /
    /// shipped default; `Some(0)` disables the fairness grant.
    pub prefill_grant_interval: Option<usize>,
    /// Preemption policy string from CLI (parsed into enum at build_server_config).
    pub preemption_policy: String,
    /// Force the legacy sequential worker, bypassing the batch scheduler.
    pub no_batch: bool,
    /// Maximum number of pending requests to batch together for prefill (default: 1).
    pub max_batch_prefill: usize,
    /// #715: `--max-batch-prefill-tokens` cap on the batched-prefill transient.
    /// `None` keeps the env override / derived default; `Some(0)` = uncapped.
    pub max_batch_prefill_tokens: Option<usize>,
    /// Decode-time storage backend requested by the CLI. `None` preserves the
    /// legacy `MLXCEL_SERVER_DECODE_STORAGE` env-var fallback.
    pub decode_storage_backend: Option<crate::server::DecodeStorageBackend>,

    // Warmup
    pub warmup: bool,

    // Default sampling
    pub temperature: f32,
    pub temperature_was_set: bool,
    pub top_k: i32,
    pub top_k_was_set: bool,
    pub top_p: f32,
    pub top_p_was_set: bool,
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

    /// Operator-facing notices explaining why [`Self::kv_cache_mode`] or
    /// [`Self::batch_kv_quant`] differ from what was requested (issue #1350).
    ///
    /// `ServerStartupInput::into_startup_config` resolves the requested mode
    /// against the model family, but it runs before
    /// [`initialize_server_logging`] installs a tracing subscriber, so a
    /// `tracing::warn!` there is written to nothing. The notices are carried
    /// here and emitted by [`start_server`] once logging exists. Empty when the
    /// requested mode was used as-is.
    pub kv_cache_mode_notices: Vec<String>,

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

    /// Resolved llama-server b10621 RoPE override (`--rope-scaling`,
    /// `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`), or `None` when
    /// the operator asked for the checkpoint's own rotation.
    ///
    /// Installed process-wide by [`start_server`] before the model worker
    /// loads anything, because every model family reads its own `config.json`
    /// inside its own loader and there is no argument to thread. See
    /// [`crate::models::rope_overrides`] for why that installation is verified
    /// afterwards rather than trusted.
    pub rope_override: Option<crate::models::rope_overrides::RopeRuntimeOverride>,
}

impl Default for ServerStartupConfig {
    fn default() -> Self {
        Self {
            reasoning_format: crate::server::ReasoningFormat::default(),
            skip_chat_parsing: false,
            no_prefill_assistant: false,
            reasoning_budget_message: None,
            model_path: PathBuf::new(),
            adapter_path: None,
            model_alias: None,
            model_aliases: Vec::new(),
            host: "127.0.0.1".to_string(),
            port: 8080,
            api_keys: Vec::new(),
            api_key_files: Vec::new(),
            // Serving-throughput default: 4 concurrent decode slots (#628).
            n_parallel: 4,
            ctx_size: 0,
            n_predict: -1,
            http_timeout: crate::server::transport::DEFAULT_HTTP_TIMEOUT_SECS,
            decode_timeout: crate::server::transport::DEFAULT_DECODE_TIMEOUT_SECS,
            api_prefix: String::new(),
            sse_ping_interval: Some(std::time::Duration::from_secs(
                crate::server::transport::DEFAULT_SSE_PING_INTERVAL_SECS as u64,
            )),
            threads_http: crate::server::transport::resolve_http_threads(
                crate::server::transport::DEFAULT_THREADS_HTTP,
                4,
            ),
            reuse_port: false,
            tls: None,
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
            embedding_model_path: None,
            embedding_batch_size: crate::server::config::DEFAULT_EMBEDDING_BATCH_SIZE,
            embedding_max_length: None,
            embedding_queue_depth: crate::server::config::DEFAULT_EMBEDDING_QUEUE_DEPTH,
            embedding_request_timeout_secs:
                crate::server::config::DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS,
            reranker_model_path: None,
            rerank_batch_size: crate::server::config::DEFAULT_RERANK_BATCH_SIZE,
            prefill_chunk_size: 512,
            batch_size_conflict: false,
            ubatch_size_provided: false,
            enable_preemption: false,
            enable_vlm_prefix_cache: false,
            cors_policy: crate::server::CorsPolicy::default(),
            // #1011: unset -> scheduler resolves the env override / default.
            prefill_grant_interval: None,
            preemption_policy: "longest-first".to_string(),
            no_batch: false,
            // Serving-throughput default: batched prefill up to 4 requests (#628).
            max_batch_prefill: 4,
            // #715: unset -> scheduler derives the token budget.
            max_batch_prefill_tokens: None,
            decode_storage_backend: None,
            chat_template: None,
            chat_template_file: None,
            enable_slots: true,
            enable_props: false,
            spm_infill: false,
            embd_normalize: None,
            embedding_serving_mode: crate::server::config::EmbeddingServingMode::Any,
            pooling: None,
            enable_metrics: false,
            warmup: true,
            temperature: 0.8,
            temperature_was_set: false,
            top_k: 40,
            top_k_was_set: false,
            top_p: 0.95,
            top_p_was_set: false,
            min_p: 0.05,
            seed: None,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
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
            kv_cache_mode_notices: Vec::new(),
            batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig::default(),
            max_kv_size: None,
            // Serving-throughput default guard (#628): `auto` paged KV budget
            // pairs with the batched-decode default. Disable with
            // `--kv-cache-budget none`.
            kv_cache_budget: Some(crate::memory_estimate::PagedBudgetDirective::Auto),
            responses_store_max_entries: 1024,
            responses_store_ttl_secs: 3600,
            conversation_store_max_entries: 256,
            conversation_store_ttl_secs: 3600,
            #[cfg(feature = "surgery")]
            surgery_config_path: None,
            max_denoising_steps: None,
            diffusion_sampler: "entropy-bound".to_string(),
            diffusion_threshold: 0.9,
            rope_override: None,
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

/// Resolve the per-request default token budget applied when a request omits
/// `max_tokens` / `n_predict`.
///
/// llama-server parity (issue #476): a negative `--n-predict` (`-1`, the
/// default) means "generate until the context window is full". It resolves to
/// the effective per-slot context window — an explicit `--ctx-size` (already
/// divided across slots into `per_slot_context_size`) when set, otherwise the
/// model's `max_position_embeddings` read from `config.json`, with the
/// historical 4096 used only when neither is available. A non-negative
/// `--n-predict N` is taken verbatim.
pub(super) fn resolve_default_max_tokens(
    n_predict: i32,
    per_slot_context_size: usize,
    model_path: &Path,
) -> usize {
    if n_predict >= 0 {
        return n_predict as usize;
    }
    if per_slot_context_size > 0 {
        return per_slot_context_size;
    }
    crate::read_model_context_window(model_path)
        .unwrap_or(crate::cli::max_tokens::DEFAULT_CONTEXT_WINDOW_FALLBACK)
}

pub(super) fn resolve_dry_penalty_last_n(value: i32) -> usize {
    if value < 0 { 0 } else { value as usize }
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

    // The ViT-backed Gemma 4 VLM and the encoder-free Gemma 4 Unified model
    // both consume `video_url` content blocks (issue #164). Kimi-VL / Kimi-VL
    // 2.5 (MoonViT 3D) also consume video via the shared Kimi media path
    // (issue #551). Mirror the dispatch in
    // `commands/generate_vlm::compute_vlm_embeddings` and add new variants here
    // when more video-capable models land.
    let video = matches!(
        model_type,
        ModelType::Gemma4VLM | ModelType::Gemma4Unified | ModelType::KimiVL | ModelType::KimiK25
    );
    if video {
        tracing::info!(
            "model_type={:?}: enabling video_url content block support",
            model_type
        );
    }

    ModelMediaSupport { video }
}

fn is_muse_glimmer_model_path(model_path: &Path) -> bool {
    matches!(
        crate::models::get_model_type(model_path),
        Ok(crate::models::ModelType::MuseGlimmerVLM)
    )
}

fn xla_backend_requested_from_env() -> bool {
    std::env::var("MLXCEL_BACKEND")
        .ok()
        .is_some_and(|backend| backend.eq_ignore_ascii_case("xla"))
}

fn muse_glimmer_distributed_requested(startup: &ServerStartupConfig) -> bool {
    startup.distributed_config.is_some()
        || startup.node_role.is_some()
        || !startup.peers.is_empty()
        || !startup.prefill_peers.is_empty()
        || !startup.decode_peers.is_empty()
        || startup.serving_bind.is_some()
}

fn validate_muse_glimmer_unsupported_startup(startup: &ServerStartupConfig) -> Result<()> {
    if !is_muse_glimmer_model_path(&startup.model_path) {
        return Ok(());
    }

    anyhow::ensure!(
        startup.adapter_path.is_none(),
        "Muse Glimmer VLM does not support LoRA/adapters; remove --adapter/--lora"
    );
    anyhow::ensure!(
        startup.draft_model_path.is_none()
            && startup.draft_kind.is_none()
            && startup.draft_block_size.is_none(),
        "Muse Glimmer VLM does not support speculative decoding or DFlash; remove \
         --draft-model/--model-draft, --draft-kind, and --draft-block-size"
    );
    anyhow::ensure!(
        startup.kv_cache_mode == mlxcel_core::cache::KVCacheMode::Fp16
            && !startup.batch_kv_quant.is_enabled(),
        "Muse Glimmer VLM does not support INT8/Turbo KV cache modes or batch KV \
         quantization because it owns mixed sliding/full caches; use fp16 KV cache \
         mode and leave --kv-bits 0"
    );
    anyhow::ensure!(
        startup.tp_size == 1,
        "Muse Glimmer VLM does not support tensor-parallel inference yet; use --tp-size 1"
    );
    anyhow::ensure!(
        startup.pp_layers.is_none()
            && startup.pp_auto.is_none()
            && !startup.pp_peer
            && !startup.enable_elastic_pp,
        "Muse Glimmer VLM does not support pipeline-parallel inference yet; remove \
         --pp-* and elastic-PP flags"
    );
    anyhow::ensure!(
        !muse_glimmer_distributed_requested(startup),
        "Muse Glimmer VLM does not support distributed or disaggregated serving yet; \
         run a single-process MLX server"
    );
    anyhow::ensure!(
        !xla_backend_requested_from_env(),
        "Muse Glimmer VLM does not support XLA/IREE/OpenXLA execution yet; unset \
         MLXCEL_BACKEND=xla"
    );

    Ok(())
}

/// The 54 names b10621 accepts on `--chat-template` in place of a Jinja template.
///
/// Its `--chat-template` takes either template *text* or one of these built-in
/// identifiers, and the help lists them. mlxcel has no built-in template
/// library: an MLX checkpoint ships its own template in `tokenizer_config.json`
/// and mlxcel renders that. Passing a bare name here would be taken as the
/// template itself, so every prompt would render to the literal string
/// `chatml`, which is why it is detected and refused rather than accepted
/// (issue #1447). Verbatim from the b10621 `--help` text.
const B10621_BUILTIN_CHAT_TEMPLATES: &[&str] = &[
    "bailing",
    "bailing-think",
    "bailing2",
    "chatglm3",
    "chatglm4",
    "chatml",
    "command-r",
    "deepseek",
    "deepseek-ocr",
    "deepseek2",
    "deepseek3",
    "exaone-moe",
    "exaone3",
    "exaone4",
    "falcon3",
    "gemma",
    "gigachat",
    "glmedge",
    "gpt-oss",
    "granite",
    "granite-4.0",
    "granite-4.1",
    "grok-2",
    "hunyuan-dense",
    "hunyuan-moe",
    "hunyuan-vl",
    "kimi-k2",
    "llama2",
    "llama2-sys",
    "llama2-sys-bos",
    "llama2-sys-strip",
    "llama3",
    "llama4",
    "megrez",
    "minicpm",
    "mistral-v1",
    "mistral-v3",
    "mistral-v3-tekken",
    "mistral-v7",
    "mistral-v7-tekken",
    "monarch",
    "openchat",
    "orion",
    "pangu-embedded",
    "phi3",
    "phi4",
    "rwkv-world",
    "seed_oss",
    "smolvlm",
    "solar-open",
    "vicuna",
    "vicuna-orca",
    "yandex",
    "zephyr",
];

/// True when `value` is one of b10621's built-in chat-template names rather
/// than a Jinja template.
#[must_use]
pub fn is_b10621_builtin_chat_template(value: &str) -> bool {
    B10621_BUILTIN_CHAT_TEMPLATES.contains(&value.trim())
}

/// Refuse a `--chat-template` value that names a b10621 built-in template.
///
/// Called from both binaries before the model reference resolves, so the
/// mistake is reported immediately rather than after a load; the same guard
/// sits inside [`resolve_chat_template`] as the backstop for any other caller.
///
/// # Errors
///
/// Returns the diagnostic when `value` is a built-in name.
pub fn ensure_chat_template_is_not_a_builtin_name(value: &str) -> Result<()> {
    if is_b10621_builtin_chat_template(value) {
        anyhow::bail!(
            "--chat-template {} names one of llama-server's built-in chat templates. \
             mlxcel has no built-in template library: an MLX checkpoint carries its own \
             template in tokenizer_config.json, which is what mlxcel renders by default, and \
             this flag takes Jinja template text. Drop the flag to use the checkpoint's own \
             template, or pass the template itself with --chat-template-file <PATH>.",
            value.trim()
        );
    }
    Ok(())
}

/// Resolve chat template from override string, file, or model's tokenizer metadata.
///
/// `--chat-template` and `--chat-template-file` write the same field upstream,
/// so whichever appears last on the command line wins there. clap gives no
/// order, so the inline template wins here and a collision is logged rather
/// than resolved in silence; the manifest records the ordering difference
/// (issue #1447).
pub(super) fn resolve_chat_template(
    template_override: Option<&str>,
    template_file: Option<&Path>,
    model_path: &Path,
) -> Result<ChatTemplateProcessor> {
    if let Some(template) = template_override {
        // A bare built-in name is not a template. Taken literally it would
        // render every prompt to the name itself, so it is refused with the
        // two ways to supply the real thing.
        ensure_chat_template_is_not_a_builtin_name(template)?;
        if template_file.is_some() {
            tracing::warn!(
                "--chat-template and --chat-template-file are both set; the inline template \
                 wins (llama-server applies whichever came last on the command line)"
            );
        }
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
    api_keys: crate::server::ApiKeys,
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
    let sampling_defaults = resolve_generation_sampling_defaults(startup);
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
        reasoning_format: startup.reasoning_format,
        skip_chat_parsing: startup.skip_chat_parsing,
        no_prefill_assistant: startup.no_prefill_assistant,
        reasoning_budget_message: startup.reasoning_budget_message.clone(),
        api_keys,
        decode_timeout_seconds: startup.decode_timeout,
        api_prefix: startup.api_prefix.clone(),
        sse_ping_interval: startup.sse_ping_interval,
        model_alias: startup.model_alias.clone(),
        model_aliases: startup.model_aliases.clone(),
        context_size,
        n_parallel: startup.n_parallel,
        enable_slots_endpoint: startup.enable_slots,
        enable_props_endpoint: startup.enable_props,
        spm_infill: startup.spm_infill,
        embd_normalize: startup.embd_normalize,
        embedding_serving_mode: startup.embedding_serving_mode,
        enable_metrics_endpoint: startup.enable_metrics,
        default_temperature: sampling_defaults.temperature,
        default_top_p: sampling_defaults.top_p,
        default_top_k: sampling_defaults.top_k,
        default_min_p: startup.min_p,
        default_repetition_penalty: startup.repeat_penalty,
        default_repetition_context_size: startup.repeat_last_n,
        default_max_tokens: resolve_default_max_tokens(
            startup.n_predict,
            context_size,
            &startup.model_path,
        ),
        default_seed: startup.seed,
        default_frequency_penalty: startup.frequency_penalty,
        default_presence_penalty: startup.presence_penalty,
        default_dry_multiplier: startup.dry_multiplier,
        default_dry_base: startup.dry_base,
        default_dry_allowed_length: startup.dry_allowed_length,
        default_dry_penalty_last_n: resolve_dry_penalty_last_n(startup.dry_penalty_last_n),
        // Left empty here on purpose: `--dry-sequence-breaker` takes token
        // strings and the sampler takes token IDs, so resolving it needs the
        // tokenizer, which `start_server` loads after this function returns. It
        // fills the field there and fails startup on a breaker it cannot
        // represent (#1103).
        default_dry_sequence_breakers: Vec::new(),
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
        embedding_model_path: startup.embedding_model_path.clone(),
        embedding_batch_size: startup.embedding_batch_size,
        embedding_max_length: startup.embedding_max_length,
        embedding_queue_depth: startup.embedding_queue_depth,
        embedding_request_timeout_secs: startup.embedding_request_timeout_secs,
        reranker_model_path: startup.reranker_model_path.clone(),
        rerank_batch_size: startup.rerank_batch_size,
        prefill_chunk_size: startup.prefill_chunk_size,
        // #1011: pass the explicit --prefill-grant-interval through untouched
        // (the scheduler resolves env / shipped default when this is None).
        prefill_grant_interval: startup.prefill_grant_interval,
        enable_preemption: startup.enable_preemption,
        preemption_policy: parse_preemption_policy(&startup.preemption_policy),
        no_batch: startup.no_batch,
        max_batch_prefill: startup.max_batch_prefill.max(1),
        // #715: pass the explicit --max-batch-prefill-tokens through untouched
        // (the scheduler resolves env / derived default when this is None).
        max_batch_prefill_tokens: startup.max_batch_prefill_tokens,
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
        cors_policy: startup.cors_policy.clone(),
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
        // Global loop-detection override (issue #432) from `MLXCEL_LOOP_DETECTION`.
        // `None` means the per-family auto-enable policy applies.
        loop_detection: resolve_loop_detection_env(),
        // Whether the loaded model is in the Gemma 4 family. Combined with the
        // per-request tool-shaped prompt flag (issues #967 and #977), this turns
        // on the loop-detection default for protected traffic; plain and
        // grammar-only requests stay disabled.
        model_is_gemma4_family: detect_gemma4_family(&startup.model_path),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ResolvedGenerationSamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

pub(super) fn resolve_generation_sampling_defaults(
    startup: &ServerStartupConfig,
) -> ResolvedGenerationSamplingDefaults {
    let config_defaults = crate::read_generation_config_defaults(&startup.model_path);
    ResolvedGenerationSamplingDefaults {
        temperature: if startup.temperature_was_set {
            startup.temperature
        } else {
            config_defaults.temperature.unwrap_or(startup.temperature)
        },
        top_p: if startup.top_p_was_set {
            startup.top_p
        } else {
            config_defaults.top_p.unwrap_or(startup.top_p)
        },
        top_k: if startup.top_k_was_set {
            startup.top_k
        } else {
            config_defaults.top_k.unwrap_or(startup.top_k)
        },
    }
}

/// Whether `model_path` loads a Gemma 4 family model (`Gemma4`, `Gemma4VLM`, or
/// `Gemma4Unified`). Used to gate the issue #432 loop-detection auto-enable. A
/// detection error (unreadable config) returns `false`, preserving the baseline.
fn detect_gemma4_family(model_path: &Path) -> bool {
    use crate::models::ModelType;
    matches!(
        crate::models::get_model_type(model_path),
        Ok(ModelType::Gemma4 | ModelType::Gemma4VLM | ModelType::Gemma4Unified)
    )
}

/// Parse the `MLXCEL_LOOP_DETECTION` global override (issue #432).
///
/// Accepted values (case-insensitive):
/// - unset / empty: `None` (the per-family auto-enable policy applies).
/// - `off` / `0` / `none` / `false` / `disabled`: force-disable for every
///   request (`Some(disabled)`), still overridable per request.
/// - `on` / `default` / `true` / `enabled`: force the recommended threshold
///   (`min=1, max=20, count=12`).
/// - `MIN,MAX,COUNT` or `MIN:MAX:COUNT`: an explicit triple, e.g. `1,20,12`.
///
/// A malformed value warns and returns `None` so a typo does not silently
/// change generation behavior.
fn resolve_loop_detection_env() -> Option<mlxcel_core::LoopDetectionConfig> {
    let raw = std::env::var("MLXCEL_LOOP_DETECTION").ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "off" | "0" | "none" | "false" | "disabled" => {
            return Some(mlxcel_core::LoopDetectionConfig::disabled());
        }
        "on" | "default" | "true" | "enabled" => {
            return Some(crate::server::request_options::LOOP_DETECTION_RECOMMENDED);
        }
        _ => {}
    }
    let parts: Vec<&str> = trimmed.split([',', ':']).map(str::trim).collect();
    if parts.len() == 3
        && let (Ok(min), Ok(max), Ok(count)) = (
            parts[0].parse::<usize>(),
            parts[1].parse::<usize>(),
            parts[2].parse::<usize>(),
        )
    {
        return Some(mlxcel_core::LoopDetectionConfig::new(min, max, count));
    }
    tracing::warn!(
        "MLXCEL_LOOP_DETECTION=\"{raw}\" is not valid; expected off/on or MIN,MAX,COUNT \
         (e.g. 1,20,12). Ignoring."
    );
    None
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
            // Warmup is text-only; no image budget to override.
            image_soft_tokens: None,
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
    // Surface the backend's GPU count at startup (epic #486, sub-issue #487).
    // On Metal this is always 1; on a CUDA multi-GPU host it reports the real
    // adapter count that `--tp-size` can target.
    tracing::info!("Detected {} GPU(s)", mlxcel_core::gpu_device_count());
    let prefix = startup.api_prefix.as_str();
    tracing::info!("Endpoints:");
    tracing::info!("  POST {prefix}/v1/chat/completions  - OpenAI chat completions");
    tracing::info!("  POST {prefix}/v1/completions       - OpenAI text completions");
    tracing::info!("  GET  {prefix}/v1/models            - List models");
    tracing::info!("  POST {prefix}/completion           - llama-server native completion");
    tracing::info!("  POST {prefix}/tokenize             - Tokenize text");
    tracing::info!("  POST {prefix}/detokenize           - Detokenize tokens");
    if startup.enable_props {
        tracing::info!("  GET  {prefix}/props                - Server properties");
    }
    if startup.enable_slots {
        tracing::info!("  GET  {prefix}/slots                - Slot status");
    }
    tracing::info!("  GET  {prefix}/health               - Health check");
}

/// Bind and serve, applying the b10621 transport options (#1432).
///
/// The listen target comes from `--host` / `--port`
/// ([`crate::server::transport::resolve_listen_target`]), the socket
/// read/write budget from `--timeout`, and TLS from
/// `--ssl-cert-file` / `--ssl-key-file`. The accept loop itself lives in
/// [`crate::server::listen`] so all three transports enforce the timeout the
/// same way.
async fn serve_http(startup: &ServerStartupConfig, app: axum::Router) -> Result<()> {
    let resolved = crate::server::transport::resolve_listen_target(&startup.host, startup.port)?;
    if let Some(warning) = &resolved.legacy_socket_warning {
        tracing::warn!("{warning}");
    }

    let tls = match startup.tls.as_ref() {
        Some(paths) => Some(crate::server::tls::build_server_config(paths)?),
        None => None,
    };

    crate::server::listen::serve(
        &resolved.target,
        app,
        crate::server::listen::ServeOptions {
            timeouts: crate::server::transport::HttpTimeouts::from_secs(startup.http_timeout),
            reuse_port: startup.reuse_port,
            tls,
        },
        |addr| log_endpoints(startup, addr),
    )
    .await
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

    // issue #1350: the KV cache mode was resolved against the model family in
    // `into_startup_config`, which runs before the subscriber above exists.
    // Report the substitution now, so the log states the mode the caches are
    // really built with rather than the one that was asked for.
    for notice in &startup.kv_cache_mode_notices {
        tracing::warn!("{notice}");
    }
    tracing::info!(
        kv_cache_mode = %startup.kv_cache_mode,
        kv_bits = startup.batch_kv_quant.bits,
        "effective KV cache mode"
    );

    // Florence-2 (issue #1073): the encoder-decoder (seq2seq) family is
    // served on its dedicated worker loop (`server/florence2_worker.rs`),
    // which the model worker thread branches into after loading the
    // checkpoint, before any decoder-only scheduler starts. The #856-era
    // startup refusal is gone; the flag below only gates the text-only
    // warmup, which cannot run against an image-task model.
    let is_florence2 = matches!(
        crate::models::get_model_type(&startup.model_path),
        Ok(crate::models::ModelType::Florence2VLM)
    );

    // Issue #688 (M1/M2 hardening): disable CUDA graph capture for hazard-family
    // models (Gemma 4) here, on the main startup thread, before any generation or
    // pipeline worker is spawned and before the first GPU eval latches MLX's
    // process-wide `use_cuda_graphs` static. (DeepSeek-V2 was on this lever from
    // #829 to #831; its collapse turned out to be the broken RMSNorm overlay, not
    // graph capture, so the family was removed.) Performing the env write on this
    // thread (rather than only at the per-load-site calls, which run inside the
    // spawned worker) keeps it sound under Rust 2024's concurrent-getenv rule. This
    // single chokepoint covers every serve path that flows through `start_server`:
    // the batched, legacy, tensor-parallel and XLA workers, the in-process
    // pipeline-parallel worker, and the remote pipeline stage worker (the
    // `serve_remote_pipeline_stage` branch below) all load `startup.model_path`.
    // Unaffected model families and an explicit `MLX_USE_CUDA_GRAPHS` operator
    // override are left untouched.
    crate::loading::maybe_disable_cuda_graphs_for_model_for_path(&startup.model_path);
    validate_muse_glimmer_unsupported_startup(&startup)?;

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

    // Install the b10621 RoPE override before anything can load a checkpoint.
    // It is process-wide because each model family reads its own `config.json`
    // inside its own loader, so there is no argument to thread; the worker
    // confirms it actually reached the model once the load returns
    // (`rope_overrides::verify_applied`).
    crate::models::rope_overrides::install(startup.rope_override)
        .map_err(|message| anyhow::anyhow!("{message}"))?;
    if let Some(over) = startup.rope_override.as_ref() {
        tracing::info!(
            "RoPE runtime override: {} (applied to the checkpoint's rope_scaling block and \
             rope_theta before the model is built)",
            over.describe()
        );
    }

    validate_parallel_context_startup(&startup)?;
    validate_pipeline_parallel_startup(&startup)?;
    let tp_support = resolve_tensor_parallel_runtime_support(&startup)?;

    // One line naming what the context and batch flags actually resolved to.
    // Until #1450 none of these was logged anywhere: `--ctx-size` reached a
    // per-slot divisor and a KV cap and the operator had no way to see either
    // number, so an aggregate that silently became 512 per slot looked exactly
    // like one that stayed at 8192. The per-slot value is the one a single
    // request is bounded by, so it is reported next to the total rather than
    // instead of it.
    {
        let slots = effective_parallel_context_slots(
            startup.n_parallel,
            startup.max_batch_size,
            startup.no_batch,
        );
        let per_slot = resolve_parallel_context_size(
            startup.ctx_size,
            startup.n_parallel,
            startup.max_batch_size,
            startup.no_batch,
        );
        tracing::info!(
            ctx_size = startup.ctx_size,
            ctx_size_per_slot = per_slot,
            context_slots = slots,
            n_parallel = startup.n_parallel,
            prefill_chunk_size = startup.prefill_chunk_size,
            max_kv_size = ?startup.max_kv_size,
            "resolved context and batch geometry (0 = the checkpoint's own trained context)"
        );
    }

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
    // Report the effective CUDA graph-cache LRU capacity now that a tracing
    // subscriber is installed (initialize_server_logging, above) and the runtime
    // device is known to be CUDA. `apply_cuda_graph_cache_default` sets the env var
    // at the top of `main()`, before any subscriber exists, so this is the first
    // point on the server path where the choice can actually be logged (issue #818).
    #[cfg(feature = "cuda")]
    if runtime.device == crate::RuntimeDevice::Gpu {
        let cache_size =
            std::env::var("MLX_CUDA_GRAPH_CACHE_SIZE").unwrap_or_else(|_| "unset".to_string());
        tracing::info!(
            "CUDA graph-cache LRU capacity: MLX_CUDA_GRAPH_CACHE_SIZE={cache_size} (mlxcel raises MLX's default of 400 to 2000 so long-lived, shape-diverse decode does not hit the cache-thrashing abort from issue #818, unless an operator override is set)"
        );
    }
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

    let api_keys = crate::server::resolve_api_keys(&startup.api_keys, &startup.api_key_files)?;
    if !api_keys.is_empty() {
        // Never the key material itself: a count is all an operator needs to
        // confirm the file was read, and it is all a log should ever carry.
        tracing::info!("API-key authentication enabled ({} keys)", api_keys.len());
    } else if matches!(
        startup.cors_policy.origins,
        crate::server::OriginPolicy::Wildcard
    ) {
        // Upstream emits the same warning at startup: an origin policy of `*`
        // with no API key is reachable by any page in the browser.
        tracing::warn!(
            "CORS is set to allow all origins ('*') and no API key is set; this can be a \
             security risk (cross-origin attacks). Set --api-key, or narrow --cors-origins / \
             --allowed-origins"
        );
    }
    let mut config = build_server_config(&startup, api_keys);

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

    // `--dry-sequence-breaker` takes token strings and the sampler takes token
    // IDs, so this is the first point in startup where the flag can be
    // resolved. Failing here rather than dropping an unrepresentable breaker
    // is deliberate: an inert breaker makes the DRY penalty stronger than the
    // operator configured, with nothing in the logs or `/props` to say so
    // (#1103).
    config.default_dry_sequence_breakers = super::dry_breakers::resolve_dry_sequence_breakers(
        &tokenizer,
        &startup.dry_sequence_breakers,
    )?;
    if !config.default_dry_sequence_breakers.is_empty() {
        // The decoded pieces are logged alongside the ids because an id on its
        // own cannot be checked. A tokenizer that prepends a word-boundary
        // marker can resolve a plausible-looking id for the wrong token, and
        // the piece is the only place that shows it.
        tracing::info!(
            breakers = ?startup.dry_sequence_breakers,
            token_ids = ?config.default_dry_sequence_breakers,
            pieces = %super::dry_breakers::describe_resolved_breakers(
                &tokenizer,
                &config.default_dry_sequence_breakers,
            ),
            "DRY sequence breakers resolved to token IDs"
        );
    }

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
    // Issue #686: the Gemma-4 thinking-channel template's thinking-OFF branch
    // is the correct interactive default (a CLOSED `<|channel>thought\n<channel|>`
    // priming scaffold matching transformers' no-`enable_thinking` render), so
    // the `has_thinking` heuristic below must not flip it on; forcing thinking
    // there produces a bare `<|turn>model\n` that greedy-collapses to `<pad>`.
    if thinking_markers.has_thinking() && !chat_template.wants_thinking_default_off() {
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
        return serve_http(&startup, app).await;
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

    if startup.warmup && is_florence2 {
        // The warmup prompt is the text literal "Hello"; the Florence-2
        // seq2seq worker rejects any request that is not a task marker with
        // exactly one image, so a warmup attempt would only log a spurious
        // failure. The worker warms on its first real request instead.
        tracing::info!("Skipping text warmup for Florence-2 (image-task seq2seq model)");
    } else if startup.warmup {
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

    // Embedding wiring (#1353): `--embedding-model` loads a second checkpoint
    // on its own worker thread; without it, an `-m` that detects as an
    // embedding kind is served on `/v1/embeddings` instead of chat (the chat
    // worker's `load_model` bails and logs, exactly like the Whisper path).
    // Naming both is a configuration error, reported before the listener
    // binds. A failing `--embedding-model` load is fatal for the same reason;
    // a failing `-m` embedding load logs and leaves the slot empty so the
    // server still answers with structured 501s.
    // b10621 options this build parses but does not yet act on (issue #1447).
    // Said once at startup rather than left silent: the manifest records them
    // as deferred with a linked issue, and an operator who passed one needs to
    // know the request path does not honour it yet.
    if config.no_prefill_assistant {
        tracing::info!(
            "--no-prefill-assistant is what mlxcel already does: a trailing assistant message \
             is answered with a fresh turn, not continued. llama-server continues it by default"
        );
    }
    if config.reasoning_budget_message.is_some() {
        tracing::warn!(
            "--reasoning-budget-message is accepted but not yet injected before the \
             end-of-thinking tag; the reasoning budget still ends the block without it"
        );
    }

    let served_chat_id = config
        .model_alias
        .clone()
        .unwrap_or_else(|| model_provider.model_id().to_string());
    // b10621 keeps every `--alias a,b,c` entry as an API-visible name; mlxcel
    // serves the first and does not yet report the rest on `/v1/models`
    // (#1438 owns the model-object `aliases` array). Say so once at startup so
    // an operator who passed a list is not left guessing which name the server
    // answers to (issue #1434).
    if config.model_aliases.len() > 1 {
        tracing::info!(
            "serving model id '{served_chat_id}'; the additional --alias entries [{}] are \
             recorded but not yet reported on /v1/models",
            config.model_aliases[1..].join(", ")
        );
    }
    // `--pooling` is installed before any embedding checkpoint is loaded,
    // because every family resolves its mode inside its own constructor
    // (#1452). The cell is cleared when the flag was not given so a previous
    // in-process server (the test harness runs several) cannot leak its
    // choice into this one.
    crate::embeddings::set_pooling_override(startup.pooling);
    let embedding_model = resolve_embedding_provider(&startup, &config, &served_chat_id)?;
    // Rerank wiring (#1356) follows the same rule, with one addition: a
    // generative reranker's checkpoint is indistinguishable from a chat
    // model's, so `--reranker-model` is the only way to reach it and naming
    // the same path in `-m` and `--reranker-model` loads it once on the rerank
    // worker instead of being an error.
    let rerank_model = resolve_rerank_provider(&startup, &config, &served_chat_id)?;
    // A mode flag that resolved no worker is a command line that cannot do
    // what it asked for, so it fails here rather than answering 501 to every
    // request for the life of the process (#1452).
    check_serving_mode(&config, embedding_model.is_some(), rerank_model.is_some())?;

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
    .with_audio_model(audio_model)
    .with_embedding_model(embedding_model)
    .with_rerank_model(rerank_model);
    let app = create_app(state);

    serve_http(&startup, app).await
}

/// Refuse a `--embeddings` / `--reranking` command line that resolved no
/// worker able to serve that mode (#1452).
///
/// b10621 restricts an existing model; mlxcel selects a dedicated worker, so
/// the flag additionally asserts that one exists. Without this check the flag
/// would parse, generation would be off, and the only route left would answer
/// 501 forever, which is the accepted-and-ignored failure epic #1431 exists to
/// remove.
pub(crate) fn check_serving_mode(
    config: &ServerConfig,
    has_embedding: bool,
    has_rerank: bool,
) -> Result<()> {
    use crate::server::config::EmbeddingServingMode;
    match config.embedding_serving_mode {
        EmbeddingServingMode::Any => Ok(()),
        EmbeddingServingMode::EmbeddingOnly if has_embedding => Ok(()),
        EmbeddingServingMode::RerankOnly if has_rerank => Ok(()),
        EmbeddingServingMode::EmbeddingOnly => anyhow::bail!(
            "--embeddings restricts this server to embedding requests, but no embedding \
             checkpoint was loaded: pass an embedding checkpoint to -m, or name one with \
             --embedding-model <path>. `mlxcel list` reports which architectures load as \
             embedders."
        ),
        EmbeddingServingMode::RerankOnly => anyhow::bail!(
            "--reranking restricts this server to rerank requests, but no reranker was loaded: \
             pass a sequence-classifier checkpoint to -m, or name a reranker with \
             --reranker-model <path>. The Qwen3 and Qwen3-VL generative rerankers are only \
             reachable through --reranker-model."
        ),
    }
}

/// Where the embedding checkpoint comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingSource {
    /// `--embedding-model <path>`: a second checkpoint next to the chat model.
    Explicit(PathBuf),
    /// `-m` itself detected as an embedding kind.
    Primary(PathBuf),
    /// No embedding model to serve.
    None,
}

/// Pick the embedding checkpoint from `--embedding-model` and the `-m`
/// detection, rejecting the combination of both.
pub(crate) fn resolve_embedding_source(
    model_path: &Path,
    embedding_model_path: Option<&Path>,
) -> Result<EmbeddingSource> {
    let primary_is_embedding = crate::models::get_model_type(model_path)
        .is_ok_and(crate::model_metadata::is_embedding_model_type);
    match (embedding_model_path, primary_is_embedding) {
        (Some(explicit), true) => anyhow::bail!(
            "two embedding models: -m {} is an embedding checkpoint and --embedding-model {} \
             was also given. Pass a chat model to -m and the embedding checkpoint to \
             --embedding-model, or pass the embedding checkpoint to -m alone.",
            model_path.display(),
            explicit.display()
        ),
        (Some(explicit), false) => Ok(EmbeddingSource::Explicit(explicit.to_path_buf())),
        (None, true) => Ok(EmbeddingSource::Primary(model_path.to_path_buf())),
        (None, false) => Ok(EmbeddingSource::None),
    }
}

/// Served id of an explicitly named side checkpoint: its directory name.
///
/// Used by both `--embedding-model` and `--reranker-model`, which each get
/// their own id next to the chat model's.
fn side_model_id_for_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| path.display().to_string())
}

/// Spawn the embedding worker for the resolved source, if any.
fn resolve_embedding_provider(
    startup: &ServerStartupConfig,
    config: &ServerConfig,
    served_chat_id: &str,
) -> Result<Option<Arc<dyn crate::server::embedding_model::EmbeddingModelProvider>>> {
    let source =
        resolve_embedding_source(&startup.model_path, config.embedding_model_path.as_deref())?;
    let (path, model_id, fatal) = match source {
        EmbeddingSource::None => return Ok(None),
        EmbeddingSource::Explicit(path) => {
            let id = side_model_id_for_path(&path);
            (path, id, true)
        }
        EmbeddingSource::Primary(path) => (path, served_chat_id.to_string(), false),
    };

    let request_timeout =
        std::time::Duration::from_secs(if config.embedding_request_timeout_secs == 0 {
            crate::server::config::DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS
        } else {
            config.embedding_request_timeout_secs
        });
    let load_options = crate::embeddings::EmbeddingLoadOptions {
        max_length: config.embedding_max_length,
    };
    tracing::info!(
        path = %path.display(),
        model_id = %model_id,
        batch_size = config.embedding_batch_size,
        "Loading embedding model for /v1/embeddings"
    );
    match crate::server::embedding_worker::EmbeddingWorkerProvider::load(
        &path,
        model_id,
        config.embedding_batch_size,
        config.embedding_queue_depth,
        request_timeout,
        load_options,
    ) {
        Ok(provider) => {
            let info = provider.info();
            tracing::info!(
                model_type = ?info.model_type,
                dim = info.dim,
                max_length = info.max_length,
                multi_vector = info.multi_vector,
                supports_images = info.supports_images,
                batch_size = info.batch_size,
                "Embedding model ready on /v1/embeddings"
            );
            Ok(Some(Arc::new(provider)))
        }
        Err(err) if fatal => Err(err.context(format!(
            "failed to load --embedding-model {}",
            path.display()
        ))),
        Err(err) => {
            tracing::error!(
                "Failed to load embedding model {}: {err:#}; /v1/embeddings will return 501",
                path.display()
            );
            Ok(None)
        }
    }
}

/// Where the reranker checkpoint comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RerankSource {
    /// `--reranker-model <path>`: a checkpoint next to the chat model.
    Explicit(PathBuf),
    /// `-m` itself detected as a sequence-classifier reranker.
    Primary(PathBuf),
    /// No reranker to serve.
    None,
}

/// Pick the reranker checkpoint from `--reranker-model` and the `-m`
/// detection.
///
/// Passing the same path to both is allowed and resolves to
/// [`RerankSource::Primary`]: the checkpoint is then loaded once, on the
/// rerank worker, which is what "when `--reranker-model` names the same path
/// as `-m`, the checkpoint is loaded once and chat stays unloaded" means. Two
/// different reranker checkpoints are a configuration error.
pub(crate) fn resolve_rerank_source(
    model_path: &Path,
    reranker_model_path: Option<&Path>,
) -> Result<RerankSource> {
    let primary_is_reranker = crate::models::get_model_type(model_path)
        .is_ok_and(crate::model_metadata::is_reranker_model_type);
    match (reranker_model_path, primary_is_reranker) {
        // The same directory in both flags is the rerank-only shape: `-m` is
        // required, and a generative reranker is only reachable through
        // `--reranker-model`, so this is how an operator asks for a server
        // that only reranks. `ModelProvider` recognizes it too and leaves the
        // chat worker without a model.
        (Some(explicit), _) if explicit == model_path => {
            Ok(RerankSource::Primary(model_path.to_path_buf()))
        }
        (Some(explicit), true) => anyhow::bail!(
            "two rerankers: -m {} is a reranker checkpoint and --reranker-model {} was also \
             given. Pass a chat model to -m and the reranker to --reranker-model, or pass the \
             reranker to -m alone.",
            model_path.display(),
            explicit.display()
        ),
        (Some(explicit), false) => Ok(RerankSource::Explicit(explicit.to_path_buf())),
        (None, true) => Ok(RerankSource::Primary(model_path.to_path_buf())),
        (None, false) => Ok(RerankSource::None),
    }
}

/// Spawn the rerank worker for the resolved source, if any.
fn resolve_rerank_provider(
    startup: &ServerStartupConfig,
    config: &ServerConfig,
    served_chat_id: &str,
) -> Result<Option<Arc<dyn crate::server::rerank_model::RerankModelProvider>>> {
    let source = resolve_rerank_source(&startup.model_path, config.reranker_model_path.as_deref())?;
    let (path, model_id, fatal) = match source {
        RerankSource::None => return Ok(None),
        RerankSource::Explicit(path) => {
            let id = side_model_id_for_path(&path);
            (path, id, true)
        }
        RerankSource::Primary(path) => (path, served_chat_id.to_string(), false),
    };

    let request_timeout =
        std::time::Duration::from_secs(if config.embedding_request_timeout_secs == 0 {
            crate::server::config::DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS
        } else {
            config.embedding_request_timeout_secs
        });
    let load_options = crate::rerank::RerankLoadOptions {
        batch_size: (config.rerank_batch_size > 0).then_some(config.rerank_batch_size),
        max_length: None,
    };
    tracing::info!(
        path = %path.display(),
        model_id = %model_id,
        batch_size = config.rerank_batch_size,
        "Loading reranker for /v1/rerank"
    );
    match crate::server::rerank_worker::RerankWorkerProvider::load(
        &path,
        model_id,
        config.embedding_queue_depth,
        request_timeout,
        load_options,
    ) {
        Ok(provider) => {
            let info = provider.info();
            tracing::info!(
                kind = info.kind.as_str(),
                model_type = %info.model_type,
                max_length = info.max_length,
                batch_size = info.batch_size,
                supports_images = info.supports_images,
                "Reranker ready on /v1/rerank"
            );
            Ok(Some(Arc::new(provider)))
        }
        Err(err) if fatal => Err(err.context(format!(
            "failed to load --reranker-model {}",
            path.display()
        ))),
        Err(err) => {
            tracing::error!(
                "Failed to load reranker {}: {err:#}; /v1/rerank will return 501",
                path.display()
            );
            Ok(None)
        }
    }
}

#[cfg(test)]
#[path = "startup_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "muse_glimmer_startup_guard_tests.rs"]
mod muse_glimmer_startup_guard_tests;
