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

//! Shared server configuration types.
//!
//! These structs are reused by route handlers, startup normalization, and the
//! model worker, so keeping them separate from the startup side effects makes
//! server policy easier to extend and test.

use std::path::PathBuf;

use crate::SamplingConfig;
use crate::distributed::ShardConfig;
use crate::distributed::TransportBackend;
use crate::distributed::pipeline::RemotePipelineRuntimeConfig;
use crate::server::batch::RequestPriority;
use crate::server::prompt_cache::key::MultimodalDigest;
use mlxcel_core::lang_analyzer::LangBiasConfig;
use mlxcel_core::sampling::LogprobsConfig;

/// Storage backend used by the server batch scheduler for decode-time state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeStorageBackend {
    /// Select paged decode automatically for workers that support it.
    #[default]
    Auto,
    /// Existing dense per-sequence KV caches.
    Dense,
    /// Paged block-table state mirrored alongside dense compatibility caches.
    Paged,
}

impl std::str::FromStr for DecodeStorageBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "dense" => Ok(Self::Dense),
            "paged" => Ok(Self::Paged),
            other => Err(format!(
                "unknown decode storage backend \"{other}\"; expected \"auto\", \"dense\", or \"paged\""
            )),
        }
    }
}

/// Per-request metadata that the scheduler needs to compose a
/// [`crate::server::prompt_cache::key::PromptCacheKey`] without re-running
/// the chat template pipeline on the worker thread.
///
/// Route handlers build this once — when
/// [`crate::server::state::AppState::prompt_cache`] is installed — and hand
/// it to the scheduler via [`ServerGenerateOptions::prompt_cache_ctx`]. When
/// `None` the scheduler falls back to its pre-cache behavior.
#[derive(Debug, Clone)]
pub struct PromptCacheRequestContext {
    /// Display model id (matches
    /// [`crate::server::state::AppState::display_model_id`]).
    pub model_id: String,
    /// LoRA adapter id; `None` for the base model.
    pub lora_id: Option<String>,
    /// Stable digest of the rendering pipeline inputs — see
    /// [`crate::server::prompt_cache::key::template_sig`].
    pub template_sig: String,
    /// Resolved session key — see
    /// [`crate::server::prompt_cache::key::resolve_session_key`]. Owned so
    /// the scheduler can compose a [`crate::server::prompt_cache::key::PromptCacheKey`]
    /// on demand without reaching back into the route layer.
    pub session_key: String,
    /// Stable digest of the request's resolved multimodal payload (image +
    /// audio bytes), built by
    /// [`crate::server::prompt_cache::key::multimodal_digest`] over the
    /// post-resolution byte slices.
    ///
    /// [`MultimodalDigest::empty`] for text-only requests, so the composed
    /// cache key stays byte-identical to the pre-#124 text path. Folding the
    /// digest into the key is what lets a future multimodal-sharing step
    /// (#124 step c) reuse image/audio prefixes without a text↔image bucket
    /// collision; until that step lifts the scheduler's `is_multimodal` gate
    /// the digest is carried but multimodal requests still take the cold path.
    pub mm_digest: MultimodalDigest,
}

/// Bridge between server request params and `mlxcel-core` `SamplingConfig`.
#[derive(Debug, Clone)]
pub struct ServerGenerateOptions {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_sequences: Option<Vec<String>>,
    /// Request priority for prefill queue ordering.
    pub priority: RequestPriority,
    /// Log probability configuration; disabled by default (zero overhead).
    pub logprobs: LogprobsConfig,
    /// per-request thinking-token budget. `None` means "inherit
    /// whatever server default is configured"; `Some(budget)` explicitly sets
    /// a value for this request (including reverting to unbounded via
    /// the raw `-1` request value, which the routes translate to a sentinel
    /// before reaching this field).
    ///
    /// Resolution precedence is performed in the route layer via
    /// [`crate::server::thinking_budget::resolve_request_budget`] so the
    /// scheduler sees a single effective value.
    pub reasoning_budget: ReasoningBudgetOverride,

    /// whether the first generated token should be treated as
    /// "already inside the `<think>` block" because the prompt primed it.
    ///
    /// `true` for chat endpoints (`/v1/chat/completions`) whose chat template
    /// renders a Qwen3-style `<think>\n` at the end of the prompt, so the
    /// model's first decoded token is reasoning content. `false` for the raw
    /// text endpoints (`/v1/completions`, `/completion`) where the prompt is
    /// free-form and the model must emit `<think>` itself before any counting
    /// begins. Without this distinction, a raw-text request with
    /// `thinking_budget_tokens > 0` would miscount ordinary answer tokens as
    /// reasoning tokens.
    pub thinking_enter_block_on_start: bool,

    /// cache-key metadata the scheduler uses to look
    /// up a stored prompt prefix and adopt its detached KV cache. `None` when
    /// the route did not install a
    /// [`crate::server::prompt_cache::PromptCacheStore`] (the feature flag is
    /// off) or when the request does not participate in prefix reuse (e.g.
    /// raw text-completion endpoints that do not render a chat template).
    pub prompt_cache_ctx: Option<PromptCacheRequestContext>,

    /// optional structured-output constraint produced by
    /// [`crate::server::structured::build_constraint_from_response_format`].
    ///
    /// `None` when the request did not supply a `response_format` of type
    /// `"json_schema"`. When `Some`, the scheduler attaches the constraint
    /// to the queued sequence and drives `compute_mask` / `consume_token`
    /// around every per-step `sample_token_optimized` call so the emitted
    /// tokens always conform to the supplied JSON schema.
    ///
    /// Wrapped in `Arc<Mutex<...>>` because the constraint mutates internal
    /// matcher state on every step and must move from the route handler
    /// across the channel into the model worker thread without a fresh
    /// build (rebuilding is expensive — see `TOK_ENV_CACHE` in
    /// `structured.rs`). The Mutex is uncontended in practice: only the
    /// worker thread that owns the sequence touches it.
    pub structured: Option<
        std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,
}

/// Per-request reasoning-budget override.
///
/// Distinct from `Option<ThinkingBudget>` because the "per-request explicitly
/// set to -1 (revert to unbounded)" case needs to be representable distinctly
/// from "no per-request override; inherit server default". The route helpers
/// normalize request bodies into this enum before the scheduler consumes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningBudgetOverride {
    /// No per-request value supplied — the scheduler should use the
    /// server-wide default from `ServerConfig::reasoning_budget`.
    #[default]
    InheritServerDefault,
    /// Per-request override resolved to this effective budget (or `None` =
    /// explicitly unrestricted).
    Explicit(Option<crate::server::thinking_budget::ThinkingBudget>),
}

/// Policy for selecting which sequence to evict when preemption is enabled
/// and the batch is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreemptionPolicy {
    /// Evict the sequence that has generated the most tokens.
    #[default]
    LongestFirst,
    /// Evict the lowest-priority sequence; break ties by longest running.
    LowestPriority,
}

/// Normalized pipeline-parallel runtime mode for the server worker.
#[derive(Debug, Clone)]
pub enum PipelineParallelRuntimeConfig {
    /// Existing single-process stage-partitioned runtime.
    InProcess {
        layers: String,
        micro_batch_size: usize,
    },
    /// Coordinator runtime that dispatches requests to remote stages.
    RemoteCoordinator(RemotePipelineRuntimeConfig),
}

impl PipelineParallelRuntimeConfig {
    pub fn describe(&self) -> String {
        match self {
            Self::InProcess {
                layers,
                micro_batch_size,
            } => {
                format!("in_process(pp_layers={layers}, pp_micro_batch_size={micro_batch_size})")
            }
            Self::RemoteCoordinator(config) => format!(
                "remote_coordinator(stages={}, transport={}, bind_address={})",
                config.stage_peers.len(),
                config.transport_backend,
                config.bind_address
            ),
        }
    }
}

/// Startup-only config for launching this process as a remote pipeline stage.
#[derive(Debug, Clone)]
pub struct RemotePipelineStageConfig {
    pub bind_address: String,
    pub stage_index: u32,
    pub num_stages: u32,
    pub upstream_peer: Option<String>,
    pub downstream_peer: Option<String>,
    pub transport_backend: TransportBackend,
}

/// Default bound for the audio worker command queue (admission control).
///
/// Each queued speech-to-text command can hold up to the 25 MiB per-request
/// payload, so a depth of `8` caps queued payload at roughly 200 MiB plus the
/// one request in flight, while still absorbing short bursts.
pub const DEFAULT_AUDIO_QUEUE_DEPTH: usize = 8;

/// Default per-request reply timeout (seconds) for the audio worker. Generous
/// upper bound for a single bounded clip; a stuck request frees its blocking
/// thread after this instead of hanging.
pub const DEFAULT_AUDIO_REQUEST_TIMEOUT_SECS: u64 = 120;

/// Server configuration derived from CLI-compatible startup arguments.
///
/// Default values intentionally track `llama-server` behavior where practical
/// so route handlers can apply one consistent set of defaults.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub model_alias: Option<String>,
    /// Effective per-slot context window in tokens (`0` = model default).
    ///
    /// Startup lowers `--ctx-size C --parallel N` to `C / N` for continuous
    /// batching, matching llama.cpp server semantics. An explicit
    /// `--max-batch-size` override becomes the divisor because it controls the
    /// maximum number of concurrent decode sequences.
    pub context_size: usize,
    pub n_parallel: usize,
    pub enable_slots_endpoint: bool,
    pub enable_props_endpoint: bool,
    pub enable_metrics_endpoint: bool,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub default_top_k: i32,
    pub default_min_p: f32,
    pub default_repetition_penalty: f32,
    pub default_repetition_context_size: usize,
    pub default_max_tokens: usize,
    pub default_seed: Option<u64>,
    pub default_frequency_penalty: f32,
    pub default_presence_penalty: f32,
    pub default_dry_multiplier: f32,
    pub default_dry_base: f32,
    pub default_dry_allowed_length: usize,
    pub default_dry_penalty_last_n: usize,
    pub draft_model_path: Option<PathBuf>,
    pub num_draft_tokens: usize,
    /// raw `--draft-kind` override string from the CLI / env
    /// var (`LLAMA_ARG_DRAFT_KIND` / `MLXCEL_DRAFT_KIND`).
    ///
    /// `None` means the server should auto-detect the drafter kind from
    /// `draft_model_path` via
    /// [`mlxcel_core::drafter::resolve_drafter_kind`], OR run the
    /// classic [`crate::SpeculativeGenerator`] path when no drafter is
    /// configured. Stored as a raw `Option<String>` because parsing
    /// only succeeds for `dflash` / `mtp` (the `internal-mtp` variant of
    /// [`mlxcel_core::drafter::DrafterKind`] is auto-detected, not
    /// user-selectable) and the parse error must surface at the
    /// dispatch site where the operator-facing error message lives.
    pub draft_kind: Option<String>,
    /// explicit `--draft-block-size` override. `None` means
    /// "use the per-kind default" — `4` for MTP, `16` for DFlash. See
    /// [`crate::cli::speculative_args::default_block_size_for_kind`].
    pub draft_block_size: Option<u32>,
    /// Maximum number of sequences in the active decode batch.
    /// Defaults to `n_parallel` (4 as of #628); the worker clamps it to 1 for
    /// model families that cannot batch (`supports_batching() == false`).
    pub max_batch_size: usize,
    /// Maximum number of requests waiting in the prefill queue.
    pub max_queue_depth: usize,
    /// Bound on the audio worker command queue (admission control). When the
    /// queue is full, new audio requests get a structured `503` instead of
    /// growing memory without bound. A `0` clamps to at least one queued
    /// command at the channel boundary. See [`DEFAULT_AUDIO_QUEUE_DEPTH`].
    pub audio_queue_depth: usize,
    /// Per-request reply timeout for the audio worker, in seconds. A stuck or
    /// pathologically slow audio request frees its blocking thread and returns
    /// a structured `504` after this. A `0` falls back to the default rather
    /// than timing out instantly. See [`DEFAULT_AUDIO_REQUEST_TIMEOUT_SECS`].
    pub audio_request_timeout_secs: u64,
    /// Number of tokens per prefill chunk. When 0, chunking is disabled and
    /// the full prompt is prefilled in a single pass.
    pub prefill_chunk_size: usize,
    /// Whether preemptive eviction is enabled. When true and the batch is
    /// full, a high-priority incoming request may evict a lower-priority
    /// or longer-running active sequence.
    pub enable_preemption: bool,
    /// Policy used to select the eviction victim.
    pub preemption_policy: PreemptionPolicy,
    /// When true, disable the batch scheduler and use the legacy sequential
    /// worker. Equivalent to `max_batch_size <= 1` for scheduling purposes
    /// but makes the intent explicit and guarantees zero scheduler overhead.
    pub no_batch: bool,
    /// Maximum number of requests to batch together for prefill.
    ///
    /// When `> 1`, the scheduler collects up to this many pending requests and
    /// runs a single batched forward pass `[batch_size, max_seq_len]` so that
    /// larger matmul operations better saturate Neural Accelerator cores.
    /// Falls back to sequential (per-request) prefill when only one request
    /// is pending or on any error.
    ///
    /// Default: 1 (no batching, backward compatible).
    /// Recommended: 4–8 on M5 Pro/Max hardware.
    pub max_batch_prefill: usize,
    /// #715: padded-token budget bounding the batched-prefill transient
    /// (`--max-batch-prefill-tokens`). `None` (the default) lets the scheduler
    /// use `MLXCEL_MAX_BATCH_PREFILL_TOKENS` or the derived default
    /// (`max_batch_prefill * prefill_chunk_size`); `Some(0)` disables the cap
    /// (uncapped); `Some(n)` sets an explicit budget.
    pub max_batch_prefill_tokens: Option<usize>,
    /// Decode-time storage backend used by the batch scheduler.
    pub decode_storage_backend: DecodeStorageBackend,
    /// Normalized pipeline-parallel runtime mode for the server worker.
    pub pipeline_parallel_runtime: Option<PipelineParallelRuntimeConfig>,
    /// When present, launch this process as a remote pipeline stage instead of
    /// the HTTP API server.
    pub remote_pipeline_stage: Option<RemotePipelineStageConfig>,
    /// Tensor-parallel loading/runtime options resolved at startup.
    pub tensor_parallel: ShardConfig,
    /// Maximum number of cached post-projection image features per loaded model.
    ///
    /// `0` disables the cache entirely. When enabled, multi-turn VLM
    /// conversations that revisit the same image can skip the vision tower and
    /// multimodal embedder on subsequent turns. Default is
    /// [`DEFAULT_VISION_CACHE_SIZE`](crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE).
    pub vision_cache_size: usize,
    /// Axis B (B8): server-wide language bias configuration, if
    /// resolved at startup from CLI flags or the `LLAMA_ARG_LANG_BIAS` env
    /// var. Every batch sequence inherits this same policy (Phase 1 single
    /// policy per batch; per-request overrides reserved for B12).
    pub lang_bias_config: Option<LangBiasConfig>,

    /// server-wide default thinking-token budget for Qwen3-family
    /// models. `None` = unrestricted reasoning (default, bit-exact baseline).
    /// Per-request `thinking_budget_tokens` overrides this value (including
    /// a per-request `-1` reverting to unbounded for that one request).
    pub reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,

    /// server-wide default chat-template kwargs resolved from
    /// `--chat-template-kwargs` and/or `LLAMA_ARG_CHAT_TEMPLATE_KWARGS`.
    ///
    /// `None` means "no server-default kwargs"; per-request kwargs may still
    /// set keys such as `preserve_thinking`. The per-request merge happens in
    /// [`crate::server::chat_template_kwargs::merge_server_and_request`] so
    /// every registered key — today `preserve_thinking`, tomorrow others —
    /// inherits the same precedence rules.
    pub chat_template_kwargs: Option<crate::server::chat_template_kwargs::ChatTemplateKwargs>,

    /// cross-request prompt-prefix KV cache policy.
    ///
    /// Defaults to the baseline policy (enabled with 2 GiB / 1024 entries /
    /// 1-hour TTL). When `enabled = false` the store is skipped entirely at
    /// startup so no memory is reserved. CLI/env parsing for the individual
    /// fields is tracked separately in for now operators set
    /// the policy via the Rust API or keep the default.
    pub prompt_cache: crate::server::prompt_cache::PromptCacheConfig,

    /// (B11): server-wide KV cache mode.
    ///
    /// Resolved from `--cache-type-k`/`--cache-type-v` (llama-server split
    /// flags) or the legacy `--kv-cache-mode` shorthand.  Defaults to
    /// `KVCacheMode::Fp16` (bit-exact baseline). The model worker uses this
    /// when constructing per-sequence `CxxGenerator` instances so that every
    /// sequence in the batch sees the same KV quantization policy.
    pub kv_cache_mode: mlxcel_core::cache::KVCacheMode,

    /// batch KV cache quantization configuration for the
    /// continuous-batching scheduler.
    ///
    /// Resolved from the `--kv-bits`, `--kv-group-size`,
    /// `--kv-quant-scheme`, and `--kv-skip-last-layer` CLI flags. When
    /// disabled (`bits == 0`) the scheduler honours the legacy
    /// [`Self::kv_cache_mode`] field. When enabled, the resolved
    /// per-layer modes from
    /// [`mlxcel_core::cache::BatchKvQuantConfig::resolve_layer_modes`]
    /// take precedence (with the last layer forced to FP16 when
    /// `skip_last_layer == true`).
    pub batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig,

    /// upper bound on the **live KV window** of plain
    /// (non-sliding) `KVCache` instances.
    ///
    /// Mirrors upstream mlx-lm's
    /// [`BatchGenerator(max_kv_size=...)`](https://github.com/ml-explore/mlx-lm/pull/1106)
    /// parameter, with the same RoPE-faithful semantics as upstream's
    /// `RotatingKVCache`: when set, the batch scheduler calls
    /// [`mlxcel_core::cache::KVCache::trim_front`] after every prefill
    /// chunk and every decode step on caches whose `live_len()` exceeds
    /// the bound. `trim_front` advances `live_start` and physically slices
    /// the buffer head — it **does not** decrement `offset`, so K vectors
    /// rotated at write-time and Q vectors rotated at the current
    /// monotonic offset continue to see the correct relative position
    /// after the cap engages. See [`mlxcel_core::cache::KVCache::trim_front`]
    /// for the full position invariant.
    ///
    /// Sliding-window models that build their own [`RotatingKVCache`]
    /// internally (Gemma 3/4, Exaone 4, RecurrentGemma, Step 3.5, gpt-oss)
    /// already enforce a model-specific window and are unaffected by this
    /// cap. Models using KV quantization modes other than `Fp16` / `Int8`
    /// (`Turbo4*` / `Turbo3*`) also bypass the cap — `--max-kv-size` is not
    /// supported in combination with Turbo KV quantization in v1. The
    /// startup warning emitted in
    /// [`crate::server::batch::BatchScheduler::with_max_kv_size`] flags
    /// both the legacy `kv_cache_mode` flag and the per-layer modes
    /// resolved from `batch_kv_quant`.
    ///
    /// Resolved from the effective per-slot `--ctx-size` and the
    /// `--max-kv-size` CLI flag / `LLAMA_ARG_MAX_KV_SIZE` env var. The
    /// explicit max-KV value is validated by
    /// [`crate::server::cli_input::resolve_max_kv_size`] against the
    /// accepted range (`0` = disabled, or
    /// `[MAX_KV_SIZE_MIN, i32::MAX]`). If both are present, the lower value
    /// wins so the configured context window remains an upper bound.
    pub max_kv_size: Option<usize>,

    /// Paged KV pool block-budget directive (epic #116 #122 b3,
    /// `--kv-cache-budget`).
    ///
    /// `None` (the default) keeps the paged pool unbounded — the
    /// behaviour-preserving path. `Some(Bytes)` / `Some(Auto)` is resolved to
    /// a concrete block count on the worker thread (where the model geometry
    /// is known) by [`crate::memory_estimate::resolve_paged_block_budget`] and
    /// installed via
    /// [`crate::server::batch::BatchScheduler::with_paged_block_budget`]. Only
    /// meaningful for pool-backed (Fp16, dense-natural-backend) sequences.
    pub kv_cache_budget: Option<crate::memory_estimate::PagedBudgetDirective>,
    /// `--enable-vlm-prefix-cache` (#124 step c). Default off. When on, the
    /// scheduler permits VLM (image/audio) chat requests to adopt and donate
    /// KV prefixes for multi-turn same-image conversations; text-only and
    /// non-VLM behavior is unchanged.
    pub enable_vlm_prefix_cache: bool,
    /// Validated CORS allow-list origins (#244). `None` keeps the historical
    /// permissive policy (reflects any `Origin`); `Some(non_empty)` restricts
    /// cross-origin requests to exactly these origins. Built once at startup
    /// from `--allowed-origins` / `MLXCEL_ALLOWED_ORIGINS` and consumed by
    /// [`crate::server::create_app`].
    pub cors_allowed_origins: Option<Vec<axum::http::HeaderValue>>,
    /// Serving role for disaggregated paged KV serving (#126 B2), derived from
    /// `--node-role`. [`ServingMode::Hybrid`] (the default) is the single-node
    /// path and is byte-identical to a server with no distributed flags.
    /// `PrefillOnly` / `DecodeOnly` select the disaggregated serving role; the
    /// worker carries the mode so the serving-role coordinator can be wired
    /// onto the live scheduler in a later step (B2b).
    ///
    /// [`ServingMode`]: crate::distributed::disaggregated::ServingMode
    pub serving_mode: crate::distributed::disaggregated::ServingMode,
    /// Prefill-node peers a decode node receives handoffs from (disaggregated
    /// serving, #126 B3b2a). Threaded to the worker for the live serving role.
    pub prefill_peers: Vec<std::net::SocketAddr>,
    /// Decode-node peers a prefill node hands off to (disaggregated serving,
    /// #126 B3b2a). The first entry is the prefill node's KV handoff target.
    pub decode_peers: Vec<std::net::SocketAddr>,
    /// This node's own serving-role transport bind address (#126 B3b2a). `Some`
    /// on a non-hybrid node enables the live prefill/decode role loop; `None`
    /// keeps the standard single-node scheduler loop.
    pub serving_bind: Option<std::net::SocketAddr>,
    /// `--max-denoising-steps` (issue #217 phase 3). Serve-level override for
    /// the DiffusionGemma per-block denoising step cap; `None` keeps the
    /// checkpoint default. Only diffusion models read it.
    pub max_denoising_steps: Option<usize>,
    /// `--diffusion-sampler` (issue #217 phase 3). `"entropy-bound"` (default)
    /// or `"confidence-threshold"`. Only diffusion models read it.
    pub diffusion_sampler: String,
    /// `--diffusion-threshold` (issue #217 phase 3). Confidence threshold for
    /// the confidence-threshold sampler. Only diffusion models read it.
    pub diffusion_threshold: f32,

    /// Global N-gram loop-detection override (issue #432), resolved from the
    /// `MLXCEL_LOOP_DETECTION` env var at startup. `None` means "operator did
    /// not set a global override" so the per-family auto-enable policy applies;
    /// `Some(cfg)` forces that configuration for every request (including an
    /// explicitly disabled one), still overridable per-request. Precedence:
    /// explicit request > this global override > family auto-enable > disabled.
    pub loop_detection: Option<mlxcel_core::LoopDetectionConfig>,

    /// Whether the loaded model is in the Gemma 4 family (`Gemma4`,
    /// `Gemma4VLM`, or `Gemma4Unified`), resolved once at startup. Enables the
    /// engine-level loop-detection default-on for the family, unconditionally:
    /// it does not require tools or a `json_schema` response_format, so plain
    /// Gemma 4 chat is covered too. Defaults to `false` so non-Gemma-4 models
    /// keep the bit-exact baseline.
    pub model_is_gemma4_family: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            timeout_seconds: 600,
            model_alias: None,
            context_size: 0,
            // Serving-throughput default: admit up to 4 concurrent decode
            // sequences so weight reads amortize across the batch (#628). The
            // worker clamps this to 1 for non-batching model families.
            n_parallel: 4,
            enable_slots_endpoint: true,
            enable_props_endpoint: false,
            enable_metrics_endpoint: false,
            default_temperature: 0.8,
            default_top_p: 0.9,
            default_top_k: 40,
            default_min_p: 0.1,
            default_repetition_penalty: 1.0,
            default_repetition_context_size: 64,
            default_max_tokens: 512,
            default_seed: None,
            default_frequency_penalty: 0.0,
            default_presence_penalty: 0.0,
            default_dry_multiplier: 0.0,
            default_dry_base: 1.75,
            default_dry_allowed_length: 2,
            default_dry_penalty_last_n: 0,
            draft_model_path: None,
            num_draft_tokens: 3,
            // default to "auto-detect from drafter config"
            // when a drafter is supplied; the classic
            // `SpeculativeGenerator` path runs when no drafter is set.
            draft_kind: None,
            draft_block_size: None,
            // Serving-throughput default: batched decode up to 4 sequences
            // (#628). Clamped to 1 by the worker for non-batching families.
            max_batch_size: 4,
            max_queue_depth: 1024,
            audio_queue_depth: DEFAULT_AUDIO_QUEUE_DEPTH,
            audio_request_timeout_secs: DEFAULT_AUDIO_REQUEST_TIMEOUT_SECS,
            prefill_chunk_size: 512,
            enable_preemption: false,
            preemption_policy: PreemptionPolicy::default(),
            no_batch: false,
            // Serving-throughput default: batched prefill of up to 4 pending
            // requests (#628). No-ops for families without batched prefill.
            max_batch_prefill: 4,
            // #715: unset -> scheduler derives `max_batch_prefill * prefill_chunk_size`.
            max_batch_prefill_tokens: None,
            decode_storage_backend: DecodeStorageBackend::Auto,
            pipeline_parallel_runtime: None,
            remote_pipeline_stage: None,
            tensor_parallel: ShardConfig::default(),
            vision_cache_size: crate::vision::feature_cache::DEFAULT_VISION_CACHE_SIZE,
            lang_bias_config: None,
            reasoning_budget: None,
            chat_template_kwargs: None,
            prompt_cache: crate::server::prompt_cache::PromptCacheConfig::default(),
            kv_cache_mode: mlxcel_core::cache::KVCacheMode::Fp16,
            batch_kv_quant: mlxcel_core::cache::BatchKvQuantConfig::default(),
            max_kv_size: None,
            // Serving-throughput default guard (#628): pair the batched-decode
            // default with an `auto` paged KV budget so admission sheds load
            // instead of OOMing. Disable with `--kv-cache-budget none`.
            kv_cache_budget: Some(crate::memory_estimate::PagedBudgetDirective::Auto),
            enable_vlm_prefix_cache: false,
            cors_allowed_origins: None,
            serving_mode: crate::distributed::disaggregated::ServingMode::Hybrid,
            prefill_peers: Vec::new(),
            decode_peers: Vec::new(),
            serving_bind: None,
            max_denoising_steps: None,
            diffusion_sampler: "entropy-bound".to_string(),
            diffusion_threshold: 0.9,
            loop_detection: None,
            model_is_gemma4_family: false,
        }
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
