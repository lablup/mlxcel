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

    /// The conversation re-rendered with `add_generation_prompt = false`
    /// (issue #1143), carried from the route to the tokenization boundary.
    ///
    /// Mirrors the `prompt` / `prompt_token_ids` split the generate request
    /// already uses (issue #633): the dispatch thread tokenizes this into
    /// [`Self::history_prefix_tokens`] and clears the string, and the
    /// scheduler tokenizes it itself only when that pre-tokenization did not
    /// happen. `None` whenever the route produced no usable history render.
    pub history_prompt: Option<String>,

    /// History-boundary token vector for this request (issue #1143).
    ///
    /// Two stages, one meaning throughout: "the leading tokens of this
    /// prompt that are guaranteed to survive into the next turn".
    ///
    /// 1. Route / dispatch layer: the tokenization of
    ///    [`crate::server::chat_request::PreparedChatRequest::history_prompt`],
    ///    the same conversation re-rendered with `add_generation_prompt =
    ///    false`. Producing it by tokenizing the history *render* is what makes
    ///    it immune to the three divergence classes in epic #1148 (generation-
    ///    prompt-only scaffolds, thinking stripped from history, and
    ///    retokenization drift between sampled ids and canonical ids).
    /// 2. Scheduler (`enqueue_request`): clipped to the longest common prefix
    ///    with the request's live prompt tokens, so the vector is by
    ///    construction a genuine prefix of what the model prefills. The clip
    ///    also drops the one or two tokens that a BPE merge across the
    ///    history/scaffold seam would otherwise make unstable.
    ///
    /// `None` when the route could not produce a usable history render
    /// (multimodal request, template fallback, non-prefix render), when no
    /// tokenizer was available on the dispatch thread, or when the clipped
    /// prefix turned out too short to be worth a snapshot.
    pub history_prefix_tokens: Option<Vec<i32>>,
}

/// Bridge between server request params and `mlxcel-core` `SamplingConfig`.
#[derive(Debug, Clone)]
pub struct ServerGenerateOptions {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_sequences: Option<Vec<String>>,
    /// b10621 `--ignore-eos` / `ignore_eos` (#1436): suppress every
    /// end-of-generation token with a `-inf` logit bias at enqueue time so
    /// the model keeps generating until the token budget or a string stop,
    /// exactly as upstream's EOG bias does. Default `false`.
    pub ignore_eos: bool,
    /// Request priority for prefill queue ordering.
    pub priority: RequestPriority,
    /// Log probability configuration; disabled by default (zero overhead).
    pub logprobs: LogprobsConfig,
    /// b10621 DRY breaker STRINGS in effect for this request (#1485):
    /// `Some` is the set the scheduler derives the breaker head map from
    /// against the vocabulary at enqueue time (the request's own strings, or
    /// the server-wide default set); `None` means the request supplied
    /// exact-id breakers (the mlxcel-native OpenAI surface) and no string
    /// derivation runs.
    pub dry_breaker_strings: Option<Vec<String>>,
    /// Resolved numeric logit biases (#1485): request `logit_bias` when
    /// present, else the server-wide `--logit-bias` set. Merged into the
    /// sequence's token-bias map at enqueue time, before the `ignore_eos`
    /// EOG suppression, upstream's bias-then-EOG order.
    pub logit_bias: Vec<(i32, f32)>,
    /// Text-keyed logit biases (#1485): tokenized (special parsing off) at
    /// enqueue time; the bias applies to every resulting token.
    pub logit_bias_texts: Vec<(String, f32)>,
    /// b10621 `post_sampling_probs` (#1485): the native route's `n_probs`
    /// report is taken from the post-sampling-chain distribution (linear
    /// probabilities) instead of the raw-logit log probabilities.
    pub post_sampling_probs: bool,
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

    /// b10621 `reasoning_control` (#1444): when the request armed realtime
    /// reasoning control, this is the shared "end reasoning now" flag the
    /// sequence's [`crate::server::thinking_budget::ThinkingState`] polls at
    /// every sampling step. `POST /v1/chat/completions/control` with
    /// `action: "reasoning_end"` sets it through the
    /// [`crate::server::completion_control::CompletionControlRegistry`].
    /// `None` means the request did not arm control; the sampler pays
    /// nothing. `Some` keeps the thinking tracker active even when no
    /// budget is configured, which is upstream's "create the budget sampler
    /// on demand" behavior.
    pub reasoning_control: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,

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

    /// Per-request context-retention overrides (#1472): b10621's `n_keep` /
    /// `n_discard` native request fields. Both `None` on every route that
    /// does not declare them; the scheduler falls back to the server-wide
    /// `--keep` and the half-window discard default.
    pub retention: RetentionOverride,

    /// Effective LoRA adapter user scales for this request (#1439): the
    /// server-default snapshot taken at admission, or the request's own
    /// `lora` field resolved through upstream's rule. `None` when the server
    /// has no runtime-LoRA state. Batches only ever contain one snapshot,
    /// and the executing worker applies it before the batch's forwards, so a
    /// concurrent `POST /lora-adapters` never changes a generation already
    /// in flight.
    pub lora_scales: Option<std::sync::Arc<Vec<f32>>>,
    /// b10621 grammar surfaces in effect for this request (#1485): the GBNF or
    /// schema the constraint above was compiled from, plus the lazy-trigger
    /// declaration.
    ///
    /// `structured` is what the scheduler runs; this is what `/props` and the
    /// response's `generation_settings` report, and the two are built from the
    /// same resolution so they cannot disagree. `None` when the request asked
    /// for no constraint, which is every request that predates #1485.
    pub grammar: Option<std::sync::Arc<crate::server::grammar::GrammarSpec>>,

    /// per-request Gemma 4 image soft-token budget.
    ///
    /// `None` means "no override": the Gemma 4 preprocessor uses the budget
    /// read from the checkpoint's `processor_config.json` at load time, which
    /// is the behavior for every request that does not set `detail` or
    /// `max_soft_tokens` on an `image_url` content part.
    ///
    /// Resolved and validated in the route layer (see
    /// [`crate::server::types::request::resolve_request_image_soft_tokens`]),
    /// so by the time the scheduler sees this field it is already known to be
    /// on the supported ladder. Ignored by every non-Gemma-4 model.
    pub image_soft_tokens: Option<usize>,
}

/// Per-request context-retention overrides (b10621 `n_keep` / `n_discard`,
/// #1472).
///
/// `n_keep` is the number of leading prompt tokens retained across a context
/// shift (`-1` = the whole initial prompt); `None` inherits the server-wide
/// `--keep`. `n_discard` is how many tokens past the retained prefix each
/// shift drops; `None` (and upstream's `0`) resolve to half of the
/// non-retained window. Validated at the route; the scheduler receives only
/// in-domain values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RetentionOverride {
    pub n_keep: Option<i64>,
    pub n_discard: Option<i64>,
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

/// Default `--embedding-batch-size`: texts per embedding forward pass.
pub const DEFAULT_EMBEDDING_BATCH_SIZE: usize = crate::embeddings::DEFAULT_EMBEDDING_BATCH_SIZE;

/// Default bound for the embedding worker command queue; tracks the audio
/// value so both single-thread workers shed load the same way.
pub const DEFAULT_EMBEDDING_QUEUE_DEPTH: usize = DEFAULT_AUDIO_QUEUE_DEPTH;

/// Default per-request reply timeout (seconds) for the embedding worker;
/// tracks the audio value.
pub const DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS: u64 = DEFAULT_AUDIO_REQUEST_TIMEOUT_SECS;

/// Default `--rerank-batch-size`: query/document pairs per forward pass.
///
/// This is the text default; a multimodal reranker lowers it to its own
/// [`crate::rerank::DEFAULT_RERANK_VL_BATCH_SIZE`] unless the flag is given,
/// because each of its rows carries a full image's worth of visual tokens.
pub const DEFAULT_RERANK_BATCH_SIZE: usize = crate::rerank::DEFAULT_RERANK_BATCH_SIZE;

/// What this server is allowed to serve, per b10621's `--embeddings` and
/// `--reranking` (#1452).
///
/// b10621 has one model and one set of weights, so those flags are a
/// server-wide restriction rather than a worker selection. mlxcel reproduces
/// the restriction half here: generation routes answer the same 501 they answer
/// with no chat model, and the body names the flag that turned them off, so an
/// operator can tell "this deployment is embedding-only" from "the chat model
/// failed to load".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingServingMode {
    /// No mode flag: every loaded worker serves its routes.
    #[default]
    Any,
    /// `--embeddings`: embedding routes only.
    EmbeddingOnly,
    /// `--reranking`: reranking routes only.
    RerankOnly,
}

impl EmbeddingServingMode {
    /// The b10621 flag that selected this mode, or `None` for [`Self::Any`].
    #[must_use]
    pub fn flag(self) -> Option<&'static str> {
        match self {
            Self::Any => None,
            Self::EmbeddingOnly => Some("--embeddings"),
            Self::RerankOnly => Some("--reranking"),
        }
    }

    /// Whether generation is refused in this mode.
    #[must_use]
    pub fn blocks_generation(self) -> bool {
        !matches!(self, Self::Any)
    }
}

/// Optional compatibility alias emitted next to `reasoning_content` on Chat
/// Completions responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningAliasField {
    /// Emit only the established `reasoning_content` field.
    None,
    /// Also emit an identical `reasoning` field.
    #[default]
    Reasoning,
}

impl ReasoningAliasField {
    /// Whether Chat Completions should duplicate reasoning into `reasoning`.
    #[must_use]
    pub const fn emits_reasoning(self) -> bool {
        matches!(self, Self::Reasoning)
    }

    /// The CLI spelling for this policy.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Reasoning => "reasoning",
        }
    }
}

impl std::fmt::Display for ReasoningAliasField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ReasoningAliasField {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "reasoning" => Ok(Self::Reasoning),
            other => Err(format!(
                "unknown reasoning alias field '{other}'; expected one of: none, reasoning"
            )),
        }
    }
}

/// Request-scoped server defaults that can be replaced without rebuilding the
/// model worker. A request clones one immutable snapshot and keeps it for its
/// whole lifetime, so a concurrent settings update cannot split prompt
/// rendering and generation across different defaults.
#[derive(Debug, Clone)]
pub struct LiveSettings {
    /// Pre-resolved language bias carried by requests. Not part of the JSON
    /// settings surface; it is derived from `lang_bias_config` before publish.
    pub(crate) resolved_token_bias: mlxcel_core::sampling::TokenBiasMap,
    pub timeout_seconds: u64,
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
    /// b10621 DRY sequence-breaker STRINGS (#1485). Live-updatable through
    /// the runtime-settings surface; the head-token data the sampler compares
    /// against is derived from these strings per request, at enqueue time,
    /// where the vocabulary is available.
    pub default_dry_sequence_breakers: Vec<String>,
    pub lang_bias_config: Option<LangBiasConfig>,
    pub reasoning_budget: Option<crate::server::thinking_budget::ThinkingBudget>,
    pub chat_template_kwargs: Option<crate::server::chat_template_kwargs::ChatTemplateKwargs>,
    pub loop_detection: Option<mlxcel_core::LoopDetectionConfig>,
    pub max_denoising_steps: Option<usize>,
    pub diffusion_sampler: String,
    pub diffusion_threshold: f32,
}

/// Server configuration derived from CLI-compatible startup arguments.
///
/// Default values intentionally track `llama-server` behavior where practical
/// so route handlers can apply one consistent set of defaults.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Vertex AI (GCP) custom-container compat routes (b10621, #1456).
    /// Resolved once at startup from `AIP_MODE=PREDICTION` and the other
    /// `AIP_*` variables; `None` (the default) mounts nothing.
    pub gcp: Option<crate::server::gcp_compat::GcpRoutes>,
    /// Where a model's thoughts are reported (b10621 `--reasoning-format` /
    /// `LLAMA_ARG_THINK`, issue #1447).
    pub reasoning_format: crate::server::ReasoningFormat,
    /// OpenRouter-compatible alias emitted next to `reasoning_content` on
    /// Chat Completions messages and deltas.
    pub reasoning_alias_field: ReasoningAliasField,
    /// b10621 `--skip-chat-parsing`: force a pure content parser, so reasoning
    /// and tool calls stay in `message.content`.
    pub skip_chat_parsing: bool,
    /// b10621 `--no-prefill-assistant`: a trailing assistant message is a
    /// complete message rather than a prefix to continue.
    pub no_prefill_assistant: bool,
    /// b10621 `--reasoning-budget-message`: text injected before the
    /// end-of-thinking tag when the reasoning budget is exhausted.
    pub reasoning_budget_message: Option<String>,
    /// Configured API keys (#1437). Empty disables authentication, matching
    /// b10621's empty `api_keys` vector. `Debug` prints a count, never key
    /// material.
    pub api_keys: crate::server::ApiKeys,
    /// Per-request decode watchdog, in seconds. Sourced from
    /// `--decode-timeout` / `MLXCEL_DECODE_TIMEOUT`. Before #1432 this was
    /// `--timeout`, which now carries the b10621 HTTP socket read/write
    /// timeout instead; see [`crate::server::transport`].
    pub decode_timeout_seconds: u64,
    /// b10621 `--api-prefix` (`LLAMA_ARG_API_PREFIX`). Empty (the default)
    /// mounts every route at the root. A non-empty value is a validated path
    /// with a leading and no trailing slash; `create_app` nests the whole
    /// route set under it.
    pub api_prefix: String,
    /// b10621 `--sse-ping-interval` (`LLAMA_ARG_SSE_PING_INTERVAL`). `None`
    /// disables the SSE comment pings (`-1` upstream).
    pub sse_ping_interval: Option<std::time::Duration>,
    /// Served model id: the first entry of `model_aliases`, or `None` when
    /// `--alias` was not given (the provider's own id is then served).
    pub model_alias: Option<String>,
    /// Every name `--alias` supplied, primary first (issue #1434).
    ///
    /// b10621 takes `--alias a,b,c` and keeps all three as API-visible names.
    /// mlxcel serves the first; the rest are carried here so the `/v1/models`
    /// `aliases` array (#1438) can report them without re-parsing the CLI
    /// value. Empty when `--alias` was not given.
    pub model_aliases: Vec<String>,
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
    /// Opt-in runtime settings management endpoint.
    pub enable_settings_endpoint: bool,
    /// `--slot-save-path`: directory for `POST /slots/:id_slot` save/restore
    /// files. `None` disables the slot actions, b10621's default (#1440).
    pub slot_save_path: Option<std::path::PathBuf>,
    /// `--tags`: informational model tags reported in the `/v1/models` model
    /// object; never used for routing, exactly b10621's contract (#1438).
    pub model_tags: Vec<String>,
    /// The b10621 multi-adapter LoRA specification (#1439): what `--lora` /
    /// `--lora-scaled` / `--lora-init-without-apply` resolved to, in listed
    /// order. Fused at model load; `GET /lora-adapters` reports it and the
    /// per-request `lora` field is validated against it. Empty when the
    /// legacy single-adapter `adapter_path` plumbing (one adapter, scale 1,
    /// applied) serves the request instead.
    pub lora_adapters: Vec<crate::lora::LoraAdapterSpec>,
    /// Runtime (unfused) LoRA serving state (#1439). `Some` when adapters
    /// serve unfused: `GET /lora-adapters` reports its live server scales,
    /// `POST /lora-adapters` replaces them, requests snapshot them at
    /// admission, and the worker applies each batch's snapshot to the shared
    /// handles the layers read. `None` under `--lora-fuse`, parallel
    /// loaders, or without adapters, where the fused-boundary refusals
    /// apply.
    pub lora_runtime: Option<std::sync::Arc<crate::lora::RuntimeLoraSet>>,
    /// `--embd-normalize`: the server-wide embedding normalization, `None`
    /// when the operator did not choose one and the checkpoint's own
    /// `normalize` flag decides (#1452).
    pub embd_normalize: Option<crate::embeddings::EmbdNormalize>,
    /// `--embeddings` / `--reranking`: generation is off and this server serves
    /// only its side-model routes (#1452). Carries the flag that turned it off
    /// so the 501 body can name it.
    pub embedding_serving_mode: EmbeddingServingMode,
    /// `--spm-infill`: use the Suffix/Prefix/Middle ordering on `POST /infill`
    /// instead of the default Prefix/Suffix/Middle (#1442).
    ///
    /// A model's FIM training fixes one of the two orderings, and prompting in
    /// the wrong one produces a fluent but wrong completion rather than an
    /// error, so this is a correctness switch and not a preference.
    pub spm_infill: bool,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub default_top_k: i32,
    pub default_min_p: f32,
    /// Server-wide default for locally typical sampling (`1.0` = disabled),
    /// set by `--typical` / `--typical-p` (#1377).
    pub default_typical_p: f32,
    /// Server-wide default for top-n-sigma (`0.0` = disabled), set by
    /// `--top-nsigma` / `--top-n-sigma` (#1436). The b10621 flag default is
    /// `-1.0` (its disabled sentinel); startup folds any non-positive or
    /// non-finite value to mlxcel's `0.0` disabled form.
    pub default_top_n_sigma: f32,
    /// Server-wide default XTC probability (`0.0` = disabled), set by
    /// `--xtc-probability` (#1436).
    pub default_xtc_probability: f32,
    /// Server-wide default XTC threshold, set by `--xtc-threshold` (#1436).
    /// b10621 domain `0.0..=1.0`; values above `0.5` make XTC inert.
    pub default_xtc_threshold: f32,
    /// b10621 `--ignore-eos` (#1436): server-wide default for suppressing
    /// end-of-generation tokens.
    pub default_ignore_eos: bool,
    /// b10621 `-r` / `--reverse-prompt` (#1436): server-wide stop strings
    /// merged into every request's stop set.
    pub default_stop_sequences: Vec<String>,
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
    /// Server-wide DRY sequence breaker STRINGS, from
    /// `--dry-sequence-breaker`, following b10621's value domain (#1485):
    /// when the flag is absent this holds the b10621 default set (`\n`,
    /// `:`, `"`, `*`); the flag's values replace it, and the `none` sentinel
    /// leaves it empty. Breaker token data (the head map the sampler
    /// consumes) is derived from these strings against the vocabulary at
    /// enqueue time, where the tokenizer lives; a request carrying its own
    /// `dry_sequence_breakers` field overrides this set wholesale.
    pub default_dry_sequence_breakers: Vec<String>,
    /// Server-wide constrained-decoding default from `--grammar`,
    /// `--grammar-file`, `--json-schema` or `--json-schema-file` (#1485).
    ///
    /// b10621 stores all four in one field and lets the last flag on the
    /// command line win; the winner is resolved in
    /// [`crate::cli::grammar_args`] before the config is built. Every request
    /// that does not carry its own non-empty `grammar` or a `json_schema`
    /// inherits this, exactly as upstream's request params start from the
    /// server defaults. `None` when no flag was given.
    pub default_grammar: Option<crate::server::grammar::GrammarSpec>,
    /// Server-wide mirostat mode (`--mirostat`, #1485): `0` disabled, `1`
    /// Mirostat, `2` Mirostat 2.0; validated to that domain at startup.
    pub default_mirostat: i32,
    /// Server-wide mirostat target entropy tau (`--mirostat-ent`, #1485).
    pub default_mirostat_tau: f32,
    /// Server-wide mirostat learning rate eta (`--mirostat-lr`, #1485).
    pub default_mirostat_eta: f32,
    /// Server-wide dynamic-temperature range (`--dynatemp-range`, #1485;
    /// `0.0` = disabled).
    pub default_dynatemp_range: f32,
    /// Server-wide dynamic-temperature exponent (`--dynatemp-exp`, #1485).
    pub default_dynatemp_exponent: f32,
    /// Server-wide adaptive-p target (`--adaptive-target`, #1485; negative =
    /// disabled). Takes effect only when the sampler list names
    /// `adaptive_p` (see [`ServerConfig::default_adaptive_p_named`]),
    /// exactly as b10621 activates the sampler solely through
    /// `params.samplers`.
    pub default_adaptive_target: f32,
    /// Server-wide adaptive-p EMA decay (`--adaptive-decay`, #1485),
    /// clamped to `0.0..=0.99` at startup as upstream clamps it at sampler
    /// init.
    pub default_adaptive_decay: f32,
    /// Whether the server-wide sampler list (`--samplers` /
    /// `--sampler-seq`) named the `adaptive_p` stage (#1485).
    pub default_adaptive_p_named: bool,
    /// Server-wide logit biases from repeated `-l` / `--logit-bias`
    /// TOKEN(+/-)BIAS values (#1485), applied to every request that does not
    /// carry its own `logit_bias` field (a request field replaces this set
    /// wholesale, as b10621's field handler clears and rebuilds the list).
    pub default_logit_bias: Vec<(i32, f32)>,
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
    /// `--embedding-model`: a second checkpoint served on `/v1/embeddings`
    /// next to the chat model. `None` means "use `-m` when it is itself an
    /// embedding checkpoint, otherwise serve no embeddings".
    pub embedding_model_path: Option<std::path::PathBuf>,
    /// `--embedding-batch-size`: texts per embedding forward pass. See
    /// [`DEFAULT_EMBEDDING_BATCH_SIZE`].
    pub embedding_batch_size: usize,
    /// `--embedding-max-length`: lowers the token cap derived from the
    /// checkpoint. `None` keeps the derived value.
    pub embedding_max_length: Option<usize>,
    /// Bound on the embedding worker command queue (admission control). See
    /// [`DEFAULT_EMBEDDING_QUEUE_DEPTH`].
    pub embedding_queue_depth: usize,
    /// Per-request reply timeout for the embedding worker, in seconds. A `0`
    /// falls back to the default. See [`DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS`].
    pub embedding_request_timeout_secs: u64,
    /// `--reranker-model`: a checkpoint served on `/v1/rerank` next to the
    /// chat model. `None` means "use `-m` when it is itself a
    /// sequence-classifier reranker, otherwise serve no reranking". The
    /// generative rerankers are only reachable through this flag: their
    /// checkpoints are indistinguishable from chat models.
    pub reranker_model_path: Option<std::path::PathBuf>,
    /// `--rerank-batch-size`: query/document pairs per forward pass. A `0`
    /// takes the loaded reranker kind's own default. See
    /// [`DEFAULT_RERANK_BATCH_SIZE`].
    pub rerank_batch_size: usize,
    /// Number of tokens per prefill chunk. When 0, chunking is disabled and
    /// the full prompt is prefilled in a single pass.
    pub prefill_chunk_size: usize,
    /// #1011 prefill fairness interval (`--prefill-grant-interval`): decode
    /// ticks a parked chunked prefill yields before the scheduler grants it
    /// one, bounding the admitted request's time to first token. `None` lets
    /// the scheduler resolve `MLXCEL_PREFILL_GRANT_INTERVAL` or the shipped
    /// default; `Some(0)` disables the grant (pre-#1011 unbounded wait).
    pub prefill_grant_interval: Option<usize>,
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
    /// parameter, with the same RoPE-faithful, attention-sink-preserving
    /// semantics as upstream's `RotatingKVCache(keep=...)`: when set, the
    /// batch scheduler calls
    /// [`mlxcel_core::cache::KVCache::trim_front_keep_sink`] after every
    /// prefill chunk and every decode step on caches whose `live_len()`
    /// exceeds the bound, pinning a small leading attention-sink prefix and
    /// dropping the excess tokens that follow it rather than the oldest
    /// tokens overall. `trim_front_keep_sink` advances `live_start` and
    /// physically rearranges the buffer accordingly, it **does not**
    /// decrement `offset`, so K vectors rotated at write-time and Q
    /// vectors rotated at the current monotonic offset continue to see the
    /// correct relative position after the cap engages. See
    /// [`mlxcel_core::cache::KVCache::trim_front_keep_sink`] for the full
    /// position invariant.
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

    /// Whether the scheduler may shift (front-discard) a sequence's context
    /// to make room at [`Self::max_kv_size`] (#1472, b10621
    /// `--context-shift`). Off (the default, upstream's too), an over-long
    /// prompt is refused at admission and a generation that reaches the bound
    /// stops with `truncated: true` and `stop_type: "limit"`.
    pub context_shift: bool,

    /// Server-wide default for the retained leading tokens on a context
    /// shift (#1472, b10621 `--keep`; `-1` = the whole initial prompt).
    /// Overridden per request by the native `n_keep` field.
    pub n_keep: i64,

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
    /// Resolved CORS policy (#244, realigned onto b10621 in #1432). Built
    /// once at startup from `--cors-origins` / `--cors-methods` /
    /// `--cors-headers` / `--cors-credentials`, or from the mlxcel-native
    /// `--allowed-origins` allow-list, and consumed by
    /// [`crate::server::create_app`].
    pub cors_policy: crate::server::CorsPolicy,
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
    /// engine-level loop-detection default-on for the family, for tool-shaped
    /// requests (issues #967 and #977). Plain and grammar-only Gemma 4 requests
    /// are not covered. Defaults to `false` so non-Gemma-4 models keep the
    /// bit-exact baseline.
    pub model_is_gemma4_family: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            gcp: None,
            reasoning_format: crate::server::ReasoningFormat::default(),
            reasoning_alias_field: ReasoningAliasField::default(),
            skip_chat_parsing: false,
            no_prefill_assistant: false,
            reasoning_budget_message: None,
            api_keys: crate::server::ApiKeys::default(),
            decode_timeout_seconds: crate::server::transport::DEFAULT_DECODE_TIMEOUT_SECS,
            api_prefix: String::new(),
            sse_ping_interval: Some(std::time::Duration::from_secs(
                crate::server::transport::DEFAULT_SSE_PING_INTERVAL_SECS as u64,
            )),
            model_alias: None,
            model_aliases: Vec::new(),
            context_size: 0,
            // Serving-throughput default: admit up to 4 concurrent decode
            // sequences so weight reads amortize across the batch (#628). The
            // worker clamps this to 1 for non-batching model families.
            n_parallel: 4,
            enable_slots_endpoint: true,
            enable_props_endpoint: false,
            slot_save_path: None,
            model_tags: Vec::new(),
            lora_adapters: Vec::new(),
            lora_runtime: None,
            spm_infill: false,
            embd_normalize: None,
            embedding_serving_mode: EmbeddingServingMode::Any,
            enable_metrics_endpoint: false,
            enable_settings_endpoint: false,
            default_temperature: 0.8,
            default_top_p: 0.95,
            default_top_k: 40,
            default_min_p: 0.05,
            default_typical_p: 1.0,
            default_top_n_sigma: 0.0,
            default_xtc_probability: 0.0,
            default_xtc_threshold: 0.1,
            default_ignore_eos: false,
            default_stop_sequences: Vec::new(),
            default_repetition_penalty: 1.0,
            default_repetition_context_size: 64,
            default_max_tokens: 512,
            default_seed: None,
            default_frequency_penalty: 0.0,
            default_presence_penalty: 0.0,
            default_dry_multiplier: 0.0,
            default_dry_base: 1.75,
            default_dry_allowed_length: 2,
            default_dry_penalty_last_n: 64,
            default_dry_sequence_breakers: crate::server::dry_breakers::default_breaker_strings(),
            default_grammar: None,
            default_mirostat: 0,
            default_mirostat_tau: 5.0,
            default_mirostat_eta: 0.1,
            default_dynatemp_range: 0.0,
            default_dynatemp_exponent: 1.0,
            default_adaptive_target: -1.0,
            default_adaptive_decay: 0.9,
            default_adaptive_p_named: false,
            default_logit_bias: Vec::new(),
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
            embedding_model_path: None,
            embedding_batch_size: DEFAULT_EMBEDDING_BATCH_SIZE,
            embedding_max_length: None,
            embedding_queue_depth: DEFAULT_EMBEDDING_QUEUE_DEPTH,
            embedding_request_timeout_secs: DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECS,
            reranker_model_path: None,
            rerank_batch_size: DEFAULT_RERANK_BATCH_SIZE,
            prefill_chunk_size: 512,
            // #1011: unset -> scheduler resolves the env override / default.
            prefill_grant_interval: None,
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
            context_shift: false,
            n_keep: 0,
            // Serving-throughput default guard (#628): pair the batched-decode
            // default with an `auto` paged KV budget so admission sheds load
            // instead of OOMing. Disable with `--kv-cache-budget none`.
            kv_cache_budget: Some(crate::memory_estimate::PagedBudgetDirective::Auto),
            enable_vlm_prefix_cache: false,
            cors_policy: crate::server::CorsPolicy::default(),
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

impl ServerConfig {
    /// Build the startup snapshot for request-scoped live settings.
    #[must_use]
    pub fn live_settings(&self) -> LiveSettings {
        LiveSettings {
            resolved_token_bias: mlxcel_core::sampling::TokenBiasMap::default(),
            timeout_seconds: crate::server::model_provider::effective_decode_timeout_seconds(
                self.decode_timeout_seconds,
            ),
            default_temperature: self.default_temperature,
            default_top_p: self.default_top_p,
            default_top_k: self.default_top_k,
            default_min_p: self.default_min_p,
            default_repetition_penalty: self.default_repetition_penalty,
            default_repetition_context_size: self.default_repetition_context_size,
            default_max_tokens: self.default_max_tokens,
            default_seed: self.default_seed,
            default_frequency_penalty: self.default_frequency_penalty,
            default_presence_penalty: self.default_presence_penalty,
            default_dry_multiplier: self.default_dry_multiplier,
            default_dry_base: self.default_dry_base,
            default_dry_allowed_length: self.default_dry_allowed_length,
            default_dry_penalty_last_n: self.default_dry_penalty_last_n,
            default_dry_sequence_breakers: self.default_dry_sequence_breakers.clone(),
            lang_bias_config: self.lang_bias_config.clone(),
            reasoning_budget: self.reasoning_budget,
            chat_template_kwargs: self.chat_template_kwargs.clone(),
            loop_detection: self.loop_detection,
            max_denoising_steps: self.max_denoising_steps,
            diffusion_sampler: self.diffusion_sampler.clone(),
            diffusion_threshold: self.diffusion_threshold,
        }
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
