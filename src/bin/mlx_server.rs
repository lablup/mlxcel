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

use clap::{Args as ClapArgs, Parser, Subcommand};
use std::path::PathBuf;

use mlxcel::cli::batch_quant_args::BatchKvQuantArgs;
use mlxcel::cli::speculative_args::{
    SpeculativeArgs, env_fallback_draft_block_size, env_fallback_draft_kind,
};
use mlxcel::cli::turbo_args::TurboKvCacheArgs;
use mlxcel::downloader::{
    DownloadArgs, DownloadOptions, download_repo, resolve_model_source_with_override,
};
use mlxcel::lang_bias::LangBiasCliArgs;
use mlxcel::server::{
    ServerStartupInput, env_fallback_apc_block_size, env_fallback_apc_enabled,
    env_fallback_apc_hash, env_fallback_apc_num_blocks, env_fallback_cache_type_k,
    env_fallback_cache_type_v, env_fallback_chat_template_kwargs, env_fallback_kv_bits,
    env_fallback_kv_group_size, env_fallback_kv_quant_scheme, env_fallback_kv_skip_last_layer,
    env_fallback_lang_bias, env_fallback_lang_bias_include_byte_fragments,
    env_fallback_prompt_cache_capacity_bytes, env_fallback_prompt_cache_enabled,
    env_fallback_prompt_cache_max_entries, env_fallback_prompt_cache_min_prefix,
    env_fallback_prompt_cache_ttl, env_fallback_reasoning_budget, long_cli_flag_was_set,
    start_server,
};

/// mlxcel-server: llama-server compatible HTTP server for MLX inference
///
/// Drop-in replacement for llama-server (llama.cpp) using Apple Silicon MLX or
/// CUDA backends. Supports OpenAI-compatible API endpoints and llama-server
/// native endpoints.
///
/// Usage modes:
///
/// 1. Legacy flag-only invocation (backward-compatible default):
///    `mlxcel-server -m models/foo --port 8080`
///    `mlxcel-server -m mlx-community/Qwen3-4B-4bit --port 8080`
///    With no subcommand, the binary boots the HTTP server using the
///    flattened server flags below. `-m/--model` accepts the same local-path
///    or HuggingFace `owner/name` repo-id values as `mlxcel serve -m`.
///
/// 2. Subcommand mode:
///    `mlxcel-server download <REPO_ID>`
///    `download` fetches a HuggingFace model snapshot using the same
///    downloader the `mlxcel` CLI uses. Server flags are
///    rejected when a subcommand is supplied.
#[derive(Parser, Debug)]
#[command(
    name = "mlxcel-server",
    author = "Lablup Inc.",
    version,
    about = "llama-server compatible HTTP server for MLX inference on Apple Silicon and CUDA GPUs",
    args_conflicts_with_subcommands = true,
    flatten_help = true,
    verbatim_doc_comment,
    after_help = "\
Tensor Parallel Runtime:
  Current multi-rank support: dense Llama, Qwen2/2.5, Qwen3, Qwen3.5 text, Gemma 3 text, Gemma 4 text, ERNIE 4.5, Hunyuan v1 Dense
  Current constraints: --tp-embedding-mode replicated, --tp-lm-head-mode replicated
                       LoRA unsupported, server batching supported for listed dense runtimes
                       except Gemma 4 E2B-style conservative fallback checkpoints

Model store:
  -m/--model accepts either a local path or a HuggingFace owner/name repo-id.
  Repo-ids are resolved exactly like `mlxcel serve -m`: legacy ./models/<name>,
  then the HuggingFace cache, then the mlxcel store, with auto-download on miss.
  Use --models-dir (or MLXCEL_MODELS_DIR) to point the mlxcel store at another
  volume; snapshots live at <root>/<owner>/<name> under that root.

Remote Pipeline Parallel Example (TCP):
  1. Generate a shared cluster config:
       CLUSTER_NAME=studio-pp \\
       TRANSPORT_BACKEND=tcp \\
       COORDINATOR_CONTROL_ADDR=192.168.1.22:19000 \\
       STAGE0_ADDR=192.168.1.22:19001 \\
       STAGE1_ADDR=192.168.1.24:19001 \\
       scripts/benchmark_pipeline_remote_rollout.sh write-config \\
         examples/distributed/generated_pipeline_remote_2node_tcp.toml

  2. Start stage-1 on machine B:
       mlxcel-server -m models/llama-3.2-1b-4bit \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id stage-1 --host 0.0.0.0 --port 18081 --no-warmup

  3. Start stage-0 on machine A:
       mlxcel-server -m models/llama-3.2-1b-4bit \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id stage-0 --host 0.0.0.0 --port 18081 --no-warmup

  4. Start the coordinator on machine A:
       mlxcel-server -m models/llama-3.2-1b-4bit --alias llama-remote-pp \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id coordinator --host 0.0.0.0 --port 18080 \\
         --parallel 2 --max-batch-size 2 --pp-micro-batch-size 2 \\
         --metrics --no-warmup

Thunderbolt mode:
  Use the same workflow with TRANSPORT_BACKEND=thunderbolt and each node's
  Thunderbolt Bridge IP (for example 169.254.x.x). The current Thunderbolt
  path uses the shared TCP transport core over the Bridge network.

Subcommands:
  download <REPO_ID>    Fetch a HuggingFace model snapshot into the global store
                        (${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>);
                        reuses an existing HuggingFace cache copy. --local-dir opts out.

See also: docs/distributed.md"
)]
struct Cli {
    /// Subcommand to run. When omitted, the binary boots the HTTP server
    /// using the flattened [`ServerArgs`] flags (legacy invocation).
    #[command(subcommand)]
    command: Option<Commands>,

    /// Server-start arguments. Mutually exclusive with `command` (enforced by
    /// `args_conflicts_with_subcommands = true` on the parent command).
    #[command(flatten)]
    server: ServerArgs,
}

/// Clap value parser: an f32 in the closed interval [0, 1].
///
/// Used by: `--diffusion-threshold` (fail fast at startup instead of
/// surfacing a per-request engine error under the confidence sampler).
fn parse_unit_interval(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("not a number: {e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("must be between 0 and 1, got {v}"))
    }
}

/// Subcommands supported by `mlxcel-server`.
///
/// The set is intentionally narrow: only operations that legitimately need to
/// share the server binary (currently just model downloading) live here. The
/// long-form server-start flags remain at the top level for full backward
/// compatibility with existing scripts and llama-server drop-in usage.
#[derive(Subcommand, Debug)]
enum Commands {
    /// Download a HuggingFace model repository snapshot
    Download(DownloadArgs),
}

/// clap value parser for `--kv-cache-budget`: a raw byte count or the literal
/// `auto` (epic #116 #122 b3).
fn parse_kv_cache_budget(s: &str) -> Result<mlxcel::memory_estimate::PagedBudgetDirective, String> {
    s.parse()
}

#[derive(ClapArgs, Debug)]
struct ServerArgs {
    /// Path to the model directory, or a HuggingFace `owner/name` repo-id.
    ///
    /// Required when running in legacy server-start mode (no subcommand).
    /// Modeled as `Option<PathBuf>` so the `download` subcommand can be
    /// invoked without supplying `-m`. An existing path is used as-is; a
    /// repo-id is resolved from a legacy `./models/<name>` directory, the
    /// HuggingFace cache, or the mlxcel store, and auto-downloaded on a miss.
    /// A bare name without a slash (e.g. `Qwen3-4B-4bit`) is resolved as
    /// `mlx-community/<name>`; override the org with the
    /// `MLXCEL_DEFAULT_ORG` environment variable.
    #[arg(
        short = 'm',
        long = "model",
        env = "LLAMA_ARG_MODEL",
        value_name = "PATH_OR_REPO_ID"
    )]
    model: Option<PathBuf>,

    /// Model-store root for resolving / downloading an `owner/name` repo-id.
    ///
    /// Sets the directory that directly holds snapshots, so a repo-id resolves
    /// to / downloads at `<PATH>/<owner>/<name>` (no extra `models/` subdir).
    /// Overrides the `MLXCEL_MODELS_DIR` environment variable. No effect when
    /// `-m/--model` is already an existing local path.
    #[arg(long, value_name = "PATH")]
    models_dir: Option<PathBuf>,

    /// Model alias (shown in API responses instead of directory name)
    #[arg(
        short = 'a',
        long = "alias",
        env = "LLAMA_ARG_ALIAS",
        value_name = "NAME"
    )]
    alias: Option<String>,

    /// Path to LoRA adapter directory
    #[arg(long = "lora", value_name = "PATH")]
    lora: Option<PathBuf>,

    /// Host address to bind to (or Unix socket path when --port 0)
    #[arg(long, env = "LLAMA_ARG_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Port number to listen on (0 = Unix socket mode using --host as socket path)
    #[arg(long, env = "LLAMA_ARG_PORT", default_value_t = 8080)]
    port: u16,

    /// Total context budget shared across parallel slots (0 = use model default)
    #[arg(
        short = 'c',
        long = "ctx-size",
        env = "LLAMA_ARG_CTX_SIZE",
        default_value_t = 0
    )]
    ctx_size: usize,

    /// Maximum tokens to predict (-1 = unlimited)
    #[arg(
        short = 'n',
        long = "predict",
        visible_alias = "n-predict",
        env = "LLAMA_ARG_N_PREDICT",
        default_value_t = -1
    )]
    predict: i32,

    /// Number of parallel request slots that share --ctx-size
    #[arg(long = "parallel", env = "LLAMA_ARG_N_PARALLEL", default_value_t = 1)]
    parallel: usize,

    /// API key for authentication
    #[arg(long = "api-key", env = "LLAMA_API_KEY", value_name = "KEY")]
    api_key: Option<String>,

    /// Path to file containing API key
    #[arg(long = "api-key-file", value_name = "PATH")]
    api_key_file: Option<PathBuf>,

    /// Request timeout in seconds
    #[arg(long, env = "LLAMA_ARG_TIMEOUT", default_value_t = 600)]
    timeout: u64,

    /// Path to drafter checkpoint for server speculative decoding
    ///
    /// Accepts the llama-server-style `--model-draft` spelling (primary) and
    /// the mlx-lm-style `--draft-model` spelling (alias, matches `mlxcel
    /// serve`) so commands copied between the two binaries work unchanged.
    #[arg(
        long = "model-draft",
        visible_alias = "draft-model",
        env = "LLAMA_ARG_MODEL_DRAFT",
        value_name = "PATH"
    )]
    model_draft: Option<PathBuf>,

    /// Maximum number of draft tokens per speculation step
    #[arg(
        long = "draft",
        visible_alias = "draft-max",
        env = "LLAMA_ARG_DRAFT_MAX",
        default_value_t = 16
    )]
    draft: usize,

    /// Maximum concurrent decode sequences; explicit value shares --ctx-size
    #[arg(long = "max-batch-size", value_name = "N")]
    max_batch_size: Option<usize>,

    /// Disable continuous batching and use the legacy sequential worker.
    ///
    /// When set, requests are processed one at a time in FIFO order with no
    /// batch scheduler overhead. Equivalent to using `--max-batch-size 1` but
    /// with explicit sequential semantics and no prefill chunking.
    #[arg(long = "no-batch")]
    no_batch: bool,

    /// Maximum number of requests waiting in the prefill queue (default: 32)
    #[arg(long = "max-queue-depth", default_value_t = 32)]
    max_queue_depth: usize,

    /// Bound on the audio worker command queue; a full queue returns 503 (default: 8)
    ///
    /// Caps how many audio (speech-to-text / text-to-speech) requests may wait
    /// behind the one in flight before admission is shed, so a burst cannot grow
    /// memory without bound (each queued command holds the full audio payload).
    /// A `0` clamps to at least one queued command.
    #[arg(
        long = "audio-queue-depth",
        env = "MLXCEL_AUDIO_QUEUE_DEPTH",
        default_value_t = 8
    )]
    audio_queue_depth: usize,

    /// Per-request audio reply timeout in seconds; 0 falls back to the default (default: 120)
    ///
    /// A stuck or pathologically slow audio request frees its blocking thread
    /// and returns a structured 504 after this, instead of hanging the worker.
    #[arg(
        long = "audio-request-timeout-secs",
        env = "MLXCEL_AUDIO_REQUEST_TIMEOUT_SECS",
        default_value_t = 120
    )]
    audio_request_timeout_secs: u64,

    /// Prefill chunk size in tokens (0 = disabled, default: 512)
    #[arg(long = "prefill-chunk-size", default_value_t = 512)]
    prefill_chunk_size: usize,

    /// Prefill batch size [llama-server alias for --prefill-chunk-size] [default: 512]
    #[arg(
        short = 'b',
        long = "batch-size",
        env = "LLAMA_ARG_BATCH_SIZE",
        value_name = "N"
    )]
    batch_size: Option<usize>,

    /// Physical micro-batch size [not applicable on Apple Silicon unified memory; ignored]
    #[arg(long = "ubatch-size", env = "LLAMA_ARG_UBATCH_SIZE", value_name = "N")]
    ubatch_size: Option<usize>,

    /// Enable preemptive eviction of lower-priority sequences
    #[arg(long = "enable-preemption")]
    enable_preemption: bool,

    /// Enable experimental VLM (image/audio) prompt-prefix cache sharing
    /// (default off). When on, multimodal chat requests may adopt and donate
    /// KV prefixes for multi-turn same-image conversations (the prefilled
    /// suffix is the newly-appended text turn). Text-only and non-VLM behavior
    /// is unchanged. Also reads `MLXCEL_ENABLE_VLM_PREFIX_CACHE` (true/false/1/0).
    #[arg(
        long = "enable-vlm-prefix-cache",
        env = "MLXCEL_ENABLE_VLM_PREFIX_CACHE"
    )]
    enable_vlm_prefix_cache: bool,

    /// Comma-separated list of allowed CORS origins (e.g.
    /// `https://app.example.com,https://admin.example.com`). When set,
    /// the server restricts cross-origin requests to exactly these origins
    /// instead of the default permissive policy that reflects any origin.
    /// Unset (default) keeps the permissive behavior. Only affects the
    /// browser-reachable TCP HTTP listener. Also reads
    /// `MLXCEL_ALLOWED_ORIGINS`.
    #[arg(
        long = "allowed-origins",
        env = "MLXCEL_ALLOWED_ORIGINS",
        value_delimiter = ',',
        value_name = "ORIGINS"
    )]
    allowed_origins: Vec<String>,

    /// Maximum denoising steps per canvas block (diffusion models only;
    /// default: the checkpoint's generation_config, typically 48)
    #[arg(long = "max-denoising-steps", value_name = "N")]
    max_denoising_steps: Option<usize>,

    /// Per-step acceptance sampler for diffusion models (diffusion models only)
    #[arg(
        long = "diffusion-sampler",
        value_name = "SAMPLER",
        default_value = "entropy-bound",
        value_parser = ["entropy-bound", "confidence-threshold"]
    )]
    diffusion_sampler: String,

    /// Confidence threshold for `--diffusion-sampler confidence-threshold`
    /// (diffusion models only)
    #[arg(
        long = "diffusion-threshold",
        value_name = "FLOAT",
        default_value_t = 0.9,
        value_parser = parse_unit_interval
    )]
    diffusion_threshold: f32,

    /// Preemption policy: "longest-first" (default) or "lowest-priority"
    #[arg(long = "preemption-policy", default_value = "longest-first")]
    preemption_policy: String,

    /// Maximum number of requests to batch together for prefill (default: 1)
    ///
    /// When > 1, the scheduler collects up to this many pending requests and
    /// runs a single batched forward pass [batch_size, max_seq_len] for better
    /// Neural Accelerator utilization. Recommended: 4-8 on M5 Pro/Max hardware.
    #[arg(long = "max-batch-prefill", default_value_t = 1)]
    max_batch_prefill: usize,

    /// Maximum KV cache size for plain (non-sliding) caches (0 = unbounded, the default).
    ///
    /// When set to `N > 0`, the batch scheduler caps each per-sequence plain
    /// `KVCache` to `N` tokens by dropping the oldest entries once `offset`
    /// exceeds the bound.
    ///
    /// Sliding-window models that already build their own `RotatingKVCache`
    /// (Gemma 3/4, Exaone 4, RecurrentGemma, Step 3.5, gpt-oss) are
    /// unaffected: their model-specific window remains the source of truth.
    ///
    /// Not supported in combination with Turbo KV quantization
    /// (`--kv-cache-mode turbo4*`); when both are set the cap is silently
    /// skipped for the Turbo-quantized layers with a startup warning.
    ///
    /// Also reads `LLAMA_ARG_MAX_KV_SIZE`.
    #[arg(
        long = "max-kv-size",
        env = "LLAMA_ARG_MAX_KV_SIZE",
        default_value_t = 0,
        value_name = "N"
    )]
    max_kv_size: usize,

    /// Paged KV-cache pool block budget: `auto` or a byte count (default: unbounded).
    ///
    /// Bounds the unified paged KV cache (epic #116): `auto` derives the cap
    /// from the memory estimate, a raw byte count sets it explicitly. Only
    /// affects pool-backed (Fp16) models under `--decode-storage-backend paged`.
    /// Also reads `MLXCEL_KV_CACHE_BUDGET`.
    #[arg(
        long = "kv-cache-budget",
        env = "MLXCEL_KV_CACHE_BUDGET",
        value_name = "BYTES|auto",
        value_parser = parse_kv_cache_budget
    )]
    kv_cache_budget: Option<mlxcel::memory_estimate::PagedBudgetDirective>,

    /// Maximum number of responses persisted by the OpenAI
    /// `/v1/responses` store (in-memory). `0` disables persistence
    /// entirely. Also reads `LLAMA_ARG_RESPONSES_STORE_MAX_ENTRIES`.
    #[arg(
        long = "responses-store-max-entries",
        env = "LLAMA_ARG_RESPONSES_STORE_MAX_ENTRIES",
        default_value_t = 1024,
        value_name = "N"
    )]
    responses_store_max_entries: usize,

    /// TTL (seconds) for in-memory Responses-API response
    /// entries. `0` disables TTL.
    /// Also reads `LLAMA_ARG_RESPONSES_STORE_TTL_SECS`.
    #[arg(
        long = "responses-store-ttl-secs",
        env = "LLAMA_ARG_RESPONSES_STORE_TTL_SECS",
        default_value_t = 3600,
        value_name = "SECS"
    )]
    responses_store_ttl_secs: u64,

    /// Maximum number of conversation transcripts persisted
    /// for the OpenAI Responses API `conversation` field. `0` disables.
    /// Also reads `LLAMA_ARG_CONVERSATION_STORE_MAX_ENTRIES`.
    #[arg(
        long = "conversation-store-max-entries",
        env = "LLAMA_ARG_CONVERSATION_STORE_MAX_ENTRIES",
        default_value_t = 256,
        value_name = "N"
    )]
    conversation_store_max_entries: usize,

    /// TTL (seconds) for conversation transcript entries.
    /// `0` disables TTL.
    /// Also reads `LLAMA_ARG_CONVERSATION_STORE_TTL_SECS`.
    #[arg(
        long = "conversation-store-ttl-secs",
        env = "LLAMA_ARG_CONVERSATION_STORE_TTL_SECS",
        default_value_t = 3600,
        value_name = "SECS"
    )]
    conversation_store_ttl_secs: u64,

    /// Override chat template (Jinja2 template string)
    #[arg(long = "chat-template", value_name = "TEMPLATE")]
    chat_template: Option<String>,

    /// Path to chat template file
    #[arg(long = "chat-template-file", value_name = "PATH")]
    chat_template_file: Option<PathBuf>,

    /// Enable /slots endpoint
    #[arg(long = "slots", overrides_with = "_no_slots", default_value_t = true)]
    slots: bool,

    /// Disable /slots endpoint
    #[arg(long = "no-slots", overrides_with = "slots", hide = true)]
    _no_slots: bool,

    /// Enable /props endpoint
    #[arg(long = "props")]
    props: bool,

    /// Enable /metrics endpoint
    #[arg(long = "metrics")]
    metrics: bool,

    /// Enable model warmup on startup
    #[arg(long = "warmup", overrides_with = "_no_warmup", default_value_t = true)]
    warmup: bool,

    /// Disable model warmup on startup
    #[arg(long = "no-warmup", overrides_with = "warmup", hide = true)]
    _no_warmup: bool,

    // Default sampling parameters.
    /// Default sampling temperature
    #[arg(long = "temp", default_value_t = 0.8)]
    temp: f32,

    /// Default top-K sampling
    #[arg(long = "top-k", env = "LLAMA_ARG_TOP_K", default_value_t = 40)]
    top_k: i32,

    /// Default top-P (nucleus) sampling
    #[arg(long = "top-p", default_value_t = 0.9)]
    top_p: f32,

    /// Default min-P sampling
    #[arg(long = "min-p", default_value_t = 0.1)]
    min_p: f32,

    /// Random seed (-1 = random)
    #[arg(short = 's', long = "seed", default_value_t = -1)]
    seed: i64,

    /// Default repetition penalty lookback window
    #[arg(long = "repeat-last-n", default_value_t = 64)]
    repeat_last_n: usize,

    /// Default repetition penalty multiplier
    #[arg(long = "repeat-penalty", default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Default presence penalty
    #[arg(long = "presence-penalty", default_value_t = 0.0)]
    presence_penalty: f32,

    /// Default frequency penalty
    #[arg(long = "frequency-penalty", default_value_t = 0.0)]
    frequency_penalty: f32,

    // DRY sampling parameters.
    /// DRY penalty multiplier (0.0 = disabled)
    #[arg(long = "dry-multiplier", default_value_t = 0.0)]
    dry_multiplier: f32,

    /// DRY exponential base
    #[arg(long = "dry-base", default_value_t = 1.75)]
    dry_base: f32,

    /// DRY minimum match length before penalty
    #[arg(long = "dry-allowed-length", default_value_t = 2)]
    dry_allowed_length: usize,

    /// DRY lookback window (-1 = full context)
    #[arg(long = "dry-penalty-last-n", default_value_t = -1)]
    dry_penalty_last_n: i32,

    /// DRY sequence breaker token strings (e.g. "\n", "\t")
    #[arg(long = "dry-sequence-breaker", value_delimiter = ',')]
    dry_sequence_breakers: Vec<String>,

    // Logging.
    /// Enable verbose (debug) logging
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    /// Disable all logging
    #[arg(long = "log-disable")]
    log_disable: bool,

    /// Log output file
    #[arg(long = "log-file", env = "LLAMA_LOG_FILE", value_name = "PATH")]
    log_file: Option<PathBuf>,

    // Distributed inference.
    /// Path to TOML cluster configuration file for distributed inference
    #[arg(long, value_name = "PATH")]
    distributed_config: Option<PathBuf>,

    /// Role this node plays in the cluster (prefill, decode, pipeline_stage, tensor_parallel_rank, pipeline_tensor_parallel, hybrid)
    #[arg(long, value_name = "ROLE")]
    node_role: Option<String>,

    /// Unique identifier for this node in the cluster
    #[arg(long, value_name = "ID")]
    node_id: Option<String>,

    /// Comma-separated list of peer addresses (host:port) for static discovery
    #[arg(long, value_delimiter = ',', value_name = "ADDR")]
    peers: Vec<std::net::SocketAddr>,

    /// Comma-separated prefill-node addresses. Decode nodes use this to identify
    /// accepted handoff sources; routers use it to select a prefill target.
    /// Consumed when `--node-role decode` or `--node-role router`.
    #[arg(long, value_delimiter = ',', value_name = "ADDR")]
    prefill_peers: Vec<std::net::SocketAddr>,

    /// Comma-separated decode-node addresses. Prefill nodes hand KV state to one
    /// of these targets; routers use it to route decode continuations.
    /// Consumed when `--node-role prefill` or `--node-role router`.
    #[arg(long, value_delimiter = ',', value_name = "ADDR")]
    decode_peers: Vec<std::net::SocketAddr>,

    /// This node's own bind address (host:port) for the disaggregated
    /// serving-role transport (#126). Required for `--node-role prefill`,
    /// `--node-role decode`, and `--node-role router`: prefill nodes receive
    /// prompt frames, decode nodes receive KV handoffs, and routers receive
    /// role-result frames.
    #[arg(long, value_name = "ADDR")]
    serving_bind: Option<std::net::SocketAddr>,

    /// Manual pipeline-parallel layer partition (e.g. "0-15,16-31")
    ///
    /// Specifies explicit layer ranges per pipeline stage. Each range is
    /// inclusive on both ends. When omitted, layers are auto-partitioned
    /// proportionally to device memory.
    #[arg(long = "pp-layers", value_name = "RANGES")]
    pp_layers: Option<String>,

    /// Micro-batch size for single-machine pipeline execution.
    #[arg(long = "pp-micro-batch-size", default_value_t = 1, value_name = "N")]
    pp_micro_batch_size: usize,

    /// Zero-config pipeline-parallel bring-up: declare the desired number of stages.
    ///
    /// When set (N >= 2), `mlxcel-server` acts as the coordinator and resolves
    /// peers either from `--cluster-peers` or via `--cluster-discovery=mdns`,
    /// allocates ports for the coordinator control plane and stage data ports
    /// if they are not explicitly provided, and emits a deterministic cluster
    /// TOML to `--cluster-config-out` before starting the server. The flag is
    /// mutually exclusive with `--distributed-config`.
    #[arg(long = "pp-auto", value_name = "N")]
    pp_auto: Option<u32>,

    /// Peer role for zero-config pipeline bring-up: register with the coordinator
    /// instead of starting a server of our own.
    ///
    /// When set, `mlxcel-server` announces its availability (either statically
    /// by registering against a known coordinator address, or via broadcast
    /// when `--cluster-discovery=mdns`) and then starts a pipeline stage
    /// service using the stage assignment the coordinator returns.
    #[arg(long = "pp-peer")]
    pp_peer: bool,

    /// Cluster discovery mechanism: "static" (default) or "mdns" for UDP broadcast.
    ///
    /// "static" consumes `--cluster-peers` verbatim. "mdns" enables opt-in
    /// LAN peer discovery via UDP broadcast. The name is retained for future
    /// zeroconf compatibility; today the implementation uses plain UDP so no
    /// extra dependency is required.
    #[arg(
        long = "cluster-discovery",
        default_value = "static",
        value_name = "MODE"
    )]
    cluster_discovery: String,

    /// Human-readable cluster name used to scope discovery and as the TOML header.
    ///
    /// Defaults to the value embedded in the generated TOML when `--pp-auto`
    /// runs (currently `mlxcel-cluster`). Peers with a mismatching name are
    /// ignored by the coordinator during mDNS discovery.
    #[arg(long = "cluster-name", value_name = "NAME")]
    cluster_name: Option<String>,

    /// Static peer addresses for zero-config bring-up (host:port, comma-separated).
    ///
    /// Each peer address should point at the control+data socket that the
    /// corresponding `mlxcel-server --pp-peer` exposes. Ignored when
    /// `--cluster-discovery=mdns` fully resolves the expected peer count.
    #[arg(long = "cluster-peers", value_delimiter = ',', value_name = "ADDR")]
    cluster_peers: Vec<std::net::SocketAddr>,

    /// UDP port for the discovery beacon when `--cluster-discovery=mdns` is used.
    #[arg(long = "cluster-discovery-port", value_name = "PORT")]
    cluster_discovery_port: Option<u16>,

    /// Coordinator control-plane bind address for zero-config bring-up (host:port).
    ///
    /// Kept deliberately distinct from the HTTP listen address so operators do
    /// not have to co-schedule two services on a single port.
    #[arg(long = "cluster-control-addr", value_name = "ADDR")]
    cluster_control_addr: Option<std::net::SocketAddr>,

    /// Output path for the emitted cluster TOML.
    ///
    /// Defaults to `<current directory>/.mlxcel/cluster.toml` when
    /// `--pp-auto` is used and this flag is omitted.
    #[arg(long = "cluster-config-out", value_name = "PATH")]
    cluster_config_out: Option<PathBuf>,

    /// Plan the cluster topology and emit the TOML without starting workers.
    ///
    /// Exits with non-zero status when port, version, or peer-count conflicts
    /// cannot be resolved. Only meaningful in combination with `--pp-auto`.
    #[arg(long = "dry-run", default_value_t = false)]
    dry_run: bool,

    /// Number of tensor-parallel ranks (must be a power of 2).
    ///
    /// Current multi-rank runtime support is limited to dense Llama, Qwen2/2.5,
    /// Qwen3, Qwen3.5 text, Gemma 3 text, Gemma 4 text, ERNIE 4.5, and
    /// Hunyuan v1 Dense models.
    #[arg(long = "tp-size", default_value_t = 1, value_name = "N")]
    tp_size: usize,

    /// MoE expert sharding mode: "expert_parallel" or "within_expert"
    #[arg(
        long = "tp-moe-mode",
        default_value = "expert_parallel",
        value_name = "MODE"
    )]
    tp_moe_mode: String,

    /// Embedding sharding mode: "vocab_parallel" or "replicated".
    ///
    /// The current in-process tensor-parallel runtime requires "replicated".
    #[arg(
        long = "tp-embedding-mode",
        default_value = "replicated",
        value_name = "MODE"
    )]
    tp_embedding_mode: String,

    /// LM head sharding mode: "vocab_parallel" or "replicated".
    ///
    /// The current in-process tensor-parallel runtime requires "replicated".
    #[arg(
        long = "tp-lm-head-mode",
        default_value = "replicated",
        value_name = "MODE"
    )]
    tp_lm_head_mode: String,

    /// Decode storage backend for continuous batching.
    ///
    /// Accepted values: `auto`, `dense`, `paged`. When omitted, the server
    /// uses `MLXCEL_SERVER_DECODE_STORAGE` if set, otherwise automatic
    /// selection.
    #[arg(long = "decode-storage-backend", value_name = "BACKEND")]
    decode_storage_backend: Option<mlxcel::server::DecodeStorageBackend>,

    // llama-server compatibility arguments (accepted but ignored).
    /// Accepted for llama-server CLI compatibility (ignored: mlxcel has no web UI)
    #[arg(long, hide = true)]
    _no_webui: bool,

    /// Accepted for llama-server CLI compatibility (ignored: mlxcel always processes templates)
    #[arg(long, hide = true)]
    _jinja: bool,

    /// Accepted for llama-server CLI compatibility (ignored: mlxcel always uses Metal)
    #[arg(long = "n-gpu-layers", hide = true)]
    _n_gpu_layers: Option<i32>,

    /// Accepted for llama-server CLI compatibility (ignored: vision projector loaded automatically)
    #[arg(long, hide = true)]
    _mmproj: Option<String>,

    /// Accepted for llama-server CLI compatibility (ignored)
    #[arg(long, hide = true)]
    _flash_attn: bool,

    /// Accepted for llama-server CLI compatibility (ignored: not applicable to MLX)
    #[arg(long, hide = true)]
    _mlock: bool,

    /// Accepted for llama-server CLI compatibility (ignored: not applicable to MLX)
    #[arg(long = "no-mmap", hide = true)]
    _no_mmap: bool,

    /// Accepted for llama-server CLI compatibility (ignored: mlxcel handles batching internally)
    #[arg(long, hide = true)]
    _cont_batching: bool,

    /// Maximum number of cached post-projection image features per loaded VLM.
    ///
    /// Multi-turn conversations that revisit the same image reuse cached
    /// vision features and skip the vision tower + multimodal embedder on
    /// subsequent turns. `0` disables caching. Default: 20.
    #[arg(long = "vision-cache-size", default_value_t = 20, value_name = "N")]
    vision_cache_size: usize,

    /// Maximum encoded bytes accepted for each image input.
    ///
    /// Also reads `LLAMA_ARG_MAX_IMAGE_PAYLOAD_SIZE`.
    #[arg(
        long = "max-image-payload-size",
        env = "LLAMA_ARG_MAX_IMAGE_PAYLOAD_SIZE",
        default_value_t = mlxcel::server::DEFAULT_MAX_IMAGE_PAYLOAD_SIZE,
        value_name = "BYTES"
    )]
    max_image_payload_size: usize,

    /// Maximum number of image inputs accepted in one request.
    ///
    /// Also reads `LLAMA_ARG_MAX_IMAGES`.
    #[arg(
        long = "max-images",
        env = "LLAMA_ARG_MAX_IMAGES",
        default_value_t = mlxcel::server::DEFAULT_MAX_IMAGES_PER_REQUEST,
        value_name = "N"
    )]
    max_images_per_request: usize,

    /// Maximum decoded image width accepted by the VLM image decoder.
    #[arg(
        long = "max-image-width",
        env = "LLAMA_ARG_MAX_IMAGE_WIDTH",
        default_value_t = mlxcel::server::DEFAULT_MAX_IMAGE_WIDTH,
        value_name = "PX"
    )]
    max_image_width: u32,

    /// Maximum decoded image height accepted by the VLM image decoder.
    #[arg(
        long = "max-image-height",
        env = "LLAMA_ARG_MAX_IMAGE_HEIGHT",
        default_value_t = mlxcel::server::DEFAULT_MAX_IMAGE_HEIGHT,
        value_name = "PX"
    )]
    max_image_height: u32,

    /// Maximum decoder allocation budget for a single image.
    #[arg(
        long = "max-image-decode-alloc-bytes",
        env = "LLAMA_ARG_MAX_IMAGE_DECODE_ALLOC_BYTES",
        default_value_t = mlxcel::server::DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES,
        value_name = "BYTES"
    )]
    max_image_decode_alloc_bytes: u64,

    /// Enable experimental elastic pipeline-parallel repartitioning.
    ///
    /// When set, `mlxcel-server` constructs a repartition coordinator that can
    /// drain in-flight requests, recompute the partition plan, and reload
    /// layer weights without a full cluster restart. Off by default.
    #[arg(long = "enable-elastic-pp", default_value_t = false)]
    enable_elastic_pp: bool,

    /// Maximum wait (seconds) for in-flight requests to drain during an
    /// elastic repartition. Only meaningful with `--enable-elastic-pp`.
    #[arg(
        long = "elastic-pp-drain-timeout",
        default_value_t = 120,
        value_name = "SECONDS"
    )]
    elastic_pp_drain_timeout: u64,

    /// Memory usage fraction above which a memory-pressure trigger fires.
    /// Values outside (0.0, 1.0] are clamped. Default: 0.92. Only meaningful
    /// with `--enable-elastic-pp`.
    #[arg(
        long = "elastic-pp-pressure-fraction",
        default_value_t = 0.92,
        value_name = "FRACTION"
    )]
    elastic_pp_pressure_fraction: f64,

    /// Cool-down (seconds) between successive memory-pressure repartition
    /// triggers on the same stage. Explicit operator triggers bypass this
    /// debounce. Default: 30. Only meaningful with `--enable-elastic-pp`.
    #[arg(
        long = "elastic-pp-cool-down",
        default_value_t = 30,
        value_name = "SECONDS"
    )]
    elastic_pp_cool_down: u64,

    /// Enable `/metrics` and advertise the port operators should scrape.
    ///
    /// Currently the Prometheus endpoint is multiplexed onto the same HTTP
    /// port as the OpenAI API. Passing this flag enables the endpoint.
    /// A warning is logged when the requested port differs from `--port`
    /// because metrics are currently served on the main HTTP listener.
    #[arg(long = "metrics-port", value_name = "PORT")]
    metrics_port: Option<u16>,

    /// Write a chrome-tracing-compatible JSON trace of pipeline scheduler
    /// actions (batch arrival, stage enter/exit, activation send/receive,
    /// admission reject) to this file for offline analysis in
    /// `chrome://tracing` or Perfetto.
    #[arg(long = "debug-pp-trace", value_name = "PATH")]
    debug_pp_trace: Option<PathBuf>,

    // Shared TurboQuant KV-cache flag group (--cache-type-k, --cache-type-v,
    // --kv-cache-mode, --turbo-boundary-v). Defined once in
    // mlxcel::cli::turbo_args so all three binaries (mlxcel generate,
    // mlxcel serve, mlxcel-server) expose identical help text and flags.
    //
    // Placed immediately before the `lang_bias` flatten so that the
    // `KV Cache (TurboQuant) Options` heading introduced by `TurboKvCacheArgs`
    // does not bleed into sibling fields below; the next `next_help_heading`
    // (`Batch KV Quantization Options`, set on `BatchKvQuantArgs`, then
    // `Language Bias Options`, set on `LangBiasCliArgs`) takes over the
    // moment the next group is parsed.
    #[command(flatten)]
    turbo: TurboKvCacheArgs,

    /// Continuous-batching KV quantization flag group
    /// (`--kv-bits`, `--kv-group-size`, `--kv-quant-scheme`,
    /// `--kv-skip-last-layer`). Defined once in
    /// `mlxcel::cli::batch_quant_args` so both server binaries
    /// (`mlxcel serve`, `mlxcel-server`) expose identical help text and
    /// flags. Not flattened on `mlxcel generate`; the offline path has no
    /// continuous-batching scheduler to feed.
    #[command(flatten)]
    batch_quant: BatchKvQuantArgs,

    /// Speculative-decoding flag group (`--draft-kind`, `--draft-block-size`).
    /// Defined once in `mlxcel::cli::speculative_args` so all three
    /// binaries (`mlxcel generate`, `mlxcel serve`, `mlxcel-server`) expose
    /// identical help text and parsing. The `--model-draft` / `--draft`
    /// flags stay above on this struct because their primary spelling is
    /// llama-server-compatible; each also carries a visible alias
    /// (`--draft-model`, `--draft-max`) so a `mlxcel serve` command line
    /// works unchanged on `mlxcel-server`. See the parity note on
    /// `SpeculativeArgs` and `ServeArgs::draft_model` / `draft_max` in
    /// `src/main.rs`.
    #[command(flatten)]
    speculative: SpeculativeArgs,

    /// Language-bias options for server-wide output
    /// steering. See `--lang-bias`, `--lang-bias-config`, `--lang-bias-policy`,
    /// and the `--lang-bias-include-*` family of flags.
    ///
    /// The `--lang-bias` flag also reads from the `LLAMA_ARG_LANG_BIAS` env var.
    /// CLI flag takes precedence over the env var.
    #[command(flatten)]
    lang_bias: LangBiasCliArgs,

    /// Default thinking-token budget for Qwen3-family models.
    ///
    /// Caps the number of tokens generated inside the `<think>...</think>`
    /// reasoning block. Matches llama.cpp `--reasoning-budget` semantics:
    ///   -1 = unrestricted (default)
    ///    0 = immediate end of thinking (force </think> on first reasoning token)
    ///    N > 0 = cap reasoning at N tokens
    ///
    /// Per-request `thinking_budget_tokens` (primary), `thinking_token_budget`
    /// (vLLM alias), or `thinking_budget` (Qwen alias) on
    /// `/v1/chat/completions` and `/completion` override this value. Also
    /// reads from `LLAMA_ARG_REASONING_BUDGET` (applied via
    /// `env_fallback_reasoning_budget`); CLI wins on conflict. Unparseable
    /// env values are warn-logged and ignored. Silently ignored for models
    /// that do not expose `<think>` / `</think>` tokens.
    #[arg(
        long = "reasoning-budget",
        default_value_t = -1,
        value_name = "N"
    )]
    reasoning_budget: i32,

    /// Default chat-template kwargs (JSON object).
    ///
    /// Forwarded verbatim as Jinja template kwargs when rendering chat
    /// conversations. Matches llama.cpp's `--chat-template-kwargs` shape.
    ///
    /// Examples:
    ///   --chat-template-kwargs '{"preserve_thinking": true}'
    ///   --chat-template-kwargs '{"enable_thinking": false, "preserve_thinking": true}'
    ///
    /// Per-request `chat_template_kwargs` (top-level or under `extra_body`)
    /// overrides server defaults on a per-key basis; unrelated server-default
    /// keys persist through the merge. The `preserve_thinking` alias is also
    /// accepted via nested `extra_body.preserve_thinking` and the OpenAI SDK's
    /// flattened root-level `extra_body={"preserve_thinking": ...}` shape.
    ///
    /// Also honors `LLAMA_ARG_CHAT_TEMPLATE_KWARGS`; CLI wins on conflict.
    /// Malformed JSON is rejected at startup with a clear error.
    ///
    /// Note: `preserve_thinking` quality benefits are validated on Qwen3.6;
    /// Qwen3 / Qwen3.5 accept the flag but were trained on the
    /// rolling-checkpoint convention.
    #[arg(
        long = "chat-template-kwargs",
        value_name = "JSON",
        verbatim_doc_comment
    )]
    chat_template_kwargs: Option<String>,

    // cross-request prompt-prefix KV cache knobs.
    /// Enable or disable the cross-request prompt-prefix KV cache (default: true).
    ///
    /// When disabled, the server performs no prefix-match lookup and no memory
    /// is reserved for the cache. Disabling eliminates any lock contention and
    /// matcher overhead.
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_ENABLED` (boolean on/off/true/false/1/0)
    /// and the llama.cpp-compat alias `LLAMA_ARG_CACHE_REUSE` when the CLI flag
    /// is not explicitly provided. CLI flag takes precedence over env vars.
    #[arg(
        long = "prompt-cache-enabled",
        default_value_t = true,
        value_name = "BOOL",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    prompt_cache_enabled: bool,

    /// Maximum byte budget for the prompt-prefix KV cache (default: 2 GiB).
    ///
    /// Inserts that would push total cache size above this threshold after LRU
    /// eviction are rejected. Setting to `0` effectively disables inserts.
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES` when the CLI flag is
    /// absent. CLI flag takes precedence.
    #[arg(long = "prompt-cache-capacity-bytes", value_name = "BYTES")]
    prompt_cache_capacity_bytes: Option<usize>,

    /// Maximum number of live entries in the prompt-prefix KV cache (default: 1024).
    ///
    /// Once the limit is reached, the least-recently-used entry is evicted to
    /// make room for a new one.
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_MAX_ENTRIES` when the CLI flag is absent.
    /// CLI flag takes precedence.
    #[arg(long = "prompt-cache-max-entries", value_name = "N")]
    prompt_cache_max_entries: Option<usize>,

    /// Time-to-live for a prompt-cache entry in seconds (default: 3600).
    ///
    /// Entries older than this value since their last hit are lazily evicted
    /// on the next lookup or on an explicit eviction pass.
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_TTL` when the CLI flag is absent.
    /// CLI flag takes precedence.
    #[arg(long = "prompt-cache-ttl", value_name = "SECONDS")]
    prompt_cache_ttl: Option<u64>,

    /// Minimum prompt-prefix length (tokens) required before an entry is cached
    /// (default: 32).
    ///
    /// Prefixes shorter than this threshold are not stored to avoid polluting the
    /// cache with tiny prefixes that cannot amortize the detach/adopt overhead.
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_MIN_PREFIX` when the CLI flag is absent.
    /// CLI flag takes precedence.
    #[arg(long = "prompt-cache-min-prefix", value_name = "N")]
    prompt_cache_min_prefix: Option<usize>,

    // Automatic Prefix Caching (APC) knobs.
    /// Enable Automatic Prefix Caching (APC) with block-granularity hash chains
    /// (default: true). Disable with `--apc-enabled=false`.
    ///
    /// APC layers on top of the existing prompt-prefix cache to enable
    /// finer-grained KV reuse with chained `(parent_hash, tokens, extra_hash)`
    /// per block. When enabled on a hybrid SSM/attention model (jamba, mamba,
    /// mamba2, nemotron_h, gated_delta, kimi_linear, qwen3_next), APC is
    /// automatically disabled at runtime since SSM state cannot be decomposed
    /// into hashable blocks.
    ///
    /// Also reads `APC_ENABLED`.
    #[arg(
        long = "apc-enabled",
        default_value_t = true,
        value_name = "BOOL",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    apc_enabled: bool,

    /// Tokens per APC block (default: 16).
    ///
    /// Smaller values increase reuse granularity at the cost of per-block
    /// hashing overhead. Also reads `APC_BLOCK_SIZE`.
    #[arg(long = "apc-block-size", value_name = "N")]
    apc_block_size: Option<usize>,

    /// Maximum number of APC block entries to track. `None` falls back to
    /// the heuristic derived from `--prompt-cache-max-entries`.
    ///
    /// Also reads `APC_NUM_BLOCKS`.
    #[arg(long = "apc-num-blocks", value_name = "N")]
    apc_num_blocks: Option<usize>,

    /// APC hash algorithm (default: `sha256`).
    ///
    /// Accepted values: `sha256`, `blake3`. SHA-256 is the default for
    /// wire-compatibility with upstream APC artifacts; BLAKE3 is faster but
    /// not wire-compatible.
    ///
    /// Also reads `APC_HASH`.
    #[arg(long = "apc-hash", value_name = "ALGO")]
    apc_hash: Option<String>,

    // Axis A weight-load surgery configuration.
    // Closed-repo references kept in a non-doc comment to avoid leaking
    // tracker URLs into `--help` text.
    /// Apply weight-load surgery configuration from a YAML file.
    ///
    /// Path to a YAML configuration file describing structural
    /// fine-tuning operations (scale / add / prune / replace /
    /// interpolate). When omitted, weight loading is bit-exact identical
    /// to the pre-surgery baseline.
    ///
    /// Also reads `MLXCEL_SURGERY`; CLI flag wins on conflict.
    ///
    /// Example:
    ///
    ///     mlxcel-server -m models/foo --surgery surgery.yaml --port 8080
    ///
    /// The supported surgery operations are summarised in the project README.
    #[cfg(feature = "surgery")]
    #[arg(long = "surgery", value_name = "FILE", env = "MLXCEL_SURGERY")]
    surgery: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Default the CUDA kernel JIT cache to a persistent, MLX-pin-scoped dir so
    // the first-run kernel compilation is paid once per machine, not every boot.
    mlxcel_core::ensure_persistent_ptx_cache();

    // Raise MLX_MAX_OPS_PER_BUFFER on pre-M5 Apple Silicon to close decode
    // command-buffer dispatch-gap idle (#353). Hardware-gated, a no-op when the
    // variable is already set, and must run before any MLX op.
    mlxcel_core::hardware::apply_metal_ops_per_buffer_default();

    match cli.command {
        // Subcommand-driven dispatch. Currently only `download`
        // exists; future operational subcommands (e.g. cache inspection) can
        // be added to [`Commands`] without touching the legacy server-start
        // path.
        Some(Commands::Download(args)) => run_download(args),
        // Legacy invocation: no subcommand → boot the HTTP server using the
        // flattened server flags. Backward-compatible with every prior
        // `mlxcel-server -m foo --port 8080 ...` invocation.
        None => start_server(build_startup_input(cli.server)?.into_startup_config()?).await,
    }
}

fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    let opts = DownloadOptions::from_args(&args);
    download_repo(opts)
}

fn build_startup_input(mut args: ServerArgs) -> anyhow::Result<ServerStartupInput> {
    // Translate `--turbo-boundary-v` into the `MLXCEL_KV_BOUNDARY_V_LAYERS`
    // env var before any caller of `mlxcel-core` constructs a cache.
    // mlxcel-core reads this env var on first cache instantiation, and the
    // write site must be upstream of any code that spawns tasks reading the
    // process environment. The tokio worker threads spawned by
    // `#[tokio::main]` are still parked at this point (no task has been
    // scheduled yet), so the only env reader is this thread. See the
    // function-level SAFETY note on `TurboKvCacheArgs::apply_to_environment`
    // for the full precondition.
    args.turbo.apply_to_environment();

    // Axis B (B7): apply `LLAMA_ARG_LANG_BIAS` env-var fallback
    // before resolving, so env-supplied values flow through the same
    // validation and normalization as CLI flags. CLI flag wins on conflict
    // (see `env_fallback_lang_bias` INFO log).
    env_fallback_lang_bias(&mut args.lang_bias);
    // env-var fallback for the byte-fragment opt-in flag.
    env_fallback_lang_bias_include_byte_fragments(&mut args.lang_bias);
    // env-var fallback for the chat-template kwargs default.
    env_fallback_chat_template_kwargs(&mut args.chat_template_kwargs);

    // Env-var fallbacks for prompt-cache knobs. Detect explicit boolean flags
    // from argv so `--prompt-cache-enabled=false` keeps CLI-over-env precedence
    // while the compiled-in default still allows env overrides.
    env_fallback_prompt_cache_enabled(
        &mut args.prompt_cache_enabled,
        long_cli_flag_was_set("prompt-cache-enabled"),
    );
    env_fallback_prompt_cache_capacity_bytes(&mut args.prompt_cache_capacity_bytes);
    env_fallback_prompt_cache_max_entries(&mut args.prompt_cache_max_entries);
    env_fallback_prompt_cache_ttl(&mut args.prompt_cache_ttl);
    env_fallback_prompt_cache_min_prefix(&mut args.prompt_cache_min_prefix);

    // env-var fallbacks for the APC knobs (parity with upstream
    // mlx-vlm `APC_*` env vars).
    env_fallback_apc_enabled(&mut args.apc_enabled, long_cli_flag_was_set("apc-enabled"));
    env_fallback_apc_block_size(&mut args.apc_block_size);
    env_fallback_apc_num_blocks(&mut args.apc_num_blocks);
    env_fallback_apc_hash(&mut args.apc_hash);

    // (B11): env-var fallbacks for KV cache type split flags.
    // LLAMA_ARG_CACHE_TYPE_K / LLAMA_ARG_CACHE_TYPE_V are the canonical env
    // vars matching llama.cpp; the clap `env = "..."` attribute on the arg
    // also reads them directly, so these helpers are only needed when the CLI
    // flag uses a different default convention (Option<String>). Since we use
    // `env = "..."` on the clap arg definition, these explicit fallback calls
    // are not strictly necessary here, clap already reads the env vars.
    // We still call them for consistency with the pattern and to allow future
    // warn-on-conflict logic (e.g. if a separate MLXCEL_* alias is added).
    env_fallback_cache_type_k(&mut args.turbo.cache_type_k);
    env_fallback_cache_type_v(&mut args.turbo.cache_type_v);
    // env-var fallbacks for the continuous-batching KV
    // quantization knobs. The flags themselves live in
    // `mlxcel::cli::batch_quant_args::BatchKvQuantArgs` (flattened above);
    // these helpers honor the warn-on-CLI-conflict pattern shared with the
    // other LLAMA_ARG_* env vars.
    env_fallback_kv_bits(&mut args.batch_quant.kv_bits);
    env_fallback_kv_group_size(&mut args.batch_quant.kv_group_size);
    env_fallback_kv_quant_scheme(&mut args.batch_quant.kv_quant_scheme);
    env_fallback_kv_skip_last_layer(&mut args.batch_quant.kv_skip_last_layer);

    // env-var fallbacks for the speculative-decoding selector
    // flags. `clap` already reads `LLAMA_ARG_DRAFT_KIND` /
    // `LLAMA_ARG_DRAFT_BLOCK_SIZE` via the `env = "..."` attr on each flag;
    // the helpers below layer the mlxcel-native `MLXCEL_DRAFT_KIND` /
    // `MLXCEL_DRAFT_BLOCK_SIZE` aliases on top with the same warn-on-conflict
    // pattern shared with the other `MLXCEL_*` / `LLAMA_ARG_*` pairs.
    env_fallback_draft_kind(&mut args.speculative.draft_kind);
    env_fallback_draft_block_size(&mut args.speculative.draft_block_size);

    // Axis B (B8): resolve once up-front so CLI errors surface before the
    // server starts listening. Baseline path returns `None` (bit-exact).
    let lang_bias_config = args
        .lang_bias
        .resolve()
        .map_err(|e| anyhow::anyhow!("--lang-bias: {e}"))?;

    let model_path = args.model.ok_or_else(|| {
        anyhow::anyhow!(
            "--model/-m is required to start the server \
             (set the LLAMA_ARG_MODEL env var or pass -m <PATH_OR_REPO_ID>)"
        )
    })?;
    let model_path = resolve_model_source_with_override(&model_path, args.models_dir.as_deref())?;

    Ok(ServerStartupInput {
        model_path,
        adapter_path: args.lora,
        model_alias: args.alias,
        host: args.host,
        port: args.port,
        api_key: args.api_key,
        api_key_file: args.api_key_file,
        n_parallel: args.parallel,
        ctx_size: args.ctx_size,
        n_predict: args.predict,
        timeout: args.timeout,
        draft_model_path: args.model_draft,
        draft_max: args.draft,
        // forward the speculative-decoding selector flags
        // resolved above via env-var fallbacks. Reconciliation into a
        // typed `DrafterKind` happens later, at the dispatch site.
        draft_kind: args.speculative.draft_kind,
        draft_block_size: args.speculative.draft_block_size,
        max_batch_size: args.max_batch_size,
        no_batch: args.no_batch,
        max_queue_depth: args.max_queue_depth,
        audio_queue_depth: args.audio_queue_depth,
        audio_request_timeout_secs: args.audio_request_timeout_secs,
        prefill_chunk_size: args.prefill_chunk_size,
        batch_size: args.batch_size,
        ubatch_size: args.ubatch_size,
        enable_preemption: args.enable_preemption,
        preemption_policy: args.preemption_policy,
        max_batch_prefill: args.max_batch_prefill,
        decode_storage_backend: args.decode_storage_backend,
        chat_template: args.chat_template,
        chat_template_file: args.chat_template_file,
        slots: args.slots,
        no_slots: args._no_slots,
        props: args.props,
        metrics: args.metrics,
        warmup: args.warmup,
        no_warmup: args._no_warmup,
        temperature: args.temp,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
        repeat_last_n: args.repeat_last_n,
        repeat_penalty: args.repeat_penalty,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
        dry_multiplier: args.dry_multiplier,
        dry_base: args.dry_base,
        dry_allowed_length: args.dry_allowed_length,
        dry_penalty_last_n: args.dry_penalty_last_n,
        dry_sequence_breakers: args.dry_sequence_breakers,
        verbose: args.verbose,
        log_disable: args.log_disable,
        log_file: args.log_file,
        distributed_config: args.distributed_config,
        node_role: args.node_role,
        node_id: args.node_id,
        peers: args.peers,
        prefill_peers: args.prefill_peers,
        decode_peers: args.decode_peers,
        serving_bind: args.serving_bind,
        pp_layers: args.pp_layers,
        pp_micro_batch_size: args.pp_micro_batch_size,
        pp_auto: args.pp_auto,
        pp_peer: args.pp_peer,
        cluster_discovery: args.cluster_discovery,
        cluster_name: args.cluster_name,
        cluster_peers: args.cluster_peers,
        cluster_discovery_port: args.cluster_discovery_port,
        cluster_control_addr: args.cluster_control_addr,
        cluster_config_out: args.cluster_config_out,
        dry_run: args.dry_run,
        tp_size: args.tp_size,
        tp_moe_mode: args.tp_moe_mode,
        tp_embedding_mode: args.tp_embedding_mode,
        tp_lm_head_mode: args.tp_lm_head_mode,
        vision_cache_size: args.vision_cache_size,
        max_image_payload_size: args.max_image_payload_size,
        max_images_per_request: args.max_images_per_request,
        max_image_width: args.max_image_width,
        max_image_height: args.max_image_height,
        max_image_decode_alloc_bytes: args.max_image_decode_alloc_bytes,
        enable_elastic_pp: args.enable_elastic_pp,
        elastic_pp_drain_timeout: args.elastic_pp_drain_timeout,
        elastic_pp_pressure_fraction: args.elastic_pp_pressure_fraction,
        elastic_pp_cool_down: args.elastic_pp_cool_down,
        metrics_port: args.metrics_port,
        debug_pp_trace: args.debug_pp_trace,
        lang_bias_config,
        // route through `env_fallback_reasoning_budget` so that
        // CLI-vs-env precedence, unparseable-env handling, and the collision
        // INFO log are handled consistently with `mlxcel serve` and with the
        // other LLAMA_ARG_* env fallbacks. (Do NOT put `env = "..."` on the
        // clap arg, that bypasses our warn-and-ignore policy for unparseable
        // values and would emit a misleading collision warning.)
        reasoning_budget: {
            let mut v = args.reasoning_budget;
            env_fallback_reasoning_budget(&mut v);
            v
        },
        chat_template_kwargs: args.chat_template_kwargs,
        // prompt-cache knobs already resolved via env-var fallbacks above.
        prompt_cache_enabled: args.prompt_cache_enabled,
        prompt_cache_capacity_bytes: args.prompt_cache_capacity_bytes,
        prompt_cache_max_entries: args.prompt_cache_max_entries,
        prompt_cache_ttl_seconds: args.prompt_cache_ttl,
        prompt_cache_min_prefix: args.prompt_cache_min_prefix,
        // APC knobs already resolved via env-var fallbacks above.
        apc_enabled: args.apc_enabled,
        apc_block_size: args.apc_block_size,
        apc_num_blocks: args.apc_num_blocks,
        apc_hash: args.apc_hash,
        // (B11): KV cache type split flags already resolved via
        // env-var fallbacks (and clap `env = "..."`) above.
        cache_type_k: args.turbo.cache_type_k,
        cache_type_v: args.turbo.cache_type_v,
        kv_cache_mode_legacy: args.turbo.kv_cache_mode,
        // continuous-batching KV quantization knobs (flattened
        // from `BatchKvQuantArgs`).
        kv_bits: args.batch_quant.kv_bits,
        kv_group_size: args.batch_quant.kv_group_size,
        kv_quant_scheme: args.batch_quant.kv_quant_scheme,
        kv_skip_last_layer: args.batch_quant.kv_skip_last_layer,
        // maximum KV cache size for plain (non-sliding) caches.
        // clap reads `LLAMA_ARG_MAX_KV_SIZE` directly via the `env = ...`
        // attribute on the flag, so no separate env-fallback helper is needed.
        max_kv_size: args.max_kv_size,
        // paged KV pool block-budget directive (#122 b3); clap parses it into
        // a `PagedBudgetDirective`, resolved to a block count on the worker.
        kv_cache_budget: args.kv_cache_budget,
        // experimental VLM prompt-prefix cache toggle (#124 step c).
        enable_vlm_prefix_cache: args.enable_vlm_prefix_cache,
        // CORS allow-list origins (#244); validated in into_startup_config.
        allowed_origins: args.allowed_origins,
        // Responses API in-memory store limits. clap reads the
        // matching `LLAMA_ARG_*` env vars directly via the `env = ...`
        // attributes on the flags.
        responses_store_max_entries: args.responses_store_max_entries,
        responses_store_ttl_secs: args.responses_store_ttl_secs,
        conversation_store_max_entries: args.conversation_store_max_entries,
        conversation_store_ttl_secs: args.conversation_store_ttl_secs,
        // (A4): forward the surgery YAML path. clap reads
        // `MLXCEL_SURGERY` directly via the `env = ...` attribute on
        // the flag, so no separate env-fallback helper is needed.
        #[cfg(feature = "surgery")]
        surgery_config_path: args.surgery,
        // serve-level block-diffusion knobs (#217 phase 3); diffusion models
        // only.
        max_denoising_steps: args.max_denoising_steps,
        diffusion_sampler: args.diffusion_sampler.clone(),
        diffusion_threshold: args.diffusion_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn parse_server_args(argv: &[&str]) -> ServerArgs {
        let cli = Cli::try_parse_from(argv).expect("mlxcel-server args should parse");
        assert!(
            cli.command.is_none(),
            "test argv should exercise legacy server-start mode"
        );
        cli.server
    }

    fn make_complete_snapshot(models_root: &Path, repo_id: &str) -> PathBuf {
        let mut dir = models_root.to_path_buf();
        for segment in repo_id.split('/') {
            dir.push(segment);
        }
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), b"{}").unwrap();
        dir
    }

    #[test]
    fn legacy_server_mode_resolves_repo_id_from_models_dir_override() {
        let tmp = tempfile::tempdir().unwrap();
        let models_root = tmp.path().join("custom-model-store");
        let repo_id = "zz-mlxcel-test-owner/zz-mlxcel-test-model";
        let expected = make_complete_snapshot(&models_root, repo_id);
        let models_root_arg = models_root.to_string_lossy().to_string();

        let args = parse_server_args(&[
            "mlxcel-server",
            "-m",
            repo_id,
            "--models-dir",
            &models_root_arg,
        ]);
        let input = build_startup_input(args).expect("repo-id should resolve from override store");

        assert_eq!(input.model_path, expected);
    }

    #[test]
    fn legacy_server_mode_keeps_existing_model_path_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        let local_model = tmp.path().join("local-model");
        fs::create_dir_all(&local_model).unwrap();
        let decoy_models_root = tmp.path().join("decoy-store");
        let local_model_arg = local_model.to_string_lossy().to_string();
        let decoy_arg = decoy_models_root.to_string_lossy().to_string();

        let args = parse_server_args(&[
            "mlxcel-server",
            "-m",
            &local_model_arg,
            "--models-dir",
            &decoy_arg,
        ]);
        let input = build_startup_input(args).expect("existing path should be accepted");

        assert_eq!(input.model_path, local_model);
    }

    // ── Drafter flag aliases (issue #464) ───────────────────────
    //
    // `mlxcel-server` uses the llama-server-style `--model-draft` /
    // `--draft` spelling as the primary flag names, and `mlxcel serve`
    // uses the mlx-lm-style `--draft-model` / `--draft-max` spelling.
    // Both binaries now accept both spellings via `visible_alias`, so a
    // command line copied from one to the other parses unchanged. These
    // tests pin that both spellings resolve to the identical `ServerArgs`
    // field value here; `src/main_tests.rs` carries the matching
    // assertions for `mlxcel serve`.

    #[test]
    fn model_draft_and_draft_model_aliases_resolve_identically() {
        let primary = parse_server_args(&["mlxcel-server", "--model-draft", "models/draft"]);
        let aliased = parse_server_args(&["mlxcel-server", "--draft-model", "models/draft"]);

        assert_eq!(
            primary.model_draft, aliased.model_draft,
            "--model-draft and its --draft-model alias must resolve to the same drafter path"
        );
        assert_eq!(primary.model_draft, Some(PathBuf::from("models/draft")));
    }

    #[test]
    fn draft_and_draft_max_aliases_resolve_identically() {
        let primary = parse_server_args(&["mlxcel-server", "--draft", "24"]);
        let aliased = parse_server_args(&["mlxcel-server", "--draft-max", "24"]);

        assert_eq!(
            primary.draft, aliased.draft,
            "--draft and its --draft-max alias must resolve to the same token budget"
        );
        assert_eq!(primary.draft, 24);
    }
}
