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

use anyhow::Context;
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::path::PathBuf;

use mlxcel::cli::batch_quant_args::BatchKvQuantArgs;
use mlxcel::cli::cache_args::CacheCompatArgs;
use mlxcel::cli::chat_compat_args::ChatCompatArgs;
use mlxcel::cli::context_args::ContextCompatArgs;
use mlxcel::cli::ggml_compat_args::{GgmlCompatArgs, read_model_layer_count};
use mlxcel::cli::logging_compat_args::LoggingCompatArgs;
use mlxcel::cli::multimodal_compat_args::MultimodalCompatArgs;
use mlxcel::cli::rope_args::RopeOverrideArgs;
use mlxcel::cli::slot_args::SlotCompatArgs;
use mlxcel::cli::spec_compat_args::SpecCompatArgs;
use mlxcel::cli::speculative_args::{
    SpeculativeArgs, env_fallback_draft_block_size, env_fallback_draft_kind,
};
use mlxcel::cli::turbo_args::{TurboKvCacheArgs, resolve_kv_cache_mode};
use mlxcel::cli::ui_compat_args::UiCompatArgs;
use mlxcel::downloader::{
    DownloadArgs, DownloadOptions, ModelSourceOptions, download_repo,
    resolve_model_source_with_options, set_offline_mode,
};
use mlxcel::lang_bias::LangBiasCliArgs;
use mlxcel::server::{
    LlamaModelSourceArgs, ServerStartupInput, env_fallback_apc_block_size,
    env_fallback_apc_enabled, env_fallback_apc_hash, env_fallback_apc_num_blocks,
    env_fallback_api_key_files, env_fallback_api_keys, env_fallback_batch_size,
    env_fallback_cache_type_k, env_fallback_cache_type_v, env_fallback_chat_template_kwargs,
    env_fallback_cors_credentials, env_fallback_draft_model, env_fallback_embedding_model,
    env_fallback_endpoint_slots, env_fallback_kv_bits, env_fallback_kv_group_size,
    env_fallback_kv_quant_scheme, env_fallback_kv_skip_last_layer, env_fallback_lang_bias,
    env_fallback_lang_bias_include_byte_fragments, env_fallback_log_file, env_fallback_offline,
    env_fallback_prompt_cache_capacity_bytes, env_fallback_prompt_cache_enabled,
    env_fallback_prompt_cache_max_entries, env_fallback_prompt_cache_min_prefix,
    env_fallback_prompt_cache_snapshot_capacity_bytes,
    env_fallback_prompt_cache_snapshot_max_entries, env_fallback_prompt_cache_snapshot_ttl,
    env_fallback_prompt_cache_ttl, env_fallback_reasoning_budget, env_fallback_reranker_model,
    env_fallback_settings_endpoint, env_fallback_ubatch_size, long_cli_flag_was_set,
    resolve_llama_model_source, start_server, superseded_model_notice,
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
    // b10621 spells the help option `-h`, `--help` and `--usage`;
    // clap's generated help argument cannot carry an alias, so it is
    // declared by hand on this struct instead (#1448).
    disable_help_flag = true,
    // b10621 accepts a space-separated negative value on every option whose
    // domain admits one (`llama-server --seed -1`), and mlxcel rejected all
    // 122 of its value-taking long options with "unexpected argument '-1'"
    // until this setting was added (#1459). `ServerArgs` is flattened into
    // this command, so the setting reaches every server flag from here.
    //
    // `allow_negative_numbers`, not `allow_hyphen_values`: the latter takes
    // ANY `-`-leading token as the pending option's value, so a mistyped
    // `--seed --moldel foo` would silently bind `--moldel` as the seed and
    // never report the typo. This one admits only tokens that lex as a
    // number, which is exactly the domain b10621 documents. Options whose
    // own domain excludes negatives (`--port: u16`, `--ctx-size: usize`,
    // `parse_unit_interval` on `--diffusion-threshold`) still reject `-1` in
    // their own value parser, which is asserted in this file's test module.
    allow_negative_numbers = true,
    verbatim_doc_comment,
    after_help = "\
Model and Runtime Support:
  Checkpoint-specific capabilities and limitations:
    https://github.com/lablup/mlxcel/blob/main/docs/supported-models.md
  Distributed setup and current constraints:
    https://github.com/lablup/mlxcel/blob/main/docs/distributed.md

Model store:
  -m/--model accepts either a local path or a HuggingFace owner/name repo-id.
  Repo-ids are resolved exactly like `mlxcel serve -m`: legacy ./models/<name>,
  then the HuggingFace cache, then the mlxcel store, with auto-download on miss.
  Use --model-store-root (or MLXCEL_MODELS_DIR) to point the mlxcel store at another
  volume; snapshots live at <root>/<owner>/<name> under that root.

Embeddings and Reranking:
  -m <embedding checkpoint> serves POST /v1/embeddings without a chat model.
  --embedding-model <checkpoint> adds embeddings beside the chat model in -m.
  -m <cross-encoder checkpoint> serves POST /v1/rerank without a chat model.
  --reranker-model <checkpoint> adds reranking beside chat and is required for
  generative rerankers. Queue depth and timeout use the --embedding-* worker
  flags for both endpoints.
  Full request schemas, examples, and supported families:
    https://github.com/lablup/mlxcel/blob/main/docs/embeddings.md

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
    /// Print usage and exit.
    ///
    /// Declared by hand, with `disable_help_flag`, rather than left to clap's
    /// generated help argument: b10621 spells this option `-h`, `--help` and
    /// `--usage`, and clap's built-in help argument cannot carry an alias
    /// through the derive. Declared first so it lands before any
    /// `next_help_heading` group (#1448).
    #[arg(
        short = 'h',
        long = "help",
        visible_alias = "usage",
        action = clap::ArgAction::Help
    )]
    help: Option<bool>,

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

    /// HuggingFace repository to serve, as `<owner>/<name>`.
    ///
    /// llama-server compatible spelling. Resolved exactly like the same value
    /// passed to `-m`: reused from `./models/`, the HuggingFace cache, or the
    /// mlxcel store, and downloaded into the mlxcel store on a miss. Takes
    /// precedence over `-m` when both are given, matching llama-server. The
    /// `:<quant>` suffix llama-server accepts selects a GGUF quantization and
    /// is rejected; MLX checkpoints carry their quantization in the repository
    /// name (`mlx-community/Qwen3-4B-4bit`).
    #[arg(
        long = "hf-repo",
        env = "LLAMA_ARG_HF_REPO",
        value_name = "<owner>/<name>"
    )]
    hf_repo: Option<String>,

    /// HuggingFace access token used when downloading a repository.
    ///
    /// Takes precedence over the `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`
    /// environment variables. The value is never rendered in `--help`, logged,
    /// or written to disk.
    #[arg(
        long = "hf-token",
        env = "HF_TOKEN",
        hide_env_values = true,
        value_name = "TOKEN"
    )]
    hf_token: Option<String>,

    /// Offline mode: resolve models from local caches only, never download.
    ///
    /// A repository identifier that is not already present in `./models/`, the
    /// HuggingFace cache, or the mlxcel store is an error instead of a
    /// download. Also reads `LLAMA_ARG_OFFLINE`.
    #[arg(long = "offline", default_value_t = false)]
    offline: bool,

    /// Accepted for llama-server CLI compatibility; rejected at startup
    /// (selects one GGUF file inside a repository, and MLX loads a whole
    /// SafeTensors snapshot).
    #[arg(
        long = "hf-file",
        env = "LLAMA_ARG_HF_FILE",
        value_name = "FILE",
        hide = true
    )]
    hf_file: Option<String>,

    /// Accepted for llama-server CLI compatibility; rejected at startup
    /// (mlxcel resolves models by repository identifier, not by URL).
    #[arg(
        long = "model-url",
        env = "LLAMA_ARG_MODEL_URL",
        value_name = "MODEL_URL",
        hide = true
    )]
    model_url: Option<String>,

    /// Accepted for llama-server CLI compatibility; rejected at startup
    /// (Docker Hub model repositories distribute GGUF artifacts).
    #[arg(
        long = "docker-repo",
        env = "LLAMA_ARG_DOCKER_REPO",
        value_name = "[<repo>/]<model>[:quant]",
        hide = true
    )]
    docker_repo: Option<String>,

    /// Model-store root for resolving / downloading an `owner/name` repo-id.
    ///
    /// Sets the directory that directly holds snapshots, so a repo-id resolves
    /// to / downloads at `<PATH>/<owner>/<name>` (no extra `models/` subdir).
    /// Overrides the `MLXCEL_MODELS_DIR` environment variable. No effect when
    /// `-m/--model` is already an existing local path. This used to be
    /// spelled `--models-dir`, which now carries b10621's router semantics
    /// (#1438).
    #[arg(long = "model-store-root", value_name = "PATH")]
    model_store_root: Option<PathBuf>,

    /// Directory containing models for the router server (default: disabled)
    #[arg(long = "models-dir", env = "LLAMA_ARG_MODELS_DIR", value_name = "PATH")]
    models_dir: Option<PathBuf>,

    /// For router server, maximum number of models to load simultaneously (0 = unlimited)
    #[arg(
        long = "models-max",
        env = "LLAMA_ARG_MODELS_MAX",
        default_value_t = 4,
        value_name = "N"
    )]
    models_max: usize,

    /// For router server, whether to automatically load models
    #[arg(
        long = "models-autoload",
        env = "LLAMA_ARG_MODELS_AUTOLOAD",
        default_value_t = true,
        overrides_with = "_no_models_autoload"
    )]
    models_autoload: bool,

    /// Disable automatic model loading in router mode
    #[arg(
        long = "no-models-autoload",
        overrides_with = "models_autoload",
        hide = true
    )]
    _no_models_autoload: bool,

    /// Path to INI file containing model presets for the router server (default: disabled)
    #[arg(
        long = "models-preset",
        env = "LLAMA_ARG_MODELS_PRESET",
        value_name = "PATH"
    )]
    models_preset: Option<PathBuf>,

    /// Set model tags, comma-separated (informational, not used for routing)
    #[arg(long = "tags", env = "LLAMA_ARG_TAGS", value_name = "TAGS")]
    tags: Option<String>,

    /// Repository revision (branch, tag, or commit hash). Defaults to `main`.
    ///
    /// Resolves the HuggingFace cache snapshot for that revision, and fetches
    /// that revision on a miss. The mlxcel store is not revision-namespaced, so
    /// a repo already present there is not reused for a revision-qualified
    /// request and the request is refused rather than answered with an unknown
    /// revision; use `--model-store-root` to give each revision its own root. Not
    /// valid when `-m/--model` is an existing local path.
    #[arg(long, value_name = "REV")]
    revision: Option<String>,

    /// Model alias (shown in API responses instead of directory name)
    #[arg(
        short = 'a',
        long = "alias",
        env = "LLAMA_ARG_ALIAS",
        value_name = "NAME"
    )]
    alias: Option<String>,

    /// Path to LoRA adapter (use comma-separated values to load multiple adapters)
    #[arg(long = "lora", visible_alias = "adapter", value_name = "FNAME")]
    lora: Option<String>,

    /// Path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...) note: use comma-separated values
    #[arg(long = "lora-scaled", value_name = "FNAME:SCALE,...")]
    lora_scaled: Option<String>,

    /// Load LoRA adapters without applying them (apply later via POST /lora-adapters)
    #[arg(long = "lora-init-without-apply")]
    lora_init_without_apply: bool,

    /// Fuse LoRA adapters into the base weights at load (mlxcel-native, zero decode overhead; runtime scale changes and per-request selection are then refused)
    #[arg(long = "lora-fuse")]
    lora_fuse: bool,

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

    /// Number of parallel request slots that share --ctx-size (default: -1, -1 = auto)
    ///
    /// b10621's `-1` (the default) lets the server choose; the automatic
    /// count resolves to 4 slots, which is also what upstream's auto picks.
    /// Sets the maximum concurrent decode batch for multi-client serving:
    /// batched decode amortizes the per-step weight reads across the batch,
    /// raising aggregate throughput and keeping time-to-first-token low under
    /// concurrent load. On CUDA the amortizing kernel covers decode batches
    /// of up to 7 rows (issue #725); keep this at 7 or below there (see
    /// docs/CONTINUOUS_BATCHING.md). The scheduler clamps this to 1 for model
    /// families that cannot batch (SSM / hybrid / mixed-cache). Use `--parallel
    /// 1` (or `--no-batch`) to restore single-slot sequential serving. Both
    /// `--parallel` and `--n-parallel` are accepted on `mlxcel serve` and on
    /// `mlxcel-server`, so this flag parses on either binary.
    #[arg(
        long = "parallel",
        visible_alias = "n-parallel",
        env = "LLAMA_ARG_N_PARALLEL",
        default_value_t = -1,
        allow_hyphen_values = true
    )]
    parallel: i64,

    /// API key for authentication; multiple keys can be given as a
    /// comma-separated list
    ///
    /// Repeatable, and every occurrence adds to the same key set, matching
    /// llama-server b10621. The value is split the way b10621 splits it: a
    /// field may be quoted to contain a comma, and whitespace is NOT trimmed,
    /// so `--api-key "a, b"` configures `a` and `" b"`.
    ///
    /// `LLAMA_API_KEY` adds to the set rather than replacing it, which is
    /// also what b10621 does (it applies environment variables first and the
    /// command line second, appending both times).
    #[arg(long = "api-key", value_name = "KEY")]
    api_key: Vec<String>,

    /// Path to a file containing API keys, one per line
    ///
    /// Lines starting with `#` are comments and blank lines are skipped;
    /// nothing else is trimmed, so trailing whitespace is part of the key.
    /// Repeatable, and `LLAMA_ARG_API_KEY_FILE` adds to the same set, matching
    /// llama-server b10621.
    #[arg(long = "api-key-file", value_name = "FNAME")]
    api_key_file: Vec<PathBuf>,

    /// HTTP socket read/write timeout in seconds (llama-server `--timeout`)
    ///
    /// Bounds how long one socket read or write may block, matching
    /// llama-server b10621 and its 3600-second default. It does NOT cancel a
    /// slow generation: that is `--decode-timeout`.
    ///
    /// BREAKING (#1432): before this, `--timeout` / `LLAMA_ARG_TIMEOUT` was
    /// the decode watchdog with a 600-second default. A command line that
    /// relied on the old meaning should now pass `--decode-timeout`.
    #[arg(
        long,
        env = "LLAMA_ARG_TIMEOUT",
        default_value_t = 3600,
        value_name = "N"
    )]
    timeout: u64,

    /// Per-request decode watchdog in seconds (mlxcel-native)
    ///
    /// Cancels a request whose generation stops producing tokens. This is the
    /// control `--timeout` carried before #1432; the default is unchanged.
    /// Also reads `MLXCEL_DECODE_TIMEOUT`.
    #[arg(
        long = "decode-timeout",
        env = "MLXCEL_DECODE_TIMEOUT",
        default_value_t = 600,
        value_name = "N"
    )]
    decode_timeout: u64,

    /// Prefix path the server serves from, without the trailing slash
    ///
    /// `--api-prefix /llama` moves every route under `/llama`, so
    /// `/v1/chat/completions` becomes `/llama/v1/chat/completions`. Matching
    /// llama-server b10621, the public health endpoints are recognised by
    /// their unprefixed paths, so `/llama/health` requires authentication when
    /// an API key is configured. Also reads `LLAMA_ARG_API_PREFIX`.
    #[arg(
        long = "api-prefix",
        env = "LLAMA_ARG_API_PREFIX",
        default_value = "",
        value_name = "PREFIX"
    )]
    api_prefix: String,

    /// Server SSE ping interval in seconds (-1 = disabled)
    ///
    /// Interval between SSE comment pings emitted while a stream stays silent,
    /// which is what keeps a reverse proxy from dropping a long prefill. Also
    /// reads `LLAMA_ARG_SSE_PING_INTERVAL`.
    #[arg(
        long = "sse-ping-interval",
        env = "LLAMA_ARG_SSE_PING_INTERVAL",
        default_value_t = 30,
        value_name = "N",
        allow_negative_numbers = true
    )]
    sse_ping_interval: i64,

    /// Number of threads used to process HTTP requests (-1 = automatic)
    ///
    /// mlxcel serves HTTP on a Tokio runtime, so this sizes the runtime's
    /// worker threads. `-1` uses llama-server b10621's own formula,
    /// `max(--parallel + 4, cores - 1)`. Also reads `LLAMA_ARG_THREADS_HTTP`.
    #[arg(
        long = "threads-http",
        env = "LLAMA_ARG_THREADS_HTTP",
        default_value_t = -1,
        value_name = "N",
        allow_negative_numbers = true
    )]
    threads_http: i64,

    /// Allow multiple sockets to bind to the same port (SO_REUSEPORT)
    ///
    /// Also reads `LLAMA_ARG_REUSE_PORT`.
    #[arg(long = "reuse-port", env = "LLAMA_ARG_REUSE_PORT")]
    reuse_port: bool,

    /// Path to a file with a PEM-encoded SSL certificate
    ///
    /// Serves HTTPS instead of HTTP. Must be given together with
    /// `--ssl-key-file`. Also reads `LLAMA_ARG_SSL_CERT_FILE`.
    #[arg(
        long = "ssl-cert-file",
        env = "LLAMA_ARG_SSL_CERT_FILE",
        value_name = "FNAME"
    )]
    ssl_cert_file: Option<PathBuf>,

    /// Path to a file with a PEM-encoded SSL private key
    ///
    /// Must be given together with `--ssl-cert-file`. Also reads
    /// `LLAMA_ARG_SSL_KEY_FILE`.
    #[arg(
        long = "ssl-key-file",
        env = "LLAMA_ARG_SSL_KEY_FILE",
        value_name = "FNAME"
    )]
    ssl_key_file: Option<PathBuf>,

    /// Allowed origins for CORS (default: `*`)
    ///
    /// Follows llama-server b10621: the value is emitted verbatim as
    /// `Access-Control-Allow-Origin`, except for `*` (which echoes the request
    /// `Origin` while `--cors-credentials` is on) and the special value
    /// `localhost` (which echoes the `Origin` only when its host is
    /// `localhost`, `127.0.0.1` or `::1`). For per-origin matching against a
    /// list, use the mlxcel-native `--allowed-origins` instead; the two are
    /// mutually exclusive. Also reads `LLAMA_ARG_CORS_ORIGINS`.
    #[arg(
        long = "cors-origins",
        env = "LLAMA_ARG_CORS_ORIGINS",
        default_value = "*",
        value_name = "ORIGINS",
        conflicts_with = "allowed_origins"
    )]
    cors_origins: String,

    /// Comma-separated list of allowed methods for CORS
    ///
    /// Sent as `Access-Control-Allow-Methods` on a preflight. Also reads
    /// `LLAMA_ARG_CORS_METHODS`.
    #[arg(
        long = "cors-methods",
        env = "LLAMA_ARG_CORS_METHODS",
        default_value = "GET, POST, DELETE, OPTIONS",
        value_name = "METHODS"
    )]
    cors_methods: String,

    /// Comma-separated list of allowed headers for CORS
    ///
    /// Sent as `Access-Control-Allow-Headers` on a preflight. Also reads
    /// `LLAMA_ARG_CORS_HEADERS`.
    #[arg(
        long = "cors-headers",
        env = "LLAMA_ARG_CORS_HEADERS",
        default_value = "*",
        value_name = "HEADERS"
    )]
    cors_headers: String,

    /// Whether to allow credentials for CORS (default: enabled)
    ///
    /// With this enabled and `--cors-origins *` (the default), the `Origin`
    /// header is echoed back and credentials are always allowed, matching
    /// llama-server b10621. Also reads `LLAMA_ARG_CORS_CREDENTIALS`.
    #[arg(
        long = "cors-credentials",
        default_value_t = true,
        overrides_with = "_no_cors_credentials"
    )]
    cors_credentials: bool,

    /// Disable CORS credentials (llama-server `--no-cors-credentials`)
    #[arg(
        long = "no-cors-credentials",
        overrides_with = "cors_credentials",
        hide = true
    )]
    _no_cors_credentials: bool,

    /// Path to drafter checkpoint for server speculative decoding
    ///
    /// Accepts the llama-server-style `--model-draft` spelling (primary) and
    /// the mlx-lm-style `--draft-model` spelling (alias, matches `mlxcel
    /// serve`) so commands copied between the two binaries work unchanged.
    ///
    /// Whether the singleton (B=1) speculative burst runs is decided by an
    /// adaptive policy: the first few requests per (target, drafter,
    /// hardware, block size) pairing are profiled with measured classic-step
    /// probes, and the pairing is enabled or declined from the measured
    /// speedup. `MLXCEL_ENABLE_MTP_B1` pins the decision, and
    /// `MLXCEL_MTP_ADAPTIVE=0` restores the static per-hardware gates. See
    /// docs/CONTINUOUS_BATCHING.md and docs/environment-variables.md.
    #[arg(
        long = "model-draft",
        visible_alias = "draft-model",
        alias = "spec-draft-model",
        env = "LLAMA_ARG_SPEC_DRAFT_MODEL",
        value_name = "PATH"
    )]
    model_draft: Option<PathBuf>,

    /// Maximum number of draft tokens per speculation step. Accepts the
    /// b10621 canonical spelling --spec-draft-n-max and its env; the removed
    /// b10621 spellings --draft / --draft-max stay as compatibility aliases
    /// (b10621 itself errors on them), and the legacy LLAMA_ARG_DRAFT_MAX
    /// env is honored as a fallback when the canonical one is unset.
    #[arg(
        long = "draft",
        visible_alias = "draft-max",
        alias = "spec-draft-n-max",
        alias = "draft-n",
        env = "LLAMA_ARG_SPEC_DRAFT_N_MAX",
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

    /// Embedding checkpoint to serve on /v1/embeddings next to the chat model
    ///
    /// A local directory or a HuggingFace `owner/name` repo-id (resolved like
    /// `-m`). Loads on its own worker thread; `-m` keeps serving chat. When
    /// `-m` itself is an embedding checkpoint this flag must be omitted (that
    /// checkpoint is then served on /v1/embeddings and chat stays unloaded).
    /// Also reads `MLXCEL_EMBEDDING_MODEL`.
    #[arg(
        long = "embedding-model",
        env = "LLAMA_ARG_EMBEDDING_MODEL",
        value_name = "PATH_OR_REPO_ID"
    )]
    embedding_model: Option<String>,

    /// Texts per embedding forward pass (default: 16)
    ///
    /// `/v1/embeddings` sorts text inputs by token length and cuts them into
    /// micro-batches of this size, each right-padded to its longest member.
    #[arg(
        long = "embedding-batch-size",
        env = "MLXCEL_EMBEDDING_BATCH_SIZE",
        default_value_t = 16
    )]
    embedding_batch_size: usize,

    /// Token cap per embedding input (default: derived from the checkpoint)
    ///
    /// Lowers the limit derived from `sentence_bert_config.json`,
    /// `tokenizer_config.json` and `config.json` (hard cap 8192). Longer
    /// inputs are truncated from the right, keeping a trailing special token.
    #[arg(
        long = "embedding-max-length",
        env = "MLXCEL_EMBEDDING_MAX_LENGTH",
        value_name = "N"
    )]
    embedding_max_length: Option<usize>,

    /// Bound on each embedding/reranking worker command queue; a full queue returns 503 (default: 8)
    #[arg(
        long = "embedding-queue-depth",
        env = "MLXCEL_EMBEDDING_QUEUE_DEPTH",
        default_value_t = 8
    )]
    embedding_queue_depth: usize,

    /// Per-request embedding/reranking reply timeout in seconds; 0 uses the default (default: 120)
    #[arg(
        long = "embedding-request-timeout-secs",
        env = "MLXCEL_EMBEDDING_REQUEST_TIMEOUT_SECS",
        default_value_t = 120
    )]
    embedding_request_timeout_secs: u64,

    /// Reranker checkpoint to serve on /v1/rerank next to the chat model
    ///
    /// A local directory or a HuggingFace `owner/name` repo-id (resolved like
    /// `-m`). Loads on its own worker thread; `-m` keeps serving chat. A
    /// one-label `ForSequenceClassification` cross-encoder is also detected
    /// from `-m` alone; the Qwen3 and Qwen3-VL generative rerankers are
    /// indistinguishable from chat checkpoints and are only reachable through
    /// this flag. Naming the same directory in `-m` and here serves that one
    /// checkpoint as a reranker and leaves chat unloaded. Also reads
    /// `MLXCEL_RERANKER_MODEL`.
    #[arg(
        long = "reranker-model",
        env = "LLAMA_ARG_RERANKER_MODEL",
        value_name = "PATH_OR_REPO_ID"
    )]
    reranker_model: Option<String>,

    /// Query/document pairs per rerank forward pass (0 = the reranker kind's default)
    ///
    /// `/v1/rerank` scores documents in micro-batches of this size. The
    /// default is 8 for a text reranker and 2 for the multimodal one, whose
    /// rows each carry a full image's worth of visual tokens.
    #[arg(
        long = "rerank-batch-size",
        env = "MLXCEL_RERANK_BATCH_SIZE",
        default_value_t = 0,
        value_name = "N"
    )]
    rerank_batch_size: usize,

    /// Prefill chunk size in tokens (0 = disabled, default: 512)
    #[arg(long = "prefill-chunk-size", default_value_t = 512)]
    prefill_chunk_size: usize,

    /// Decode ticks a parked chunked prefill yields before it is granted one
    /// (#1011).
    ///
    /// A prompt longer than `--prefill-chunk-size` admitted next to a busy
    /// decode batch runs one chunk and is then parked. This bounds how long it
    /// stays parked: after N consecutive decode ticks the next tick is granted
    /// to the prefill, so a C-chunk prompt reaches its first token within
    /// `C * (N + 1)` ticks however long the batch keeps decoding. The price is
    /// paid by the decoding streams, whose mean inter-token latency during that
    /// window rises by roughly one chunk forward per N decode steps, so this is
    /// the TTFT-versus-ITL dial: lower is faster to first token and noisier for
    /// everyone else. `0` disables the grant and restores the pre-#1011
    /// behaviour, in which a parked prefill waits for the batch to drain and
    /// its time to first token has no bound. Env:
    /// `MLXCEL_PREFILL_GRANT_INTERVAL` (the flag wins).
    #[arg(long = "prefill-grant-interval", value_name = "N")]
    prefill_grant_interval: Option<usize>,

    /// Prefill batch size [llama-server alias for --prefill-chunk-size] [default: 512]
    #[arg(
        short = 'b',
        long = "batch-size",
        env = "LLAMA_ARG_BATCH",
        value_name = "N"
    )]
    batch_size: Option<usize>,

    /// Physical micro-batch size [not applicable on Apple Silicon unified memory; ignored]
    #[arg(long = "ubatch-size", env = "LLAMA_ARG_UBATCH", value_name = "N")]
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

    /// Maximum number of requests to batch together for prefill (default: 4)
    ///
    /// When > 1, the scheduler collects up to this many pending requests and
    /// runs a single batched forward pass [batch_size, max_seq_len] for better
    /// Neural Accelerator utilization and lower time-to-first-token under
    /// concurrent arrivals. Only engages for model families that opt into
    /// batched prefill (`supports_batched_prefill()`); other families fall back
    /// to sequential prefill automatically. Set to 1 to disable.
    #[arg(long = "max-batch-prefill", default_value_t = 4)]
    max_batch_prefill: usize,

    /// Cap on the transient memory of a single batched prefill (#715).
    ///
    /// The batched-prefill path pads every prompt in a cohort to the window's
    /// longest prompt L and materializes a stacked `[B, L, L]` FP32 attention
    /// mask, an O(B*L^2) transient. This caps the drained window by total
    /// padded tokens (rows * L): rows past the budget spill to the next tick
    /// and prefill via the chunked single-sequence path. Unset derives the
    /// default `2 * max_batch_prefill * prefill_chunk_size` (2 * 4 * 512 = 4096).
    /// `0` disables the cap (uncapped). Env: `MLXCEL_MAX_BATCH_PREFILL_TOKENS`
    /// overrides both.
    #[arg(long = "max-batch-prefill-tokens")]
    max_batch_prefill_tokens: Option<usize>,

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

    /// Paged KV-cache pool block budget: `auto` (default), a byte count, or `none`.
    ///
    /// Bounds the unified paged KV cache (epic #116): `auto` derives the cap
    /// from the memory estimate, a raw byte count sets it explicitly, and
    /// `none` / `0` leaves the pool unbounded. Only affects pool-backed (Fp16)
    /// models under the paged decode backend (the `--parallel > 1` default);
    /// dense-backend workers ignore it. Defaults to `auto` so the batched-decode
    /// default cannot run concurrent full-context sequences into an OOM abort;
    /// admission returns clean backpressure instead. Also reads
    /// `MLXCEL_KV_CACHE_BUDGET`.
    #[arg(
        long = "kv-cache-budget",
        env = "MLXCEL_KV_CACHE_BUDGET",
        value_name = "BYTES|auto|none",
        default_value = "auto",
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

    /// Approximate byte budget for the OpenAI `/v1/responses` store.
    /// `0` keeps response storage enabled but immediately evicts every stored
    /// response. Also reads `MLXCEL_RESPONSES_STORE_MAX_BYTES`.
    #[arg(
        long = "responses-store-max-bytes",
        env = "MLXCEL_RESPONSES_STORE_MAX_BYTES",
        default_value_t = mlxcel::server::responses_store::DEFAULT_RESPONSES_STORE_MAX_BYTES,
        value_name = "BYTES"
    )]
    responses_store_max_bytes: usize,

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

    /// Approximate byte budget for conversation transcripts.
    /// `0` keeps the conversation store enabled but immediately evicts every
    /// transcript. Also reads `MLXCEL_CONVERSATION_STORE_MAX_BYTES`.
    #[arg(
        long = "conversation-store-max-bytes",
        env = "MLXCEL_CONVERSATION_STORE_MAX_BYTES",
        default_value_t = mlxcel::server::conversation_store::DEFAULT_CONVERSATION_STORE_MAX_BYTES,
        value_name = "BYTES"
    )]
    conversation_store_max_bytes: usize,

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
    #[arg(
        long = "chat-template",
        env = "LLAMA_ARG_CHAT_TEMPLATE",
        value_name = "TEMPLATE"
    )]
    chat_template: Option<String>,

    /// Path to chat template file
    #[arg(
        long = "chat-template-file",
        env = "LLAMA_ARG_CHAT_TEMPLATE_FILE",
        value_name = "PATH"
    )]
    chat_template_file: Option<PathBuf>,

    /// Alias emitted next to reasoning_content on Chat Completions responses
    #[arg(
        long = "reasoning-alias-field",
        env = "MLXCEL_REASONING_ALIAS_FIELD",
        default_value_t = mlxcel::server::ReasoningAliasField::default(),
        value_name = "none|reasoning"
    )]
    reasoning_alias_field: mlxcel::server::ReasoningAliasField,

    /// Enable /slots endpoint
    #[arg(long = "slots", default_value_t = true, overrides_with = "_no_slots")]
    slots: bool,

    /// Disable /slots endpoint
    #[arg(long = "no-slots", overrides_with = "slots", hide = true)]
    _no_slots: bool,

    /// Enable /props endpoint
    #[arg(long = "props", env = "LLAMA_ARG_ENDPOINT_PROPS")]
    props: bool,

    /// Enable /metrics endpoint
    #[arg(long = "metrics", env = "LLAMA_ARG_ENDPOINT_METRICS")]
    metrics: bool,

    /// Enable authenticated GET/PATCH /v1/settings endpoints
    #[arg(long = "settings")]
    settings: bool,

    /// b10621 `--sleep-idle-seconds`: sleep after this many idle seconds,
    /// freeing the model until the next request wakes it (-1 disables).
    #[arg(
        long = "sleep-idle-seconds",
        value_name = "SECONDS",
        default_value_t = -1,
        allow_negative_numbers = true,
        hide = true
    )]
    sleep_idle_seconds: i64,

    /// Path to save slot kv cache (default: disabled)
    #[arg(long = "slot-save-path", value_name = "PATH")]
    slot_save_path: Option<PathBuf>,

    /// Enable model warmup on startup
    #[arg(long = "warmup", overrides_with = "_no_warmup", default_value_t = true)]
    warmup: bool,

    /// Disable model warmup on startup
    #[arg(long = "no-warmup", overrides_with = "warmup", hide = true)]
    _no_warmup: bool,

    // Default sampling parameters.
    /// Default sampling temperature
    #[arg(long = "temp", visible_alias = "temperature", default_value_t = 0.8)]
    temp: f32,

    /// Default top-K sampling
    #[arg(long = "top-k", env = "LLAMA_ARG_TOP_K", default_value_t = 40)]
    top_k: i32,

    /// Default top-P (nucleus) sampling
    #[arg(long = "top-p", default_value_t = 0.95)]
    top_p: f32,

    /// Default min-P sampling
    #[arg(long = "min-p", default_value_t = 0.05)]
    min_p: f32,

    /// Default locally typical sampling, parameter p (1.0 = disabled)
    #[arg(
        long = "typical",
        alias = "typical-p",
        value_name = "N",
        default_value_t = 1.0
    )]
    typical_p: f32,

    /// Default top-n-sigma sampling (-1.0 or 0.0 = disabled, b10621 sentinel)
    #[arg(
        long = "top-nsigma",
        alias = "top-n-sigma",
        value_name = "N",
        default_value_t = -1.0,
        allow_negative_numbers = true
    )]
    top_n_sigma: f32,

    /// Default XTC removal probability (0.0 = disabled)
    #[arg(long = "xtc-probability", value_name = "N", default_value_t = 0.0)]
    xtc_probability: f32,

    /// Default XTC probability threshold (values above 0.5 make XTC inert)
    #[arg(long = "xtc-threshold", value_name = "N", default_value_t = 0.1)]
    xtc_threshold: f32,

    /// Suppress end-of-generation tokens so generation runs to the token budget or a stop string (b10621 --ignore-eos)
    #[arg(long = "ignore-eos")]
    ignore_eos: bool,

    /// Add a server-wide stop string, merged into every request's stop set (repeatable; b10621 -r / --reverse-prompt)
    #[arg(short = 'r', long = "reverse-prompt", value_name = "PROMPT")]
    reverse_prompt: Vec<String>,

    /// Sampler chain order. mlxcel's chain order is fixed to b10621's default; any other order is rejected at startup
    #[arg(long = "samplers", value_name = "SAMPLERS")]
    samplers: Option<String>,

    /// Sampler chain order in b10621's single-character form; same fixed order rule as --samplers
    #[arg(long = "sampler-seq", alias = "sampling-seq", value_name = "SEQUENCE")]
    sampler_seq: Option<String>,

    /// Random seed (-1 = random; folded into b10621's uint32 seed space, so -2 is the deterministic seed 4294967294)
    #[arg(
        short = 's',
        long = "seed",
        default_value = "-1",
        allow_negative_numbers = true,
        value_parser = mlxcel::server::cli_input::parse_seed_arg
    )]
    seed: i128,

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

    /// DRY lookback window (0 = disable DRY, matching b10621; negatives
    /// rejected at parse time exactly as b10621 does)
    #[arg(long = "dry-penalty-last-n", default_value_t = 64, value_parser = clap::value_parser!(i32).range(0..))]
    dry_penalty_last_n: i32,

    /// DRY sequence breaker strings (default: "\n", ":", "\"", "*"; "none" = no breakers)
    ///
    /// Sets the server-wide default that a request without its own
    /// `dry_sequence_breakers` field inherits; a request that sends the field
    /// overrides it. When the flag is absent, b10621's default breaker set
    /// (`\n`, `:`, `"`, `*`) applies; giving the flag replaces that default
    /// with exactly the given values, and the sentinel value `none` runs DRY
    /// with no breakers at all, both matching b10621 (#1485).
    ///
    /// A breaker no longer has to encode to one token: breaker token data is
    /// derived by scanning the vocabulary for tokens whose decoded text
    /// carries the breaker string (upstream's
    /// `get_overlapping_token_sequences`), so multi-character and
    /// multi-token breakers work as they do in b10621. Strings longer than
    /// 40 characters are truncated with a warning, upstream's cap. The
    /// escapes `\n`, `\t`, `\r` and `\\` are interpreted, since a shell does
    /// not expand them inside quotes; any other backslash sequence is taken
    /// literally. The value is comma-separated, so a comma cannot itself be
    /// a breaker.
    ///
    /// The singular `--dry-sequence-breaker` is the primary spelling on both
    /// server binaries, matching llama-server. The plural
    /// `--dry-sequence-breakers` is accepted as an alias on both, so no
    /// command line that worked before stops working.
    #[arg(
        long = "dry-sequence-breaker",
        visible_alias = "dry-sequence-breakers",
        value_delimiter = ','
    )]
    dry_sequence_breakers: Vec<String>,

    /// BNF-like grammar to constrain generations (see samples in grammars/ dir)
    #[arg(long = "grammar", value_name = "GRAMMAR")]
    grammar: Option<String>,

    /// file to read grammar from
    #[arg(long = "grammar-file", value_name = "FNAME")]
    grammar_file: Option<std::path::PathBuf>,

    /// JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead
    #[arg(short = 'j', long = "json-schema", value_name = "SCHEMA")]
    json_schema: Option<String>,

    /// File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead
    #[arg(long = "json-schema-file", value_name = "FILE")]
    json_schema_file: Option<std::path::PathBuf>,

    /// Mirostat sampling (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0). Replaces the truncation samplers and penalties while active, as in b10621
    #[arg(long = "mirostat", value_name = "N", default_value_t = 0)]
    mirostat: i32,

    /// Mirostat learning rate, parameter eta
    #[arg(long = "mirostat-lr", value_name = "N", default_value_t = 0.1)]
    mirostat_eta: f32,

    /// Mirostat target entropy, parameter tau
    #[arg(long = "mirostat-ent", value_name = "N", default_value_t = 5.0)]
    mirostat_tau: f32,

    /// Dynamic temperature range (0.0 = disabled)
    #[arg(long = "dynatemp-range", value_name = "N", default_value_t = 0.0)]
    dynatemp_range: f32,

    /// Dynamic temperature exponent
    #[arg(long = "dynatemp-exp", value_name = "N", default_value_t = 1.0)]
    dynatemp_exponent: f32,

    /// adaptive-p: select tokens near this probability (0.0 to 1.0; negative = disabled). Runs only when the sampler list names adaptive_p, as in b10621
    #[arg(
        long = "adaptive-target",
        value_name = "N",
        default_value_t = -1.0,
        allow_negative_numbers = true
    )]
    adaptive_target: f32,

    /// adaptive-p: decay rate for target adaptation over time (0.0 to 0.99)
    #[arg(long = "adaptive-decay", value_name = "N", default_value_t = 0.9)]
    adaptive_decay: f32,

    /// Bias a token id in every request: TOKEN_ID(+/-)BIAS, e.g. 15043+1 raises token 15043, 15043-1 lowers it (repeatable; b10621 -l / --logit-bias)
    #[arg(short = 'l', long = "logit-bias", value_name = "TOKEN_ID(+/-)BIAS")]
    logit_bias: Vec<String>,

    // Logging.
    /// Enable verbose logging: every mlxcel message.
    ///
    /// Equivalent to the top `--verbosity` tier, and unconditional: a
    /// command-line `--verbose` beats `RUST_LOG`, matching b10621's `-v`
    /// (#1448). `--log-verbose` is the b10621 twin spelling.
    #[arg(short = 'v', long = "verbose", visible_alias = "log-verbose")]
    verbose: bool,

    /// Disable all logging
    #[arg(long = "log-disable")]
    log_disable: bool,

    /// Log output file
    #[arg(long = "log-file", env = "LLAMA_ARG_LOG_FILE", value_name = "PATH")]
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

    /// llama-server b10621 GGML runtime, placement, and memory options
    /// (`--n-gpu-layers`, `--split-mode`, `--mlock`, `--numa`, `--rpc`, the
    /// CPU thread-pool knobs, ...). Every one is hidden and its value is
    /// classified at startup: inert values are accepted, values whose b10621
    /// meaning mlxcel cannot reproduce are rejected with a diagnostic. Defined
    /// once in `mlxcel::cli::ggml_compat_args` so both server binaries accept
    /// exactly the same set (issue #1445).
    #[command(flatten)]
    ggml_compat: GgmlCompatArgs,

    /// b10621 speculative compatibility surface (#1433), defined once in
    /// `mlxcel::cli::spec_compat_args` so both server binaries accept the
    /// same spellings and classify the same values.
    #[command(flatten)]
    spec_compat: SpecCompatArgs,

    /// llama-server b10621 chat-template, reasoning, and output-parsing
    /// options (`--reasoning`, `--reasoning-format`, `--skip-chat-parsing`,
    /// `--prefill-assistant`, ...). Hidden compatibility surfaces, classified
    /// at startup like the GGML group. Defined once in
    /// `mlxcel::cli::chat_compat_args` so both server binaries accept exactly
    /// the same set (issue #1447).
    #[command(flatten)]
    chat_compat: ChatCompatArgs,

    /// llama-server b10621 multimodal projector and media options
    /// (`--mmproj`, `--mmproj-url`, `--mmproj-auto`, `--mmproj-offload`,
    /// `--mmproj-device`, `--image-min-tokens`, `--image-max-tokens`,
    /// `--mtmd-batch-max-tokens`, `--media-path`). Every projector flag is a
    /// hidden compatibility surface classified at startup like the GGML group;
    /// `--media-path` is a real mlxcel feature and is visible. Defined once in
    /// `mlxcel::cli::multimodal_compat_args` so both server binaries accept
    /// exactly the same set (issue #1451).
    #[command(flatten)]
    multimodal_compat: MultimodalCompatArgs,

    /// b10621 Web UI / tools / MCP / CORS-proxy / agent compatibility
    /// surface, shared with `mlxcel serve` through
    /// `mlxcel::cli::ui_compat_args`: inert forms accepted, enabling forms
    /// refused at startup (issue #1435).
    #[command(flatten)]
    ui_compat: UiCompatArgs,

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

    /// RoPE / YaRN runtime-override flag group (`--rope-scaling`,
    /// `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`, and the five
    /// `--yarn-*` knobs). Defined once in `mlxcel::cli::rope_args` so both
    /// server binaries (`mlxcel serve`, `mlxcel-server`) accept the same
    /// llama-server b10621 command line. Resolved before the model is loaded;
    /// see `ServerStartupConfig::rope_override`.
    #[command(flatten)]
    rope: RopeOverrideArgs,

    /// Prompt-cache and continuous-batching flag group (`--cache-prompt`,
    /// `--no-cache-prompt`, `--cache-reuse`, `--cache-ram`, `--cont-batching`,
    /// `--no-cont-batching`). Defined once in `mlxcel::cli::cache_args` so both
    /// server binaries accept the same llama-server b10621 command line.
    #[command(flatten)]
    cache_compat: CacheCompatArgs,

    /// Context-retention flag group (`--context-shift`, `--no-context-shift`,
    /// `--keep`, `--swa-full`). Defined once in `mlxcel::cli::context_args` so
    /// both server binaries accept the same llama-server b10621 command line.
    #[command(flatten)]
    context_compat: ContextCompatArgs,

    /// Slot-state and context-checkpoint flag group (`--cache-idle-slots`,
    /// `--slot-prompt-similarity`, `--kv-unified`, `--ctx-checkpoints`,
    /// `--checkpoint-min-step`). Defined once in `mlxcel::cli::slot_args` so
    /// both server binaries refuse the same command lines with one message.
    #[command(flatten)]
    slot_compat: SlotCompatArgs,

    /// Fill-in-the-middle flag group (`--spm-infill`). Defined once in
    /// `mlxcel::cli::infill_args` so both server binaries accept the same
    /// llama-server b10621 command line.
    #[command(flatten)]
    infill: mlxcel::cli::infill_args::InfillArgs,

    /// llama-server b10621 embedding and reranking mode flag group
    /// (`--embedding`, `--rerank`, `--pooling`, `--embd-normalize`). Defined
    /// once in `mlxcel::cli::embedding_compat_args` so both server binaries
    /// accept the same command line.
    #[command(flatten)]
    embedding_compat: mlxcel::cli::embedding_compat_args::EmbeddingCompatArgs,

    /// llama-server b10621 logging, introspection, and built-in preset
    /// options (`--log-colors`, `--log-prefix`, `--log-timestamps`,
    /// `--verbosity`, `--cache-list`, `--completion-bash`, `--log-prompts-dir`
    /// and the twelve GGUF model presets). Defined once in
    /// `mlxcel::cli::logging_compat_args` so both server binaries accept
    /// exactly the same set (issue #1448).
    #[command(flatten)]
    logging_compat: LoggingCompatArgs,

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
    /// reads from canonical `LLAMA_ARG_THINK_BUDGET` and legacy
    /// `LLAMA_ARG_REASONING_BUDGET` (applied via
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
    /// when the CLI flag is not explicitly provided. `LLAMA_ARG_CACHE_REUSE`
    /// is a separate integer minimum-chunk setting; only `0` is supported.
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

    /// Disable the prompt-prefix KV cache (shorthand for
    /// `--prompt-cache-enabled=false`).
    ///
    /// The prompt cache is on by default; this flag is a clean opt-out that
    /// overrides `--prompt-cache-enabled` and `MLXCEL_PROMPT_CACHE_ENABLED`.
    /// When set, repeated shared prefixes
    /// (for example a long system prompt) are re-prefilled every request.
    #[arg(long = "no-prompt-cache")]
    no_prompt_cache: bool,

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

    /// Byte budget for the exact-prefix snapshot store (default: 512 MiB).
    ///
    /// Snapshot-only families (SSM / linear-attention) park a whole recurrent
    /// state per conversation. That state scales with model width, not with
    /// prompt length: a few MiB on a small model, 300 MB or more on a 30B-class
    /// one. The 512 MiB default suits small and medium models; for a large one,
    /// size the store from measurement: read `snapshot_bytes` on
    /// `/v1/cache/stats` after one turn, then multiply by the number of entries
    /// you expect to hold at once. Count turns as well as concurrent sessions,
    /// because a conversation keeps one entry per turn until its snapshots
    /// start superseding each other (see `--prompt-cache-snapshot-max-entries`).
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_SNAPSHOT_CAPACITY_BYTES` when the CLI
    /// flag is absent. CLI flag takes precedence.
    #[arg(long = "prompt-cache-snapshot-capacity-bytes", value_name = "BYTES")]
    prompt_cache_snapshot_capacity_bytes: Option<usize>,

    /// Maximum number of live snapshot entries (default: 4096).
    ///
    /// Once the limit is reached, the least-recently-used snapshot is evicted.
    /// A conversation's turns collapse to one entry once each turn's snapshot
    /// extends the previous turn's token vector, which the current donate path
    /// does not yet produce, so budget for turns as well as for concurrent
    /// conversations until turn-boundary capture lands.
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_SNAPSHOT_MAX_ENTRIES` when the CLI flag
    /// is absent. CLI flag takes precedence.
    #[arg(long = "prompt-cache-snapshot-max-entries", value_name = "N")]
    prompt_cache_snapshot_max_entries: Option<usize>,

    /// Time-to-live for a snapshot entry in seconds (default: 7200).
    ///
    /// Snapshots outlive detached KV entries by default because multi-turn
    /// chat has longer gaps between turns than a burst of similar prompts.
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_SNAPSHOT_TTL` when the CLI flag is
    /// absent. CLI flag takes precedence.
    #[arg(long = "prompt-cache-snapshot-ttl", value_name = "SECONDS")]
    prompt_cache_snapshot_ttl: Option<u64>,

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

fn main() -> anyhow::Result<()> {
    // Hidden machine interface for the llama-server b10621 compatibility
    // manifest (issue #1443): `mlxcel-server --dump-flag-surface` prints the
    // complete clap surface, hidden compatibility arguments included, as
    // deterministic JSON and exits. Intercepted before `Cli::parse` and
    // matched positionally over `args_os` (never `args`, which panics on a
    // non-UTF-8 argument), so the operator-facing `--help` surface and
    // ordinary argument values are unaffected. See `src/cli/flag_surface.rs`
    // for the contract and consumers.
    let raw_args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if mlxcel::cli::flag_surface::dump_requested(&raw_args, 1) {
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        println!(
            "{}",
            mlxcel::cli::flag_surface::flag_surface_json("mlxcel-server", &mut cmd)
        );
        return Ok(());
    }

    // llama.cpp writes its multi-letter options with one dash (`-hf`, `-hft`,
    // `-mu`); clap reads that as a cluster of one-letter shorts, so `-hf`
    // parses as `-h -f` and renders `--help` with exit status 0. Rewrite those
    // exact tokens to their long spellings before clap sees them, so a
    // llama-server command line reaches the real option (issue #1434).
    let cli = {
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        let args =
            mlxcel::cli::llama_short_flags::expand_llama_short_options(&mut cmd, raw_args, 1);
        Cli::parse_from(args)
    };

    // b10621 introspection options (issue #1448): `--cache-list` reports the
    // model store and `--completion-bash` prints a completion script, both
    // before any model is resolved and both exiting 0, exactly as upstream's
    // parser-level handlers do. Placed here so neither needs `-m` and neither
    // pays for the runtime, the MLX environment defaults, or a model load.
    if let Some(action) = cli.server.logging_compat.early_action() {
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        print!(
            "{}",
            mlxcel::cli::logging_compat_args::render_early_action(
                action,
                "mlxcel-server",
                "mlxcel-server",
                &mut cmd,
                cli.server.model_store_root.as_deref(),
            )
        );
        return Ok(());
    }

    // Default the CUDA kernel JIT cache to a persistent, MLX-pin-scoped dir so
    // the first-run kernel compilation is paid once per machine, not every boot.
    mlxcel_core::ensure_persistent_ptx_cache();

    // Raise MLX_MAX_OPS_PER_BUFFER on pre-M5 Apple Silicon to close decode
    // command-buffer dispatch-gap idle (#353). Hardware-gated, a no-op when the
    // variable is already set, and must run before any MLX op.
    mlxcel_core::hardware::apply_metal_ops_per_buffer_default();

    // On a CUDA build, raise MLX_CUDA_GRAPH_CACHE_SIZE so long-lived,
    // shape-diverse decode (especially speculative/batched) does not abort on
    // MLX's fatal "Cache thrashing" throw at the default capacity (#818). No-op
    // off CUDA, a no-op when the variable is already set, and must run before
    // any MLX op.
    mlxcel_core::hardware::apply_cuda_graph_cache_default();

    // Publish autotuned CUDA kernel knobs (qmm CTA tile, multirow-qmv row
    // window) into the environment the patched MLX kernels read (#906). Inert
    // unless MLXCEL_AUTOTUNE is set and a tuned entry exists, never overwrites
    // an operator-set variable, and must run before any MLX op or worker thread.
    for (var, value) in mlxcel_core::autotune::ops::apply_tuned_cuda_kernel_env(
        mlxcel_core::autotune::ops::cuda_kernel_knobs::TILE_M_CAP_BLACKWELL,
    ) {
        tracing::info!("autotune: applied {var}={value} from the tactic cache");
    }

    // The Tokio runtime is built here rather than by `#[tokio::main]` so
    // `--threads-http` can size its worker pool (#1432); llama-server b10621
    // sizes its own HTTP thread pool from the same flag. Building it after the
    // environment mutations above also keeps those on a single-threaded main
    // with no runtime in existence, which is stricter than the previous
    // "workers are still parked" argument.
    let workers = mlxcel::server::transport::resolve_http_threads(
        cli.server.threads_http,
        mlxcel::server::resolve_n_parallel(cli.server.parallel).unwrap_or(4),
    );
    let runtime = mlxcel::server::transport::build_http_runtime(workers)
        .context("failed to build the HTTP runtime; check --threads-http")?;

    runtime.block_on(async move {
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
    })
}

fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    let opts = DownloadOptions::from_args(&args);
    download_repo(opts)
}

fn build_startup_input(mut args: ServerArgs) -> anyhow::Result<ServerStartupInput> {
    env_fallback_settings_endpoint(&mut args.settings, long_cli_flag_was_set("settings"));
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
    env_fallback_reasoning_budget(
        &mut args.reasoning_budget,
        long_cli_flag_was_set("reasoning-budget"),
    );

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
    env_fallback_prompt_cache_snapshot_capacity_bytes(
        &mut args.prompt_cache_snapshot_capacity_bytes,
    );
    env_fallback_prompt_cache_snapshot_max_entries(&mut args.prompt_cache_snapshot_max_entries);
    env_fallback_prompt_cache_snapshot_ttl(&mut args.prompt_cache_snapshot_ttl);

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
    mlxcel::cli::speculative_args::env_fallback_draft_max(
        &mut args.draft,
        mlxcel::server::long_cli_flag_was_set("draft")
            || mlxcel::server::long_cli_flag_was_set("draft-max")
            || mlxcel::server::long_cli_flag_was_set("spec-draft-n-max")
            || mlxcel::server::long_cli_flag_was_set("draft-n"),
    );
    env_fallback_draft_model(&mut args.model_draft);
    env_fallback_batch_size(&mut args.batch_size);
    env_fallback_ubatch_size(&mut args.ubatch_size);
    env_fallback_log_file(&mut args.log_file);
    env_fallback_endpoint_slots(
        &mut args.slots,
        long_cli_flag_was_set("slots"),
        long_cli_flag_was_set("no-slots"),
    );
    env_fallback_api_keys(&mut args.api_key);
    env_fallback_api_key_files(&mut args.api_key_file);
    env_fallback_cors_credentials(
        &mut args.cors_credentials,
        long_cli_flag_was_set("cors-credentials"),
        long_cli_flag_was_set("no-cors-credentials"),
    );

    // Axis B (B8): resolve once up-front so CLI errors surface before the
    // server starts listening. Baseline path returns `None` (bit-exact).
    let lang_bias_config = args
        .lang_bias
        .resolve()
        .map_err(|e| anyhow::anyhow!("--lang-bias: {e}"))?;

    // llama-server model-source translation (issue #1434): pick the reference
    // from `-m` / `--hf-repo`, and refuse `--docker-repo`, `--model-url` and
    // `--hf-file` here rather than accepting them and resolving something
    // else. `--offline` and `--hf-token` then shape how the reference is
    // resolved.
    // b10621 GGML runtime / placement / memory options (issue #1445): every
    // value is classified before the model reference is even resolved, so
    // `--numa distribute` or `--rpc host:1` is reported immediately rather
    // than after a multi-gigabyte download. Inert values are accepted
    // silently; a value whose upstream meaning mlxcel cannot reproduce stops
    // startup with a diagnostic naming the option, the value, the limitation,
    // and the mlxcel alternative.
    args.ggml_compat
        .apply_env_bindings()
        .map_err(|(var, raw)| anyhow::anyhow!("{var} has an invalid boolean value {raw:?}"))?;
    args.ggml_compat
        .ensure_inert_before_model()
        .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;

    // b10621 logging and preset options (issue #1448): `--log-prompts-dir` and
    // the twelve GGUF presets are refused here, before the model reference is
    // resolved, so a copied llama-server command line fails in under a second
    // with the mlxcel equivalent rather than after a multi-gigabyte download.
    args.logging_compat
        .ensure_supported()
        .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;
    let log_format = args
        .logging_compat
        .resolve_format(mlxcel::cli::logging_compat_args::verbosity_was_set_on_cli())
        .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;
    // Register credentials before the subscriber exists so no log sink can
    // ever hold an API key or a repository token (#1448).
    mlxcel::cli::logging_compat_args::register_credentials_for_redaction(
        &args.api_key,
        &args.api_key_file,
        args.hf_token.as_deref(),
    );
    // The subscriber itself is installed inside `start_server`, which is
    // reached only after the model reference resolves. Validate and create the
    // destination here instead, so an unwritable `--log-file` is reported
    // before a multi-gigabyte download rather than after it (#1448).
    mlxcel::server::logging::precheck_log_destination(args.log_disable, args.log_file.as_deref())?;

    // b10621 speculative compatibility options (#1433): classified before the
    // model load like the GGML group above. n-gram / lookup speculation, GGML
    // draft-side process placement, and draft-sampler thresholds mlxcel's
    // MTP / DFlash verification does not use are rejected with a diagnostic;
    // the inert forms (and the n-gram tuning knobs, inert while --spec-type
    // stays none) are accepted.
    args.spec_compat
        .apply_env_bindings()
        .map_err(|(var, raw)| anyhow::anyhow!("{var} has an invalid boolean value {raw:?}"))?;
    args.spec_compat
        .ensure_inert()
        .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;
    // --spec-type translation (#1433): `none` disables speculation exactly
    // as b10621 does (the explicit selector stops the draft-sidecar type
    // inference), and draft-mtp / draft-dflash translate onto --draft-kind,
    // erroring deterministically when they conflict with an explicit one.
    let spec_type = args
        .spec_compat
        .resolved_spec_type()
        .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;
    if spec_type.disable_speculation && args.model_draft.is_some() {
        tracing::warn!(
            "--spec-type none disables speculative decoding (b10621 semantics); ignoring the configured draft model"
        );
        args.model_draft = None;
    }
    if let Some(kind) = spec_type.draft_kind {
        match args.speculative.draft_kind.as_deref() {
            None => args.speculative.draft_kind = Some(kind.to_string()),
            Some(existing) if existing == kind => {}
            Some(existing) => {
                anyhow::bail!(
                    "--spec-type draft-{kind} conflicts with --draft-kind {existing}: pick one"
                );
            }
        }
    }

    // b10621 multimodal projector / media options (issue #1451): classified
    // before the model reference resolves, like the GGML and chat-template
    // groups, so `--mmproj projector.gguf` is reported immediately rather than
    // after a multi-gigabyte download. The requested image-token budget and the
    // `--media-path` root are installed process-wide here, before the first
    // load and before the server accepts a request that could name a local
    // file.
    args.multimodal_compat
        .apply_env_bindings()
        .map_err(|(var, raw)| anyhow::anyhow!("{var} has an invalid boolean value {raw:?}"))?;
    args.multimodal_compat
        .ensure_inert()
        .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;

    // b10621 Web UI / tools / MCP / CORS-proxy / agent surface (issue
    // #1435): the inert forms are accepted, every enabling form fails here,
    // before the model reference resolves.
    args.ui_compat
        .apply_env_bindings()
        .map_err(|(var, raw)| anyhow::anyhow!("{var} has an invalid boolean value {raw:?}"))?;
    args.ui_compat
        .ensure_inert()
        .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;
    let image_token_bounds = args
        .multimodal_compat
        .image_token_bounds()
        .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;
    mlxcel::vision::image_token_overrides::install(
        mlxcel::vision::image_token_overrides::ImageTokenOverride::from_bounds(
            image_token_bounds.min_tokens,
            image_token_bounds.max_tokens,
        ),
    )
    .map_err(|message| anyhow::anyhow!("{message}"))?;
    mlxcel::server::media_root::install_media_root(
        args.multimodal_compat
            .resolve_media_root()
            .map_err(|message| anyhow::anyhow!("{message}"))?,
    )
    .map_err(|message| anyhow::anyhow!("{message}"))?;
    mlxcel::server::configure_media_admission(
        args.multimodal_compat.media_admission().is_disabled(),
    );
    mlxcel::server::configure_private_media_urls(
        mlxcel::server::private_media_urls_allowed_from_env(),
    );

    // b10621 chat-template / reasoning / parsing options (issue #1447):
    // classified before the model reference resolves, like the GGML group, so
    // `--reasoning-format legacy` is reported immediately.
    args.chat_compat
        .apply_env_bindings()
        .map_err(|(var, raw)| anyhow::anyhow!("{var} has an invalid boolean value {raw:?}"))?;
    let chat_compat = args
        .chat_compat
        .resolve()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // b10621's `--chat-template` takes either template text or one of its
    // built-in identifiers; mlxcel has no built-in library, so a bare name
    // would become the template itself. Checked before the model resolves so
    // the mistake is reported immediately (issue #1447).
    if let Some(template) = args.chat_template.as_deref() {
        mlxcel::server::ensure_chat_template_is_not_a_builtin_name(template)?;
    }

    // The KV cache type is model-independent too, so validate it here rather
    // than leaving `--cache-type-k q8_0` to be reported after a multi-gigabyte
    // download (issue #1445). The resolved mode is recomputed later against the
    // loaded model's family, which can only substitute a supported mode for
    // another supported one; this pass exists to reject the value outright.
    resolve_kv_cache_mode(
        args.turbo.cache_type_k.as_deref(),
        args.turbo.cache_type_v.as_deref(),
        args.turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    env_fallback_offline(&mut args.offline);
    // One process-wide flag, as in b10621, so the fetch sites reached from
    // inside a loader (the moondream starmie tokenizer, request-path media
    // URLs) honour --offline too, not just the `-m` resolver.
    set_offline_mode(args.offline);
    // b10621 router mode (#1438): `--models-dir` with no model argument
    // serves the router surface and never resolves a checkpoint here.
    let router_mode = args.models_dir.is_some()
        && args.model.is_none()
        && args.hf_repo.is_none()
        && args.model_url.is_none()
        && args.docker_repo.is_none();
    // The #1438 migration guard, raised BEFORE model resolution so the old
    // store-root combination fails in milliseconds instead of after a
    // multi-gigabyte download. `into_startup_config` re-checks it for
    // programmatic callers.
    if args.models_dir.is_some() && !router_mode {
        anyhow::bail!(
            "--models-dir now selects llama-server b10621 router-mode model discovery and cannot \
             be combined with a model argument. To set the mlxcel model-store root (its old \
             meaning), use --model-store-root <PATH> or the MLXCEL_MODELS_DIR environment \
             variable; to start the router, drop the model argument"
        );
    }
    let model_path = if router_mode {
        PathBuf::new()
    } else {
        let source = resolve_llama_model_source(&LlamaModelSourceArgs {
            model: args.model.clone(),
            hf_repo: args.hf_repo.clone(),
            hf_file: args.hf_file.clone(),
            model_url: args.model_url.clone(),
            docker_repo: args.docker_repo.clone(),
        })?;
        if let Some(superseded) = source.superseded_model.as_deref() {
            tracing::info!("{}", superseded_model_notice(&source.reference, superseded));
        }
        resolve_model_source_with_options(
            &source.reference,
            ModelSourceOptions {
                models_dir: args.model_store_root.as_deref(),
                revision: args.revision.as_deref(),
                token: args.hf_token.as_deref(),
                offline: args.offline,
            },
        )
        // Name the flag the operator actually typed. Without this a failure to
        // resolve an `--hf-repo` value reports it as a `-m` problem.
        .with_context(|| format!("resolving {} {}", source.origin, source.reference.display()))?
    };

    // The `--gpu-layers` half of the b10621 GGML classification, deferred
    // until the checkpoint is known: only its layer count distinguishes a full
    // offload (inert, since mlxcel always runs every layer on the accelerator)
    // from a partial one (issue #1445).
    if !router_mode {
        args.ggml_compat
            .ensure_inert(read_model_layer_count(&model_path))
            .map_err(|rejection| anyhow::anyhow!("{rejection}"))?;
    }

    // `--embedding-model` accepts the same path-or-repo-id shapes as `-m`
    // and resolves through the same store lookup / auto-download.
    env_fallback_embedding_model(&mut args.embedding_model);
    let embedding_model_path = args
        .embedding_model
        .as_deref()
        .map(|value| {
            resolve_model_source_with_options(
                std::path::Path::new(value),
                ModelSourceOptions {
                    models_dir: args.model_store_root.as_deref(),
                    revision: args.revision.as_deref(),
                    token: args.hf_token.as_deref(),
                    offline: args.offline,
                },
            )
        })
        .transpose()?;

    // `--reranker-model` accepts the same path-or-repo-id shapes as `-m`.
    env_fallback_reranker_model(&mut args.reranker_model);
    let reranker_model_path = args
        .reranker_model
        .as_deref()
        .map(|value| {
            resolve_model_source_with_options(
                std::path::Path::new(value),
                ModelSourceOptions {
                    models_dir: args.model_store_root.as_deref(),
                    revision: args.revision.as_deref(),
                    token: args.hf_token.as_deref(),
                    offline: args.offline,
                },
            )
        })
        .transpose()?;

    Ok(ServerStartupInput {
        chat_compat,
        reasoning_alias_field: args.reasoning_alias_field,
        model_path,
        adapter_path: None,
        lora: args.lora.clone(),
        lora_scaled: args.lora_scaled.clone(),
        lora_init_without_apply: args.lora_init_without_apply,
        lora_fuse: args.lora_fuse,
        model_alias: args.alias,
        host: args.host,
        port: args.port,
        api_keys: args.api_key,
        api_key_files: args.api_key_file,
        n_parallel: mlxcel::server::resolve_n_parallel(args.parallel)
            .map_err(|message| anyhow::anyhow!("{message}"))?,
        ctx_size: args.ctx_size,
        n_predict: args.predict,
        // HTTP transport (#1432). `timeout` is now the socket read/write
        // budget; the decode watchdog moved to `--decode-timeout`. Both
        // "was set" flags include the environment binding, because a
        // deployment that only ever set `LLAMA_ARG_TIMEOUT` is exactly the one
        // whose meaning changed.
        timeout: args.timeout,
        timeout_was_set: long_cli_flag_was_set("timeout")
            || std::env::var_os("LLAMA_ARG_TIMEOUT").is_some(),
        decode_timeout: args.decode_timeout,
        sleep_idle_seconds: args.sleep_idle_seconds,
        decode_timeout_was_set: long_cli_flag_was_set("decode-timeout")
            || std::env::var_os("MLXCEL_DECODE_TIMEOUT").is_some(),
        api_prefix: args.api_prefix,
        sse_ping_interval: args.sse_ping_interval,
        threads_http: args.threads_http,
        reuse_port: args.reuse_port,
        ssl_cert_file: args.ssl_cert_file,
        ssl_key_file: args.ssl_key_file,
        cors_origins: args.cors_origins,
        cors_methods: args.cors_methods,
        cors_headers: args.cors_headers,
        cors_credentials: args.cors_credentials,
        no_cors_credentials: args._no_cors_credentials,
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
        embedding_model_path,
        embedding_batch_size: args.embedding_batch_size,
        embedding_max_length: args.embedding_max_length,
        embedding_queue_depth: args.embedding_queue_depth,
        embedding_request_timeout_secs: args.embedding_request_timeout_secs,
        reranker_model_path,
        rerank_batch_size: args.rerank_batch_size,
        prefill_chunk_size: args.prefill_chunk_size,
        prefill_grant_interval: args.prefill_grant_interval,
        batch_size: args.batch_size,
        ubatch_size: args.ubatch_size,
        enable_preemption: args.enable_preemption,
        preemption_policy: args.preemption_policy,
        max_batch_prefill: args.max_batch_prefill,
        max_batch_prefill_tokens: args.max_batch_prefill_tokens,
        decode_storage_backend: args.decode_storage_backend,
        chat_template: args.chat_template,
        chat_template_file: args.chat_template_file,
        slots: args.slots,
        no_slots: args._no_slots,
        props: args.props,
        metrics: args.metrics,
        settings: args.settings,
        slot_save_path: args.slot_save_path,
        router_models_dir: args.models_dir.clone(),
        models_max: args.models_max,
        models_autoload: args.models_autoload,
        models_preset: args.models_preset.clone(),
        tags: args.tags.clone(),
        model_store_root: args.model_store_root.clone(),
        warmup: args.warmup,
        no_warmup: args._no_warmup,
        temperature: args.temp,
        temperature_was_set: long_cli_flag_was_set("temp") || long_cli_flag_was_set("temperature"),
        top_k: args.top_k,
        top_k_was_set: long_cli_flag_was_set("top-k")
            || std::env::var_os("LLAMA_ARG_TOP_K").is_some(),
        top_p: args.top_p,
        top_p_was_set: long_cli_flag_was_set("top-p"),
        min_p: args.min_p,
        typical_p: args.typical_p,
        top_n_sigma: args.top_n_sigma,
        xtc_probability: args.xtc_probability,
        xtc_threshold: args.xtc_threshold,
        ignore_eos: args.ignore_eos,
        reverse_prompt: args.reverse_prompt.clone(),
        samplers: args.samplers.clone(),
        sampler_seq: args.sampler_seq.clone(),
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
        grammar_source: {
            // b10621 lets the last of --grammar / --grammar-file /
            // --json-schema / --json-schema-file on the command line win,
            // because all four write one field there (#1485).
            use clap::CommandFactory;
            let mut cmd = Cli::command();
            mlxcel::cli::grammar_args::resolve_grammar_source(
                &mut cmd,
                &std::env::args_os().collect::<Vec<_>>(),
                1,
                args.grammar,
                args.grammar_file,
                args.json_schema,
                args.json_schema_file,
            )
        },
        mirostat: args.mirostat,
        mirostat_tau: args.mirostat_tau,
        mirostat_eta: args.mirostat_eta,
        dynatemp_range: args.dynatemp_range,
        dynatemp_exponent: args.dynatemp_exponent,
        adaptive_target: args.adaptive_target,
        adaptive_decay: args.adaptive_decay,
        logit_bias: args.logit_bias,
        verbose: args.verbose,
        log_disable: args.log_disable,
        log_file: args.log_file,
        log_format,
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
        reasoning_budget: args.reasoning_budget,
        chat_template_kwargs: args.chat_template_kwargs,
        // prompt-cache knobs already resolved via env-var fallbacks above.
        // `--no-prompt-cache` is the highest-precedence opt-out: it wins over
        // both `--prompt-cache-enabled` and the env-var fallbacks.
        prompt_cache_enabled: args.prompt_cache_enabled && !args.no_prompt_cache,
        prompt_cache_capacity_bytes: args.prompt_cache_capacity_bytes,
        prompt_cache_max_entries: args.prompt_cache_max_entries,
        prompt_cache_ttl_seconds: args.prompt_cache_ttl,
        prompt_cache_min_prefix: args.prompt_cache_min_prefix,
        prompt_cache_snapshot_capacity_bytes: args.prompt_cache_snapshot_capacity_bytes,
        prompt_cache_snapshot_max_entries: args.prompt_cache_snapshot_max_entries,
        prompt_cache_snapshot_ttl_seconds: args.prompt_cache_snapshot_ttl,
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
        // Responses API in-memory store limits. clap reads the matching env
        // vars directly via the `env = ...` attributes on the flags.
        responses_store_max_entries: args.responses_store_max_entries,
        responses_store_max_bytes: args.responses_store_max_bytes,
        responses_store_ttl_secs: args.responses_store_ttl_secs,
        conversation_store_max_entries: args.conversation_store_max_entries,
        conversation_store_max_bytes: args.conversation_store_max_bytes,
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
        rope: args.rope.clone(),
        cache_compat: args.cache_compat.clone(),
        context_compat: args.context_compat.clone(),
        slot_compat: args.slot_compat.clone(),
        infill: args.infill.clone(),
        embedding_compat: args.embedding_compat.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    static CLI_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    struct ScopedEnv(Vec<(&'static str, Option<OsString>)>);

    impl ScopedEnv {
        fn set(values: &[(&'static str, &str)]) -> Self {
            let saved = values
                .iter()
                .map(|(key, _)| (*key, std::env::var_os(key)))
                .collect();
            for (key, value) in values {
                // SAFETY: every env-mutating test in this binary holds CLI_ENV_LOCK.
                unsafe { std::env::set_var(key, value) };
            }
            Self(saved)
        }

        fn unset(keys: &[&'static str]) -> Self {
            let saved = keys
                .iter()
                .map(|key| (*key, std::env::var_os(key)))
                .collect();
            for key in keys {
                // SAFETY: every env-mutating test in this binary holds CLI_ENV_LOCK.
                unsafe { std::env::remove_var(key) };
            }
            Self(saved)
        }
    }

    impl Drop for ScopedEnv {
        fn drop(&mut self) {
            for (key, value) in self.0.drain(..) {
                // SAFETY: the guard is dropped before the outer CLI_ENV_LOCK guard.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

    #[test]
    fn help_explains_embedding_and_reranking_server_modes() {
        use clap::CommandFactory;

        let mut command = Cli::command();
        let help = command.render_long_help().to_string();
        for expected in [
            "POST /v1/embeddings",
            "--embedding-model",
            "POST /v1/rerank",
            "--reranker-model",
            "--embedding-* worker",
            "docs/embeddings.md",
            "docs/supported-models.md",
            "docs/distributed.md",
        ] {
            assert!(
                help.contains(expected),
                "mlxcel-server --help is missing {expected:?}:\n{help}"
            );
        }
        for model_specific in ["Current multi-rank support", "Gemma 4 E2B-style"] {
            assert!(
                !help.contains(model_specific),
                "mlxcel-server --help must leave model-specific guidance in the model catalog; found {model_specific:?} in:\n{help}"
            );
        }
    }

    fn parse_server_args(argv: &[&str]) -> ServerArgs {
        let cli = Cli::try_parse_from(argv).expect("mlxcel-server args should parse");
        assert!(
            cli.command.is_none(),
            "test argv should exercise legacy server-start mode"
        );
        cli.server
    }

    #[test]
    fn settings_cli_mlxcel_server_help_exposes_flag() {
        use clap::CommandFactory;

        let mut command = Cli::command();
        let help = command.render_long_help().to_string();
        assert!(
            help.contains("--settings"),
            "`mlxcel-server --help` is missing --settings:\n{help}"
        );
    }

    #[test]
    fn settings_cli_mlxcel_server_defaults_off_and_propagates_explicit_flag() {
        let _lock = CLI_ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("CLI env lock");
        let _settings_env = ScopedEnv::unset(&["MLXCEL_ENABLE_SETTINGS_ENDPOINT"]);

        let models_dir = tempfile::tempdir().expect("models directory");
        let models_dir = models_dir.path().to_string_lossy();
        let default = parse_server_args(&["mlxcel-server", "--models-dir", &models_dir]);
        assert!(!default.settings, "the settings endpoint must default off");
        let default = build_startup_input(default).expect("default startup input");
        assert!(!default.settings, "the default must remain off downstream");

        let enabled =
            parse_server_args(&["mlxcel-server", "--models-dir", &models_dir, "--settings"]);
        assert!(enabled.settings, "--settings must propagate through clap");
        let enabled = build_startup_input(enabled).expect("enabled startup input");
        assert!(enabled.settings, "--settings must reach startup input");
    }

    #[test]
    fn store_byte_budget_flags_parse() {
        let args = parse_server_args(&[
            "mlxcel-server",
            "-m",
            "models/foo",
            "--responses-store-max-bytes",
            "12345",
            "--conversation-store-max-bytes",
            "67890",
        ]);

        assert_eq!(args.responses_store_max_bytes, 12_345);
        assert_eq!(args.conversation_store_max_bytes, 67_890);
    }

    #[test]
    fn store_byte_budget_envs_parse_when_flags_are_absent() {
        let _lock = CLI_ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env = ScopedEnv::set(&[
            ("MLXCEL_RESPONSES_STORE_MAX_BYTES", "11111"),
            ("MLXCEL_CONVERSATION_STORE_MAX_BYTES", "22222"),
        ]);

        let args = parse_server_args(&["mlxcel-server", "-m", "models/foo"]);

        assert_eq!(args.responses_store_max_bytes, 11_111);
        assert_eq!(args.conversation_store_max_bytes, 22_222);
    }

    #[test]
    fn reasoning_alias_field_defaults_parses_and_rejects_unknown_values() {
        assert_eq!(
            parse_server_args(&["mlxcel-server", "-m", "models/foo"]).reasoning_alias_field,
            mlxcel::server::ReasoningAliasField::Reasoning
        );
        for (value, expected) in [
            ("reasoning", mlxcel::server::ReasoningAliasField::Reasoning),
            ("none", mlxcel::server::ReasoningAliasField::None),
        ] {
            assert_eq!(
                parse_server_args(&[
                    "mlxcel-server",
                    "-m",
                    "models/foo",
                    "--reasoning-alias-field",
                    value,
                ])
                .reasoning_alias_field,
                expected,
                "{value}"
            );
        }

        let error = Cli::try_parse_from([
            "mlxcel-server",
            "-m",
            "models/foo",
            "--reasoning-alias-field",
            "other",
        ])
        .expect_err("unknown reasoning alias field must be rejected");
        let text = error.to_string();
        assert!(text.contains("--reasoning-alias-field"), "{text}");
        assert!(text.contains("none, reasoning"), "{text}");
    }

    // ── llama-server model-source flags (issue #1434) ───────────────

    #[test]
    fn warmup_defaults_to_enabled_and_no_warmup_disables_it() {
        // b10621: "whether to perform warmup with an empty run (default:
        // enabled)". Both spellings must parse and `--no-warmup` must win,
        // because it is the only thing that makes `--no-warmup` more than an
        // accepted-and-ignored flag.
        let default = parse_server_args(&["mlxcel-server", "-m", "models/foo"]);
        assert!(default.warmup, "warmup defaults to enabled");
        assert!(!default._no_warmup);

        let disabled = parse_server_args(&["mlxcel-server", "-m", "models/foo", "--no-warmup"]);
        assert!(
            !mlxcel::server::resolve_compat_toggle(disabled.warmup, disabled._no_warmup),
            "--no-warmup must disable the warmup pass"
        );

        let re_enabled = parse_server_args(&[
            "mlxcel-server",
            "-m",
            "models/foo",
            "--no-warmup",
            "--warmup",
        ]);
        assert!(
            mlxcel::server::resolve_compat_toggle(re_enabled.warmup, re_enabled._no_warmup),
            "a later --warmup must override an earlier --no-warmup"
        );
    }

    #[test]
    fn model_source_compatibility_flags_parse_into_their_own_fields() {
        let args = parse_server_args(&[
            "mlxcel-server",
            "--hf-repo",
            "mlx-community/Qwen3-4B-4bit",
            "--hf-token",
            "hf_example",
            "--offline",
            "--hf-file",
            "model.gguf",
            "--model-url",
            "https://example.com/model.gguf",
            "--docker-repo",
            "ai/gemma3",
        ]);
        assert_eq!(args.hf_repo.as_deref(), Some("mlx-community/Qwen3-4B-4bit"));
        assert_eq!(args.hf_token.as_deref(), Some("hf_example"));
        assert!(args.offline);
        assert_eq!(args.hf_file.as_deref(), Some("model.gguf"));
        assert_eq!(
            args.model_url.as_deref(),
            Some("https://example.com/model.gguf")
        );
        assert_eq!(args.docker_repo.as_deref(), Some("ai/gemma3"));
        // `-m` is no longer required when another source is supplied.
        assert_eq!(args.model, None);
    }

    #[test]
    fn temp_and_temperature_aliases_resolve_identically() {
        let primary = parse_server_args(&["mlxcel-server", "-m", "models/foo", "--temp", "0.37"]);
        let alias =
            parse_server_args(&["mlxcel-server", "-m", "models/foo", "--temperature", "0.37"]);
        assert_eq!(primary.temp, alias.temp);
    }

    #[test]
    fn canonical_llama_envs_and_endpoint_precedence_parse_together() {
        let _lock = CLI_ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // `LLAMA_ARG_LOG_FILE` is set process-wide for the length of this
        // test, and `--log-file` is bound to it through clap on both binaries.
        // Any other test in this binary that builds a logging configuration
        // while the guard is live therefore opens the path, and a relative one
        // lands in the repository root: running the suite used to leave a
        // zero-byte `canonical.log` behind in the working tree. An absolute
        // temporary path keeps the parsing assertion intact and the artifact
        // out of the tree.
        let log_file = std::env::temp_dir().join("mlxcel-canonical-llama-env.log");
        let log_file_arg = log_file.to_string_lossy().into_owned();
        let _env = ScopedEnv::set(&[
            ("LLAMA_ARG_BATCH", "111"),
            ("LLAMA_ARG_BATCH_SIZE", "999"),
            ("LLAMA_ARG_UBATCH", "22"),
            ("LLAMA_ARG_UBATCH_SIZE", "999"),
            ("LLAMA_ARG_SPEC_DRAFT_MODEL", "models/canonical-draft"),
            ("LLAMA_ARG_MODEL_DRAFT", "models/legacy-draft"),
            ("LLAMA_ARG_LOG_FILE", log_file_arg.as_str()),
            ("LLAMA_LOG_FILE", "legacy.log"),
            ("LLAMA_ARG_CHAT_TEMPLATE", "{{ messages }}"),
            ("LLAMA_ARG_CHAT_TEMPLATE_FILE", "canonical.jinja"),
            ("LLAMA_ARG_ENDPOINT_METRICS", "true"),
            ("LLAMA_ARG_ENDPOINT_PROPS", "true"),
            ("LLAMA_ARG_ENDPOINT_SLOTS", "false"),
        ]);

        let mut args = parse_server_args(&["mlxcel-server", "-m", "models/foo"]);
        env_fallback_batch_size(&mut args.batch_size);
        env_fallback_ubatch_size(&mut args.ubatch_size);
        env_fallback_draft_model(&mut args.model_draft);
        env_fallback_log_file(&mut args.log_file);
        env_fallback_endpoint_slots(&mut args.slots, false, false);
        assert_eq!(args.batch_size, Some(111));
        assert_eq!(args.ubatch_size, Some(22));
        assert_eq!(
            args.model_draft.as_deref(),
            Some(Path::new("models/canonical-draft"))
        );
        assert_eq!(args.log_file.as_deref(), Some(log_file.as_path()));
        assert_eq!(args.chat_template.as_deref(), Some("{{ messages }}"));
        assert_eq!(
            args.chat_template_file.as_deref(),
            Some(Path::new("canonical.jinja"))
        );
        assert!(args.metrics);
        assert!(args.props);
        assert!(!args.slots);

        let mut cli_args = parse_server_args(&[
            "mlxcel-server",
            "-m",
            "models/foo",
            "--batch-size",
            "333",
            "--slots",
        ]);
        env_fallback_endpoint_slots(&mut cli_args.slots, true, false);
        assert_eq!(cli_args.batch_size, Some(333));
        assert!(cli_args.slots);

        let mut no_slots_args =
            parse_server_args(&["mlxcel-server", "-m", "models/foo", "--no-slots"]);
        env_fallback_endpoint_slots(&mut no_slots_args.slots, false, true);
        assert!(no_slots_args._no_slots);
        assert!(!(no_slots_args.slots && !no_slots_args._no_slots));
    }

    #[test]
    fn slots_flags_use_last_occurrence() {
        let disabled =
            parse_server_args(&["mlxcel-server", "-m", "models/foo", "--slots", "--no-slots"]);
        assert!(!(disabled.slots && !disabled._no_slots));

        let enabled =
            parse_server_args(&["mlxcel-server", "-m", "models/foo", "--no-slots", "--slots"]);
        assert!(enabled.slots && !enabled._no_slots);
    }

    fn make_complete_snapshot(models_root: &Path, repo_id: &str) -> PathBuf {
        let mut dir = models_root.to_path_buf();
        for segment in repo_id.split('/') {
            dir.push(segment);
        }
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), b"{}").unwrap();
        // The resolver's offline completeness gate (downloader/completeness.rs,
        // issue #465) also requires at least one non-zero weight file; a
        // config-only directory is classified as an interrupted download and
        // triggers a network re-fetch.
        fs::write(dir.join("model.safetensors"), b"stub-weights").unwrap();
        dir
    }

    #[test]
    fn legacy_server_mode_resolves_repo_id_from_models_dir_override() {
        let tmp = tempfile::tempdir().unwrap();
        let models_root = tmp.path().join("custom-model-store");
        let repo_id = "zz-mlxcel-test-owner/zz-mlxcel-test-model";
        let expected = make_complete_snapshot(&models_root, repo_id);
        let models_root_arg = models_root.to_string_lossy().to_string();

        // #1438: the store-root override moved to --model-store-root;
        // --models-dir now selects router mode.
        let args = parse_server_args(&[
            "mlxcel-server",
            "-m",
            repo_id,
            "--model-store-root",
            &models_root_arg,
        ]);
        let input = build_startup_input(args).expect("repo-id should resolve from override store");

        assert_eq!(input.model_path, expected);
    }

    #[test]
    fn models_dir_with_a_model_argument_fails_with_the_migration_diagnostic() {
        // #1438: the old `--models-dir <store>` + `-m` combination must fail
        // loudly with the replacement named, never silently pick a meaning.
        let tmp = tempfile::tempdir().unwrap();
        let dir_arg = tmp.path().to_string_lossy().to_string();
        let args = parse_server_args(&[
            "mlxcel-server",
            "-m",
            "models/foo",
            "--models-dir",
            &dir_arg,
        ]);
        let err = build_startup_input(args)
            .expect_err("the migration guard must refuse the combination before resolution");
        assert!(err.to_string().contains("--model-store-root"), "{err}");
    }

    #[test]
    fn serving_throughput_defaults_are_on_out_of_the_box() {
        // #628: shipped defaults enable multi-client batching machinery.
        let args = parse_server_args(&["mlxcel-server", "-m", "models/foo"]);
        assert_eq!(
            args.parallel, -1,
            "b10621's -1 (auto) is the shipped default (#1472)"
        );
        assert_eq!(
            mlxcel::server::resolve_n_parallel(args.parallel).expect("auto resolves"),
            4,
            "auto resolves to 4 slots, matching upstream's auto and the #628 default"
        );
        assert_eq!(
            args.max_batch_prefill, 4,
            "batched prefill should default to 4"
        );
        assert!(
            args.prompt_cache_enabled,
            "prompt cache should be on by default"
        );
        assert!(
            !args.no_prompt_cache,
            "no-prompt-cache opt-out defaults off"
        );
    }

    #[test]
    fn parallel_one_escape_hatch_still_parses() {
        // #628: `--parallel 1` remains a full single-slot escape hatch.
        let args = parse_server_args(&["mlxcel-server", "-m", "models/foo", "--parallel", "1"]);
        assert_eq!(args.parallel, 1);
    }

    #[test]
    fn no_prompt_cache_flag_disables_prompt_cache() {
        // #628: `--no-prompt-cache` overrides the default-on prompt cache.
        let tmp = tempfile::tempdir().unwrap();
        let local_model = tmp.path().join("local-model");
        fs::create_dir_all(&local_model).unwrap();
        let local_model_arg = local_model.to_string_lossy().to_string();

        let on = build_startup_input(parse_server_args(&[
            "mlxcel-server",
            "-m",
            &local_model_arg,
        ]))
        .expect("default args should build");
        assert!(on.prompt_cache_enabled, "prompt cache on by default");

        let off = build_startup_input(parse_server_args(&[
            "mlxcel-server",
            "-m",
            &local_model_arg,
            "--no-prompt-cache",
        ]))
        .expect("--no-prompt-cache args should build");
        assert!(
            !off.prompt_cache_enabled,
            "--no-prompt-cache must disable the prompt cache"
        );
    }

    #[test]
    fn kv_cache_budget_defaults_to_auto() {
        // #628: the batched-decode default pairs with an `auto` paged KV budget
        // guard so admission sheds load instead of OOMing.
        use mlxcel::memory_estimate::PagedBudgetDirective;
        let args = parse_server_args(&["mlxcel-server", "-m", "models/foo"]);
        assert_eq!(args.kv_cache_budget, Some(PagedBudgetDirective::Auto));
    }

    #[test]
    fn kv_cache_budget_explicit_disable_and_bytes_parse() {
        // #628: escape hatches. `none` and `0` disable the guard (unbounded);
        // an explicit byte count sets a hard cap.
        use mlxcel::memory_estimate::PagedBudgetDirective;
        let none = parse_server_args(&[
            "mlxcel-server",
            "-m",
            "models/foo",
            "--kv-cache-budget",
            "none",
        ]);
        assert_eq!(none.kv_cache_budget, Some(PagedBudgetDirective::Disabled));

        let zero = parse_server_args(&[
            "mlxcel-server",
            "-m",
            "models/foo",
            "--kv-cache-budget",
            "0",
        ]);
        assert_eq!(zero.kv_cache_budget, Some(PagedBudgetDirective::Disabled));

        let bytes = parse_server_args(&[
            "mlxcel-server",
            "-m",
            "models/foo",
            "--kv-cache-budget",
            "8589934592",
        ]);
        assert_eq!(
            bytes.kv_cache_budget,
            Some(PagedBudgetDirective::Bytes(8_589_934_592))
        );
    }

    #[test]
    fn legacy_server_mode_keeps_existing_model_path_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        let local_model = tmp.path().join("local-model");
        fs::create_dir_all(&local_model).unwrap();
        let decoy_models_root = tmp.path().join("decoy-store");
        let local_model_arg = local_model.to_string_lossy().to_string();
        let decoy_arg = decoy_models_root.to_string_lossy().to_string();

        // #1438: --models-dir is router mode now; the store-root decoy uses
        // the replacement spelling and must still not affect a local path.
        let args = parse_server_args(&[
            "mlxcel-server",
            "-m",
            &local_model_arg,
            "--model-store-root",
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
    fn spec_draft_spellings_resolve_to_the_draft_controls() {
        // b10621's canonical spellings (#1433): --spec-draft-model is the
        // real model flag and --spec-draft-n-max the real token cap; the
        // removed --draft / --draft-n / --draft-max spellings stay accepted
        // as aliases of the same field (b10621 itself errors on them).
        let canonical = parse_server_args(&["mlxcel-server", "--spec-draft-model", "models/draft"]);
        assert_eq!(canonical.model_draft, Some(PathBuf::from("models/draft")));
        for spelling in ["--spec-draft-n-max", "--draft-n"] {
            let parsed = parse_server_args(&["mlxcel-server", spelling, "24"]);
            assert_eq!(parsed.draft, 24, "{spelling} must set the draft-token cap");
        }
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

    // ── Server flag spelling parity (issue #1109) ───────────────
    //
    // Four flag spellings had drifted between this binary and `mlxcel
    // serve`. `--parallel` / `--n-parallel` was the worst case: neither
    // spelling worked on both binaries, so a command line copied between
    // them failed to parse even though both flags read the same
    // `LLAMA_ARG_N_PARALLEL` env var. These tests pin that each spelling
    // now resolves to the identical `ServerArgs` field value here;
    // `src/main_tests.rs` carries the matching assertions for `mlxcel
    // serve`, and `tests/cli_help_consistency.rs` asserts the two
    // binaries accept the same set of spellings so a fifth divergence
    // cannot land silently.

    /// Every long name, alias, and short form on `mlxcel-server` must be
    /// distinct. clap detects a duplicate itself, but only behind a
    /// `debug_assert` in `Command::_build_self`, and `[profile.test-fast]`
    /// inherits `release`, so `debug-assertions` is off in the profile this
    /// repository verifies with. A `visible_alias` that collided with an
    /// existing flag would be silently last-wins rather than a panic, and
    /// issue #1109 added two of them here.
    #[test]
    fn server_flag_names_and_aliases_are_unique() {
        use clap::CommandFactory;

        let command = Cli::command();

        let mut longs: Vec<String> = Vec::new();
        let mut shorts: Vec<char> = Vec::new();
        for arg in command.get_arguments() {
            if let Some(long) = arg.get_long() {
                longs.push(long.to_string());
            }
            if let Some(aliases) = arg.get_all_aliases() {
                longs.extend(aliases.into_iter().map(str::to_string));
            }
            if let Some(short) = arg.get_short() {
                shorts.push(short);
            }
            if let Some(aliases) = arg.get_all_short_aliases() {
                shorts.extend(aliases);
            }
        }

        let duplicate_longs = duplicates(&longs);
        assert!(
            duplicate_longs.is_empty(),
            "`mlxcel-server` declares these long names more than once: \
             {duplicate_longs:?}. clap resolves a duplicate silently in this profile, so \
             the second definition would shadow the first with no error."
        );

        let duplicate_shorts = duplicates(&shorts);
        assert!(
            duplicate_shorts.is_empty(),
            "`mlxcel-server` declares these short forms more than once: {duplicate_shorts:?}"
        );
    }

    /// Values appearing more than once in `items`, sorted and deduplicated.
    fn duplicates<T: Clone + Ord>(items: &[T]) -> Vec<T> {
        let mut sorted = items.to_vec();
        sorted.sort();
        let mut repeated: Vec<T> = sorted
            .windows(2)
            .filter(|w| w[0] == w[1])
            .map(|w| w[0].clone())
            .collect();
        repeated.dedup();
        repeated
    }

    #[test]
    fn parallel_and_n_parallel_aliases_resolve_identically() {
        let primary = parse_server_args(&["mlxcel-server", "--parallel", "2"]);
        let aliased = parse_server_args(&["mlxcel-server", "--n-parallel", "2"]);

        assert_eq!(
            primary.parallel, aliased.parallel,
            "--parallel and its --n-parallel alias must resolve to the same slot count"
        );
        assert_eq!(primary.parallel, 2);
    }

    #[test]
    fn lora_and_adapter_aliases_resolve_identically() {
        let primary = parse_server_args(&["mlxcel-server", "--lora", "adapters/foo"]);
        let aliased = parse_server_args(&["mlxcel-server", "--adapter", "adapters/foo"]);

        assert_eq!(
            primary.lora, aliased.lora,
            "--lora and its --adapter alias must resolve to the same adapter path"
        );
        assert_eq!(primary.lora.as_deref(), Some("adapters/foo"));
    }

    #[test]
    fn dry_sequence_breaker_singular_and_plural_aliases_resolve_identically() {
        let primary = parse_server_args(&["mlxcel-server", "--dry-sequence-breaker", "a,b"]);
        let aliased = parse_server_args(&["mlxcel-server", "--dry-sequence-breakers", "a,b"]);

        assert_eq!(
            primary.dry_sequence_breakers, aliased.dry_sequence_breakers,
            "both DRY breaker spellings must resolve to the same breaker list"
        );
        assert_eq!(
            primary.dry_sequence_breakers,
            vec!["a".to_string(), "b".to_string()]
        );
    }
    // ── Space-separated negative values (issue #1459) ───────────────────
    //
    // b10621 accepts `llama-server --seed -1`, and mlxcel rejected the space
    // form on all 122 of its value-taking long options with "unexpected
    // argument '-1' found" until `allow_negative_numbers` landed on the root
    // command. These tests pin the fix on the command the shipped binary
    // actually parses with, and pin the two properties that separate
    // `allow_negative_numbers` from `allow_hyphen_values`: a mistyped flag
    // after an option is still reported rather than swallowed as its value,
    // and an option whose domain excludes negatives still fails in its value
    // parser instead of silently accepting `-1`.

    /// Long options that decline a space-separated value of ANY sign because
    /// they declare `require_equals`, so `--opt -1` failing on them is not a
    /// negative-number defect. Both are mlxcel-only cache knobs with no
    /// b10621 counterpart, so no llama-server command line reaches them.
    const REQUIRE_EQUALS_LONGS: [&str; 2] = ["--apc-enabled", "--prompt-cache-enabled"];

    /// True when clap rejects `argv` with the `unexpected argument` error that
    /// every value-taking option produced for a space-separated `-1` before
    /// #1459.
    fn rejected_as_unexpected_argument(argv: &[&str]) -> bool {
        match Cli::try_parse_from(argv) {
            Ok(_) => false,
            Err(err) => err.kind() == clap::error::ErrorKind::UnknownArgument,
        }
    }

    #[test]
    fn every_value_taking_long_option_accepts_a_space_separated_negative_number() {
        use clap::CommandFactory;

        let mut command = Cli::command();
        command.build();
        let options: Vec<(String, bool)> = command
            .get_arguments()
            .filter(|arg| arg.get_num_args().is_some_and(|r| r.takes_values()))
            .filter_map(|arg| {
                arg.get_long()
                    .map(|long| (format!("--{long}"), arg.is_require_equals_set()))
            })
            .collect();

        let mut swept = 0usize;
        let mut exempt: Vec<String> = Vec::new();
        let mut rejected: Vec<String> = Vec::new();
        for (long, require_equals) in &options {
            if *require_equals {
                // Prove the exemption rather than assume it: an option that
                // requires `=` must decline a space-separated positive value
                // too, otherwise it is a real negative-number rejection
                // hiding behind this branch.
                assert!(
                    Cli::try_parse_from(["mlxcel-server", long, "1"]).is_err(),
                    "{long} declares require_equals but bound a space-separated positive \
                     value; the #1459 sweep exemption no longer describes it"
                );
                exempt.push(long.clone());
                continue;
            }
            swept += 1;
            if rejected_as_unexpected_argument(&["mlxcel-server", long, "-1"]) {
                rejected.push(long.clone());
            }
        }

        assert!(
            rejected.is_empty(),
            "these mlxcel-server options still reject a space-separated negative value that \
             b10621 accepts: {rejected:?}"
        );
        exempt.sort();
        assert_eq!(
            exempt, REQUIRE_EQUALS_LONGS,
            "the set of options that take no space-separated value has changed. If a b10621 \
             option gained require_equals, that is a compatibility regression; if a new \
             mlxcel-only knob gained it, extend REQUIRE_EQUALS_LONGS deliberately"
        );
        // 122 value-taking long options at the time of the fix, minus the two
        // require_equals knobs. The floor guards against a sweep that silently
        // stops enumerating.
        assert!(
            swept >= 100,
            "only {swept} value-taking long options were swept; the enumeration has collapsed"
        );
    }

    #[test]
    fn negative_seed_predict_and_reasoning_budget_parse_in_space_form() {
        for argv in [
            &["mlxcel-server", "--seed", "-1"][..],
            &["mlxcel-server", "-s", "-1"][..],
        ] {
            assert_eq!(
                parse_server_args(argv).seed,
                -1,
                "{argv:?} must bind -1 to the seed"
            );
        }
        for argv in [
            &["mlxcel-server", "--predict", "-1"][..],
            &["mlxcel-server", "-n", "-1"][..],
            &["mlxcel-server", "--n-predict", "-1"][..],
        ] {
            assert_eq!(
                parse_server_args(argv).predict,
                -1,
                "{argv:?} must bind -1 to the predict budget"
            );
        }
        assert_eq!(
            parse_server_args(&["mlxcel-server", "--reasoning-budget", "-1"]).reasoning_budget,
            -1
        );

        // -1 is also the default for all three, so a distinct negative is what
        // proves the token was consumed as the value rather than dropped.
        assert_eq!(
            parse_server_args(&["mlxcel-server", "--seed", "-7"]).seed,
            -7
        );
        assert_eq!(
            parse_server_args(&["mlxcel-server", "--predict", "-7"]).predict,
            -7
        );
        assert_eq!(
            parse_server_args(&["mlxcel-server", "--reasoning-budget", "-7"]).reasoning_budget,
            -7
        );
    }

    #[test]
    fn a_negative_float_parses_in_space_form() {
        let args = parse_server_args(&["mlxcel-server", "--presence-penalty", "-1.5"]);
        assert!(
            (args.presence_penalty - (-1.5)).abs() < f32::EPSILON,
            "--presence-penalty -1.5 must bind -1.5, got {}",
            args.presence_penalty
        );
    }

    #[test]
    fn equals_form_negative_values_still_parse() {
        assert_eq!(parse_server_args(&["mlxcel-server", "--seed=-7"]).seed, -7);
        assert_eq!(
            parse_server_args(&["mlxcel-server", "--predict=-7"]).predict,
            -7
        );
        assert_eq!(
            parse_server_args(&["mlxcel-server", "--n-predict=-7"]).predict,
            -7
        );
        assert_eq!(
            parse_server_args(&["mlxcel-server", "--reasoning-budget=-7"]).reasoning_budget,
            -7
        );
    }

    #[test]
    fn a_negative_value_for_a_non_negative_option_fails_in_the_value_parser() {
        for (long, argv) in [
            ("--port", &["mlxcel-server", "--port", "-1"][..]),
            ("--ctx-size", &["mlxcel-server", "--ctx-size", "-1"][..]),
            // Custom `parse_unit_interval` parser rather than a clap numeric
            // range, so it exercises the other rejection path.
            (
                "--diffusion-threshold",
                &["mlxcel-server", "--diffusion-threshold", "-1"][..],
            ),
        ] {
            let err = Cli::try_parse_from(argv)
                .expect_err("a negative value outside the option's domain must be rejected");
            assert_eq!(
                err.kind(),
                clap::error::ErrorKind::ValueValidation,
                "{long} must fail in its value parser, not as an unknown argument"
            );
            let text = err.to_string();
            assert!(
                text.contains(long) && text.contains("-1"),
                "the {long} rejection must name the option and the offending value, got: {text}"
            );
            assert!(
                !text.contains("unexpected argument"),
                "{long} must no longer report the value as an unexpected argument, got: {text}"
            );
        }
    }

    #[test]
    fn a_negative_parallel_resolves_to_the_automatic_slot_count() {
        // Since #1472 `--parallel` carries b10621's whole value domain: `-1`
        // (the default) is auto and resolves to 4 slots, matching upstream's
        // own auto; other negatives and zero are refused at resolution with a
        // message naming the option, the value, and the auto form.
        let args = parse_server_args(&["mlxcel-server", "--parallel", "-1"]);
        assert_eq!(args.parallel, -1);
        assert_eq!(
            mlxcel::server::resolve_n_parallel(args.parallel).expect("auto resolves"),
            4
        );

        for bad in [0, -2] {
            let err = mlxcel::server::resolve_n_parallel(bad)
                .expect_err("only -1 and positive counts are in domain");
            assert!(
                err.contains("--parallel") && err.contains(&bad.to_string()),
                "the rejection must name the option and the value, got: {err}"
            );
        }
    }

    #[test]
    fn a_negative_number_with_no_option_awaiting_a_value_is_still_an_error() {
        for argv in [
            &["mlxcel-server", "-1"][..],
            // The seed consumes the first negative; the second has nothing
            // pending and must not be silently absorbed.
            &["mlxcel-server", "--seed", "-1", "-2"][..],
        ] {
            let err = Cli::try_parse_from(argv)
                .expect_err("a stray negative number must not be silently consumed");
            assert_eq!(
                err.kind(),
                clap::error::ErrorKind::UnknownArgument,
                "{argv:?} must report the stray negative number"
            );
        }
    }

    #[test]
    fn a_mistyped_flag_after_an_option_is_reported_not_swallowed() {
        // This is the whole reason the fix is `allow_negative_numbers` and not
        // `allow_hyphen_values`: the latter takes any `-`-leading token as the
        // pending option's value, so this typo would bind `--moldel` to the
        // seed and never be reported.
        let err = Cli::try_parse_from(["mlxcel-server", "--seed", "--moldel", "foo"])
            .expect_err("a mistyped flag must not be swallowed as the seed value");
        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
        assert!(
            err.to_string().contains("--moldel"),
            "the diagnostic must name the mistyped flag, got: {err}"
        );
    }

    #[test]
    fn the_download_subcommand_does_not_take_negative_numbers() {
        // `allow_negative_numbers` is a per-command setting that does not
        // propagate into subcommands, and `download` deliberately keeps the
        // default: it has one positional repo-id and no numeric option, so a
        // leading `-1` there is a typo rather than a value (#1459).
        let err = Cli::try_parse_from(["mlxcel-server", "download", "-1"])
            .expect_err("`download -1` has no numeric option to bind to");
        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    }
}
