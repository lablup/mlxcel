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

use clap::{Args, Parser, Subcommand};
use mlxcel::tokenizer::MlxcelTokenizer;
use std::path::PathBuf;

mod commands;
use mlxcel::cli::batch_quant_args::BatchKvQuantArgs;
use mlxcel::cli::speculative_args::SpeculativeArgs;
use mlxcel::cli::turbo_args::TurboKvCacheArgs;
use mlxcel::downloader::DownloadArgs;
use mlxcel::lang_bias::LangBiasCliArgs;

/// mlxcel: High-performance LLM/VLM/VLA inference on Apple Silicon and CUDA GPUs
///
/// A Rust implementation for running Large Language Models, Vision-Language
/// Models, and Vision-Language-Action Models efficiently on Apple Silicon and
/// CUDA GPUs using the MLX framework.
#[derive(Parser, Debug)]
#[command(
    name = "mlxcel",
    author = "Lablup Inc.",
    version,
    about = "High-performance LLM/VLM/VLA inference on Apple Silicon and CUDA GPUs",
    long_about = None,
    after_help = "\
Environment Variables:
  MLXCEL_DEVICE          Runtime device: \"gpu\" (default), \"cpu\"
  MLXCEL_WIRED_LIMIT     Apple Silicon wired memory limit
                           unset/\"max\" — use MLX gpu_max_memory_size (default)
                           \"0\"/\"none\" — disable the wired limit
                           \"96GB\" — explicit limit (supports GB, MB, or bytes)
  MLXCEL_MEMORY_LIMIT    Soft MLX allocator memory cap (fails fast on overflow)
                           unset/\"0\"/\"none\" — let MLX use its backend default (default)
                           \"32GB\" — explicit limit (supports GB, MB, or bytes)

Tensor Parallel Runtime:
  Current multi-rank support: dense Llama, Qwen2/2.5, Qwen3, Qwen3.5 text, Gemma 3 text, Gemma 4 text, ERNIE 4.5, Hunyuan v1 Dense
  Current constraints: --tp-embedding-mode replicated, --tp-lm-head-mode replicated
                       LoRA unsupported, server batching supported for listed dense runtimes
                       except Gemma 4 E2B-style conservative fallback checkpoints

For more information, visit: https://github.com/lablup/mlxcel"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a model: interactive chat, or one-shot generation with `-p`.
    ///
    /// Mirrors `ollama run` / mlx-lm ergonomics. Pass a HuggingFace
    /// `owner/name` repo-id or a local model directory (auto-downloaded and
    /// resolved exactly like `mlxcel generate -m`):
    ///
    ///     mlxcel run mlx-community/Qwen3-4B-4bit            # interactive chat
    ///     mlxcel run mlx-community/Qwen3-4B-4bit -p "Hi"    # one-shot, then exit
    ///     mlxcel run Qwen3-4B-4bit                          # bare name -> mlx-community/Qwen3-4B-4bit
    ///     mlxcel run                                        # default model, interactive
    ///
    /// A bare name without a slash (e.g. `Qwen3-4B-4bit`) is resolved as
    /// `mlx-community/<name>`; override the org with `MLXCEL_DEFAULT_ORG`.
    ///
    /// With no `-p/--prompt`, `run` drops into the interactive multi-turn chat
    /// REPL. With `-p`, it produces a single completion and exits — identical
    /// to the equivalent `mlxcel generate` invocation. With no model argument,
    /// it falls back to the default model
    /// `mlx-community/Llama-3.2-3B-Instruct-4bit` (mlx-lm parity).
    Run(commands::RunArgs),

    /// Generate text from a prompt
    #[command(visible_alias = "gen")]
    Generate(GenerateArgs),

    /// Start an OpenAI/llama-server compatible HTTP server
    Serve(ServeArgs),

    /// List downloaded models in the local store
    #[command(visible_alias = "ls")]
    List(ListArgs),

    /// List supported model architectures
    #[command(visible_alias = "supported")]
    Arch(ArchArgs),

    /// Print a pre-load memory budget for a model without running generation.
    ///
    /// Reports the byte breakdown for weights / KV cache / runtime
    /// activation headroom and compares against available unified
    /// memory. Use this before launching `mlxcel generate` or `mlxcel
    /// serve` to decide whether a model + context length will fit.
    ///
    /// Examples:
    ///
    ///     mlxcel inspect models/llama-3.2-1b-4bit
    ///     mlxcel inspect models/llama-3.2-1b-4bit --max-tokens 32768
    ///     mlxcel inspect models/llama-3.2-1b-4bit --cache-type-k int8 --cache-type-v int8
    Inspect(InspectArgs),

    /// Download a HuggingFace model repository snapshot
    Download(DownloadArgs),

    /// Detect objects in an image with an RT-DETRv2 model.
    ///
    /// Loads an RT-DETRv2 object-detection checkpoint and prints bounding
    /// boxes (`l, t, r, b`), class labels, and confidences in original-image
    /// pixel coordinates. Detection models output boxes rather than a token
    /// stream, so this is a separate surface from `generate`.
    ///
    /// Examples:
    ///
    ///     mlxcel detect -m models/docling-layout-heron-mlx-bf16 -i page.png
    ///     mlxcel detect -m models/rt-detr-v2 -i img.jpg --threshold 0.5 --format json
    Detect(DetectArgs),

    /// Remove a downloaded model from the global store.
    ///
    /// Deletes `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>`
    /// for the given repo-id and frees the space. Prompts for confirmation
    /// unless `--yes` is passed. Models that exist only in the read-only
    /// HuggingFace cache are reported but never deleted (mlxcel does not manage
    /// the HuggingFace cache).
    ///
    /// Examples:
    ///
    ///     mlxcel rm mlx-community/Qwen3-4B-4bit
    ///     mlxcel rm mlx-community/Qwen3-4B-4bit --yes
    Rm(RmArgs),
}

/// Arguments for `mlxcel list`.
///
/// `list` enumerates the models you have downloaded into the global store,
/// mirroring `ollama list`. The default table shows NAME / SIZE / MODIFIED;
/// `-v/--verbose` restores the absolute PATH column. `--json` and `-q/--quiet`
/// give machine-readable and pipe-friendly output. The supported
/// model-architecture catalog lives under the separate `mlxcel arch` verb.
#[derive(Args, Debug)]
pub(crate) struct ListArgs {
    /// Model-store root to list instead of the default location.
    ///
    /// Lists snapshots directly under `<PATH>/<owner>/<name>` (no extra
    /// `models/` subdir). Overrides the `MLXCEL_MODELS_DIR` environment
    /// variable.
    #[arg(long, value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,

    /// Emit a JSON array of `{repo_id, size_bytes, path, modified}` (modified is
    /// Unix epoch seconds, or null). Disables the table, header, and styling.
    ///
    /// Mutually exclusive with `--quiet` and `--verbose`.
    #[arg(long, conflicts_with_all = ["quiet", "verbose"])]
    pub(crate) json: bool,

    /// Print only repo-ids, one per line — no header or columns — so the output
    /// pipes cleanly (e.g. `mlxcel list -q | xargs -n1 mlxcel rm`).
    ///
    /// Mutually exclusive with `--json` and `--verbose`.
    #[arg(short, long, conflicts_with = "verbose")]
    pub(crate) quiet: bool,

    /// Append the absolute on-disk PATH column back to the default table.
    #[arg(short, long)]
    pub(crate) verbose: bool,

    /// Sort order: `name` (repo-id, default), `size` (largest first), or
    /// `modified` (most-recent first; unknown mtimes last). Applies to the
    /// table and `--json` alike.
    #[arg(long, value_enum, default_value_t = commands::models::SortKey::Name)]
    pub(crate) sort: commands::models::SortKey,
}

/// Arguments for `mlxcel arch`.
///
/// Currently takes no flags; the empty struct lets the verb grow options
/// later without a breaking signature change.
#[derive(Args, Debug)]
pub(crate) struct ArchArgs {}

/// Arguments for `mlxcel rm`.
#[derive(Args, Debug)]
pub(crate) struct RmArgs {
    /// HuggingFace repository id to remove, e.g. `mlx-community/Qwen3-4B-4bit`.
    #[arg(value_name = "REPO_ID")]
    pub(crate) repo_id: String,

    /// Model-store root to remove from instead of the default location.
    ///
    /// Removes the snapshot at `<PATH>/<owner>/<name>` (no extra `models/`
    /// subdir). Overrides the `MLXCEL_MODELS_DIR` environment variable. The
    /// path is used verbatim as the store root: deletion only ever touches
    /// `<PATH>/<owner>/<name>`, so point it at a real model store.
    #[arg(long, value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,

    /// Skip the interactive confirmation prompt.
    #[arg(long, short = 'y', default_value_t = false)]
    pub(crate) yes: bool,

    /// Repository revision used only to locate an HF-cache snapshot when the
    /// model is not in the mlxcel store (defaults to `main`). The mlxcel store
    /// itself is not revision-namespaced.
    #[arg(long, value_name = "REV")]
    pub(crate) revision: Option<String>,
}

#[derive(Args, Debug)]
pub(crate) struct GenerateArgs {
    #[command(flatten)]
    pub(crate) model: ModelOptions,

    #[command(flatten)]
    pub(crate) generation: GenerationOptions,

    #[command(flatten)]
    pub(crate) sampling: SamplingOptions,

    #[command(flatten)]
    pub(crate) pipeline_parallel: PipelineParallelOptions,

    #[command(flatten)]
    pub(crate) tensor_parallel: TensorParallelOptions,

    #[command(flatten)]
    pub(crate) lang_bias: LangBiasCliArgs,

    /// Speculative-decoding flag group (`--draft-kind`, `--draft-block-size`).
    /// The `--draft-model` and `--num-draft-tokens` flags stay on
    /// [`ModelOptions`] above to keep their existing scope and to preserve
    /// the llama-server-compat naming on `mlxcel-server`. See
    /// [`SpeculativeArgs`] for the rationale.
    #[command(flatten)]
    pub(crate) speculative: SpeculativeArgs,

    // Axis A weight-load surgery configuration.
    // The closed-repo references stay in this non-doc comment so the
    // user-facing `--help` block does not advertise tracker URLs (see
    // `tests/cli_help_consistency.rs::FORBIDDEN_SUBSTRINGS`).
    /// Apply weight-load surgery configuration from a YAML file.
    ///
    /// Path to a YAML configuration file describing structural
    /// fine-tuning operations (scale / add / prune / replace /
    /// interpolate). When omitted, model loading is bit-exact identical
    /// to the pre-surgery baseline — no extra work, no observable
    /// difference in generated tokens for any seed.
    ///
    /// Example:
    ///
    ///     mlxcel generate -m models/foo --surgery surgery.yaml -p "Hello"
    ///
    /// The supported surgery operations are summarised in the project README.
    #[cfg(feature = "surgery")]
    #[arg(long = "surgery", value_name = "FILE", env = "MLXCEL_SURGERY")]
    pub(crate) surgery: Option<PathBuf>,
}

/// Model loading options
#[derive(Args, Debug)]
#[command(next_help_heading = "Model Options")]
pub(crate) struct ModelOptions {
    /// Path to a local model directory, or a HuggingFace `owner/name`
    /// repo-id to auto-download.
    ///
    /// An existing local path is used as-is. Otherwise an `owner/name`
    /// repo-id (e.g. `mlx-community/Qwen3-4B-4bit`) is resolved from a
    /// legacy `./models/<name>` directory, the HuggingFace cache, or the
    /// mlxcel store, and downloaded into the mlxcel store on a miss so it
    /// runs from any directory. A bare name without a slash (e.g.
    /// `Qwen3-4B-4bit`) is resolved as `mlx-community/<name>`; override
    /// the org with the `MLXCEL_DEFAULT_ORG` environment variable.
    #[arg(short, long, value_name = "PATH_OR_REPO_ID")]
    pub(crate) model: PathBuf,

    /// Model-store root for resolving / downloading an `owner/name` repo-id.
    ///
    /// Sets the directory that directly holds snapshots, so a repo-id resolves
    /// to / downloads at `<PATH>/<owner>/<name>` (no extra `models/` subdir).
    /// Overrides the `MLXCEL_MODELS_DIR` environment variable. No effect when
    /// the model argument is already an existing local path.
    #[arg(long, value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,

    /// Path to LoRA adapter directory (optional)
    #[arg(long, value_name = "PATH")]
    pub(crate) adapter: Option<PathBuf>,

    /// Path to draft model for classic offline speculative decoding (optional)
    #[arg(long, value_name = "PATH")]
    pub(crate) draft_model: Option<PathBuf>,

    /// Number of draft tokens per speculation step (default: 3)
    #[arg(long, default_value_t = 3, value_name = "N")]
    pub(crate) num_draft_tokens: usize,
}

/// Text generation options
#[derive(Args, Debug)]
#[command(next_help_heading = "Generation Options")]
pub(crate) struct GenerationOptions {
    /// The prompt to generate text from.
    ///
    /// When omitted, `mlxcel generate` drops into an interactive multi-turn
    /// chat REPL (streaming output, `/bye` / `/clear` / `/?` slash commands,
    /// and `"""` multiline blocks) instead of running a single completion —
    /// mirroring `mlx_lm.chat` / `ollama run`.
    #[arg(short, long, value_name = "TEXT")]
    pub(crate) prompt: Option<String>,

    /// Image file paths for vision-language models (VLM)
    #[arg(long, value_name = "PATH", num_args = 1..)]
    pub(crate) image: Vec<PathBuf>,

    /// Audio file path for audio-language models (e.g. Gemma4 with audio)
    #[arg(long, value_name = "PATH")]
    pub(crate) audio: Option<PathBuf>,

    /// Video file paths for VLMs that support video inputs (e.g. Gemma4
    /// with video). Pass the flag multiple times for multiple videos:
    /// `--video clip1.mp4 --video clip2.mp4`. Frame extraction requires
    /// `ffmpeg` on PATH.
    #[arg(long, value_name = "PATH", num_args = 1..)]
    pub(crate) video: Vec<PathBuf>,

    /// Target sampling FPS for `--video` decoding. Frames are
    /// uniformly resampled to this rate before being fed to the
    /// vision tower. Defaults to 2.0 to match the upstream mlx-vlm
    /// behaviour.
    #[arg(long, value_name = "FLOAT", default_value_t = 2.0)]
    pub(crate) fps: f64,

    /// Maximum number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 100, value_name = "N")]
    pub(crate) max_tokens: usize,

    /// Enable profiling mode with detailed timing breakdown
    #[arg(long, default_value_t = false)]
    pub(crate) profile: bool,

    /// Disable automatic chat template application
    #[arg(long, default_value_t = false)]
    pub(crate) no_chat_template: bool,

    /// Print the recommended quantization mode for this model on the current hardware.
    ///
    /// Detects Apple Silicon generation and available memory, estimates model
    /// parameter count from config.json, then suggests the optimal quantization
    /// (int8, int4, or fp16). On M5 hardware with sufficient memory, INT8 is
    /// recommended because the Neural Accelerator delivers ~2x throughput over
    /// FP16 for 8-bit integer matmuls.
    ///
    /// Also warns when the model uses BFloat16 weights on M5 hardware, since
    /// the Neural Accelerator does not support BFloat16 computation.
    #[arg(long, default_value_t = false)]
    pub(crate) recommend_quant: bool,

    /// Run a pre-load memory estimate; abort when the model will not fit.
    ///
    /// Computes weights + KV cache + runtime activation headroom and
    /// compares the total against available unified memory. When the
    /// total exceeds the available budget, generation aborts with a
    /// clear error pointing at the over-budget figure. Use `--force`
    /// (alias `--no-memory-check`) to bypass the preflight when you
    /// have verified the estimate is conservative.
    ///
    /// Always runs before model load. Safe to combine with
    /// `--recommend-quant`: both consume the same estimator.
    #[arg(long, default_value_t = false)]
    pub(crate) estimate_memory: bool,

    /// Bypass the `--estimate-memory` preflight abort.
    ///
    /// When `--estimate-memory` would refuse to load a model because
    /// total > available, `--force` (or its alias `--no-memory-check`)
    /// downgrades the abort to a warning and continues. No-op when
    /// `--estimate-memory` is not set.
    #[arg(long = "force", alias = "no-memory-check", default_value_t = false)]
    pub(crate) force_memory: bool,

    // Shared TurboQuant KV-cache flag group (--cache-type-k, --cache-type-v,
    // --kv-cache-mode, --turbo-boundary-v). Defined once in
    // mlxcel::cli::turbo_args so all three binaries (mlxcel generate,
    // mlxcel serve, mlxcel-server) expose identical help text and flags.
    #[command(flatten)]
    pub(crate) turbo: TurboKvCacheArgs,

    // Block-diffusion flag group (--max-denoising-steps,
    // --diffusion-sampler, ...). Only affects diffusion models such as
    // DiffusionGemma; autoregressive models ignore it.
    #[command(flatten)]
    pub(crate) diffusion: DiffusionCliOptions,
}

/// Arguments for the `mlxcel inspect` subcommand.
///
/// Read-only: prints the unified memory estimate (weights / KV cache /
/// runtime headroom / total vs available unified memory) and exits
/// without loading the model. Mirrors the relevant flag surface of
/// `mlxcel generate` so an operator can sanity-check a configuration
/// before launching a real load.
#[derive(Args, Debug)]
#[command(next_help_heading = "Inspect Options")]
pub(crate) struct InspectArgs {
    /// Path to a local model directory, or a HuggingFace `owner/name`
    /// repo-id to auto-download.
    ///
    /// An existing local path is used as-is; an `owner/name` repo-id
    /// (e.g. `mlx-community/Qwen3-4B-4bit`) is resolved from a legacy
    /// `./models/<name>` directory, the HuggingFace cache, or the mlxcel
    /// store, and downloaded into the mlxcel store on a miss. A bare name
    /// without a slash (e.g. `Qwen3-4B-4bit`) is resolved as
    /// `mlx-community/<name>`; override the org with the
    /// `MLXCEL_DEFAULT_ORG` environment variable.
    #[arg(short, long, value_name = "PATH_OR_REPO_ID")]
    pub(crate) model: std::path::PathBuf,

    /// Model-store root for resolving / downloading an `owner/name` repo-id.
    ///
    /// Sets the directory that directly holds snapshots, so a repo-id resolves
    /// to / downloads at `<PATH>/<owner>/<name>` (no extra `models/` subdir).
    /// Overrides the `MLXCEL_MODELS_DIR` environment variable. No effect when
    /// the model argument is already an existing local path.
    #[arg(long, value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,

    /// Maximum number of tokens to estimate KV cache for.
    ///
    /// Treated as the context length input to the KV cache estimator.
    /// Larger values yield larger KV-cache totals. Defaults to 8192 to
    /// match the historical sizing used by `--recommend-quant`.
    #[arg(short = 'n', long, default_value_t = 8192, value_name = "N")]
    pub(crate) max_tokens: u64,

    /// Batch size used in the KV-cache estimate. Defaults to 1 for
    /// interactive use.
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub(crate) batch: u64,

    /// Quantization mode label (does not affect the byte total — the
    /// safetensors header is taken at face value because mlxcel
    /// quantizes lazily). One of: default, fp16, int8, int4.
    #[arg(long, default_value = "default", value_name = "MODE")]
    pub(crate) quant: String,

    // Shared TurboQuant KV-cache flag group — gives `inspect` the same
    // `--cache-type-k` / `--cache-type-v` surface as `generate` so the
    // estimate matches what the loaded model would actually allocate.
    #[command(flatten)]
    pub(crate) turbo: TurboKvCacheArgs,
}

/// Arguments for the `mlxcel detect` subcommand.
#[derive(Args, Debug)]
pub(crate) struct DetectArgs {
    /// Path to the RT-DETRv2 model directory (config.json + safetensors).
    #[arg(short, long, value_name = "PATH")]
    pub(crate) model: std::path::PathBuf,

    /// Path to the input image file (PNG, JPEG, etc.).
    #[arg(short, long, value_name = "PATH")]
    pub(crate) image: std::path::PathBuf,

    /// Confidence threshold in [0, 1]; detections below it are dropped.
    #[arg(short, long, default_value_t = 0.3, value_name = "FLOAT")]
    pub(crate) threshold: f32,

    /// Output format.
    #[arg(long, value_enum, default_value_t = commands::detect::OutputFormat::Text)]
    pub(crate) format: commands::detect::OutputFormat,
}

/// Sampling strategy options
#[derive(Args, Debug)]
#[command(next_help_heading = "Sampling Options")]
pub(crate) struct SamplingOptions {
    /// Sampling temperature (0.0 = greedy, higher = more random)
    #[arg(short, long, default_value_t = 0.0, value_name = "FLOAT")]
    pub(crate) temp: f32,

    /// Top-P (nucleus) sampling threshold
    #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
    pub(crate) top_p: f32,

    /// Top-K sampling limit
    #[arg(long, default_value_t = 0, value_name = "K")]
    pub(crate) top_k: i32,

    /// Min-P sampling threshold (0.0 = disabled)
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    pub(crate) min_p: f32,

    /// Repetition penalty multiplier
    #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
    pub(crate) repetition_penalty: f32,

    /// DRY (Don't Repeat Yourself) penalty multiplier (0.0 = disabled)
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    pub(crate) dry_multiplier: f32,

    /// DRY exponential base for penalty scaling
    #[arg(long, default_value_t = 1.75, value_name = "FLOAT")]
    pub(crate) dry_base: f32,

    /// DRY minimum match length before penalty applies
    #[arg(long, default_value_t = 2, value_name = "N")]
    pub(crate) dry_allowed_length: usize,

    /// DRY lookback window size (0 = use full history)
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub(crate) dry_penalty_last_n: usize,

    /// Random seed for MLX's global RNG. Makes sampled generation
    /// reproducible, including the random canvas noise of diffusion models
    /// (e.g. DiffusionGemma). Unset = nondeterministic.
    #[arg(long, value_name = "N")]
    pub(crate) seed: Option<u64>,
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

/// Block-diffusion generation options.
///
/// These flags only affect diffusion models (e.g. DiffusionGemma); ordinary
/// autoregressive models ignore them. They mirror the mlx-vlm diffusion
/// flag surface.
#[derive(Args, Debug)]
#[command(next_help_heading = "Diffusion Options")]
pub(crate) struct DiffusionCliOptions {
    /// Maximum denoising steps per canvas block (diffusion models only;
    /// default: the checkpoint's generation_config, typically 48)
    #[arg(long = "max-denoising-steps", value_name = "N")]
    pub(crate) max_denoising_steps: Option<usize>,

    /// Per-step acceptance sampler for diffusion models
    #[arg(
        long = "diffusion-sampler",
        value_name = "SAMPLER",
        default_value = "entropy-bound",
        value_parser = ["entropy-bound", "confidence-threshold"]
    )]
    pub(crate) diffusion_sampler: String,

    /// Confidence threshold for `--diffusion-sampler confidence-threshold`
    #[arg(
        long = "diffusion-threshold",
        value_name = "FLOAT",
        default_value_t = 0.9,
        value_parser = parse_unit_interval
    )]
    pub(crate) diffusion_threshold: f32,

    /// Smallest canvas allocated for the generation tail (diffusion only)
    #[arg(
        long = "diffusion-min-canvas-length",
        value_name = "N",
        default_value_t = 64
    )]
    pub(crate) diffusion_min_canvas_length: usize,

    /// Cap on the per-block canvas length (diffusion only; default: the
    /// model's canvas_length, typically 256)
    #[arg(long = "diffusion-max-canvas-length", value_name = "N")]
    pub(crate) diffusion_max_canvas_length: Option<usize>,

    /// Always allocate the model's full canvas length per block (diffusion
    /// only)
    #[arg(long = "diffusion-full-canvas", default_value_t = false)]
    pub(crate) diffusion_full_canvas: bool,
}

// Manual `Default` kept in lock-step with the `#[arg(default_value*)]`
// attributes above, same contract as the parallelism option groups.
impl Default for DiffusionCliOptions {
    fn default() -> Self {
        Self {
            max_denoising_steps: None,
            diffusion_sampler: "entropy-bound".to_string(),
            diffusion_threshold: 0.9,
            diffusion_min_canvas_length: 64,
            diffusion_max_canvas_length: None,
            diffusion_full_canvas: false,
        }
    }
}

/// Tensor-parallel options
#[derive(Args, Debug)]
#[command(next_help_heading = "Tensor Parallel Options")]
pub(crate) struct TensorParallelOptions {
    /// Number of tensor-parallel ranks (must be a power of 2).
    ///
    /// Current multi-rank runtime support is limited to dense Llama, Qwen2/2.5,
    /// Qwen3, Qwen3.5 text, Gemma 3 text, Gemma 4 text, ERNIE 4.5, and
    /// Hunyuan v1 Dense models.
    #[arg(long = "tp-size", default_value_t = 1, value_name = "N")]
    pub(crate) tp_size: usize,

    /// MoE expert sharding mode: "expert_parallel" or "within_expert"
    #[arg(
        long = "tp-moe-mode",
        default_value = "expert_parallel",
        value_name = "MODE"
    )]
    pub(crate) tp_moe_mode: String,

    /// Embedding sharding mode: "vocab_parallel" or "replicated".
    ///
    /// The current in-process tensor-parallel runtime requires "replicated".
    #[arg(
        long = "tp-embedding-mode",
        default_value = "replicated",
        value_name = "MODE"
    )]
    pub(crate) tp_embedding_mode: String,

    /// LM head sharding mode: "vocab_parallel" or "replicated".
    ///
    /// The current in-process tensor-parallel runtime requires "replicated".
    #[arg(
        long = "tp-lm-head-mode",
        default_value = "replicated",
        value_name = "MODE"
    )]
    pub(crate) tp_lm_head_mode: String,
}

/// Pipeline-parallel options
#[derive(Args, Debug)]
#[command(next_help_heading = "Pipeline Parallel Options")]
pub(crate) struct PipelineParallelOptions {
    /// Number of pipeline stages. Values <= 1 disable pipeline mode.
    #[arg(long = "pp-size", default_value_t = 1, value_name = "N")]
    pub(crate) pp_size: usize,

    /// Manual pipeline-parallel layer partition (e.g. "0-15,16-31").
    ///
    /// When omitted and `--pp-size >= 2`, the CLI path auto-partitions
    /// the model into equal-capacity in-process stages.
    #[arg(long = "pp-layers", value_name = "RANGES")]
    pub(crate) pp_layers: Option<String>,

    /// Micro-batch size for pipeline parallelism.
    #[arg(long = "pp-micro-batch-size", default_value_t = 1, value_name = "N")]
    pub(crate) pp_micro_batch_size: usize,
}

// `Default` impls for the parallelism option groups so the `mlxcel run`
// dispatcher (`commands::run`) can build a `GenerateArgs` while leaving these
// advanced groups at their inert single-device defaults — `run` deliberately
// does not expose tensor/pipeline parallelism. These MUST stay in lock-step
// with the `#[arg(default_value*)]` attributes above; the
// `run_defaults_match_clap_defaults` test in `main_tests.rs` fails the build if
// they ever drift.

impl Default for TensorParallelOptions {
    fn default() -> Self {
        Self {
            tp_size: 1,
            tp_moe_mode: "expert_parallel".to_string(),
            tp_embedding_mode: "replicated".to_string(),
            tp_lm_head_mode: "replicated".to_string(),
        }
    }
}

impl Default for PipelineParallelOptions {
    fn default() -> Self {
        Self {
            pp_size: 1,
            pp_layers: None,
            pp_micro_batch_size: 1,
        }
    }
}

/// clap value parser for `--kv-cache-budget`: a raw byte count or the
/// literal `auto` (epic #116 #122 b3). Delegates to
/// [`mlxcel::memory_estimate::PagedBudgetDirective`]'s `FromStr`.
fn parse_kv_cache_budget(s: &str) -> Result<mlxcel::memory_estimate::PagedBudgetDirective, String> {
    s.parse()
}

/// Server options
#[derive(Args, Debug)]
#[command(after_help = "\
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
       mlxcel serve -m models/llama-3.2-1b-4bit \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id stage-1 --host 0.0.0.0 --port 18081 --no-warmup

  3. Start stage-0 on machine A:
       mlxcel serve -m models/llama-3.2-1b-4bit \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id stage-0 --host 0.0.0.0 --port 18081 --no-warmup

  4. Start the coordinator on machine A:
       mlxcel serve -m models/llama-3.2-1b-4bit --alias llama-remote-pp \\
         --distributed-config examples/distributed/generated_pipeline_remote_2node_tcp.toml \\
         --node-id coordinator --host 0.0.0.0 --port 18080 \\
         --parallel 2 --max-batch-size 2 --pp-micro-batch-size 2 \\
         --metrics --no-warmup

Thunderbolt mode:
  Use the same workflow with TRANSPORT_BACKEND=thunderbolt and each node's
  Thunderbolt Bridge IP (for example 169.254.x.x). The current Thunderbolt
  path uses the shared TCP transport core over the Bridge network.

See also: docs/distributed.md")]
pub(crate) struct ServeArgs {
    /// Path to a local model directory, or a HuggingFace `owner/name`
    /// repo-id to auto-download.
    ///
    /// An existing local path is used as-is; an `owner/name` repo-id
    /// (e.g. `mlx-community/Qwen3-4B-4bit`) is resolved from a legacy
    /// `./models/<name>` directory, the HuggingFace cache, or the mlxcel
    /// store, and downloaded into the mlxcel store on a miss. A bare name
    /// without a slash (e.g. `Qwen3-4B-4bit`) is resolved as
    /// `mlx-community/<name>`; override the org with the
    /// `MLXCEL_DEFAULT_ORG` environment variable.
    #[arg(short, long, env = "LLAMA_ARG_MODEL", value_name = "PATH_OR_REPO_ID")]
    model: PathBuf,

    /// Model-store root for resolving / downloading an `owner/name` repo-id.
    ///
    /// Sets the directory that directly holds snapshots, so a repo-id resolves
    /// to / downloads at `<PATH>/<owner>/<name>` (no extra `models/` subdir).
    /// Overrides the `MLXCEL_MODELS_DIR` environment variable. No effect when
    /// the model argument is already an existing local path.
    #[arg(long, value_name = "PATH")]
    models_dir: Option<PathBuf>,

    /// Path to LoRA adapter directory
    #[arg(long, visible_alias = "lora", value_name = "PATH")]
    adapter: Option<PathBuf>,

    /// Model alias (shown in API responses instead of directory name)
    #[arg(short = 'a', long, env = "LLAMA_ARG_ALIAS", value_name = "NAME")]
    alias: Option<String>,

    /// Host address to bind to (or Unix socket path when --port 0)
    #[arg(long, env = "LLAMA_ARG_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Port number to listen on (0 = Unix socket mode using --host as socket path)
    #[arg(long, env = "LLAMA_ARG_PORT", default_value_t = 8080)]
    port: u16,

    /// API key for authentication
    #[arg(long, env = "LLAMA_API_KEY", value_name = "KEY")]
    api_key: Option<String>,

    /// Path to file containing API key
    #[arg(long, value_name = "PATH")]
    api_key_file: Option<PathBuf>,

    /// Number of parallel request slots that share --ctx-size
    #[arg(long, env = "LLAMA_ARG_N_PARALLEL", default_value_t = 1)]
    n_parallel: usize,

    /// Total context budget shared across parallel slots (0 = use model default)
    #[arg(long, env = "LLAMA_ARG_CTX_SIZE", default_value_t = 0)]
    ctx_size: usize,

    /// Maximum tokens to predict (-1 = unlimited)
    #[arg(long = "n-predict", env = "LLAMA_ARG_N_PREDICT", default_value_t = -1)]
    n_predict: i32,

    /// Path to drafter checkpoint for server speculative decoding
    #[arg(long, value_name = "PATH")]
    draft_model: Option<PathBuf>,

    /// Maximum number of draft tokens per speculation step
    #[arg(long, env = "LLAMA_ARG_DRAFT_MAX", default_value_t = 16)]
    draft_max: usize,

    /// Maximum concurrent decode sequences; explicit value shares --ctx-size
    #[arg(long, value_name = "N")]
    max_batch_size: Option<usize>,

    /// Disable continuous batching and use the legacy sequential worker.
    ///
    /// When set, requests are processed one at a time in FIFO order with no
    /// batch scheduler overhead. Equivalent to using `--max-batch-size 1` but
    /// with explicit sequential semantics and no prefill chunking.
    #[arg(long)]
    no_batch: bool,

    /// Maximum number of requests waiting in the prefill queue (default: 32)
    #[arg(long, default_value_t = 32)]
    max_queue_depth: usize,

    /// Prefill chunk size in tokens (0 = disabled, default: 512)
    ///
    /// When set, long prompts are broken into chunks of this size and
    /// decode steps are interleaved between chunks to prevent latency
    /// spikes for active sequences.
    #[arg(long, default_value_t = 512)]
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
    ///
    /// When enabled and the batch is full, a high-priority incoming
    /// request may evict a lower-priority active sequence (which will
    /// be re-queued for re-prefill).
    #[arg(long)]
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

    /// Preemption policy: "longest-first" (default) or "lowest-priority"
    ///
    /// Controls which active sequence is evicted when preemption is
    /// triggered. "longest-first" evicts the sequence with the most
    /// generated tokens; "lowest-priority" evicts the lowest-priority
    /// sequence (ties broken by longest).
    #[arg(long, default_value = "longest-first")]
    preemption_policy: String,

    /// Maximum number of requests to batch together for prefill (default: 1)
    ///
    /// When > 1, the scheduler collects up to this many pending requests and
    /// runs a single batched forward pass [batch_size, max_seq_len] for better
    /// Neural Accelerator utilization. Falls back to sequential prefill when
    /// only one request is pending. Recommended: 4-8 on M5 Pro/Max hardware.
    #[arg(long, default_value_t = 1)]
    max_batch_prefill: usize,

    /// Maximum KV cache size for plain (non-sliding) caches (0 = unbounded, the default).
    ///
    /// When set to `N > 0`, the batch scheduler caps each per-sequence plain
    /// `KVCache` to `N` tokens by dropping the oldest entries once `offset`
    /// exceeds the bound. Mirrors upstream mlx-lm's
    /// `BatchGenerator(max_kv_size=N)` parameter. Sliding-window
    /// models (Gemma 3/4, Exaone 4, RecurrentGemma, Step 3.5, gpt-oss) keep
    /// their model-specific window. Not supported with Turbo KV quantization.
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
    /// Bounds the unified paged KV cache (epic #116) so the server evicts cold
    /// cross-request prompt prefixes (then preempts running sequences) instead
    /// of growing the pool without limit. `auto` derives the cap from the
    /// memory estimate (`(available - weights - activation) / per-block bytes`);
    /// a raw byte count (e.g. `8589934592` for 8 GiB) sets an explicit cap.
    /// Only affects pool-backed (Fp16, dense-natural-backend) models. Model-owned
    /// and quantized families keep dense caches and ignore it. Requires
    /// `--decode-storage-backend paged` to have any effect.
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
    /// entirely, in which case `GET /v1/responses/:id` and
    /// `previous_response_id` chaining return 400.
    /// Also reads `LLAMA_ARG_RESPONSES_STORE_MAX_ENTRIES`.
    #[arg(
        long = "responses-store-max-entries",
        env = "LLAMA_ARG_RESPONSES_STORE_MAX_ENTRIES",
        default_value_t = 1024,
        value_name = "N"
    )]
    responses_store_max_entries: usize,

    /// TTL (seconds) for in-memory Responses-API response
    /// entries. `0` disables TTL — entries are evicted only when the
    /// max-entries cap is hit.
    /// Also reads `LLAMA_ARG_RESPONSES_STORE_TTL_SECS`.
    #[arg(
        long = "responses-store-ttl-secs",
        env = "LLAMA_ARG_RESPONSES_STORE_TTL_SECS",
        default_value_t = 3600,
        value_name = "SECS"
    )]
    responses_store_ttl_secs: u64,

    /// Maximum number of conversation transcripts persisted
    /// for the OpenAI Responses API `conversation` field. `0` disables
    /// the conversation store; requests referencing `conversation` are
    /// still accepted but operate against an empty transcript.
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

    /// Request timeout in seconds
    #[arg(long, env = "LLAMA_ARG_TIMEOUT", default_value_t = 600)]
    timeout: u64,

    /// Override chat template (Jinja2 template string)
    #[arg(long, value_name = "TEMPLATE")]
    chat_template: Option<String>,

    /// Path to chat template file
    #[arg(long, value_name = "PATH")]
    chat_template_file: Option<PathBuf>,

    /// Enable /slots endpoint
    #[arg(long = "slots", overrides_with = "_no_slots", default_value_t = true)]
    slots: bool,

    /// Disable /slots endpoint
    #[arg(long = "no-slots", overrides_with = "slots", hide = true)]
    _no_slots: bool,

    /// Enable /props endpoint
    #[arg(long)]
    props: bool,

    /// Enable /metrics endpoint
    #[arg(long)]
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
    #[arg(long, env = "LLAMA_ARG_TOP_K", default_value_t = 40)]
    top_k: i32,

    /// Default top-P (nucleus) sampling
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Default min-P sampling
    #[arg(long, default_value_t = 0.1)]
    min_p: f32,

    /// Random seed (-1 = random)
    #[arg(short = 's', long, default_value_t = -1)]
    seed: i64,

    /// Default repetition penalty lookback window
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Default repetition penalty multiplier
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Default presence penalty
    #[arg(long, default_value_t = 0.0)]
    presence_penalty: f32,

    /// Default frequency penalty
    #[arg(long, default_value_t = 0.0)]
    frequency_penalty: f32,

    // DRY sampling parameters.
    /// DRY penalty multiplier (0.0 = disabled)
    #[arg(long, default_value_t = 0.0)]
    dry_multiplier: f32,

    /// DRY exponential base
    #[arg(long, default_value_t = 1.75)]
    dry_base: f32,

    /// DRY minimum match length before penalty
    #[arg(long, default_value_t = 2)]
    dry_allowed_length: usize,

    /// DRY lookback window (-1 = full context)
    #[arg(long, default_value_t = -1)]
    dry_penalty_last_n: i32,

    /// DRY sequence breaker token strings (e.g. "\n", "\t")
    #[arg(long, value_delimiter = ',')]
    dry_sequence_breakers: Vec<String>,

    // Logging.
    /// Enable verbose (debug) logging
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Disable all logging
    #[arg(long)]
    log_disable: bool,

    /// Log output file
    #[arg(long, env = "LLAMA_LOG_FILE", value_name = "PATH")]
    log_file: Option<PathBuf>,

    // Distributed inference.
    /// Path to TOML cluster configuration file for distributed inference
    #[arg(long, value_name = "PATH")]
    distributed_config: Option<std::path::PathBuf>,

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

    /// Micro-batch size for pipeline parallelism
    ///
    /// Splits incoming batches into micro-batches of this size to fill the
    /// pipeline and reduce bubble time. Smaller values improve pipeline
    /// utilization but add scheduling overhead. Default: 1.
    #[arg(long = "pp-micro-batch-size", default_value_t = 1, value_name = "N")]
    pp_micro_batch_size: usize,

    /// Zero-config pipeline bring-up: coordinator declares N stages.
    ///
    /// See the "Pipeline parallelism" section of `docs/distributed.md` for the
    /// full operator workflow. Mutually exclusive with
    /// `--distributed-config`.
    #[arg(long = "pp-auto", value_name = "N")]
    pp_auto: Option<u32>,

    /// Zero-config pipeline bring-up: run as a peer that joins a coordinator.
    #[arg(long = "pp-peer")]
    pp_peer: bool,

    /// Cluster discovery mechanism: "static" (default) or "mdns" for UDP broadcast.
    #[arg(
        long = "cluster-discovery",
        default_value = "static",
        value_name = "MODE"
    )]
    cluster_discovery: String,

    /// Human-readable cluster name for discovery and TOML header.
    #[arg(long = "cluster-name", value_name = "NAME")]
    cluster_name: Option<String>,

    /// Static peer addresses for zero-config bring-up (host:port, comma-separated).
    #[arg(long = "cluster-peers", value_delimiter = ',', value_name = "ADDR")]
    cluster_peers: Vec<std::net::SocketAddr>,

    /// UDP port for the discovery beacon when `--cluster-discovery=mdns`.
    #[arg(long = "cluster-discovery-port", value_name = "PORT")]
    cluster_discovery_port: Option<u16>,

    /// Coordinator control-plane bind address for zero-config bring-up (host:port).
    #[arg(long = "cluster-control-addr", value_name = "ADDR")]
    cluster_control_addr: Option<std::net::SocketAddr>,

    /// Output path for the emitted cluster TOML (default: `.mlxcel/cluster.toml`).
    #[arg(long = "cluster-config-out", value_name = "PATH")]
    cluster_config_out: Option<PathBuf>,

    /// Plan the cluster topology and emit the TOML without starting workers.
    #[arg(long = "dry-run", default_value_t = false)]
    dry_run: bool,

    /// Number of tensor-parallel ranks (must be a power of 2).
    ///
    /// When set to N > 1, model weights are sharded across N in-process ranks.
    /// Current multi-rank runtime support is limited to dense Llama, Qwen2/2.5,
    /// Qwen3, Gemma 3 text, ERNIE 4.5, and Hunyuan v1 Dense models.
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

    // llama-server compatibility arguments (accepted but ignored).
    /// Accepted for llama-server CLI compatibility (ignored — mlxcel has no web UI)
    #[arg(long, hide = true)]
    _no_webui: bool,

    /// Accepted for llama-server CLI compatibility (ignored — mlxcel always processes templates)
    #[arg(long, hide = true)]
    _jinja: bool,

    /// Accepted for llama-server CLI compatibility (ignored — mlxcel always uses Metal)
    #[arg(long = "n-gpu-layers", hide = true)]
    _n_gpu_layers: Option<i32>,

    /// Accepted for llama-server CLI compatibility (ignored — vision projector loaded automatically)
    #[arg(long, hide = true)]
    _mmproj: Option<String>,

    /// Accepted for llama-server CLI compatibility (ignored)
    #[arg(long, hide = true)]
    _flash_attn: bool,

    /// Accepted for llama-server CLI compatibility (ignored — not applicable to MLX)
    #[arg(long, hide = true)]
    _mlock: bool,

    /// Accepted for llama-server CLI compatibility (ignored — not applicable to MLX)
    #[arg(long = "no-mmap", hide = true)]
    _no_mmap: bool,

    /// Accepted for llama-server CLI compatibility (ignored — mlxcel handles batching internally)
    #[arg(long, hide = true)]
    _cont_batching: bool,

    /// Decode storage backend for continuous batching.
    ///
    /// Accepted values: `auto`, `dense`, `paged`. When omitted, the server
    /// uses `MLXCEL_SERVER_DECODE_STORAGE` if set, otherwise automatic
    /// selection.
    #[arg(long = "decode-storage-backend", value_name = "BACKEND")]
    decode_storage_backend: Option<mlxcel::server::DecodeStorageBackend>,

    /// Maximum number of cached post-projection image features per loaded VLM.
    ///
    /// Multi-turn conversations that revisit the same image can reuse cached
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
    #[arg(long = "enable-elastic-pp", default_value_t = false)]
    enable_elastic_pp: bool,

    /// Drain timeout (seconds) for elastic repartitioning.
    #[arg(
        long = "elastic-pp-drain-timeout",
        default_value_t = 120,
        value_name = "SECONDS"
    )]
    elastic_pp_drain_timeout: u64,

    /// Memory-pressure trigger fraction for elastic repartitioning.
    #[arg(
        long = "elastic-pp-pressure-fraction",
        default_value_t = 0.92,
        value_name = "FRACTION"
    )]
    elastic_pp_pressure_fraction: f64,

    /// Cool-down (seconds) between memory-pressure repartition triggers.
    #[arg(
        long = "elastic-pp-cool-down",
        default_value_t = 30,
        value_name = "SECONDS"
    )]
    elastic_pp_cool_down: u64,

    /// Enable `/metrics` and advertise this port as the scrape target.
    #[arg(long = "metrics-port", value_name = "PORT")]
    metrics_port: Option<u16>,

    /// Write chrome-tracing JSON for pipeline scheduler actions.
    #[arg(long = "debug-pp-trace", value_name = "PATH")]
    debug_pp_trace: Option<PathBuf>,

    /// Run a pre-load memory estimate; refuse to start when the model will not fit.
    ///
    /// Computes weights + KV cache + runtime activation headroom and
    /// compares the total against available unified memory before the
    /// server begins loading. When the total exceeds the available
    /// budget the process exits with a non-zero status and a clear
    /// over-budget message. Use `--force` (alias `--no-memory-check`)
    /// to bypass.
    #[arg(long, default_value_t = false)]
    estimate_memory: bool,

    /// Bypass the `--estimate-memory` preflight abort.
    ///
    /// When `--estimate-memory` would refuse to start the server
    /// because total > available, `--force` (or its alias
    /// `--no-memory-check`) downgrades the abort to a warning and
    /// continues. No-op when `--estimate-memory` is not set.
    #[arg(long = "force", alias = "no-memory-check", default_value_t = false)]
    force_memory: bool,

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
    /// identical help text and parsing. The `--draft-model` /
    /// `--draft-max` flags stay above on this struct because they have
    /// different naming on `mlxcel-server` (`--model-draft`) for
    /// llama-server CLI compatibility.
    #[command(flatten)]
    speculative: SpeculativeArgs,

    /// Language-bias options for server-wide output
    /// steering. Mirrors the same flags exposed on the `generate` subcommand.
    ///
    /// The `--lang-bias` flag also reads from the `LLAMA_ARG_LANG_BIAS` env var
    /// when running as a server. CLI flag takes precedence.
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
    /// Also honors `LLAMA_ARG_REASONING_BUDGET`; CLI flag wins on conflict.
    /// Per-request `thinking_budget_tokens` / `thinking_token_budget` /
    /// `thinking_budget` on `/v1/chat/completions` or `/completion` overrides
    /// this value.
    #[arg(long = "reasoning-budget", default_value_t = -1, value_name = "N")]
    reasoning_budget: i32,

    /// Default chat-template kwargs (JSON object).
    ///
    /// Forwarded verbatim as Jinja template kwargs when rendering chat
    /// conversations. Matches llama.cpp's `--chat-template-kwargs` shape so
    /// a client switching from llama-server needs no request changes.
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
    /// Malformed JSON causes the server to refuse to start with a clear
    /// error. Non-thinking models silently ignore `preserve_thinking`.
    ///
    /// Note: quality benefits of `preserve_thinking` are only validated on
    /// Qwen3.6. Qwen3 / Qwen3.5 accept the flag but were not trained on
    /// multi-turn thinking traces.
    #[arg(long = "chat-template-kwargs", value_name = "JSON")]
    chat_template_kwargs: Option<String>,

    // cross-request prompt-prefix KV cache knobs.
    /// Enable or disable the cross-request prompt-prefix KV cache (default: true).
    ///
    /// When disabled, the server performs no prefix-match lookup and no memory
    /// is reserved for the cache. Disabling eliminates lock contention and
    /// matcher overhead.
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_ENABLED` (on/off/true/false/1/0) and
    /// the llama.cpp-compat alias `LLAMA_ARG_CACHE_REUSE` when the CLI flag
    /// is absent. CLI flag takes precedence over env vars.
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
    /// Also reads `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES` when the CLI flag is absent.
    #[arg(long = "prompt-cache-capacity-bytes", value_name = "BYTES")]
    prompt_cache_capacity_bytes: Option<usize>,

    /// Maximum number of live entries in the prompt-prefix KV cache (default: 1024).
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_MAX_ENTRIES` when the CLI flag is absent.
    #[arg(long = "prompt-cache-max-entries", value_name = "N")]
    prompt_cache_max_entries: Option<usize>,

    /// Time-to-live for a prompt-cache entry in seconds (default: 3600).
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_TTL` when the CLI flag is absent.
    #[arg(long = "prompt-cache-ttl", value_name = "SECONDS")]
    prompt_cache_ttl: Option<u64>,

    /// Minimum prompt-prefix length (tokens) required before caching (default: 32).
    ///
    /// Also reads `MLXCEL_PROMPT_CACHE_MIN_PREFIX` when the CLI flag is absent.
    #[arg(long = "prompt-cache-min-prefix", value_name = "N")]
    prompt_cache_min_prefix: Option<usize>,

    // Automatic Prefix Caching (APC) knobs.
    /// Enable Automatic Prefix Caching (APC) with block-granularity hash chains
    /// (default: true). Disable with `--apc-enabled=false`.
    ///
    /// APC layers on top of the existing prompt-prefix cache to enable
    /// finer-grained KV reuse: without it, a stored prefix is reusable only
    /// when it is fully contained in the new request, so requests that share
    /// a long system prompt but diverge afterwards never reuse KV. With the
    /// non-consuming paged adoption a partial match shares the prefix blocks
    /// without copying or destroying the stored entry, so APC is on by
    /// default. When enabled on a hybrid SSM/attention model (jamba, mamba,
    /// mamba2, nemotron_h, gated_delta, kimi_linear, qwen3_next), APC is
    /// automatically disabled at runtime since SSM state cannot be
    /// decomposed into hashable blocks.
    ///
    /// Also reads `APC_ENABLED` (parity with upstream `mlx-vlm`).
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
    /// to the pre-surgery baseline — every served request runs against
    /// unmodified weights, so the server's response stream is unchanged.
    ///
    /// Also reads `MLXCEL_SURGERY`; CLI flag wins on conflict.
    ///
    /// Example:
    ///
    ///     mlxcel serve -m models/foo --surgery surgery.yaml
    ///
    /// The supported surgery operations are summarised in the project README.
    #[cfg(feature = "surgery")]
    #[arg(long = "surgery", value_name = "FILE", env = "MLXCEL_SURGERY")]
    pub(crate) surgery: Option<PathBuf>,

    // Block-diffusion serve-level flag group (--max-denoising-steps,
    // --diffusion-sampler, --diffusion-threshold). Only affects diffusion
    // models such as DiffusionGemma; autoregressive models ignore it. The flags
    // set the per-request diffusion defaults for the single-stream worker loop.
    #[command(flatten)]
    pub(crate) diffusion: DiffusionServeOptions,
}

/// Serve-level block-diffusion options.
///
/// A focused subset of [`DiffusionCliOptions`] exposing only the knobs that
/// the single-stream diffusion serving loop honors per request
/// (`--diffusion-sampler`, `--diffusion-threshold`, `--max-denoising-steps`).
/// They only affect diffusion models (e.g. DiffusionGemma); ordinary
/// autoregressive models ignore them. Canvas-shaping flags from the generate
/// CLI are intentionally not exposed in serve mode.
#[derive(Args, Debug)]
#[command(next_help_heading = "Diffusion Options")]
pub(crate) struct DiffusionServeOptions {
    /// Maximum denoising steps per canvas block (diffusion models only;
    /// default: the checkpoint's generation_config, typically 48)
    #[arg(long = "max-denoising-steps", value_name = "N")]
    pub(crate) max_denoising_steps: Option<usize>,

    /// Per-step acceptance sampler for diffusion models
    #[arg(
        long = "diffusion-sampler",
        value_name = "SAMPLER",
        default_value = "entropy-bound",
        value_parser = ["entropy-bound", "confidence-threshold"]
    )]
    pub(crate) diffusion_sampler: String,

    /// Confidence threshold for `--diffusion-sampler confidence-threshold`
    /// (diffusion models only)
    #[arg(
        long = "diffusion-threshold",
        value_name = "FLOAT",
        default_value_t = 0.9,
        value_parser = parse_unit_interval
    )]
    pub(crate) diffusion_threshold: f32,
}

// Manual `Default` kept in lock-step with the `#[arg(default_value*)]`
// attributes above, same contract as the other serve option groups.
impl Default for DiffusionServeOptions {
    fn default() -> Self {
        Self {
            max_denoising_steps: None,
            diffusion_sampler: "entropy-bound".to_string(),
            diffusion_threshold: 0.9,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Default the CUDA kernel JIT cache to a persistent, MLX-pin-scoped dir so
    // the first-run kernel compilation is paid once per machine, not every boot.
    mlxcel_core::ensure_persistent_ptx_cache();

    match cli.command {
        Commands::Run(args) => commands::run_run(args),
        Commands::Generate(args) => commands::run_generate(args),
        Commands::Serve(args) => commands::run_serve(args),
        Commands::List(args) => commands::run_list_local(args.models_dir.as_deref(), &args),
        Commands::Arch(_) => {
            print_supported_models();
            Ok(())
        }
        Commands::Inspect(args) => commands::run_inspect(args),
        Commands::Download(args) => commands::run_download(args),
        Commands::Detect(args) => commands::run_detect(args),
        Commands::Rm(args) => commands::run_remove(
            &args.repo_id,
            args.yes,
            args.revision.as_deref(),
            args.models_dir.as_deref(),
        ),
    }
}

/// Preferred family ordering for the `mlxcel arch` output. Any family that
/// appears in `ModelType::family()` but is missing from this slice is
/// appended after these, sorted alphabetically — so the output remains
/// exhaustive even if a new family is introduced without updating this
/// table. The same drift is also caught at test time by
/// `family_order_is_exhaustive` in `main_tests.rs`, which makes the
/// missing-family case a CI failure rather than a silently-reordered
/// section.
const FAMILY_ORDER: &[&str] = &[
    "Llama",
    "Qwen",
    "Gemma",
    "Mistral",
    "Phi",
    "DeepSeek",
    "Cohere",
    "InternLM",
    "GLM",
    "ERNIE",
    "Hunyuan",
    "Granite",
    "ExaOne",
    "Solar",
    "OLMo",
    "Nemotron",
    "MoE (other)",
    "Mamba / SSM",
    "Hybrid",
    "Falcon",
    "LFM2",
    "PLaMo",
    "RWKV",
    "Specialized",
    "Llama VLM",
    "Qwen VLM",
    "Gemma VLM",
    "Mistral VLM",
    "Phi VLM",
    "Cohere VLM",
    "Nemotron VLM",
    "Other VLM",
];

fn print_supported_models() {
    let mut out = String::new();
    // Writing to a `String` cannot fail — `fmt::Write` for `String` is
    // infallible — so `expect` is appropriate here.
    write_supported_models(&mut out).expect("writing to a String cannot fail");
    print!("{out}");
}

/// Render the human-readable `mlxcel arch` output into `out`.
///
/// Separated from [`print_supported_models`] so unit tests can capture the
/// exact bytes that the CLI would print without spawning a subprocess.
fn write_supported_models<W: std::fmt::Write>(out: &mut W) -> std::fmt::Result {
    use mlxcel::models::ALL_MODEL_TYPES;

    writeln!(
        out,
        "Supported Model Architectures ({}):",
        ALL_MODEL_TYPES.len()
    )?;
    writeln!(out)?;

    // Bucket variants by family in declaration order. Members within a
    // family stay in their `ALL_MODEL_TYPES` order (which mirrors the
    // enum), so the output is deterministic across builds.
    let mut buckets: Vec<(&'static str, Vec<&'static str>)> = Vec::new();
    for &mt in ALL_MODEL_TYPES {
        let (display, family) = mt.metadata();
        if let Some(existing) = buckets.iter_mut().find(|(f, _)| *f == family) {
            existing.1.push(display);
        } else {
            buckets.push((family, vec![display]));
        }
    }

    // Sort by FAMILY_ORDER index, with unknown families appended
    // alphabetically. This keeps the rendered output stable while still
    // tolerating a new family that the order table has not learned about
    // yet (the test `family_order_is_exhaustive` flags such drift).
    buckets.sort_by(|a, b| {
        let ai = FAMILY_ORDER.iter().position(|&f| f == a.0);
        let bi = FAMILY_ORDER.iter().position(|&f| f == b.0);
        match (ai, bi) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.0.cmp(b.0),
        }
    });

    for (family, members) in &buckets {
        writeln!(out, "{family}:")?;
        for name in members {
            writeln!(out, "  - {name}")?;
        }
        writeln!(out)?;
    }

    Ok(())
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
