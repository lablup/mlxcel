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

//! llama-server b10621 logging, introspection, and built-in model preset
//! options (issue #1448, epic #1431).
//!
//! Three groups share this module because they share one rule: each option is
//! either implemented for real or refused out loud, and nothing is accepted
//! and ignored.
//!
//! # Logging format and verbosity
//!
//! `--log-colors`, `--log-prefix` / `--no-log-prefix`, `--log-timestamps` /
//! `--no-log-timestamps` and `--verbosity` / `--log-verbosity` are visible
//! mlxcel options that drive the real subscriber. Their precedence, the
//! destination rules for `--log-file` / `--log-disable`, and the log-file
//! permission policy live in [`crate::server::logging`].
//!
//! # Introspection
//!
//! `--cache-list` lists the checkpoints in mlxcel's model store in b10621's
//! output format and exits; `--completion-bash` prints a source-able bash
//! completion script built from the live clap surface. Both run before any
//! model is resolved, so neither needs `-m`.
//!
//! # Presets
//!
//! b10621's twelve `--*-default` / `--*-spec` flags each rewrite
//! `params.model.hf_repo` to a **GGUF** repository under `ggml-org` and then
//! overwrite the port, context, batch, parallelism, and sampling block.
//! mlxcel loads MLX SafeTensors and cannot read GGUF at all, so there is no
//! honest way to accept one of these: mapping only the checkpoint would
//! silently drop the parameter block, and mapping the parameter block would
//! silently serve a different quantization than the operator named.
//!
//! Every preset is therefore hidden, accepted by the parser, and refused at
//! startup with the exact `mlxcel-server` command line that reaches the
//! nearest MLX checkpoint. The refusal is the deliverable: an operator who
//! copied a llama-server invocation learns which mlx-community repository to
//! use instead of watching the server come up serving something else.
//!
//! `--spec-default` is the odd one out: it configures no model, it enables
//! b10621's n-gram-modulo drafter. mlxcel's drafters are `mtp` and `dflash`,
//! both checkpoint-backed, so the flag is refused with a pointer at
//! `--draft-kind`.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference: <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>

use std::path::{Path, PathBuf};

use clap::Args;

use crate::cli::ggml_compat_args::env_bool_pair;
use crate::server::logging::{LogColors, LogFormatOptions};

/// One b10621 preset: its flag, what upstream serves, and the mlxcel
/// equivalent an operator should use instead.
///
/// `mlxcel_repo` names a checkpoint that exists in the `mlx-community`
/// HuggingFace organization and that mlxcel's `-m` resolver accepts directly.
/// `extra` carries the non-model half of the upstream preset that an operator
/// would otherwise lose silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresetMapping {
    /// The b10621 flag, with dashes.
    pub flag: &'static str,
    /// The GGUF repository upstream would download.
    pub upstream_repo: &'static str,
    /// The nearest MLX checkpoint, or `None` when the preset configures no
    /// model at all (`--spec-default`).
    pub mlxcel_repo: Option<&'static str>,
    /// Draft checkpoint for the two speculative presets.
    pub mlxcel_draft_repo: Option<&'static str>,
    /// The rest of the upstream preset, rendered as mlxcel flags.
    pub extra: &'static str,
}

/// Every b10621 preset, in help order.
///
/// The MLX targets were checked against the HuggingFace model API while
/// writing #1448; each one is a published `mlx-community` conversion of the
/// same base model upstream ships as GGUF. They are named in diagnostics
/// only, never resolved implicitly, so a repository that later disappears
/// degrades to a stale suggestion rather than a failed startup.
pub const PRESETS: &[PresetMapping] = &[
    PresetMapping {
        flag: "--embd-gemma-default",
        upstream_repo: "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
        mlxcel_repo: Some("mlx-community/embeddinggemma-300m-4bit"),
        mlxcel_draft_repo: None,
        extra: "--embedding --port 8011",
    },
    PresetMapping {
        flag: "--fim-qwen-1.5b-default",
        upstream_repo: "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF",
        mlxcel_repo: Some("mlx-community/Qwen2.5-Coder-1.5B-8bit"),
        mlxcel_draft_repo: None,
        extra: "--port 8012",
    },
    PresetMapping {
        flag: "--fim-qwen-3b-default",
        upstream_repo: "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF",
        mlxcel_repo: Some("mlx-community/Qwen2.5-Coder-3B-8bit"),
        mlxcel_draft_repo: None,
        extra: "--port 8012",
    },
    PresetMapping {
        flag: "--fim-qwen-7b-default",
        upstream_repo: "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF",
        mlxcel_repo: Some("mlx-community/Qwen2.5-Coder-7B-8bit"),
        mlxcel_draft_repo: None,
        extra: "--port 8012",
    },
    PresetMapping {
        flag: "--fim-qwen-7b-spec",
        upstream_repo: "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF",
        mlxcel_repo: Some("mlx-community/Qwen2.5-Coder-7B-8bit"),
        mlxcel_draft_repo: Some("mlx-community/Qwen2.5-Coder-0.5B-8bit"),
        extra: "--port 8012",
    },
    PresetMapping {
        flag: "--fim-qwen-14b-spec",
        upstream_repo: "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF",
        mlxcel_repo: Some("mlx-community/Qwen2.5-Coder-14B-8bit"),
        mlxcel_draft_repo: Some("mlx-community/Qwen2.5-Coder-0.5B-8bit"),
        extra: "--port 8012",
    },
    PresetMapping {
        flag: "--fim-qwen-30b-default",
        upstream_repo: "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF",
        mlxcel_repo: Some("mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"),
        mlxcel_draft_repo: None,
        extra: "--port 8012",
    },
    PresetMapping {
        flag: "--gpt-oss-20b-default",
        upstream_repo: "ggml-org/gpt-oss-20b-GGUF",
        mlxcel_repo: Some("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        mlxcel_draft_repo: None,
        extra: "--jinja --port 8013 --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.01",
    },
    PresetMapping {
        flag: "--gpt-oss-120b-default",
        upstream_repo: "ggml-org/gpt-oss-120b-GGUF",
        mlxcel_repo: Some("mlx-community/gpt-oss-120b-MXFP4-Q8"),
        mlxcel_draft_repo: None,
        extra: "--jinja --port 8013 --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.01",
    },
    PresetMapping {
        flag: "--vision-gemma-4b-default",
        upstream_repo: "ggml-org/gemma-3-4b-it-qat-GGUF",
        mlxcel_repo: Some("mlx-community/gemma-3-4b-it-qat-4bit"),
        mlxcel_draft_repo: None,
        extra: "--jinja --port 8014",
    },
    PresetMapping {
        flag: "--vision-gemma-12b-default",
        upstream_repo: "ggml-org/gemma-3-12b-it-qat-GGUF",
        mlxcel_repo: Some("mlx-community/gemma-3-12b-it-qat-4bit"),
        mlxcel_draft_repo: None,
        extra: "--jinja --port 8014",
    },
    PresetMapping {
        flag: "--spec-default",
        upstream_repo: "(no model; enables the n-gram-modulo drafter)",
        mlxcel_repo: None,
        mlxcel_draft_repo: None,
        extra: "",
    },
];

/// A b10621 option this build accepts for compatibility but refuses to honor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggingCompatRejection {
    /// The option spelling that was refused.
    pub option: &'static str,
    /// Operator-facing explanation, ending with what to do instead.
    pub detail: String,
}

impl std::fmt::Display for LoggingCompatRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.option, self.detail)
    }
}

/// An option that must run and exit before anything else happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyAction {
    /// `--cache-list` / `-cl`.
    CacheList,
    /// `--completion-bash`.
    CompletionBash,
}

/// llama-server b10621 logging, introspection, and preset options.
///
/// Flattened into both server binaries so the two surfaces cannot drift; see
/// `tests/cli_help_consistency.rs`.
#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "Logging and Introspection (llama-server compatibility)")]
pub struct LoggingCompatArgs {
    /// Colored log output: `on`, `off`, or `auto` (default `auto`).
    ///
    /// `auto` colors only when the log sink is a terminal, so `--log-file`
    /// never receives ANSI escapes.
    #[arg(
        long = "log-colors",
        env = "LLAMA_ARG_LOG_COLORS",
        value_name = "on|off|auto",
        default_value = "auto"
    )]
    pub log_colors: String,

    /// Include the level tag in each log line (the default).
    #[arg(
        long = "log-prefix",
        overrides_with = "no_log_prefix",
        action = clap::ArgAction::SetTrue
    )]
    pub log_prefix: bool,

    /// Drop the level tag from each log line.
    #[arg(
        long = "no-log-prefix",
        overrides_with = "log_prefix",
        action = clap::ArgAction::SetTrue
    )]
    pub no_log_prefix: bool,

    /// Include a timestamp in each log line (the default).
    #[arg(
        long = "log-timestamps",
        overrides_with = "no_log_timestamps",
        action = clap::ArgAction::SetTrue
    )]
    pub log_timestamps: bool,

    /// Drop the timestamp from each log line.
    #[arg(
        long = "no-log-timestamps",
        overrides_with = "log_timestamps",
        action = clap::ArgAction::SetTrue
    )]
    pub no_log_timestamps: bool,

    /// Verbosity threshold: 0 error, 2 warning, 3 info, 4 debug, 5 trace.
    ///
    /// Messages above the threshold are dropped, so a larger number means
    /// more output. `-lv` is accepted as the llama-server short spelling.
    /// A command-line value beats `RUST_LOG`; `LLAMA_ARG_LOG_VERBOSITY` does
    /// not.
    #[arg(
        long = "verbosity",
        visible_alias = "log-verbosity",
        env = "LLAMA_ARG_LOG_VERBOSITY",
        value_name = "N",
        default_value = "3",
        allow_negative_numbers = true
    )]
    pub verbosity: i32,

    /// llama-server `--log-prompts-dir`. Accepted by the parser and refused
    /// at startup; mlxcel does not write request prompts to disk.
    #[arg(long = "log-prompts-dir", value_name = "PATH", hide = true)]
    pub log_prompts_dir: Option<PathBuf>,

    /// List the checkpoints in mlxcel's model store and exit.
    #[arg(long = "cache-list", action = clap::ArgAction::SetTrue)]
    pub cache_list: bool,

    /// Print a source-able bash completion script and exit.
    #[arg(long = "completion-bash", action = clap::ArgAction::SetTrue)]
    pub completion_bash: bool,

    /// b10621 built-in EmbeddingGemma preset. Refused at startup.
    #[arg(long = "embd-gemma-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub embd_gemma_default: bool,

    /// b10621 built-in Qwen 2.5 Coder 1.5B preset. Refused at startup.
    #[arg(long = "fim-qwen-1.5b-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub fim_qwen_1_5b_default: bool,

    /// b10621 built-in Qwen 2.5 Coder 3B preset. Refused at startup.
    #[arg(long = "fim-qwen-3b-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub fim_qwen_3b_default: bool,

    /// b10621 built-in Qwen 2.5 Coder 7B preset. Refused at startup.
    #[arg(long = "fim-qwen-7b-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub fim_qwen_7b_default: bool,

    /// b10621 built-in Qwen 2.5 Coder 7B + 0.5B draft preset. Refused at startup.
    #[arg(long = "fim-qwen-7b-spec", action = clap::ArgAction::SetTrue, hide = true)]
    pub fim_qwen_7b_spec: bool,

    /// b10621 built-in Qwen 2.5 Coder 14B + 0.5B draft preset. Refused at startup.
    #[arg(long = "fim-qwen-14b-spec", action = clap::ArgAction::SetTrue, hide = true)]
    pub fim_qwen_14b_spec: bool,

    /// b10621 built-in Qwen 3 Coder 30B A3B preset. Refused at startup.
    #[arg(long = "fim-qwen-30b-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub fim_qwen_30b_default: bool,

    /// b10621 built-in gpt-oss-20b preset. Refused at startup.
    #[arg(long = "gpt-oss-20b-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub gpt_oss_20b_default: bool,

    /// b10621 built-in gpt-oss-120b preset. Refused at startup.
    #[arg(long = "gpt-oss-120b-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub gpt_oss_120b_default: bool,

    /// b10621 built-in Gemma 3 4B QAT vision preset. Refused at startup.
    #[arg(long = "vision-gemma-4b-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub vision_gemma_4b_default: bool,

    /// b10621 built-in Gemma 3 12B QAT vision preset. Refused at startup.
    #[arg(long = "vision-gemma-12b-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub vision_gemma_12b_default: bool,

    /// b10621 default speculative configuration. Refused at startup.
    #[arg(long = "spec-default", action = clap::ArgAction::SetTrue, hide = true)]
    pub spec_default: bool,
}

impl Default for LoggingCompatArgs {
    /// The parse-time defaults, spelled out rather than derived.
    ///
    /// A derived `Default` would leave `log_colors` empty and `verbosity` at
    /// zero, neither of which any command line can produce: clap fills both
    /// from `default_value` before this struct is ever handed on. Spelling
    /// them here keeps a hand-built value (a test, a future embedder)
    /// resolvable instead of failing `LogColors::parse` on an empty string.
    fn default() -> Self {
        Self {
            log_colors: "auto".to_owned(),
            log_prefix: false,
            no_log_prefix: false,
            log_timestamps: false,
            no_log_timestamps: false,
            verbosity: crate::server::logging::DEFAULT_VERBOSITY,
            log_prompts_dir: None,
            cache_list: false,
            completion_bash: false,
            embd_gemma_default: false,
            fim_qwen_1_5b_default: false,
            fim_qwen_3b_default: false,
            fim_qwen_7b_default: false,
            fim_qwen_7b_spec: false,
            fim_qwen_14b_spec: false,
            fim_qwen_30b_default: false,
            gpt_oss_20b_default: false,
            gpt_oss_120b_default: false,
            vision_gemma_4b_default: false,
            vision_gemma_12b_default: false,
            spec_default: false,
        }
    }
}

impl LoggingCompatArgs {
    /// Which preset flags were passed, in help order.
    fn requested_presets(&self) -> Vec<&'static PresetMapping> {
        let set = [
            self.embd_gemma_default,
            self.fim_qwen_1_5b_default,
            self.fim_qwen_3b_default,
            self.fim_qwen_7b_default,
            self.fim_qwen_7b_spec,
            self.fim_qwen_14b_spec,
            self.fim_qwen_30b_default,
            self.gpt_oss_20b_default,
            self.gpt_oss_120b_default,
            self.vision_gemma_4b_default,
            self.vision_gemma_12b_default,
            self.spec_default,
        ];
        debug_assert_eq!(set.len(), PRESETS.len());
        PRESETS
            .iter()
            .zip(set)
            .filter_map(|(preset, requested)| requested.then_some(preset))
            .collect()
    }

    /// The `--cache-list` / `--completion-bash` action to run before the
    /// model is resolved, if any. `--cache-list` wins when both are given,
    /// matching b10621, whose `--cache-list` handler calls `exit(0)` from
    /// inside the parser.
    #[must_use]
    pub fn early_action(&self) -> Option<EarlyAction> {
        if self.cache_list {
            return Some(EarlyAction::CacheList);
        }
        if self.completion_bash {
            return Some(EarlyAction::CompletionBash);
        }
        None
    }

    /// Refuse every option in this group that mlxcel accepts for
    /// compatibility but will not honor.
    ///
    /// Called before the model is resolved, so a copied llama-server command
    /// line fails in under a second rather than after a multi-gigabyte
    /// download.
    pub fn ensure_supported(&self) -> Result<(), LoggingCompatRejection> {
        // Value domains first, so an unknown `--log-colors` or a malformed
        // `LLAMA_ARG_LOG_PREFIX` stops startup at the same early point a
        // refused option does. `resolve_format` is pure apart from reading the
        // environment, and the caller re-derives it later; the flag argument
        // is irrelevant to validation.
        self.resolve_format(false)?;

        if let Some(dir) = self.log_prompts_dir.as_deref() {
            return Err(LoggingCompatRejection {
                option: "--log-prompts-dir",
                detail: format!(
                    "mlxcel does not write request prompts to disk, so {} would stay empty \
                     while the operator believed prompts were being captured. Prompt text is \
                     user data and a plaintext copy of it on the log volume is a disclosure \
                     surface mlxcel declines to create. For request-level debugging use \
                     --log-file with --verbosity 4, which records request metadata \
                     (route, slot, token counts, timings) and no prompt bodies.",
                    dir.display()
                ),
            });
        }

        let presets = self.requested_presets();
        let Some(preset) = presets.first() else {
            return Ok(());
        };
        Err(LoggingCompatRejection {
            option: preset.flag,
            detail: preset_detail(preset),
        })
    }

    /// Resolve the format half of this group into the value the server
    /// startup path carries.
    ///
    /// `verbosity_from_cli` must be true when `--verbosity` (or one of its
    /// spellings) appeared on the command line rather than arriving through
    /// `LLAMA_ARG_LOG_VERBOSITY` or the compiled-in default; the caller knows
    /// that and this module does not. See [`crate::server::logging`] for why
    /// the distinction matters.
    pub fn resolve_format(
        &self,
        verbosity_from_cli: bool,
    ) -> Result<LogFormatOptions, LoggingCompatRejection> {
        let colors =
            LogColors::parse(&self.log_colors).map_err(|detail| LoggingCompatRejection {
                option: "--log-colors",
                detail,
            })?;

        let prefix = resolve_bool_pair(
            "LLAMA_ARG_LOG_PREFIX",
            self.log_prefix,
            self.no_log_prefix,
            true,
        )
        .map_err(|raw| LoggingCompatRejection {
            option: "--log-prefix",
            detail: format!(
                "LLAMA_ARG_LOG_PREFIX={raw:?} is not a boolean; expected one of \
                 on/enabled/true/1 or off/disabled/false/0"
            ),
        })?;
        let timestamps = resolve_bool_pair(
            "LLAMA_ARG_LOG_TIMESTAMPS",
            self.log_timestamps,
            self.no_log_timestamps,
            true,
        )
        .map_err(|raw| LoggingCompatRejection {
            option: "--log-timestamps",
            detail: format!(
                "LLAMA_ARG_LOG_TIMESTAMPS={raw:?} is not a boolean; expected one of \
                 on/enabled/true/1 or off/disabled/false/0"
            ),
        })?;

        let env_verbosity = (!verbosity_from_cli
            && std::env::var("LLAMA_ARG_LOG_VERBOSITY").is_ok())
        .then_some(self.verbosity);

        Ok(LogFormatOptions {
            colors,
            prefix,
            timestamps,
            cli_verbosity: verbosity_from_cli.then_some(self.verbosity),
            env_verbosity,
        })
    }
}

/// True when `--verbosity` (any of its three spellings) appeared on the
/// command line rather than arriving through `LLAMA_ARG_LOG_VERBOSITY` or the
/// compiled-in default.
///
/// The raw argv is read, not the rewritten one, so the llama.cpp short form
/// `-lv` is checked explicitly: `crate::cli::llama_short_flags` rewrites it to
/// `--verbosity` before clap parses, and that rewrite is invisible here.
#[must_use]
pub fn verbosity_was_set_on_cli() -> bool {
    if crate::server::long_cli_flag_was_set("verbosity")
        || crate::server::long_cli_flag_was_set("log-verbosity")
    {
        return true;
    }
    std::env::args_os().any(|arg| arg == "-lv")
}

/// Render the output of an [`EarlyAction`], to be printed before the process
/// exits with status 0.
///
/// `exe` is the command word a completion registers against and `subject` the
/// invocation whose options it lists; see
/// [`crate::cli::completion::bash_completion_script`]. `models_dir` is the
/// inline `--models-dir <path>` flag, so `--cache-list` reports the store the
/// same invocation would load from.
#[must_use]
pub fn render_early_action(
    action: EarlyAction,
    exe: &str,
    subject: &str,
    cmd: &mut clap::Command,
    models_dir: Option<&Path>,
) -> String {
    match action {
        EarlyAction::CacheList => crate::cli::cache_list::render_store_cache_list(models_dir),
        EarlyAction::CompletionBash => {
            crate::cli::completion::bash_completion_script(exe, subject, cmd)
        }
    }
}

/// Register the credentials a server run holds so no log sink can ever
/// contain them (issue #1448 acceptance criterion).
///
/// `api_keys` are the `--api-key` values, `api_key_files` the
/// `--api-key-file` paths whose contents are keys too, and `hf_token` the
/// `--hf-token` value. `HF_TOKEN` and `LLAMA_API_KEY` are read from the
/// environment here rather than at the two call sites, so a binary that
/// forgets one of them still gets the environment half.
///
/// Call before `crate::server::logging::install`; a key registered later
/// cannot redact a line already written.
pub fn register_credentials_for_redaction(
    api_keys: &[String],
    api_key_files: &[PathBuf],
    hf_token: Option<&str>,
) {
    use crate::server::logging::register_log_secret;
    for key in api_keys {
        register_log_secret(key);
    }
    for path in api_key_files {
        let Ok(contents) = std::fs::read_to_string(path) else {
            continue;
        };
        for line in contents.lines() {
            register_log_secret(line);
        }
    }
    if let Some(token) = hf_token {
        register_log_secret(token);
    }
    for var in ["HF_TOKEN", "LLAMA_API_KEY", "MLXCEL_API_KEY"] {
        if let Ok(value) = std::env::var(var) {
            register_log_secret(&value);
        }
    }
    // `LLAMA_ARG_API_KEY_FILE` names a file whose contents are keys. It is
    // folded into `--api-key-file` by `env_fallback_api_key_files`, but that
    // runs later than this call on the `mlxcel serve` path, so read it here as
    // well rather than leaving one binary's environment half uncovered.
    if let Ok(path) = std::env::var("LLAMA_ARG_API_KEY_FILE")
        && let Ok(contents) = std::fs::read_to_string(&path)
    {
        for line in contents.lines() {
            register_log_secret(line);
        }
    }
}

/// Render the operator-facing half of a preset refusal.
fn preset_detail(preset: &PresetMapping) -> String {
    let Some(repo) = preset.mlxcel_repo else {
        return "b10621's default speculative configuration is its n-gram-modulo drafter, \
                which predicts tokens from the context itself. mlxcel's drafters are \
                checkpoint-backed: pass --draft-kind mtp with an MTP-capable target, or \
                --draft-kind dflash --model-draft <path-or-repo-id> with a small draft \
                model. See docs/llama-server-compat.md."
            .to_owned();
    };
    let draft = preset
        .mlxcel_draft_repo
        .map(|d| format!(" --model-draft {d} --draft-kind dflash"))
        .unwrap_or_default();
    format!(
        "this preset downloads the GGUF repository {upstream}, which mlxcel cannot read: \
         mlxcel serves MLX SafeTensors. Accepting the flag would either drop the preset's \
         port, context, batch and sampling settings or serve a different quantization than \
         you named, so it is refused instead. The nearest MLX checkpoint is {repo}; fetch \
         and serve it with:\n    mlxcel download {repo}\n    mlxcel-server -m {repo}{draft} \
         {extra}",
        upstream = preset.upstream_repo,
        extra = preset.extra,
    )
}

/// Resolve a b10621 `--x` / `--no-x` pair against its environment binding.
///
/// The command line wins; then `LLAMA_ARG_*` / `LLAMA_ARG_NO_*` read with
/// b10621's own truthiness rules (see
/// [`crate::cli::ggml_compat_args::env_bool_pair`]); then `default_value`.
/// Returns `Err(raw)` for an environment value b10621's `parse_bool_value`
/// would throw on, so a typo fails loudly instead of picking a side.
fn resolve_bool_pair(
    var: &str,
    enabled: bool,
    disabled: bool,
    default_value: bool,
) -> Result<bool, String> {
    if enabled {
        return Ok(true);
    }
    if disabled {
        return Ok(false);
    }
    match env_bool_pair(var) {
        Some(Ok(value)) => Ok(value),
        Some(Err(raw)) => Err(raw),
        None => Ok(default_value),
    }
}

#[cfg(test)]
#[path = "logging_compat_args_tests.rs"]
mod tests;
