// Copyright 2025-2026 Lablup Inc.
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

//! `mlxcel run` verb (epic #92, issue #95): the ollama-style entry point.
//!
//! Since epic #2166 Phase 5 (issue #2173, ADR 0007 "CLI front end") `run` is
//! an in-process client of the model server, the way llama-cli is a client of
//! llama-server but without the loopback socket: it starts the server's model
//! worker in this process at one slot
//! ([`mlxcel::server::in_process::InProcessServer`]) and submits chat requests
//! through the `/v1/chat/completions` request path. That path renders the
//! chat template, builds the worker options, splits reasoning from content
//! with the server's `StreamFilter`, parses tool calls and applies stop
//! strings, so `run` keeps only terminal I/O and REPL commands:
//!
//! * **no `-p/--prompt`**: the interactive multi-turn chat REPL
//!   ([`crate::commands::run_chat`], issue #96);
//! * **with `-p`**: one chat turn, printed as it streams.
//!
//! The equivalence invariant replaces the old "same code as `generate`"
//! promise: for the same request, `mlxcel run -p` and `mlxcel-server` produce
//! the same tokens, because they run the same request path and the same
//! engine. `mlxcel-engine-parity` checks it (arm `e:run`, greedy, under
//! `MLXCEL_SDPA_DETERMINISTIC=1`). Every sampling flag maps onto the
//! `mlxcel-server` option of the same meaning
//! ([`crate::commands::cli_server`]); defaults follow ADR 0007's decision
//! table, with greedy kept as the CLI base sampler.
//!
//! One-shot modes the chat server does not serve stay on the `generate`
//! one-shot flow until Phase 6 (#2176): block-diffusion, Florence-2,
//! Nemotron-Parse and VoiceChat checkpoints, `--output-audio`,
//! `--layout-detections`, `--audio`, `--video`, `--profile`,
//! `--estimate-memory` and `--recommend-quant`. `--image` goes through the
//! server like a chat client's `image_url` part.
//!
//! ## Default-model fallback
//!
//! When no model argument is supplied, `run` falls back to [`DEFAULT_MODEL`].
//! The repo-id is auto-downloaded into the mlxcel global store on first use by
//! the shared resolver, so `mlxcel run` with no arguments works from any
//! directory.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use mlxcel::cli::speculative_args::SpeculativeArgs;
use mlxcel::server::chat_template::ChatMessage;
use mlxcel::server::in_process::InProcessServer;

use super::chat_transcript::{Turn, messages_json};
use mlxcel::cli::in_process_client::{
    CliServerSettings, ServerSamplingOptions, chat_request_body, completion_request_body,
};

use super::cli_server::{cli_flag_was_set, resolve_cli_kv_cache_mode, settings_from_flags};
use super::cli_turn::{TurnDisplay, TurnPrinter, run_cancellable};
use crate::{GenerateArgs, GenerationOptions, ModelOptions, SamplingOptions};

/// Default model used when `mlxcel run` is invoked without a model argument.
///
/// `gemma-4-e2b-it-4bit` is a small, instruction-tuned checkpoint that
/// downloads quickly and runs in a modest memory budget, so `mlxcel run` with
/// no arguments gives a usable model out of the box. Documented in the `run`
/// `--help` text and the project README.
pub(crate) const DEFAULT_MODEL: &str = "mlx-community/gemma-4-e2b-it-4bit";

/// Arguments for `mlxcel run`.
///
/// `run` takes a model (repo-id or local path) and either streams an
/// interactive chat (no `-p`) or prints a one-shot completion (`-p "..."`).
/// The model argument is **optional**: omitting it loads
/// [`DEFAULT_MODEL`]. Sampling and generation flags are the *same* clap groups
/// [`GenerateArgs`] flattens ([`GenerationOptions`] / [`SamplingOptions`]), so
/// `--help` and behavior stay in lock-step with `mlxcel generate` and no flag
/// is duplicated.
#[derive(Args, Debug)]
#[command(next_help_heading = "Run Options")]
pub(crate) struct RunArgs {
    /// Model to run: a local directory **or** a HuggingFace `owner/name`
    /// repo-id to auto-download (resolved exactly like `mlxcel generate -m`).
    ///
    /// Optional: when omitted, `mlxcel run` falls back to the default model
    /// `mlx-community/gemma-4-e2b-it-4bit` and auto-downloads it into the
    /// mlxcel store on first use. Given as a positional argument so
    /// `mlxcel run <repo-id>` reads like `ollama run`.
    /// A bare name without a slash (e.g. `Qwen3-4B-4bit`) is resolved as
    /// `mlx-community/<name>`; override the org with the `MLXCEL_DEFAULT_ORG`
    /// environment variable.
    #[arg(value_name = "MODEL_OR_REPO_ID")]
    pub(crate) model: Option<PathBuf>,

    /// Model-store root for resolving / downloading an `owner/name` repo-id.
    ///
    /// Sets the directory that directly holds snapshots, so a repo-id resolves
    /// to / downloads at `<PATH>/<owner>/<name>` (no extra `models/` subdir).
    /// Overrides the `MLXCEL_MODELS_DIR` environment variable. No effect when
    /// the model argument is already an existing local path.
    #[arg(long, value_name = "PATH")]
    pub(crate) models_dir: Option<PathBuf>,

    /// Repository revision (branch, tag, or commit hash). Defaults to `main`.
    ///
    /// Resolves the HuggingFace cache snapshot for that revision, and fetches
    /// that revision on a miss. The mlxcel store is not revision-namespaced, so
    /// a repo already present there is not reused for a revision-qualified
    /// request and the request is refused rather than answered with an unknown
    /// revision; use `--models-dir` to give each revision its own root. Not
    /// valid when the model argument is an existing local path.
    #[arg(long, value_name = "REV")]
    pub(crate) revision: Option<String>,

    /// Path to LoRA adapter directory (optional). Mirrors `mlxcel generate
    /// --adapter`.
    #[arg(long, value_name = "PATH")]
    pub(crate) adapter: Option<PathBuf>,

    /// Generation options shared verbatim with `mlxcel generate` (`-p/--prompt`,
    /// `-n/--max-tokens`, image/audio/video inputs, `--no-chat-template`, the
    /// TurboQuant KV-cache flags, …). Omitting `-p/--prompt` drops into the
    /// interactive chat REPL.
    #[command(flatten)]
    pub(crate) generation: GenerationOptions,

    /// Sampling options shared verbatim with `mlxcel generate` (temperature,
    /// top-k/p, min-p, repetition + DRY penalties).
    #[command(flatten)]
    pub(crate) sampling: SamplingOptions,

    /// The `mlxcel-server` sampling options the generate group lacks
    /// (frequency/presence penalties, XTC, Mirostat, dynamic temperature, DRY
    /// sequence breakers, stop strings).
    #[command(flatten)]
    pub(crate) server_sampling: ServerSamplingOptions,

    /// Speculative drafter checkpoint (a DFlash drafter or an MTP head),
    /// served through the server's speculative burst exactly as
    /// `mlxcel-server --draft-model` serves it. `--draft-kind` picks the kind
    /// when the checkpoint does not say.
    #[arg(long, value_name = "PATH")]
    pub(crate) draft_model: Option<PathBuf>,

    /// `--draft-kind` (`dflash` or `mtp`) and `--draft-block-size`, shared
    /// with `mlxcel serve`.
    #[command(flatten)]
    pub(crate) speculative: SpeculativeArgs,
}

impl RunArgs {
    /// Lower the `run` flag surface onto a full [`GenerateArgs`] for the
    /// one-shot modes that stay on `generate` until Phase 6, filling the model
    /// (default-model fallback) and leaving the parallelism, language-bias and
    /// surgery groups at their clap defaults.
    fn into_generate_args(self) -> GenerateArgs {
        let model = self.model.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));

        GenerateArgs {
            model: ModelOptions {
                model,
                models_dir: self.models_dir,
                revision: self.revision,
                adapter: self.adapter,
                draft_model: self.draft_model,
                num_draft_tokens: 3,
            },
            generation: self.generation,
            sampling: self.sampling,
            pipeline_parallel: crate::PipelineParallelOptions::default(),
            tensor_parallel: crate::TensorParallelOptions::default(),
            lang_bias: mlxcel::lang_bias::LangBiasCliArgs::default(),
            speculative: self.speculative,
            prompt_lookup: crate::PromptLookupOptions::default(),
            #[cfg(feature = "surgery")]
            surgery: None,
        }
    }

    /// The in-process server settings this command line asks for.
    pub(crate) fn server_settings(&self) -> Result<CliServerSettings> {
        let kv_cache_mode = resolve_cli_kv_cache_mode(&self.generation.turbo)?;
        let mut settings = settings_from_flags(
            &self.sampling,
            self.server_sampling.clone(),
            self.generation.max_tokens,
            kv_cache_mode,
            &cli_flag_was_set,
        );
        settings.adapter = self.adapter.clone();
        settings.draft_model = self.draft_model.clone();
        settings.draft_kind = self.speculative.draft_kind.clone();
        settings.draft_block_size = self.speculative.draft_block_size;
        Ok(settings)
    }
}

/// Whether a `-p` run asks for a one-shot mode the chat server does not
/// serve, so it stays on the `generate` flow (Phase 6, #2176).
fn one_shot_stays_on_generate(generation: &GenerationOptions, model_path: &Path) -> bool {
    if generation.output_audio.is_some()
        || generation.layout_detections.is_some()
        || generation.profile
        || generation.estimate_memory
        || generation.recommend_quant
        || !generation.video.is_empty()
        || generation.audio.is_some()
    {
        return true;
    }
    use mlxcel::models::ModelType;
    mlxcel::models::get_model_type(model_path).is_ok_and(|model_type| {
        matches!(
            model_type,
            ModelType::DiffusionGemma
                | ModelType::Llada2Moe
                | ModelType::Florence2VLM
                | ModelType::NemotronParseVLM
                | ModelType::NemotronVoiceChat
        )
    })
}

/// Handle `mlxcel run`.
///
/// No prompt: the chat REPL (issue #96). With `-p`: one chat turn through the
/// in-process server, or the `generate` one-shot flow for the modes
/// [`one_shot_stays_on_generate`] lists.
pub(crate) fn run_run(args: RunArgs) -> Result<()> {
    if args.generation.prompt.is_none() {
        // `generate`'s no-prompt checks (VoiceChat, `--layout-detections`,
        // `--output-audio`) and the REPL dispatch.
        if args.generation.output_audio.is_some()
            || args.generation.layout_detections.is_some()
            || args.generation.audio.is_some()
        {
            return crate::commands::run_generate(args.into_generate_args());
        }
        let opts = crate::commands::chat::ChatOptions {
            model: args
                .model
                .clone()
                .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL)),
            models_dir: args.models_dir.clone(),
            revision: args.revision.clone(),
            server: args.server_settings()?,
            no_chat_template: args.generation.no_chat_template,
            show_reasoning: args.generation.show_reasoning,
            images: args.generation.image.clone(),
        };
        return crate::commands::run_chat(opts);
    }
    run_once(args)
}

/// One `-p` turn through the in-process server.
fn run_once(args: RunArgs) -> Result<()> {
    let prompt = args.generation.prompt.clone().unwrap_or_default();
    let runtime = mlxcel::initialize_runtime_checked()?;
    super::generate::print_runtime_setup(&runtime);
    let requested = args
        .model
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let model_path = mlxcel::downloader::resolve_model_source_with_override(
        &requested,
        args.models_dir.as_deref(),
        args.revision.as_deref(),
    )?;
    if one_shot_stays_on_generate(&args.generation, &model_path) {
        return crate::commands::run_generate(args.into_generate_args());
    }

    let mut settings = args.server_settings()?;
    // One request has nothing to reuse, so the prompt cache stays off (ADR
    // 0007: on for chat clients, which `run -p` is not); the warmup is the
    // one-token pass the `generate` flow always ran before its generation.
    settings.prompt_cache = false;
    settings.warmup = true;
    settings.kv_cache_mode = mlxcel::cli::turbo_args::resolve_and_announce_kv_cache_mode(
        settings.kv_cache_mode,
        &model_path,
    );

    println!("Loading model from {model_path:?}...");
    let load_start = Instant::now();
    let server = InProcessServer::start(&settings.startup(&model_path))?;
    println!(
        "Model loaded in {:.2}s.",
        load_start.elapsed().as_secs_f64()
    );

    let display = TurnDisplay {
        show_reasoning: args.generation.show_reasoning,
    };
    let mut printer = TurnPrinter::new(display);
    println!("Generating...");
    let turn = run_cancellable(|cancel| {
        if args.generation.no_chat_template {
            let request = mlxcel::server::in_process::chat::completion_request_from_json(
                completion_request_body(&prompt, &settings),
            )?;
            server.complete(request, cancel, |delta| printer.on_delta(delta))
        } else {
            let messages = messages_json(&[Turn {
                message: ChatMessage {
                    role: "user".to_string(),
                    content: prompt.clone(),
                },
                images: args.generation.image.clone(),
            }])?;
            let request = mlxcel::server::in_process::chat::chat_request_from_json(
                chat_request_body(messages, &settings),
            )?;
            server.chat(request, cancel, |delta| printer.on_delta(delta))
        }
    })?;
    printer.finish(&turn, "Re-run");
    println!();
    let seconds = turn.result.generation_only_ms as f64 / 1000.0;
    let rate = if seconds > 0.0 {
        turn.result.completion_tokens as f64 / seconds
    } else {
        0.0
    };
    println!(
        "[Generated {} tokens in {seconds:.2}s = {rate:.2} tok/s]",
        turn.result.completion_tokens
    );
    if let Some(spec) = turn.result.speculative.as_ref() {
        println!(
            "[Speculative {:?}: {} rounds, {}/{} drafted tokens accepted]",
            spec.draft_kind, spec.draft_rounds, spec.draft_n_accepted, spec.draft_n
        );
    }
    server.shutdown()
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod tests;
