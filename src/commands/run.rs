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

//! `mlxcel run` verb (epic #92, issue #95): the ollama-style entry
//! point.
//!
//! `run` is the capstone of the unified download + run epic. It is a thin
//! dispatcher that forks **no** model-loading or generation code, it builds a
//! [`GenerateArgs`] from its own (deliberately small) flag surface and hands it
//! straight to [`crate::commands::run_generate`], which already routes:
//!
//! * **no `-p/--prompt`** → the interactive multi-turn chat REPL
//!   ([`crate::commands::run_chat`], issue #96), and
//! * **with `-p`** → the historical one-shot `generate` flow
//!   (`run_generate_once`), including the repo-id-aware `-m` resolver
//!   ([`mlxcel::downloader::resolve_model_source`], issue #94).
//!
//! Routing through `run_generate` (rather than re-implementing the
//! prompt/no-prompt branch) is what guarantees `mlxcel run <repo-id> -p "..."`
//! produces byte-identical output to the equivalent `mlxcel generate -m
//! <repo-id> -p "..."` invocation, they execute the same code.
//!
//! ## Default-model fallback
//!
//! When no model argument is supplied, `run` falls back to [`DEFAULT_MODEL`].
//! The repo-id is auto-downloaded into the mlxcel global store on first use by
//! the shared resolver, so `mlxcel run` with no arguments works from any
//! directory.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

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
}

impl RunArgs {
    /// Lower the `run` flag surface onto a full [`GenerateArgs`], filling the
    /// model (default-model fallback) and leaving every advanced flag group
    /// (`tensor_parallel` / `pipeline_parallel` / `speculative` / `lang_bias`
    /// / `surgery`) at its clap default; `run` intentionally does not expose
    /// them, matching the minimal `ollama run` surface. The resulting
    /// `GenerateArgs` is then driven by the unchanged `run_generate` dispatch.
    fn into_generate_args(self) -> GenerateArgs {
        let model = self.model.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));

        GenerateArgs {
            model: ModelOptions {
                model,
                models_dir: self.models_dir,
                adapter: self.adapter,
                // `run` does not surface offline speculative decoding; keep the
                // same defaults `mlxcel generate` uses when the flags are absent.
                draft_model: None,
                num_draft_tokens: 3,
            },
            generation: self.generation,
            sampling: self.sampling,
            pipeline_parallel: crate::PipelineParallelOptions::default(),
            tensor_parallel: crate::TensorParallelOptions::default(),
            lang_bias: mlxcel::lang_bias::LangBiasCliArgs::default(),
            speculative: mlxcel::cli::speculative_args::SpeculativeArgs::default(),
            #[cfg(feature = "surgery")]
            surgery: None,
        }
    }
}

/// Handle `mlxcel run`.
///
/// Resolves the default model when none is given, then dispatches through the
/// shared [`crate::commands::run_generate`] path: no prompt → interactive chat
/// REPL (issue #96); `-p` → one-shot generation (the historical `generate`
/// flow). Model resolution / auto-download is performed by the same
/// [`mlxcel::downloader::resolve_model_source`] resolver (issue #94) those paths
/// already use.
pub(crate) fn run_run(args: RunArgs) -> Result<()> {
    crate::commands::run_generate(args.into_generate_args())
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod tests;
