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

//! Argument-parsing and lowering tests for the `mlxcel run` verb (issue #95).
//!
//! These cover the clap surface and the `RunArgs -> GenerateArgs` lowering
//! only; they never load a model or run generation (the dispatch into
//! `run_generate` is exercised end-to-end by the real-model integration
//! suites). The point is to lock the three documented `run` behaviors —
//! one-shot (`-p`), no-prompt REPL, and default-model fallback — at the
//! arg-construction boundary.

use std::path::PathBuf;

use clap::Parser;

use super::{DEFAULT_MODEL, RunArgs};
use crate::{Cli, Commands};

/// Parse a `mlxcel run ...` command line into [`RunArgs`], panicking with the
/// clap error if it does not parse.
fn parse_run(argv: &[&str]) -> RunArgs {
    let cli = Cli::try_parse_from(argv).expect("run command line should parse");
    match cli.command {
        Commands::Run(args) => args,
        other => panic!("expected a Run command, got {other:?}"),
    }
}

#[test]
fn run_with_repo_id_and_prompt_is_one_shot() {
    // `mlxcel run <repo-id> -p "..."` → model + prompt populated, so the
    // shared `run_generate` dispatch takes the one-shot branch.
    let args = parse_run(&[
        "mlxcel",
        "run",
        "mlx-community/Qwen3-4B-4bit",
        "-p",
        "Hello, world!",
    ]);
    assert_eq!(
        args.model.as_deref(),
        Some(std::path::Path::new("mlx-community/Qwen3-4B-4bit"))
    );
    assert_eq!(args.generation.prompt.as_deref(), Some("Hello, world!"));

    // The lowered GenerateArgs must carry the same model + prompt verbatim.
    let gen_args = args.into_generate_args();
    assert_eq!(
        gen_args.model.model,
        PathBuf::from("mlx-community/Qwen3-4B-4bit")
    );
    assert_eq!(gen_args.generation.prompt.as_deref(), Some("Hello, world!"));
}

#[test]
fn run_with_repo_id_no_prompt_enters_repl() {
    // `mlxcel run <repo-id>` (no -p) → prompt is None, which `run_generate`
    // routes into the interactive chat REPL (issue #96).
    let args = parse_run(&["mlxcel", "run", "mlx-community/Qwen3-4B-4bit"]);
    assert!(
        args.generation.prompt.is_none(),
        "no -p must leave prompt None so dispatch enters the REPL"
    );

    let gen_args = args.into_generate_args();
    assert_eq!(
        gen_args.model.model,
        PathBuf::from("mlx-community/Qwen3-4B-4bit")
    );
    assert!(gen_args.generation.prompt.is_none());
}

#[test]
fn run_with_no_model_falls_back_to_default_model() {
    // `mlxcel run` (no model arg) → default-model fallback (mlx-lm style).
    let args = parse_run(&["mlxcel", "run"]);
    assert!(
        args.model.is_none(),
        "model is optional; omitting it must parse to None"
    );

    let gen_args = args.into_generate_args();
    assert_eq!(
        gen_args.model.model,
        PathBuf::from(DEFAULT_MODEL),
        "absent model must lower to the documented default repo-id"
    );
    // No prompt either → the default model is run interactively.
    assert!(gen_args.generation.prompt.is_none());
}

#[test]
fn run_no_model_with_prompt_uses_default_model_one_shot() {
    // `mlxcel run -p "..."` → default model, one-shot.
    let args = parse_run(&["mlxcel", "run", "-p", "Hi"]);
    assert!(args.model.is_none());
    assert_eq!(args.generation.prompt.as_deref(), Some("Hi"));

    let gen_args = args.into_generate_args();
    assert_eq!(gen_args.model.model, PathBuf::from(DEFAULT_MODEL));
    assert_eq!(gen_args.generation.prompt.as_deref(), Some("Hi"));
}

#[test]
fn default_model_matches_mlx_lm() {
    // Locks the documented default to the mlx-lm `DEFAULT_MODEL` value. If a
    // future change picks a different default, this test forces the README /
    // help text to be updated deliberately.
    assert_eq!(DEFAULT_MODEL, "mlx-community/Llama-3.2-3B-Instruct-4bit");
}

#[test]
fn run_shares_generate_sampling_and_generation_flags() {
    // The sampling/generation flags must be the *same* clap groups `generate`
    // uses, so a value provided to `run` round-trips into the lowered
    // GenerateArgs unchanged.
    let args = parse_run(&[
        "mlxcel",
        "run",
        "mlx-community/Qwen3-4B-4bit",
        "-p",
        "x",
        "-n",
        "42",
        "--temp",
        "0.7",
        "--top-p",
        "0.95",
        "--no-chat-template",
    ]);
    let gen_args = args.into_generate_args();
    assert_eq!(gen_args.generation.max_tokens, 42);
    assert!(gen_args.generation.no_chat_template);
    assert_eq!(gen_args.sampling.temp, 0.7);
    assert_eq!(gen_args.sampling.top_p, 0.95);
}

#[test]
fn run_accepts_adapter_flag() {
    let args = parse_run(&[
        "mlxcel",
        "run",
        "mlx-community/Qwen3-4B-4bit",
        "--adapter",
        "adapters/lora",
    ]);
    let gen_args = args.into_generate_args();
    assert_eq!(gen_args.model.adapter, Some(PathBuf::from("adapters/lora")));
}

#[test]
fn run_lowers_advanced_groups_to_inert_defaults() {
    // `run` does not expose tensor/pipeline parallelism or speculative
    // decoding; the lowered GenerateArgs must leave them at the same inert
    // single-device defaults `generate` uses when those flags are absent, so
    // the dispatched one-shot path behaves identically to a plain `generate`.
    let gen_args = parse_run(&["mlxcel", "run", "mlx-community/Qwen3-4B-4bit", "-p", "x"])
        .into_generate_args();
    assert_eq!(gen_args.tensor_parallel.tp_size, 1);
    assert_eq!(gen_args.pipeline_parallel.pp_size, 1);
    assert_eq!(gen_args.pipeline_parallel.pp_micro_batch_size, 1);
    assert!(gen_args.model.draft_model.is_none());
    assert_eq!(gen_args.model.num_draft_tokens, 3);
    assert!(gen_args.speculative.draft_kind.is_none());
}
