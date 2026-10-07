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

//! Single-stream (B=1) decode benchmark for both decode paths (issue #2167).
//!
//! `mlxcel-bench-decode` times only `CxxGenerator`, so the path the server
//! decodes on has no decode benchmark. This binary measures both on the same
//! synthesized prompt:
//!
//! - `cli`: `CxxGenerator::generate_with_stats` after a warmup pass, exactly
//!   `mlxcel-bench-decode`'s measured call, so its numbers are the ones that
//!   tool reports for the same model, prompt length and token budget;
//! - `server`: the `mlxcel-server` model worker and `BatchScheduler`, driven
//!   in-process with pre-tokenized ids so HTTP and tokenization stay outside
//!   the timed region, one request at a time.
//!
//! Each measurement prints a human-readable line and an `[engine-bench]` JSON
//! line (parsed by `scripts/engine_bench_rounds.py`), and with `--csv` appends
//! a CSV row. `--prefill-chunk` sets the prefill chunk on both paths (the CLI
//! through `MLXCEL_PREFILL_CHUNK`, the server through `--prefill-chunk-size`)
//! and `--decode-storage` picks the server's decode storage, which is what the
//! ADR 0007 decision-table A/Bs vary. One measured pass per context per
//! process; run several processes, interleaved, for numbers.

#[path = "bench_engine/output.rs"]
mod output;

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use mlxcel::server::DecodeStorageBackend;
use mlxcel::server::engine_probe::cases::neutral_params;
use mlxcel::server::engine_probe::cli_engine::{bench_like_bench_decode, suppress_eos_for_cli};
use mlxcel::server::engine_probe::prompt::{cap_prompt_len, synthesize_prompt_tokens};
use mlxcel::server::engine_probe::{ServerEngine, ServerEngineOptions, ServerEngineRequest};
use mlxcel_core::cache::KVCacheMode;

use output::Measurement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PathArg {
    Cli,
    Server,
    Both,
}

/// B=1 decode benchmark: `CxxGenerator` vs the in-process server scheduler.
#[derive(Parser, Debug)]
#[command(name = "mlxcel-bench-engine")]
struct Args {
    /// Path to the model directory.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Which decode path(s) to measure.
    #[arg(long, value_enum, default_value_t = PathArg::Both)]
    path: PathArg,

    /// Synthesized prompt lengths in tokens, one measured pass each. The
    /// defaults are the short (a few hundred keys) and long (about 8K keys)
    /// contexts of the ADR 0007 baseline.
    #[arg(long, value_delimiter = ',', default_values_t = [256usize, 8192])]
    prompt_tokens: Vec<usize>,

    /// Tokens generated in the measured pass.
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,

    /// Tokens generated in the warmup pass that precedes each measured pass.
    #[arg(long, default_value_t = 16)]
    warmup_tokens: usize,

    /// Prefill chunk for both paths. Unset keeps each path's default (CLI
    /// 2048 or `MLXCEL_PREFILL_CHUNK`, server 512). The CLI side is applied
    /// through `MLXCEL_PREFILL_CHUNK` and conflicts with a different value
    /// already in the environment.
    #[arg(long, value_name = "N")]
    prefill_chunk: Option<usize>,

    /// Server decode storage: auto (the server default), dense, or paged.
    #[arg(long, default_value = "auto", value_parser = DecodeStorageBackend::from_str)]
    decode_storage: DecodeStorageBackend,

    /// Suppress end-of-generation tokens on both paths so every pass spends
    /// the full budget. Off by default: on the server the EOS bias puts the
    /// request on the per-row sampler, which is not the path a plain request
    /// decodes on.
    #[arg(long)]
    ignore_eos: bool,

    /// Append one CSV row per measurement to this file.
    #[arg(long, value_name = "PATH")]
    csv: Option<PathBuf>,

    /// Free-form label recorded with every measurement (for example an A/B
    /// arm name).
    #[arg(long, default_value = "")]
    label: String,
}

/// What to do about `MLXCEL_PREFILL_CHUNK` for `--prefill-chunk`: `Ok(Some)`
/// is the value to set, `Ok(None)` means nothing to do, and a different value
/// already in the environment is an error (two sources for one setting would
/// leave the run's chunk ambiguous).
fn cli_chunk_env_action(chunk: Option<usize>, existing: Option<&str>) -> Result<Option<String>> {
    let Some(chunk) = chunk else {
        return Ok(None);
    };
    anyhow::ensure!(chunk > 0, "--prefill-chunk must be at least 1");
    match existing {
        Some(existing) if existing.trim() != chunk.to_string() => anyhow::bail!(
            "--prefill-chunk {chunk} conflicts with MLXCEL_PREFILL_CHUNK={existing} in the \
             environment; drop one of them"
        ),
        Some(_) => Ok(None),
        None => Ok(Some(chunk.to_string())),
    }
}

/// Apply `--prefill-chunk` to the CLI path before anything reads it.
fn apply_cli_prefill_chunk(chunk: Option<usize>) -> Result<()> {
    let existing = std::env::var("MLXCEL_PREFILL_CHUNK").ok();
    if let Some(value) = cli_chunk_env_action(chunk, existing.as_deref())? {
        // SAFETY: called first thing in `main`, before the runtime, the model
        // loader or any worker thread exists, so no other thread can be
        // reading the environment concurrently.
        unsafe { std::env::set_var("MLXCEL_PREFILL_CHUNK", value) }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let run_cli = matches!(args.path, PathArg::Cli | PathArg::Both);
    let run_server = matches!(args.path, PathArg::Server | PathArg::Both);
    if run_cli {
        apply_cli_prefill_chunk(args.prefill_chunk)?;
    }
    mlxcel_core::hardware::apply_metal_ops_per_buffer_default();
    mlxcel_core::hardware::apply_cuda_graph_cache_default();
    mlxcel_core::hardware::apply_cuda_sdpa_cache_default();
    mlxcel_core::hardware::apply_cuda_graph_budget_default(Some(&args.model));
    let _runtime = mlxcel::initialize_runtime();

    let tokenizer = mlxcel::tokenizer::load_tokenizer(&args.model)
        .with_context(|| format!("failed to load tokenizer from {}", args.model.display()))?;
    let max_context = mlxcel::read_model_context_window(&args.model);
    let mut prompts = Vec::with_capacity(args.prompt_tokens.len());
    for &target in &args.prompt_tokens {
        let len = cap_prompt_len(target, max_context, args.max_tokens);
        prompts.push((target, synthesize_prompt_tokens(&tokenizer, len)?));
    }
    let sampling = mlxcel::sampling::build_sampling_config(neutral_params(
        mlxcel::read_eos_token_ids(&args.model),
    ));
    let mut measurements: Vec<Measurement> = Vec::new();

    if run_cli {
        let (model, _) = mlxcel::load_model(&args.model).context("failed to load model")?;
        let mut cli_sampling = sampling.clone();
        if args.ignore_eos {
            suppress_eos_for_cli(&mut cli_sampling, &model);
        }
        let chunk = mlxcel_core::generate::prefill_chunk_len();
        for (target, tokens) in &prompts {
            let (generated, stats) = bench_like_bench_decode(
                &model,
                tokens,
                args.max_tokens,
                args.warmup_tokens,
                &cli_sampling,
                KVCacheMode::Fp16,
            );
            let m = Measurement::from_cli(&args.model, *target, &stats, generated.len(), chunk);
            m.emit(&args.label, args.max_tokens);
            measurements.push(m);
        }
    }
    mlxcel_core::clear_memory_cache();

    if run_server {
        let engine = ServerEngine::start(
            &args.model,
            ServerEngineOptions {
                decode_storage: args.decode_storage,
                prefill_chunk_size: args.prefill_chunk,
                prompt_cache: true,
            },
        )?;
        let storage = engine.effective_decode_storage();
        let chunk = engine.prefill_chunk_size();
        for (target, tokens) in &prompts {
            let request = |max_tokens| ServerEngineRequest {
                prompt_tokens: tokens,
                sampling: sampling.clone(),
                max_tokens,
                ignore_eos: args.ignore_eos,
                prompt_cache: None,
            };
            if args.warmup_tokens > 0 {
                engine.run(&request(args.warmup_tokens))?;
            }
            let run = engine.run(&request(args.max_tokens))?;
            let m =
                Measurement::from_server(&args.model, *target, tokens.len(), &run, chunk, storage);
            m.emit(&args.label, args.max_tokens);
            measurements.push(m);
        }
        engine.shutdown()?;
    }

    output::print_side_by_side(&measurements);
    if let Some(path) = args.csv.as_ref() {
        output::append_csv(path, &measurements, &args.label, args.max_tokens)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["mlxcel-bench-engine", "--model", "models/m"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("arguments parse")
    }

    #[test]
    fn defaults_measure_short_and_long_context_on_both_paths() {
        let args = parse(&[]);
        assert_eq!(args.path, PathArg::Both);
        assert_eq!(args.prompt_tokens, vec![256, 8192]);
        assert_eq!(args.max_tokens, 128);
        assert_eq!(args.warmup_tokens, 16);
        assert_eq!(args.prefill_chunk, None);
        assert_eq!(args.decode_storage, DecodeStorageBackend::Auto);
        assert!(!args.ignore_eos);
    }

    #[test]
    fn storage_chunk_and_prompt_lengths_parse() {
        let args = parse(&[
            "--path",
            "server",
            "--decode-storage",
            "paged",
            "--prefill-chunk",
            "2048",
            "--prompt-tokens",
            "512,4096",
        ]);
        assert_eq!(args.path, PathArg::Server);
        assert_eq!(args.decode_storage, DecodeStorageBackend::Paged);
        assert_eq!(args.prefill_chunk, Some(2048));
        assert_eq!(args.prompt_tokens, vec![512, 4096]);
        assert!(
            Args::try_parse_from([
                "mlxcel-bench-engine",
                "-m",
                "m",
                "--decode-storage",
                "bogus"
            ])
            .is_err()
        );
        assert!(Args::try_parse_from(["mlxcel-bench-engine", "--path", "cli"]).is_err());
    }

    #[test]
    fn chunk_env_is_set_only_when_requested_and_unset() {
        assert_eq!(cli_chunk_env_action(None, None).unwrap(), None);
        assert_eq!(cli_chunk_env_action(None, Some("512")).unwrap(), None);
        assert_eq!(
            cli_chunk_env_action(Some(2048), None).unwrap(),
            Some("2048".to_string())
        );
    }

    #[test]
    fn chunk_env_agreeing_value_is_accepted_and_conflict_is_an_error() {
        assert_eq!(
            cli_chunk_env_action(Some(512), Some(" 512 ")).unwrap(),
            None
        );
        let err = cli_chunk_env_action(Some(2048), Some("512"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("conflicts with MLXCEL_PREFILL_CHUNK=512"),
            "{err}"
        );
    }

    #[test]
    fn zero_chunk_is_rejected() {
        let err = cli_chunk_env_action(Some(0), None).unwrap_err().to_string();
        assert!(err.contains("at least 1"), "{err}");
    }
}
