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

//! Cross-path decode parity harness (issue #2167, epic #2166).
//!
//! Runs one prompt and one `SamplingConfig` through
//!
//! - (a) `CxxGenerator`, the way `mlxcel generate` calls it,
//! - (b) the server `BatchScheduler` in-process at B=1 with dense decode
//!   storage, and
//! - (c) the same at B=1 with paged decode storage,
//!
//! and prints, for every pair, `identical` or the first divergent token index
//! with the id each side produced. On (b) it also compares a prompt-cache miss
//! (the request arriving cold) against a hit (the same request after a priming
//! request stored its history prefix), stating how each side's prefill was
//! partitioned.
//!
//! The run records a baseline: divergence is expected before epic #2166 lands
//! and exits 0. `--expect-identical` turns any divergence into exit code 1 so
//! later phases can gate on it. Run under `MLXCEL_SDPA_DETERMINISTIC=1`
//! (`make engine-parity` sets it); `--expect-identical` refuses to run
//! without it.

#[path = "engine_parity/report.rs"]
mod report;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use mlxcel::server::DecodeStorageBackend;
use mlxcel::server::engine_probe::cases::{
    cache_prime_len, default_dry_breaker_ids, greedy_case, seeded_penalties_case,
};
use mlxcel::server::engine_probe::cli_engine::generate_like_cli;
use mlxcel::server::engine_probe::prompt::render_chat_prompt;
use mlxcel::server::engine_probe::{
    ParityCase, ProbeCacheKey, ServerEngine, ServerEngineOptions, ServerEngineRequest,
    describe_prefill_partition,
};
use mlxcel_core::cache::KVCacheMode;

use report::{PairRow, PathStream, Report, Side};

const DEFAULT_PROMPT: &str = "Write a short story about a lighthouse keeper who finds a message \
in a bottle. Describe the weather that morning, the keeper's daily routine, and what the message \
says, in about three paragraphs.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CaseName {
    /// Greedy decoding, no penalties.
    Greedy,
    /// Seeded sampling with repetition/frequency/presence penalties and DRY.
    Seeded,
}

/// Cross-path decode parity harness: CLI vs server B=1 dense vs paged.
#[derive(Parser, Debug)]
#[command(name = "mlxcel-engine-parity")]
struct Args {
    /// Path to the model directory.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// User prompt, rendered through the chat template as one user turn.
    #[arg(short = 'p', long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Tokens to generate per run.
    #[arg(short = 'n', long, default_value_t = 64)]
    max_tokens: usize,

    /// Seed for the seeded sampling case.
    #[arg(long, default_value_t = 1234)]
    seed: u64,

    /// Cases to run.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = [CaseName::Greedy, CaseName::Seeded])]
    cases: Vec<CaseName>,

    /// Use the prompt verbatim instead of applying the chat template.
    #[arg(long)]
    no_chat_template: bool,

    /// Server `--prefill-chunk-size` for (b) and (c). Unset keeps the server
    /// default (512).
    #[arg(long, value_name = "N")]
    server_prefill_chunk: Option<usize>,

    /// Skip the prompt-cache miss/hit comparison.
    #[arg(long)]
    no_prompt_cache_case: bool,

    /// Exit with status 1 when any compared pair diverges.
    #[arg(long)]
    expect_identical: bool,

    /// Also write every token stream and row as JSON to this path.
    #[arg(long, value_name = "PATH")]
    json: Option<PathBuf>,
}

fn build_cases(args: &Args, stop_ids: &[i32], breakers: &[i32]) -> Vec<ParityCase> {
    args.cases
        .iter()
        .map(|case| match case {
            CaseName::Greedy => greedy_case(stop_ids.to_vec()),
            CaseName::Seeded => {
                seeded_penalties_case(stop_ids.to_vec(), breakers.to_vec(), args.seed)
            }
        })
        .collect()
}

fn server_options(args: &Args, storage: DecodeStorageBackend, cache: bool) -> ServerEngineOptions {
    ServerEngineOptions {
        decode_storage: storage,
        prefill_chunk_size: args.server_prefill_chunk,
        prompt_cache: cache,
    }
}

fn request<'a>(
    tokens: &'a [i32],
    case: &ParityCase,
    max_tokens: usize,
    cache: Option<ProbeCacheKey>,
) -> ServerEngineRequest<'a> {
    ServerEngineRequest {
        prompt_tokens: tokens,
        sampling: case.sampling.clone(),
        max_tokens,
        ignore_eos: false,
        prompt_cache: cache,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sdpa_deterministic = std::env::var("MLXCEL_SDPA_DETERMINISTIC").is_ok_and(|v| v == "1");
    if args.expect_identical && !sdpa_deterministic {
        anyhow::bail!(
            "--expect-identical needs MLXCEL_SDPA_DETERMINISTIC=1 (CUDA decode SDPA is \
             otherwise free to change its reduction order between runs)"
        );
    }
    mlxcel_core::hardware::apply_metal_ops_per_buffer_default();
    mlxcel_core::hardware::apply_cuda_graph_cache_default();
    mlxcel_core::hardware::apply_cuda_sdpa_cache_default();
    mlxcel_core::hardware::apply_cuda_graph_budget_default(Some(&args.model));
    let _runtime = mlxcel::initialize_runtime();

    let tokenizer = mlxcel::tokenizer::load_tokenizer(&args.model)
        .with_context(|| format!("failed to load tokenizer from {}", args.model.display()))?;
    let prompt = render_chat_prompt(&args.model, &tokenizer, &args.prompt, args.no_chat_template)?;
    let stop_ids = mlxcel::read_eos_token_ids(&args.model);
    let breakers = default_dry_breaker_ids(&tokenizer);
    let cases = build_cases(&args, &stop_ids, &breakers);
    let mut report = Report::new(&args.model, prompt.tokens.len(), sdpa_deterministic);

    // (a) CLI: load once, run every case, release the model before the server
    // worker loads its own copy.
    {
        let (model, _) = mlxcel::load_model(&args.model).context("failed to load model")?;
        for case in &cases {
            let tokens = generate_like_cli(
                &model,
                &args.model,
                &prompt.tokens,
                args.max_tokens,
                &case.sampling,
                KVCacheMode::Fp16,
            )?;
            report.push_stream(case.name, Side::Cli, PathStream::ran(tokens, "cli"));
        }
    }
    mlxcel_core::clear_memory_cache();

    // (b) dense and (c) paged, prompt cache off: the engine comparison proper.
    for (side, storage) in [
        (Side::Dense, DecodeStorageBackend::Dense),
        (Side::Paged, DecodeStorageBackend::Paged),
    ] {
        let engine = ServerEngine::start(&args.model, server_options(&args, storage, false))?;
        let effective = engine.effective_decode_storage();
        let chunk = engine.prefill_chunk_size();
        for case in &cases {
            let stream = if effective != storage {
                PathStream::not_applicable(format!(
                    "paged decode unsupported for this model (worker fell back to {effective:?})"
                ))
            } else {
                let run = engine.run(&request(&prompt.tokens, case, args.max_tokens, None))?;
                let partition =
                    describe_prefill_partition(prompt.tokens.len(), run.cached_tokens, None, chunk);
                PathStream::ran(run.tokens, &format!("{effective:?} pc=off {partition}"))
            };
            report.push_stream(case.name, side, stream);
        }
        engine.shutdown()?;
    }

    // (b) with the prompt cache on. Miss: the request arrives cold. Hit: a
    // priming request over the history prefix stores that prefix first (the
    // shape a follow-up chat turn finds), then the same request adopts it and
    // prefills only the rest. Each side has its own session key, so neither
    // can see the other's entries.
    if !args.no_prompt_cache_case {
        let engine = ServerEngine::start(
            &args.model,
            server_options(&args, DecodeStorageBackend::Dense, true),
        )?;
        let chunk = engine.prefill_chunk_size();
        let boundary = prompt
            .history_tokens
            .as_deref()
            .and_then(|history| engine.history_boundary(history, &prompt.tokens));
        let prime_len = cache_prime_len(&prompt.tokens, prompt.history_tokens.as_deref());
        for case in &cases {
            for side in [Side::CacheMiss, Side::CacheHit] {
                let key = ProbeCacheKey {
                    session: format!("engine-parity-{}-{side:?}", case.name),
                    history_tokens: prompt.history_tokens.clone(),
                };
                let mut primed = String::new();
                if side == Side::CacheHit {
                    let prime_key = ProbeCacheKey {
                        history_tokens: None,
                        ..key.clone()
                    };
                    let prime = engine.run(&request(
                        &prompt.tokens[..prime_len],
                        case,
                        1,
                        Some(prime_key),
                    ))?;
                    // KV-store inserts are counted; recurrent-state snapshots
                    // are not, so the adoption on the next request (`cached=`)
                    // is the evidence that priming worked.
                    primed = format!(
                        " primed[0..{prime_len}){}",
                        prime
                            .prompt_cache_reject
                            .map_or_else(String::new, |r| format!(" reject={r}"))
                    );
                }
                let run = engine.run(&request(&prompt.tokens, case, args.max_tokens, Some(key)))?;
                let partition = describe_prefill_partition(
                    prompt.tokens.len(),
                    run.cached_tokens,
                    boundary,
                    chunk,
                );
                let label = format!(
                    "Dense pc=on cached={} forwarded={} {partition}{primed}{}",
                    run.cached_tokens,
                    run.forwarded_prefill_tokens,
                    run.prompt_cache_reject
                        .map_or_else(String::new, |r| format!(" reject={r}"))
                );
                report.push_stream(case.name, side, PathStream::ran(run.tokens, &label));
            }
        }
        engine.shutdown()?;
    }

    let rows: Vec<PairRow> = report.rows();
    report.print(&rows);
    if let Some(path) = args.json.as_ref() {
        report.write_json(path, &rows)?;
        println!("wrote {}", path.display());
    }
    let diverged = rows.iter().filter(|row| row.diverged()).count();
    if args.expect_identical && diverged > 0 {
        eprintln!("--expect-identical: {diverged} pair(s) diverged");
        std::process::exit(1);
    }
    Ok(())
}
