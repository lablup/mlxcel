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

/// Smallest prompt (in tokens) the harness can run. Every path needs at least
/// one token to prefill.
const MIN_PROMPT_TOKENS: usize = 1;

/// Smallest prompt the prompt-cache miss/hit comparison can run. The priming
/// request stores a strict prefix of the prompt and the measured request still
/// needs one token to prefill, so a prompt shorter than this would hand the
/// scheduler a zero-length priming request.
const MIN_PROMPT_TOKENS_FOR_CACHE_CASE: usize = 2;

/// Reject a prompt that cannot drive every requested path, before any model is
/// loaded or any worker is started.
fn validate_prompt_text(prompt: &str) -> Result<()> {
    anyhow::ensure!(
        !prompt.trim().is_empty(),
        "--prompt is empty; give the harness a non-empty prompt"
    );
    Ok(())
}

/// Reject a rendered prompt too short for the requested cases, with the reason
/// and the remedy, instead of failing later on a zero-length prime request.
fn validate_prompt_tokens(token_count: usize, prompt_cache_case: bool) -> Result<()> {
    let needed = if prompt_cache_case {
        MIN_PROMPT_TOKENS_FOR_CACHE_CASE
    } else {
        MIN_PROMPT_TOKENS
    };
    anyhow::ensure!(
        token_count >= needed,
        "the rendered prompt is {token_count} token(s) long but at least {needed} {} needed{}; \
         give --prompt more text",
        if needed == 1 { "is" } else { "are" },
        if prompt_cache_case {
            " (the prompt-cache comparison primes a strict prefix and still prefills one token; \
             --no-prompt-cache-case lifts the second requirement)"
        } else {
            ""
        },
    );
    Ok(())
}

/// `--expect-identical` compares token streams byte for byte, which only means
/// something when the CUDA decode SDPA reduction order is pinned.
fn validate_expect_identical(expect_identical: bool, sdpa_deterministic: bool) -> Result<()> {
    anyhow::ensure!(
        !expect_identical || sdpa_deterministic,
        "--expect-identical needs MLXCEL_SDPA_DETERMINISTIC=1 (CUDA decode SDPA is \
         otherwise free to change its reduction order between runs)"
    );
    Ok(())
}

/// Whether the run should exit non-zero: only `--expect-identical` turns a
/// divergence into a failure, the baseline run itself always exits 0.
fn gate_failed(expect_identical: bool, diverged: usize) -> bool {
    expect_identical && diverged > 0
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sdpa_deterministic = std::env::var("MLXCEL_SDPA_DETERMINISTIC").is_ok_and(|v| v == "1");
    validate_expect_identical(args.expect_identical, sdpa_deterministic)?;
    validate_prompt_text(&args.prompt)?;
    mlxcel_core::hardware::apply_metal_ops_per_buffer_default();
    mlxcel_core::hardware::apply_cuda_graph_cache_default();
    mlxcel_core::hardware::apply_cuda_sdpa_cache_default();
    mlxcel_core::hardware::apply_cuda_graph_budget_default(Some(&args.model));
    let _runtime = mlxcel::initialize_runtime();

    let tokenizer = mlxcel::tokenizer::load_tokenizer(&args.model)
        .with_context(|| format!("failed to load tokenizer from {}", args.model.display()))?;
    let prompt = render_chat_prompt(&args.model, &tokenizer, &args.prompt, args.no_chat_template)?;
    validate_prompt_tokens(prompt.tokens.len(), !args.no_prompt_cache_case)?;
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
                // Model-owned KV families resolve to paged storage but decode
                // a lone sequence through their dense caches, so the kernel
                // evidence goes next to the storage name.
                let kernel = match (effective, run.paged_decode_launches) {
                    (DecodeStorageBackend::Paged, 0) => {
                        " paged-kernel-launches=0 (B=1 decode ran dense attention)".to_string()
                    }
                    (DecodeStorageBackend::Paged, n) => format!(" paged-kernel-launches={n}"),
                    _ => String::new(),
                };
                PathStream::ran(
                    run.tokens,
                    &format!("{effective:?} pc=off {partition}{kernel}"),
                )
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
    if gate_failed(args.expect_identical, diverged) {
        eprintln!("--expect-identical: {diverged} pair(s) diverged");
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["mlxcel-engine-parity", "--model", "models/m"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("arguments parse")
    }

    #[test]
    fn defaults_run_both_cases_with_the_prompt_cache_comparison() {
        let args = parse(&[]);
        assert_eq!(args.cases, vec![CaseName::Greedy, CaseName::Seeded]);
        assert_eq!(args.max_tokens, 64);
        assert_eq!(args.seed, 1234);
        assert!(!args.expect_identical);
        assert!(!args.no_prompt_cache_case);
        assert_eq!(args.server_prefill_chunk, None);
    }

    #[test]
    fn model_is_required_and_cases_parse_as_a_list() {
        assert!(Args::try_parse_from(["mlxcel-engine-parity"]).is_err());
        let args = parse(&["--cases", "seeded"]);
        assert_eq!(args.cases, vec![CaseName::Seeded]);
        let args = parse(&["--cases", "greedy,seeded"]);
        assert_eq!(args.cases, vec![CaseName::Greedy, CaseName::Seeded]);
        assert!(
            Args::try_parse_from(["mlxcel-engine-parity", "-m", "m", "--cases", "bogus"]).is_err()
        );
    }

    #[test]
    fn build_cases_follow_the_requested_order_and_seed() {
        let args = parse(&["--cases", "seeded,greedy", "--seed", "9"]);
        let cases = build_cases(&args, &[2], &[198]);
        let names: Vec<_> = cases.iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["seeded-penalties-dry", "greedy"]);
        assert_eq!(cases[0].sampling.seed, Some(9));
        assert_eq!(cases[0].sampling.dry_sequence_breakers, vec![198]);
        assert_eq!(cases[1].sampling.temperature, 0.0);
    }

    #[test]
    fn server_options_carry_storage_chunk_and_cache_flag() {
        let args = parse(&["--server-prefill-chunk", "2048"]);
        let options = server_options(&args, DecodeStorageBackend::Paged, true);
        assert_eq!(options.decode_storage, DecodeStorageBackend::Paged);
        assert_eq!(options.prefill_chunk_size, Some(2048));
        assert!(options.prompt_cache);
        let options = server_options(&parse(&[]), DecodeStorageBackend::Dense, false);
        assert_eq!(options.prefill_chunk_size, None);
        assert!(!options.prompt_cache);
    }

    #[test]
    fn request_borrows_the_prompt_and_clones_the_sampling() {
        let case = greedy_case(vec![2]);
        let tokens = [1, 2, 3];
        let req = request(&tokens, &case, 16, None);
        assert_eq!(req.prompt_tokens, &tokens);
        assert_eq!(req.max_tokens, 16);
        assert!(!req.ignore_eos);
        assert!(req.prompt_cache.is_none());
    }

    #[test]
    fn empty_prompt_text_is_rejected_up_front() {
        assert!(validate_prompt_text("hello").is_ok());
        let err = validate_prompt_text("").unwrap_err().to_string();
        assert!(err.contains("--prompt is empty"), "{err}");
        assert!(validate_prompt_text("  \n\t ").is_err());
    }

    #[test]
    fn too_short_prompt_is_rejected_before_the_prime_request() {
        // The cache comparison needs a strict prefix to prime plus one token to
        // prefill, so a one-token prompt would have produced a zero-length
        // prime request.
        assert_eq!(cache_prime_len(&[7], None), 0);
        let err = validate_prompt_tokens(1, true).unwrap_err().to_string();
        assert!(err.contains("at least 2 are needed"), "{err}");
        assert!(err.contains("--no-prompt-cache-case"), "{err}");
        assert!(validate_prompt_tokens(0, true).is_err());
        assert!(validate_prompt_tokens(2, true).is_ok());
        assert!(cache_prime_len(&[7, 8], None) >= 1);
    }

    #[test]
    fn single_token_prompt_runs_when_the_cache_case_is_off() {
        assert!(validate_prompt_tokens(1, false).is_ok());
        let err = validate_prompt_tokens(0, false).unwrap_err().to_string();
        assert!(err.contains("at least 1 is needed"), "{err}");
        assert!(!err.contains("--no-prompt-cache-case"), "{err}");
    }

    #[test]
    fn expect_identical_requires_deterministic_sdpa() {
        assert!(validate_expect_identical(false, false).is_ok());
        assert!(validate_expect_identical(false, true).is_ok());
        assert!(validate_expect_identical(true, true).is_ok());
        let err = validate_expect_identical(true, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("MLXCEL_SDPA_DETERMINISTIC=1"), "{err}");
    }

    #[test]
    fn only_expect_identical_turns_divergence_into_failure() {
        assert!(!gate_failed(false, 5));
        assert!(!gate_failed(true, 0));
        assert!(gate_failed(true, 1));
    }
}
