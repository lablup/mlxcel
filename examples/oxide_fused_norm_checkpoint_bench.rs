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

//! End-to-end run of one checkpoint that records which implementation of the
//! fused residual-add + RMSNorm served it (issue #2834).
//!
//! `mlxcel-bench-engine` already measures B=1 prefill and decode on both
//! front ends, but it prints nothing about the route, and the route is the
//! variable this experiment changes: graph path (`MLXCEL_FUSED_ADD_RMSNORM`
//! unset), the existing fused kernel (`=1`), or the cuda-oxide node behind it
//! (`=1` plus `MLXCEL_CUDA_OXIDE_FUSED_NORM=1`, in a build made with
//! `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1`). All three are process-wide, so one
//! process is one arm. This example runs the same two engine probes
//! `mlxcel-bench-engine` runs (`bench_like_bench_decode` for the CLI path, the
//! in-process `ServerEngine` for the scheduler), adds a batch of concurrent
//! requests on the scheduler, and prints after each measured pass:
//!
//! - the adapter's route counters over that pass, so a run that claims the
//!   cuda-oxide route shows launches and zero existing-kernel joins;
//! - MLX's peak memory over the pass (reset before it), which on a unified
//!   memory host is the figure RSS does not show;
//! - the generated token ids, so greedy output can be compared across arms.
//!
//! ```text
//! MLXCEL_FUSED_ADD_RMSNORM=1 MLXCEL_CUDA_OXIDE_FUSED_NORM=1 \
//!   gpu-lock run --tag issue-2834-e2e -- target/release/examples/oxide_fused_norm_checkpoint_bench \
//!   --model models/llama-3.2-1b-instruct-bf16 --path server --batch 4 \
//!   --prompt-tokens 128 --label oxide
//! ```
//!
//! One `[oxide-ckpt-bench] {json}` line per measured pass; `--csv` appends
//! the same fields as a row. Decode throughput follows docs/benchmarks.md: the
//! first token comes out of prefill, so a request of N tokens contributes N-1
//! to the decode window. For a batch the window runs from the earliest first
//! token to the last completion and the numerator is the sum over rows.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Barrier;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use mlxcel::server::engine_probe::cases::neutral_params;
use mlxcel::server::engine_probe::cli_engine::bench_like_bench_decode;
use mlxcel::server::engine_probe::prompt::{cap_prompt_len, synthesize_prompt_tokens};
use mlxcel::server::engine_probe::{ServerEngine, ServerEngineOptions, ServerEngineRequest};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::fused_norm_route::{FusedNormRouteSnapshot, fused_norm_route_snapshot};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PathArg {
    Cli,
    Server,
}

/// End-to-end checkpoint run with fused-norm route markers.
#[derive(Parser, Debug)]
#[command(name = "oxide_fused_norm_checkpoint_bench")]
struct Args {
    /// Path to the model directory.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Decode path: the raw-completion client or the in-process scheduler.
    #[arg(long, value_enum)]
    path: PathArg,

    /// Concurrent requests on the scheduler. The CLI path is B=1 only.
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Synthesized prompt lengths in tokens, one measured pass each.
    #[arg(long, value_delimiter = ',', default_values_t = [128usize, 2048])]
    prompt_tokens: Vec<usize>,

    /// Tokens generated in the measured pass.
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,

    /// Tokens generated in the warmup pass that precedes each measured pass.
    #[arg(long, default_value_t = 16)]
    warmup_tokens: usize,

    /// Arm name recorded with every measurement.
    #[arg(long, default_value = "")]
    label: String,

    /// The route the counters must show for the run to count: `graph`,
    /// `fused` or `oxide`. A mismatch exits non-zero after printing the record.
    #[arg(long)]
    expect_route: Option<String>,

    /// Append one CSV row per measured pass to this file.
    #[arg(long, value_name = "PATH")]
    csv: Option<PathBuf>,
}

/// Counter movement over one measured pass.
#[derive(Debug, Clone, Copy, Default)]
struct RouteDelta {
    built: bool,
    installed: bool,
    oxide_routes: u64,
    native_routes: u64,
    launches: u64,
    last_fallback: &'static str,
}

impl RouteDelta {
    fn between(
        before: Option<FusedNormRouteSnapshot>,
        after: Option<FusedNormRouteSnapshot>,
    ) -> Self {
        match (before, after) {
            (Some(b), Some(a)) => Self {
                built: true,
                installed: a.installed,
                oxide_routes: a.oxide_routes - b.oxide_routes,
                native_routes: a.native_routes - b.native_routes,
                launches: a.launches - b.launches,
                last_fallback: a.last_fallback,
            },
            _ => Self {
                last_fallback: "adapter_absent",
                ..Self::default()
            },
        }
    }

    /// The implementation that served the pass, by the same rule as
    /// `FusedNormRouteSnapshot::route`. Without the adapter nothing counts the
    /// existing kernel, so the route cannot be told apart from the graph path.
    fn route(&self) -> &'static str {
        if !self.built {
            return "unknown";
        }
        match (self.oxide_routes, self.native_routes) {
            (0, 0) => "graph",
            (0, _) => "fused",
            (_, 0) if self.launches > 0 => "oxide",
            _ => "mixed",
        }
    }
}

struct Pass {
    path: &'static str,
    batch: usize,
    prompt_target: usize,
    prompt_tokens: usize,
    /// Per-row generated tokens.
    tokens: Vec<Vec<i32>>,
    /// Median per-row time to the first token.
    ttft_ms: f64,
    /// Decode window: first token (earliest row) to last completion.
    decode_ms: f64,
    decode_tok_s: f64,
    peak_memory_bytes: u64,
    active_memory_bytes: u64,
    route: RouteDelta,
}

fn tokens_digest(rows: &[Vec<i32>]) -> String {
    let mut hasher = Sha256::new();
    for row in rows {
        hasher.update((row.len() as u64).to_le_bytes());
        for id in row {
            hasher.update(id.to_le_bytes());
        }
    }
    let digest = hasher.finalize();
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

const CSV_HEADER: &str = "label,model,path,batch,prompt_target_len,prompt_tokens,max_tokens,\
generated_tokens,ttft_ms,prefill_tok_s,decode_ms,decode_tok_s,peak_memory_bytes,\
active_memory_bytes,route,adapter_built,installed,oxide_routes,native_routes,launches,\
last_fallback,env_fused,env_oxide,env_cuda_graphs,tokens_sha256_16,first_row_tokens";

impl Pass {
    fn fields(&self, args: &Args) -> Vec<(&'static str, String, bool)> {
        let generated: usize = self.tokens.iter().map(Vec::len).sum();
        let prefill_tok_s = if self.ttft_ms > 0.0 {
            (self.prompt_tokens * self.batch) as f64 / (self.ttft_ms / 1000.0)
        } else {
            0.0
        };
        let env = |name: &str| std::env::var(name).unwrap_or_default();
        let first_row = self.tokens.first().map_or_else(String::new, |row| {
            row.iter().map(i32::to_string).collect::<Vec<_>>().join(" ")
        });
        let model = args.model.file_name().map_or_else(
            || args.model.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        // (name, value, quote as a JSON string)
        vec![
            ("label", args.label.clone(), true),
            ("model", model, true),
            ("path", self.path.to_string(), true),
            ("batch", self.batch.to_string(), false),
            ("prompt_target_len", self.prompt_target.to_string(), false),
            ("prompt_tokens", self.prompt_tokens.to_string(), false),
            ("max_tokens", args.max_tokens.to_string(), false),
            ("generated_tokens", generated.to_string(), false),
            ("ttft_ms", format!("{:.3}", self.ttft_ms), false),
            ("prefill_tok_s", format!("{prefill_tok_s:.2}"), false),
            ("decode_ms", format!("{:.3}", self.decode_ms), false),
            ("decode_tok_s", format!("{:.3}", self.decode_tok_s), false),
            (
                "peak_memory_bytes",
                self.peak_memory_bytes.to_string(),
                false,
            ),
            (
                "active_memory_bytes",
                self.active_memory_bytes.to_string(),
                false,
            ),
            ("route", self.route.route().to_string(), true),
            ("adapter_built", self.route.built.to_string(), false),
            ("installed", self.route.installed.to_string(), false),
            ("oxide_routes", self.route.oxide_routes.to_string(), false),
            ("native_routes", self.route.native_routes.to_string(), false),
            ("launches", self.route.launches.to_string(), false),
            ("last_fallback", self.route.last_fallback.to_string(), true),
            ("env_fused", env("MLXCEL_FUSED_ADD_RMSNORM"), true),
            ("env_oxide", env("MLXCEL_CUDA_OXIDE_FUSED_NORM"), true),
            ("env_cuda_graphs", env("MLX_USE_CUDA_GRAPHS"), true),
            ("tokens_sha256_16", tokens_digest(&self.tokens), true),
            ("first_row_tokens", first_row, true),
        ]
    }

    fn emit(&self, args: &Args) -> Result<()> {
        let fields = self.fields(args);
        let json = fields
            .iter()
            .map(|(name, value, quote)| {
                if *quote {
                    format!("\"{name}\":{}", serde_json::Value::String(value.clone()))
                } else {
                    format!("\"{name}\":{value}")
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        println!("[oxide-ckpt-bench] {{{json}}}");
        if let Some(path) = &args.csv {
            let exists = path.exists();
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("cannot open {}", path.display()))?;
            if !exists {
                writeln!(file, "{CSV_HEADER}")?;
            }
            let row = fields
                .iter()
                .map(|(_, value, _)| value.replace(',', ";"))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(file, "{row}")?;
        }
        Ok(())
    }
}

/// Row `row` of a batch: the same deterministic corpus, started `17 * row`
/// tokens later so the rows differ, with the first token (BOS) kept.
fn batch_prompts(corpus: &[i32], len: usize, batch: usize) -> Vec<Vec<i32>> {
    (0..batch)
        .map(|row| {
            let offset = 17 * row;
            let mut tokens = corpus[offset..offset + len].to_vec();
            tokens[0] = corpus[0];
            tokens
        })
        .collect()
}

fn run_cli(args: &Args, prompts: &[(usize, Vec<i32>)], passes: &mut Vec<Pass>) -> Result<()> {
    let (model, _) = mlxcel::load_model(&args.model).context("failed to load model")?;
    let sampling = mlxcel::sampling::build_sampling_config(neutral_params(
        mlxcel::read_eos_token_ids(&args.model),
    ));
    for (target, corpus) in prompts {
        let tokens = &batch_prompts(corpus, cap_len(corpus, 1), 1)[0];
        // Warm up outside the measured window, exactly as the measured call
        // would, then measure with the peak counter and the route counters
        // reset to this pass.
        if args.warmup_tokens > 0 {
            bench_like_bench_decode(
                &model,
                tokens,
                args.warmup_tokens,
                0,
                &sampling,
                KVCacheMode::Fp16,
            )
            .context("warmup generation failed")?;
        }
        mlxcel_core::synchronize_default();
        mlxcel_core::memory::reset_peak_memory();
        let before = fused_norm_route_snapshot();
        let (generated, stats) = bench_like_bench_decode(
            &model,
            tokens,
            args.max_tokens,
            0,
            &sampling,
            KVCacheMode::Fp16,
        )
        .with_context(|| format!("cli pass failed at {target} prompt tokens"))?;
        let route = RouteDelta::between(before, fused_norm_route_snapshot());
        passes.push(Pass {
            path: "cli",
            batch: 1,
            prompt_target: *target,
            prompt_tokens: stats.prompt_tokens,
            ttft_ms: stats.prefill_time_ms,
            decode_ms: stats.decode_time_ms,
            decode_tok_s: stats.decode_tok_per_sec,
            tokens: vec![generated],
            peak_memory_bytes: mlxcel_core::memory::peak_memory(),
            active_memory_bytes: mlxcel_core::memory::active_memory(),
            route,
        });
        passes.last().expect("just pushed").emit(args)?;
    }
    Ok(())
}

/// The longest prompt `corpus` can give every row of a batch of `batch`.
fn cap_len(corpus: &[i32], batch: usize) -> usize {
    corpus.len() - 17 * (batch - 1)
}

struct RowRun {
    tokens: Vec<i32>,
    prompt_tokens: usize,
    /// Milliseconds from the batch start to this row's first token and to its
    /// completion.
    first_at_ms: f64,
    done_at_ms: f64,
    ttft_ms: f64,
}

fn run_batch(
    engine: &ServerEngine,
    prompts: &[Vec<i32>],
    sampling: &mlxcel_core::generate::SamplingConfig,
    max_tokens: usize,
) -> Result<Vec<RowRun>> {
    let barrier = Barrier::new(prompts.len());
    let origin = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .map(|tokens| {
                let barrier = &barrier;
                scope.spawn(move || -> Result<RowRun> {
                    barrier.wait();
                    let sent_ms = origin.elapsed().as_secs_f64() * 1000.0;
                    let run = engine.run(&ServerEngineRequest {
                        prompt_tokens: tokens,
                        sampling: sampling.clone(),
                        max_tokens,
                        ignore_eos: false,
                        prompt_cache: None,
                    })?;
                    Ok(RowRun {
                        first_at_ms: sent_ms + run.ttft_ms,
                        done_at_ms: sent_ms + run.ttft_ms + run.decode_ms,
                        ttft_ms: run.ttft_ms,
                        prompt_tokens: run.prompt_tokens,
                        tokens: run.tokens,
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("request thread panicked"))
            .collect()
    })
}

fn run_server(args: &Args, prompts: &[(usize, Vec<i32>)], passes: &mut Vec<Pass>) -> Result<()> {
    let engine = ServerEngine::start(&args.model, ServerEngineOptions::default())?;
    let sampling = mlxcel::sampling::build_sampling_config(neutral_params(
        mlxcel::read_eos_token_ids(&args.model),
    ));
    for (target, corpus) in prompts {
        let rows = batch_prompts(corpus, cap_len(corpus, args.batch), args.batch);
        if args.warmup_tokens > 0 {
            run_batch(&engine, &rows, &sampling, args.warmup_tokens)?;
        }
        mlxcel_core::memory::reset_peak_memory();
        let before = fused_norm_route_snapshot();
        let runs = run_batch(&engine, &rows, &sampling, args.max_tokens)
            .with_context(|| format!("server pass failed at {target} prompt tokens"))?;
        let route = RouteDelta::between(before, fused_norm_route_snapshot());
        let first = runs
            .iter()
            .map(|r| r.first_at_ms)
            .fold(f64::INFINITY, f64::min);
        let done = runs.iter().map(|r| r.done_at_ms).fold(0.0, f64::max);
        let window_tokens: usize = runs.iter().map(|r| r.tokens.len().saturating_sub(1)).sum();
        let decode_ms = done - first;
        let mut ttfts: Vec<f64> = runs.iter().map(|r| r.ttft_ms).collect();
        passes.push(Pass {
            path: "server",
            batch: args.batch,
            prompt_target: *target,
            prompt_tokens: runs[0].prompt_tokens,
            ttft_ms: median(&mut ttfts),
            decode_ms,
            decode_tok_s: mlxcel_core::generate::decode_tok_per_sec(window_tokens, decode_ms),
            tokens: runs.into_iter().map(|r| r.tokens).collect(),
            peak_memory_bytes: mlxcel_core::memory::peak_memory(),
            active_memory_bytes: mlxcel_core::memory::active_memory(),
            route,
        });
        passes.last().expect("just pushed").emit(args)?;
    }
    engine.shutdown()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.batch == 0 || (args.path == PathArg::Cli && args.batch != 1) {
        bail!("--batch must be 1 on the cli path and at least 1 on the server path");
    }
    mlxcel::cli::mla_latent_option::init_mla_latent_quant(None)?;
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
        // Extra tokens so every row of a batch can start at its own offset.
        let corpus = synthesize_prompt_tokens(&tokenizer, len + 17 * (args.batch - 1))?;
        prompts.push((target, corpus));
    }

    let mut passes = Vec::new();
    match args.path {
        PathArg::Cli => run_cli(&args, &prompts, &mut passes)?,
        PathArg::Server => run_server(&args, &prompts, &mut passes)?,
    }

    if let Some(expected) = &args.expect_route {
        let wrong: Vec<String> = passes
            .iter()
            .filter(|p| p.route.route() != expected)
            .map(|p| format!("{} prompt {}: {}", p.path, p.prompt_target, p.route.route()))
            .collect();
        if !wrong.is_empty() {
            bail!("expected route {expected}, got {}", wrong.join(", "));
        }
    }
    Ok(())
}
