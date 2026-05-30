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

//! Perf benchmark harness for the speculative drafter pairings shipped by
//! (sub-9).
//!
//! ## Scope today
//!
//! Captures **no-drafter baseline** throughput numbers for the two reachable
//! target models (`models/qwen3.5-4b-4bit`, `models/gemma-4-31b-it-4bit`).
//! These numbers are the denominator of every "speedup vs no-drafter" cell
//! in `docs/model_tests.md::Speculative drafters` and can be
//! captured today against real on-disk checkpoints.
//!
//! ## Scope deferred to
//!
//! Speculative-decoding numerators (`--kind mtp` / `--kind dflash`) require:
//!
//! 1. **A public way to construct the speculative target's per-layer cache**.
//!    For Qwen 3.5 that is `Qwen3NextCache` (the `pub(crate) fn make_caches`
//!    on `Qwen35Model`). For Gemma 4 that is the per-`SequenceId` cache slot
//!    on `Gemma4Wrapper`. Both are currently binary-private.
//! 2. **An `MtpTarget` impl on `Gemma4Wrapper`**. The hooks
//!    (`forward_with_speculative_sinks`, `rollback_speculative_cache`) are
//!    all public, but the trait adapter that wires them to
//!    `prefill_and_seed` / `verify_forward` / `verify_finalize` is the work
//!    explicitly deferred.
//! 3. **A lazy-bind fix for `DFlashDrafter`**. The upstream
//!    `z-lab/Qwen3.5-4B-DFlash` checkpoint omits `embed_tokens.weight`
//!    because upstream Python binds to the target's `embed_tokens` at
//!    `bind()` time, but the Rust loader currently requires the weight at
//!    construction. See `tests/speculative_parity.rs` for the diagnostic.
//!
//! All three are scoped into follow-up. Until they land, this binary
//! prints a clear `[DEFERRED]` row for the speculative paths instead of
//! silently emitting fake numbers.
//!
//! ## Invocation
//!
//! ```bash
//! # Baseline-only run (works today):
//! ./target/release/speculative_bench \
//!     --target models/qwen3.5-4b-4bit \
//!     --kind none \
//!     --batch 1 \
//!     --max-tokens 96 \
//!     --prompt "Explain Apple Silicon's unified memory in one short paragraph."
//!
//! # Full sweep across pairings (B = 1, 2, 4):
//! ./target/release/speculative_bench --sweep
//! ```
//!
//! Streaming output is enabled via `eprintln!` for the progress lines, so
//! piping through `tee` (e.g. `2>&1 | tee /tmp/bench.log`) preserves the
//! per-row timings even if the harness is interrupted mid-sweep.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use mlxcel::tokenizer::MlxcelTokenizer;
use mlxcel::{LanguageModel, SamplingConfig, initialize_runtime, load_model};
use mlxcel_core::generate::{CxxGenerator, GenerationStats};

/// Default 17-token prompt that matches the upstream MTP perf-table conditions
/// (`references/mlx-vlm/README.md`). Token count is approximate (depends on
/// the tokenizer), but the prompt structure is the same: a short instruction
/// + a moderately information-dense follow-on.
const DEFAULT_PROMPT: &str =
    "Explain Apple Silicon's unified memory architecture in one short paragraph.";

/// Default max-new-tokens cap. Matches the upstream perf-table conditions and
/// keeps each row's wall-clock under 3 minutes on M-class hardware.
const DEFAULT_MAX_TOKENS: usize = 96;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BenchKind {
    /// No-drafter baseline. The denominator of the speedup column. Always
    /// reachable today via `LanguageModel::forward`.
    None,
    /// MTP speculative path (Gemma 4 assistant). DEFERRED: requires
    /// `MtpTarget` impl on `Gemma4Wrapper` (follow-up).
    Mtp,
    /// DFlash speculative path (Qwen 3.5 DFlash). DEFERRED: requires
    /// (a) public cache-construction API on `Qwen35Model` and (b) lazy-bind
    /// fix on `DFlashDrafter` (follow-up).
    Dflash,
}

impl std::fmt::Display for BenchKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchKind::None => f.write_str("none"),
            BenchKind::Mtp => f.write_str("mtp"),
            BenchKind::Dflash => f.write_str("dflash"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "speculative_bench",
    about = "Speculative drafter perf benchmark.",
    long_about = "Captures no-drafter baseline tok/s for the speculative \
                  target models. Speculative paths are scaffolded but \
                  deferred to follow-up — see the module docs."
)]
struct Args {
    /// Path to the target model directory.
    #[arg(long, conflicts_with = "sweep")]
    target: Option<PathBuf>,

    /// Path to the drafter model directory (ignored when `--kind none`).
    #[arg(long)]
    draft: Option<PathBuf>,

    /// Speculative drafter kind. Default: `none` (baseline).
    #[arg(long, default_value = "none", value_enum)]
    kind: BenchKind,

    /// Batch size. The baseline path always runs B=1 today; this argument is
    /// recorded into the output table so a future B>1 run produces a
    /// consistent shape.
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Max-new-tokens cap (per row).
    #[arg(long, default_value_t = DEFAULT_MAX_TOKENS)]
    max_tokens: usize,

    /// Prompt to feed the model. Defaults to a 17-token-ish instruction.
    #[arg(long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Block size for the speculative path (ignored when `--kind none`).
    /// Mirrors the upstream defaults: 4 for MTP, 16 for DFlash.
    #[arg(long)]
    block_size: Option<u32>,

    /// Run the full sweep across reachable pairings and emit the Markdown
    /// table. Equivalent to `--target X --kind Y --batch B` over the
    /// catalog of pairings.
    #[arg(long, default_value_t = false)]
    sweep: bool,
}

/// A single sweep row's result.
///
/// `decode_ms` and `generated_tokens` are recorded for diagnostic / future
/// downstream consumers (e.g. a JSON export that pairs the table cell with
/// raw timing) even though they don't appear in the rendered Markdown
/// table. Allow `dead_code` so a future refactor that adds a JSON export
/// surface doesn't need to first cleanup these fields.
#[allow(dead_code)]
struct Row {
    pairing: String,
    target_dir: PathBuf,
    kind: BenchKind,
    batch: usize,
    block_size: Option<u32>,
    /// Tok/s, or `None` when the run was deferred / skipped.
    tok_per_sec: Option<f64>,
    /// Decode wall-clock in milliseconds (excludes prefill).
    decode_ms: Option<f64>,
    /// Number of generated tokens actually emitted.
    generated_tokens: Option<usize>,
    /// Speedup vs no-drafter baseline for the same `(target, batch)`. Filled
    /// after all rows are collected.
    speedup_vs_baseline: Option<f64>,
    /// `None` when the row ran successfully; otherwise a short message.
    status_note: Option<String>,
}

impl Row {
    fn deferred(
        pairing: &str,
        target: &Path,
        kind: BenchKind,
        batch: usize,
        block_size: Option<u32>,
        note: &str,
    ) -> Self {
        Self {
            pairing: pairing.to_string(),
            target_dir: target.to_path_buf(),
            kind,
            batch,
            block_size,
            tok_per_sec: None,
            decode_ms: None,
            generated_tokens: None,
            speedup_vs_baseline: None,
            status_note: Some(note.to_string()),
        }
    }
}

/// Repository of reachable pairings, parallel to `tests/speculative_parity.rs`.
struct Pairing {
    /// Human-readable name shown in the table.
    name: &'static str,
    target_subdir: &'static str,
    /// `Some(...)` for speculative pairings; `None` for baseline-only rows.
    draft_subdir: Option<&'static str>,
    kind: BenchKind,
    /// Drafter's `block_size` config setting.
    block_size: Option<u32>,
}

const REACHABLE_PAIRINGS: &[Pairing] = &[
    // Qwen 3.5 4B family — baseline + DFlash drafter.
    Pairing {
        name: "Qwen 3.5 4B (no drafter)",
        target_subdir: "qwen3.5-4b-4bit",
        draft_subdir: None,
        kind: BenchKind::None,
        block_size: None,
    },
    Pairing {
        name: "Qwen 3.5 4B + DFlash",
        target_subdir: "qwen3.5-4b-4bit",
        draft_subdir: Some("Qwen3.5-4B-DFlash"),
        kind: BenchKind::Dflash,
        block_size: Some(16),
    },
    // Gemma 4 31B family — baseline + MTP assistant drafter.
    Pairing {
        name: "Gemma 4 31B (no drafter)",
        target_subdir: "gemma-4-31b-it-4bit",
        draft_subdir: None,
        kind: BenchKind::None,
        block_size: None,
    },
    Pairing {
        name: "Gemma 4 31B + MTP assistant",
        target_subdir: "gemma-4-31b-it-4bit",
        draft_subdir: Some("gemma-4-31B-it-assistant-bf16"),
        kind: BenchKind::Mtp,
        block_size: Some(4),
    },
];

/// Resolve a model directory against the canonical `models/` layout,
/// matching `tests/common::repo_model_dir`. Falls back to a sibling
/// `mlxcel-internal` checkout (`../mlxcel-internal/models/<name>`) so the
/// binary can be run from a `git worktree`-created secondary working tree
/// even though `target/` and `models/` live in the primary tree.
fn resolve_model_dir(name: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let primary = manifest_dir.join("models").join(name);
    if primary.exists() {
        return primary;
    }
    let shared = manifest_dir
        .parent()
        .map(|p| p.join("mlxcel-internal").join("models").join(name))
        .unwrap_or_else(|| primary.clone());
    if shared.exists() {
        return shared;
    }
    primary
}

/// Encode a prompt with a tokenizer, mirroring `tests/tensor_parallel_real_models.rs`.
fn encode_prompt(tokenizer: &MlxcelTokenizer, prompt: &str) -> Vec<i32> {
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    tokenizer
        .encode(prompt, add_special)
        .expect("tokenizer.encode must succeed on a valid utf-8 prompt")
        .into_iter()
        .map(|t| t as i32)
        .collect()
}

/// Run the no-drafter baseline against a real on-disk target. Returns the
/// decode wall-clock and the number of generated tokens so the caller can
/// fill in a `Row`.
///
/// Streaming progress is logged via `eprintln!` so the parent shell sees
/// the row finish even on long real-model runs (avoids the 600s stream
/// watchdog the orchestrator notes).
fn run_baseline(target_dir: &Path, prompt: &str, max_tokens: usize) -> Result<(f64, usize)> {
    eprintln!("[bench/baseline] Loading target from {:?}", target_dir);

    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (model, tokenizer) = load_model(target_dir).context("load_model failed")?;
    let prompt_tokens = encode_prompt(&tokenizer, prompt);
    eprintln!(
        "[bench/baseline] Prompt {} tokens, max_new {}",
        prompt_tokens.len(),
        max_tokens
    );

    let num_layers = model.num_layers();

    // Warm-up: a single forward to pull lazy MLX kernels onto the GPU before
    // the timed run. Upstream MLX defers Metal kernel compilation until the
    // first call; without the warm-up the first generation reports an
    // inflated decode time. Bound the warm-up to 4 new tokens so it adds
    // negligible total wall-clock.
    {
        eprintln!("[bench/baseline] Warm-up (4 tokens)...");
        let mut warmup_gen = CxxGenerator::new(num_layers);
        let _ = warmup_gen.generate(&model, &prompt_tokens, 4, &SamplingConfig::greedy());
        mlxcel_core::synchronize_default();
    }

    eprintln!("[bench/baseline] Timed run starts");
    let mut generator = CxxGenerator::new(num_layers);
    let started = Instant::now();
    let (tokens, stats): (Vec<i32>, GenerationStats) = generator.generate_with_stats(
        &model,
        &prompt_tokens,
        max_tokens,
        &SamplingConfig::greedy(),
    );
    mlxcel_core::synchronize_default();
    let elapsed = started.elapsed();
    // `generate_with_stats` returns only the generated tokens (not the
    // prompt prepended) in `tokens`, and `GenerationStats::generated_tokens`
    // carries the count internally. The wall-clock used for tok/s is the
    // `decode_time_ms` field of GenerationStats, which excludes the
    // prefill (matches upstream `_dflash_rounds` / `_mtp_rounds` perf
    // measurement).
    let _ = elapsed; // kept for diagnostic logging only; tok/s uses stats.decode_*
    // `gen` is a reserved Rust 2024 keyword, so we use `generated` here.
    let generated = tokens.len();
    eprintln!(
        "[bench/baseline] Done: prompt={} prefill_ms={:.1} decode_ms={:.1} \
         generated={} tok/s={:.1} (wall {:.1}s)",
        stats.prompt_tokens,
        stats.prefill_time_ms,
        stats.decode_time_ms,
        generated,
        stats.decode_tok_per_sec,
        elapsed.as_secs_f64(),
    );
    Ok((stats.decode_time_ms, generated))
}

/// Render a Markdown perf table from the collected rows. Output goes to
/// stdout; the parent script captures this and pastes it into
/// `docs/model_tests.md`.
fn print_markdown_table(rows: &[Row]) {
    println!();
    println!("### Speculative drafter perf table");
    println!();
    println!("| Pairing | Kind | B | block_size | tok/s | speedup vs no-drafter | status |");
    println!("|---------|------|---|------------|-------|------------------------|--------|");
    for row in rows {
        let tok_s_cell = match row.tok_per_sec {
            Some(t) => format!("{t:.1}"),
            None => "—".to_string(),
        };
        let speedup_cell = match row.speedup_vs_baseline {
            Some(s) => format!("{s:.2}×"),
            None => "—".to_string(),
        };
        let block_cell = match row.block_size {
            Some(b) => b.to_string(),
            None => "—".to_string(),
        };
        let status_cell = row.status_note.as_deref().unwrap_or("ok").to_string();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            row.pairing, row.kind, row.batch, block_cell, tok_s_cell, speedup_cell, status_cell,
        );
    }
    println!();
    println!("Note: speculative rows are deferred to follow-up — see");
    println!("`docs/model_tests.md::Speculative drafters` for the");
    println!("wiring details. Baseline rows are real perf numbers captured on");
    println!("the host this binary ran on.");
}

/// Fill in `speedup_vs_baseline` for every row against the matching
/// `(target, batch)` baseline row, when both numerator and denominator are
/// available.
fn compute_speedups(rows: &mut [Row]) {
    // Build an index of baseline tok/s by (target_dir, batch).
    let baseline: std::collections::HashMap<(PathBuf, usize), f64> = rows
        .iter()
        .filter_map(|r| {
            if matches!(r.kind, BenchKind::None) {
                r.tok_per_sec.map(|t| ((r.target_dir.clone(), r.batch), t))
            } else {
                None
            }
        })
        .collect();
    for row in rows.iter_mut() {
        if matches!(row.kind, BenchKind::None) {
            // Baseline rows have speedup 1.00× by definition; render that
            // explicitly so the table cell is never empty for a successful
            // baseline.
            if row.tok_per_sec.is_some() {
                row.speedup_vs_baseline = Some(1.0);
            }
            continue;
        }
        let key = (row.target_dir.clone(), row.batch);
        if let (Some(this), Some(&base)) = (row.tok_per_sec, baseline.get(&key))
            && base > 0.0
        {
            row.speedup_vs_baseline = Some(this / base);
        }
    }
}

/// Build a `Row` for a single pairing using the supplied prompt + max_tokens.
fn bench_one_pairing(p: &Pairing, prompt: &str, batch: usize, max_tokens: usize) -> Row {
    let target_path = resolve_model_dir(p.target_subdir);
    if !target_path.exists() {
        return Row::deferred(
            p.name,
            &target_path,
            p.kind,
            batch,
            p.block_size,
            "target checkpoint missing on disk",
        );
    }

    // Speculative pairings still need their drafter directory on disk for
    // the eventual end-to-end run; report missing drafter as DEFERRED with
    // the dedicated note so the table makes the limitation visible.
    if let Some(draft_sub) = p.draft_subdir {
        let draft_path = resolve_model_dir(draft_sub);
        if !draft_path.exists() {
            return Row::deferred(
                p.name,
                &target_path,
                p.kind,
                batch,
                p.block_size,
                "drafter checkpoint missing on disk",
            );
        }
    }

    match p.kind {
        BenchKind::None => match run_baseline(&target_path, prompt, max_tokens) {
            Ok((decode_ms, generated)) => Row {
                pairing: p.name.to_string(),
                target_dir: target_path,
                kind: p.kind,
                batch,
                block_size: p.block_size,
                tok_per_sec: if decode_ms > 0.0 {
                    Some(generated as f64 / (decode_ms / 1000.0))
                } else {
                    None
                },
                decode_ms: Some(decode_ms),
                generated_tokens: Some(generated),
                speedup_vs_baseline: None,
                status_note: None,
            },
            Err(e) => Row::deferred(
                p.name,
                &target_path,
                p.kind,
                batch,
                p.block_size,
                &format!("baseline run failed: {e}"),
            ),
        },
        BenchKind::Mtp => Row::deferred(
            p.name,
            &target_path,
            p.kind,
            batch,
            p.block_size,
            "DEFERRED — needs MtpTarget for Gemma4Wrapper",
        ),
        BenchKind::Dflash => Row::deferred(
            p.name,
            &target_path,
            p.kind,
            batch,
            p.block_size,
            "DEFERRED — DFlash loader + public Qwen3NextCache API",
        ),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut rows: Vec<Row> = Vec::new();
    if args.sweep {
        eprintln!(
            "[bench] Sweep mode: benching {} pairings at B=1 with max_tokens={}",
            REACHABLE_PAIRINGS.len(),
            args.max_tokens,
        );
        for p in REACHABLE_PAIRINGS {
            let row = bench_one_pairing(p, &args.prompt, args.batch, args.max_tokens);
            eprintln!(
                "[bench] Finished pairing: {} -> tok/s={:?} status={:?}",
                row.pairing, row.tok_per_sec, row.status_note,
            );
            rows.push(row);
        }
    } else {
        let target = args
            .target
            .as_deref()
            .context("--target is required when --sweep is not set")?
            .to_path_buf();
        let pairing_name = format!("{} ({})", target.display(), args.kind);
        let synthetic = Pairing {
            name: Box::leak(pairing_name.into_boxed_str()),
            target_subdir: Box::leak(
                target
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "(target)".to_string())
                    .into_boxed_str(),
            ),
            draft_subdir: args
                .draft
                .as_ref()
                .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
                .map(|s| -> &'static str { Box::leak(s.into_boxed_str()) }),
            kind: args.kind,
            block_size: args.block_size,
        };
        // The synthetic pairing references the canonical layout via
        // `resolve_model_dir`; pass the explicit `--target` directly so an
        // operator who points at a non-canonical path is honored.
        let row = if synthetic.target_subdir == target.file_name().unwrap().to_string_lossy() {
            // Default canonical path; fall back to `bench_one_pairing` which
            // re-resolves via `resolve_model_dir`.
            bench_one_pairing(&synthetic, &args.prompt, args.batch, args.max_tokens)
        } else {
            // Operator passed an absolute path that does not match the
            // canonical `models/<name>` layout; honor it directly.
            match args.kind {
                BenchKind::None => match run_baseline(&target, &args.prompt, args.max_tokens) {
                    Ok((decode_ms, generated)) => Row {
                        pairing: synthetic.name.to_string(),
                        target_dir: target.clone(),
                        kind: args.kind,
                        batch: args.batch,
                        block_size: args.block_size,
                        tok_per_sec: if decode_ms > 0.0 {
                            Some(generated as f64 / (decode_ms / 1000.0))
                        } else {
                            None
                        },
                        decode_ms: Some(decode_ms),
                        generated_tokens: Some(generated),
                        speedup_vs_baseline: None,
                        status_note: None,
                    },
                    Err(e) => Row::deferred(
                        synthetic.name,
                        &target,
                        args.kind,
                        args.batch,
                        args.block_size,
                        &format!("baseline run failed: {e}"),
                    ),
                },
                BenchKind::Mtp | BenchKind::Dflash => Row::deferred(
                    synthetic.name,
                    &target,
                    args.kind,
                    args.batch,
                    args.block_size,
                    "DEFERRED — see module docs",
                ),
            }
        };
        rows.push(row);
    }

    compute_speedups(&mut rows);
    print_markdown_table(&rows);
    Ok(())
}
