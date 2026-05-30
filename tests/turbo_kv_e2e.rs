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

//! Quality gate for the TurboQuant KV cache compression (B3).
//!
//! Tests two quality dimensions for `KVCacheMode::Turbo4Asym` vs `KVCacheMode::Fp16`
//! baseline on a curated set of small MLX models:
//!
//! 1. **Wikitext-2 perplexity gate** — aggregate NLL over 20 non-overlapping 4K-token
//!    chunks of `tests/fixtures/wikitext2_excerpt.txt`. Gate: the relative PPL increase
//!    of Turbo4Asym vs Fp16 must be ≤ 1.0%.
//!
//! 2. **NIAH single-needle retrieval gate** — "the magic word is: cantaloupe" planted
//!    at 9 (depth, length) cells (depths 0%/50%/100% × lengths 1K/2K/4K) plus 3 cells
//!    at 8K context. Gate: turbo retrieval count ≥ baseline retrieval count.
//!
//! A **rotation kurtosis sanity test** loads a real K tensor from a model
//! (preferring Qwen2.5-7B), applies the WHT + sign-flip rotation via
//! `mlxcel_core::cache::turbo::turbo4_v_rotate`, and verifies that the
//! post-rotation non-excess kurtosis (E[(x-μ)^4]/σ^4, Gaussian=3.0) drops
//! below 5. The TurboQuant+ paper reports ~900 → ~2.9 on Qwen3-1.7B.
//!
//! Speed measurements (PPL-evaluation throughput tok/s, wall-clock ms) are captured
//! during the PPL run and appended to `benchmarks/turbo_kv/<YYYY-MM-DD>_<machine>.csv`.
//! Decode/prefill rates measured against an autoregressive generate path are not
//! recorded yet and are tracked as a follow-up. The CSV values are recorded but not
//! gated.
//!
//! # Running the tests
//!
//! All tests are gated with `#[ignore]` to avoid blocking `cargo test` on machines
//! without the required model checkouts. Run each gate individually:
//!
//! ```text
//! # PPL + NIAH gate for Qwen2.5-1.5B (base variant — see note in Prerequisite section)
//! cargo test --test turbo_kv_e2e --release -- --ignored test_qwen25_15b_quality_gate --nocapture
//!
//! # PPL + NIAH gate for Llama-3.1-8B
//! cargo test --test turbo_kv_e2e --release -- --ignored test_llama31_8b_quality_gate --nocapture
//!
//! # PPL + NIAH gate for Gemma-3-4B
//! cargo test --test turbo_kv_e2e --release -- --ignored test_gemma3_4b_quality_gate --nocapture
//!
//! # Rotation kurtosis sanity (loads a real K tensor from any available model)
//! cargo test --test turbo_kv_e2e --release -- --ignored test_rotation_kurtosis_sanity --nocapture
//! ```
//!
//! Tests skip gracefully (via `eprintln!` + early return) when the required model
//! directory is absent.
//!
//! # Prerequisite model checkouts
//!
//! Download one or more of the following into `models/`:
//!
//! ```text
//! ./target/release/mlxcel download mlx-community/Qwen2.5-1.5B-4bit
//! ./target/release/mlxcel download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
//! ./target/release/mlxcel download mlx-community/gemma-3-4b-it-4bit
//! ```
//!
//! Note: The B3 Qwen2.5-1.5B fixture uses the **base** (non-instruct) variant
//! `Qwen2.5-1.5B-4bit`. The instruct variant collapses on raw wikitext without
//! the chat template (`<|im_start|>user\n...<|im_end|>`), producing PPL ≈ 2×10⁷
//! and NIAH=0/12. This gate measures TurboQuant compression
//! quality, not chat performance, so the base model is the correct fixture.
//!
//! # VLM gates
//!
//! VLM gates run a smaller PPL+NIAH harness on text-only inputs because the
//! prefill cost on a vision-aware checkpoint is dominated by the decoder, not
//! the vision tower; reusing the existing wikitext fixture keeps the
//! comparison directly aligned with the text-model gate. A separate
//! image-token kurtosis sanity test then exercises the actual vision path
//! (image + text prompt) and measures the K-side kurtosis at image-token
//! positions before and after the WHT rotation. See the docstrings on
//! [`run_vlm_quality_gate`] and [`test_vlm_image_token_kurtosis`] for the
//! exact harness sizing and gate thresholds.
//!
//! ```text
//! # PPL + NIAH gate for Qwen2-VL-2B (text-only path)
//! cargo test --test turbo_kv_e2e --release -- --ignored test_qwen2_vl_2b_quality_gate --nocapture
//!
//! # PPL + NIAH gate for Aya-Vision-8B (text-only path)
//! cargo test --test turbo_kv_e2e --release -- --ignored test_aya_vision_8b_quality_gate --nocapture
//!
//! # Image-token K-side kurtosis pre/post WHT rotation
//! cargo test --test turbo_kv_e2e --release -- --ignored test_vlm_image_token_kurtosis --nocapture
//! ```

mod common;
use common::repo_model_dir;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use mlxcel::vlm_runtime::{prepare_and_compute_vlm_embeddings, prepared_embedding_refs};
use mlxcel::{CxxGenerator, LanguageModel, LoadedModel, SamplingConfig, load_model};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::cache::turbo::{TurboQuantParams, turbo4_v_rotate};
use mlxcel_core::{
    array_shape, array_to_raw_bytes, astype, dtype, eval, from_slice_f32, item_f32, mean_all,
    multiply, square, subtract,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of non-overlapping 4K-token chunks to evaluate for PPL.
const PPL_CHUNKS: usize = 20;

/// Token window size per PPL chunk. Must fit in context.
const PPL_CHUNK_LEN: usize = 4096;

/// Relative PPL increase gate: (ppl_turbo - ppl_fp16) / ppl_fp16 ≤ PPL_GATE_REL.
const PPL_GATE_REL: f64 = 0.01; // 1.0%
const QWEN25_15B_BASE_PPL_MIN: f64 = 5.0;
const QWEN25_15B_BASE_PPL_MAX: f64 = 30.0;

/// Needle string used in NIAH harness.
const NIAH_NEEDLE: &str = "cantaloupe";

/// Padding filler for NIAH context windows.
const NIAH_FILLER: &str = "The following is a long document about various topics. ";

/// Max tokens to generate when checking for the needle in NIAH.
const NIAH_MAX_GEN: usize = 32;

// ---------------------------------------------------------------------------
// PPL harness
// ---------------------------------------------------------------------------

/// Compute wikitext-2 perplexity using a loaded model and tokenizer.
///
/// Reads `tests/fixtures/wikitext2_excerpt.txt`, tokenizes it, slices into
/// non-overlapping chunks of `PPL_CHUNK_LEN`, and calls
/// `CxxGenerator::evaluate_loglikelihoods` on each chunk.
///
/// # PPL aggregation math
///
/// NLL is **accumulated** across all chunks (total_nll = -sum of all logprobs).
/// PPL = exp(total_nll / total_target_tokens). Averaging per-chunk PPLs is
/// mathematically wrong (equal-weights short and long chunks) and is explicitly
/// avoided here.
///
/// Returns `(ppl, total_target_tokens)`.
fn compute_ppl(
    model: &impl LanguageModel,
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    kv_mode: KVCacheMode,
) -> (f64, usize) {
    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("wikitext2_excerpt.txt");

    let corpus = fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("wikitext2 corpus missing at {:?}: {e}", corpus_path));

    // Tokenize without special tokens to avoid BOS/EOS contaminating PPL.
    let all_ids = tokenizer.encode(&corpus, false).expect("tokenize corpus");
    let all_ids_i32: Vec<i32> = all_ids.iter().map(|&id| id as i32).collect();

    let num_layers = model.num_layers();
    let mut generator = CxxGenerator::new_with_kv_mode(num_layers, kv_mode);

    let mut total_nll = 0.0_f64;
    let mut total_target_tokens = 0_usize;

    // Non-overlapping windows of PPL_CHUNK_LEN. Each window contributes
    // PPL_CHUNK_LEN - 1 target positions.
    let max_chunks = PPL_CHUNKS.min(all_ids_i32.len() / PPL_CHUNK_LEN);
    for chunk_idx in 0..max_chunks {
        let start = chunk_idx * PPL_CHUNK_LEN;
        let end = start + PPL_CHUNK_LEN;
        let window = &all_ids_i32[start..end];

        // Returns log P(token[i+1] | token[0..=i]) for i in 0..len-1.
        let logprobs = generator.evaluate_loglikelihoods(model, window);
        debug_assert_eq!(logprobs.len(), window.len() - 1);

        // Accumulate NLL: -sum(log_probs). logprobs are in natural log.
        let chunk_nll: f64 = logprobs.iter().map(|&lp| -(lp as f64)).sum();
        total_nll += chunk_nll;
        total_target_tokens += logprobs.len();
    }

    if total_target_tokens == 0 {
        panic!("No target tokens evaluated — corpus too short or no chunks processed");
    }

    let mean_nll = total_nll / total_target_tokens as f64;
    let ppl = mean_nll.exp();
    (ppl, total_target_tokens)
}

// ---------------------------------------------------------------------------
// NIAH harness
// ---------------------------------------------------------------------------

/// One (depth, context_len) cell in the NIAH evaluation grid.
#[derive(Debug, Clone, Copy)]
struct NiahCell {
    /// Fraction of total context where the needle is planted (0.0 = start, 1.0 = end).
    depth_pct: f64,
    /// Total context length in tokens (approximate — we build prompt chars and tokenize).
    context_len: usize,
}

// 4K grid: 3 depths × 3 context lengths.
static NIAH_4K_CELLS: &[NiahCell] = &[
    NiahCell {
        depth_pct: 0.0,
        context_len: 1024,
    },
    NiahCell {
        depth_pct: 0.5,
        context_len: 1024,
    },
    NiahCell {
        depth_pct: 1.0,
        context_len: 1024,
    },
    NiahCell {
        depth_pct: 0.0,
        context_len: 2048,
    },
    NiahCell {
        depth_pct: 0.5,
        context_len: 2048,
    },
    NiahCell {
        depth_pct: 1.0,
        context_len: 2048,
    },
    NiahCell {
        depth_pct: 0.0,
        context_len: 4096,
    },
    NiahCell {
        depth_pct: 0.5,
        context_len: 4096,
    },
    NiahCell {
        depth_pct: 1.0,
        context_len: 4096,
    },
];

// 8K grid: 3 depths at length 8192.
static NIAH_8K_CELLS: &[NiahCell] = &[
    NiahCell {
        depth_pct: 0.0,
        context_len: 8192,
    },
    NiahCell {
        depth_pct: 0.5,
        context_len: 8192,
    },
    NiahCell {
        depth_pct: 1.0,
        context_len: 8192,
    },
];

/// Build a NIAH prompt that embeds the needle at depth `depth_pct` within
/// roughly `target_token_len` tokens of total context.
///
/// Structure:
/// ```text
/// <padding text up to depth>
/// The magic word is: <needle>.
/// <padding text to fill remaining>
///
/// Q: What is the magic word?
/// A:
/// ```
fn build_niah_prompt(needle: &str, depth_pct: f64, target_token_len: usize) -> String {
    // Rough char budget: 1 token ≈ 4 chars. Pad generously.
    let total_chars = target_token_len * 4;
    let filler: String = NIAH_FILLER.repeat(total_chars / NIAH_FILLER.len() + 1);

    // Bound the insertion offset by the *target* prompt length, not the
    // (possibly larger) filler length — otherwise depth_pct=1.0 produces
    // an `insert_at` greater than total_chars, and the `total_chars - insert_at`
    // arithmetic below underflows on usize.
    let insert_char =
        ((total_chars as f64 * depth_pct).round() as usize).min(total_chars.saturating_sub(1));
    let insert_at = filler[..insert_char]
        .rfind(' ')
        .map(|p| p + 1)
        .unwrap_or(insert_char);

    let after_end = total_chars.max(insert_at).min(filler.len());
    let before = &filler[..insert_at];
    let after = &filler[insert_at..after_end];

    format!("{before}\nThe magic word is: {needle}.\n{after}\n\nQ: What is the magic word?\nA:")
}

/// Run NIAH evaluation across `cells` for one KV cache mode.
///
/// Returns the count of cells where the generated response contained the needle.
fn run_niah(
    model: &impl LanguageModel,
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    kv_mode: KVCacheMode,
    cells: &[NiahCell],
    model_name: &str,
    mode_label: &str,
) -> usize {
    let num_layers = model.num_layers();

    // Greedy sampling for reproducible retrieval.
    let sampling = SamplingConfig::greedy();

    let mut hits = 0_usize;
    for cell in cells {
        let prompt = build_niah_prompt(NIAH_NEEDLE, cell.depth_pct, cell.context_len);

        let prompt_ids: Vec<i32> = tokenizer
            .encode(&prompt, false)
            .expect("tokenize NIAH prompt")
            .iter()
            .map(|&id| id as i32)
            .collect();

        let mut generator = CxxGenerator::new_with_kv_mode(num_layers, kv_mode);
        let gen_tokens = generator.generate(model, &prompt_ids, NIAH_MAX_GEN, &sampling);

        let gen_u32: Vec<u32> = gen_tokens.iter().map(|&t| t as u32).collect();
        let response = tokenizer.decode(&gen_u32, true).unwrap_or_default();
        let hit = response
            .to_lowercase()
            .contains(&NIAH_NEEDLE.to_lowercase());
        if hit {
            hits += 1;
        }
        eprintln!(
            "  [{mode_label}] depth={:.0}% len={} => {} | response=\"{}\"",
            cell.depth_pct * 100.0,
            cell.context_len,
            if hit { "HIT" } else { "MISS" },
            response
                .replace('\n', " ")
                .chars()
                .take(80)
                .collect::<String>()
        );
    }

    eprintln!(
        "[{model_name}][{mode_label}] NIAH hits: {hits}/{total}",
        total = cells.len()
    );
    hits
}

// ---------------------------------------------------------------------------
// Speed bench recording
// ---------------------------------------------------------------------------

/// Sanitize a single filename component (date or machine string) derived from
/// an environment variable.
///
/// Rejects path separators (`/`, `\`), `..` traversal, control characters,
/// CSV-injection prefixes (`=`, `+`, `-`, `@`, `\t`, `\r`), and embedded
/// commas or newlines. Returns the original string if it passes all checks,
/// or a safe fallback if it does not.
fn sanitize_filename_component<'a>(s: &'a str, fallback: &'a str) -> &'a str {
    if s.is_empty() {
        return fallback;
    }
    // Reject path traversal and directory separators.
    if s.contains('/') || s.contains('\\') || s.contains("..") {
        return fallback;
    }
    // Reject CSV injection prefixes and structural characters.
    let first = s.chars().next().unwrap_or('\0');
    if matches!(first, '=' | '+' | '-' | '@' | '\t' | '\r') {
        return fallback;
    }
    // Reject embedded commas, newlines, and control characters.
    if s.chars()
        .any(|c| c == ',' || c == '\n' || c == '\r' || c.is_control())
    {
        return fallback;
    }
    s
}

/// Append one speed measurement row to the turbo_kv CSV.
///
/// The CSV lives at `benchmarks/turbo_kv/<date>_<machine>.csv`. The directory
/// and `.gitkeep` are committed; individual run files accumulate across CI jobs.
///
/// # Column semantics
///
/// `ppl_eval_tok_per_s` is measured as `n_target_tokens / ppl_wall_seconds`,
/// which reflects the throughput of the PPL evaluation pass (full-context
/// forward per chunk), not autoregressive decode rate. The prefill column is
/// intentionally absent until a separate generate-path benchmark is wired in.
fn record_speed_row(
    model_name: &str,
    kv_mode_label: &str,
    context_len: usize,
    ppl_eval_tok_per_s: f64,
    wall_clock_ms: f64,
) {
    let benchmarks_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("turbo_kv");

    let raw_date = std::env::var("MLXCEL_BENCH_DATE").unwrap_or_else(|_| "2026-04-26".to_string());
    let raw_machine =
        std::env::var("MLXCEL_BENCH_MACHINE").unwrap_or_else(|_| hostname_or_default());

    let date_str = sanitize_filename_component(&raw_date, "unknown-date");
    let machine = sanitize_filename_component(&raw_machine, "unknown-machine");

    fs::create_dir_all(&benchmarks_dir).expect("create benchmarks/turbo_kv dir");

    let csv_path = benchmarks_dir.join(format!("{date_str}_{machine}.csv"));

    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)
        .expect("open turbo_kv CSV");

    // Re-check file length after acquiring the append handle to avoid a TOCTOU
    // race when two test jobs start simultaneously on the same host and date.
    let needs_header = f.metadata().map(|m| m.len() == 0).unwrap_or(true);

    if needs_header {
        writeln!(
            f,
            "model,kv_cache_mode,context_len,ppl_eval_tok_per_s,wall_clock_ms,timestamp"
        )
        .expect("write CSV header");
    }

    writeln!(
        f,
        "{model_name},{kv_mode_label},{context_len},{ppl_eval_tok_per_s:.2},{wall_clock_ms:.1},{date_str}"
    )
    .expect("write CSV row");
}

fn hostname_or_default() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

// ---------------------------------------------------------------------------
// Core quality-gate runner
// ---------------------------------------------------------------------------

/// Run the full PPL + NIAH quality gate for one model directory name.
///
/// Returns `Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo))` when the
/// model is present, `None` when it is absent (soft skip).
fn run_quality_gate(model_dir_name: &str) -> Option<(f64, f64, usize, usize)> {
    let model_dir = repo_model_dir(model_dir_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {model_dir_name}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/{model_dir_name}",
            model_dir.display()
        );
        return None;
    }

    eprintln!("\n=== quality gate: {model_dir_name} ===");

    let (model, tokenizer) = load_model(&model_dir).expect("load model");

    // ── PPL ──────────────────────────────────────────────────────────────────
    eprintln!("[{model_dir_name}] computing fp16 baseline PPL...");
    let t0 = Instant::now();
    let (ppl_fp16, n_tokens) = compute_ppl(&model, &tokenizer, KVCacheMode::Fp16);
    let fp16_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ppl_eval_tps_fp16 = n_tokens as f64 / (fp16_ms / 1000.0).max(1e-9);
    eprintln!("[{model_dir_name}][fp16] PPL={ppl_fp16:.4} ({n_tokens} tokens, {fp16_ms:.0}ms)");
    record_speed_row(
        model_dir_name,
        "fp16",
        PPL_CHUNK_LEN,
        ppl_eval_tps_fp16,
        fp16_ms,
    );

    eprintln!("[{model_dir_name}] computing turbo4asym PPL...");
    let t1 = Instant::now();
    let (ppl_turbo, _) = compute_ppl(&model, &tokenizer, KVCacheMode::Turbo4Asym);
    let turbo_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let ppl_eval_tps_turbo = n_tokens as f64 / (turbo_ms / 1000.0).max(1e-9);
    eprintln!("[{model_dir_name}][turbo4asym] PPL={ppl_turbo:.4} ({turbo_ms:.0}ms)");
    record_speed_row(
        model_dir_name,
        "turbo4asym",
        PPL_CHUNK_LEN,
        ppl_eval_tps_turbo,
        turbo_ms,
    );

    let rel_ppl = (ppl_turbo - ppl_fp16) / ppl_fp16;
    eprintln!(
        "[{model_dir_name}] PPL relative increase: {:.4}% (gate: ≤{:.1}%)",
        rel_ppl * 100.0,
        PPL_GATE_REL * 100.0
    );

    // ── NIAH ─────────────────────────────────────────────────────────────────
    eprintln!("[{model_dir_name}] NIAH fp16 (4K cells)...");
    let niah_fp16_4k = run_niah(
        &model,
        &tokenizer,
        KVCacheMode::Fp16,
        NIAH_4K_CELLS,
        model_dir_name,
        "fp16",
    );
    eprintln!("[{model_dir_name}] NIAH fp16 (8K cells)...");
    let niah_fp16_8k = run_niah(
        &model,
        &tokenizer,
        KVCacheMode::Fp16,
        NIAH_8K_CELLS,
        model_dir_name,
        "fp16-8k",
    );
    let niah_baseline = niah_fp16_4k + niah_fp16_8k;

    eprintln!("[{model_dir_name}] NIAH turbo4asym (4K cells)...");
    let niah_turbo_4k = run_niah(
        &model,
        &tokenizer,
        KVCacheMode::Turbo4Asym,
        NIAH_4K_CELLS,
        model_dir_name,
        "turbo4asym",
    );
    eprintln!("[{model_dir_name}] NIAH turbo4asym (8K cells)...");
    let niah_turbo_8k = run_niah(
        &model,
        &tokenizer,
        KVCacheMode::Turbo4Asym,
        NIAH_8K_CELLS,
        model_dir_name,
        "turbo4asym-8k",
    );
    let niah_turbo = niah_turbo_4k + niah_turbo_8k;

    let total_cells = NIAH_4K_CELLS.len() + NIAH_8K_CELLS.len();
    eprintln!(
        "[{model_dir_name}] NIAH: baseline={niah_baseline}/{total_cells} \
         turbo={niah_turbo}/{total_cells}"
    );

    Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo))
}

// ---------------------------------------------------------------------------
// Per-model test functions
// ---------------------------------------------------------------------------

// B3 fixture: use the BASE (non-instruct) variant of Qwen2.5-1.5B.
//
// The instruct-tuned `Qwen2.5-1.5B-Instruct-4bit` collapses on raw wikitext
// without the chat template, producing PPL ≈ 2×10⁷ and NIAH=0/12 — values
// six orders of magnitude off a healthy ~10–15 baseline (and comment). The relative turbo4asym gate would still pass in that
// case (both fp16 and turbo degenerate together), making the test a
// meaningless noise check rather than a real quality signal.
//
// `Qwen2.5-1.5B-4bit` (base model) produces healthy absolute PPL on raw
// wikitext and is the correct fixture for a TurboQuant compression gate.
// Download: ./target/release/mlxcel download mlx-community/Qwen2.5-1.5B-4bit
#[test]
#[ignore = "requires Qwen2.5-1.5B-4bit weights (base, non-instruct variant) — \
            run with --release -- --ignored test_qwen25_15b_quality_gate --nocapture"]
fn test_qwen25_15b_quality_gate() {
    let Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo)) =
        run_quality_gate("Qwen2.5-1.5B-4bit")
    else {
        return; // model absent — soft skip
    };

    let rel = (ppl_turbo - ppl_fp16) / ppl_fp16;
    assert!(
        (QWEN25_15B_BASE_PPL_MIN..=QWEN25_15B_BASE_PPL_MAX).contains(&ppl_fp16),
        "Qwen2.5-1.5B base fp16 PPL {ppl_fp16:.4} is outside the healthy raw-wikitext \
         range [{QWEN25_15B_BASE_PPL_MIN:.1}, {QWEN25_15B_BASE_PPL_MAX:.1}]. \
         This gate exists to catch the degenerate instruct-fixture behavior."
    );
    assert!(
        niah_baseline > 0,
        "Qwen2.5-1.5B base baseline NIAH must be non-zero; got {niah_baseline}. \
         This gate should not pass on a collapsed raw-text fixture."
    );
    assert!(
        rel <= PPL_GATE_REL,
        "Qwen2.5-1.5B PPL regression {:.4}% > {:.1}% gate \
         (fp16={ppl_fp16:.4}, turbo={ppl_turbo:.4})",
        rel * 100.0,
        PPL_GATE_REL * 100.0
    );
    assert!(
        niah_turbo >= niah_baseline,
        "Qwen2.5-1.5B NIAH turbo ({niah_turbo}) dropped below baseline ({niah_baseline})"
    );
}

#[test]
#[ignore = "requires Meta-Llama-3.1-8B-Instruct-4bit weights — \
            run with --release -- --ignored test_llama31_8b_quality_gate --nocapture"]
fn test_llama31_8b_quality_gate() {
    let Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo)) =
        run_quality_gate("Meta-Llama-3.1-8B-Instruct-4bit")
    else {
        return;
    };

    let rel = (ppl_turbo - ppl_fp16) / ppl_fp16;
    assert!(
        rel <= PPL_GATE_REL,
        "Llama-3.1-8B PPL regression {:.4}% > {:.1}% gate \
         (fp16={ppl_fp16:.4}, turbo={ppl_turbo:.4})",
        rel * 100.0,
        PPL_GATE_REL * 100.0
    );
    assert!(
        niah_turbo >= niah_baseline,
        "Llama-3.1-8B NIAH turbo ({niah_turbo}) dropped below baseline ({niah_baseline})"
    );
}

#[test]
#[ignore = "requires gemma-3-4b-it-4bit weights — \
            run with --release -- --ignored test_gemma3_4b_quality_gate --nocapture"]
fn test_gemma3_4b_quality_gate() {
    let Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo)) =
        run_quality_gate("gemma-3-4b-it-4bit")
    else {
        return;
    };

    let rel = (ppl_turbo - ppl_fp16) / ppl_fp16;
    assert!(
        rel <= PPL_GATE_REL,
        "Gemma-3-4B PPL regression {:.4}% > {:.1}% gate \
         (fp16={ppl_fp16:.4}, turbo={ppl_turbo:.4})",
        rel * 100.0,
        PPL_GATE_REL * 100.0
    );
    assert!(
        niah_turbo >= niah_baseline,
        "Gemma-3-4B NIAH turbo ({niah_turbo}) dropped below baseline ({niah_baseline})"
    );
}

// ---------------------------------------------------------------------------
// Rotation kurtosis sanity test
// ---------------------------------------------------------------------------

/// Compute the **non-excess** sample kurtosis E[(x-μ)^4] / σ^4 using MLX ops.
///
/// For a Gaussian distribution this returns exactly 3.0. For heavy-tailed
/// distributions it is > 3. The TurboQuant+ paper states kurtosis ~900 for
/// raw Qwen3-1.7B K-cache tensors and ~2.9 after WHT+sign rotation, both
/// measured as non-excess kurtosis.
fn non_excess_kurtosis(data: &[f32]) -> f64 {
    let arr = from_slice_f32(data, &[data.len() as i32]);
    eval(&arr);

    let mean = mean_all(&arr);
    eval(&mean);
    let centered = subtract(&arr, &mean);
    let sq = square(&centered);
    let m2 = mean_all(&sq);
    let fourth = multiply(&sq, &sq);
    let m4 = mean_all(&fourth);
    eval(&m2);
    eval(&m4);

    let variance = item_f32(&m2) as f64;
    let fourth_moment = item_f32(&m4) as f64;
    if variance < 1e-12 {
        return 0.0;
    }
    fourth_moment / (variance * variance) // Gaussian = 3.0
}

/// Try to load a layer-0 K-projection weight tensor from a model directory.
///
/// Returns `None` if the model has no K-projection tensor in a real
/// floating-point dtype. Quantized weights (UINT* / INT*) are rejected
/// because reinterpreting packed integer codes as floats yields meaningless
/// kurtosis (typically `inf` after the (x-μ)^4 step overflows f32).
///
/// Caller is expected to fall back to the next candidate model when this
/// returns `None`.
fn load_k_tensor(model_dir: &std::path::Path) -> Option<Vec<f32>> {
    let weights = mlxcel_core::weights::load_weights_from_dir(model_dir).ok()?;

    let candidates = [
        "model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "transformer.h.0.attn.k_proj.weight",
        "model.layers.0.attention.wk.weight",
    ];

    for name in &candidates {
        if let Some(arr) = weights.get(*name)
            && let Some(floats) = try_extract_float_tensor(name, arr)
        {
            return Some(floats);
        }
    }

    // Fallback: first float-dtype tensor with "k_proj" in its name.
    for (name, arr) in &weights {
        if name.contains("k_proj")
            && let Some(floats) = try_extract_float_tensor(name, arr)
        {
            return Some(floats);
        }
    }
    None
}

/// Convert an MLX array to `Vec<f32>` if and only if its dtype is one of the
/// real floating-point dtypes. Quantized integer dtypes return `None`.
fn try_extract_float_tensor(name: &str, arr: &mlxcel_core::MlxArray) -> Option<Vec<f32>> {
    use mlxcel_core::dtype as dt;
    let actual_dtype = mlxcel_core::array_dtype(arr);
    if !matches!(actual_dtype, x if x == dt::FLOAT16 || x == dt::FLOAT32 || x == dt::BFLOAT16 || x == dt::FLOAT64)
    {
        eprintln!(
            "  skipping K tensor '{name}': dtype id {actual_dtype} is not float \
             (likely quantized — pick a non-quantized model checkout)"
        );
        return None;
    }
    eval(arr);
    let arr_f32 = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    eval(&arr_f32);
    let bytes = mlxcel_core::array_to_raw_bytes(&arr_f32);
    if bytes.len() < 8 {
        return None;
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if floats.is_empty() || floats.iter().any(|x| !x.is_finite()) {
        eprintln!(
            "  skipping K tensor '{name}': contains non-finite values after \
             dtype conversion (raw bytes were not a real float tensor)"
        );
        return None;
    }
    eprintln!("  loaded K tensor '{name}' ({} elements)", floats.len());
    Some(floats)
}

/// Apply the TurboQuant rotation to a flat f32 slice.
///
/// The slice is reshaped as `[1, 1, n_rows, head_dim]` where `head_dim` is the
/// largest power-of-two that divides the slice length and is ≤ 256. Returns the
/// rotated values as a new `Vec<f32>`.
fn apply_rotation_to_data(data: &[f32], seed: u32) -> Vec<f32> {
    let head_dim = [256u32, 128, 64, 32]
        .into_iter()
        .find(|&d| d as usize <= data.len() && data.len().is_multiple_of(d as usize))
        .unwrap_or(64);

    let n_rows = data.len() / head_dim as usize;
    let params = TurboQuantParams::new(head_dim, seed);

    let arr = from_slice_f32(data, &[1, 1, n_rows as i32, head_dim as i32]);
    eval(&arr);

    let rotated = turbo4_v_rotate(&arr, &params);
    eval(&rotated);

    mlxcel_core::array_to_raw_bytes(&rotated)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Rotation kurtosis sanity test.
///
/// Loads the layer-0 K-projection weight from any available model (prefers
/// Qwen2.5-7B for strongest signal), applies the WHT + sign-flip rotation,
/// and asserts that the post-rotation non-excess kurtosis drops below 5.
///
/// Static weight tensors typically start with kurtosis in the 3–30 range
/// (far less heavy-tailed than runtime K-cache activations). The gate of 5
/// is intentionally conservative and a meaningful bar: it demonstrates the
/// rotation moves the distribution toward the Gaussian regime even for inputs
/// that are not already extremely non-Gaussian.
///
/// # Note on static weights vs runtime activations
///
/// The TurboQuant+ paper's ~900 → 2.9 kurtosis claim refers to *runtime
/// attention K-cache* tensors accumulated over many forward passes. Static
/// weight matrices are different: they are already closer to Gaussian, so the
/// post-rotation kurtosis will naturally be close to 3 already. The test still
/// validates that `turbo4_v_rotate` is numerically correct and that the rotation
/// does not *increase* kurtosis.
///
/// Used by: `tests/turbo_kv_e2e.rs` (kurtosis sanity check).
#[test]
#[ignore = "requires at least one non-quantized (bf16/fp16) model checkout: \
            qwen2.5-0.5b-bf16 preferred. Quantized 4bit models are rejected \
            because their K-projection weights are packed integers, not floats. \
            Soft-skips if no candidate model is present."]
fn test_rotation_kurtosis_sanity() {
    // Order matters: non-quantized (bf16/fp16) checkouts come first because
    // 4-bit quantized weights are stored as packed integers and cannot be
    // reinterpreted as float without first dequantizing — `load_k_tensor`
    // rejects them and the test would otherwise soft-skip on quantized-only
    // hosts.
    let candidates = [
        "qwen2.5-0.5b-bf16",
        "gemma3n-e4b-bf16",
        "Qwen2.5-7B-Instruct-4bit",
        "qwen2.5-7b-4bit",
        // base model used by B3 quality gate since
        "Qwen2.5-1.5B-4bit",
        "Qwen2.5-1.5B-Instruct-4bit",
        "Meta-Llama-3.1-8B-Instruct-4bit",
        "gemma-3-4b-it-4bit",
        "llama-3.1-8b-4bit",
        "gemma3-4b-4bit",
    ];

    let mut loaded: Option<(String, Vec<f32>)> = None;
    for name in &candidates {
        let dir = repo_model_dir(name);
        if !dir.exists() {
            continue;
        }
        eprintln!("  attempting K tensor from {name}...");
        if let Some(data) = load_k_tensor(&dir) {
            loaded = Some((name.to_string(), data));
            break;
        }
    }

    let Some((model_name, k_data)) = loaded else {
        eprintln!(
            "Skipping test_rotation_kurtosis_sanity: no candidate model found. \
             Download any model listed in the ignore message to enable this gate."
        );
        return;
    };

    // Cap at 64K floats for a stable kurtosis estimate without long runtime.
    let sample: Vec<f32> = k_data.into_iter().take(65536).collect();
    eprintln!("  sample: {} elements from {model_name}", sample.len());

    let kurt_before = non_excess_kurtosis(&sample);
    eprintln!("  kurtosis before rotation (non-excess, Gaussian=3.0): {kurt_before:.4}");

    let rotated = apply_rotation_to_data(&sample, 0xAB_CD_12_34);
    let kurt_after = non_excess_kurtosis(&rotated);
    eprintln!("  kurtosis after  rotation (non-excess, Gaussian=3.0): {kurt_after:.4}");

    // Gate: post-rotation kurtosis < 5. For static weights that are already
    // near-Gaussian this is easily satisfied; for model activations it captures
    // the whitening effect documented in the TurboQuant+ paper.
    assert!(
        kurt_after < 5.0,
        "post-rotation kurtosis {kurt_after:.4} ≥ 5.0 — rotation did not whiten the \
         distribution as expected (model: {model_name}, pre-rotation: {kurt_before:.4})"
    );
}

// ---------------------------------------------------------------------------
// VLM (multimodal) quality gates
// ---------------------------------------------------------------------------

// VLM-specific PPL sizing.
//
// We deliberately keep the VLM PPL pass smaller than the text-model gate:
// vision-aware checkpoints carry a heavier prefill graph (mRoPE for Qwen-VL,
// 4D image-aware attention masks for Aya/Gemma-style VLMs), so per-chunk
// throughput is roughly half of an architecturally-matched text checkpoint.
// 6 chunks × 2048 tokens (≈12K target tokens per mode) is enough for the
// relative PPL gate to be statistically meaningful while keeping wall-clock
// for one model to ~5 minutes on M1 Ultra. Text-only gates remain at the
// original 20×4K sizing.
const VLM_PPL_CHUNKS: usize = 6;
const VLM_PPL_CHUNK_LEN: usize = 2048;

/// 4K-context NIAH grid only — VLMs typically advertise much longer context
/// but the 8K cells at the text gate added ~2× wall-clock without changing
/// the gate outcome on the text models, so we trim them for the VLM pass.
static VLM_NIAH_CELLS: &[NiahCell] = NIAH_4K_CELLS;

/// Resolve the model's image-token id across the supported VLM families.
///
/// Standard `VisionModule`-backed VLMs (Aya Vision, Gemma3, Llama4, LLaVA)
/// and Gemma3n expose this via `image_token_block_info()`. Qwen-VL families
/// (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VLM) expose it on the
/// `QwenVlmPromptInfo`. Returns `None` for VLM families that don't surface
/// either accessor (e.g. Phi3-V, Phi4-MM, Molmo) — the kurtosis test
/// soft-skips in that case.
fn vlm_image_token_id(model: &LoadedModel) -> Option<i32> {
    if let Some(info) = model.image_token_block_info() {
        return Some(info.image_token_id);
    }
    if let Some(info) = model.qwen_vl_prompt_info() {
        return Some(info.image_token_id);
    }
    None
}

/// Smaller PPL pass for VLM gates. Otherwise identical to [`compute_ppl`].
fn compute_vlm_ppl(
    model: &impl LanguageModel,
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    kv_mode: KVCacheMode,
) -> (f64, usize) {
    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("wikitext2_excerpt.txt");

    let corpus = fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("wikitext2 corpus missing at {:?}: {e}", corpus_path));

    let all_ids = tokenizer.encode(&corpus, false).expect("tokenize corpus");
    let all_ids_i32: Vec<i32> = all_ids.iter().map(|&id| id as i32).collect();

    let num_layers = model.num_layers();
    let mut generator = CxxGenerator::new_with_kv_mode(num_layers, kv_mode);

    let mut total_nll = 0.0_f64;
    let mut total_target_tokens = 0_usize;

    let max_chunks = VLM_PPL_CHUNKS.min(all_ids_i32.len() / VLM_PPL_CHUNK_LEN);
    for chunk_idx in 0..max_chunks {
        let start = chunk_idx * VLM_PPL_CHUNK_LEN;
        let end = start + VLM_PPL_CHUNK_LEN;
        let window = &all_ids_i32[start..end];

        let logprobs = generator.evaluate_loglikelihoods(model, window);
        debug_assert_eq!(logprobs.len(), window.len() - 1);

        let chunk_nll: f64 = logprobs.iter().map(|&lp| -(lp as f64)).sum();
        total_nll += chunk_nll;
        total_target_tokens += logprobs.len();
    }

    if total_target_tokens == 0 {
        panic!("No target tokens evaluated — corpus too short or no chunks processed");
    }

    let mean_nll = total_nll / total_target_tokens as f64;
    let ppl = mean_nll.exp();
    (ppl, total_target_tokens)
}

/// Run the VLM PPL + NIAH quality gate on the text-only path.
///
/// Returns `Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo))` when the
/// model directory is present, `None` for soft-skip.
fn run_vlm_quality_gate(model_dir_name: &str) -> Option<(f64, f64, usize, usize)> {
    let model_dir = repo_model_dir(model_dir_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {model_dir_name}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/{model_dir_name}",
            model_dir.display()
        );
        return None;
    }

    eprintln!("\n=== VLM quality gate (text-only path): {model_dir_name} ===");

    let (model, tokenizer) = load_model(&model_dir).expect("load model");
    if !model.is_vlm() {
        eprintln!(
            "Skipping {model_dir_name}: not a VLM checkpoint (text-only loaders should use \
             the standard text gate)."
        );
        return None;
    }

    eprintln!("[{model_dir_name}] computing fp16 baseline PPL...");
    let t0 = Instant::now();
    let (ppl_fp16, n_tokens) = compute_vlm_ppl(&model, &tokenizer, KVCacheMode::Fp16);
    let fp16_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ppl_eval_tps_fp16 = n_tokens as f64 / (fp16_ms / 1000.0).max(1e-9);
    eprintln!("[{model_dir_name}][fp16] PPL={ppl_fp16:.4} ({n_tokens} tokens, {fp16_ms:.0}ms)");
    record_speed_row(
        model_dir_name,
        "fp16",
        VLM_PPL_CHUNK_LEN,
        ppl_eval_tps_fp16,
        fp16_ms,
    );

    eprintln!("[{model_dir_name}] computing turbo4asym PPL...");
    let t1 = Instant::now();
    let (ppl_turbo, _) = compute_vlm_ppl(&model, &tokenizer, KVCacheMode::Turbo4Asym);
    let turbo_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let ppl_eval_tps_turbo = n_tokens as f64 / (turbo_ms / 1000.0).max(1e-9);
    eprintln!("[{model_dir_name}][turbo4asym] PPL={ppl_turbo:.4} ({turbo_ms:.0}ms)");
    record_speed_row(
        model_dir_name,
        "turbo4asym",
        VLM_PPL_CHUNK_LEN,
        ppl_eval_tps_turbo,
        turbo_ms,
    );

    let rel_ppl = (ppl_turbo - ppl_fp16) / ppl_fp16;
    eprintln!(
        "[{model_dir_name}] PPL relative increase: {:.4}% (gate: ≤{:.1}%)",
        rel_ppl * 100.0,
        PPL_GATE_REL * 100.0
    );

    eprintln!("[{model_dir_name}] NIAH fp16 (4K cells)...");
    let niah_baseline = run_niah(
        &model,
        &tokenizer,
        KVCacheMode::Fp16,
        VLM_NIAH_CELLS,
        model_dir_name,
        "fp16",
    );
    eprintln!("[{model_dir_name}] NIAH turbo4asym (4K cells)...");
    let niah_turbo = run_niah(
        &model,
        &tokenizer,
        KVCacheMode::Turbo4Asym,
        VLM_NIAH_CELLS,
        model_dir_name,
        "turbo4asym",
    );

    let total_cells = VLM_NIAH_CELLS.len();
    eprintln!(
        "[{model_dir_name}] NIAH: baseline={niah_baseline}/{total_cells} \
         turbo={niah_turbo}/{total_cells}"
    );

    Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo))
}

/// Convert an `MlxArray` of arbitrary float dtype to a flat `Vec<f32>`.
///
/// Used by the image-token kurtosis path to read out a slice of the K cache
/// after one prefill pass on an image+text prompt. We always cast to FP32
/// before extracting bytes so the kurtosis math runs in a single, predictable
/// dtype regardless of the cache's underlying storage (FP16 / BF16 / FP32).
fn array_to_f32_vec(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    eval(arr);
    let arr_f32 = astype(arr, dtype::FLOAT32);
    eval(&arr_f32);
    array_to_raw_bytes(&arr_f32)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Locate the **contiguous** image-token span in `prompt_tokens`.
///
/// Returns `(start, len)` where `prompt_tokens[start..start+len]` is the
/// inclusive run of `image_token_id`, or `None` if no image tokens are
/// present. For all VLM families currently shipped in mlxcel that expose an
/// `image_token_id`, the runtime inserts image tokens as a contiguous block
/// — so we assert that and bail loudly if the prompt ever turns out to
/// interleave them.
fn find_image_token_span(prompt_tokens: &[i32], image_token_id: i32) -> Option<(usize, usize)> {
    let first = prompt_tokens.iter().position(|&t| t == image_token_id)?;
    let last = prompt_tokens.iter().rposition(|&t| t == image_token_id)?;
    let span_len = last - first + 1;
    let actual_count = prompt_tokens[first..=last]
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();
    assert_eq!(
        actual_count, span_len,
        "image tokens are not contiguous in the prompt — this kurtosis helper assumes \
         a single contiguous image block (got {actual_count} image tokens across a \
         span of {span_len} positions)",
    );
    Some((first, span_len))
}

/// One end-to-end image-token K-side kurtosis measurement on a real VLM.
///
/// Builds an image+text prompt against `model_dir_name`, runs prefill once
/// through `model.forward_with_embeddings()` with FP16 KV cache, detaches the
/// layer-0 K tensor, slices the rows that correspond to image tokens, and
/// returns the non-excess kurtosis pre- and post-WHT rotation along with the
/// number of image-token positions sampled.
///
/// Used by: [`test_vlm_image_token_kurtosis`]
fn measure_vlm_image_token_kurtosis(model_dir_name: &str) -> Option<(f64, f64, usize)> {
    let model_dir = repo_model_dir(model_dir_name);
    if !model_dir.exists() {
        eprintln!(
            "  {model_dir_name}: directory not found at {}, skipping",
            model_dir.display()
        );
        return None;
    }

    eprintln!("  loading {model_dir_name}...");
    let (model, tokenizer) = load_model(&model_dir).ok()?;
    if !model.is_vlm() {
        eprintln!("  {model_dir_name}: not a VLM checkpoint, skipping");
        return None;
    }
    let image_token_id = vlm_image_token_id(&model)?;
    eprintln!("  {model_dir_name}: image_token_id={image_token_id}");

    let image_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("test_image.png");
    let image = image::open(&image_path).expect("load test image fixture");

    // Use a short, deliberately bland text prompt — we want the image tokens
    // to dominate the cache rows we slice, not the text.
    let prompt_text = "Describe this image briefly.";
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode(prompt_text, true)
        .expect("tokenize VLM prompt")
        .iter()
        .map(|&id| id as i32)
        .collect();

    let prepared = prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        prompt_text,
        std::slice::from_ref(&image),
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect()
        },
    )
    .ok()
    .flatten()?;

    let Some((image_start, image_len)) = find_image_token_span(&prompt_tokens, image_token_id)
    else {
        eprintln!(
            "  {model_dir_name}: no image tokens after VLM preparation — \
             the prompt template likely did not insert any (image_token_id={image_token_id})"
        );
        return None;
    };
    eprintln!(
        "  {model_dir_name}: prompt has {} tokens with image span [{image_start}..{}) \
         ({image_len} image tokens)",
        prompt_tokens.len(),
        image_start + image_len
    );

    let (inputs_embeds, mask) =
        prepared_embedding_refs(&prepared.embeddings).expect("prepared embeddings missing");

    let num_layers = model.num_layers();
    let mut generator = CxxGenerator::new_with_kv_mode(num_layers, KVCacheMode::Fp16);

    let input_ids_arr =
        mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);

    // Single prefill pass populates `caches[layer]` with FP16 K (and V).
    let logits = model.forward_with_embeddings(
        &input_ids_arr,
        Some(inputs_embeds),
        generator.caches_mut(),
        mask,
    );
    eval(&logits);

    let detached = generator.caches_mut()[0].clone_handle();
    let k_full = detached.keys()?;
    let k_shape = array_shape(k_full);
    if k_shape.len() != 4 {
        eprintln!(
            "  {model_dir_name}: unexpected K cache shape {k_shape:?} \
             (expected 4-D [B, H, T, D])"
        );
        return None;
    }
    let (b, h, _t, head_dim) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    let stop_t = (image_start + image_len) as i32;
    let k_image = mlxcel_core::slice(
        k_full,
        &[0, 0, image_start as i32, 0],
        &[b, h, stop_t, head_dim],
    );
    eval(&k_image);

    // Kurtosis pre-rotation over the image-only slice.
    let k_image_f32 = array_to_f32_vec(&k_image);
    let kurt_before = non_excess_kurtosis(&k_image_f32);

    // Apply the same WHT + sign rotation the runtime would use, then measure
    // post-rotation kurtosis. `turbo4_v_rotate` requires the last dim to
    // match `TurboQuantParams::head_dim`, so we build params with the cache's
    // own head_dim (a power-of-two ≤256 on every VLM we ship today).
    let params = TurboQuantParams::new(head_dim as u32, 0xAB_CD_12_34);
    let rotated = turbo4_v_rotate(&k_image, &params);
    eval(&rotated);
    let rotated_f32 = array_to_f32_vec(&rotated);
    let kurt_after = non_excess_kurtosis(&rotated_f32);

    Some((kurt_before, kurt_after, image_len))
}

/// VLM image-token K-side kurtosis sanity test.
///
/// establishes that the WHT + sign-flip rotation drops K-side
/// kurtosis on **text** activations (TurboQuant+ paper reports ~900 → ~2.9
/// on Qwen3-1.7B). This test repeats the measurement on the **image-token**
/// slice of the K cache for every locally-cached VLM checkpoint to confirm
/// the rotation behaves the same way on visually-derived activations.
///
/// Pre-conditions are intentionally loose: the test soft-skips if no VLM
/// checkpoint with a discoverable `image_token_id` is present.
///
/// # Gates
///
/// - `kurt_after < kurt_before` — rotation must not *increase* heavy-tailedness.
/// - `kurt_after < kurt_before * 0.5` **OR** `kurt_after < 4.0` —
///   rotation must drop kurtosis by ≥50%, OR land within the near-Gaussian
///   regime (Gaussian = 3.0 exactly, +1 slack for sample-noise on the small
///   image-token slice). The OR-floor handles VLM families like Pixtral
///   whose image-token K embeddings are already close to Gaussian
///   pre-rotation (measured ~5.3 on M1 Ultra with 4096-token image grid),
///   leaving the rotation no headroom for a 50% drop while still moving
///   the slice closer to Gaussian.
#[test]
#[ignore = "requires at least one VLM checkpoint locally — \
            run with --release -- --ignored test_vlm_image_token_kurtosis --nocapture"]
fn test_vlm_image_token_kurtosis() {
    // Order matters: smaller checkpoints first so the soft-skip walk is fast
    // when only the larger ones are available; aya-vision-8b and gemma-4-e4b
    // are both healthy 4-bit VLM fixtures shipped during the follow-up.
    let candidates = [
        "qwen2-vl-2b-4bit",
        "qwen2.5-vl-3b-4bit",
        "aya-vision-8b",
        "gemma-4-e4b-it-4bit",
        "gemma3-4b-4bit",
        "phi-3.5-vision-4bit",
        "pixtral-12b-4bit",
    ];

    let mut results: Vec<(String, f64, f64, usize)> = Vec::new();
    for name in &candidates {
        if let Some((before, after, image_len)) = measure_vlm_image_token_kurtosis(name) {
            eprintln!(
                "  [{name}] image-token K kurtosis: before={before:.4} \
                 after={after:.4} (image_tokens={image_len})"
            );
            results.push((name.to_string(), before, after, image_len));
        }
    }

    if results.is_empty() {
        eprintln!(
            "Skipping test_vlm_image_token_kurtosis: no VLM checkpoint with a \
             discoverable image_token_id was found locally."
        );
        return;
    }

    eprintln!("\n  Summary: VLM image-token K-side kurtosis pre/post WHT rotation");
    for (model_name, kurt_before, kurt_after, image_tokens) in &results {
        eprintln!(
            "    {model_name:<24} n={image_tokens:<5} before={kurt_before:>10.4} → after={kurt_after:>8.4} \
             ({:.1}% of original)",
            kurt_after / kurt_before * 100.0
        );
    }

    // Gate every collected measurement so any VLM family that regresses is
    // surfaced rather than masked by a healthier sibling.
    for (model_name, kurt_before, kurt_after, _) in &results {
        assert!(
            kurt_before.is_finite() && kurt_after.is_finite(),
            "kurtosis must be finite (model={model_name}, \
             before={kurt_before}, after={kurt_after})"
        );
        assert!(
            kurt_after < kurt_before,
            "VLM image-token K rotation should not *increase* kurtosis: \
             before={kurt_before:.4} after={kurt_after:.4} (model={model_name})"
        );
        let drop_ratio = kurt_after / kurt_before;
        let near_gaussian = *kurt_after < 4.0;
        assert!(
            drop_ratio < 0.5 || near_gaussian,
            "VLM image-token K rotation only reduced kurtosis from {kurt_before:.4} to \
             {kurt_after:.4} ({:.1}% of original); expected ≤50% drop OR post-rotation \
             < 4.0 (model={model_name})",
            drop_ratio * 100.0,
        );
    }
}

/// VLM Turbo4Asym B3 quality gate — Qwen2-VL 2B, text-only path.
///
/// Surrogate for the Qwen2.5-VL 3B fixture suggested in same Qwen-VL
/// architecture (insert/expand image tokens via mRoPE-aware runtime) at a
/// smaller size that loads under typical M1 Ultra dev RAM. Gates the relative
/// PPL increase ≤ [`PPL_GATE_REL`] and NIAH retrieval not regressing.
#[test]
#[ignore = "requires qwen2-vl-2b-4bit weights — \
            run with --release -- --ignored test_qwen2_vl_2b_quality_gate --nocapture"]
fn test_qwen2_vl_2b_quality_gate() {
    let Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo)) =
        run_vlm_quality_gate("qwen2-vl-2b-4bit")
    else {
        return;
    };

    let rel = (ppl_turbo - ppl_fp16) / ppl_fp16;
    assert!(
        ppl_fp16.is_finite() && ppl_fp16 > 0.0,
        "qwen2-vl-2b fp16 PPL must be finite and positive; got {ppl_fp16}"
    );
    assert!(
        rel <= PPL_GATE_REL,
        "Qwen2-VL-2B PPL regression {:.4}% > {:.1}% gate \
         (fp16={ppl_fp16:.4}, turbo={ppl_turbo:.4})",
        rel * 100.0,
        PPL_GATE_REL * 100.0
    );
    assert!(
        niah_turbo >= niah_baseline,
        "Qwen2-VL-2B NIAH turbo ({niah_turbo}) dropped below baseline ({niah_baseline})"
    );
}

/// VLM Turbo4Asym B3 quality gate — Aya-Vision 8B, text-only path.
///
/// Aya-Vision is a Cohere2-decoder + SigLIP vision tower checkpoint (the
/// "Standard" VLM family in `LoadedModel::vlm_runtime`). Together with
/// Qwen2-VL 2B this satisfies the "at least 2 VLM models" criterion.
#[test]
#[ignore = "requires aya-vision-8b weights — \
            run with --release -- --ignored test_aya_vision_8b_quality_gate --nocapture"]
fn test_aya_vision_8b_quality_gate() {
    let Some((ppl_fp16, ppl_turbo, niah_baseline, niah_turbo)) =
        run_vlm_quality_gate("aya-vision-8b")
    else {
        return;
    };

    let rel = (ppl_turbo - ppl_fp16) / ppl_fp16;
    assert!(
        ppl_fp16.is_finite() && ppl_fp16 > 0.0,
        "aya-vision-8b fp16 PPL must be finite and positive; got {ppl_fp16}"
    );
    assert!(
        rel <= PPL_GATE_REL,
        "Aya-Vision-8B PPL regression {:.4}% > {:.1}% gate \
         (fp16={ppl_fp16:.4}, turbo={ppl_turbo:.4})",
        rel * 100.0,
        PPL_GATE_REL * 100.0
    );
    assert!(
        niah_turbo >= niah_baseline,
        "Aya-Vision-8B NIAH turbo ({niah_turbo}) dropped below baseline ({niah_baseline})"
    );
}
